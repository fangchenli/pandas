#!/usr/bin/env python3
"""Probe: radix-partitioned parallel hash join over our nogil Cython kernel.

Partition both sides by key-hash (partition_by_key, the groupby-win kernel) so
matching keys co-locate in the same bucket, then build+probe each bucket on a
thread pool (low-level build_join_table_i8 + probe_*_chunk, single-threaded per
bucket -> parallel across buckets), concat the remapped indices. Parallelizes
the BUILD and the probe, which the current single-build kernel does not.

Compares: current inner_join_indexers_i8 vs partitioned vs Polars, on the big
TPC-H join (lineitem 18M >< orders 4.5M). Validates the matched pair multiset.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import time

import duckdb
import numpy as np
import polars as pl

from pandas._libs.lazy_groupby import partition_by_key
from pandas._libs.lazy_join import (
    build_join_table_i8,
    probe_count_chunk,
    probe_fill_chunk,
)

from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

_POOL = ThreadPoolExecutor(max_workers=min(16, __import__("os").cpu_count() or 4))


def _bucket_join(lk, rk, l_rows, r_rows):
    """Inner join within one bucket; return (global_left, global_right) or None."""
    nl, nr = len(l_rows), len(r_rows)
    if nl == 0 or nr == 0:
        return None
    # build on the smaller side
    if nr <= nl:
        bk = np.ascontiguousarray(rk[r_rows], dtype=np.int64)
        pk = np.ascontiguousarray(lk[l_rows], dtype=np.int64)
        build_right = True
    else:
        bk = np.ascontiguousarray(lk[l_rows], dtype=np.int64)
        pk = np.ascontiguousarray(rk[r_rows], dtype=np.int64)
        build_right = False
    slot_key, slot_gid, counts, offsets, group_rows = build_join_table_i8(bk)
    total = probe_count_chunk(pk, 0, len(pk), slot_key, slot_gid, counts)
    if total == 0:
        return None
    out_probe = np.empty(total, dtype=np.int64)
    out_build = np.empty(total, dtype=np.int64)
    probe_fill_chunk(
        pk,
        0,
        len(pk),
        slot_key,
        slot_gid,
        counts,
        offsets,
        group_rows,
        out_probe,
        out_build,
        0,
    )
    if build_right:  # probe=left, build=right
        return l_rows[out_probe], r_rows[out_build]
    return l_rows[out_build], r_rows[out_probe]  # probe=right, build=left


def partitioned_join(lk, rk, n_buckets=16):
    lk_u = np.ascontiguousarray(lk.view(np.uint64))
    rk_u = np.ascontiguousarray(rk.view(np.uint64))
    perm_l, off_l = partition_by_key(lk_u, n_buckets)
    perm_r, off_r = partition_by_key(rk_u, n_buckets)

    def do(b):
        return _bucket_join(
            lk,
            rk,
            perm_l[off_l[b] : off_l[b + 1]],
            perm_r[off_r[b] : off_r[b + 1]],
        )

    parts = [p for p in _POOL.map(do, range(n_buckets)) if p is not None]
    if not parts:
        e = np.empty(0, dtype=np.int64)
        return e, e
    left_idx = np.concatenate([p[0] for p in parts])
    right_idx = np.concatenate([p[1] for p in parts])
    return left_idx, right_idx


def med(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def checksum(lk, rk, li, ri):
    """Order-independent fingerprint of the matched pair multiset."""
    # pair = (left_key value, left_row, right_row) — sum of a mix
    return (
        len(li),
        int(lk[li].sum()),
        int(
            (li.astype(np.int64) * 1315423911 + ri.astype(np.int64)).sum()
            & ((1 << 62) - 1)
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=3.0)
    ap.add_argument("-b", type=int, default=16)
    args = ap.parse_args()
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={args.sf})")
    lk = np.ascontiguousarray(
        con.execute("SELECT l_orderkey FROM lineitem").df()["l_orderkey"].to_numpy(),
        dtype=np.int64,
    )
    ok = np.ascontiguousarray(
        con.execute("SELECT o_orderkey FROM orders").df()["o_orderkey"].to_numpy(),
        dtype=np.int64,
    )
    con.close()
    print(f"sf={args.sf} buckets={args.b}  li={len(lk):,}  ord={len(ok):,}")

    cur = inner_join_indexers_i8(lk, ok, max_hit_fraction=None)
    par = partitioned_join(lk, ok, args.b)
    cs_cur = checksum(lk, ok, cur[0], cur[1])
    cs_par = checksum(lk, ok, par[0], par[1])
    print("correct:", cs_cur == cs_par, "| cur", cs_cur, "| par", cs_par)

    li_pl = pl.DataFrame({"l_orderkey": lk})
    ord_pl = pl.DataFrame({"o_orderkey": ok})

    def pol():
        return (
            li_pl.lazy()
            .join(ord_pl.lazy(), left_on="l_orderkey", right_on="o_orderkey")
            .collect()
        )

    cur_ms = med(lambda: inner_join_indexers_i8(lk, ok, max_hit_fraction=None))
    par_ms = med(lambda: partitioned_join(lk, ok, args.b))
    pol_ms = med(pol)
    print(f"\nCURRENT inner_join_indexers_i8 : {cur_ms:7.1f} ms")
    print(f"PARTITIONED parallel join      : {par_ms:7.1f} ms")
    print(f"POLARS join (keys only)        : {pol_ms:7.1f} ms")


if __name__ == "__main__":
    main()
