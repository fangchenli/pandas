#!/usr/bin/env python3
"""Probe (generalization): does buffer-resident join->agg parity HOLD beyond the
simple 2-table low-card case? Two harder shapes from BUFFER_JOIN_AGG_PROBE.md's
open caveats:

  HIGHCARD: lineitem |><| orders -> group_by(l_orderkey).sum   (~4.5M groups @
            SF-3 -> the group stage routes to the parallel lazy_groupby kernel,
            not a single arrow group_by).
  CHAIN:    customer |><| orders |><| lineitem -> group_by(o_orderdate).sum
            (q3 backbone -> TWO joins; each intermediate still materializes in
            the current engine; tests whether the buffer path holds through a
            chain).

Both are order-insensitive (a group terminates the pipeline), so the buffer
path joins in whichever direction is cheapest (build on the smaller side) and
ignores row order -- the safest fusion class, same as the simple probe.

Paths per scenario (same asserted-equal result):
  POLARS         native lazy chain->group->agg
  LIVE           our engine, collect(use_physical_planner=True)
  BUFFER         lazy_join indices -> pc.take only-surviving-cols -> group
                 (parallel lazy_groupby for HIGHCARD, arrow group_by for CHAIN)

Read-only. SF-3 default. Run:
    python pandas/lazy/benchmarks/probe_buffer_join_agg_gen.py --sf 3
"""

from __future__ import annotations

import argparse
import time

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

import pandas as pd
from pandas.lazy import col
from pandas.lazy.backends.numpy.join import inner_join_indexers_i8


def _t(fn, n=5):
    out, ts = None, []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), out


def _join_idx(left_keys, right_keys):
    """(left_idx, right_idx) building on the smaller side (order irrelevant)."""
    if len(right_keys) <= len(left_keys):
        li, ri = inner_join_indexers_i8(left_keys, right_keys, max_hit_fraction=None)
        return li, ri
    ri, li = inner_join_indexers_i8(right_keys, left_keys, max_hit_fraction=None)
    return li, ri


_POOL = None


def _pool(n_buckets):
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        import os

        _POOL = ThreadPoolExecutor(max_workers=min(n_buckets, os.cpu_count() or 4))
    return _POOL


def _parallel_group_sum(key_i64: np.ndarray, val_pa, n_buckets=16):
    """Replicate the engine's parallel partitioned hash-agg (high cardinality).

    Uses a persistent pool (as the engine does) so the timing is not penalized
    by per-call thread spin-up.
    """
    from pandas._libs.lazy_groupby import partition_by_key

    comb = np.ascontiguousarray(key_i64.astype(np.uint64))
    perm, off = partition_by_key(comb, n_buckets)
    perm_pa = pa.array(perm)
    key_pa = pa.array(key_i64)
    tbl = pa.table({"k": key_pa, "v": val_pa})

    def _bucket(i):
        lo, hi = int(off[i]), int(off[i + 1])
        if hi == lo:
            return None
        sub = tbl.take(perm_pa[lo:hi])
        return sub.group_by("k").aggregate([("v", "sum")])

    parts = [
        p for p in _pool(n_buckets).map(_bucket, range(n_buckets)) if p and p.num_rows
    ]
    return pa.concat_tables(parts)


def load(sf):
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    li = con.execute("SELECT l_orderkey, l_extendedprice FROM lineitem").df()
    orders = con.execute("SELECT o_orderkey, o_custkey, o_orderdate FROM orders").df()
    cust = con.execute("SELECT c_custkey FROM customer").df()
    con.close()
    od = pd.to_datetime(orders["o_orderdate"]).to_numpy().astype("datetime64[D]")
    orders["o_orderdate"] = od.astype("int64")
    li["l_extendedprice"] = li["l_extendedprice"].astype("float64")
    return li, orders, cust


def as_series(tbl, key, valcol, name="o_orderdate"):
    return pd.Series(
        tbl[valcol].to_numpy(),
        index=pd.Index(tbl[key].to_numpy(), name=name),
    ).sort_index()


def run_highcard(li, orders, n):
    print("\n=== HIGHCARD: lineitem |><| orders -> group_by(l_orderkey).sum ===")
    lk = li["l_orderkey"].to_numpy()
    ok = orders["o_orderkey"].to_numpy()
    price = li["l_extendedprice"].to_numpy()
    price_pa = pa.array(price)

    ref = (
        pd.DataFrame({"l_orderkey": lk, "l_extendedprice": price})
        .merge(
            pd.DataFrame({"o_orderkey": ok}),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .groupby("l_orderkey")["l_extendedprice"]
        .sum()
        .sort_index()
    )
    ngroups = ref.shape[0]

    li_pl = pl.DataFrame({"l_orderkey": lk, "l_extendedprice": price})
    ord_pl = pl.DataFrame({"o_orderkey": ok})

    def polars_run():
        return (
            li_pl.lazy()
            .join(ord_pl.lazy(), left_on="l_orderkey", right_on="o_orderkey")
            .group_by("l_orderkey")
            .agg(pl.col("l_extendedprice").sum())
            .collect()
        )

    def live_run():
        return (
            li.select()
            .join(
                orders.select("o_orderkey"), left_on="l_orderkey", right_on="o_orderkey"
            )
            .group_by("l_orderkey")
            .agg(col("l_extendedprice").sum().alias("s"))
            .collect(use_physical_planner=True)
        )

    def buffer_run():
        li_i, _ = _join_idx(lk, ok)
        g_key = pc.take(pa.array(lk), pa.array(li_i))
        g_val = pc.take(price_pa, pa.array(li_i))
        t = _parallel_group_sum(g_key.to_numpy(), g_val)
        return as_series(t, "k", "v_sum", name="l_orderkey")

    return _report(
        ngroups,
        n,
        polars_run,
        live_run,
        buffer_run,
        ref,
        "l_orderkey",
        "l_extendedprice",
    )


def run_chain(li, orders, cust, n):
    print("\n=== CHAIN: customer><orders><lineitem -> group_by(orderdate).sum ===")
    lk = li["l_orderkey"].to_numpy()
    price = li["l_extendedprice"].to_numpy()
    ok = orders["o_orderkey"].to_numpy()
    ock = orders["o_custkey"].to_numpy()
    odate = orders["o_orderdate"].to_numpy()
    ck = cust["c_custkey"].to_numpy()
    price_pa, odate_pa, ok_pa = pa.array(price), pa.array(odate), pa.array(ok)

    li_df = pd.DataFrame({"l_orderkey": lk, "l_extendedprice": price})
    ord_df = pd.DataFrame({"o_orderkey": ok, "o_custkey": ock, "o_orderdate": odate})
    cust_df = pd.DataFrame({"c_custkey": ck})
    ref = (
        cust_df.merge(ord_df, left_on="c_custkey", right_on="o_custkey")
        .merge(li_df, left_on="o_orderkey", right_on="l_orderkey")
        .groupby("o_orderdate")["l_extendedprice"]
        .sum()
        .sort_index()
    )
    ngroups = ref.shape[0]

    li_pl = pl.DataFrame(li_df)
    ord_pl = pl.DataFrame(ord_df)
    cust_pl = pl.DataFrame(cust_df)

    def polars_run():
        return (
            cust_pl.lazy()
            .join(ord_pl.lazy(), left_on="c_custkey", right_on="o_custkey")
            .join(li_pl.lazy(), left_on="o_orderkey", right_on="l_orderkey")
            .group_by("o_orderdate")
            .agg(pl.col("l_extendedprice").sum())
            .collect()
        )

    def live_run():
        return (
            cust_df.select()
            .join(orders.select(), left_on="c_custkey", right_on="o_custkey")
            .join(li.select(), left_on="o_orderkey", right_on="l_orderkey")
            .group_by("o_orderdate")
            .agg(col("l_extendedprice").sum().alias("s"))
            .collect(use_physical_planner=True)
        )

    def buffer_run():
        # join1: customer |><| orders  (build on customer, the small side)
        o_i, _c_i = _join_idx(ock, ck)  # (orders_idx, cust_idx)
        ab_okey = pc.take(ok_pa, pa.array(o_i))  # surviving: next join key
        ab_date = pc.take(odate_pa, pa.array(o_i))  # surviving: final group key
        # join2: AB |><| lineitem  on orderkey
        ab_okey_np = ab_okey.to_numpy()
        l_i, ab_i = _join_idx(lk, ab_okey_np)  # (lineitem_idx, ab_idx)
        g_price = pc.take(price_pa, pa.array(l_i))
        g_date = pc.take(ab_date, pa.array(ab_i))
        t = pa.table({"o_orderdate": g_date, "l_extendedprice": g_price})
        grouped = t.group_by("o_orderdate").aggregate([("l_extendedprice", "sum")])
        return as_series(grouped, "o_orderdate", "l_extendedprice_sum")

    return _report(
        ngroups, n, polars_run, live_run, buffer_run, ref, "o_orderdate", None
    )


def _vals(out, keyname):
    """Sorted aggregate-value array from any path's output (order-insensitive)."""
    if isinstance(out, pd.Series):
        return np.sort(out.to_numpy())
    if isinstance(out, pl.DataFrame):
        return np.sort(out[out.columns[-1]].to_numpy())
    # pandas DataFrame (LIVE): last column is the agg
    return np.sort(out.iloc[:, -1].to_numpy())


def _report(ngroups, n, polars_run, live_run, buffer_run, ref, keyname, valname):
    print(f"groups={ngroups:,}")
    ref_v = np.sort(ref.to_numpy())
    res = {}
    for name, fn in [
        ("POLARS", polars_run),
        ("LIVE", live_run),
        ("BUFFER", buffer_run),
    ]:
        ms, out = _t(fn, n)
        got = _vals(out, keyname)
        assert got.shape == ref_v.shape and np.allclose(got, ref_v, rtol=1e-6), (
            f"{name} mismatch ({got.shape} vs {ref_v.shape})"
        )
        res[name] = ms
    pl_ms = res["POLARS"]
    print(f"{'path':<10}{'ms':>9}{'x vs PL':>10}")
    for k in ("POLARS", "LIVE", "BUFFER"):
        print(f"{k:<10}{res[k]:>9.1f}{pl_ms / res[k]:>9.2f}x")
    print(f"  BUFFER vs LIVE: {res['LIVE'] / res['BUFFER']:.2f}x")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=3.0)
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--scenario", choices=["highcard", "chain", "both"], default="both")
    args = ap.parse_args()

    print(f"pyarrow {pa.__version__}  polars {pl.__version__}  sf={args.sf}")
    li, orders, cust = load(args.sf)
    print(f"lineitem={len(li):,}  orders={len(orders):,}  customer={len(cust):,}")

    if args.scenario in ("highcard", "both"):
        run_highcard(li, orders, args.n)
    if args.scenario in ("chain", "both"):
        run_chain(li, orders, cust, args.n)


if __name__ == "__main__":
    main()
