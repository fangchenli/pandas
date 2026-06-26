#!/usr/bin/env python3
"""Probe: parallel string-key factorize aggregate vs Arrow group_by + Polars.

Validates the string-key group-by kernel (docs/STRING_HASH_AGGREGATE_KERNEL.md).
Builds q10's group input and groups by its 7 keys (incl. 5 strings) summing
revenue, three ways:
  - arrow single group_by  (the fallback path)
  - the kernel             (hash_string_col/hash_int_col -> partition_by_key ->
                            bucket_factorize -> aggregate on codes -> attach keys)
  - polars                 (reference)

Result (SF-3, 8 threads): the parallel factorize path reaches Polars parity on
the group operator (~30 ms vs ~57 ms single). The win is parallelising the
string hashing; keys are compared by a 128-bit hash (exact in practice).
"""

from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

from pandas._libs.lazy_groupby import (
    bucket_factorize,
    hash_int_col,
    hash_string_col,
    partition_by_key,
)

POOL = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))
NT = min(16, os.cpu_count() or 4)


def med(fn, n=9):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def str_buffers(arr):
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    bufs = arr.buffers()
    off_dt = np.int64 if pa.types.is_large_string(arr.type) else np.int32
    offsets = np.frombuffer(bufs[1], dtype=off_dt, count=len(arr) + 1).astype(np.int64)
    data = (
        np.frombuffer(bufs[2], dtype=np.uint8)
        if bufs[2] is not None
        else np.zeros(0, np.uint8)
    )
    return np.ascontiguousarray(offsets), np.ascontiguousarray(data)


def int_arr(arr):
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    npv = arr.to_numpy(zero_copy_only=False)
    if npv.dtype.kind in "Mf":
        return np.ascontiguousarray(npv.view(np.int64))
    return np.ascontiguousarray(npv.astype(np.int64, copy=False))


def spans(n, t):
    s = (n + t - 1) // t
    return [(i * s, min(n, (i + 1) * s)) for i in range(t) if i * s < n]


def factorize(table, int_keys, str_keys, nb=16, par=True):
    n = table.num_rows
    ic = {c: int_arr(table.column(c)) for c in int_keys}
    sc = {c: str_buffers(table.column(c)) for c in str_keys}
    acc = np.empty(n, np.uint64)
    acc2 = np.empty(n, np.uint64)

    def hr(lohi):
        lo, hi = lohi
        first = True
        for c in int_keys:
            hash_int_col(ic[c], acc, acc2, first, lo, hi)
            first = False
        for c in str_keys:
            o, d = sc[c]
            hash_string_col(o, d, acc, acc2, first, lo, hi)
            first = False

    if par:
        list(POOL.map(hr, spans(n, NT)))
    else:
        hr((0, n))
    perm, off = partition_by_key(np.ascontiguousarray(acc), nb)
    codes = np.empty(n, np.int64)
    res = list(
        POOL.map(
            lambda b: bucket_factorize(
                perm, int(off[b]), int(off[b + 1]), acc, acc2, codes
            ),
            range(nb),
        )
    )
    base = 0
    reps_list = []
    for b in range(nb):
        ng, rep = res[b]
        if ng:
            if base:
                codes[perm[off[b] : off[b + 1]]] += base
            reps_list.append(rep)
            base += ng
    reps = np.concatenate(reps_list) if reps_list else np.empty(0, np.int64)
    return codes, reps


def kernel(table, int_keys, str_keys, valcol, par=True):
    codes, reps = factorize(table, int_keys, str_keys, par=par)
    rev = np.ascontiguousarray(
        pc.cast(table.column(valcol), pa.float64()).to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    sums = np.bincount(codes, weights=rev, minlength=int(codes.max()) + 1)
    grp = int_keys + str_keys
    out = {c: table.column(c).take(pa.array(reps)) for c in grp}
    out[valcol + "_sum"] = pa.array(sums)
    return pa.table(out)


sf = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
con = duckdb.connect()
con.execute("INSTALL tpch;LOAD tpch")
con.execute(f"CALL dbgen(sf={sf})")
_SQL = """
SELECT c_custkey, c_name, CAST(c_acctbal AS DOUBLE) c_acctbal, c_phone,
       n_name, c_address, c_comment,
       l_extendedprice * (1 - l_discount) AS revenue
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R' AND c_nationkey = n_nationkey
"""
df = con.execute(_SQL).fetch_arrow_table()
con.close()
allkeys = [
    "c_custkey",
    "c_name",
    "c_acctbal",
    "c_phone",
    "n_name",
    "c_address",
    "c_comment",
]
ik = ["c_custkey", "c_acctbal"]
sk = ["c_name", "c_phone", "n_name", "c_address", "c_comment"]
print(f"rows={df.num_rows:,} threads={NT}")
ref = df.group_by(allkeys).aggregate([("revenue", "sum")])
ker = kernel(df, ik, sk, "revenue")
sa = np.sort(pc.cast(ref.column("revenue_sum"), pa.float64()).to_numpy())
sb = np.sort(ker.column("revenue_sum").to_numpy())
print(f"correct={ref.num_rows == ker.num_rows and np.allclose(sa, sb)}")


def _arrow():
    return df.group_by(allkeys).aggregate([("revenue", "sum")])


def _pol():
    return pl.from_arrow(df).group_by(allkeys).agg(pl.col("revenue").sum())


def _ker_serial():
    return kernel(df, ik, sk, "revenue", par=False)


def _ker_par():
    return kernel(df, ik, sk, "revenue", par=True)


print(f"arrow single      : {med(_arrow):6.1f} ms")
print(f"kernel serial-hash: {med(_ker_serial):6.1f} ms")
print(f"kernel par-hash   : {med(_ker_par):6.1f} ms")
print(f"polars            : {med(_pol):6.1f} ms")
