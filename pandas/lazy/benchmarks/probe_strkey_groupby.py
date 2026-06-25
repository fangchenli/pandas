#!/usr/bin/env python3
"""Probe: can we parallelize q10's high-card STRING-key group-by?

q10 groups the joined output by 7 customer/nation keys incl. 3 long strings
(c_name, c_address, c_comment). Measures, on the q10 group input:
  A single full 7-key group_by (current fallback)
  B single group by c_custkey only (the integer superkey) -- no string hashing
  C parallel partition-by-custkey + per-bucket group (narrow [custkey,revenue])
  + Polars full and custkey-only.

Finding (docs/ENGINE_GAP_REFRAMING.md): the string-key group is NOT a clean
catchable lever. Naive parallel (partition + take of wide string cols into
buckets) REGRESSES via the string-gather tax; the only fast thing (group by the
integer superkey, B ~10ms vs A ~62ms) needs superkey metadata we don't track,
and any path that hashes/moves/min's the strings hits a ~50ms floor. Polars's
~30ms comes from genuinely parallel string hashing.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from pandas._libs.lazy_groupby import partition_by_key

POOL = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))


def med(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


sf = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
con = duckdb.connect()
con.execute("INSTALL tpch;LOAD tpch")
con.execute(f"CALL dbgen(sf={sf})")
_Q10_INPUT = """
SELECT c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment,
       l_extendedprice * (1 - l_discount) AS revenue
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R' AND c_nationkey = n_nationkey
"""
df = con.execute(_Q10_INPUT).fetch_arrow_table()
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
aggs = [("revenue", "sum")]
ck = df.column("c_custkey").combine_chunks().to_numpy().astype(np.int64)
print(f"rows={df.num_rows:,} distinct custkey={len(np.unique(ck)):,}")


# A) single full (7 keys)
def single_full():
    return df.group_by(allkeys).aggregate(aggs)


# B) single on custkey only (narrow) -- floor of grouping work, no strings
narrow = df.select(["c_custkey", "revenue"])


def single_ck():
    return narrow.group_by(["c_custkey"]).aggregate(aggs)


# C) parallel partition by custkey, narrow [custkey,revenue], group custkey
def par_ck(nb=16):
    comb = np.ascontiguousarray(ck.view(np.uint64))
    perm, off = partition_by_key(comb, nb)
    pp = pa.array(perm)

    def b(i):
        lo, hi = int(off[i]), int(off[i + 1])
        if hi == lo:
            return None
        return narrow.take(pp[lo:hi]).group_by(["c_custkey"]).aggregate(aggs)

    return pa.concat_tables(
        [p for p in POOL.map(b, range(nb)) if p is not None and p.num_rows]
    )


pdf = pl.from_arrow(df)


def pol_full():
    return pdf.group_by(allkeys).agg(pl.col("revenue").sum())


def pol_ck():
    return pdf.group_by(["c_custkey"]).agg(pl.col("revenue").sum())


print(f"A single full(7key) : {med(single_full):6.1f} ms")
print(f"B single custkey    : {med(single_ck):6.1f} ms")
print(f"C par   custkey     : {med(par_ck):6.1f} ms")
print(f"  polars full(7key) : {med(pol_full):6.1f} ms")
print(f"  polars custkey    : {med(pol_ck):6.1f} ms")
