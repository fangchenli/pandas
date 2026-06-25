#!/usr/bin/env python3
"""Decompose WHERE lazy-pandas time goes on a join query vs Polars in-memory.
Both engines materialize. cProfile our engine; polars per-node profile (in-mem).
"""

import argparse
import cProfile
import io
import pstats
import sys
import time

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, "pandas/lazy/benchmarks")
import bench_tpch as B


def med(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=3.0)
    ap.add_argument("-q", type=int, default=10)
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={args.sf})")
    t = B.load_tables(con)
    plt = {k: pl.from_pandas(v) for k, v in t.items()}
    con.close()
    lp_fn, pl_fn = B.QUERIES[args.q]

    a = med(lambda: lp_fn(t).collect(use_physical_planner=True))
    b = med(lambda: pl_fn(plt))
    print(f"q{args.q} sf={args.sf}: lazy {a:.1f}ms  pl-mem {b:.1f}ms  ({b / a:.2f}x)\n")

    # polars in-memory per-node profile
    print("=== POLARS in-memory per-node (ms) ===")
    captured = {}
    oc = pl.LazyFrame.collect

    def cap(self, *aa, **kk):
        captured["lf"] = self
        return oc(self, *aa, **kk)

    pl.LazyFrame.collect = cap
    pl_fn(plt)
    pl.LazyFrame.collect = oc
    _, prof = captured["lf"].profile()
    rows = [(r["node"], r["end"] - r["start"]) for r in prof.iter_rows(named=True)]
    rows.sort(key=lambda x: -x[1])
    for node, dur in rows[:15]:
        print(f"  {dur / 1000:8.2f} ms  {node}")

    print("\n=== LAZY-PANDAS cProfile (top cumulative, filtered) ===")
    pr = cProfile.Profile()
    pr.enable()
    lp_fn(t).collect(use_physical_planner=True)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(40)
    for line in s.getvalue().splitlines():
        low = line.lower()
        if any(
            k in low
            for k in (
                "merge",
                "join",
                "groupby",
                "group_by",
                "aggregate",
                "take",
                "factor",
                "concat",
                "sort",
                "argsort",
                "to_numpy",
                "to_pandas",
                "pyarrow",
                "_execute",
                "physical",
                "hashtable",
                "combine_chunks",
                "from_pandas",
                "asarray",
                "array(",
                "lazy_",
                "/ops/",
                "gather",
            )
        ):
            print(line)


if __name__ == "__main__":
    main()
