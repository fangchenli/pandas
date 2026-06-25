#!/usr/bin/env python3
"""Three-way TPC-H comparison to resolve which Polars engine our gap is vs:
  (A) lazy-pandas  (use_physical_planner=True)
  (B) Polars DEFAULT .collect()        -> in-memory engine (materializing)
  (C) Polars engine="streaming"        -> new streaming engine (fused/pipelined)
Run on the join-heavy queries. Reports B/A and C/A ratios (>1 = polars faster).
"""

import argparse
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
    ap.add_argument("-q", "--queries", default="3,5,7,8,9,10,18")
    args = ap.parse_args()
    qs = [int(x) for x in args.queries.split(",")]

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={args.sf})")
    t = B.load_tables(con)
    plt = {k: pl.from_pandas(v) for k, v in t.items()}
    con.close()

    orig = pl.LazyFrame.collect

    def make_streaming():
        def cap(self, *a, **k):
            k.pop("engine", None)
            return orig(self, *a, engine="streaming", **k)

        return cap

    print(f"sf={args.sf}  (>1.0x = polars faster than lazy-pandas)")
    print(
        f"{'q':>3} {'lazy(ms)':>10} {'pl-mem(ms)':>11} {'pl-strm(ms)':>12} "
        f"{'mem/lazy':>9} {'strm/lazy':>10}"
    )
    for q in qs:
        lp_fn, pl_fn = B.QUERIES[q]
        try:
            a = med(lambda: lp_fn(t).collect(use_physical_planner=True))
        except Exception as e:
            print(f"{q:>3}  lazy FAILED: {e!r}")
            continue
        pl.LazyFrame.collect = orig
        b = med(lambda: pl_fn(plt))
        pl.LazyFrame.collect = make_streaming()
        try:
            c = med(lambda: pl_fn(plt))
        except Exception as e:
            c = float("nan")
            print(f"   (q{q} streaming err: {e!r})")
        pl.LazyFrame.collect = orig
        print(f"{q:>3} {a:>10.1f} {b:>11.1f} {c:>12.1f} {b / a:>8.2f}x {c / a:>9.2f}x")


if __name__ == "__main__":
    main()
