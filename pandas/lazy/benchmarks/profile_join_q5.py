#!/usr/bin/env python3
"""Side-by-side profile of q5 (6-table join chain -> low-card group -> sort):
Polars per-node profiler vs cProfile of lazy-pandas. Shows directly where each
engine spends time."""

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
    ap.add_argument("-q", type=int, default=5)
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={args.sf})")
    t = B.load_tables(con)
    lp_fn, pl_fn = B.QUERIES[args.q]

    # convert polars tables once (native vs native)
    plt = {k: pl.from_pandas(v) for k, v in t.items()}

    lp_ms = med(lambda: lp_fn(t).collect(use_physical_planner=True))
    pl_ms = med(lambda: pl_fn(plt))
    print(
        f"q{args.q} sf={args.sf}: lazy {lp_ms:.1f} ms   polars {pl_ms:.1f} ms"
        f"   ({pl_ms / lp_ms:.2f}x)"
    )

    # ---- Polars per-node profile ----
    # Rebuild the polars lazy plan without the terminal .collect() so we can
    # call .profile(). pl_q* ends in .collect(); reuse its body via the lazy
    # plan: re-run pl_fn but intercept. Simpler: rebuild inline isn't possible,
    # so call .profile on a lazy version by stripping collect through the frame.
    print("\n=== POLARS per-node profile (us) ===")
    try:
        # pl_fn returns a collected DataFrame; we need the LazyFrame. Rebuild
        # by calling the same ops but ending at the lazy frame: monkeypatch
        # collect to capture self.
        captured = {}
        orig_collect = pl.LazyFrame.collect

        def cap(self, *a, **k):
            captured["lf"] = self
            return orig_collect(self, *a, **k)

        pl.LazyFrame.collect = cap
        pl_fn(plt)
        pl.LazyFrame.collect = orig_collect
        lf = captured["lf"]
        _, prof = lf.profile()
        prof = prof.sort("node")
        rows = []
        for r in prof.iter_rows(named=True):
            dur = r["end"] - r["start"]
            rows.append((r["node"], dur))
        rows.sort(key=lambda x: -x[1])
        for node, dur in rows:
            print(f"  {dur / 1000:8.2f} ms  {node}")
    except Exception as e:
        print("  polars profile failed:", repr(e))

    # ---- lazy-pandas cProfile ----
    print("\n=== LAZY-PANDAS cProfile (top cumulative) ===")
    pr = cProfile.Profile()
    pr.enable()
    lp_fn(t).collect(use_physical_planner=True)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    out = s.getvalue()
    # trim to the meaningful lines
    for line in out.splitlines():
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
                "pa.",
                "pyarrow",
                "_execute",
                "physical",
                "hash",
                "combine_chunks",
            )
        ):
            print(line)


if __name__ == "__main__":
    main()
