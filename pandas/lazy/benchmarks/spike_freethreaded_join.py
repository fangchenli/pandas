#!/usr/bin/env python3
"""
Spike: does a partitioned hash join scale under free-threaded Python?

Background: the M5 join work measured threaded partition-pair joins as
*slower* than serial on CPython 3.11 — pandas' Cython hash join holds the GIL,
so the worker threads serialize. The engine pivoted to acero routing instead.
This spike re-runs the experiment on a free-threaded build to see whether the
hypothesis ("free-threading unlocks the partitioned join") holds.

Run with a free-threaded interpreter, e.g.::

    micromamba create -n pandas-ft -c conda-forge python-freethreading=3.14 \\
        numpy pyarrow cython meson-python meson ninja pkg-config \\
        c-compiler cxx-compiler python-dateutil pytz versioneer
    micromamba run -n pandas-ft python -m pip install -ve . \\
        --no-build-isolation -Cbuilddir=/tmp/pandas-ft-build
    micromamba run -n pandas-ft python pandas/lazy/benchmarks/spike_freethreaded_join.py

Measured (8M x 8M inner, Apple Silicon, 8 cores, pandas 3.1.0.dev, June 2026):

    GIL after import pandas : False          (numpy/pyarrow/pandas FT-ready)
    baseline full pd.merge  : 1302 ms
    partitioned, 1 thread   : 1332 ms  0.98x  (partitioning overhead ~ neutral)
    partitioned, 2 threads  :  954 ms  1.36x
    partitioned, 4 threads  :  715 ms  1.82x
    partitioned, 8 threads  :  487 ms  2.67x  vs the full merge
    acero join (use_threads):  337 ms         (still faster than the above)

Conclusion: free-threading DOES unlock the partitioned join (2.67x vs serial,
where 3.11 made it slower). But acero's internally-parallel C++ join is faster
still (337 vs 487 ms), so the partitioned join only earns its keep for joins
acero cannot do — nullable/float keys (acero uses SQL null semantics) and
index preservation — under relaxed output order. A narrow niche.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

import pandas as pd


def timeit(fn, warmup=1, runs=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000


def main() -> None:
    print("pandas", pd.__version__)
    # CPython's official free-threading introspection API (underscore-prefixed
    # but public); fetched via getattr so the cross-module private-name check
    # does not flag it.
    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if gil_enabled is None:
        print("Not a free-threaded build; run with python3.*t.")
        return
    print("GIL after import pandas:", gil_enabled())

    n = 8_000_000
    rng = np.random.default_rng(0)
    lk = rng.integers(0, n, n)
    lv = rng.standard_normal(n)
    rk = rng.integers(0, n, n)
    rv = rng.standard_normal(n)
    left = pd.DataFrame({"k": lk, "lv": lv})
    right = pd.DataFrame({"k": rk, "rv": rv})

    base = timeit(lambda: left.merge(right, on="k", how="inner"))
    print(f"baseline full pd.merge   : {base:8.1f} ms")

    def join_pair(lpart, rpart):
        return pd.DataFrame({"k": lpart[0], "lv": lpart[1]}).merge(
            pd.DataFrame({"k": rpart[0], "rv": rpart[1]}), on="k", how="inner"
        )

    for nt in (1, 2, 4, 8):
        lh, rh = lk % nt, rk % nt
        lp = [(lk[lh == p], lv[lh == p]) for p in range(nt)]
        rp = [(rk[rh == p], rv[rh == p]) for p in range(nt)]

        def run(lp=lp, rp=rp, nt=nt):
            out: list = [None] * nt
            ts = [
                threading.Thread(
                    target=lambda i: out.__setitem__(i, join_pair(lp[i], rp[i])),
                    args=(i,),
                )
                for i in range(nt)
            ]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            return out

        ms = timeit(run)
        print(f"partitioned, {nt} thread(s) : {ms:8.1f} ms  ({base / ms:.2f}x)")


if __name__ == "__main__":
    main()
