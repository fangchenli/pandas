#!/usr/bin/env python3
"""
EXPERIMENT: where does the join gap (0.30x vs Polars) come from?

Decomposes an inner equi-join across every substrate, thread count
controlled, for two contrasting shapes:

  exploding   left=N, right=N//10, cardinality=N//100  (bench shape; each
              key fans out ~100x10 -> ~100M output rows at N=10M)
  selective   left=N, right=N, unique keys             (1:1, the shape our
              Cython kernel is designed for)

Substrates:
  pd_merge    pandas pd.merge (what our engine falls back to; single-thread)
  cyth_1      our lazy_join CSR kernel, single thread
  cyth_N      our lazy_join CSR kernel, self-threaded (<=8)
  acero_1     pyarrow Table.join (acero), 1 cpu
  acero_N     pyarrow Table.join (acero), all cpus
  lp          our engine .join().collect(use_physical_planner=True)
  polars_1    Polars, POLARS_MAX_THREADS=1
  polars_N    Polars, all threads

Subprocess per (engine, threads) so POLARS_MAX_THREADS / Arrow cpu count
are set before import.

Usage::

    python exp_join_decomp.py
    python exp_join_decomp.py --worker polars 1   # internal: one config
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

SIZES = [1_000_000, 10_000_000]
SHAPES = ["exploding", "selective"]
REPS = 5
WARMUP = 1
NCPU = os.cpu_count() or 8


def gen(n: int, shape: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    if shape == "exploding":
        n_right = n // 10
        card = max(1, n // 100)
        lk = rng.integers(0, card, n).astype(np.int64)
        rk = rng.integers(0, card, n_right).astype(np.int64)
    else:  # selective: 1:1 unique keys, right is a shuffled subset-ish
        lk = np.arange(n, dtype=np.int64)
        rng.shuffle(lk)
        rk = np.arange(n, dtype=np.int64)
        rng.shuffle(rk)
    lv = rng.standard_normal(len(lk))
    rv = rng.standard_normal(len(rk))
    return lk, lv, rk, rv


def _best(fn) -> float:
    for _ in range(WARMUP):
        fn()
    best = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def run_pd_merge(data):
    import pandas as pd

    lk, lv, rk, rv = data
    left = pd.DataFrame({"key": lk, "left_val": lv})
    right = pd.DataFrame({"key": rk, "right_val": rv})
    return lambda: pd.merge(left, right, on="key", how="inner")


def run_cython(data, threaded):
    from pandas._libs.lazy_join import (
        build_join_table_i8,
        probe_count_chunk,
        probe_fill_chunk,
    )

    lk, lv, rk, rv = data
    lk = np.ascontiguousarray(lk)
    rk = np.ascontiguousarray(rk)

    def single():
        slot_key, slot_gid, counts, offsets, group_rows = build_join_table_i8(rk)
        n = len(lk)
        total = probe_count_chunk(lk, 0, n, slot_key, slot_gid, counts)
        ol = np.empty(total, np.int64)
        orr = np.empty(total, np.int64)
        probe_fill_chunk(
            lk, 0, n, slot_key, slot_gid, counts, offsets, group_rows, ol, orr, 0
        )
        # gather payloads (1 col each side) — the real join cost includes this
        return lv[ol], rv[orr]

    if not threaded:
        return single

    from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

    def threaded_run():
        res = inner_join_indexers_i8(lk, rk, max_hit_fraction=None)
        ol, orr = res
        return lv[ol], rv[orr]

    return threaded_run


def run_acero(data, ncpu):
    import pyarrow as pa

    pa.set_cpu_count(ncpu)
    lk, lv, rk, rv = data
    left = pa.table({"key": lk, "left_val": lv})
    right = pa.table({"key": rk, "right_val": rv})
    return lambda: left.join(right, keys="key", join_type="inner")


def run_lp(data):
    import pandas as pd
    from pandas.lazy import col  # noqa: F401

    lk, lv, rk, rv = data
    left = pd.DataFrame({"key": lk, "left_val": lv})
    right = pd.DataFrame({"key": rk, "right_val": rv})
    return lambda: (
        left.select()
        .join(right.select(), on="key", how="inner")
        .collect(use_physical_planner=True)
    )


def run_polars(data):
    import polars as pl

    lk, lv, rk, rv = data
    left = pl.DataFrame({"key": lk, "left_val": lv})
    right = pl.DataFrame({"key": rk, "right_val": rv})
    return lambda: left.lazy().join(right.lazy(), on="key", how="inner").collect()


def worker(engine: str, threads: int) -> None:
    for n in SIZES:
        for shape in SHAPES:
            data = gen(n, shape)
            try:
                if engine == "pd_merge":
                    fn = run_pd_merge(data)
                elif engine == "cython":
                    fn = run_cython(data, threaded=threads > 1)
                elif engine == "acero":
                    fn = run_acero(data, threads if threads > 1 else 1)
                elif engine == "lp":
                    fn = run_lp(data)
                elif engine == "polars":
                    fn = run_polars(data)
                else:
                    raise ValueError(engine)
                ms = _best(fn)
            except Exception as exc:
                print(
                    f"ERR {engine} x{threads} {shape} {n}: "
                    f"{type(exc).__name__}: {str(exc)[:160]}",
                    file=sys.stderr,
                )
                continue
            print(
                "RESULT "
                + json.dumps(
                    {
                        "engine": engine,
                        "threads": threads,
                        "shape": shape,
                        "n": n,
                        "ms": ms,
                    }
                ),
                flush=True,
            )


def configs():
    return [
        ("pd_merge", 1, {}),
        ("cython", 1, {}),
        ("cython", NCPU, {}),
        ("acero", 1, {}),
        ("acero", NCPU, {}),
        ("lp", 1, {}),
        ("polars", 1, {"POLARS_MAX_THREADS": "1"}),
        ("polars", NCPU, {"POLARS_MAX_THREADS": str(NCPU)}),
    ]


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]))
        return

    rows = []
    bindir = os.path.dirname(sys.executable)
    for engine, threads, env in configs():
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            engine,
            str(threads),
        ]
        e = dict(os.environ, **env)
        e["PATH"] = bindir + os.pathsep + e.get("PATH", "")
        print(f"[running {engine} x{threads}] ...", flush=True)
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=e,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False,
        )
        got = 0
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                rows.append(json.loads(ln[7:]))
                got += 1
        if got == 0:
            print("  !! no results; stderr tail:")
            print("\n".join(p.stderr.strip().splitlines()[-6:]))

    def get(engine, threads, shape, n):
        for r in rows:
            if (r["engine"], r["threads"], r["shape"], r["n"]) == (
                engine,
                threads,
                shape,
                n,
            ):
                return r["ms"]
        return None

    for shape in SHAPES:
        print(f"\n=== inner join [{shape}]  (ms, best of {REPS}, NCPU={NCPU}) ===\n")
        hdr = [
            "N",
            "pd_merge",
            "cyth_1",
            f"cyth_{NCPU}",
            "acero_1",
            f"acero_{NCPU}",
            "lp",
            "polars_1",
            f"polars_{NCPU}",
        ]
        print("  ".join(f"{h:>9}" for h in hdr))
        for n in SIZES:
            cells = [
                f"{n // 1_000_000}M",
                get("pd_merge", 1, shape, n),
                get("cython", 1, shape, n),
                get("cython", NCPU, shape, n),
                get("acero", 1, shape, n),
                get("acero", NCPU, shape, n),
                get("lp", 1, shape, n),
                get("polars", 1, shape, n),
                get("polars", NCPU, shape, n),
            ]
            out = [f"{cells[0]:>9}"]
            out.extend(
                f"{c:9.1f}" if isinstance(c, float) else f"{'-':>9}" for c in cells[1:]
            )
            print("  ".join(out))
        # decomposition
        print(f"\n  decomposition [{shape}] (x = times faster):")
        for n in SIZES:
            pm = get("pd_merge", 1, shape, n)
            cy_n = get("cython", NCPU, shape, n)
            pl1 = get("polars", 1, shape, n)
            pl_n = get("polars", NCPU, shape, n)
            lp = get("lp", 1, shape, n)

            def r(x, y):
                return f"{x / y:.2f}x" if x and y else "-"

            print(
                f"    {n // 1_000_000:>3}M  polars_par={r(pl1, pl_n):>7}  "
                f"lp/polars={r(pl_n, lp):>7}  bestkernel/polars={r(pl_n, cy_n):>7}  "
                f"pdmerge/polars={r(pl_n, pm):>7}"
            )

    with open("exp_join_decomp_results.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n[wrote exp_join_decomp_results.json]")


if __name__ == "__main__":
    main()
