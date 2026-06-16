#!/usr/bin/env python3
"""
EXPERIMENT: where does the Polars gap come from, and how much does morsel
streaming (Acero) vs codegen close it?

Runs the SAME three logical shapes — the worst non-limit benchmark
categories — through every execution substrate, with thread count
controlled, so the gap decomposes cleanly:

  raw_pc      whole-array pyarrow.compute chain (full N-row intermediates)
  acero_1     Acero declaration, single thread  (same kernels, morsel-streamed)
  acero_N     Acero declaration, all threads
  fused_1     our Cython fused filter+agg kernel, single thread
  fused_N     our Cython fused kernel, Python-threaded over row slices
  lp          our actual LazyDataFrame.collect(use_physical_planner=True)
  polars_1    Polars, POLARS_MAX_THREADS=1
  polars_N    Polars, all threads

Decomposition the table supports:
  streaming win   = raw_pc / acero_1          (cache-residency, no codegen)
  parallel win    = engine_1 / engine_N       (multi-core)
  per-core gap    = acero_1 / polars_1        (the codegen-addressable residual)
  framework cost  = lp - raw_pc               (our plan/execute overhead)

Each (engine, threads) runs in its own subprocess so POLARS_MAX_THREADS and
the Arrow CPU count are set correctly before import.

Usage::

    python exp_kernel_fusion.py                     # full matrix, 1M + 10M
    python exp_kernel_fusion.py --worker polars 1   # internal: one config
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

SIZES = [1_000_000, 10_000_000]
OPS = ["proj", "filt_sum", "filt6"]
REPS = 7
WARMUP = 2
NCPU = os.cpu_count() or 8


def gen(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "a": rng.standard_normal(n),  # ~half > 0
        "b": rng.standard_normal(n),
        "i": rng.integers(0, 1000, n).astype(np.int64),
    }


def _best(fn) -> float:
    for _ in range(WARMUP):
        fn()
    best = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


# --------------------------------------------------------------------------
# raw whole-array pyarrow.compute
# --------------------------------------------------------------------------
def run_pc(data, op):
    import pyarrow as pa
    import pyarrow.compute as pc

    a, b, i = pa.array(data["a"]), pa.array(data["b"]), pa.array(data["i"])
    if op == "proj":
        return lambda: pc.add(a, b)
    if op == "filt_sum":
        return lambda: pc.sum(pc.filter(a, pc.greater(a, 0.0))).as_py()
    if op == "filt6":

        def f():
            m = pc.and_(pc.and_(pc.greater(a, 0.0), pc.less(i, 500)), pc.less(b, 0.5))
            return pc.sum(m).as_py()  # count of True

        return f
    raise ValueError(op)


# --------------------------------------------------------------------------
# Acero declaration (morsel-streamed); threads via use_threads
# --------------------------------------------------------------------------
def run_acero(data, op, use_threads):
    import pyarrow as pa
    import pyarrow.acero as ac
    import pyarrow.compute as pc

    tbl = pa.table({"a": data["a"], "b": data["b"], "i": data["i"]})

    def decl_for(op):
        src = ac.Declaration("table_source", ac.TableSourceNodeOptions(tbl))
        if op == "proj":
            return ac.Declaration.from_sequence(
                [
                    src,
                    ac.Declaration(
                        "project",
                        ac.ProjectNodeOptions(
                            [pc.add(pc.field("a"), pc.field("b"))], ["c"]
                        ),
                    ),
                ]
            )
        if op == "filt_sum":
            return ac.Declaration.from_sequence(
                [
                    src,
                    ac.Declaration("filter", ac.FilterNodeOptions(pc.field("a") > 0.0)),
                    ac.Declaration(
                        "aggregate", ac.AggregateNodeOptions([("a", "sum", None, "s")])
                    ),
                ]
            )
        if op == "filt6":
            pred = (pc.field("a") > 0.0) & (pc.field("i") < 500) & (pc.field("b") < 0.5)
            return ac.Declaration.from_sequence(
                [
                    src,
                    ac.Declaration("filter", ac.FilterNodeOptions(pred)),
                    ac.Declaration(
                        "aggregate",
                        ac.AggregateNodeOptions([("a", "count", None, "n")]),
                    ),
                ]
            )
        raise ValueError(op)

    d = decl_for(op)
    return lambda: d.to_table(use_threads=use_threads)


# --------------------------------------------------------------------------
# our Cython fused filter+aggregate kernel; threaded over row slices
# --------------------------------------------------------------------------
def run_fused(data, op, nthreads):
    from concurrent.futures import ThreadPoolExecutor

    from pandas._libs.lazy_fused_agg import fused_filter_aggs

    a = np.ascontiguousarray(data["a"])
    b = np.ascontiguousarray(data["b"])
    i = np.ascontiguousarray(data["i"])
    n = a.shape[0]
    NEG = np.float64(-np.inf)
    POS = np.float64(np.inf)
    eps0 = np.nextafter(0.0, np.inf)  # strict >0
    I64MIN = np.iinfo(np.int64).min

    if op == "filt_sum":
        i64_cols, i64_lo, i64_hi = [], np.empty(0, np.int64), np.empty(0, np.int64)
        f64_cols = [a]
        f64_lo = np.array([eps0])
        f64_hi = np.array([POS])
        agg_kinds = np.array([0], np.int64)  # sum(a)
        agg_a, agg_b = [a], [None]
    elif op == "filt6":
        i64_cols = [i]
        i64_lo = np.array([I64MIN], np.int64)
        i64_hi = np.array([499], np.int64)
        f64_cols = [a, b]
        f64_lo = np.array([eps0, NEG])
        f64_hi = np.array([POS, np.nextafter(0.5, NEG)])  # strict b<0.5
        agg_kinds = np.array([2], np.int64)  # count
        agg_a, agg_b = [a], [None]
    else:
        return None  # fused does aggregation, not projection

    def one(start, end):
        out, kept = fused_filter_aggs(
            i64_cols,
            i64_lo,
            i64_hi,
            f64_cols,
            f64_lo,
            f64_hi,
            agg_kinds,
            agg_a,
            agg_b,
            start,
            end,
        )
        return out, kept

    if nthreads == 1:
        return lambda: one(0, n)

    bounds = [(k * n // nthreads, (k + 1) * n // nthreads) for k in range(nthreads)]
    ex = ThreadPoolExecutor(max_workers=nthreads)

    def f():
        res = list(ex.map(lambda se: one(*se), bounds))
        total = np.sum([r[0] for r in res], axis=0)
        return total

    return f


# --------------------------------------------------------------------------
# our actual LazyDataFrame path (framework overhead included)
# --------------------------------------------------------------------------
def run_lp(data, op):
    import pandas as pd
    from pandas.lazy import col

    pdf = pd.DataFrame(data)
    if op == "proj":
        return lambda: pdf.select((col("a") + col("b")).alias("c")).collect(
            use_physical_planner=True
        )
    if op == "filt_sum":
        return lambda: (
            pdf.select()
            .filter(col("a") > 0)
            .select(col("a").sum().alias("s"))
            .collect(use_physical_planner=True)
        )
    if op == "filt6":
        return lambda: (
            pdf.select()
            .filter((col("a") > 0) & (col("i") < 500) & (col("b") < 0.5))
            .select(col("a").count().alias("n"))
            .collect(use_physical_planner=True)
        )
    raise ValueError(op)


# --------------------------------------------------------------------------
# Polars; threads set by POLARS_MAX_THREADS env before import
# --------------------------------------------------------------------------
def run_polars(data, op):
    import polars as pl

    df = pl.DataFrame(data)
    if op == "proj":
        return lambda: (
            df.lazy().select((pl.col("a") + pl.col("b")).alias("c")).collect()
        )
    if op == "filt_sum":
        return lambda: (
            df.lazy().filter(pl.col("a") > 0).select(pl.col("a").sum()).collect()
        )
    if op == "filt6":
        return lambda: (
            df.lazy()
            .filter((pl.col("a") > 0) & (pl.col("i") < 500) & (pl.col("b") < 0.5))
            .select(pl.len())
            .collect()
        )
    raise ValueError(op)


def worker(engine: str, threads: int) -> None:
    """Run every op x size for one (engine, threads); print JSON lines."""
    if engine == "acero":
        import pyarrow as pa

        pa.set_cpu_count(threads if threads > 1 else 1)
    for n in SIZES:
        data = gen(n)
        for op in OPS:
            try:
                if engine == "pc":
                    fn = run_pc(data, op)
                elif engine == "acero":
                    fn = run_acero(data, op, use_threads=threads > 1)
                elif engine == "fused":
                    fn = run_fused(data, op, threads)
                elif engine == "lp":
                    fn = run_lp(data, op)
                elif engine == "polars":
                    fn = run_polars(data, op)
                else:
                    raise ValueError(engine)
                if fn is None:
                    continue
                ms = _best(fn)
            except Exception as exc:  # isolate per op; keep the rest of the matrix
                print(
                    f"ERR {engine} x{threads} {op} {n}: "
                    f"{type(exc).__name__}: {str(exc)[:160]}",
                    file=sys.stderr,
                )
                continue
            print(
                "RESULT "
                + json.dumps(
                    {"engine": engine, "threads": threads, "op": op, "n": n, "ms": ms}
                ),
                flush=True,
            )


# config matrix: (engine, threads, env)
def configs():
    return [
        ("pc", 1, {}),
        ("acero", 1, {}),
        ("acero", NCPU, {}),
        ("fused", 1, {}),
        ("fused", NCPU, {}),
        ("lp", 1, {}),
        ("polars", 1, {"POLARS_MAX_THREADS": "1"}),
        ("polars", NCPU, {"POLARS_MAX_THREADS": str(NCPU)}),
    ]


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]))
        return

    rows = []
    for engine, threads, env in configs():
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            engine,
            str(threads),
        ]
        e = dict(os.environ, **env)
        # Ensure the active interpreter's bin (cython etc.) is on PATH so the
        # meson editable rebuild can find its tools in the subprocess.
        bindir = os.path.dirname(sys.executable)
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
            print("\n".join(p.stderr.strip().splitlines()[-8:]))

    def get(engine, threads, op, n):
        for r in rows:
            if (r["engine"], r["threads"], r["op"], r["n"]) == (engine, threads, op, n):
                return r["ms"]
        return None

    print(f"\n=== Kernel-fusion experiment (NCPU={NCPU}, best of {REPS}) ===\n")
    hdr = [
        "op",
        "N",
        "raw_pc",
        "acero_1",
        f"acero_{NCPU}",
        "fused_1",
        f"fused_{NCPU}",
        "lp",
        "polars_1",
        f"polars_{NCPU}",
    ]
    print("  ".join(f"{h:>9}" for h in hdr))
    for n in SIZES:
        for op in OPS:
            cells = [
                op,
                f"{n // 1_000_000}M",
                get("pc", 1, op, n),
                get("acero", 1, op, n),
                get("acero", NCPU, op, n),
                get("fused", 1, op, n),
                get("fused", NCPU, op, n),
                get("lp", 1, op, n),
                get("polars", 1, op, n),
                get("polars", NCPU, op, n),
            ]
            out = [f"{cells[0]:>9}", f"{cells[1]:>9}"]
            out.extend(
                f"{c:9.1f}" if isinstance(c, float) else f"{'-':>9}" for c in cells[2:]
            )
            print("  ".join(out))

    print("\n=== decomposition (x = how many times faster) ===\n")
    print(
        f"{'op':>9} {'N':>4}  {'stream':>8} {'par(ac)':>8} {'par(pl)':>8} "
        f"{'percore':>8} {'fused/pl':>9} {'fwk_ovh':>9}"
    )
    for n in SIZES:
        for op in OPS:
            pc1 = get("pc", 1, op, n)
            ac1 = get("acero", 1, op, n)
            ac_n = get("acero", NCPU, op, n)
            pl1 = get("polars", 1, op, n)
            pl_n = get("polars", NCPU, op, n)
            fu_n = get("fused", NCPU, op, n)
            lp = get("lp", 1, op, n)

            def ratio(x, y):
                return f"{x / y:8.2f}" if x and y else f"{'-':>8}"

            stream = ratio(pc1, ac1)
            par_ac = ratio(ac1, ac_n)
            par_pl = ratio(pl1, pl_n)
            percore = ratio(ac1, pl1)
            fused_pl = f"{fu_n / pl_n:9.2f}" if fu_n and pl_n else f"{'-':>9}"
            fwk = f"{(lp - pc1):8.1f}m" if lp and pc1 else f"{'-':>9}"
            print(
                f"{op:>9} {n // 1_000_000:>3}M  {stream} {par_ac} {par_pl} "
                f"{percore} {fused_pl} {fwk}"
            )

    with open("exp_kernel_fusion_results.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n[wrote exp_kernel_fusion_results.json]")


if __name__ == "__main__":
    main()
