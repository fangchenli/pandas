"""Perf snapshot of the DataFusion lowering vs Polars (+ the lazy engine) on the
22 TPC-H queries.

Two timing modes (both best-of-3, warm):

* default — full ``translate_datafusion.run`` per call: a fresh SessionContext
  that *re-registers* every source frame each run. This charges the Arrow
  registration boundary on every query; Polars, timed on pre-converted frames,
  pays it once. So the default ratio is a cold/total number, not execution.
* ``--execute-only`` — register all sources ONCE and reuse the ctx (via
  ``run(ldf, ctx, sources)``), so only lowering+execution is timed. This is the
  fair comparison; at SF-1 the geomean lands ~parity with Polars, i.e. the
  default-mode gap is the registration boundary, not the engine. Caveat: a
  reused ctx accumulates the materialized-subplan tables of shared-subplan
  queries (q2/q11/q15/q21) across the repeated timing runs, biasing those slow.

Correctness of the lowering is checked separately by ``validate_datafusion.py``.

    PATH=<pandas-dev>/bin:$PATH python bench_datafusion.py --sf 1
    PATH=<pandas-dev>/bin:$PATH python bench_datafusion.py --sf 1 --execute-only
"""

from __future__ import annotations

import argparse
import math

import bench_tpch as B
from datafusion import SessionContext
import translate_datafusion as T


def _geomean(xs: list) -> float:
    xs = [x for x in xs if x and not math.isnan(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sf", type=float, default=1.0, help="TPC-H scale factor")
    ap.add_argument(
        "--execute-only",
        action="store_true",
        help="pre-register sources once, reuse the ctx (fair; excludes registration)",
    )
    args = ap.parse_args()

    import polars as pl

    con = B.make_duckdb(args.sf)
    tables = B.load_tables(con)
    pl_tables = {n: pl.from_pandas(df) for n, df in tables.items()}

    ctx = sources = None
    if args.execute_only:
        ctx = SessionContext()
        sources = T.register_sources(ctx, tables)

    mode = (
        "execute-only (sources pre-registered)"
        if args.execute_only
        else "full run (registration each call)"
    )
    print(f"DataFusion lowering vs Polars | SF-{args.sf} | {mode}")
    print(f"lineitem={len(tables['lineitem']):,} rows\n")
    print(f"{'q':>4} {'DF(ms)':>9} {'PL(ms)':>9} {'LZ(ms)':>9} {'PL/DF':>7}")
    print("-" * 46)

    ratios = []
    for n in sorted(B.QUERIES):
        lp_fn, pl_fn = B.QUERIES[n]

        def df_call(t, lp_fn=lp_fn):
            return T.run(lp_fn(t), ctx, sources)

        try:
            df_ms = B.time_query(df_call, tables)
        except Exception as e:
            print(f"q{n:<3} {'ERR':>9}  {type(e).__name__}: {str(e)[:30]}")
            continue
        try:
            pl_ms = B.time_query(pl_fn, pl_tables)
        except Exception:
            pl_ms = float("nan")
        try:
            lz_ms = B.time_query(
                lambda t, lp_fn=lp_fn: lp_fn(t).collect(use_physical_planner=True),
                tables,
            )
        except Exception:
            lz_ms = float("nan")
        r = pl_ms / df_ms if df_ms else float("nan")
        ratios.append(r)
        print(f"q{n:<3} {df_ms:9.1f} {pl_ms:9.1f} {lz_ms:9.1f} {r:7.2f}")

    print(f"\ngeomean PL/DF = {_geomean(ratios):.2f}  (>1 = DF faster than Polars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
