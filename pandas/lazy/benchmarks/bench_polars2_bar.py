"""Four-way TPC-H snapshot against the Polars 2.0 bar.

Per query (validated vs DuckDB):
  LZ     lazy-pandas Cython/Python engine (collect(use_physical_planner=True))
  DF     lazy LogicalPlan lowered into DataFusion, execute-only (sources
         registered once per query outside the timed region)
  PLs    Polars 2.0 default = streaming engine
  PLm    Polars 2.0 in-memory engine (the pre-2.0 default, i.e. the bar every
         prior campaign number was measured against)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_tpch as B
from datafusion import SessionContext
import polars as pl
import translate_datafusion as T


def geomean(xs):
    xs = [x for x in xs if x and not math.isnan(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--skip-lz", action="store_true")
    args = ap.parse_args()
    which = (
        [int(q) for q in args.queries.split(",")] if args.queries else sorted(B.QUERIES)
    )

    t0 = time.perf_counter()
    con = B.make_duckdb(args.sf)
    tables = B.load_tables(con)
    pl_tables = {n: pl.from_pandas(df) for n, df in tables.items()}
    print(
        f"polars {pl.__version__} | SF-{args.sf} | "
        f"lineitem={len(tables['lineitem']):,} | "
        f"gen {time.perf_counter() - t0:.1f}s | threads={pl.thread_pool_size()}"
    )
    hdr = (
        f"{'q':>4} {'valid':>12} {'LZ':>8} {'DF':>8} {'PLs':>8} {'PLm':>8} | "
        f"{'PLs/LZ':>7} {'PLs/DF':>7} {'PLm/PLs':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    r_lz, r_df, r_mem = [], [], []
    rows = []
    for n in which:
        lp_fn, pl_fn = B.QUERIES[n]
        ref = con.execute(f"PRAGMA tpch({n})").df()
        valid = []
        # --- Polars streaming (2.0 default)
        pl.Config.set_engine_affinity("streaming")
        try:
            ok, _ = B.validate(pl_fn(pl_tables).to_pandas(), ref)
            valid.append("s" if ok else "S!")
            pls = B.time_query(pl_fn, pl_tables)
        except Exception as e:
            valid.append(f"Serr:{type(e).__name__}")
            pls = float("nan")
        # --- Polars in-memory
        pl.Config.set_engine_affinity("in-memory")
        try:
            plm = B.time_query(pl_fn, pl_tables)
        except Exception:
            plm = float("nan")
        pl.Config.set_engine_affinity("streaming")
        # --- lazy engine
        lz = float("nan")
        if not args.skip_lz:
            try:
                res = lp_fn(tables).collect(use_physical_planner=True)
                ok, msg = B.validate(res, ref)
                valid.append("z" if ok else f"Z!{msg[:12]}")
                lz = B.time_query(
                    lambda t: lp_fn(t).collect(use_physical_planner=True), tables
                )
            except Exception as e:
                valid.append(f"Zerr:{type(e).__name__}")
        # --- DataFusion lowering, execute-only
        try:
            ctx = SessionContext()
            sources = T.register_sources(ctx, tables)
            res = T.run(lp_fn(tables), ctx, sources)
            ok, msg = B.validate(res, ref)
            valid.append("d" if ok else f"D!{msg[:12]}")
            df = B.time_query(lambda t: T.run(lp_fn(t), ctx, sources), tables)
        except Exception as e:
            valid.append(f"Derr:{type(e).__name__}")
            df = float("nan")
        a = pls / lz if lz else float("nan")
        b = pls / df if df else float("nan")
        c = plm / pls if pls else float("nan")
        r_lz.append(a)
        r_df.append(b)
        r_mem.append(c)
        rows.append((n, lz, df, pls, plm))
        print(
            f"q{n:<3} {' '.join(valid):>12} {lz:8.1f} {df:8.1f} "
            f"{pls:8.1f} {plm:8.1f} | {a:7.2f} {b:7.2f} {c:8.2f}",
            flush=True,
        )
    print("-" * len(hdr))
    tot = lambda i: sum(r[i] for r in rows if not math.isnan(r[i]))
    print(
        f"geomean  PLs/LZ={geomean(r_lz):.2f}  PLs/DF={geomean(r_df):.2f}  "
        f"PLm/PLs={geomean(r_mem):.2f}"
    )
    print(
        f"totals ms LZ={tot(1):.0f} DF={tot(2):.0f} PLs={tot(3):.0f} PLm={tot(4):.0f}"
    )


if __name__ == "__main__":
    main()
