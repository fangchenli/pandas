"""All-22 TPC-H through the owned Rust prototype engine (lazy_engine_rs) vs
Polars 2.0 streaming. Plan compiled + Arrow tables built once per query
(outside the timed region); timed = Rust execute + to_pandas. Validated vs
DuckDB."""

from __future__ import annotations

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import bench_tpch as B
import polars as pl
import translate as R


def geomean(xs):
    xs = [x for x in xs if x and not math.isnan(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("--queries", default=None)
    args = ap.parse_args()
    which = (
        [int(q) for q in args.queries.split(",")] if args.queries else sorted(B.QUERIES)
    )
    con = B.make_duckdb(args.sf)
    tables = B.load_tables(con)
    pl_tables = {n: pl.from_pandas(df) for n, df in tables.items()}
    pl.Config.set_engine_affinity("streaming")
    print(f"Rust prototype vs Polars {pl.__version__} streaming | SF-{args.sf}")
    print(f"{'q':>4} {'valid':>8} {'RS':>8} {'PLs':>8} | {'PLs/RS':>7}")
    rs_r = []
    for n in which:
        lp_fn, pl_fn = B.QUERIES[n]
        ref = con.execute(f"PRAGMA tpch({n})").df()
        try:
            jp, tbls, schema = R.compile_plan(lp_fn(tables))
            res = R.exec_compiled(jp, tbls, schema)
            ok, msg = B.validate(res, ref)
            v = "ok" if ok else f"BAD:{msg[:10]}"
            rs = B.time_query(lambda _: R.exec_compiled(jp, tbls, schema), None)
        except Exception as e:
            v = f"err:{type(e).__name__}"
            rs = float("nan")
        pls = B.time_query(pl_fn, pl_tables)
        r = pls / rs if rs else float("nan")
        rs_r.append(r)
        print(f"q{n:<3} {v:>8} {rs:8.1f} {pls:8.1f} | {r:7.2f}", flush=True)
    print(f"geomean PLs/RS = {geomean(rs_r):.2f}")


if __name__ == "__main__":
    main()
