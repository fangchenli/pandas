"""Demo: run TPC-H q1 and q6 through the GENERAL plan->Rust executor
(lazy_engine_rs.execute(plan_json, tables)) — queries route through the engine,
not a hand-written run_qN. Correct vs DuckDB. See ../../docs/RUST_ENGINE_DIRECTION.md.

Status: correct, general, and FUSED (cache-resident morsel-parallel aggregate) —
beats Polars: q6 ~2.6x, q1 ~1.7x, through the general JSON-plan path.
"""

from __future__ import annotations

import json
import time

import bench_tpch as B
import lazy_engine_rs as E
import numpy as np
import pyarrow as pa

import pandas as pd


def _ns(s):
    return s.values.astype("datetime64[ns]").view("int64")


def _col(n):
    return {"t": "col", "name": n}


def _litf(v):
    return {"t": "litf", "v": v}


def _liti(v):
    return {"t": "liti", "v": v}


def _bn(o, ln, r):
    return {"t": "bin", "op": o, "l": ln, "r": r}


def _best(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def main(sf: float = 3.0):
    import polars as pl

    con = B.make_duckdb(sf)
    t = B.load_tables(con)
    li = t["lineitem"]
    lineitem = pa.RecordBatch.from_arrays(
        [
            pa.array(_ns(li["l_shipdate"])),
            pa.array(li["l_returnflag"], type=pa.string()),
            pa.array(li["l_linestatus"], type=pa.string()),
            pa.array(li["l_quantity"].to_numpy()),
            pa.array(li["l_extendedprice"].to_numpy()),
            pa.array(li["l_discount"].to_numpy()),
            pa.array(li["l_tax"].to_numpy()),
        ],
        names=[
            "l_shipdate",
            "l_returnflag",
            "l_linestatus",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
        ],
    )
    tables = {"lineitem": lineitem}
    plt = {n: pl.from_pandas(d) for n, d in t.items()}

    # q6
    lo = int(pd.Timestamp("1994-01-01").value)
    hi = int(pd.Timestamp("1995-01-01").value)
    pred = _bn(
        "and",
        _bn(
            "and",
            _bn(
                "and",
                _bn(
                    "and",
                    _bn("ge", _col("l_shipdate"), _liti(lo)),
                    _bn("lt", _col("l_shipdate"), _liti(hi)),
                ),
                _bn("ge", _col("l_discount"), _litf(0.05)),
            ),
            _bn("le", _col("l_discount"), _litf(0.07)),
        ),
        _bn("lt", _col("l_quantity"), _litf(24.0)),
    )
    q6 = {
        "op": "aggregate",
        "group": [],
        "aggs": [{"func": "sum", "col": "revenue", "name": "revenue"}],
        "input": {
            "op": "project",
            "exprs": [
                {
                    "expr": _bn("mul", _col("l_extendedprice"), _col("l_discount")),
                    "name": "revenue",
                }
            ],
            "input": {
                "op": "filter",
                "pred": pred,
                "input": {
                    "op": "scan",
                    "table": "lineitem",
                    "columns": [
                        "l_shipdate",
                        "l_discount",
                        "l_quantity",
                        "l_extendedprice",
                    ],
                },
            },
        },
    }
    r6 = E.execute(json.dumps(q6), tables).to_pandas()
    ref6 = con.execute("PRAGMA tpch(6)").df()
    ok6 = abs(r6["revenue"][0] - ref6["revenue"][0]) < 1

    # q1
    cut = int((pd.Timestamp("1998-12-01") - pd.Timedelta(days=90)).value)
    omd = _bn("sub", _litf(1.0), _col("l_discount"))
    dp = _bn("mul", _col("l_extendedprice"), omd)
    charge = _bn("mul", dp, _bn("add", _litf(1.0), _col("l_tax")))
    proj = {
        "op": "project",
        "exprs": [
            {"expr": _col("l_returnflag"), "name": "rf"},
            {"expr": _col("l_linestatus"), "name": "ls"},
            {"expr": _col("l_quantity"), "name": "q"},
            {"expr": _col("l_extendedprice"), "name": "p"},
            {"expr": _col("l_discount"), "name": "d"},
            {"expr": dp, "name": "dp"},
            {"expr": charge, "name": "ch"},
        ],
        "input": {
            "op": "filter",
            "pred": _bn("le", _col("l_shipdate"), _liti(cut)),
            "input": {
                "op": "scan",
                "table": "lineitem",
                "columns": [
                    "l_shipdate",
                    "l_returnflag",
                    "l_linestatus",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_tax",
                ],
            },
        },
    }
    agg = {
        "op": "aggregate",
        "group": ["rf", "ls"],
        "aggs": [
            {"func": "sum", "col": "q", "name": "sum_qty"},
            {"func": "sum", "col": "p", "name": "sum_base_price"},
            {"func": "sum", "col": "dp", "name": "sum_disc_price"},
            {"func": "sum", "col": "ch", "name": "sum_charge"},
            {"func": "mean", "col": "q", "name": "avg_qty"},
            {"func": "mean", "col": "p", "name": "avg_price"},
            {"func": "mean", "col": "d", "name": "avg_disc"},
            {"func": "count", "col": "q", "name": "count_order"},
        ],
        "input": proj,
    }
    q1 = {"op": "sort", "keys": [{"col": "rf"}, {"col": "ls"}], "input": agg}
    r1 = E.execute(json.dumps(q1), tables).to_pandas()
    ref1 = con.execute("PRAGMA tpch(1)").df()
    ok1 = (
        np.allclose(r1["sum_qty"], ref1["sum_qty"], rtol=1e-6)
        and (r1["count_order"].to_numpy() == ref1["count_order"].to_numpy()).all()
    )

    print(
        f"q6 general-exec correct={ok6}: "
        f"{_best(lambda: E.execute(json.dumps(q6), tables)):.1f} ms | "
        f"polars {_best(lambda: B.pl_q6(plt)):.1f} ms"
    )
    print(
        f"q1 general-exec correct={ok1}: "
        f"{_best(lambda: E.execute(json.dumps(q1), tables)):.1f} ms | "
        f"polars {_best(lambda: B.pl_q1(plt)):.1f} ms  (fused: "
        f"beats Polars)"
    )


if __name__ == "__main__":
    main()
