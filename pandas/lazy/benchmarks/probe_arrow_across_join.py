#!/usr/bin/env python3
"""
PROBE: does keeping the pipeline in Arrow across the join (Acero end-to-end,
converting only the small aggregate back to pandas) beat our current
pd.merge/NumPy path on a real join->groupby->topn TPC-H query (q3)?

Thesis (from exp_join_decomp.py): Acero's join is fast but its win dies at the
Arrow->NumPy round-trip on the big intermediate. If the intermediate is
consumed by a reduction and never leaves Arrow, the win should land.
"""

from __future__ import annotations

import time

import bench_tpch as bt
import polars as pl
import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.compute as pc

import pandas as pd

CUT = pd.Timestamp("1995-03-15")


def best(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    b = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1000, r


def acero_e2e(tabs):
    """q3 with all joins+filters+aggregate in Acero; only the small grouped
    result crosses back to pandas for the sort+limit."""
    cut = pa.scalar(CUT.to_datetime64(), type=pa.timestamp("us"))

    def src(t):
        return ac.Declaration("table_source", ac.TableSourceNodeOptions(t))

    cust = ac.Declaration(
        "filter",
        ac.FilterNodeOptions(pc.equal(pc.field("c_mktsegment"), "BUILDING")),
        inputs=[src(tabs["customer"])],
    )
    cust = ac.Declaration(
        "project",
        ac.ProjectNodeOptions([pc.field("c_custkey")], ["c_custkey"]),
        inputs=[cust],
    )
    orders = ac.Declaration(
        "filter",
        ac.FilterNodeOptions(pc.less(pc.field("o_orderdate"), cut)),
        inputs=[src(tabs["orders"])],
    )
    orders = ac.Declaration(
        "project",
        ac.ProjectNodeOptions(
            [
                pc.field("o_custkey"),
                pc.field("o_orderkey"),
                pc.field("o_orderdate"),
                pc.field("o_shippriority"),
            ],
            ["o_custkey", "o_orderkey", "o_orderdate", "o_shippriority"],
        ),
        inputs=[orders],
    )
    li = ac.Declaration(
        "filter",
        ac.FilterNodeOptions(pc.greater(pc.field("l_shipdate"), cut)),
        inputs=[src(tabs["lineitem"])],
    )
    li = ac.Declaration(
        "project",
        ac.ProjectNodeOptions(
            [
                pc.field("l_orderkey"),
                pc.field("l_extendedprice") * (pc.scalar(1.0) - pc.field("l_discount")),
            ],
            ["l_orderkey", "revpart"],
        ),
        inputs=[li],
    )
    j1 = ac.Declaration(
        "hashjoin",
        ac.HashJoinNodeOptions(
            "inner", left_keys=["c_custkey"], right_keys=["o_custkey"]
        ),
        inputs=[cust, orders],
    )
    j2 = ac.Declaration(
        "hashjoin",
        ac.HashJoinNodeOptions(
            "inner", left_keys=["o_orderkey"], right_keys=["l_orderkey"]
        ),
        inputs=[j1, li],
    )
    agg = ac.Declaration(
        "aggregate",
        ac.AggregateNodeOptions(
            [("revpart", "hash_sum", None, "revenue")],
            keys=["l_orderkey", "o_orderdate", "o_shippriority"],
        ),
        inputs=[j2],
    )
    if globals().get("_JOINS_ONLY"):
        return j2.to_table(use_threads=True)
    grouped = agg.to_table(use_threads=True)  # Arrow throughout
    # only the (small) grouped result crosses back to pandas
    df = grouped.to_pandas()
    return df.sort_values(["revenue", "o_orderdate"], ascending=[False, True]).head(10)


def main():
    import sys

    sf = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(f"[gen SF-{sf:g} ...]", flush=True)
    con = bt.make_duckdb(sf)
    # q3 touches only these three tables; load just them to bound memory.
    needed = ("customer", "orders", "lineitem")
    all_tabs = bt.load_tables(con)
    tables = {n: all_tabs[n] for n in needed}
    atabs = {n: pa.Table.from_pandas(tables[n], preserve_index=False) for n in needed}
    ptabs = {n: pl.from_pandas(tables[n]) for n in needed}

    # 1) our engine (pd.merge / numpy). bt._lp wraps each DataFrame as lazy.
    def lp_run():
        return bt.lp_q3(dict(tables)).collect(use_physical_planner=True)

    t_lp, r_lp = best(lp_run)
    t_ac, r_ac = best(lambda: acero_e2e(atabs))
    globals()["_JOINS_ONLY"] = True
    t_acj, _ = best(lambda: acero_e2e(atabs))
    globals()["_JOINS_ONLY"] = False
    t_pl, r_pl = best(lambda: bt.pl_q3(ptabs))

    # validate vs duckdb reference
    con.execute("LOAD tpch")
    ref = con.execute("PRAGMA tpch(3)").df()
    ok_lp, _ = bt.validate(r_lp, ref)
    ok_ac, _ = bt.validate(r_ac.reset_index(drop=True), ref)

    print(f"\nq3 SF-{sf:g}  (validated vs DuckDB)")
    print(f"  lp (engine, pd.merge/numpy) : {t_lp:8.1f} ms   valid={ok_lp}")
    print(f"  acero end-to-end (arrow)    : {t_ac:8.1f} ms   valid={ok_ac}")
    print(f"  acero joins only (no agg)   : {t_acj:8.1f} ms")
    print(f"  polars                      : {t_pl:8.1f} ms")
    print(
        f"\n  acero/lp = {t_lp / t_ac:.2f}x   polars/lp = {t_pl / t_lp:.2f}x"
        f"   acero-agg-share = {(t_ac - t_acj) / t_ac:.0%}"
    )


if __name__ == "__main__":
    main()
