#!/usr/bin/env python3
"""
TPC-H (PDS-H) benchmark for lazy pandas vs Polars, validated against DuckDB.

TPC-H is the decision-support benchmark Polars markets against pandas/Dask/
Spark/DuckDB (22 queries over an 8-table schema with multi-table joins,
aggregations, sorting, and subqueries). It stress-tests the *whole* engine on
realistic analytical pipelines, unlike the H2O group-by/join microbenchmark.

Data and reference results both come from DuckDB's first-party ``tpch``
extension: ``CALL dbgen(sf=...)`` generates the tables, ``PRAGMA tpch(n)`` is
the authoritative result for query n. Each lazy-pandas query is validated
against that reference, then timed against Polars.

Fairness: each engine is timed on its **native** input doing **only the
query** - lazy pandas on the pandas tables, Polars on Polars tables converted
once up front (not inside the timed loop). Timing Polars' ``from_pandas`` per
run would charge it a pandas->Arrow conversion (~75% of a query's time at
SF-1) that the already-native lazy-pandas path does not pay.

Queries are hand-translated from the TPC-H SQL into the lazy DataFrame API
(reference: pola.rs/polars-benchmark). Implemented queries are registered in
``QUERIES``; the suite grows incrementally.

Usage::

    python pandas/lazy/benchmarks/bench_tpch.py --sf 1
    python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --queries 1,6
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import pandas as pd
from pandas.lazy import (
    col,
    lit,
    when,
)

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import duckdb

TABLES = [
    "lineitem",
    "orders",
    "customer",
    "supplier",
    "part",
    "partsupp",
    "nation",
    "region",
]


# ---------------------------------------------------------------------------
# Data generation (via DuckDB dbgen) + the DuckDB reference connection
# ---------------------------------------------------------------------------
def make_duckdb(sf: float) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    return con


def load_tables(con) -> dict[str, pd.DataFrame]:
    """Export each TPC-H table to a pandas DataFrame.

    Decimal columns are cast to float64 (TPC-H permits float arithmetic; the
    validation uses a tolerance), and date columns to datetime64.
    """
    tables = {}
    for name in TABLES:
        df = con.execute(f"SELECT * FROM {name}").df()
        for c in df.columns:
            if (
                df[c].dtype == object
                and len(df)
                and isinstance(df[c].iloc[0], __import__("decimal").Decimal)
            ):
                df[c] = df[c].astype("float64")
        tables[name] = df
    return tables


# ---------------------------------------------------------------------------
# Queries: QUERIES[n] = (lazy_pandas_fn(tables) -> LazyDataFrame, polars_fn)
# ---------------------------------------------------------------------------
def _lp(df):
    return df.select()


# --- Q1: Pricing Summary Report --------------------------------------------
def lp_q1(t):
    cutoff = pd.Timestamp("1998-12-01") - pd.Timedelta(days=90)
    li = t["lineitem"]
    disc_price = col("l_extendedprice") * (1 - col("l_discount"))
    charge = col("l_extendedprice") * (1 - col("l_discount")) * (1 + col("l_tax"))
    return (
        _lp(li)
        .filter(col("l_shipdate") <= cutoff)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            col("l_quantity").sum().alias("sum_qty"),
            col("l_extendedprice").sum().alias("sum_base_price"),
            disc_price.sum().alias("sum_disc_price"),
            charge.sum().alias("sum_charge"),
            col("l_quantity").mean().alias("avg_qty"),
            col("l_extendedprice").mean().alias("avg_price"),
            col("l_discount").mean().alias("avg_disc"),
            col("l_quantity").count().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )


def pl_q1(t):
    cutoff = pd.Timestamp("1998-12-01") - pd.Timedelta(days=90)
    li = t["lineitem"].lazy()
    return (
        li.filter(pl.col("l_shipdate") <= cutoff)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.sum("l_quantity").alias("sum_qty"),
            pl.sum("l_extendedprice").alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .alias("sum_disc_price"),
            (
                pl.col("l_extendedprice")
                * (1 - pl.col("l_discount"))
                * (1 + pl.col("l_tax"))
            )
            .sum()
            .alias("sum_charge"),
            pl.mean("l_quantity").alias("avg_qty"),
            pl.mean("l_extendedprice").alias("avg_price"),
            pl.mean("l_discount").alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
        .collect()
    )


# --- Q6: Forecasting Revenue Change ----------------------------------------
def lp_q6(t):
    li = t["lineitem"]
    return (
        _lp(li)
        .filter(
            (col("l_shipdate") >= pd.Timestamp("1994-01-01"))
            & (col("l_shipdate") < pd.Timestamp("1995-01-01"))
            & (col("l_discount") >= 0.05)
            & (col("l_discount") <= 0.07)
            & (col("l_quantity") < 24)
        )
        .select((col("l_extendedprice") * col("l_discount")).alias("revenue"))
        .sum()
    )


def pl_q6(t):
    li = t["lineitem"].lazy()
    return (
        li.filter(
            (pl.col("l_shipdate") >= pd.Timestamp("1994-01-01"))
            & (pl.col("l_shipdate") < pd.Timestamp("1995-01-01"))
            & (pl.col("l_discount") >= 0.05)
            & (pl.col("l_discount") <= 0.07)
            & (pl.col("l_quantity") < 24)
        )
        .select((pl.col("l_extendedprice") * pl.col("l_discount")).alias("revenue"))
        .sum()
        .collect()
    )


# --- Q3: Shipping Priority (3-table join) ----------------------------------
def lp_q3(t):
    cut = pd.Timestamp("1995-03-15")
    c = _lp(t["customer"]).filter(col("c_mktsegment") == "BUILDING")
    o = _lp(t["orders"]).filter(col("o_orderdate") < cut)
    li = _lp(t["lineitem"]).filter(col("l_shipdate") > cut)
    return (
        c.join(o, left_on="c_custkey", right_on="o_custkey")
        .join(li, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("l_orderkey", "o_orderdate", "o_shippriority")
        .agg((col("l_extendedprice") * (1 - col("l_discount"))).sum().alias("revenue"))
        .sort("revenue", "o_orderdate", descending=[True, False])
        .limit(10)
    )


def pl_q3(t):
    cut = pd.Timestamp("1995-03-15")
    c = t["customer"].lazy().filter(pl.col("c_mktsegment") == "BUILDING")
    o = t["orders"].lazy().filter(pl.col("o_orderdate") < cut)
    li = t["lineitem"].lazy().filter(pl.col("l_shipdate") > cut)
    return (
        c.join(o, left_on="c_custkey", right_on="o_custkey")
        .join(li, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .alias("revenue")
        )
        .sort(["revenue", "o_orderdate"], descending=[True, False])
        .limit(10)
        .collect()
    )


# --- Q10: Returned Item Reporting (4-table join) ---------------------------
def lp_q10(t):
    lo = pd.Timestamp("1993-10-01")
    hi = pd.Timestamp("1994-01-01")
    c = _lp(t["customer"])
    o = _lp(t["orders"]).filter((col("o_orderdate") >= lo) & (col("o_orderdate") < hi))
    li = _lp(t["lineitem"]).filter(col("l_returnflag") == "R")
    n = _lp(t["nation"])
    return (
        c.join(o, left_on="c_custkey", right_on="o_custkey")
        .join(li, left_on="o_orderkey", right_on="l_orderkey")
        .join(n, left_on="c_nationkey", right_on="n_nationkey")
        .group_by(
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        )
        .agg((col("l_extendedprice") * (1 - col("l_discount"))).sum().alias("revenue"))
        .sort("revenue", descending=True)
        .limit(20)
    )


def pl_q10(t):
    lo = pd.Timestamp("1993-10-01")
    hi = pd.Timestamp("1994-01-01")
    c = t["customer"].lazy()
    o = (
        t["orders"]
        .lazy()
        .filter((pl.col("o_orderdate") >= lo) & (pl.col("o_orderdate") < hi))
    )
    li = t["lineitem"].lazy().filter(pl.col("l_returnflag") == "R")
    n = t["nation"].lazy()
    return (
        c.join(o, left_on="c_custkey", right_on="o_custkey")
        .join(li, left_on="o_orderkey", right_on="l_orderkey")
        .join(n, left_on="c_nationkey", right_on="n_nationkey")
        .group_by(
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        )
        .agg(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .alias("revenue")
        )
        .sort("revenue", descending=True)
        .limit(20)
        .collect()
    )


# --- Q5: Local Supplier Volume (6-table join) ------------------------------
def lp_q5(t):
    lo, hi = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")
    o = _lp(t["orders"]).filter((col("o_orderdate") >= lo) & (col("o_orderdate") < hi))
    region = _lp(t["region"]).filter(col("r_name") == "ASIA")
    return (
        _lp(t["customer"])
        .join(o, left_on="c_custkey", right_on="o_custkey")
        .join(_lp(t["lineitem"]), left_on="o_orderkey", right_on="l_orderkey")
        .join(_lp(t["supplier"]), left_on="l_suppkey", right_on="s_suppkey")
        .filter(col("c_nationkey") == col("s_nationkey"))
        .join(_lp(t["nation"]), left_on="s_nationkey", right_on="n_nationkey")
        .join(region, left_on="n_regionkey", right_on="r_regionkey")
        .group_by("n_name")
        .agg((col("l_extendedprice") * (1 - col("l_discount"))).sum().alias("revenue"))
        .sort("revenue", descending=True)
    )


def pl_q5(t):
    lo, hi = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")
    o = (
        t["orders"]
        .lazy()
        .filter((pl.col("o_orderdate") >= lo) & (pl.col("o_orderdate") < hi))
    )
    region = t["region"].lazy().filter(pl.col("r_name") == "ASIA")
    return (
        t["customer"]
        .lazy()
        .join(o, left_on="c_custkey", right_on="o_custkey")
        .join(
            t["lineitem"].lazy(),
            left_on="o_orderkey",
            right_on="l_orderkey",
        )
        .join(
            t["supplier"].lazy(),
            left_on="l_suppkey",
            right_on="s_suppkey",
        )
        .filter(pl.col("c_nationkey") == pl.col("s_nationkey"))
        .join(
            t["nation"].lazy(),
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
        .join(region, left_on="n_regionkey", right_on="r_regionkey")
        .group_by("n_name")
        .agg(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .alias("revenue")
        )
        .sort("revenue", descending=True)
        .collect()
    )


# --- Q12: Shipping Modes (conditional aggregation) -------------------------
def lp_q12(t):
    lo, hi = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")
    li = _lp(t["lineitem"]).filter(
        col("l_shipmode").isin(["MAIL", "SHIP"])
        & (col("l_commitdate") < col("l_receiptdate"))
        & (col("l_shipdate") < col("l_commitdate"))
        & (col("l_receiptdate") >= lo)
        & (col("l_receiptdate") < hi)
    )
    is_high = col("o_orderpriority").isin(["1-URGENT", "2-HIGH"])
    high = when(is_high).then(1).otherwise(0)
    low = when(is_high).then(0).otherwise(1)
    return (
        _lp(t["orders"])
        .join(li, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("l_shipmode")
        .agg(
            high.sum().alias("high_line_count"),
            low.sum().alias("low_line_count"),
        )
        .sort("l_shipmode")
    )


def pl_q12(t):
    lo, hi = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")
    li = (
        t["lineitem"]
        .lazy()
        .filter(
            pl.col("l_shipmode").is_in(["MAIL", "SHIP"])
            & (pl.col("l_commitdate") < pl.col("l_receiptdate"))
            & (pl.col("l_shipdate") < pl.col("l_commitdate"))
            & (pl.col("l_receiptdate") >= lo)
            & (pl.col("l_receiptdate") < hi)
        )
    )
    is_high = pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])
    return (
        t["orders"]
        .lazy()
        .join(li, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("l_shipmode")
        .agg(
            pl.when(is_high).then(1).otherwise(0).sum().alias("high_line_count"),
            pl.when(is_high).then(0).otherwise(1).sum().alias("low_line_count"),
        )
        .sort("l_shipmode")
        .collect()
    )


# --- Q4: Order Priority Checking (semi-join via distinct + inner join) ------
def lp_q4(t):
    lo, hi = pd.Timestamp("1993-07-01"), pd.Timestamp("1993-10-01")
    matching = (
        _lp(t["lineitem"])
        .filter(col("l_commitdate") < col("l_receiptdate"))
        .select(col("l_orderkey"))
        .distinct()
    )
    return (
        _lp(t["orders"])
        .filter((col("o_orderdate") >= lo) & (col("o_orderdate") < hi))
        .join(matching, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("o_orderpriority")
        .agg(col("o_orderkey").count().alias("order_count"))
        .sort("o_orderpriority")
    )


def pl_q4(t):
    lo, hi = pd.Timestamp("1993-07-01"), pd.Timestamp("1993-10-01")
    matching = (
        t["lineitem"]
        .lazy()
        .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
        .select("l_orderkey")
    )
    return (
        t["orders"]
        .lazy()
        .filter((pl.col("o_orderdate") >= lo) & (pl.col("o_orderdate") < hi))
        .join(matching, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .group_by("o_orderpriority")
        .agg(pl.len().alias("order_count"))
        .sort("o_orderpriority")
        .collect()
    )


# --- Q14: Promotion Effect (global ratio of conditional sums) --------------
def lp_q14(t):
    lo, hi = pd.Timestamp("1995-09-01"), pd.Timestamp("1995-10-01")
    li = _lp(t["lineitem"]).filter((col("l_shipdate") >= lo) & (col("l_shipdate") < hi))
    disc = col("l_extendedprice") * (1 - col("l_discount"))
    promo = when(col("p_type").str.startswith("PROMO")).then(disc).otherwise(0.0)
    return (
        li.join(_lp(t["part"]), left_on="l_partkey", right_on="p_partkey")
        .with_columns(lit(1).alias("__g"))
        .group_by("__g")
        .agg((100.0 * promo.sum() / disc.sum()).alias("promo_revenue"))
        .select(col("promo_revenue"))
    )


def pl_q14(t):
    lo, hi = pd.Timestamp("1995-09-01"), pd.Timestamp("1995-10-01")
    li = (
        t["lineitem"]
        .lazy()
        .filter((pl.col("l_shipdate") >= lo) & (pl.col("l_shipdate") < hi))
    )
    disc = pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
    promo = pl.when(pl.col("p_type").str.starts_with("PROMO")).then(disc).otherwise(0.0)
    return (
        li.join(t["part"].lazy(), left_on="l_partkey", right_on="p_partkey")
        .select((100.0 * promo.sum() / disc.sum()).alias("promo_revenue"))
        .collect()
    )


QUERIES = {
    1: (lp_q1, pl_q1),
    3: (lp_q3, pl_q3),
    4: (lp_q4, pl_q4),
    5: (lp_q5, pl_q5),
    6: (lp_q6, pl_q6),
    10: (lp_q10, pl_q10),
    12: (lp_q12, pl_q12),
    14: (lp_q14, pl_q14),
}


# ---------------------------------------------------------------------------
# Validation + timing
# ---------------------------------------------------------------------------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows and columns so two results compare regardless of order."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.reindex(sorted(df.columns), axis=1)
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def validate(lazy_df: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[bool, str]:
    a, b = _normalize(lazy_df), _normalize(ref_df)
    if a.shape != b.shape:
        return False, f"shape {a.shape} != ref {b.shape}"
    for col_a, col_b in zip(a.columns, b.columns, strict=True):
        sa, sb = a[col_a], b[col_b]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            if not np.allclose(
                sa.to_numpy(dtype="float64"),
                sb.to_numpy(dtype="float64"),
                rtol=1e-3,
                atol=1e-2,
            ):
                return False, f"numeric mismatch in {col_a}"
        elif not (
            sa.astype(str)
            .reset_index(drop=True)
            .equals(sb.astype(str).reset_index(drop=True))
        ):
            return False, f"value mismatch in {col_a}"
    return True, "ok"


def time_query(fn, arg, warm=1, runs=3):
    for _ in range(warm):
        fn(arg)
    best = float("inf")
    for _ in range(runs):
        s = time.perf_counter()
        fn(arg)
        best = min(best, (time.perf_counter() - s) * 1000)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("--queries", default=None, help="comma list, e.g. 1,6")
    args = ap.parse_args()

    which = (
        [int(q) for q in args.queries.split(",")] if args.queries else sorted(QUERIES)
    )

    print("TPC-H (PDS-H) — lazy pandas vs Polars, validated vs DuckDB")
    print(f"scale factor: {args.sf}  polars={'yes' if HAS_POLARS else 'NO'}")
    t0 = time.perf_counter()
    con = make_duckdb(args.sf)
    tables = load_tables(con)
    # Pre-convert the Polars tables ONCE, outside the timed query: each engine
    # is timed on its native input doing only the query (lazy pandas on the
    # pandas frames, Polars on Polars frames). Timing Polars' from_pandas per
    # run would charge it a pandas->Arrow conversion the native lazy path never
    # pays - not a fair engine comparison.
    pl_tables = (
        {name: pl.from_pandas(df) for name, df in tables.items()}
        if HAS_POLARS
        else None
    )
    print(
        f"[generated SF-{args.sf} in {time.perf_counter() - t0:.1f}s; "
        f"lineitem={len(tables['lineitem']):,} rows]\n"
    )

    print(f"{'query':>6} {'valid':>7} {'LP (ms)':>10} {'PL (ms)':>10} {'PL/LP':>7}")
    print("-" * 46)
    for n in which:
        if n not in QUERIES:
            print(f"{'q' + str(n):>6}   not implemented")
            continue
        lp_fn, pl_fn = QUERIES[n]
        try:
            lp_res = lp_fn(tables).collect(use_physical_planner=True)
            ref = con.execute(f"PRAGMA tpch({n})").df()
            ok, msg = validate(lp_res, ref)
            lp_ms = time_query(
                lambda t: lp_fn(t).collect(use_physical_planner=True), tables
            )
            pl_ms = time_query(pl_fn, pl_tables) if HAS_POLARS else float("nan")
            ratio = pl_ms / lp_ms if lp_ms else float("nan")
            flag = "OK" if ok else f"FAIL:{msg}"
            print(
                f"{'q' + str(n):>6} {flag:>7} {lp_ms:10.1f} {pl_ms:10.1f} {ratio:7.2f}"
            )
        except Exception as e:
            print(f"{'q' + str(n):>6}   ERROR: {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
