#!/usr/bin/env python3
"""
H2O.ai db-benchmark ("Database-like ops benchmark") for lazy pandas vs Polars.

The de-facto cross-engine DataFrame benchmark (maintained by DuckDB Labs:
https://github.com/duckdblabs/db-benchmark). It measures the two primitives a
DataFrame engine lives or dies on - group-by and join - with no SQL parser
needed, so it maps directly onto our lazy API. See docs/BENCHMARK_SUITES.md.

This is a faithful Python port of the h2oai data schema and the 10 group-by +
5 join queries, run against both lazy pandas and Polars at a configurable
scale, replicating the benchmark's cold/hot ("run twice") convention.

Queries our engine does not yet support (grouped median, aggregation
arithmetic, grouped top-k + explode, correlation) are reported as
``unsupported`` rather than hidden - the honest competitive picture plus a
precise gap list.

Usage::

    python pandas/lazy/benchmarks/bench_h2o.py            # 1e7 (~0.5 GB)
    python pandas/lazy/benchmarks/bench_h2o.py --scale 1e8
    python pandas/lazy/benchmarks/bench_h2o.py --task groupby
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import pandas as pd
from pandas.lazy import col

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# ---------------------------------------------------------------------------
# Data generation (h2oai schema)
# ---------------------------------------------------------------------------
def _labels(codes: np.ndarray, prefix: str) -> np.ndarray:
    """Map integer codes (1..k) to string labels via a vectorized take."""
    k = int(codes.max())
    levels = np.array([f"{prefix}{i}" for i in range(k + 1)], dtype=object)
    return levels[codes]


def generate_groupby(n: int, k: int, seed: int = 0) -> pd.DataFrame:
    """G1_<n>_<k> group-by dataset.

    id1, id2 are k-level string keys; id3 is a high-cardinality (n/k) string
    key; id4, id5 are k-level integer keys; id6 is a high-cardinality integer
    key; v1, v2 are small integers; v3 is a float. Matches the h2oai schema.
    """
    rng = np.random.default_rng(seed)
    nk = max(n // k, 1)
    return pd.DataFrame(
        {
            "id1": _labels(rng.integers(1, k + 1, n), "id"),
            "id2": _labels(rng.integers(1, k + 1, n), "id"),
            "id3": _labels(rng.integers(1, nk + 1, n), "id"),
            "id4": rng.integers(1, k + 1, n),
            "id5": rng.integers(1, k + 1, n),
            "id6": rng.integers(1, nk + 1, n),
            "v1": rng.integers(1, 6, n),
            "v2": rng.integers(1, 16, n),
            "v3": np.round(rng.uniform(0, 100, n), 6),
        }
    )


def generate_join(n: int, seed: int = 0) -> dict[str, pd.DataFrame]:
    """LHS table x plus small/medium/big right tables (h2oai join schema).

    Right-table ids are 1..size (each once), and x's ids are sampled from the
    same range, so every join is referentially 1:1. id4/id5/id6 are the string
    forms of id1/id2/id3 (string-key join variants).
    """
    rng = np.random.default_rng(seed)
    n_small = max(n // 1_000_000, 1)
    n_med = max(n // 1_000, 1)
    n_big = n

    x = pd.DataFrame(
        {
            "id1": rng.integers(1, n_small + 1, n),
            "id2": rng.integers(1, n_med + 1, n),
            "id3": rng.integers(1, n_big + 1, n),
            "v1": np.round(rng.uniform(0, 100, n), 6),
        }
    )
    x["id4"] = _labels(x["id1"].to_numpy(), "id")
    x["id5"] = _labels(x["id2"].to_numpy(), "id")

    small = pd.DataFrame({"id1": np.arange(1, n_small + 1)})
    small["id4"] = _labels(small["id1"].to_numpy(), "id")
    small["v2"] = np.round(rng.uniform(0, 100, n_small), 6)

    medium = pd.DataFrame({"id2": np.arange(1, n_med + 1)})
    medium["id1"] = rng.integers(1, n_small + 1, n_med)
    medium["id5"] = _labels(medium["id2"].to_numpy(), "id")
    medium["v2"] = np.round(rng.uniform(0, 100, n_med), 6)

    big = pd.DataFrame({"id3": np.arange(1, n_big + 1)})
    big["id1"] = rng.integers(1, n_small + 1, n_big)
    big["id2"] = rng.integers(1, n_med + 1, n_big)
    big["v2"] = np.round(rng.uniform(0, 100, n_big), 6)
    return {"x": x, "small": small, "medium": medium, "big": big}


# ---------------------------------------------------------------------------
# Queries: each entry is (name, lazy_pandas_fn, polars_fn). A lazy_pandas_fn
# may be None for a query the engine does not yet support.
# ---------------------------------------------------------------------------
UNSUPPORTED = "unsupported"


def _lp(df):
    return df.select()


def groupby_queries():
    def lp1(d):
        return _lp(d).group_by("id1").agg(col("v1").sum().alias("v1"))

    def lp2(d):
        return _lp(d).group_by("id1", "id2").agg(col("v1").sum().alias("v1"))

    def lp3(d):
        return (
            _lp(d)
            .group_by("id3")
            .agg(col("v1").sum().alias("v1"), col("v3").mean().alias("v3"))
        )

    def lp4(d):
        return (
            _lp(d)
            .group_by("id4")
            .agg(
                col("v1").mean().alias("v1"),
                col("v2").mean().alias("v2"),
                col("v3").mean().alias("v3"),
            )
        )

    def lp5(d):
        return (
            _lp(d)
            .group_by("id6")
            .agg(
                col("v1").sum().alias("v1"),
                col("v2").sum().alias("v2"),
                col("v3").sum().alias("v3"),
            )
        )

    def lp6(d):
        return (
            _lp(d)
            .group_by("id4", "id5")
            .agg(
                col("v3").median().alias("v3_median"),
                col("v3").std().alias("v3_std"),
            )
        )

    def lp7(d):
        return (
            _lp(d)
            .group_by("id3")
            .agg((col("v1").max() - col("v2").min()).alias("range_v1_v2"))
        )

    def lp8(d):
        return (
            _lp(d)
            .select(col("id6"), col("v3"))
            .sort("v3", descending=True)
            .group_by("id6")
            .head(2)
        )

    def lp9(d):
        return (
            _lp(d)
            .group_by("id2", "id4")
            .agg((col("v1").corr(col("v2")) ** 2).alias("r2"))
        )

    def lp10(d):
        return (
            _lp(d)
            .group_by("id1", "id2", "id3", "id4", "id5", "id6")
            .agg(col("v3").sum().alias("v3"), col("v1").count().alias("count"))
        )

    def pl1(d):
        return d.lazy().group_by("id1").agg(pl.sum("v1")).collect()

    def pl2(d):
        return d.lazy().group_by("id1", "id2").agg(pl.sum("v1")).collect()

    def pl3(d):
        return (
            d.lazy()
            .group_by("id3")
            .agg(pl.sum("v1"), pl.mean("v3").alias("v3"))
            .collect()
        )

    def pl4(d):
        return (
            d.lazy()
            .group_by("id4")
            .agg(pl.mean("v1"), pl.mean("v2"), pl.mean("v3"))
            .collect()
        )

    def pl5(d):
        return (
            d.lazy()
            .group_by("id6")
            .agg(pl.sum("v1"), pl.sum("v2"), pl.sum("v3"))
            .collect()
        )

    def pl6(d):
        return (
            d.lazy()
            .group_by("id4", "id5")
            .agg(pl.median("v3").alias("v3_median"), pl.std("v3").alias("v3_std"))
            .collect()
        )

    def pl7(d):
        return (
            d.lazy()
            .group_by("id3")
            .agg((pl.max("v1") - pl.min("v2")).alias("range_v1_v2"))
            .collect()
        )

    def pl8(d):
        return (
            d.lazy()
            .group_by("id6")
            .agg(pl.col("v3").top_k(2).alias("largest2_v3"))
            .explode("largest2_v3")
            .collect()
        )

    def pl9(d):
        return (
            d.lazy()
            .group_by("id2", "id4")
            .agg((pl.corr("v1", "v2") ** 2).alias("r2"))
            .collect()
        )

    def pl10(d):
        return (
            d.lazy()
            .group_by("id1", "id2", "id3", "id4", "id5", "id6")
            .agg(pl.sum("v3"), pl.len().alias("count"))
            .collect()
        )

    return [
        ("q1: sum(v1) by id1", lp1, pl1),
        ("q2: sum(v1) by id1,id2", lp2, pl2),
        ("q3: sum(v1),mean(v3) by id3", lp3, pl3),
        ("q4: mean(v1,v2,v3) by id4", lp4, pl4),
        ("q5: sum(v1,v2,v3) by id6", lp5, pl5),
        ("q6: median(v3),std(v3) by id4,id5", lp6, pl6),  # grouped median
        ("q7: max(v1)-min(v2) by id3", lp7, pl7),  # agg arithmetic
        ("q8: top2 v3 by id6", lp8, pl8),  # grouped top-k (sort + head)
        ("q9: corr(v1,v2)^2 by id2,id4", lp9, pl9),  # correlation
        ("q10: sum(v3),count by id1..id6", lp10, pl10),
    ]


def join_queries(tables):
    small, medium, big = tables["small"], tables["medium"], tables["big"]

    def lp1(x):
        return _lp(x).join(small.select(), on="id1", how="inner")

    def lp2(x):
        return _lp(x).join(medium.select(), on="id2", how="inner")

    def lp3(x):
        return _lp(x).join(medium.select(), on="id2", how="left")

    def lp4(x):
        return _lp(x).join(medium.select(), on="id5", how="inner")

    def lp5(x):
        return _lp(x).join(big.select(), on="id3", how="inner")

    pl_small = pl.from_pandas(small) if HAS_POLARS else None
    pl_medium = pl.from_pandas(medium) if HAS_POLARS else None
    pl_big = pl.from_pandas(big) if HAS_POLARS else None

    def pj1(x):
        return x.lazy().join(pl_small.lazy(), on="id1", how="inner").collect()

    def pj2(x):
        return x.lazy().join(pl_medium.lazy(), on="id2", how="inner").collect()

    def pj3(x):
        return x.lazy().join(pl_medium.lazy(), on="id2", how="left").collect()

    def pj4(x):
        return x.lazy().join(pl_medium.lazy(), on="id5", how="inner").collect()

    def pj5(x):
        return x.lazy().join(pl_big.lazy(), on="id3", how="inner").collect()

    return [
        ("q1: inner join small on id1", lp1, pj1),
        ("q2: inner join medium on id2", lp2, pj2),
        ("q3: left join medium on id2", lp3, pj3),
        ("q4: inner join medium on id5", lp4, pj4),
        ("q5: inner join big on id3", lp5, pj5),
    ]


# ---------------------------------------------------------------------------
# Timing (cold + hot, the h2oai "run twice" convention)
# ---------------------------------------------------------------------------
def _collect(result):
    """Force materialization for a lazy-pandas or Polars result.

    Our LazyDataFrame collects with ``order="relaxed"`` - the H2O benchmark
    does not constrain output row order (neither does Polars), so this is the
    fair comparison and lets joins use the acero path. Polars results arrive
    already materialized from their query function.
    """
    from pandas.lazy.frame import LazyDataFrame

    if isinstance(result, LazyDataFrame):
        return result.collect(use_physical_planner=True, order="relaxed")
    return result


def time_query(fn, arg, warm_runs=4):
    """Return (cold_ms, hot_ms, nrows).

    cold = the very first run (all caches cold - includes one-time process
    costs: acero thread-pool init and the per-column dictionary-encoding key
    cache build). hot = min of ``warm_runs`` subsequent runs, i.e. steady
    state once those one-time costs are amortized. The original "run twice"
    convention conflated the two: for the first Arrow-routed query in a
    process, *both* of its runs sat in the warm-up tail (e.g. string-key q1
    measured 114 ms when steady state is ~13 ms).
    """
    start = time.perf_counter()
    out = _collect(fn(arg))
    cold = (time.perf_counter() - start) * 1000
    hot = float("inf")
    for _ in range(warm_runs):
        s = time.perf_counter()
        out = _collect(fn(arg))
        hot = min(hot, (time.perf_counter() - s) * 1000)
    return cold, hot, len(out)


def run_task(name, queries, arg, label):
    print(f"\n## {name} ({label})\n")
    print(f"{'query':38} {'LP cold':>9} {'LP hot':>9} {'PL hot':>9} {'hotx':>6}")
    print("-" * 76)
    for qname, lp_fn, pl_fn in queries:
        # Polars
        pl_hot = None
        if HAS_POLARS and pl_fn is not None:
            try:
                _, pl_hot, _ = time_query(pl_fn, arg["pl"])
            except Exception as e:
                pl_hot = f"ERR:{type(e).__name__}"
        # lazy pandas
        if lp_fn is None:
            lp_cold = lp_hot = UNSUPPORTED
            ratio = ""
        else:
            try:
                lp_cold, lp_hot, _ = time_query(lp_fn, arg["lp"])
                ratio = (
                    f"{pl_hot / lp_hot:.2f}"
                    if isinstance(pl_hot, float) and lp_hot
                    else ""
                )
            except Exception as e:
                lp_cold = lp_hot = f"ERR:{type(e).__name__}"
                ratio = ""

        def fmt(v):
            return f"{v:9.1f}" if isinstance(v, float) else f"{v!s:>9}"

        print(f"{qname:38} {fmt(lp_cold)} {fmt(lp_hot)} {fmt(pl_hot)} {ratio:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1e7", help="rows, e.g. 1e7 or 1e8")
    ap.add_argument("--task", choices=["groupby", "join", "all"], default="all")
    ap.add_argument("--k", type=int, default=100, help="group-by key cardinality")
    args = ap.parse_args()

    n = int(float(args.scale))
    print("H2O.ai db-benchmark — lazy pandas vs Polars")
    print(f"scale: {n:,} rows  (k={args.k})  polars={'yes' if HAS_POLARS else 'NO'}")
    print("hotx = Polars_hot / LazyPandas_hot  (>1 means lazy pandas faster)")

    if args.task in ("groupby", "all"):
        t = time.perf_counter()
        df = generate_groupby(n, args.k)
        print(f"\n[generated group-by data in {time.perf_counter() - t:.1f}s]")
        arg = {"lp": df, "pl": pl.from_pandas(df) if HAS_POLARS else None}
        run_task("Group-by", groupby_queries(), arg, f"{n:,} rows")

    if args.task in ("join", "all"):
        t = time.perf_counter()
        tables = generate_join(n)
        print(f"\n[generated join data in {time.perf_counter() - t:.1f}s]")
        arg = {
            "lp": tables["x"],
            "pl": pl.from_pandas(tables["x"]) if HAS_POLARS else None,
        }
        run_task("Join", join_queries(tables), arg, f"{n:,} rows")


if __name__ == "__main__":
    main()
