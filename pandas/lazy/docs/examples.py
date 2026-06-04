#!/usr/bin/env python3
"""
Lazy pandas — runnable tour.

A self-contained demonstration of the lazy execution prototype for the
design proposal in PROPOSAL.md. Requires only pandas (this branch) and
pyarrow; writes temporary files to a scratch directory it cleans up.

Usage:
    python pandas/lazy/docs/examples.py
"""

from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

import pandas as pd
from pandas.lazy import (
    col,
    lit,
    scan,
    when,
)

RNG = np.random.default_rng(42)


def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def make_events(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": RNG.choice(["east", "west", "north", "south"], n),
            "status": RNG.choice(["ok", "error", "timeout"], n, p=[0.9, 0.07, 0.03]),
            "value": RNG.exponential(100.0, n).round(2),
            "user": RNG.integers(0, n // 10 or 1, n),
        }
    )


def example_basics() -> None:
    header("1. Basics: build lazily, inspect, collect")
    df = make_events(1_000)

    ldf = (
        df.select()  # enter lazy mode
        .filter(col("status") == "ok")
        .with_columns((col("value") * 0.01).alias("value_pct"))
        .select("region", "value_pct")
    )
    print("Nothing has executed yet. The optimized plan:\n")
    print(ldf.explain())
    result = ldf.collect()  # an ordinary pd.DataFrame
    print(f"\ncollect() -> {type(result).__name__}, {len(result)} rows")
    print(result.head(3))


def example_expressions() -> None:
    header("2. Expressions: when/then, strings, window functions")
    df = pd.DataFrame(
        {
            "name": ["  Alice", "bob ", "CAROL", "dan", "  eve "],
            "score": [82, 95, 58, 73, 95],
        }
    )
    result = (
        df.select()
        .with_columns(
            col("name").str.strip().str.lower().alias("name"),
            when(col("score") >= 90)
            .then(lit("A"))
            .when(col("score") >= 70)
            .then(lit("B"))
            .otherwise(lit("C"))
            .alias("grade"),
            (-col("score")).rank().alias("rank"),  # rank by descending score
        )
        .sort("rank")
        .collect()
    )
    print(result)


def example_groupby_join() -> None:
    header("3. Group-by and join")
    events = make_events(10_000)
    regions = pd.DataFrame(
        {
            "region": ["east", "west", "north", "south"],
            "manager": ["amy", "ben", "cho", "dee"],
        }
    )
    result = (
        events.select()
        .filter(col("status") != "error")
        .group_by("region")
        .agg(
            col("value").sum().alias("total"),
            col("value").mean().alias("avg"),
            col("user").n_unique().alias("users"),
        )
        .join(regions.select(), on="region")
        .sort("total", descending=True)
        .collect()
    )
    print(result)


def example_optimizer() -> None:
    header("4. The optimizer at work")
    df = make_events(1_000)
    ldf = (
        df.select()
        .filter(col("value") > 10)
        .filter(col("status") == "ok")  # fused with the filter above
        .with_columns((col("value") * 2).alias("v2"), lit(1).alias("unused"))
        .select("region", "v2")  # 'unused' is pruned
        .sort("v2", descending=True)
        .head(5)  # Sort+Limit -> TopK
    )
    print("As written:\n")
    print(ldf.explain(optimized=False))
    print("\nAfter optimization (filters fused, projection pruned, TopK):\n")
    print(ldf.explain())


def example_scan(tmp: Path) -> None:
    header("5. Lazy file scans with pushdown (Parquet, glob)")
    for i in range(4):
        make_events(50_000).to_parquet(tmp / f"events_{i}.parquet")

    ldf = (
        scan(str(tmp / "events_*.parquet"))
        .filter(col("value") > 400)  # pushed into the Parquet reader
        .select("region", "value")  # only these columns are read
        .head(5)
    )
    print(ldf.explain(physical=True))
    print(ldf.collect(use_physical_planner=True))


def example_streaming(tmp: Path) -> None:
    header("6. Streaming execution: batches + early termination")
    path = tmp / "big.parquet"
    make_events(500_000).to_parquet(path)

    ldf = scan(str(path)).filter(col("status") == "timeout")
    batches = ldf.collect(use_physical_planner=True, streaming=True, batch_size=65536)
    total = 0
    for i, batch in enumerate(batches):
        total += len(batch)
        print(f"  batch {i}: {len(batch):>6} rows (running total {total})")
    print(f"Streamed {total} matching rows without materializing the scan.")

    t0 = time.perf_counter()
    head = scan(str(path)).head(10).collect(use_physical_planner=True)
    print(
        f"head(10) with early termination: {len(head)} rows "
        f"in {(time.perf_counter() - t0) * 1e3:.1f} ms"
    )


def example_pipeline_benchmark() -> None:
    header("7. Where lazy wins: a fused multi-filter pipeline (1M rows)")
    df = make_events(1_000_000).convert_dtypes(dtype_backend="pyarrow")

    def eager() -> pd.DataFrame:
        out = df[df["value"] > 20]
        out = out[out["status"] == "ok"]
        out = out[out["region"].isin(["east", "west"])]
        return out[["region", "value"]]

    lazy_q = (
        df.select()
        .filter(col("value") > 20)
        .filter(col("status") == "ok")
        .filter(col("region").isin(["east", "west"]))
        .select("region", "value")
    )

    eager(), lazy_q.collect(use_physical_planner=True)  # warmup
    t0 = time.perf_counter()
    e = eager()
    t_eager = time.perf_counter() - t0
    t0 = time.perf_counter()
    lazy = lazy_q.collect(use_physical_planner=True)
    t_lazy = time.perf_counter() - t0
    assert len(e) == len(lazy)
    print(f"  eager pandas : {t_eager * 1e3:7.1f} ms")
    print(f"  lazy pandas  : {t_lazy * 1e3:7.1f} ms  ({t_eager / t_lazy:.1f}x)")
    print("  (single simple ops favor eager — lazy pays off on pipelines)")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="lazy_pandas_demo_"))
    try:
        example_basics()
        example_expressions()
        example_groupby_join()
        example_optimizer()
        example_scan(tmp)
        example_streaming(tmp)
        example_pipeline_benchmark()
        header("Done. See PROPOSAL.md and ARCHITECTURE.md for the design.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
