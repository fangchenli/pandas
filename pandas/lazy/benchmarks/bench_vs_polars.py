#!/usr/bin/env python3
"""
Benchmark: Lazy Pandas vs Polars

Compares performance of lazy pandas against Polars for common operations:
- File scanning (Parquet, CSV)
- Filtering and projection
- Aggregations (groupby)
- Joins
- String operations
- Window functions

Usage:
    python pandas/lazy/benchmarks/bench_vs_polars.py
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import platform
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from shared import (
    DATA_DIR,
    materialize_result,
)

import pandas as pd
from pandas.lazy import (
    col,
    scan,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Configuration
# =============================================================================
WARMUP_RUNS = 3
TIMED_RUNS = 7
SEED = 42

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    category: str
    operation: str
    data_rows: int
    lazy_pandas_ms: float
    polars_ms: float
    speedup: float  # polars_ms / lazy_pandas_ms (>1 means lazy pandas faster)
    notes: str = ""


# =============================================================================
# Helper Functions
# =============================================================================


def benchmark_operation(
    op_func: Callable, warmup: int = WARMUP_RUNS, runs: int = TIMED_RUNS
) -> float:
    """Run benchmark with warmup, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        result = op_func()
        materialize_result(result)

    gc.collect()
    gc.disable()

    try:
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            result = op_func()
            materialize_result(result)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    finally:
        gc.enable()

    return float(np.median(times))


def generate_test_data(
    n_rows: int, seed: int = SEED
) -> tuple[pd.DataFrame, pl.DataFrame]:
    """Generate matching test data for pandas and Polars."""
    rng = np.random.default_rng(seed)

    data = {
        "id": np.arange(n_rows),
        "group": rng.choice(["A", "B", "C", "D", "E"], n_rows),
        "category": rng.choice([f"cat_{i}" for i in range(100)], n_rows),
        "value1": rng.standard_normal(n_rows),
        "value2": rng.standard_normal(n_rows),
        "int_val": rng.integers(0, 1000, n_rows),
        "text": rng.choice(
            [
                "hello world",
                "foo bar baz",
                "test string",
                "sample text",
                "random words",
            ],
            n_rows,
        ),
    }

    pdf = pd.DataFrame(data)
    pldf = pl.DataFrame(data)

    return pdf, pldf


def generate_join_data(
    left_size: int, right_size: int, key_cardinality: int, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Generate join test data."""
    rng = np.random.default_rng(seed)

    # Left table
    left_data = {
        "key": rng.integers(0, key_cardinality, left_size),
        "left_val": rng.standard_normal(left_size),
    }
    pdf_left = pd.DataFrame(left_data)
    pldf_left = pl.DataFrame(left_data)

    # Right table
    right_data = {
        "key": rng.integers(0, key_cardinality, right_size),
        "right_val": rng.standard_normal(right_size),
    }
    pdf_right = pd.DataFrame(right_data)
    pldf_right = pl.DataFrame(right_data)

    return pdf_left, pdf_right, pldf_left, pldf_right


# =============================================================================
# Benchmark Categories
# =============================================================================


def benchmark_filter_project(
    pdf: pd.DataFrame, pldf: pl.DataFrame
) -> list[BenchmarkResult]:
    """Benchmark filter and projection operations."""
    results = []
    n_rows = len(pdf)

    # Simple filter
    def lazy_pandas_filter():
        return pdf.select().filter(col("value1") > 0).collect(use_physical_planner=True)

    def polars_filter():
        return pldf.lazy().filter(pl.col("value1") > 0).collect()

    lp_time = benchmark_operation(lazy_pandas_filter)
    pl_time = benchmark_operation(polars_filter)
    results.append(
        BenchmarkResult(
            category="filter_project",
            operation="filter(value1 > 0)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # Filter + select columns
    def lazy_pandas_filter_select():
        return (
            pdf.select()
            .filter(col("value1") > 0)
            .select("id", "group", "value1")
            .collect(use_physical_planner=True)
        )

    def polars_filter_select():
        return (
            pldf.lazy()
            .filter(pl.col("value1") > 0)
            .select("id", "group", "value1")
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_filter_select)
    pl_time = benchmark_operation(polars_filter_select)
    results.append(
        BenchmarkResult(
            category="filter_project",
            operation="filter + select(3 cols)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # Multiple filters (AND)
    def lazy_pandas_multi_filter():
        return (
            pdf.select()
            .filter((col("value1") > 0) & (col("int_val") < 500))
            .collect(use_physical_planner=True)
        )

    def polars_multi_filter():
        return (
            pldf.lazy()
            .filter((pl.col("value1") > 0) & (pl.col("int_val") < 500))
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_multi_filter)
    pl_time = benchmark_operation(polars_multi_filter)
    results.append(
        BenchmarkResult(
            category="filter_project",
            operation="filter(val1 > 0 AND int_val < 500)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # With computed column
    def lazy_pandas_with_column():
        return (
            pdf.select()
            .with_columns((col("value1") + col("value2")).alias("sum_vals"))
            .collect(use_physical_planner=True)
        )

    def polars_with_column():
        return (
            pldf.lazy()
            .with_columns((pl.col("value1") + pl.col("value2")).alias("sum_vals"))
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_with_column)
    pl_time = benchmark_operation(polars_with_column)
    results.append(
        BenchmarkResult(
            category="filter_project",
            operation="with_columns(value1 + value2)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # Filter -> scalar aggregate. The single-pass mask+accumulate shape that
    # avoids materializing the filtered intermediate (the structural Polars
    # win; see docs/MATERIALIZATION_EXPERIMENT.md). Routes to the fused
    # PhysicalFusedFilterAgg kernel rather than the row-compacting pipeline.
    def lazy_pandas_filter_sum():
        return (
            pdf.select()
            .filter(col("value1") > 0)
            .select(col("value1").sum().alias("s"))
            .collect(use_physical_planner=True)
        )

    def polars_filter_sum():
        return (
            pldf.lazy().filter(pl.col("value1") > 0).select(pl.col("value1").sum())
        ).collect()

    lp_time = benchmark_operation(lazy_pandas_filter_sum)
    pl_time = benchmark_operation(polars_filter_sum)
    results.append(
        BenchmarkResult(
            category="filter_project",
            operation="filter(value1 > 0).select(sum)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    def lazy_pandas_filter_count():
        return (
            pdf.select()
            .filter((col("value1") > 0) & (col("int_val") < 500))
            .select(col("value1").count().alias("n"))
            .collect(use_physical_planner=True)
        )

    def polars_filter_count():
        return (
            pldf.lazy()
            .filter((pl.col("value1") > 0) & (pl.col("int_val") < 500))
            .select(pl.col("value1").count())
        ).collect()

    lp_time = benchmark_operation(lazy_pandas_filter_count)
    pl_time = benchmark_operation(polars_filter_count)
    results.append(
        BenchmarkResult(
            category="filter_project",
            operation="filter(val1>0 AND int_val<500).select(count)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    def lazy_pandas_filter_prodsum():
        return (
            pdf.select()
            .filter(col("value1") > 0)
            .select((col("value1") * col("value2")).sum().alias("p"))
            .collect(use_physical_planner=True)
        )

    def polars_filter_prodsum():
        return (
            pldf.lazy()
            .filter(pl.col("value1") > 0)
            .select((pl.col("value1") * pl.col("value2")).sum())
        ).collect()

    lp_time = benchmark_operation(lazy_pandas_filter_prodsum)
    pl_time = benchmark_operation(polars_filter_prodsum)
    results.append(
        BenchmarkResult(
            category="filter_project",
            operation="filter(value1 > 0).select(sum(v1*v2))",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    return results


def benchmark_aggregations(
    pdf: pd.DataFrame, pldf: pl.DataFrame
) -> list[BenchmarkResult]:
    """Benchmark aggregation operations."""
    results = []
    n_rows = len(pdf)

    # Simple groupby sum
    def lazy_pandas_groupby_sum():
        return (
            pdf.select()
            .group_by("group")
            .agg(col("value1").sum().alias("sum"))
            .collect(use_physical_planner=True)
        )

    def polars_groupby_sum():
        return (
            pldf.lazy()
            .group_by("group")
            .agg(pl.col("value1").sum().alias("sum"))
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_groupby_sum)
    pl_time = benchmark_operation(polars_groupby_sum)
    results.append(
        BenchmarkResult(
            category="aggregation",
            operation="groupby(group).sum(value1)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # Groupby with multiple aggregations
    def lazy_pandas_groupby_multi():
        return (
            pdf.select()
            .group_by("group")
            .agg(
                col("value1").sum().alias("sum_v1"),
                col("value1").mean().alias("mean_v1"),
                col("value2").max().alias("max_v2"),
                col("int_val").count().alias("count"),
            )
            .collect(use_physical_planner=True)
        )

    def polars_groupby_multi():
        return (
            pldf.lazy()
            .group_by("group")
            .agg(
                pl.col("value1").sum().alias("sum_v1"),
                pl.col("value1").mean().alias("mean_v1"),
                pl.col("value2").max().alias("max_v2"),
                pl.col("int_val").count().alias("count"),
            )
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_groupby_multi)
    pl_time = benchmark_operation(polars_groupby_multi)
    results.append(
        BenchmarkResult(
            category="aggregation",
            operation="groupby(group).agg(sum, mean, max, count)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # High cardinality groupby
    def lazy_pandas_groupby_highcard():
        return (
            pdf.select()
            .group_by("category")
            .agg(col("value1").sum().alias("sum"))
            .collect(use_physical_planner=True)
        )

    def polars_groupby_highcard():
        return (
            pldf.lazy()
            .group_by("category")
            .agg(pl.col("value1").sum().alias("sum"))
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_groupby_highcard)
    pl_time = benchmark_operation(polars_groupby_highcard)
    results.append(
        BenchmarkResult(
            category="aggregation",
            operation="groupby(category[100]).sum()",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    return results


def benchmark_joins(
    pdf_left: pd.DataFrame,
    pdf_right: pd.DataFrame,
    pldf_left: pl.DataFrame,
    pldf_right: pl.DataFrame,
) -> list[BenchmarkResult]:
    """Benchmark join operations."""
    results = []
    left_rows = len(pdf_left)
    right_rows = len(pdf_right)

    # Inner join
    def lazy_pandas_inner_join():
        return (
            pdf_left.select()
            .join(pdf_right.select(), on="key", how="inner")
            .collect(use_physical_planner=True)
        )

    def polars_inner_join():
        return pldf_left.lazy().join(pldf_right.lazy(), on="key", how="inner").collect()

    lp_time = benchmark_operation(lazy_pandas_inner_join)
    pl_time = benchmark_operation(polars_inner_join)
    results.append(
        BenchmarkResult(
            category="join",
            operation=f"inner_join({left_rows}x{right_rows})",
            data_rows=left_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # Left join
    def lazy_pandas_left_join():
        return (
            pdf_left.select()
            .join(pdf_right.select(), on="key", how="left")
            .collect(use_physical_planner=True)
        )

    def polars_left_join():
        return pldf_left.lazy().join(pldf_right.lazy(), on="key", how="left").collect()

    lp_time = benchmark_operation(lazy_pandas_left_join)
    pl_time = benchmark_operation(polars_left_join)
    results.append(
        BenchmarkResult(
            category="join",
            operation=f"left_join({left_rows}x{right_rows})",
            data_rows=left_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    return results


def benchmark_string_ops(
    pdf: pd.DataFrame, pldf: pl.DataFrame
) -> list[BenchmarkResult]:
    """Benchmark string operations."""
    results = []
    n_rows = len(pdf)

    # String contains
    def lazy_pandas_str_contains():
        return (
            pdf.select()
            .filter(col("text").str.contains("foo"))
            .collect(use_physical_planner=True)
        )

    def polars_str_contains():
        return pldf.lazy().filter(pl.col("text").str.contains("foo")).collect()

    lp_time = benchmark_operation(lazy_pandas_str_contains)
    pl_time = benchmark_operation(polars_str_contains)
    results.append(
        BenchmarkResult(
            category="string",
            operation="filter(text.contains('foo'))",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # String lower
    def lazy_pandas_str_lower():
        return (
            pdf.select()
            .with_columns(col("text").str.lower().alias("text_lower"))
            .collect(use_physical_planner=True)
        )

    def polars_str_lower():
        return (
            pldf.lazy()
            .with_columns(pl.col("text").str.to_lowercase().alias("text_lower"))
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_str_lower)
    pl_time = benchmark_operation(polars_str_lower)
    results.append(
        BenchmarkResult(
            category="string",
            operation="with_columns(text.lower())",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # String length
    def lazy_pandas_str_len():
        return (
            pdf.select()
            .with_columns(col("text").str.len().alias("text_len"))
            .collect(use_physical_planner=True)
        )

    def polars_str_len():
        return (
            pldf.lazy()
            .with_columns(pl.col("text").str.len_chars().alias("text_len"))
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_str_len)
    pl_time = benchmark_operation(polars_str_len)
    results.append(
        BenchmarkResult(
            category="string",
            operation="with_columns(text.len())",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    return results


def benchmark_parquet_scan() -> list[BenchmarkResult]:
    """Benchmark Parquet file scanning."""
    results = []

    # Find parquet files
    parquet_files = sorted(DATA_DIR.glob("yellow_tripdata_*.parquet"))

    if not parquet_files:
        print("  No parquet files found, skipping parquet scan benchmarks")
        return results

    single_file = parquet_files[0]
    glob_pattern = str(DATA_DIR / "yellow_tripdata_*.parquet")

    # Get row count
    n_rows = len(pd.read_parquet(single_file))

    # Single file scan + filter + head (use streaming for lazy pandas)
    def lazy_pandas_scan_filter_head():
        return (
            scan(single_file)
            .filter(col("fare_amount") > 20)
            .head(1000)
            .collect(streaming=True, use_physical_planner=True)
        )

    def polars_scan_filter_head():
        return (
            pl.scan_parquet(single_file)
            .filter(pl.col("fare_amount") > 20)
            .head(1000)
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_scan_filter_head)
    pl_time = benchmark_operation(polars_scan_filter_head)
    results.append(
        BenchmarkResult(
            category="parquet_scan",
            operation="scan.filter.head(1000)",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # Single file scan + select + filter
    def lazy_pandas_scan_select_filter():
        return (
            scan(single_file)
            .select("fare_amount", "tip_amount", "total_amount")
            .filter(col("fare_amount") > 20)
            .collect(streaming=True, use_physical_planner=True)
        )

    def polars_scan_select_filter():
        return (
            pl.scan_parquet(single_file)
            .select("fare_amount", "tip_amount", "total_amount")
            .filter(pl.col("fare_amount") > 20)
            .collect()
        )

    lp_time = benchmark_operation(lazy_pandas_scan_select_filter)
    pl_time = benchmark_operation(polars_scan_select_filter)
    results.append(
        BenchmarkResult(
            category="parquet_scan",
            operation="scan.select(3).filter",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # Multi-file glob scan + head
    if len(parquet_files) > 1:

        def lazy_pandas_glob_head():
            return (
                scan(glob_pattern)
                .head(1000)
                .collect(streaming=True, use_physical_planner=True)
            )

        def polars_glob_head():
            return pl.scan_parquet(glob_pattern).head(1000).collect()

        lp_time = benchmark_operation(lazy_pandas_glob_head)
        pl_time = benchmark_operation(polars_glob_head)
        results.append(
            BenchmarkResult(
                category="parquet_scan",
                operation="glob_scan.head(1000)",
                data_rows=n_rows * len(parquet_files),
                lazy_pandas_ms=lp_time,
                polars_ms=pl_time,
                speedup=pl_time / lp_time if lp_time > 0 else 0,
            )
        )

    return results


def benchmark_head_limit(
    pdf: pd.DataFrame, pldf: pl.DataFrame
) -> list[BenchmarkResult]:
    """Benchmark head/limit operations."""
    results = []
    n_rows = len(pdf)

    for limit in [10, 100, 1000]:

        def lazy_pandas_head(lim=limit):
            return pdf.select().head(lim).collect(use_physical_planner=True)

        def polars_head(lim=limit):
            return pldf.lazy().head(lim).collect()

        lp_time = benchmark_operation(lazy_pandas_head)
        pl_time = benchmark_operation(polars_head)
        results.append(
            BenchmarkResult(
                category="limit",
                operation=f"head({limit})",
                data_rows=n_rows,
                lazy_pandas_ms=lp_time,
                polars_ms=pl_time,
                speedup=pl_time / lp_time if lp_time > 0 else 0,
            )
        )

    return results


def benchmark_sort(pdf: pd.DataFrame, pldf: pl.DataFrame) -> list[BenchmarkResult]:
    """Benchmark sort operations.

    Two regimes, deliberately: the mixed frame (string payload columns, plus a
    string key for one multi-key case) is the realistic string-gather-bound
    path; an all-numeric frame isolates the radix argsort and radix lexsort
    kernels from that gather wall. Without the numeric cases the suite cannot
    observe the numeric-sort work at all.
    """
    results = []
    n_rows = len(pdf)

    def add(operation, lp_fn, pl_fn):
        lp = benchmark_operation(lp_fn)
        pl_ms = benchmark_operation(pl_fn)
        results.append(
            BenchmarkResult(
                category="sort",
                operation=operation,
                data_rows=n_rows,
                lazy_pandas_ms=lp,
                polars_ms=pl_ms,
                speedup=pl_ms / lp if lp > 0 else 0,
            )
        )

    # --- Mixed frame (string payload columns are gathered on every sort) ---
    add(
        "sort(value1)",
        lambda: pdf.select().sort("value1").collect(use_physical_planner=True),
        lambda: pldf.lazy().sort("value1").collect(),
    )
    # String primary key -> radix lexsort cannot apply, falls back to Arrow.
    add(
        "sort(group[str], value1)",
        lambda: pdf.select().sort("group", "value1").collect(use_physical_planner=True),
        lambda: pldf.lazy().sort("group", "value1").collect(),
    )
    # Numeric multi-key on the same mixed frame -> radix lexsort fires; string
    # payload columns are still gathered, so this isolates the argsort win.
    add(
        "sort(int_val, value1)",
        lambda: (
            pdf.select().sort("int_val", "value1").collect(use_physical_planner=True)
        ),
        lambda: pldf.lazy().sort("int_val", "value1").collect(),
    )

    # --- All-numeric frame (no string gather; pure radix kernel timing) ---
    rng = np.random.default_rng(SEED + 1)
    num = pd.DataFrame(
        {
            "key_int": rng.integers(0, 1000, n_rows).astype("int64"),
            "num_a": rng.standard_normal(n_rows),
            "num_b": rng.standard_normal(n_rows),
        }
    )
    num_pl = pl.from_pandas(num)
    add(
        "sort(num_a) [numeric-only]",
        lambda: num.select().sort("num_a").collect(use_physical_planner=True),
        lambda: num_pl.lazy().sort("num_a").collect(),
    )
    add(
        "sort(key_int, num_a) [numeric-only]",
        lambda: (
            num.select().sort("key_int", "num_a").collect(use_physical_planner=True)
        ),
        lambda: num_pl.lazy().sort("key_int", "num_a").collect(),
    )

    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================


def run_benchmarks() -> list[BenchmarkResult]:
    """Run all benchmarks and return results."""
    all_results = []

    print(f"pandas={pd.__version__}, polars={pl.__version__}")
    print(f"Warmup: {WARMUP_RUNS}, Timed runs: {TIMED_RUNS}")
    print("=" * 70)

    # Test sizes
    sizes = [1_000_000, 10_000_000]

    for size in sizes:
        print(f"\n{'=' * 70}")
        print(f"DATA SIZE: {size:,} rows")
        print("=" * 70)

        # Generate test data
        print("\nGenerating test data...")
        pdf, pldf = generate_test_data(size)

        def print_result(r: BenchmarkResult) -> None:
            status = "LP faster" if r.speedup > 1 else "Polars faster"
            print(
                f"  {r.operation}: LP={r.lazy_pandas_ms:.2f}ms, "
                f"PL={r.polars_ms:.2f}ms, {r.speedup:.2f}x ({status})"
            )

        # Filter and projection
        print("\n--- Filter & Projection ---")
        results = benchmark_filter_project(pdf, pldf)
        for r in results:
            print_result(r)
        all_results.extend(results)

        # Aggregations
        print("\n--- Aggregations ---")
        results = benchmark_aggregations(pdf, pldf)
        for r in results:
            print_result(r)
        all_results.extend(results)

        # String operations
        print("\n--- String Operations ---")
        results = benchmark_string_ops(pdf, pldf)
        for r in results:
            print_result(r)
        all_results.extend(results)

        # Head/limit
        print("\n--- Head/Limit ---")
        results = benchmark_head_limit(pdf, pldf)
        for r in results:
            print_result(r)
        all_results.extend(results)

        # Sort
        print("\n--- Sort ---")
        results = benchmark_sort(pdf, pldf)
        for r in results:
            print_result(r)
        all_results.extend(results)

        # Joins
        print("\n--- Joins ---")
        pdf_left, pdf_right, pldf_left, pldf_right = generate_join_data(
            size, size // 10, size // 100
        )
        results = benchmark_joins(pdf_left, pdf_right, pldf_left, pldf_right)
        for r in results:
            print_result(r)
        all_results.extend(results)

    # Parquet scan benchmarks
    print(f"\n{'=' * 70}")
    print("PARQUET SCAN BENCHMARKS")
    print("=" * 70)
    results = benchmark_parquet_scan()
    for r in results:
        status = "LP faster" if r.speedup > 1 else "Polars faster"
        print(
            f"  {r.operation}: LP={r.lazy_pandas_ms:.2f}ms, "
            f"PL={r.polars_ms:.2f}ms, {r.speedup:.2f}x ({status})"
        )
    all_results.extend(results)

    # Engine-era cases (June 2026): category-key aggregation (Arrow
    # dictionary routing) and the join->groupby composite (decision
    # layer routes the order-free join to acero).
    print(f"\n{'=' * 70}")
    print("ENGINE PIPELINE BENCHMARKS")
    print("=" * 70)
    for n_rows in [1_000_000, 10_000_000]:
        results = benchmark_engine_pipelines(n_rows)
        for r in results:
            print_result(r)
        all_results.extend(results)

    return all_results


def benchmark_engine_pipelines(n_rows: int) -> list[BenchmarkResult]:
    """Composite patterns exercising the pipeline engine's routing."""
    rng = np.random.default_rng(123)
    results = []

    # Category-key groupby: Categorical flows as Arrow dictionary
    pdf = pd.DataFrame(
        {
            "g": pd.Categorical(rng.choice(["A", "B", "C", "D", "E"], n_rows)),
            "v": rng.standard_normal(n_rows),
        }
    )
    pldf = pl.from_pandas(pdf)

    def lp_cat_groupby():
        return (
            pdf.select()
            .group_by("g")
            .agg(col("v").sum().alias("s"))
            .collect(use_physical_planner=True)
        )

    def pl_cat_groupby():
        return pldf.lazy().group_by("g").agg(pl.col("v").sum().alias("s")).collect()

    lp_time = benchmark_operation(lp_cat_groupby)
    pl_time = benchmark_operation(pl_cat_groupby)
    results.append(
        BenchmarkResult(
            category="engine_pipeline",
            operation="groupby(category-dtype key).sum",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    # join -> groupby: the order-free join routes to acero
    n_right = max(n_rows // 10, 1)
    left = pd.DataFrame(
        {
            "k": rng.integers(0, max(n_rows // 20, 1), n_rows),
            "v": rng.standard_normal(n_rows),
        }
    )
    right = pd.DataFrame(
        {
            "k": rng.integers(0, max(n_rows // 20, 1), n_right),
            "w": rng.standard_normal(n_right),
        }
    )
    pl_left, pl_right = pl.from_pandas(left), pl.from_pandas(right)

    def lp_join_groupby():
        return (
            left.select()
            .join(right.select(), on="k")
            .group_by("k")
            .agg(col("v").sum().alias("s"), col("w").mean().alias("m"))
            .collect(use_physical_planner=True)
        )

    def pl_join_groupby():
        return (
            pl_left.lazy()
            .join(pl_right.lazy(), on="k")
            .group_by("k")
            .agg(pl.col("v").sum().alias("s"), pl.col("w").mean().alias("m"))
            .collect()
        )

    lp_time = benchmark_operation(lp_join_groupby)
    pl_time = benchmark_operation(pl_join_groupby)
    results.append(
        BenchmarkResult(
            category="engine_pipeline",
            operation="join -> groupby",
            data_rows=n_rows,
            lazy_pandas_ms=lp_time,
            polars_ms=pl_time,
            speedup=pl_time / lp_time if lp_time > 0 else 0,
        )
    )

    return results


def generate_report(results: list[BenchmarkResult]) -> str:
    """Generate markdown report from results."""
    lines = [
        "# Lazy Pandas vs Polars Benchmark Report",
        "",
        "## Environment",
        "",
        f"- pandas version: {pd.__version__}",
        f"- polars version: {pl.__version__}",
        f"- Python version: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        f"- Warmup runs: {WARMUP_RUNS}",
        f"- Timed runs: {TIMED_RUNS}",
        "",
        "## Methodology",
        "",
        "Lazy pandas timings use the **physical engine** "
        "(`collect(use_physical_planner=True)`). Reports generated before "
        "June 2026 measured the eager evaluation path for every category "
        "except parquet scans and are not directly comparable. The test "
        "data is mixed-dtype (numeric + string columns), so conversion "
        "boundaries are part of what is measured. Absolute timings vary "
        "with machine load; treat ratios as the signal.",
        "",
        "## Summary",
        "",
        "**Speedup interpretation:** Values > 1.0 mean lazy pandas is faster, "
        "< 1.0 mean Polars is faster.",
        "",
    ]

    # Group by category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    # Summary table
    lines.append("### Category Summary")
    lines.append("")
    lines.append("| Category | Avg Speedup | Best LP | Best Polars |")
    lines.append("|----------|-------------|---------|-------------|")

    for cat, cat_results in categories.items():
        speedups = [r.speedup for r in cat_results]
        avg = np.mean(speedups)
        best_lp = max(speedups)
        best_pl = min(speedups)
        lines.append(f"| {cat} | {avg:.2f}x | {best_lp:.2f}x | {best_pl:.2f}x |")

    lines.append("")

    # Detailed results by category
    for cat, cat_results in categories.items():
        lines.append(f"## {cat.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |")
        lines.append("|-----------|------|------------------|-------------|---------|")

        for r in cat_results:
            speedup_str = (
                f"**{r.speedup:.2f}x**" if r.speedup > 1 else f"*{r.speedup:.2f}x*"
            )
            lines.append(
                f"| {r.operation} | {r.data_rows:,} | "
                f"{r.lazy_pandas_ms:.2f} | {r.polars_ms:.2f} | {speedup_str} |"
            )

        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Find where lazy pandas wins
    lp_wins = [r for r in results if r.speedup > 1.0]
    pl_wins = [r for r in results if r.speedup < 1.0]

    if lp_wins:
        lines.append("### Where Lazy Pandas is Faster")
        lines.append("")
        sorted_lp = sorted(lp_wins, key=lambda x: x.speedup, reverse=True)[:5]
        lines.extend(
            f"- **{r.operation}** ({r.category}): {r.speedup:.2f}x faster"
            for r in sorted_lp
        )
        lines.append("")

    if pl_wins:
        lines.append("### Where Polars is Faster")
        lines.append("")
        sorted_pl = sorted(pl_wins, key=lambda x: x.speedup)[:5]
        lines.extend(
            f"- **{r.operation}** ({r.category}): {1 / r.speedup:.2f}x faster"
            for r in sorted_pl
        )
        lines.append("")

    return "\n".join(lines)


def main():
    print("Lazy Pandas vs Polars Benchmark")
    print("=" * 70)

    results = run_benchmarks()

    # Save results
    print("\n" + "=" * 70)
    print("Generating report...")
    print("=" * 70)

    # Save JSON
    results_dict = [
        {
            "category": r.category,
            "operation": r.operation,
            "data_rows": r.data_rows,
            "lazy_pandas_ms": r.lazy_pandas_ms,
            "polars_ms": r.polars_ms,
            "speedup": r.speedup,
        }
        for r in results
    ]

    with open("benchmark_lazy_vs_polars_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print("Results saved to benchmark_lazy_vs_polars_results.json")

    # Generate and save report
    report = generate_report(results)
    with open("LAZY_VS_POLARS_BENCHMARK.md", "w") as f:
        f.write(report)
    print("Report saved to LAZY_VS_POLARS_BENCHMARK.md")

    # Print summary
    print("\n" + report)


if __name__ == "__main__":
    main()
