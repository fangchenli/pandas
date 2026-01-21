#!/usr/bin/env python
"""
Shared benchmark utilities for lazy pandas.

This module provides common functions used across benchmark files to avoid
code duplication:
- timeit: Benchmark timing utility with warmup and statistics
- Data generation functions for various benchmark scenarios
- Path utilities for data directories
"""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Path utilities
# =============================================================================

# Standard data directory relative to the pandas root
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def ensure_data_dir() -> Path:
    """Ensure the data directory exists and return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


# =============================================================================
# Timing utilities
# =============================================================================


def timeit(
    func: Callable[[], Any],
    n_runs: int = 5,
    warmup: int = 1,
    disable_gc: bool = False,
) -> tuple[float, float]:
    """
    Run function multiple times and return (mean, std) in milliseconds.

    Parameters
    ----------
    func : Callable
        Function to benchmark (should take no arguments)
    n_runs : int, default 5
        Number of timed runs
    warmup : int, default 1
        Number of warmup runs (not timed)
    disable_gc : bool, default False
        Whether to disable garbage collection during timing

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation in milliseconds
    """
    # Warmup runs
    for _ in range(warmup):
        func()

    if disable_gc:
        gc.collect()
        gc.disable()

    try:
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    finally:
        if disable_gc:
            gc.enable()

    return float(np.mean(times)), float(np.std(times))


def benchmark(
    func: Callable[[], Any],
    n_runs: int = 5,
    n_warmup: int = 2,
) -> dict[str, float]:
    """
    Benchmark a function and return detailed timing statistics.

    Parameters
    ----------
    func : Callable
        Function to benchmark
    n_runs : int, default 5
        Number of timed runs
    n_warmup : int, default 2
        Number of warmup runs

    Returns
    -------
    dict[str, float]
        Dictionary with mean_ms, std_ms, and min_ms
    """
    # Warmup
    for _ in range(n_warmup):
        func()

    gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    finally:
        gc.enable()

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
    }


def materialize_result(result: Any) -> Any:
    """
    Ensure a result is fully materialized (handles generators from streaming).

    Parameters
    ----------
    result : Any
        The result from collect(), may be a generator

    Returns
    -------
    Any
        Materialized result (concatenated DataFrame if generator)
    """
    import types

    if isinstance(result, types.GeneratorType):
        chunks = list(result)
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()
    return result


# =============================================================================
# Data generation utilities
# =============================================================================


def create_test_data(
    n_rows: int,
    use_arrow: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create basic test DataFrame with numeric and string columns.

    Parameters
    ----------
    n_rows : int
        Number of rows
    use_arrow : bool, default False
        Whether to use Arrow-backed dtypes
    seed : int, default 42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Test DataFrame with columns: a (int), b (float), c (str)
    """
    rng = np.random.default_rng(seed)

    data = {
        "a": rng.integers(0, 100, n_rows),
        "b": rng.random(n_rows),
        "c": rng.choice(["foo", "bar", "baz"], n_rows),
    }

    df = pd.DataFrame(data)

    if use_arrow:
        df = df.astype(
            {
                "a": "int64[pyarrow]",
                "b": "double[pyarrow]",
                "c": "string[pyarrow]",
            }
        )

    return df


def create_grouped_data(
    n_rows: int,
    n_groups: int = 100,
    use_arrow: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create test data with grouping columns.

    Parameters
    ----------
    n_rows : int
        Number of rows
    n_groups : int, default 100
        Number of unique values in group_a column
    use_arrow : bool, default False
        Whether to use Arrow-backed dtypes
    seed : int, default 42
        Random seed

    Returns
    -------
    pd.DataFrame
        Test DataFrame with columns: group_a, group_b, value1, value2, value3
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "group_a": rng.integers(0, n_groups, n_rows),
            "group_b": rng.choice(["X", "Y", "Z"], n_rows),
            "value1": rng.random(n_rows) * 100,
            "value2": rng.random(n_rows) * 1000,
            "value3": rng.integers(0, 100, n_rows),
        }
    )

    if use_arrow:
        df = df.astype(
            {
                "group_a": "int64[pyarrow]",
                "group_b": "string[pyarrow]",
                "value1": "double[pyarrow]",
                "value2": "double[pyarrow]",
                "value3": "int64[pyarrow]",
            }
        )

    return df


def create_sales_data(
    n_rows: int,
    use_arrow: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create realistic sales data for pipeline benchmarks.

    Parameters
    ----------
    n_rows : int
        Number of rows
    use_arrow : bool, default False
        Whether to use Arrow-backed dtypes
    seed : int, default 42
        Random seed

    Returns
    -------
    pd.DataFrame
        Sales data with columns: order_id, region, category, quantity,
        unit_price, discount, year, month
    """
    rng = np.random.default_rng(seed)

    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Clothing", "Food", "Home", "Sports"]

    df = pd.DataFrame(
        {
            "order_id": np.arange(n_rows),
            "region": rng.choice(regions, n_rows),
            "category": rng.choice(categories, n_rows),
            "quantity": rng.integers(1, 100, n_rows),
            "unit_price": rng.uniform(10, 500, n_rows).round(2),
            "discount": rng.uniform(0, 0.3, n_rows).round(2),
            "year": rng.choice([2022, 2023, 2024], n_rows),
            "month": rng.integers(1, 13, n_rows),
        }
    )

    if use_arrow:
        df = df.astype(
            {
                "order_id": "int64[pyarrow]",
                "region": "string[pyarrow]",
                "category": "string[pyarrow]",
                "quantity": "int64[pyarrow]",
                "unit_price": "double[pyarrow]",
                "discount": "double[pyarrow]",
                "year": "int64[pyarrow]",
                "month": "int64[pyarrow]",
            }
        )

    return df


def create_string_data(
    n_rows: int,
    use_arrow: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create DataFrame with string columns for string operation benchmarks.

    Parameters
    ----------
    n_rows : int
        Number of rows
    use_arrow : bool, default False
        Whether to use Arrow-backed dtypes
    seed : int, default 42
        Random seed

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: text, id
    """
    rng = np.random.default_rng(seed)

    prefixes = ["foo", "bar", "baz", "qux"]
    suffixes = ["_a", "_b", "_c", "_d"]

    strings = [
        rng.choice(prefixes) + str(i % 1000) + rng.choice(suffixes)
        for i in range(n_rows)
    ]

    df = pd.DataFrame({"text": strings, "id": np.arange(n_rows)})

    if use_arrow:
        df = df.astype({"text": "string[pyarrow]", "id": "int64[pyarrow]"})

    return df


def create_join_data(
    n_left: int,
    n_right: int,
    n_keys: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create two DataFrames for join testing.

    Parameters
    ----------
    n_left : int
        Number of rows in left DataFrame
    n_right : int
        Number of rows in right DataFrame
    n_keys : int
        Number of unique key values
    seed : int, default 42
        Random seed

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Left and right DataFrames with key column and value columns
    """
    rng = np.random.default_rng(seed)

    left = pd.DataFrame(
        {
            "key": rng.integers(0, n_keys, n_left),
            "left_val1": rng.random(n_left),
            "left_val2": rng.random(n_left),
        }
    )

    right = pd.DataFrame(
        {
            "key": rng.integers(0, n_keys, n_right),
            "right_val1": rng.random(n_right),
            "right_val2": rng.random(n_right),
        }
    )

    return left, right


def create_wide_data(
    n_rows: int,
    n_cols: int = 50,
    use_arrow: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create wide DataFrame with many columns for projection benchmarks.

    Parameters
    ----------
    n_rows : int
        Number of rows
    n_cols : int, default 50
        Number of columns (plus filter_col)
    use_arrow : bool, default False
        Whether to use Arrow-backed dtypes
    seed : int, default 42
        Random seed

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with col_0 to col_{n-1} and filter_col
    """
    rng = np.random.default_rng(seed)

    data = {f"col_{i}": rng.random(n_rows) for i in range(n_cols)}
    data["filter_col"] = rng.random(n_rows)
    df = pd.DataFrame(data)

    if use_arrow:
        df = df.astype(dict.fromkeys(df.columns, "double[pyarrow]"))

    return df


# =============================================================================
# Output utilities
# =============================================================================


def print_header(title: str, width: int = 70) -> None:
    """Print a benchmark section header."""
    print("=" * width)
    print(title)
    print("=" * width)
    print()


def print_subheader(title: str, width: int = 70) -> None:
    """Print a benchmark subsection header."""
    print(f"\n{'-' * 3} {title} {'-' * 3}")


def print_result(
    label: str,
    mean_ms: float,
    std_ms: float,
    width: int = 22,
) -> None:
    """Print a single benchmark result line."""
    print(f"{label:<{width}} {mean_ms:8.2f} ms ± {std_ms:.2f} ms")


def print_speedup(
    baseline_ms: float,
    test_ms: float,
    label: str = "Speedup",
) -> None:
    """Print speedup ratio."""
    speedup = baseline_ms / test_ms if test_ms > 0 else float("inf")
    print(f"  {label}: {speedup:.2f}x")
