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


# =============================================================================
# Hardware info collection
# =============================================================================


def get_hardware_info() -> dict[str, Any]:
    """
    Collect hardware and environment info for benchmark reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary with system info including:
        - platform, python_version, numpy_version, pandas_version
        - cpu_count, cpu_brand (if available)
        - memory_total_gb
        - timestamp
    """
    from datetime import (
        datetime,
        timezone,
    )
    import platform
    import sys

    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cpu_count": None,
        "cpu_brand": None,
        "memory_total_gb": None,
    }

    # CPU count
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    # CPU brand (macOS/Linux)
    try:
        import subprocess

        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["cpu_brand"] = result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_brand"] = line.split(":")[1].strip()
                        break
    except Exception:
        pass

    # Total memory
    try:
        if platform.system() == "Darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["memory_total_gb"] = round(int(result.stdout.strip()) / 1e9, 1)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["memory_total_gb"] = round(kb / 1e6, 1)
                        break
    except Exception:
        pass

    return info


# =============================================================================
# JSON result utilities
# =============================================================================


def save_benchmark_results(
    results: dict[str, Any],
    output_path: Path | str,
    include_hardware: bool = True,
) -> Path:
    """
    Save benchmark results to JSON with optional hardware info.

    Parameters
    ----------
    results : dict[str, Any]
        Benchmark results to save
    output_path : Path or str
        Path to output JSON file
    include_hardware : bool, default True
        Whether to include hardware info in output

    Returns
    -------
    Path
        Path to saved file
    """
    import json

    output_path = Path(output_path)

    output = {
        "results": results,
    }

    if include_hardware:
        output["hardware"] = get_hardware_info()

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output_path


def load_benchmark_results(input_path: Path | str) -> dict[str, Any]:
    """
    Load benchmark results from JSON.

    Parameters
    ----------
    input_path : Path or str
        Path to input JSON file

    Returns
    -------
    dict[str, Any]
        Loaded benchmark results
    """
    import json

    with open(input_path) as f:
        return json.load(f)


# =============================================================================
# CI baseline comparison
# =============================================================================


def compare_to_baseline(
    current: dict[str, float],
    baseline: dict[str, float],
    threshold_pct: float = 10.0,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Compare current results to baseline and detect regressions.

    Parameters
    ----------
    current : dict[str, float]
        Current benchmark results (metric_name -> value_ms)
    baseline : dict[str, float]
        Baseline benchmark results (metric_name -> value_ms)
    threshold_pct : float, default 10.0
        Regression threshold as percentage (fail if slower by more than this)

    Returns
    -------
    tuple[bool, list[dict]]
        (passed, comparisons) where passed is False if any regression detected
        comparisons is a list of dicts with metric, baseline, current, diff_pct, status
    """
    comparisons = []
    passed = True

    for metric in sorted(set(current.keys()) & set(baseline.keys())):
        curr_val = current[metric]
        base_val = baseline[metric]

        if base_val > 0:
            diff_pct = ((curr_val - base_val) / base_val) * 100
        else:
            diff_pct = 0.0

        if diff_pct > threshold_pct:
            status = "REGRESSION"
            passed = False
        elif diff_pct < -threshold_pct:
            status = "IMPROVEMENT"
        else:
            status = "OK"

        comparisons.append(
            {
                "metric": metric,
                "baseline_ms": base_val,
                "current_ms": curr_val,
                "diff_pct": diff_pct,
                "status": status,
            }
        )

    return passed, comparisons


def print_comparison_table(comparisons: list[dict[str, Any]]) -> None:
    """Print comparison results as a table."""
    print(
        f"\n{'Metric':<40} {'Baseline':>10} {'Current':>10} {'Diff':>10} {'Status':>12}"
    )
    print("-" * 85)

    for c in comparisons:
        diff_str = f"{c['diff_pct']:+.1f}%"
        status_color = (
            "❌"
            if c["status"] == "REGRESSION"
            else "✅"
            if c["status"] == "IMPROVEMENT"
            else "  "
        )
        print(
            f"{c['metric']:<40} {c['baseline_ms']:>10.2f} {c['current_ms']:>10.2f} "
            f"{diff_str:>10} {status_color} {c['status']}"
        )


def run_ci_comparison(
    current_results: dict[str, float],
    baseline_path: Path | str,
    threshold_pct: float = 10.0,
    output_path: Path | str | None = None,
) -> bool:
    """
    Run CI comparison against baseline file.

    Parameters
    ----------
    current_results : dict[str, float]
        Current benchmark results
    baseline_path : Path or str
        Path to baseline JSON file
    threshold_pct : float, default 10.0
        Regression threshold percentage
    output_path : Path or str, optional
        Path to save comparison results

    Returns
    -------
    bool
        True if all checks passed (no regressions), False otherwise
    """
    baseline_path = Path(baseline_path)

    if not baseline_path.exists():
        print(f"⚠️  No baseline found at {baseline_path}, saving current as baseline")
        save_benchmark_results({"metrics": current_results}, baseline_path)
        return True

    baseline_data = load_benchmark_results(baseline_path)
    baseline_metrics = baseline_data.get("results", {}).get("metrics", {})

    if not baseline_metrics:
        print("⚠️  Baseline file has no metrics, saving current as baseline")
        save_benchmark_results({"metrics": current_results}, baseline_path)
        return True

    passed, comparisons = compare_to_baseline(
        current_results, baseline_metrics, threshold_pct
    )

    print_comparison_table(comparisons)

    if output_path:
        save_benchmark_results(
            {
                "metrics": current_results,
                "comparisons": comparisons,
                "passed": passed,
                "threshold_pct": threshold_pct,
            },
            output_path,
        )

    if passed:
        print(f"\n✅ All benchmarks within {threshold_pct}% threshold")
    else:
        regressions = [c for c in comparisons if c["status"] == "REGRESSION"]
        print(f"\n❌ {len(regressions)} regression(s) detected (>{threshold_pct}%)")

    return passed
