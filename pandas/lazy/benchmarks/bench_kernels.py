#!/usr/bin/env python
"""
Benchmark: Vectorized kernel operations.

Tests performance of vectorized lazy pandas kernels vs eager pandas:
- Join kernel (NumPy hash_join vs DataFrame.merge)
- Rolling window functions (sliding_window_view vs rolling)
- Cumulative functions (vectorized vs pandas cumsum/cummin/cummax)
- Fill functions (vectorized ffill/bfill vs pandas)
"""

from collections.abc import Callable
import time

import numpy as np

import pandas as pd


def timeit(func: Callable, n_runs: int = 7, warmup: int = 3) -> tuple[float, float]:
    """Run function multiple times and return (mean, std) in milliseconds."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def print_comparison(name: str, eager_time: float, lazy_time: float) -> None:
    """Print comparison with speedup calculation."""
    speedup = eager_time / lazy_time if lazy_time > 0 else float("inf")
    winner = "Lazy" if speedup > 1 else "Eager"
    print(f"  {name}:")
    print(f"    Eager: {eager_time:8.2f} ms")
    print(f"    Lazy:  {lazy_time:8.2f} ms")
    print(f"    Speedup: {speedup:.2f}x ({winner} wins)")


# =============================================================================
# Join Kernel Benchmarks
# =============================================================================


def create_join_data(
    n_left: int, n_right: int, n_keys: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create two DataFrames for join testing."""
    rng = np.random.default_rng(42)

    left = pd.DataFrame(
        {
            "key": rng.integers(0, n_keys, n_left),
            "left_val1": rng.random(n_left),
            "left_val2": rng.random(n_left),
            "left_int": rng.integers(0, 1000, n_left),
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


def benchmark_joins(n_rows: int) -> None:
    """Benchmark join operations with NumPy kernel."""
    print(f"\n--- JOIN OPERATIONS ({n_rows:,} rows) ---")

    n_keys = n_rows // 10  # 10% unique keys
    left, right = create_join_data(n_rows, n_rows // 2, n_keys)

    print(f"Left: {len(left):,} rows, Right: {len(right):,} rows, {n_keys} unique keys")

    join_types = ["inner", "left", "right", "outer"]

    for jtype in join_types:
        # Lazy (uses NumPy kernel now)
        def lazy_join(how=jtype):
            return left.select().join(right.select(), on="key", how=how).collect()

        # Eager
        def eager_join(how=jtype):
            return left.merge(right, on="key", how=how)

        lazy_mean, _ = timeit(lazy_join)
        eager_mean, _ = timeit(eager_join)

        print_comparison(f"{jtype.capitalize()} Join", eager_mean, lazy_mean)


def benchmark_multi_key_joins(n_rows: int) -> None:
    """Benchmark multi-key join operations."""
    print(f"\n--- MULTI-KEY JOIN ({n_rows:,} rows) ---")

    rng = np.random.default_rng(42)
    n_keys = n_rows // 20

    left = pd.DataFrame(
        {
            "key1": rng.integers(0, n_keys, n_rows),
            "key2": rng.choice(["A", "B", "C", "D", "E"], n_rows),
            "value": rng.random(n_rows),
        }
    )

    right_size = n_rows // 3
    right = pd.DataFrame(
        {
            "key1": rng.integers(0, n_keys, right_size),
            "key2": rng.choice(["A", "B", "C", "D", "E"], right_size),
            "score": rng.random(right_size),
        }
    )

    print(f"Left: {len(left):,} rows, Right: {len(right):,} rows")

    # Lazy
    def lazy_join():
        return left.select().join(right.select(), on=["key1", "key2"]).collect()

    # Eager
    def eager_join():
        return left.merge(right, on=["key1", "key2"])

    lazy_mean, _ = timeit(lazy_join)
    eager_mean, _ = timeit(eager_join)

    print_comparison("Multi-key Inner Join", eager_mean, lazy_mean)


# =============================================================================
# Rolling Window Kernel Benchmarks
# =============================================================================


def create_rolling_data(n_rows: int, na_ratio: float = 0.05) -> np.ndarray:
    """Create test array for rolling window benchmarks."""
    rng = np.random.default_rng(42)

    values = rng.random(n_rows) * 100
    na_mask = rng.random(n_rows) < na_ratio
    values[na_mask] = np.nan

    return values


def benchmark_rolling(n_rows: int, window: int = 50) -> None:
    """Benchmark rolling window operations."""
    print(f"\n--- ROLLING OPERATIONS ({n_rows:,} rows, window={window}) ---")

    arr = create_rolling_data(n_rows)
    series = pd.Series(arr)

    from pandas.lazy.backends import dispatch_kernel

    operations = [
        ("rolling_sum", "sum"),
        ("rolling_mean", "mean"),
        ("rolling_min", "min"),
        ("rolling_max", "max"),
        ("rolling_std", "std"),
    ]

    for kernel_name, agg_name in operations:
        # Lazy (vectorized kernel)
        def lazy_rolling(kn=kernel_name):
            return dispatch_kernel(kn, "numpy", arr, window)

        # Eager
        def eager_rolling(agg=agg_name):
            return getattr(series.rolling(window), agg)()

        lazy_mean, _ = timeit(lazy_rolling)
        eager_mean, _ = timeit(eager_rolling)

        print_comparison(kernel_name, eager_mean, lazy_mean)


# =============================================================================
# Cumulative Function Benchmarks
# =============================================================================


def benchmark_cumulative(n_rows: int) -> None:
    """Benchmark cumulative operations."""
    print(f"\n--- CUMULATIVE OPERATIONS ({n_rows:,} rows) ---")

    rng = np.random.default_rng(42)
    values = rng.random(n_rows) * 100
    # Add some NaN
    na_mask = rng.random(n_rows) < 0.02
    values[na_mask] = np.nan

    series = pd.Series(values)

    from pandas.lazy.backends import dispatch_kernel

    operations = [
        ("cumulative_sum", "cumsum"),
        ("cumulative_min", "cummin"),
        ("cumulative_max", "cummax"),
    ]

    for kernel_name, pandas_method in operations:
        # Lazy (vectorized kernel)
        def lazy_cumulative(kn=kernel_name):
            return dispatch_kernel(kn, "numpy", values)

        # Eager
        def eager_cumulative(method=pandas_method):
            return getattr(series, method)()

        lazy_mean, _ = timeit(lazy_cumulative)
        eager_mean, _ = timeit(eager_cumulative)

        print_comparison(kernel_name, eager_mean, lazy_mean)


# =============================================================================
# Fill Function Benchmarks
# =============================================================================


def benchmark_fill(n_rows: int) -> None:
    """Benchmark ffill/bfill operations."""
    print(f"\n--- FILL OPERATIONS ({n_rows:,} rows) ---")

    rng = np.random.default_rng(42)
    values = rng.random(n_rows) * 100
    # Add significant NaN (~10%)
    na_mask = rng.random(n_rows) < 0.10
    values[na_mask] = np.nan

    series = pd.Series(values)

    from pandas.lazy.backends import dispatch_kernel

    operations = ["ffill", "bfill"]

    for op_name in operations:
        # Lazy (vectorized kernel)
        def lazy_fill(op=op_name):
            return dispatch_kernel(op, "numpy", values.copy())

        # Eager
        def eager_fill(op=op_name):
            return getattr(series, op)()

        lazy_mean, _ = timeit(lazy_fill)
        eager_mean, _ = timeit(eager_fill)

        print_comparison(op_name, eager_mean, lazy_mean)


# =============================================================================
# Main
# =============================================================================


def run_benchmarks():
    """Run all kernel benchmarks."""
    print("=" * 80)
    print("VECTORIZED KERNEL BENCHMARKS")
    print("=" * 80)
    print()
    print("Compares vectorized lazy pandas kernels vs eager pandas operations.")
    print()

    sizes = [100_000, 500_000, 1_000_000]

    for n_rows in sizes:
        print(f"\n{'=' * 80}")
        print(f"Dataset Size: {n_rows:,} rows")
        print("=" * 80)

        # Join benchmarks
        benchmark_joins(n_rows)
        benchmark_multi_key_joins(n_rows)

        # Rolling benchmarks
        benchmark_rolling(n_rows)

        # Cumulative benchmarks
        benchmark_cumulative(n_rows)

        # Fill benchmarks
        benchmark_fill(n_rows)

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Vectorized Kernel Performance:

JOINS:
- NumPy hash_join kernel uses vectorized factorize for composite keys
- Avoids DataFrame creation overhead when using physical planner
- Performance competitive with pandas merge

ROLLING WINDOWS:
- Uses numpy sliding_window_view for O(n) performance
- Vectorized aggregation over all windows at once
- Handles NaN with vectorized masks

CUMULATIVE:
- Vectorized np.cumsum/cummin/cummax with NaN handling
- Fills NaN with neutral values before accumulation

FILL (ffill/bfill):
- Uses np.maximum.accumulate trick for index propagation
- O(n) vectorized instead of O(n) loop with conditionals
- Significant speedup over pandas for large arrays

Key Takeaways:
1. Lazy kernels benefit from avoiding intermediate DataFrames
2. NumPy vectorization typically matches or beats pandas
3. Biggest wins are in operations with complex NaN handling
4. Join performance depends heavily on key cardinality
""")


if __name__ == "__main__":
    run_benchmarks()
