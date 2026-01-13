#!/usr/bin/env python
"""
Benchmark: Filter operations

Compares lazy vs eager execution for filter/where operations.
"""

from collections.abc import Callable
import time

import numpy as np

import pandas as pd


def timeit(func: Callable, n_runs: int = 5, warmup: int = 1) -> tuple[float, float]:
    """Run function multiple times and return (mean, std) in milliseconds."""
    # Warmup
    for _ in range(warmup):
        func()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def create_test_data(n_rows: int, use_arrow: bool = False) -> pd.DataFrame:
    """Create test DataFrame with specified size."""
    rng = np.random.default_rng(42)

    data = {
        "a": rng.integers(0, 100, n_rows),
        "b": rng.random(n_rows),
        "c": rng.choice(["foo", "bar", "baz"], n_rows),
    }

    df = pd.DataFrame(data)

    if use_arrow:
        # Convert to Arrow-backed dtypes
        df = df.astype(
            {
                "a": "int64[pyarrow]",
                "b": "double[pyarrow]",
                "c": "string[pyarrow]",
            }
        )

    return df


def bench_eager_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager filter: execute immediately."""
    return df[df["a"] > 50]


def bench_lazy_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy filter: build plan, then execute."""
    from pandas.lazy import col

    lf = df.select()  # Enter lazy mode
    lf = lf.filter(col("a") > 50)
    return lf.collect()


def bench_eager_chained_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager chained filters."""
    result = df[df["a"] > 50]
    result = result[result["b"] < 0.5]
    return result


def bench_lazy_chained_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy chained filters (should be optimized to single filter)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 50)
    lf = lf.filter(col("b") < 0.5)
    return lf.collect()


def run_benchmarks():
    """Run all filter benchmarks."""
    sizes = [10_000, 100_000, 1_000_000]

    print("=" * 70)
    print("FILTER BENCHMARKS")
    print("=" * 70)
    print()

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows")
        print("=" * 70)

        # NumPy-backed DataFrame
        df_numpy = create_test_data(n_rows, use_arrow=False)

        # Arrow-backed DataFrame
        df_arrow = create_test_data(n_rows, use_arrow=True)

        # Simple filter
        print("\n--- Simple Filter (a > 50) ---")

        mean, std = timeit(lambda: bench_eager_filter(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_filter(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_filter(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_filter(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Chained filters
        print("\n--- Chained Filters (a > 50 AND b < 0.5) ---")

        mean, std = timeit(lambda: bench_eager_chained_filter(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_chained_filter(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_chained_filter(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_chained_filter(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")


if __name__ == "__main__":
    run_benchmarks()
