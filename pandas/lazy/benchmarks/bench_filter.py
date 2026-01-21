#!/usr/bin/env python
"""
Benchmark: Filter operations

Compares lazy vs eager execution for filter/where operations.
"""

import pandas as pd
from pandas.lazy.benchmarks.shared import (
    create_test_data,
    timeit,
)


def bench_eager_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager filter: execute immediately."""
    return df[df["a"] > 50]


def bench_lazy_filter(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy filter: build plan, then execute."""
    from pandas.lazy import col

    lf = df.select()  # Enter lazy mode
    lf = lf.filter(col("a") > 50)
    return lf.collect(use_physical_planner=use_physical_planner)


def bench_eager_chained_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager chained filters."""
    result = df[df["a"] > 50]
    result = result[result["b"] < 0.5]
    return result


def bench_lazy_chained_filter(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy chained filters (should be optimized to single filter)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 50)
    lf = lf.filter(col("b") < 0.5)
    return lf.collect(use_physical_planner=use_physical_planner)


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
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_filter(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_filter(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_filter(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_filter(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_filter(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Chained filters
        print("\n--- Chained Filters (a > 50 AND b < 0.5) ---")

        mean, std = timeit(lambda: bench_eager_chained_filter(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_chained_filter(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_chained_filter(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_chained_filter(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_chained_filter(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_chained_filter(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")


if __name__ == "__main__":
    run_benchmarks()
