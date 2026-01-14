#!/usr/bin/env python
"""
Benchmark: Select/Project operations

Compares lazy vs eager execution for column selection and computed columns.
"""

from collections.abc import Callable
import time

import numpy as np

import pandas as pd


def timeit(func: Callable, n_runs: int = 5, warmup: int = 1) -> tuple[float, float]:
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


def create_test_data(
    n_rows: int, n_cols: int = 10, use_arrow: bool = False
) -> pd.DataFrame:
    """Create test DataFrame with specified size."""
    rng = np.random.default_rng(42)

    data = {f"col_{i}": rng.random(n_rows) for i in range(n_cols)}
    df = pd.DataFrame(data)

    if use_arrow:
        df = df.astype(dict.fromkeys(df.columns, "double[pyarrow]"))

    return df


def bench_eager_select_few(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: select 3 columns from 10."""
    return df[["col_0", "col_1", "col_2"]]


def bench_lazy_select_few(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: select 3 columns from 10."""
    lf = df.select("col_0", "col_1", "col_2")
    return lf.collect(use_physical_planner=use_physical_planner)


def bench_eager_computed_column(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: add computed column."""
    result = df.copy()
    result["computed"] = df["col_0"] + df["col_1"] * 2
    return result


def bench_lazy_computed_column(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: add computed column."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns((col("col_0") + col("col_1") * 2).alias("computed"))
    return lf.collect(use_physical_planner=use_physical_planner)


def bench_eager_multiple_computed(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: add multiple computed columns."""
    result = df.copy()
    result["sum_01"] = df["col_0"] + df["col_1"]
    result["diff_23"] = df["col_2"] - df["col_3"]
    result["prod_45"] = df["col_4"] * df["col_5"]
    return result


def bench_lazy_multiple_computed(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: add multiple computed columns."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(
        (col("col_0") + col("col_1")).alias("sum_01"),
        (col("col_2") - col("col_3")).alias("diff_23"),
        (col("col_4") * col("col_5")).alias("prod_45"),
    )
    return lf.collect(use_physical_planner=use_physical_planner)


def bench_eager_filter_then_select(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: filter then select (not optimized)."""
    result = df[df["col_0"] > 0.5]
    return result[["col_0", "col_1", "col_2"]]


def bench_lazy_filter_then_select(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: filter then select (projection pushdown possible)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("col_0") > 0.5)
    lf = lf.select("col_0", "col_1", "col_2")
    return lf.collect(use_physical_planner=use_physical_planner)


def run_benchmarks():
    """Run all select benchmarks."""
    sizes = [10_000, 100_000, 1_000_000]

    print("=" * 70)
    print("SELECT/PROJECT BENCHMARKS")
    print("=" * 70)

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows x 10 columns")
        print("=" * 70)

        df_numpy = create_test_data(n_rows, use_arrow=False)
        df_arrow = create_test_data(n_rows, use_arrow=True)

        # Column selection
        print("\n--- Select 3 columns from 10 ---")

        mean, std = timeit(lambda: bench_eager_select_few(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_select_few(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_select_few(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_select_few(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_select_few(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_select_few(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Single computed column
        print("\n--- Add computed column (col_0 + col_1 * 2) ---")

        mean, std = timeit(lambda: bench_eager_computed_column(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_computed_column(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_computed_column(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_computed_column(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_computed_column(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_computed_column(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Multiple computed columns
        print("\n--- Add 3 computed columns ---")

        mean, std = timeit(lambda: bench_eager_multiple_computed(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_multiple_computed(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_multiple_computed(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_multiple_computed(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_multiple_computed(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_multiple_computed(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Filter then select
        print("\n--- Filter then select (col_0 > 0.5, select 3 cols) ---")

        mean, std = timeit(lambda: bench_eager_filter_then_select(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_filter_then_select(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_filter_then_select(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_filter_then_select(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_filter_then_select(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: bench_lazy_filter_then_select(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")


if __name__ == "__main__":
    run_benchmarks()
