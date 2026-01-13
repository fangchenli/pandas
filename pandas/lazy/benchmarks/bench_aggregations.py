#!/usr/bin/env python
"""
Benchmark: Aggregation operations

Tests aggregations with filter pipelines.
Note: The lazy API returns DataFrames, so aggregations that return scalars
work differently than in eager mode.
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


def create_grouped_data(
    n_rows: int, n_groups: int = 100, use_arrow: bool = False
) -> pd.DataFrame:
    """Create data with grouping columns."""
    rng = np.random.default_rng(42)

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


# =============================================================================
# Filter then compute (common aggregation-like pipeline)
# =============================================================================


def eager_filter_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Filter then compute derived values."""
    result = df[df["value1"] > 50].copy()
    result["derived"] = result["value1"] * result["value2"] / 100
    return result


def lazy_filter_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Filter then compute derived values."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("value1") > 50)
    lf = lf.with_columns((col("value1") * col("value2") / 100).alias("derived"))
    return lf.collect()


# =============================================================================
# Multiple filters + compute
# =============================================================================


def eager_multi_filter_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Multiple filters then compute."""
    result = df[df["group_b"] == "X"]
    result = result[result["value1"] > 30]
    result = result[result["value2"] < 800]
    result = result.copy()
    result["score"] = result["value1"] + result["value2"] * 0.1
    return result


def lazy_multi_filter_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Multiple filters then compute (filter fusion)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("group_b") == "X")
    lf = lf.filter(col("value1") > 30)
    lf = lf.filter(col("value2") < 800)
    lf = lf.with_columns((col("value1") + col("value2") * 0.1).alias("score"))
    return lf.collect()


# =============================================================================
# Compute then filter
# =============================================================================


def eager_compute_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Compute then filter on computed value."""
    result = df.copy()
    result["ratio"] = result["value1"] / (result["value2"] + 1)
    return result[result["ratio"] > 0.1]


def lazy_compute_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Compute then filter on computed value."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns((col("value1") / (col("value2") + 1)).alias("ratio"))
    lf = lf.filter(col("ratio") > 0.1)
    return lf.collect()


# =============================================================================
# Complex multi-step pipeline
# =============================================================================


def eager_complex_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Multi-step ETL-like pipeline."""
    # Step 1: Filter by group
    result = df[df["group_b"].isin(["X", "Y"])]
    # Step 2: Compute multiple columns
    result = result.copy()
    result["total"] = result["value1"] + result["value2"]
    result["weighted"] = result["value1"] * result["value3"]
    # Step 3: Filter by computed
    result = result[result["total"] > 500]
    # Step 4: Final computation
    result = result.copy()
    result["final_score"] = result["weighted"] / result["total"]
    return result


def lazy_complex_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Multi-step ETL-like pipeline."""
    from pandas.lazy import col

    lf = df.select()
    # Step 1: Filter by group (using OR since is_in may not exist)
    lf = lf.filter((col("group_b") == "X") | (col("group_b") == "Y"))
    # Step 2: Compute multiple columns
    total = col("value1") + col("value2")
    weighted = col("value1") * col("value3")
    lf = lf.with_columns(
        total.alias("total"),
        weighted.alias("weighted"),
    )
    # Step 3: Filter by computed
    lf = lf.filter(col("total") > 500)
    # Step 4: Final computation
    lf = lf.with_columns((col("weighted") / col("total")).alias("final_score"))
    return lf.collect()


# =============================================================================
# Select subset + compute
# =============================================================================


def eager_select_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Select columns then compute."""
    result = df[["value1", "value2", "value3"]].copy()
    result["combined"] = result["value1"] * result["value2"] + result["value3"]
    return result


def lazy_select_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Select columns then compute."""
    from pandas.lazy import col

    lf = df.select("value1", "value2", "value3")
    lf = lf.with_columns(
        (col("value1") * col("value2") + col("value3")).alias("combined")
    )
    return lf.collect()


# =============================================================================
# Multiple computed columns in one pass
# =============================================================================


def eager_multi_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Compute multiple columns."""
    result = df.copy()
    result["sum_12"] = result["value1"] + result["value2"]
    result["diff_12"] = result["value1"] - result["value2"]
    result["prod_13"] = result["value1"] * result["value3"]
    result["ratio_23"] = result["value2"] / (result["value3"] + 1)
    result["flag"] = result["value1"] > 50
    return result


def lazy_multi_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Compute multiple columns in one with_columns."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(
        (col("value1") + col("value2")).alias("sum_12"),
        (col("value1") - col("value2")).alias("diff_12"),
        (col("value1") * col("value3")).alias("prod_13"),
        (col("value2") / (col("value3") + 1)).alias("ratio_23"),
        (col("value1") > 50).alias("flag"),
    )
    return lf.collect()


def run_benchmarks():
    """Run all aggregation-like benchmarks."""
    sizes = [10_000, 100_000, 1_000_000]

    print("=" * 70)
    print("AGGREGATION-LIKE PIPELINE BENCHMARKS")
    print("=" * 70)
    print()

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows")
        print("=" * 70)

        df_numpy = create_grouped_data(n_rows, use_arrow=False)
        df_arrow = create_grouped_data(n_rows, use_arrow=True)

        # Filter then compute
        print("\n--- Filter → Compute ---")

        mean, std = timeit(lambda: eager_filter_compute(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_filter_compute(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_filter_compute(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_filter_compute(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Multiple filters + compute
        print("\n--- 3 Filters → Compute (fusion opportunity) ---")

        mean, std = timeit(lambda: eager_multi_filter_compute(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_multi_filter_compute(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_multi_filter_compute(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_multi_filter_compute(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Compute then filter
        print("\n--- Compute → Filter ---")

        mean, std = timeit(lambda: eager_compute_filter(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_compute_filter(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_compute_filter(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_compute_filter(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Complex pipeline
        print("\n--- Complex: Filter → Compute → Filter → Compute ---")

        mean, std = timeit(lambda: eager_complex_pipeline(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_complex_pipeline(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_complex_pipeline(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_complex_pipeline(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Select + compute
        print("\n--- Select 3 cols → Compute ---")

        mean, std = timeit(lambda: eager_select_compute(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_select_compute(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_select_compute(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_select_compute(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Multi-compute
        print("\n--- 5 computed columns in one pass ---")

        mean, std = timeit(lambda: eager_multi_compute(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_multi_compute(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_multi_compute(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_multi_compute(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")


if __name__ == "__main__":
    run_benchmarks()
