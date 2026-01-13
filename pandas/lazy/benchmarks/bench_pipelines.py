#!/usr/bin/env python
"""
Benchmark: Complex multi-step pipelines

Tests realistic query patterns that chain multiple operations.
These scenarios benefit most from lazy execution optimizations.
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


def create_sales_data(n_rows: int, use_arrow: bool = False) -> pd.DataFrame:
    """Create realistic sales data."""
    rng = np.random.default_rng(42)

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


# =============================================================================
# Pipeline 1: Filter -> Compute -> Filter
# =============================================================================


def eager_pipeline_filter_compute_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Filter by region, compute total, filter by amount."""
    # Step 1: Filter by region
    result = df[df["region"] == "North"]
    # Step 2: Compute total price
    result = result.copy()
    result["total"] = (
        result["quantity"] * result["unit_price"] * (1 - result["discount"])
    )
    # Step 3: Filter high-value orders
    result = result[result["total"] > 1000]
    return result


def lazy_pipeline_filter_compute_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Filter by region, compute total, filter by amount."""
    from pandas.lazy import col

    lf = df.select()
    # Step 1: Filter by region
    lf = lf.filter(col("region") == "North")
    # Step 2: Compute total price
    lf = lf.with_columns(
        (col("quantity") * col("unit_price") * (1 - col("discount"))).alias("total")
    )
    # Step 3: Filter high-value orders
    lf = lf.filter(col("total") > 1000)
    return lf.collect()


# =============================================================================
# Pipeline 2: Multiple filters with OR conditions
# =============================================================================


def eager_pipeline_complex_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Complex filter with multiple conditions."""
    mask = (
        ((df["region"] == "North") & (df["category"] == "Electronics"))
        | ((df["region"] == "South") & (df["unit_price"] > 200))
        | ((df["year"] == 2024) & (df["quantity"] > 50))
    )
    return df[mask]


def lazy_pipeline_complex_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Complex filter with multiple conditions."""
    from pandas.lazy import col

    lf = df.select()
    condition = (
        ((col("region") == "North") & (col("category") == "Electronics"))
        | ((col("region") == "South") & (col("unit_price") > 200))
        | ((col("year") == 2024) & (col("quantity") > 50))
    )
    lf = lf.filter(condition)
    return lf.collect()


# =============================================================================
# Pipeline 3: Compute multiple derived columns
# =============================================================================


def eager_pipeline_multi_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Compute multiple derived columns."""
    result = df.copy()
    result["gross_total"] = result["quantity"] * result["unit_price"]
    result["discount_amount"] = result["gross_total"] * result["discount"]
    result["net_total"] = result["gross_total"] - result["discount_amount"]
    result["avg_unit_net"] = result["net_total"] / result["quantity"]
    result["is_high_value"] = result["net_total"] > 500
    return result


def lazy_pipeline_multi_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Compute multiple derived columns."""
    from pandas.lazy import col

    lf = df.select()
    gross = col("quantity") * col("unit_price")
    discount_amt = gross * col("discount")
    net = gross - discount_amt

    lf = lf.with_columns(
        gross.alias("gross_total"),
        discount_amt.alias("discount_amount"),
        net.alias("net_total"),
        (net / col("quantity")).alias("avg_unit_net"),
        (net > 500).alias("is_high_value"),
    )
    return lf.collect()


# =============================================================================
# Pipeline 4: Filter -> Select subset -> Compute
# =============================================================================


def eager_pipeline_filter_select_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Filter, select columns, then compute."""
    # Filter
    result = df[df["year"] == 2024]
    # Select subset of columns
    result = result[["region", "category", "quantity", "unit_price", "discount"]]
    # Compute
    result = result.copy()
    result["revenue"] = (
        result["quantity"] * result["unit_price"] * (1 - result["discount"])
    )
    return result


def lazy_pipeline_filter_select_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Filter, select columns, then compute."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("year") == 2024)
    lf = lf.select("region", "category", "quantity", "unit_price", "discount")
    lf = lf.with_columns(
        (col("quantity") * col("unit_price") * (1 - col("discount"))).alias("revenue")
    )
    return lf.collect()


# =============================================================================
# Pipeline 5: Chained string operations with filter
# =============================================================================


def eager_pipeline_string_ops(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: String operations with filtering."""
    result = df.copy()
    # Uppercase region
    result["region_upper"] = result["region"].str.upper()
    # Check category prefix
    result["is_e_category"] = result["category"].str.startswith("E")
    # Filter
    result = result[result["is_e_category"]]
    # More string ops
    result = result.copy()
    result["region_cat"] = result["region_upper"] + "_" + result["category"]
    return result


def lazy_pipeline_string_ops(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: String operations with filtering."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(
        col("region").str.upper().alias("region_upper"),
        col("category").str.startswith("E").alias("is_e_category"),
    )
    lf = lf.filter(col("is_e_category"))
    lf = lf.with_columns(
        (col("region_upper") + "_" + col("category")).alias("region_cat")
    )
    return lf.collect()


# =============================================================================
# Pipeline 6: Sequential filters (filter fusion opportunity)
# =============================================================================


def eager_pipeline_sequential_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Multiple sequential filters."""
    result = df[df["region"] == "North"]
    result = result[result["year"] == 2024]
    result = result[result["quantity"] > 20]
    result = result[result["unit_price"] > 100]
    result = result[result["discount"] < 0.2]
    return result


def lazy_pipeline_sequential_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Multiple sequential filters (fusion opportunity)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("region") == "North")
    lf = lf.filter(col("year") == 2024)
    lf = lf.filter(col("quantity") > 20)
    lf = lf.filter(col("unit_price") > 100)
    lf = lf.filter(col("discount") < 0.2)
    return lf.collect()


# =============================================================================
# Pipeline 7: Wide to narrow projection
# =============================================================================


def create_wide_data(
    n_rows: int, n_cols: int = 50, use_arrow: bool = False
) -> pd.DataFrame:
    """Create wide DataFrame with many columns."""
    rng = np.random.default_rng(42)

    data = {f"col_{i}": rng.random(n_rows) for i in range(n_cols)}
    data["filter_col"] = rng.random(n_rows)
    df = pd.DataFrame(data)

    if use_arrow:
        df = df.astype(dict.fromkeys(df.columns, "double[pyarrow]"))

    return df


def eager_pipeline_wide_to_narrow(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Filter wide DataFrame, select few columns."""
    result = df[df["filter_col"] > 0.5]
    return result[["col_0", "col_1", "col_2"]]


def lazy_pipeline_wide_to_narrow(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Filter wide DataFrame, select few columns (projection pushdown)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("filter_col") > 0.5)
    lf = lf.select("col_0", "col_1", "col_2")
    return lf.collect()


def run_benchmarks():
    """Run all pipeline benchmarks."""
    sizes = [10_000, 100_000, 1_000_000]

    print("=" * 70)
    print("COMPLEX PIPELINE BENCHMARKS")
    print("=" * 70)
    print()
    print("Tests realistic multi-step query patterns.")
    print()

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows")
        print("=" * 70)

        df_numpy = create_sales_data(n_rows, use_arrow=False)
        df_arrow = create_sales_data(n_rows, use_arrow=True)

        # Pipeline 1: Filter -> Compute -> Filter
        print("\n--- Pipeline 1: Filter → Compute → Filter ---")

        mean, std = timeit(lambda: eager_pipeline_filter_compute_filter(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_filter_compute_filter(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_pipeline_filter_compute_filter(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_filter_compute_filter(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Pipeline 2: Complex filter
        print("\n--- Pipeline 2: Complex OR filter ---")

        mean, std = timeit(lambda: eager_pipeline_complex_filter(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_complex_filter(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_pipeline_complex_filter(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_complex_filter(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Pipeline 3: Multiple computed columns
        print("\n--- Pipeline 3: 5 computed columns ---")

        mean, std = timeit(lambda: eager_pipeline_multi_compute(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_multi_compute(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_pipeline_multi_compute(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_multi_compute(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Pipeline 4: Filter -> Select -> Compute
        print("\n--- Pipeline 4: Filter → Select → Compute ---")

        mean, std = timeit(lambda: eager_pipeline_filter_select_compute(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_filter_select_compute(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_pipeline_filter_select_compute(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_filter_select_compute(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Pipeline 5: String operations
        print("\n--- Pipeline 5: String ops + filter ---")

        mean, std = timeit(lambda: eager_pipeline_string_ops(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_string_ops(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_pipeline_string_ops(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_string_ops(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Pipeline 6: Sequential filters
        print("\n--- Pipeline 6: 5 sequential filters (fusion) ---")

        mean, std = timeit(lambda: eager_pipeline_sequential_filters(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_sequential_filters(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_pipeline_sequential_filters(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_sequential_filters(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Pipeline 7: Wide to narrow
        print("\n--- Pipeline 7: Wide (50 cols) → Filter → Select 3 ---")

        df_wide_numpy = create_wide_data(n_rows, n_cols=50, use_arrow=False)
        df_wide_arrow = create_wide_data(n_rows, n_cols=50, use_arrow=True)

        mean, std = timeit(lambda: eager_pipeline_wide_to_narrow(df_wide_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_wide_to_narrow(df_wide_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_pipeline_wide_to_narrow(df_wide_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_pipeline_wide_to_narrow(df_wide_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")


if __name__ == "__main__":
    run_benchmarks()
