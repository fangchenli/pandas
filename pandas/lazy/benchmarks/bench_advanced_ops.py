#!/usr/bin/env python
"""
Benchmark: Advanced operations (GroupBy, Joins, Window functions).

Tests operations that weren't comprehensively benchmarked before.
"""

import numpy as np

import pandas as pd
from pandas.lazy.benchmarks.shared import (
    create_join_data,
    timeit,
)


def create_test_data(n_rows: int, n_groups: int = 100) -> pd.DataFrame:
    """Create test DataFrame with groupable columns."""
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "group_a": rng.integers(0, n_groups, n_rows),
            "group_b": rng.choice(["X", "Y", "Z"], n_rows),
            "value1": rng.random(n_rows) * 100,
            "value2": rng.random(n_rows) * 100,
            "value3": rng.integers(0, 1000, n_rows),
            "timestamp": pd.date_range("2020-01-01", periods=n_rows, freq="s"),
        }
    )


# =============================================================================
# GroupBy Benchmarks
# =============================================================================


def eager_groupby_single_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Single aggregation."""
    return df.groupby("group_a")["value1"].sum().reset_index()


def lazy_groupby_single_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Single aggregation."""
    from pandas.lazy import col

    return (
        df.select()
        .group_by("group_a")
        .agg(col("value1").sum().alias("value1"))
        .collect()
    )


def eager_groupby_multi_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Multiple aggregations."""
    return (
        df.groupby("group_a")
        .agg(
            {
                "value1": "sum",
                "value2": "mean",
                "value3": ["min", "max"],
            }
        )
        .reset_index()
    )


def lazy_groupby_multi_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Multiple aggregations."""
    from pandas.lazy import col

    return (
        df.select()
        .group_by("group_a")
        .agg(
            col("value1").sum().alias("value1_sum"),
            col("value2").mean().alias("value2_mean"),
            col("value3").min().alias("value3_min"),
            col("value3").max().alias("value3_max"),
        )
        .collect()
    )


def eager_groupby_filter_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Filter then groupby."""
    result = df[df["value1"] > 50]
    return result.groupby("group_a")["value2"].sum().reset_index()


def lazy_groupby_filter_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Filter then groupby (predicate pushdown opportunity)."""
    from pandas.lazy import col

    return (
        df.select()
        .filter(col("value1") > 50)
        .group_by("group_a")
        .agg(col("value2").sum().alias("value2"))
        .collect()
    )


def eager_groupby_multi_key(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Groupby with multiple keys."""
    return df.groupby(["group_a", "group_b"])["value1"].sum().reset_index()


def lazy_groupby_multi_key(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Groupby with multiple keys."""
    from pandas.lazy import col

    return (
        df.select()
        .group_by("group_a", "group_b")
        .agg(col("value1").sum().alias("value1"))
        .collect()
    )


# =============================================================================
# Join Benchmarks
# =============================================================================


def eager_inner_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Eager: Inner join."""
    return left.merge(right, on="key", how="inner")


def lazy_inner_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Inner join."""
    return left.select().join(right.select(), on="key", how="inner").collect()


def eager_left_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Eager: Left join."""
    return left.merge(right, on="key", how="left")


def lazy_left_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Left join."""
    return left.select().join(right.select(), on="key", how="left").collect()


def eager_join_then_filter(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Eager: Join then filter."""
    result = left.merge(right, on="key", how="inner")
    return result[result["left_val1"] > 0.5]


def lazy_join_then_filter(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Join then filter (predicate pushdown opportunity)."""
    from pandas.lazy import col

    return (
        left.select()
        .join(right.select(), on="key", how="inner")
        .filter(col("left_val1") > 0.5)
        .collect()
    )


def eager_join_select(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Eager: Join then select columns."""
    result = left.merge(right, on="key", how="inner")
    return result[["key", "left_val1", "right_val1"]]


def lazy_join_select(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Join then select (projection pruning opportunity)."""
    return (
        left.select()
        .join(right.select(), on="key", how="inner")
        .select("key", "left_val1", "right_val1")
        .collect()
    )


# =============================================================================
# Sort/OrderBy Benchmarks
# =============================================================================


def eager_sort_single(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Sort by single column."""
    return df.sort_values("value1")


def lazy_sort_single(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Sort by single column."""
    return df.select().sort("value1").collect()


def eager_sort_multi(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Sort by multiple columns."""
    return df.sort_values(["group_a", "value1"], ascending=[True, False])


def lazy_sort_multi(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Sort by multiple columns."""
    return (
        df.select()
        .sort("group_a", ascending=True)
        .sort("value1", descending=True)
        .collect()
    )


def eager_sort_limit(df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """Eager: Sort then limit (top-k)."""
    return df.sort_values("value1", ascending=False).head(k)


def lazy_sort_limit(df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """Lazy: Sort then limit (TopK optimization)."""
    return df.select().sort("value1", descending=True).limit(k).collect()


def run_benchmarks():
    """Run advanced operations benchmarks."""
    print("=" * 80)
    print("ADVANCED OPERATIONS BENCHMARKS")
    print("=" * 80)
    print()
    print("Tests GroupBy, Join, and Sort operations.")
    print()

    sizes = [100_000, 500_000, 1_000_000]

    for n_rows in sizes:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {n_rows:,} rows")
        print("=" * 80)

        df = create_test_data(n_rows, n_groups=100)

        # =================================================================
        # GroupBy benchmarks
        # =================================================================
        print("\n--- GROUPBY OPERATIONS ---")

        print("\n1. Single aggregation (100 groups)")
        mean, std = timeit(lambda: eager_groupby_single_agg(df))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_groupby_single_agg(df))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n2. Multiple aggregations (sum, mean, min, max)")
        mean, std = timeit(lambda: eager_groupby_multi_agg(df))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_groupby_multi_agg(df))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n3. Filter then groupby")
        mean, std = timeit(lambda: eager_groupby_filter_agg(df))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_groupby_filter_agg(df))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n4. Multi-key groupby")
        mean, std = timeit(lambda: eager_groupby_multi_key(df))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_groupby_multi_key(df))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        # =================================================================
        # Join benchmarks
        # =================================================================
        print("\n--- JOIN OPERATIONS ---")

        # Create join test data
        n_keys = n_rows // 100  # 1% unique keys
        left, right = create_join_data(n_rows, n_rows // 2, n_keys)

        print(f"\nJoin: {len(left):,} rows x {len(right):,} rows, {n_keys} unique keys")

        print("\n5. Inner join")
        mean, std = timeit(lambda: eager_inner_join(left, right))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_inner_join(left, right))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n6. Left join")
        mean, std = timeit(lambda: eager_left_join(left, right))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_left_join(left, right))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n7. Join then filter")
        mean, std = timeit(lambda: eager_join_then_filter(left, right))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_join_then_filter(left, right))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n8. Join then select columns")
        mean, std = timeit(lambda: eager_join_select(left, right))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_join_select(left, right))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        # =================================================================
        # Sort benchmarks
        # =================================================================
        print("\n--- SORT OPERATIONS ---")

        print("\n9. Sort by single column")
        mean, std = timeit(lambda: eager_sort_single(df))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_sort_single(df))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n10. Sort + Limit 100 (Top-K)")
        mean, std = timeit(lambda: eager_sort_limit(df, 100))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_sort_limit(df, 100))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n11. Sort + Limit 1000 (Top-K)")
        mean, std = timeit(lambda: eager_sort_limit(df, 1000))
        print(f"Eager: {mean:8.2f} ms ± {std:.2f} ms")
        mean, std = timeit(lambda: lazy_sort_limit(df, 1000))
        print(f"Lazy:  {mean:8.2f} ms ± {std:.2f} ms")

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Advanced Operations Performance:

GROUPBY:
- Lazy groupby benefits from filter pushdown (filter before group)
- Multi-aggregation can be optimized to single pass
- Performance depends on number of groups and aggregation complexity

JOINS:
- Join performance dominated by join algorithm, not lazy vs eager
- Lazy benefits from:
  - Predicate pushdown (filter before join reduces rows)
  - Projection pruning (fewer columns to materialize)
- Many-to-many joins can explode in size

SORT:
- Full sort: Similar performance (both use NumPy/pandas sort)
- Sort+Limit: Lazy significantly faster with TopK optimization
  - TopK uses heap-based partial sort O(n log k) vs O(n log n)
  - Bigger win for small k and large n

When Lazy Wins for Advanced Ops:
1. Filter -> GroupBy (pushdown)
2. Sort -> Limit with small k (TopK)
3. Join -> Filter -> Select (combined optimizations)
4. Complex pipelines with multiple operations
""")


if __name__ == "__main__":
    run_benchmarks()
