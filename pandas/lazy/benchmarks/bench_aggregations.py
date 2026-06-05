#!/usr/bin/env python
"""
Benchmark: Aggregation operations

Tests aggregations with filter pipelines.
Note: The lazy API returns DataFrames, so aggregations that return scalars
work differently than in eager mode.
"""

import sys

from shared import (
    baseline_cli,
    create_grouped_data,
    scale_sizes,
    timeit,
)

import pandas as pd

# Sizes below this produce sub-millisecond timings whose noise would
# make regression gates flaky; printed but not recorded as metrics.
METRIC_MIN_ROWS = 100_000

# =============================================================================
# Filter then compute (common aggregation-like pipeline)
# =============================================================================


def eager_filter_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Filter then compute derived values."""
    result = df[df["value1"] > 50].copy()
    result["derived"] = result["value1"] * result["value2"] / 100
    return result


def lazy_filter_compute(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Filter then compute derived values."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("value1") > 50)
    lf = lf.with_columns((col("value1") * col("value2") / 100).alias("derived"))
    return lf.collect(use_physical_planner=use_physical_planner)


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


def lazy_multi_filter_compute(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Multiple filters then compute (filter fusion)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("group_b") == "X")
    lf = lf.filter(col("value1") > 30)
    lf = lf.filter(col("value2") < 800)
    lf = lf.with_columns((col("value1") + col("value2") * 0.1).alias("score"))
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Compute then filter
# =============================================================================


def eager_compute_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Compute then filter on computed value."""
    result = df.copy()
    result["ratio"] = result["value1"] / (result["value2"] + 1)
    return result[result["ratio"] > 0.1]


def lazy_compute_filter(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Compute then filter on computed value."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns((col("value1") / (col("value2") + 1)).alias("ratio"))
    lf = lf.filter(col("ratio") > 0.1)
    return lf.collect(use_physical_planner=use_physical_planner)


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


def lazy_complex_pipeline(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
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
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Select subset + compute
# =============================================================================


def eager_select_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Select columns then compute."""
    result = df[["value1", "value2", "value3"]].copy()
    result["combined"] = result["value1"] * result["value2"] + result["value3"]
    return result


def lazy_select_compute(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Select columns then compute."""
    from pandas.lazy import col

    lf = df.select("value1", "value2", "value3")
    lf = lf.with_columns(
        (col("value1") * col("value2") + col("value3")).alias("combined")
    )
    return lf.collect(use_physical_planner=use_physical_planner)


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


def lazy_multi_compute(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
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
    return lf.collect(use_physical_planner=use_physical_planner)


def run_benchmarks() -> int:
    """Run all aggregation pipeline benchmarks."""
    sizes = scale_sizes([10_000, 100_000, 1_000_000])
    metrics: dict[str, float] = {}

    print("=" * 70)
    print("AGGREGATION PIPELINE BENCHMARKS")
    print("=" * 70)

    pipelines = [
        ("filter_compute", eager_filter_compute, lazy_filter_compute),
        ("multi_filter_compute", eager_multi_filter_compute, lazy_multi_filter_compute),
        ("compute_filter", eager_compute_filter, lazy_compute_filter),
        ("complex_pipeline", eager_complex_pipeline, lazy_complex_pipeline),
        ("select_compute", eager_select_compute, lazy_select_compute),
        ("multi_compute", eager_multi_compute, lazy_multi_compute),
    ]

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows")
        print("=" * 70)

        for backend_label, use_arrow in (("numpy", False), ("arrow", True)):
            df = create_grouped_data(n_rows, use_arrow=use_arrow)
            print(f"\n--- {backend_label} backend ---")

            for name, eager_fn, lazy_fn in pipelines:
                variants = [
                    ("eager", lambda: eager_fn(df)),
                    ("lazy", lambda: lazy_fn(df)),
                    ("physical", lambda: lazy_fn(df, use_physical_planner=True)),
                ]
                for variant, fn in variants:
                    mean, std = timeit(fn)
                    print(f"{name:22s} {variant:9s} {mean:8.2f} ms ± {std:.2f} ms")
                    if n_rows >= METRIC_MIN_ROWS:
                        size_label = f"{n_rows // 1000}k"
                        key = f"agg_{name}_{backend_label}_{variant}_{size_label}"
                        metrics[key] = mean

    return baseline_cli(metrics, "aggregations")


if __name__ == "__main__":
    sys.exit(run_benchmarks())
