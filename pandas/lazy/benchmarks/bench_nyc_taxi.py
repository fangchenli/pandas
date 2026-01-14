"""
NYC Taxi Dataset Benchmark: Comparing DataFrame Processing Approaches

This benchmark compares four approaches to processing the NYC Yellow Taxi dataset:
1. Polars (lazy execution with query optimization)
2. Eager pandas (traditional pandas operations)
3. Lazy pandas without physical planner (logical optimization only)
4. Lazy pandas with physical planner (full optimization + kernel dispatch)

Dataset: NYC Yellow Taxi Trip Records
Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Usage:
    python pandas/lazy/benchmarks/bench_nyc_taxi.py [--months 1] [--runs 3]

Requirements:
    pip install polars pyarrow
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl

    import pandas as pd

# =============================================================================
# Data Loading
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download_data(year: int = 2024, month: int = 1) -> Path:
    """Download NYC Taxi parquet file if not present."""
    import urllib.request

    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    filepath = DATA_DIR / filename

    if not filepath.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")

    return filepath


def download_multiple_months(
    year: int = 2024, months: list[int] | None = None
) -> list[Path]:
    """Download multiple months of NYC Taxi data."""
    if months is None:
        months = [1]
    return [download_data(year, month) for month in months]


def load_pandas(filepath: Path | list[Path]) -> pd.DataFrame:
    """Load data as pandas DataFrame."""
    import pandas as pd

    if isinstance(filepath, list):
        dfs = [pd.read_parquet(f, engine="pyarrow") for f in filepath]
        return pd.concat(dfs, ignore_index=True)
    return pd.read_parquet(filepath, engine="pyarrow")


def load_polars(filepath: Path | list[Path]) -> pl.LazyFrame:
    """Load data as Polars LazyFrame."""
    import polars as pl

    if isinstance(filepath, list):
        return pl.scan_parquet(filepath)
    return pl.scan_parquet(filepath)


# =============================================================================
# Workflow Definition
# =============================================================================
#
# This workflow represents a typical analytics pipeline:
# 1. Filter: Select trips meeting certain criteria
# 2. Compute: Calculate derived metrics
# 3. Aggregate: Group by dimensions and compute statistics
# 4. Sort: Order results by key metric
#
# Operations used:
# - Filter pushdown (multiple filters)
# - Projection pushdown (select subset of columns)
# - Computed columns (arithmetic expressions)
# - Group by aggregation
# - Sorting with limit (TopK optimization opportunity)


def workflow_polars(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Polars implementation of the analytics workflow.

    Polars uses lazy evaluation with automatic query optimization.
    """
    import polars as pl

    result = (
        lf
        # Filter: Valid trips with reasonable values
        .filter(pl.col("trip_distance") > 0)
        .filter(pl.col("trip_distance") < 100)  # Exclude outliers
        .filter(pl.col("fare_amount") > 0)
        .filter(pl.col("fare_amount") < 500)
        .filter(pl.col("passenger_count") > 0)
        .filter(pl.col("payment_type").is_in([1, 2]))  # Credit/Cash only
        # Select columns we need (projection)
        .select(
            [
                "PULocationID",
                "trip_distance",
                "fare_amount",
                "tip_amount",
                "total_amount",
                "passenger_count",
                "tpep_pickup_datetime",
            ]
        )
        # Compute derived metrics
        .with_columns(
            [
                (pl.col("fare_amount") / pl.col("trip_distance")).alias(
                    "fare_per_mile"
                ),
                (pl.col("tip_amount") / pl.col("fare_amount") * 100).alias(
                    "tip_percentage"
                ),
                pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            ]
        )
        # Filter computed column
        .filter(pl.col("fare_per_mile") < 50)  # Remove extreme outliers
        # Aggregate by pickup location and hour
        .group_by(["PULocationID", "pickup_hour"])
        .agg(
            [
                pl.len().alias("trip_count"),
                pl.col("trip_distance").mean().alias("avg_distance"),
                pl.col("fare_amount").sum().alias("total_fare"),
                pl.col("tip_percentage").mean().alias("avg_tip_pct"),
                pl.col("passenger_count").mean().alias("avg_passengers"),
            ]
        )
        # Sort by total fare descending, take top 100
        .sort("total_fare", descending=True)
        .limit(100)
    )

    return result.collect()


def workflow_pandas_eager(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eager pandas implementation of the analytics workflow.

    This is the traditional pandas approach - operations execute immediately.
    """

    # Filter: Valid trips with reasonable values
    df = df[df["trip_distance"] > 0]
    df = df[df["trip_distance"] < 100]
    df = df[df["fare_amount"] > 0]
    df = df[df["fare_amount"] < 500]
    df = df[df["passenger_count"] > 0]
    df = df[df["payment_type"].isin([1, 2])]

    # Select columns we need
    df = df[
        [
            "PULocationID",
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "total_amount",
            "passenger_count",
            "tpep_pickup_datetime",
        ]
    ].copy()

    # Compute derived metrics
    df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"]
    df["tip_percentage"] = df["tip_amount"] / df["fare_amount"] * 100
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

    # Filter computed column
    df = df[df["fare_per_mile"] < 50]

    # Aggregate by pickup location and hour
    result = (
        df.groupby(["PULocationID", "pickup_hour"], as_index=False)
        .agg(
            trip_count=("trip_distance", "count"),
            avg_distance=("trip_distance", "mean"),
            total_fare=("fare_amount", "sum"),
            avg_tip_pct=("tip_percentage", "mean"),
            avg_passengers=("passenger_count", "mean"),
        )
        .sort_values("total_fare", ascending=False)
        .head(100)
    )

    return result


def workflow_lazy_pandas(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """
    Lazy pandas implementation of the analytics workflow.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    use_physical_planner : bool
        If True, use physical planner for execution.
        If False, use eager evaluation with logical optimization only.
    """
    from pandas.lazy import col

    # Enter lazy mode
    lf = df.select()

    # Filter: Valid trips with reasonable values
    lf = lf.filter(col("trip_distance") > 0)
    lf = lf.filter(col("trip_distance") < 100)
    lf = lf.filter(col("fare_amount") > 0)
    lf = lf.filter(col("fare_amount") < 500)
    lf = lf.filter(col("passenger_count") > 0)
    lf = lf.filter(col("payment_type").isin([1, 2]))  # Credit/Cash only

    # Select columns we need
    lf = lf.select(
        "PULocationID",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "passenger_count",
        "tpep_pickup_datetime",
    )

    # Compute derived metrics
    # Note: dt.hour is a property in lazy pandas (not a method like Polars)
    lf = lf.with_columns(
        (col("fare_amount") / col("trip_distance")).alias("fare_per_mile"),
        (col("tip_amount") / col("fare_amount") * 100).alias("tip_percentage"),
        col("tpep_pickup_datetime").dt.hour.alias("pickup_hour"),
    )

    # Filter computed column
    lf = lf.filter(col("fare_per_mile") < 50)

    # Aggregate by pickup location and hour
    lf = lf.group_by("PULocationID", "pickup_hour").agg(
        col("trip_distance").count().alias("trip_count"),
        col("trip_distance").mean().alias("avg_distance"),
        col("fare_amount").sum().alias("total_fare"),
        col("tip_percentage").mean().alias("avg_tip_pct"),
        col("passenger_count").mean().alias("avg_passengers"),
    )

    # Sort by total fare descending, take top 100
    lf = lf.sort("total_fare", descending=True).head(100)

    # Execute with specified planner
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Benchmarking
# =============================================================================


def timeit(func, *args, warmup: int = 1, runs: int = 3, **kwargs):
    """Time a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
        gc.collect()

    # Timed runs
    times = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return result, times


def run_benchmarks(filepath: Path | list[Path], runs: int = 3) -> dict:
    """Run all benchmarks and return results."""
    results = {}

    if isinstance(filepath, list):
        print(f"\nLoading data from {len(filepath)} files...")
        for f in filepath:
            print(f"  - {f.name}")
    else:
        print(f"\nLoading data from {filepath}...")

    # Load data for each framework
    df_pandas = load_pandas(filepath)
    lf_polars = load_polars(filepath)

    print(f"Loaded {len(df_pandas):,} rows")
    print(f"\nRunning benchmarks ({runs} runs each)...\n")

    # 1. Polars
    print("1. Polars (lazy)...")
    try:
        result_polars, times_polars = timeit(workflow_polars, lf_polars, runs=runs)
        results["polars"] = {
            "times": times_polars,
            "mean": np.mean(times_polars),
            "std": np.std(times_polars),
            "rows": len(result_polars),
        }
        mean = results["polars"]["mean"]
        std = results["polars"]["std"]
        print(f"   Mean: {mean:.1f} ms (±{std:.1f})")
    except Exception as e:
        print(f"   FAILED: {e}")
        results["polars"] = {"error": str(e)}

    # 2. Eager pandas
    print("2. Eager pandas...")
    try:
        # Make a copy for each run since we modify in place
        def run_eager():
            return workflow_pandas_eager(df_pandas.copy())

        result_eager, times_eager = timeit(run_eager, runs=runs)
        results["pandas_eager"] = {
            "times": times_eager,
            "mean": np.mean(times_eager),
            "std": np.std(times_eager),
            "rows": len(result_eager),
        }
        mean = results["pandas_eager"]["mean"]
        std = results["pandas_eager"]["std"]
        print(f"   Mean: {mean:.1f} ms (±{std:.1f})")
    except Exception as e:
        print(f"   FAILED: {e}")
        results["pandas_eager"] = {"error": str(e)}

    # 3. Lazy pandas without physical planner
    print("3. Lazy pandas (no physical planner)...")
    try:
        result_lazy, times_lazy = timeit(
            workflow_lazy_pandas, df_pandas, use_physical_planner=False, runs=runs
        )
        results["lazy_no_physical"] = {
            "times": times_lazy,
            "mean": np.mean(times_lazy),
            "std": np.std(times_lazy),
            "rows": len(result_lazy),
        }
        mean = results["lazy_no_physical"]["mean"]
        std = results["lazy_no_physical"]["std"]
        print(f"   Mean: {mean:.1f} ms (±{std:.1f})")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["lazy_no_physical"] = {"error": str(e)}

    # 4. Lazy pandas with physical planner
    print("4. Lazy pandas (with physical planner)...")
    try:
        result_physical, times_physical = timeit(
            workflow_lazy_pandas, df_pandas, use_physical_planner=True, runs=runs
        )
        results["lazy_physical"] = {
            "times": times_physical,
            "mean": np.mean(times_physical),
            "std": np.std(times_physical),
            "rows": len(result_physical),
        }
        mean = results["lazy_physical"]["mean"]
        std = results["lazy_physical"]["std"]
        print(f"   Mean: {mean:.1f} ms (±{std:.1f})")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["lazy_physical"] = {"error": str(e)}

    return results


def print_summary(results: dict) -> None:
    """Print a summary comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Find baseline (polars or eager pandas)
    baseline_key = (
        "polars"
        if "polars" in results and "error" not in results["polars"]
        else "pandas_eager"
    )
    baseline = results.get(baseline_key, {}).get("mean", 1)

    print(f"\n{'Approach':<35} {'Mean (ms)':>12} {'Std':>10} {'vs Polars':>12}")
    print("-" * 70)

    order = ["polars", "pandas_eager", "lazy_no_physical", "lazy_physical"]
    names = {
        "polars": "Polars (lazy)",
        "pandas_eager": "Pandas (eager)",
        "lazy_no_physical": "Lazy pandas (no physical)",
        "lazy_physical": "Lazy pandas (physical planner)",
    }

    for key in order:
        if key not in results:
            continue
        r = results[key]
        name = names[key]
        if "error" in r:
            print(f"{name:<35} {'FAILED':>12}")
        else:
            ratio = r["mean"] / baseline if baseline else 0
            print(f"{name:<35} {r['mean']:>12.1f} {r['std']:>10.1f} {ratio:>11.2f}x")

    print("-" * 70)

    # Additional insights
    if "pandas_eager" in results and "lazy_physical" in results:
        if (
            "error" not in results["pandas_eager"]
            and "error" not in results["lazy_physical"]
        ):
            speedup = results["pandas_eager"]["mean"] / results["lazy_physical"]["mean"]
            print(f"\nLazy pandas (physical) vs Eager pandas: {speedup:.2f}x speedup")

    if "lazy_no_physical" in results and "lazy_physical" in results:
        if (
            "error" not in results["lazy_no_physical"]
            and "error" not in results["lazy_physical"]
        ):
            improvement = (
                results["lazy_no_physical"]["mean"] / results["lazy_physical"]["mean"]
            )
            print(f"Physical planner improvement: {improvement:.2f}x")


# =============================================================================
# Result Validation
# =============================================================================


def validate_results(filepath: Path) -> None:
    """Validate that all approaches produce equivalent results."""
    print("\nValidating results...")

    df_pandas = load_pandas(filepath)
    lf_polars = load_polars(filepath)

    # Run all workflows
    result_polars = workflow_polars(lf_polars)
    result_eager = workflow_pandas_eager(df_pandas.copy())

    try:
        result_lazy = workflow_lazy_pandas(df_pandas, use_physical_planner=False)
        result_physical = workflow_lazy_pandas(df_pandas, use_physical_planner=True)
    except Exception as e:
        print(f"Lazy pandas failed: {e}")
        return

    # Compare row counts
    print("\nRow counts:")
    print(f"  Polars:                    {len(result_polars)}")
    print(f"  Pandas eager:              {len(result_eager)}")
    print(f"  Lazy pandas (no physical): {len(result_lazy)}")
    print(f"  Lazy pandas (physical):    {len(result_physical)}")

    # Compare total_fare sum (should be very close)
    print("\nTotal fare sum:")
    print(f"  Polars:                    {result_polars['total_fare'].sum():,.2f}")
    print(f"  Pandas eager:              {result_eager['total_fare'].sum():,.2f}")
    print(f"  Lazy pandas (no physical): {result_lazy['total_fare'].sum():,.2f}")
    print(f"  Lazy pandas (physical):    {result_physical['total_fare'].sum():,.2f}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NYC Taxi Dataset Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--year", type=int, default=2024, help="Year of data (default: 2024)"
    )
    parser.add_argument(
        "--month", type=int, default=1, help="Single month of data (default: 1)"
    )
    parser.add_argument(
        "--months",
        type=str,
        default=None,
        help="Multiple months, e.g., '1-6' or '1,2,3' (overrides --month)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate results match"
    )

    args = parser.parse_args()

    # Parse months argument
    if args.months:
        if "-" in args.months:
            start, end = map(int, args.months.split("-"))
            months = list(range(start, end + 1))
        else:
            months = [int(m) for m in args.months.split(",")]
        filepaths = download_multiple_months(args.year, months)
    else:
        filepaths = download_data(args.year, args.month)

    if args.validate:
        validate_results(filepaths)
    else:
        results = run_benchmarks(filepaths, runs=args.runs)
        print_summary(results)


if __name__ == "__main__":
    main()
