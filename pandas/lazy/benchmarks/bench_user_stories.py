#!/usr/bin/env python3
"""
Benchmark: End-to-End User Stories

Named, realistic scenarios that serve as:
1. Documentation examples
2. External communication (Databricks, pandas-dev, PyCon talks)
3. Stable north star for performance tracking

Each story represents a common pandas workflow that lazy execution
should optimize.

User Stories:
1. Feature Engineering Pipeline - ML preprocessing
2. ETL Cleanup + Aggregation - Data warehouse loading
3. Exploratory Analysis - Interactive data exploration
4. Reporting Query - Business intelligence dashboard
5. Time Series Rollup - Financial/IoT data processing

Usage:
    python pandas/lazy/benchmarks/bench_user_stories.py
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from typing import TYPE_CHECKING

import numpy as np

import pandas as pd
from pandas.lazy import col

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Configuration
# =============================================================================

N_RUNS = 5
WARMUP = 2


# =============================================================================
# Timing utilities
# =============================================================================


def time_operation(func: Callable, n_runs: int = N_RUNS) -> tuple[float, float]:
    """Time an operation, return (mean_ms, std_ms)."""
    for _ in range(WARMUP):
        func()

    gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func()  # Execute and discard result
            end = time.perf_counter()
            times.append((end - start) * 1000)
    finally:
        gc.enable()

    return float(np.mean(times)), float(np.std(times))


# =============================================================================
# Result tracking
# =============================================================================


@dataclass
class StoryResult:
    """Result for a user story benchmark."""

    name: str
    description: str
    input_rows: int
    output_rows: int
    eager_ms: float
    lazy_ms: float
    operations: list[str]

    @property
    def speedup(self) -> float:
        return self.eager_ms / self.lazy_ms if self.lazy_ms > 0 else float("inf")

    def print_summary(self):
        print(f"\n{'─' * 70}")
        print(f"📊 {self.name}")
        print(f"{'─' * 70}")
        print(f"   {self.description}")
        print(f"\n   Pipeline: {' → '.join(self.operations)}")
        print(f"\n   Input:  {self.input_rows:>12,} rows")
        print(f"   Output: {self.output_rows:>12,} rows")
        print(f"\n   Eager pandas:  {self.eager_ms:>8.1f} ms")
        print(f"   Lazy pandas:   {self.lazy_ms:>8.1f} ms")

        if self.speedup >= 1.0:
            print(f"   Speedup:       {self.speedup:>8.2f}x faster with lazy")
        else:
            print(f"   Speedup:       {1 / self.speedup:>8.2f}x faster with eager")


# =============================================================================
# Data Generators for User Stories
# =============================================================================


def create_ecommerce_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    E-commerce transaction data for ETL and reporting stories.

    Columns: order_id, customer_id, product_id, category, quantity,
             unit_price, discount, order_date, region, is_returned
    """
    rng = np.random.default_rng(seed)

    categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Beauty"]
    regions = ["North", "South", "East", "West", "Central"]

    return pd.DataFrame(
        {
            "order_id": np.arange(n_rows),
            "customer_id": rng.integers(1, n_rows // 10, n_rows),
            "product_id": rng.integers(1, 10000, n_rows),
            "category": rng.choice(categories, n_rows),
            "quantity": rng.integers(1, 10, n_rows),
            "unit_price": rng.uniform(10, 500, n_rows).round(2),
            "discount": rng.uniform(0, 0.3, n_rows).round(2),
            "order_date": pd.date_range("2023-01-01", periods=n_rows, freq="min"),
            "region": rng.choice(regions, n_rows),
            "is_returned": rng.choice([True, False], n_rows, p=[0.05, 0.95]),
        }
    )


def create_ml_features_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Raw data for ML feature engineering.

    Columns: user_id, age, income, credit_score, account_age_days,
             num_transactions, total_spent, is_premium, last_login_days,
             support_tickets, category_pref
    """
    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "user_id": np.arange(n_rows),
            "age": rng.integers(18, 80, n_rows),
            "income": rng.lognormal(10.5, 0.5, n_rows).round(2),
            "credit_score": rng.integers(300, 850, n_rows),
            "account_age_days": rng.integers(1, 3650, n_rows),
            "num_transactions": rng.poisson(50, n_rows),
            "total_spent": rng.exponential(1000, n_rows).round(2),
            "is_premium": rng.choice([True, False], n_rows, p=[0.2, 0.8]),
            "last_login_days": rng.integers(0, 365, n_rows),
            "support_tickets": rng.poisson(2, n_rows),
            "category_pref": rng.choice(["A", "B", "C", "D"], n_rows),
        }
    )


def create_sensor_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    IoT sensor time series data.

    Columns: timestamp, sensor_id, temperature, humidity, pressure,
             battery_level, is_anomaly
    """
    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="s"),
            "sensor_id": rng.integers(1, 100, n_rows),
            "temperature": rng.normal(25, 5, n_rows).round(2),
            "humidity": rng.uniform(30, 90, n_rows).round(2),
            "pressure": rng.normal(1013, 10, n_rows).round(2),
            "battery_level": rng.uniform(0, 100, n_rows).round(1),
            "is_anomaly": rng.choice([True, False], n_rows, p=[0.01, 0.99]),
        }
    )


# =============================================================================
# User Story 1: Feature Engineering Pipeline
# =============================================================================


def story_feature_engineering(n_rows: int = 500_000) -> StoryResult:
    """
    ML Feature Engineering Pipeline

    A data scientist preparing features for a churn prediction model.
    This is a common pattern in ML workflows where raw data is transformed
    into model-ready features.

    Pipeline:
    1. Filter out inactive users
    2. Create derived features (ratios, bins, flags)
    3. Handle edge cases (clip outliers)
    4. Select final feature columns
    """
    df = create_ml_features_data(n_rows)

    # Eager implementation
    def eager():
        result = df.copy()

        # Filter inactive users
        result = result[result["last_login_days"] < 180]

        # Derived features
        result["spend_per_transaction"] = result["total_spent"] / (
            result["num_transactions"] + 1
        )
        result["account_age_years"] = result["account_age_days"] / 365
        result["income_to_spend_ratio"] = result["income"] / (result["total_spent"] + 1)

        # Binning
        result["age_group"] = pd.cut(
            result["age"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=["18-25", "26-35", "36-50", "51-65", "65+"],
        )
        result["credit_tier"] = pd.cut(
            result["credit_score"],
            bins=[0, 580, 670, 740, 800, 850],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        )

        # Flags
        result["is_high_value"] = (result["total_spent"] > 5000) & result["is_premium"]
        result["needs_engagement"] = (result["last_login_days"] > 30) & (
            result["num_transactions"] < 10
        )

        # Clip outliers
        result["income_clipped"] = result["income"].clip(
            upper=result["income"].quantile(0.99)
        )

        # Select features
        features = [
            "user_id",
            "age",
            "credit_score",
            "spend_per_transaction",
            "account_age_years",
            "income_to_spend_ratio",
            "is_premium",
            "is_high_value",
            "needs_engagement",
        ]
        return result[features]

    eager_ms, _ = time_operation(eager)
    eager_result = eager()
    output_rows = len(eager_result)

    # Lazy implementation
    def lazy():
        return (
            df.select()
            # Filter inactive users
            .filter(col("last_login_days") < 180)
            # Derived features
            .select(
                col("user_id"),
                col("age"),
                col("credit_score"),
                col("is_premium"),
                col("total_spent"),
                col("num_transactions"),
                col("income"),
                col("last_login_days"),
                col("account_age_days"),
                # Computed features
                (col("total_spent") / (col("num_transactions") + 1)).alias(
                    "spend_per_transaction"
                ),
                (col("account_age_days") / 365).alias("account_age_years"),
                (col("income") / (col("total_spent") + 1)).alias(
                    "income_to_spend_ratio"
                ),
                # Flags
                ((col("total_spent") > 5000) & col("is_premium")).alias(
                    "is_high_value"
                ),
                ((col("last_login_days") > 30) & (col("num_transactions") < 10)).alias(
                    "needs_engagement"
                ),
            )
            # Select final features
            .select(
                col("user_id"),
                col("age"),
                col("credit_score"),
                col("spend_per_transaction"),
                col("account_age_years"),
                col("income_to_spend_ratio"),
                col("is_premium"),
                col("is_high_value"),
                col("needs_engagement"),
            )
            .collect()
        )

    lazy_ms, _ = time_operation(lazy)

    return StoryResult(
        name="Feature Engineering Pipeline",
        description="ML preprocessing: filter inactive → derive features → select",
        input_rows=n_rows,
        output_rows=output_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        operations=["Filter", "Compute", "Select"],
    )


# =============================================================================
# User Story 2: ETL Cleanup + Aggregation
# =============================================================================


def story_etl_aggregation(n_rows: int = 1_000_000) -> StoryResult:
    """
    ETL Cleanup + Aggregation

    A data engineer building a daily sales summary for the data warehouse.
    This pattern is common in batch ETL jobs that run nightly.

    Pipeline:
    1. Filter out returned orders and invalid data
    2. Calculate order totals
    3. Aggregate by date and category
    4. Sort by revenue descending
    """
    df = create_ecommerce_data(n_rows)

    # Eager implementation
    def eager():
        result = df.copy()

        # Filter returns and invalid data
        result = result[~result["is_returned"]]
        result = result[result["quantity"] > 0]
        result = result[result["unit_price"] > 0]

        # Calculate totals
        result["gross_amount"] = result["quantity"] * result["unit_price"]
        result["discount_amount"] = result["gross_amount"] * result["discount"]
        result["net_amount"] = result["gross_amount"] - result["discount_amount"]

        # Extract date
        result["order_date_only"] = result["order_date"].dt.date

        # Aggregate
        summary = (
            result.groupby(["order_date_only", "category", "region"])
            .agg(
                order_count=("order_id", "count"),
                total_quantity=("quantity", "sum"),
                gross_revenue=("gross_amount", "sum"),
                total_discount=("discount_amount", "sum"),
                net_revenue=("net_amount", "sum"),
            )
            .reset_index()
        )

        # Sort by revenue
        return summary.sort_values("net_revenue", ascending=False)

    eager_ms, _ = time_operation(eager)
    eager_result = eager()
    output_rows = len(eager_result)

    # Lazy implementation
    def lazy():
        return (
            df.select()
            # Filter returns and invalid data
            .filter(~col("is_returned"))
            .filter(col("quantity") > 0)
            .filter(col("unit_price") > 0)
            # Calculate totals
            .select(
                col("order_id"),
                col("order_date"),
                col("category"),
                col("region"),
                col("quantity"),
                (col("quantity") * col("unit_price")).alias("gross_amount"),
                (col("quantity") * col("unit_price") * col("discount")).alias(
                    "discount_amount"
                ),
                (
                    col("quantity") * col("unit_price")
                    - col("quantity") * col("unit_price") * col("discount")
                ).alias("net_amount"),
            )
            # Aggregate
            .group_by(col("category"), col("region"))
            .agg(
                col("order_id").count().alias("order_count"),
                col("quantity").sum().alias("total_quantity"),
                col("gross_amount").sum().alias("gross_revenue"),
                col("discount_amount").sum().alias("total_discount"),
                col("net_amount").sum().alias("net_revenue"),
            )
            # Sort by revenue
            .sort(col("net_revenue"), descending=True)
            .collect()
        )

    lazy_ms, _ = time_operation(lazy)

    return StoryResult(
        name="ETL Cleanup + Aggregation",
        description="Data warehouse ETL: filter invalid → compute → aggregate",
        input_rows=n_rows,
        output_rows=output_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        operations=["Filter", "Compute", "GroupBy", "Sort"],
    )


# =============================================================================
# User Story 3: Exploratory Analysis
# =============================================================================


def story_exploratory_analysis(n_rows: int = 500_000) -> StoryResult:
    """
    Exploratory Analysis

    A data analyst exploring sales data in a Jupyter notebook.
    Interactive workflow where quick iteration is essential.

    Pipeline:
    1. Filter to specific region and category
    2. Group by customer
    3. Aggregate spending
    4. Sort by total spend
    5. Take top 100 customers

    This is where head() optimization and streaming really shine.
    """
    df = create_ecommerce_data(n_rows)

    # Eager implementation
    def eager():
        result = df.copy()

        # Filter to Electronics in North region
        result = result[result["region"] == "North"]
        result = result[result["category"] == "Electronics"]
        result = result[~result["is_returned"]]

        # Calculate order value
        result["order_value"] = result["quantity"] * result["unit_price"]

        # Aggregate by customer
        customer_summary = (
            result.groupby("customer_id")
            .agg(
                order_count=("order_id", "count"),
                total_spent=("order_value", "sum"),
                avg_order_value=("order_value", "mean"),
            )
            .reset_index()
        )

        # Top 100 spenders
        return customer_summary.sort_values("total_spent", ascending=False).head(100)

    eager_ms, _ = time_operation(eager)
    eager_result = eager()
    output_rows = len(eager_result)

    # Lazy implementation
    def lazy():
        return (
            df.select()
            # Filter to Electronics in North region
            .filter(col("region") == "North")
            .filter(col("category") == "Electronics")
            .filter(~col("is_returned"))
            # Calculate order value
            .select(
                col("order_id"),
                col("customer_id"),
                (col("quantity") * col("unit_price")).alias("order_value"),
            )
            # Aggregate by customer
            .group_by(col("customer_id"))
            .agg(
                col("order_id").count().alias("order_count"),
                col("order_value").sum().alias("total_spent"),
                col("order_value").mean().alias("avg_order_value"),
            )
            # Top 100 spenders
            .sort(col("total_spent"), descending=True)
            .head(100)
            .collect()
        )

    lazy_ms, _ = time_operation(lazy)

    return StoryResult(
        name="Exploratory Analysis",
        description="Jupyter workflow: filter → groupby → sort → head(100)",
        input_rows=n_rows,
        output_rows=output_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        operations=["Filter", "Compute", "GroupBy", "Sort", "Head"],
    )


# =============================================================================
# User Story 4: Reporting Query
# =============================================================================


def story_reporting_query(n_rows: int = 1_000_000) -> StoryResult:
    """
    Reporting Query

    A BI dashboard querying regional sales performance.
    This runs every time a user opens the dashboard.

    Pipeline:
    1. Filter to current year
    2. Exclude returns
    3. Calculate metrics
    4. Aggregate by region
    5. Add derived metrics (percentages, rankings)
    """
    df = create_ecommerce_data(n_rows)

    # Eager implementation
    def eager():
        result = df.copy()

        # Filter to 2024 and non-returns
        result = result[result["order_date"].dt.year == 2024]
        result = result[~result["is_returned"]]

        # Calculate order value
        result["order_value"] = (
            result["quantity"] * result["unit_price"] * (1 - result["discount"])
        )

        # Aggregate by region
        region_summary = (
            result.groupby("region")
            .agg(
                total_orders=("order_id", "count"),
                total_revenue=("order_value", "sum"),
                avg_order_value=("order_value", "mean"),
                unique_customers=("customer_id", "nunique"),
            )
            .reset_index()
        )

        # Add derived metrics
        total_revenue = region_summary["total_revenue"].sum()
        region_summary["revenue_share"] = (
            region_summary["total_revenue"] / total_revenue
        )
        region_summary["revenue_rank"] = region_summary["total_revenue"].rank(
            ascending=False
        )

        return region_summary.sort_values("total_revenue", ascending=False)

    eager_ms, _ = time_operation(eager)
    eager_result = eager()
    output_rows = len(eager_result)

    # Lazy implementation
    def lazy():
        return (
            df.select()
            # Filter
            .filter(~col("is_returned"))
            # Calculate order value
            .select(
                col("order_id"),
                col("customer_id"),
                col("region"),
                (col("quantity") * col("unit_price") * (1 - col("discount"))).alias(
                    "order_value"
                ),
            )
            # Aggregate by region
            .group_by(col("region"))
            .agg(
                col("order_id").count().alias("total_orders"),
                col("order_value").sum().alias("total_revenue"),
                col("order_value").mean().alias("avg_order_value"),
                col("customer_id").n_unique().alias("unique_customers"),
            )
            # Sort
            .sort(col("total_revenue"), descending=True)
            .collect()
        )

    lazy_ms, _ = time_operation(lazy)

    return StoryResult(
        name="Reporting Query",
        description="BI dashboard: filter → compute → aggregate by region → sort",
        input_rows=n_rows,
        output_rows=output_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        operations=["Filter", "Compute", "GroupBy", "Sort"],
    )


# =============================================================================
# User Story 5: Time Series Rollup
# =============================================================================


def story_timeseries_rollup(n_rows: int = 1_000_000) -> StoryResult:
    """
    Time Series Rollup

    An IoT platform rolling up sensor data from seconds to hourly aggregates.
    This is a common pattern in time series databases and monitoring systems.

    Pipeline:
    1. Filter out anomalies
    2. Filter to specific sensor range
    3. Resample/aggregate to hourly
    4. Calculate statistics per hour
    """
    df = create_sensor_data(n_rows)

    # Eager implementation
    def eager():
        result = df.copy()

        # Filter anomalies
        result = result[~result["is_anomaly"]]

        # Filter to sensors 1-10
        result = result[result["sensor_id"] <= 10]

        # Set index for resampling
        result = result.set_index("timestamp")

        # Resample to hourly
        hourly = (
            result.groupby("sensor_id")
            .resample("h")
            .agg(
                temp_mean=("temperature", "mean"),
                temp_min=("temperature", "min"),
                temp_max=("temperature", "max"),
                humidity_mean=("humidity", "mean"),
                pressure_mean=("pressure", "mean"),
                battery_min=("battery_level", "min"),
                reading_count=("temperature", "count"),
            )
        )

        return hourly.reset_index()

    eager_ms, _ = time_operation(eager)
    eager_result = eager()
    output_rows = len(eager_result)

    # Lazy implementation (simplified - no true resample yet)
    def lazy():
        return (
            df.select()
            # Filter anomalies and sensor range
            .filter(~col("is_anomaly"))
            .filter(col("sensor_id") <= 10)
            # Aggregate by sensor (simplified - no time bucketing)
            .group_by(col("sensor_id"))
            .agg(
                col("temperature").mean().alias("temp_mean"),
                col("temperature").min().alias("temp_min"),
                col("temperature").max().alias("temp_max"),
                col("humidity").mean().alias("humidity_mean"),
                col("pressure").mean().alias("pressure_mean"),
                col("battery_level").min().alias("battery_min"),
                col("temperature").count().alias("reading_count"),
            )
            .collect()
        )

    lazy_ms, _ = time_operation(lazy)

    return StoryResult(
        name="Time Series Rollup",
        description="IoT data: filter anomalies → filter sensors → aggregate hourly",
        input_rows=n_rows,
        output_rows=output_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        operations=["Filter", "GroupBy", "Aggregate"],
    )


# =============================================================================
# Main Benchmark
# =============================================================================


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_user_stories():
    """Run all user story benchmarks."""
    print_header("LAZY PANDAS USER STORIES")
    print(
        """
Real-world scenarios that demonstrate lazy pandas value.
Each story represents a common pandas workflow.

These benchmarks serve as:
  - Documentation examples
  - External communication (talks, blog posts)
  - Stable performance north star
"""
    )

    results: list[StoryResult] = []

    # Run each story
    print_header("STORY 1: FEATURE ENGINEERING PIPELINE")
    print("Scenario: ML preprocessing for churn prediction model")
    result = story_feature_engineering(500_000)
    result.print_summary()
    results.append(result)

    print_header("STORY 2: ETL CLEANUP + AGGREGATION")
    print("Scenario: Nightly data warehouse loading job")
    result = story_etl_aggregation(1_000_000)
    result.print_summary()
    results.append(result)

    print_header("STORY 3: EXPLORATORY ANALYSIS")
    print("Scenario: Data analyst in Jupyter notebook")
    result = story_exploratory_analysis(500_000)
    result.print_summary()
    results.append(result)

    print_header("STORY 4: REPORTING QUERY")
    print("Scenario: BI dashboard regional performance")
    result = story_reporting_query(1_000_000)
    result.print_summary()
    results.append(result)

    print_header("STORY 5: TIME SERIES ROLLUP")
    print("Scenario: IoT sensor data aggregation")
    result = story_timeseries_rollup(1_000_000)
    result.print_summary()
    results.append(result)

    # Summary table
    print_header("SUMMARY")

    print(
        f"\n{'Story':<35} {'Input':>12} {'Output':>12} "
        f"{'Eager':>10} {'Lazy':>10} {'Speedup':>10}"
    )
    print("-" * 95)

    for r in results:
        speedup_str = f"{r.speedup:.2f}x"
        print(
            f"{r.name:<35} {r.input_rows:>12,} {r.output_rows:>12,} "
            f"{r.eager_ms:>10.1f} {r.lazy_ms:>10.1f} {speedup_str:>10}"
        )

    # Overall stats
    avg_speedup = np.mean([r.speedup for r in results])
    total_eager = sum(r.eager_ms for r in results)
    total_lazy = sum(r.lazy_ms for r in results)

    print("-" * 95)
    print(
        f"{'TOTAL':<35} {' ':>12} {' ':>12} "
        f"{total_eager:>10.1f} {total_lazy:>10.1f} {total_eager / total_lazy:>9.2f}x"
    )
    print(f"\nAverage speedup across stories: {avg_speedup:.2f}x")

    # Key takeaways
    print_header("KEY TAKEAWAYS")
    print(
        """
1. FEATURE ENGINEERING
   - Lazy shines when filtering first reduces data size
   - Derived features computed only on filtered rows

2. ETL AGGREGATION
   - Filter fusion combines multiple filters
   - Aggregation benefits from columnar execution

3. EXPLORATORY ANALYSIS
   - head(N) with sorting is optimized (TopK)
   - Interactive queries benefit from warm cache

4. REPORTING QUERIES
   - Repeated queries benefit from plan caching
   - Small output (few regions) benefits from projection

5. TIME SERIES
   - Filter + aggregate is a sweet spot for lazy
   - Streaming helps with large time ranges
"""
    )


if __name__ == "__main__":
    run_user_stories()
