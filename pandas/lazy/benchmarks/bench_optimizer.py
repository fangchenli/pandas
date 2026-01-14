#!/usr/bin/env python
"""
Benchmark: Optimization pass effectiveness.

Measures:
- Actual speedup from each optimization pass
- Planning overhead (cost of optimization)
- Which query patterns benefit most from each optimization
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas.lazy.frame import LazyDataFrame


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


def create_test_data(n_rows: int) -> pd.DataFrame:
    """Create test DataFrame."""
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
            "c": rng.random(n_rows),
            "d": rng.integers(0, 100, n_rows),
            "e": rng.integers(0, 100, n_rows),
            "category": rng.choice(["A", "B", "C", "D"], n_rows),
        }
    )


def create_custom_optimizer(pass_classes: list):
    """Create optimizer with specific passes only."""
    from pandas.lazy.optimize import Optimizer

    passes = [cls() for cls in pass_classes]
    return Optimizer(passes=passes)


def measure_planning_overhead(df: pd.DataFrame, query_builder: Callable) -> dict:
    """
    Measure planning vs execution time separately.

    Returns dict with:
    - plan_time_ms: Time to build and optimize plan
    - exec_time_ms: Time to execute optimized plan
    - total_time_ms: Total time
    """
    from pandas.lazy.optimize import Optimizer

    # Build lazy frame
    lf = query_builder(df)

    # Measure plan building/optimization
    optimizer = Optimizer()

    start = time.perf_counter()
    plan = lf._plan
    _ = optimizer.optimize(plan)
    plan_time = (time.perf_counter() - start) * 1000

    # Measure execution
    start = time.perf_counter()
    _ = lf.collect()  # Full execution
    total_time = (time.perf_counter() - start) * 1000

    # Note: exec_time includes plan time since collect() does both
    return {
        "plan_time_ms": plan_time,
        "total_time_ms": total_time,
    }


# =============================================================================
# Query patterns for each optimization pass
# =============================================================================


def query_filter_fusion(df: pd.DataFrame) -> LazyDataFrame:
    """5 sequential filters - benefits from FilterFusion."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 0.1)
    lf = lf.filter(col("b") > 0.2)
    lf = lf.filter(col("c") > 0.3)
    lf = lf.filter(col("d") > 10)
    lf = lf.filter(col("e") > 20)
    return lf


def query_predicate_pushdown(df: pd.DataFrame) -> LazyDataFrame:
    """Compute then filter - benefits from PredicatePushdown."""
    from pandas.lazy import col

    lf = df.select()
    # Compute columns
    lf = lf.with_columns(
        (col("a") + col("b")).alias("x"),
        (col("c") * col("d")).alias("y"),
    )
    # Filter on original column (can be pushed down)
    lf = lf.filter(col("a") > 0.5)
    return lf


def query_projection_pruning(df: pd.DataFrame) -> LazyDataFrame:
    """Select few columns after filter - benefits from ProjectionPruning."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 0.5)
    lf = lf.select("a", "b")  # Only need 2 of 6 columns
    return lf


def query_limit_pushdown(df: pd.DataFrame) -> LazyDataFrame:
    """Filter then limit - benefits from LimitPushdown."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 0.5)
    lf = lf.limit(100)
    return lf


def query_constant_folding(df: pd.DataFrame) -> LazyDataFrame:
    """Expressions with constants - benefits from ConstantFolding."""
    from pandas.lazy import (
        col,
        lit,
    )

    lf = df.select()
    # Expression with constant that can be folded
    lf = lf.with_columns(
        (col("a") + lit(1) + lit(2)).alias("x"),  # 1+2 can be folded
        (col("b") * lit(2) * lit(3)).alias("y"),  # 2*3 can be folded
    )
    return lf


def query_sort_limit_topk(df: pd.DataFrame) -> LazyDataFrame:
    """Sort then limit - benefits from SortLimitToTopK."""

    lf = df.select()
    lf = lf.sort("a")
    lf = lf.limit(10)
    return lf


def query_dead_code_elimination(df: pd.DataFrame) -> LazyDataFrame:
    """Redundant filter conditions - benefits from DeadCodeElimination."""
    from pandas.lazy import col

    lf = df.select()
    # Filter that's always true for our data (a is 0-1 range)
    lf = lf.filter(col("a") >= 0)  # Always true, can be eliminated
    lf = lf.filter(col("a") < 100)  # Always true, can be eliminated
    lf = lf.filter(col("a") > 0.5)  # Actual filter
    return lf


def query_aggregate_pushdown(df: pd.DataFrame) -> LazyDataFrame:
    """Groupby with passthrough projection - benefits from AggregatePushdown."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.select("a", "b", "category")  # Pass-through projection
    lf = lf.groupby("category").agg(col("a").sum(), col("b").mean())
    return lf


def query_combined_optimizations(df: pd.DataFrame) -> LazyDataFrame:
    """Complex query benefiting from multiple optimizations."""
    from pandas.lazy import col

    lf = df.select()
    # Multiple filters (fusion)
    lf = lf.filter(col("a") > 0.2)
    lf = lf.filter(col("b") > 0.3)
    # Compute
    lf = lf.with_columns((col("c") * col("d")).alias("total"))
    # Filter on computed + original (predicate pushdown for original)
    lf = lf.filter(col("a") > 0.5)  # Can be pushed
    lf = lf.filter(col("total") > 10)  # Cannot be pushed
    # Select subset (projection pruning)
    lf = lf.select("a", "total")
    # Sort and limit (TopK)
    lf = lf.sort("total", descending=True)
    lf = lf.limit(100)
    return lf


def benchmark_with_passes(
    df: pd.DataFrame,
    query_builder: Callable,
    pass_configs: dict[str, list],
    n_runs: int = 5,
) -> dict[str, tuple[float, float]]:
    """
    Run query with different optimizer configurations.

    Parameters
    ----------
    df : DataFrame
        Input data
    query_builder : Callable
        Function that builds the lazy query
    pass_configs : dict
        Mapping from config name to list of pass classes
    n_runs : int
        Number of benchmark runs

    Returns
    -------
    dict
        Mapping from config name to (mean_ms, std_ms)
    """
    results = {}

    for name, passes in pass_configs.items():
        if passes is None:
            # No optimization
            def run():
                lf = query_builder(df)
                return lf.collect(optimize=False)
        else:
            optimizer = create_custom_optimizer(passes)

            def run(opt=optimizer):
                lf = query_builder(df)
                # Apply custom optimizer
                optimized_plan = opt.optimize(lf._plan)
                # Create new LazyDataFrame with optimized plan
                from pandas.lazy.frame import LazyDataFrame

                optimized_lf = LazyDataFrame.__new__(LazyDataFrame)
                optimized_lf._plan = optimized_plan
                return optimized_lf.collect(optimize=False)

        mean, std = timeit(run, n_runs=n_runs)
        results[name] = (mean, std)

    return results


def run_benchmarks():
    """Run optimization pass effectiveness benchmarks."""
    from pandas.lazy.optimize import (
        AggregatePushdown,
        ConstantFolding,
        DeadCodeElimination,
        FilterFusion,
        LimitPushdown,
        PredicatePushdown,
        ProjectionPruning,
        SortLimitToTopK,
    )

    print("=" * 80)
    print("OPTIMIZATION PASS EFFECTIVENESS BENCHMARKS")
    print("=" * 80)
    print()
    print("Measures actual speedup from each optimization pass.")
    print("Compares: No optimization vs Individual passes vs All passes")
    print()

    sizes = [100_000, 500_000, 1_000_000]

    # All default passes in order
    all_passes = [
        ConstantFolding,
        DeadCodeElimination,
        FilterFusion,
        PredicatePushdown,
        AggregatePushdown,
        ProjectionPruning,
        LimitPushdown,
        SortLimitToTopK,
    ]

    for n_rows in sizes:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {n_rows:,} rows")
        print("=" * 80)

        df = create_test_data(n_rows)

        # =================================================================
        # Test 1: FilterFusion effectiveness
        # =================================================================
        print("\n--- FilterFusion: 5 sequential filters ---")

        configs = {
            "No Opt": None,
            "FilterFusion only": [FilterFusion],
            "All passes": all_passes,
        }

        results = benchmark_with_passes(df, query_filter_fusion, configs)
        baseline = results["No Opt"][0]

        for name, (mean, std) in results.items():
            speedup = baseline / mean if mean > 0 else 0
            print(
                f"{name:20s}: {mean:8.2f} ms ± {std:.2f} ms  (speedup: {speedup:.2f}x)"
            )

        # =================================================================
        # Test 2: PredicatePushdown effectiveness
        # =================================================================
        print("\n--- PredicatePushdown: Compute then filter ---")

        configs = {
            "No Opt": None,
            "Pushdown only": [PredicatePushdown],
            "All passes": all_passes,
        }

        results = benchmark_with_passes(df, query_predicate_pushdown, configs)
        baseline = results["No Opt"][0]

        for name, (mean, std) in results.items():
            speedup = baseline / mean if mean > 0 else 0
            print(
                f"{name:20s}: {mean:8.2f} ms ± {std:.2f} ms  (speedup: {speedup:.2f}x)"
            )

        # =================================================================
        # Test 3: ProjectionPruning effectiveness
        # =================================================================
        print("\n--- ProjectionPruning: Select 2 of 6 columns ---")

        configs = {
            "No Opt": None,
            "Pruning only": [ProjectionPruning],
            "All passes": all_passes,
        }

        results = benchmark_with_passes(df, query_projection_pruning, configs)
        baseline = results["No Opt"][0]

        for name, (mean, std) in results.items():
            speedup = baseline / mean if mean > 0 else 0
            print(
                f"{name:20s}: {mean:8.2f} ms ± {std:.2f} ms  (speedup: {speedup:.2f}x)"
            )

        # =================================================================
        # Test 4: SortLimitToTopK effectiveness
        # =================================================================
        print("\n--- SortLimitToTopK: Sort + Limit 10 ---")

        configs = {
            "No Opt": None,
            "TopK only": [SortLimitToTopK],
            "All passes": all_passes,
        }

        results = benchmark_with_passes(df, query_sort_limit_topk, configs)
        baseline = results["No Opt"][0]

        for name, (mean, std) in results.items():
            speedup = baseline / mean if mean > 0 else 0
            print(
                f"{name:20s}: {mean:8.2f} ms ± {std:.2f} ms  (speedup: {speedup:.2f}x)"
            )

        # =================================================================
        # Test 5: ConstantFolding effectiveness
        # =================================================================
        print("\n--- ConstantFolding: Expressions with literals ---")

        configs = {
            "No Opt": None,
            "Folding only": [ConstantFolding],
            "All passes": all_passes,
        }

        results = benchmark_with_passes(df, query_constant_folding, configs)
        baseline = results["No Opt"][0]

        for name, (mean, std) in results.items():
            speedup = baseline / mean if mean > 0 else 0
            print(
                f"{name:20s}: {mean:8.2f} ms ± {std:.2f} ms  (speedup: {speedup:.2f}x)"
            )

        # =================================================================
        # Test 6: DeadCodeElimination effectiveness
        # =================================================================
        print("\n--- DeadCodeElimination: Filter(True) ---")

        configs = {
            "No Opt": None,
            "DCE only": [DeadCodeElimination],
            "All passes": all_passes,
        }

        results = benchmark_with_passes(df, query_dead_code_elimination, configs)
        baseline = results["No Opt"][0]

        for name, (mean, std) in results.items():
            speedup = baseline / mean if mean > 0 else 0
            print(
                f"{name:20s}: {mean:8.2f} ms ± {std:.2f} ms  (speedup: {speedup:.2f}x)"
            )

        # =================================================================
        # Test 7: Combined optimizations
        # =================================================================
        print("\n--- Combined: Complex query (all optimizations) ---")

        configs = {
            "No Opt": None,
            "FilterFusion": [FilterFusion],
            "Pushdown": [PredicatePushdown],
            "Pruning": [ProjectionPruning],
            "TopK": [SortLimitToTopK],
            "All passes": all_passes,
        }

        results = benchmark_with_passes(df, query_combined_optimizations, configs)
        baseline = results["No Opt"][0]

        for name, (mean, std) in results.items():
            speedup = baseline / mean if mean > 0 else 0
            print(
                f"{name:20s}: {mean:8.2f} ms ± {std:.2f} ms  (speedup: {speedup:.2f}x)"
            )

    # =================================================================
    # Planning overhead analysis
    # =================================================================
    print("\n" + "=" * 80)
    print("PLANNING OVERHEAD ANALYSIS")
    print("=" * 80)
    print()

    # Use medium dataset
    df = create_test_data(500_000)

    queries = [
        (
            "Simple filter",
            lambda d: d.select().filter(
                __import__("pandas.lazy", fromlist=["col"]).col("a") > 0.5
            ),
        ),
        ("5 sequential filters", query_filter_fusion),
        ("Complex query", query_combined_optimizations),
    ]

    print(
        "Query                     Plan Time (ms)   Total Time (ms)   Plan Overhead %"
    )
    print("-" * 75)

    for name, builder in queries:
        try:
            overhead = measure_planning_overhead(df, builder)
            plan_pct = (overhead["plan_time_ms"] / overhead["total_time_ms"]) * 100
            print(
                f"{name:25s} {overhead['plan_time_ms']:10.3f}       "
                f"{overhead['total_time_ms']:10.2f}          {plan_pct:6.1f}%"
            )
        except Exception as e:
            print(f"{name:25s} Error: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Optimization Pass Impact:

1. FILTER FUSION: High impact for sequential filters.
   Reduces N filter operations to 1 combined filter.

2. PREDICATE PUSHDOWN: Impact depends on selectivity.
   More benefit when filter is highly selective (few rows pass).

3. PROJECTION PRUNING: Impact depends on column ratio.
   More benefit when selecting few columns from wide DataFrame.

4. SORT+LIMIT to TopK: Very high impact for small K.
   Avoids full sort - uses heap-based partial sort.

5. CONSTANT FOLDING: Low impact for most queries.
   Mainly useful for complex expressions with many literals.

6. DEAD CODE ELIMINATION: Low but consistent impact.
   Removes no-op operations that would waste cycles.

Planning Overhead:
- Planning cost is typically < 1% of total execution time
- More passes = slightly more planning time
- Planning cost is amortized for repeated queries
""")


if __name__ == "__main__":
    run_benchmarks()
