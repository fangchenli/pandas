#!/usr/bin/env python3
"""
Benchmark: Planning Phase Breakdown

Measures the overhead of each phase in lazy execution separately:
1. Logical plan construction (building the plan tree)
2. Optimization passes (each pass individually)
3. Physical planning (logical -> physical conversion)
4. Execution (actual data processing)

This helps identify where the 2-5x overhead on small data comes from,
and guides "fast path" heuristics (skip optimizer, skip CSE, etc.).

Usage:
    python pandas/lazy/benchmarks/bench_planning_phases.py
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from typing import TYPE_CHECKING

import numpy as np

import pandas as pd
from pandas.lazy import col
from pandas.lazy.optimize import (
    AggregatePushdown,
    CommonSubexpressionElimination,
    ConstantFolding,
    ConversionElimination,
    DeadCodeElimination,
    EngineSelection,
    ExpressionSimplification,
    FilterFusion,
    LimitPushdown,
    PredicatePushdown,
    ProjectionPruning,
    SortLimitToTopK,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Timing utilities
# =============================================================================


def time_function_ns(func: Callable, n_runs: int = 100) -> tuple[float, float]:
    """
    Time a function in nanoseconds for fine-grained measurement.

    Returns (mean_ns, std_ns).
    """
    # Warmup
    for _ in range(5):
        func()

    gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(n_runs):
            start = time.perf_counter_ns()
            func()
            end = time.perf_counter_ns()
            times.append(end - start)
    finally:
        gc.enable()

    return float(np.mean(times)), float(np.std(times))


def time_function_us(func: Callable, n_runs: int = 50) -> tuple[float, float]:
    """Time a function, return (mean_us, std_us) in microseconds."""
    mean_ns, std_ns = time_function_ns(func, n_runs)
    return mean_ns / 1000, std_ns / 1000


def time_function_ms(func: Callable, n_runs: int = 20) -> tuple[float, float]:
    """Time a function, return (mean_ms, std_ms) in milliseconds."""
    mean_ns, std_ns = time_function_ns(func, n_runs)
    return mean_ns / 1_000_000, std_ns / 1_000_000


# =============================================================================
# Data generation
# =============================================================================


def create_test_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Create test DataFrame with various column types."""
    rng = np.random.default_rng(seed)

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


# =============================================================================
# Query builders - return LazyDataFrame without executing
# =============================================================================


def simple_filter(df: pd.DataFrame):
    """Single filter."""
    return df.select().filter(col("a") > 0.5)


def chained_filters(df: pd.DataFrame):
    """Multiple chained filters."""
    return (
        df.select().filter(col("a") > 0.5).filter(col("b") < 0.8).filter(col("c") > 0.2)
    )


def filter_project(df: pd.DataFrame):
    """Filter then project."""
    return df.select().filter(col("a") > 0.5).select(col("a"), col("b"), col("c"))


def complex_expression(df: pd.DataFrame):
    """Complex computed columns."""
    return df.select().select(
        col("a"),
        (col("a") + col("b") * 2).alias("computed1"),
        ((col("c") - col("a")) / (col("b") + 1)).alias("computed2"),
        (col("d") * col("e")).alias("computed3"),
    )


def filter_compute_filter(df: pd.DataFrame):
    """Filter -> compute -> filter pattern."""
    return (
        df.select()
        .filter(col("a") > 0.3)
        .select(col("a"), col("b"), (col("a") + col("b")).alias("sum_ab"))
        .filter(col("sum_ab") > 1.0)
    )


def aggregation_query(df: pd.DataFrame):
    """GroupBy aggregation."""
    return (
        df.select()
        .group_by(col("category"))
        .agg(
            col("a").mean().alias("avg_a"),
            col("b").sum().alias("total_b"),
            col("d").max().alias("max_d"),
        )
    )


def sort_limit(df: pd.DataFrame):
    """Sort with limit (TopK candidate)."""
    return df.select().sort(col("a"), descending=True).head(100)


def cse_candidate(df: pd.DataFrame):
    """Query with common subexpressions."""
    expensive = col("a") * col("b") + col("c")
    return df.select().select(
        expensive.alias("expr1"),
        (expensive * 2).alias("expr2"),
        (expensive + 1).alias("expr3"),
    )


QUERIES = {
    "simple_filter": simple_filter,
    "chained_filters": chained_filters,
    "filter_project": filter_project,
    "complex_expression": complex_expression,
    "filter_compute_filter": filter_compute_filter,
    "aggregation": aggregation_query,
    "sort_limit": sort_limit,
    "cse_candidate": cse_candidate,
}


# =============================================================================
# Phase benchmarks
# =============================================================================


@dataclass
class PhaseTimings:
    """Timing breakdown for each phase."""

    query_name: str
    n_rows: int
    plan_construction_us: float
    optimization_total_us: float
    physical_planning_us: float
    execution_ms: float
    # Per-pass breakdown
    pass_timings_us: dict[str, float]


def benchmark_phases(
    df: pd.DataFrame, query_builder: Callable, query_name: str
) -> PhaseTimings:
    """
    Benchmark each phase of lazy execution separately.

    Returns detailed timing breakdown.
    """
    from pandas.lazy.physical import PhysicalPlanner

    # Phase 1: Logical plan construction
    def build_plan():
        lf = query_builder(df)
        return lf._plan

    plan_us, _ = time_function_us(build_plan, n_runs=100)

    # Get the plan for subsequent phases
    lf = query_builder(df)
    plan = lf._plan

    # Phase 2: Individual optimization passes
    pass_timings = {}
    all_passes = [
        ("ConstantFolding", ConstantFolding()),
        ("ExpressionSimplification", ExpressionSimplification()),
        ("FilterFusion", FilterFusion()),
        ("PredicatePushdown", PredicatePushdown()),
        ("AggregatePushdown", AggregatePushdown()),
        ("ProjectionPruning", ProjectionPruning()),
        ("LimitPushdown", LimitPushdown()),
        ("SortLimitToTopK", SortLimitToTopK()),
        ("CSE", CommonSubexpressionElimination()),
        ("DeadCodeElimination", DeadCodeElimination()),
        ("EngineSelection", EngineSelection()),
        ("ConversionElimination", ConversionElimination()),
    ]

    current_plan = plan
    total_opt_us = 0.0

    for pass_name, pass_instance in all_passes:

        def run_pass(p=pass_instance, pl=current_plan):
            return p.optimize(pl)

        pass_us, _ = time_function_us(run_pass, n_runs=50)
        pass_timings[pass_name] = pass_us
        total_opt_us += pass_us
        current_plan = pass_instance.optimize(current_plan)

    optimized_plan = current_plan

    # Phase 3: Physical planning
    def physical_plan():
        planner = PhysicalPlanner()
        return planner.plan(optimized_plan)

    physical_us, _ = time_function_us(physical_plan, n_runs=50)

    # Phase 4: Execution (includes physical planning in real usage)
    def execute():
        return lf.collect()

    exec_ms, _ = time_function_ms(execute, n_runs=10)

    return PhaseTimings(
        query_name=query_name,
        n_rows=len(df),
        plan_construction_us=plan_us,
        optimization_total_us=total_opt_us,
        physical_planning_us=physical_us,
        execution_ms=exec_ms,
        pass_timings_us=pass_timings,
    )


def benchmark_skip_optimizer(df: pd.DataFrame, query_builder: Callable) -> dict:
    """Compare execution with and without optimization."""

    # With optimization (default)
    def with_opt():
        return query_builder(df).collect(optimize=True)

    # Without optimization
    def without_opt():
        return query_builder(df).collect(optimize=False)

    with_opt_ms, _ = time_function_ms(with_opt, n_runs=10)
    without_opt_ms, _ = time_function_ms(without_opt, n_runs=10)

    return {
        "with_optimization_ms": with_opt_ms,
        "without_optimization_ms": without_opt_ms,
        "optimization_overhead_ms": with_opt_ms - without_opt_ms,
    }


# =============================================================================
# Main benchmark
# =============================================================================


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_phase_breakdown(timings: PhaseTimings):
    """Print detailed phase breakdown."""
    total_planning_us = (
        timings.plan_construction_us
        + timings.optimization_total_us
        + timings.physical_planning_us
    )
    total_planning_ms = total_planning_us / 1000
    exec_ms = timings.execution_ms

    print(f"\n--- {timings.query_name} ({timings.n_rows:,} rows) ---")
    print(f"{'Phase':<30} {'Time':>12} {'% of Planning':>15}")
    print("-" * 60)

    pct_plan = timings.plan_construction_us / total_planning_us * 100
    pct_opt = timings.optimization_total_us / total_planning_us * 100
    pct_phys = timings.physical_planning_us / total_planning_us * 100

    plan_us = timings.plan_construction_us
    opt_us = timings.optimization_total_us
    phys_us = timings.physical_planning_us
    print(f"{'Plan construction':<30} {plan_us:>9.1f} us {pct_plan:>10.1f}%")
    print(f"{'Optimization (total)':<30} {opt_us:>9.1f} us {pct_opt:>10.1f}%")
    print(f"{'Physical planning':<30} {phys_us:>9.1f} us {pct_phys:>10.1f}%")
    print("-" * 60)
    print(f"{'Total planning overhead':<30} {total_planning_ms:>9.3f} ms")
    print(f"{'Execution':<30} {exec_ms:>9.3f} ms")
    print(f"{'Planning / Execution ratio':<30} {total_planning_ms / exec_ms:>9.1%}")


def print_pass_breakdown(timings: PhaseTimings):
    """Print per-pass timing breakdown."""
    print("\n  Optimization pass breakdown:")
    print(f"  {'Pass':<28} {'Time (us)':>10} {'%':>8}")
    print("  " + "-" * 50)

    total = timings.optimization_total_us
    sorted_passes = sorted(
        timings.pass_timings_us.items(), key=lambda x: x[1], reverse=True
    )

    for pass_name, pass_us in sorted_passes:
        pct = pass_us / total * 100 if total > 0 else 0
        print(f"  {pass_name:<28} {pass_us:>10.1f} {pct:>7.1f}%")


def run_benchmarks():
    """Run all planning phase benchmarks."""
    print_header("LAZY PANDAS PLANNING PHASE BENCHMARK")
    print("\nThis benchmark breaks down where lazy execution overhead comes from.")
    print("Understanding this helps design 'fast path' heuristics.\n")

    # Test different data sizes
    sizes = [1_000, 10_000, 100_000, 1_000_000]

    # First: Show phase breakdown for each query type at 100K rows
    print_header("PHASE BREAKDOWN BY QUERY TYPE (100K rows)")

    df_100k = create_test_data(100_000)
    for query_name, query_builder in QUERIES.items():
        try:
            timings = benchmark_phases(df_100k, query_builder, query_name)
            print_phase_breakdown(timings)
            print_pass_breakdown(timings)
        except Exception as e:
            print(f"\n--- {query_name} ---")
            print(f"  Error: {e}")

    # Second: Show how planning overhead scales with data size
    print_header("PLANNING OVERHEAD VS DATA SIZE")
    print("\nQuery: chained_filters (3 sequential filters)")
    print(f"{'Rows':<12} {'Planning (ms)':<15} {'Execution (ms)':<15} {'Ratio':>10}")
    print("-" * 55)

    for size in sizes:
        df = create_test_data(size)
        timings = benchmark_phases(df, chained_filters, "chained_filters")
        planning_ms = (
            timings.plan_construction_us
            + timings.optimization_total_us
            + timings.physical_planning_us
        ) / 1000
        ratio = planning_ms / timings.execution_ms
        exec_ms = timings.execution_ms
        print(f"{size:<12,} {planning_ms:<15.3f} {exec_ms:<15.3f} {ratio:>9.1%}")

    # Third: Compare with vs without optimizer
    print_header("OPTIMIZATION OVERHEAD (skip optimizer?)")
    print("\nComparing collect(optimize=True) vs collect(optimize=False)")
    print(f"{'Query':<25} {'With Opt (ms)':<15} {'No Opt (ms)':<15} {'Overhead':>12}")
    print("-" * 70)

    df_100k = create_test_data(100_000)
    for query_name, query_builder in QUERIES.items():
        try:
            results = benchmark_skip_optimizer(df_100k, query_builder)
            overhead = results["optimization_overhead_ms"]
            print(
                f"{query_name:<25} "
                f"{results['with_optimization_ms']:<15.3f} "
                f"{results['without_optimization_ms']:<15.3f} "
                f"{overhead:>+11.3f} ms"
            )
        except Exception as e:
            print(f"{query_name:<25} Error: {e}")

    # Fourth: Identify most expensive passes
    print_header("MOST EXPENSIVE OPTIMIZATION PASSES (averaged across queries)")

    pass_totals: dict[str, list[float]] = {}
    df_100k = create_test_data(100_000)

    for query_name, query_builder in QUERIES.items():
        try:
            timings = benchmark_phases(df_100k, query_builder, query_name)
            for pass_name, pass_us in timings.pass_timings_us.items():
                if pass_name not in pass_totals:
                    pass_totals[pass_name] = []
                pass_totals[pass_name].append(pass_us)
        except Exception:
            pass

    print(f"\n{'Pass':<30} {'Avg (us)':<12} {'Max (us)':<12}")
    print("-" * 55)

    sorted_passes = sorted(
        pass_totals.items(), key=lambda x: np.mean(x[1]), reverse=True
    )
    for pass_name, times in sorted_passes:
        avg = np.mean(times)
        max_time = np.max(times)
        print(f"{pass_name:<30} {avg:<12.1f} {max_time:<12.1f}")

    # Summary and recommendations
    print_header("RECOMMENDATIONS")
    print(
        """
Based on the benchmark results:

1. PLANNING OVERHEAD BREAKDOWN
   - Plan construction: Usually < 10% of planning time
   - Optimization passes: Usually 60-80% of planning time
   - Physical planning: Usually 10-30% of planning time

2. FAST PATH CANDIDATES
   - Simple queries (single filter/select): Skip optimizer entirely
   - Small data (< 10K rows): Consider eager execution
   - No CSE opportunities: Skip CommonSubexpressionElimination

3. EXPENSIVE PASSES TO CONSIDER SKIPPING
   - CSE: Often most expensive, only helps with repeated subexpressions
   - EngineSelection: Can skip if using single backend
   - PredicatePushdown: Can skip for flat queries (no nested operations)

4. ADAPTIVE EXECUTION HEURISTICS
   - If data < 10K rows AND query is simple: use eager path
   - If no filter operations: skip FilterFusion, PredicatePushdown
   - If no limits: skip LimitPushdown, SortLimitToTopK
"""
    )


if __name__ == "__main__":
    run_benchmarks()
