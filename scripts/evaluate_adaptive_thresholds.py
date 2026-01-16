#!/usr/bin/env python
"""
Evaluate the adaptive threshold system for lazy pandas.

This script compares execution performance under three configurations:
1. Default thresholds (static)
2. Calibrated thresholds (from benchmark calibration)
3. Adaptive thresholds (runtime-adjusted)

It simulates various workload patterns to test how well the adaptive
system learns optimal thresholds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import numpy as np

import pandas as pd
from pandas.lazy import col
from pandas.lazy.optimize.adaptive import (
    get_adaptive_manager,
    reset_adaptive_manager,
)
from pandas.lazy.optimize.config import (
    ThresholdConfig,
    reset_threshold_config,
    set_threshold_config,
)


@dataclass
class WorkloadResult:
    """Result from running a workload."""

    name: str
    config_type: str
    total_time_ms: float
    operations: int
    avg_time_ms: float


def create_test_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Create test DataFrame with mixed types."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "value": rng.standard_normal(n_rows),
            "category": rng.choice(["A", "B", "C", "D"], n_rows),
            "flag": rng.choice([True, False], n_rows),
        }
    )


def run_filter_workload(
    df: pd.DataFrame,
    n_iterations: int = 50,
) -> tuple[float, int]:
    """
    Run filter operations and return total time and count.

    Uses lazy evaluation with .select() and .collect().
    Uses physical planner for execution (required for adaptive thresholds).
    """
    total_time = 0.0
    operations = 0

    for i in range(n_iterations):
        # Vary the filter threshold to test different sizes
        threshold = 0.5 + (i % 10) * 0.05  # 0.5 to 0.95

        start = time.perf_counter()
        _ = (
            df.select()
            .filter(col("value") > threshold)
            .collect(use_physical_planner=True)
        )
        elapsed = time.perf_counter() - start

        total_time += elapsed * 1000  # Convert to ms
        operations += 1

        # Also do compound filters
        start = time.perf_counter()
        _ = (
            df.select()
            .filter((col("value") > 0) & col("flag"))
            .collect(use_physical_planner=True)
        )
        elapsed = time.perf_counter() - start

        total_time += elapsed * 1000
        operations += 1

    return total_time, operations


def run_mixed_workload(
    df: pd.DataFrame,
    n_iterations: int = 30,
) -> tuple[float, int]:
    """
    Run a mix of operations to test adaptive learning.
    Uses physical planner for execution.
    """
    total_time = 0.0
    operations = 0

    for _ in range(n_iterations):
        # Filter
        start = time.perf_counter()
        _ = df.select().filter(col("value") > 0).collect(use_physical_planner=True)
        total_time += (time.perf_counter() - start) * 1000
        operations += 1

        # Projection with computation
        start = time.perf_counter()
        _ = (
            df.select()
            .with_columns(
                (col("value") * 2 + 1).alias("computed"),
                (col("value") ** 2).alias("squared"),
            )
            .collect(use_physical_planner=True)
        )
        total_time += (time.perf_counter() - start) * 1000
        operations += 1

        # Filter + projection
        start = time.perf_counter()
        _ = (
            df.select()
            .filter(col("category") == "A")
            .with_columns((col("value") + 10).alias("adjusted"))
            .collect(use_physical_planner=True)
        )
        total_time += (time.perf_counter() - start) * 1000
        operations += 1

    return total_time, operations


def run_varying_size_workload(
    sizes: list[int],
    n_iterations_per_size: int = 10,
) -> tuple[float, int]:
    """
    Run operations on varying data sizes to test threshold adaptation.
    Uses physical planner for execution.
    """
    total_time = 0.0
    operations = 0

    for size in sizes:
        df = create_test_data(size)

        for _ in range(n_iterations_per_size):
            start = time.perf_counter()
            _ = df.select().filter(col("value") > 0).collect(use_physical_planner=True)
            total_time += (time.perf_counter() - start) * 1000
            operations += 1

    return total_time, operations


def evaluate_config(
    workload_name: str,
    config_name: str,
    config: ThresholdConfig | None,
    adaptive: bool,
    workload_fn,
) -> tuple[WorkloadResult, dict | None]:
    """Evaluate a workload with a specific configuration.

    Returns the result and optionally the adaptive statistics (if adaptive=True).
    """
    # Reset state
    reset_threshold_config()
    reset_adaptive_manager()

    if config is not None:
        set_threshold_config(config)

    # Set adaptive mode if requested
    if adaptive:
        pd.set_option("compute.lazy.adaptive_thresholds", True)
    else:
        pd.set_option("compute.lazy.adaptive_thresholds", False)

    # Warmup
    warmup_df = create_test_data(10_000)
    _ = warmup_df.select().filter(col("value") > 0).collect(use_physical_planner=True)

    # Run workload
    total_time, operations = workload_fn()

    # Capture adaptive stats before reset (if adaptive)
    adaptive_stats = None
    if adaptive:
        adaptive_stats = get_adaptive_manager().get_statistics()

    # Reset
    pd.set_option("compute.lazy.adaptive_thresholds", False)
    reset_threshold_config()

    result = WorkloadResult(
        name=workload_name,
        config_type=config_name,
        total_time_ms=total_time,
        operations=operations,
        avg_time_ms=total_time / operations if operations > 0 else 0,
    )
    return result, adaptive_stats


def print_results(results: list[WorkloadResult]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Group by workload
    workloads = {}
    for r in results:
        if r.name not in workloads:
            workloads[r.name] = []
        workloads[r.name].append(r)

    for workload_name, workload_results in workloads.items():
        print(f"\n{workload_name}")
        print("-" * 50)
        print(f"{'Config':<20} {'Total (ms)':<15} {'Avg (ms)':<15}")
        print("-" * 50)

        # Sort by total time
        workload_results.sort(key=lambda x: x.total_time_ms)
        baseline = workload_results[0].total_time_ms

        for r in workload_results:
            speedup = baseline / r.total_time_ms if r.total_time_ms > 0 else 0
            speedup_str = f"({speedup:.2f}x)" if speedup != 1.0 else "(baseline)"
            print(
                f"{r.config_type:<20} {r.total_time_ms:>10.2f}     "
                f"{r.avg_time_ms:>10.3f}     {speedup_str}"
            )


def print_adaptive_statistics(stats: dict | None) -> None:
    """Print statistics collected by the adaptive manager."""
    print("\n" + "=" * 70)
    print("ADAPTIVE THRESHOLD STATISTICS")
    print("=" * 70)

    if stats is None:
        print("\nNo adaptive statistics available.")
        return

    print(f"\nTotal executions recorded: {stats['total_executions']}")

    if stats["operations"]:
        print("\nPer-operation statistics:")
        for op, op_stats in stats["operations"].items():
            print(f"\n  {op}:")
            print(f"    Arrow samples: {op_stats['arrow_samples']}")
            print(f"    NumPy samples: {op_stats['numpy_samples']}")
            arrow_tp = op_stats["arrow_throughput_ema"]
            numpy_tp = op_stats["numpy_throughput_ema"]
            print(f"    Arrow throughput (rows/ms): {arrow_tp:.2f}")
            print(f"    NumPy throughput (rows/ms): {numpy_tp:.2f}")
            print(f"    Crossover estimate: {op_stats['crossover_estimate']}")
            print(f"    Confidence: {op_stats['confidence']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptive threshold system")
    parser.add_argument(
        "--size",
        type=int,
        default=100_000,
        help="Default data size for workloads",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations per workload",
    )
    parser.add_argument(
        "--workload",
        choices=["filter", "mixed", "varying", "all"],
        default="all",
        help="Which workload to run",
    )
    args = parser.parse_args()

    print("Adaptive Threshold Evaluation")
    print("=" * 70)
    print(f"Data size: {args.size:,} rows")
    print(f"Iterations: {args.iterations}")

    # Create test data
    df = create_test_data(args.size)

    # Define configurations to test
    configs = [
        ("Default", None, False),
        ("Adaptive", None, True),
        # Optimistic config (low thresholds - prefer Arrow)
        ("LowThreshold", ThresholdConfig(filter_arrow_threshold=10_000), False),
        # Conservative config (high thresholds - prefer NumPy)
        ("HighThreshold", ThresholdConfig(filter_arrow_threshold=200_000), False),
    ]

    results = []
    adaptive_stats = None

    # Run filter workload
    if args.workload in ["filter", "all"]:
        print("\n\nRunning filter workload...")
        for config_name, config, adaptive in configs:
            print(f"  Testing {config_name}...")
            result, stats = evaluate_config(
                "Filter Workload",
                config_name,
                config,
                adaptive,
                lambda df=df, n=args.iterations: run_filter_workload(df, n),
            )
            results.append(result)
            if stats is not None:
                adaptive_stats = stats

    # Run mixed workload
    if args.workload in ["mixed", "all"]:
        print("\n\nRunning mixed workload...")
        for config_name, config, adaptive in configs:
            print(f"  Testing {config_name}...")
            result, stats = evaluate_config(
                "Mixed Workload",
                config_name,
                config,
                adaptive,
                lambda df=df, n=args.iterations: run_mixed_workload(df, n),
            )
            results.append(result)
            if stats is not None:
                adaptive_stats = stats

    # Run varying size workload
    if args.workload in ["varying", "all"]:
        print("\n\nRunning varying size workload...")
        sizes = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
        for config_name, config, adaptive in configs:
            print(f"  Testing {config_name}...")
            result, stats = evaluate_config(
                "Varying Size Workload",
                config_name,
                config,
                adaptive,
                lambda s=sizes: run_varying_size_workload(s, n_iterations_per_size=5),
            )
            results.append(result)
            if stats is not None:
                adaptive_stats = stats

    # Print results
    print_results(results)

    # Show adaptive statistics
    print_adaptive_statistics(adaptive_stats)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The adaptive system should perform close to or better than any fixed
configuration across different workloads. Key observations:

1. For consistent workloads (same data sizes), adaptive overhead is minimal
2. For varying workloads, adaptive learns and adjusts thresholds
3. Initial operations may use suboptimal thresholds until enough data collected
""")


if __name__ == "__main__":
    main()
