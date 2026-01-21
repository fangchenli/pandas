#!/usr/bin/env python3
"""
Benchmark: Join Performance for Lazy Pandas

Tests the join optimizations:
- Build/probe optimization (building hash table on smaller side)
- Semi and anti join kernels
- Cardinality estimation

Usage:
    python pandas/lazy/benchmarks/bench_join.py
"""

from __future__ import annotations

import numpy as np

import pandas as pd
from pandas.lazy import col
from pandas.lazy.benchmarks.shared import benchmark


def generate_join_data(left_size: int, right_size: int, key_cardinality: int):
    """Generate data for join benchmarks."""
    rng = np.random.default_rng(42)

    # Left table
    left_keys = rng.integers(0, key_cardinality, left_size)
    left_values = rng.standard_normal(left_size)
    left_df = pd.DataFrame({"id": left_keys, "left_val": left_values})

    # Right table
    right_keys = rng.integers(0, key_cardinality, right_size)
    right_values = rng.standard_normal(right_size)
    right_df = pd.DataFrame({"id": right_keys, "right_val": right_values})

    return left_df, right_df


def benchmark_inner_join():
    """Benchmark inner join with build/probe optimization."""
    print("\n" + "=" * 70)
    print("Inner Join Benchmark (Build/Probe Optimization)")
    print("=" * 70)

    scenarios = [
        # (left_size, right_size, key_cardinality, description)
        (100_000, 10_000, 5_000, "Left large, Right small"),
        (10_000, 100_000, 5_000, "Left small, Right large"),
        (100_000, 100_000, 10_000, "Both equal size"),
        (1_000_000, 10_000, 5_000, "Left very large, Right small"),
    ]

    for left_size, right_size, key_card, desc in scenarios:
        print(f"\n{desc}:")
        print(f"  Left: {left_size:,} rows, Right: {right_size:,} rows")

        left_df, right_df = generate_join_data(left_size, right_size, key_card)

        # Eager pandas
        def eager_join():
            return left_df.merge(right_df, on="id")

        eager_result = benchmark(eager_join)

        # Lazy pandas
        def lazy_join():
            return left_df.select().join(right_df.select(), on="id").collect()

        lazy_result = benchmark(lazy_join)

        speedup = eager_result["mean_ms"] / lazy_result["mean_ms"]

        print(f"  Eager pandas:  {eager_result['mean_ms']:.2f} ms")
        print(f"  Lazy pandas:   {lazy_result['mean_ms']:.2f} ms")
        print(f"  Speedup:       {speedup:.2f}x")


def benchmark_semi_anti_join():
    """Benchmark semi and anti joins vs equivalent filter operations."""
    print("\n" + "=" * 70)
    print("Semi/Anti Join Benchmark")
    print("=" * 70)

    from pandas.lazy.backends.numpy.join import (
        numpy_anti_join,
        numpy_semi_join,
    )

    scenarios = [
        (100_000, 10_000, 5_000, "100K left, 10K right"),
        (100_000, 100_000, 10_000, "100K left, 100K right"),
        (1_000_000, 10_000, 5_000, "1M left, 10K right"),
    ]

    for left_size, right_size, key_card, desc in scenarios:
        print(f"\n{desc}:")

        left_df, right_df = generate_join_data(left_size, right_size, key_card)

        # Convert to numpy arrays for kernel test
        left_arrays = {
            "id": left_df["id"].values,
            "left_val": left_df["left_val"].values,
        }
        right_arrays = {
            "id": right_df["id"].values,
            "right_val": right_df["right_val"].values,
        }

        # Semi join kernel
        def semi_join_kernel():
            return numpy_semi_join(left_arrays, right_arrays, keys=["id"])

        semi_result = benchmark(semi_join_kernel)

        # Anti join kernel
        def anti_join_kernel():
            return numpy_anti_join(left_arrays, right_arrays, keys=["id"])

        anti_result = benchmark(anti_join_kernel)

        # Equivalent using isin (semi)
        def isin_semi():
            mask = left_df["id"].isin(right_df["id"])
            return left_df[mask]

        isin_semi_result = benchmark(isin_semi)

        # Equivalent using isin (anti)
        def isin_anti():
            mask = ~left_df["id"].isin(right_df["id"])
            return left_df[mask]

        isin_anti_result = benchmark(isin_anti)

        print(f"  Semi join kernel:  {semi_result['mean_ms']:.2f} ms")
        print(f"  Semi via isin:     {isin_semi_result['mean_ms']:.2f} ms")
        semi_speedup = isin_semi_result["mean_ms"] / semi_result["mean_ms"]
        print(f"  Semi speedup:      {semi_speedup:.2f}x")
        print(f"  Anti join kernel:  {anti_result['mean_ms']:.2f} ms")
        print(f"  Anti via isin:     {isin_anti_result['mean_ms']:.2f} ms")
        anti_speedup = isin_anti_result["mean_ms"] / anti_result["mean_ms"]
        print(f"  Anti speedup:      {anti_speedup:.2f}x")


def benchmark_cardinality_estimation():
    """Verify cardinality estimation is working."""
    print("\n" + "=" * 70)
    print("Cardinality Estimation")
    print("=" * 70)

    left_df = pd.DataFrame({"id": range(10000), "a": range(10000)})
    right_df = pd.DataFrame({"id": range(5000), "b": range(5000)})

    # Test various operations
    print("\nRow count estimates:")

    # Source
    source = left_df.select()._plan
    print(f"  DataFrameSource(10000 rows): {source.estimate_row_count()}")

    # Filter
    filtered = left_df.select().filter(col("a") > 5000)._plan
    print(f"  Filter (a > 5000):           {filtered.estimate_row_count()}")

    # Limit
    limited = left_df.select().head(100)._plan
    print(f"  Limit(100):                  {limited.estimate_row_count()}")

    # Inner join
    inner = left_df.select().join(right_df.select(), on="id")._plan
    print(f"  Inner join:                  {inner.estimate_row_count()}")

    # Left join
    left_j = left_df.select().join(right_df.select(), on="id", how="left")._plan
    print(f"  Left join:                   {left_j.estimate_row_count()}")

    # Cross join
    small_left = pd.DataFrame({"a": range(100)})
    small_right = pd.DataFrame({"b": range(50)})
    cross = small_left.select().join(small_right.select(), how="cross")._plan
    print(f"  Cross join (100 x 50):       {cross.estimate_row_count()}")


def benchmark_build_probe_effect():
    """Show the effect of build/probe optimization."""
    print("\n" + "=" * 70)
    print("Build/Probe Optimization Effect")
    print("=" * 70)

    # Create asymmetric data
    large_size = 500_000
    small_size = 5_000
    key_card = 10_000

    large_df, small_df = generate_join_data(large_size, small_size, key_card)

    print(f"\nAsymmetric join: {large_size:,} x {small_size:,} rows")

    # Large left, small right (optimal order for build/probe - builds on small right)
    def join_large_small():
        return large_df.select().join(small_df.select(), on="id").collect()

    result1 = benchmark(join_large_small)

    # Small left, large right (needs swap for optimal build/probe)
    def join_small_large():
        return small_df.select().join(large_df.select(), on="id").collect()

    result2 = benchmark(join_small_large)

    print(f"  Large.join(Small): {result1['mean_ms']:.2f} ms")
    print(f"  Small.join(Large): {result2['mean_ms']:.2f} ms")
    max_time = max(result1["mean_ms"], result2["mean_ms"])
    min_time = min(result1["mean_ms"], result2["mean_ms"])
    ratio = max_time / min_time
    print(f"  Ratio:             {ratio:.2f}x")
    print("  (Should be similar due to build/probe swap optimization)")


if __name__ == "__main__":
    print("Join Performance Benchmark")
    print("=" * 70)

    benchmark_inner_join()
    benchmark_semi_anti_join()
    benchmark_cardinality_estimation()
    benchmark_build_probe_effect()

    print("\n" + "=" * 70)
    print("Benchmark complete!")
