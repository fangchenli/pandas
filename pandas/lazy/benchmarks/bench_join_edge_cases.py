#!/usr/bin/env python3
"""
Benchmark: Join Edge Cases (Skew, Cardinality Mismatch, Wrong-Side Build)

Tests join performance under challenging conditions:
1. Highly skewed keys - one key dominates (hot key problem)
2. Tiny build vs large probe - asymmetric sizes
3. Wrong-side build - intentionally suboptimal hash table side
4. Cardinality explosion - many-to-many with high fanout

These scenarios stress:
- Spill-to-disk logic (skewed partitions)
- Build/probe side selection
- Grace hash join partitioning
- Runtime adaptation (fallback to sort-merge)

Usage:
    python pandas/lazy/benchmarks/bench_join_edge_cases.py
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from typing import TYPE_CHECKING

import numpy as np

import pandas as pd

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
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    finally:
        gc.enable()

    return float(np.mean(times)), float(np.std(times))


# =============================================================================
# Skewed data generation
# =============================================================================


def create_uniform_keys(n_rows: int, n_keys: int, seed: int = 42) -> np.ndarray:
    """Create uniformly distributed keys."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_keys, n_rows)


def create_skewed_keys(
    n_rows: int,
    n_keys: int,
    hot_key_fraction: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Create skewed keys where one key gets hot_key_fraction of rows.

    Parameters
    ----------
    n_rows : int
        Total number of rows
    n_keys : int
        Number of unique keys
    hot_key_fraction : float
        Fraction of rows that have the hot key (0.5 = 50%)
    seed : int
        Random seed

    Returns
    -------
    np.ndarray
        Array of keys with skewed distribution
    """
    rng = np.random.default_rng(seed)

    n_hot = int(n_rows * hot_key_fraction)
    n_cold = n_rows - n_hot

    # Hot key is always 0
    hot_keys = np.zeros(n_hot, dtype=np.int64)

    # Cold keys are uniformly distributed among remaining keys
    cold_keys = rng.integers(1, n_keys, n_cold)

    # Combine and shuffle
    keys = np.concatenate([hot_keys, cold_keys])
    rng.shuffle(keys)

    return keys


def create_zipf_keys(
    n_rows: int, n_keys: int, alpha: float = 1.5, seed: int = 42
) -> np.ndarray:
    """
    Create Zipf-distributed keys (more realistic skew).

    Alpha controls skew: higher = more skewed
    - alpha=1.0: mild skew
    - alpha=1.5: moderate skew
    - alpha=2.0: heavy skew
    """
    rng = np.random.default_rng(seed)

    # Generate Zipf distribution
    ranks = np.arange(1, n_keys + 1)
    weights = 1.0 / (ranks**alpha)
    weights /= weights.sum()

    # Sample keys according to weights
    keys = rng.choice(n_keys, size=n_rows, p=weights)
    return keys


def create_join_dataframe(
    n_rows: int, keys: np.ndarray, prefix: str, seed: int = 42
) -> pd.DataFrame:
    """Create a DataFrame with given keys and value columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "key": keys,
            f"{prefix}_val1": rng.random(n_rows),
            f"{prefix}_val2": rng.random(n_rows),
            f"{prefix}_id": np.arange(n_rows),
        }
    )


# =============================================================================
# Benchmark results
# =============================================================================


@dataclass
class JoinResult:
    """Result for a join benchmark."""

    scenario: str
    left_rows: int
    right_rows: int
    result_rows: int
    eager_ms: float
    lazy_ms: float
    description: str = ""

    @property
    def speedup(self) -> float:
        return self.eager_ms / self.lazy_ms if self.lazy_ms > 0 else float("inf")

    @property
    def fanout(self) -> float:
        """Result rows / max(left, right) - measures explosion."""
        return self.result_rows / max(self.left_rows, self.right_rows)


# =============================================================================
# Join benchmarks
# =============================================================================


def benchmark_uniform_join(left_rows: int, right_rows: int, n_keys: int) -> JoinResult:
    """Baseline: uniform key distribution."""
    left_keys = create_uniform_keys(left_rows, n_keys, seed=42)
    right_keys = create_uniform_keys(right_rows, n_keys, seed=43)

    left_df = create_join_dataframe(left_rows, left_keys, "left", seed=44)
    right_df = create_join_dataframe(right_rows, right_keys, "right", seed=45)

    # Eager
    def eager():
        return left_df.merge(right_df, on="key")

    eager_ms, _ = time_operation(eager)
    result_rows = len(left_df.merge(right_df, on="key"))

    # Lazy
    def lazy():
        return left_df.select().join(right_df.select(), on="key").collect()

    lazy_ms, _ = time_operation(lazy)

    return JoinResult(
        scenario="uniform",
        left_rows=left_rows,
        right_rows=right_rows,
        result_rows=result_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        description=f"Uniform keys, {n_keys} unique",
    )


def benchmark_skewed_join(
    left_rows: int, right_rows: int, n_keys: int, hot_fraction: float = 0.5
) -> JoinResult:
    """Skewed: one key dominates one side."""
    # Left side is skewed
    left_keys = create_skewed_keys(left_rows, n_keys, hot_fraction, seed=42)
    # Right side is uniform
    right_keys = create_uniform_keys(right_rows, n_keys, seed=43)

    left_df = create_join_dataframe(left_rows, left_keys, "left", seed=44)
    right_df = create_join_dataframe(right_rows, right_keys, "right", seed=45)

    # Eager
    def eager():
        return left_df.merge(right_df, on="key")

    eager_ms, _ = time_operation(eager)
    result_rows = len(left_df.merge(right_df, on="key"))

    # Lazy
    def lazy():
        return left_df.select().join(right_df.select(), on="key").collect()

    lazy_ms, _ = time_operation(lazy)

    return JoinResult(
        scenario=f"skewed_{int(hot_fraction * 100)}pct",
        left_rows=left_rows,
        right_rows=right_rows,
        result_rows=result_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        description=f"Hot key has {hot_fraction:.0%} of left rows",
    )


def benchmark_both_skewed_join(
    left_rows: int, right_rows: int, n_keys: int, hot_fraction: float = 0.5
) -> JoinResult:
    """Both sides skewed on same key - worst case for hash join."""
    left_keys = create_skewed_keys(left_rows, n_keys, hot_fraction, seed=42)
    right_keys = create_skewed_keys(right_rows, n_keys, hot_fraction, seed=43)

    left_df = create_join_dataframe(left_rows, left_keys, "left", seed=44)
    right_df = create_join_dataframe(right_rows, right_keys, "right", seed=45)

    # Eager
    def eager():
        return left_df.merge(right_df, on="key")

    eager_ms, _ = time_operation(eager)
    result_rows = len(left_df.merge(right_df, on="key"))

    # Lazy
    def lazy():
        return left_df.select().join(right_df.select(), on="key").collect()

    lazy_ms, _ = time_operation(lazy)

    return JoinResult(
        scenario="both_skewed",
        left_rows=left_rows,
        right_rows=right_rows,
        result_rows=result_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        description=f"Both sides {hot_fraction:.0%} skewed - WORST CASE",
    )


def benchmark_zipf_join(
    left_rows: int, right_rows: int, n_keys: int, alpha: float = 1.5
) -> JoinResult:
    """Zipf distribution - realistic skew pattern."""
    left_keys = create_zipf_keys(left_rows, n_keys, alpha, seed=42)
    right_keys = create_zipf_keys(right_rows, n_keys, alpha, seed=43)

    left_df = create_join_dataframe(left_rows, left_keys, "left", seed=44)
    right_df = create_join_dataframe(right_rows, right_keys, "right", seed=45)

    # Eager
    def eager():
        return left_df.merge(right_df, on="key")

    eager_ms, _ = time_operation(eager)
    result_rows = len(left_df.merge(right_df, on="key"))

    # Lazy
    def lazy():
        return left_df.select().join(right_df.select(), on="key").collect()

    lazy_ms, _ = time_operation(lazy)

    return JoinResult(
        scenario=f"zipf_alpha{alpha}",
        left_rows=left_rows,
        right_rows=right_rows,
        result_rows=result_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        description=f"Zipf distribution (alpha={alpha})",
    )


def benchmark_tiny_build_large_probe(
    build_rows: int, probe_rows: int, n_keys: int
) -> JoinResult:
    """Tiny build side, large probe side - optimal for hash join."""
    # Small dimension table
    build_keys = np.arange(n_keys)  # Unique keys
    build_df = create_join_dataframe(
        build_rows, build_keys[:build_rows], "dim", seed=42
    )

    # Large fact table with foreign keys
    probe_keys = create_uniform_keys(probe_rows, build_rows, seed=43)
    probe_df = create_join_dataframe(probe_rows, probe_keys, "fact", seed=44)

    # Eager
    def eager():
        return probe_df.merge(build_df, on="key")

    eager_ms, _ = time_operation(eager)
    result_rows = len(probe_df.merge(build_df, on="key"))

    # Lazy
    def lazy():
        return probe_df.select().join(build_df.select(), on="key").collect()

    lazy_ms, _ = time_operation(lazy)

    return JoinResult(
        scenario="tiny_build",
        left_rows=probe_rows,
        right_rows=build_rows,
        result_rows=result_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        description=f"Probe {probe_rows:,} x Build {build_rows:,} (optimal)",
    )


def benchmark_wrong_side_build(
    small_rows: int, large_rows: int, n_keys: int
) -> JoinResult:
    """
    Wrong side build - intentionally put large table on build side.

    This tests if the optimizer correctly swaps build/probe sides.
    """
    small_keys = create_uniform_keys(small_rows, n_keys, seed=42)
    large_keys = create_uniform_keys(large_rows, n_keys, seed=43)

    small_df = create_join_dataframe(small_rows, small_keys, "small", seed=44)
    large_df = create_join_dataframe(large_rows, large_keys, "large", seed=45)

    # Eager - large on left (pandas will handle this)
    def eager():
        return large_df.merge(small_df, on="key")

    eager_ms, _ = time_operation(eager)
    result_rows = len(large_df.merge(small_df, on="key"))

    # Lazy - large on left, optimizer should swap
    def lazy():
        return large_df.select().join(small_df.select(), on="key").collect()

    lazy_ms, _ = time_operation(lazy)

    return JoinResult(
        scenario="wrong_side",
        left_rows=large_rows,
        right_rows=small_rows,
        result_rows=result_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        description=f"Large ({large_rows:,}) left, Small ({small_rows:,}) right",
    )


def benchmark_cardinality_explosion(
    left_rows: int, right_rows: int, duplicates_per_key: int
) -> JoinResult:
    """
    Many-to-many join with high fanout.

    Each key appears multiple times on both sides, causing result explosion.
    """
    n_keys = min(left_rows, right_rows) // duplicates_per_key

    # Create keys with duplicates
    rng = np.random.default_rng(42)
    left_keys = np.repeat(np.arange(n_keys), duplicates_per_key)
    rng.shuffle(left_keys)
    left_keys = left_keys[:left_rows]

    rng = np.random.default_rng(43)
    right_keys = np.repeat(np.arange(n_keys), duplicates_per_key)
    rng.shuffle(right_keys)
    right_keys = right_keys[:right_rows]

    left_df = create_join_dataframe(left_rows, left_keys, "left", seed=44)
    right_df = create_join_dataframe(right_rows, right_keys, "right", seed=45)

    # Eager
    def eager():
        return left_df.merge(right_df, on="key")

    eager_ms, _ = time_operation(eager)
    result_rows = len(left_df.merge(right_df, on="key"))

    # Lazy
    def lazy():
        return left_df.select().join(right_df.select(), on="key").collect()

    lazy_ms, _ = time_operation(lazy)

    return JoinResult(
        scenario=f"explosion_{duplicates_per_key}x",
        left_rows=left_rows,
        right_rows=right_rows,
        result_rows=result_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        description=f"{duplicates_per_key}x duplicates per key",
    )


def benchmark_no_matches(left_rows: int, right_rows: int) -> JoinResult:
    """No matching keys - worst case for probe."""
    # Disjoint key ranges
    left_keys = np.arange(0, left_rows)
    right_keys = np.arange(left_rows, left_rows + right_rows)

    left_df = create_join_dataframe(left_rows, left_keys, "left", seed=42)
    right_df = create_join_dataframe(right_rows, right_keys, "right", seed=43)

    # Eager
    def eager():
        return left_df.merge(right_df, on="key")

    eager_ms, _ = time_operation(eager)
    result_rows = 0

    # Lazy
    def lazy():
        return left_df.select().join(right_df.select(), on="key").collect()

    lazy_ms, _ = time_operation(lazy)

    return JoinResult(
        scenario="no_matches",
        left_rows=left_rows,
        right_rows=right_rows,
        result_rows=result_rows,
        eager_ms=eager_ms,
        lazy_ms=lazy_ms,
        description="Disjoint keys - 0 matches",
    )


# =============================================================================
# Main benchmark
# =============================================================================


def print_header(title: str):
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)


def print_results_table(results: list[JoinResult]):
    """Print results as a table."""
    print(
        f"\n{'Scenario':<20} {'Left':>10} {'Right':>10} {'Result':>12} "
        f"{'Eager':>10} {'Lazy':>10} {'Speedup':>10} {'Fanout':>8}"
    )
    print("-" * 95)

    for r in results:
        speedup_str = f"{r.speedup:.2f}x"
        print(
            f"{r.scenario:<20} {r.left_rows:>10,} {r.right_rows:>10,} "
            f"{r.result_rows:>12,} {r.eager_ms:>10.1f} {r.lazy_ms:>10.1f} "
            f"{speedup_str:>10} {r.fanout:>8.1f}"
        )
        if r.description:
            print(f"  └─ {r.description}")


def run_join_edge_case_benchmarks():
    """Run all join edge case benchmarks."""
    print_header("JOIN EDGE CASE BENCHMARK")
    print(
        """
This benchmark tests join performance under challenging conditions:
- Skewed keys (hot key problem)
- Size mismatch (tiny build vs large probe)
- Wrong-side build (optimizer should swap)
- Cardinality explosion (many-to-many)

These scenarios stress spill logic, build/probe selection, and adaptive execution.
"""
    )

    # Configuration (reduced for memory safety)
    left_rows = 50_000
    right_rows = 50_000
    n_keys = 5_000

    results: list[JoinResult] = []

    # Baseline: uniform distribution
    print_header("BASELINE: UNIFORM KEY DISTRIBUTION")
    result = benchmark_uniform_join(left_rows, right_rows, n_keys)
    results.append(result)
    print_results_table([result])

    # Skew benchmarks
    print_header("KEY SKEW SCENARIOS")

    skew_results = []

    # One side skewed
    for hot_fraction in [0.3, 0.5, 0.7, 0.9]:
        result = benchmark_skewed_join(left_rows, right_rows, n_keys, hot_fraction)
        skew_results.append(result)
        results.append(result)

    # Both sides skewed (worst case)
    result = benchmark_both_skewed_join(left_rows, right_rows, n_keys, 0.5)
    skew_results.append(result)
    results.append(result)

    # Zipf distribution
    for alpha in [1.0, 1.5, 2.0]:
        result = benchmark_zipf_join(left_rows, right_rows, n_keys, alpha)
        skew_results.append(result)
        results.append(result)

    print_results_table(skew_results)

    # Size mismatch benchmarks
    print_header("SIZE MISMATCH SCENARIOS")

    size_results = []

    # Tiny build, large probe (optimal)
    result = benchmark_tiny_build_large_probe(1_000, 200_000, 1_000)
    size_results.append(result)
    results.append(result)

    result = benchmark_tiny_build_large_probe(5_000, 200_000, 5_000)
    size_results.append(result)
    results.append(result)

    # Wrong side build (should be swapped by optimizer)
    result = benchmark_wrong_side_build(5_000, 200_000, 5_000)
    size_results.append(result)
    results.append(result)

    print_results_table(size_results)

    # Cardinality explosion
    print_header("CARDINALITY EXPLOSION (MANY-TO-MANY)")

    explosion_results = []

    for dup in [2, 5, 10]:
        # Use smaller sizes for higher duplicates to avoid memory issues
        rows = 30_000 if dup <= 5 else 20_000
        result = benchmark_cardinality_explosion(rows, rows, dup)
        explosion_results.append(result)
        results.append(result)

    print_results_table(explosion_results)

    # Edge cases
    print_header("EDGE CASES")

    edge_results = []

    # No matches
    result = benchmark_no_matches(50_000, 50_000)
    edge_results.append(result)
    results.append(result)

    print_results_table(edge_results)

    # Summary
    print_header("SUMMARY")

    print("\nBy scenario type:")
    print(f"  Uniform baseline:     speedup = {results[0].speedup:.2f}x")

    skew_speedups = [
        r.speedup for r in results if "skew" in r.scenario or "zipf" in r.scenario
    ]
    if skew_speedups:
        print(f"  Skewed keys:          speedup = {np.mean(skew_speedups):.2f}x (avg)")

    size_speedups = [
        r.speedup for r in results if "tiny" in r.scenario or "wrong" in r.scenario
    ]
    if size_speedups:
        print(f"  Size mismatch:        speedup = {np.mean(size_speedups):.2f}x (avg)")

    explosion_speedups = [r.speedup for r in results if "explosion" in r.scenario]
    if explosion_speedups:
        avg = np.mean(explosion_speedups)
        print(f"  Cardinality explosion: speedup = {avg:.2f}x (avg)")

    # Recommendations
    print_header("RECOMMENDATIONS")
    print(
        """
Based on join edge case analysis:

1. SKEW HANDLING
   - Detect skew during partitioning (empty partition ratio, max partition size)
   - Consider salting hot keys to spread load
   - Fall back to sort-merge for pathological skew

2. BUILD/PROBE SELECTION
   - ALWAYS build hash table on smaller side
   - Estimate cardinality before deciding
   - Consider sampling for unknown sizes

3. CARDINALITY EXPLOSION
   - Monitor result size during execution
   - Consider early termination for exploding joins
   - Use streaming output to avoid memory pressure

4. GRACE HASH JOIN TUNING
   - More partitions for skewed data (128-256 vs default 64)
   - Smaller partitions = better skew tolerance
   - Track partition size statistics for adaptive decisions

5. ADAPTIVE EXECUTION
   - Start with hash join
   - Monitor spill behavior
   - Switch to sort-merge if pathological:
     * Max partition > memory budget
     * Skew ratio > 10x
     * Empty partitions > 50%
"""
    )


if __name__ == "__main__":
    run_join_edge_case_benchmarks()
