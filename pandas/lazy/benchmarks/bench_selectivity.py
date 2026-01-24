#!/usr/bin/env python3
"""
Benchmark: Selectivity-Sensitive Performance

Measures performance across different filter selectivities:
- 1% selectivity (highly selective, few rows pass)
- 10% selectivity
- 50% selectivity
- 90% selectivity (low selectivity, most rows pass)

This matters because:
1. Filter fusion + streaming shine at LOW selectivity (early termination)
2. Branch misprediction + memory bandwidth dominate at HIGH selectivity
3. Helps tune heuristics like "is predicate pushdown worth it?"

Polars and DuckDB both use selectivity internally for optimization decisions.

Usage:
    python pandas/lazy/benchmarks/bench_selectivity.py
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

SELECTIVITIES = [0.01, 0.10, 0.50, 0.90]  # 1%, 10%, 50%, 90%
ROW_COUNTS = [100_000, 1_000_000]
N_RUNS = 5
WARMUP = 2


# =============================================================================
# Timing utilities
# =============================================================================


def time_operation(func: Callable, n_runs: int = N_RUNS) -> tuple[float, float]:
    """Time an operation, return (mean_ms, std_ms)."""
    # Warmup
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
# Data generation with controlled selectivity
# =============================================================================


def create_selectivity_data(
    n_rows: int, selectivity: float, seed: int = 42
) -> pd.DataFrame:
    """
    Create DataFrame where filtering on 'filter_col > 0' has given selectivity.

    Parameters
    ----------
    n_rows : int
        Number of rows
    selectivity : float
        Fraction of rows that pass the filter (0.0 to 1.0)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        DataFrame with filter_col designed for target selectivity
    """
    rng = np.random.default_rng(seed)

    # Create filter column: selectivity fraction are > 0
    n_pass = int(n_rows * selectivity)
    filter_col = np.zeros(n_rows)
    filter_col[:n_pass] = rng.random(n_pass) + 0.001  # Positive values
    filter_col[n_pass:] = -rng.random(n_rows - n_pass) - 0.001  # Negative values
    rng.shuffle(filter_col)

    return pd.DataFrame(
        {
            "filter_col": filter_col,
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
            "c": rng.random(n_rows),
            "d": rng.integers(0, 1000, n_rows),
            "category": rng.choice(["A", "B", "C", "D"], n_rows),
        }
    )


# =============================================================================
# Benchmark results
# =============================================================================


@dataclass
class SelectivityResult:
    """Result for a single selectivity test."""

    operation: str
    n_rows: int
    selectivity: float
    rows_passing: int
    eager_ms: float
    lazy_ms: float

    @property
    def speedup(self) -> float:
        """Lazy speedup over eager (>1 means lazy is faster)."""
        return self.eager_ms / self.lazy_ms if self.lazy_ms > 0 else float("inf")

    @property
    def rows_per_ms(self) -> float:
        """Throughput in rows per millisecond."""
        return self.n_rows / self.lazy_ms if self.lazy_ms > 0 else 0


# =============================================================================
# Selectivity benchmarks
# =============================================================================


def benchmark_simple_filter(
    df: pd.DataFrame, selectivity: float
) -> tuple[float, float]:
    """Benchmark simple filter at given selectivity."""

    # Eager
    def eager():
        return df[df["filter_col"] > 0]

    eager_ms, _ = time_operation(eager)

    # Lazy
    def lazy():
        return df.select().filter(col("filter_col") > 0).collect()

    lazy_ms, _ = time_operation(lazy)

    return eager_ms, lazy_ms


def benchmark_filter_then_compute(
    df: pd.DataFrame, selectivity: float
) -> tuple[float, float]:
    """Benchmark filter followed by computation."""

    # Eager
    def eager():
        filtered = df[df["filter_col"] > 0]
        return filtered.assign(
            computed=filtered["a"] + filtered["b"] * 2,
            ratio=filtered["a"] / (filtered["b"] + 1),
        )

    eager_ms, _ = time_operation(eager)

    # Lazy
    def lazy():
        return (
            df.select()
            .filter(col("filter_col") > 0)
            .select(
                col("filter_col"),
                col("a"),
                col("b"),
                col("c"),
                col("d"),
                col("category"),
                (col("a") + col("b") * 2).alias("computed"),
                (col("a") / (col("b") + 1)).alias("ratio"),
            )
            .collect()
        )

    lazy_ms, _ = time_operation(lazy)

    return eager_ms, lazy_ms


def benchmark_chained_filters(
    df: pd.DataFrame, selectivity: float
) -> tuple[float, float]:
    """
    Benchmark chained filters.

    Each filter has sqrt(selectivity) so combined is ~selectivity.
    """
    # Create thresholds for chained filtering
    # If we want final selectivity S, and we have 2 independent filters,
    # each should have selectivity sqrt(S)
    single_sel = np.sqrt(selectivity)

    # For filter_col > 0, we already have 'selectivity' rows passing
    # Add a second filter on 'a' column
    threshold = 1 - single_sel  # a > threshold gives single_sel fraction

    # Eager
    def eager():
        return df[(df["filter_col"] > 0) & (df["a"] > threshold)]

    eager_ms, _ = time_operation(eager)

    # Lazy (filters should be fused)
    def lazy():
        return (
            df.select()
            .filter(col("filter_col") > 0)
            .filter(col("a") > threshold)
            .collect()
        )

    lazy_ms, _ = time_operation(lazy)

    return eager_ms, lazy_ms


def benchmark_filter_then_agg(
    df: pd.DataFrame, selectivity: float
) -> tuple[float, float]:
    """Benchmark filter followed by aggregation."""

    # Eager
    def eager():
        filtered = df[df["filter_col"] > 0]
        return filtered.groupby("category").agg({"a": "mean", "b": "sum", "d": "max"})

    eager_ms, _ = time_operation(eager)

    # Lazy
    def lazy():
        return (
            df.select()
            .filter(col("filter_col") > 0)
            .group_by(col("category"))
            .agg(
                col("a").mean().alias("a_mean"),
                col("b").sum().alias("b_sum"),
                col("d").max().alias("d_max"),
            )
            .collect()
        )

    lazy_ms, _ = time_operation(lazy)

    return eager_ms, lazy_ms


def benchmark_filter_then_head(
    df: pd.DataFrame, selectivity: float, head_n: int = 100
) -> tuple[float, float]:
    """
    Benchmark filter followed by head().

    This is where low selectivity + streaming really shines.
    """

    # Eager
    def eager():
        return df[df["filter_col"] > 0].head(head_n)

    eager_ms, _ = time_operation(eager)

    # Lazy
    def lazy():
        return df.select().filter(col("filter_col") > 0).head(head_n).collect()

    lazy_ms, _ = time_operation(lazy)

    return eager_ms, lazy_ms


def benchmark_filter_streaming(
    df: pd.DataFrame, selectivity: float
) -> tuple[float, float]:
    """Benchmark streaming filter execution."""

    # Non-streaming lazy
    def non_streaming():
        return df.select().filter(col("filter_col") > 0).collect()

    non_streaming_ms, _ = time_operation(non_streaming)

    # Streaming lazy
    def streaming():
        results = list(
            df.select()
            .filter(col("filter_col") > 0)
            .collect(streaming=True, batch_size=65536, use_physical_planner=True)
        )
        return pd.concat(results) if results else pd.DataFrame()

    streaming_ms, _ = time_operation(streaming)

    return non_streaming_ms, streaming_ms


# =============================================================================
# Main benchmark
# =============================================================================


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_selectivity_table(results: list[SelectivityResult], title: str):
    """Print results as a table grouped by selectivity."""
    print(f"\n{title}")
    print("-" * 80)
    print(
        f"{'Selectivity':>12} {'Rows Pass':>12} {'Eager (ms)':>12} "
        f"{'Lazy (ms)':>12} {'Speedup':>10} {'Rows/ms':>12}"
    )
    print("-" * 80)

    for r in results:
        speedup_str = f"{r.speedup:.2f}x"
        if r.speedup > 1:
            speedup_str = f"+{speedup_str}"
        print(
            f"{r.selectivity:>11.0%} {r.rows_passing:>12,} {r.eager_ms:>12.2f} "
            f"{r.lazy_ms:>12.2f} {speedup_str:>10} {r.rows_per_ms:>12,.0f}"
        )


def run_selectivity_benchmarks():
    """Run all selectivity benchmarks."""
    print_header("SELECTIVITY-SENSITIVE BENCHMARK")
    print(
        """
This benchmark measures how filter selectivity affects performance.

Selectivity = fraction of rows that pass the filter
  - 1%: Highly selective (few rows pass)
  - 90%: Low selectivity (most rows pass)

Key insights:
  - LOW selectivity: Early termination, streaming, fusion shine
  - HIGH selectivity: Memory bandwidth, branch prediction dominate
"""
    )

    for n_rows in ROW_COUNTS:
        print_header(f"DATA SIZE: {n_rows:,} rows")

        # Simple filter benchmark
        results = []
        for sel in SELECTIVITIES:
            df = create_selectivity_data(n_rows, sel)
            rows_pass = int(n_rows * sel)
            eager_ms, lazy_ms = benchmark_simple_filter(df, sel)
            results.append(
                SelectivityResult(
                    operation="simple_filter",
                    n_rows=n_rows,
                    selectivity=sel,
                    rows_passing=rows_pass,
                    eager_ms=eager_ms,
                    lazy_ms=lazy_ms,
                )
            )
        print_selectivity_table(results, "Simple Filter: df[df['col'] > 0]")

        # Filter then compute
        results = []
        for sel in SELECTIVITIES:
            df = create_selectivity_data(n_rows, sel)
            rows_pass = int(n_rows * sel)
            eager_ms, lazy_ms = benchmark_filter_then_compute(df, sel)
            results.append(
                SelectivityResult(
                    operation="filter_compute",
                    n_rows=n_rows,
                    selectivity=sel,
                    rows_passing=rows_pass,
                    eager_ms=eager_ms,
                    lazy_ms=lazy_ms,
                )
            )
        print_selectivity_table(results, "Filter → Compute: filter then add columns")

        # Chained filters
        results = []
        for sel in SELECTIVITIES:
            df = create_selectivity_data(n_rows, sel)
            rows_pass = int(n_rows * sel)  # Approximate
            eager_ms, lazy_ms = benchmark_chained_filters(df, sel)
            results.append(
                SelectivityResult(
                    operation="chained_filters",
                    n_rows=n_rows,
                    selectivity=sel,
                    rows_passing=rows_pass,
                    eager_ms=eager_ms,
                    lazy_ms=lazy_ms,
                )
            )
        print_selectivity_table(results, "Chained Filters: filter1 AND filter2")

        # Filter then aggregate
        results = []
        for sel in SELECTIVITIES:
            df = create_selectivity_data(n_rows, sel)
            rows_pass = int(n_rows * sel)
            eager_ms, lazy_ms = benchmark_filter_then_agg(df, sel)
            results.append(
                SelectivityResult(
                    operation="filter_agg",
                    n_rows=n_rows,
                    selectivity=sel,
                    rows_passing=rows_pass,
                    eager_ms=eager_ms,
                    lazy_ms=lazy_ms,
                )
            )
        print_selectivity_table(results, "Filter → Aggregate: filter then groupby")

        # Filter then head (early termination opportunity)
        results = []
        for sel in SELECTIVITIES:
            df = create_selectivity_data(n_rows, sel)
            rows_pass = int(n_rows * sel)
            eager_ms, lazy_ms = benchmark_filter_then_head(df, sel, head_n=100)
            results.append(
                SelectivityResult(
                    operation="filter_head",
                    n_rows=n_rows,
                    selectivity=sel,
                    rows_passing=rows_pass,
                    eager_ms=eager_ms,
                    lazy_ms=lazy_ms,
                )
            )
        print_selectivity_table(results, "Filter → Head(100): early termination")

    # Selectivity crossover analysis
    print_header("SELECTIVITY CROSSOVER ANALYSIS")
    print(
        """
Finding the selectivity at which lazy becomes faster than eager.
This helps tune heuristics for "is predicate pushdown worth it?"
"""
    )

    # Test many selectivities to find crossover
    test_selectivities = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99]

    print(f"\n{'Selectivity':>12} {'Eager (ms)':>12} {'Lazy (ms)':>12} {'Winner':>10}")
    print("-" * 50)

    crossover_point = None
    prev_winner = None

    for sel in test_selectivities:
        df = create_selectivity_data(1_000_000, sel)
        eager_ms, lazy_ms = benchmark_simple_filter(df, sel)
        winner = "LAZY" if lazy_ms < eager_ms else "EAGER"

        if prev_winner and winner != prev_winner:
            crossover_point = sel

        print(f"{sel:>11.1%} {eager_ms:>12.2f} {lazy_ms:>12.2f} {winner:>10}")
        prev_winner = winner

    if crossover_point:
        print(f"\nCrossover point: ~{crossover_point:.0%} selectivity")

    # Streaming vs non-streaming by selectivity
    print_header("STREAMING BENEFIT BY SELECTIVITY")
    print(
        """
Comparing streaming vs non-streaming execution at different selectivities.
Streaming should help more at LOW selectivity (early termination opportunity).
"""
    )

    print(
        f"\n{'Selectivity':>12} {'Non-Stream':>12} {'Streaming':>12} "
        f"{'Speedup':>10} {'Benefit':>15}"
    )
    print("-" * 65)

    for sel in SELECTIVITIES:
        df = create_selectivity_data(1_000_000, sel)
        non_stream_ms, stream_ms = benchmark_filter_streaming(df, sel)
        speedup = non_stream_ms / stream_ms if stream_ms > 0 else 0

        if speedup > 1.1:
            benefit = "STREAMING WINS"
        elif speedup < 0.9:
            benefit = "NON-STREAM WINS"
        else:
            benefit = "~EQUAL"

        print(
            f"{sel:>11.0%} {non_stream_ms:>12.2f} {stream_ms:>12.2f} "
            f"{speedup:>9.2f}x {benefit:>15}"
        )

    # Summary and recommendations
    print_header("RECOMMENDATIONS")
    print(
        """
Based on selectivity analysis:

1. LOW SELECTIVITY (< 10%)
   - Lazy execution shines (filter fusion, early termination)
   - Streaming provides additional benefit
   - Predicate pushdown is ALWAYS worth it

2. MEDIUM SELECTIVITY (10-50%)
   - Lazy and eager are comparable
   - Benefit depends on subsequent operations
   - Streaming may or may not help

3. HIGH SELECTIVITY (> 50%)
   - Eager may be faster for simple operations
   - Lazy still wins for complex pipelines
   - Memory bandwidth becomes the bottleneck

4. HEURISTICS FOR OPTIMIZATION
   - If estimated selectivity < 10%: Aggressive pushdown
   - If estimated selectivity > 50%: Consider skipping pushdown
   - If followed by head(N): Always push filters

5. CARDINALITY ESTIMATION
   - Accurate selectivity estimation is critical
   - Consider histogram-based estimation
   - Sample-based estimation for unknown predicates
"""
    )


if __name__ == "__main__":
    run_selectivity_benchmarks()
