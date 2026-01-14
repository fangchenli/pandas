#!/usr/bin/env python
"""
Benchmark: Memory usage analysis for eager vs lazy execution.

Measures:
- Peak memory usage during query execution
- Memory allocation patterns
- Memory efficiency of lazy vs eager for complex pipelines
"""

from collections.abc import Callable
import gc
import sys
import time
import tracemalloc

import numpy as np

import pandas as pd


def measure_memory(
    func: Callable, n_runs: int = 3
) -> tuple[float, float, float, float]:
    """
    Measure memory usage of a function.

    Returns
    -------
    tuple[float, float, float, float]
        (peak_mb, current_mb, alloc_mb, time_ms)
        - peak_mb: Peak memory usage during execution
        - current_mb: Memory used after execution
        - alloc_mb: Total bytes allocated
        - time_ms: Execution time in milliseconds
    """
    # Force garbage collection for clean baseline
    gc.collect()

    peaks = []
    currents = []
    allocs = []
    times = []

    for _ in range(n_runs):
        gc.collect()

        tracemalloc.start()
        start = time.perf_counter()

        result = func()

        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Keep result alive for accurate current memory
        _ = result

        peaks.append(peak / (1024 * 1024))  # Convert to MB
        currents.append(current / (1024 * 1024))
        allocs.append(peak / (1024 * 1024))  # Peak approximates total alloc
        times.append((end - start) * 1000)

    return (
        np.mean(peaks),
        np.mean(currents),
        np.mean(allocs),
        np.mean(times),
    )


def get_object_size(obj) -> float:
    """Get approximate size of object in MB."""
    return sys.getsizeof(obj) / (1024 * 1024)


def create_test_data(n_rows: int, use_arrow: bool = False) -> pd.DataFrame:
    """Create test DataFrame."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
            "c": rng.random(n_rows),
            "d": rng.integers(0, 100, n_rows),
            "e": rng.integers(0, 100, n_rows),
            "category": rng.choice(["A", "B", "C", "D"], n_rows),
            "flag": rng.choice([True, False], n_rows),
        }
    )

    if use_arrow:
        df = df.astype(
            {
                "a": "double[pyarrow]",
                "b": "double[pyarrow]",
                "c": "double[pyarrow]",
                "d": "int64[pyarrow]",
                "e": "int64[pyarrow]",
                "category": "string[pyarrow]",
                "flag": "bool[pyarrow]",
            }
        )

    return df


# =============================================================================
# Scenario 1: Simple filter (baseline)
# =============================================================================


def eager_simple_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Simple filter."""
    return df[df["a"] > 0.5]


def lazy_simple_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Simple filter."""
    from pandas.lazy import col

    return df.select().filter(col("a") > 0.5).collect()


# =============================================================================
# Scenario 2: Chained filters (memory for intermediate results)
# =============================================================================


def eager_chained_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Chained filters create intermediate DataFrames."""
    result = df[df["a"] > 0.5]
    result = result[result["b"] > 0.3]
    result = result[result["c"] > 0.2]
    result = result[result["d"] > 20]
    result = result[result["e"] > 30]
    return result


def lazy_chained_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Chained filters fused into single operation."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 0.5)
    lf = lf.filter(col("b") > 0.3)
    lf = lf.filter(col("c") > 0.2)
    lf = lf.filter(col("d") > 20)
    lf = lf.filter(col("e") > 30)
    return lf.collect()


# =============================================================================
# Scenario 3: Computed columns (memory for new arrays)
# =============================================================================


def eager_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Multiple computed columns."""
    result = df.copy()
    result["x"] = result["a"] + result["b"]
    result["y"] = result["c"] * result["d"]
    result["z"] = result["x"] / result["y"]
    result["w"] = result["z"] ** 2
    result["v"] = result["w"] - result["e"]
    return result


def lazy_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Multiple computed columns."""
    from pandas.lazy import col

    lf = df.select()
    x = col("a") + col("b")
    y = col("c") * col("d")
    z = x / y
    w = z**2
    v = w - col("e")

    lf = lf.with_columns(
        x.alias("x"),
        y.alias("y"),
        z.alias("z"),
        w.alias("w"),
        v.alias("v"),
    )
    return lf.collect()


# =============================================================================
# Scenario 4: Filter then compute (projection pushdown opportunity)
# =============================================================================


def eager_filter_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Filter then compute."""
    result = df[df["a"] > 0.9]  # ~10% selectivity
    result = result.copy()
    result["x"] = result["b"] * result["c"]
    result["y"] = result["d"] + result["e"]
    return result[["x", "y"]]


def lazy_filter_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Filter then compute with optimization."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 0.9)
    lf = lf.with_columns(
        (col("b") * col("c")).alias("x"),
        (col("d") + col("e")).alias("y"),
    )
    lf = lf.select("x", "y")
    return lf.collect()


# =============================================================================
# Scenario 5: Wide DataFrame to narrow (projection pruning)
# =============================================================================


def create_wide_data(n_rows: int, n_cols: int = 100) -> pd.DataFrame:
    """Create wide DataFrame."""
    rng = np.random.default_rng(42)
    data = {f"col_{i}": rng.random(n_rows) for i in range(n_cols)}
    return pd.DataFrame(data)


def eager_wide_to_narrow(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Select few columns from wide DataFrame."""
    return df[["col_0", "col_1", "col_2"]]


def lazy_wide_to_narrow(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Select few columns (projection pruning)."""
    return df.select().select("col_0", "col_1", "col_2").collect()


# =============================================================================
# Scenario 6: Complex pipeline (realistic ETL)
# =============================================================================


def eager_complex_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Complex ETL pipeline."""
    # Filter
    result = df[df["a"] > 0.3]
    # Compute
    result = result.copy()
    result["total"] = result["b"] * result["c"]
    result["score"] = result["d"] / (result["e"] + 1)
    # Filter again
    result = result[result["total"] > 0.2]
    result = result[result["score"] > 0.5]
    # Select subset
    return result[["total", "score", "category"]]


def lazy_complex_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: Complex ETL pipeline."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 0.3)
    lf = lf.with_columns(
        (col("b") * col("c")).alias("total"),
        (col("d") / (col("e") + 1)).alias("score"),
    )
    lf = lf.filter(col("total") > 0.2)
    lf = lf.filter(col("score") > 0.5)
    lf = lf.select("total", "score", "category")
    return lf.collect()


def run_benchmarks():
    """Run memory benchmarks."""
    print("=" * 80)
    print("MEMORY USAGE BENCHMARKS: Eager vs Lazy Execution")
    print("=" * 80)
    print()
    print("Measures peak memory usage during query execution.")
    print("Lower is better - lazy should reduce intermediate allocations.")
    print()

    sizes = [100_000, 500_000, 1_000_000]

    for n_rows in sizes:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {n_rows:,} rows")
        print("=" * 80)

        # Create test data
        df = create_test_data(n_rows, use_arrow=False)
        df_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"DataFrame size: {df_size:.2f} MB")
        print()

        # Scenario 1: Simple filter
        print("--- Scenario 1: Simple filter (a > 0.5) ---")
        peak, curr, alloc, time_ms = measure_memory(lambda: eager_simple_filter(df))
        print(f"Eager: peak={peak:.2f} MB, time={time_ms:.2f} ms")

        peak, curr, alloc, time_ms = measure_memory(lambda: lazy_simple_filter(df))
        print(f"Lazy:  peak={peak:.2f} MB, time={time_ms:.2f} ms")
        print()

        # Scenario 2: Chained filters
        print("--- Scenario 2: 5 chained filters ---")
        peak, curr, alloc, time_ms = measure_memory(lambda: eager_chained_filters(df))
        print(f"Eager: peak={peak:.2f} MB, time={time_ms:.2f} ms")

        peak, curr, alloc, time_ms = measure_memory(lambda: lazy_chained_filters(df))
        print(f"Lazy:  peak={peak:.2f} MB, time={time_ms:.2f} ms")
        print()

        # Scenario 3: Computed columns
        print("--- Scenario 3: 5 computed columns ---")
        peak, curr, alloc, time_ms = measure_memory(lambda: eager_computed_columns(df))
        print(f"Eager: peak={peak:.2f} MB, time={time_ms:.2f} ms")

        peak, curr, alloc, time_ms = measure_memory(lambda: lazy_computed_columns(df))
        print(f"Lazy:  peak={peak:.2f} MB, time={time_ms:.2f} ms")
        print()

        # Scenario 4: Filter then compute
        print("--- Scenario 4: Filter (~10% sel) then compute ---")
        peak, curr, alloc, time_ms = measure_memory(lambda: eager_filter_compute(df))
        print(f"Eager: peak={peak:.2f} MB, time={time_ms:.2f} ms")

        peak, curr, alloc, time_ms = measure_memory(lambda: lazy_filter_compute(df))
        print(f"Lazy:  peak={peak:.2f} MB, time={time_ms:.2f} ms")
        print()

        # Scenario 5: Wide to narrow
        print("--- Scenario 5: Wide (100 cols) to narrow (3 cols) ---")
        df_wide = create_wide_data(n_rows, n_cols=100)
        wide_size = df_wide.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"Wide DataFrame size: {wide_size:.2f} MB")

        peak, curr, alloc, time_ms = measure_memory(
            lambda: eager_wide_to_narrow(df_wide)
        )
        print(f"Eager: peak={peak:.2f} MB, time={time_ms:.2f} ms")

        peak, curr, alloc, time_ms = measure_memory(
            lambda: lazy_wide_to_narrow(df_wide)
        )
        print(f"Lazy:  peak={peak:.2f} MB, time={time_ms:.2f} ms")
        print()

        # Scenario 6: Complex pipeline
        print("--- Scenario 6: Complex ETL pipeline ---")
        peak, curr, alloc, time_ms = measure_memory(lambda: eager_complex_pipeline(df))
        print(f"Eager: peak={peak:.2f} MB, time={time_ms:.2f} ms")

        peak, curr, alloc, time_ms = measure_memory(lambda: lazy_complex_pipeline(df))
        print(f"Lazy:  peak={peak:.2f} MB, time={time_ms:.2f} ms")

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Memory Benefits of Lazy Execution:

1. CHAINED FILTERS: Lazy fuses multiple filters into single pass,
   avoiding intermediate DataFrame allocations.

2. COMPUTED COLUMNS: Similar - lazy can compute all columns in
   single pass without intermediate result storage.

3. FILTER+COMPUTE: Lazy with predicate pushdown filters BEFORE
   expensive computations, reducing working set size.

4. WIDE TO NARROW: Projection pruning means lazy only materializes
   needed columns, not entire wide DataFrame.

5. COMPLEX PIPELINES: Combination of above optimizations compounds
   memory savings for realistic ETL workloads.

Note: Memory profiling has overhead. Relative comparisons matter
more than absolute numbers.
""")


if __name__ == "__main__":
    run_benchmarks()
