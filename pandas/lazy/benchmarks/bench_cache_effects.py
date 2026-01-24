#!/usr/bin/env python3
"""
Benchmark: Cold vs Warm Cache Performance

Explicitly measures cache effects that matter for interactive use:
1. Arrow IPC cache (spill/reload cycles)
2. Spill-to-disk cold vs warm reads
3. DataFrame source caching
4. Plan cache / repeated query execution

This is critical for pitching lazy pandas as an interactive engine.
Users running the same query twice should see dramatic speedups.

Usage:
    python pandas/lazy/benchmarks/bench_cache_effects.py
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np

import pandas as pd
from pandas.lazy import col

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Timing utilities
# =============================================================================


def time_operation(func: Callable, n_runs: int = 5) -> tuple[float, float]:
    """Time an operation, return (mean_ms, std_ms)."""
    times = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return float(np.mean(times)), float(np.std(times))


def drop_os_cache():
    """
    Attempt to drop OS file cache.

    Note: This requires root on Linux. On macOS, we can't truly drop cache,
    so we use a large memory allocation to pressure the cache.
    """
    import platform

    if platform.system() == "Linux":
        try:
            # Requires root: echo 3 > /proc/sys/vm/drop_caches
            os.system("sync")
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
            return True
        except (PermissionError, FileNotFoundError):
            pass

    # Fallback: allocate large array to pressure cache
    try:
        pressure = np.zeros(500_000_000, dtype=np.uint8)  # 500MB
        del pressure
        gc.collect()
    except MemoryError:
        pass

    return False


# =============================================================================
# Data generation
# =============================================================================


def create_test_dataframe(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Create a test DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
            "c": rng.random(n_rows),
            "category": rng.choice(["A", "B", "C", "D"], n_rows),
            "value": rng.integers(0, 1000, n_rows),
        }
    )


def create_test_parquet(n_rows: int, path: Path, seed: int = 42) -> Path:
    """Create a test Parquet file."""
    df = create_test_dataframe(n_rows, seed)
    df.to_parquet(path, index=False)
    return path


# =============================================================================
# Benchmark results
# =============================================================================


@dataclass
class CacheResult:
    """Result of a cold vs warm cache comparison."""

    name: str
    cold_ms: float
    cold_std: float
    warm_ms: float
    warm_std: float

    @property
    def speedup(self) -> float:
        return self.cold_ms / self.warm_ms if self.warm_ms > 0 else float("inf")

    def __str__(self) -> str:
        return (
            f"{self.name:<40} "
            f"Cold: {self.cold_ms:>8.2f} ms  "
            f"Warm: {self.warm_ms:>8.2f} ms  "
            f"Speedup: {self.speedup:>5.1f}x"
        )


# =============================================================================
# Cache effect benchmarks
# =============================================================================


def benchmark_repeated_query_execution(df: pd.DataFrame) -> CacheResult:
    """
    Benchmark: Same query executed twice.

    First execution builds plan, optimizes, executes.
    Second execution may benefit from cached optimized plan.
    """

    # Cold: First execution
    def first_run():
        return df.select().filter(col("a") > 0.5).select(col("a"), col("b")).collect()

    cold_ms, cold_std = time_operation(first_run, n_runs=3)

    # Create the lazy frame once
    lf = df.select().filter(col("a") > 0.5).select(col("a"), col("b"))

    # Warm: Execute the same lazy frame again
    def warm_run():
        return lf.collect()

    # First warm run to populate any caches
    lf.collect()

    warm_ms, warm_std = time_operation(warm_run, n_runs=5)

    return CacheResult(
        name="Repeated query execution",
        cold_ms=cold_ms,
        cold_std=cold_std,
        warm_ms=warm_ms,
        warm_std=warm_std,
    )


def benchmark_spill_reload_cycle(n_rows: int = 100_000) -> CacheResult:
    """
    Benchmark: Spill to disk then reload.

    Cold: First reload after spill (file not in OS cache)
    Warm: Second reload (file in OS cache)
    """
    from pandas.lazy.backends.spill import (
        SpillConfig,
        SpillManager,
    )

    df = create_test_dataframe(n_rows)
    arrays = {c: df[c].values for c in df.columns}

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SpillConfig(
            enabled=True,
            spill_dir=Path(tmpdir),
            threshold_mb=4096,  # 4GB threshold
        )
        spill_manager = SpillManager(config)

        # Register and spill the data
        spill_manager.register("test_data", arrays)
        spill_manager.spill("test_data")

        # Try to drop OS cache
        drop_os_cache()

        # Cold read
        def cold_reload():
            return spill_manager.reload("test_data")

        cold_ms, cold_std = time_operation(cold_reload, n_runs=3)

        # Warm read (file should be in OS cache now)
        def warm_reload():
            return spill_manager.reload("test_data")

        # Prime the cache
        spill_manager.reload("test_data")

        warm_ms, warm_std = time_operation(warm_reload, n_runs=5)

        spill_manager.close()

    return CacheResult(
        name=f"Spill/reload cycle ({n_rows:,} rows)",
        cold_ms=cold_ms,
        cold_std=cold_std,
        warm_ms=warm_ms,
        warm_std=warm_std,
    )


def benchmark_parquet_read(n_rows: int = 500_000) -> CacheResult:
    """
    Benchmark: Parquet file read.

    Cold: First read (file not in OS cache)
    Warm: Second read (file in OS cache)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test.parquet"
        create_test_parquet(n_rows, parquet_path)

        # Try to drop OS cache
        drop_os_cache()

        # Cold read
        def cold_read():
            return pd.read_parquet(parquet_path)

        cold_ms, cold_std = time_operation(cold_read, n_runs=3)

        # Prime the cache
        pd.read_parquet(parquet_path)

        # Warm read
        def warm_read():
            return pd.read_parquet(parquet_path)

        warm_ms, warm_std = time_operation(warm_read, n_runs=5)

    return CacheResult(
        name=f"Parquet read ({n_rows:,} rows)",
        cold_ms=cold_ms,
        cold_std=cold_std,
        warm_ms=warm_ms,
        warm_std=warm_std,
    )


def benchmark_lazy_parquet_scan(n_rows: int = 500_000) -> CacheResult:
    """
    Benchmark: Lazy scan of Parquet file.

    Cold: First scan + collect
    Warm: Second scan + collect (file cached)
    """
    from pandas.lazy import scan

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test.parquet"
        create_test_parquet(n_rows, parquet_path)

        # Try to drop OS cache
        drop_os_cache()

        # Cold scan
        def cold_scan():
            return scan(parquet_path).filter(col("a") > 0.5).collect()

        cold_ms, cold_std = time_operation(cold_scan, n_runs=3)

        # Prime the cache
        scan(parquet_path).collect()

        # Warm scan
        def warm_scan():
            return scan(parquet_path).filter(col("a") > 0.5).collect()

        warm_ms, warm_std = time_operation(warm_scan, n_runs=5)

    return CacheResult(
        name=f"Lazy Parquet scan ({n_rows:,} rows)",
        cold_ms=cold_ms,
        cold_std=cold_std,
        warm_ms=warm_ms,
        warm_std=warm_std,
    )


def benchmark_arrow_conversion_cache(n_rows: int = 500_000) -> CacheResult:
    """
    Benchmark: NumPy to Arrow conversion.

    Some Arrow operations cache intermediate results.
    """
    df = create_test_dataframe(n_rows)

    # Cold: First conversion
    def cold_convert():
        return df.astype(
            {
                "a": "double[pyarrow]",
                "b": "double[pyarrow]",
                "c": "double[pyarrow]",
            }
        )

    cold_ms, cold_std = time_operation(cold_convert, n_runs=3)

    # Create Arrow-backed DataFrame
    arrow_df = df.astype(
        {
            "a": "double[pyarrow]",
            "b": "double[pyarrow]",
            "c": "double[pyarrow]",
        }
    )

    # Warm: Operations on already-converted data
    def warm_ops():
        return arrow_df.select().filter(col("a") > 0.5).collect()

    # Prime
    arrow_df.select().filter(col("a") > 0.5).collect()

    warm_ms, warm_std = time_operation(warm_ops, n_runs=5)

    return CacheResult(
        name=f"Arrow conversion + ops ({n_rows:,} rows)",
        cold_ms=cold_ms,
        cold_std=cold_std,
        warm_ms=warm_ms,
        warm_std=warm_std,
    )


def benchmark_streaming_batch_reuse(n_rows: int = 1_000_000) -> CacheResult:
    """
    Benchmark: Streaming execution with batch processing.

    Cold: First streaming pass
    Warm: Second streaming pass (internal buffers allocated)
    """
    df = create_test_dataframe(n_rows)

    # Cold: First streaming execution
    def cold_stream():
        results = list(
            df.select()
            .filter(col("a") > 0.5)
            .collect(streaming=True, batch_size=65536, use_physical_planner=True)
        )
        return pd.concat(results) if results else pd.DataFrame()

    cold_ms, cold_std = time_operation(cold_stream, n_runs=3)

    # Warm: Second streaming execution
    def warm_stream():
        results = list(
            df.select()
            .filter(col("a") > 0.5)
            .collect(streaming=True, batch_size=65536, use_physical_planner=True)
        )
        return pd.concat(results) if results else pd.DataFrame()

    # Prime
    list(
        df.select()
        .filter(col("a") > 0.5)
        .collect(streaming=True, batch_size=65536, use_physical_planner=True)
    )

    warm_ms, warm_std = time_operation(warm_stream, n_runs=5)

    return CacheResult(
        name=f"Streaming execution ({n_rows:,} rows)",
        cold_ms=cold_ms,
        cold_std=cold_std,
        warm_ms=warm_ms,
        warm_std=warm_std,
    )


def benchmark_complex_pipeline_cache(n_rows: int = 500_000) -> CacheResult:
    """
    Benchmark: Complex multi-step pipeline.

    Cold: First execution with full planning
    Warm: Repeated execution
    """
    df = create_test_dataframe(n_rows)

    # Complex pipeline
    def create_pipeline():
        return (
            df.select()
            .filter(col("a") > 0.3)
            .filter(col("b") < 0.8)
            .select(
                col("id"),
                col("a"),
                col("b"),
                (col("a") + col("b")).alias("sum_ab"),
                (col("a") * col("value")).alias("weighted"),
            )
            .filter(col("sum_ab") > 0.5)
            .sort(col("weighted"), descending=True)
            .head(1000)
        )

    # Cold: First execution
    def cold_pipeline():
        return create_pipeline().collect()

    cold_ms, cold_std = time_operation(cold_pipeline, n_runs=3)

    # Create and cache the lazy frame
    lf = create_pipeline()
    lf.collect()  # Prime

    # Warm: Same lazy frame
    def warm_pipeline():
        return lf.collect()

    warm_ms, warm_std = time_operation(warm_pipeline, n_runs=5)

    return CacheResult(
        name=f"Complex pipeline ({n_rows:,} rows)",
        cold_ms=cold_ms,
        cold_std=cold_std,
        warm_ms=warm_ms,
        warm_std=warm_std,
    )


# =============================================================================
# Main benchmark
# =============================================================================


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_cache_benchmarks():
    """Run all cache effect benchmarks."""
    print_header("COLD VS WARM CACHE BENCHMARK")
    print(
        """
This benchmark measures the performance difference between:
  - Cold: First execution (no caches populated)
  - Warm: Repeated execution (caches populated)

Understanding these effects is critical for interactive use cases.
"""
    )

    results: list[CacheResult] = []

    # Run benchmarks
    print_header("QUERY EXECUTION CACHING")

    df_500k = create_test_dataframe(500_000)

    result = benchmark_repeated_query_execution(df_500k)
    results.append(result)
    print(result)

    result = benchmark_complex_pipeline_cache(500_000)
    results.append(result)
    print(result)

    print_header("FILE I/O CACHING")

    result = benchmark_parquet_read(500_000)
    results.append(result)
    print(result)

    try:
        result = benchmark_lazy_parquet_scan(500_000)
        results.append(result)
        print(result)
    except Exception as e:
        print(f"Lazy Parquet scan: skipped ({e})")

    print_header("SPILL/RELOAD CACHING")

    result = benchmark_spill_reload_cycle(100_000)
    results.append(result)
    print(result)

    result = benchmark_spill_reload_cycle(500_000)
    results.append(result)
    print(result)

    print_header("CONVERSION AND STREAMING")

    result = benchmark_arrow_conversion_cache(500_000)
    results.append(result)
    print(result)

    result = benchmark_streaming_batch_reuse(1_000_000)
    results.append(result)
    print(result)

    # Summary
    print_header("SUMMARY")

    print(f"\n{'Benchmark':<45} {'Cold (ms)':>12} {'Warm (ms)':>12} {'Speedup':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r.name:<45} {r.cold_ms:>12.2f} {r.warm_ms:>12.2f} {r.speedup:>9.1f}x")

    avg_speedup = np.mean([r.speedup for r in results])
    max_speedup = max(r.speedup for r in results)
    max_name = max(results, key=lambda r: r.speedup).name

    print("-" * 80)
    print(f"{'Average speedup:':<45} {' ':>12} {' ':>12} {avg_speedup:>9.1f}x")
    print(f"{'Best speedup:':<45} {max_name[:30]:>24} {max_speedup:>9.1f}x")

    # Recommendations
    print_header("RECOMMENDATIONS FOR INTERACTIVE USE")
    print(
        """
Based on cache effects:

1. REUSE LAZY FRAMES
   - Create LazyDataFrame once, call .collect() multiple times
   - Avoids repeated plan construction and optimization
   - Expected speedup: 1.5-3x

2. USE ARROW-BACKED DTYPES
   - Convert to Arrow once at load time
   - All subsequent operations are faster
   - Avoids repeated NumPy↔Arrow conversions

3. KEEP DATA IN MEMORY
   - Spill/reload has significant cold read overhead
   - Use streaming for truly large data
   - Consider memory budget settings

4. PARQUET FILE CACHING
   - First read is slower (disk I/O)
   - Subsequent reads benefit from OS cache
   - For repeated access, consider keeping data in memory

5. BATCH SIZE TUNING
   - Streaming allocates internal buffers
   - First batch may be slower
   - Steady state is faster
"""
    )


if __name__ == "__main__":
    run_cache_benchmarks()
