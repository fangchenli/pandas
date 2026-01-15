#!/usr/bin/env python
"""
Calibration benchmark suite for lazy pandas execution thresholds.

This script runs a series of micro-benchmarks to determine optimal threshold
values for the lazy execution engine. The results can be saved to a JSON file
that can be loaded by ThresholdConfig.

Usage:
    python scripts/calibrate_lazy_thresholds.py --output calibration.json

    # Run quick calibration (fewer iterations)
    python scripts/calibrate_lazy_thresholds.py --quick --output calibration.json

    # Run specific benchmarks only
    python scripts/calibrate_lazy_thresholds.py --benchmarks filter,groupby

The output file can be loaded via:
    from pandas.lazy.optimize.config import ThresholdConfig
    config = ThresholdConfig.from_file("calibration.json")
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
import json
import platform
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import pandas as pd

# Create a random generator for reproducible benchmarks
_rng = np.random.default_rng(42)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    crossover_point: int | float | None
    confidence: float
    raw_data: list[dict[str, Any]]
    recommendation: Any


def get_hardware_info() -> dict[str, Any]:
    """Get hardware and software information."""
    import multiprocessing

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "cores": multiprocessing.cpu_count(),
            "platform": platform.system(),
            "platform_version": platform.version(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "pandas": pd.__version__,
        },
    }


def timeit(func, n_runs: int = 5, warmup: int = 2) -> tuple[float, float]:
    """
    Time a function and return mean and std in milliseconds.

    Parameters
    ----------
    func : callable
        Function to time.
    n_runs : int
        Number of timed runs.
    warmup : int
        Number of warmup runs.

    Returns
    -------
    tuple[float, float]
        Mean and std in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def find_crossover(
    test_func,
    sizes: list[int],
    n_runs: int = 5,
) -> tuple[int | None, float, list[dict]]:
    """
    Find the crossover point where method B becomes faster than method A.

    Parameters
    ----------
    test_func : callable
        Function that takes (size) and returns (time_a, time_b).
    sizes : list[int]
        Sizes to test.
    n_runs : int
        Number of runs per size.

    Returns
    -------
    tuple[int | None, float, list[dict]]
        (crossover_point, confidence, raw_data)
    """
    raw_data = []

    for size in sizes:
        time_a, std_a = timeit(lambda s=size: test_func(s, "a"), n_runs)
        time_b, std_b = timeit(lambda s=size: test_func(s, "b"), n_runs)

        raw_data.append(
            {
                "size": size,
                "method_a_ms": round(time_a, 3),
                "method_a_std": round(std_a, 3),
                "method_b_ms": round(time_b, 3),
                "method_b_std": round(std_b, 3),
                "winner": "a" if time_a < time_b else "b",
                "speedup": round(time_a / time_b, 2) if time_b > 0 else float("inf"),
            }
        )

    # Find crossover point (where B becomes faster)
    crossover = None
    for i, d in enumerate(raw_data):
        if d["winner"] == "b":
            crossover = d["size"]
            break

    # Calculate confidence based on consistency
    if crossover is not None:
        # Check how many subsequent points also favor B
        subsequent_b = sum(1 for d in raw_data[i:] if d["winner"] == "b")
        confidence = subsequent_b / len(raw_data[i:]) if len(raw_data[i:]) > 0 else 0.0
    else:
        confidence = 0.0

    return crossover, confidence, raw_data


# =============================================================================
# Individual Benchmark Functions
# =============================================================================


def benchmark_filter_backend(quick: bool = False) -> BenchmarkResult:
    """
    Benchmark NumPy vs Arrow filtering.

    Tests when Arrow filter becomes faster than NumPy boolean indexing.
    """
    print("Running filter backend benchmark...")

    sizes = (
        [5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000]
        if not quick
        else [10000, 50000, 100000, 250000]
    )

    def test_filter(size: int, method: str) -> None:
        # Create test data
        arr = _rng.standard_normal(size)

        if method == "a":
            # NumPy boolean indexing
            mask = arr > 0
            _ = arr[mask]
        else:
            # Arrow filter
            pa_arr = pa.array(arr)
            pa_mask = pc.greater(pa_arr, 0)
            _ = pc.filter(pa_arr, pa_mask)

    crossover, confidence, raw_data = find_crossover(test_filter, sizes)

    # Add buffer to crossover for safety
    recommendation = crossover if crossover else 50_000
    if crossover and confidence > 0.8:
        recommendation = int(crossover * 0.9)  # 10% margin

    return BenchmarkResult(
        name="filter_backend",
        crossover_point=crossover,
        confidence=confidence,
        raw_data=raw_data,
        recommendation=recommendation,
    )


def benchmark_filter_multi_condition(quick: bool = False) -> BenchmarkResult:
    """
    Benchmark filter with multiple conditions.

    Tests how threshold changes with number of conditions.
    """
    print("Running filter multi-condition benchmark...")

    sizes = [10000, 50000, 100000, 250000] if not quick else [25000, 100000]
    n_conditions_list = [1, 2, 3, 5]

    raw_data = []

    for n_cond in n_conditions_list:
        for size in sizes:
            arr = _rng.standard_normal(size)

            def numpy_filter():
                masks = [arr > -1 + 0.5 * i for i in range(n_cond)]
                combined = masks[0]
                for m in masks[1:]:
                    combined = combined & m
                return arr[combined]

            def arrow_filter():
                pa_arr = pa.array(arr)
                masks = [pc.greater(pa_arr, -1 + 0.5 * i) for i in range(n_cond)]
                combined = masks[0]
                for m in masks[1:]:
                    combined = pc.and_(combined, m)
                return pc.filter(pa_arr, combined)

            time_np, _ = timeit(numpy_filter)
            time_arr, _ = timeit(arrow_filter)

            raw_data.append(
                {
                    "size": size,
                    "n_conditions": n_cond,
                    "numpy_ms": round(time_np, 3),
                    "arrow_ms": round(time_arr, 3),
                    "winner": "numpy" if time_np < time_arr else "arrow",
                    "speedup": round(time_np / time_arr, 2),
                }
            )

    # Analyze: how much does each condition lower the threshold?
    # Find crossover for each condition count
    crossovers = {}
    for n_cond in n_conditions_list:
        cond_data = [d for d in raw_data if d["n_conditions"] == n_cond]
        for d in cond_data:
            if d["winner"] == "arrow":
                crossovers[n_cond] = d["size"]
                break

    return BenchmarkResult(
        name="filter_multi_condition",
        crossover_point=None,
        confidence=0.9,
        raw_data=raw_data,
        recommendation={
            "base_threshold": crossovers.get(1, 50000),
            "per_condition_reduction": 10000,  # Reduce by 10K per condition
            "crossovers_by_condition": crossovers,
        },
    )


def benchmark_groupby_backend(quick: bool = False) -> BenchmarkResult:
    """
    Benchmark Pandas Cython vs Arrow groupby.

    Tests when Arrow groupby becomes faster.
    """
    print("Running groupby backend benchmark...")

    # Test matrix: (rows, groups)
    configs = (
        [
            (10000, 10),
            (10000, 100),
            (50000, 100),
            (100000, 100),
            (100000, 1000),
            (500000, 1000),
            (1000000, 1000),
        ]
        if not quick
        else [(10000, 100), (100000, 100), (500000, 1000)]
    )

    raw_data = []

    for n_rows, n_groups in configs:
        # Create test data
        keys = _rng.integers(0, n_groups, size=n_rows)
        values = _rng.standard_normal(n_rows)
        df = pd.DataFrame({"key": keys, "value": values})
        table = pa.table({"key": keys, "value": values})

        def pandas_groupby():
            return df.groupby("key")["value"].sum()

        def arrow_groupby():
            return table.group_by("key").aggregate([("value", "hash_sum")])

        time_pd, _ = timeit(pandas_groupby)
        time_arr, _ = timeit(arrow_groupby)

        raw_data.append(
            {
                "n_rows": n_rows,
                "n_groups": n_groups,
                "pandas_ms": round(time_pd, 3),
                "arrow_ms": round(time_arr, 3),
                "winner": "pandas" if time_pd < time_arr else "arrow",
                "speedup": round(time_pd / time_arr, 2),
            }
        )

    # Find row threshold where Arrow wins (for any group count)
    row_crossover = None
    for d in raw_data:
        if d["winner"] == "arrow":
            row_crossover = d["n_rows"]
            break

    # Find group crossover (minimum groups where Arrow wins)
    group_crossover = None
    for d in sorted(raw_data, key=lambda x: x["n_groups"]):
        if d["winner"] == "arrow":
            group_crossover = d["n_groups"]
            break

    return BenchmarkResult(
        name="groupby_backend",
        crossover_point=row_crossover,
        confidence=0.85,
        raw_data=raw_data,
        recommendation={
            "row_threshold": row_crossover or 100000,
            "cardinality_threshold": group_crossover or 100,
        },
    )


def benchmark_numexpr_fusion(quick: bool = False) -> BenchmarkResult:
    """
    Benchmark NumPy vs NumExpr for expression fusion.

    Tests when NumExpr fusion becomes beneficial.
    """
    print("Running numexpr fusion benchmark...")

    try:
        import numexpr as ne
    except ImportError:
        print("  NumExpr not installed, skipping...")
        return BenchmarkResult(
            name="numexpr_fusion",
            crossover_point=None,
            confidence=0.0,
            raw_data=[],
            recommendation={"min_elements": 100000, "min_operations": 2},
        )

    sizes = (
        [10000, 25000, 50000, 75000, 100000, 250000, 500000, 1000000]
        if not quick
        else [25000, 100000, 500000]
    )

    raw_data = []

    for size in sizes:
        a = _rng.standard_normal(size)
        b = _rng.standard_normal(size)
        c = _rng.standard_normal(size)
        local_vars = {"a": a, "b": b, "c": c}

        # Test with increasing operation counts
        for n_ops in [2, 3, 5]:
            # Capture variables for closures
            _a, _b, _c = a, b, c
            _local = local_vars

            def numpy_eval(_a=_a, _b=_b, _c=_c, _n=n_ops):
                if _n == 2:
                    return _a + _b
                elif _n == 3:
                    return _a + _b * _c
                else:
                    return (_a + _b) * (_c - _a) / (_b + 0.1)

            def numexpr_eval(_local=_local, _n=n_ops):
                if _n == 2:
                    return ne.evaluate("a + b", local_dict=_local)
                elif _n == 3:
                    return ne.evaluate("a + b * c", local_dict=_local)
                else:
                    return ne.evaluate(
                        "(a + b) * (c - a) / (b + 0.1)", local_dict=_local
                    )

            time_np, _ = timeit(numpy_eval)
            time_ne, _ = timeit(numexpr_eval)

            raw_data.append(
                {
                    "size": size,
                    "n_operations": n_ops,
                    "numpy_ms": round(time_np, 3),
                    "numexpr_ms": round(time_ne, 3),
                    "winner": "numpy" if time_np < time_ne else "numexpr",
                    "speedup": round(time_np / time_ne, 2),
                }
            )

    # Find crossover for 2 operations (minimum useful)
    crossover = None
    for d in raw_data:
        if d["n_operations"] == 2 and d["winner"] == "numexpr":
            crossover = d["size"]
            break

    return BenchmarkResult(
        name="numexpr_fusion",
        crossover_point=crossover,
        confidence=0.9,
        raw_data=raw_data,
        recommendation={
            "min_elements": crossover or 100000,
            "min_operations": 2,
        },
    )


def benchmark_parallel_projection(quick: bool = False) -> BenchmarkResult:
    """
    Benchmark sequential vs parallel projection.

    Tests when ThreadPoolExecutor parallelization becomes beneficial.
    """
    print("Running parallel projection benchmark...")

    from concurrent.futures import ThreadPoolExecutor

    n_expressions_list = [2, 4, 6, 8, 10, 12, 16] if not quick else [4, 8, 12]
    row_count = 100000

    raw_data = []

    for n_expr in n_expressions_list:
        # Create test data and expressions
        arrays = {f"col_{i}": _rng.standard_normal(row_count) for i in range(n_expr)}

        def sequential_eval():
            results = {}
            for name, arr in arrays.items():
                results[f"{name}_sq"] = arr**2
            return results

        def parallel_eval():
            def square_col(item):
                name, arr = item
                return f"{name}_sq", arr**2

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = dict(executor.map(square_col, arrays.items()))
            return results

        time_seq, _ = timeit(sequential_eval)
        time_par, _ = timeit(parallel_eval)

        raw_data.append(
            {
                "n_expressions": n_expr,
                "sequential_ms": round(time_seq, 3),
                "parallel_ms": round(time_par, 3),
                "winner": "sequential" if time_seq < time_par else "parallel",
                "speedup": round(time_seq / time_par, 2),
            }
        )

    # Find crossover (where parallel becomes faster)
    crossover = None
    for d in raw_data:
        if d["winner"] == "parallel":
            crossover = d["n_expressions"]
            break

    return BenchmarkResult(
        name="parallel_projection",
        crossover_point=crossover,
        confidence=0.8,
        raw_data=raw_data,
        recommendation=crossover or 8,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def run_calibration(
    benchmarks: list[str] | None = None,
    quick: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Run calibration benchmarks and return results.

    Parameters
    ----------
    benchmarks : list[str] or None
        List of benchmark names to run, or None for all.
    quick : bool
        If True, run quick calibration with fewer iterations.
    output_path : str or None
        Path to save results JSON.

    Returns
    -------
    dict
        Calibration results including recommended configuration.
    """
    all_benchmarks = {
        "filter": benchmark_filter_backend,
        "filter_multi": benchmark_filter_multi_condition,
        "groupby": benchmark_groupby_backend,
        "numexpr": benchmark_numexpr_fusion,
        "parallel": benchmark_parallel_projection,
    }

    if benchmarks is None:
        benchmarks = list(all_benchmarks.keys())

    print(f"Running calibration benchmarks: {benchmarks}")
    print(f"Mode: {'quick' if quick else 'full'}")
    print()

    results = {}
    for name in benchmarks:
        if name in all_benchmarks:
            result = all_benchmarks[name](quick=quick)
            results[result.name] = {
                "crossover_point": result.crossover_point,
                "confidence": result.confidence,
                "raw_data": result.raw_data,
                "recommendation": result.recommendation,
            }
            print()

    # Build recommended configuration
    recommended_config = {
        "filter_arrow_threshold": 50000,
        "groupby_arrow_row_threshold": 100000,
        "groupby_arrow_cardinality_threshold": 100,
        "numexpr_min_elements": 100000,
        "numexpr_min_operations": 2,
        "parallel_expr_threshold": 8,
    }

    # Update from benchmark results
    if "filter_backend" in results:
        rec = results["filter_backend"]["recommendation"]
        if rec:
            recommended_config["filter_arrow_threshold"] = rec

    if "groupby_backend" in results:
        rec = results["groupby_backend"]["recommendation"]
        if isinstance(rec, dict):
            recommended_config["groupby_arrow_row_threshold"] = rec.get(
                "row_threshold", 100000
            )
            recommended_config["groupby_arrow_cardinality_threshold"] = rec.get(
                "cardinality_threshold", 100
            )

    if "numexpr_fusion" in results:
        rec = results["numexpr_fusion"]["recommendation"]
        if isinstance(rec, dict):
            recommended_config["numexpr_min_elements"] = rec.get("min_elements", 100000)
            recommended_config["numexpr_min_operations"] = rec.get("min_operations", 2)

    if "parallel_projection" in results:
        rec = results["parallel_projection"]["recommendation"]
        if rec:
            recommended_config["parallel_expr_threshold"] = rec

    # Build final output
    output = {
        "metadata": get_hardware_info(),
        "benchmarks": results,
        "recommended_config": recommended_config,
    }

    # Save if path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 60)
    for key, value in recommended_config.items():
        print(f"  {key}: {value}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate lazy pandas execution thresholds"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick calibration with fewer iterations",
    )
    parser.add_argument(
        "--benchmarks",
        "-b",
        type=str,
        help="Comma-separated list of benchmarks to run "
        "(filter,filter_multi,groupby,numexpr,parallel)",
    )

    args = parser.parse_args()

    benchmarks = None
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    run_calibration(
        benchmarks=benchmarks,
        quick=args.quick,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
