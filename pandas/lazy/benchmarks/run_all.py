#!/usr/bin/env python
"""
Run all lazy pandas benchmarks.

Usage:
    python run_all.py           # Run all benchmarks
    python run_all.py --quick   # Run with smaller data sizes
    python run_all.py filter    # Run specific benchmark
"""

import argparse
from pathlib import Path
import subprocess
import sys

BENCHMARKS = [
    # Planning overhead analysis (diagnostic)
    "bench_planning_phases.py",
    "bench_optimizer_quality.py",
    "bench_cache_effects.py",
    "bench_selectivity.py",
    "bench_join_edge_cases.py",
    "bench_user_stories.py",
    # Core operations
    "bench_kernel_overhead.py",
    "bench_conversion.py",
    "bench_filter.py",
    "bench_select.py",
    "bench_arithmetic.py",
    "bench_string_ops.py",
    "bench_expressions.py",
    "bench_aggregations.py",
    "bench_pipelines.py",
    "bench_advanced_ops.py",
    "bench_kernels.py",
    # Advanced benchmarks
    "bench_join.py",
    "bench_streaming.py",
]

# Benchmarks that require external data or dependencies
OPTIONAL_BENCHMARKS = [
    "bench_vs_polars.py",  # Requires polars
    "bench_nyc_taxi.py",  # Requires NYC taxi data
]


def run_benchmark(script: Path, python_path: str) -> int:
    """Run a single benchmark script."""
    print("\n")
    print("#" * 70)
    print(f"# Running: {script.name}")
    print("#" * 70)
    print()

    result = subprocess.run(
        [python_path, str(script)],
        check=False,
        cwd=script.parent,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run lazy pandas benchmarks")
    parser.add_argument(
        "benchmark",
        nargs="?",
        help="Specific benchmark to run (e.g., 'filter', 'string')",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include optional benchmarks (polars comparison, NYC taxi)",
    )
    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent
    all_benchmarks = BENCHMARKS + (OPTIONAL_BENCHMARKS if args.all else [])

    # Filter benchmarks if specific one requested
    if args.benchmark:
        matching = [b for b in all_benchmarks if args.benchmark in b]
        if not matching:
            print(f"No benchmark matching '{args.benchmark}'")
            print(f"Available: {', '.join(all_benchmarks)}")
            return 1
        benchmarks = matching
    else:
        benchmarks = all_benchmarks if args.all else BENCHMARKS

    # Run benchmarks
    failed = []
    for bench in benchmarks:
        script = benchmark_dir / bench
        if not script.exists():
            print(f"Warning: {script} not found")
            continue

        returncode = run_benchmark(script, args.python)
        if returncode != 0:
            failed.append(bench)

    # Summary
    print("\n")
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total benchmarks run: {len(benchmarks)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    else:
        print("All benchmarks completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
