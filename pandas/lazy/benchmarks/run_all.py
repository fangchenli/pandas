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
    args = parser.parse_args()

    benchmark_dir = Path(__file__).parent

    # Filter benchmarks if specific one requested
    if args.benchmark:
        matching = [b for b in BENCHMARKS if args.benchmark in b]
        if not matching:
            print(f"No benchmark matching '{args.benchmark}'")
            print(f"Available: {', '.join(BENCHMARKS)}")
            return 1
        benchmarks = matching
    else:
        benchmarks = BENCHMARKS

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
