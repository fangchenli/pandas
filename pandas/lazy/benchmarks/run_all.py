#!/usr/bin/env python
"""
Run all lazy pandas benchmarks.

Usage:
    python run_all.py                     # Run all benchmarks
    python run_all.py filter              # Run specific benchmark
    python run_all.py --all               # Include optional benchmarks
    python run_all.py --quick             # ~10x smaller data (pre-push smoke)
    python run_all.py --update-baseline   # Re-record regression baselines
    python run_all.py --threshold 10      # Stricter regression gate

Baseline flags are forwarded to baseline-aware benchmarks (those that
end with shared.baseline_cli); a regression beyond the threshold makes
this script exit non-zero. Quick mode (env LAZY_BENCH_QUICK=1) scales
data sizes down in benchmarks that use shared.scale_sizes and gates
against separate baselines/<name>.quick.json files.
"""

import argparse
import os
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
    "bench_sort.py",
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
    "bench_1tb_transactions.py",  # Requires S3 access or generated data
]

# Benchmarks that support the baseline/regression flags
# (--baseline-dir / --update-baseline / --threshold via shared.baseline_cli)
BASELINE_AWARE = {
    "bench_aggregations.py",
    "bench_filter.py",
    "bench_join.py",
    "bench_sort.py",
    "bench_streaming.py",
}


def run_benchmark(
    script: Path, python_path: str, extra_args: list[str] | None = None
) -> int:
    """Run a single benchmark script."""
    print("\n")
    print("#" * 70)
    print(f"# Running: {script.name}")
    print("#" * 70)
    print()

    result = subprocess.run(
        [python_path, str(script), *(extra_args or [])],
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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with ~10x smaller data sizes (fast pre-push smoke); "
        "gates against separate .quick baselines",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Re-record regression baselines for baseline-aware benchmarks",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Regression threshold percentage for baseline-aware benchmarks",
    )
    args = parser.parse_args()

    if args.quick:
        os.environ["LAZY_BENCH_QUICK"] = "1"

    baseline_args: list[str] = []
    if args.update_baseline:
        baseline_args.append("--update-baseline")
    if args.threshold is not None:
        baseline_args += ["--threshold", str(args.threshold)]

    benchmark_dir = Path(__file__).parent
    all_benchmarks = BENCHMARKS + (OPTIONAL_BENCHMARKS if args.all else [])

    # Filter benchmarks if specific one requested. An exact name match
    # (e.g. "join" -> bench_join.py) wins over substring matches so that
    # "join" does not also pull in bench_join_edge_cases.py.
    if args.benchmark:
        exact = f"bench_{args.benchmark}.py"
        if exact in all_benchmarks:
            benchmarks = [exact]
        else:
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

        extra = baseline_args if bench in BASELINE_AWARE else None
        returncode = run_benchmark(script, args.python, extra)
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
