#!/usr/bin/env python
"""
Benchmark: Filter operations

Compares lazy vs eager execution for filter/where operations across
NumPy- and Arrow-backed data. Baseline-aware: ends with baseline_cli
(--update-baseline / --threshold; exits non-zero on regression).
"""

import sys

from shared import (
    baseline_cli,
    create_test_data,
    scale_sizes,
    timeit,
)

import pandas as pd

# Sizes below this produce sub-millisecond timings whose run-to-run
# noise would make regression gates flaky; they are printed but not
# recorded as baseline metrics.
METRIC_MIN_ROWS = 100_000


def bench_eager_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager filter: execute immediately."""
    return df[df["a"] > 50]


def bench_lazy_filter(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy filter: build plan, then execute."""
    from pandas.lazy import col

    lf = df.select()  # Enter lazy mode
    lf = lf.filter(col("a") > 50)
    return lf.collect(use_physical_planner=use_physical_planner)


def bench_eager_chained_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager chained filters."""
    result = df[df["a"] > 50]
    result = result[result["b"] < 0.5]
    return result


def bench_lazy_chained_filter(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy chained filters (should be optimized to single filter)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 50)
    lf = lf.filter(col("b") < 0.5)
    return lf.collect(use_physical_planner=use_physical_planner)


def run_benchmarks() -> int:
    """Run all filter benchmarks."""
    sizes = scale_sizes([10_000, 100_000, 1_000_000])
    metrics: dict[str, float] = {}

    print("=" * 70)
    print("FILTER BENCHMARKS")
    print("=" * 70)
    print()

    cases = [
        ("simple_eager", bench_eager_filter, {}),
        ("simple_lazy", bench_lazy_filter, {}),
        ("simple_physical", bench_lazy_filter, {"use_physical_planner": True}),
        ("chained_eager", bench_eager_chained_filter, {}),
        ("chained_lazy", bench_lazy_chained_filter, {}),
        ("chained_physical", bench_lazy_chained_filter, {"use_physical_planner": True}),
    ]

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows")
        print("=" * 70)

        for backend_label, use_arrow in (("numpy", False), ("arrow", True)):
            df = create_test_data(n_rows, use_arrow=use_arrow)
            print(f"\n--- {backend_label} backend ---")
            for case_label, fn, kwargs in cases:
                mean, std = timeit(lambda fn=fn, kwargs=kwargs: fn(df, **kwargs))
                print(f"{case_label:24s} {mean:8.2f} ms ± {std:.2f} ms")
                if n_rows >= METRIC_MIN_ROWS:
                    size_label = f"{n_rows // 1000}k"
                    metrics[f"filter_{case_label}_{backend_label}_{size_label}"] = mean

    return baseline_cli(metrics, "filter")


if __name__ == "__main__":
    sys.exit(run_benchmarks())
