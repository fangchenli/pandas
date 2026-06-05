#!/usr/bin/env python3
"""
Benchmark: Sort Operations

Covers the sort paths of the physical engine against eager pandas:
single-key (parallel chunked argsort + parallel gather), multi-key
(Arrow table sort vs np.lexsort), descending (stable reversed-view
mapping), and tie-heavy keys (stable-sort contract overhead).

This benchmark is also the template for the baseline/regression
workflow: it collects metrics and finishes with baseline_cli(), so

    python bench_sort.py --update-baseline    # record baseline
    python bench_sort.py                      # compare, exit 1 on regression
    python bench_sort.py --threshold 10       # stricter gate

Usage:
    python pandas/lazy/benchmarks/bench_sort.py
"""

import sys

import numpy as np
from shared import (
    baseline_cli,
    benchmark,
    print_header,
    print_result,
    print_speedup,
)

import pandas as pd

SIZES = [1_000_000, 10_000_000]


def make_sort_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "key_f": rng.standard_normal(n),
            "key_i": rng.integers(0, 100, n),  # heavy ties
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        }
    )


def bench_case(metrics, name, eager_fn, lazy_fn) -> None:
    eager = benchmark(eager_fn, n_runs=3, n_warmup=1)
    lazy = benchmark(lazy_fn, n_runs=3, n_warmup=1)
    metrics[f"{name}_eager"] = eager["min_ms"]
    metrics[f"{name}_physical"] = lazy["min_ms"]
    print_result(f"  eager  {name}", eager["mean_ms"], eager["std_ms"])
    print_result(f"  lazy   {name}", lazy["mean_ms"], lazy["std_ms"])
    print_speedup(eager["min_ms"], lazy["min_ms"], label="speedup vs eager")


def main() -> int:
    metrics: dict[str, float] = {}

    for n in SIZES:
        label = f"{n // 1_000_000}m"
        df = make_sort_data(n)
        print_header(f"Sort benchmarks ({n:,} rows, 5 columns)")

        bench_case(
            metrics,
            f"sort_single_float_{label}",
            lambda df=df: df.sort_values("key_f", kind="stable"),
            lambda df=df: df.select().sort("key_f").collect(use_physical_planner=True),
        )
        bench_case(
            metrics,
            f"sort_single_int_ties_{label}",
            lambda df=df: df.sort_values("key_i", kind="stable"),
            lambda df=df: df.select().sort("key_i").collect(use_physical_planner=True),
        )
        bench_case(
            metrics,
            f"sort_descending_{label}",
            lambda df=df: df.sort_values("key_f", ascending=False, kind="stable"),
            lambda df=df: (
                df.select()
                .sort("key_f", descending=True)
                .collect(use_physical_planner=True)
            ),
        )
        bench_case(
            metrics,
            f"sort_multikey_{label}",
            lambda df=df: df.sort_values(["key_i", "key_f"], kind="stable"),
            lambda df=df: (
                df.select().sort("key_i", "key_f").collect(use_physical_planner=True)
            ),
        )
        bench_case(
            metrics,
            f"sort_head_topk_{label}",
            lambda df=df: df.sort_values("key_f", kind="stable").head(10),
            lambda df=df: (
                df.select().sort("key_f").head(10).collect(use_physical_planner=True)
            ),
        )

    return baseline_cli(metrics, "sort")


if __name__ == "__main__":
    sys.exit(main())
