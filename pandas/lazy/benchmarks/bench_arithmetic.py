#!/usr/bin/env python
"""
Benchmark: Arithmetic operations

Compares lazy vs eager execution for arithmetic on numeric columns.
Tests both simple and chained/complex expressions.
"""

from collections.abc import Callable
import time

import numpy as np

import pandas as pd


def timeit(func: Callable, n_runs: int = 5, warmup: int = 1) -> tuple[float, float]:
    """Run function multiple times and return (mean, std) in milliseconds."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def create_numeric_data(n_rows: int, use_arrow: bool = False) -> pd.DataFrame:
    """Create DataFrame with numeric columns."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
            "c": rng.random(n_rows),
            "d": rng.integers(1, 100, n_rows),  # int column
        }
    )

    if use_arrow:
        df = df.astype(
            {
                "a": "double[pyarrow]",
                "b": "double[pyarrow]",
                "c": "double[pyarrow]",
                "d": "int64[pyarrow]",
            }
        )

    return df


def bench_eager_simple_add(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: simple addition."""
    result = df.copy()
    result["sum_ab"] = df["a"] + df["b"]
    return result


def bench_lazy_simple_add(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: simple addition."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns((col("a") + col("b")).alias("sum_ab"))
    return lf.collect()


def bench_eager_chained_arithmetic(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: chained arithmetic (a + b) * c."""
    result = df.copy()
    result["result"] = (df["a"] + df["b"]) * df["c"]
    return result


def bench_lazy_chained_arithmetic(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: chained arithmetic (a + b) * c."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(((col("a") + col("b")) * col("c")).alias("result"))
    return lf.collect()


def bench_eager_complex_expression(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: complex expression with multiple operations."""
    result = df.copy()
    result["complex"] = (df["a"] * 2 + df["b"] ** 2 - df["c"] / df["d"]) * 100
    return result


def bench_lazy_complex_expression(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: complex expression with multiple operations."""
    from pandas.lazy import col

    lf = df.select()
    expr = (col("a") * 2 + col("b") ** 2 - col("c") / col("d")) * 100
    lf = lf.with_columns(expr.alias("complex"))
    return lf.collect()


def bench_eager_multiple_expressions(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: multiple expressions in sequence."""
    result = df.copy()
    result["expr1"] = df["a"] + df["b"]
    result["expr2"] = df["a"] - df["b"]
    result["expr3"] = df["a"] * df["b"]
    result["expr4"] = df["a"] / (df["b"] + 0.001)  # Avoid div by zero
    return result


def bench_lazy_multiple_expressions(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: multiple expressions (can be computed in parallel)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(
        (col("a") + col("b")).alias("expr1"),
        (col("a") - col("b")).alias("expr2"),
        (col("a") * col("b")).alias("expr3"),
        (col("a") / (col("b") + 0.001)).alias("expr4"),
    )
    return lf.collect()


def bench_eager_filter_then_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: filter then compute."""
    filtered = df[df["a"] > 0.5]
    result = filtered.copy()
    result["computed"] = filtered["b"] * 2
    return result


def bench_lazy_filter_then_compute(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: filter then compute (predicate pushdown)."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 0.5)
    lf = lf.with_columns((col("b") * 2).alias("computed"))
    return lf.collect()


def bench_eager_compute_then_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: compute then filter on computed value."""
    result = df.copy()
    result["computed"] = df["a"] + df["b"]
    return result[result["computed"] > 1.0]


def bench_lazy_compute_then_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: compute then filter on computed value."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns((col("a") + col("b")).alias("computed"))
    lf = lf.filter(col("computed") > 1.0)
    return lf.collect()


def run_benchmarks():
    """Run all arithmetic benchmarks."""
    sizes = [10_000, 100_000, 1_000_000, 5_000_000]

    print("=" * 70)
    print("ARITHMETIC OPERATION BENCHMARKS")
    print("=" * 70)

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows")
        print("=" * 70)

        df_numpy = create_numeric_data(n_rows, use_arrow=False)
        df_arrow = create_numeric_data(n_rows, use_arrow=True)

        # Simple addition
        print("\n--- Simple addition (a + b) ---")

        mean, std = timeit(lambda: bench_eager_simple_add(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_simple_add(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_simple_add(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_simple_add(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Chained arithmetic
        print("\n--- Chained arithmetic ((a + b) * c) ---")

        mean, std = timeit(lambda: bench_eager_chained_arithmetic(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_chained_arithmetic(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_chained_arithmetic(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_chained_arithmetic(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Complex expression
        print("\n--- Complex expression (a*2 + b**2 - c/d) * 100 ---")

        mean, std = timeit(lambda: bench_eager_complex_expression(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_complex_expression(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_complex_expression(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_complex_expression(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Multiple expressions
        print("\n--- Multiple expressions (4 computed columns) ---")

        mean, std = timeit(lambda: bench_eager_multiple_expressions(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_multiple_expressions(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_multiple_expressions(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_multiple_expressions(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")

        # Filter then compute
        print("\n--- Filter (a > 0.5) then compute (b * 2) ---")

        mean, std = timeit(lambda: bench_eager_filter_then_compute(df_numpy))
        print(f"Eager (NumPy):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_filter_then_compute(df_numpy))
        print(f"Lazy (NumPy):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_filter_then_compute(df_arrow))
        print(f"Eager (Arrow):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_filter_then_compute(df_arrow))
        print(f"Lazy (Arrow):   {mean:8.2f} ms ± {std:.2f} ms")


if __name__ == "__main__":
    run_benchmarks()
