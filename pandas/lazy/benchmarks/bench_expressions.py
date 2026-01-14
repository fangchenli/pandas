#!/usr/bin/env python
"""
Benchmark: Complex expression evaluation

Tests deeply nested and complex arithmetic/logical expressions.
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
    """Create DataFrame with numeric columns for expression testing."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
            "c": rng.random(n_rows),
            "d": rng.random(n_rows),
            "e": rng.random(n_rows),
            "x": rng.integers(1, 100, n_rows),
            "y": rng.integers(1, 100, n_rows),
            "z": rng.integers(1, 100, n_rows),
        }
    )

    if use_arrow:
        df = df.astype(
            {
                "a": "double[pyarrow]",
                "b": "double[pyarrow]",
                "c": "double[pyarrow]",
                "d": "double[pyarrow]",
                "e": "double[pyarrow]",
                "x": "int64[pyarrow]",
                "y": "int64[pyarrow]",
                "z": "int64[pyarrow]",
            }
        )

    return df


# =============================================================================
# Deeply nested arithmetic
# =============================================================================


def eager_nested_arithmetic(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Deeply nested arithmetic expression."""
    result = df.copy()
    result["nested"] = (
        ((df["a"] + df["b"]) * (df["c"] - df["d"]))
        / ((df["e"] + 1) * (df["a"] * df["b"] + df["c"] * df["d"]))
    ) * 100
    return result


def lazy_nested_arithmetic(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Deeply nested arithmetic expression."""
    from pandas.lazy import col

    lf = df.select()
    expr = (
        ((col("a") + col("b")) * (col("c") - col("d")))
        / ((col("e") + 1) * (col("a") * col("b") + col("c") * col("d")))
    ) * 100
    lf = lf.with_columns(expr.alias("nested"))
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Polynomial expression
# =============================================================================


def eager_polynomial(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Polynomial expression (ax^3 + bx^2 + cx + d)."""
    result = df.copy()
    x = df["a"]
    result["poly"] = df["b"] * x**3 + df["c"] * x**2 + df["d"] * x + df["e"]
    return result


def lazy_polynomial(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Polynomial expression."""
    from pandas.lazy import col

    lf = df.select()
    x = col("a")
    poly = col("b") * x**3 + col("c") * x**2 + col("d") * x + col("e")
    lf = lf.with_columns(poly.alias("poly"))
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Mixed int/float operations
# =============================================================================


def eager_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Mixed integer and float operations."""
    result = df.copy()
    result["mixed"] = (df["a"] * df["x"] + df["b"] * df["y"]) / (df["z"] + 1) + df[
        "c"
    ] * (df["x"] - df["y"])
    return result


def lazy_mixed_types(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Mixed integer and float operations."""
    from pandas.lazy import col

    lf = df.select()
    expr = (col("a") * col("x") + col("b") * col("y")) / (col("z") + 1) + col("c") * (
        col("x") - col("y")
    )
    lf = lf.with_columns(expr.alias("mixed"))
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Complex boolean expression
# =============================================================================


def eager_complex_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Complex boolean expression."""
    result = df.copy()
    result["bool_result"] = (
        ((df["a"] > 0.5) & (df["b"] < 0.5))
        | ((df["c"] > 0.3) & (df["d"] < 0.7) & (df["e"] > 0.2))
        | ((df["x"] > 50) & (df["y"] < 50))
    )
    return result


def lazy_complex_boolean(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Complex boolean expression."""
    from pandas.lazy import col

    lf = df.select()
    expr = (
        ((col("a") > 0.5) & (col("b") < 0.5))
        | ((col("c") > 0.3) & (col("d") < 0.7) & (col("e") > 0.2))
        | ((col("x") > 50) & (col("y") < 50))
    )
    lf = lf.with_columns(expr.alias("bool_result"))
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Multiple interdependent expressions
# =============================================================================


def eager_interdependent(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Expressions that depend on each other."""
    result = df.copy()
    result["step1"] = df["a"] + df["b"]
    result["step2"] = result["step1"] * df["c"]
    result["step3"] = result["step2"] / (df["d"] + 1)
    result["step4"] = result["step3"] + result["step1"]
    result["final"] = result["step4"] * df["e"]
    return result


def lazy_interdependent(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Expressions that depend on each other."""
    from pandas.lazy import col

    lf = df.select()
    step1 = col("a") + col("b")
    step2 = step1 * col("c")
    step3 = step2 / (col("d") + 1)
    step4 = step3 + step1
    final = step4 * col("e")

    lf = lf.with_columns(
        step1.alias("step1"),
        step2.alias("step2"),
        step3.alias("step3"),
        step4.alias("step4"),
        final.alias("final"),
    )
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Expression with comparison chain
# =============================================================================


def eager_comparison_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Chained comparisons for binning."""
    result = df.copy()
    # Simulate case_when binning
    result["bin"] = np.where(
        df["a"] < 0.2,
        1,
        np.where(
            df["a"] < 0.4, 2, np.where(df["a"] < 0.6, 3, np.where(df["a"] < 0.8, 4, 5))
        ),
    )
    return result


def lazy_comparison_chain(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Chained comparisons using when/then/otherwise."""
    from pandas.lazy import (
        col,
        lit,
        when,
    )

    lf = df.select()
    bin_expr = (
        when(col("a") < 0.2)
        .then(lit(1))
        .when(col("a") < 0.4)
        .then(lit(2))
        .when(col("a") < 0.6)
        .then(lit(3))
        .when(col("a") < 0.8)
        .then(lit(4))
        .otherwise(lit(5))
    )
    lf = lf.with_columns(bin_expr.alias("bin"))
    return lf.collect(use_physical_planner=use_physical_planner)


# =============================================================================
# Wide expression (many operations)
# =============================================================================


def eager_wide_expression(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: Single expression with many operations."""
    result = df.copy()
    # Use squared instead of abs to match lazy version
    result["wide"] = (
        df["a"]
        + df["b"]
        - df["c"] * df["d"] / (df["e"] + 0.1)
        + df["a"] * df["b"]
        + df["c"] * df["d"]
        + df["e"] * df["a"]
        + (df["x"] / (df["y"] + 1)) * df["z"]
        + (df["a"] - df["b"]) ** 2
        + (df["c"] - df["d"]) ** 2
    )
    return result


def lazy_wide_expression(
    df: pd.DataFrame, use_physical_planner: bool = False
) -> pd.DataFrame:
    """Lazy: Single expression with many operations."""
    from pandas.lazy import col

    lf = df.select()
    a, b, c, d, e = col("a"), col("b"), col("c"), col("d"), col("e")
    x, y, z = col("x"), col("y"), col("z")

    # Note: abs() not available on Expr, use simpler expression
    expr = (
        a
        + b
        - c * d / (e + 0.1)
        + a * b
        + c * d
        + e * a
        + (x / (y + 1)) * z
        + (a - b) * (a - b)
        + (c - d) * (c - d)  # squared instead of abs
    )
    lf = lf.with_columns(expr.alias("wide"))
    return lf.collect(use_physical_planner=use_physical_planner)


def run_benchmarks():
    """Run all expression benchmarks."""
    sizes = [10_000, 100_000, 1_000_000]

    print("=" * 70)
    print("COMPLEX EXPRESSION BENCHMARKS")
    print("=" * 70)
    print()

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows")
        print("=" * 70)

        df_numpy = create_numeric_data(n_rows, use_arrow=False)
        df_arrow = create_numeric_data(n_rows, use_arrow=True)

        # Deeply nested arithmetic
        print("\n--- Deeply nested arithmetic ---")

        mean, std = timeit(lambda: eager_nested_arithmetic(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_nested_arithmetic(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_nested_arithmetic(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_nested_arithmetic(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_nested_arithmetic(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_nested_arithmetic(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Polynomial
        print("\n--- Polynomial (ax³ + bx² + cx + d) ---")

        mean, std = timeit(lambda: eager_polynomial(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_polynomial(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_polynomial(df_numpy, use_physical_planner=True))
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_polynomial(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_polynomial(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_polynomial(df_arrow, use_physical_planner=True))
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Mixed types
        print("\n--- Mixed int/float operations ---")

        mean, std = timeit(lambda: eager_mixed_types(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_mixed_types(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_mixed_types(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_mixed_types(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_mixed_types(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_mixed_types(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Complex boolean
        print("\n--- Complex boolean expression ---")

        mean, std = timeit(lambda: eager_complex_boolean(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_complex_boolean(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_complex_boolean(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_complex_boolean(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_complex_boolean(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_complex_boolean(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Interdependent expressions
        print("\n--- 5 interdependent expressions ---")

        mean, std = timeit(lambda: eager_interdependent(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_interdependent(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_interdependent(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_interdependent(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_interdependent(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_interdependent(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Case when / comparison chain
        print("\n--- Case-when binning (5 bins) ---")

        mean, std = timeit(lambda: eager_comparison_chain(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_comparison_chain(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_comparison_chain(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_comparison_chain(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_comparison_chain(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_comparison_chain(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")

        # Wide expression
        print("\n--- Wide expression (12+ operations) ---")

        mean, std = timeit(lambda: eager_wide_expression(df_numpy))
        print(f"Eager (NumPy):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_wide_expression(df_numpy))
        print(f"Lazy (NumPy):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_wide_expression(df_numpy, use_physical_planner=True)
        )
        print(f"Lazy+Phys (NumPy):    {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: eager_wide_expression(df_arrow))
        print(f"Eager (Arrow):        {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: lazy_wide_expression(df_arrow))
        print(f"Lazy (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(
            lambda: lazy_wide_expression(df_arrow, use_physical_planner=True)
        )
        print(f"Lazy+Phys (Arrow):    {mean:8.2f} ms ± {std:.2f} ms")


if __name__ == "__main__":
    run_benchmarks()
