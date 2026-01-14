#!/usr/bin/env python
"""
Profile the physical planner to identify performance bottlenecks.

This script uses cProfile and line_profiler to understand where
the physical planner spends its time.
"""

import cProfile
from io import StringIO
import pstats

import numpy as np

import pandas as pd


def create_test_data(n_rows: int = 100_000, use_arrow: bool = True) -> pd.DataFrame:
    """Create test DataFrame."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
            "c": rng.random(n_rows),
            "d": rng.integers(1, 100, n_rows),
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


def run_lazy_query(df: pd.DataFrame, use_physical_planner: bool = False):
    """Run a moderately complex lazy query."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("a") > 0.5)
    lf = lf.with_columns(
        (col("a") + col("b")).alias("sum_ab"),
        (col("a") * col("b")).alias("prod_ab"),
        (col("c") / (col("d") + 1)).alias("ratio"),
    )
    lf = lf.filter(col("sum_ab") > 1.0)
    return lf.collect(use_physical_planner=use_physical_planner)


def profile_comparison():
    """Compare profiles of logical-only vs physical planner."""
    print("=" * 70)
    print("PROFILING PHYSICAL PLANNER")
    print("=" * 70)

    # Create test data
    df_arrow = create_test_data(100_000, use_arrow=True)

    # Warmup
    run_lazy_query(df_arrow, use_physical_planner=False)
    run_lazy_query(df_arrow, use_physical_planner=True)

    # Profile logical-only
    print("\n" + "=" * 70)
    print("LOGICAL OPTIMIZER ONLY (use_physical_planner=False)")
    print("=" * 70)

    profiler_logical = cProfile.Profile()
    profiler_logical.enable()
    for _ in range(10):
        run_lazy_query(df_arrow, use_physical_planner=False)
    profiler_logical.disable()

    s = StringIO()
    ps = pstats.Stats(profiler_logical, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    # Profile physical planner
    print("\n" + "=" * 70)
    print("WITH PHYSICAL PLANNER (use_physical_planner=True)")
    print("=" * 70)

    profiler_physical = cProfile.Profile()
    profiler_physical.enable()
    for _ in range(10):
        run_lazy_query(df_arrow, use_physical_planner=True)
    profiler_physical.disable()

    s = StringIO()
    ps = pstats.Stats(profiler_physical, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


def profile_physical_planner_detailed():
    """Detailed profiling of physical planner components."""
    print("\n" + "=" * 70)
    print("DETAILED PHYSICAL PLANNER PROFILING")
    print("=" * 70)

    from pandas.lazy import col
    from pandas.lazy.optimize import Optimizer
    from pandas.lazy.physical import (
        PhysicalPlanner,
        execute_physical_plan,
    )

    df = create_test_data(100_000, use_arrow=True)

    # Build the lazy frame
    lf = df.select()
    lf = lf.filter(col("a") > 0.5)
    lf = lf.with_columns(
        (col("a") + col("b")).alias("sum_ab"),
        (col("a") * col("b")).alias("prod_ab"),
        (col("c") / (col("d") + 1)).alias("ratio"),
    )
    lf = lf.filter(col("sum_ab") > 1.0)

    # Get optimized plan
    optimized = Optimizer().optimize(lf._plan)

    # Profile just the physical planning phase
    print("\n--- Physical Plan Creation (100 iterations) ---")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(100):
        planner = PhysicalPlanner(preferred_backend="auto")
        plan = planner.plan(optimized)
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    # Profile just the execution phase
    print("\n--- Physical Plan Execution (10 iterations) ---")

    planner = PhysicalPlanner(preferred_backend="auto")
    plan = planner.plan(optimized)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):
        execute_physical_plan(plan, preferred_backend="auto", strict=False)
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


def profile_type_inference():
    """Profile type inference specifically."""
    print("\n" + "=" * 70)
    print("TYPE INFERENCE PROFILING")
    print("=" * 70)

    from pandas.lazy.ir import (
        Call,
        FieldRef,
    )
    from pandas.lazy.types import (
        LazyDtype,
        infer_expr_dtype,
    )

    df = create_test_data(100_000, use_arrow=True)

    # Build schema
    schema = {}
    for col_name in df.columns:
        dtype = df[col_name].dtype
        schema[col_name] = LazyDtype.from_pandas_dtype(dtype)

    # Create a complex expression IR
    expr = Call(
        "add",
        (
            Call("multiply", (FieldRef("a"), FieldRef("b")), {}),
            Call("divide", (FieldRef("c"), FieldRef("d")), {}),
        ),
        {},
    )

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(1000):
        infer_expr_dtype(expr, schema)
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())


def profile_backend_selection():
    """Profile backend selection logic."""
    print("\n" + "=" * 70)
    print("BACKEND SELECTION PROFILING")
    print("=" * 70)

    from pandas.lazy.optimize.engine import decide_expr_backend

    # Simulate many backend decisions
    funcs = ["add", "multiply", "divide", "filter", "str_lower", "sum", "mean"]
    backends = ["arrow", "numpy"]

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10000):
        for func in funcs:
            for input_backend in backends:
                decide_expr_backend(func, input_backend, "auto")
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())


def profile_array_eval():
    """Profile array evaluator specifically."""
    print("\n" + "=" * 70)
    print("ARRAY EVALUATOR PROFILING")
    print("=" * 70)

    import pyarrow as pa

    from pandas.lazy.backends.array_eval import ArrayEvaluator
    from pandas.lazy.ir import (
        Call,
        FieldRef,
    )

    # Create test arrays
    n = 100_000
    rng = np.random.default_rng(42)
    arrays = {
        "a": pa.array(rng.random(n)),
        "b": pa.array(rng.random(n)),
        "c": pa.array(rng.random(n)),
        "d": pa.array(rng.integers(1, 100, n)),
    }

    # Create expression: (a + b) * c / d
    expr = Call(
        "divide",
        (
            Call(
                "multiply",
                (Call("add", (FieldRef("a"), FieldRef("b")), {}), FieldRef("c")),
                {},
            ),
            FieldRef("d"),
        ),
        {},
    )

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(100):
        evaluator = ArrayEvaluator(arrays, preferred_backend="auto")
        evaluator.evaluate(expr)
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    profile_comparison()
    profile_physical_planner_detailed()
    profile_type_inference()
    profile_backend_selection()
    profile_array_eval()
