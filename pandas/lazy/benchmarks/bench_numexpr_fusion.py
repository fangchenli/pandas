#!/usr/bin/env python
"""
Benchmark NumExpr expression fusion performance.

This script compares:
1. NumPy element-wise operations (baseline)
2. NumExpr fused operations
3. Lazy evaluation with and without fusion
"""

import time

import numpy as np

import pandas as pd


def benchmark_raw_operations(n_rows: int = 1_000_000):
    """Benchmark raw NumPy vs NumExpr operations."""
    print("=" * 70)
    print(f"RAW OPERATION BENCHMARKS (n={n_rows:,})")
    print("=" * 70)

    rng = np.random.default_rng(42)
    a = rng.random(n_rows)
    b = rng.random(n_rows)
    c = rng.random(n_rows)
    d = rng.random(n_rows) + 0.1  # Avoid division by zero

    # Expression: (a + b) * c / d
    n_iterations = 20

    # NumPy element-wise
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = (a + b) * c / d
    numpy_time = (time.perf_counter() - start) / n_iterations * 1000

    # NumExpr fused
    try:
        import numexpr as ne

        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = ne.evaluate("(a + b) * c / d")
        numexpr_time = (time.perf_counter() - start) / n_iterations * 1000

        print("\nExpression: (a + b) * c / d")
        print(f"  NumPy:   {numpy_time:.2f}ms")
        print(f"  NumExpr: {numexpr_time:.2f}ms")
        print(f"  Speedup: {numpy_time / numexpr_time:.2f}x")

    except ImportError:
        print("\nNumExpr not installed - skipping raw benchmark")
        return

    # More complex expression: (a + b) * c / d + (a - b) ** 2
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = (a + b) * c / d + (a - b) ** 2
    numpy_time = (time.perf_counter() - start) / n_iterations * 1000

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = ne.evaluate("(a + b) * c / d + (a - b) ** 2")
    numexpr_time = (time.perf_counter() - start) / n_iterations * 1000

    print("\nExpression: (a + b) * c / d + (a - b) ** 2")
    print(f"  NumPy:   {numpy_time:.2f}ms")
    print(f"  NumExpr: {numexpr_time:.2f}ms")
    print(f"  Speedup: {numpy_time / numexpr_time:.2f}x")

    # Comparison with boolean result: (a > 0.5) & (b < 0.5) | (c > 0.3)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = (a > 0.5) & (b < 0.5) | (c > 0.3)
    numpy_time = (time.perf_counter() - start) / n_iterations * 1000

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = ne.evaluate("(a > 0.5) & (b < 0.5) | (c > 0.3)")
    numexpr_time = (time.perf_counter() - start) / n_iterations * 1000

    print("\nExpression: (a > 0.5) & (b < 0.5) | (c > 0.3)")
    print(f"  NumPy:   {numpy_time:.2f}ms")
    print(f"  NumExpr: {numexpr_time:.2f}ms")
    print(f"  Speedup: {numpy_time / numexpr_time:.2f}x")


def benchmark_fusion_module(n_rows: int = 1_000_000):
    """Benchmark the fusion module directly."""
    print("\n" + "=" * 70)
    print(f"FUSION MODULE BENCHMARKS (n={n_rows:,})")
    print("=" * 70)

    try:
        from pandas.lazy.backends.numexpr_fusion import (
            build_numexpr_string,
            can_fuse_expression,
            count_fuseable_ops,
        )
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )
    except ImportError as e:
        print(f"Import error: {e}")
        return

    rng = np.random.default_rng(42)
    arrays = {
        "a": rng.random(n_rows),
        "b": rng.random(n_rows),
        "c": rng.random(n_rows),
        "d": rng.random(n_rows) + 0.1,
    }

    # Build IR for (a + b) * c / d
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

    # Check if expression can be fused
    print("\nExpression IR: (a + b) * c / d")
    print(f"  Fuseable ops: {count_fuseable_ops(expr)}")
    print(f"  Can fuse: {can_fuse_expression(expr, arrays)}")

    # Build numexpr string
    var_map: dict[str, str] = {}
    expr_str = build_numexpr_string(expr, var_map)
    print(f"  NumExpr string: {expr_str}")
    print(f"  Variable map: {var_map}")

    # Benchmark fusion
    n_iterations = 20

    # Direct NumPy evaluation
    from pandas.lazy.backends.array_eval import ArrayEvaluator

    evaluator_no_ne = ArrayEvaluator(
        arrays, preferred_backend="numpy", use_numexpr=False
    )
    start = time.perf_counter()
    for _ in range(n_iterations):
        evaluator_no_ne.evaluate(expr)
    no_ne_time = (time.perf_counter() - start) / n_iterations * 1000

    # NumExpr fusion
    evaluator_ne = ArrayEvaluator(arrays, preferred_backend="numpy", use_numexpr=True)
    start = time.perf_counter()
    for _ in range(n_iterations):
        evaluator_ne.evaluate(expr)
    ne_time = (time.perf_counter() - start) / n_iterations * 1000

    print("\nArrayEvaluator benchmark:")
    print(f"  Without NumExpr: {no_ne_time:.2f}ms")
    print(f"  With NumExpr:    {ne_time:.2f}ms")
    print(f"  Speedup:         {no_ne_time / ne_time:.2f}x")


def benchmark_lazy_evaluation(n_rows: int = 1_000_000):
    """Benchmark lazy evaluation with and without NumExpr."""
    print("\n" + "=" * 70)
    print(f"LAZY EVALUATION BENCHMARKS (n={n_rows:,})")
    print("=" * 70)

    from pandas.lazy import col

    rng = np.random.default_rng(42)

    # Create DataFrame with NumPy backend (for NumExpr compatibility)
    df = pd.DataFrame(
        {
            "a": rng.random(n_rows),
            "b": rng.random(n_rows),
            "c": rng.random(n_rows),
            "d": rng.random(n_rows) + 0.1,
        }
    )

    # Build lazy expression: (a + b) * c / d
    lf = df.select().with_columns(
        ((col("a") + col("b")) * col("c") / col("d")).alias("result")
    )

    n_iterations = 10

    # Lazy evaluation without physical planner
    start = time.perf_counter()
    for _ in range(n_iterations):
        lf.collect(use_physical_planner=False)
    no_physical_time = (time.perf_counter() - start) / n_iterations * 1000

    # Lazy evaluation with physical planner
    start = time.perf_counter()
    for _ in range(n_iterations):
        lf.collect(use_physical_planner=True)
    physical_time = (time.perf_counter() - start) / n_iterations * 1000

    print("\nLazy expression: (a + b) * c / d")
    print(f"  Without physical planner: {no_physical_time:.2f}ms")
    print(f"  With physical planner:    {physical_time:.2f}ms")
    print(f"  Speedup:                  {no_physical_time / physical_time:.2f}x")

    # More complex expression with filter
    lf_complex = (
        df.select()
        .filter(col("a") > 0.5)
        .with_columns(
            ((col("a") + col("b")) * col("c") / col("d")).alias("result1"),
            ((col("a") - col("b")) ** 2).alias("result2"),
        )
        .filter(col("result1") > 0.5)
    )

    start = time.perf_counter()
    for _ in range(n_iterations):
        lf_complex.collect(use_physical_planner=False)
    no_physical_time = (time.perf_counter() - start) / n_iterations * 1000

    start = time.perf_counter()
    for _ in range(n_iterations):
        lf_complex.collect(use_physical_planner=True)
    physical_time = (time.perf_counter() - start) / n_iterations * 1000

    print("\nComplex lazy expression with filters:")
    print(f"  Without physical planner: {no_physical_time:.2f}ms")
    print(f"  With physical planner:    {physical_time:.2f}ms")
    print(f"  Speedup:                  {no_physical_time / physical_time:.2f}x")


def benchmark_size_threshold(sizes: list[int] | None = None):
    """Benchmark different array sizes to find optimal threshold."""
    print("\n" + "=" * 70)
    print("SIZE THRESHOLD ANALYSIS")
    print("=" * 70)

    if sizes is None:
        sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]

    try:
        import numexpr as ne
    except ImportError:
        print("NumExpr not installed - skipping threshold analysis")
        return

    results = []

    for n in sizes:
        rng = np.random.default_rng(42)
        a = rng.random(n)
        b = rng.random(n)
        c = rng.random(n)
        d = rng.random(n) + 0.1

        n_iterations = 20

        # NumPy
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = (a + b) * c / d
        numpy_time = (time.perf_counter() - start) / n_iterations * 1000

        # NumExpr
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = ne.evaluate("(a + b) * c / d")
        numexpr_time = (time.perf_counter() - start) / n_iterations * 1000

        speedup = numpy_time / numexpr_time
        results.append((n, numpy_time, numexpr_time, speedup))

    print(f"\n{'Size':>12} {'NumPy (ms)':>12} {'NumExpr (ms)':>14} {'Speedup':>10}")
    print("-" * 50)
    for n, np_time, ne_time, speedup in results:
        print(f"{n:>12,} {np_time:>12.2f} {ne_time:>14.2f} {speedup:>10.2f}x")


def benchmark_arrow_cached_functions(n_rows: int = 1_000_000):
    """Benchmark Arrow cached function calls vs direct calls."""
    print("\n" + "=" * 70)
    print(f"ARROW CACHED FUNCTION BENCHMARKS (n={n_rows:,})")
    print("=" * 70)

    import pyarrow as pa
    import pyarrow.compute as pc

    rng = np.random.default_rng(42)
    a = pa.array(rng.random(n_rows))
    b = pa.array(rng.random(n_rows))
    c = pa.array(rng.random(n_rows))
    d = pa.array(rng.random(n_rows) + 0.1)

    n_iterations = 50

    # Direct pc.add/multiply/divide calls
    start = time.perf_counter()
    for _ in range(n_iterations):
        t1 = pc.add(a, b)
        t2 = pc.multiply(t1, c)
        _ = pc.divide(t2, d)
    direct_time = (time.perf_counter() - start) / n_iterations * 1000

    # Cached function references
    add_fn = pc.get_function("add")
    mul_fn = pc.get_function("multiply")
    div_fn = pc.get_function("divide")

    start = time.perf_counter()
    for _ in range(n_iterations):
        t1 = add_fn.call([a, b])
        t2 = mul_fn.call([t1, c])
        _ = div_fn.call([t2, d])
    cached_time = (time.perf_counter() - start) / n_iterations * 1000

    print("\nExpression: (a + b) * c / d")
    print(f"  Direct pc.* calls:  {direct_time:.2f}ms")
    print(f"  Cached fn.call():   {cached_time:.2f}ms")
    print(f"  Speedup:            {direct_time / cached_time:.2f}x")

    # Test with ArrayEvaluator
    try:
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": a, "b": b, "c": c, "d": d}

        # Build IR for (a + b) * c / d
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

        evaluator = ArrayEvaluator(arrays, preferred_backend="arrow")
        start = time.perf_counter()
        for _ in range(n_iterations):
            evaluator.evaluate(expr)
        eval_time = (time.perf_counter() - start) / n_iterations * 1000

        print(f"\nArrayEvaluator (Arrow): {eval_time:.2f}ms")
        print(f"  vs direct calls:      {direct_time / eval_time:.2f}x")

    except ImportError as e:
        print(f"\nArrayEvaluator test skipped: {e}")


if __name__ == "__main__":
    benchmark_raw_operations()
    benchmark_fusion_module()
    benchmark_arrow_cached_functions()
    benchmark_lazy_evaluation()
    benchmark_size_threshold()
