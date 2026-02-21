#!/usr/bin/env python3
"""
Demo: pd.compile — graph breaks handled transparently.

Every example here includes operations that would fail with naive proxy
tracing (len, shape, iloc, apply, iterrows). The system materializes at
graph break points and resumes tracing.
"""

import time

import numpy as np

import pandas as pd
from pandas.compile import (
    PandasBackend,
    Tracer,
    compile,
)

# ============================================================
# Sample data
# ============================================================
sales_df = pd.DataFrame(
    {
        "id": range(1, 9),
        "region": ["East", "West", "East", "West", "East", "West", "East", "West"],
        "product": ["A", "B", "A", "A", "B", "B", "A", "A"],
        "price": [100, 250, 150, 300, 50, 175, 225, 400],
        "quantity": [10, 5, 8, 3, 20, 7, 4, 2],
    }
)

regions_df = pd.DataFrame(
    {
        "region": ["East", "West"],
        "manager": ["Alice", "Bob"],
        "budget": [50000, 75000],
    }
)

empty_df = pd.DataFrame(
    {
        "id": pd.Series([], dtype="int64"),
        "region": pd.Series([], dtype="object"),
        "price": pd.Series([], dtype="int64"),
    }
)

SEP = "-" * 72

print("=" * 72)
print("  PANDAS JIT: Transparent Graph Break Handling")
print("=" * 72)
print()


# ============================================================
# Test 1: len(df) in control flow
# ============================================================
print(SEP)
print("  TEST 1: if len(df) > 0  (data-dependent control flow)")
print(SEP)


@compile(backend=PandasBackend())
def safe_aggregate(df):
    """Only aggregate if there are rows after filtering."""
    filtered = df[df["price"] > 100]
    if len(filtered) > 0:
        result = filtered.groupby("region").sum()
        return result
    else:
        return filtered


result = safe_aggregate(sales_df)
print("\n  Result (non-empty case):")
print(f"  {result.to_string(index=False)}")

result_empty = safe_aggregate(empty_df)
print("\n  Result (empty case):")
print(f"  {result_empty.to_string(index=False)}")

print("\n  Execution plan:")
print(safe_aggregate.explain(sales_df))


# ============================================================
# Test 2: Chained graph breaks
# ============================================================
print(f"\n{SEP}")
print("  TEST 2: Multiple graph breaks in one function")
print(SEP)


@compile(backend=PandasBackend())
def multi_break(df):
    """Multiple materializations in one function."""
    expensive = df[df["price"] > 100]

    n_expensive = len(expensive)
    print(f"    -> After filter: {n_expensive} rows")

    expensive["total"] = expensive["price"] * expensive["quantity"]
    top = expensive.sort_values("total", ascending=False)

    max_total = top["total"].max()
    print(f"    -> Max total: {max_total}")

    threshold = max_total * 0.5
    significant = top[top["total"] > threshold]

    return significant


result = multi_break(sales_df)
print("\n  Result:")
print(f"  {result.to_string(index=False)}")
print("\n  Execution plan:")
print(multi_break.explain(sales_df))


# ============================================================
# Test 3: df.apply() — full graph break with resume
# ============================================================
print(f"\n{SEP}")
print("  TEST 3: df.apply() — opaque Python function")
print(SEP)


@compile(backend=PandasBackend())
def with_apply(df):
    """apply() can't be traced, but we handle it."""
    filtered = df[df["price"] > 50]

    filtered["price_tier"] = filtered["price"].apply(
        lambda x: "premium" if x > 200 else "standard"
    )

    premium = filtered[filtered["price_tier"] == "premium"]
    return premium[["region", "product", "price", "price_tier"]]


result = with_apply(sales_df)
print("\n  Result:")
print(f"  {result.to_string(index=False)}")


# ============================================================
# Test 4: df.shape and df.iloc — positional access
# ============================================================
print(f"\n{SEP}")
print("  TEST 4: df.shape and df.iloc[] — positional indexing")
print(SEP)


@compile(backend=PandasBackend())
def shape_and_iloc(df):
    """shape and iloc both require materialization."""
    sorted_df = df.sort_values("price", ascending=False)

    n_rows, n_cols = sorted_df.shape
    print(f"    -> Shape: {n_rows} rows x {n_cols} cols")

    top_half = sorted_df.iloc[: n_rows // 2]

    return top_half[["region", "product", "price"]]


result = shape_and_iloc(sales_df)
print("\n  Result (top half by price):")
print(f"  {result.to_string(index=False)}")


# ============================================================
# Test 5: iterrows — full eager fallback
# ============================================================
print(f"\n{SEP}")
print("  TEST 5: iterrows() — imperative loop")
print(SEP)


@compile(backend=PandasBackend())
def with_iterrows(df):
    """iterrows is fundamentally imperative, but we handle it."""
    filtered = df[df["price"] > 100]

    tags = []
    for _, row in filtered.iterrows():
        if row["price"] > 200:
            tags.append("high")
        else:
            tags.append("mid")

    result = (
        filtered._ensure_materialized().copy()
        if hasattr(filtered, "_ensure_materialized")
        else filtered.copy()
    )
    result["tag"] = tags
    return result


result = with_iterrows(sales_df)
print("\n  Result:")
print(f"  {result.to_string(index=False)}")


# ============================================================
# Test 6: Complex real-world pipeline
# ============================================================
print(f"\n{SEP}")
print("  TEST 6: Complex real-world pipeline with multiple breaks")
print(SEP)


@compile(backend=PandasBackend())
def real_world_pipeline(sales, regions):
    """A realistic pipeline mixing compiled and eager ops."""
    merged = sales.merge(regions, on="region")
    merged["revenue"] = merged["price"] * merged["quantity"]
    high_value = merged[merged["revenue"] > 200]

    n = len(high_value)
    print(f"    -> High-value sales: {n}")

    if n == 0:
        return high_value

    by_manager = (
        high_value.groupby("manager")
        .agg(
            total_revenue=("revenue", "sum"),
            n_sales=("id", "count"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    return by_manager


result = real_world_pipeline(sales_df, regions_df)
print("\n  Result:")
print(f"  {result.to_string(index=False)}")
print("\n  Execution plan:")
print(real_world_pipeline.explain(sales_df, regions_df))


# ============================================================
# Test 7: Context manager with graph breaks
# ============================================================
print(f"\n{SEP}")
print("  TEST 7: Context manager with graph breaks")
print(SEP)

with Tracer(backend=PandasBackend()) as t:
    df = t.input(sales_df, "sales")

    expensive = df[df["price"] > 100]

    n = len(expensive)
    print(f"    -> Expensive items: {n}")

    summary = expensive.groupby("region").agg(
        avg_price=("price", "mean"),
        total_qty=("quantity", "sum"),
    )

    t.output(summary)

print("\n  Result:")
print(f"  {t.result().to_string(index=False)}")
print("\n  Execution plan:")
print(t.explain())


# ============================================================
# Test 8: Guard system — cache hit
# ============================================================
print(f"\n{SEP}")
print("  TEST 8: Guard system — plan caching")
print(SEP)


@compile(backend=PandasBackend())
def cacheable(df):
    return df[df["price"] > 100].groupby("region").sum()


t0 = time.perf_counter()
result1 = cacheable(sales_df)
t1 = time.perf_counter()
trace_time = t1 - t0

t0 = time.perf_counter()
result2 = cacheable(sales_df)
t1 = time.perf_counter()
cached_time = t1 - t0

print(f"  First call (trace):  {trace_time * 1000:.2f}ms")
print(f"  Second call (cache): {cached_time * 1000:.2f}ms")
print(f"  Cache entries: {len(cacheable._cached_plans)}")
assert result1.equals(result2), "Results should be identical"
print("  Results match: OK")


# ============================================================
# Test 9: Straight-line code (no breaks — single compiled segment)
# ============================================================
print(f"\n{SEP}")
print("  TEST 9: Straight-line code — single compiled segment (optimal)")
print(SEP)


@compile(backend=PandasBackend())
def pure_pipeline(df):
    """No graph breaks — compiles to a single Substrait plan."""
    return (
        df.assign(total=lambda x: x["price"] * x["quantity"])[
            ["region", "product", "total"]
        ]
        .query("total > 500")
        .groupby("region")
        .agg(revenue=("total", "sum"))
        .sort_values("revenue", ascending=False)
    )


result = pure_pipeline(sales_df)
print("\n  Result:")
print(f"  {result.to_string(index=False)}")
print("\n  Execution plan:")
plan_str = pure_pipeline.explain(sales_df)
print(plan_str)
n_segments = plan_str.count("COMPILED")
print(f"\n  Segments: {n_segments} (ideal: 1 = fully compiled)")


# ============================================================
# Test 10: Correctness verification — JIT vs pandas
# ============================================================
print(f"\n{SEP}")
print("  TEST 10: Correctness verification — JIT vs pandas")
print(SEP)


def process_pandas(df):
    """Pure pandas version."""
    filtered = df[df["price"] > 100]
    filtered = filtered.copy()
    filtered["total"] = filtered["price"] * filtered["quantity"]
    result = (
        filtered.groupby("region")
        .agg(
            revenue=("total", "sum"),
            count=("id", "count"),
        )
        .sort_values("revenue", ascending=False)
    )
    return result


@compile(backend=PandasBackend())
def process_jit(df):
    """JIT version — same code."""
    filtered = df[df["price"] > 100]
    filtered["total"] = filtered["price"] * filtered["quantity"]
    result = (
        filtered.groupby("region")
        .agg(
            revenue=("total", "sum"),
            count=("id", "count"),
        )
        .sort_values("revenue", ascending=False)
    )
    return result


pandas_result = process_pandas(sales_df)
jit_result = process_jit(sales_df)

print("\n  Pandas result:")
print(f"  {pandas_result.to_string()}")
print("\n  JIT result:")
print(f"  {jit_result.to_string(index=False)}")

for col in pandas_result.columns:
    pd_vals = pandas_result[col].values
    jit_vals = jit_result[col].values
    match = np.array_equal(pd_vals, jit_vals)
    print(f"  Column '{col}': {'OK' if match else 'MISMATCH'}")


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 72}")
print("  SUMMARY")
print(f"{'=' * 72}")
print("""
  Transparent materialization:
    len(df)                   -> Materializes, returns int, resumes
    if len(df) > 0:           -> Works! Both branches traceable
    df.shape                  -> Materializes, returns tuple
    df.iloc[:]                -> Materializes, returns TracedDataFrame
    df.apply(fn)              -> Materializes, runs fn, resumes
    df.iterrows()             -> Materializes, iterates real data
    Multiple breaks           -> Each creates a new segment
    Plan caching              -> Guard on schema, reuse compiled plan
    Straight-line code        -> Still compiles to single segment
""")
