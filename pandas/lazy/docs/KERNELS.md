# Lazy Pandas Kernel Reference

## Overview

This document describes which operations use which backend (Arrow, NumPy, or pandas Cython fallback) in the lazy pandas physical planner.

## Physical Plan Operators

| Operator | Primary Backend | Fallback | Notes |
|----------|----------------|----------|-------|
| `PhysicalScan` | Direct extraction | N/A | Extracts arrays from DataFrame |
| `PhysicalParquetScan` | PyArrow Parquet | N/A | Lazy Parquet with pushdown |
| `PhysicalFilter` | Arrow/NumPy kernel | pandas DataFrame | Uses `filter` kernel |
| `PhysicalProject` | Arrow/NumPy kernel | pandas Evaluator | Expression evaluation |
| `PhysicalHashAggregate` | Arrow/NumPy kernel | pandas groupby | Uses `groupby_*` kernels |
| `PhysicalSort` | Cython radix argsort / radix lexsort | `np.lexsort` | Numeric keys via `pandas._libs.lazy_radix` (single-key parallel radix, multi-key + factorized-string lexsort); Arrow/`np.lexsort` fallback |
| `PhysicalTopK` | Arrow/NumPy kernel | Full sort | Uses `select_k_unstable` kernel |
| `PhysicalLimit` | Direct slicing | N/A | No kernel needed |
| `PhysicalDistinct` | Arrow/NumPy kernel | sortedness-aware first-occurrence kernel (diff-based when sorted, hash-based `duplicated()` otherwise — no argsort) | Uses `unique_indices` kernel |
| `PhysicalGroupByHead` | pandas `groupby().head(n)` | — | Group-wise head; with a preceding sort gives top-k rows per group |
| `PhysicalHashJoin` | **`pd.merge` (in-memory equi-joins)** | Arrow/NumPy `hash_join`, Grace (size-triggered) | `pd.merge` is primary (eager semantics, fastest); acero/indexer/Grace are fallbacks |
| `PhysicalConvert` | Direct conversion | N/A | Backend conversion |

## Kernel Categories

### Arithmetic Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `add` | `np.add` | `pc.add` | Equivalent |
| `subtract` | `np.subtract` | `pc.subtract` | Equivalent |
| `multiply` | `np.multiply` | `pc.multiply` | Equivalent |
| `divide` | `np.divide` | `pc.divide` | Equivalent |
| `floor_divide` | `np.floor_divide` | `pc.divide` + `pc.floor` | NumPy faster |
| `modulo` | `np.mod` | `pc.call_function` | NumPy faster |
| `power` | `np.power` | `pc.power` | Equivalent |
| `negate` | `np.negative` | `pc.negate` | Equivalent |
| `abs` | `np.abs` | `pc.abs` | Equivalent |

### Comparison Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `equal` | `np.equal` | `pc.equal` | Equivalent |
| `not_equal` | `np.not_equal` | `pc.not_equal` | Equivalent |
| `less` | `np.less` | `pc.less` | Equivalent |
| `less_equal` | `np.less_equal` | `pc.less_equal` | Equivalent |
| `greater` | `np.greater` | `pc.greater` | Equivalent |
| `greater_equal` | `np.greater_equal` | `pc.greater_equal` | Equivalent |

### Logical Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `and_` | `np.logical_and` | `pc.and_` | Equivalent |
| `or_` | `np.logical_or` | `pc.or_` | Equivalent |
| `invert` | `np.logical_not` | `pc.invert` | Equivalent |

### String Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `str_lower` | pandas StringMethods | `pc.utf8_lower` | **Arrow 2-10x faster** |
| `str_upper` | pandas StringMethods | `pc.utf8_upper` | **Arrow 2-10x faster** |
| `str_len` | pandas StringMethods | `pc.utf8_length` | **Arrow 2-10x faster** |
| `str_strip` | pandas StringMethods | `pc.utf8_trim_whitespace` | **Arrow 2-10x faster** |
| `str_contains` | pandas StringMethods | `pc.match_substring_regex` | **Arrow only** |
| `str_startswith` | pandas StringMethods | `pc.starts_with` | **Arrow only** |
| `str_endswith` | pandas StringMethods | `pc.ends_with` | **Arrow only** |
| `str_replace` | pandas StringMethods | `pc.replace_substring_regex` | **Arrow only** |
| `str_slice` | pandas StringMethods | `pc.utf8_slice_codeunits` | **Arrow only** |
| `str_lstrip` | pandas StringMethods | `pc.utf8_ltrim_whitespace` | **Arrow only** |
| `str_rstrip` | pandas StringMethods | `pc.utf8_rtrim_whitespace` | **Arrow only** |

### Null Handling

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `is_null` | `np.isnan` | `pc.is_null` | Arrow handles all nullables |
| `is_not_null` | `~np.isnan` | `pc.is_valid` | Arrow handles all nullables |
| `fill_null` | `np.where` | `pc.fill_null` | Equivalent |
| `coalesce` | N/A | `pc.coalesce` | **Arrow only** |

### Aggregation Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `sum` | `np.nansum` | `pc.sum` | Equivalent |
| `mean` | `np.nanmean` | `pc.mean` | Equivalent |
| `min` | `np.nanmin` | `pc.min` | Equivalent |
| `max` | `np.nanmax` | `pc.max` | Equivalent |
| `count` | `np.count_nonzero` | `pc.count` | Equivalent |
| `std` | `np.nanstd` | `pc.stddev` | Equivalent |
| `var` | `np.nanvar` | `pc.variance` | Equivalent |
| `first` | `arr[0]` | `pc.first` | Equivalent |
| `last` | `arr[-1]` | `pc.last` | Equivalent |
| `n_unique` | `len(np.unique)` | `pc.count_distinct` | Equivalent |

### GroupBy Aggregations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `groupby_sum` | `np.bincount` | `pa.Table.group_by` | Arrow faster for many groups |
| `groupby_mean` | bincount + divide | `pa.Table.group_by` | Arrow faster |
| `groupby_min` | `np.minimum.at` | `pa.Table.group_by` | Equivalent |
| `groupby_max` | `np.maximum.at` | `pa.Table.group_by` | Equivalent |
| `groupby_count` | `np.bincount` | `pa.Table.group_by` | Equivalent |
| `groupby_std` | Two-pass algorithm | `pa.Table.group_by` | Equivalent |
| `groupby_var` | Two-pass algorithm | `pa.Table.group_by` | Equivalent |
| `groupby_median` | pandas grouped median | *(no Arrow kernel)* | Exact; forces the NumPy path (Arrow has only approximate t-digest) |
| `groupby_first` | Index-based | `pa.Table.group_by` | Equivalent |
| `groupby_last` | Index-based | `pa.Table.group_by` | Equivalent |

### Rolling Window Operations

| Kernel | NumPy (fallback) | Bottleneck (fast path) | Performance Notes |
|--------|------------------|------------------------|-------------------|
| `rolling_sum` | O(n) cumsum | `bn.move_sum` | **Bottleneck ~100% of pandas** |
| `rolling_mean` | O(n) cumsum | `bn.move_mean` | **Bottleneck ~100% of pandas** |
| `rolling_min` | sliding_window_view | `bn.move_min` | **Bottleneck ~100% of pandas** |
| `rolling_max` | sliding_window_view | `bn.move_max` | **Bottleneck ~100% of pandas** |
| `rolling_std` | O(n) cumsum variance | `bn.move_std` | **Bottleneck ~100% of pandas** |
| `rolling_var` | O(n) cumsum variance | `bn.move_var` | **Bottleneck ~100% of pandas** |
| `rolling_median` | sliding_window_view + nanmedian | `bn.move_median` | **Bottleneck only efficient** |
| `rolling_argmin` | sliding_window_view | `bn.move_argmin` | Index of min in window |
| `rolling_argmax` | sliding_window_view | `bn.move_argmax` | Index of max in window |
| `rolling_rank` | N/A | `bn.move_rank` | **Bottleneck required** |

**Note:** When Bottleneck is installed and `pd.set_option("compute.use_bottleneck", True)`, rolling operations use the C-optimized Bottleneck functions for ~100% of pandas Cython performance. Without Bottleneck, the pure NumPy fallback achieves 54-96% of pandas performance.

### Cumulative Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `cumulative_sum` | `np.nancumsum` | `pc.cumulative_sum` | NumPy slightly faster |
| `cumulative_min` | `np.minimum.accumulate` | `pc.cumulative_min` | Equivalent |
| `cumulative_max` | `np.maximum.accumulate` | `pc.cumulative_max` | Equivalent |
| `cumulative_prod` | `np.nancumprod` | `pc.cumulative_prod` | NumPy slightly faster |

### Fill Operations

| Kernel | NumPy (fallback) | Bottleneck (fast path) | Arrow | Performance Notes |
|--------|------------------|------------------------|-------|-------------------|
| `ffill` | `np.maximum.accumulate` trick | `bn.push` | `pc.fill_null_forward` | **Bottleneck faster** |
| `bfill` | reverse + accumulate | `bn.push` (reversed) | `pc.fill_null_backward` | **Bottleneck faster** |
| `interpolate_linear` | `np.interp` | N/A | N/A | NumPy only |

**Note:** With Bottleneck, ffill/bfill operations are significantly faster than the pure NumPy fallback.

### Join Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `hash_join` | Factorize-based | `pa.Table.join` | Fallback only — the primary in-memory path is `pd.merge` (see below) |
| `inner_join` | Via hash_join | Via hash_join | Equivalent |
| `left_join` | Via hash_join | Via hash_join | Equivalent |
| `right_join` | Via hash_join | Via hash_join | Equivalent |
| `outer_join` | Via hash_join | Via hash_join | Equivalent |
| `semi_join` | N/A | `pa.Table.join` | **Arrow only** |
| `anti_join` | N/A | `pa.Table.join` | **Arrow only** |
| `cross_join` | DataFrame fallback | DataFrame fallback | Always uses pandas |

### Datetime Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `dt_year` | View cast | `pc.year` | Arrow slightly faster |
| `dt_month` | View cast | `pc.month` | Arrow slightly faster |
| `dt_day` | View cast | `pc.day` | Arrow slightly faster |
| `dt_hour` | View cast | `pc.hour` | Arrow slightly faster |
| `dt_minute` | View cast | `pc.minute` | Arrow slightly faster |
| `dt_second` | View cast | `pc.second` | Arrow slightly faster |
| `dt_day_of_week` | View cast | `pc.day_of_week` | Equivalent |
| `dt_day_of_year` | Calculation | `pc.day_of_year` | Equivalent |
| `dt_strftime` | `strftime` loop | `pc.strftime` | **Arrow faster** |

### Sort/Selection Operations

| Kernel | NumPy | Arrow | Performance Notes |
|--------|-------|-------|-------------------|
| `sort_indices` | Cython LSD radix argsort (`lazy_radix`), thread-parallel ≥1M; k-way merge / `np.argsort` fallback | `pc.sort_indices` | Radix ~130 ms/10M float64, faster than Polars `arg_sort` |
| `take` | `arr[indices]` | `pc.take` | Equivalent |
| `select_k_unstable` | `np.argpartition` | `pc.select_k_unstable` | NumPy slightly faster |
| `unique` | sortedness-aware first-occurrence kernel (diff-based when sorted, hash-based `duplicated()` otherwise — no argsort) | `pc.unique` | Equivalent |
| `unique_indices` | sortedness-aware first-occurrence kernel (diff-based when sorted, hash-based `duplicated()` otherwise — no argsort) with return_index | N/A | **NumPy only** |

## Backend Routing

The backend router (`pandas/lazy/backends/router.py`) decides which backend to use:

### Arrow-Preferred Operations
These operations are **always** routed to Arrow when available (2-10x faster):
```python
ARROW_PREFERRED_OPS = {
    "str_lower", "str_upper", "str_len", "str_strip",
    "str_lstrip", "str_rstrip", "str_contains",
    "str_startswith", "str_endswith", "str_replace", "str_slice",
    "is_null", "is_not_null", "fill_null", "coalesce",
}
```

### Neutral Operations
These operations follow the input data format:
- Arithmetic: `add`, `subtract`, `multiply`, `divide`, etc.
- Comparison: `equal`, `less`, `greater`, etc.
- Logical: `and_`, `or_`, `invert`
- Aggregations: `sum`, `mean`, `min`, `max`, etc.

### Backend routing for group-by
`groupby_prefers_arrow` routes a hash aggregation to Arrow/acero when every
aggregation has an Arrow groupby kernel **and** (a relevant column is
Arrow-backed **or** all relevant columns are numeric). So numeric-keyed
group-by goes to acero (NumPy→Arrow numeric conversion is zero-copy), not the
pandas NumPy path. The exception is `median`, which has no Arrow kernel and is
therefore forced to the exact NumPy `groupby_median`.

### Bottleneck-Accelerated Operations
When Bottleneck is installed (`pip install bottleneck`) and enabled:
- Rolling window operations: `rolling_sum`, `rolling_mean`, `rolling_min`, `rolling_max`, `rolling_std`, `rolling_var`, `rolling_median`, `rolling_argmin`, `rolling_argmax`, `rolling_rank`
- Fill operations: `ffill`, `bfill`

## Performance Summary

Standings below are vs **Polars** on the H2O db-benchmark
(`benchmarks/H2O_BENCHMARK.md`) unless noted vs pandas.

### Where Lazy Pandas Wins (vs Polars)
1. **Sort (numeric)**: radix argsort, faster than Polars `arg_sort`
2. **Group-by**: 6 of 10 H2O queries beat Polars (string-key sums up to ~7x,
   `corr` ~3x, agg-arithmetic ~2x)
3. **String operations on Arrow data**: faster than pandas
4. **Filter pushdown / projection pruning**: reduces work before expensive ops
5. **TopK optimization**: O(n log k) vs O(n log n) for small k

### Where Lazy Pandas is Competitive
1. **Joins**: route to `pd.merge` (≈ pandas merge speed by construction);
   vs Polars 0.14–1.30x on H2O depending on shape (Polars' parallel hash join
   wins on string-key/left joins)
2. **Rolling operations (with Bottleneck)**: ~100% of pandas Cython
3. **Arithmetic/comparison**: equivalent to pandas
4. **Fill operations (with Bottleneck)**: ~100% of pandas

### Where Lazy Pandas is Slower (NumPy fallback only)
Without Bottleneck installed:
1. **ffill/bfill**: ~25% of pandas (Cython vs pure NumPy)
2. **Rolling sum/mean**: 65-77% of pandas
3. **Rolling std**: 54-64% of pandas

**Recommendation**: Install Bottleneck for optimal performance: `pip install bottleneck`

## Fallback Behavior

When a kernel is unavailable:
1. **Strict mode** (`strict=True`): Raises `NotImplementedError`
2. **Normal mode**: Falls back to pandas DataFrame operations with warning

## Parallelism

The physical planner supports parallel execution at multiple levels:

### Parallel Expression Evaluation

When a projection has many expressions (≥8 by default), expressions are evaluated
in parallel using `ThreadPoolExecutor`. Each thread gets its own `ArrayEvaluator`
to avoid contention.

**Configuration**: `ExecutionContext.parallel_threshold` (default: 8)

### Parallel Join Sides

Left and right sides of a hash join are executed in parallel, allowing independent
subplans to overlap. This is particularly beneficial for complex joins where both
sides involve significant computation.

### Iterative Query Optimization

The optimizer runs passes iteratively (up to 3 iterations by default) until a
fixpoint is reached. This allows compound optimizations where one pass creates
opportunities for another.

### Scaling Behavior

| Dataset Size | Physical Planner Improvement |
|-------------|------------------------------|
| ~3M rows | ~1.0x (overhead dominates) |
| ~10M rows | ~1.4x |
| ~20M rows | ~1.7x |

**Note**: Parallelism benefits scale with data size. For small datasets (<3M rows),
the overhead may exceed the benefit.

## Lazy File Scanning

The `scan()` function enables lazy reading of files with optimization pushdown.

### Supported Formats

| Format | Status | Pushdown Support |
|--------|--------|------------------|
| Parquet | ✅ Implemented | Predicate (incl. row-group statistics) + Projection |
| CSV | ✅ Implemented | Projection |
| JSON | 🚧 Planned | Projection only |

### Path Types

| Path Type | Example | Notes |
|-----------|---------|-------|
| Local file | `scan("data.parquet")` | Single file |
| Glob pattern | `scan("data/*.parquet")` | Multiple files |
| S3 URL | `scan("s3://bucket/data.parquet")` | Requires fsspec |
| GCS URL | `scan("gs://bucket/data.parquet")` | Requires gcsfs |
| HTTPS URL | `scan("https://example.com/data.parquet")` | Direct download |

### Predicate Pushdown

Filters are converted to PyArrow compute expressions and pushed to the Parquet reader:

```python
from pandas.lazy import scan, col

# Filter is pushed to Parquet reader
ldf = scan("data.parquet").filter(col("value") > 100)
result = ldf.collect(use_physical_planner=True)
```

Supported predicates:
- Comparison: `>`, `>=`, `<`, `<=`, `==`, `!=`
- Logical: `&` (and), `|` (or), `~` (not)

### Projection Pushdown

Only columns actually used in the query are read from the file:

```python
# Only reads columns "a" and "b" from Parquet
ldf = scan("data.parquet").select("a", "b")
result = ldf.collect(use_physical_planner=True)
```

## Files Reference

| File | Description |
|------|-------------|
| `pandas/lazy/scan.py` | `scan()` function and format detection |
| `pandas/lazy/physical.py` | Physical operators |
| `pandas/lazy/backends/__init__.py` | Kernel registry and dispatch |
| `pandas/lazy/backends/_bottleneck.py` | Bottleneck integration and configuration |
| `pandas/lazy/backends/router.py` | Backend routing logic |
| `pandas/lazy/backends/array_eval.py` | Expression evaluation |
| `pandas/lazy/backends/numpy/core.py` | NumPy arithmetic/comparison kernels |
| `pandas/lazy/backends/numpy/string.py` | NumPy string kernels |
| `pandas/lazy/backends/numpy/datetime.py` | NumPy datetime kernels |
| `pandas/lazy/backends/numpy/groupby.py` | NumPy groupby kernels |
| `pandas/lazy/backends/numpy/join.py` | NumPy join kernel (hash_join) |
| `pandas/lazy/backends/numpy/misc.py` | Rolling, cumulative, fill kernels (with Bottleneck) |
| `pandas/lazy/backends/arrow/core.py` | Arrow arithmetic/comparison kernels |
| `pandas/lazy/backends/arrow/string.py` | Arrow string kernels |
| `pandas/lazy/backends/arrow/datetime.py` | Arrow datetime kernels |
| `pandas/lazy/backends/arrow/groupby.py` | Arrow groupby kernels |
| `pandas/lazy/backends/arrow/join.py` | Arrow join kernel |
