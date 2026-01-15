# Lazy Pandas Optimization: Next Steps

## Current Performance Gap Analysis

### Benchmark Results (NYC Taxi, ~3M rows)
- **Polars**: ~210 ms
- **Lazy Pandas (physical)**: ~430 ms (2.0x slower)
- **Eager Pandas**: ~550 ms

### Detailed Benchmarking Results

#### GroupBy Performance (Arrow vs Pandas)

| Rows | Groups | Dtype | Pandas (ms) | Arrow (ms) | Winner | Speedup |
|------|--------|-------|-------------|------------|--------|---------|
| 10K | 10 | int64 | 0.97 | 13.19 | **Pandas** | 13.5x |
| 10K | 10 | float64 | 0.32 | 0.34 | **Pandas** | 1.1x |
| 10K | 100 | int64 | 0.20 | 0.13 | Arrow | 1.6x |
| 10K | 100 | float64 | 0.20 | 0.12 | Arrow | 1.7x |
| 50K | 10 | float64 | 0.51 | 0.35 | Arrow | 1.4x |
| 100K | 10 | float64 | 0.95 | 0.65 | Arrow | 1.5x |
| 100K | 100 | float64 | 1.51 | 0.66 | Arrow | 2.3x |
| 500K | 1000 | float64 | 5.70 | 2.80 | Arrow | 2.0x |
| 1M | 1000 | float64 | 14.8 | 5.6 | Arrow | 2.6x |

**Key Finding**: Arrow groupby is faster in almost all cases. **Pandas Cython wins only for very small data (<10K rows) with few groups (<100) and int64 dtype**. This is due to Arrow's JIT/setup overhead dominating at tiny scales.

#### Filter Performance (Arrow vs NumPy)

| Scenario | NumPy | Arrow | Speedup |
|----------|-------|-------|---------|
| 1M rows, 50% selectivity | 5.2ms | 3.3ms | 1.6x |
| 3M rows, 10% selectivity | 8.4ms | 5.3ms | 1.6x |
| 3M rows, 3 conditions | 31.2ms | 13.5ms | **2.3x** |

**Key Finding**: Arrow is consistently faster for filters, especially with multiple conditions.

### When to Use Each Backend

| Operation | Use Arrow When | Use NumPy/Pandas When |
|-----------|---------------|----------------------|
| GroupBy | Almost always (>10K rows OR >100 groups) | <10K rows AND <100 groups AND int64 |
| Filter | Multiple conditions OR >1M rows | Simple single condition |
| Arithmetic | String/datetime involved | Pure numeric operations |

**Simplified Rule for GroupBy**: Use Arrow by default. The Cython path only wins in a narrow corner case (tiny data, few groups, integer dtype) that's rarely encountered in real workloads.

## Proposed Optimizations

### Phase 1: Smart Backend Selection (High Impact)

**Goal**: Choose optimal backend based on data characteristics, not just operation type.

1. **Cost-Based Backend Selection**
   - Track row count and cardinality estimates through the plan
   - Choose Arrow for groupby when: rows > 100K OR groups > 100
   - Choose Arrow for filters when: multiple conditions OR rows > 1M
   - Add heuristics to `EngineSelection` pass

2. **Arrow GroupBy with Thresholds**
   - Current: Always uses pandas groupby via DataFrame fallback
   - Proposed: Use `pyarrow.Table.group_by()` for large data
   - **Key**: Only when row_count > 100K to avoid Arrow overhead
   - Expected speedup: 5-10x for large datasets

3. **Filter Fusion Before Execution**
   - Current: Multiple filter nodes executed sequentially
   - Proposed: Fuse consecutive filters into single compound predicate
   - Use Arrow's `pc.filter()` with complex expression trees
   - Expected speedup: 1.5-2x for multi-filter queries

### Phase 2: Reduce Overhead (Medium Impact)

4. **Lazy Column Extraction**
   - Current: `PhysicalScan` extracts ALL columns from DataFrame
   - Proposed: Only extract columns that will actually be used
   - Requires: Propagate "needed columns" from ProjectionPruning to PhysicalScan

5. **Eliminate DataFrame Round-trips**
   - Current: Some operations (join, groupby) convert to DataFrame
   - Proposed: Stay in array format, only convert at final result
   - Identify and fix all DataFrame construction points

6. **Row Count Propagation**
   - Add `estimated_rows` to Schema or plan nodes
   - Use for backend selection decisions
   - Update after filter operations (selectivity estimation)

### Phase 3: Advanced (Lower Priority)

7. **Predicate Pushdown to Parquet Reader**
   - If source is Parquet file, push filters to PyArrow reader
   - Avoid reading unnecessary row groups

8. **Streaming/Chunked Execution**
   - Process data in chunks for very large datasets
   - Reduces peak memory usage

## Implementation Plan

### Step 1: Arrow GroupBy Integration (Highest ROI)
- Add row count threshold check (>100K rows)
- Implement direct Arrow groupby path in `PhysicalHashAggregate`
- Expected: 5-10x speedup for large groupby operations

### Step 2: Filter Fusion at Physical Level
- Fuse consecutive `PhysicalFilter` nodes
- Use single Arrow `pc.filter()` call with compound expression
- Expected: 1.5-2x speedup for multi-filter queries

### Step 3: Lazy Column Extraction
- Modify `PhysicalScan` to accept needed columns
- Thread column requirements from optimizer
- Expected: Reduced memory and extraction time

### Step 4: Cost-Based Engine Selection
- Add row count estimates to plan nodes
- Update `EngineSelection` to use estimates
- Dynamic threshold-based backend selection

## Current Results (After Arrow-Backed DataFrame Output)

| Operation | Data Size | Physical vs Pandas | Notes |
|-----------|-----------|-------------------|-------|
| GroupBy | 3M rows | **1.04x** (similar) | Scales well with data size |
| GroupBy | 1M rows | **0.74x** (35% faster) | NumPy kernel is efficient |
| Filter | 3M rows | 1.25-1.46x slower | Improved with Arrow-backed output |
| Filter + GroupBy | 3M rows | 1.15x slower | Benefits from both optimizations |

### Key Insights

1. **GroupBy is faster** than pandas due to efficient NumPy kernel dispatch
2. **Filter overhead reduced** by using Arrow-backed DataFrame output
3. **Arrow-backed DataFrame** conversion is ~15x faster than NumPy-backed (0.23ms vs 3.43ms for 800K rows)
4. Output dtype changes: `float64` → `double[pyarrow]`, `int64` → `int64[pyarrow]`

### Implemented Optimizations

1. ✅ **Arrow filter path for large datasets** (>50K rows)
   - Converts NumPy arrays to Arrow for filtering
   - Uses `pa.table.filter()` for batch filtering
   - Arrow filter is ~30% faster than NumPy for multi-condition filters

2. ✅ **Filter fusion at logical level**
   - Multiple consecutive filters are fused into single compound predicate
   - Reduces plan traversal overhead

3. ✅ **Arrow-backed DataFrame output** (NEW)
   - Uses `table.to_pandas(types_mapper=pd.ArrowDtype)` for near-zero-copy conversion
   - Reduces DataFrame conversion time from ~3.4ms to ~0.2ms for 800K rows (~15-18x faster)
   - Output columns use Arrow-backed dtypes (`double[pyarrow]`, `int64[pyarrow]`)
   - Controlled by `use_arrow_dtype` parameter in `arrays_to_dataframe()`

#### Why Zero-Copy Works

When using `types_mapper=pd.ArrowDtype`:
1. pandas creates an `ArrowExtensionArray` that **wraps** the existing PyArrow array
2. `ArrowExtensionArray` stores a reference to the `ChunkedArray` in `_pa_array`
3. The underlying Arrow memory buffers are **shared, not copied**

When using default `to_pandas()`:
1. Must allocate new NumPy arrays with contiguous memory
2. Must convert Arrow null bitmaps to `np.nan` or `pd.NA`
3. Must copy all values from Arrow buffers to NumPy buffers

**Verification** (buffer addresses match = zero-copy):
```python
arr = pa.array([1.0, 2.0, 3.0])
table = pa.table({"a": arr})
df = table.to_pandas(types_mapper=pd.ArrowDtype)

# Get buffer addresses
original_addr = arr.buffers()[1].address
pandas_addr = df['a'].array._pa_array.chunk(0).buffers()[1].address
assert original_addr == pandas_addr  # Same memory!
```

**References**:
- [Arrow Python Pandas Integration](https://arrow.apache.org/docs/python/pandas.html)
- [pandas PyArrow Functionality](https://pandas.pydata.org/docs/user_guide/pyarrow.html)

4. ✅ **Bottleneck acceleration for rolling/fill operations** (NEW)
   - Integrated Bottleneck library for C-optimized rolling window functions
   - Uses `bn.move_sum`, `bn.move_mean`, `bn.move_std`, etc. when available
   - Uses `bn.push` for ffill/bfill operations
   - Falls back to pure NumPy when Bottleneck not installed
   - Controlled by `pd.set_option("compute.use_bottleneck", True/False)`
   - Configuration module: `pandas/lazy/backends/_bottleneck.py`

5. ✅ **Lazy Parquet scanning with predicate/projection pushdown** (NEW)
   - Added `scan()` function for lazy file reading
   - `ParquetSource` logical plan node with predicate and projection fields
   - `PhysicalParquetScan` converts predicates to PyArrow compute expressions
   - Supports local paths, glob patterns (`*.parquet`), and URLs (s3://, gs://)
   - Predicate pushdown enables row group filtering in Parquet
   - Projection pushdown reads only required columns

6. ✅ **Advanced expression simplification** (NEW)
   - **De Morgan's Laws**: `~(a & b) → ~a | ~b`, `~(a | b) → ~a & ~b`
   - **Self-cancellation**: `x - x → 0`, `x / x → 1`
   - **Comparison simplifications**: `x == x → True`, `x != x → False`, `x < x → False`, etc.
   - **Logical idempotence**: `x & x → x`, `x | x → x`
   - Enables more aggressive constant folding and dead code elimination
   - Located in `ExpressionSimplification` pass in `pandas/lazy/optimize/passes.py`

7. ✅ **Threshold Configuration System** (NEW)
   - Centralized `ThresholdConfig` dataclass for all execution thresholds
   - Supports loading/saving to JSON for hardware-specific calibration
   - Integrates with `ExecutionContext` for runtime access
   - Configuration module: `pandas/lazy/optimize/config.py`
   - Threshold catalog: `pandas/lazy/optimize/THRESHOLD_CATALOG.md`

8. ✅ **Calibration Benchmark Suite** (NEW)
   - Script to determine optimal thresholds for current hardware
   - Benchmarks: filter backend, groupby backend, numexpr fusion, parallelization
   - Outputs JSON consumable by `ThresholdConfig.from_file()`
   - Script: `scripts/calibrate_lazy_thresholds.py`

9. ✅ **Pandas Options Integration** (NEW)
   - Lazy thresholds exposed via `pd.set_option("compute.lazy.*")`
   - Available options:
     - `compute.lazy.filter_arrow_threshold` (default: 50,000)
     - `compute.lazy.groupby_arrow_row_threshold` (default: 100,000)
     - `compute.lazy.groupby_arrow_cardinality_threshold` (default: 100)
     - `compute.lazy.parallel_expr_threshold` (default: 8)
     - `compute.lazy.numexpr_min_elements` (default: 100,000)
   - Supports `pd.option_context()` for temporary changes
   - Attribute-style access: `pd.options.compute.lazy.filter_arrow_threshold`

### Next Optimization Opportunities

1. **Streaming execution** for very large datasets
2. **Row group statistics** for smarter predicate pushdown in Parquet
3. **Dynamic threshold adjustment** based on runtime statistics

## Expected Results

| Metric | Current | After Phase 1 | After All |
|--------|---------|---------------|-----------|
| Gap to Polars | 2.0x | 1.4-1.6x | 1.2-1.4x |
| vs Eager Pandas | 1.3x | 3-4x | 4-5x |

## Decision Matrix

| Rows | Groups | Filters | Best Backend |
|------|--------|---------|--------------|
| <100K | <100 | Few | NumPy/Pandas |
| <100K | >100 | Few | Arrow |
| >100K | Any | Few | Arrow |
| Any | Any | Many (>3) | Arrow |

## Files to Modify

1. `pandas/lazy/physical.py` - PhysicalHashAggregate (Arrow path), PhysicalFilter (fusion)
2. `pandas/lazy/backends/arrow/groupby.py` - Direct Arrow groupby
3. `pandas/lazy/optimize/engine.py` - Cost-based selection with row estimates
4. `pandas/lazy/types.py` - Add row count estimate to Schema
