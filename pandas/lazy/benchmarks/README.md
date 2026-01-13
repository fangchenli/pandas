# Lazy Pandas Benchmarks

This directory contains standalone benchmarks comparing lazy vs eager execution
and Arrow vs NumPy backends.

## Running Benchmarks

```bash
# Run all benchmarks
python run_all.py

# Run specific benchmark
python bench_filter.py
python bench_select.py
python bench_arithmetic.py
python bench_string_ops.py
python bench_expressions.py
python bench_aggregations.py
python bench_pipelines.py
python bench_kernel_overhead.py
python bench_conversion.py
```

## Benchmark Suite

| Benchmark | Description |
|-----------|-------------|
| `bench_kernel_overhead.py` | Kernel dispatch overhead vs direct calls |
| `bench_conversion.py` | Arrow ↔ NumPy conversion costs |
| `bench_filter.py` | Filter and chained filter operations |
| `bench_select.py` | Column selection and computed columns |
| `bench_arithmetic.py` | Numeric arithmetic expressions |
| `bench_string_ops.py` | String operations (lower, contains, replace) |
| `bench_expressions.py` | Complex nested expressions, polynomials, case-when |
| `bench_aggregations.py` | Filter/compute pipelines with multiple steps |
| `bench_pipelines.py` | Realistic multi-step ETL-like query patterns |

## Key Findings

### Kernel Dispatch Overhead
- ~0.1-0.5μs per call - negligible for arrays > 10K elements

### Array Conversion Cost
- Float/Int: Near zero (zero-copy possible)
- Strings: Expensive (14ms for 1M elements NumPy→Arrow, 28ms Arrow→NumPy)

### Where Lazy Execution Wins

1. **Sequential Filters (Filter Fusion)**
   - 5 sequential filters at 1M rows: Lazy Arrow 12.7ms vs Eager Arrow 22ms (**1.7x faster**)
   - Filter fusion combines predicates into single pass

2. **Multi-filter Pipelines with Compute**
   - 3 filters + compute at 1M rows: Lazy Arrow 17.6ms vs Eager Arrow 21.4ms

3. **Large Arrow Data**
   - Arrow backend has lower overhead in lazy path
   - Lazy Arrow often competitive with eager Arrow at scale

### Where Eager Execution Wins

1. **Simple Operations**
   - Single filter/select: Eager is 2-5x faster due to no planning overhead
   - Simple arithmetic: Eager NumPy beats lazy

2. **Small Datasets**
   - Fixed overhead dominates under ~10K rows

3. **NumPy Data Paths**
   - Lazy overhead higher for NumPy conversion

### Performance at Scale (1M rows)

| Operation | Eager NumPy | Lazy NumPy | Eager Arrow | Lazy Arrow |
|-----------|-------------|------------|-------------|------------|
| 5 sequential filters | 22.8ms | 18.9ms | 22.0ms | **12.7ms** |
| Complex OR filter | 28.2ms | 37.2ms | 26.4ms | 29.3ms |
| Filter→Compute→Filter | 23.4ms | 35.5ms | 23.9ms | 25.8ms |
| 5 computed columns | 9.8ms | 70.0ms | 13.4ms | 51.7ms |
| Deeply nested arithmetic | 16.2ms | 48.5ms | 29.4ms | 30.3ms |
| Complex boolean expr | 10.2ms | 26.3ms | **3.0ms** | 6.1ms |

### Recommendations

1. **Use Arrow-backed dtypes** with lazy for best performance
2. **Batch operations** in lazy mode to amortize planning overhead
3. **Use lazy for**:
   - Multiple chained filters (fusion benefit)
   - Complex pipelines with filter+compute
   - Arrow data at scale
4. **Use eager for**:
   - Simple single operations
   - Small datasets
   - NumPy data with simple ops

## Notes

- Benchmarks use `np.random.default_rng(42)` for reproducible data
- Times are mean ± std over 5 runs with 1 warmup
- Tested on Apple Silicon Mac with pandas development build
