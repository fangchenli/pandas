# Lazy Pandas vs Polars Benchmark Report

## Environment

- pandas version: 3.1.0.dev0+1220.g55b305e82a
- polars version: 1.37.1
- Python version: 3.11.14
- Platform: macOS-26.5.1-arm64-arm-64bit
- Warmup runs: 3
- Timed runs: 7

## Methodology

Lazy pandas timings use the **physical engine** (`collect(use_physical_planner=True)`). Reports generated before June 2026 measured the eager evaluation path for every category except parquet scans and are not directly comparable. The test data is mixed-dtype (numeric + string columns), so conversion boundaries are part of what is measured. Absolute timings vary with machine load; treat ratios as the signal.

## Summary

**Speedup interpretation:** Values > 1.0 mean lazy pandas is faster, < 1.0 mean Polars is faster.

### Category Summary

| Category | Avg Speedup | Best LP | Best Polars |
|----------|-------------|---------|-------------|
| filter_project | 0.21x | 0.38x | 0.07x |
| aggregation | 1.14x | 3.36x | 0.35x |
| string | 1.66x | 4.52x | 0.37x |
| limit | 0.03x | 0.04x | 0.02x |
| sort | 0.74x | 1.28x | 0.41x |
| join | 0.35x | 0.46x | 0.29x |
| parquet_scan | 0.90x | 1.64x | 0.43x |
| engine_pipeline | 0.44x | 0.59x | 0.20x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 22.07 | 3.37 | *0.15x* |
| filter + select(3 cols) | 1,000,000 | 5.25 | 1.50 | *0.29x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 14.48 | 2.80 | *0.19x* |
| with_columns(value1 + value2) | 1,000,000 | 8.03 | 0.53 | *0.07x* |
| filter(value1 > 0) | 10,000,000 | 264.12 | 40.92 | *0.15x* |
| filter + select(3 cols) | 10,000,000 | 43.54 | 16.50 | *0.38x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 122.08 | 29.49 | *0.24x* |
| with_columns(value1 + value2) | 10,000,000 | 55.72 | 10.40 | *0.19x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 7.20 | 2.52 | *0.35x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 13.04 | 11.02 | *0.85x* |
| groupby(category[100]).sum() | 1,000,000 | 5.68 | 2.68 | *0.47x* |
| groupby(group).sum(value1) | 10,000,000 | 22.58 | 20.72 | *0.92x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 35.25 | 118.29 | **3.36x** |
| groupby(category[100]).sum() | 10,000,000 | 16.89 | 15.36 | *0.91x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 14.74 | 17.54 | **1.19x** |
| with_columns(text.lower()) | 1,000,000 | 14.15 | 63.91 | **4.52x** |
| with_columns(text.len()) | 1,000,000 | 12.09 | 4.48 | *0.37x* |
| filter(text.contains('foo')) | 10,000,000 | 117.70 | 182.70 | **1.55x** |
| with_columns(text.lower()) | 10,000,000 | 321.97 | 629.51 | **1.96x** |
| with_columns(text.len()) | 10,000,000 | 126.09 | 47.19 | *0.37x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.30 | 0.01 | *0.03x* |
| head(100) | 1,000,000 | 0.28 | 0.01 | *0.02x* |
| head(1000) | 1,000,000 | 0.29 | 0.01 | *0.03x* |
| head(10) | 10,000,000 | 0.28 | 0.01 | *0.04x* |
| head(100) | 10,000,000 | 0.28 | 0.01 | *0.03x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.03x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 72.28 | 34.56 | *0.48x* |
| sort(group[str], value1) | 1,000,000 | 121.28 | 88.08 | *0.73x* |
| sort(int_val, value1) | 1,000,000 | 83.10 | 45.33 | *0.55x* |
| sort(num_a) [numeric-only] | 1,000,000 | 29.40 | 23.49 | *0.80x* |
| sort(key_int, num_a) [numeric-only] | 1,000,000 | 46.82 | 33.89 | *0.72x* |
| sort(value1) | 10,000,000 | 1147.55 | 476.22 | *0.41x* |
| sort(group[str], value1) | 10,000,000 | 1614.73 | 2066.68 | **1.28x** |
| sort(int_val, value1) | 10,000,000 | 1526.73 | 688.02 | *0.45x* |
| sort(num_a) [numeric-only] | 10,000,000 | 412.61 | 443.59 | **1.08x** |
| sort(key_int, num_a) [numeric-only] | 10,000,000 | 912.66 | 784.92 | *0.86x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 121.04 | 35.59 | *0.29x* |
| left_join(1000000x100000) | 1,000,000 | 94.65 | 43.22 | *0.46x* |
| inner_join(10000000x1000000) | 10,000,000 | 2499.14 | 804.47 | *0.32x* |
| left_join(10000000x1000000) | 10,000,000 | 2232.65 | 732.14 | *0.33x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 61.56 | 39.08 | *0.63x* |
| scan.select(3).filter | 2,964,624 | 19.19 | 8.19 | *0.43x* |
| glob_scan.head(1000) | 17,787,744 | 6.69 | 10.99 | **1.64x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.59 | 1.49 | *0.20x* |
| join -> groupby | 1,000,000 | 55.07 | 32.25 | *0.59x* |
| groupby(category-dtype key).sum | 10,000,000 | 22.08 | 9.71 | *0.44x* |
| join -> groupby | 10,000,000 | 885.07 | 471.18 | *0.53x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 4.52x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 3.36x faster
- **with_columns(text.lower())** (string): 1.96x faster
- **glob_scan.head(1000)** (parquet_scan): 1.64x faster
- **filter(text.contains('foo'))** (string): 1.55x faster

### Where Polars is Faster

- **head(100)** (limit): 42.07x faster
- **head(10)** (limit): 39.90x faster
- **head(1000)** (limit): 39.49x faster
- **head(1000)** (limit): 38.61x faster
- **head(100)** (limit): 30.20x faster
