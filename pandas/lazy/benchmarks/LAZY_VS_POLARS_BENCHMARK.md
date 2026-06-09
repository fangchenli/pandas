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
| filter_project | 0.22x | 0.41x | 0.11x |
| aggregation | 1.16x | 3.30x | 0.34x |
| string | 2.09x | 4.86x | 0.37x |
| limit | 0.02x | 0.03x | 0.02x |
| sort | 0.42x | 0.47x | 0.32x |
| join | 0.35x | 0.42x | 0.28x |
| parquet_scan | 0.86x | 1.54x | 0.42x |
| engine_pipeline | 0.43x | 0.59x | 0.21x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 20.39 | 3.37 | *0.17x* |
| filter + select(3 cols) | 1,000,000 | 5.12 | 1.45 | *0.28x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.25 | 2.79 | *0.21x* |
| with_columns(value1 + value2) | 1,000,000 | 5.75 | 0.64 | *0.11x* |
| filter(value1 > 0) | 10,000,000 | 221.17 | 36.30 | *0.16x* |
| filter + select(3 cols) | 10,000,000 | 42.76 | 17.50 | *0.41x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 124.04 | 27.93 | *0.23x* |
| with_columns(value1 + value2) | 10,000,000 | 52.58 | 9.75 | *0.19x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 7.30 | 2.50 | *0.34x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 13.17 | 10.64 | *0.81x* |
| groupby(category[100]).sum() | 1,000,000 | 5.78 | 2.25 | *0.39x* |
| groupby(group).sum(value1) | 10,000,000 | 18.68 | 18.47 | *0.99x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 35.71 | 117.83 | **3.30x** |
| groupby(category[100]).sum() | 10,000,000 | 16.48 | 19.16 | **1.16x** |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 15.08 | 17.21 | **1.14x** |
| with_columns(text.lower()) | 1,000,000 | 13.50 | 58.26 | **4.31x** |
| with_columns(text.len()) | 1,000,000 | 12.18 | 4.54 | *0.37x* |
| filter(text.contains('foo')) | 10,000,000 | 119.10 | 175.23 | **1.47x** |
| with_columns(text.lower()) | 10,000,000 | 127.35 | 619.06 | **4.86x** |
| with_columns(text.len()) | 10,000,000 | 121.65 | 47.41 | *0.39x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.32 | 0.01 | *0.02x* |
| head(100) | 1,000,000 | 0.29 | 0.01 | *0.02x* |
| head(1000) | 1,000,000 | 0.29 | 0.01 | *0.03x* |
| head(10) | 10,000,000 | 0.30 | 0.01 | *0.02x* |
| head(100) | 10,000,000 | 0.28 | 0.01 | *0.02x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.03x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 70.85 | 33.22 | *0.47x* |
| sort(group, value1) | 1,000,000 | 257.67 | 83.12 | *0.32x* |
| sort(value1) | 10,000,000 | 1030.11 | 460.44 | *0.45x* |
| sort(group, value1) | 10,000,000 | 3187.63 | 1429.10 | *0.45x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 123.94 | 34.90 | *0.28x* |
| left_join(1000000x100000) | 1,000,000 | 102.16 | 42.61 | *0.42x* |
| inner_join(10000000x1000000) | 10,000,000 | 1802.02 | 554.65 | *0.31x* |
| left_join(10000000x1000000) | 10,000,000 | 2045.09 | 770.56 | *0.38x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 59.82 | 36.65 | *0.61x* |
| scan.select(3).filter | 2,964,624 | 21.36 | 8.94 | *0.42x* |
| glob_scan.head(1000) | 17,787,744 | 6.94 | 10.65 | **1.54x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.53 | 1.59 | *0.21x* |
| join -> groupby | 1,000,000 | 63.04 | 29.70 | *0.47x* |
| groupby(category-dtype key).sum | 10,000,000 | 20.61 | 9.23 | *0.45x* |
| join -> groupby | 10,000,000 | 775.84 | 460.17 | *0.59x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 4.86x faster
- **with_columns(text.lower())** (string): 4.31x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 3.30x faster
- **glob_scan.head(1000)** (parquet_scan): 1.54x faster
- **filter(text.contains('foo'))** (string): 1.47x faster

### Where Polars is Faster

- **head(10)** (limit): 44.35x faster
- **head(100)** (limit): 42.87x faster
- **head(100)** (limit): 41.67x faster
- **head(10)** (limit): 41.56x faster
- **head(1000)** (limit): 39.77x faster
