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
| filter_project | 0.21x | 0.37x | 0.08x |
| aggregation | 0.34x | 1.19x | 0.07x |
| string | 2.18x | 5.25x | 0.42x |
| limit | 0.02x | 0.03x | 0.02x |
| sort | 0.42x | 0.47x | 0.37x |
| join | 0.29x | 0.35x | 0.20x |
| parquet_scan | 0.26x | 0.62x | 0.07x |
| engine_pipeline | 0.42x | 0.58x | 0.20x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 18.21 | 3.42 | *0.19x* |
| filter + select(3 cols) | 1,000,000 | 5.16 | 1.44 | *0.28x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 12.49 | 2.70 | *0.22x* |
| with_columns(value1 + value2) | 1,000,000 | 6.20 | 0.48 | *0.08x* |
| filter(value1 > 0) | 10,000,000 | 202.70 | 33.39 | *0.16x* |
| filter + select(3 cols) | 10,000,000 | 44.18 | 16.44 | *0.37x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 114.97 | 27.69 | *0.24x* |
| with_columns(value1 + value2) | 10,000,000 | 56.53 | 9.42 | *0.17x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 31.50 | 2.77 | *0.09x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 35.90 | 11.38 | *0.32x* |
| groupby(category[100]).sum() | 1,000,000 | 35.12 | 2.37 | *0.07x* |
| groupby(group).sum(value1) | 10,000,000 | 79.38 | 18.07 | *0.23x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 97.28 | 115.70 | **1.19x** |
| groupby(category[100]).sum() | 10,000,000 | 90.79 | 16.09 | *0.18x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 13.20 | 18.52 | **1.40x** |
| with_columns(text.lower()) | 1,000,000 | 14.44 | 54.52 | **3.77x** |
| with_columns(text.len()) | 1,000,000 | 10.86 | 4.59 | *0.42x* |
| filter(text.contains('foo')) | 10,000,000 | 102.35 | 176.46 | **1.72x** |
| with_columns(text.lower()) | 10,000,000 | 121.86 | 639.78 | **5.25x** |
| with_columns(text.len()) | 10,000,000 | 97.63 | 49.13 | *0.50x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.31 | 0.01 | *0.02x* |
| head(100) | 1,000,000 | 0.30 | 0.01 | *0.02x* |
| head(1000) | 1,000,000 | 0.30 | 0.01 | *0.02x* |
| head(10) | 10,000,000 | 0.29 | 0.01 | *0.02x* |
| head(100) | 10,000,000 | 0.33 | 0.01 | *0.02x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.03x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 83.22 | 34.11 | *0.41x* |
| sort(group, value1) | 1,000,000 | 253.56 | 94.65 | *0.37x* |
| sort(value1) | 10,000,000 | 1057.05 | 459.67 | *0.43x* |
| sort(group, value1) | 10,000,000 | 3110.95 | 1471.92 | *0.47x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 131.24 | 39.29 | *0.30x* |
| left_join(1000000x100000) | 1,000,000 | 125.91 | 43.86 | *0.35x* |
| inner_join(10000000x1000000) | 10,000,000 | 2491.02 | 777.28 | *0.31x* |
| left_join(10000000x1000000) | 10,000,000 | 3223.64 | 635.97 | *0.20x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 55.38 | 34.56 | *0.62x* |
| scan.select(3).filter | 2,964,624 | 83.90 | 7.87 | *0.09x* |
| glob_scan.head(1000) | 17,787,744 | 145.13 | 10.52 | *0.07x* |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.22 | 1.48 | *0.20x* |
| join -> groupby | 1,000,000 | 63.82 | 30.06 | *0.47x* |
| groupby(category-dtype key).sum | 10,000,000 | 21.52 | 9.26 | *0.43x* |
| join -> groupby | 10,000,000 | 761.22 | 443.06 | *0.58x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 5.25x faster
- **with_columns(text.lower())** (string): 3.77x faster
- **filter(text.contains('foo'))** (string): 1.72x faster
- **filter(text.contains('foo'))** (string): 1.40x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 1.19x faster

### Where Polars is Faster

- **head(100)** (limit): 48.30x faster
- **head(100)** (limit): 43.13x faster
- **head(10)** (limit): 42.62x faster
- **head(10)** (limit): 42.19x faster
- **head(1000)** (limit): 40.82x faster
