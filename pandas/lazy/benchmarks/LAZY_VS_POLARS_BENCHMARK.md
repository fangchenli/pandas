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
| filter_project | 0.21x | 0.40x | 0.09x |
| aggregation | 1.16x | 3.35x | 0.40x |
| string | 2.14x | 5.30x | 0.40x |
| limit | 0.03x | 0.03x | 0.02x |
| sort | 0.75x | 1.09x | 0.53x |
| join | 0.37x | 0.51x | 0.29x |
| parquet_scan | 0.88x | 1.52x | 0.42x |
| engine_pipeline | 0.43x | 0.56x | 0.19x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 19.95 | 3.53 | *0.18x* |
| filter + select(3 cols) | 1,000,000 | 5.10 | 1.48 | *0.29x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.66 | 2.74 | *0.20x* |
| with_columns(value1 + value2) | 1,000,000 | 5.49 | 0.51 | *0.09x* |
| filter(value1 > 0) | 10,000,000 | 215.06 | 32.64 | *0.15x* |
| filter + select(3 cols) | 10,000,000 | 42.26 | 16.74 | *0.40x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 124.11 | 26.72 | *0.22x* |
| with_columns(value1 + value2) | 10,000,000 | 49.39 | 9.41 | *0.19x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 7.41 | 2.93 | *0.40x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 13.24 | 11.16 | *0.84x* |
| groupby(category[100]).sum() | 1,000,000 | 6.30 | 3.04 | *0.48x* |
| groupby(group).sum(value1) | 10,000,000 | 19.11 | 18.28 | *0.96x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 34.69 | 116.37 | **3.35x** |
| groupby(category[100]).sum() | 10,000,000 | 16.53 | 15.66 | *0.95x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 14.95 | 17.35 | **1.16x** |
| with_columns(text.lower()) | 1,000,000 | 14.38 | 59.42 | **4.13x** |
| with_columns(text.len()) | 1,000,000 | 11.83 | 4.75 | *0.40x* |
| filter(text.contains('foo')) | 10,000,000 | 121.95 | 174.64 | **1.43x** |
| with_columns(text.lower()) | 10,000,000 | 117.85 | 624.46 | **5.30x** |
| with_columns(text.len()) | 10,000,000 | 112.28 | 48.06 | *0.43x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.31 | 0.01 | *0.02x* |
| head(100) | 1,000,000 | 0.31 | 0.01 | *0.02x* |
| head(1000) | 1,000,000 | 0.30 | 0.01 | *0.03x* |
| head(10) | 10,000,000 | 0.30 | 0.01 | *0.03x* |
| head(100) | 10,000,000 | 0.28 | 0.01 | *0.03x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.03x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 67.32 | 35.75 | *0.53x* |
| sort(group[str], value1) | 1,000,000 | 115.48 | 88.30 | *0.76x* |
| sort(int_val, value1) | 1,000,000 | 86.93 | 52.07 | *0.60x* |
| sort(num_a) [numeric-only] | 1,000,000 | 30.30 | 22.56 | *0.74x* |
| sort(key_int, num_a) [numeric-only] | 1,000,000 | 45.14 | 34.81 | *0.77x* |
| sort(value1) | 10,000,000 | 868.84 | 459.04 | *0.53x* |
| sort(group[str], value1) | 10,000,000 | 1361.64 | 1487.55 | **1.09x** |
| sort(int_val, value1) | 10,000,000 | 1073.45 | 651.96 | *0.61x* |
| sort(num_a) [numeric-only] | 10,000,000 | 312.83 | 304.51 | *0.97x* |
| sort(key_int, num_a) [numeric-only] | 10,000,000 | 548.78 | 499.87 | *0.91x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 91.65 | 35.60 | *0.39x* |
| left_join(1000000x100000) | 1,000,000 | 99.91 | 50.80 | *0.51x* |
| inner_join(10000000x1000000) | 10,000,000 | 1880.29 | 545.24 | *0.29x* |
| left_join(10000000x1000000) | 10,000,000 | 2125.50 | 650.79 | *0.31x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 52.68 | 36.08 | *0.68x* |
| scan.select(3).filter | 2,964,624 | 18.71 | 7.81 | *0.42x* |
| glob_scan.head(1000) | 17,787,744 | 6.91 | 10.52 | **1.52x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.10 | 1.36 | *0.19x* |
| join -> groupby | 1,000,000 | 53.90 | 30.17 | *0.56x* |
| groupby(category-dtype key).sum | 10,000,000 | 21.91 | 9.51 | *0.43x* |
| join -> groupby | 10,000,000 | 859.97 | 448.86 | *0.52x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 5.30x faster
- **with_columns(text.lower())** (string): 4.13x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 3.35x faster
- **glob_scan.head(1000)** (parquet_scan): 1.52x faster
- **filter(text.contains('foo'))** (string): 1.43x faster

### Where Polars is Faster

- **head(100)** (limit): 43.81x faster
- **head(10)** (limit): 40.40x faster
- **head(100)** (limit): 39.57x faster
- **head(1000)** (limit): 38.90x faster
- **head(10)** (limit): 38.31x faster
