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
| filter_project | 0.23x | 0.39x | 0.09x |
| aggregation | 1.13x | 3.29x | 0.38x |
| string | 2.13x | 5.27x | 0.34x |
| limit | 0.03x | 0.03x | 0.02x |
| sort | 0.60x | 0.90x | 0.32x |
| join | 0.33x | 0.41x | 0.29x |
| parquet_scan | 0.83x | 1.55x | 0.38x |
| engine_pipeline | 0.44x | 0.62x | 0.21x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 21.12 | 6.49 | *0.31x* |
| filter + select(3 cols) | 1,000,000 | 6.25 | 1.55 | *0.25x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 14.04 | 2.95 | *0.21x* |
| with_columns(value1 + value2) | 1,000,000 | 5.90 | 0.51 | *0.09x* |
| filter(value1 > 0) | 10,000,000 | 215.93 | 34.52 | *0.16x* |
| filter + select(3 cols) | 10,000,000 | 42.82 | 16.71 | *0.39x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 122.03 | 27.04 | *0.22x* |
| with_columns(value1 + value2) | 10,000,000 | 49.31 | 10.32 | *0.21x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 7.03 | 2.75 | *0.39x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 13.09 | 10.07 | *0.77x* |
| groupby(category[100]).sum() | 1,000,000 | 5.87 | 2.21 | *0.38x* |
| groupby(group).sum(value1) | 10,000,000 | 18.83 | 18.65 | *0.99x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 34.46 | 113.42 | **3.29x** |
| groupby(category[100]).sum() | 10,000,000 | 16.46 | 15.37 | *0.93x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 15.32 | 17.84 | **1.16x** |
| with_columns(text.lower()) | 1,000,000 | 13.91 | 57.20 | **4.11x** |
| with_columns(text.len()) | 1,000,000 | 13.32 | 4.55 | *0.34x* |
| filter(text.contains('foo')) | 10,000,000 | 117.58 | 175.83 | **1.50x** |
| with_columns(text.lower()) | 10,000,000 | 122.57 | 645.52 | **5.27x** |
| with_columns(text.len()) | 10,000,000 | 117.13 | 47.77 | *0.41x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.31 | 0.01 | *0.03x* |
| head(100) | 1,000,000 | 0.28 | 0.01 | *0.03x* |
| head(1000) | 1,000,000 | 0.30 | 0.01 | *0.02x* |
| head(10) | 10,000,000 | 0.28 | 0.01 | *0.03x* |
| head(100) | 10,000,000 | 0.28 | 0.01 | *0.02x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.03x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 67.39 | 35.10 | *0.52x* |
| sort(group[str], value1) | 1,000,000 | 253.81 | 81.67 | *0.32x* |
| sort(int_val, value1) | 1,000,000 | 90.73 | 45.50 | *0.50x* |
| sort(num_a) [numeric-only] | 1,000,000 | 32.34 | 23.38 | *0.72x* |
| sort(key_int, num_a) [numeric-only] | 1,000,000 | 48.29 | 34.18 | *0.71x* |
| sort(value1) | 10,000,000 | 936.41 | 462.80 | *0.49x* |
| sort(group[str], value1) | 10,000,000 | 3150.13 | 1454.44 | *0.46x* |
| sort(int_val, value1) | 10,000,000 | 1089.08 | 628.11 | *0.58x* |
| sort(num_a) [numeric-only] | 10,000,000 | 375.48 | 308.89 | *0.82x* |
| sort(key_int, num_a) [numeric-only] | 10,000,000 | 565.84 | 511.67 | *0.90x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 120.35 | 34.59 | *0.29x* |
| left_join(1000000x100000) | 1,000,000 | 102.55 | 41.90 | *0.41x* |
| inner_join(10000000x1000000) | 10,000,000 | 1699.29 | 561.80 | *0.33x* |
| left_join(10000000x1000000) | 10,000,000 | 2205.61 | 669.16 | *0.30x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 62.01 | 34.51 | *0.56x* |
| scan.select(3).filter | 2,964,624 | 20.53 | 7.77 | *0.38x* |
| glob_scan.head(1000) | 17,787,744 | 6.78 | 10.49 | **1.55x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.24 | 1.49 | *0.21x* |
| join -> groupby | 1,000,000 | 62.91 | 29.72 | *0.47x* |
| groupby(category-dtype key).sum | 10,000,000 | 21.68 | 9.96 | *0.46x* |
| join -> groupby | 10,000,000 | 776.28 | 477.93 | *0.62x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 5.27x faster
- **with_columns(text.lower())** (string): 4.11x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 3.29x faster
- **glob_scan.head(1000)** (parquet_scan): 1.55x faster
- **filter(text.contains('foo'))** (string): 1.50x faster

### Where Polars is Faster

- **head(100)** (limit): 43.14x faster
- **head(1000)** (limit): 40.76x faster
- **head(10)** (limit): 39.83x faster
- **head(100)** (limit): 39.30x faster
- **head(1000)** (limit): 39.05x faster
