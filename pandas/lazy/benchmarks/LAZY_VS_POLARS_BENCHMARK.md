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
| filter_project | 0.23x | 0.39x | 0.08x |
| aggregation | 1.17x | 3.47x | 0.35x |
| string | 2.07x | 4.38x | 0.38x |
| limit | 0.03x | 0.03x | 0.02x |
| sort | 0.39x | 0.47x | 0.33x |
| join | 0.22x | 0.32x | 0.16x |
| parquet_scan | 0.86x | 1.55x | 0.42x |
| engine_pipeline | 0.42x | 0.58x | 0.19x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 18.16 | 3.95 | *0.22x* |
| filter + select(3 cols) | 1,000,000 | 5.72 | 1.62 | *0.28x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 12.72 | 2.85 | *0.22x* |
| with_columns(value1 + value2) | 1,000,000 | 6.72 | 0.51 | *0.08x* |
| filter(value1 > 0) | 10,000,000 | 202.52 | 37.67 | *0.19x* |
| filter + select(3 cols) | 10,000,000 | 44.41 | 17.14 | *0.39x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 116.67 | 28.44 | *0.24x* |
| with_columns(value1 + value2) | 10,000,000 | 61.42 | 12.28 | *0.20x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 6.99 | 2.43 | *0.35x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 13.54 | 10.92 | *0.81x* |
| groupby(category[100]).sum() | 1,000,000 | 5.74 | 2.31 | *0.40x* |
| groupby(group).sum(value1) | 10,000,000 | 21.96 | 22.51 | **1.02x** |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 35.83 | 124.49 | **3.47x** |
| groupby(category[100]).sum() | 10,000,000 | 16.59 | 15.58 | *0.94x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 13.03 | 17.36 | **1.33x** |
| with_columns(text.lower()) | 1,000,000 | 15.06 | 63.56 | **4.22x** |
| with_columns(text.len()) | 1,000,000 | 11.44 | 4.49 | *0.39x* |
| filter(text.contains('foo')) | 10,000,000 | 102.33 | 175.44 | **1.71x** |
| with_columns(text.lower()) | 10,000,000 | 140.75 | 616.17 | **4.38x** |
| with_columns(text.len()) | 10,000,000 | 126.85 | 47.97 | *0.38x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.31 | 0.01 | *0.02x* |
| head(100) | 1,000,000 | 0.30 | 0.01 | *0.02x* |
| head(1000) | 1,000,000 | 0.29 | 0.01 | *0.03x* |
| head(10) | 10,000,000 | 0.29 | 0.01 | *0.02x* |
| head(100) | 10,000,000 | 0.29 | 0.01 | *0.02x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.03x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 85.60 | 34.91 | *0.41x* |
| sort(group, value1) | 1,000,000 | 253.43 | 82.95 | *0.33x* |
| sort(value1) | 10,000,000 | 1343.15 | 479.45 | *0.36x* |
| sort(group, value1) | 10,000,000 | 3218.20 | 1525.29 | *0.47x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 166.36 | 34.51 | *0.21x* |
| left_join(1000000x100000) | 1,000,000 | 132.19 | 41.97 | *0.32x* |
| inner_join(10000000x1000000) | 10,000,000 | 3471.31 | 622.57 | *0.18x* |
| left_join(10000000x1000000) | 10,000,000 | 4049.07 | 633.87 | *0.16x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 54.75 | 34.29 | *0.63x* |
| scan.select(3).filter | 2,964,624 | 18.52 | 7.76 | *0.42x* |
| glob_scan.head(1000) | 17,787,744 | 6.66 | 10.29 | **1.55x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.03 | 1.35 | *0.19x* |
| join -> groupby | 1,000,000 | 62.92 | 29.25 | *0.46x* |
| groupby(category-dtype key).sum | 10,000,000 | 21.00 | 9.27 | *0.44x* |
| join -> groupby | 10,000,000 | 813.71 | 473.53 | *0.58x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 4.38x faster
- **with_columns(text.lower())** (string): 4.22x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 3.47x faster
- **filter(text.contains('foo'))** (string): 1.71x faster
- **glob_scan.head(1000)** (parquet_scan): 1.55x faster

### Where Polars is Faster

- **head(100)** (limit): 42.28x faster
- **head(100)** (limit): 42.27x faster
- **head(10)** (limit): 40.74x faster
- **head(10)** (limit): 40.64x faster
- **head(1000)** (limit): 38.38x faster
