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
| filter_project | 0.57x | 2.17x | 0.09x |
| aggregation | 1.10x | 3.16x | 0.35x |
| string | 2.16x | 4.93x | 0.34x |
| limit | 0.02x | 0.03x | 0.02x |
| sort | 0.71x | 1.05x | 0.48x |
| join | 0.40x | 0.44x | 0.35x |
| parquet_scan | 0.86x | 1.63x | 0.42x |
| engine_pipeline | 0.51x | 0.74x | 0.20x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 10.13 | 3.43 | *0.34x* |
| filter + select(3 cols) | 1,000,000 | 5.10 | 1.46 | *0.29x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 11.15 | 2.71 | *0.24x* |
| with_columns(value1 + value2) | 1,000,000 | 5.66 | 0.53 | *0.09x* |
| filter(value1 > 0).select(sum) | 1,000,000 | 2.76 | 0.66 | *0.24x* |
| filter(val1>0 AND int_val<500).select(count) | 1,000,000 | 1.51 | 0.89 | *0.59x* |
| filter(value1 > 0).select(sum(v1*v2)) | 1,000,000 | 2.64 | 1.18 | *0.45x* |
| filter(value1 > 0) | 10,000,000 | 86.22 | 38.00 | *0.44x* |
| filter + select(3 cols) | 10,000,000 | 47.18 | 17.34 | *0.37x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 60.75 | 31.48 | *0.52x* |
| with_columns(value1 + value2) | 10,000,000 | 54.73 | 10.10 | *0.18x* |
| filter(value1 > 0).select(sum) | 10,000,000 | 10.85 | 8.15 | *0.75x* |
| filter(val1>0 AND int_val<500).select(count) | 10,000,000 | 4.06 | 8.79 | **2.17x** |
| filter(value1 > 0).select(sum(v1*v2)) | 10,000,000 | 11.97 | 16.32 | **1.36x** |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 7.21 | 2.53 | *0.35x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 13.10 | 10.30 | *0.79x* |
| groupby(category[100]).sum() | 1,000,000 | 5.81 | 2.21 | *0.38x* |
| groupby(group).sum(value1) | 10,000,000 | 18.75 | 19.10 | **1.02x** |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 37.20 | 117.45 | **3.16x** |
| groupby(category[100]).sum() | 10,000,000 | 17.64 | 15.72 | *0.89x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 13.48 | 17.21 | **1.28x** |
| with_columns(text.lower()) | 1,000,000 | 13.60 | 61.09 | **4.49x** |
| with_columns(text.len()) | 1,000,000 | 13.36 | 4.50 | *0.34x* |
| filter(text.contains('foo')) | 10,000,000 | 114.32 | 174.82 | **1.53x** |
| with_columns(text.lower()) | 10,000,000 | 125.70 | 620.10 | **4.93x** |
| with_columns(text.len()) | 10,000,000 | 129.71 | 47.58 | *0.37x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.30 | 0.01 | *0.03x* |
| head(100) | 1,000,000 | 0.28 | 0.01 | *0.03x* |
| head(1000) | 1,000,000 | 0.28 | 0.01 | *0.02x* |
| head(10) | 10,000,000 | 0.32 | 0.01 | *0.02x* |
| head(100) | 10,000,000 | 0.28 | 0.01 | *0.02x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.03x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 69.72 | 33.99 | *0.49x* |
| sort(group[str], value1) | 1,000,000 | 115.69 | 85.58 | *0.74x* |
| sort(int_val, value1) | 1,000,000 | 86.19 | 44.90 | *0.52x* |
| sort(num_a) [numeric-only] | 1,000,000 | 33.03 | 22.70 | *0.69x* |
| sort(key_int, num_a) [numeric-only] | 1,000,000 | 49.37 | 33.63 | *0.68x* |
| sort(value1) | 10,000,000 | 965.19 | 460.42 | *0.48x* |
| sort(group[str], value1) | 10,000,000 | 1413.38 | 1490.85 | **1.05x** |
| sort(int_val, value1) | 10,000,000 | 1255.82 | 645.83 | *0.51x* |
| sort(num_a) [numeric-only] | 10,000,000 | 309.35 | 301.20 | *0.97x* |
| sort(key_int, num_a) [numeric-only] | 10,000,000 | 549.64 | 514.35 | *0.94x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 79.29 | 34.41 | *0.43x* |
| left_join(1000000x100000) | 1,000,000 | 93.03 | 41.31 | *0.44x* |
| inner_join(10000000x1000000) | 10,000,000 | 1460.40 | 554.38 | *0.38x* |
| left_join(10000000x1000000) | 10,000,000 | 1876.36 | 656.52 | *0.35x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 65.06 | 34.10 | *0.52x* |
| scan.select(3).filter | 2,964,624 | 18.80 | 7.96 | *0.42x* |
| glob_scan.head(1000) | 17,787,744 | 6.65 | 10.83 | **1.63x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.16 | 1.40 | *0.20x* |
| join -> groupby | 1,000,000 | 38.11 | 28.38 | *0.74x* |
| groupby(category-dtype key).sum | 10,000,000 | 21.71 | 9.22 | *0.42x* |
| join -> groupby | 10,000,000 | 674.34 | 451.23 | *0.67x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 4.93x faster
- **with_columns(text.lower())** (string): 4.49x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 3.16x faster
- **filter(val1>0 AND int_val<500).select(count)** (filter_project): 2.17x faster
- **glob_scan.head(1000)** (parquet_scan): 1.63x faster

### Where Polars is Faster

- **head(10)** (limit): 47.73x faster
- **head(100)** (limit): 41.91x faster
- **head(1000)** (limit): 40.56x faster
- **head(100)** (limit): 39.94x faster
- **head(10)** (limit): 38.44x faster
