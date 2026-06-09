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
| filter_project | 0.25x | 0.50x | 0.10x |
| aggregation | 1.11x | 3.30x | 0.32x |
| string | 1.84x | 4.43x | 0.36x |
| limit | 0.03x | 0.03x | 0.02x |
| sort | 0.76x | 1.12x | 0.48x |
| join | 0.31x | 0.38x | 0.23x |
| parquet_scan | 0.81x | 1.44x | 0.43x |
| engine_pipeline | 0.45x | 0.58x | 0.19x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 20.00 | 3.53 | *0.18x* |
| filter + select(3 cols) | 1,000,000 | 4.95 | 2.48 | *0.50x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.36 | 2.92 | *0.22x* |
| with_columns(value1 + value2) | 1,000,000 | 5.22 | 0.50 | *0.10x* |
| filter(value1 > 0) | 10,000,000 | 223.52 | 42.00 | *0.19x* |
| filter + select(3 cols) | 10,000,000 | 45.47 | 19.34 | *0.43x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 155.60 | 33.96 | *0.22x* |
| with_columns(value1 + value2) | 10,000,000 | 52.70 | 9.49 | *0.18x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 7.20 | 2.55 | *0.35x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 12.98 | 10.89 | *0.84x* |
| groupby(category[100]).sum() | 1,000,000 | 7.37 | 2.39 | *0.32x* |
| groupby(group).sum(value1) | 10,000,000 | 22.23 | 20.06 | *0.90x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 39.98 | 131.97 | **3.30x** |
| groupby(category[100]).sum() | 10,000,000 | 17.43 | 16.63 | *0.95x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 15.33 | 19.15 | **1.25x** |
| with_columns(text.lower()) | 1,000,000 | 18.70 | 60.24 | **3.22x** |
| with_columns(text.len()) | 1,000,000 | 12.00 | 4.61 | *0.38x* |
| filter(text.contains('foo')) | 10,000,000 | 135.99 | 192.67 | **1.42x** |
| with_columns(text.lower()) | 10,000,000 | 140.31 | 621.68 | **4.43x** |
| with_columns(text.len()) | 10,000,000 | 143.51 | 51.02 | *0.36x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.31 | 0.01 | *0.03x* |
| head(100) | 1,000,000 | 0.30 | 0.01 | *0.03x* |
| head(1000) | 1,000,000 | 0.31 | 0.01 | *0.02x* |
| head(10) | 10,000,000 | 0.31 | 0.01 | *0.03x* |
| head(100) | 10,000,000 | 0.31 | 0.01 | *0.03x* |
| head(1000) | 10,000,000 | 0.31 | 0.01 | *0.02x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 73.84 | 40.94 | *0.55x* |
| sort(group[str], value1) | 1,000,000 | 113.11 | 85.52 | *0.76x* |
| sort(int_val, value1) | 1,000,000 | 93.04 | 49.14 | *0.53x* |
| sort(num_a) [numeric-only] | 1,000,000 | 31.19 | 24.75 | *0.79x* |
| sort(key_int, num_a) [numeric-only] | 1,000,000 | 47.18 | 34.68 | *0.74x* |
| sort(value1) | 10,000,000 | 1129.68 | 539.07 | *0.48x* |
| sort(group[str], value1) | 10,000,000 | 1517.79 | 1707.47 | **1.12x** |
| sort(int_val, value1) | 10,000,000 | 1204.34 | 713.14 | *0.59x* |
| sort(num_a) [numeric-only] | 10,000,000 | 344.40 | 386.69 | **1.12x** |
| sort(key_int, num_a) [numeric-only] | 10,000,000 | 613.25 | 544.42 | *0.89x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 123.66 | 35.05 | *0.28x* |
| left_join(1000000x100000) | 1,000,000 | 111.44 | 42.86 | *0.38x* |
| inner_join(10000000x1000000) | 10,000,000 | 2071.48 | 718.24 | *0.35x* |
| left_join(10000000x1000000) | 10,000,000 | 3041.83 | 712.04 | *0.23x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 61.98 | 35.55 | *0.57x* |
| scan.select(3).filter | 2,964,624 | 19.28 | 8.32 | *0.43x* |
| glob_scan.head(1000) | 17,787,744 | 7.92 | 11.40 | **1.44x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.92 | 1.54 | *0.19x* |
| join -> groupby | 1,000,000 | 66.76 | 38.50 | *0.58x* |
| groupby(category-dtype key).sum | 10,000,000 | 21.23 | 9.68 | *0.46x* |
| join -> groupby | 10,000,000 | 824.94 | 479.09 | *0.58x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 4.43x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 3.30x faster
- **with_columns(text.lower())** (string): 3.22x faster
- **glob_scan.head(1000)** (parquet_scan): 1.44x faster
- **filter(text.contains('foo'))** (string): 1.42x faster

### Where Polars is Faster

- **head(1000)** (limit): 41.14x faster
- **head(1000)** (limit): 40.58x faster
- **head(100)** (limit): 39.89x faster
- **head(100)** (limit): 38.91x faster
- **head(10)** (limit): 38.67x faster
