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
| filter_project | 0.21x | 0.37x | 0.10x |
| aggregation | 1.08x | 3.06x | 0.35x |
| string | 2.38x | 5.93x | 0.35x |
| limit | 0.03x | 0.03x | 0.03x |
| sort | 0.74x | 1.08x | 0.53x |
| join | 0.40x | 0.51x | 0.30x |
| parquet_scan | 0.92x | 1.66x | 0.42x |
| engine_pipeline | 0.42x | 0.52x | 0.21x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 19.73 | 3.23 | *0.16x* |
| filter + select(3 cols) | 1,000,000 | 5.31 | 1.44 | *0.27x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.78 | 2.71 | *0.20x* |
| with_columns(value1 + value2) | 1,000,000 | 5.62 | 0.56 | *0.10x* |
| filter(value1 > 0) | 10,000,000 | 211.24 | 32.09 | *0.15x* |
| filter + select(3 cols) | 10,000,000 | 41.58 | 15.52 | *0.37x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 121.20 | 26.04 | *0.21x* |
| with_columns(value1 + value2) | 10,000,000 | 47.00 | 10.01 | *0.21x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 7.01 | 2.43 | *0.35x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 12.98 | 10.69 | *0.82x* |
| groupby(category[100]).sum() | 1,000,000 | 5.83 | 2.13 | *0.37x* |
| groupby(group).sum(value1) | 10,000,000 | 18.59 | 17.75 | *0.96x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 37.87 | 115.90 | **3.06x** |
| groupby(category[100]).sum() | 10,000,000 | 16.27 | 15.05 | *0.92x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 14.70 | 17.12 | **1.16x** |
| with_columns(text.lower()) | 1,000,000 | 13.23 | 64.99 | **4.91x** |
| with_columns(text.len()) | 1,000,000 | 12.92 | 4.49 | *0.35x* |
| filter(text.contains('foo')) | 10,000,000 | 114.56 | 172.55 | **1.51x** |
| with_columns(text.lower()) | 10,000,000 | 111.74 | 662.08 | **5.93x** |
| with_columns(text.len()) | 10,000,000 | 111.39 | 47.03 | *0.42x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.31 | 0.01 | *0.03x* |
| head(100) | 1,000,000 | 0.29 | 0.01 | *0.03x* |
| head(1000) | 1,000,000 | 0.29 | 0.01 | *0.03x* |
| head(10) | 10,000,000 | 0.28 | 0.01 | *0.03x* |
| head(100) | 10,000,000 | 0.28 | 0.01 | *0.03x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.03x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 65.27 | 34.73 | *0.53x* |
| sort(group[str], value1) | 1,000,000 | 113.92 | 82.47 | *0.72x* |
| sort(int_val, value1) | 1,000,000 | 83.49 | 44.77 | *0.54x* |
| sort(num_a) [numeric-only] | 1,000,000 | 29.53 | 22.59 | *0.76x* |
| sort(key_int, num_a) [numeric-only] | 1,000,000 | 46.94 | 33.83 | *0.72x* |
| sort(value1) | 10,000,000 | 803.28 | 435.54 | *0.54x* |
| sort(group[str], value1) | 10,000,000 | 1284.24 | 1384.88 | **1.08x** |
| sort(int_val, value1) | 10,000,000 | 1040.69 | 616.13 | *0.59x* |
| sort(num_a) [numeric-only] | 10,000,000 | 306.44 | 298.89 | *0.98x* |
| sort(key_int, num_a) [numeric-only] | 10,000,000 | 549.16 | 489.54 | *0.89x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 87.94 | 34.72 | *0.39x* |
| left_join(1000000x100000) | 1,000,000 | 87.86 | 44.91 | *0.51x* |
| inner_join(10000000x1000000) | 10,000,000 | 1769.32 | 539.22 | *0.30x* |
| left_join(10000000x1000000) | 10,000,000 | 1619.61 | 645.71 | *0.40x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 50.96 | 34.94 | *0.69x* |
| scan.select(3).filter | 2,964,624 | 18.38 | 7.72 | *0.42x* |
| glob_scan.head(1000) | 17,787,744 | 6.56 | 10.87 | **1.66x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.12 | 1.50 | *0.21x* |
| join -> groupby | 1,000,000 | 55.75 | 28.72 | *0.52x* |
| groupby(category-dtype key).sum | 10,000,000 | 20.77 | 9.09 | *0.44x* |
| join -> groupby | 10,000,000 | 830.12 | 429.10 | *0.52x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 5.93x faster
- **with_columns(text.lower())** (string): 4.91x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 3.06x faster
- **glob_scan.head(1000)** (parquet_scan): 1.66x faster
- **filter(text.contains('foo'))** (string): 1.51x faster

### Where Polars is Faster

- **head(100)** (limit): 39.96x faster
- **head(1000)** (limit): 39.11x faster
- **head(1000)** (limit): 38.82x faster
- **head(100)** (limit): 38.58x faster
- **head(10)** (limit): 37.76x faster
