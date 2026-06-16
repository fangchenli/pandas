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
| filter_project | 0.53x | 2.29x | 0.10x |
| aggregation | 1.49x | 4.66x | 0.17x |
| string | 1.65x | 3.61x | 0.30x |
| limit | 0.02x | 0.03x | 0.02x |
| sort | 0.72x | 0.97x | 0.38x |
| join | 0.38x | 0.55x | 0.29x |
| parquet_scan | 0.82x | 1.44x | 0.41x |
| engine_pipeline | 0.44x | 0.59x | 0.18x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 16.59 | 5.31 | *0.32x* |
| filter + select(3 cols) | 1,000,000 | 11.95 | 2.17 | *0.18x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 15.13 | 4.43 | *0.29x* |
| with_columns(value1 + value2) | 1,000,000 | 6.97 | 0.72 | *0.10x* |
| filter(value1 > 0).select(sum) | 1,000,000 | 5.29 | 0.94 | *0.18x* |
| filter(val1>0 AND int_val<500).select(count) | 1,000,000 | 2.51 | 1.21 | *0.48x* |
| filter(value1 > 0).select(sum(v1*v2)) | 1,000,000 | 5.57 | 2.83 | *0.51x* |
| filter(value1 > 0) | 10,000,000 | 87.92 | 34.25 | *0.39x* |
| filter + select(3 cols) | 10,000,000 | 45.88 | 17.60 | *0.38x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 72.90 | 37.20 | *0.51x* |
| with_columns(value1 + value2) | 10,000,000 | 61.99 | 10.85 | *0.18x* |
| filter(value1 > 0).select(sum) | 10,000,000 | 12.32 | 9.06 | *0.74x* |
| filter(val1>0 AND int_val<500).select(count) | 10,000,000 | 6.34 | 14.54 | **2.29x** |
| filter(value1 > 0).select(sum(v1*v2)) | 10,000,000 | 22.46 | 18.50 | *0.82x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 16.79 | 2.88 | *0.17x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 14.46 | 14.89 | **1.03x** |
| groupby(category[100]).sum() | 1,000,000 | 7.57 | 4.68 | *0.62x* |
| groupby(group).sum(value1) | 10,000,000 | 26.89 | 34.79 | **1.29x** |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 50.70 | 236.24 | **4.66x** |
| groupby(category[100]).sum() | 10,000,000 | 24.87 | 28.34 | **1.14x** |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 20.94 | 20.38 | *0.97x* |
| with_columns(text.lower()) | 1,000,000 | 23.46 | 84.60 | **3.61x** |
| with_columns(text.len()) | 1,000,000 | 15.72 | 4.70 | *0.30x* |
| filter(text.contains('foo')) | 10,000,000 | 147.74 | 194.83 | **1.32x** |
| with_columns(text.lower()) | 10,000,000 | 183.78 | 618.56 | **3.37x** |
| with_columns(text.len()) | 10,000,000 | 165.06 | 51.72 | *0.31x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.34 | 0.01 | *0.03x* |
| head(100) | 1,000,000 | 0.33 | 0.01 | *0.02x* |
| head(1000) | 1,000,000 | 0.37 | 0.01 | *0.02x* |
| head(10) | 10,000,000 | 0.31 | 0.01 | *0.02x* |
| head(100) | 10,000,000 | 0.29 | 0.01 | *0.02x* |
| head(1000) | 10,000,000 | 0.30 | 0.01 | *0.02x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 80.44 | 44.35 | *0.55x* |
| sort(group[str], value1) | 1,000,000 | 187.91 | 168.41 | *0.90x* |
| sort(int_val, value1) | 1,000,000 | 129.68 | 87.21 | *0.67x* |
| sort(num_a) [numeric-only] | 1,000,000 | 46.58 | 31.87 | *0.68x* |
| sort(key_int, num_a) [numeric-only] | 1,000,000 | 73.60 | 48.89 | *0.66x* |
| sort(value1) | 10,000,000 | 1233.68 | 470.49 | *0.38x* |
| sort(group[str], value1) | 10,000,000 | 1714.21 | 1659.52 | *0.97x* |
| sort(int_val, value1) | 10,000,000 | 1267.74 | 673.59 | *0.53x* |
| sort(num_a) [numeric-only] | 10,000,000 | 337.33 | 311.87 | *0.92x* |
| sort(key_int, num_a) [numeric-only] | 10,000,000 | 593.35 | 527.54 | *0.89x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 248.83 | 73.27 | *0.29x* |
| left_join(1000000x100000) | 1,000,000 | 176.46 | 96.94 | *0.55x* |
| inner_join(10000000x1000000) | 10,000,000 | 2157.16 | 668.05 | *0.31x* |
| left_join(10000000x1000000) | 10,000,000 | 2089.87 | 734.54 | *0.35x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 64.51 | 39.42 | *0.61x* |
| scan.select(3).filter | 2,964,624 | 19.19 | 7.87 | *0.41x* |
| glob_scan.head(1000) | 17,787,744 | 7.41 | 10.70 | **1.44x** |

## Engine Pipeline

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(category-dtype key).sum | 1,000,000 | 7.52 | 1.32 | *0.18x* |
| join -> groupby | 1,000,000 | 58.86 | 31.45 | *0.53x* |
| groupby(category-dtype key).sum | 10,000,000 | 21.44 | 9.73 | *0.45x* |
| join -> groupby | 10,000,000 | 927.32 | 545.28 | *0.59x* |

## Key Findings

### Where Lazy Pandas is Faster

- **groupby(group).agg(sum, mean, max, count)** (aggregation): 4.66x faster
- **with_columns(text.lower())** (string): 3.61x faster
- **with_columns(text.lower())** (string): 3.37x faster
- **filter(val1>0 AND int_val<500).select(count)** (filter_project): 2.29x faster
- **glob_scan.head(1000)** (parquet_scan): 1.44x faster

### Where Polars is Faster

- **head(1000)** (limit): 45.03x faster
- **head(100)** (limit): 44.59x faster
- **head(1000)** (limit): 41.88x faster
- **head(100)** (limit): 41.12x faster
- **head(10)** (limit): 40.13x faster
