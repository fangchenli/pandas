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
| filter_project | 0.14x | 0.24x | 0.03x |
| aggregation | 0.42x | 1.47x | 0.08x |
| string | 1.31x | 3.17x | 0.37x |
| limit | 0.02x | 0.03x | 0.02x |
| sort | 0.28x | 0.29x | 0.26x |
| join | 0.29x | 0.36x | 0.22x |
| parquet_scan | 0.28x | 0.69x | 0.08x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 19.73 | 3.39 | *0.17x* |
| filter + select(3 cols) | 1,000,000 | 42.02 | 1.44 | *0.03x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.12 | 2.74 | *0.21x* |
| with_columns(value1 + value2) | 1,000,000 | 6.45 | 0.64 | *0.10x* |
| filter(value1 > 0) | 10,000,000 | 209.90 | 33.99 | *0.16x* |
| filter + select(3 cols) | 10,000,000 | 437.96 | 16.88 | *0.04x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 114.89 | 27.76 | *0.24x* |
| with_columns(value1 + value2) | 10,000,000 | 58.44 | 9.76 | *0.17x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 24.42 | 2.48 | *0.10x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 29.83 | 10.37 | *0.35x* |
| groupby(category[100]).sum() | 1,000,000 | 29.62 | 2.51 | *0.08x* |
| groupby(group).sum(value1) | 10,000,000 | 62.37 | 18.17 | *0.29x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 77.97 | 114.73 | **1.47x** |
| groupby(category[100]).sum() | 10,000,000 | 75.58 | 15.27 | *0.20x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 46.26 | 17.34 | *0.37x* |
| with_columns(text.lower()) | 1,000,000 | 20.94 | 66.43 | **3.17x** |
| with_columns(text.len()) | 1,000,000 | 8.84 | 4.50 | *0.51x* |
| filter(text.contains('foo')) | 10,000,000 | 453.30 | 171.91 | *0.38x* |
| with_columns(text.lower()) | 10,000,000 | 217.87 | 626.56 | **2.88x** |
| with_columns(text.len()) | 10,000,000 | 88.27 | 47.76 | *0.54x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.31 | 0.01 | *0.02x* |
| head(100) | 1,000,000 | 0.29 | 0.01 | *0.02x* |
| head(1000) | 1,000,000 | 0.28 | 0.01 | *0.02x* |
| head(10) | 10,000,000 | 0.28 | 0.01 | *0.03x* |
| head(100) | 10,000,000 | 0.28 | 0.01 | *0.02x* |
| head(1000) | 10,000,000 | 0.28 | 0.01 | *0.02x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 119.24 | 34.98 | *0.29x* |
| sort(group, value1) | 1,000,000 | 298.88 | 84.73 | *0.28x* |
| sort(value1) | 10,000,000 | 1527.86 | 450.61 | *0.29x* |
| sort(group, value1) | 10,000,000 | 5486.30 | 1440.18 | *0.26x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 130.91 | 39.75 | *0.30x* |
| left_join(1000000x100000) | 1,000,000 | 129.05 | 46.05 | *0.36x* |
| inner_join(10000000x1000000) | 10,000,000 | 2492.63 | 743.46 | *0.30x* |
| left_join(10000000x1000000) | 10,000,000 | 3090.16 | 671.59 | *0.22x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 52.83 | 36.24 | *0.69x* |
| scan.select(3).filter | 2,964,624 | 86.66 | 7.59 | *0.09x* |
| glob_scan.head(1000) | 17,787,744 | 138.07 | 10.87 | *0.08x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 3.17x faster
- **with_columns(text.lower())** (string): 2.88x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 1.47x faster

### Where Polars is Faster

- **head(100)** (limit): 41.93x faster
- **head(10)** (limit): 41.28x faster
- **head(1000)** (limit): 40.87x faster
- **head(100)** (limit): 40.71x faster
- **head(1000)** (limit): 40.51x faster
