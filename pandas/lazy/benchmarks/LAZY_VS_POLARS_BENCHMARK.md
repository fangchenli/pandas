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
| filter_project | 0.13x | 0.22x | 0.04x |
| aggregation | 0.39x | 1.35x | 0.09x |
| string | 1.20x | 2.85x | 0.37x |
| limit | 0.00x | 0.01x | 0.00x |
| sort | 0.29x | 0.30x | 0.28x |
| join | 0.30x | 0.36x | 0.21x |
| parquet_scan | 0.28x | 0.67x | 0.07x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 20.10 | 3.43 | *0.17x* |
| filter + select(3 cols) | 1,000,000 | 41.30 | 1.46 | *0.04x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.64 | 2.68 | *0.20x* |
| with_columns(value1 + value2) | 1,000,000 | 6.77 | 0.52 | *0.08x* |
| filter(value1 > 0) | 10,000,000 | 210.80 | 32.51 | *0.15x* |
| filter + select(3 cols) | 10,000,000 | 441.92 | 16.09 | *0.04x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 123.58 | 26.88 | *0.22x* |
| with_columns(value1 + value2) | 10,000,000 | 69.75 | 10.05 | *0.14x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 23.20 | 2.73 | *0.12x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 29.22 | 10.56 | *0.36x* |
| groupby(category[100]).sum() | 1,000,000 | 29.09 | 2.49 | *0.09x* |
| groupby(group).sum(value1) | 10,000,000 | 69.24 | 17.88 | *0.26x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 85.23 | 114.85 | **1.35x** |
| groupby(category[100]).sum() | 10,000,000 | 81.86 | 15.33 | *0.19x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 46.85 | 17.57 | *0.38x* |
| with_columns(text.lower()) | 1,000,000 | 21.34 | 55.48 | **2.60x** |
| with_columns(text.len()) | 1,000,000 | 9.30 | 4.49 | *0.48x* |
| filter(text.contains('foo')) | 10,000,000 | 458.22 | 170.39 | *0.37x* |
| with_columns(text.lower()) | 10,000,000 | 226.61 | 646.50 | **2.85x** |
| with_columns(text.len()) | 10,000,000 | 94.71 | 47.02 | *0.50x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 1.09 | 0.01 | *0.01x* |
| head(100) | 1,000,000 | 1.09 | 0.01 | *0.01x* |
| head(1000) | 1,000,000 | 1.08 | 0.01 | *0.01x* |
| head(10) | 10,000,000 | 9.10 | 0.01 | *0.00x* |
| head(100) | 10,000,000 | 9.09 | 0.01 | *0.00x* |
| head(1000) | 10,000,000 | 10.91 | 0.01 | *0.00x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 112.35 | 33.88 | *0.30x* |
| sort(group, value1) | 1,000,000 | 283.49 | 81.33 | *0.29x* |
| sort(value1) | 10,000,000 | 1525.31 | 442.32 | *0.29x* |
| sort(group, value1) | 10,000,000 | 4998.89 | 1381.62 | *0.28x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 125.79 | 38.13 | *0.30x* |
| left_join(1000000x100000) | 1,000,000 | 124.63 | 45.03 | *0.36x* |
| inner_join(10000000x1000000) | 10,000,000 | 2357.90 | 725.95 | *0.31x* |
| left_join(10000000x1000000) | 10,000,000 | 2961.34 | 618.49 | *0.21x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 49.90 | 33.37 | *0.67x* |
| scan.select(3).filter | 2,964,624 | 82.61 | 7.64 | *0.09x* |
| glob_scan.head(1000) | 17,787,744 | 147.43 | 10.36 | *0.07x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 2.85x faster
- **with_columns(text.lower())** (string): 2.60x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 1.35x faster

### Where Polars is Faster

- **head(100)** (limit): 1371.84x faster
- **head(10)** (limit): 1355.91x faster
- **head(1000)** (limit): 1335.88x faster
- **head(100)** (limit): 163.86x faster
- **head(1000)** (limit): 150.20x faster
