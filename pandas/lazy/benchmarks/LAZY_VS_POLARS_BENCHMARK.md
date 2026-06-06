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
| filter_project | 0.22x | 0.39x | 0.08x |
| aggregation | 0.53x | 2.11x | 0.08x |
| string | 1.10x | 2.43x | 0.42x |
| limit | 0.03x | 0.05x | 0.01x |
| sort | 0.27x | 0.34x | 0.22x |
| join | 0.31x | 0.39x | 0.21x |
| parquet_scan | 0.30x | 0.74x | 0.08x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 22.86 | 3.64 | *0.16x* |
| filter + select(3 cols) | 1,000,000 | 9.21 | 1.94 | *0.21x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 14.07 | 3.25 | *0.23x* |
| with_columns(value1 + value2) | 1,000,000 | 8.19 | 0.63 | *0.08x* |
| filter(value1 > 0) | 10,000,000 | 278.72 | 62.78 | *0.23x* |
| filter + select(3 cols) | 10,000,000 | 57.57 | 22.43 | *0.39x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 125.72 | 47.40 | *0.38x* |
| with_columns(value1 + value2) | 10,000,000 | 122.84 | 13.65 | *0.11x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 24.90 | 3.39 | *0.14x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 34.15 | 13.51 | *0.40x* |
| groupby(category[100]).sum() | 1,000,000 | 32.73 | 2.78 | *0.08x* |
| groupby(group).sum(value1) | 10,000,000 | 130.92 | 29.77 | *0.23x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 100.20 | 210.97 | **2.11x** |
| groupby(category[100]).sum() | 10,000,000 | 87.24 | 21.06 | *0.24x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 50.45 | 21.16 | *0.42x* |
| with_columns(text.lower()) | 1,000,000 | 25.28 | 61.40 | **2.43x** |
| with_columns(text.len()) | 1,000,000 | 10.44 | 4.66 | *0.45x* |
| filter(text.contains('foo')) | 10,000,000 | 513.79 | 232.66 | *0.45x* |
| with_columns(text.lower()) | 10,000,000 | 284.61 | 664.56 | **2.34x** |
| with_columns(text.len()) | 10,000,000 | 101.43 | 51.57 | *0.51x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 0.32 | 0.01 | *0.03x* |
| head(100) | 1,000,000 | 0.30 | 0.01 | *0.03x* |
| head(1000) | 1,000,000 | 0.33 | 0.01 | *0.03x* |
| head(10) | 10,000,000 | 1.12 | 0.01 | *0.01x* |
| head(100) | 10,000,000 | 0.35 | 0.02 | *0.05x* |
| head(1000) | 10,000,000 | 0.33 | 0.01 | *0.04x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 150.33 | 50.99 | *0.34x* |
| sort(group, value1) | 1,000,000 | 472.31 | 118.48 | *0.25x* |
| sort(value1) | 10,000,000 | 2590.81 | 564.83 | *0.22x* |
| sort(group, value1) | 10,000,000 | 5976.63 | 1599.59 | *0.27x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 206.40 | 43.36 | *0.21x* |
| left_join(1000000x100000) | 1,000,000 | 173.60 | 66.77 | *0.38x* |
| inner_join(10000000x1000000) | 10,000,000 | 2489.11 | 618.48 | *0.25x* |
| left_join(10000000x1000000) | 10,000,000 | 3997.95 | 1550.08 | *0.39x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 70.77 | 52.16 | *0.74x* |
| scan.select(3).filter | 2,964,624 | 137.13 | 12.39 | *0.09x* |
| glob_scan.head(1000) | 17,787,744 | 203.57 | 16.32 | *0.08x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 2.43x faster
- **with_columns(text.lower())** (string): 2.34x faster
- **groupby(group).agg(sum, mean, max, count)** (aggregation): 2.11x faster

### Where Polars is Faster

- **head(10)** (limit): 84.15x faster
- **head(1000)** (limit): 39.88x faster
- **head(100)** (limit): 37.36x faster
- **head(10)** (limit): 35.76x faster
- **head(1000)** (limit): 28.49x faster
