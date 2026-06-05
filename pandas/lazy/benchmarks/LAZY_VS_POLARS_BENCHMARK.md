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
| filter_project | 0.13x | 0.22x | 0.03x |
| aggregation | 0.07x | 0.12x | 0.03x |
| string | 1.22x | 2.89x | 0.37x |
| limit | 0.00x | 0.01x | 0.00x |
| sort | 0.28x | 0.30x | 0.27x |
| join | 0.30x | 0.36x | 0.26x |
| parquet_scan | 0.28x | 0.66x | 0.08x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 20.15 | 3.28 | *0.16x* |
| filter + select(3 cols) | 1,000,000 | 42.18 | 1.45 | *0.03x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.56 | 2.71 | *0.20x* |
| with_columns(value1 + value2) | 1,000,000 | 6.80 | 0.47 | *0.07x* |
| filter(value1 > 0) | 10,000,000 | 214.14 | 33.12 | *0.15x* |
| filter + select(3 cols) | 10,000,000 | 445.47 | 16.81 | *0.04x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 125.63 | 27.87 | *0.22x* |
| with_columns(value1 + value2) | 10,000,000 | 65.83 | 9.70 | *0.15x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 31.12 | 2.58 | *0.08x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 100.41 | 10.85 | *0.11x* |
| groupby(category[100]).sum() | 1,000,000 | 50.34 | 2.17 | *0.04x* |
| groupby(group).sum(value1) | 10,000,000 | 290.88 | 18.29 | *0.06x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 975.63 | 116.16 | *0.12x* |
| groupby(category[100]).sum() | 10,000,000 | 497.62 | 15.45 | *0.03x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 46.57 | 17.35 | *0.37x* |
| with_columns(text.lower()) | 1,000,000 | 21.39 | 61.78 | **2.89x** |
| with_columns(text.len()) | 1,000,000 | 9.29 | 4.48 | *0.48x* |
| filter(text.contains('foo')) | 10,000,000 | 465.33 | 174.13 | *0.37x* |
| with_columns(text.lower()) | 10,000,000 | 228.20 | 623.02 | **2.73x** |
| with_columns(text.len()) | 10,000,000 | 97.68 | 47.54 | *0.49x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 1.08 | 0.01 | *0.01x* |
| head(100) | 1,000,000 | 1.08 | 0.01 | *0.01x* |
| head(1000) | 1,000,000 | 1.20 | 0.01 | *0.01x* |
| head(10) | 10,000,000 | 9.08 | 0.01 | *0.00x* |
| head(100) | 10,000,000 | 9.25 | 0.01 | *0.00x* |
| head(1000) | 10,000,000 | 9.09 | 0.01 | *0.00x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 112.98 | 33.79 | *0.30x* |
| sort(group, value1) | 1,000,000 | 283.75 | 81.13 | *0.29x* |
| sort(value1) | 10,000,000 | 1630.80 | 446.39 | *0.27x* |
| sort(group, value1) | 10,000,000 | 5064.32 | 1387.32 | *0.27x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 125.31 | 37.60 | *0.30x* |
| left_join(1000000x100000) | 1,000,000 | 124.17 | 45.17 | *0.36x* |
| inner_join(10000000x1000000) | 10,000,000 | 2410.45 | 704.31 | *0.29x* |
| left_join(10000000x1000000) | 10,000,000 | 3009.86 | 790.18 | *0.26x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 50.32 | 33.30 | *0.66x* |
| scan.select(3).filter | 2,964,624 | 87.18 | 7.66 | *0.09x* |
| glob_scan.head(1000) | 17,787,744 | 136.95 | 10.36 | *0.08x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 2.89x faster
- **with_columns(text.lower())** (string): 2.73x faster

### Where Polars is Faster

- **head(100)** (limit): 1328.51x faster
- **head(10)** (limit): 1320.60x faster
- **head(1000)** (limit): 1033.87x faster
- **head(100)** (limit): 156.79x faster
- **head(1000)** (limit): 152.10x faster
