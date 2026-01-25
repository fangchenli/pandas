# Lazy Pandas vs Polars Benchmark Report

## Environment

- pandas version: 3.0.0rc1+64.g341630917a.dirty
- polars version: 1.37.1
- Python version: 3.11.14
- Platform: macOS-26.2-arm64-arm-64bit
- Warmup runs: 3
- Timed runs: 7

## Summary

**Speedup interpretation:** Values > 1.0 mean lazy pandas is faster, < 1.0 mean Polars is faster.

### Category Summary

| Category | Avg Speedup | Best LP | Best Polars |
|----------|-------------|---------|-------------|
| filter_project | 0.14x | 0.29x | 0.03x |
| aggregation | 0.19x | 0.43x | 0.08x |
| string | 0.86x | 2.07x | 0.17x |
| limit | 0.00x | 0.00x | 0.00x |
| sort | 0.22x | 0.26x | 0.18x |
| join | 0.15x | 0.16x | 0.14x |
| parquet_scan | 0.28x | 0.68x | 0.07x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 16.02 | 3.40 | *0.21x* |
| filter + select(3 cols) | 1,000,000 | 16.97 | 1.45 | *0.09x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 10.52 | 3.00 | *0.29x* |
| with_columns(value1 + value2) | 1,000,000 | 14.72 | 0.50 | *0.03x* |
| filter(value1 > 0) | 10,000,000 | 213.96 | 37.06 | *0.17x* |
| filter + select(3 cols) | 10,000,000 | 203.60 | 17.15 | *0.08x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 126.57 | 28.28 | *0.22x* |
| with_columns(value1 + value2) | 10,000,000 | 175.90 | 9.86 | *0.06x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 22.24 | 2.49 | *0.11x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 29.91 | 10.42 | *0.35x* |
| groupby(category[100]).sum() | 1,000,000 | 20.56 | 2.18 | *0.11x* |
| groupby(group).sum(value1) | 10,000,000 | 212.43 | 18.21 | *0.09x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 281.90 | 120.87 | *0.43x* |
| groupby(category[100]).sum() | 10,000,000 | 194.46 | 16.43 | *0.08x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 46.78 | 17.22 | *0.37x* |
| with_columns(text.lower()) | 1,000,000 | 25.83 | 53.37 | **2.07x** |
| with_columns(text.len()) | 1,000,000 | 29.10 | 5.03 | *0.17x* |
| filter(text.contains('foo')) | 10,000,000 | 486.44 | 175.76 | *0.36x* |
| with_columns(text.lower()) | 10,000,000 | 316.08 | 613.93 | **1.94x** |
| with_columns(text.len()) | 10,000,000 | 201.72 | 47.65 | *0.24x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 4.17 | 0.01 | *0.00x* |
| head(100) | 1,000,000 | 3.57 | 0.01 | *0.00x* |
| head(1000) | 1,000,000 | 3.87 | 0.01 | *0.00x* |
| head(10) | 10,000,000 | 39.40 | 0.01 | *0.00x* |
| head(100) | 10,000,000 | 38.22 | 0.01 | *0.00x* |
| head(1000) | 10,000,000 | 39.61 | 0.01 | *0.00x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 151.78 | 39.07 | *0.26x* |
| sort(group, value1) | 1,000,000 | 357.97 | 82.72 | *0.23x* |
| sort(value1) | 10,000,000 | 2749.93 | 492.77 | *0.18x* |
| sort(group, value1) | 10,000,000 | 7331.36 | 1455.51 | *0.20x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 308.25 | 42.11 | *0.14x* |
| left_join(1000000x100000) | 1,000,000 | 305.69 | 47.31 | *0.15x* |
| inner_join(10000000x1000000) | 10,000,000 | 5472.74 | 896.51 | *0.16x* |
| left_join(10000000x1000000) | 10,000,000 | 5720.46 | 931.51 | *0.16x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 52.63 | 35.92 | *0.68x* |
| scan.select(3).filter | 2,964,624 | 86.63 | 8.76 | *0.10x* |
| glob_scan.head(1000) | 17,787,744 | 148.07 | 10.42 | *0.07x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 2.07x faster
- **with_columns(text.lower())** (string): 1.94x faster

### Where Polars is Faster

- **head(10)** (limit): 5628.97x faster
- **head(100)** (limit): 5152.53x faster
- **head(1000)** (limit): 4591.96x faster
- **head(1000)** (limit): 509.73x faster
- **head(10)** (limit): 487.98x faster
