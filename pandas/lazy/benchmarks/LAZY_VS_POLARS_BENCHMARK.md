# Lazy Pandas vs Polars Benchmark Report

## Environment

- pandas version: 3.1.0.dev0+1220.g55b305e82a
- polars version: 1.37.1
- Python version: 3.11.14
- Platform: macOS-26.5.1-arm64-arm-64bit
- Warmup runs: 3
- Timed runs: 7

## Methodology

Lazy pandas timings use the **physical engine**
(`collect(use_physical_planner=True)`). Reports generated before June 2026
measured the eager evaluation path for every category except parquet scans
and are not directly comparable. The test data is mixed-dtype (numeric +
string columns), so conversion boundaries are part of what is measured.
Absolute timings vary with machine load; treat ratios as the signal.

## Summary

**Speedup interpretation:** Values > 1.0 mean lazy pandas is faster, < 1.0 mean Polars is faster.

### Category Summary

| Category | Avg Speedup | Best LP | Best Polars |
|----------|-------------|---------|-------------|
| filter_project | 0.13x | 0.21x | 0.03x |
| aggregation | 0.08x | 0.12x | 0.04x |
| string | 1.07x | 2.43x | 0.37x |
| limit | 0.00x | 0.01x | 0.00x |
| sort | 0.34x | 0.45x | 0.25x |
| join | 0.06x | 0.07x | 0.05x |
| parquet_scan | 0.28x | 0.67x | 0.07x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 20.94 | 3.59 | *0.17x* |
| filter + select(3 cols) | 1,000,000 | 43.57 | 1.45 | *0.03x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.81 | 2.79 | *0.20x* |
| with_columns(value1 + value2) | 1,000,000 | 8.67 | 0.63 | *0.07x* |
| filter(value1 > 0) | 10,000,000 | 291.55 | 45.02 | *0.15x* |
| filter + select(3 cols) | 10,000,000 | 696.14 | 23.34 | *0.03x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 159.36 | 32.95 | *0.21x* |
| with_columns(value1 + value2) | 10,000,000 | 85.82 | 12.23 | *0.14x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 31.97 | 3.11 | *0.10x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 101.99 | 10.94 | *0.11x* |
| groupby(category[100]).sum() | 1,000,000 | 49.92 | 3.18 | *0.06x* |
| groupby(group).sum(value1) | 10,000,000 | 388.88 | 24.07 | *0.06x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 1393.63 | 165.30 | *0.12x* |
| groupby(category[100]).sum() | 10,000,000 | 672.97 | 24.18 | *0.04x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 48.24 | 17.86 | *0.37x* |
| with_columns(text.lower()) | 1,000,000 | 22.00 | 53.51 | **2.43x** |
| with_columns(text.len()) | 1,000,000 | 10.08 | 4.62 | *0.46x* |
| filter(text.contains('foo')) | 10,000,000 | 638.99 | 239.14 | *0.37x* |
| with_columns(text.lower()) | 10,000,000 | 313.30 | 741.90 | **2.37x** |
| with_columns(text.len()) | 10,000,000 | 142.68 | 60.58 | *0.42x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 1.39 | 0.01 | *0.01x* |
| head(100) | 1,000,000 | 1.20 | 0.01 | *0.01x* |
| head(1000) | 1,000,000 | 1.19 | 0.01 | *0.01x* |
| head(10) | 10,000,000 | 14.11 | 0.01 | *0.00x* |
| head(100) | 10,000,000 | 14.06 | 0.01 | *0.00x* |
| head(1000) | 10,000,000 | 13.62 | 0.01 | *0.00x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 132.99 | 46.96 | *0.35x* |
| sort(group, value1) | 1,000,000 | 348.57 | 156.90 | *0.45x* |
| sort(value1) | 10,000,000 | 2933.97 | 918.92 | *0.31x* |
| sort(group, value1) | 10,000,000 | 9594.00 | 2398.24 | *0.25x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 662.07 | 43.68 | *0.07x* |
| left_join(1000000x100000) | 1,000,000 | 1202.41 | 64.64 | *0.05x* |
| inner_join(10000000x1000000) | 10,000,000 | 13002.82 | 769.43 | *0.06x* |
| left_join(10000000x1000000) | 10,000,000 | 12904.97 | 817.74 | *0.06x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 52.48 | 35.33 | *0.67x* |
| scan.select(3).filter | 2,964,624 | 84.11 | 7.65 | *0.09x* |
| glob_scan.head(1000) | 17,787,744 | 159.93 | 11.00 | *0.07x* |

## Key Findings

### Where Lazy Pandas is Faster

- **with_columns(text.lower())** (string): 2.43x faster
- **with_columns(text.lower())** (string): 2.37x faster

### Where Polars is Faster

- **head(100)** (limit): 1721.15x faster
- **head(1000)** (limit): 1535.07x faster
- **head(10)** (limit): 1172.22x faster
- **head(10)** (limit): 172.57x faster
- **head(100)** (limit): 169.65x faster
