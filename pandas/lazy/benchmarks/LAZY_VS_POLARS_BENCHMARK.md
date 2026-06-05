# Lazy Pandas vs Polars Benchmark Report

## Environment

- pandas version: 3.1.0.dev0+1220.g55b305e82a
- polars version: 1.37.1
- Python version: 3.11.14
- Platform: macOS-26.5.1-arm64-arm-64bit
- Warmup runs: 3
- Timed runs: 7

## Methodology Note (2026-06-05)

This run measures the **physical engine**
(`collect(use_physical_planner=True)`). Earlier revisions of this report
(January 2026) unknowingly measured the *eager evaluation path* — plain
`collect()` — for every category except parquet scans, so historical
numbers are not directly comparable.

Measuring both paths on the same day surfaced the dominant cost on this
benchmark's mixed-dtype data: **the physical engine converts pass-through
object-string columns to Arrow on output even when no operation touches
them**. Isolated: `with_columns(v1+v2)` at 10M rows takes 55 ms on a
numeric-only frame but 379 ms with one untouched object-string column
present (eager path: 50 ms). The data here carries several string columns,
so physical-engine numbers below are dominated by that conversion, not by
kernel speed. Eliminating pass-through conversion is the top engine
priority (see docs/ROADMAP.md); on numeric-only data (see `bench_sort.py`)
the physical engine beats eager pandas across every sort shape.

Same-day eager-path reference points at 10M rows for comparison with the
table below: filter 183 ms, with_columns 87 ms, groupby-sum 208 ms,
str.lower 253 ms (a 2.6x *win* over Polars on the eager path),
single-key sort 2848 ms, inner join 1541 ms.

## Summary

**Speedup interpretation:** Values > 1.0 mean lazy pandas is faster, < 1.0 mean Polars is faster.

### Category Summary

| Category | Avg Speedup | Best LP | Best Polars |
|----------|-------------|---------|-------------|
| filter_project | 0.11x | 0.23x | 0.00x |
| aggregation | 0.08x | 0.11x | 0.03x |
| string | 0.27x | 0.42x | 0.04x |
| limit | 0.00x | 0.01x | 0.00x |
| sort | 0.20x | 0.24x | 0.17x |
| join | 0.06x | 0.08x | 0.05x |
| parquet_scan | 0.27x | 0.65x | 0.07x |

## Filter Project

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(value1 > 0) | 1,000,000 | 20.48 | 3.92 | *0.19x* |
| filter + select(3 cols) | 1,000,000 | 58.51 | 1.61 | *0.03x* |
| filter(val1 > 0 AND int_val < 500) | 1,000,000 | 13.83 | 2.74 | *0.20x* |
| with_columns(value1 + value2) | 1,000,000 | 104.43 | 0.51 | *0.00x* |
| filter(value1 > 0) | 10,000,000 | 219.44 | 36.86 | *0.17x* |
| filter + select(3 cols) | 10,000,000 | 591.91 | 16.86 | *0.03x* |
| filter(val1 > 0 AND int_val < 500) | 10,000,000 | 126.70 | 28.62 | *0.23x* |
| with_columns(value1 + value2) | 10,000,000 | 1225.70 | 9.72 | *0.01x* |

## Aggregation

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| groupby(group).sum(value1) | 1,000,000 | 30.89 | 2.60 | *0.08x* |
| groupby(group).agg(sum, mean, max, count) | 1,000,000 | 108.12 | 11.89 | *0.11x* |
| groupby(category[100]).sum() | 1,000,000 | 51.01 | 2.43 | *0.05x* |
| groupby(group).sum(value1) | 10,000,000 | 288.41 | 18.52 | *0.06x* |
| groupby(group).agg(sum, mean, max, count) | 10,000,000 | 971.37 | 111.05 | *0.11x* |
| groupby(category[100]).sum() | 10,000,000 | 494.50 | 15.90 | *0.03x* |

## String

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| filter(text.contains('foo')) | 1,000,000 | 48.53 | 17.66 | *0.36x* |
| with_columns(text.lower()) | 1,000,000 | 155.68 | 65.43 | *0.42x* |
| with_columns(text.len()) | 1,000,000 | 107.99 | 4.47 | *0.04x* |
| filter(text.contains('foo')) | 10,000,000 | 470.06 | 174.03 | *0.37x* |
| with_columns(text.lower()) | 10,000,000 | 1670.18 | 654.30 | *0.39x* |
| with_columns(text.len()) | 10,000,000 | 1095.39 | 47.62 | *0.04x* |

## Limit

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| head(10) | 1,000,000 | 1.37 | 0.01 | *0.01x* |
| head(100) | 1,000,000 | 1.48 | 0.01 | *0.01x* |
| head(1000) | 1,000,000 | 1.38 | 0.01 | *0.01x* |
| head(10) | 10,000,000 | 9.42 | 0.01 | *0.00x* |
| head(100) | 10,000,000 | 9.65 | 0.01 | *0.00x* |
| head(1000) | 10,000,000 | 9.76 | 0.01 | *0.00x* |

## Sort

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| sort(value1) | 1,000,000 | 215.47 | 36.65 | *0.17x* |
| sort(group, value1) | 1,000,000 | 398.00 | 87.43 | *0.22x* |
| sort(value1) | 10,000,000 | 2672.76 | 465.47 | *0.17x* |
| sort(group, value1) | 10,000,000 | 6197.65 | 1483.53 | *0.24x* |

## Join

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| inner_join(1000000x100000) | 1,000,000 | 607.27 | 38.84 | *0.06x* |
| left_join(1000000x100000) | 1,000,000 | 771.78 | 46.29 | *0.06x* |
| inner_join(10000000x1000000) | 10,000,000 | 8585.52 | 712.38 | *0.08x* |
| left_join(10000000x1000000) | 10,000,000 | 11902.45 | 631.37 | *0.05x* |

## Parquet Scan

| Operation | Rows | Lazy Pandas (ms) | Polars (ms) | Speedup |
|-----------|------|------------------|-------------|---------|
| scan.filter.head(1000) | 2,964,624 | 51.08 | 33.18 | *0.65x* |
| scan.select(3).filter | 2,964,624 | 85.28 | 7.64 | *0.09x* |
| glob_scan.head(1000) | 17,787,744 | 144.45 | 10.06 | *0.07x* |

## Key Findings

### Where Polars is Faster

- **head(100)** (limit): 1331.62x faster
- **head(10)** (limit): 1255.82x faster
- **head(1000)** (limit): 1232.44x faster
- **with_columns(value1 + value2)** (filter_project): 203.89x faster
- **head(100)** (limit): 190.41x faster
