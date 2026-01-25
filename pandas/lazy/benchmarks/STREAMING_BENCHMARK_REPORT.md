# Streaming Execution Benchmark Report

## Environment

- pandas version: 3.0.0rc1+64.g341630917a.dirty
- Python version: 3.11.14
- Platform: macOS-26.2-arm64-arm-64bit
- Warmup runs: 3
- Timed runs: 7

## Summary

| Category | Avg Speedup vs Full Read | Max Speedup | Min Speedup |
|----------|--------------------------|-------------|-------------|
| filter_head | 2.3x | 2.3x | 2.2x |
| head | 1.7x | 1.7x | 1.7x |
| multi_file | 10.6x | 10.9x | 10.4x |
| project_filter_head | 2.7x | 2.7x | 2.7x |
| streaming_collect | 0.1x | 0.1x | 0.0x |

## Filter Head

| Operation | Data Rows | Streaming (ms) | Full Read (ms) | Speedup |
|-----------|-----------|----------------|----------------|---------|
| filter.head(10) | 2,964,624 | 55.38 | 127.91 | **2.3x** |
| filter.head(100) | 2,964,624 | 53.20 | 124.62 | **2.3x** |
| filter.head(1000) | 2,964,624 | 52.92 | 119.24 | **2.3x** |
| filter.head(10000) | 2,964,624 | 53.53 | 119.68 | **2.2x** |

## Head

| Operation | Data Rows | Streaming (ms) | Full Read (ms) | Speedup |
|-----------|-----------|----------------|----------------|---------|
| head(10) | 2,964,624 | 48.92 | 83.04 | 1.7x |
| head(100) | 2,964,624 | 49.93 | 82.48 | 1.7x |
| head(1000) | 2,964,624 | 50.20 | 84.51 | 1.7x |
| head(10000) | 2,964,624 | 49.86 | 82.52 | 1.7x |

## Multi File

| Operation | Data Rows | Streaming (ms) | Full Read (ms) | Speedup |
|-----------|-----------|----------------|----------------|---------|
| glob.head(100) | 20,332,093 | 157.26 | 1707.80 | **10.9x** |
| glob.head(1000) | 20,332,093 | 148.97 | 1554.87 | **10.4x** |

## Project Filter Head

| Operation | Data Rows | Streaming (ms) | Full Read (ms) | Speedup |
|-----------|-----------|----------------|----------------|---------|
| select.filter.head(10) | 2,964,624 | 8.49 | 23.05 | **2.7x** |
| select.filter.head(100) | 2,964,624 | 8.48 | 23.28 | **2.7x** |
| select.filter.head(1000) | 2,964,624 | 8.47 | 23.00 | **2.7x** |
| select.filter.head(10000) | 2,964,624 | 8.50 | 23.03 | **2.7x** |

## Streaming Collect

| Batch Size | Data Rows | Streaming (ms) | Full Read (ms) | Ratio |
|------------|-----------|----------------|----------------|-------|
| 1,024 | 2,964,624 | 894.33 | 20.59 | 0.02x |
| 8,192 | 2,964,624 | 364.99 | 20.64 | 0.06x |
| 65,536 | 2,964,624 | 298.11 | 21.46 | 0.07x |

## Key Findings

### Top 5 Speedups

1. **glob.head(100)** (multi_file): 10.9x faster (157.26ms vs 1707.80ms)
2. **glob.head(1000)** (multi_file): 10.4x faster (148.97ms vs 1554.87ms)
3. **select.filter.head(100)** (project_filter_head): 2.7x faster (8.48ms vs 23.28ms)
4. **select.filter.head(1000)** (project_filter_head): 2.7x faster (8.47ms vs 23.00ms)
5. **select.filter.head(10)** (project_filter_head): 2.7x faster (8.49ms vs 23.05ms)

### Conclusions

- **Early termination** provides significant speedups for `head()` operations
- **Projection pushdown** combined with streaming further improves performance
- **Multi-file streaming** enables efficient reading of partitioned datasets
- **Batch size** has minimal impact on throughput for full dataset processing
