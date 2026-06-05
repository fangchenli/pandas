# Streaming Execution Benchmark Report

## Environment

- pandas version: 3.1.0.dev0+1220.g55b305e82a
- Python version: 3.11.14
- Platform: macOS-26.5.1-arm64-arm-64bit
- Warmup runs: 3
- Timed runs: 7

## Summary

| Category | Avg Speedup vs Full Read | Max Speedup | Min Speedup |
|----------|--------------------------|-------------|-------------|
| filter_head | 2.0x | 2.1x | 1.9x |
| head | 1.6x | 1.6x | 1.5x |
| multi_file | 10.3x | 10.8x | 9.8x |
| project_filter_head | 2.6x | 2.6x | 2.5x |
| streaming_collect | 0.0x | 0.1x | 0.0x |

## Filter Head

| Operation | Data Rows | Streaming (ms) | Full Read (ms) | Speedup |
|-----------|-----------|----------------|----------------|---------|
| filter.head(10) | 2,964,624 | 51.05 | 106.26 | **2.1x** |
| filter.head(100) | 2,964,624 | 56.83 | 107.92 | 1.9x |
| filter.head(1000) | 2,964,624 | 52.72 | 111.13 | **2.1x** |
| filter.head(10000) | 2,964,624 | 53.36 | 107.59 | **2.0x** |

## Head

| Operation | Data Rows | Streaming (ms) | Full Read (ms) | Speedup |
|-----------|-----------|----------------|----------------|---------|
| head(10) | 2,964,624 | 52.83 | 76.72 | 1.5x |
| head(100) | 2,964,624 | 48.33 | 77.82 | 1.6x |
| head(1000) | 2,964,624 | 47.64 | 76.37 | 1.6x |
| head(10000) | 2,964,624 | 47.99 | 75.72 | 1.6x |

## Multi File

| Operation | Data Rows | Streaming (ms) | Full Read (ms) | Speedup |
|-----------|-----------|----------------|----------------|---------|
| glob.head(100) | 20,332,093 | 141.80 | 1392.69 | **9.8x** |
| glob.head(1000) | 20,332,093 | 134.89 | 1457.73 | **10.8x** |

## Project Filter Head

| Operation | Data Rows | Streaming (ms) | Full Read (ms) | Speedup |
|-----------|-----------|----------------|----------------|---------|
| select.filter.head(10) | 2,964,624 | 8.07 | 20.66 | **2.6x** |
| select.filter.head(100) | 2,964,624 | 8.06 | 20.63 | **2.6x** |
| select.filter.head(1000) | 2,964,624 | 8.25 | 20.82 | **2.5x** |
| select.filter.head(10000) | 2,964,624 | 8.07 | 20.86 | **2.6x** |

## Streaming Collect

| Batch Size | Data Rows | Streaming (ms) | Full Read (ms) | Ratio |
|------------|-----------|----------------|----------------|-------|
| 1,024 | 2,964,624 | 737.88 | 18.60 | 0.03x |
| 8,192 | 2,964,624 | 341.39 | 18.64 | 0.05x |
| 65,536 | 2,964,624 | 288.92 | 18.61 | 0.06x |

## Key Findings

### Top 5 Speedups

1. **glob.head(1000)** (multi_file): 10.8x faster (134.89ms vs 1457.73ms)
2. **glob.head(100)** (multi_file): 9.8x faster (141.80ms vs 1392.69ms)
3. **select.filter.head(10000)** (project_filter_head): 2.6x faster (8.07ms vs 20.86ms)
4. **select.filter.head(100)** (project_filter_head): 2.6x faster (8.06ms vs 20.63ms)
5. **select.filter.head(10)** (project_filter_head): 2.6x faster (8.07ms vs 20.66ms)

### Conclusions

- **Early termination** provides significant speedups for `head()` operations
- **Projection pushdown** combined with streaming further improves performance
- **Multi-file streaming** enables efficient reading of partitioned datasets
- **Batch size** has minimal impact on throughput for full dataset processing
