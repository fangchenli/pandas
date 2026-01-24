# Lazy Pandas Benchmark Suite

Comprehensive performance benchmarks for lazy pandas execution engine, comparing lazy vs eager execution, Arrow vs NumPy backends, and competitive analysis against Polars and DuckDB.

## Quick Start

```bash
# Run all core benchmarks
python run_all.py

# Run specific benchmark
python run_all.py filter
python run_all.py join

# Run with optional benchmarks (Polars comparison, NYC Taxi)
python run_all.py --all

# Use specific Python interpreter
python run_all.py --python /path/to/python
```

## Benchmark Categories

### Planning Overhead Analysis

| Benchmark | File | Description |
|-----------|------|-------------|
| Planning Phases | `bench_planning_phases.py` | **Critical** - Breaks down lazy overhead into plan construction, optimization passes, and physical planning |
| Optimizer Quality | `bench_optimizer_quality.py` | **Critical** - Measures plan quality (nodes reduced, filters fused, etc.) for regression detection |
| Cache Effects | `bench_cache_effects.py` | **Critical** - Cold vs warm cache for interactive use (query, file I/O, spill) |
| Selectivity | `bench_selectivity.py` | **Critical** - Performance at 1%, 10%, 50%, 90% selectivity |
| Join Edge Cases | `bench_join_edge_cases.py` | **Critical** - Skew, cardinality mismatch, wrong-side build |

### Core Operations

| Benchmark | File | Description |
|-----------|------|-------------|
| Kernel Overhead | `bench_kernel_overhead.py` | Measures dispatch overhead vs direct NumPy/Arrow calls |
| Conversion | `bench_conversion.py` | Arrow ↔ NumPy conversion costs |
| Filter | `bench_filter.py` | Single and chained filter operations |
| Select | `bench_select.py` | Column selection and projection |
| Arithmetic | `bench_arithmetic.py` | Numeric expressions and computations |
| String Ops | `bench_string_ops.py` | String methods (lower, contains, replace) |
| Expressions | `bench_expressions.py` | Complex nested expressions, case-when |
| Aggregations | `bench_aggregations.py` | GroupBy and aggregate operations |

### Advanced Operations

| Benchmark | File | Description |
|-----------|------|-------------|
| Pipelines | `bench_pipelines.py` | Multi-step ETL-like query patterns |
| Advanced Ops | `bench_advanced_ops.py` | Window functions, distinct, complex queries |
| Kernels | `bench_kernels.py` | Individual kernel performance |
| Join | `bench_join.py` | Join operations with build/probe optimization |
| Streaming | `bench_streaming.py` | Streaming execution and early termination |
| Memory | `bench_memory.py` | Memory usage and spill-to-disk |
| Optimizer | `bench_optimizer.py` | Query optimizer performance |
| NumExpr Fusion | `bench_numexpr_fusion.py` | Expression fusion with NumExpr |

### Competitive Analysis

| Benchmark | File | Description | Requirements |
|-----------|------|-------------|--------------|
| vs Polars | `bench_vs_polars.py` | Head-to-head comparison with Polars | `pip install polars` |
| NYC Taxi | `bench_nyc_taxi.py` | Real-world dataset benchmark | NYC Taxi parquet files |
| 1TB Transactions | `bench_1tb_transactions.py` | Larger-than-memory workloads | S3 access or generated data |

## Benchmark Infrastructure

### Shared Utilities (`shared.py`)

#### Timing Functions

```python
from pandas.lazy.benchmarks.shared import timeit, benchmark

# Simple timing (returns mean, std in ms)
mean_ms, std_ms = timeit(lambda: df.filter(...).collect())

# Detailed timing (returns dict with mean, std, min)
result = benchmark(lambda: df.filter(...).collect())
print(f"Mean: {result['mean_ms']:.2f} ms")
```

#### Data Generation

```python
from pandas.lazy.benchmarks.shared import (
    create_test_data,      # Basic numeric + string columns
    create_grouped_data,   # Data with grouping columns
    create_sales_data,     # Realistic sales/ETL data
    create_string_data,    # String-heavy data
    create_join_data,      # Left/right tables for joins
    create_wide_data,      # Many columns for projection tests
)

# Create 1M rows with Arrow-backed dtypes
df = create_test_data(n_rows=1_000_000, use_arrow=True, seed=42)

# Create join data
left, right = create_join_data(n_left=100_000, n_right=10_000, n_keys=5000)
```

#### Output Helpers

```python
from pandas.lazy.benchmarks.shared import (
    print_header,      # Section headers
    print_subheader,   # Subsection headers
    print_result,      # "Label    X.XX ms ± Y.YY ms"
    print_speedup,     # "Speedup: N.NNx"
)
```

### Data Directory

Benchmarks use `pandas/data/` for test datasets:

```python
from pandas.lazy.benchmarks.shared import DATA_DIR, ensure_data_dir

# DATA_DIR = /path/to/pandas/data/
data_path = ensure_data_dir()  # Creates if needed
```

## Benchmark Structure

Each benchmark file follows a consistent pattern:

```python
#!/usr/bin/env python3
"""
Benchmark: <Name>

<Description of what this benchmark tests>

Usage:
    python pandas/lazy/benchmarks/bench_<name>.py
"""

from pandas.lazy.benchmarks.shared import (
    benchmark,
    create_test_data,
    print_header,
    print_result,
)

def benchmark_<operation>():
    """Benchmark description."""
    print_header("Operation Name")

    for size in [10_000, 100_000, 1_000_000]:
        df = create_test_data(size, use_arrow=True)

        # Eager baseline
        eager_result = benchmark(lambda: df.query("a > 50"))

        # Lazy execution
        lazy_result = benchmark(lambda: df.select().filter(...).collect())

        print_result("Eager", eager_result["mean_ms"], eager_result["std_ms"])
        print_result("Lazy", lazy_result["mean_ms"], lazy_result["std_ms"])

if __name__ == "__main__":
    benchmark_<operation>()
```

## Planning Phase Analysis

The `bench_planning_phases.py` benchmark is critical for understanding lazy overhead:

```
Lazy Execution Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Plan Construction    │ Build logical plan tree from API     │
├─────────────────────────┼───────────────────────────────────────┤
│ 2. Optimization         │ 12 passes: ConstantFolding,          │
│                         │ FilterFusion, PredicatePushdown,     │
│                         │ ProjectionPruning, CSE, etc.         │
├─────────────────────────┼───────────────────────────────────────┤
│ 3. Physical Planning    │ Convert logical → physical plan      │
│                         │ Apply operator fusion                 │
├─────────────────────────┼───────────────────────────────────────┤
│ 4. Execution            │ Actually process data                 │
└─────────────────────────┴───────────────────────────────────────┘
```

### Measuring Each Phase Independently

```python
from pandas.lazy.optimize import Optimizer
from pandas.lazy.physical import PhysicalPlanner

# Phase 1: Plan construction only
lf = df.select().filter(col("a") > 0.5)
plan = lf._plan  # No execution

# Phase 2: Optimization only
optimizer = Optimizer()
optimized = optimizer.optimize(plan)

# Phase 3: Physical planning only
planner = PhysicalPlanner()
physical = planner.plan(optimized)

# Phase 4: Full execution
result = lf.collect()
```

### Per-Pass Timing

```python
from pandas.lazy.optimize import (
    ConstantFolding, FilterFusion, PredicatePushdown,
    CommonSubexpressionElimination, ...
)

# Time each pass individually
for pass_cls in [ConstantFolding, FilterFusion, PredicatePushdown, ...]:
    pass_instance = pass_cls()
    # time: pass_instance.optimize(plan)
```

### Fast Path Heuristics

Based on planning phase analysis, consider skipping:

| Condition | Skip | Savings |
|-----------|------|---------|
| Single filter/select | Entire optimizer | 60-80% of overhead |
| No repeated expressions | CSE pass | 10-20% of overhead |
| Flat query (no nesting) | PredicatePushdown | 5-10% of overhead |
| No limits in query | LimitPushdown, SortLimitToTopK | 5% of overhead |
| Data < 10K rows | Use eager execution | 100% of overhead |

## Optimizer Quality Metrics

The `bench_optimizer_quality.py` benchmark measures plan quality, not speed:

```json
{
  "total_nodes_before": 62,
  "total_nodes_after": 39,
  "total_nodes_reduced": 23,
  "filters_fused": 9,
  "projects_merged": 13,
  "topk_conversions": 3
}
```

### What Each Pass Should Do

| Pass | Expected Effect | Test Query |
|------|-----------------|------------|
| FilterFusion | 3 filters → 1 filter | `filter_fusion_3` |
| SortLimitToTopK | Sort+Limit → TopK | `sort_limit_topk` |
| DeadCodeElimination | Remove unused projections | `dead_code` |
| ProjectionPruning | Prune unused columns | `projection_pruning` |
| PredicatePushdown | Push filters toward scan | `predicate_pushdown` |

### Regression Detection

Run this benchmark when:
- Adding new optimization passes
- Refactoring existing passes
- "Simplifying" code that might affect pushdown

If expectations fail, the optimizer has regressed:

```
Query                      Metric              Expected   Actual   Status
-------------------------------------------------------------------------
filter_fusion_3            filters_fused              2        2     PASS
sort_limit_topk            sort_limit_to_topk      True     True     PASS
```

## Cold vs Warm Cache Effects

The `bench_cache_effects.py` benchmark measures interactive performance:

| Scenario | Cold | Warm | Speedup |
|----------|------|------|---------|
| Repeated query | 19 ms | 6 ms | 3.1x |
| Parquet read | 34 ms | 10 ms | 3.5x |
| Spill/reload | 1 ms | 0.01 ms | 100x+ |
| Streaming | 22 ms | 11 ms | 2.0x |

### Why This Matters for Interactive Use

```
User session:
  Query 1 (cold):  df.filter(...).collect()  → 20ms
  Query 2 (warm):  df.filter(...).collect()  → 6ms   (3x faster!)
  Query 3 (warm):  df.filter(...).collect()  → 6ms
```

### Key Recommendations

1. **Reuse LazyDataFrame objects** - don't recreate for same query
2. **Convert to Arrow once** at load time
3. **Keep frequently accessed data in memory**
4. **Parquet benefits from OS file cache** on repeated reads

## Selectivity Analysis

The `bench_selectivity.py` benchmark measures performance across filter selectivities:

| Selectivity | Rows Passing | Streaming Speedup | Notes |
|-------------|--------------|-------------------|-------|
| 1% | 10K of 1M | 2.1x | Early termination shines |
| 10% | 100K of 1M | 2.0x | Streaming still helps |
| 50% | 500K of 1M | 1.0x | No streaming benefit |
| 90% | 900K of 1M | 1.7x | Memory bandwidth limited |

### Key Findings

1. **Streaming excels at low selectivity** - 2x faster at 1-10% selectivity
2. **Eager wins for simple filters** - Lazy overhead dominates simple operations
3. **Lazy improves relatively at high selectivity** - Gap narrows (0.28x → 0.70x)

### Optimization Heuristics

```
Selectivity     Action
──────────────────────────────────────────
< 10%           Aggressive pushdown + streaming
10-50%          Push filters, streaming optional
> 50%           Consider keeping predicates local
+ head(N)       ALWAYS push filters (early termination)
```

### Cardinality Estimation

For optimizer decisions, estimate selectivity using:
- Histogram-based estimation (for known columns)
- Sample-based estimation (for complex predicates)
- Default 10% assumption (when unknown)

## Join Edge Cases

The `bench_join_edge_cases.py` benchmark tests challenging join scenarios:

### Scenarios Tested

| Scenario | Description | Challenge |
|----------|-------------|-----------|
| Uniform | Baseline - evenly distributed keys | None (control) |
| Skewed 50% | One key has 50% of rows | Hot partition problem |
| Both skewed | Both sides skewed on same key | Worst case for hash join |
| Zipf | Realistic power-law distribution | Natural skew pattern |
| Tiny build | 1K build × 500K probe | Optimal hash join case |
| Wrong-side | Large left, small right | Tests optimizer swap |
| Explosion | Many-to-many with duplicates | Result size explosion |
| No matches | Disjoint key ranges | Wasted probe effort |

### What This Exposes

1. **Skew handling** - Does Grace hash join detect pathological partitions?
2. **Build/probe selection** - Does optimizer pick smaller side for build?
3. **Spill behavior** - How does performance degrade under memory pressure?
4. **Adaptive execution** - Does it fall back to sort-merge when needed?

### Grace Hash Join Health Indicators

```
Healthy:
  - Partition sizes uniform (skew ratio < 3x)
  - Few empty partitions (< 10%)
  - Max partition fits in memory

Pathological (consider sort-merge fallback):
  - Skew ratio > 10x
  - Empty partitions > 50%
  - Max partition > memory budget
  - Recursive spilling required
```

## Key Metrics

### What We Measure

1. **Execution Time** - Wall-clock time in milliseconds
2. **Memory Usage** - Peak memory consumption
3. **Throughput** - Rows/second for streaming operations
4. **Speedup Ratio** - Baseline time / test time

### Methodology

- **Warmup runs**: 1-3 runs before timing (JIT compilation, caching)
- **Timed runs**: 5-7 runs for statistical significance
- **GC disabled**: During timing to reduce variance
- **Reproducible data**: Fixed random seed (42) for all generated data

## Expected Results

### Where Lazy Wins

| Scenario | Typical Speedup | Reason |
|----------|-----------------|--------|
| Sequential filters | 1.5-2x | Filter fusion |
| Filter + head(N) | 10-100x | Early termination |
| Multi-step pipelines | 1.2-1.5x | Reduced materialization |
| Large Arrow data | 1.1-1.3x | Arrow-native execution |
| Memory-constrained | N/A | Streaming + spill |

### Where Eager Wins

| Scenario | Typical Slowdown | Reason |
|----------|------------------|--------|
| Single operations | 2-5x slower | Planning overhead |
| Small data (<10K) | 2-10x slower | Fixed overhead dominates |
| NumPy data paths | 1.5-2x slower | Conversion overhead |

### Performance Guidelines

```
Data Size        Recommendation
─────────────────────────────────
< 10K rows       Use eager (overhead dominates)
10K - 100K       Profile both (workload dependent)
100K - 1M        Lazy often wins for pipelines
> 1M rows        Lazy + streaming recommended
> Memory         Lazy + streaming required
```

## Adding New Benchmarks

### 1. Create Benchmark File

```python
#!/usr/bin/env python3
"""
Benchmark: <New Feature>

<Description>

Usage:
    python pandas/lazy/benchmarks/bench_new_feature.py
"""

from pandas.lazy.benchmarks.shared import benchmark, create_test_data

def benchmark_new_feature():
    # Implementation
    pass

if __name__ == "__main__":
    benchmark_new_feature()
```

### 2. Add to run_all.py

```python
BENCHMARKS = [
    # ... existing benchmarks ...
    "bench_new_feature.py",
]
```

### 3. Update Documentation

Add entry to the benchmark table in this file and `README.md`.

## Profiling

For detailed profiling beyond benchmarks:

```python
# Profile physical planner
python profile_physical_planner.py

# Use Python profiler
python -m cProfile -s cumtime bench_filter.py

# Memory profiling
python -m memory_profiler bench_memory.py
```

## Continuous Integration

Benchmarks are designed to run in CI for regression detection:

```yaml
# Example CI configuration
benchmark:
  script:
    - python run_all.py
  artifacts:
    paths:
      - benchmark_results.json
```

## Results Storage

Benchmark results can be saved as JSON for tracking:

```python
import json
from pathlib import Path

results = {
    "timestamp": "2024-01-23T12:00:00",
    "commit": "abc123",
    "benchmarks": {
        "filter_1m_rows": {"mean_ms": 12.5, "std_ms": 0.8},
        # ...
    }
}

Path("benchmark_results.json").write_text(json.dumps(results, indent=2))
```

Existing result files:
- `benchmark_lazy_vs_polars_results.json` - Polars comparison
- `benchmark_streaming_results.json` - Streaming benchmarks

## Hardware Considerations

Results vary significantly by hardware:

| Factor | Impact |
|--------|--------|
| CPU cores | Parallel operations scale |
| L3 cache | Filter/scan performance |
| Memory bandwidth | Large data operations |
| SSD speed | Spill-to-disk operations |
| NUMA topology | Multi-socket scaling |

Document your hardware when reporting results:
```python
import platform
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")
print(f"Python: {platform.python_version()}")
```
