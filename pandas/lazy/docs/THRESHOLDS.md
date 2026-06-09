# Lazy Pandas: Threshold Catalog for Cost-Based Optimization

> **M2 note**: the cost model has one home now. Option-backed runtime
> thresholds (`compute.lazy.*`) are defined in
> `pandas/lazy/optimize/config.py` (registered in
> `pandas/core/config_init.py`); engine decision constants — morsel
> size, parallel-kernel minimums, key-encoding thresholds — live in
> **`pandas/lazy/cost.py`** with their measured provenance, and the
> scattered modules import from there. Calibration tunes that module,
> not per-file constants. `compute.lazy.morsel_size` is the newest key.

This document catalogs all threshold points in the lazy pandas execution engine
that are used for cost-based optimization decisions.

## Overview

The lazy pandas engine makes runtime decisions based on data characteristics.
These decisions use configurable thresholds that can be:

1. **Set globally** via `pd.set_option("compute.lazy.*")`
2. **Loaded from a file** via `ThresholdConfig.from_file()`
3. **Calibrated for hardware** via `scripts/calibrate_lazy_thresholds.py`
4. **Adapted at runtime** via the adaptive threshold system

## Quick Start

```python
import pandas as pd

# View current thresholds
pd.get_option("compute.lazy.filter_arrow_threshold")  # 50000

# Set a custom threshold
pd.set_option("compute.lazy.filter_arrow_threshold", 25000)

# Use calibrated thresholds from file
from pandas.lazy.optimize.config import ThresholdConfig, set_threshold_config
config = ThresholdConfig.from_file("calibrated_thresholds.json")
set_threshold_config(config)

# Enable adaptive thresholds (experimental)
pd.set_option("compute.lazy.adaptive_thresholds", True)
```

## All Configurable Thresholds

### Backend Selection Thresholds

| Option | Default | Type | Description |
|--------|---------|------|-------------|
| `compute.lazy.filter_arrow_threshold` | 50,000 | int | Row count above which Arrow filter is used instead of NumPy |
| `compute.lazy.groupby_arrow_row_threshold` | 100,000 | int | Row count above which Arrow groupby is preferred |
| `compute.lazy.groupby_arrow_cardinality_threshold` | 100 | int | Group count above which Arrow groupby is preferred |
| `compute.lazy.arrow_majority_fraction` | 0.5 | float | Fraction of Arrow columns needed to prefer Arrow backend |
| `compute.lazy.morsel_size` | 131,072 | int or None | Morsel (batch) size for pipeline execution; `None` uses the cost-model default (`MORSEL_SIZE`) |

> The engine cost constants that are **not** option-backed (`MIN_PARALLEL_ROWS`,
> `MAX_WORKERS`, `PARALLEL_SORT_MIN_ROWS`, `PARALLEL_TAKE_MIN_ROWS`,
> `ARROW_MULTIKEY_SORT_MIN_ROWS`, `MIN_ENCODE_ROWS`, …) live in
> `pandas/lazy/cost.py` with measured provenance; change them there, not via
> options.

### Parallelization Thresholds

| Option | Default | Type | Description |
|--------|---------|------|-------------|
| `compute.lazy.parallel_expr_threshold` | 8 | int | Minimum expressions before parallelizing projections |
| `compute.lazy.parallel_chunk_size` | 65,536 | int | Batch size for streaming execution (L3 cache friendly) |

### NumExpr Fusion Thresholds

| Option | Default | Type | Description |
|--------|---------|------|-------------|
| `compute.lazy.numexpr_min_elements` | 100,000 | int | Minimum array elements for NumExpr to be beneficial |
| `compute.lazy.numexpr_min_operations` | 2 | int | Minimum fusible operations for NumExpr benefit |
| `compute.lazy.numexpr_large_array_threshold` | 1,000,000 | int | Size where memory bandwidth becomes dominant |

### Adaptive Threshold System

| Option | Default | Type | Description |
|--------|---------|------|-------------|
| `compute.lazy.adaptive_thresholds` | False | bool | Enable runtime-adaptive threshold tuning |

## Detailed Threshold Specifications

### T1: Filter Arrow Threshold (`filter_arrow_threshold`)

**Purpose**: Determine when to use Arrow's SIMD-optimized filter vs NumPy.

**Decision Logic**:
```python
if row_count > filter_arrow_threshold:
    use_arrow_filter()
else:
    use_numpy_filter()
```

**Factors Affecting Optimal Value**:
- Row count (primary factor)
- Number of filter conditions (more conditions → lower threshold)
- Selectivity (low selectivity → Arrow more beneficial)
- Data types (string filters always use Arrow)

**Benchmark Findings**:
- Arrow has ~10ms setup overhead
- Crossover typically 10K-50K rows depending on hardware
- Arrow is 2-3x faster for large data due to SIMD

---

### T2/T3: GroupBy Arrow Thresholds

**Purpose**: Determine when Arrow's native groupby outperforms pandas Cython.

**Decision Logic**:
```python
if row_count > groupby_arrow_row_threshold:
    use_arrow_groupby()
elif n_groups > groupby_arrow_cardinality_threshold:
    use_arrow_groupby()
else:
    use_pandas_groupby()
```

**Benchmark Findings**:
```
| Rows  | Groups | Pandas (ms) | Arrow (ms) | Winner      |
|-------|--------|-------------|------------|-------------|
| 10K   | 10     | 0.97        | 13.19      | Pandas 13x  |
| 10K   | 100    | 0.20        | 0.13       | Arrow 1.6x  |
| 100K  | 100    | 1.51        | 0.66       | Arrow 2.3x  |
| 1M    | 1000   | 14.8        | 5.6        | Arrow 2.6x  |
```

**Key Insight**: Arrow has ~10-15ms JIT overhead. Optimal when:
- rows > 100K, OR
- rows > 10K AND groups > 100

---

### T4: Parallel Expression Threshold (`parallel_expr_threshold`)

**Purpose**: Determine when ThreadPoolExecutor overhead is worth it.

**Decision Logic**:
```python
if n_expressions >= parallel_expr_threshold:
    parallelize_projection()
else:
    execute_sequentially()
```

**Factors Affecting Optimal Value**:
- Number of expressions
- Expression complexity
- Array size (larger arrays benefit more)
- CPU core count

**Benchmark Findings**:
- ThreadPoolExecutor overhead: ~1-2ms per submission
- Typically beneficial at 8+ expressions with large data

---

### T5: Parallel Chunk Size (`parallel_chunk_size`)

**Purpose**: Set batch size for streaming execution.

**Considerations**:
- Default 65,536 is L3 cache friendly
- Smaller batches = lower memory, more overhead
- Larger batches = higher memory, less overhead

---

### T6/T7/T8: NumExpr Fusion Thresholds

**Purpose**: Determine when NumExpr's expression compilation is beneficial.

**Decision Logic**:
```python
should_use = (
    array_size >= numexpr_min_elements and
    n_operations >= numexpr_min_operations
)

if array_size >= numexpr_large_array_threshold:
    benefit_factor = 1.2  # Memory-bound, NumExpr very effective
elif array_size >= numexpr_min_elements:
    benefit_factor = 1.0  # Standard benefit
else:
    benefit_factor = 0.5  # Limited benefit
```

**Benchmark Findings**:
- NumExpr has ~0.5ms compilation overhead
- Break-even at ~100K elements for simple expressions
- 1.5-3x speedup for large arrays due to cache efficiency

---

## Configuration System

### ThresholdConfig Class

The configuration system is implemented in `pandas/lazy/optimize/config.py`:

```python
from pandas.lazy.optimize.config import (
    ThresholdConfig,
    get_threshold_config,
    set_threshold_config,
    reset_threshold_config,
)

# Create custom config
config = ThresholdConfig(
    filter_arrow_threshold=25_000,
    groupby_arrow_row_threshold=50_000,
    parallel_expr_threshold=10,
)

# Set as global default
set_threshold_config(config)

# Save to file
config.to_file("my_thresholds.json")

# Load from file
loaded = ThresholdConfig.from_file("my_thresholds.json")
```

### Integration with pandas Options

All thresholds are accessible via pandas' standard options system:

```python
import pandas as pd

# Get current value
pd.get_option("compute.lazy.filter_arrow_threshold")

# Set new value
pd.set_option("compute.lazy.filter_arrow_threshold", 25000)

# Temporary override
with pd.option_context("compute.lazy.filter_arrow_threshold", 10000):
    result = ldf.filter(col("x") > 0).collect()

# Reset to default
pd.reset_option("compute.lazy.filter_arrow_threshold")

# View all lazy options
pd.describe_option("compute.lazy")
```

---

## Calibration System

### Running Calibration Benchmarks

```bash
python scripts/calibrate_lazy_thresholds.py --output my_config.json
```

The calibration script:
1. Benchmarks each operation at various data sizes
2. Detects crossover points where backends become equal
3. Outputs optimal thresholds as JSON

### Calibration Output Format

```json
{
    "metadata": {
        "timestamp": "2026-01-15T10:30:00Z",
        "hardware": {
            "cpu": "Apple M1 Pro",
            "cores": 10,
            "memory_gb": 32
        },
        "software": {
            "python": "3.12.0",
            "numpy": "2.0.0",
            "pyarrow": "16.0.0"
        }
    },
    "thresholds": {
        "filter_arrow_threshold": 9000,
        "groupby_arrow_row_threshold": 10000,
        "numexpr_min_elements": 250000,
        "parallel_expr_threshold": 10
    }
}
```

---

## Adaptive Threshold System

The adaptive system automatically tunes thresholds based on runtime statistics.

### How It Works

1. **Statistics Collection**: Records execution times for each operation/backend
2. **EMA Smoothing**: Uses exponential moving average to smooth measurements
3. **Crossover Detection**: Estimates where backends have equal performance
4. **Threshold Adjustment**: Adjusts thresholds based on observed performance

### Usage

```python
import pandas as pd

# Enable adaptive thresholds
pd.set_option("compute.lazy.adaptive_thresholds", True)

# Run workload - system learns optimal thresholds
for _ in range(100):
    result = df.select().filter(col("x") > 0).collect(use_physical_planner=True)

# Check learned statistics
from pandas.lazy.optimize.adaptive import get_adaptive_manager
stats = get_adaptive_manager().get_statistics()
print(stats)
```

### Adaptive Statistics

```python
{
    "total_executions": 100,
    "operations": {
        "filter": {
            "arrow_samples": 80,
            "numpy_samples": 20,
            "arrow_throughput_ema": 150000.0,  # rows/ms
            "numpy_throughput_ema": 80000.0,
            "crossover_estimate": 45000,
            "confidence": 0.9
        }
    }
}
```

---

## Implementation Status

### Completed

- [x] Phase 1: Threshold identification and documentation
- [x] Phase 2: Configuration system (`ThresholdConfig`)
- [x] Phase 2: pandas options integration (`pd.set_option`)
- [x] Phase 3: Calibration benchmark suite
- [x] Phase 4: Adaptive threshold system

### Files

| File | Description |
|------|-------------|
| `pandas/lazy/optimize/config.py` | ThresholdConfig class and global config management |
| `pandas/lazy/optimize/adaptive.py` | Adaptive threshold manager |
| `pandas/core/config_init.py` | pandas options registration |
| `scripts/calibrate_lazy_thresholds.py` | Calibration benchmark suite |
| `scripts/evaluate_adaptive_thresholds.py` | Adaptive system evaluation |

---

## Best Practices

### For Most Users

Use the default thresholds - they work well for typical workloads:

```python
# Just use lazy pandas normally
result = df.select().filter(col("x") > 0).collect()
```

### For Performance-Critical Applications

Run calibration on your specific hardware:

```bash
python scripts/calibrate_lazy_thresholds.py --output config.json
```

Then load the calibrated config:

```python
from pandas.lazy.optimize.config import ThresholdConfig, set_threshold_config
config = ThresholdConfig.from_file("config.json")
set_threshold_config(config)
```

### For Variable Workloads

Consider enabling adaptive thresholds:

```python
pd.set_option("compute.lazy.adaptive_thresholds", True)
```

Note: Adaptive thresholds have some overhead and work best for long-running
applications where the workload varies over time.
