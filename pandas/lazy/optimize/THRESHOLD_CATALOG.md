# Lazy Pandas: Threshold Catalog for Cost-Based Optimization

This document catalogs all threshold points in the lazy pandas execution engine that require calibration through benchmarking.

## Overview

The lazy pandas engine makes numerous runtime decisions based on data characteristics. Currently, these decisions use hardcoded thresholds derived from initial benchmarking. A systematic calibration system would allow these thresholds to be tuned for different hardware configurations and PyArrow/NumPy versions.

## Threshold Categories

### Category 1: Backend Selection Thresholds

These thresholds determine when to use Arrow vs NumPy/Pandas backends.

| ID | Name | Current Value | Location | Parameters | Description |
|----|------|---------------|----------|------------|-------------|
| T1 | `filter_arrow_threshold` | 50,000 rows | `physical.py:796` | row_count | Use Arrow filter when rows > threshold |
| T2 | `groupby_arrow_threshold` | 100,000 rows | (proposed) | row_count | Use Arrow groupby when rows > threshold |
| T3 | `groupby_cardinality_threshold` | 100 groups | (proposed) | n_groups | Use Arrow groupby when groups > threshold |

### Category 2: Parallelization Thresholds

These thresholds determine when parallel execution is beneficial.

| ID | Name | Current Value | Location | Parameters | Description |
|----|------|---------------|----------|------------|-------------|
| T4 | `parallel_expr_threshold` | 8 expressions | `physical.py:201` | n_expressions | Parallelize projection when expressions >= threshold |
| T5 | `parallel_chunk_size` | 65,536 rows | `physical.py:1580` | batch_size | Chunk size for streaming execution |

### Category 3: Fusion Thresholds

These thresholds determine when expression fusion is beneficial.

| ID | Name | Current Value | Location | Parameters | Description |
|----|------|---------------|----------|------------|-------------|
| T6 | `numexpr_min_elements` | 100,000 | `numexpr_fusion.py:55` | array_size | Minimum elements for NumExpr to be beneficial |
| T7 | `numexpr_min_operations` | 2 | `numexpr_fusion.py:209` | n_operations | Minimum fusible operations for NumExpr |
| T8 | `numexpr_large_array` | 1,000,000 | `numexpr_fusion.py:460` | array_size | Threshold where memory bandwidth dominates |

### Category 4: Conversion Cost Thresholds

These thresholds determine when format conversion is worth the cost.

| ID | Name | Current Value | Location | Parameters | Description |
|----|------|---------------|----------|------------|-------------|
| T9 | `arrow_majority_fraction` | 0.5 | `router.py:186` | column_ratio | Fraction of Arrow columns to prefer Arrow backend |
| T10 | `conversion_break_even` | (TBD) | `router.py` | row_count, n_cols | Rows where conversion cost equals operation savings |

## Detailed Threshold Specifications

### T1: Filter Arrow Threshold

**Purpose**: Determine when to convert NumPy arrays to Arrow for filtering.

**Current Logic** (physical.py:796-798):
```python
n_rows = len(first_arr)
if n_rows > 50_000:
    use_arrow_filter = True
```

**Factors to Calibrate**:
- Row count crossover point
- Number of filter conditions (more conditions → lower threshold)
- Selectivity (low selectivity → Arrow more beneficial)
- Data types (string filters always use Arrow)

**Benchmark Parameters**:
```python
{
    "row_counts": [1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    "n_conditions": [1, 2, 3, 5],
    "selectivity": [0.01, 0.1, 0.5, 0.9],
    "dtypes": ["int64", "float64", "string"],
}
```

**Expected Output**:
```python
filter_thresholds = {
    "base_threshold": 50000,
    "per_condition_adjustment": -10000,  # Lower threshold per extra condition
    "selectivity_adjustment": {...},
}
```

---

### T2/T3: GroupBy Arrow Thresholds

**Purpose**: Determine when Arrow's native groupby outperforms Pandas Cython.

**Current Logic**: Static (always uses pandas), proposed to be dynamic.

**Benchmark Data** (from OPTIMIZATION_NEXT_STEPS.md):
```
| Rows  | Groups | Pandas (ms) | Arrow (ms) | Winner      |
|-------|--------|-------------|------------|-------------|
| 10K   | 10     | 0.97        | 13.19      | Pandas 13x  |
| 10K   | 100    | 0.20        | 0.13       | Arrow 1.6x  |
| 100K  | 100    | 1.51        | 0.66       | Arrow 2.3x  |
| 1M    | 1000   | 14.8        | 5.6        | Arrow 2.6x  |
```

**Key Insight**: Arrow has ~10-15ms JIT/setup overhead. Crossover occurs when:
- rows > 10K AND groups > 100, OR
- rows > 100K (regardless of groups)

**Factors to Calibrate**:
- Row count crossover point
- Cardinality (number of groups) crossover point
- Number of aggregation columns
- Aggregation function type (sum vs mean vs std)
- Data type (int64 vs float64)

**Benchmark Parameters**:
```python
{
    "row_counts": [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000],
    "n_groups": [10, 100, 1000, 10000],
    "n_agg_columns": [1, 3, 5, 10],
    "agg_funcs": ["sum", "mean", "min", "max", "std", "count"],
    "dtypes": ["int64", "float64"],
}
```

**Expected Output**:
```python
groupby_thresholds = {
    "row_threshold": 100000,
    "cardinality_threshold": 100,
    "combined_rule": "rows > 10000 AND groups > 100",
}
```

---

### T4: Parallel Expression Threshold

**Purpose**: Determine when ThreadPoolExecutor overhead is worth parallelizing projections.

**Current Logic** (physical.py:201-204):
```python
parallel_threshold: int = 8
use_parallel = n_exprs >= context.parallel_threshold
```

**Factors to Calibrate**:
- Number of expressions crossover point
- Expression complexity (simple vs complex)
- Array size (larger arrays benefit more from parallelism)
- CPU core count

**Benchmark Parameters**:
```python
{
    "n_expressions": [2, 4, 6, 8, 10, 15, 20],
    "row_counts": [10000, 100000, 1000000],
    "expr_complexity": ["simple", "moderate", "complex"],
    "n_cores": [4, 8, 16],
}
```

**Expected Output**:
```python
parallel_thresholds = {
    "min_expressions": 8,
    "min_rows_per_expr": 10000,
    "complexity_adjustment": {"simple": +2, "complex": -2},
}
```

---

### T6/T7/T8: NumExpr Fusion Thresholds

**Purpose**: Determine when NumExpr's expression compilation is beneficial.

**Current Logic** (numexpr_fusion.py:55, 209, 457-462):
```python
MIN_ELEMENTS_FOR_NUMEXPR = 100_000

if array_size < MIN_ELEMENTS_FOR_NUMEXPR:
    size_factor = 0.5
elif array_size < 1_000_000:
    size_factor = 1.0
else:
    size_factor = 1.2
```

**Factors to Calibrate**:
- Minimum array size for NumExpr benefit
- Minimum operations for fusion benefit
- Size brackets for benefit scaling
- Expression complexity (depth of expression tree)

**Benchmark Parameters**:
```python
{
    "array_sizes": [1000, 10000, 50000, 100000, 500000, 1000000, 5000000],
    "n_operations": [1, 2, 3, 5, 10],
    "operation_types": ["add", "multiply", "compare", "mixed"],
    "dtypes": ["float64", "int64"],
}
```

**Expected Output**:
```python
numexpr_thresholds = {
    "min_elements": 100000,
    "min_operations": 2,
    "size_brackets": [
        {"max_size": 100000, "factor": 0.5},
        {"max_size": 1000000, "factor": 1.0},
        {"max_size": float("inf"), "factor": 1.2},
    ],
}
```

---

## Configuration Schema

The calibration benchmark suite should produce a configuration in this format:

```python
@dataclass
class ThresholdConfig:
    """Configuration for all execution thresholds."""

    # Backend selection
    filter_arrow_threshold: int = 50_000
    groupby_arrow_row_threshold: int = 100_000
    groupby_arrow_cardinality_threshold: int = 100

    # Parallelization
    parallel_expr_threshold: int = 8
    parallel_chunk_size: int = 65_536

    # NumExpr fusion
    numexpr_min_elements: int = 100_000
    numexpr_min_operations: int = 2
    numexpr_large_array_threshold: int = 1_000_000

    # Size factors for benefit estimation
    numexpr_size_factors: dict[str, float] = field(default_factory=lambda: {
        "small": 0.5,   # < numexpr_min_elements
        "medium": 1.0,  # numexpr_min_elements to numexpr_large_array_threshold
        "large": 1.2,   # > numexpr_large_array_threshold
    })

    # Conversion costs
    arrow_majority_fraction: float = 0.5

    @classmethod
    def from_file(cls, path: str) -> "ThresholdConfig":
        """Load configuration from JSON/YAML file."""
        ...

    def to_file(self, path: str) -> None:
        """Save configuration to file."""
        ...
```

## Calibration Output Format

The benchmark suite should output results in this format:

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
            "pyarrow": "16.0.0",
            "numexpr": "2.9.0"
        }
    },
    "benchmarks": {
        "filter_backend": {
            "crossover_rows": 47500,
            "confidence": 0.95,
            "raw_data": [...]
        },
        "groupby_backend": {
            "crossover_rows": 95000,
            "crossover_cardinality": 85,
            "confidence": 0.92,
            "raw_data": [...]
        },
        ...
    },
    "recommended_config": {
        "filter_arrow_threshold": 50000,
        "groupby_arrow_row_threshold": 100000,
        "groupby_arrow_cardinality_threshold": 100,
        ...
    }
}
```

## Implementation Phases

### Phase 1: Threshold Identification (This Document) ✅
- Catalog all decision points
- Document current thresholds
- Define calibration parameters

### Phase 2: Configuration System
- Create `ThresholdConfig` dataclass
- Add `get_threshold()` helper functions
- Wire thresholds through `ExecutionContext`
- Add `pd.set_option("compute.lazy.*")` integration

### Phase 3: Calibration Benchmark Suite
- Create `scripts/calibrate_thresholds.py`
- Implement micro-benchmarks for each threshold
- Implement crossover point detection
- Generate configuration file output

## Next Steps

1. **Step 2**: Build the configuration system
   - Create `pandas/lazy/optimize/config.py`
   - Define `ThresholdConfig` dataclass
   - Wire into `ExecutionContext`

2. **Step 3**: Build calibration suite
   - Create `scripts/calibrate_lazy_thresholds.py`
   - Implement benchmarks for each threshold category
   - Add crossover point detection algorithm
