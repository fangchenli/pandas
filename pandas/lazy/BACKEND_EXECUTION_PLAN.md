# Backend-Specific Execution Plan

## Overview

This document describes the plan to implement backend-specific execution (Arrow vs NumPy)
for the lazy pandas physical planner. When `use_physical_planner=True`, instead of calling
pandas DataFrame/Series operations, we will:

1. Work with raw arrays (`dict[str, Array]`) instead of DataFrames
2. Call Arrow compute kernels or NumPy operations directly
3. Track the index as a special `__index__` column
4. Only construct the final DataFrame at `collect()` time

## Design Decisions

| Decision | Choice |
|----------|--------|
| Granularity | Per-node, with overrides for high-value ops |
| Format detection | Check `dtype` for `ArrowDtype` |
| Mixed formats | Allow (some columns Arrow, some NumPy) |
| Index handling | Store as `__index__` column(s) |
| Intermediate data | `dict[str, Array]` not `DataFrame` |
| Override threshold | Any string/null ops → use Arrow |

## Architecture

```
PhysicalScan
    │
    ▼
dict[str, Array]  ←── {"__index__": [...], "col_a": [...], "col_b": [...]}
    │
    ▼
PhysicalFilter (backend=arrow)
    │
    ▼
dict[str, Array]  ←── filtered arrays
    │
    ▼
PhysicalProject (backend=numpy)
    │
    ▼
dict[str, Array]  ←── projected arrays
    │
    ▼
collect() → pd.DataFrame  ←── Restore index, construct DataFrame once
```

## New Type Definitions

```python
# pandas/lazy/backends/types.py

from typing import Union
import numpy as np
import pyarrow as pa

# Array types we work with
# NOTE: pandas Arrow-backed columns use pa.ChunkedArray internally
# (accessed via ArrowExtensionArray._ndarray)
ArrayLike = Union[np.ndarray, pa.Array, pa.ChunkedArray]

# For type checking any PyArrow array type
PyArrowArray = Union[pa.Array, pa.ChunkedArray]

# The intermediate data structure passed between nodes
ArrayDict = dict[str, ArrayLike]

# Index column naming convention
INDEX_COL_PREFIX = "__index__"
INDEX_COL_NAME = "__index__"  # Single index
# MultiIndex: __index_0__, __index_1__, etc.
```

## ChunkedArray Handling

pandas Arrow-backed columns store data as `pa.ChunkedArray` (not single `pa.Array`).
When extracting from a DataFrame:

```python
df = pd.DataFrame({"a": pd.array(["x", "y"], dtype="string[pyarrow]")})
arr = df["a"].array._ndarray  # This is a pa.ChunkedArray
```

**Strategy: Keep ChunkedArray, let PyArrow handle it**

Most `pyarrow.compute` functions accept `ChunkedArray` directly:
```python
pc.utf8_lower(chunked)   # Works
pc.add(chunked, 1)       # Works
pc.filter(chunked, mask) # Works
```

Only call `combine_chunks()` when an operation requires contiguous memory.

```python
# pandas/lazy/backends/convert.py

def extract_arrow_array(col) -> PyArrowArray:
    """Extract Arrow array from pandas column."""
    if hasattr(col, "array") and hasattr(col.array, "_ndarray"):
        # pandas Series with ArrowExtensionArray
        return col.array._ndarray
    elif hasattr(col, "_ndarray"):
        # ArrowExtensionArray directly
        return col._ndarray
    return col

def to_contiguous(arr: PyArrowArray) -> pa.Array:
    """Unchunk if needed (only when required)."""
    if isinstance(arr, pa.ChunkedArray):
        return arr.combine_chunks()
    return arr
```

## ChunkedArray Alignment

Binary operations on two `ChunkedArray`s may have misaligned chunk boundaries:

```python
arr1 = pa.chunked_array([[1, 2], [3, 4, 5]])      # chunks: [2, 3]
arr2 = pa.chunked_array([[10, 20, 30], [40, 50]]) # chunks: [3, 2]
```

**Strategy: Trust PyArrow + verify in debug mode**

- `pyarrow.compute` functions handle misaligned chunks internally
- Columns from the same DataFrame scan should naturally be aligned
- Add verification in debug/strict mode to catch issues early

```python
# pandas/lazy/backends/convert.py

def verify_chunk_alignment(arrays: ArrayDict) -> bool:
    """Verify all ChunkedArrays have compatible chunk structures (debug only)."""
    chunked = [
        arr for arr in arrays.values()
        if isinstance(arr, pa.ChunkedArray)
    ]

    if len(chunked) < 2:
        return True

    # Check all have same number of chunks and lengths
    num_chunks = chunked[0].num_chunks
    if not all(arr.num_chunks == num_chunks for arr in chunked):
        return False

    for i in range(num_chunks):
        lengths = [arr.chunk(i).length for i in chunked]
        if len(set(lengths)) > 1:
            return False

    return True
```

**When alignment could break:**
- After filter (mask applied per-chunk, result may have different structure)
- After join (combining arrays from different sources)
- After concat (appending chunks from different arrays)

For these cases, PyArrow handles it internally. Only optimize if profiling shows issues.

## Directory Structure

```
pandas/lazy/
├── backends/
│   ├── __init__.py         # Backend registry and utilities
│   ├── types.py            # Type definitions (ArrayLike, ArrayDict)
│   ├── router.py           # Backend decision logic
│   ├── convert.py          # Format conversion utilities
│   ├── arrow_kernels.py    # Arrow compute implementations
│   └── numpy_kernels.py    # NumPy implementations
├── physical.py             # Updated to return ArrayDict
├── eval.py                 # Keep for fallback
└── ...
```

## Implementation Phases

### Phase 1: Infrastructure (`pandas/lazy/backends/`)

**Files to create:**

1. **`types.py`** - Type definitions
   ```python
   ArrayLike = Union[np.ndarray, pa.Array, pa.ChunkedArray]
   ArrayDict = dict[str, ArrayLike]
   INDEX_COL_PREFIX = "__index__"
   ```

2. **`convert.py`** - Format conversion utilities
   ```python
   def is_arrow_backed(arr: ArrayLike) -> bool:
       """Check if array is Arrow-backed."""

   def to_arrow(arr: ArrayLike) -> pa.Array:
       """Convert array to Arrow format."""

   def to_numpy(arr: ArrayLike) -> np.ndarray:
       """Convert array to NumPy format."""

   def get_column_formats(arrays: ArrayDict) -> dict[str, str]:
       """Get format ("arrow" or "numpy") for each column."""
   ```

3. **`router.py`** - Backend decision logic
   ```python
   # Operation classifications
   ARROW_PREFERRED_OPS = {
       "str_lower", "str_upper", "str_contains", "str_replace",
       "str_slice", "str_strip", "str_len", "str_startswith", "str_endswith",
       "is_null", "is_not_null", "fill_null", "coalesce",
   }

   NUMPY_PREFERRED_OPS = set()  # Nothing forced to NumPy yet

   NEUTRAL_OPS = {
       "add", "subtract", "multiply", "divide", "equal", "less", "greater",
       # ... follow input format
   }

   def decide_node_backend(
       node: PhysicalPlan,
       input_formats: dict[str, str],
       context: ExecutionContext,
   ) -> Literal["arrow", "numpy"]:
       """Decide backend for a physical node."""
   ```

### Phase 2: Index Handling

**Update `PhysicalScan.execute()`:**

```python
def execute(self, context: ExecutionContext) -> ArrayDict:
    df = self.df
    arrays = {}

    # Extract data columns
    for col in df.columns:
        arrays[col] = df[col].array  # Get underlying array

    # Extract index as column(s)
    if isinstance(df.index, pd.MultiIndex):
        for i in range(df.index.nlevels):
            col_name = f"__index_{i}__"
            arrays[col_name] = df.index.get_level_values(i).to_numpy()
        context.index_names = list(df.index.names)
    else:
        arrays["__index__"] = df.index.to_numpy()
        context.index_names = [df.index.name]

    return arrays
```

**Update `ExecutionContext`:**

```python
@dataclass
class ExecutionContext:
    preferred_backend: Literal["auto", "arrow", "numpy"] = "auto"
    strict: bool = False
    cache: dict[int, Any] = field(default_factory=dict)

    # New: track index metadata
    index_names: list[str | None] = field(default_factory=list)
    index_is_multi: bool = False
```

**Final DataFrame construction:**

```python
def arrays_to_dataframe(arrays: ArrayDict, context: ExecutionContext) -> pd.DataFrame:
    """Convert ArrayDict back to DataFrame with proper index."""

    # Separate index columns from data columns
    index_cols = {}
    data_cols = {}

    for name, arr in arrays.items():
        if name.startswith("__index"):
            index_cols[name] = arr
        else:
            data_cols[name] = arr

    # Build DataFrame from data columns
    df = pd.DataFrame(data_cols)

    # Restore index
    if len(index_cols) == 0:
        # No index was stored (e.g., after aggregation)
        pass
    elif len(index_cols) == 1 and "__index__" in index_cols:
        # Simple index
        df.index = pd.Index(index_cols["__index__"], name=context.index_names[0])
    else:
        # MultiIndex
        levels = []
        for i in range(len(index_cols)):
            levels.append(index_cols[f"__index_{i}__"])
        df.index = pd.MultiIndex.from_arrays(levels, names=context.index_names)

    return df
```

### Phase 3: Kernel Dispatcher

**`backends/__init__.py`:**

```python
from typing import Callable
from pandas.lazy.backends.types import ArrayLike

# Kernel registry: (function_name, backend) -> implementation
_KERNELS: dict[tuple[str, str], Callable] = {}

def register_kernel(func_name: str, backend: str):
    """Decorator to register a kernel implementation."""
    def decorator(fn: Callable) -> Callable:
        _KERNELS[(func_name, backend)] = fn
        return fn
    return decorator

def get_kernel(func_name: str, backend: str) -> Callable | None:
    """Get kernel implementation, or None if not found."""
    return _KERNELS.get((func_name, backend))

def dispatch_kernel(
    func_name: str,
    backend: str,
    *args,
    **kwargs
) -> ArrayLike:
    """Dispatch to appropriate kernel."""
    kernel = get_kernel(func_name, backend)
    if kernel is None:
        raise NotImplementedError(
            f"No {backend} kernel for {func_name}"
        )
    return kernel(*args, **kwargs)
```

### Phase 4: Arrow Kernels (High-Value)

**`backends/arrow_kernels.py`:**

```python
import pyarrow as pa
import pyarrow.compute as pc
from pandas.lazy.backends import register_kernel

# String operations - biggest performance wins

@register_kernel("str_lower", "arrow")
def arrow_str_lower(arr: pa.Array) -> pa.Array:
    return pc.utf8_lower(arr)

@register_kernel("str_upper", "arrow")
def arrow_str_upper(arr: pa.Array) -> pa.Array:
    return pc.utf8_upper(arr)

@register_kernel("str_len", "arrow")
def arrow_str_len(arr: pa.Array) -> pa.Array:
    return pc.utf8_length(arr)

@register_kernel("str_contains", "arrow")
def arrow_str_contains(arr: pa.Array, pattern: str, regex: bool = True) -> pa.Array:
    if regex:
        return pc.match_substring_regex(arr, pattern)
    else:
        return pc.match_substring(arr, pattern)

@register_kernel("str_replace", "arrow")
def arrow_str_replace(
    arr: pa.Array, pattern: str, replacement: str, regex: bool = True
) -> pa.Array:
    if regex:
        return pc.replace_substring_regex(arr, pattern, replacement)
    else:
        return pc.replace_substring(arr, pattern, replacement)

@register_kernel("str_strip", "arrow")
def arrow_str_strip(arr: pa.Array) -> pa.Array:
    return pc.utf8_trim_whitespace(arr)

@register_kernel("str_startswith", "arrow")
def arrow_str_startswith(arr: pa.Array, pattern: str) -> pa.Array:
    return pc.starts_with(arr, pattern)

@register_kernel("str_endswith", "arrow")
def arrow_str_endswith(arr: pa.Array, pattern: str) -> pa.Array:
    return pc.ends_with(arr, pattern)

# Null operations - native Arrow support

@register_kernel("is_null", "arrow")
def arrow_is_null(arr: pa.Array) -> pa.Array:
    return pc.is_null(arr)

@register_kernel("is_not_null", "arrow")
def arrow_is_not_null(arr: pa.Array) -> pa.Array:
    return pc.is_valid(arr)

@register_kernel("fill_null", "arrow")
def arrow_fill_null(arr: pa.Array, fill_value) -> pa.Array:
    return pc.fill_null(arr, fill_value)

@register_kernel("coalesce", "arrow")
def arrow_coalesce(*arrays: pa.Array) -> pa.Array:
    return pc.coalesce(*arrays)

# Arithmetic operations

@register_kernel("add", "arrow")
def arrow_add(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.add(left, right)

@register_kernel("subtract", "arrow")
def arrow_subtract(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.subtract(left, right)

@register_kernel("multiply", "arrow")
def arrow_multiply(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.multiply(left, right)

@register_kernel("divide", "arrow")
def arrow_divide(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.divide(left, right)

# Comparison operations

@register_kernel("equal", "arrow")
def arrow_equal(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.equal(left, right)

@register_kernel("not_equal", "arrow")
def arrow_not_equal(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.not_equal(left, right)

@register_kernel("less", "arrow")
def arrow_less(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.less(left, right)

@register_kernel("less_equal", "arrow")
def arrow_less_equal(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.less_equal(left, right)

@register_kernel("greater", "arrow")
def arrow_greater(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.greater(left, right)

@register_kernel("greater_equal", "arrow")
def arrow_greater_equal(left: pa.Array, right: pa.Array | float) -> pa.Array:
    return pc.greater_equal(left, right)

# Aggregations

@register_kernel("sum", "arrow")
def arrow_sum(arr: pa.Array) -> float:
    return pc.sum(arr).as_py()

@register_kernel("mean", "arrow")
def arrow_mean(arr: pa.Array) -> float:
    return pc.mean(arr).as_py()

@register_kernel("min", "arrow")
def arrow_min(arr: pa.Array) -> float:
    return pc.min(arr).as_py()

@register_kernel("max", "arrow")
def arrow_max(arr: pa.Array) -> float:
    return pc.max(arr).as_py()

@register_kernel("count", "arrow")
def arrow_count(arr: pa.Array) -> int:
    return pc.count(arr).as_py()
```

### Phase 5: NumPy Kernels

**`backends/numpy_kernels.py`:**

```python
import numpy as np
from pandas.lazy.backends import register_kernel

# Arithmetic operations

@register_kernel("add", "numpy")
def numpy_add(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.add(left, right)

@register_kernel("subtract", "numpy")
def numpy_subtract(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.subtract(left, right)

@register_kernel("multiply", "numpy")
def numpy_multiply(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.multiply(left, right)

@register_kernel("divide", "numpy")
def numpy_divide(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.divide(left, right)

# Comparison operations

@register_kernel("equal", "numpy")
def numpy_equal(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.equal(left, right)

@register_kernel("not_equal", "numpy")
def numpy_not_equal(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.not_equal(left, right)

@register_kernel("less", "numpy")
def numpy_less(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.less(left, right)

@register_kernel("less_equal", "numpy")
def numpy_less_equal(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.less_equal(left, right)

@register_kernel("greater", "numpy")
def numpy_greater(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.greater(left, right)

@register_kernel("greater_equal", "numpy")
def numpy_greater_equal(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.greater_equal(left, right)

# Aggregations

@register_kernel("sum", "numpy")
def numpy_sum(arr: np.ndarray) -> float:
    return np.nansum(arr)

@register_kernel("mean", "numpy")
def numpy_mean(arr: np.ndarray) -> float:
    return np.nanmean(arr)

@register_kernel("min", "numpy")
def numpy_min(arr: np.ndarray) -> float:
    return np.nanmin(arr)

@register_kernel("max", "numpy")
def numpy_max(arr: np.ndarray) -> float:
    return np.nanmax(arr)

@register_kernel("count", "numpy")
def numpy_count(arr: np.ndarray) -> int:
    return np.count_nonzero(~np.isnan(arr)) if arr.dtype.kind == 'f' else len(arr)

# String operations (via pandas StringMethods as fallback)
# These are slow in NumPy, so we don't implement them - router should prefer Arrow
```

### Phase 6: Update Physical Node Execution

**Change physical nodes to work with `ArrayDict`:**

```python
# Before (current)
def execute(self, context: ExecutionContext) -> DataFrame:
    input_df = self.input.execute(context)
    ...

# After
def execute(self, context: ExecutionContext) -> ArrayDict:
    input_arrays = self.input.execute(context)
    ...
```

**Example: PhysicalFilter**

```python
@dataclass
class PhysicalFilter(PhysicalPlan):
    input: PhysicalPlan
    predicate: Expr
    schema: Schema
    backend: Literal["auto", "arrow", "numpy"] = "auto"

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.convert import to_arrow, to_numpy

        input_arrays = self.input.execute(context)

        # Evaluate predicate to get boolean mask
        mask = self._evaluate_predicate(input_arrays, context)

        # Filter all arrays (including index)
        result = {}
        for name, arr in input_arrays.items():
            if isinstance(arr, pa.Array):
                # Arrow filter
                result[name] = pc.filter(arr, mask)
            else:
                # NumPy filter
                result[name] = arr[mask]

        return result

    def _evaluate_predicate(self, arrays: ArrayDict, context) -> ArrayLike:
        """Evaluate predicate expression to boolean array."""
        # Use kernel dispatcher for expression evaluation
        ...
```

**Example: PhysicalProject**

```python
@dataclass
class PhysicalProject(PhysicalPlan):
    input: PhysicalPlan
    exprs: tuple[Expr, ...]
    schema: Schema
    backend: Literal["auto", "arrow", "numpy"] = "auto"

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas.lazy.backends.router import should_override_backend
        from pandas.lazy.expr import extract_output_name

        input_arrays = self.input.execute(context)

        result = {}

        # Preserve index columns
        for name, arr in input_arrays.items():
            if name.startswith("__index"):
                result[name] = arr

        # Evaluate each expression
        for expr in self.exprs:
            output_name = extract_output_name(expr)

            # Check if this expression needs backend override
            expr_backend = self.backend
            if should_override_backend(expr):
                expr_backend = "arrow"  # String/null ops force Arrow

            result[output_name] = self._evaluate_expr(
                expr, input_arrays, expr_backend, context
            )

        return result
```

### Phase 7: Expression Evaluator for Arrays

Create a new evaluator that works on `ArrayDict` instead of `DataFrame`:

**`backends/array_eval.py`:**

```python
class ArrayEvaluator:
    """Evaluates IR expressions against ArrayDict."""

    def __init__(self, arrays: ArrayDict, backend: str = "auto"):
        self._arrays = arrays
        self._backend = backend

    def evaluate(self, node: IRNode) -> ArrayLike:
        """Evaluate an IR node and return array result."""
        from pandas.lazy.ir import Alias, Call, Cast, FieldRef, Literal

        if isinstance(node, FieldRef):
            return self._arrays[node.name]

        elif isinstance(node, Literal):
            return node.value

        elif isinstance(node, Alias):
            return self.evaluate(node.arg)

        elif isinstance(node, Call):
            return self._evaluate_call(node)

        elif isinstance(node, Cast):
            arr = self.evaluate(node.arg)
            return self._cast(arr, node.target_dtype)

        else:
            raise NotImplementedError(f"Unknown IR node: {type(node)}")

    def _evaluate_call(self, node: Call) -> ArrayLike:
        """Evaluate a function call."""
        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.router import ARROW_PREFERRED_OPS

        func = node.function
        args = [self.evaluate(arg) for arg in node.args]

        # Determine backend for this operation
        if func in ARROW_PREFERRED_OPS:
            backend = "arrow"
        elif self._backend == "auto":
            # Follow format of first array argument
            backend = self._infer_backend(args)
        else:
            backend = self._backend

        # Convert args to target backend if needed
        converted_args = [self._ensure_backend(a, backend) for a in args]

        # Dispatch to kernel
        return dispatch_kernel(func, backend, *converted_args, **node.kwargs)
```

## Updated Execution Flow

```
LazyDataFrame.collect(use_physical_planner=True)
    │
    ▼
Optimizer.optimize(logical_plan)
    │
    ▼
PhysicalPlanner.plan(optimized_plan)
    │  - For each node, decide backend
    │  - ARROW_PREFERRED_OPS force Arrow
    │  - Others follow input format or user preference
    ▼
PhysicalPlan tree with backend hints
    │
    ▼
execute_physical_plan(physical_plan, context)
    │
    ▼
PhysicalScan.execute()
    │  - Extract DataFrame to ArrayDict
    │  - Store index as __index__ column(s)
    ▼
ArrayDict: {"__index__": [...], "col_a": [...], ...}
    │
    ▼
PhysicalFilter.execute()
    │  - Evaluate predicate using ArrayEvaluator
    │  - Filter all arrays with boolean mask
    ▼
PhysicalProject.execute()
    │  - Evaluate expressions using ArrayEvaluator
    │  - Dispatch to Arrow/NumPy kernels
    ▼
... more nodes ...
    │
    ▼
Final ArrayDict
    │
    ▼
arrays_to_dataframe(arrays, context)
    │  - Separate index columns from data
    │  - Build DataFrame
    │  - Restore Index/MultiIndex with names
    ▼
pd.DataFrame (returned to user)
```

## Testing Strategy

1. **Unit tests for kernels**: Test each Arrow/NumPy kernel independently
2. **Backend equivalence**: Run same query with both backends, compare results
3. **Index preservation**: Test that index survives all operations correctly
4. **MultiIndex**: Specific tests for MultiIndex handling
5. **Mixed formats**: Test columns with different backends in same ArrayDict
6. **Override behavior**: Test that string ops always use Arrow

## Implementation Order

1. **Phase 1**: Create `backends/` directory and type definitions
2. **Phase 2**: Implement index extraction and restoration
3. **Phase 3**: Kernel registry infrastructure
4. **Phase 4**: Arrow string kernels (biggest win)
5. **Phase 5**: NumPy kernels for arithmetic/comparison
6. **Phase 6**: Update PhysicalScan to return ArrayDict
7. **Phase 7**: Update other physical nodes
8. **Phase 8**: Array-based expression evaluator
9. **Phase 9**: Integration testing
10. **Phase 10**: Performance benchmarking

## Open Questions (To Resolve During Implementation)

1. **Chunked arrays**: Should we unchunk `pa.ChunkedArray` or handle natively?
   - Recommendation: Unchunk for simplicity, optimize later if needed

2. **Type preservation**: How to preserve nullable int types through operations?
   - Recommendation: Track original dtypes in context, restore at end

3. **Memory management**: When to copy vs view arrays?
   - Recommendation: Default to views, copy only when mutation needed

4. **Error handling**: What to do when kernel doesn't exist?
   - Recommendation: Fall back to pandas Evaluator with warning (unless strict mode)
