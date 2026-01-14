"""
Physical plan representation for lazy pandas.

This module defines physical plan nodes that represent concrete execution
strategies. Unlike logical plans which describe *what* to compute, physical
plans describe *how* to compute it, including:

- Which backend (Arrow/NumPy) to use for each operation
- Which algorithm to use (e.g., hash join vs merge join)
- Memory and parallelism hints

The physical planner converts an optimized logical plan into a physical plan,
making execution decisions based on:
- Data characteristics (types, cardinality estimates)
- Operation requirements (some ops require specific backends)
- User preferences (engine hints)
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.lazy.backends.types import ArrayDict
    from pandas.lazy.expr import Expr
    from pandas.lazy.plan import LogicalPlan
    from pandas.lazy.types import Schema


# =============================================================================
# Physical Plan Nodes
# =============================================================================


@dataclass
class PhysicalPlan(ABC):
    """
    Base class for physical plan nodes.

    Physical plans represent concrete execution strategies with backend
    and algorithm choices already made.
    """

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute this physical plan node.

        Parameters
        ----------
        context : ExecutionContext
            Execution context with runtime state.

        Returns
        -------
        ArrayDict
            The result as a dictionary mapping column names to arrays.
            Index columns are stored as "__index__" or "__index_N__".
        """
        ...

    @abstractmethod
    def children(self) -> list[PhysicalPlan]:
        """Return child plan nodes."""
        ...

    @property
    @abstractmethod
    def output_schema(self) -> Schema:
        """Return the output schema of this node."""
        ...


@dataclass
class ExecutionContext:
    """
    Runtime context for physical plan execution.

    Holds state needed during execution, like intermediate results
    and configuration.
    """

    # Preferred backend for operations that support multiple
    preferred_backend: Literal["auto", "arrow", "numpy"] = "auto"

    # Whether to use strict mode (fail on fallbacks)
    strict: bool = False

    # Cache for intermediate results (for CSE)
    cache: dict[int, Any] = field(default_factory=dict)

    # Index metadata (populated by PhysicalScan)
    index_names: list[str | None] = field(default_factory=list)
    index_is_multi: bool = False


# =============================================================================
# Scan Nodes (Data Sources)
# =============================================================================


@dataclass
class PhysicalScan(PhysicalPlan):
    """
    Physical scan of a DataFrame source.

    This is a leaf node that reads from an in-memory DataFrame.
    Extracts arrays from DataFrame and stores index as special columns.
    """

    df: DataFrame
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        import pandas as pd
        from pandas.lazy.backends.convert import extract_array
        from pandas.lazy.backends.types import index_col_name

        arrays: ArrayDict = {}

        # Extract data columns
        for col in self.df.columns:
            arrays[col] = extract_array(self.df[col])

        # Extract index as special column(s)
        if isinstance(self.df.index, pd.MultiIndex):
            context.index_is_multi = True
            context.index_names = list(self.df.index.names)
            for i in range(self.df.index.nlevels):
                col_name = index_col_name(i)
                arrays[col_name] = self.df.index.get_level_values(i).to_numpy()
        else:
            context.index_is_multi = False
            context.index_names = [self.df.index.name]
            arrays[index_col_name()] = self.df.index.to_numpy()

        return arrays

    def children(self) -> list[PhysicalPlan]:
        return []

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Projection Nodes
# =============================================================================


@dataclass
class PhysicalProject(PhysicalPlan):
    """
    Physical projection (column selection/computation).

    Executes expressions to produce output columns.
    """

    input: PhysicalPlan
    exprs: tuple[Expr, ...]
    schema: Schema
    backend: Literal["auto", "arrow", "numpy"] = "auto"

    def execute(self, context: ExecutionContext) -> ArrayDict:
        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.types import is_index_col
        from pandas.lazy.expr import extract_output_name

        input_arrays = self.input.execute(context)

        # Use ArrayEvaluator for direct array-based evaluation
        evaluator = ArrayEvaluator(input_arrays, preferred_backend=self.backend)

        result: ArrayDict = {}

        # Preserve index columns
        for name, arr in input_arrays.items():
            if is_index_col(name):
                result[name] = arr

        # Evaluate expressions
        for expr in self.exprs:
            name = extract_output_name(expr)
            value = evaluator.evaluate(expr._ir)
            # Ensure result is an array (not scalar for column expressions)
            if not isinstance(value, (np.ndarray, pa.Array, pa.ChunkedArray)):
                # Scalar result - broadcast to array length
                arr_len = len(next(iter(input_arrays.values())))
                value = np.full(arr_len, value)
            result[name] = value

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Filter Nodes
# =============================================================================


@dataclass
class PhysicalFilter(PhysicalPlan):
    """
    Physical filter (row selection).

    Applies a predicate to filter rows.
    """

    input: PhysicalPlan
    predicate: Expr
    schema: Schema
    backend: Literal["auto", "arrow", "numpy"] = "auto"

    def execute(self, context: ExecutionContext) -> ArrayDict:
        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.convert import get_array_backend
        from pandas.lazy.backends.types import is_index_col

        input_arrays = self.input.execute(context)

        # Use ArrayEvaluator for predicate evaluation
        evaluator = ArrayEvaluator(input_arrays, preferred_backend=self.backend)
        mask = evaluator.evaluate(self.predicate._ir)

        # Separate data columns from index columns
        data_arrays = {k: v for k, v in input_arrays.items() if not is_index_col(k)}
        index_arrays = {k: v for k, v in input_arrays.items() if is_index_col(k)}

        # Check if all data arrays are Arrow-backed for batch filtering
        all_data_arrow = len(data_arrays) > 0 and all(
            isinstance(arr, (pa.Array, pa.ChunkedArray)) for arr in data_arrays.values()
        )

        # Ensure mask is a proper array for filtering
        if isinstance(mask, (pa.Array, pa.ChunkedArray)):
            mask_arr = mask.to_numpy(zero_copy_only=False)
            pa_mask = mask
        elif isinstance(mask, np.ndarray):
            mask_arr = mask
            pa_mask = None
        else:
            mask_arr = np.asarray(mask)
            pa_mask = None

        result: ArrayDict = {}

        if all_data_arrow:
            # Batch filter data columns using Arrow Table for efficiency
            if pa_mask is None:
                pa_mask = pa.array(mask_arr)

            # Build table and filter in one operation
            table = pa.table(data_arrays)
            filtered_table = table.filter(pa_mask)

            # Extract result arrays
            for name in data_arrays.keys():
                result[name] = filtered_table.column(name).combine_chunks()
        else:
            # Filter data arrays individually
            for name, arr in data_arrays.items():
                backend = get_array_backend(arr)
                result[name] = dispatch_kernel("filter", backend, arr, mask_arr)

        # Always filter index columns with numpy (they're typically numpy arrays)
        for name, arr in index_arrays.items():
            backend = get_array_backend(arr)
            result[name] = dispatch_kernel("filter", backend, arr, mask_arr)

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Aggregation Nodes
# =============================================================================


@dataclass
class PhysicalHashAggregate(PhysicalPlan):
    """
    Hash-based aggregation.

    Uses hash table for grouping - good for high cardinality.
    Uses kernel dispatch for efficient aggregation on Arrow or NumPy arrays.
    """

    input: PhysicalPlan
    group_by: tuple[Expr, ...]
    agg_exprs: tuple[Expr, ...]
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas.lazy.backends.convert import (
            get_array_backend,
        )
        from pandas.lazy.expr import extract_output_name
        from pandas.lazy.ir import (
            Alias,
            Call,
            FieldRef,
        )

        input_arrays = self.input.execute(context)

        # Extract group-by column names
        group_cols = [extract_output_name(e) for e in self.group_by]

        # Build aggregation list: (output_name, input_col, agg_func)
        agg_specs: list[tuple[str, str, str]] = []
        for expr in self.agg_exprs:
            output_name = extract_output_name(expr)
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if isinstance(ir, Call) and ir.is_aggregate:
                if ir.args and isinstance(ir.args[0], FieldRef):
                    col_name = ir.args[0].name
                    agg_func = ir.function
                    agg_specs.append((output_name, col_name, agg_func))

        # Determine backend from input arrays
        first_col = next(iter(input_arrays.values()))
        backend = get_array_backend(first_col)

        if not group_cols:
            # Global aggregation - no grouping keys
            return self._execute_global_aggregation(
                input_arrays, agg_specs, backend, context
            )

        # Grouped aggregation using kernel dispatch
        return self._execute_grouped_aggregation(
            input_arrays, group_cols, agg_specs, backend
        )

    def _execute_global_aggregation(
        self,
        input_arrays: ArrayDict,
        agg_specs: list[tuple[str, str, str]],
        backend: str,
        context: ExecutionContext,
    ) -> ArrayDict:
        """Execute aggregation without grouping (global aggregates)."""
        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends import dispatch_kernel

        result: ArrayDict = {}

        for output_name, col_name, agg_func in agg_specs:
            arr = input_arrays[col_name]

            # Use kernel dispatch for aggregations
            kernel_name = agg_func  # e.g., "sum", "mean", "min", "max"
            try:
                value = dispatch_kernel(kernel_name, backend, arr)
                # Scalar results - wrap in single-element array
                if isinstance(value, (pa.Scalar,)):
                    value = value.as_py()
                if backend == "arrow":
                    result[output_name] = pa.array([value])
                else:
                    result[output_name] = np.array([value])
            except NotImplementedError:
                # Fallback for unsupported aggregations
                if backend == "arrow":
                    np_arr = arr.to_numpy(zero_copy_only=False)
                else:
                    np_arr = arr
                value = getattr(np, agg_func)(np_arr)
                if backend == "arrow":
                    result[output_name] = pa.array([value])
                else:
                    result[output_name] = np.array([value])

        return result

    def _execute_grouped_aggregation(
        self,
        input_arrays: ArrayDict,
        group_cols: list[str],
        agg_specs: list[tuple[str, str, str]],
        backend: str,
    ) -> ArrayDict:
        """Execute grouped aggregation using kernel dispatch."""

        # For single-key groupby, use optimized single-key kernels
        # For multi-key, use hash_aggregate kernel
        if len(group_cols) == 1:
            return self._execute_single_key_groupby(
                input_arrays, group_cols[0], agg_specs, backend
            )
        else:
            return self._execute_multi_key_groupby(
                input_arrays, group_cols, agg_specs, backend
            )

    def _execute_single_key_groupby(
        self,
        input_arrays: ArrayDict,
        group_col: str,
        agg_specs: list[tuple[str, str, str]],
        backend: str,
    ) -> ArrayDict:
        """Execute single-key groupby using optimized kernels."""
        import pyarrow as pa

        from pandas.lazy.backends import (
            dispatch_kernel,
            has_kernel,
        )
        from pandas.lazy.backends.convert import ensure_backend

        key_arr = input_arrays[group_col]
        result: ArrayDict = {}
        unique_keys = None

        # Ensure arrays match the expected backend
        if backend == "numpy":
            key_arr = ensure_backend(key_arr, "numpy")
        elif backend == "arrow":
            key_arr = ensure_backend(key_arr, "arrow")

        for output_name, col_name, agg_func in agg_specs:
            value_arr = input_arrays[col_name]

            # Ensure value array matches backend
            if backend == "numpy":
                value_arr = ensure_backend(value_arr, "numpy")
            elif backend == "arrow":
                value_arr = ensure_backend(value_arr, "arrow")

            kernel_name = f"groupby_{agg_func}"

            if has_kernel(kernel_name, backend):
                keys, values = dispatch_kernel(kernel_name, backend, key_arr, value_arr)
                if unique_keys is None:
                    unique_keys = keys
                result[output_name] = values
            else:
                # Fallback to numpy if kernel not available
                np_keys = ensure_backend(key_arr, "numpy")
                np_values = ensure_backend(value_arr, "numpy")

                kernel_name_np = f"groupby_{agg_func}"
                if has_kernel(kernel_name_np, "numpy"):
                    keys, values = dispatch_kernel(
                        kernel_name_np, "numpy", np_keys, np_values
                    )
                    if unique_keys is None:
                        if backend == "arrow":
                            unique_keys = pa.array(keys)
                        else:
                            unique_keys = keys
                    if backend == "arrow":
                        result[output_name] = pa.array(values)
                    else:
                        result[output_name] = values
                else:
                    raise NotImplementedError(
                        f"No kernel for groupby_{agg_func} in {backend} or numpy"
                    )

        # Add group key to result first (for correct column ordering)
        ordered_result: ArrayDict = {group_col: unique_keys}
        ordered_result.update(result)

        return ordered_result

    def _execute_multi_key_groupby(
        self,
        input_arrays: ArrayDict,
        group_cols: list[str],
        agg_specs: list[tuple[str, str, str]],
        backend: str,
    ) -> ArrayDict:
        """Execute multi-key groupby using hash_aggregate kernel."""
        import pyarrow as pa

        from pandas.lazy.backends import (
            dispatch_kernel,
            has_kernel,
        )
        from pandas.lazy.backends.convert import ensure_backend

        if backend == "arrow" and has_kernel("group_by", "arrow"):
            # Use Arrow's native table-based group_by
            return self._execute_arrow_table_groupby(
                input_arrays, group_cols, agg_specs
            )

        # Fallback to hash_aggregate kernel
        key_arrays = [input_arrays[col] for col in group_cols]
        value_arrays = [input_arrays[spec[1]] for spec in agg_specs]
        agg_funcs = [spec[2] for spec in agg_specs]

        # Ensure arrays match the expected backend
        if backend == "numpy":
            key_arrays = [ensure_backend(arr, "numpy") for arr in key_arrays]
            value_arrays = [ensure_backend(arr, "numpy") for arr in value_arrays]
        elif backend == "arrow":
            key_arrays = [ensure_backend(arr, "arrow") for arr in key_arrays]
            value_arrays = [ensure_backend(arr, "arrow") for arr in value_arrays]

        if has_kernel("hash_aggregate", backend):
            result_keys, result_values = dispatch_kernel(
                "hash_aggregate", backend, key_arrays, value_arrays, agg_funcs
            )
        else:
            # Convert to numpy and use numpy kernel
            np_keys = [ensure_backend(arr, "numpy") for arr in key_arrays]
            np_values = [ensure_backend(arr, "numpy") for arr in value_arrays]

            result_keys, result_values = dispatch_kernel(
                "hash_aggregate", "numpy", np_keys, np_values, agg_funcs
            )

            if backend == "arrow":
                result_keys = [pa.array(k) for k in result_keys]
                result_values = [pa.array(v) for v in result_values]

        # Build result dict
        result: ArrayDict = {}
        for col, arr in zip(group_cols, result_keys, strict=False):
            result[col] = arr
        for (output_name, _, _), arr in zip(agg_specs, result_values, strict=False):
            result[output_name] = arr

        return result

    def _execute_arrow_table_groupby(
        self,
        input_arrays: ArrayDict,
        group_cols: list[str],
        agg_specs: list[tuple[str, str, str]],
    ) -> ArrayDict:
        """Execute groupby using Arrow's native table group_by."""
        import pyarrow as pa

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.types import is_index_col

        # Build Arrow table from arrays (excluding index columns)
        columns = {k: v for k, v in input_arrays.items() if not is_index_col(k)}
        table = pa.table(columns)

        # Execute groupby
        result_table = dispatch_kernel(
            "group_by", "arrow", table, group_cols, agg_specs
        )

        # Convert result table back to ArrayDict
        result: ArrayDict = {}
        for col_name in result_table.column_names:
            result[col_name] = result_table.column(col_name)

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Sort Nodes
# =============================================================================


@dataclass
class PhysicalSort(PhysicalPlan):
    """
    Physical sort operation.

    Sorts by specified columns with configurable algorithm.
    """

    input: PhysicalPlan
    by: tuple[Expr, ...]
    descending: tuple[bool, ...]
    schema: Schema
    algorithm: Literal["quicksort", "mergesort", "heapsort", "stable"] = "quicksort"

    def execute(self, context: ExecutionContext) -> ArrayDict:
        import numpy as np

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.convert import get_array_backend
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )

        input_arrays = self.input.execute(context)

        # Handle simple single-column sort using kernels directly
        if len(self.by) == 1:
            expr = self.by[0]
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg

            if isinstance(ir, FieldRef):
                # Simple column sort - use kernel directly
                col_name = ir.name
                arr = input_arrays[col_name]
                backend = get_array_backend(arr)
                descending = self.descending[0]

                # Get sort indices using kernel
                if backend == "arrow":
                    order = "descending" if descending else "ascending"
                    sort_indices = dispatch_kernel(
                        "array_sort_indices", backend, arr, order=order
                    )
                else:
                    sort_indices = dispatch_kernel(
                        "sort_indices", backend, arr, descending=descending
                    )

                # Apply indices to all arrays using take kernel
                result: ArrayDict = {}
                for name, col_arr in input_arrays.items():
                    col_backend = get_array_backend(col_arr)
                    result[name] = dispatch_kernel(
                        "take", col_backend, col_arr, sort_indices
                    )

                return result

        # Multi-column or computed expression sort - use evaluator
        evaluator = ArrayEvaluator(input_arrays, preferred_backend="auto")

        # Evaluate sort keys
        sort_key_arrays = []
        for i, expr in enumerate(self.by):
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if isinstance(ir, FieldRef):
                sort_key_arrays.append(input_arrays[ir.name])
            else:
                sort_key_arrays.append(evaluator.evaluate(ir))

        # Multi-column sort using lexsort (NumPy)
        # Convert all sort keys to NumPy for lexsort
        np_keys = []
        for i, (arr, desc) in enumerate(
            zip(reversed(sort_key_arrays), reversed(self.descending), strict=False)
        ):
            if hasattr(arr, "to_numpy"):
                np_arr = arr.to_numpy(zero_copy_only=False)
            else:
                np_arr = np.asarray(arr)
            # For descending, negate numeric arrays (lexsort sorts ascending)
            if desc and np.issubdtype(np_arr.dtype, np.number):
                np_arr = -np_arr
            np_keys.append(np_arr)

        sort_indices = np.lexsort(np_keys)

        # Apply indices to all arrays
        result: ArrayDict = {}
        for name, arr in input_arrays.items():
            backend = get_array_backend(arr)
            result[name] = dispatch_kernel("take", backend, arr, sort_indices)

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# TopK Node
# =============================================================================


@dataclass
class PhysicalTopK(PhysicalPlan):
    """
    Physical TopK operation.

    Efficiently returns top K rows without full sort using nsmallest/nlargest.
    """

    input: PhysicalPlan
    k: int
    by: tuple[Expr, ...]
    descending: tuple[bool, ...]
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        import numpy as np

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.convert import get_array_backend
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )

        input_arrays = self.input.execute(context)

        if self.k == 0:
            # Return empty arrays
            result: ArrayDict = {}
            for name, arr in input_arrays.items():
                if hasattr(arr, "slice"):
                    result[name] = arr.slice(0, 0)
                else:
                    result[name] = arr[:0]
            return result

        # Get array length
        first_arr = next(iter(input_arrays.values()))
        arr_len = len(first_arr)

        if self.k >= arr_len:
            # Need full sort - delegate to sort behavior
            from pandas.lazy.backends.convert import get_array_backend

            if len(self.by) == 1:
                expr = self.by[0]
                ir = expr._ir
                if isinstance(ir, Alias):
                    ir = ir.arg
                if isinstance(ir, FieldRef):
                    col_name = ir.name
                    arr = input_arrays[col_name]
                    backend = get_array_backend(arr)
                    descending = self.descending[0]

                    if backend == "arrow":
                        order = "descending" if descending else "ascending"
                        sort_indices = dispatch_kernel(
                            "array_sort_indices", backend, arr, order=order
                        )
                    else:
                        sort_indices = dispatch_kernel(
                            "sort_indices", backend, arr, descending=descending
                        )

                    result: ArrayDict = {}
                    for name, col_arr in input_arrays.items():
                        col_backend = get_array_backend(col_arr)
                        result[name] = dispatch_kernel(
                            "take", col_backend, col_arr, sort_indices
                        )
                    return result

            # Fall back to multi-key sort for full result
            evaluator = ArrayEvaluator(input_arrays, preferred_backend="auto")
            sort_key_arrays = []
            for expr in self.by:
                ir = expr._ir
                if isinstance(ir, Alias):
                    ir = ir.arg
                if isinstance(ir, FieldRef):
                    sort_key_arrays.append(input_arrays[ir.name])
                else:
                    sort_key_arrays.append(evaluator.evaluate(ir))

            np_keys = []
            rev_arrays = reversed(sort_key_arrays)
            rev_desc = reversed(self.descending)
            for arr, desc in zip(rev_arrays, rev_desc, strict=True):
                if hasattr(arr, "to_numpy"):
                    np_arr = arr.to_numpy(zero_copy_only=False)
                else:
                    np_arr = np.asarray(arr)
                if desc and np.issubdtype(np_arr.dtype, np.number):
                    np_arr = -np_arr
                np_keys.append(np_arr)

            sort_indices = np.lexsort(np_keys)

            result: ArrayDict = {}
            for name, arr in input_arrays.items():
                backend = get_array_backend(arr)
                result[name] = dispatch_kernel("take", backend, arr, sort_indices)
            return result

        # Use select_k for efficient TopK when k < arr_len
        if len(self.by) == 1:
            expr = self.by[0]
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg

            if isinstance(ir, FieldRef):
                # Simple single-column TopK - use select_k kernel
                col_name = ir.name
                arr = input_arrays[col_name]
                backend = get_array_backend(arr)
                descending = self.descending[0]

                if backend == "numpy":
                    order = "descending" if descending else "ascending"
                    topk_indices = dispatch_kernel(
                        "select_k_unstable", backend, arr, self.k, order=order
                    )
                else:
                    # Arrow: use sort_indices + slice
                    order = "descending" if descending else "ascending"
                    topk_indices = dispatch_kernel(
                        "select_k_unstable",
                        backend,
                        arr,
                        self.k,
                        sort_keys=[("", order)],
                    )

                result: ArrayDict = {}
                for name, col_arr in input_arrays.items():
                    col_backend = get_array_backend(col_arr)
                    result[name] = dispatch_kernel(
                        "take", col_backend, col_arr, topk_indices
                    )
                return result

        # Multi-column TopK - evaluate all keys and use argpartition + sort
        evaluator = ArrayEvaluator(input_arrays, preferred_backend="auto")

        sort_key_arrays = []
        for expr in self.by:
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if isinstance(ir, FieldRef):
                sort_key_arrays.append(input_arrays[ir.name])
            else:
                sort_key_arrays.append(evaluator.evaluate(ir))

        # Multi-column: use lexsort to get full ordering, then take top k
        np_keys = []
        rev_arrays = reversed(sort_key_arrays)
        rev_desc = reversed(self.descending)
        for arr, desc in zip(rev_arrays, rev_desc, strict=True):
            if hasattr(arr, "to_numpy"):
                np_arr = arr.to_numpy(zero_copy_only=False)
            else:
                np_arr = np.asarray(arr)
            if desc and np.issubdtype(np_arr.dtype, np.number):
                np_arr = -np_arr
            np_keys.append(np_arr)

        sort_indices = np.lexsort(np_keys)[: self.k]

        result: ArrayDict = {}
        for name, arr in input_arrays.items():
            backend = get_array_backend(arr)
            result[name] = dispatch_kernel("take", backend, arr, sort_indices)

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Limit Node
# =============================================================================


@dataclass
class PhysicalLimit(PhysicalPlan):
    """
    Physical limit (row count restriction).
    """

    input: PhysicalPlan
    n: int
    offset: int
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        input_arrays = self.input.execute(context)

        # Determine array length from first array
        first_arr = next(iter(input_arrays.values()))
        arr_len = len(first_arr)

        # Calculate slice indices
        if self.offset == -1:
            # Tail operation (offset=-1 is a special marker)
            start = max(0, arr_len - self.n)
            end = arr_len
        elif self.offset > 0:
            # Skip + limit
            start = self.offset
            end = min(self.offset + self.n, arr_len)
        else:
            # Simple head/limit
            start = 0
            end = min(self.n, arr_len)

        # Slice all arrays
        result: ArrayDict = {}
        for name, arr in input_arrays.items():
            if hasattr(arr, "slice"):
                # Arrow array
                result[name] = arr.slice(start, end - start)
            else:
                # NumPy array
                result[name] = arr[start:end]

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Distinct Node
# =============================================================================


@dataclass
class PhysicalDistinct(PhysicalPlan):
    """
    Physical distinct (duplicate removal).
    """

    input: PhysicalPlan
    subset: tuple[str, ...] | None
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        import numpy as np

        from pandas.lazy.backends import (
            dispatch_kernel,
            has_kernel,
        )
        from pandas.lazy.backends.convert import get_array_backend
        from pandas.lazy.backends.types import is_index_col

        input_arrays = self.input.execute(context)

        # Get columns to check for distinctness
        if self.subset:
            check_cols = list(self.subset)
        else:
            # All data columns (exclude index columns)
            check_cols = [
                name for name in input_arrays.keys() if not is_index_col(name)
            ]

        # Single column distinct - use unique kernel directly
        if len(check_cols) == 1:
            col_name = check_cols[0]
            arr = input_arrays[col_name]
            backend = get_array_backend(arr)

            # Get unique values and find first occurrence indices
            if backend == "numpy" and has_kernel("unique_indices", "numpy"):
                _, unique_indices = dispatch_kernel("unique_indices", backend, arr)
                # Sort indices to maintain original order
                unique_indices = np.sort(unique_indices)
            else:
                # Fallback: compute indices manually
                if hasattr(arr, "to_numpy"):
                    np_arr = arr.to_numpy(zero_copy_only=False)
                else:
                    np_arr = np.asarray(arr)
                _, unique_indices = np.unique(np_arr, return_index=True)
                unique_indices = np.sort(unique_indices)

            # Apply indices to all arrays
            result: ArrayDict = {}
            for name, col_arr in input_arrays.items():
                col_backend = get_array_backend(col_arr)
                result[name] = dispatch_kernel(
                    "take", col_backend, col_arr, unique_indices
                )
            return result

        # Multi-column distinct - need to combine columns for uniqueness check
        # Convert relevant columns to numpy and use structured array approach
        np_arrays = []
        for col_name in check_cols:
            arr = input_arrays[col_name]
            if hasattr(arr, "to_numpy"):
                np_arr = arr.to_numpy(zero_copy_only=False)
            else:
                np_arr = np.asarray(arr)
            np_arrays.append(np_arr)

        # Use pandas-style approach: hash-based deduplication
        # Create a combined view for uniqueness testing
        n_rows = len(np_arrays[0])

        # Build a structured array for unique check
        dtypes = [(f"f{i}", arr.dtype) for i, arr in enumerate(np_arrays)]
        structured = np.empty(n_rows, dtype=dtypes)
        for i, arr in enumerate(np_arrays):
            structured[f"f{i}"] = arr

        # Get unique indices
        _, unique_indices = np.unique(structured, return_index=True)
        unique_indices = np.sort(unique_indices)

        # Apply indices to all arrays
        result: ArrayDict = {}
        for name, arr in input_arrays.items():
            backend = get_array_backend(arr)
            result[name] = dispatch_kernel("take", backend, arr, unique_indices)

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Join Nodes
# =============================================================================


@dataclass
class PhysicalHashJoin(PhysicalPlan):
    """
    Hash-based join.

    Good for equi-joins with reasonable key cardinality.
    Uses kernel dispatch for efficient join on Arrow or NumPy arrays.
    """

    left: PhysicalPlan
    right: PhysicalPlan
    on: tuple[str, ...] | None
    left_on: tuple[str, ...] | None
    right_on: tuple[str, ...] | None
    how: Literal["inner", "left", "right", "outer", "cross"]
    suffix: tuple[str, str]
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas.lazy.backends import has_kernel
        from pandas.lazy.backends.convert import (
            get_array_backend,
        )
        from pandas.lazy.backends.types import is_index_col

        left_arrays = self.left.execute(context)
        right_arrays = self.right.execute(context)

        # Exclude index columns from join
        left_data = {k: v for k, v in left_arrays.items() if not is_index_col(k)}
        right_data = {k: v for k, v in right_arrays.items() if not is_index_col(k)}

        # Determine backend
        first_col = next(iter(left_data.values()))
        backend = get_array_backend(first_col)

        # Cross join requires special handling (no kernel, use DataFrame)
        if self.how == "cross":
            return self._execute_cross_join(left_data, right_data, context)

        # Try Arrow kernel for Arrow backend
        if backend == "arrow" and has_kernel("hash_join", "arrow"):
            return self._execute_arrow_join(left_data, right_data)

        # Fallback to DataFrame-based join
        return self._execute_dataframe_join(left_arrays, right_arrays, context)

    def _execute_arrow_join(
        self,
        left_data: ArrayDict,
        right_data: ArrayDict,
    ) -> ArrayDict:
        """Execute join using Arrow's native join."""
        import pyarrow as pa

        from pandas.lazy.backends import dispatch_kernel

        # Build Arrow tables
        left_table = pa.table(left_data)
        right_table = pa.table(right_data)

        # Determine keys
        if self.on is not None:
            keys = list(self.on)
            left_keys = None
            right_keys = None
        else:
            keys = None
            left_keys = list(self.left_on) if self.left_on else None
            right_keys = list(self.right_on) if self.right_on else None

        # Execute join
        result_table = dispatch_kernel(
            "hash_join",
            "arrow",
            left_table,
            right_table,
            keys=keys,
            left_keys=left_keys,
            right_keys=right_keys,
            join_type=self.how,
            left_suffix=self.suffix[0],
            right_suffix=self.suffix[1],
        )

        # Convert result table to ArrayDict
        result: ArrayDict = {}
        for col_name in result_table.column_names:
            result[col_name] = result_table.column(col_name)

        return result

    def _execute_dataframe_join(
        self,
        left_arrays: ArrayDict,
        right_arrays: ArrayDict,
        context: ExecutionContext,
    ) -> ArrayDict:
        """Fallback to DataFrame-based join."""
        from pandas.lazy.backends.convert import (
            arrays_to_dataframe,
            dataframe_to_arrays,
        )
        from pandas.lazy.backends.types import is_index_col

        left_df = arrays_to_dataframe(
            left_arrays,
            index_names=context.index_names,
            index_is_multi=context.index_is_multi,
        )
        right_df = arrays_to_dataframe(
            right_arrays,
            index_names=context.index_names,
            index_is_multi=context.index_is_multi,
        )

        if self.on is not None:
            result_df = left_df.merge(
                right_df,
                on=list(self.on),
                how=self.how,
                suffixes=self.suffix,
            )
        else:
            result_df = left_df.merge(
                right_df,
                left_on=list(self.left_on) if self.left_on else None,
                right_on=list(self.right_on) if self.right_on else None,
                how=self.how,
                suffixes=self.suffix,
            )

        arrays, _, _ = dataframe_to_arrays(result_df)
        return {k: v for k, v in arrays.items() if not is_index_col(k)}

    def _execute_cross_join(
        self,
        left_data: ArrayDict,
        right_data: ArrayDict,
        context: ExecutionContext,
    ) -> ArrayDict:
        """Execute cross join (Cartesian product)."""
        from pandas.lazy.backends.convert import (
            arrays_to_dataframe,
            dataframe_to_arrays,
        )
        from pandas.lazy.backends.types import is_index_col

        left_df = arrays_to_dataframe(
            left_data,
            index_names=context.index_names,
            index_is_multi=context.index_is_multi,
        )
        right_df = arrays_to_dataframe(
            right_data,
            index_names=context.index_names,
            index_is_multi=context.index_is_multi,
        )

        result_df = left_df.merge(right_df, how="cross", suffixes=self.suffix)
        arrays, _, _ = dataframe_to_arrays(result_df)
        return {k: v for k, v in arrays.items() if not is_index_col(k)}

    def children(self) -> list[PhysicalPlan]:
        return [self.left, self.right]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Convert Node
# =============================================================================


@dataclass
class PhysicalConvert(PhysicalPlan):
    """
    Physical conversion between backends (Arrow <-> NumPy).

    Converts all arrays to the target backend format.
    """

    input: PhysicalPlan
    target_backend: Literal["arrow", "numpy"]
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas.lazy.backends.convert import ensure_backend

        input_arrays = self.input.execute(context)

        # Convert all arrays to target backend
        result: ArrayDict = {}
        for name, arr in input_arrays.items():
            result[name] = ensure_backend(arr, self.target_backend)

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


# =============================================================================
# Physical Planner
# =============================================================================


class PhysicalPlanner:
    """
    Converts optimized logical plans to physical plans.

    The planner makes execution decisions based on:
    - Data characteristics
    - Operation requirements
    - User preferences
    """

    def __init__(
        self,
        preferred_backend: Literal["auto", "arrow", "numpy"] = "auto",
    ) -> None:
        self.preferred_backend = preferred_backend

    def plan(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        """
        Convert a logical plan to a physical plan.

        Parameters
        ----------
        logical_plan : LogicalPlan
            The optimized logical plan.

        Returns
        -------
        PhysicalPlan
            The physical execution plan.
        """
        from pandas.lazy.plan import (
            Aggregate,
            Convert,
            DataFrameSource,
            Distinct,
            Filter,
            Join,
            Limit,
            Project,
            Sort,
            TopK,
        )

        if isinstance(logical_plan, DataFrameSource):
            return self._plan_scan(logical_plan)

        elif isinstance(logical_plan, Project):
            return self._plan_project(logical_plan)

        elif isinstance(logical_plan, Filter):
            return self._plan_filter(logical_plan)

        elif isinstance(logical_plan, Aggregate):
            return self._plan_aggregate(logical_plan)

        elif isinstance(logical_plan, Sort):
            return self._plan_sort(logical_plan)

        elif isinstance(logical_plan, TopK):
            return self._plan_topk(logical_plan)

        elif isinstance(logical_plan, Limit):
            return self._plan_limit(logical_plan)

        elif isinstance(logical_plan, Distinct):
            return self._plan_distinct(logical_plan)

        elif isinstance(logical_plan, Join):
            return self._plan_join(logical_plan)

        elif isinstance(logical_plan, Convert):
            return self._plan_convert(logical_plan)

        else:
            raise NotImplementedError(
                f"Physical planning not implemented for: {type(logical_plan)}"
            )

    def _plan_scan(self, node) -> PhysicalScan:
        """Plan a DataFrameSource."""
        return PhysicalScan(
            df=node.df,
            schema=node.resolve_schema(),
        )

    def _plan_project(self, node) -> PhysicalProject:
        """Plan a Project."""
        return PhysicalProject(
            input=self.plan(node.input),
            exprs=node.exprs,
            schema=node.resolve_schema(),
            backend=self._choose_backend_for_exprs(node.exprs),
        )

    def _plan_filter(self, node) -> PhysicalFilter:
        """Plan a Filter."""
        return PhysicalFilter(
            input=self.plan(node.input),
            predicate=node.predicate,
            schema=node.resolve_schema(),
            backend=self._choose_backend_for_exprs((node.predicate,)),
        )

    def _plan_aggregate(self, node) -> PhysicalHashAggregate:
        """Plan an Aggregate."""
        # For now, always use hash aggregate
        # Future: could choose between hash/sort aggregate based on statistics
        return PhysicalHashAggregate(
            input=self.plan(node.input),
            group_by=node.group_by,
            agg_exprs=node.agg_exprs,
            schema=node.resolve_schema(),
        )

    def _plan_sort(self, node) -> PhysicalSort:
        """Plan a Sort."""
        # Future: choose algorithm based on data size
        return PhysicalSort(
            input=self.plan(node.input),
            by=node.by,
            descending=node.descending,
            schema=node.resolve_schema(),
            algorithm="quicksort",
        )

    def _plan_topk(self, node) -> PhysicalTopK:
        """Plan a TopK."""
        return PhysicalTopK(
            input=self.plan(node.input),
            k=node.k,
            by=node.by,
            descending=node.descending,
            schema=node.resolve_schema(),
        )

    def _plan_limit(self, node) -> PhysicalLimit:
        """Plan a Limit."""
        return PhysicalLimit(
            input=self.plan(node.input),
            n=node.n,
            offset=node.offset,
            schema=node.resolve_schema(),
        )

    def _plan_distinct(self, node) -> PhysicalDistinct:
        """Plan a Distinct."""
        return PhysicalDistinct(
            input=self.plan(node.input),
            subset=node.subset,
            schema=node.resolve_schema(),
        )

    def _plan_join(self, node) -> PhysicalHashJoin:
        """Plan a Join."""
        # For now, always use hash join
        # Future: could choose between hash/merge/nested-loop based on statistics
        return PhysicalHashJoin(
            left=self.plan(node.left),
            right=self.plan(node.right),
            on=node.on,
            left_on=node.left_on,
            right_on=node.right_on,
            how=node.how,
            suffix=node.suffix,
            schema=node.resolve_schema(),
        )

    def _plan_convert(self, node) -> PhysicalConvert:
        """Plan a Convert (backend conversion)."""
        return PhysicalConvert(
            input=self.plan(node.input),
            target_backend=node.target_backend,
            schema=node.resolve_schema(),
        )

    def _choose_backend_for_exprs(
        self, exprs: tuple[Expr, ...]
    ) -> Literal["auto", "arrow", "numpy"]:
        """
        Choose the best backend for a set of expressions.

        For now, returns "auto". Future versions could analyze
        expressions and choose based on operation requirements.
        """
        # TODO: Implement backend selection based on expression analysis
        return self.preferred_backend


# =============================================================================
# Execution Entry Point
# =============================================================================


def execute_physical_plan(
    plan: PhysicalPlan,
    *,
    preferred_backend: Literal["auto", "arrow", "numpy"] = "auto",
    strict: bool = False,
) -> DataFrame:
    """
    Execute a physical plan and return the result.

    Parameters
    ----------
    plan : PhysicalPlan
        The physical plan to execute.
    preferred_backend : {"auto", "arrow", "numpy"}
        Preferred execution backend.
    strict : bool
        If True, fail on backend fallbacks.

    Returns
    -------
    DataFrame
        The execution result.
    """
    from pandas.lazy.backends.convert import arrays_to_dataframe

    context = ExecutionContext(
        preferred_backend=preferred_backend,
        strict=strict,
    )

    # Execute and get ArrayDict
    arrays = plan.execute(context)

    # Convert ArrayDict back to DataFrame with proper index
    return arrays_to_dataframe(
        arrays,
        index_names=context.index_names,
        index_is_multi=context.index_is_multi,
    )
