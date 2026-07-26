"""Physical plan — sort_limit operators (split from physical.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pandas.lazy.backends import (
    dispatch_kernel,
    has_kernel,
)
from pandas.lazy.backends.array_eval import ArrayEvaluator
from pandas.lazy.backends.convert import (
    get_array_backend,
    to_arrow,
)
from pandas.lazy.backends.numpy.core import radix_lexsort
from pandas.lazy.backends.spill import ExternalSorter
from pandas.lazy.backends.types import (
    ArrayDict,
    is_index_col,
)
from pandas.lazy.cost import ARROW_MULTIKEY_SORT_MIN_ROWS
from pandas.lazy.ir import (
    Alias,
    FieldRef,
)
from pandas.lazy.physical.base import (
    ExecutionContext,
    PhysicalPlan,
    get_ordered_columns,
)
from pandas.lazy.physical.join import take_all_columns

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas.lazy.expr import Expr
    from pandas.lazy.types import Schema


def _keys_for_radix_lexsort(sort_key_arrays) -> list | None:
    """Code each sort key into a radix-sortable NumPy array, or None.

    Numeric keys pass through. String/object keys are factorized with
    ``sort=True`` into order-preserving integer codes, then cast to float64
    with nulls as NaN — so the radix lexsort's float path handles
    null-placement (nulls last) and per-key descending exactly, with no
    special casing. Returns None if any key is a kind the coder does not
    cover (datetime, etc.), so the caller falls back to Arrow's table sort.
    """

    import pandas as pd

    keys: list = []
    for arr in sort_key_arrays:
        np_arr = (
            arr.to_numpy(zero_copy_only=False)
            if hasattr(arr, "to_numpy")
            else np.asarray(arr)
        )
        kind = np_arr.dtype.kind
        if kind in ("i", "u", "f"):
            keys.append(np_arr)
        elif kind in ("O", "U", "S"):
            try:
                codes, _ = pd.factorize(np_arr, sort=True)
            except (TypeError, ValueError):
                return None  # unsortable/mixed object column
            fcodes = codes.astype(np.float64)
            if (codes == -1).any():
                fcodes[codes == -1] = np.nan  # nulls -> sort last via NaN
            keys.append(fcodes)
        else:
            return None  # datetime/bool/etc.: let Arrow handle it
    return keys


def _is_arrow_sortable(arr) -> bool:
    """Whether an array can be routed through Arrow's table sort."""

    if isinstance(arr, (pa.Array, pa.ChunkedArray)):
        return True
    if isinstance(arr, np.ndarray):
        # Numeric / boolean / datetime convert zero-copy or cheaply
        return arr.dtype.kind in ("i", "u", "f", "b", "M")
    return False


@dataclass
class PhysicalSort(PhysicalPlan):
    """
    Physical sort operation.

    Sorts by specified columns with configurable algorithm.

    External Merge Sort
    -------------------
    When spilling is enabled and input data exceeds the operator memory budget,
    this operator uses external merge sort:

    1. Read input in batches (streaming if available)
    2. Sort each batch in memory
    3. Spill sorted batches (runs) to disk
    4. K-way merge the sorted runs to produce final output

    This allows sorting datasets larger than available memory.
    """

    input: PhysicalPlan
    by: tuple[Expr, ...]
    descending: tuple[bool, ...]
    schema: Schema
    algorithm: Literal["quicksort", "mergesort", "heapsort", "stable"] = "quicksort"

    def execute(self, context: ExecutionContext) -> ArrayDict:
        # Check if we should use external sort (spill-enabled out-of-core sorting)
        if context.spill_enabled:
            # Try streaming external sort if input supports it
            if hasattr(self.input, "execute_batches"):
                return self._execute_external_sort(context)

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

                # Apply indices to all arrays (parallel for large data)
                return take_all_columns(input_arrays, sort_indices)

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

        n_rows = len(sort_key_arrays[0]) if sort_key_arrays else 0

        # Multi-key sort: a radix-based lexsort (stable per-key radix,
        # composing permutations) is ~6x faster than Arrow's table
        # sort_indices at 10M for numeric keys, and ~3.7x faster for string
        # keys once they are factorized into order-preserving codes. Code
        # each key into a radix-sortable NumPy array; if every key codes, take
        # this path, else fall back to Arrow.
        if n_rows >= ARROW_MULTIKEY_SORT_MIN_ROWS:
            radix_keys = _keys_for_radix_lexsort(sort_key_arrays)
            if radix_keys is not None:
                sort_indices = radix_lexsort(radix_keys, self.descending)
                return take_all_columns(input_arrays, sort_indices)

        # Large multi-key sort: route through Arrow's table-level
        # sort_indices, which is multi-threaded (~1.65x over np.lexsort
        # at 10M rows) and supports mixed ascending/descending natively.
        if n_rows >= ARROW_MULTIKEY_SORT_MIN_ROWS and all(
            _is_arrow_sortable(arr) for arr in sort_key_arrays
        ):
            try:
                key_table = pa.table(
                    {
                        f"__sort_key_{i}__": to_arrow(arr)
                        for i, arr in enumerate(sort_key_arrays)
                    }
                )
                sort_keys = [
                    (f"__sort_key_{i}__", "descending" if desc else "ascending")
                    for i, desc in enumerate(self.descending)
                ]
                sort_indices = pc.sort_indices(key_table, sort_keys=sort_keys)
                return take_all_columns(input_arrays, sort_indices)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                # Unsupported key type combination - fall back to lexsort
                pass

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

        # Apply indices to all arrays (parallel for large data)
        return take_all_columns(input_arrays, sort_indices)

    def _execute_external_sort(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute sort using external merge sort for larger-than-memory data.

        Uses the spill manager's ExternalSorter to:
        1. Process input in batches
        2. Sort and spill each batch as a sorted run
        3. K-way merge all runs to produce final sorted output
        """

        spill_manager = context.spill_manager
        if spill_manager is None:
            # Fallback to in-memory sort if spill manager not available
            input_arrays = self.input.execute(context)
            return self._sort_in_memory(input_arrays, context)

        # Extract sort key column names
        sort_keys: list[str] = []
        for expr in self.by:
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if isinstance(ir, FieldRef):
                sort_keys.append(ir.name)
            else:
                # Complex expression - need to evaluate
                # Fall back to in-memory sort for now
                input_arrays = self.input.execute(context)
                return self._sort_in_memory(input_arrays, context)

        # Get operator budget from spill config
        spill_config = context._spill_config
        run_size_bytes = spill_config.operator_budget_mb * 1024 * 1024

        # Create external sorter
        sorter = ExternalSorter(
            spill_manager=spill_manager,
            name="sort",
            run_size_bytes=run_size_bytes,
        )

        # Process input in batches
        for batch in self.input.execute_batches(context):
            sorter.add_batch(batch, sort_keys=sort_keys)

        # Finish sorting and get merged result
        result = sorter.finish(sort_keys=sort_keys)

        # Handle descending sort by reversing
        # Note: ExternalSorter sorts ascending; we reverse for descending
        if any(self.descending):
            result = self._maybe_reverse(result, sort_keys)

        return result

    def _sort_in_memory(
        self, input_arrays: ArrayDict, context: ExecutionContext
    ) -> ArrayDict:
        """In-memory sort fallback."""

        # Handle simple single-column sort using kernels directly
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

        # Multi-column sort
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
        for arr, desc in zip(
            reversed(sort_key_arrays), reversed(self.descending), strict=False
        ):
            if hasattr(arr, "to_numpy"):
                np_arr = arr.to_numpy(zero_copy_only=False)
            else:
                np_arr = np.asarray(arr)
            if desc and np.issubdtype(np_arr.dtype, np.number):
                np_arr = -np_arr
            np_keys.append(np_arr)

        sort_indices = np.lexsort(np_keys)

        result = {}
        for name, arr in input_arrays.items():
            backend = get_array_backend(arr)
            result[name] = dispatch_kernel("take", backend, arr, sort_indices)

        return result

    def _maybe_reverse(self, arrays: ArrayDict, sort_keys: list[str]) -> ArrayDict:
        """Reverse arrays if descending sort is needed."""

        # Check if all sort keys are descending
        all_descending = all(self.descending)
        if not all_descending:
            # Mixed ascending/descending - can't simply reverse
            # This case is handled by negating keys during external sort
            # For now, return as-is (ascending)
            return arrays

        # Reverse all arrays
        result: ArrayDict = {}
        for name, arr in arrays.items():
            if hasattr(arr, "to_pylist"):
                # Arrow array - convert to numpy, reverse, convert back

                np_arr = np.asarray(arr.to_pylist())
                result[name] = pa.array(np_arr[::-1].copy())
            else:
                result[name] = arr[::-1].copy()

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema

    @property
    def is_pipeline_breaker(self) -> bool:
        """Sort requires all input to determine global order."""
        return True


# =============================================================================
# TopK Node
# =============================================================================


@dataclass
class PhysicalTopK(PhysicalPlan):
    """
    Physical TopK operation.

    Efficiently returns top K rows without full sort using:
    - Single-key: np.argpartition for O(n) selection
    - Multi-key: heap-based streaming for O(n log k) selection
    - Streaming input: Processes batches incrementally using heap

    Streaming Execution
    -------------------
    For streaming inputs with small k, uses heap-based selection:
    - Maintains a heap of size k as batches arrive
    - O(n log k) time complexity instead of O(n log n) for full sort
    - Memory efficient: only stores k rows at a time
    """

    input: PhysicalPlan
    k: int
    by: tuple[Expr, ...]
    descending: tuple[bool, ...]
    schema: Schema

    @property
    def supports_streaming(self) -> bool:
        # TopK with streaming uses heap-based selection
        # Result isn't streaming, but input can be processed in streaming
        return False

    def execute(self, context: ExecutionContext) -> ArrayDict:
        # For small k with streaming input, use vectorized streaming approach
        if self.k > 0 and self.k <= 10000 and self.input.supports_streaming:
            return self._execute_streaming_topk(context)
        return self._execute_materialized_topk(context)

    def _execute_streaming_topk(self, context: ExecutionContext) -> ArrayDict:
        """
        Vectorized streaming top-k selection.

        Processes input batches incrementally using vectorized operations:
        1. For each batch, select top-k using select_k kernel
        2. Concatenate partial results (O(k * num_batches) rows)
        3. Final top-k selection on merged partials

        This is O(n) per batch for selection, much faster than row-by-row heap.
        """

        partial_results: list[ArrayDict] = []

        for batch in self.input.execute_batches(context):
            # Skip empty batches
            first_arr = next(iter(batch.values()))
            if len(first_arr) == 0:
                continue

            # Select top-k from this batch
            batch_topk = self._select_k_from_arrays(batch, self.k)
            if batch_topk:
                partial_results.append(batch_topk)

        if not partial_results:
            return self._make_empty_result(context)

        # If only one batch, return directly
        if len(partial_results) == 1:
            return partial_results[0]

        # Concatenate all partial results
        merged = self._concat_array_dicts(partial_results)

        # Final top-k selection on merged partials
        return self._select_k_from_arrays(merged, self.k)

    def _concat_array_dicts(self, arrays_list: list[ArrayDict]) -> ArrayDict:
        """Concatenate multiple ArrayDicts into one."""

        if not arrays_list:
            return {}

        result: ArrayDict = {}
        col_names = list(arrays_list[0].keys())

        for col_name in col_names:
            arrays = [d[col_name] for d in arrays_list if col_name in d]
            if not arrays:
                continue

            backend = get_array_backend(arrays[0])

            if backend == "arrow":
                # Concatenate Arrow arrays
                chunked = pa.chunked_array(arrays)
                result[col_name] = chunked.combine_chunks()
            else:
                # Concatenate NumPy arrays
                result[col_name] = np.concatenate(arrays)

        return result

    def _select_k_from_arrays(self, arrays: ArrayDict, k: int) -> ArrayDict:
        """
        Select top-k rows from arrays using vectorized operations.

        For single-key: uses select_k_unstable kernel (O(n) partitioning)
        For multi-key: uses lexsort + take (O(n log n) but vectorized)
        """

        first_arr = next(iter(arrays.values()))
        arr_len = len(first_arr)

        if arr_len == 0:
            return {}

        # If k >= array length, return all rows (sorted)
        if k >= arr_len:
            return self._sort_arrays(arrays)

        # Single-key case: use select_k_unstable kernel
        if len(self.by) == 1:
            expr = self.by[0]
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg

            if isinstance(ir, FieldRef):
                col_name = ir.name
                arr = arrays[col_name]
                backend = get_array_backend(arr)
                descending = self.descending[0]
                order = "descending" if descending else "ascending"

                if backend == "numpy":
                    topk_indices = dispatch_kernel(
                        "select_k_unstable", backend, arr, k, order=order
                    )
                else:
                    topk_indices = dispatch_kernel(
                        "select_k_unstable", backend, arr, k, sort_keys=[("", order)]
                    )

                result: ArrayDict = {}
                for name, col_arr in arrays.items():
                    col_backend = get_array_backend(col_arr)
                    result[name] = dispatch_kernel(
                        "take", col_backend, col_arr, topk_indices
                    )
                return result

        # Multi-key case: lexsort + take top k
        evaluator = ArrayEvaluator(arrays, preferred_backend="auto")
        sort_key_arrays = []

        for expr in self.by:
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if isinstance(ir, FieldRef):
                sort_key_arrays.append(arrays[ir.name])
            else:
                sort_key_arrays.append(evaluator.evaluate(ir))

        # Build keys for lexsort (reversed order, with negation for descending)
        np_keys = []
        for arr, desc in zip(
            reversed(sort_key_arrays), reversed(self.descending), strict=True
        ):
            if hasattr(arr, "to_numpy"):
                np_arr = arr.to_numpy(zero_copy_only=False)
            else:
                np_arr = np.asarray(arr)
            if desc and np.issubdtype(np_arr.dtype, np.number):
                np_arr = -np_arr
            np_keys.append(np_arr)

        sort_indices = np.lexsort(np_keys)[:k]

        result = {}
        for name, arr in arrays.items():
            backend = get_array_backend(arr)
            result[name] = dispatch_kernel("take", backend, arr, sort_indices)

        return result

    def _sort_arrays(self, arrays: ArrayDict) -> ArrayDict:
        """Sort arrays by the sort keys (used when k >= array length)."""

        evaluator = ArrayEvaluator(arrays, preferred_backend="auto")
        sort_key_arrays = []

        for expr in self.by:
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if isinstance(ir, FieldRef):
                sort_key_arrays.append(arrays[ir.name])
            else:
                sort_key_arrays.append(evaluator.evaluate(ir))

        np_keys = []
        for arr, desc in zip(
            reversed(sort_key_arrays), reversed(self.descending), strict=True
        ):
            if hasattr(arr, "to_numpy"):
                np_arr = arr.to_numpy(zero_copy_only=False)
            else:
                np_arr = np.asarray(arr)
            if desc and np.issubdtype(np_arr.dtype, np.number):
                np_arr = -np_arr
            np_keys.append(np_arr)

        sort_indices = np.lexsort(np_keys)

        result: ArrayDict = {}
        for name, arr in arrays.items():
            backend = get_array_backend(arr)
            result[name] = dispatch_kernel("take", backend, arr, sort_indices)

        return result

    def _make_empty_result(self, context: ExecutionContext) -> ArrayDict:
        """Create empty result arrays matching schema."""

        result: ArrayDict = {}
        for name in self.schema.names:
            # Default to empty numpy array
            result[name] = np.array([])
        return result

    def _execute_materialized_topk(self, context: ExecutionContext) -> ArrayDict:
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

    Streaming Execution
    -------------------
    This operator supports streaming execution with early termination.
    For head() operations, it stops reading from the input once enough
    rows have been collected, providing significant speedups for large
    datasets.

    Note: Tail operations (offset=-1) require all data and cannot
    benefit from early termination in streaming mode.
    """

    input: PhysicalPlan
    n: int
    offset: int
    schema: Schema

    @property
    def supports_streaming(self) -> bool:
        # Tail operations need all data, so don't support streaming
        # for offset=-1 (special marker for tail)
        if self.offset == -1:
            return False
        return self.input.supports_streaming

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Stream limited batches with early termination.

        For head() operations, stops reading from input once the
        limit is reached. For skip+limit operations, skips the
        required rows before yielding results.

        Yields
        ------
        ArrayDict
            Batches limited to the requested number of rows.
        """
        rows_seen = 0
        rows_yielded = 0

        for batch in self.input.execute_batches(context):
            if not batch:
                continue

            first_arr = next(iter(batch.values()))
            batch_len = len(first_arr)

            if batch_len == 0:
                continue

            # Handle offset (skip rows)
            if self.offset > 0 and rows_seen + batch_len <= self.offset:
                # Skip entire batch
                rows_seen += batch_len
                continue

            # Calculate what to keep from this batch
            skip_start = max(0, self.offset - rows_seen) if self.offset > 0 else 0
            keep_count = min(batch_len - skip_start, self.n - rows_yielded)

            if keep_count > 0:
                # Slice the batch
                sliced = self._slice_batch(batch, skip_start, skip_start + keep_count)
                yield sliced
                rows_yielded += keep_count

            rows_seen += batch_len

            # Early termination - we have enough rows!
            if rows_yielded >= self.n:
                return

    def _slice_batch(self, batch: ArrayDict, start: int, end: int) -> ArrayDict:
        """Slice all arrays in a batch."""
        result: ArrayDict = {}
        for name, arr in batch.items():
            if hasattr(arr, "slice"):
                # Arrow array
                result[name] = arr.slice(start, end - start)
            else:
                # NumPy array
                result[name] = arr[start:end]
        return result

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute limit and return all results at once.

        For streaming-capable inputs, uses execute_batches() with early
        termination for efficiency. For tail operations or non-streaming
        inputs, falls back to materializing all data first.
        """
        # Tail operation needs all data
        if self.offset == -1:
            return self._execute_tail(context)

        # For streaming sources, use batched execution with early termination
        if self.input.supports_streaming:
            return self._materialize_batches(context)

        # Non-streaming fallback: get all data, then slice
        return self._execute_slice(context)

    def _execute_tail(self, context: ExecutionContext) -> ArrayDict:
        """Execute tail operation (needs all data)."""
        input_arrays = self.input.execute(context)
        first_arr = next(iter(input_arrays.values()))
        arr_len = len(first_arr)

        start = max(0, arr_len - self.n)
        end = arr_len

        return self._slice_batch(input_arrays, start, end)

    def _execute_slice(self, context: ExecutionContext) -> ArrayDict:
        """Execute as a simple slice on materialized data."""
        input_arrays = self.input.execute(context)
        first_arr = next(iter(input_arrays.values()))
        arr_len = len(first_arr)

        if self.offset > 0:
            start = self.offset
            end = min(self.offset + self.n, arr_len)
        else:
            start = 0
            end = min(self.n, arr_len)

        return self._slice_batch(input_arrays, start, end)

    def _materialize_batches(self, context: ExecutionContext) -> ArrayDict:
        """Materialize batches from execute_batches into single result."""

        batches = list(self.execute_batches(context))

        if not batches:
            # Return empty result
            result: ArrayDict = {}
            for col_name in self.schema:
                result[col_name] = pa.array([])
            return result

        if len(batches) == 1:
            return batches[0]

        # Concatenate all batches preserving column order
        result: ArrayDict = {}
        for col_name in get_ordered_columns(batches):
            chunks = [batch[col_name] for batch in batches if col_name in batch]
            if chunks:
                if hasattr(chunks[0], "type"):
                    # Arrow arrays - concatenate
                    chunked = pa.chunked_array(chunks)
                    result[col_name] = chunked.combine_chunks()
                else:
                    # NumPy arrays

                    result[col_name] = np.concatenate(chunks)

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

        # Defensive: an empty input (no rows materialized upstream, e.g. a
        # fully-filtered scan) must return empty, not crash on the
        # structured-array path below.
        if not input_arrays or not check_cols:
            return dict(input_arrays)

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

    @property
    def is_pipeline_breaker(self) -> bool:
        """Distinct requires all input to deduplicate globally."""
        return True


@dataclass
class PhysicalGroupByHead(PhysicalPlan):
    """Keep the first ``n`` rows of each group (group-wise head).

    With a preceding sort this yields top-k rows per group as plain rows. The
    kept row positions come from pandas' ``groupby(...).head(n)`` over just the
    key columns (preserving input order); all columns are then gathered by
    those positions, so payload backends are preserved.
    """

    input: PhysicalPlan
    group_keys: tuple[str, ...]
    n: int
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas import DataFrame

        input_arrays = self.input.execute(context)
        keys = list(self.group_keys)
        key_data = {}
        for name in keys:
            arr = input_arrays[name]
            key_data[name] = (
                arr.to_numpy(zero_copy_only=False)
                if hasattr(arr, "to_numpy")
                else np.asarray(arr)
            )
        # head(n) over just the key columns gives the original positions of the
        # first n rows of each group, in input order.
        key_df = DataFrame(key_data, copy=False)
        keep = key_df.groupby(keys, sort=False).head(self.n).index.to_numpy()

        return {
            name: dispatch_kernel("take", get_array_backend(arr), arr, keep)
            for name, arr in input_arrays.items()
        }

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema

    @property
    def is_pipeline_breaker(self) -> bool:
        """Group-wise head needs all rows of each group present."""
        return True
