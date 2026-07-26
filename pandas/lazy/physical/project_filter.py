"""Physical plan — project_filter operators (split from physical.py)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pandas.lazy.backends import dispatch_kernel
from pandas.lazy.backends.array_eval import ArrayEvaluator
from pandas.lazy.backends.convert import (
    get_array_backend,
    to_arrow,
)
from pandas.lazy.backends.memory_pool import PoolingStrategy
from pandas.lazy.backends.types import (
    ArrayDict,
    is_index_col,
)
from pandas.lazy.expr import (
    Expr,
    extract_output_name,
)
from pandas.lazy.physical.base import (
    ExecutionContext,
    PhysicalPlan,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas.lazy.types import Schema

# =============================================================================
# Projection Nodes
# =============================================================================


@dataclass
class PhysicalProject(PhysicalPlan):
    """
    Physical projection (column selection/computation).

    Executes expressions to produce output columns.

    Streaming Execution
    -------------------
    This operator supports streaming execution. It passes through batches
    from the input, evaluating projection expressions on each batch
    independently.
    """

    input: PhysicalPlan
    exprs: tuple[Expr, ...]
    schema: Schema
    backend: Literal["auto", "arrow", "numpy"] = "auto"

    @property
    def supports_streaming(self) -> bool:
        return self.input.supports_streaming

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Stream projected batches from input.

        Evaluates projection expressions on each batch independently.

        Yields
        ------
        ArrayDict
            Projected batches from the input.
        """
        for batch in self.input.execute_batches(context):
            yield self._project_batch(batch, context)

    def _project_batch(
        self, input_arrays: ArrayDict, context: ExecutionContext
    ) -> ArrayDict:
        """Evaluate projection expressions on a single batch."""

        result: ArrayDict = {}

        # Preserve index columns
        for name, arr in input_arrays.items():
            if is_index_col(name):
                result[name] = arr

        # Determine if we should parallelize
        n_exprs = len(self.exprs)
        use_parallel = n_exprs >= context.parallel_threshold

        if use_parallel:
            result.update(self._evaluate_parallel(input_arrays, context, np, pa))
        else:
            result.update(self._evaluate_sequential(input_arrays, context, np, pa))

        return result

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute projection and return all results at once.

        Delegates to _project_batch for the actual projection logic.
        """
        input_arrays = self.input.execute(context)
        return self._project_batch(input_arrays, context)

    def _evaluate_sequential(
        self,
        input_arrays: ArrayDict,
        context: ExecutionContext,
        np,
        pa,
    ) -> ArrayDict:
        """Evaluate expressions sequentially."""

        evaluator = ArrayEvaluator(
            input_arrays,
            preferred_backend=self.backend,
            pooling_strategy=(
                PoolingStrategy.SCRATCH
                if self.backend != "arrow"
                else PoolingStrategy.NONE
            ),
        )

        result: ArrayDict = {}
        arr_len = len(next(iter(input_arrays.values()))) if input_arrays else 0

        for expr in self.exprs:
            name = extract_output_name(expr)
            value = evaluator.evaluate(expr._ir)
            # Ensure result is an array (not scalar for column expressions)
            if not isinstance(value, (np.ndarray, pa.Array, pa.ChunkedArray)):
                value = np.full(arr_len, value)
            result[name] = value

        return result

    def _evaluate_parallel(
        self,
        input_arrays: ArrayDict,
        context: ExecutionContext,
        np,
        pa,
    ) -> ArrayDict:
        """Evaluate expressions in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor

        arr_len = len(next(iter(input_arrays.values()))) if input_arrays else 0

        def evaluate_expr(expr):
            """Evaluate a single expression (thread worker function)."""
            # Each thread gets its own evaluator to avoid contention
            evaluator = ArrayEvaluator(
                input_arrays,
                preferred_backend=self.backend,
                pooling_strategy=(
                    PoolingStrategy.SCRATCH
                    if self.backend != "arrow"
                    else PoolingStrategy.NONE
                ),
            )
            name = extract_output_name(expr)
            value = evaluator.evaluate(expr._ir)
            # Ensure result is an array
            if not isinstance(value, (np.ndarray, pa.Array, pa.ChunkedArray)):
                value = np.full(arr_len, value)
            return name, value

        # Determine worker count

        n_workers = context.n_workers
        if n_workers is None:
            # Use CPU count but cap at number of expressions
            n_workers = min(os.cpu_count() or 4, len(self.exprs))

        result: ArrayDict = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(evaluate_expr, expr) for expr in self.exprs]
            for future in futures:
                name, value = future.result()
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

    Streaming Execution
    -------------------
    This operator supports streaming execution. It passes through batches
    from the input, applying the predicate filter to each batch independently.
    Empty batches (where all rows are filtered out) are skipped.
    """

    input: PhysicalPlan
    predicate: Expr
    schema: Schema
    backend: Literal["auto", "arrow", "numpy"] = "auto"
    # Set by the engine's decision layer when the choice is decidable
    # from the schema alone (all data columns Arrow-backed). Mixed
    # backends stay a runtime threshold decision (actual row count is
    # strictly better information there).
    planned_backend: str | None = None

    @property
    def supports_streaming(self) -> bool:
        return self.input.supports_streaming

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Stream filtered batches from input.

        Applies the filter predicate to each batch independently and
        yields non-empty filtered results.

        Yields
        ------
        ArrayDict
            Filtered batches from the input.
        """
        for batch in self.input.execute_batches(context):
            filtered = self._filter_batch(batch, context)
            # Skip empty batches
            if filtered:
                first_arr = next(iter(filtered.values()))
                if len(first_arr) > 0:
                    yield filtered

    def _filter_batch(
        self, input_arrays: ArrayDict, context: ExecutionContext
    ) -> ArrayDict:
        """Apply predicate filter to a single batch."""
        import time

        # Separate data columns from index columns
        data_arrays = {k: v for k, v in input_arrays.items() if not is_index_col(k)}
        index_arrays = {k: v for k, v in input_arrays.items() if is_index_col(k)}

        if not data_arrays:
            return input_arrays

        # Get row count for statistics
        first_arr = next(iter(data_arrays.values()))
        n_rows = len(first_arr)

        # Check if all data arrays are Arrow-backed
        all_data_arrow = all(
            isinstance(arr, (pa.Array, pa.ChunkedArray)) for arr in data_arrays.values()
        )

        # For streaming, prefer Arrow path as data is typically Arrow from scan.
        # The decision layer may have planned this from the schema already.
        use_arrow_filter = all_data_arrow or self.planned_backend == "arrow"
        if not use_arrow_filter:
            # Use threshold config to determine if Arrow filter is beneficial
            threshold = context.threshold_config.filter_arrow_threshold
            if n_rows > threshold:
                use_arrow_filter = True
                data_arrays = {
                    k: to_arrow(v) if isinstance(v, np.ndarray) else v
                    for k, v in data_arrays.items()
                }

        start_time = time.perf_counter()

        if use_arrow_filter:
            table = pa.table(data_arrays)
            evaluator = ArrayEvaluator(dict(data_arrays), preferred_backend="arrow")
            mask = evaluator.evaluate(self.predicate._ir)

            if isinstance(mask, np.ndarray):
                pa_mask = pa.array(mask)
            elif isinstance(mask, (pa.Array, pa.ChunkedArray)):
                pa_mask = mask
            else:
                pa_mask = pa.array(np.asarray(mask))

            filtered_table = table.filter(pa_mask)

            result: ArrayDict = {}
            for name in data_arrays.keys():
                # Keep as ChunkedArray - PyArrow ops work on ChunkedArray directly
                # This avoids expensive memory consolidation when there are many chunks
                result[name] = filtered_table.column(name)

            for name, arr in index_arrays.items():
                if isinstance(arr, np.ndarray):
                    result[name] = pc.filter(pa.array(arr), pa_mask)
                elif isinstance(arr, (pa.Array, pa.ChunkedArray)):
                    result[name] = pc.filter(arr, pa_mask)
                else:
                    backend = get_array_backend(arr)
                    mask_np = pa_mask.to_numpy(zero_copy_only=False)
                    result[name] = dispatch_kernel("filter", backend, arr, mask_np)

            # Record execution statistics for adaptive thresholds
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            context.record_execution("filter", "arrow", n_rows, elapsed_ms)

            return result

        # NumPy path
        evaluator = ArrayEvaluator(input_arrays, preferred_backend=self.backend)
        mask = evaluator.evaluate(self.predicate._ir)

        if isinstance(mask, (pa.Array, pa.ChunkedArray)):
            mask_arr = mask.to_numpy(zero_copy_only=False)
        elif isinstance(mask, np.ndarray):
            mask_arr = mask
        else:
            mask_arr = np.asarray(mask)

        indices = np.nonzero(mask_arr)[0]

        result: ArrayDict = {}
        for name, arr in data_arrays.items():
            if isinstance(arr, np.ndarray):
                result[name] = arr.take(indices)
            else:
                backend = get_array_backend(arr)
                result[name] = dispatch_kernel("filter", backend, arr, mask_arr)

        for name, arr in index_arrays.items():
            if isinstance(arr, np.ndarray):
                result[name] = arr.take(indices)
            else:
                backend = get_array_backend(arr)
                result[name] = dispatch_kernel("filter", backend, arr, mask_arr)

        # Record execution statistics for adaptive thresholds
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        context.record_execution("filter", "numpy", n_rows, elapsed_ms)

        return result

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute filter and return all results at once.

        Delegates to _filter_batch for the actual filtering logic.
        """
        input_arrays = self.input.execute(context)
        return self._filter_batch(input_arrays, context)

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema
