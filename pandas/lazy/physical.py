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
    from collections.abc import Iterator

    import numpy as np

    from pandas import DataFrame
    from pandas.lazy.backends.spill import (
        SpillConfig,
        SpillManager,
    )
    from pandas.lazy.backends.types import ArrayDict
    from pandas.lazy.expr import Expr
    from pandas.lazy.plan import LogicalPlan
    from pandas.lazy.types import Schema


# =============================================================================
# Helper Functions
# =============================================================================


def _get_ordered_columns(batches: list) -> list[str]:
    """
    Get column names in deterministic order from a list of batches.

    When concatenating batches, we need to iterate over columns in a
    consistent order. Simply using `set()` would produce non-deterministic
    ordering because set iteration order depends on hash values.

    This function preserves the column order from the first batch that
    contains each column, ensuring deterministic output column order.

    Parameters
    ----------
    batches : list of ArrayDict
        List of batches to get column names from.

    Returns
    -------
    list of str
        Column names in deterministic order (preserving first occurrence order).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for batch in batches:
        for col_name in batch.keys():
            if col_name not in seen:
                seen.add(col_name)
                ordered.append(col_name)
    return ordered


# =============================================================================
# Physical Plan Nodes
# =============================================================================


@dataclass
class PhysicalPlan(ABC):
    """
    Base class for physical plan nodes.

    Physical plans represent concrete execution strategies with backend
    and algorithm choices already made.

    Streaming Execution
    -------------------
    Physical plan nodes can support streaming execution through the
    `execute_batches()` method. This enables:

    - Memory efficiency: Process data larger than RAM
    - Early termination: Stop reading when limit is satisfied
    - Better cache locality: Process in L3-cache-friendly batches

    Operators that support streaming should override `execute_batches()`
    and set `supports_streaming` to True. The default implementation
    calls `execute()` and yields a single batch.
    """

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute this physical plan node and return all results at once.

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

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Execute this physical plan node and yield batches of results.

        This method enables streaming execution for memory efficiency
        and early termination. Override this method for operators that
        can process data in batches (e.g., scan, filter, project, limit).

        The default implementation calls execute() and yields a single batch.
        Operators that need all data (e.g., sort, aggregate) should use
        the default implementation.

        Parameters
        ----------
        context : ExecutionContext
            Execution context with runtime state.

        Yields
        ------
        ArrayDict
            Batches of results, each as a dictionary mapping column names
            to arrays.

        Notes
        -----
        Streaming operators should:
        1. Override this method to yield batches from input
        2. Set `supports_streaming` property to True
        3. Check `context.batch_size` for suggested batch size

        Pipeline breakers (sort, aggregate, distinct) should use the
        default implementation which materializes all data.
        """
        yield self.execute(context)

    @property
    def supports_streaming(self) -> bool:
        """
        Whether this operator supports batch-by-batch streaming execution.

        Returns True if this operator can process data incrementally
        without needing all input data at once. Pipeline breakers
        (sort, aggregate, distinct, join) return False.

        Returns
        -------
        bool
            True if streaming is supported, False otherwise.
        """
        return False

    @property
    def is_pipeline_breaker(self) -> bool:
        """
        Whether this operator breaks the streaming pipeline.

        Pipeline breakers are operators that must consume ALL input data
        before producing ANY output. This creates a materialization boundary
        in the execution pipeline.

        The physical planner wraps inputs to breaker operators with explicit
        PhysicalMaterialize nodes, making boundaries visible in the plan.

        Returns
        -------
        bool
            True if this operator requires all input before producing output.
        """
        return False

    def _materialize_input(
        self, input_plan: PhysicalPlan, context: ExecutionContext
    ) -> ArrayDict:
        """
        Materialize input from a potentially streaming source.

        This helper should be used by pipeline breakers (sort, aggregate,
        distinct, join) that need all data before processing.

        Parameters
        ----------
        input_plan : PhysicalPlan
            The input plan to execute.
        context : ExecutionContext
            Execution context.

        Returns
        -------
        ArrayDict
            Materialized data from the input.
        """
        return input_plan.execute(context)

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

    The threshold_config parameter provides centralized access to all
    execution thresholds. If not provided, uses the global default.

    Adaptive Thresholds
    -------------------
    When adaptive_thresholds is True, the context collects execution
    statistics and adjusts thresholds based on observed performance.
    This is useful for workloads where optimal thresholds vary.
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
    # Whether to preserve index during operations (for groupby, join)
    preserve_index: bool = False
    # Whether user explicitly set an index via set_index()
    # When True, index is always reconstructed regardless of preserve_index
    user_set_index: bool = False

    # Relaxed output ordering (collect(order="relaxed")). When True, the
    # final result's row order is unspecified, which lets the decision
    # layer route order-preserving joins to acero's parallel hash join.
    # Intermediate order-dependent values (shift/cum_*) are still computed
    # over a well-defined order; only the terminal output order is relaxed.
    order_relaxed: bool = False

    # Parallelism configuration
    # Number of workers for parallel execution (None = auto based on CPU count)
    n_workers: int | None = None
    # Minimum number of expressions before parallelizing (overhead threshold)
    # ThreadPoolExecutor has ~1-2ms overhead per submission, so only parallelize
    # when there are enough heavy expressions to amortize this cost
    parallel_threshold: int = 8

    # Streaming configuration
    # Batch size for streaming execution (64K rows is L3 cache friendly)
    batch_size: int = 65536
    # Whether streaming execution is enabled
    streaming_enabled: bool = False

    # Threshold configuration (None = use global default)
    # This provides centralized access to all execution thresholds
    _threshold_config: Any = field(default=None, repr=False)

    # Adaptive threshold configuration
    # When enabled, collects execution statistics and adjusts thresholds
    adaptive_thresholds: bool = False

    # Spill configuration for out-of-core processing
    # When enabled, intermediate results can be spilled to disk under memory pressure
    _spill_manager: SpillManager | None = field(default=None, repr=False)
    _spill_config: SpillConfig | None = field(default=None, repr=False)

    def clone_for_subplan(self) -> ExecutionContext:
        """
        Create an independent context for executing a subplan concurrently.

        Configuration (backend preference, strict mode, parallelism,
        streaming, thresholds, spill settings) is carried over; mutable
        per-execution state (CSE cache, index metadata, user_set_index)
        starts fresh so concurrent subplans cannot race on it. Callers
        decide which subplan's index metadata to propagate back.
        """
        return ExecutionContext(
            preferred_backend=self.preferred_backend,
            strict=self.strict,
            preserve_index=self.preserve_index,
            order_relaxed=self.order_relaxed,
            n_workers=self.n_workers,
            parallel_threshold=self.parallel_threshold,
            batch_size=self.batch_size,
            streaming_enabled=self.streaming_enabled,
            _threshold_config=self._threshold_config,
            adaptive_thresholds=self.adaptive_thresholds,
            _spill_manager=self._spill_manager,
            _spill_config=self._spill_config,
        )

    @property
    def spill_manager(self) -> SpillManager | None:
        """
        Get the spill manager, creating one if spill config is set.

        The spill manager handles disk spilling of intermediate results
        when memory pressure is detected. Used by memory-intensive operators
        like sort, join, and groupby.
        """
        if self._spill_manager is not None:
            return self._spill_manager

        if self._spill_config is not None and self._spill_config.enabled:
            from pandas.lazy.backends.spill import SpillManager

            self._spill_manager = SpillManager(self._spill_config)
            return self._spill_manager

        return None

    @property
    def spill_enabled(self) -> bool:
        """Check if spilling is enabled."""
        return self._spill_config is not None and self._spill_config.enabled

    def check_memory_pressure(self) -> bool:
        """
        Check memory pressure and spill if needed.

        Returns True if spilling occurred.
        """
        manager = self.spill_manager
        if manager is None:
            return False

        spilled = manager.check_memory_pressure()
        return len(spilled) > 0

    @property
    def threshold_config(self) -> Any:
        """Get the threshold configuration, using global default if not set."""
        if self._threshold_config is not None:
            return self._threshold_config

        # If adaptive thresholds enabled, get adapted config
        if self.adaptive_thresholds:
            from pandas.lazy.optimize.adaptive import get_adaptive_config

            return get_adaptive_config()

        from pandas.lazy.optimize.config import get_threshold_config

        return get_threshold_config()

    def record_execution(
        self,
        operation: str,
        backend: str,
        rows: int,
        time_ms: float,
    ) -> None:
        """
        Record an execution event for adaptive threshold tuning.

        Parameters
        ----------
        operation : str
            The operation type ("filter", "groupby", "projection", "numexpr").
        backend : str
            The backend used ("arrow" or "numpy").
        rows : int
            Number of rows processed.
        time_ms : float
            Execution time in milliseconds.
        """
        if self.adaptive_thresholds:
            from pandas.lazy.optimize.adaptive import record_execution

            record_execution(operation, backend, rows, time_ms)


# =============================================================================
# Materialization Boundary
# =============================================================================


@dataclass
class PhysicalMaterialize(PhysicalPlan):
    """
    Explicit materialization boundary in the execution pipeline.

    This node forces full materialization of its input, converting a streaming
    iterator into a complete in-memory (or spilled) dataset. It serves as an
    explicit boundary between streaming pipelines.

    Purpose
    -------
    Making materialization explicit enables:

    1. **Clear pipeline boundaries**: The plan clearly shows where streaming
       stops and full data access is required.

    2. **Centralized spill management**: All spill logic happens here, not
       scattered across operator implementations.

    3. **Operator fusion boundaries**: Fusion can safely combine operators
       within a pipeline but knows not to cross Materialize nodes.

    4. **Backend conversion points**: Backend switches can be pinned to
       materialization points for efficiency.

    5. **Explain/debug clarity**: Query plans clearly show where data
       is fully buffered.

    When Materialize is Inserted
    ----------------------------
    The physical planner inserts Materialize nodes before:

    - **Sort**: Needs all rows to determine global order
    - **Distinct**: Needs all values to deduplicate globally
    - **Aggregate**: Needs all rows per group (without partial agg)
    - **HashJoin build side**: Must build complete hash table before probe

    Example
    -------
    Before (implicit materialization inside Sort):
        Sort(input=Filter(Project(Scan)))

    After (explicit boundary):
        Sort(input=Materialize(Filter(Project(Scan)), reason="sort"))

    The execution result is identical, but now:
    - Spill decisions are centralized in Materialize
    - explain() shows the boundary clearly
    - Fusion knows not to cross it

    Parameters
    ----------
    input : PhysicalPlan
        The input plan to materialize.
    reason : str
        Why materialization is required. Used for debugging and explain().
        Common values: "sort", "distinct", "aggregate", "hash_join_build",
        "backend_convert".

    Attributes
    ----------
    is_pipeline_breaker : bool
        Always True - this node is the pipeline breaker.
    supports_streaming : bool
        Always False - output is fully materialized.
    """

    input: PhysicalPlan
    reason: str

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Materialize all input batches into a single result.

        This consumes the entire input (via execute_batches if streaming,
        otherwise via execute) and returns the complete dataset.
        """
        import numpy as np
        import pyarrow as pa

        # If input doesn't support streaming, just execute directly
        if not self.input.supports_streaming:
            return self.input.execute(context)

        # Consume all batches from streaming input
        batches: list[ArrayDict] = list(self.input.execute_batches(context))

        if not batches:
            return {}

        if len(batches) == 1:
            return batches[0]

        # Concatenate all batches
        columns = _get_ordered_columns(batches)
        result: ArrayDict = {}

        for col in columns:
            arrays_to_concat = [batch[col] for batch in batches if col in batch]
            if not arrays_to_concat:
                continue

            first = arrays_to_concat[0]
            if isinstance(first, (pa.Array, pa.ChunkedArray)):
                # Arrow concat
                result[col] = pa.concat_arrays(
                    [
                        arr if isinstance(arr, pa.Array) else arr.combine_chunks()
                        for arr in arrays_to_concat
                    ]
                )
            else:
                # NumPy concat
                result[col] = np.concatenate(arrays_to_concat)

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.input.output_schema

    @property
    def is_pipeline_breaker(self) -> bool:
        """Materialize is always a pipeline breaker by definition."""
        return True

    @property
    def supports_streaming(self) -> bool:
        """Materialize does not support streaming output."""
        return False


# =============================================================================
# Fused Pipeline (Operator Fusion)
# =============================================================================


@dataclass
class FusedOperation:
    """
    A single operation within a fused pipeline.

    Parameters
    ----------
    op_type : str
        Type of operation: "filter", "project", or "limit".
    predicate : Expr or None
        For filters: the predicate to evaluate.
    exprs : tuple[Expr, ...] or None
        For projects: the expressions to compute.
    limit_n : int or None
        For limits: number of rows to return.
    """

    op_type: Literal["filter", "project", "limit"]
    predicate: Expr | None = None
    exprs: tuple[Expr, ...] | None = None
    limit_n: int | None = None


@dataclass
class PhysicalFusedPipeline(PhysicalPlan):
    """
    Fused execution of multiple operators in a single pass.

    Operator fusion combines chains of Filter, Project, and Limit operators
    into a single physical operator that processes data in one pass. This
    provides significant performance benefits:

    1. **Reduced allocations**: No intermediate arrays between operators
    2. **Better cache locality**: Data stays hot in CPU cache
    3. **Early termination**: Filter+Limit can stop as soon as enough rows pass
    4. **Simplified spilling**: Fewer intermediates to track

    Fuseable Patterns
    -----------------
    - Filter → Project: Evaluate predicate first, only compute expressions
      for rows that pass the filter
    - Project → Project: Combine expressions into single evaluation
    - Filter → Limit: Stop processing as soon as limit is reached
    - Filter → Filter: Combine predicates with AND

    Example
    -------
    Before fusion:
        Scan → Filter(x > 0) → Project(x, y*2) → Filter(y < 10) → Limit(100)

    After fusion:
        Scan → FusedPipeline([
            Filter(x > 0),
            Project(x, y*2),
            Filter(y < 10),
            Limit(100)
        ])

    The fused pipeline processes batches:
    1. Load batch from scan
    2. Apply Filter(x > 0) to get mask
    3. For passing rows, compute y*2
    4. Apply Filter(y < 10) on computed values
    5. Emit rows until 100 reached, then stop

    Parameters
    ----------
    input : PhysicalPlan
        The input plan (typically a scan or another non-fuseable operator).
    operations : tuple[FusedOperation, ...]
        Sequence of operations to apply in order.
    schema : Schema
        Output schema after all operations.
    """

    input: PhysicalPlan
    operations: tuple[FusedOperation, ...]
    schema: Schema

    @property
    def supports_streaming(self) -> bool:
        # Fused pipeline supports streaming if input does
        return self.input.supports_streaming

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """Execute fused pipeline on full input."""
        input_arrays = self.input.execute(context)
        return self._execute_fused(input_arrays, context)

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """Stream fused pipeline with early termination for limits."""
        # Check if we have a limit operation
        limit_n = None
        for op in self.operations:
            if op.op_type == "limit" and op.limit_n is not None:
                limit_n = op.limit_n
                break

        rows_yielded = 0

        for batch in self.input.execute_batches(context):
            if not batch:
                continue

            remaining = limit_n - rows_yielded if limit_n else None
            result = self._execute_fused(batch, context, remaining=remaining)

            if result:
                first_arr = next(iter(result.values()))
                batch_len = len(first_arr)
                if batch_len > 0:
                    yield result
                    rows_yielded += batch_len

                    # Early termination for limit
                    if limit_n and rows_yielded >= limit_n:
                        return

    def _execute_fused(
        self,
        input_arrays: ArrayDict,
        context: ExecutionContext,
        remaining: int | None = None,
    ) -> ArrayDict:
        """
        Execute all fused operations on input arrays.

        Parameters
        ----------
        input_arrays : ArrayDict
            Input arrays to process.
        context : ExecutionContext
            Execution context.
        remaining : int or None
            For streaming with limit: how many more rows we need.

        Returns
        -------
        ArrayDict
            Result after applying all operations.
        """
        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.types import is_index_col
        from pandas.lazy.expr import extract_output_name

        if not input_arrays:
            return {}

        # Track current arrays and mask through the pipeline
        current_arrays = input_arrays
        current_mask: np.ndarray | None = None  # Boolean mask of valid rows

        for op in self.operations:
            if op.op_type == "filter":
                # Evaluate predicate
                evaluator = ArrayEvaluator(current_arrays, preferred_backend="auto")
                pred_result = evaluator.evaluate(op.predicate._ir)

                # Convert to numpy mask
                if isinstance(pred_result, (pa.Array, pa.ChunkedArray)):
                    new_mask = pred_result.to_numpy(zero_copy_only=False)
                else:
                    new_mask = np.asarray(pred_result)

                # Combine with existing mask
                if current_mask is not None:
                    current_mask = current_mask & new_mask
                else:
                    current_mask = new_mask

            elif op.op_type == "project":
                # Apply current mask to arrays before projection — but only
                # to the columns the projection actually references (plus
                # index columns). Masking first and projecting second spent
                # ~60% of filter+select runtime filtering payload columns
                # (often strings) that the projection immediately dropped.
                if current_mask is not None:
                    if op.exprs:
                        from pandas.lazy.optimize.utils import (
                            get_referenced_columns,
                        )

                        needed: set[str] = set()
                        for expr in op.exprs:
                            needed |= get_referenced_columns(expr)
                        if needed.issubset(current_arrays.keys()):
                            current_arrays = {
                                name: arr
                                for name, arr in current_arrays.items()
                                if name in needed or is_index_col(name)
                            }
                        # else: a reference is missing (should not happen);
                        # fall through and mask everything - safe fallback
                    current_arrays = self._apply_mask(current_arrays, current_mask)
                    current_mask = None  # Mask is now applied

                # Evaluate expressions
                evaluator = ArrayEvaluator(current_arrays, preferred_backend="auto")
                result: ArrayDict = {}

                # Keep index columns
                for name, arr in current_arrays.items():
                    if is_index_col(name):
                        result[name] = arr

                # Evaluate projection expressions
                if current_arrays:
                    arr_len = len(next(iter(current_arrays.values())))
                else:
                    arr_len = 0
                for expr in op.exprs:
                    name = extract_output_name(expr)
                    value = evaluator.evaluate(expr._ir)
                    if not isinstance(value, (np.ndarray, pa.Array, pa.ChunkedArray)):
                        value = np.full(arr_len, value)
                    result[name] = value

                current_arrays = result

            elif op.op_type == "limit":
                # Apply any pending mask first
                if current_mask is not None:
                    current_arrays = self._apply_mask(current_arrays, current_mask)
                    current_mask = None

                # Apply limit
                limit_n = op.limit_n
                if remaining is not None:
                    limit_n = min(limit_n, remaining)

                if current_arrays:
                    first_arr = next(iter(current_arrays.values()))
                    if len(first_arr) > limit_n:
                        current_arrays = self._slice_arrays(current_arrays, 0, limit_n)

        # Apply any remaining mask
        if current_mask is not None:
            current_arrays = self._apply_mask(current_arrays, current_mask)

        return current_arrays

    def _apply_mask(self, arrays: ArrayDict, mask: np.ndarray) -> ArrayDict:
        """Apply boolean mask to all arrays.

        Converts the mask to indices once and delegates to
        _take_all_columns: backend-preserving gathers (pc.take handles
        ChunkedArray directly - the previous per-column loop paid a
        combine_chunks copy and rebuilt the Arrow mask per column) and
        threshold-gated parallel fan-out across columns.
        """
        import numpy as np

        indices = np.flatnonzero(np.asarray(mask))
        return _take_all_columns(arrays, indices)

    def _slice_arrays(self, arrays: ArrayDict, start: int, end: int) -> ArrayDict:
        """Slice all arrays."""
        result: ArrayDict = {}
        for name, arr in arrays.items():
            result[name] = arr[start:end]
        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


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

        # Always extract source index as special column(s)
        # This is needed for:
        # 1. reset_index() to work (adds index as column)
        # 2. Filters/joins to track row correspondence
        # Whether the index is reconstructed at the end depends on preserve_index
        # OR whether there's an explicit set_index() in the plan (which marks
        # the context as having a user-specified index).
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


@dataclass
class PhysicalParquetScan(PhysicalPlan):
    """
    Physical scan of Parquet file(s).

    Supports predicate and projection pushdown to minimize I/O.
    Uses PyArrow for efficient Parquet reading.

    Streaming Execution
    -------------------
    This operator supports streaming execution via `execute_batches()`.
    When a `limit` is set, it enables early termination - only reading
    enough row groups to satisfy the limit. This can provide significant
    speedups for `head()` operations on large files.

    Parameters
    ----------
    path : str
        Path to Parquet file(s). Supports local paths, globs, and URLs.
    schema : Schema
        Output schema (after column pruning if applicable).
    columns : tuple[str, ...] | None
        Columns to read. None means all columns.
    predicate : Expr | None
        Filter predicate to push down to Parquet reader.
    limit : int | None
        Maximum number of rows to return. Enables early termination
        during streaming execution.
    """

    path: str
    schema: Schema
    columns: tuple[str, ...] | None = None
    predicate: Expr | None = None
    limit: int | None = None

    @property
    def supports_streaming(self) -> bool:
        return True

    def _resolve_paths(self) -> str | list[str]:
        """Resolve path, expanding glob patterns if needed."""
        path = self.path
        if "*" in path and "://" not in path:
            import glob as glob_module

            files = sorted(glob_module.glob(path))
            if not files:
                raise FileNotFoundError(f"No files found matching pattern: {path}")
            return files
        return path

    def _execute_small_limit_batches(
        self, paths: str | list[str], context: ExecutionContext
    ) -> Iterator[ArrayDict]:
        """Direct ParquetFile streaming for small unfiltered limits.

        Reads files in order with iter_batches, stopping as soon as the
        limit is satisfied - no Dataset scanner, no fragment readahead.
        """
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        from pandas.lazy.backends.types import INDEX_COL_NAME

        file_list = [paths] if isinstance(paths, str) else list(paths)
        columns = list(self.columns) if self.columns else None
        remaining = self.limit
        row_offset = 0

        for path in file_list:
            if remaining <= 0:
                return
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=max(remaining, 1024), columns=columns
            ):
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                arrays: ArrayDict = {
                    name: batch.column(name) for name in batch.schema.names
                }
                arrays[INDEX_COL_NAME] = pa.array(
                    np.arange(row_offset, row_offset + batch.num_rows, dtype=np.int64)
                )
                context.index_is_multi = False
                context.index_names = [None]
                yield arrays
                row_offset += batch.num_rows
                remaining -= batch.num_rows
                if remaining <= 0:
                    return

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Stream batches from Parquet file(s).

        Uses PyArrow's Dataset API for efficient batch iteration with
        predicate pushdown. Supports early termination when a limit
        is set.

        Yields
        ------
        ArrayDict
            Batches of data from the Parquet file(s).
        """
        import pyarrow as pa
        import pyarrow.dataset as ds

        from pandas.lazy.backends.types import INDEX_COL_NAME

        # Resolve file paths
        paths = self._resolve_paths()

        # Create dataset (handles single file, list of files, or directory)
        dataset = ds.dataset(paths, format="parquet")

        # Build scanner with pushdown optimizations
        filter_expr = None
        if self.predicate is not None:
            filter_expr = self._build_arrow_filters(self.predicate)

        # Small unfiltered limits bypass the Dataset scanner entirely:
        # the scanner pays ~50 ms of fixed startup on a multi-file
        # dataset (fragment readahead opens and pre-decodes across all
        # files) regardless of batch size, while ParquetFile.iter_batches
        # reads the first 1,000 rows of the first file in ~6 ms. Streams
        # files in order, stopping at the limit.
        if (
            self.predicate is None
            and self.limit is not None
            and self.limit <= context.batch_size
        ):
            yield from self._execute_small_limit_batches(paths, context)
            return

        scanner = dataset.scanner(
            columns=list(self.columns) if self.columns else None,
            filter=filter_expr,
            batch_size=context.batch_size,
        )

        rows_yielded = 0
        row_offset = 0  # Track row offset for index generation

        for batch in scanner.to_batches():
            batch_len = batch.num_rows
            if batch_len == 0:
                continue

            # Check limit for early termination
            if self.limit is not None:
                remaining = self.limit - rows_yielded
                if remaining <= 0:
                    return
                if batch_len > remaining:
                    # Slice the batch to exact limit
                    batch = batch.slice(0, remaining)
                    batch_len = remaining

            # Convert RecordBatch to ArrayDict
            arrays: ArrayDict = {}
            for col_name in batch.schema.names:
                arrays[col_name] = batch.column(col_name)

            # Generate index column for this batch. np.arange, not a
            # Python range: pa.array(range(...)) iterates one Python int
            # per row - it dominated full-scan time (104 ms for a
            # 2.9M-row file whose raw threaded read costs 9 ms).
            import numpy as np

            arrays[INDEX_COL_NAME] = pa.array(
                np.arange(row_offset, row_offset + batch_len, dtype=np.int64)
            )
            context.index_is_multi = False
            context.index_names = [None]

            yield arrays

            rows_yielded += batch_len
            row_offset += batch_len

            # Early termination check
            if self.limit is not None and rows_yielded >= self.limit:
                return

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute and return all data at once.

        For non-streaming execution or when downstream operators need
        all data. Materializes batches from execute_batches() into
        a single ArrayDict.
        """
        import pyarrow as pa

        from pandas.lazy.backends.types import INDEX_COL_NAME

        # Collect all batches
        batches = list(self.execute_batches(context))

        if not batches:
            # Return empty ArrayDict with correct schema
            arrays: ArrayDict = {}
            for col_name in self.schema.fields:
                arrays[col_name] = pa.array([])
            arrays[INDEX_COL_NAME] = pa.array([], type=pa.int64())
            return arrays

        if len(batches) == 1:
            return batches[0]

        # Concatenate all batches preserving column order
        result: ArrayDict = {}
        for col_name in _get_ordered_columns(batches):
            chunks = [batch[col_name] for batch in batches if col_name in batch]
            if chunks:
                # Combine into single contiguous array
                chunked = pa.chunked_array(chunks)
                result[col_name] = chunked.combine_chunks()

        return result

    def _build_arrow_filters(self, predicate: Expr) -> list | None:
        """
        Convert lazy predicate to PyArrow filter expression.

        PyArrow supports filter pushdown in the format:
        - Simple: [("col", "op", value)]
        - Compound: [[("col1", ">", 5)], [("col2", "<", 10)]]  # OR of ANDs

        Returns None if predicate cannot be pushed down.
        """

        ir = predicate._ir

        # Try to convert to PyArrow compute expression for row group filtering
        arrow_expr = self._ir_to_arrow_expr(ir)
        if arrow_expr is not None:
            # Return as PyArrow compute expression
            return arrow_expr

        return None

    def _ir_to_arrow_expr(self, ir):
        """Convert IR node to PyArrow compute expression."""
        import pyarrow.compute as pc

        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        if isinstance(ir, FieldRef):
            return pc.field(ir.name)

        if isinstance(ir, Literal):
            return ir.value

        if isinstance(ir, Call):
            # Binary comparison operators
            if ir.function == "greater" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.greater(left, right)

            elif ir.function == "greater_equal" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.greater_equal(left, right)

            elif ir.function == "less" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.less(left, right)

            elif ir.function == "less_equal" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.less_equal(left, right)

            elif ir.function == "equal" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.equal(left, right)

            elif ir.function == "not_equal" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.not_equal(left, right)

            # Logical operators
            # Use and_kleene/or_kleene which handle expression types correctly
            elif ir.function == "and_" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.and_kleene(left, right)

            elif ir.function == "or_" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.or_kleene(left, right)

            elif ir.function == "invert" and len(ir.args) == 1:
                arg = self._ir_to_arrow_expr(ir.args[0])
                if arg is not None:
                    return pc.invert(arg)

            # Null-checking operators - can use row group null_count statistics
            elif ir.function == "is_null" and len(ir.args) == 1:
                arg = self._ir_to_arrow_expr(ir.args[0])
                if arg is not None:
                    return pc.is_null(arg)

            elif ir.function == "is_not_null" and len(ir.args) == 1:
                arg = self._ir_to_arrow_expr(ir.args[0])
                if arg is not None:
                    return pc.is_valid(arg)

            # isin operator - enables row group filtering on dict/categorical
            elif ir.function == "isin" and len(ir.args) >= 1:
                import pyarrow as pa

                col = self._ir_to_arrow_expr(ir.args[0])
                # Values can be in args[1] or kwargs["values"]
                values = ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("values")
                if col is not None and values is not None:
                    # Convert values to PyArrow array for is_in
                    try:
                        if isinstance(values, (list, tuple)):
                            value_set = pa.array(values)
                        else:
                            value_set = pa.array(list(values))
                        return pc.is_in(col, value_set=value_set)
                    except (TypeError, pa.ArrowInvalid):
                        # Cannot convert values to Arrow, fall back
                        pass

            # String operations - enable row group filtering on string columns
            elif ir.function == "str_startswith" and len(ir.args) >= 1:
                col = self._ir_to_arrow_expr(ir.args[0])
                prefix_arg = ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("prefix")
                # Extract value if it's a Literal
                if isinstance(prefix_arg, Literal):
                    prefix = prefix_arg.value
                else:
                    prefix = prefix_arg
                if col is not None and isinstance(prefix, str):
                    return pc.starts_with(col, prefix)

            elif ir.function == "str_endswith" and len(ir.args) >= 1:
                col = self._ir_to_arrow_expr(ir.args[0])
                suffix_arg = ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("suffix")
                # Extract value if it's a Literal
                if isinstance(suffix_arg, Literal):
                    suffix = suffix_arg.value
                else:
                    suffix = suffix_arg
                if col is not None and isinstance(suffix, str):
                    return pc.ends_with(col, suffix)

            elif ir.function == "str_contains" and len(ir.args) >= 1:
                col = self._ir_to_arrow_expr(ir.args[0])
                pattern_arg = (
                    ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("pattern")
                )
                # Extract value if it's a Literal
                if isinstance(pattern_arg, Literal):
                    pattern = pattern_arg.value
                else:
                    pattern = pattern_arg
                if col is not None and isinstance(pattern, str):
                    # Use match_substring for contains
                    return pc.match_substring(col, pattern)

        # Cannot convert this expression
        return None

    def children(self) -> list[PhysicalPlan]:
        return []

    @property
    def output_schema(self) -> Schema:
        return self.schema


@dataclass
class PhysicalCSVScan(PhysicalPlan):
    """
    Physical scan of CSV file(s).

    Supports projection pushdown to minimize I/O. Predicates are applied
    after reading since CSV doesn't support native predicate pushdown.

    Uses PyArrow CSV reader for efficient batch processing.

    Parameters
    ----------
    path : str
        Path to CSV file(s). Supports local paths and globs.
    schema : Schema
        Output schema (after column pruning if applicable).
    columns : tuple[str, ...] | None
        Columns to read. None means all columns.
    predicate : Expr | None
        Filter predicate to apply after reading.
    limit : int | None
        Maximum number of rows to return. Enables early termination.
    sep : str
        Delimiter/separator character.
    header : bool
        Whether the file has a header row.
    skip_rows : int
        Number of rows to skip at the start.
    n_rows : int | None
        Maximum number of rows to read from source.
    """

    path: str
    schema: Schema
    columns: tuple[str, ...] | None = None
    predicate: Expr | None = None
    limit: int | None = None
    sep: str = ","
    header: bool = True
    skip_rows: int = 0
    n_rows: int | None = None

    @property
    def supports_streaming(self) -> bool:
        return True

    def _resolve_paths(self) -> str | list[str]:
        """Resolve path, expanding glob patterns if needed."""
        path = self.path
        if "*" in path and "://" not in path:
            import glob as glob_module

            files = sorted(glob_module.glob(path))
            if not files:
                raise FileNotFoundError(f"No files found matching pattern: {path}")
            return files
        return path

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Stream batches from CSV file(s).

        Uses PyArrow's CSV streaming reader for efficient batch iteration.
        Supports early termination when a limit is set.

        Yields
        ------
        ArrayDict
            Batches of data from the CSV file(s).
        """
        import pyarrow as pa
        from pyarrow import csv

        from pandas.lazy.backends.types import INDEX_COL_NAME

        # Resolve file paths
        paths = self._resolve_paths()
        if isinstance(paths, str):
            paths = [paths]

        # Configure CSV read options
        read_options = csv.ReadOptions(
            skip_rows=self.skip_rows,
            autogenerate_column_names=not self.header,
            block_size=context.batch_size * 1024,  # Approximate bytes per batch
        )
        parse_options = csv.ParseOptions(delimiter=self.sep)

        rows_yielded = 0
        row_offset = 0

        for file_path in paths:
            # Create streaming reader
            reader = csv.open_csv(
                file_path,
                read_options=read_options,
                parse_options=parse_options,
            )

            for batch in reader:
                batch_len = batch.num_rows
                if batch_len == 0:
                    continue

                # Apply n_rows limit from source
                if self.n_rows is not None:
                    remaining = self.n_rows - rows_yielded
                    if remaining <= 0:
                        return
                    if batch_len > remaining:
                        batch = batch.slice(0, remaining)
                        batch_len = remaining

                # Check limit for early termination
                if self.limit is not None:
                    remaining = self.limit - rows_yielded
                    if remaining <= 0:
                        return
                    if batch_len > remaining:
                        batch = batch.slice(0, remaining)
                        batch_len = remaining

                # Convert RecordBatch to ArrayDict
                arrays: ArrayDict = {}

                # If columns specified, only include those
                columns_to_read = (
                    list(self.columns) if self.columns else batch.schema.names
                )
                for col_name in columns_to_read:
                    if col_name in batch.schema.names:
                        arrays[col_name] = batch.column(col_name)

                # Generate index column for this batch
                arrays[INDEX_COL_NAME] = pa.array(
                    range(row_offset, row_offset + batch_len)
                )
                context.index_is_multi = False
                context.index_names = [None]

                yield arrays

                rows_yielded += batch_len
                row_offset += batch_len

                # Early termination check
                if self.limit is not None and rows_yielded >= self.limit:
                    return
                if self.n_rows is not None and rows_yielded >= self.n_rows:
                    return

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute and return all data at once.

        For non-streaming execution or when downstream operators need
        all data. Materializes batches from execute_batches() into
        a single ArrayDict.
        """
        import pyarrow as pa

        from pandas.lazy.backends.types import INDEX_COL_NAME

        # Collect all batches
        batches = list(self.execute_batches(context))

        if not batches:
            # Return empty ArrayDict with correct schema
            arrays: ArrayDict = {}
            for col_name in self.schema.fields:
                arrays[col_name] = pa.array([])
            arrays[INDEX_COL_NAME] = pa.array([], type=pa.int64())
            return arrays

        if len(batches) == 1:
            return batches[0]

        # Concatenate all batches preserving column order
        result: ArrayDict = {}
        for col_name in _get_ordered_columns(batches):
            chunks = [batch[col_name] for batch in batches if col_name in batch]
            if chunks:
                # Combine into single contiguous array
                chunked = pa.chunked_array(chunks)
                result[col_name] = chunked.combine_chunks()

        return result

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
        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends.types import is_index_col

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
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.memory_pool import PoolingStrategy
        from pandas.lazy.expr import extract_output_name

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

        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.memory_pool import PoolingStrategy
        from pandas.lazy.expr import extract_output_name

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
        import os

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

        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.convert import (
            get_array_backend,
            to_arrow,
        )
        from pandas.lazy.backends.types import is_index_col

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
            import pyarrow.compute as pc

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


# =============================================================================
# Aggregation Nodes
# =============================================================================

# Merge functions for two-phase streaming aggregation
# Maps: original_agg -> merge_agg (to combine partial results)
_STREAMING_MERGE_FUNCS: dict[str, str] = {
    "sum": "sum",  # sum of partial sums
    "count": "sum",  # sum of partial counts
    "min": "min",  # min of partial mins
    "max": "max",  # max of partial maxes
    "first": "first",  # first of firsts (batch order preserved)
    "last": "last",  # last of lasts (batch order preserved)
    # mean handled specially: track (sum, count), then divide
}


@dataclass
class PhysicalHashAggregate(PhysicalPlan):
    """
    Hash-based aggregation.

    Uses hash table for grouping - good for high cardinality.
    Uses kernel dispatch for efficient aggregation on Arrow or NumPy arrays.

    Arrow GroupBy Optimization
    --------------------------
    When input data is Arrow-backed (e.g., from Parquet scan), this operator
    uses PyArrow's native group_by() function which provides:

    - Zero-copy aggregation on Arrow tables
    - Multi-threaded execution for most aggregations
    - Efficient memory usage through Arrow's columnar format
    - Hardware-optimized SIMD operations in Arrow's C++ backend

    The Arrow path is automatically selected when:
    1. Input arrays are PyArrow arrays (from Parquet scan or Arrow-backed DataFrame)
    2. All aggregation functions are supported by PyArrow

    Supported Arrow aggregations: sum, mean, min, max, count, first, last,
    std, var, any, all, count_distinct/nunique, prod
    """

    input: PhysicalPlan
    group_by: tuple[Expr, ...]
    agg_exprs: tuple[Expr, ...]
    schema: Schema
    # Set by the engine's decision layer (engine/decisions.py) at plan
    # time from the input schema's per-column storage backends; when
    # None, the runtime relevant-columns logic below decides.
    planned_backend: str | None = None

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

        # Extract group-by column names and aggregation specs early
        # (needed to decide if streaming is beneficial)
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

        # Try streaming aggregation for memory-efficient processing of large inputs
        # (e.g., Concat nodes that scan multiple files).
        #
        # Streaming is only possible for "mergeable" aggregations where partial
        # results can be combined algebraically:
        # - sum: sum of partial sums
        # - count: sum of partial counts
        # - min/max: min/max of partial min/max
        # - mean: sum of sums / sum of counts
        # - first/last: first of firsts / last of lasts (requires ordering)
        #
        # Non-mergeable aggregations (nunique, std, var) require full materialization
        # because there's no algebraic way to merge partial results without
        # maintaining per-group state that can grow unbounded.
        #
        # Trade-off: Streaming uses less peak memory but has ~2x merge overhead.
        # For data that fits in RAM, non-streaming is faster.
        # For data exceeding RAM, streaming prevents OOM errors.
        if (
            self.input.supports_streaming
            and group_cols
            and self._can_stream_aggregation(agg_specs)
        ):
            return self._execute_streaming_aggregation(group_cols, agg_specs, context)

        input_arrays = self.input.execute(context)

        # Backend choice. The decision layer (engine/decisions.py) plans
        # this from the input schema when possible; otherwise fall back
        # to inspecting the materialized arrays at runtime.
        #
        # Either way the rule is the same: decide from the columns this
        # aggregation actually uses - the group keys and the aggregation
        # value columns. Unrelated payload columns must not influence
        # the choice (a previous version fell back to the backend of the
        # *first* input column, which sent Arrow-string-keyed groupbys
        # down the NumPy path - factorizing the key through object
        # arrays, ~5x slower than Arrow's table groupby). Arrow is
        # preferred when any relevant column is Arrow-backed; string
        # group keys are exactly where Arrow's hash aggregation wins.
        if self.planned_backend is not None:
            backend = self.planned_backend
        else:
            value_cols = {col for _, col, _ in agg_specs}
            relevant_cols = value_cols.union(group_cols or ())
            relevant_backends = {
                get_array_backend(input_arrays[col])
                for col in relevant_cols
                if col in input_arrays
            }
            if "arrow" in relevant_backends:
                backend = "arrow"
            elif relevant_backends:
                backend = "numpy"
            else:
                first_col = next(iter(input_arrays.values()))
                backend = get_array_backend(first_col)

        if not group_cols:
            # Global aggregation - no grouping keys
            return self._execute_global_aggregation(
                input_arrays, agg_specs, backend, context
            )

        # Grouped aggregation - prefer Arrow table-based groupby for efficiency
        return self._execute_grouped_aggregation(
            input_arrays, group_cols, agg_specs, backend, context
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
        context: ExecutionContext,
    ) -> ArrayDict:
        """
        Execute grouped aggregation using kernel dispatch.

        When data is Arrow-backed, uses PyArrow's native Table.group_by()
        for both single-key and multi-key groupby. This provides:
        - Zero-copy aggregation (no conversion overhead)
        - Multi-threaded execution
        - Efficient SIMD operations from Arrow's C++ backend

        When preserve_index is enabled in context, group keys are stored
        as index columns (using __index__ naming convention) rather than
        as regular data columns, mimicking pandas groupby behavior.
        """
        from pandas.lazy.backends import has_kernel

        # For Arrow data, prefer the Arrow Table-based groupby path
        # This handles both single-key and multi-key cases efficiently
        if backend == "arrow" and has_kernel("group_by", "arrow"):
            if self._can_use_arrow_groupby(agg_specs):
                # Check if we have nunique - PyArrow's hash_count_distinct is slow
                # Use pandas Cython path for nunique (6x faster)
                nunique_aggs = [
                    spec
                    for spec in agg_specs
                    if spec[2] in ("nunique", "n_unique", "count_distinct")
                ]
                other_aggs = [
                    spec
                    for spec in agg_specs
                    if spec[2] not in ("nunique", "n_unique", "count_distinct")
                ]

                if nunique_aggs:
                    # Hybrid path: pandas Cython for nunique, Arrow for rest
                    return self._execute_hybrid_groupby(
                        input_arrays,
                        group_cols,
                        nunique_aggs,
                        other_aggs,
                        context,
                    )

                return self._execute_arrow_table_groupby(
                    input_arrays, group_cols, agg_specs, context
                )

        # Note: The NumPy path is vectorized and efficient (using factorize +
        # get_group_index + duplicated + bincount for nunique).

        # For single-key groupby, use optimized single-key kernels
        # For multi-key, use hash_aggregate kernel
        if len(group_cols) == 1:
            return self._execute_single_key_groupby(
                input_arrays, group_cols[0], agg_specs, backend, context
            )
        else:
            return self._execute_multi_key_groupby(
                input_arrays, group_cols, agg_specs, backend, context
            )

    def _can_use_arrow_groupby(self, agg_specs: list[tuple[str, str, str]]) -> bool:
        """Check if all aggregations are supported by Arrow's group_by."""
        # PyArrow's hash aggregation supports these functions
        arrow_supported_aggs = {
            "sum",
            "mean",
            "min",
            "max",
            "count",
            "first",
            "last",
            "std",
            "var",
            "any",
            "all",
            "count_distinct",
            "nunique",
            "n_unique",
            "prod",
        }
        return all(agg_func in arrow_supported_aggs for _, _, agg_func in agg_specs)

    def _execute_single_key_groupby(
        self,
        input_arrays: ArrayDict,
        group_col: str,
        agg_specs: list[tuple[str, str, str]],
        backend: str,
        context: ExecutionContext,
    ) -> ArrayDict:
        """Execute single-key groupby using optimized kernels."""
        import pyarrow as pa

        from pandas.lazy.backends import (
            dispatch_kernel,
            has_kernel,
        )
        from pandas.lazy.backends.convert import ensure_backend
        from pandas.lazy.backends.types import INDEX_COL_NAME

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

        # Store group key - either as index column or data column
        # When preserve_index=True, group keys become the index (pandas-style)
        ordered_result: ArrayDict = {}

        if context.preserve_index:
            # Store group key as index column for index reconstruction
            ordered_result[INDEX_COL_NAME] = unique_keys
            context.index_names = [group_col]
            context.index_is_multi = False
        else:
            # Default: store as regular data column
            ordered_result[group_col] = unique_keys

        ordered_result.update(result)

        return ordered_result

    def _execute_multi_key_groupby(
        self,
        input_arrays: ArrayDict,
        group_cols: list[str],
        agg_specs: list[tuple[str, str, str]],
        backend: str,
        context: ExecutionContext,
    ) -> ArrayDict:
        """Execute multi-key groupby using hash_aggregate kernel."""
        import pyarrow as pa

        from pandas.lazy.backends import (
            dispatch_kernel,
            has_kernel,
        )
        from pandas.lazy.backends.convert import ensure_backend
        from pandas.lazy.backends.types import (
            INDEX_COL_NAME,
            index_col_name,
        )

        if backend == "arrow" and has_kernel("group_by", "arrow"):
            # Use Arrow's native table-based group_by
            return self._execute_arrow_table_groupby(
                input_arrays, group_cols, agg_specs, context
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

        if context.preserve_index:
            # Store group keys as index columns for index reconstruction
            is_multi = len(group_cols) > 1
            for i, (col, arr) in enumerate(zip(group_cols, result_keys, strict=False)):
                # Use INDEX_COL_NAME for single key, index_col_name(i) for multi-key
                idx_col = index_col_name(i) if is_multi else INDEX_COL_NAME
                result[idx_col] = arr
            context.index_names = list(group_cols)
            context.index_is_multi = is_multi
        else:
            # Default: store as regular data columns
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
        context: ExecutionContext,
    ) -> ArrayDict:
        """
        Execute groupby using Arrow's native table group_by.

        This is the preferred path for Arrow data because:
        1. Zero-copy table construction from existing Arrow arrays
        2. Multi-threaded aggregation (except first/last)
        3. Vectorized SIMD operations in Arrow's C++ backend
        4. Memory-efficient columnar processing
        """
        import pyarrow as pa

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.types import (
            INDEX_COL_NAME,
            index_col_name,
            is_index_col,
        )

        # Build Arrow table from arrays (excluding index columns)
        columns = {k: v for k, v in input_arrays.items() if not is_index_col(k)}
        table = pa.table(columns)

        # Execute groupby using Arrow's native group_by
        result_table = dispatch_kernel(
            "group_by", "arrow", table, group_cols, agg_specs
        )

        # Convert result table back to ArrayDict
        # Combine chunks for consistency (Table.column returns ChunkedArray)
        result: ArrayDict = {}

        if context.preserve_index:
            # Store group keys as index columns for index reconstruction
            is_multi = len(group_cols) > 1
            for i, col in enumerate(group_cols):
                chunked = result_table.column(col)
                # Use INDEX_COL_NAME for single key, index_col_name(i) for multi-key
                idx_col = index_col_name(i) if is_multi else INDEX_COL_NAME
                if isinstance(chunked, pa.ChunkedArray):
                    result[idx_col] = chunked.combine_chunks()
                else:
                    result[idx_col] = chunked
            context.index_names = list(group_cols)
            context.index_is_multi = is_multi

            # Add aggregation result columns (skip group cols)
            for col_name in result_table.column_names:
                if col_name not in group_cols:
                    chunked = result_table.column(col_name)
                    if isinstance(chunked, pa.ChunkedArray):
                        result[col_name] = chunked.combine_chunks()
                    else:
                        result[col_name] = chunked
        else:
            # Default: store group keys as regular data columns
            for col_name in result_table.column_names:
                chunked = result_table.column(col_name)
                if isinstance(chunked, pa.ChunkedArray):
                    result[col_name] = chunked.combine_chunks()
                else:
                    result[col_name] = chunked

        return result

    def _execute_hybrid_groupby(
        self,
        input_arrays: ArrayDict,
        group_cols: list[str],
        nunique_aggs: list[tuple[str, str, str]],
        other_aggs: list[tuple[str, str, str]],
        context: ExecutionContext,
    ) -> ArrayDict:
        """
        Execute groupby with hybrid approach: pandas Cython for nunique, Arrow for rest.

        PyArrow's hash_count_distinct is ~6x slower than pandas' Cython-based
        approach using factorize + get_group_index + duplicated + bincount.
        This method uses pandas internals for nunique aggregations while
        leveraging Arrow's efficient hash aggregation for other operations.

        Parameters
        ----------
        input_arrays : ArrayDict
            Input arrays including group columns and value columns.
        group_cols : list[str]
            Column names to group by.
        nunique_aggs : list[tuple[str, str, str]]
            Aggregation specs for nunique: (output_name, input_col, agg_func)
        other_aggs : list[tuple[str, str, str]]
            Aggregation specs for non-nunique operations.
        context : ExecutionContext
            Execution context with settings like preserve_index.

        Returns
        -------
        ArrayDict
            Result with group keys and all aggregated values.
        """
        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.types import (
            INDEX_COL_NAME,
            index_col_name,
            is_index_col,
        )

        def _factorize(arr):
            """Factorize array using appropriate backend, returning NumPy codes."""
            if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                codes, uniques, n_uniques = dispatch_kernel("factorize", "arrow", arr)
                # Convert Arrow codes to NumPy for downstream indexing operations
                return codes.to_numpy(), uniques, n_uniques
            else:
                if hasattr(arr, "to_numpy"):
                    arr = arr.to_numpy(zero_copy_only=False)
                else:
                    arr = np.asarray(arr)
                return dispatch_kernel("factorize", "numpy", arr)

        result: ArrayDict = {}

        # Step 1: Factorize group keys to get group codes
        # For single key, factorize directly; for multi-key, create compound index
        if len(group_cols) == 1:
            group_arr = input_arrays[group_cols[0]]
            group_codes, group_uniques, n_groups = _factorize(group_arr)
            # Convert numpy uniques to arrow for consistency
            if not isinstance(group_uniques, pa.Array):
                group_uniques_arrow = pa.array(group_uniques)
            else:
                group_uniques_arrow = group_uniques
        else:
            # Multi-key: factorize each key and create compound group index
            key_codes_list = []
            shapes = []

            for col in group_cols:
                arr = input_arrays[col]
                codes, _, n_unique = _factorize(arr)
                key_codes_list.append(codes)
                shapes.append(n_unique)

            # Create compound group index
            group_codes = dispatch_kernel(
                "get_group_index", "numpy", key_codes_list, tuple(shapes)
            )

            # Re-factorize to get contiguous group codes
            group_codes, _, n_groups = dispatch_kernel(
                "factorize", "numpy", group_codes
            )

            # Note: group_uniques_arrow not set for multi-key case
            # We handle this in the else branch below

        # Step 2: Compute nunique for each nunique aggregation
        for output_name, col_name, _ in nunique_aggs:
            value_arr = input_arrays[col_name]
            value_codes, _, n_values = _factorize(value_arr)

            # Create compound (group, value) index
            compound_index = dispatch_kernel(
                "get_group_index",
                "numpy",
                [group_codes, value_codes],
                (n_groups, n_values),
            )

            # Find duplicates using Cython hash table
            is_dup = dispatch_kernel("duplicated", "numpy", compound_index)

            # Count unique values per group
            nunique_result = np.bincount(
                group_codes[~is_dup], minlength=n_groups
            ).astype(np.int64)

            result[output_name] = pa.array(nunique_result)

        # Step 3: Execute other aggregations with Arrow (if any)
        if other_aggs:
            # Build Arrow table
            columns = {k: v for k, v in input_arrays.items() if not is_index_col(k)}
            table = pa.table(columns)

            # Execute groupby for non-nunique aggregations
            other_result_table = dispatch_kernel(
                "group_by", "arrow", table, group_cols, other_aggs
            )

            # Extract results
            for output_name, _, _ in other_aggs:
                chunked = other_result_table.column(output_name)
                if isinstance(chunked, pa.ChunkedArray):
                    result[output_name] = chunked.combine_chunks()
                else:
                    result[output_name] = chunked

            # Use group keys from Arrow result (guaranteed same order)
            if context.preserve_index:
                is_multi = len(group_cols) > 1
                for i, col in enumerate(group_cols):
                    chunked = other_result_table.column(col)
                    idx_col = index_col_name(i) if is_multi else INDEX_COL_NAME
                    if isinstance(chunked, pa.ChunkedArray):
                        result[idx_col] = chunked.combine_chunks()
                    else:
                        result[idx_col] = chunked
                context.index_names = list(group_cols)
                context.index_is_multi = is_multi
            else:
                for col in group_cols:
                    chunked = other_result_table.column(col)
                    if isinstance(chunked, pa.ChunkedArray):
                        result[col] = chunked.combine_chunks()
                    else:
                        result[col] = chunked
        # Only nunique aggregations - add group keys from our factorization
        elif len(group_cols) == 1:
            if context.preserve_index:
                result[INDEX_COL_NAME] = group_uniques_arrow
                context.index_names = list(group_cols)
                context.index_is_multi = False
            else:
                result[group_cols[0]] = group_uniques_arrow
        else:
            # Multi-key: need to reconstruct unique key combinations
            # This is a fallback - in practice, mixed aggs are common
            # For pure nunique multi-key, fall back to Arrow path
            # (slower but correct)
            return self._execute_arrow_table_groupby(
                input_arrays, group_cols, nunique_aggs, context
            )

        return result

    # =========================================================================
    # Streaming Aggregation (v2) - Array-based, uses optimized kernels
    # =========================================================================

    def _can_stream_aggregation(self, agg_specs: list[tuple[str, str, str]]) -> bool:
        """
        Check if all aggregations can be computed in streaming mode.

        Streaming requires that partial results can be merged algebraically.
        Non-mergeable aggregations (nunique, std, var) require full data.

        Parameters
        ----------
        agg_specs : list[tuple[str, str, str]]
            Aggregation specs: (output_name, input_col, agg_func)

        Returns
        -------
        bool
            True if all aggregations are streaming-compatible.
        """
        mergeable_aggs = set(_STREAMING_MERGE_FUNCS.keys()) | {"mean"}
        return all(agg_func in mergeable_aggs for _, _, agg_func in agg_specs)

    def _execute_streaming_aggregation(
        self,
        group_cols: list[str],
        agg_specs: list[tuple[str, str, str]],
        context: ExecutionContext,
    ) -> ArrayDict:
        """
        Execute aggregation in streaming mode using two-phase map-reduce.

        This method keeps data as Arrow arrays throughout and uses the same
        optimized Arrow group_by kernel for both phases:

        Phase 1 (Map): For each batch, compute partial aggregates per group
        Phase 2 (Reduce): Concatenate partials and re-aggregate with merge funcs

        This enables memory-efficient aggregation for streaming sources like
        ConcatNode (multi-file scans) without materializing all data at once.

        Parameters
        ----------
        group_cols : list[str]
            Column names to group by.
        agg_specs : list[tuple[str, str, str]]
            Aggregation specs: (output_name, input_col, agg_func)
        context : ExecutionContext
            Execution context.

        Returns
        -------
        ArrayDict
            Aggregated result with group keys and aggregated values.
        """
        import pyarrow as pa

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.types import (
            INDEX_COL_NAME,
            index_col_name,
            is_index_col,
        )

        # Separate mean aggregations (need special handling)
        mean_aggs: list[tuple[str, str]] = []  # (output_name, input_col)
        direct_aggs: list[tuple[str, str, str]] = []  # (output_name, input_col, agg)

        for output_name, col_name, agg_func in agg_specs:
            if agg_func == "mean":
                mean_aggs.append((output_name, col_name))
            else:
                direct_aggs.append((output_name, col_name, agg_func))

        # Build per-batch aggregation specs
        # For mean, we need to compute sum and count, then divide at the end
        batch_agg_specs: list[tuple[str, str, str]] = list(direct_aggs)
        for output_name, col_name in mean_aggs:
            batch_agg_specs.append((f"__sum_{output_name}__", col_name, "sum"))
            batch_agg_specs.append((f"__count_{output_name}__", col_name, "count"))

        # Phase 1: Per-batch aggregation
        partial_tables: list[pa.Table] = []

        for batch in self.input.execute_batches(context):
            # Skip empty batches
            first_arr = next(iter(batch.values()))
            if len(first_arr) == 0:
                continue

            # Build Arrow table from batch (exclude index columns)
            columns = {k: v for k, v in batch.items() if not is_index_col(k)}
            batch_table = pa.table(columns)

            # Aggregate this batch using Arrow's optimized group_by
            partial = dispatch_kernel(
                "group_by", "arrow", batch_table, group_cols, batch_agg_specs
            )
            partial_tables.append(partial)

        # Handle empty result
        if not partial_tables:
            return self._make_empty_aggregate_result(group_cols, agg_specs, context)

        # Single batch - no merge needed
        if len(partial_tables) == 1:
            result_table = partial_tables[0]
        else:
            # Phase 2: Merge partial results
            # Concatenate all partial tables
            merged_table = pa.concat_tables(partial_tables)

            # Build merge aggregation specs
            merge_specs: list[tuple[str, str, str]] = []
            for output_name, _, agg_func in direct_aggs:
                merge_func = _STREAMING_MERGE_FUNCS[agg_func]
                merge_specs.append((output_name, output_name, merge_func))
            for output_name, _ in mean_aggs:
                # Sum the partial sums, sum the partial counts
                merge_specs.append(
                    (f"__sum_{output_name}__", f"__sum_{output_name}__", "sum")
                )
                merge_specs.append(
                    (f"__count_{output_name}__", f"__count_{output_name}__", "sum")
                )

            # Re-aggregate with merge functions
            result_table = dispatch_kernel(
                "group_by", "arrow", merged_table, group_cols, merge_specs
            )

        # Post-process: compute mean from sum/count
        if mean_aggs:
            # We need to compute mean = sum / count and remove intermediate columns
            result_columns = {
                name: result_table.column(name)
                for name in result_table.column_names
                if not name.startswith("__")
            }

            for output_name, _ in mean_aggs:
                sum_col = result_table.column(f"__sum_{output_name}__")
                count_col = result_table.column(f"__count_{output_name}__")
                # Compute mean (handle chunked arrays)
                import pyarrow.compute as pc

                mean_arr = pc.divide(
                    pc.cast(sum_col, pa.float64()), pc.cast(count_col, pa.float64())
                )
                if isinstance(mean_arr, pa.ChunkedArray):
                    mean_arr = mean_arr.combine_chunks()
                result_columns[output_name] = mean_arr

            # Rebuild table with correct column order
            final_columns = []
            final_names = []
            for col in group_cols:
                final_columns.append(result_columns[col])
                final_names.append(col)
            for output_name, _, _ in agg_specs:
                final_columns.append(result_columns[output_name])
                final_names.append(output_name)
            result_table = pa.table(dict(zip(final_names, final_columns, strict=True)))

        # Convert result table to ArrayDict
        result: ArrayDict = {}

        if context.preserve_index:
            is_multi = len(group_cols) > 1
            for i, col in enumerate(group_cols):
                idx_col = index_col_name(i) if is_multi else INDEX_COL_NAME
                chunked = result_table.column(col)
                if isinstance(chunked, pa.ChunkedArray):
                    result[idx_col] = chunked.combine_chunks()
                else:
                    result[idx_col] = chunked
            context.index_names = list(group_cols)
            context.index_is_multi = is_multi
        else:
            for col in group_cols:
                chunked = result_table.column(col)
                if isinstance(chunked, pa.ChunkedArray):
                    result[col] = chunked.combine_chunks()
                else:
                    result[col] = chunked

        for output_name, _, _ in agg_specs:
            chunked = result_table.column(output_name)
            if isinstance(chunked, pa.ChunkedArray):
                result[output_name] = chunked.combine_chunks()
            else:
                result[output_name] = chunked

        return result

    def _make_empty_aggregate_result(
        self,
        group_cols: list[str],
        agg_specs: list[tuple[str, str, str]],
        context: ExecutionContext,
    ) -> ArrayDict:
        """Create empty result for aggregation with no input rows."""
        import pyarrow as pa

        from pandas.lazy.backends.types import (
            INDEX_COL_NAME,
            index_col_name,
        )

        result: ArrayDict = {}

        if context.preserve_index:
            is_multi = len(group_cols) > 1
            for i, col in enumerate(group_cols):
                idx_col = index_col_name(i) if is_multi else INDEX_COL_NAME
                result[idx_col] = pa.array([])
            context.index_names = list(group_cols)
            context.index_is_multi = is_multi
        else:
            for col in group_cols:
                result[col] = pa.array([])

        for output_name, _, _ in agg_specs:
            result[output_name] = pa.array([])

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema

    @property
    def is_pipeline_breaker(self) -> bool:
        """Aggregate requires all rows per group before emitting results."""
        return True


# =============================================================================
# Sort Nodes
# =============================================================================

# Minimum rows before per-column gather is parallelized. Both np.take and
# pc.take release the GIL, so a thread pool over columns scales (~2.5x for
# 4 columns at 10M rows).
from pandas.lazy.cost import (
    ARROW_MULTIKEY_SORT_MIN_ROWS,
    PARALLEL_TAKE_MIN_ROWS,
)

# Minimum rows before multi-key sort routes through Arrow's table-level
# sort_indices (multi-threaded; ~1.65x over np.lexsort at 10M rows).


def _take_all_columns(input_arrays: ArrayDict, indices) -> ArrayDict:
    """
    Apply gather indices to every column, in parallel for large data.

    indices may be a NumPy array or an Arrow array; it is converted once
    for the backends that need it rather than per column.
    """
    from concurrent.futures import ThreadPoolExecutor
    import os

    import numpy as np

    from pandas.lazy.backends import dispatch_kernel
    from pandas.lazy.backends.convert import get_array_backend

    backends = {name: get_array_backend(arr) for name, arr in input_arrays.items()}
    np_indices = indices
    if not isinstance(indices, np.ndarray) and any(
        b == "numpy" for b in backends.values()
    ):
        np_indices = indices.to_numpy(zero_copy_only=False)
        if np_indices.dtype == np.uint64:
            # Arrow sort_indices yields uint64; np.take requires a
            # safe-castable (signed) index dtype
            np_indices = np_indices.astype(np.int64)

    def take_one(item):
        name, arr = item
        backend = backends[name]
        idx = np_indices if backend == "numpy" else indices
        return name, dispatch_kernel("take", backend, arr, idx)

    items = list(input_arrays.items())
    if len(items) >= 2 and len(indices) >= PARALLEL_TAKE_MIN_ROWS:
        n_workers = min(8, os.cpu_count() or 4, len(items))
        with ThreadPoolExecutor(n_workers) as ex:
            return dict(ex.map(take_one, items))
    return dict(map(take_one, items))


def _is_arrow_sortable(arr) -> bool:
    """Whether an array can be routed through Arrow's table sort."""
    import numpy as np
    import pyarrow as pa

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
        import numpy as np

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.convert import get_array_backend
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )

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
                return _take_all_columns(input_arrays, sort_indices)

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

        # Large multi-key sort: route through Arrow's table-level
        # sort_indices, which is multi-threaded (~1.65x over np.lexsort
        # at 10M rows) and supports mixed ascending/descending natively.
        if n_rows >= ARROW_MULTIKEY_SORT_MIN_ROWS and all(
            _is_arrow_sortable(arr) for arr in sort_key_arrays
        ):
            try:
                import pyarrow as pa
                import pyarrow.compute as pc

                from pandas.lazy.backends.convert import to_arrow

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
                return _take_all_columns(input_arrays, sort_indices)
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
        return _take_all_columns(input_arrays, sort_indices)

    def _execute_external_sort(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute sort using external merge sort for larger-than-memory data.

        Uses the spill manager's ExternalSorter to:
        1. Process input in batches
        2. Sort and spill each batch as a sorted run
        3. K-way merge all runs to produce final sorted output
        """
        from pandas.lazy.backends.spill import ExternalSorter
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )

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
        import numpy as np

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.convert import get_array_backend
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )

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
        import numpy as np

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
                import pyarrow as pa

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
        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends.convert import get_array_backend

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
        import numpy as np

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.convert import get_array_backend
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )

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
        import numpy as np

        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.convert import get_array_backend
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )

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
        import numpy as np

        result: ArrayDict = {}
        for name in self.schema.names:
            # Default to empty numpy array
            result[name] = np.array([])
        return result

    def _execute_materialized_topk(self, context: ExecutionContext) -> ArrayDict:
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
        import pyarrow as pa

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
        for col_name in _get_ordered_columns(batches):
            chunks = [batch[col_name] for batch in batches if col_name in batch]
            if chunks:
                if hasattr(chunks[0], "type"):
                    # Arrow arrays - concatenate
                    chunked = pa.chunked_array(chunks)
                    result[col_name] = chunked.combine_chunks()
                else:
                    # NumPy arrays
                    import numpy as np

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

    @property
    def is_pipeline_breaker(self) -> bool:
        """Distinct requires all input to deduplicate globally."""
        return True


# =============================================================================
# Join Nodes
# =============================================================================


@dataclass
class PhysicalHashJoin(PhysicalPlan):
    """
    Hash-based join with build/probe optimization.

    Good for equi-joins with reasonable key cardinality.
    Uses kernel dispatch for efficient join on Arrow or NumPy arrays.

    Build/Probe Optimization
    ------------------------
    For inner joins and semi/anti joins, the hash table is built on the
    smaller side for optimal performance. This reduces memory usage and
    improves cache locality.

    - Inner join: Build on smaller side, probe with larger
    - Left join: Must build on right (probe side is left)
    - Right join: Must build on left (probe side is right)
    - Outer join: No optimization (need both sides)

    Grace Hash Join (Spill-Enabled)
    -------------------------------
    When spilling is enabled and data exceeds memory budget, this operator
    uses Grace hash join:

    1. Partition both sides by hash of join key(s) into N partitions
    2. Spill partitions to disk
    3. For each partition pair (i, i): load into memory and perform regular hash join
    4. Concatenate results from all partition joins

    This allows joining datasets larger than available memory.
    """

    left: PhysicalPlan
    right: PhysicalPlan
    on: tuple[str, ...] | None
    left_on: tuple[str, ...] | None
    right_on: tuple[str, ...] | None
    how: Literal["inner", "left", "right", "outer", "cross", "semi", "anti"]
    suffix: tuple[str, str]
    schema: Schema

    # Estimated row counts for build/probe optimization (set by planner)
    left_rows_estimate: int | None = None
    right_rows_estimate: int | None = None
    # Set by the decision layer when this join feeds an order-insensitive
    # sink (groupby/sort/distinct) and key types are acero-safe: routes
    # to Arrow's internally-parallel hash join (~4x at 10M x 1M). Joins
    # whose row order is observable keep the indexer path, whose order
    # matches eager pd.merge by construction.
    planned_acero: bool = False

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas.lazy.backends import has_kernel
        from pandas.lazy.backends.convert import (
            get_array_backend,
        )
        from pandas.lazy.backends.types import is_index_col

        # Check if we should use Grace hash join (spill-enabled out-of-core join)
        if context.spill_enabled and self.how in ("inner", "left", "right"):
            # Grace hash join only supports equi-joins
            if self.on is not None or (
                self.left_on is not None and self.right_on is not None
            ):
                return self._execute_grace_hash_join(context)

        # Execute left and right sides in parallel
        left_arrays, right_arrays = self._execute_sides_parallel(context)

        # Acero fast path (planned): only when the index is not observed
        # downstream - index columns cannot survive an order-destroying
        # join meaningfully, and the default RangeIndex is regenerated
        # at output anyway.
        if (
            self.planned_acero
            and not context.preserve_index
            and not context.user_set_index
        ):
            return self._execute_acero_join(left_arrays, right_arrays)

        # Separate index columns - we'll include left index in join to preserve it
        left_index_cols = {k: v for k, v in left_arrays.items() if is_index_col(k)}
        left_data = {k: v for k, v in left_arrays.items() if not is_index_col(k)}
        right_data = {k: v for k, v in right_arrays.items() if not is_index_col(k)}

        # Include left index columns in left_data so they get the same row selection
        # This allows index preservation when preserve_index=True is used
        left_data_with_index = {**left_data, **left_index_cols}

        # Determine backend
        first_col = next(iter(left_data.values()))
        backend = get_array_backend(first_col)

        # Cross join requires special handling (no kernel, use DataFrame)
        if self.how == "cross":
            return self._execute_cross_join(left_data_with_index, right_data, context)

        # Semi and anti joins
        if self.how in ("semi", "anti"):
            return self._execute_semi_anti_join(
                left_data_with_index, right_data, backend
            )

        # For inner joins, apply build/probe optimization
        # Build hash table on smaller side for better performance
        # Note: For inner joins, we don't include index columns in the swap
        # because when sides are swapped, the "left" becomes the original right
        # We'll add left index columns back after getting the result
        swapped = False
        if self.how == "inner":
            orig_left_data = left_data
            left_data, right_data = self._maybe_swap_for_build_probe(
                left_data, right_data
            )
            swapped = left_data is not orig_left_data

        # Prepare data for join - include index for non-swapped cases
        if self.how == "inner" and swapped:
            # When swapped, we need to track which rows from original left are kept
            # Don't include left index cols in the join data - we'll add them back
            join_left_data = left_data
            join_right_data = {**right_data, **left_index_cols}
        else:
            # Normal case - include left index in left data
            join_left_data = left_data_with_index
            join_right_data = right_data

        # Try Arrow kernel for Arrow backend
        if backend == "arrow" and has_kernel("hash_join", "arrow"):
            result = self._execute_arrow_join(
                join_left_data, join_right_data, swapped=swapped
            )
            return self._reorder_columns(result)

        # Try NumPy kernel for NumPy backend
        if backend == "numpy" and has_kernel("hash_join", "numpy"):
            result = self._execute_numpy_join(
                join_left_data, join_right_data, swapped=swapped
            )
            return self._reorder_columns(result)

        # Fallback to DataFrame-based join
        return self._execute_dataframe_join(left_arrays, right_arrays, context)

    def _reorder_columns(self, result: ArrayDict) -> ArrayDict:
        """Reorder result columns to match schema order."""
        schema_cols = self.schema.names
        # Filter to only columns that exist in result
        ordered = {}
        for col in schema_cols:
            if col in result:
                ordered[col] = result[col]
        # Add any extra columns not in schema (shouldn't happen, but be safe)
        for col in result:
            if col not in ordered:
                ordered[col] = result[col]
        return ordered

    def _maybe_swap_for_build_probe(
        self,
        left_data: ArrayDict,
        right_data: ArrayDict,
    ) -> tuple[ArrayDict, ArrayDict]:
        """
        Swap left and right if right is smaller (build/probe optimization).

        For inner joins, it's more efficient to build the hash table on
        the smaller side and probe with the larger side.

        Returns
        -------
        tuple[ArrayDict, ArrayDict]
            (left_data, right_data) potentially swapped
        """
        # Get row counts
        left_rows = len(next(iter(left_data.values()))) if left_data else 0
        right_rows = len(next(iter(right_data.values()))) if right_data else 0

        # Use estimates if available (more accurate for pre-filtered data)
        if self.left_rows_estimate is not None:
            left_rows = self.left_rows_estimate
        if self.right_rows_estimate is not None:
            right_rows = self.right_rows_estimate

        # Swap if right is smaller (for inner join, order doesn't matter)
        # This makes right the build side (smaller = better for hash table)
        if right_rows < left_rows:
            # No swap needed - right is already smaller
            return left_data, right_data
        # Swap so build side (right) is smaller
        # For inner joins with same keys, this is semantically equivalent
        # but we need to handle left_on/right_on swap if different
        elif self.on is not None or (self.left_on == self.right_on):
            # Same keys - can swap freely
            return right_data, left_data
        else:
            # Different keys - don't swap to preserve semantics
            return left_data, right_data

    def _execute_semi_anti_join(
        self,
        left_data: ArrayDict,
        right_data: ArrayDict,
        backend: str,
    ) -> ArrayDict:
        """Execute semi or anti join."""
        import numpy as np

        from pandas.lazy.backends import (
            dispatch_kernel,
            has_kernel,
        )

        kernel_name = "semi_join" if self.how == "semi" else "anti_join"

        # Determine keys
        if self.on is not None:
            keys = list(self.on)
            left_keys = None
            right_keys = None
        else:
            keys = None
            left_keys = list(self.left_on) if self.left_on else None
            right_keys = list(self.right_on) if self.right_on else None

        # Try Arrow kernel
        if backend == "arrow" and has_kernel(kernel_name, "arrow"):
            import pyarrow as pa

            left_table = pa.table(left_data)
            right_table = pa.table(right_data)

            result_table = dispatch_kernel(
                kernel_name,
                "arrow",
                left_table,
                right_table,
                keys=keys,
                left_keys=left_keys,
                right_keys=right_keys,
            )

            return {col: result_table.column(col) for col in result_table.column_names}

        # Try NumPy kernel
        if has_kernel(kernel_name, "numpy"):
            left_arrays = {k: np.asarray(v) for k, v in left_data.items()}
            right_arrays = {k: np.asarray(v) for k, v in right_data.items()}

            return dispatch_kernel(
                kernel_name,
                "numpy",
                left_arrays,
                right_arrays,
                keys=keys,
                left_keys=left_keys,
                right_keys=right_keys,
            )

        # Fallback: simulate with inner join and filter
        raise NotImplementedError(f"{kernel_name} not available for backend {backend}")

    def _execute_acero_join(
        self,
        left_arrays: ArrayDict,
        right_arrays: ArrayDict,
    ) -> ArrayDict:
        """Order-free join through Arrow's internally-parallel hash join.

        Used only when the decision layer planned it (consumer sink is
        order-insensitive, key types acero-safe) and the runtime index
        guards passed. Index columns are dropped: row order does not
        survive, and the caller's sink does not observe it.
        """
        import pyarrow as pa

        from pandas.lazy.backends.convert import to_arrow
        from pandas.lazy.backends.types import is_index_col

        def to_table(arrays: ArrayDict) -> pa.Table:
            cols = {}
            for name, arr in arrays.items():
                if is_index_col(name):
                    continue
                if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                    cols[name] = arr
                else:
                    cols[name] = to_arrow(arr)
            return pa.table(cols)

        left_table = to_table(left_arrays)
        right_table = to_table(right_arrays)

        if self.on is not None:
            keys = list(self.on)
            right_keys = keys
        else:
            keys = list(self.left_on)
            right_keys = list(self.right_on)

        join_type = "inner" if self.how == "inner" else "left outer"
        result = left_table.join(
            right_table,
            keys=keys,
            right_keys=right_keys,
            join_type=join_type,
            left_suffix=self.suffix[0],
            right_suffix=self.suffix[1],
            use_threads=True,
        )
        out: ArrayDict = {name: result.column(name) for name in result.column_names}
        return self._reorder_columns(out)

    def _execute_arrow_join(
        self,
        left_data: ArrayDict,
        right_data: ArrayDict,
        swapped: bool = False,
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

        # When sides are swapped for build/probe optimization, we need to swap
        # the suffixes too so that the original left side still gets _x suffix
        # and original right side still gets _y suffix
        if swapped:
            left_suffix = self.suffix[1]  # Original right -> now left, use _y
            right_suffix = self.suffix[0]  # Original left -> now right, use _x
        else:
            left_suffix = self.suffix[0]
            right_suffix = self.suffix[1]

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
            left_suffix=left_suffix,
            right_suffix=right_suffix,
        )

        # Convert result table to ArrayDict
        result: ArrayDict = {}
        for col_name in result_table.column_names:
            result[col_name] = result_table.column(col_name)

        return result

    def _execute_numpy_join(
        self,
        left_data: ArrayDict,
        right_data: ArrayDict,
        swapped: bool = False,
    ) -> ArrayDict:
        """Execute join using the indexer-based hash join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        # Pass arrays through in their native backend: the kernel computes
        # indexers from key columns only and gathers payload columns with
        # backend-preserving takes. (A previous np.asarray here converted
        # Arrow string columns to object arrays on input - a full copy of
        # every pass-through column.)
        left_arrays = dict(left_data)
        right_arrays = dict(right_data)

        # Determine keys
        if self.on is not None:
            keys = list(self.on)
            left_keys = None
            right_keys = None
        else:
            keys = None
            left_keys = list(self.left_on) if self.left_on else None
            right_keys = list(self.right_on) if self.right_on else None

        # When sides are swapped for build/probe optimization, we need to swap
        # the suffixes too so that the original left side still gets _x suffix
        # and original right side still gets _y suffix
        if swapped:
            left_suffix = self.suffix[1]  # Original right -> now left, use _y
            right_suffix = self.suffix[0]  # Original left -> now right, use _x
        else:
            left_suffix = self.suffix[0]
            right_suffix = self.suffix[1]

        # Execute join
        result = dispatch_kernel(
            "hash_join",
            "numpy",
            left_arrays,
            right_arrays,
            keys=keys,
            left_keys=left_keys,
            right_keys=right_keys,
            join_type=self.how,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
        )

        return result

    def _execute_grace_hash_join(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute join using Grace hash join for larger-than-memory data.

        Grace hash join partitions both sides by hash of join keys, spills
        partitions to disk, then joins matching partition pairs in memory.

        Runtime Adaptation
        ------------------
        After partitioning, if the join detects pathological behavior
        (skewed partitions, max partition exceeds budget), it falls back
        to sort-merge join which has more predictable I/O patterns.
        """
        from pandas.lazy.backends.spill import GraceHashJoiner
        from pandas.lazy.backends.types import is_index_col

        spill_manager = context.spill_manager
        if spill_manager is None:
            # Fallback to regular in-memory join
            left_arrays, right_arrays = self._execute_sides_parallel(context)
            return self._execute_dataframe_join(left_arrays, right_arrays, context)

        # Execute both sides to get full data
        # For truly out-of-core, we'd want streaming here too
        left_arrays, right_arrays = self._execute_sides_parallel(context)

        # Separate index columns
        left_index_cols = {k: v for k, v in left_arrays.items() if is_index_col(k)}
        left_data = {k: v for k, v in left_arrays.items() if not is_index_col(k)}
        right_data = {k: v for k, v in right_arrays.items() if not is_index_col(k)}

        # Determine join keys
        if self.on is not None:
            left_keys = list(self.on)
            right_keys = list(self.on)
        else:
            left_keys = list(self.left_on) if self.left_on else []
            right_keys = list(self.right_on) if self.right_on else []

        # Determine number of partitions based on data size and memory budget
        spill_config = context._spill_config
        operator_budget = spill_config.operator_budget_mb * 1024 * 1024

        # Estimate size of left + right data
        from pandas.lazy.backends.spill import get_arrays_bytes

        left_size = get_arrays_bytes(left_data)
        right_size = get_arrays_bytes(right_data)
        total_size = left_size + right_size

        # Calculate number of partitions needed
        # Each partition pair should fit in operator budget
        num_partitions = max(4, int(total_size / operator_budget) + 1)
        # Cap at reasonable number to avoid too many small files
        num_partitions = min(num_partitions, 256)

        # Create Grace hash joiner
        joiner = GraceHashJoiner(
            spill_manager=spill_manager,
            num_partitions=num_partitions,
            name="join",
        )

        # Partition both sides
        joiner.partition_left(left_data, left_keys)
        joiner.partition_right(right_data, right_keys)

        # Check for pathological behavior after partitioning
        # If detected, fall back to sort-merge join
        if joiner.is_pathological(operator_budget):
            import warnings

            stats = joiner.get_stats()
            warnings.warn(
                f"Hash join detected pathological spill behavior "
                f"(skew_ratio={stats['skew_ratio']:.1f}, "
                f"empty_partitions={stats['empty_partitions']}/{num_partitions * 2}). "
                f"Falling back to sort-merge join.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._execute_sort_merge_fallback(
                left_data, right_data, left_keys, right_keys, context
            )

        # Perform join across all partition pairs
        result = joiner.join(how=self.how)

        # Add back left index columns if present
        # Note: index ordering may not be preserved in partitioned join
        if left_index_cols:
            # For now, we don't preserve index in Grace hash join
            # This could be added by including index cols in partition data
            pass

        return self._reorder_columns(result)

    def _execute_sort_merge_fallback(
        self,
        left_data: ArrayDict,
        right_data: ArrayDict,
        left_keys: list[str],
        right_keys: list[str],
        context: ExecutionContext,
    ) -> ArrayDict:
        """
        Fall back to sort-merge join when hash join becomes pathological.

        Sort-merge join has more predictable I/O patterns and handles
        skewed data better than hash join with many partitions.
        """
        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.convert import get_array_backend

        # Sort both sides by join keys
        def sort_by_keys(arrays: ArrayDict, keys: list[str]) -> ArrayDict:
            if not keys:
                return arrays
            key_arr = arrays[keys[0]]
            backend = get_array_backend(key_arr)
            sort_indices = dispatch_kernel("argsort", backend, key_arr)
            result: ArrayDict = {}
            for name, arr in arrays.items():
                arr_backend = get_array_backend(arr)
                result[name] = dispatch_kernel("take", arr_backend, arr, sort_indices)
            return result

        left_sorted = sort_by_keys(left_data, left_keys)
        right_sorted = sort_by_keys(right_data, right_keys)

        # Use PhysicalSortMergeJoin's merge logic
        # Create a temporary instance to reuse the merge code
        temp_join = PhysicalSortMergeJoin(
            left=self.left,  # Not used, just for schema
            right=self.right,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            how=self.how,
            suffix=self.suffix,
            schema=self.schema,
        )

        return temp_join._merge_sorted(
            left_sorted, right_sorted, left_keys, right_keys, context
        )

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

    def _execute_sides_parallel(
        self, context: ExecutionContext
    ) -> tuple[ArrayDict, ArrayDict]:
        """
        Execute left and right sides of join in parallel.

        For large plans, this can provide significant speedup by overlapping
        the execution of independent subplans.

        Note: Each side gets its own ExecutionContext to avoid race conditions
        when modifying index metadata. The left side's index metadata is
        preserved in the main context for index preservation during joins.
        """
        from concurrent.futures import ThreadPoolExecutor

        # Create separate contexts for each side to avoid race conditions,
        # carrying over the parent's configuration (backend, thresholds,
        # preserve_index, spill settings)
        left_context = context.clone_for_subplan()
        right_context = context.clone_for_subplan()

        def execute_left():
            return self.left.execute(left_context)

        def execute_right():
            return self.right.execute(right_context)

        # Use ThreadPoolExecutor with 2 workers for left and right
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(execute_left)
            right_future = executor.submit(execute_right)

            left_arrays = left_future.result()
            right_arrays = right_future.result()

        # Preserve left side's index metadata in main context for index preservation
        # This allows joins to preserve the left table's index when preserve_index=True
        context.index_names = left_context.index_names
        context.index_is_multi = left_context.index_is_multi

        return left_arrays, right_arrays

    def children(self) -> list[PhysicalPlan]:
        return [self.left, self.right]

    @property
    def output_schema(self) -> Schema:
        return self.schema

    @property
    def is_pipeline_breaker(self) -> bool:
        """HashJoin requires build side fully materialized before probe."""
        return True


@dataclass
class PhysicalSortMergeJoin(PhysicalPlan):
    """
    Sort-merge join implementation.

    This join algorithm:
    1. Sorts both sides by join keys
    2. Merges the sorted streams using a merge cursor

    Use Cases
    ---------
    Sort-merge join is preferred when:
    - Data is already sorted on join keys (free sort)
    - Hash join spilling becomes pathological (many recursive spills)
    - Join keys have high cardinality with few duplicates
    - Memory is constrained and predictable I/O is preferred

    The PhysicalHashJoin can fall back to this implementation at runtime
    when it detects pathological spill behavior.

    Parameters
    ----------
    left : PhysicalPlan
        Left input (will be sorted if not already).
    right : PhysicalPlan
        Right input (will be sorted if not already).
    on : tuple[str, ...] or None
        Column names to join on (same name in both sides).
    left_on : tuple[str, ...] or None
        Column names from left side for join.
    right_on : tuple[str, ...] or None
        Column names from right side for join.
    how : str
        Join type: "inner", "left", "right", "outer".
    suffix : tuple[str, str]
        Suffixes for overlapping column names.
    schema : Schema
        Output schema.
    """

    left: PhysicalPlan
    right: PhysicalPlan
    on: tuple[str, ...] | None
    left_on: tuple[str, ...] | None
    right_on: tuple[str, ...] | None
    how: Literal["inner", "left", "right", "outer"]
    suffix: tuple[str, str]
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """Execute sort-merge join."""
        # Execute both sides
        left_arrays = self.left.execute(context)
        right_arrays = self.right.execute(context)

        # Determine join keys
        if self.on is not None:
            left_keys = list(self.on)
            right_keys = list(self.on)
        else:
            left_keys = list(self.left_on) if self.left_on else []
            right_keys = list(self.right_on) if self.right_on else []

        if not left_keys or not right_keys:
            raise ValueError("Sort-merge join requires join keys")

        # Sort both sides by join keys
        left_sorted = self._sort_by_keys(left_arrays, left_keys, context)
        right_sorted = self._sort_by_keys(right_arrays, right_keys, context)

        # Perform merge join
        result = self._merge_sorted(
            left_sorted, right_sorted, left_keys, right_keys, context
        )

        return result

    def _sort_by_keys(
        self,
        arrays: ArrayDict,
        keys: list[str],
        context: ExecutionContext,
    ) -> ArrayDict:
        """Sort arrays by the specified keys."""
        from pandas.lazy.backends import dispatch_kernel
        from pandas.lazy.backends.convert import get_array_backend

        if not keys:
            return arrays

        # Get sort key array
        key_arr = arrays[keys[0]]
        backend = get_array_backend(key_arr)

        # Get sort indices
        sort_indices = dispatch_kernel("argsort", backend, key_arr)

        # Apply sort indices to all arrays
        result: ArrayDict = {}
        for name, arr in arrays.items():
            arr_backend = get_array_backend(arr)
            result[name] = dispatch_kernel("take", arr_backend, arr, sort_indices)

        return result

    def _merge_sorted(
        self,
        left: ArrayDict,
        right: ArrayDict,
        left_keys: list[str],
        right_keys: list[str],
        context: ExecutionContext,
    ) -> ArrayDict:
        """Merge two sorted arrays on join keys."""
        import numpy as np

        from pandas.lazy.backends.types import is_index_col

        # Get key arrays (use first key for now)
        left_key = left[left_keys[0]]
        right_key = right[right_keys[0]]

        # Convert to numpy for merge logic
        if hasattr(left_key, "to_numpy"):
            left_key_np = left_key.to_numpy()
        else:
            left_key_np = np.asarray(left_key)

        if hasattr(right_key, "to_numpy"):
            right_key_np = right_key.to_numpy()
        else:
            right_key_np = np.asarray(right_key)

        # Perform merge to get matching indices
        left_indices, right_indices = self._merge_indices(
            left_key_np, right_key_np, self.how
        )

        # Build result arrays
        result: ArrayDict = {}

        # Add left columns
        for name, arr in left.items():
            if is_index_col(name):
                continue
            # Handle null indices (-1) for outer joins
            if hasattr(arr, "to_numpy"):
                arr_np = arr.to_numpy()
            else:
                arr_np = np.asarray(arr)

            # Create result array
            out_name = name
            if name in right and name not in left_keys:
                out_name = name + self.suffix[0]

            result[out_name] = self._take_with_nulls(arr_np, left_indices)

        # Add right columns
        for name, arr in right.items():
            if is_index_col(name):
                continue
            if name in right_keys and name in left_keys:
                # Join key already added from left
                continue

            if hasattr(arr, "to_numpy"):
                arr_np = arr.to_numpy()
            else:
                arr_np = np.asarray(arr)

            out_name = name
            if name in left and name not in right_keys:
                out_name = name + self.suffix[1]

            result[out_name] = self._take_with_nulls(arr_np, right_indices)

        return result

    def _merge_indices(
        self,
        left_key: np.ndarray,
        right_key: np.ndarray,
        how: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute matching indices for sorted arrays using merge algorithm.

        Returns (left_indices, right_indices) where -1 indicates null (no match).
        """
        import numpy as np

        left_idx = []
        right_idx = []

        i, j = 0, 0
        n_left, n_right = len(left_key), len(right_key)

        while i < n_left and j < n_right:
            left_val = left_key[i]
            right_val = right_key[j]

            # Handle NaN comparison
            left_is_nan = left_val != left_val  # NaN check
            right_is_nan = right_val != right_val

            if left_is_nan and right_is_nan:
                # Both NaN - don't match (SQL semantics)
                if how in ("left", "outer"):
                    left_idx.append(i)
                    right_idx.append(-1)
                i += 1
                if how in ("right", "outer"):
                    left_idx.append(-1)
                    right_idx.append(j)
                j += 1
            elif left_is_nan or left_val < right_val:
                # Left value is smaller or NaN
                if how in ("left", "outer"):
                    left_idx.append(i)
                    right_idx.append(-1)
                i += 1
            elif right_is_nan or left_val > right_val:
                # Right value is smaller or NaN
                if how in ("right", "outer"):
                    left_idx.append(-1)
                    right_idx.append(j)
                j += 1
            else:
                # Equal values - find all matches
                # Count duplicates on both sides
                left_start = i
                while i < n_left and left_key[i] == left_val:
                    i += 1
                left_end = i

                right_start = j
                while j < n_right and right_key[j] == right_val:
                    j += 1
                right_end = j

                # Cross product of matching rows
                for li in range(left_start, left_end):
                    for ri in range(right_start, right_end):
                        left_idx.append(li)
                        right_idx.append(ri)

        # Handle remaining left rows
        while i < n_left:
            if how in ("left", "outer"):
                left_idx.append(i)
                right_idx.append(-1)
            i += 1

        # Handle remaining right rows
        while j < n_right:
            if how in ("right", "outer"):
                left_idx.append(-1)
                right_idx.append(j)
            j += 1

        return np.array(left_idx, dtype=np.intp), np.array(right_idx, dtype=np.intp)

    def _take_with_nulls(self, arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Take from array, handling -1 as null."""
        import numpy as np

        # Create output array
        if np.issubdtype(arr.dtype, np.floating):
            result = np.empty(len(indices), dtype=arr.dtype)
            result[:] = np.nan
        elif np.issubdtype(arr.dtype, np.integer):
            # Convert to float to support NaN
            result = np.empty(len(indices), dtype=np.float64)
            result[:] = np.nan
        else:
            # Object dtype for strings etc
            result = np.empty(len(indices), dtype=object)
            result[:] = None

        # Fill valid indices
        valid = indices >= 0
        result[valid] = arr[indices[valid]]

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.left, self.right]

    @property
    def output_schema(self) -> Schema:
        return self.schema

    @property
    def is_pipeline_breaker(self) -> bool:
        """Sort-merge join requires both sides sorted (pipeline breaker)."""
        return True


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


@dataclass
class PhysicalSetIndex(PhysicalPlan):
    """
    Physical set_index operation.

    Marks columns to be used as the index. The actual index setting
    happens during arrays_to_dataframe conversion when preserve_index=True.
    """

    input: PhysicalPlan
    keys: tuple[str, ...]
    drop: bool
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas.lazy.backends.types import (
            INDEX_COL_NAME,
            index_col_name,
        )

        input_arrays = self.input.execute(context)
        result: ArrayDict = {}

        # Mark that user explicitly set an index
        # This ensures the index is reconstructed even if preserve_index=False
        context.user_set_index = True

        # Copy all non-key columns to result
        for name, arr in input_arrays.items():
            if name in self.keys and self.drop:
                continue  # Skip key columns when drop=True
            result[name] = arr

        # Store key columns as index columns
        if len(self.keys) == 1:
            # Single index
            key_name = self.keys[0]
            if key_name in input_arrays:
                result[INDEX_COL_NAME] = input_arrays[key_name]
                context.index_names = [key_name]
                context.index_is_multi = False
        else:
            # MultiIndex
            context.index_names = list(self.keys)
            context.index_is_multi = True
            for i, key_name in enumerate(self.keys):
                if key_name in input_arrays:
                    result[index_col_name(i)] = input_arrays[key_name]

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


@dataclass
class PhysicalResetIndex(PhysicalPlan):
    """
    Physical reset_index operation.

    Converts index columns back to regular columns and sets a fresh RangeIndex.
    """

    input: PhysicalPlan
    drop: bool
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from pandas.lazy.backends.types import (
            INDEX_COL_NAME,
            index_col_name,
            is_index_col,
        )

        input_arrays = self.input.execute(context)
        result: ArrayDict = {}

        # Get current index info
        index_names = context.index_names or [None]
        index_is_multi = context.index_is_multi

        if not self.drop:
            # Add index columns as regular data columns
            if index_is_multi:
                for i, idx_name in enumerate(index_names):
                    idx_col = index_col_name(i)
                    if idx_col in input_arrays:
                        col_name = idx_name if idx_name else f"level_{i}"
                        result[col_name] = input_arrays[idx_col]
            elif INDEX_COL_NAME in input_arrays:
                idx_name = index_names[0] if index_names else None
                col_name = idx_name if idx_name else "index"
                result[col_name] = input_arrays[INDEX_COL_NAME]

        # Copy all non-index columns
        for name, arr in input_arrays.items():
            if not is_index_col(name):
                result[name] = arr

        # Reset context index info (will use RangeIndex)
        context.index_names = [None]
        context.index_is_multi = False

        return result

    def children(self) -> list[PhysicalPlan]:
        return [self.input]

    @property
    def output_schema(self) -> Schema:
        return self.schema


@dataclass
class PhysicalConcat(PhysicalPlan):
    """
    Physical concatenation of multiple inputs.

    Supports streaming execution - yields batches from each input
    sequentially. This allows processing multiple sources without
    materializing all data at once.

    Parameters
    ----------
    inputs : tuple[PhysicalPlan, ...]
        Physical plans to concatenate.
    schema : Schema
        Output schema (same as each input's schema).
    """

    inputs: tuple[PhysicalPlan, ...]
    schema: Schema

    @property
    def supports_streaming(self) -> bool:
        # Concat supports streaming if all inputs support streaming
        return all(inp.supports_streaming for inp in self.inputs)

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Stream batches from all inputs sequentially.

        This is the preferred execution path as it doesn't require
        materializing all data at once.
        """
        for input_plan in self.inputs:
            if input_plan.supports_streaming:
                yield from input_plan.execute_batches(context)
            else:
                # For non-streaming inputs, wrap in single batch
                yield input_plan.execute(context)

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute concatenation by collecting all batches.

        Inputs are independent subtrees (typically per-file scans), so the
        materializing path executes them in parallel; ordering of the
        concatenated output is preserved by collecting results per input.

        For better memory efficiency, prefer using execute_batches()
        in downstream operators.
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        import numpy as np
        import pyarrow as pa

        if len(self.inputs) > 1:
            # Each input gets an independent context: scans/set_index mutate
            # index metadata, which must not race across concurrent inputs.
            # The first input's metadata wins, matching Concat.resolve_schema.
            input_contexts = [context.clone_for_subplan() for _ in self.inputs]
            n_workers = min(8, os.cpu_count() or 4, len(self.inputs))
            with ThreadPoolExecutor(n_workers) as ex:
                input_results = list(
                    ex.map(
                        lambda pair: pair[0].execute(pair[1]),
                        zip(self.inputs, input_contexts, strict=True),
                    )
                )
            context.index_names = input_contexts[0].index_names
            context.index_is_multi = input_contexts[0].index_is_multi
            context.user_set_index = input_contexts[0].user_set_index
        else:
            input_results = [plan.execute(context) for plan in self.inputs]

        all_arrays: dict[str, list] = {}
        for batch in input_results:
            for name, arr in batch.items():
                if name not in all_arrays:
                    all_arrays[name] = []
                all_arrays[name].append(arr)

        # Concatenate arrays
        result: ArrayDict = {}
        for name, arrays in all_arrays.items():
            if len(arrays) == 0:
                continue
            if len(arrays) == 1:
                result[name] = arrays[0]
            elif isinstance(arrays[0], (pa.Array, pa.ChunkedArray)):
                # Concatenate Arrow arrays/chunked arrays without copying:
                # flatten everything into one ChunkedArray
                chunks: list[pa.Array] = []
                for arr in arrays:
                    if isinstance(arr, pa.ChunkedArray):
                        chunks.extend(arr.chunks)
                    else:
                        chunks.append(arr)
                result[name] = pa.chunked_array(chunks)
            else:
                # Concatenate NumPy arrays
                result[name] = np.concatenate(arrays)

        return result

    def children(self) -> list[PhysicalPlan]:
        return list(self.inputs)

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

    Pipeline Boundaries
    -------------------
    The planner inserts explicit PhysicalMaterialize nodes before pipeline
    breaker operators (sort, aggregate, distinct, join build side). This
    makes materialization points visible in the physical plan for:

    - Clear debugging and explain output
    - Centralized spill management
    - Correct fusion boundaries
    - Backend conversion points
    """

    def __init__(
        self,
        preferred_backend: Literal["auto", "arrow", "numpy"] = "auto",
    ) -> None:
        self.preferred_backend = preferred_backend

    def _materialize_for_breaker(
        self,
        logical_input: LogicalPlan,
        reason: str,
    ) -> PhysicalPlan:
        """
        Plan an input and wrap in Materialize node for a pipeline breaker.

        This ensures inputs to pipeline breakers (sort, aggregate, distinct,
        join) are explicitly materialized, making the boundary visible in
        the plan.

        Parameters
        ----------
        logical_input : LogicalPlan
            The logical input to plan and materialize.
        reason : str
            Why materialization is needed (for debugging/explain).

        Returns
        -------
        PhysicalPlan
            The input wrapped in PhysicalMaterialize.
        """
        physical_input = self._plan_recursive(logical_input)
        return PhysicalMaterialize(input=physical_input, reason=reason)

    def plan(
        self, logical_plan: LogicalPlan, *, enable_fusion: bool = True
    ) -> PhysicalPlan:
        """
        Convert a logical plan to a physical plan.

        Parameters
        ----------
        logical_plan : LogicalPlan
            The optimized logical plan.
        enable_fusion : bool, default True
            If True, apply operator fusion as a post-processing step.
            Fusion combines chains of Filter/Project/Limit into single
            fused operators for better performance.

        Returns
        -------
        PhysicalPlan
            The physical execution plan.
        """
        physical_plan = self._plan_recursive(logical_plan)

        if enable_fusion:
            physical_plan = self._apply_fusion(physical_plan)

        return physical_plan

    def _plan_recursive(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        """Recursively convert logical plan to physical plan."""
        from pandas.lazy.plan import (
            Aggregate,
            Concat,
            Convert,
            CSVSource,
            DataFrameSource,
            Distinct,
            Filter,
            Join,
            Limit,
            ParquetSource,
            Project,
            ResetIndex,
            SetIndex,
            Sort,
            TopK,
        )

        if isinstance(logical_plan, DataFrameSource):
            return self._plan_scan(logical_plan)

        elif isinstance(logical_plan, ParquetSource):
            return self._plan_parquet_scan(logical_plan)

        elif isinstance(logical_plan, CSVSource):
            return self._plan_csv_scan(logical_plan)

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

        elif isinstance(logical_plan, SetIndex):
            return self._plan_set_index(logical_plan)

        elif isinstance(logical_plan, ResetIndex):
            return self._plan_reset_index(logical_plan)

        elif isinstance(logical_plan, Concat):
            return self._plan_concat(logical_plan)

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

    def _plan_parquet_scan(self, node) -> PhysicalParquetScan:
        """Plan a ParquetSource."""
        return PhysicalParquetScan(
            path=node.path,
            schema=node.resolve_schema(),
            columns=node.columns,
            predicate=node.predicate,
            limit=node.limit,
        )

    def _plan_csv_scan(self, node) -> PhysicalPlan:
        """Plan a CSVSource.

        Unlike Parquet, CSV doesn't support native predicate pushdown.
        If there's a predicate, we wrap the scan with a PhysicalFilter.
        """
        scan = PhysicalCSVScan(
            path=node.path,
            schema=node.resolve_schema(),
            columns=node.columns,
            predicate=None,  # Predicate applied by filter, not scan
            sep=node.sep,
            header=node.header,
            skip_rows=node.skip_rows,
            n_rows=node.n_rows,
        )

        # If there's a predicate, add a filter step after scanning
        if node.predicate is not None:
            return PhysicalFilter(
                input=scan,
                predicate=node.predicate,
                schema=node.resolve_schema(),
                backend=self._choose_backend_for_exprs((node.predicate,)),
            )

        return scan

    def _plan_project(self, node) -> PhysicalProject:
        """Plan a Project."""
        return PhysicalProject(
            input=self._plan_recursive(node.input),
            exprs=node.exprs,
            schema=node.resolve_schema(),
            backend=self._choose_backend_for_exprs(node.exprs),
        )

    def _plan_filter(self, node) -> PhysicalFilter:
        """Plan a Filter."""
        return PhysicalFilter(
            input=self._plan_recursive(node.input),
            predicate=node.predicate,
            schema=node.resolve_schema(),
            backend=self._choose_backend_for_exprs((node.predicate,)),
        )

    def _plan_aggregate(self, node) -> PhysicalHashAggregate:
        """Plan an Aggregate."""
        # Aggregate is a pipeline breaker - needs all rows per group
        # Wrap input in Materialize to make boundary explicit
        return PhysicalHashAggregate(
            input=self._materialize_for_breaker(node.input, "aggregate"),
            group_by=node.group_by,
            agg_exprs=node.agg_exprs,
            schema=node.resolve_schema(),
        )

    def _plan_sort(self, node) -> PhysicalSort:
        """Plan a Sort."""
        # Sort is a pipeline breaker - needs all data for global ordering
        # Wrap input in Materialize to make boundary explicit
        return PhysicalSort(
            input=self._materialize_for_breaker(node.input, "sort"),
            by=node.by,
            descending=node.descending,
            schema=node.resolve_schema(),
            algorithm="quicksort",
        )

    def _plan_topk(self, node) -> PhysicalTopK:
        """Plan a TopK."""
        return PhysicalTopK(
            input=self._plan_recursive(node.input),
            k=node.k,
            by=node.by,
            descending=node.descending,
            schema=node.resolve_schema(),
        )

    def _plan_limit(self, node) -> PhysicalLimit:
        """Plan a Limit."""
        return PhysicalLimit(
            input=self._plan_recursive(node.input),
            n=node.n,
            offset=node.offset,
            schema=node.resolve_schema(),
        )

    def _plan_distinct(self, node) -> PhysicalDistinct:
        """Plan a Distinct."""
        # Distinct is a pipeline breaker - needs all values to deduplicate
        # Wrap input in Materialize to make boundary explicit
        return PhysicalDistinct(
            input=self._materialize_for_breaker(node.input, "distinct"),
            subset=node.subset,
            schema=node.resolve_schema(),
        )

    def _plan_join(self, node) -> PhysicalHashJoin:
        """Plan a Join."""
        # Get row count estimates for build/probe optimization
        left_rows = node.left.estimate_row_count()
        right_rows = node.right.estimate_row_count()

        # Hash join: build side must be fully materialized to build hash table
        # Currently we materialize both sides; future optimization could stream
        # the probe side through the join.
        #
        # Note: PhysicalHashJoin internally chooses which side is build vs probe
        # based on row estimates. Both sides need materialization for now.
        return PhysicalHashJoin(
            left=self._materialize_for_breaker(node.left, "hash_join_build"),
            right=self._materialize_for_breaker(node.right, "hash_join_build"),
            on=node.on,
            left_on=node.left_on,
            right_on=node.right_on,
            how=node.how,
            suffix=node.suffix,
            schema=node.resolve_schema(),
            left_rows_estimate=left_rows,
            right_rows_estimate=right_rows,
        )

    def _plan_convert(self, node) -> PhysicalConvert:
        """Plan a Convert (backend conversion)."""
        return PhysicalConvert(
            input=self._plan_recursive(node.input),
            target_backend=node.target_backend,
            schema=node.resolve_schema(),
        )

    def _plan_set_index(self, node) -> PhysicalSetIndex:
        """Plan a SetIndex."""
        return PhysicalSetIndex(
            input=self._plan_recursive(node.input),
            keys=node.keys,
            drop=node.drop,
            schema=node.resolve_schema(),
        )

    def _plan_reset_index(self, node) -> PhysicalResetIndex:
        """Plan a ResetIndex."""
        return PhysicalResetIndex(
            input=self._plan_recursive(node.input),
            drop=node.drop,
            schema=node.resolve_schema(),
        )

    def _plan_concat(self, node) -> PhysicalConcat:
        """Plan a Concat."""
        return PhysicalConcat(
            inputs=tuple(self._plan_recursive(inp) for inp in node.inputs),
            schema=node.resolve_schema(),
        )

    def _apply_fusion(self, plan: PhysicalPlan) -> PhysicalPlan:
        """
        Apply operator fusion to optimize the physical plan.

        Detects chains of fuseable operators (Filter, Project, Limit) and
        combines them into single PhysicalFusedPipeline operators.

        Fusion Rules
        ------------
        1. Filter → Project: Fuse to evaluate predicate before expressions
        2. Project → Project: Combine into single projection
        3. Filter → Filter: Combine predicates (AND)
        4. Filter → Limit: Short-circuit when limit is reached
        5. Project → Limit: Fuse to stop early
        6. Any combination of above

        Fusion Boundaries
        -----------------
        Fusion stops at:
        - Pipeline breakers (Sort, Aggregate, Join, Distinct)
        - Materialize nodes
        - Scan nodes (fusion starts fresh after)

        Parameters
        ----------
        plan : PhysicalPlan
            The physical plan to optimize.

        Returns
        -------
        PhysicalPlan
            Optimized plan with fused operators.
        """
        # First, recursively apply fusion to children
        plan = self._apply_fusion_to_children(plan)

        # Then check if this node can be fused with its input
        return self._try_fuse(plan)

    def _apply_fusion_to_children(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Recursively apply fusion to all children of a plan node."""
        from dataclasses import replace

        children = plan.children()
        if not children:
            return plan

        # Recursively optimize children
        new_children = [self._apply_fusion(child) for child in children]

        # If no changes, return original
        if all(new is old for new, old in zip(new_children, children, strict=True)):
            return plan

        # Create new node with optimized children
        if hasattr(plan, "input") and len(new_children) == 1:
            return replace(plan, input=new_children[0])
        elif hasattr(plan, "left") and hasattr(plan, "right"):
            if len(new_children) == 2:
                return replace(plan, left=new_children[0], right=new_children[1])
        elif hasattr(plan, "inputs"):
            return replace(plan, inputs=tuple(new_children))
        else:
            # Unknown structure, return as-is
            return plan

    def _try_fuse(self, plan: PhysicalPlan) -> PhysicalPlan:
        """
        Try to fuse this plan node with its input(s) into a FusedPipeline.

        Only Filter, Project, and Limit can be fused.
        """
        # Check if this is a fuseable operator
        if not isinstance(plan, (PhysicalFilter, PhysicalProject, PhysicalLimit)):
            return plan

        # Don't fuse tail operations (offset=-1)
        if isinstance(plan, PhysicalLimit) and plan.offset == -1:
            return plan

        # Collect the chain of fuseable operations
        operations: list[FusedOperation] = []
        current = plan
        base_input = None

        while True:
            if isinstance(current, PhysicalFilter):
                operations.append(
                    FusedOperation(op_type="filter", predicate=current.predicate)
                )
                current = current.input

            elif isinstance(current, PhysicalProject):
                operations.append(
                    FusedOperation(op_type="project", exprs=current.exprs)
                )
                current = current.input

            elif isinstance(current, PhysicalLimit) and current.offset != -1:
                # Only fuse head() operations, not tail()
                operations.append(FusedOperation(op_type="limit", limit_n=current.n))
                current = current.input

            elif isinstance(current, PhysicalFusedPipeline):
                # Already fused - absorb its operations. They are stored
                # in execution (bottom-up) order, but this collection loop
                # builds a top-down list that is reversed at the end, so
                # they must be added reversed to survive that reversal.
                # (Getting this wrong made later projections drop columns
                # that earlier fused filters still referenced.)
                operations.extend(reversed(current.operations))
                current = current.input

            else:
                # Not fuseable - this is the base input
                base_input = current
                break

        # Reverse to get operations in execution order (bottom-up to top-down)
        operations.reverse()

        # If only one operation and no fusion benefit, return original
        if len(operations) == 1:
            return plan

        # Check if fusion is beneficial
        # At minimum we need: Filter+Project, Filter+Limit, or Project+Limit
        has_filter = any(op.op_type == "filter" for op in operations)
        has_project = any(op.op_type == "project" for op in operations)
        has_limit = any(op.op_type == "limit" for op in operations)

        # Fusion is beneficial if we have at least two different types
        # or multiple of the same type that can be combined
        beneficial = (
            (has_filter and has_project)
            or (has_filter and has_limit)
            or (has_project and has_limit)
            or sum(1 for op in operations if op.op_type == "filter") > 1
            or sum(1 for op in operations if op.op_type == "project") > 1
        )

        if not beneficial:
            return plan

        return PhysicalFusedPipeline(
            input=base_input,
            operations=tuple(operations),
            schema=plan.output_schema,
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
    preserve_index: bool = False,
    order_relaxed: bool = False,
    spill_config: SpillConfig | None = None,
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
    preserve_index : bool, default False
        If True, preserve the original DataFrame index.
    order_relaxed : bool, default False
        If True, the final output row order is unspecified, which lets
        order-preserving joins route to acero's parallel hash join.
    spill_config : SpillConfig or None, default None
        Configuration for disk spilling under memory pressure.
        When enabled, intermediate results can be spilled to disk.

    Returns
    -------
    DataFrame
        The execution result.
    """
    import pandas as pd
    from pandas.lazy.backends.convert import arrays_to_dataframe

    # Check if adaptive thresholds are enabled
    adaptive_enabled = pd.get_option("compute.lazy.adaptive_thresholds")

    context = ExecutionContext(
        preferred_backend=preferred_backend,
        strict=strict,
        adaptive_thresholds=adaptive_enabled,
        preserve_index=preserve_index,
        order_relaxed=order_relaxed,
        _spill_config=spill_config,
    )

    # Execute through the pipeline engine (docs/ENGINE_DESIGN.md, M1):
    # the plan compiles to an explicit pipeline graph; every node still
    # runs its own execute(), so behavior is identical to the previous
    # direct recursion.
    from pandas.lazy.engine import execute_as_pipelines

    arrays = execute_as_pipelines(plan, context)

    # Convert ArrayDict back to DataFrame with proper index
    # Reconstruct index if preserve_index=True OR user explicitly called set_index()
    should_reconstruct_index = preserve_index or context.user_set_index
    return arrays_to_dataframe(
        arrays,
        index_names=context.index_names,
        index_is_multi=context.index_is_multi,
        preserve_index=should_reconstruct_index,
        schema=plan.output_schema,
    )
