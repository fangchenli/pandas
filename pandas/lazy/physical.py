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
import dataclasses
import os
import threading
from types import SimpleNamespace

# Execute-once caching of shared subplans (see PhysicalPlanner.plan). The
# q15 regression that kept this off is diagnosed and fixed: the childless
# wrapper hid its inner subtree from the fusion/chain post-passes.
_SUBPLAN_CACHE_ENABLED = True
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

    from pandas import DataFrame
    from pandas.lazy.types import Schema


# =============================================================================
# Helper Functions
# =============================================================================


import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pandas.lazy.backends import (
    dispatch_kernel,
    has_kernel,
)
from pandas.lazy.backends.array_eval import ArrayEvaluator
from pandas.lazy.backends.convert import (
    arrays_to_dataframe,
    dataframe_to_arrays,
    ensure_backend,
    extract_array,
    get_array_backend,
    to_arrow,
)
from pandas.lazy.backends.memory_pool import PoolingStrategy
from pandas.lazy.backends.numpy.core import radix_lexsort
from pandas.lazy.backends.spill import (
    ExternalSorter,
    GraceHashJoiner,
    SpillConfig,
    SpillManager,
    get_arrays_bytes,
)
from pandas.lazy.backends.types import (
    INDEX_COL_NAME,
    ArrayDict,
    index_col_name,
    is_index_col,
)
from pandas.lazy.cost import (
    ARROW_MULTIKEY_SORT_MIN_ROWS,
    PARALLEL_TAKE_MIN_ROWS,
)
from pandas.lazy.expr import (
    Expr,
    extract_output_name,
)
from pandas.lazy.ir import (
    Alias,
    Call,
    FieldRef,
)
from pandas.lazy.ir import (
    Literal as IRLiteral,  # distinct from typing.Literal used in annotations
)
from pandas.lazy.plan import (
    Aggregate,
    Concat,
    Convert,
    CSVSource,
    DataFrameSource,
    Distinct,
    Filter,
    GroupByHead,
    Join,
    Limit,
    LogicalPlan,
    ParquetSource,
    Project,
    ResetIndex,
    SetIndex,
    Sort,
    TopK,
)


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

    # Results of common (shared) subplans, computed once per collect and
    # reused by every consumer (see PhysicalCachedSubplan). Unlike the CSE
    # cache this is deliberately SHARED across clone_for_subplan contexts -
    # concurrent feeders of different breakers may request the same shared
    # subplan; the lock serializes its first computation.
    subplan_cache: dict[int, Any] = field(default_factory=dict)
    subplan_lock: Any = field(default_factory=threading.Lock)

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
            subplan_cache=self.subplan_cache,
            subplan_lock=self.subplan_lock,
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
                # Arrow concat as a ChunkedArray: concat_arrays/combine_chunks
                # on string columns overflows int32 offsets past 2 GB (TPC-H
                # q13's o_comment at SF-100 — "offset overflow while
                # concatenating arrays"). Chunked output avoids the copy AND
                # the overflow; downstream kernels accept ChunkedArray.
                chunks: list = []
                for arr in arrays_to_concat:
                    if isinstance(arr, pa.ChunkedArray):
                        chunks.extend(arr.chunks)
                    else:
                        chunks.append(arr)
                result[col] = pa.chunked_array(chunks)
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

        # Pushdown is BEST-EFFORT: _build_arrow_filters returns None for
        # predicates Arrow dataset expressions can't represent (e.g. regex
        # contains). The Filter node was already removed from the plan by
        # the optimizer, so when conversion fails the predicate MUST be
        # applied here per batch — silently dropping it returned every row
        # (TPC-H q16/q13 through scans gave wrong results).
        fallback_predicate = (
            self.predicate._ir
            if self.predicate is not None and filter_expr is None
            else None
        )

        rows_yielded = 0
        row_offset = 0  # Track row offset for index generation

        for batch in scanner.to_batches():
            batch_len = batch.num_rows
            if batch_len == 0:
                continue

            if fallback_predicate is not None:
                from pandas.lazy.backends.array_eval import ArrayEvaluator

                cols0 = {n: batch.column(n) for n in batch.schema.names}
                mask = ArrayEvaluator(cols0, preferred_backend="auto").evaluate(
                    fallback_predicate
                )
                if isinstance(mask, (pa.Array, pa.ChunkedArray)):
                    mask = mask.to_numpy(zero_copy_only=False)
                mask = np.asarray(mask, dtype=bool)
                batch = batch.filter(pa.array(mask))
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

        if isinstance(ir, FieldRef):
            return pc.field(ir.name)

        if isinstance(ir, IRLiteral):
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
                if isinstance(prefix_arg, IRLiteral):
                    prefix = prefix_arg.value
                else:
                    prefix = prefix_arg
                if col is not None and isinstance(prefix, str):
                    return pc.starts_with(col, prefix)

            elif ir.function == "str_endswith" and len(ir.args) >= 1:
                col = self._ir_to_arrow_expr(ir.args[0])
                suffix_arg = ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("suffix")
                # Extract value if it's a Literal
                if isinstance(suffix_arg, IRLiteral):
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
                if isinstance(pattern_arg, IRLiteral):
                    pattern = pattern_arg.value
                else:
                    pattern = pattern_arg
                if col is not None and isinstance(pattern, str):
                    # Expr.str.contains defaults to REGEX semantics (pandas
                    # str.contains), but pc.match_substring is LITERAL — a
                    # regex pattern pushed as a literal silently filters
                    # everything out (TPC-H q16's 'Customer.*Complaints'
                    # matched 0 rows through a scan). Push down only when
                    # the match is literal-safe; otherwise leave the filter
                    # in-engine where the regex kernel evaluates it.
                    regex = ir.kwargs.get("regex", True)
                    if not regex or not (set(".^$*+?{}[]\\|()") & set(pattern)):
                        return pc.match_substring(col, pattern)
                    return None

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
        from pyarrow import csv

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


def _array_is_numeric(arr) -> bool:
    """True for an int/float/bool NumPy or Arrow column."""
    dt = getattr(arr, "dtype", None)
    if dt is not None and hasattr(dt, "kind"):
        return dt.kind in "iufb"
    t = getattr(arr, "type", None)
    if t is not None:
        return (
            pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)
        )
    return False


# Parallel partitioned grouped-aggregate (docs/PARALLEL_GROUPBY_SCOPE.md).
# Arrow's Table.group_by is single-threaded but has the best serial hash-agg
# kernel measured (2.4x Polars' serial). Polars wins high-cardinality group-bys
# purely by parallelism. So: hash-partition rows by the group key so every group
# lands wholly in one bucket, run Arrow's group_by on each bucket on its own
# thread (pyarrow releases the GIL), then concat — no cross-bucket merge. Beats
# Polars ~1.9x on q20's shape (52 vs 102 ms SF-3) and is bit-exact vs the single
# Arrow group_by. Module toggle for controlled A/B.
_PARALLEL_GROUPBY = True
_PARALLEL_GROUPBY_BUCKETS = 16
# Below this row count the partition + thread-spinup overhead outweighs the
# parallel speedup; the single Arrow group_by stays faster.
_PARALLEL_GROUPBY_MIN_ROWS = 300_000
# Minimum sampled distinct-key ratio to go parallel: few groups => Arrow's
# single group_by is already fast and partitioning is pure overhead.
_PARALLEL_GROUPBY_MIN_RATIO = 0.15
_GROUPBY_POOL = None

# Fuse an inner join feeding a hash aggregate: instead of materializing the full
# join output and re-reading it in the aggregate (the physical plan shows a
# Materialize breaker between the join and the group), compute the join indices
# and gather ONLY the columns the aggregate touches (group keys + agg values)
# straight into the group's Arrow table. Safe because a group is order-
# insensitive (it reorders rows anyway), so this dodges the eager row-order
# contract entirely. Probed in docs/BUFFER_JOIN_AGG_PROBE.md: 0.43x->~1.0x vs
# Polars on the agg-terminated shape, scoped to the low-cardinality final group
# (high-card falls through to the existing parallel groupby path unchanged).
# Module toggle for controlled A/B. DEFAULT-OFF: the node is correct and
# regression-free, but its direct-adjacency match (HashAggregate over a bare
# inner HashJoin) only fires on q4 of TPC-H — real queries put a filter/project
# or a JoinChain between the join and the group (see docs/BUFFER_JOIN_AGG_PROBE.md
# "Engine integration"). Kept as the validated foundation for those extensions.
_FUSE_JOIN_AGG = False


def _groupby_pool():
    """Shared thread pool for per-bucket Arrow group_by (created once)."""
    global _GROUPBY_POOL
    if _GROUPBY_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        import os

        _GROUPBY_POOL = ThreadPoolExecutor(
            max_workers=min(_PARALLEL_GROUPBY_BUCKETS, (os.cpu_count() or 4)),
            thread_name_prefix="lazy-groupby",
        )
    return _GROUPBY_POOL


def _partition_key_arrays(table, group_cols: list[str]):
    """Extract the group-key columns as int64 numpy arrays for partitioning.

    Returns a list of int64 arrays, or None if the keys are not cheaply packable
    (non-integer/temporal dtype, or any nulls — fall back to single group_by).
    """
    arrays = []
    for c in group_cols:
        chunked = table.column(c)
        if chunked.null_count:
            return None
        if isinstance(chunked, pa.ChunkedArray):
            chunked = chunked.combine_chunks()
        t = chunked.type
        if not (pa.types.is_integer(t) or pa.types.is_temporal(t)):
            return None
        npv = chunked.to_numpy(zero_copy_only=False)
        if npv.dtype.kind == "M":
            iv = npv.view(np.int64)
        else:
            iv = npv.astype(np.int64, copy=False)
        arrays.append(iv)
    return arrays


def _combine_partition_keys(key_arrays):
    """Combine int64 key arrays into one uint64 partition key per row.

    Collisions are harmless: a tuple always maps to one value, so a group never
    splits across buckets; distinct tuples colliding only co-locate groups.
    """
    comb = key_arrays[0].astype(np.uint64)
    for a in key_arrays[1:]:
        comb = comb * np.uint64(1000003) + a.astype(np.uint64)
    return comb


# ---------------------------------------------------------------------------
# Parallel string-key group-by via the lazy_groupby factorize kernel
# (docs/STRING_HASH_AGGREGATE_KERNEL.md). Arrow's group_by on raw string keys is
# single-threaded and ~2x Polars on high-cardinality string groups; dictionary-
# encoding doesn't pay when keys are near-unique. So factorize the mixed
# (string + numeric/temporal) key into dense int64 codes IN PARALLEL, aggregate
# on the cheap codes via the existing Arrow kernel (covers every agg func, null
# semantics, decode), then attach the real key columns at each group's
# representative row. Reaches Polars parity on q10's group (~30 vs ~57 ms).
# Module toggle for controlled A/B.
_STRING_HASH_GROUPBY = True


def _hash_key_int64(chunked):
    """Return a list of int64 arrays representing one numeric/temporal/boolean/
    decimal key column for hashing, or None if the dtype isn't supported.

    Most dtypes yield one array; ``decimal128``/``decimal256`` yield two/four
    (the raw value bytes viewed as int64 halves — equal decimals have equal
    bytes, matching Arrow's group_by). Floats are bit-viewed (so -0.0 and +0.0
    stay distinct, as in Arrow); any NaN float key returns None (Arrow/pandas
    NaN-group semantics differ, so fall back).
    """
    t = chunked.type
    if pa.types.is_integer(t) or pa.types.is_temporal(t):
        npv = chunked.to_numpy(zero_copy_only=False)
        if npv.dtype.kind == "M":
            return [np.ascontiguousarray(npv.view(np.int64))]
        return [np.ascontiguousarray(npv.astype(np.int64, copy=False))]
    if pa.types.is_boolean(t):
        return [
            np.ascontiguousarray(
                chunked.to_numpy(zero_copy_only=False).astype(np.int64)
            )
        ]
    if pa.types.is_floating(t):
        f = chunked.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        if np.isnan(f).any():
            return None  # Arrow/pandas float-NaN group semantics differ; fall back
        return [np.ascontiguousarray(f).view(np.int64)]
    if pa.types.is_decimal(t):
        # A decimal128/256 value is byte_width raw bytes (a two's-complement
        # scaled integer); equal values have equal bytes within a column (same
        # scale), so hash the bytes as int64 halves. Requires zero offset.
        if chunked.offset != 0:
            return None
        bufs = chunked.buffers()
        if len(bufs) < 2 or bufs[1] is None:
            return None
        k = t.byte_width // 8  # 2 for decimal128, 4 for decimal256
        raw = np.frombuffer(bufs[1], dtype=np.int64, count=len(chunked) * k)
        raw = raw.reshape(len(chunked), k)
        return [np.ascontiguousarray(raw[:, j]) for j in range(k)]
    return None


def _hash_str_buffers(chunked):
    """Return (int64 offsets, uint8 data) for an Arrow string column, or None.

    Only zero-offset, non-null arrays are handled (else fall back).
    """
    if chunked.null_count or chunked.offset != 0:
        return None
    bufs = chunked.buffers()
    if len(bufs) < 3 or bufs[1] is None:
        return None
    off_dt = np.int64 if pa.types.is_large_string(chunked.type) else np.int32
    offsets = np.frombuffer(bufs[1], dtype=off_dt, count=len(chunked) + 1)
    offsets = np.ascontiguousarray(offsets.astype(np.int64, copy=False))
    if bufs[2] is None:
        data = np.zeros(0, dtype=np.uint8)
    else:
        data = np.ascontiguousarray(np.frombuffer(bufs[2], dtype=np.uint8))
    return offsets, data


def _classify_group_keys(table):
    """Split a table's columns into (int_like, string) key sources for the
    factorize kernel, or return None if any key dtype is unsupported / null.

    int_like: list of (name, int64 array); string: list of (name, (offsets, data)).
    """
    int_like: list = []
    strings: list = []
    for c in table.column_names:
        ch = table.column(c)
        if isinstance(ch, pa.ChunkedArray):
            ch = ch.combine_chunks()
        if ch.null_count:
            return None
        t = ch.type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            sb = _hash_str_buffers(ch)
            if sb is None:
                return None
            strings.append((c, sb))
        else:
            parts = _hash_key_int64(ch)
            if parts is None:
                return None
            int_like.extend((c, p) for p in parts)
    return int_like, strings


def _combined_key_hash(int_like, strings, n, parallel):
    """Compute the 128-bit combined row hash (acc, acc2) over the key sources,
    parallelising the string hashing across row ranges when ``parallel``.
    """
    from pandas._libs.lazy_groupby import (
        hash_int_col,
        hash_string_col,
    )

    acc = np.empty(n, dtype=np.uint64)
    acc2 = np.empty(n, dtype=np.uint64)

    def hash_range(lo, hi):
        first = True
        for _, iv in int_like:
            hash_int_col(iv, acc, acc2, first, lo, hi)
            first = False
        for _, (off, data) in strings:
            hash_string_col(off, data, acc, acc2, first, lo, hi)
            first = False

    if parallel and n >= _PARALLEL_GROUPBY_MIN_ROWS:
        nt = min(_PARALLEL_GROUPBY_BUCKETS, (os.cpu_count() or 4))
        step = (n + nt - 1) // nt
        spans = [(i * step, min(n, (i + 1) * step)) for i in range(nt) if i * step < n]
        list(_groupby_pool().map(lambda s: hash_range(s[0], s[1]), spans))
    else:
        hash_range(0, n)
    return acc, acc2


def _factorize_from_classified(int_like, strings, n):
    """Factorize a classified mixed key into dense int64 codes in parallel.

    Returns (codes int64[n], reps int64[n_groups]). ``reps[c]`` is a row whose
    key equals group ``c`` (to recover the real key values). Groups are exact
    (128-bit key hash; see docs/STRING_HASH_AGGREGATE_KERNEL.md).
    """
    from pandas._libs.lazy_groupby import (
        bucket_factorize,
        partition_by_key,
    )

    acc, acc2 = _combined_key_hash(int_like, strings, n, parallel=True)
    nb = _PARALLEL_GROUPBY_BUCKETS
    perm, off = partition_by_key(np.ascontiguousarray(acc), nb)
    codes = np.empty(n, dtype=np.int64)

    def fac(b):
        return bucket_factorize(perm, int(off[b]), int(off[b + 1]), acc, acc2, codes)

    parts = list(_groupby_pool().map(fac, range(nb)))
    # offset each bucket's local codes by its global base; concat reps in order
    base = 0
    reps_list = []
    for b in range(nb):
        ng, rep = parts[b]
        if ng:
            if base:
                sl = perm[off[b] : off[b + 1]]
                codes[sl] += base
            reps_list.append(rep)
            base += ng
    reps = np.concatenate(reps_list) if reps_list else np.empty(0, dtype=np.int64)
    return codes, reps


def _string_hash_grouped_table(table, group_cols, agg_specs):
    """Parallel string-key grouped aggregate: factorize keys -> codes, aggregate
    on the codes via the Arrow kernel, attach real keys at the rep rows.

    Returns the result ``pa.Table`` (same schema as the single group_by) or None
    to fall back. Cardinality-gated on a cheap sample FIRST so low-card string
    groups (e.g. q1) don't pay to touch the full key columns.
    """
    n = table.num_rows

    # Cheap pre-check: any string key at all? (schema only, no data touched.)
    schema = table.schema
    if not any(
        pa.types.is_string(schema.field(c).type)
        or pa.types.is_large_string(schema.field(c).type)
        for c in group_cols
    ):
        return None

    sub = table.select(group_cols)

    # Cardinality gate on a strided sample BEFORE combining the full columns
    # (factorize only pays with many groups; the full combine_chunks on a wide
    # 18M-row string key is exactly the q1 cost we must avoid when rejecting).
    step = max(1, n // 8192)
    idx = pa.array(np.arange(0, n, step, dtype=np.int64))
    s_cls = _classify_group_keys(sub.take(idx))
    if s_cls is None:
        return None
    s_int, s_str = s_cls
    if not s_str:
        return None
    sacc, _ = _combined_key_hash(s_int, s_str, len(idx), parallel=False)
    if len(np.unique(sacc)) / len(sacc) < _PARALLEL_GROUPBY_MIN_RATIO:
        return None

    # Gate passed: classify the full columns once and factorize.
    cls = _classify_group_keys(sub)
    if cls is None:
        return None
    int_like, strings = cls
    if not strings:
        return None
    codes, reps = _factorize_from_classified(int_like, strings, n)

    code_name = "__lazy_grp_code__"
    value_cols = list(dict.fromkeys(in_col for _, in_col, _ in agg_specs))
    cols = {code_name: pa.array(codes)}
    for vc in value_cols:
        cols[vc] = table.column(vc)
    code_table = pa.table(cols)
    agg = dispatch_kernel("group_by", "arrow", code_table, [code_name], agg_specs)

    res_codes = agg.column(code_name).to_numpy()
    key_rows = pa.array(reps[res_codes])
    out: dict = {}
    for c in group_cols:
        out[c] = table.column(c).take(key_rows)
    for name in agg.column_names:
        if name != code_name:
            out[name] = agg.column(name)
    return pa.table(out)


def groupby_prefers_arrow(
    relevant_backends: set[str], all_numeric: bool, agg_funcs: set[str]
) -> bool:
    """Whether a hash aggregation should run on Arrow/acero rather than NumPy.

    Acero's internally-parallel hash aggregation beats the pandas-backed NumPy
    path at *every* size measured (e.g. ~10x at 10M rows, and still faster at
    1k) - and converting NumPy-resident *numeric* columns to Arrow is
    zero-copy. So numeric-keyed aggregation goes to Arrow even when the data is
    NumPy-backed, not only when a column is already Arrow-backed.

    Gated on every aggregation having an Arrow groupby kernel: ``median`` has
    none, so a query using it stays on NumPy (where pandas computes it).
    """
    if not agg_funcs or not all(has_kernel(f"groupby_{f}", "arrow") for f in agg_funcs):
        return False
    if agg_funcs & {"n_unique", "nunique"}:
        # acero count_distinct measured 2.4x slower than the packed-dedup
        # NumPy kernel on TPC-H q21's 6M-row shape (448 vs 183 ms).
        return False
    if "arrow" in relevant_backends:
        return True
    return all_numeric


def _rebatch_fixed(batches, block_rows: int = 2_000_000):
    """Re-slice a batch stream into EXACT fixed-size blocks.

    Parquet scanner batch boundaries vary run to run (readahead-dependent),
    so per-batch partial float sums were nondeterministic — measured at
    SF-300: 1.8M of 3M group sums differed between two identical runs,
    which broke TPC-H q15's ``total_revenue == max`` equality when its
    shared subquery was computed twice. Fixed block boundaries make the
    summation order a pure function of the row stream.
    """
    import numpy as np

    buf: list = []
    rows = 0

    def emit(n):
        nonlocal buf, rows
        names = buf[0].keys()
        merged = {}
        for name in names:
            parts = [b[name] for b in buf]
            if isinstance(parts[0], (pa.Array, pa.ChunkedArray)):
                chunks = []
                for p in parts:
                    chunks.extend(p.chunks if isinstance(p, pa.ChunkedArray) else [p])
                merged[name] = pa.chunked_array(chunks)
            else:
                merged[name] = np.concatenate(parts) if len(parts) > 1 else parts[0]
        out = {
            k: v.slice(0, n) if hasattr(v, "slice") else v[:n]
            for k, v in merged.items()
        }
        rest = {
            k: v.slice(n) if hasattr(v, "slice") else v[n:] for k, v in merged.items()
        }
        buf = [rest] if rows - n > 0 else []
        rows = rows - n
        return out

    for b in batches:
        if not b:
            continue
        n = len(next(iter(b.values())))
        if n == 0:
            continue
        buf.append(b)
        rows += n
        while rows >= block_rows:
            yield emit(block_rows)
    if rows:
        yield emit(rows)


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
            present = [c for c in relevant_cols if c in input_arrays]
            relevant_backends = {get_array_backend(input_arrays[c]) for c in present}
            all_numeric = bool(present) and all(
                _array_is_numeric(input_arrays[c]) for c in present
            )
            agg_funcs = {f for _, _, f in agg_specs}
            if groupby_prefers_arrow(relevant_backends, all_numeric, agg_funcs):
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

        result: ArrayDict = {}

        for output_name, col_name, agg_func in agg_specs:
            arr = input_arrays[col_name]
            # Backend must follow the column's actual array, not a single
            # frame-wide choice: a mixed frame (NumPy numerics + Arrow strings)
            # would otherwise route a NumPy column to an Arrow kernel (whose
            # NaN-skip helper expects ``arr.type``).
            arr_backend = get_array_backend(arr)

            kernel_name = agg_func  # e.g., "sum", "mean", "min", "max"
            try:
                value = dispatch_kernel(kernel_name, arr_backend, arr)
                # Scalar results - wrap in single-element array
                if isinstance(value, (pa.Scalar,)):
                    value = value.as_py()
                if arr_backend == "arrow":
                    result[output_name] = pa.array([value])
                else:
                    result[output_name] = np.array([value])
            except NotImplementedError:
                # Fallback for unsupported aggregations
                if arr_backend == "arrow":
                    np_arr = arr.to_numpy(zero_copy_only=False)
                else:
                    np_arr = arr
                value = getattr(np, agg_func)(np_arr)
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

    @staticmethod
    def _string_key_bytes(input_arrays: ArrayDict, group_cols: list[str]) -> int:
        """Total bytes of string-typed group-key columns, any container."""
        total = 0
        for c in group_cols:
            arr = input_arrays.get(c)
            if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                if (
                    pa.types.is_string(arr.type)
                    or pa.types.is_binary(arr.type)
                    or pa.types.is_large_string(arr.type)
                ):
                    total += arr.nbytes
            else:
                dt = str(getattr(arr, "dtype", "")).lower()
                if "string" in dt or "object" in dt or "binary" in dt:
                    total += int(getattr(arr, "nbytes", 0))
        return total

    def _grouped_arrow_table(self, table, group_cols, agg_specs):
        """Run the Arrow group_by, parallel-partitioned when it pays off.

        Returns the result ``pa.Table`` (group keys + aggregate columns), with
        identical schema to the single ``group_by`` kernel either way. Falls
        back to the single kernel on small inputs, non-packable keys, count-
        distinct (Arrow's is slow; handled by the hybrid path), or any error.
        """

        def single():
            return dispatch_kernel("group_by", "arrow", table, group_cols, agg_specs)

        n_rows = table.num_rows
        _nuniq = {"nunique", "n_unique", "count_distinct"}
        if (
            not _PARALLEL_GROUPBY
            or n_rows < _PARALLEL_GROUPBY_MIN_ROWS
            or any(f in _nuniq for _, _, f in agg_specs)
        ):
            return single()

        # String-keyed groups: the int partition path below can't pack string
        # keys, so they would fall to the single-threaded kernel. Route them to
        # the parallel factorize kernel instead (no-op/fall-through for all-
        # numeric keys, which _partition_key_arrays handles).
        if _STRING_HASH_GROUPBY:
            try:
                res = _string_hash_grouped_table(table, group_cols, agg_specs)
                if res is not None:
                    return res
            except Exception:
                pass  # any failure must not change results

        key_arrays = _partition_key_arrays(table, group_cols)
        if key_arrays is None:
            return single()

        # Cardinality gate FIRST, on a cheap strided sample — before building the
        # full combined key. Parallel partitioning only pays with many groups;
        # for few groups Arrow's single group_by is already fast and the full
        # comb + partition would be pure overhead (the q1 low-card case).
        n = len(key_arrays[0])
        step = max(1, n // 8192)
        samp = _combine_partition_keys([a[::step] for a in key_arrays])
        if len(np.unique(samp)) / len(samp) < _PARALLEL_GROUPBY_MIN_RATIO:
            return single()

        try:
            from pandas._libs.lazy_groupby import partition_by_key

            comb = _combine_partition_keys(key_arrays)

            # Narrow to just the columns group_by touches before the per-bucket
            # take — otherwise wide (post-join) inputs pay to copy every payload
            # column into each of the T buckets (the q17/q18 regression).
            needed = list(dict.fromkeys(group_cols + [c for _, c, _ in agg_specs]))
            if len(needed) < table.num_columns:
                table = table.select(needed)

            t_buckets = _PARALLEL_GROUPBY_BUCKETS
            perm, off = partition_by_key(np.ascontiguousarray(comb), t_buckets)
            perm_pa = pa.array(perm)

            def _bucket(i):
                lo = int(off[i])
                hi = int(off[i + 1])
                if hi == lo:
                    return None
                sub = table.take(perm_pa[lo:hi])
                return dispatch_kernel("group_by", "arrow", sub, group_cols, agg_specs)

            parts = [
                p
                for p in _groupby_pool().map(_bucket, range(t_buckets))
                if p is not None and p.num_rows
            ]
            if not parts:
                return single()
            # Groups are disjoint across buckets, so a plain concat is the full
            # result — no re-aggregation/merge needed.
            return pa.concat_tables(parts)
        except Exception:
            # Any failure (kernel, take, concat) must not change results.
            return single()

    def _execute_arrow_table_groupby(
        self,
        input_arrays: ArrayDict,
        group_cols: list[str],
        agg_specs: list[tuple[str, str, str]],
        context: ExecutionContext,
    ) -> ArrayDict:
        """
        Execute groupby using Arrow's native table group_by.

        Every arrow-groupby route MUST pass through here: acero's
        hash_aggregate row table ABORTS the process (uncatchable C++
        std::length_error) when group-key string payload nears int32
        limits (q10 at SF-300: ~20M rows x 7 customer key columns incl.
        comments). Such inputs fall back to the pandas groupby path.

        This is the preferred path for Arrow data because:
        1. Zero-copy table construction from existing Arrow arrays
        2. Multi-threaded aggregation (except first/last)
        3. Vectorized SIMD operations in Arrow's C++ backend
        4. Memory-efficient columnar processing
        """

        # Build Arrow table from arrays (excluding index columns)
        if self._string_key_bytes(input_arrays, group_cols) >= 1_500_000_000:
            return self._execute_multi_key_groupby(
                input_arrays, group_cols, agg_specs, "numpy", context
            )

        columns = {k: v for k, v in input_arrays.items() if not is_index_col(k)}
        table = pa.table(columns)

        # Execute groupby using Arrow's native group_by (single-threaded), or the
        # parallel partitioned path on large high-cardinality inputs.
        result_table = self._grouped_arrow_table(table, group_cols, agg_specs)

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

    def can_preaggregate(self) -> bool:
        """True when this aggregation can consume pre-aggregated morsel
        batches (G4): grouped, every agg expr is a simple mergeable
        Call(FieldRef) the streaming map/reduce path handles."""
        group_cols = [extract_output_name(e) for e in self.group_by]
        if not group_cols:
            return False
        specs: list[tuple[str, str, str]] = []
        for expr in self.agg_exprs:
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if (
                isinstance(ir, Call)
                and ir.is_aggregate
                and ir.args
                and isinstance(ir.args[0], FieldRef)
            ):
                specs.append((extract_output_name(expr), ir.args[0].name, ir.function))
            else:
                return False  # non-simple agg: streaming would drop it
        return self._can_stream_aggregation(specs)

    def partial_aggregate_batch(self, batch: ArrayDict) -> ArrayDict:
        """Phase-1 partial aggregation of one morsel (G4 worker-side).

        Same per-batch spec as the streaming map/reduce path (mean emits
        __sum_/__count_ columns); the sink later merges partials only.
        """
        group_cols = [extract_output_name(e) for e in self.group_by]
        agg_specs: list[tuple[str, str, str]] = []
        for expr in self.agg_exprs:
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            agg_specs.append((extract_output_name(expr), ir.args[0].name, ir.function))
        batch_specs: list[tuple[str, str, str]] = []
        for output_name, col_name, agg_func in agg_specs:
            if agg_func == "mean":
                batch_specs.append((f"__sum_{output_name}__", col_name, "sum"))
                batch_specs.append((f"__count_{output_name}__", col_name, "count"))
            else:
                batch_specs.append((output_name, col_name, agg_func))
        columns = {k: v for k, v in batch.items() if not is_index_col(k)}
        table = dispatch_kernel(
            "group_by", "arrow", pa.table(columns), group_cols, batch_specs
        )
        return {name: table.column(name) for name in table.column_names}

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

        # G4: morsel workers may have partial-aggregated already — the
        # incoming batches ARE phase-1 partials (mean already decomposed
        # into __sum_/__count_); skip straight to the merge.
        if getattr(self.input, "preaggregated", False):
            for batch in self.input.execute_batches(context):
                if not batch:
                    continue
                columns = {k: v for k, v in batch.items() if not is_index_col(k)}
                if columns and len(next(iter(columns.values()))):
                    partial_tables.append(pa.table(columns))
            return self._merge_streaming_partials(
                partial_tables, group_cols, agg_specs, mean_aggs, direct_aggs, context
            )

        for batch in _rebatch_fixed(self.input.execute_batches(context)):
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

            # Reduction-quality bail (G4): with a high-cardinality group key
            # the partials barely shrink, so map/reduce re-aggregates nearly
            # the full input twice (measured: q3's ~3M-group key regressed
            # 1.07x). When the input is pre-batched in memory (the G4
            # adapter — materializing is an option there, unlike file
            # streams), a poor first reduction falls back to one-shot
            # aggregation over the concatenated input.
            if (
                not partial_tables
                and getattr(self.input, "batches", None) is not None
                and partial.num_rows * 2 > len(first_arr)
            ):
                arrays = self.input.execute(context)
                return self._execute_grouped_aggregation(
                    arrays, group_cols, agg_specs, "arrow", context
                )
            partial_tables.append(partial)

        return self._merge_streaming_partials(
            partial_tables, group_cols, agg_specs, mean_aggs, direct_aggs, context
        )

    def _merge_streaming_partials(
        self,
        partial_tables: list,
        group_cols: list[str],
        agg_specs: list[tuple[str, str, str]],
        mean_aggs: list[tuple[str, str]],
        direct_aggs: list[tuple[str, str, str]],
        context: ExecutionContext,
    ) -> ArrayDict:
        """Phase 2 of streaming aggregation: merge partials and finish."""
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

# Minimum rows before multi-key sort routes through Arrow's table-level
# sort_indices (multi-threaded; ~1.65x over np.lexsort at 10M rows).


def _unwrap_materialize(plan):
    """Look through the PhysicalMaterialize bookkeeping wrapper."""
    if isinstance(plan, PhysicalMaterialize):
        return plan.input
    return plan


class _CompositeJoin:
    """Late-materialized join-chain result: bases + row indices, no payloads.

    Represents the output of a chain of inner joins as the original base
    ArrayDicts plus one row-index array per base (``None`` = identity) and a
    column→base map. Payload columns are gathered exactly once, in
    ``gather()``, by the topmost join of the chain.
    """

    def __init__(self, bases, indices, col_map):
        self.bases = bases  # list[ArrayDict]
        self.indices = indices  # list[np.ndarray | None]
        self.col_map = col_map  # dict[str, int] — column name -> base pos

    @classmethod
    def from_arrays(cls, arrays: ArrayDict) -> _CompositeJoin:
        data = {n: a for n, a in arrays.items() if not is_index_col(n)}
        return cls([data], [None], dict.fromkeys(data, 0))

    def column(self, name: str):
        """The (gathered) values of one column — used for join keys only."""
        b = self.col_map.get(name)
        if b is None:
            return None
        arr = self.bases[b].get(name)
        idx = self.indices[b]
        if idx is None or not isinstance(arr, np.ndarray):
            return arr if idx is None else None
        return arr[idx]

    def extend(self, chain_rows, right_arrays, right_idx, drop_right_keys):
        """One more chain step: re-map every base through ``chain_rows`` and
        append the right side with its own row indices. ``drop_right_keys``
        are shared-name join keys already provided by the left side."""
        new_indices = [
            chain_rows if idx is None else idx[chain_rows] for idx in self.indices
        ]
        new_indices.append(right_idx)
        bases = [*self.bases, right_arrays]
        col_map = dict(self.col_map)
        b_right = len(bases) - 1
        for n in right_arrays:
            if n not in drop_right_keys:
                col_map[n] = b_right
        return _CompositeJoin(bases, new_indices, col_map)

    def gather(self) -> ArrayDict:
        """Materialize: one parallel gather per base, columns in merge order."""
        per_base: list[ArrayDict] = []
        for b, (base, idx) in enumerate(zip(self.bases, self.indices, strict=True)):
            cols = {n: a for n, a in base.items() if self.col_map.get(n) == b}
            per_base.append(cols if idx is None else _take_all_columns(cols, idx))
        return {n: per_base[b][n] for n, b in self.col_map.items()}


def _take_all_columns(input_arrays: ArrayDict, indices) -> ArrayDict:
    """
    Apply gather indices to every column, in parallel for large data.

    indices may be a NumPy array or an Arrow array; it is converted once
    for the backends that need it rather than per column.
    """
    from concurrent.futures import ThreadPoolExecutor
    import os

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
                return _take_all_columns(input_arrays, sort_indices)

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
        for col_name in _get_ordered_columns(batches):
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

        # Execute left and right sides in parallel. Every join path needs both
        # sides materialized (a hash join needs the full build side; this
        # engine does not stream join inputs), so we do it once up front and
        # then choose the join strategy from the actual materialized sizes.
        left_arrays, right_arrays = self._execute_sides_parallel(context)
        return self._join_arrays(context, left_arrays, right_arrays)

    def _join_arrays(
        self,
        context: ExecutionContext,
        left_arrays: ArrayDict,
        right_arrays: ArrayDict,
    ) -> ArrayDict:
        """Join two materialized sides — the whole strategy ladder.

        Split from ``execute`` so PhysicalJoinChain can run any chain step on
        already-materialized arrays with exactly this node's semantics
        (spill-aware Grace, pd.merge/Cython fast paths, acero, indexer).
        """
        equi = self.on is not None or (
            self.left_on is not None and self.right_on is not None
        )

        # Out-of-core: spill to a Grace hash join, but only when the
        # materialized inputs actually exceed the operator memory budget.
        # Size-triggered (not merely spill-enabled), so turning spilling on no
        # longer pessimizes small joins - those fall through to the fast
        # in-memory pd.merge below; only genuinely large joins partition/spill.
        if (
            context.spill_enabled
            and context.spill_manager is not None
            and self.how in ("inner", "left", "right")
            and equi
        ):
            budget = context._spill_config.operator_budget_mb * 1024 * 1024
            in_memory_bytes = get_arrays_bytes(left_arrays) + get_arrays_bytes(
                right_arrays
            )
            if in_memory_bytes > budget:
                return self._execute_grace_hash_join(context, left_arrays, right_arrays)

        # pd.merge fast path. pd.merge IS the eager-pandas semantics this join
        # promises to match - so it is row-order- and null-correct by
        # construction (no acero-safe key gate needed) - and it is the fastest
        # path measured on the H2O db-benchmark: it beats both the custom
        # indexer hash join and Arrow/acero by 2-7x, because acero's parallel
        # join is undone by the Arrow<->pandas round-trip on payload columns.
        # Used for in-memory equi-joins whose index is not observed downstream
        # (the default RangeIndex regenerates at output); index-observing joins
        # keep the indexer path below, which carries the left index.
        if (
            self.how in ("inner", "left", "right", "outer")
            and equi
            and not context.preserve_index
            and not context.user_set_index
        ):
            return self._execute_pandas_merge(left_arrays, right_arrays)

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

    def _join_key_arrays_i64(self, left_get, right_get):
        """Resolve this join's key pair(s) to one int64 array per side.

        One int key pair passes through; TWO int key pairs pack into a
        single int64 key — ``(k1 - lo1) * span2 + (k2 - lo2)`` with offsets
        and spans computed over BOTH sides so equal pairs map to equal
        packed values (the same trick as the packed ``n_unique`` kernel).
        This lets composite-key joins (TPC-H's ``partsupp`` steps) use the
        Cython kernel and join chains. Returns ``(lk64, rk64, key_pairs)``
        or ``None`` (>2 keys, non-int/uint64 keys, empty side, shared-name
        dtype mismatch, or a packed range that would not fit int63).
        """
        if self.on is not None:
            pairs = [(k, k) for k in self.on]
        elif self.left_on is not None and self.right_on is not None:
            if len(self.left_on) != len(self.right_on):
                return None
            pairs = list(zip(self.left_on, self.right_on, strict=True))
        else:
            return None
        if len(pairs) not in (1, 2):
            return None
        larrs, rarrs = [], []
        for ln, rn in pairs:
            la, ra = left_get(ln), right_get(rn)
            if not (
                isinstance(la, np.ndarray)
                and isinstance(ra, np.ndarray)
                and la.dtype.kind in "iu"
                and ra.dtype.kind in "iu"
                and la.dtype != np.dtype("uint64")
                and ra.dtype != np.dtype("uint64")
                and len(la)
                and len(ra)
            ):
                return None
            if ln == rn and la.dtype != ra.dtype:
                # Shared key column is gathered from one side; mismatched
                # dtypes would diverge from pd.merge's upcast result.
                return None
            larrs.append(la)
            rarrs.append(ra)
        if len(pairs) == 1:
            return (
                np.ascontiguousarray(larrs[0], dtype=np.int64),
                np.ascontiguousarray(rarrs[0], dtype=np.int64),
                pairs,
            )
        lo1 = min(int(larrs[0].min()), int(rarrs[0].min()))
        hi1 = max(int(larrs[0].max()), int(rarrs[0].max()))
        lo2 = min(int(larrs[1].min()), int(rarrs[1].min()))
        hi2 = max(int(larrs[1].max()), int(rarrs[1].max()))
        span2 = hi2 - lo2 + 1
        if span2 <= 0 or (hi1 - lo1 + 1) > (2**62) // span2:
            return None
        lk = (larrs[0].astype(np.int64) - lo1) * span2 + (
            larrs[1].astype(np.int64) - lo2
        )
        rk = (rarrs[0].astype(np.int64) - lo1) * span2 + (
            rarrs[1].astype(np.int64) - lo2
        )
        return lk, rk, pairs

    def _compose_step(
        self,
        left_comp: _CompositeJoin,
        right_arrays: ArrayDict,
        preserve_order: bool = True,
    ) -> _CompositeJoin | None:
        """Late-materialization chain step: join without gathering payloads.

        Given the running chain as a ``_CompositeJoin`` and this step's
        materialized right side, return the extended composite — or ``None``
        when this step is ineligible at runtime (the chain driver then
        materializes and joins through ``_join_arrays``, degrading
        gracefully).

        Eligibility per step mirrors ``_try_cython_join``: inner, single
        NumPy-int key, no overlapping payload names.
        """
        if self.how != "inner":
            return None
        resolved = self._join_key_arrays_i64(left_comp.column, right_arrays.get)
        if resolved is None:
            return None
        lk64, rk64, key_pairs = resolved
        shared = {rn for ln, rn in key_pairs if ln == rn}
        if (set(left_comp.col_map) & set(right_arrays)) - shared:
            return None  # driver materializes and joins via _join_arrays

        from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

        if len(rk64) <= 4 * len(lk64):
            # Natural: build right, probe the chain in row order — preserves
            # pd.merge's cascade order by construction.
            chain_rows, right_idx = inner_join_indexers_i8(lk64, rk64)
        elif preserve_order:
            right_probe, chain_build = inner_join_indexers_i8(rk64, lk64)
            order = np.argsort(chain_build, kind="stable")
            chain_rows = chain_build[order]
            right_idx = right_probe[order]
        else:
            # Order-free chain: keep the kernel's probe-major order.
            right_idx, chain_rows = inner_join_indexers_i8(rk64, lk64)

        return left_comp.extend(chain_rows, right_arrays, right_idx, shared)

    def _try_cython_join(
        self,
        left_arrays: ArrayDict,
        right_arrays: ArrayDict,
    ) -> ArrayDict | None:
        """Single-pass Cython hash join for inner equi-joins on one int key.

        ``pd.merge``'s factorize-then-join machinery is the dominant cost of
        analytical join pipelines (measured: ~52% of TPC-H q5 in
        ``_factorize_keys`` alone). ``pandas._libs.lazy_join`` replaces it
        with a CSR-grouped hash table and a probe whose count/fill passes
        are ``nogil`` and thread-parallel — measured 7-11x over ``pd.merge``
        on the dominant TPC-H join shapes at the indexer level.

        Row order is exactly ``pd.merge``'s inner order. The kernel builds on
        the RIGHT and probes the LEFT in row order, which IS that order; when
        the right side is much larger we build on the LEFT instead (hash
        builds should be on the small side) and restore left-row-major order
        with one stable integer argsort of the output indices.

        Falls through to ``pd.merge`` (returns None) unless: inner join,
        exactly one key pair, both keys NumPy integer (uint64 excluded — its
        upper range cannot round-trip through int64), and no overlapping
        payload names (keeps output naming trivially identical to pd.merge).
        """
        if self.how != "inner":
            return None
        left_cols = {n: a for n, a in left_arrays.items() if not is_index_col(n)}
        right_cols = {n: a for n, a in right_arrays.items() if not is_index_col(n)}
        resolved = self._join_key_arrays_i64(left_cols.get, right_cols.get)
        if resolved is None:
            return None
        lk64, rk64, key_pairs = resolved
        shared = {rn for ln, rn in key_pairs if ln == rn}
        if (set(left_cols) & set(right_cols)) - shared:
            return None

        from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

        # Payload-width-aware gating (measured, benchmarks/exp_join_decomp.py).
        # The threaded CSR probe stays in NumPy and beats pd.merge ~1.9-2.3x —
        # and Polars ~1.3x — on *narrow* large high-fanout joins (3 gathered
        # columns: 10M exploding 1303->965 ms end-to-end). But the per-column
        # gather of an exploded *moderate* payload loses to pd.merge's
        # consolidated block take, and a tight interleaved A/B showed TPC-H
        # q7/q21 (lineitem joins, moderate payload, high hit) regress ~2-6%
        # when routed onto the kernel. So relax the size bail ONLY for very
        # narrow joins where the win is large and unambiguous; everything else
        # keeps the original behavior exactly (no regression on the validated
        # workload).
        n_gather = len(left_cols) + sum(1 for n in right_cols if n not in shared)
        NARROW_PAYLOAD = 4
        if n_gather <= NARROW_PAYLOAD:
            cap = None  # narrow: kernel wins at any size/selectivity
        else:
            # Wide: exact original gate (500k size bail + 0.5 selectivity).
            if min(len(lk64), len(rk64)) > 500_000:
                return None
            cap = 0.5
        if len(rk64) <= 4 * len(lk64):
            # Natural direction: build right, probe left → pd.merge order.
            result = inner_join_indexers_i8(lk64, rk64, max_hit_fraction=cap)
            if result is None:
                return None
            left_idx, right_idx = result
        else:
            # Right side much larger: build on the (small) left, probe with
            # the right, then restore left-row-major order. NumPy's stable
            # sort on integers is a radix sort.
            result = inner_join_indexers_i8(rk64, lk64, max_hit_fraction=cap)
            if result is None:
                return None
            right_probe, left_build = result
            order = np.argsort(left_build, kind="stable")
            left_idx = left_build[order]
            right_idx = right_probe[order]

        out: ArrayDict = {}
        out.update(_take_all_columns(left_cols, left_idx))
        right_gather = {n: a for n, a in right_cols.items() if n not in shared}
        out.update(_take_all_columns(right_gather, right_idx))
        return out

    def _execute_pandas_merge(
        self,
        left_arrays: ArrayDict,
        right_arrays: ArrayDict,
    ) -> ArrayDict:
        """Equi-join through ``pd.merge`` - the eager semantics, fastest path.

        Index columns are dropped (the merged row order is itself the eager
        order; the default RangeIndex regenerates at output). Arrow-backed
        columns ride through as ArrowExtensionArray so string keys/payload
        stay columnar.
        """
        import pandas as pd
        from pandas import DataFrame
        from pandas.arrays import ArrowExtensionArray

        fast = self._try_cython_join(left_arrays, right_arrays)
        if fast is not None:
            return self._reorder_columns(fast)

        def to_frame(arrays: ArrayDict) -> DataFrame:
            cols = {}
            for name, arr in arrays.items():
                if is_index_col(name):
                    continue
                if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                    ca = (
                        arr
                        if isinstance(arr, pa.ChunkedArray)
                        else pa.chunked_array([arr])
                    )
                    # int32-offset string/binary columns overflow past 2 GB
                    # inside merge's pyarrow take (SEGFAULT, not a clean
                    # raise — TPC-H q2/q18 at SF-100). Upcast to 64-bit
                    # offsets before merge; only the offsets buffer is
                    # rebuilt, not the string bytes.
                    if ca.type == pa.string():
                        ca = ca.cast(pa.large_string())
                    elif ca.type == pa.binary():
                        ca = ca.cast(pa.large_binary())
                    cols[name] = ArrowExtensionArray(ca)
                else:
                    cols[name] = arr
            return DataFrame(cols, copy=False)

        left_df = to_frame(left_arrays)
        right_df = to_frame(right_arrays)
        if self.on is not None:
            merged = pd.merge(
                left_df,
                right_df,
                on=list(self.on),
                how=self.how,
                suffixes=self.suffix,
            )
        else:
            merged = pd.merge(
                left_df,
                right_df,
                left_on=list(self.left_on),
                right_on=list(self.right_on),
                how=self.how,
                suffixes=self.suffix,
            )
        out, _, _ = dataframe_to_arrays(merged)
        return self._reorder_columns(out)

    def _execute_arrow_join(
        self,
        left_data: ArrayDict,
        right_data: ArrayDict,
        swapped: bool = False,
    ) -> ArrayDict:
        """Execute join using Arrow's native join."""

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

    def _execute_grace_hash_join(
        self,
        context: ExecutionContext,
        left_arrays: ArrayDict | None = None,
        right_arrays: ArrayDict | None = None,
    ) -> ArrayDict:
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

        # Sides are normally materialized by the caller (execute) and passed
        # in so the size-based spill decision can be made; materialize here
        # only when called directly without them.
        if left_arrays is None or right_arrays is None:
            left_arrays, right_arrays = self._execute_sides_parallel(context)

        spill_manager = context.spill_manager
        if spill_manager is None:
            # Fallback to regular in-memory join
            return self._execute_dataframe_join(left_arrays, right_arrays, context)

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
            suffix=self.suffix,
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


class _FusedAggSpec:
    """Plan-time translation of filter+project+aggregate onto the fused
    Cython kernel (pandas._libs.lazy_fused_agg). Built by
    PhysicalPlanner._fuse_filter_aggregates; None fields mean untranslatable.
    """

    def __init__(self):
        self.i64_preds = []  # (col, lo, hi) closed int64 ranges
        self.f64_preds = []  # (col, lo, hi) closed float64 ranges
        self.aggs = []  # (out_name, kind, col_a, col_b, col_c)
        self.mean_outs = {}  # out_name -> (sum_slot, count_slot)
        self.group_cols = []  # source columns of the group keys (or empty)


@dataclass
class PhysicalFusedFilterAgg(PhysicalPlan):
    """Single-pass fused filter+aggregate over a scan (no group keys).

    The probe behind this: q6's fused C loop ran 10.8 ms on 8 threads vs
    70 ms through materializing operators (~26 ms Polars). Bails to the
    original subplan at runtime when inputs are not cleanly numeric
    (nulls/NaNs would diverge from pandas skip-NaN semantics).
    """

    scan: PhysicalPlan
    spec: object
    fallback: PhysicalPlan
    schema: Schema

    def children(self) -> list[PhysicalPlan]:
        return [self.scan]

    @property
    def output_schema(self) -> Schema:
        return self.schema

    @staticmethod
    def _derive_group_codes(arrays, spec):
        """(codes, cards, key_decode, total_card) or None (bail to fallback)."""
        import numpy as _np

        # Dense group codes: int64 keys with a small value range, or
        # 1-byte arrow strings (the data buffer IS the code array — q1's
        # returnflag/linestatus; dictionary_encode at 18M rows costs
        # ~194 ms, the buffer view is free).
        key_codes = []
        key_decode = []  # ("int", minv) | ("byte",)
        cards = []
        for name in spec.group_cols:
            arr = arrays.get(name)
            if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                ca = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
                if (
                    pa.types.is_string(ca.type)
                    and ca.null_count == 0
                    and len(ca.buffers()) >= 3
                    and ca.buffers()[2] is not None
                    and ca.buffers()[2].size == len(ca)
                ):
                    codes = _np.frombuffer(ca.buffers()[2], dtype=_np.uint8).astype(
                        _np.int64
                    )
                    key_codes.append(codes)
                    key_decode.append(("byte",))
                    cards.append(256)
                    continue
                if ca.null_count == 0 and pa.types.is_integer(ca.type):
                    arr = ca.to_numpy(zero_copy_only=False)
                else:
                    return None
            a = _np.asarray(arr)
            if a.dtype.kind not in "iu":
                return None
            mn, mx = int(a.min()), int(a.max())
            card = mx - mn + 1
            if card > 2_000_000:
                return None
            key_codes.append(a.astype(_np.int64) - mn)
            key_decode.append(("int", mn))
            cards.append(card)
        total_card = 1
        for c in cards:
            total_card *= c
            if total_card > 4_000_000:
                return None
        codes = key_codes[0].copy()
        for j in range(1, len(key_codes)):
            codes *= cards[j]
            codes += key_codes[j]
        codes = _np.ascontiguousarray(codes)
        return codes, cards, key_decode, total_card

    def _execute_grouped_fused(
        self,
        arrays,
        spec,
        group_info,
        i64_cols,
        il,
        ih,
        f64_cols,
        fl,
        fh,
        kinds_arr,
        agg_a,
        agg_b,
        agg_c,
        n,
        n_workers,
        context,
    ):
        import numpy as _np

        from pandas._libs.lazy_fused_agg import fused_filter_group_aggs

        codes, cards, key_decode, total_card = group_info

        from concurrent.futures import ThreadPoolExecutor

        def run(s):
            return fused_filter_group_aggs(
                i64_cols,
                il,
                ih,
                f64_cols,
                fl,
                fh,
                kinds_arr,
                agg_a,
                agg_b,
                agg_c,
                codes,
                total_card,
                s[0],
                s[1],
            )

        if n < 1_000_000 or n_workers == 1:
            out, counts = run((0, n))
        else:
            bounds = [
                (i * n // n_workers, (i + 1) * n // n_workers) for i in range(n_workers)
            ]
            with ThreadPoolExecutor(n_workers) as ex:
                parts = list(ex.map(run, bounds))
            out = parts[0][0]
            counts = parts[0][1]
            for po, pc_ in parts[1:]:
                counts += pc_
                for k, kind in enumerate(kinds_arr):
                    if kind in (0, 1, 5, 6):
                        out[k] += po[k]
                    elif kind == 3:
                        out[k] = _np.fmin(out[k], po[k])
                    elif kind == 4:
                        out[k] = _np.fmax(out[k], po[k])
            for k, kind in enumerate(kinds_arr):
                if kind == 2:
                    out[k] = counts.astype(_np.float64)

        idx = _np.flatnonzero(counts > 0)
        if len(idx) == 0:
            return self.fallback.execute(context)
        built: dict = {}
        rem = idx.copy()
        for j in range(len(spec.group_cols) - 1, -1, -1):
            cj = rem % cards[j]
            rem = rem // cards[j]
            dec = key_decode[j]
            if dec[0] == "int":
                built[spec.group_cols[j]] = cj + dec[1]
            else:
                built[spec.group_cols[j]] = _np.array(
                    [chr(b) for b in cj], dtype=object
                )
        slots = {}
        for k, (oname, kind, _a, _b, _c) in enumerate(spec.aggs):
            slots[oname] = (out[k][idx], kind)
        for oname, kind, _a, _b, _c in spec.aggs:
            if oname in spec.mean_outs or oname.startswith("__fused_"):
                continue
            vals, knd = slots[oname]
            built[oname] = vals.astype(_np.int64) if knd == 2 else vals
        for oname, (s_slot, c_slot) in spec.mean_outs.items():
            cnt = out[c_slot][idx]
            built[oname] = _np.where(cnt > 0, out[s_slot][idx] / cnt, _np.nan)
        result: ArrayDict = {}
        for name in self.schema.names:
            if name in built:
                result[name] = built[name]
        return result

    def execute(self, context: ExecutionContext) -> ArrayDict:
        import numpy as _np

        from pandas._libs.lazy_fused_agg import fused_filter_aggs

        arrays = self.scan.execute(context)
        spec = self.spec

        # Grouped: derive group codes FIRST — the cardinality cap is the
        # common bail (q20's 600k x 30k key space), and bailing after the
        # predicate/agg fetches wasted ~100 ms of copies per run.
        group_info = None
        if spec.group_cols:
            group_info = self._derive_group_codes(arrays, spec)
            if group_info is None:
                return self.fallback.execute(context)

        def as_np(name):
            """(int64/float64 view, ns_per_unit) — datetime literals are
            baked in NANOSECONDS at plan time; columns may be us/ms/s."""
            arr = arrays.get(name)
            if arr is None:
                return None, 1
            if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                if arr.null_count:
                    return None, 1
                arr = arr.to_numpy(zero_copy_only=False)
            arr = _np.asarray(arr)
            if arr.dtype.kind == "M":
                unit = _np.datetime_data(arr.dtype)[0]
                scale = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 10**9}.get(unit)
                if scale is None:
                    return None, 1
                return _np.ascontiguousarray(arr).view("int64"), scale
            return arr, 1

        i64_cols, i64_lo, i64_hi = [], [], []
        for col, lo, hi in spec.i64_preds:
            a, scale = as_np(col)
            if a is None or a.dtype != _np.int64:
                return self.fallback.execute(context)
            i64_cols.append(_np.ascontiguousarray(a, dtype=_np.int64))
            # ns-baked bounds -> column unit; ceil for lo, floor for hi so
            # the closed range stays exact for whole-unit boundaries.
            i64_lo.append(-(-lo // scale) if lo > -(2**62) else lo)
            i64_hi.append(hi // scale if hi < 2**62 else hi)
        f64_cols, f64_lo, f64_hi = [], [], []
        for col, lo, hi in spec.f64_preds:
            a, _ = as_np(col)
            if a is None or a.dtype.kind != "f":
                return self.fallback.execute(context)
            f64_cols.append(_np.ascontiguousarray(a, dtype=_np.float64))
            f64_lo.append(lo)
            f64_hi.append(hi)
        agg_kinds, agg_a, agg_b, agg_c = [], [], [], []
        _f64_memo: dict = {}

        def fetch_f64(name):
            # Memoized per column: q1 references the same 4 columns from
            # ~10 agg slots — repeating the isnan pass and the arrow->numpy
            # copy per slot cost more than the fused kernel itself.
            if name in _f64_memo:
                return _f64_memo[name]
            a, _ = as_np(name)
            if a is None or a.dtype.kind != "f" or _np.isnan(a).any():
                res = None
            else:
                res = _np.ascontiguousarray(a, dtype=_np.float64)
            _f64_memo[name] = res
            return res

        for _out, kind, ca, cb, cc in spec.aggs:
            if kind == 2:
                agg_a.append(None)
                agg_b.append(None)
                agg_c.append(None)
            else:
                cols = []
                for cn in (ca, cb, cc):
                    if cn is None:
                        cols.append(None)
                        continue
                    f = fetch_f64(cn)
                    if f is None:
                        return self.fallback.execute(context)
                    cols.append(f)
                agg_a.append(cols[0])
                agg_b.append(cols[1])
                agg_c.append(cols[2])
            agg_kinds.append(kind)
        n = len(next(iter(arrays.values()))) if arrays else 0
        if n == 0:
            return self.fallback.execute(context)
        kinds_arr = _np.asarray(agg_kinds, dtype=_np.int64)
        il = _np.asarray(i64_lo, dtype=_np.int64)
        ih = _np.asarray(i64_hi, dtype=_np.int64)
        fl = _np.asarray(f64_lo, dtype=_np.float64)
        fh = _np.asarray(f64_hi, dtype=_np.float64)

        from concurrent.futures import ThreadPoolExecutor
        import os as _os

        n_workers = min(8, _os.cpu_count() or 1)

        if spec.group_cols:
            return self._execute_grouped_fused(
                arrays,
                spec,
                group_info,
                i64_cols,
                il,
                ih,
                f64_cols,
                fl,
                fh,
                kinds_arr,
                agg_a,
                agg_b,
                agg_c,
                n,
                n_workers,
                context,
            )

        for k, kind in enumerate(agg_kinds):
            if agg_a[k] is None and kind != 2:
                return self.fallback.execute(context)
            if agg_a[k] is None:
                agg_a[k] = _np.empty(0)
        if n < 1_000_000 or n_workers == 1:
            out, _ = fused_filter_aggs(
                i64_cols, il, ih, f64_cols, fl, fh, kinds_arr, agg_a, agg_b, 0, n
            )
        else:
            bounds = [
                (i * n // n_workers, (i + 1) * n // n_workers) for i in range(n_workers)
            ]
            with ThreadPoolExecutor(n_workers) as ex:
                parts = list(
                    ex.map(
                        lambda s: fused_filter_aggs(
                            i64_cols,
                            il,
                            ih,
                            f64_cols,
                            fl,
                            fh,
                            kinds_arr,
                            agg_a,
                            agg_b,
                            s[0],
                            s[1],
                        ),
                        bounds,
                    )
                )
            out = _np.zeros(len(agg_kinds))
            for k, kind in enumerate(agg_kinds):
                vals = [p[0][k] for p in parts]
                if kind in (0, 1, 2):
                    out[k] = _np.nansum(vals)
                elif kind == 3:
                    out[k] = _np.nanmin(vals)
                else:
                    out[k] = _np.nanmax(vals)
        result: ArrayDict = {}
        slots = {}
        for k, (oname, kind, _a, _b, _c) in enumerate(spec.aggs):
            slots[oname] = out[k]
        for oname, kind, _a, _b, _c in spec.aggs:
            if oname in spec.mean_outs:
                continue
            if oname.startswith("__fused_"):
                continue
            if kind == 2:
                result[oname] = np.array([int(slots[oname])], dtype=np.int64)
            else:
                result[oname] = np.array([slots[oname]], dtype=np.float64)
        for oname, (s_slot, c_slot) in spec.mean_outs.items():
            cnt = out[c_slot]
            result[oname] = np.array([out[s_slot] / cnt if cnt else float("nan")])
        return result


@dataclass
class PhysicalFusedJoinAgg(PhysicalPlan):
    """Inner join feeding a hash aggregate, fused to skip the join's full
    materialization.

    The plan for ``agg(join(a, b))`` puts a Materialize breaker between the
    join and the group, so the join's entire output is built and re-read by the
    aggregate — but the aggregate only ever touches the group keys and the
    aggregate value columns; the rest of each joined row is built and
    discarded. This node computes the join indices and gathers ONLY those
    surviving columns straight off the indices (Arrow ``take``), then runs the
    unchanged group path on the narrow result.

    Safe because a hash aggregate is order-insensitive — it reorders rows
    anyway — so there is no eager row-order contract to preserve across the
    join (unlike a join feeding a sort/limit/plain collect). Probed in
    docs/BUFFER_JOIN_AGG_PROBE.md: ~0.43x->1.0x vs Polars on the agg-terminated
    shape; scoped to a low-cardinality final group (high-card falls through to
    the same parallel-groupby kernel the plain path uses, since the group runs
    through the identical ``_execute_grouped_aggregation``).

    ``join`` and ``agg`` are the original nodes, kept only for their parameter
    helpers (key resolution, group execution); ``left``/``right`` are the join
    sides the pipeline compiler rebinds to precomputed inputs at runtime.
    ``fallback`` is the original ``agg`` subtree, run if the fused gather turns
    out inapplicable at runtime (rare given the planner's static gate).
    """

    left: PhysicalPlan
    right: PhysicalPlan
    join: PhysicalPlan
    agg: PhysicalPlan
    fallback: PhysicalPlan
    schema: Schema

    def children(self) -> list[PhysicalPlan]:
        return [self.left, self.right]

    @property
    def output_schema(self) -> Schema:
        return self.schema

    @property
    def is_pipeline_breaker(self) -> bool:
        return True

    def _needed_columns(self) -> set[str]:
        """Every source column the group keys and aggregates reference."""
        from pandas.lazy.optimize.utils import get_referenced_columns

        needed: set[str] = set()
        for e in self.agg.group_by:
            needed |= get_referenced_columns(e)
        for e in self.agg.agg_exprs:
            needed |= get_referenced_columns(e)
        return needed

    def _build_narrow(
        self,
        left_arrays: ArrayDict,
        right_arrays: ArrayDict,
    ) -> ArrayDict | None:
        """Join indices + gather of only the referenced columns, as Arrow."""
        left_cols = {n: a for n, a in left_arrays.items() if not is_index_col(n)}
        right_cols = {n: a for n, a in right_arrays.items() if not is_index_col(n)}

        resolved = self.join._join_key_arrays_i64(left_cols.get, right_cols.get)
        if resolved is None:
            return None
        lk64, rk64, key_pairs = resolved
        shared = {rn for ln, rn in key_pairs if ln == rn}
        if (set(left_cols) & set(right_cols)) - shared:
            return None  # overlapping payload names would need suffixing

        needed = self._needed_columns()
        if any(c not in left_cols and c not in right_cols for c in needed):
            return None

        from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

        if len(rk64) <= 4 * len(lk64):
            res = inner_join_indexers_i8(lk64, rk64, max_hit_fraction=None)
            if res is None:
                return None
            left_idx, right_idx = res
        else:
            # Build on the (smaller) left, probe with the right — no reorder
            # pass, the downstream group is order-insensitive.
            res = inner_join_indexers_i8(rk64, lk64, max_hit_fraction=None)
            if res is None:
                return None
            right_idx, left_idx = res

        left_idx_pa = pa.array(left_idx)
        right_idx_pa = pa.array(right_idx)
        out: ArrayDict = {}
        for c in needed:
            if c in left_cols:
                src, idx_pa = left_cols[c], left_idx_pa
            else:
                src, idx_pa = right_cols[c], right_idx_pa
            src_pa = (
                src if isinstance(src, (pa.Array, pa.ChunkedArray)) else pa.array(src)
            )
            out[c] = pc.take(src_pa, idx_pa)
        return out

    def execute(self, context: ExecutionContext) -> ArrayDict:
        if not _FUSE_JOIN_AGG:
            return self.fallback.execute(context)
        left_arrays = self.left.execute(context)
        right_arrays = self.right.execute(context)
        narrow = self._build_narrow(left_arrays, right_arrays)
        if narrow is None:
            return self.fallback.execute(context)
        # Re-run the original aggregate over the narrow gathered input so its
        # FULL logic (computed aggregates, backend choice, cardinality gate /
        # parallel kernel, index/result formatting) is reused unchanged — the
        # result is identical to grouping a materialized join, the only
        # difference is that the discarded payload columns were never gathered.
        from pandas.lazy.engine.pipeline import PrecomputedInput

        bound = dataclasses.replace(
            self.agg, input=PrecomputedInput(narrow, schema=None)
        )
        return bound.execute(context)


@dataclass
class PhysicalCachedSubplan(PhysicalPlan):
    """Execute-once wrapper for a subplan shared by multiple consumers.

    LazyFrame reuse makes the logical plan a DAG (the same Filter/Join
    subtree object feeding several branches, e.g. TPC-H q21's ``late``
    filter or q2's 4-table ``base`` join), but pipelines executed each
    consumer's copy independently. The planner wraps each shared non-source
    subtree in ONE of these (memoized by logical node identity); whichever
    consumer pipeline reaches it first computes, the rest reuse the cached
    arrays via the context-shared ``subplan_cache``. Childless on purpose:
    the pipeline compiler treats it as a source, and the inner subtree
    executes through the ordinary direct ``execute`` path.
    """

    inner: PhysicalPlan
    key: int
    schema: Schema

    def children(self) -> list[PhysicalPlan]:
        return []

    @property
    def output_schema(self) -> Schema:
        return self.schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        with context.subplan_lock:
            if self.key not in context.subplan_cache:
                # Execute the inner subtree through the full pipeline engine
                # (not direct recursion) so it keeps morsel parallelism and
                # the decision layer — direct execution cost q15 3.5x.
                from pandas.lazy.engine.pipeline import execute_as_pipelines

                context.subplan_cache[self.key] = execute_as_pipelines(
                    self.inner, context
                )
            return context.subplan_cache[self.key]

    def __repr__(self) -> str:
        return f"PhysicalCachedSubplan({type(self.inner).__name__})"


@dataclass
class PhysicalJoinChain(PhysicalPlan):
    """Late-materialization chain of inner joins (one breaker, N base inputs).

    The planner collapses ``Join(Join(Join(A,B),C),D)`` trees of statically
    eligible joins (inner, single-key equi) into one of these so the pipeline
    executor feeds it the BASE relations instead of pre-materializing each
    intermediate join: the cascade of intermediate payload gathers is what
    dominates multi-join pipelines (measured 2.2x on TPC-H q7's chain).

    Execution composes the Cython kernel's indexers step by step
    (``PhysicalHashJoin._compose_step``), gathering only each step's probe
    key column, and gathers every payload column exactly once at the end —
    in pd.merge's cascade order by construction. Any step that is ineligible
    at runtime materializes the running chain and joins through the original
    node's ``_join_arrays`` (full strategy ladder), so semantics are always
    exactly those of the original nested joins. Spill / index-preserving
    contexts skip composition entirely and run the cascade.
    """

    bases: tuple[PhysicalPlan, ...]  # base inputs, left-most first
    steps: tuple[PhysicalHashJoin, ...]  # original nodes, bottom-up
    schema: Schema
    # Set by the decision layer when this chain feeds an order-insensitive
    # sink (groupby/sort/topk/distinct): composition then keeps the kernel's
    # natural probe-major order, skipping the stable-argsort restoration of
    # pd.merge's cascade order AND making the final gather sequential-ish on
    # the big base - the difference between losing and winning on full-hit
    # chains (measured: q18 1.83x loss order-preserving -> win order-free).
    order_free: bool = False

    def children(self) -> list[PhysicalPlan]:
        return list(self.bases)

    @property
    def output_schema(self) -> Schema:
        return self.schema

    @property
    def is_pipeline_breaker(self) -> bool:
        return True

    def execute(self, context: ExecutionContext) -> ArrayDict:
        from concurrent.futures import ThreadPoolExecutor

        # Materialize all base relations (in parallel — they are independent;
        # mirrors _execute_sides_parallel, left-most context wins for index
        # metadata like nested joins did).
        contexts = [context.clone_for_subplan() for _ in self.bases]
        with ThreadPoolExecutor(max_workers=min(4, len(self.bases))) as ex:
            base_arrays = list(
                ex.map(
                    lambda bc: bc[0].execute(bc[1]),
                    zip(self.bases, contexts, strict=True),
                )
            )
        context.index_names = contexts[0].index_names
        context.index_is_multi = contexts[0].index_is_multi

        compose_ok = (
            not context.spill_enabled
            and not context.preserve_index
            and not context.user_set_index
        )
        acc: ArrayDict | None = None
        comp: _CompositeJoin | None = (
            _CompositeJoin.from_arrays(base_arrays[0]) if compose_ok else None
        )
        if not compose_ok:
            acc = base_arrays[0]

        for step, right in zip(self.steps, base_arrays[1:], strict=True):
            if comp is not None:
                right_data = {n: a for n, a in right.items() if not is_index_col(n)}
                extended = step._compose_step(
                    comp, right_data, preserve_order=not self.order_free
                )
                if extended is not None:
                    comp = extended
                    continue
                # Runtime-ineligible step: materialize and continue eagerly.
                acc = comp.gather()
                comp = None
            acc = step._join_arrays(context, acc, right)

        if comp is not None:
            return self.steps[-1]._reorder_columns(comp.gather())
        return acc

    def __repr__(self) -> str:
        return f"PhysicalJoinChain({len(self.bases)} bases)"


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
        # Common-subplan detection: LazyFrame reuse makes the logical plan a
        # DAG (preserved through optimization by PlanVisitor's visit memo);
        # shared non-source subtrees can be planned once and wrapped in an
        # execute-once PhysicalCachedSubplan. DEFAULT-OFF pending diagnosis:
        # the wrapper wins modestly where expected (q21 0.96x) but makes
        # q15's *surrounding* graph ~3.5x slower through a mechanism not yet
        # understood (probed and ruled out: the aggregate itself, the order
        # contract, morsel-parallelism loss — inner runs through the full
        # pipeline engine). See ROADMAP "common-subplan caching".
        self._shared_ids = (
            self._find_shared_subplans(logical_plan)
            if _SUBPLAN_CACHE_ENABLED
            else set()
        )
        self._shared_wrappers: dict[int, PhysicalCachedSubplan] = {}

        physical_plan = self._plan_recursive(logical_plan)

        if enable_fusion:
            physical_plan = self._apply_fusion(physical_plan)

        physical_plan = self._collapse_join_chains(physical_plan)
        physical_plan = self._fuse_filter_aggregates(physical_plan)
        physical_plan = self._fuse_join_aggregates(physical_plan)

        # PhysicalCachedSubplan is childless (the pipeline compiler must
        # treat it as a source), so the tree-walking post-passes above never
        # see its inner subtree — without this, a shared subtree loses
        # operator fusion and chain collapse (measured: q15's shared filter
        # ran unfused at 663 ms vs 44 ms fused). Apply them to each wrapper's
        # inner explicitly; nested shared wrappers are in the dict too.
        for wrapper in getattr(self, "_shared_wrappers", {}).values():
            inner = wrapper.inner
            if enable_fusion:
                inner = self._apply_fusion(inner)
            inner = self._collapse_join_chains(inner)
            inner = self._fuse_filter_aggregates(inner)
            wrapper.inner = self._fuse_join_aggregates(inner)

        return physical_plan

    @staticmethod
    def _find_shared_subplans(plan: LogicalPlan) -> set[int]:
        """ids of non-source logical nodes referenced by 2+ parents."""
        from pandas.lazy.plan import (
            CSVSource,
            DataFrameSource,
            ParquetSource,
        )

        counts: dict[int, int] = {}

        def walk(node) -> None:
            nid = id(node)
            counts[nid] = counts.get(nid, 0) + 1
            if counts[nid] == 1:  # recurse each subtree once (DAG-safe)
                for child in node.children():
                    walk(child)

        walk(plan)
        shared: set[int] = set()
        seen: set[int] = set()

        def collect(node) -> None:
            nid = id(node)
            if nid in seen:
                return
            seen.add(nid)
            if counts.get(nid, 0) >= 2 and not isinstance(
                node,
                (DataFrameSource, ParquetSource, CSVSource),
            ):
                # Aggregates were once excluded here (q15 anomaly hunt — the
                # real cause was the childless wrapper hiding its inner from
                # the fusion post-passes, since fixed). The exclusion later
                # turned HARMFUL: in scan mode q15's shared root optimizes to
                # the Aggregate itself, sharing silently failed, `rev` was
                # computed twice with ULP-different float sums, and the
                # total_revenue == mx equality returned EMPTY at SF-300.
                shared.add(nid)
            for child in node.children():
                collect(child)

        collect(plan)
        return shared

    def _fuse_filter_aggregates(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Collapse ungrouped scan->filter/project->aggregate subtrees onto
        the fused single-pass Cython kernel (PhysicalFusedFilterAgg).

        Conservative: every predicate conjunct must be a numeric/datetime
        range over a source column and every aggregate must resolve
        (through the fused projections) to sum/sum-of-product/count/min/
        max/mean of source columns; anything else leaves the plan
        untouched. The original subtree rides along as runtime fallback.
        """
        children = plan.children()
        if children:
            new_children = [self._fuse_filter_aggregates(c) for c in children]
            if any(n is not o for n, o in zip(new_children, children, strict=True)):
                if isinstance(plan, PhysicalHashJoin):
                    plan = dataclasses.replace(
                        plan, left=new_children[0], right=new_children[1]
                    )
                elif isinstance(plan, PhysicalConcat):
                    plan = dataclasses.replace(plan, inputs=tuple(new_children))
                elif isinstance(plan, PhysicalJoinChain):
                    plan = dataclasses.replace(plan, bases=tuple(new_children))
                else:
                    plan = dataclasses.replace(plan, input=new_children[0])

        # Shape A: scan -> fused(filter/project) -> HashAggregate.
        if isinstance(plan, PhysicalHashAggregate):
            node = plan.input
            if isinstance(node, PhysicalMaterialize):
                node = node.input
            if not isinstance(node, PhysicalFusedPipeline):
                return plan
            if not isinstance(node.input, PhysicalScan):
                return plan
            spec = self._translate_fused_agg(node, plan)
            if spec is None:
                return plan
            return PhysicalFusedFilterAgg(
                scan=node.input,
                spec=spec,
                fallback=plan,
                schema=plan.output_schema,
            )

        # Shape B: scan -> fused(filter/project, ..., scalar-agg project).
        # An ungrouped `select(col.sum())` lowers to a reducing project at
        # the tail of the fused pipeline rather than a HashAggregate node, so
        # Shape A misses it and it falls to the row-compacting generic
        # pipeline (measured 5x slower than the kernel — see
        # docs/MATERIALIZATION_EXPERIMENT.md). Peel the terminal aggregate
        # project off, reuse the same translator with a no-group shim.
        if isinstance(plan, PhysicalFusedPipeline) and isinstance(
            plan.input, PhysicalScan
        ):
            ops = plan.operations
            if not ops or ops[-1].op_type != "project":
                return plan
            term_exprs = list(ops[-1].exprs or ())

            def _is_agg_expr(e):
                ir = e._ir.arg if isinstance(e._ir, Alias) else e._ir
                return isinstance(ir, Call) and ir.is_aggregate

            if not term_exprs or not all(_is_agg_expr(e) for e in term_exprs):
                return plan
            prefix = dataclasses.replace(plan, operations=ops[:-1])
            shim = SimpleNamespace(group_by=[], agg_exprs=term_exprs)
            spec = self._translate_fused_agg(prefix, shim)
            if spec is None:
                return plan
            return PhysicalFusedFilterAgg(
                scan=plan.input,
                spec=spec,
                fallback=plan,
                schema=plan.output_schema,
            )

        return plan

    @staticmethod
    def _join_keys_integer(join: PhysicalHashJoin) -> bool:
        """Whether the join's key columns are int (packable by the kernel).

        Static gate so we only build a fused node for joins whose keys the
        Cython indexer can actually pack — string/datetime-keyed joins would
        always fall back at runtime (re-running the sides), a regression.
        """
        if join.on is not None:
            lkeys, rkeys = list(join.on), list(join.on)
        elif join.left_on is not None and join.right_on is not None:
            lkeys, rkeys = list(join.left_on), list(join.right_on)
        else:
            return False
        if len(lkeys) not in (1, 2) or len(lkeys) != len(rkeys):
            return False
        try:
            ls, rs = join.left.output_schema, join.right.output_schema
            for sch, keys in ((ls, lkeys), (rs, rkeys)):
                for k in keys:
                    dt = getattr(sch[k], "numpy_dtype", None)
                    if dt is None or dt.kind not in "iu" or dt == np.dtype("uint64"):
                        return False
        except (KeyError, AttributeError, TypeError):
            return False
        return True

    def _fuse_join_aggregates(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Collapse ``HashAggregate(Materialize?(inner HashJoin))`` onto a
        single PhysicalFusedJoinAgg that gathers only the group/agg columns off
        the join indices, skipping the full-join materialization.

        Static gate: inner join with int keys feeding a non-empty group. The
        aggregate may be anything (re-run unchanged over the narrow input);
        runtime ``_build_narrow`` falls back if a referenced column cannot be
        resolved off one side. See PhysicalFusedJoinAgg /
        docs/BUFFER_JOIN_AGG_PROBE.md.
        """
        if not _FUSE_JOIN_AGG:
            return plan

        children = plan.children()
        if children:
            new_children = [self._fuse_join_aggregates(c) for c in children]
            if any(n is not o for n, o in zip(new_children, children, strict=True)):
                if isinstance(plan, (PhysicalHashJoin, PhysicalFusedJoinAgg)):
                    plan = dataclasses.replace(
                        plan, left=new_children[0], right=new_children[1]
                    )
                elif isinstance(plan, PhysicalConcat):
                    plan = dataclasses.replace(plan, inputs=tuple(new_children))
                elif isinstance(plan, PhysicalJoinChain):
                    plan = dataclasses.replace(plan, bases=tuple(new_children))
                else:
                    plan = dataclasses.replace(plan, input=new_children[0])

        if not isinstance(plan, PhysicalHashAggregate) or not plan.group_by:
            return plan
        node = plan.input
        if isinstance(node, PhysicalMaterialize):
            node = node.input
        if not isinstance(node, PhysicalHashJoin) or node.how != "inner":
            return plan
        if not self._join_keys_integer(node):
            return plan

        return PhysicalFusedJoinAgg(
            left=node.left,
            right=node.right,
            join=node,
            agg=plan,
            fallback=plan,
            schema=plan.output_schema,
        )

    @staticmethod
    def _translate_fused_agg(fused, agg):
        import numpy as _np

        INT_MIN, INT_MAX = -(2**63) + 1, 2**63 - 1

        # Predicate slot (int64 vs float64) follows the COLUMN dtype, not
        # the literal: `l_quantity < 24` is an int literal over a float64
        # column — int-range semantics (hi = 23) would silently change the
        # result, and the runtime dtype check would bail to the fallback.
        col_kinds: dict = {}
        try:
            schema = fused.input.output_schema
            for name in schema.names:
                dt = getattr(schema[name], "numpy_dtype", None)
                if dt is not None:
                    col_kinds[name] = dt.kind
        except Exception:
            pass

        def lit_value(ir):
            if not isinstance(ir, IRLiteral):
                return None
            v = ir.value
            if hasattr(v, "value") and hasattr(v, "tz"):  # Timestamp
                return ("i", int(v.value)) if v.tz is None else None
            if isinstance(v, _np.datetime64):
                return ("i", int(_np.datetime64(v, "ns").view("int64")))
            if isinstance(v, (bool,)):
                return None
            if isinstance(v, (int, _np.integer)):
                return ("n", int(v))
            if isinstance(v, (float, _np.floating)):
                return ("f", float(v))
            return None

        env: dict = {}

        def resolve(ir):
            if isinstance(ir, Alias):
                return resolve(ir.arg)
            if isinstance(ir, FieldRef):
                return env.get(ir.name, ir)
            if isinstance(ir, Call):
                return Call(ir.function, tuple(resolve(a) for a in ir.args))
            return ir

        spec = _FusedAggSpec()
        i64r: dict = {}
        f64r: dict = {}

        def add_range(col_ir, op, val):
            if not isinstance(col_ir, FieldRef):
                return False
            kind, v = val
            name = col_ir.name
            ck = col_kinds.get(name)
            if ck == "f" and kind in ("n", "f"):
                kind = "f"
            elif ck in ("i", "m", "M") and kind == "n":
                kind = "i"
            elif ck is None:
                return False
            if kind == "i" or (kind == "n" and op in ("ge", "gt", "le", "lt", "eq")):
                lo, hi = i64r.get(name, (INT_MIN, INT_MAX))
                iv = int(v)
                if op == "ge":
                    lo = max(lo, iv)
                elif op == "gt":
                    lo = max(lo, iv + 1)
                elif op == "le":
                    hi = min(hi, iv)
                elif op == "lt":
                    hi = min(hi, iv - 1)
                else:
                    lo, hi = max(lo, iv), min(hi, iv)
                i64r[name] = (lo, hi)
                return True
            if kind in ("f", "n"):
                lo, hi = f64r.get(name, (-_np.inf, _np.inf))
                fv = float(v)
                if op == "ge":
                    lo = max(lo, fv)
                elif op == "gt":
                    lo = max(lo, _np.nextafter(fv, _np.inf))
                elif op == "le":
                    hi = min(hi, fv)
                elif op == "lt":
                    hi = min(hi, _np.nextafter(fv, -_np.inf))
                else:
                    lo, hi = max(lo, fv), min(hi, fv)
                f64r[name] = (lo, hi)
                return True
            return False

        CMP = {
            "greater_equal": "ge",
            "greater": "gt",
            "less_equal": "le",
            "less": "lt",
            "equal": "eq",
        }
        FLIP = {"ge": "le", "gt": "lt", "le": "ge", "lt": "gt", "eq": "eq"}

        def conjuncts(ir, out):
            if isinstance(ir, Call) and ir.function == "and_":
                for a in ir.args:
                    conjuncts(a, out)
            else:
                out.append(ir)

        for op in fused.operations:
            if op.op_type == "filter":
                pred = getattr(op, "predicate", None) or getattr(op, "expr", None)
                if pred is None:
                    return None
                parts: list = []
                conjuncts(resolve(pred._ir), parts)
                for c in parts:
                    if not (isinstance(c, Call) and len(c.args) == 2):
                        return None
                    a, b = c.args
                    cmp = CMP.get(c.function)
                    if cmp is None:
                        return None
                    va, vb = lit_value(a), lit_value(b)
                    if vb is not None and not isinstance(a, IRLiteral):
                        if not add_range(a, cmp, vb):
                            return None
                    elif va is not None and not isinstance(b, IRLiteral):
                        if not add_range(b, FLIP[cmp], va):
                            return None
                    else:
                        return None
            elif op.op_type == "project":
                new_env = {}
                for e in op.exprs or ():
                    ir = e._ir
                    name = extract_output_name(e)
                    if isinstance(ir, Alias):
                        ir = ir.arg
                    new_env[name] = resolve(ir)
                env = new_env
            else:
                return None  # limit etc: not fusable

        for g in agg.group_by:
            g_ir = resolve(g._ir if not isinstance(g._ir, Alias) else g._ir.arg)
            if isinstance(g_ir, Alias):
                g_ir = g_ir.arg
            if not isinstance(g_ir, FieldRef):
                return None
            spec.group_cols.append(g_ir.name)

        AGGK = {"sum": 0, "count": 2, "min": 3, "max": 4}
        for e in agg.agg_exprs:
            out_name = extract_output_name(e)
            ir = e._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if not (isinstance(ir, Call) and ir.is_aggregate and ir.args):
                return None
            fn = ir.function
            arg = resolve(ir.args[0])

            def one_minus(x):
                # subtract(1, F) or subtract(F-from-lit ...): match 1 - F
                if (
                    isinstance(x, Call)
                    and x.function == "subtract"
                    and len(x.args) == 2
                    and isinstance(x.args[0], IRLiteral)
                    and x.args[0].value == 1
                    and isinstance(x.args[1], FieldRef)
                ):
                    return x.args[1].name
                return None

            def one_plus(x):
                if isinstance(x, Call) and x.function == "add" and len(x.args) == 2:
                    a0, a1 = x.args
                    if (
                        isinstance(a0, IRLiteral)
                        and a0.value == 1
                        and isinstance(a1, FieldRef)
                    ):
                        return a1.name
                    if (
                        isinstance(a1, IRLiteral)
                        and a1.value == 1
                        and isinstance(a0, FieldRef)
                    ):
                        return a0.name
                return None

            def classify(a):
                """-> (kind, col_a, col_b, col_c) or None."""
                if isinstance(a, FieldRef):
                    return (0, a.name, None, None)
                if not (isinstance(a, Call) and a.function == "multiply"):
                    return None
                if len(a.args) != 2:
                    return None
                x, y = a.args
                if isinstance(x, FieldRef) and isinstance(y, FieldRef):
                    return (1, x.name, y.name, None)
                if isinstance(x, FieldRef) and one_minus(y) is not None:
                    return (5, x.name, one_minus(y), None)
                if isinstance(y, FieldRef) and one_minus(x) is not None:
                    return (5, y.name, one_minus(x), None)
                # (F * (1-F)) * (1+F) in either nesting order
                inner, outer = (x, y) if isinstance(x, Call) else (y, x)
                if (
                    isinstance(inner, Call)
                    and inner.function == "multiply"
                    and len(inner.args) == 2
                ):
                    sub = classify(inner)
                    if sub is not None and sub[0] == 5:
                        cc = one_plus(outer)
                        if cc is not None:
                            return (6, sub[1], sub[2], cc)
                return None

            cls = classify(arg)
            if fn == "count":
                spec.aggs.append((out_name, 2, None, None, None))
            elif fn == "sum":
                if cls is None:
                    return None
                k, ca, cb, cc = cls
                kind = 1 if k == 1 else (k if k in (5, 6) else 0)
                spec.aggs.append((out_name, kind, ca, cb, cc))
            elif fn in ("min", "max"):
                if cls is None or cls[0] != 0:
                    return None
                spec.aggs.append((out_name, AGGK[fn], cls[1], None, None))
            elif fn == "mean":
                if cls is None or cls[0] != 0:
                    return None
                s_slot = len(spec.aggs)
                spec.aggs.append((f"__fused_sum_{out_name}", 0, cls[1], None, None))
                c_slot = len(spec.aggs)
                spec.aggs.append((f"__fused_cnt_{out_name}", 2, None, None, None))
                spec.mean_outs[out_name] = (s_slot, c_slot)
            else:
                return None
        for name, (lo, hi) in i64r.items():
            spec.i64_preds.append((name, lo, hi))
        for name, (lo, hi) in f64r.items():
            spec.f64_preds.append((name, lo, hi))
        if not spec.aggs:
            return None
        if not spec.group_cols and any(a[1] in (5, 6) for a in spec.aggs):
            return None  # product-minus forms only in the grouped kernel
        if spec.group_cols:
            # GROUPED FUSION IS OFF — measured losses across the board
            # (controlled on/off at SF-3): q1 1.05x (11-slot per-row
            # scatter can't auto-vectorize; acero's SIMD grouped agg wins),
            # q15 1.20x and q20 1.26x (full-column fetch copies + group-code
            # derivation cost more than the already-fused morsel-parallel
            # baseline on selective filters). The grouped kernel and
            # translation stay as tested infrastructure; revisiting needs
            # zero-copy column access and/or SIMD scatter. Ungrouped
            # (q6-class) fusion is a clean 0.31x win and stays on.
            return None
        return spec

    def _collapse_join_chains(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Collapse left-deep trees of eligible inner joins into one
        PhysicalJoinChain breaker (late materialization — see that class).

        Statically eligible step: inner, single-key equi. Runtime gates
        (key dtypes, name overlap, spill/index contexts) live in the chain
        node, which degrades per step to the original join semantics.
        """

        def is_step(node) -> bool:
            if not (isinstance(node, PhysicalHashJoin) and node.how == "inner"):
                return False
            if node.on is not None:
                pairs = [(k, k) for k in node.on]
            elif (
                node.left_on is not None
                and node.right_on is not None
                and len(node.left_on) == len(node.right_on)
            ):
                pairs = list(zip(node.left_on, node.right_on, strict=True))
            else:
                return False
            if len(pairs) == 1:
                return True
            if len(pairs) != 2:
                return False
            # Two-key steps collapse only when both keys are integer (they
            # pack into one int64 at runtime). Non-int composite keys keep
            # the nested joins so the decision layer can still route them
            # to acero (chains have no acero fallback inside).
            try:
                ls = node.left.output_schema
                rs = node.right.output_schema
                for ln, rn in pairs:
                    for schema, name in ((ls, ln), (rs, rn)):
                        dt = schema[name]
                        np_dt = getattr(dt, "numpy_dtype", None)
                        if np_dt is None or np_dt.kind not in "iu":
                            return False
            except Exception:
                return False
            return True

        def rewrite(node: PhysicalPlan) -> PhysicalPlan:
            inner = _unwrap_materialize(node)
            if is_step(inner) and is_step(_unwrap_materialize(inner.left)):
                # Collect the maximal left-deep chain bottom-up.
                steps: list[PhysicalHashJoin] = []
                cur = inner
                while is_step(cur):
                    steps.append(cur)
                    nxt = _unwrap_materialize(cur.left)
                    if not is_step(nxt):
                        break
                    cur = nxt
                steps.reverse()  # bottom-up
                bases = [rewrite(steps[0].left)] + [rewrite(s.right) for s in steps]
                chain = PhysicalJoinChain(
                    bases=tuple(bases),
                    steps=tuple(steps),
                    schema=inner.output_schema,
                )
                if isinstance(node, PhysicalMaterialize):
                    return dataclasses.replace(node, input=chain)
                return chain
            children = node.children()
            if not children:
                return node
            new_children = [rewrite(c) for c in children]
            if all(n is o for n, o in zip(new_children, children, strict=True)):
                return node
            if isinstance(node, PhysicalHashJoin):
                return dataclasses.replace(
                    node, left=new_children[0], right=new_children[1]
                )
            if isinstance(node, PhysicalConcat):
                return dataclasses.replace(node, inputs=tuple(new_children))
            return dataclasses.replace(node, input=new_children[0])

        return rewrite(plan)

    def _plan_recursive(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        """Recursively convert logical plan to physical plan."""

        nid = id(logical_plan)
        shared = getattr(self, "_shared_ids", None)
        if shared is not None and nid in shared:
            wrapper = self._shared_wrappers.get(nid)
            if wrapper is None:
                shared.discard(nid)  # avoid recursing into this hook
                inner = self._plan_recursive(logical_plan)
                shared.add(nid)
                wrapper = PhysicalCachedSubplan(
                    inner=inner, key=nid, schema=logical_plan.resolve_schema()
                )
                self._shared_wrappers[nid] = wrapper
            return wrapper

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

        elif isinstance(logical_plan, GroupByHead):
            return self._plan_group_by_head(logical_plan)

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

    def _plan_group_by_head(self, node) -> PhysicalGroupByHead:
        """Plan a GroupByHead (group-wise head; a pipeline breaker)."""
        group_keys = tuple(extract_output_name(e) for e in node.group_by)
        return PhysicalGroupByHead(
            input=self._materialize_for_breaker(node.input, "group_by_head"),
            group_keys=group_keys,
            n=node.n,
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

        # A single FILTER always fuses: PhysicalFusedPipeline is what carries
        # morsel parallelism and prune-before-mask, so a bare PhysicalFilter
        # over a large input is ~15-20x slower than the same filter fused
        # (q15's shared filter: 691 vs 44 ms; q20's filter-on-join-output:
        # 929 ms bare). Single project/limit chains keep the no-benefit
        # exemption — they have no mask/prune work to parallelize.
        if len(operations) == 1 and operations[0].op_type == "filter":
            return PhysicalFusedPipeline(
                input=base_input,
                operations=tuple(operations),
                schema=plan.output_schema,
            )
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
    # Pipeline breakers materialize every output column from a fresh
    # take/aggregate, so the result can be assembled without the
    # block-consolidation copy (no column aliases user data).
    materialized_output = isinstance(
        plan,
        (
            PhysicalSort,
            PhysicalHashAggregate,
            PhysicalDistinct,
            PhysicalTopK,
            PhysicalHashJoin,
        ),
    )
    return arrays_to_dataframe(
        arrays,
        index_names=context.index_names,
        index_is_multi=context.index_is_multi,
        preserve_index=should_reconstruct_index,
        schema=plan.output_schema,
        materialized_output=materialized_output,
    )
