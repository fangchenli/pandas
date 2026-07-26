"""Physical plan base classes: PhysicalPlan, ExecutionContext, PhysicalMaterialize."""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    dataclass,
    field,
)
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np
import pyarrow as pa

from pandas.lazy.backends.spill import (
    SpillConfig,
    SpillManager,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas.lazy.backends.types import ArrayDict
    from pandas.lazy.types import Schema


def get_ordered_columns(batches: list) -> list[str]:
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
        columns = get_ordered_columns(batches)
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
