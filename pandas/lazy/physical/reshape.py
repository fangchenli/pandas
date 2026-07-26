"""Physical plan — reshape operators (split from physical.py)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pyarrow as pa

from pandas.lazy.backends.convert import ensure_backend
from pandas.lazy.backends.types import (
    INDEX_COL_NAME,
    ArrayDict,
    index_col_name,
    is_index_col,
)
from pandas.lazy.physical.base import (
    ExecutionContext,
    PhysicalPlan,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas.lazy.types import Schema


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
                if len({c.type for c in chunks}) > 1:
                    # Mixed Arrow types across inputs (e.g. int32 + int64):
                    # chunked_array rejects them. Let Arrow promote to a common
                    # type via a permissive table concat instead of crashing.
                    tbl = pa.concat_tables(
                        [pa.table({name: c}) for c in chunks],
                        promote_options="permissive",
                    )
                    result[name] = tbl.column(name)
                else:
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
