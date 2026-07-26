"""Physical plan — join operators (split from physical.py)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pyarrow as pa

from pandas.lazy.backends import (
    dispatch_kernel,
    has_kernel,
)
from pandas.lazy.backends.convert import (
    arrays_to_dataframe,
    dataframe_to_arrays,
    get_array_backend,
    to_arrow,
)
from pandas.lazy.backends.spill import (
    GraceHashJoiner,
    get_arrays_bytes,
)
from pandas.lazy.backends.types import (
    ArrayDict,
    is_index_col,
)
from pandas.lazy.cost import PARALLEL_TAKE_MIN_ROWS
from pandas.lazy.physical.base import (
    ExecutionContext,
    PhysicalPlan,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.lazy.types import Schema

# Rust (PyO3+rayon) fused index-gen for inner single-int-key joins. The Cython
# join can't beat Polars (no-OpenMP: no real threads / no fused SIMD gather); a
# Rust kernel computes the join indices with real threads, then the engine
# gathers via `take_all_columns`.
# DEFAULT-OFF: this *index-only* wire REGRESSES all-22 at SF-3 (geo-mean
# 0.43x->0.41x; q18 +51%, q4 +44%, q17 +36%) — the Rust index-gen is fast, but
# the engine's per-column `np.take` gather (two passes, post-materialization) is
# slower than pd.merge's consolidated single-pass block take on wide TPC-H joins.
# The Rust *fused* kernel that beats Polars gathers INSIDE Rust (one pass, direct
# output, column-major, no transpose) — that's the win, and capturing it needs a
# fused-both-sides numeric-gather wire (next step), not index-only. Kept as a
# validated foundation (all-22 still validate vs DuckDB with it on). Opt-in:
# `PANDAS_LAZY_RUST_JOIN=1`. See docs/JOIN_KERNEL_REBUILD_PROBE.md.
_RUST_JOIN = os.environ.get("PANDAS_LAZY_RUST_JOIN", "0") == "1"
try:
    import lazyjoin_rs as _rust_join
except ImportError:
    _rust_join = None


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
            per_base.append(cols if idx is None else take_all_columns(cols, idx))
        return {n: per_base[b][n] for n, b in self.col_map.items()}


def take_all_columns(input_arrays: ArrayDict, indices) -> ArrayDict:
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
        # Only inner joins spill: the Grace path's single-sided partitions need
        # null-padding of the missing side, which is not implemented (it would
        # crash / drop columns), so left/right/outer fall through to the
        # in-memory pd.merge below rather than partitioning.
        if (
            context.spill_enabled
            and context.spill_manager is not None
            and self.how == "inner"
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

        # Rust fast path: real-threaded fused index-gen (beats Cython, which the
        # no-OpenMP build caps), then the engine gathers every column in parallel
        # (`take_all_columns`, all dtypes). General inner join, pd.merge order.
        if _RUST_JOIN and _rust_join is not None:
            li = np.ascontiguousarray(lk64)
            ri_keys = np.ascontiguousarray(rk64)
            left_idx, right_idx = _rust_join.join_indices_i64(li, ri_keys)
            out: ArrayDict = {}
            out.update(take_all_columns(left_cols, left_idx))
            right_gather = {n: a for n, a in right_cols.items() if n not in shared}
            out.update(take_all_columns(right_gather, right_idx))
            return out

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
        out.update(take_all_columns(left_cols, left_idx))
        right_gather = {n: a for n, a in right_cols.items() if n not in shared}
        out.update(take_all_columns(right_gather, right_idx))
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
