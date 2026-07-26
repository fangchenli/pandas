"""Physical plan — fused operators (split from physical.py)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pandas.lazy.backends.array_eval import ArrayEvaluator
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
from pandas.lazy.physical.join import take_all_columns

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas.lazy.types import Schema

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
        take_all_columns: backend-preserving gathers (pc.take handles
        ChunkedArray directly - the previous per-column loop paid a
        combine_chunks copy and rebuilt the Arrow mask per column) and
        threshold-gated parallel fan-out across columns.
        """

        indices = np.flatnonzero(np.asarray(mask))
        return take_all_columns(arrays, indices)

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
FUSE_JOIN_AGG = False


class FusedAggSpec:
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
        if not FUSE_JOIN_AGG:
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
