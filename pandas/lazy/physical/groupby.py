"""Physical plan — groupby operators (split from physical.py)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pandas.lazy.backends import (
    dispatch_kernel,
    has_kernel,
)
from pandas.lazy.backends.convert import (
    ensure_backend,
    get_array_backend,
)
from pandas.lazy.backends.types import (
    INDEX_COL_NAME,
    ArrayDict,
    index_col_name,
    is_index_col,
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
from pandas.lazy.physical.base import (
    ExecutionContext,
    PhysicalPlan,
)

if TYPE_CHECKING:
    from pandas.lazy.types import Schema


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
# Direct-address group-by (numpy-native): for a single BOUNDED int key,
# sum/count/mean over high-card groups, accumulate straight into a dense array
# indexed by (key - kmin) via np.bincount — one C pass per column, no hash, no
# sort, no permutation. Wired at the NUMPY level (reads the numpy input arrays
# directly, builds only the small arrow output) so it pays NO arrow<->numpy
# round-trip on the 18M-row input — the tax that sank the earlier arrow-table
# wiring (docs/Q18_DECOMP.md). Gated to a bounded dense accumulator and high
# cardinality. DEFAULT-ON: in-context A/B is a clean win — q18 0.70x (1.43x,
# -191 ms at SF-3), all 22 exact, no regressions. Module toggle for A/B.
_DIRECT_ADDRESS_GROUPBY = True
# Per-accumulator span ceiling (span * 8 bytes ~ 1 GB at 128M); falls back at
# larger scale where the dense array would be too big / too sparse.
_DIRECT_ADDRESS_MAX_SPAN = 128_000_000


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


def _dense_int_keys(arr):
    """Return a contiguous int64 numpy view of ``arr`` (numpy or Arrow) if it is
    a null-free integer column, else None. Numpy input is used directly (no
    copy when already int64) — the whole point is to avoid the Arrow round-trip.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind not in "iu":
            return None
        return np.ascontiguousarray(arr.astype(np.int64, copy=False))
    if isinstance(arr, (pa.Array, pa.ChunkedArray)):
        if arr.null_count or not pa.types.is_integer(arr.type):
            return None
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        return np.ascontiguousarray(
            arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        )
    return None


def _dense_float_vals(arr):
    """Return a float64 numpy view of ``arr`` (numpy or Arrow) if it is a
    null-free, NaN-free floating column, else None. Int columns return None:
    Arrow sums int->int64 exactly, but float-weight bincount would lose
    precision above 2**53, so int value columns fall back."""
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind != "f":
            return None
        v = np.ascontiguousarray(arr.astype(np.float64, copy=False))
    elif isinstance(arr, (pa.Array, pa.ChunkedArray)):
        if arr.null_count or not pa.types.is_floating(arr.type):
            return None
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        v = np.ascontiguousarray(arr.to_numpy(zero_copy_only=False).astype(np.float64))
    else:
        return None
    if np.isnan(v).any():  # NaN-skip semantics differ from accumulate-every-row
        return None
    return v


def _col_has_missing(arr) -> bool:
    """True if ``arr`` (numpy or Arrow) has any null OR float NaN. count()/sum()
    skip missing, but the dense accumulator counts/sums EVERY row — so any
    missing in a referenced value column means the fast path would disagree
    (e.g. a LEFT JOIN fills unmatched numeric columns with NaN: q13's
    count(o_orderkey))."""
    if isinstance(arr, np.ndarray):
        return arr.dtype.kind == "f" and bool(np.isnan(arr).any())
    if isinstance(arr, (pa.Array, pa.ChunkedArray)):
        if arr.null_count:
            return True
        if pa.types.is_floating(arr.type):
            return bool(pc.any(pc.is_nan(arr)).as_py())
    return False


def _direct_address_grouped_arrays(input_arrays, group_cols, agg_specs):
    """Numpy-native dense direct-address grouped aggregate for a single bounded
    int key. Reads the numpy input columns directly and returns a result
    ``ArrayDict`` of Arrow columns (matching the arrow group_by output schema),
    or ``None`` to fall back. See ``_DIRECT_ADDRESS_GROUPBY``.

    Gates: one integer key, no nulls; aggs only sum/count/mean; sum/mean value
    columns floating, null- and NaN-free; count column present and null-free;
    key span bounded and dense (``span <= 2*n``); group count high (strided
    sample >= the parallel ratio gate — low-card loses to Arrow's tiny hash).
    """
    if len(group_cols) != 1:
        return None
    if not {f for _, _, f in agg_specs} <= {"sum", "count", "mean"}:
        return None

    keys = _dense_int_keys(input_arrays.get(group_cols[0]))
    if keys is None or len(keys) == 0:
        return None
    n = len(keys)
    kmin = int(keys.min())
    span = int(keys.max()) - kmin + 1
    if span <= 0 or span > _DIRECT_ADDRESS_MAX_SPAN or span > 2 * n:
        return None

    # Cardinality gate on a cheap strided sample: skip low-card (Arrow wins).
    step = max(1, n // 8192)
    samp = keys[::step]
    if len(np.unique(samp)) / len(samp) < _PARALLEL_GROUPBY_MIN_RATIO:
        return None

    fvals: dict[str, np.ndarray] = {}
    for _, in_col, func in agg_specs:
        if func == "count":
            # count = count of non-null/non-NaN values; our dense counts are
            # group sizes, so any missing in the count column disagrees. A LEFT
            # JOIN fills unmatched numeric cols with NaN (q13's count(o_orderkey)),
            # so this must check numpy NaN too, not just Arrow nulls.
            if _col_has_missing(input_arrays.get(in_col)):
                return None
            continue
        if in_col in fvals:
            continue
        v = _dense_float_vals(input_arrays.get(in_col))
        if v is None:
            return None
        fvals[in_col] = v

    idx = keys - kmin
    counts = np.bincount(idx, minlength=span)
    present = counts > 0
    reps = np.nonzero(present)[0]
    sums = {c: np.bincount(idx, weights=v, minlength=span) for c, v in fvals.items()}

    out: ArrayDict = {}
    out[group_cols[0]] = pa.array((reps + kmin).astype(keys.dtype, copy=False))
    for out_name, in_col, func in agg_specs:
        if func == "count":
            out[out_name] = pa.array(counts[present])
        elif func == "sum":
            out[out_name] = pa.array(sums[in_col][present])
        else:  # mean
            out[out_name] = pa.array(sums[in_col][present] / counts[present])
    return out


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

        # Numpy-native direct-address fast path: a single bounded int key with
        # sum/count/mean over high-card groups accumulates via np.bincount on the
        # numpy input columns directly, with no Arrow<->numpy round-trip on the
        # (large) input — only the small grouped output is built as Arrow. Beats
        # the partition-parallel arrow path ~1.5x on q18's inner group; falls
        # through (None) for any unsupported key/agg/range (docs/Q18_DECOMP.md).
        if _DIRECT_ADDRESS_GROUPBY:
            try:
                da = _direct_address_grouped_arrays(input_arrays, group_cols, agg_specs)
            except Exception:
                da = None  # any failure must not change results
            if da is not None:
                gcol = group_cols[0]
                if context.preserve_index:
                    context.index_names = list(group_cols)
                    context.index_is_multi = False
                    return {
                        INDEX_COL_NAME: da[gcol],
                        **{name: arr for name, arr in da.items() if name != gcol},
                    }
                return dict(da)

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

        # acero's hash_sum returns null for a group whose values are all null;
        # pandas sum (min_count=0) returns 0 there. Fill so the two agree.
        for out_name, _, func in agg_specs:
            if func == "sum" and out_name in result:
                result[out_name] = pc.fill_null(result[out_name], 0)

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
