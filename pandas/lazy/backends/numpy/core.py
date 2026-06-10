"""
Core NumPy kernel implementations.

This module contains basic operations:
- Arithmetic: add, subtract, multiply, divide, etc.
- Comparison: equal, not_equal, less, greater, etc.
- Logical: and, or, invert
- Null handling: is_null, is_not_null, fill_null
- Aggregation: sum, mean, min, max, count, std, var
- Filter, sort, take, unique operations
- Cast/type conversion
- If-else / where operations
"""

import numpy as np

from pandas.lazy.backends import register_kernel

# =============================================================================
# Arithmetic Operations
# =============================================================================


@register_kernel("add", "numpy")
def numpy_add(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Add two arrays or array and scalar."""
    return np.add(left, right)


@register_kernel("subtract", "numpy")
def numpy_subtract(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Subtract two arrays or array and scalar."""
    return np.subtract(left, right)


@register_kernel("multiply", "numpy")
def numpy_multiply(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Multiply two arrays or array and scalar."""
    return np.multiply(left, right)


@register_kernel("divide", "numpy")
def numpy_divide(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Divide two arrays or array and scalar."""
    return np.divide(left, right)


@register_kernel("floor_divide", "numpy")
def numpy_floor_divide(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Floor divide two arrays or array and scalar."""
    return np.floor_divide(left, right)


@register_kernel("modulo", "numpy")
def numpy_modulo(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Modulo of two arrays or array and scalar."""
    return np.mod(left, right)


@register_kernel("power", "numpy")
def numpy_power(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Raise array to power."""
    return np.power(left, right)


@register_kernel("negate", "numpy")
def numpy_negate(arr: np.ndarray) -> np.ndarray:
    """Negate array."""
    return np.negative(arr)


@register_kernel("abs", "numpy")
def numpy_abs(arr: np.ndarray) -> np.ndarray:
    """Absolute value."""
    return np.abs(arr)


# =============================================================================
# Comparison Operations
# =============================================================================


@register_kernel("equal", "numpy")
def numpy_equal(left: np.ndarray, right: np.ndarray | float | int | str) -> np.ndarray:
    """Check equality."""
    return np.equal(left, right)


@register_kernel("not_equal", "numpy")
def numpy_not_equal(
    left: np.ndarray, right: np.ndarray | float | int | str
) -> np.ndarray:
    """Check inequality."""
    return np.not_equal(left, right)


@register_kernel("less", "numpy")
def numpy_less(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Check less than."""
    return np.less(left, right)


@register_kernel("less_equal", "numpy")
def numpy_less_equal(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Check less than or equal."""
    return np.less_equal(left, right)


@register_kernel("greater", "numpy")
def numpy_greater(left: np.ndarray, right: np.ndarray | float | int) -> np.ndarray:
    """Check greater than."""
    return np.greater(left, right)


@register_kernel("greater_equal", "numpy")
def numpy_greater_equal(
    left: np.ndarray, right: np.ndarray | float | int
) -> np.ndarray:
    """Check greater than or equal."""
    return np.greater_equal(left, right)


# =============================================================================
# Logical Operations
# =============================================================================


@register_kernel("and_", "numpy")
def numpy_and(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Logical AND."""
    return np.logical_and(left, right)


@register_kernel("or_", "numpy")
def numpy_or(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Logical OR."""
    return np.logical_or(left, right)


@register_kernel("invert", "numpy")
def numpy_invert(arr: np.ndarray) -> np.ndarray:
    """Logical NOT."""
    return np.logical_not(arr)


# =============================================================================
# Null Operations
# =============================================================================


@register_kernel("is_null", "numpy")
def numpy_is_null(arr: np.ndarray) -> np.ndarray:
    """Check for null/NaN values."""
    # Handle both numeric NaN and object None
    if arr.dtype.kind in ("f", "c"):  # float or complex
        return np.isnan(arr)
    elif arr.dtype.kind == "O":  # object
        return np.array(
            [x is None or (isinstance(x, float) and np.isnan(x)) for x in arr],
            dtype=bool,
        )
    else:
        # Integer and other types don't have NaN
        return np.zeros(len(arr), dtype=bool)


@register_kernel("is_not_null", "numpy")
def numpy_is_not_null(arr: np.ndarray) -> np.ndarray:
    """Check for non-null values."""
    return ~numpy_is_null(arr)


@register_kernel("fill_null", "numpy")
def numpy_fill_null(arr: np.ndarray, fill_value) -> np.ndarray:
    """Fill null values with a scalar or array."""
    result = arr.copy()
    mask = numpy_is_null(arr)
    if isinstance(fill_value, np.ndarray):
        result[mask] = fill_value[mask]
    else:
        result[mask] = fill_value
    return result


@register_kernel("isin", "numpy")
def numpy_isin(arr: np.ndarray, *, values: list) -> np.ndarray:
    """
    Check if values are contained in a set of values.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    values : list
        Set of values to check membership against.

    Returns
    -------
    np.ndarray
        Boolean array indicating membership.
    """
    return np.isin(arr, values)


# =============================================================================
# Aggregation Operations
# =============================================================================


@register_kernel("sum", "numpy")
def numpy_sum(arr: np.ndarray):
    """Sum of array elements (ignoring NaN)."""
    return np.nansum(arr)


@register_kernel("mean", "numpy")
def numpy_mean(arr: np.ndarray):
    """Mean of array elements (ignoring NaN)."""
    return np.nanmean(arr)


@register_kernel("min", "numpy")
def numpy_min(arr: np.ndarray):
    """Minimum of array elements (ignoring NaN)."""
    return np.nanmin(arr)


@register_kernel("max", "numpy")
def numpy_max(arr: np.ndarray):
    """Maximum of array elements (ignoring NaN)."""
    return np.nanmax(arr)


@register_kernel("count", "numpy")
def numpy_count(arr: np.ndarray) -> int:
    """Count non-null elements."""
    if arr.dtype.kind in ("f", "c"):
        return int(np.sum(~np.isnan(arr)))
    elif arr.dtype.kind == "O":
        return sum(
            1
            for x in arr
            if x is not None and not (isinstance(x, float) and np.isnan(x))
        )
    else:
        return len(arr)


@register_kernel("std", "numpy")
def numpy_std(arr: np.ndarray, *, ddof: int = 1):
    """Standard deviation (ignoring NaN)."""
    return np.nanstd(arr, ddof=ddof)


@register_kernel("var", "numpy")
def numpy_var(arr: np.ndarray, *, ddof: int = 1):
    """Variance (ignoring NaN)."""
    return np.nanvar(arr, ddof=ddof)


@register_kernel("n_unique", "numpy")
def numpy_n_unique(arr: np.ndarray) -> int:
    """Count unique values."""
    return len(np.unique(arr))


@register_kernel("median", "numpy")
def numpy_median(arr: np.ndarray):
    """Median of array elements (ignoring NaN)."""
    return np.nanmedian(arr)


@register_kernel("any", "numpy")
def numpy_any(arr: np.ndarray) -> bool:
    """Check if any element is True/truthy."""
    return bool(np.any(arr))


@register_kernel("all", "numpy")
def numpy_all(arr: np.ndarray) -> bool:
    """Check if all elements are True/truthy."""
    return bool(np.all(arr))


@register_kernel("quantile", "numpy")
def numpy_quantile(arr: np.ndarray, q: float = 0.5, interpolation: str = "linear"):
    """
    Compute quantile of array elements.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    q : float, default 0.5
        Quantile to compute (0.0 to 1.0).
    interpolation : str, default "linear"
        Interpolation method.

    Returns
    -------
    scalar
        Quantile value.
    """
    return np.nanquantile(arr, q, method=interpolation)


@register_kernel("product", "numpy")
def numpy_product(arr: np.ndarray):
    """Product of array elements (ignoring NaN)."""
    return np.nanprod(arr)


@register_kernel("first", "numpy")
def numpy_first(arr: np.ndarray):
    """Get first non-null value."""
    if len(arr) == 0:
        return None
    # Find first non-NaN for numeric, or first non-None for object
    if arr.dtype == object:
        for val in arr:
            if val is not None:
                return val
    else:
        for val in arr:
            if not np.isnan(val):
                return val
    return None


@register_kernel("last", "numpy")
def numpy_last(arr: np.ndarray):
    """Get last non-null value."""
    if len(arr) == 0:
        return None
    # Find last non-NaN for numeric, or last non-None for object
    if arr.dtype == object:
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] is not None:
                return arr[i]
    else:
        for i in range(len(arr) - 1, -1, -1):
            if not np.isnan(arr[i]):
                return arr[i]
    return None


# =============================================================================
# Filter Operation
# =============================================================================


@register_kernel("filter", "numpy")
def numpy_filter(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Filter array by boolean mask."""
    return arr[mask]


# =============================================================================
# String Operations
# Note: NumPy string operations are generally slower than Arrow.
# The router should prefer Arrow for string ops, but we provide
# these for completeness when data is already NumPy.
# =============================================================================


@register_kernel("str_lower", "numpy")
def numpy_str_lower(arr: np.ndarray) -> np.ndarray:
    """Convert strings to lowercase."""
    return np.char.lower(arr)


@register_kernel("str_upper", "numpy")
def numpy_str_upper(arr: np.ndarray) -> np.ndarray:
    """Convert strings to uppercase."""
    return np.char.upper(arr)


@register_kernel("str_len", "numpy")
def numpy_str_len(arr: np.ndarray) -> np.ndarray:
    """Get string lengths."""
    return np.char.str_len(arr)


@register_kernel("str_strip", "numpy")
def numpy_str_strip(arr: np.ndarray) -> np.ndarray:
    """Strip whitespace from both ends."""
    return np.char.strip(arr)


# =============================================================================
# Sort Operations
# =============================================================================


# Minimum rows before chunked parallel argsort pays for its merge overhead.
# Measured crossover vs serial stable argsort: floats win from ~200K rows,
# ints break even around 500K and win above.
from pandas.lazy.cost import PARALLEL_SORT_MIN_ROWS


def _parallel_argsort(arr: np.ndarray) -> np.ndarray:
    """
    Multi-threaded *stable* argsort: sorted runs + k-way segment merge.

    Phase 1: contiguous chunks stable-argsort concurrently (np.argsort
    releases the GIL). Phase 2: pivot values sampled from the first run
    split every run with ``searchsorted(..., side="right")`` — all ties
    with a pivot land left of the boundary — partitioning the output
    into disjoint value segments; each segment gathers its run slices
    *in run order* and stable-argsorts locally, in parallel.

    Stability: runs are contiguous original-index ranges in order, so
    global stable tie order == (run, within-run position) == position
    in the run-ordered gather; the local stable argsort breaks value
    ties by exactly that position. Equivalent permutation to
    ``np.argsort(arr, kind="stable")``.

    This replaces a cascading pairwise merge (log2(k) full passes over
    all rows). DuckDB's 2025 sort redesign measured the k-way shape at
    ~6.5x vs ~3.5x for cascading two-way at 8 threads; here the segment
    merge is one parallel compare-heavy pass plus one concatenation.
    """
    from concurrent.futures import ThreadPoolExecutor
    import os

    n_chunks = min(8, os.cpu_count() or 4)
    bounds = np.linspace(0, len(arr), n_chunks + 1, dtype=np.int64)

    def sort_chunk(i: int) -> np.ndarray:
        lo, hi = bounds[i], bounds[i + 1]
        return np.argsort(arr[lo:hi], kind="stable") + lo

    with ThreadPoolExecutor(n_chunks) as ex:
        runs = list(ex.map(sort_chunk, range(n_chunks)))

    # Pivots: evenly spaced sample from the first run's sorted values.
    # Approximate splitters are fine — segments need not be equal-sized,
    # only value-disjoint; heavy duplicate keys skew one segment, which
    # costs balance, not correctness.
    n_segments = n_chunks
    first_keys = arr[runs[0]]
    pivot_pos = np.linspace(0, len(first_keys) - 1, n_segments + 1, dtype=np.int64)[
        1:-1
    ]
    pivots = first_keys[pivot_pos]

    # Per-run split positions for each pivot (side="right": pivot-equal
    # values stay left of the boundary in every run).
    splits = np.empty((len(runs), n_segments + 1), dtype=np.int64)
    splits[:, 0] = 0
    for r, run in enumerate(runs):
        splits[r, 1:-1] = np.searchsorted(arr[run], pivots, side="right")
        splits[r, -1] = len(run)

    def merge_segment(j: int) -> np.ndarray:
        parts = [run[splits[r, j] : splits[r, j + 1]] for r, run in enumerate(runs)]
        candidates = np.concatenate(parts)
        if len(candidates) == 0:
            return candidates
        order = np.argsort(arr[candidates], kind="stable")
        return candidates[order]

    with ThreadPoolExecutor(n_segments) as ex:
        segments = list(ex.map(merge_segment, range(n_segments)))

    return np.concatenate(segments)


_SIGN64 = np.uint64(0x8000000000000000)

# Below this the radix runs single-threaded. Measured crossover: the
# parallel path already wins ~1.2x at 500K and ~1.7x at 1M; 1M is the
# robust threshold (the 500K margin is thin enough to invert under load).
RADIX_PARALLEL_MIN_ROWS = 1_000_000
_RADIX_BITS = 11
_RADIX_BUCKETS = 1 << _RADIX_BITS
_RADIX_MASK = np.uint64(_RADIX_BUCKETS - 1)
_RADIX_PASSES = (64 + _RADIX_BITS - 1) // _RADIX_BITS


def _radix_sort_parallel(keys: np.ndarray, n_threads: int) -> np.ndarray:
    """Thread-parallel stable LSD radix argsort over ``uint64`` keys.

    pandas avoids OpenMP, so the parallelism is driven here: the input is
    split into ``n_threads`` contiguous chunks and each pass runs in three
    steps — a parallel per-chunk histogram, a serial combine that turns the
    local histograms into each chunk's per-digit write offsets, and a
    parallel scatter where every chunk writes to its own disjoint slice of
    each bucket. Chunks are ordered contiguous ranges, so the result is the
    same stable permutation the serial kernel produces.

    The nogil phase kernels live in ``pandas._libs.lazy_radix``; the GIL is
    released for their hot loops, so the worker threads run concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor

    from pandas._libs.lazy_radix import (
        histogram_chunk,
        scatter_chunk,
    )

    n = len(keys)
    p = n_threads
    bounds = [(n * c) // p for c in range(p + 1)]
    keys = keys.copy()  # consumed: each pass scatters keys into the buffer
    idx = np.arange(n, dtype=np.int64)
    kb = np.empty(n, dtype=np.uint64)
    ib = np.empty(n, dtype=np.int64)

    with ThreadPoolExecutor(p) as ex:
        for chunk_pass in range(_RADIX_PASSES):
            shift = chunk_pass * _RADIX_BITS
            local_hist = np.zeros((p, _RADIX_BUCKETS), dtype=np.int64)

            list(
                ex.map(
                    lambda c, sh=shift: histogram_chunk(
                        keys, bounds[c], bounds[c + 1], sh, _RADIX_MASK, local_hist[c]
                    ),
                    range(p),
                )
            )

            # Combine: offset[c, b] = (start of bucket b) + (count of b in
            # chunks before c). Bucket starts are the exclusive prefix sum of
            # per-bucket totals; the per-chunk part is the exclusive prefix
            # sum down the chunk axis.
            bucket_base = np.zeros(_RADIX_BUCKETS, dtype=np.int64)
            np.cumsum(local_hist.sum(axis=0)[:-1], out=bucket_base[1:])
            thread_prefix = np.zeros((p, _RADIX_BUCKETS), dtype=np.int64)
            np.cumsum(local_hist[:-1], axis=0, out=thread_prefix[1:])
            offsets = np.ascontiguousarray(bucket_base[None, :] + thread_prefix)

            list(
                ex.map(
                    lambda c, sh=shift: scatter_chunk(
                        keys,
                        idx,
                        bounds[c],
                        bounds[c + 1],
                        sh,
                        _RADIX_MASK,
                        offsets[c],
                        kb,
                        ib,
                    ),
                    range(p),
                )
            )

            keys, kb = kb, keys
            idx, ib = ib, idx

    return idx


def _radix_argsort(arr: np.ndarray) -> np.ndarray | None:
    """Stable ascending argsort via the Cython LSD radix kernel.

    Builds order-preserving ``uint64`` keys (vectorized) and defers the
    radix passes to ``pandas._libs.lazy_radix``. Order-preservation:

    - unsigned ints widen to ``uint64`` unchanged;
    - signed ints widen to ``int64`` and flip the sign bit;
    - floats widen to ``float64``; a set sign bit (negative) flips all
      bits, an unset one flips just the sign bit, giving a monotonic
      ``uint64`` image of IEEE-754 order. NaN never reaches here — the
      caller (`numpy_sort_indices`) replaces it with +/-inf first.

    Returns ``None`` for dtypes the kernel does not cover, so the caller
    falls back to the NumPy k-way merge.
    """
    kind = arr.dtype.kind
    if kind == "f":
        u = np.ascontiguousarray(arr, dtype=np.float64).view(np.uint64)
        # -0.0's bit pattern is exactly the sign bit (_SIGN64). It equals
        # +0.0, so it must not sort before it; excluding it from the
        # negative branch maps its key onto +0.0's, keeping the tie stable.
        neg = ((u >> np.uint64(63)) != 0) & (u != _SIGN64)
        keys = np.where(neg, ~u, u | _SIGN64).astype(np.uint64, copy=False)
    elif kind == "i":
        u = np.ascontiguousarray(arr, dtype=np.int64).view(np.uint64)
        keys = u ^ _SIGN64
    elif kind == "u":
        keys = np.ascontiguousarray(arr, dtype=np.uint64)
    else:
        return None

    if len(keys) >= RADIX_PARALLEL_MIN_ROWS:
        import os

        n_threads = min(8, os.cpu_count() or 1)
        if n_threads > 1:
            return _radix_sort_parallel(keys, n_threads)

    from pandas._libs.lazy_radix import radix_argsort_u64

    return radix_argsort_u64(keys)


def _argsort(arr: np.ndarray) -> np.ndarray:
    """Stable argsort, accelerated for large numeric arrays.

    Stability is part of the lazy sort contract: it is the only tie
    order on which the eager evaluator, the NumPy engine, and the Arrow
    engine (whose sorts are stable) can all agree.

    Large integer/unsigned/float arrays go through the Cython LSD radix
    argsort (`_radix_argsort`): thread-parallel above
    ``RADIX_PARALLEL_MIN_ROWS`` (~130 ms at 10M float64, beating Polars),
    serial below. The parallel k-way merge is the fallback if the kernel
    is unavailable or raises, and ``np.argsort`` handles small or
    non-numeric arrays.
    """
    if len(arr) >= PARALLEL_SORT_MIN_ROWS and arr.dtype.kind in ("i", "u", "f"):
        try:
            result = _radix_argsort(arr)
        except Exception:
            result = None
        if result is not None:
            return result
        return _parallel_argsort(arr)
    return np.argsort(arr, kind="stable")


@register_kernel("sort_indices", "numpy")
def numpy_sort_indices(
    arr: np.ndarray,
    *,
    descending: bool = False,
    null_placement: str = "at_end",
) -> np.ndarray:
    """
    Return indices that would sort the array.

    Parameters
    ----------
    arr : np.ndarray
        Array to sort.
    descending : bool, default False
        Sort in descending order.
    null_placement : str, default "at_end"
        Where to place nulls: "at_start" or "at_end".

    Returns
    -------
    np.ndarray
        Indices that would sort the array.
    """
    # Handle NaN placement for float arrays
    if arr.dtype.kind in ("f", "c"):
        mask = np.isnan(arr)
        has_nan = np.any(mask)
    else:
        has_nan = False

    if has_nan:
        # Create a copy for sorting with NaN replaced
        sort_arr = arr.copy()
        if null_placement == "at_end":
            fill_val = np.inf if not descending else -np.inf
        else:  # at_start
            fill_val = -np.inf if not descending else np.inf
        sort_arr[mask] = fill_val
    else:
        sort_arr = arr

    if descending:
        # Stable descending: a plain indices[::-1] would reverse the
        # order of tied rows. Stable-argsort the reversed view and map
        # positions back; ties keep their original relative order.
        n = len(sort_arr)
        indices = (n - 1 - _argsort(sort_arr[::-1]))[::-1]
    else:
        indices = _argsort(sort_arr)

    return indices


@register_kernel("array_sort_indices", "numpy")
def numpy_array_sort_indices(
    arr: np.ndarray,
    *,
    order: str = "ascending",
    null_placement: str = "at_end",
) -> np.ndarray:
    """
    Return indices that would sort the array (simpler API).

    Parameters
    ----------
    arr : np.ndarray
        Array to sort.
    order : str, default "ascending"
        Sort order: "ascending" or "descending".
    null_placement : str, default "at_end"
        Where to place nulls: "at_start" or "at_end".

    Returns
    -------
    np.ndarray
        Indices that would sort the array.
    """
    return numpy_sort_indices(
        arr,
        descending=(order == "descending"),
        null_placement=null_placement,
    )


def radix_lexsort(keys: list[np.ndarray], descending: tuple[bool, ...]) -> np.ndarray:
    """Stable multi-key argsort by composing per-key radix sorts.

    Lexicographic order results from stably sorting by each key from least-
    to most-significant: the final, most-significant sort groups by the
    primary key while every earlier sort survives as the tie-breaker. Each
    per-key sort goes through ``numpy_sort_indices`` (NaN placement,
    descending, and the parallel radix kernel for large numeric keys), so
    per-key direction is handled natively — no lexsort negate trick.

    ``keys`` are primary-first (``keys[0]`` is the primary sort key). At 10M
    rows / 2 numeric keys this is ~6x faster than Arrow's table
    ``sort_indices`` (382 ms vs 2200 ms) and beats NumPy's ``lexsort`` by
    ~26x. Caller guarantees every key is a numeric NumPy array.

    The least-significant-key-first stable composition is the textbook
    radix approach to multi-field keys (Knuth, TAOCP Vol. 3, §5.2.5) and is
    exactly the contract of ``numpy.lexsort`` (last key primary, stable) —
    https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html —
    which this matches while sourcing each per-key sort from the parallel
    radix kernel instead of NumPy's timsort.
    """
    perm: np.ndarray | None = None
    for i in range(len(keys) - 1, -1, -1):
        key_in_order = keys[i] if perm is None else keys[i][perm]
        sub = numpy_sort_indices(key_in_order, descending=descending[i])
        perm = sub if perm is None else perm[sub]
    assert perm is not None  # keys is non-empty for a multi-key sort
    return perm


# =============================================================================
# Take/Gather Operations
# =============================================================================


@register_kernel("take", "numpy")
def numpy_take(
    arr: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Select elements from array by indices.

    Parameters
    ----------
    arr : np.ndarray
        Source array.
    indices : np.ndarray
        Indices to select.

    Returns
    -------
    np.ndarray
        Elements at the specified indices.
    """
    return np.take(arr, indices)


# =============================================================================
# Unique/Distinct Operations
# =============================================================================


@register_kernel("unique", "numpy")
def numpy_unique(arr: np.ndarray) -> np.ndarray:
    """
    Return unique values in the array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Unique values (sorted).
    """
    return np.unique(arr)


@register_kernel("unique_indices", "numpy")
def numpy_unique_indices(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return unique values and their first occurrence indices.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    tuple of np.ndarray
        (unique_values, indices of first occurrence, ascending)
    """
    # Hash-based first-occurrence detection (pandas duplicated) instead of
    # np.unique(return_index=True): return_index forces a stable argsort,
    # which is what made Distinct scale superlinearly (TPC-H q22 at SF-3:
    # 5.4 s -> 0.3 s on the 13.5M-row probe, 17x). Indices come out already
    # ascending (= first-occurrence order).
    import pandas as pd

    mask = pd.Series(arr).duplicated().to_numpy()
    indices = np.flatnonzero(~mask)
    return arr[indices], indices


# =============================================================================
# Set Membership Operations
# =============================================================================


@register_kernel("is_in", "numpy")
def numpy_is_in(
    arr: np.ndarray,
    value_set: np.ndarray,
) -> np.ndarray:
    """
    Check if elements are in a set of values.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    value_set : np.ndarray
        Set of values to check membership against.

    Returns
    -------
    np.ndarray
        Boolean array indicating membership.
    """
    return np.isin(arr, value_set)


# =============================================================================
# Ranking Operations (for TopK)
# =============================================================================


@register_kernel("rank", "numpy")
def numpy_rank(
    arr: np.ndarray,
    *,
    method: str = "min",
    ascending: bool = True,
) -> np.ndarray:
    """
    Compute numerical rank of each element.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    method : str, default "min"
        How to handle ties: "min", "max", "first", "dense", "average".
    ascending : bool, default True
        Rank in ascending order.

    Returns
    -------
    np.ndarray
        Ranks as float64 array.
    """
    from scipy import stats as scipy_stats

    if not ascending:
        arr = -arr

    if method == "min":
        ranks = scipy_stats.rankdata(arr, method="min")
    elif method == "max":
        ranks = scipy_stats.rankdata(arr, method="max")
    elif method == "first":
        ranks = scipy_stats.rankdata(arr, method="ordinal")
    elif method == "dense":
        ranks = scipy_stats.rankdata(arr, method="dense")
    else:  # average
        ranks = scipy_stats.rankdata(arr, method="average")

    return ranks


@register_kernel("argpartition", "numpy")
def numpy_argpartition(
    arr: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Return indices of k smallest elements (unordered).

    Useful for TopK operations - faster than full sort.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    k : int
        Number of elements.

    Returns
    -------
    np.ndarray
        Indices of k smallest elements (not sorted).
    """
    if k >= len(arr):
        return np.arange(len(arr))
    return np.argpartition(arr, k)[:k]


@register_kernel("select_k_unstable", "numpy")
def numpy_select_k(
    arr: np.ndarray,
    k: int,
    *,
    order: str = "ascending",
) -> np.ndarray:
    """
    Select indices of k smallest or largest elements.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    k : int
        Number of elements to select.
    order : str, default "ascending"
        "ascending" for smallest, "descending" for largest.

    Returns
    -------
    np.ndarray
        Indices of selected elements (sorted by value).
    """
    if k >= len(arr):
        indices = np.argsort(arr)
        if order == "descending":
            indices = indices[::-1]
        return indices

    if order == "ascending":
        # k smallest
        partitioned_indices = np.argpartition(arr, k)[:k]
        # Sort the k indices by their values
        sorted_order = np.argsort(arr[partitioned_indices])
        return partitioned_indices[sorted_order]
    else:
        # k largest
        partitioned_indices = np.argpartition(arr, -k)[-k:]
        # Sort in descending order
        sorted_order = np.argsort(arr[partitioned_indices])[::-1]
        return partitioned_indices[sorted_order]


# =============================================================================
# Cast/Type Conversion Operations
# =============================================================================


@register_kernel("cast", "numpy")
def numpy_cast(
    arr: np.ndarray,
    target_type: np.dtype,
    *,
    safe: bool = True,
) -> np.ndarray:
    """
    Cast array to a different type.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    target_type : np.dtype
        Target NumPy data type.
    safe : bool, default True
        If True, use safe casting mode.

    Returns
    -------
    np.ndarray
        Cast array.
    """
    casting = "safe" if safe else "unsafe"
    return arr.astype(target_type, casting=casting, copy=True)


# =============================================================================
# If-Else / Where Operations
# =============================================================================


@register_kernel("if_else", "numpy")
def numpy_if_else(
    condition: np.ndarray,
    true_value: np.ndarray | float | int,
    false_value: np.ndarray | float | int,
) -> np.ndarray:
    """
    Choose elements based on condition.

    Parameters
    ----------
    condition : np.ndarray
        Boolean condition array.
    true_value : np.ndarray or scalar
        Values to use where condition is True.
    false_value : np.ndarray or scalar
        Values to use where condition is False.

    Returns
    -------
    np.ndarray
        Result array.
    """
    return np.where(condition, true_value, false_value)


@register_kernel("case_when", "numpy")
def numpy_case_when(
    *args,
) -> np.ndarray:
    """
    Case-when expression (multiple conditions).

    Parameters
    ----------
    *args
        Alternating (condition, value) pairs, ending with a default value.
        E.g., (cond1, val1, cond2, val2, default)

    Returns
    -------
    np.ndarray
        Result array.
    """
    if len(args) < 3:
        raise ValueError(
            "case_when requires at least one condition-value pair and a default"
        )

    # Last arg is the default
    default = args[-1]

    # Start with default and work backwards
    if isinstance(default, np.ndarray):
        result = default.copy()
    else:
        # Need to determine size from conditions
        size = len(args[0])
        result = np.full(size, default)

    # Apply conditions in reverse order (later conditions override earlier)
    for i in range(len(args) - 3, -1, -2):
        condition = args[i]
        value = args[i + 1]
        result = np.where(condition, value, result)

    return result


# =============================================================================
