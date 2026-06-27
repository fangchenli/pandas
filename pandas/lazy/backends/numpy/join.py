"""
NumPy Join kernel implementations.

This module contains join operations (inner, left, right, outer, cross).

Uses vectorized factorize-based composite key creation for performance.
"""

from __future__ import annotations

import numpy as np

from pandas.lazy.backends import (
    dispatch_kernel,
    register_kernel,
)

# Join Operations
# =============================================================================


def _create_composite_key_vectorized(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Create composite key codes from multiple arrays using vectorized factorize.

    This is O(n) per column with no Python loops over rows.

    Parameters
    ----------
    arrays : list[np.ndarray]
        Arrays to combine into composite keys.

    Returns
    -------
    np.ndarray
        Integer array of composite key codes.
    """
    if len(arrays) == 0:
        return np.array([], dtype=np.int64)

    n = len(arrays[0])
    if n == 0:
        return np.array([], dtype=np.int64)

    if len(arrays) == 1:
        # Single key - just factorize it
        codes, _, _ = dispatch_kernel("factorize", "numpy", arrays[0])
        return codes.astype(np.int64)

    # Multiple keys - create composite codes using factorize
    # Similar to groupby: code0 + code1*n0 + code2*n0*n1 + ...
    all_codes = []
    multipliers = []
    current_multiplier = 1

    for arr in arrays:
        codes, uniques, _ = dispatch_kernel("factorize", "numpy", arr)
        all_codes.append(codes)
        multipliers.append(current_multiplier)
        current_multiplier *= len(uniques) + 1  # +1 for potential -1 codes (NA)

    # Create composite codes vectorized
    composite = np.zeros(n, dtype=np.int64)
    for codes, mult in zip(all_codes, multipliers, strict=False):
        # Handle -1 codes (NA values) by shifting to positive
        shifted_codes = (codes + 1).astype(np.int64)
        composite += shifted_codes * mult

    return composite


def _key_for_indexer(arr):
    """
    Normalize a key column for pandas' join-indexer machinery.

    pandas' _factorize_keys handles NumPy arrays and ExtensionArrays;
    raw PyArrow arrays are wrapped zero-copy as ArrowExtensionArray so
    only the *key* columns ever interact with pandas — payload columns
    are gathered in their native format afterwards.
    """
    import pyarrow as pa

    if isinstance(arr, (pa.Array, pa.ChunkedArray)):
        import pandas as pd

        chunked = pa.chunked_array([arr]) if isinstance(arr, pa.Array) else arr
        return pd.arrays.ArrowExtensionArray(chunked)
    return arr


def _get_join_indexers(
    left_keys: list,
    right_keys: list,
    join_type: str,
    *,
    n_left: int,
    n_right: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (left_indices, right_indices) via pandas' Cython hash join.

    -1 in an indexer means "no match" (missing row). pandas may return
    None for an indexer to mean "all rows in original order"; that is
    normalized to an explicit arange here.
    """
    from pandas.core.reshape.merge import get_join_indexers

    left_idx, right_idx = get_join_indexers(
        left_keys, right_keys, sort=False, how=join_type
    )
    if left_idx is None:
        left_idx = np.arange(n_left, dtype=np.intp)
    if right_idx is None:
        right_idx = np.arange(n_right, dtype=np.intp)
    return left_idx, right_idx


@register_kernel("hash_join", "numpy")
def numpy_hash_join(
    left_arrays: dict[str, np.ndarray],
    right_arrays: dict[str, np.ndarray],
    *,
    keys: list[str] | None = None,
    left_keys: list[str] | None = None,
    right_keys: list[str] | None = None,
    join_type: str = "inner",
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> dict[str, np.ndarray]:
    """
    Perform a hash join between two sets of NumPy arrays.

    Uses vectorized factorize-based key matching for performance.

    Parameters
    ----------
    left_arrays : dict[str, np.ndarray]
        Left table as dict of column name to array.
    right_arrays : dict[str, np.ndarray]
        Right table as dict of column name to array.
    keys : list[str] or None
        Column names to join on (used for both tables).
    left_keys : list[str] or None
        Column names from left table to join on.
    right_keys : list[str] or None
        Column names from right table to join on.
    join_type : str, default "inner"
        Type of join: "inner", "left", "right", "outer".
    left_suffix : str, default ""
        Suffix to add to left column names for disambiguation.
    right_suffix : str, default "_right"
        Suffix to add to right column names for disambiguation.

    Returns
    -------
    dict[str, np.ndarray]
        Result of the join operation.
    """
    # Determine keys
    if keys is not None:
        left_key_cols = keys
        right_key_cols = keys
    else:
        left_key_cols = left_keys or []
        right_key_cols = right_keys or []

    # Compute join indexers with pandas' Cython merge machinery — the
    # same code path eager pd.merge uses. This replaced a pure-NumPy
    # factorize/searchsorted implementation that was ~6x slower at
    # 1M x 100K rows, and it makes row-order and NaN-key semantics match
    # the eager evaluator by construction.
    left_indices, right_indices = _get_join_indexers(
        [_key_for_indexer(left_arrays[k]) for k in left_key_cols],
        [_key_for_indexer(right_arrays[k]) for k in right_key_cols],
        join_type,
        n_left=len(next(iter(left_arrays.values()))) if left_arrays else 0,
        n_right=len(next(iter(right_arrays.values()))) if right_arrays else 0,
    )

    # Build result arrays
    result = {}

    # Determine column name handling
    left_cols = set(left_arrays.keys())
    right_cols = set(right_arrays.keys())
    common_cols = left_cols & right_cols - set(left_key_cols)

    # Add left columns
    for col, arr in left_arrays.items():
        out_col = col
        if col in common_cols:
            out_col = col + left_suffix

        if col in left_key_cols:
            # Key column - coalesce from left and right for outer join
            result_arr = _take_key_column_coalesced(
                arr,
                left_indices,
                right_arrays.get(col, None) if col in right_key_cols else None,
                right_indices,
            )
            result[col] = result_arr
        else:
            # Non-key column from left
            result_arr = _take_with_missing(arr, left_indices)
            result[out_col] = result_arr

    # Add right columns (excluding key columns)
    for col, arr in right_arrays.items():
        if col in right_key_cols:
            continue  # Skip key columns (already included from left)

        out_col = col
        if col in common_cols:
            out_col = col + right_suffix

        result_arr = _take_with_missing(arr, right_indices)
        result[out_col] = result_arr

    return result


def _take_key_column_coalesced(
    left_arr: np.ndarray,
    left_indices: np.ndarray,
    right_arr: np.ndarray | None,
    right_indices: np.ndarray,
) -> np.ndarray:
    """
    Take key column values, coalescing from right when left is missing.

    This is vectorized - no Python loops.
    """
    import pyarrow as pa

    # Key columns may be Arrow-backed; the coalescing logic below is
    # NumPy-based, so convert keys (only keys) up front.
    if isinstance(left_arr, (pa.Array, pa.ChunkedArray)):
        left_arr = left_arr.to_numpy(zero_copy_only=False)
    if right_arr is not None and isinstance(right_arr, (pa.Array, pa.ChunkedArray)):
        right_arr = right_arr.to_numpy(zero_copy_only=False)

    n = len(left_indices)
    if n == 0:
        return np.array([], dtype=left_arr.dtype)

    # Start with left values where available
    left_valid = left_indices >= 0
    right_valid = right_indices >= 0

    # Check if we have any missing values
    all_left_valid = left_valid.all()
    if all_left_valid:
        # No missing values - simple take, preserve dtype
        return left_arr[left_indices]

    # Has missing values - need to handle NaN/None
    if np.issubdtype(left_arr.dtype, np.floating):
        result = np.full(n, np.nan, dtype=left_arr.dtype)
    elif np.issubdtype(left_arr.dtype, np.integer):
        # Need float to handle NaN
        result = np.full(n, np.nan, dtype=np.float64)
    else:
        result = np.empty(n, dtype=object)
        result[:] = None

    # Fill from left where valid
    result[left_valid] = left_arr[left_indices[left_valid]]

    # Fill from right where left is missing but right is valid
    if right_arr is not None:
        need_right = ~left_valid & right_valid
        result[need_right] = right_arr[right_indices[need_right]]

    return result


def _take_with_missing(arr, indices: np.ndarray):
    """
    Take elements from array, handling -1 as missing.

    Arrow arrays are gathered natively with pc.take (missing indices
    become nulls) so payload columns never round-trip through NumPy
    object arrays.

    Parameters
    ----------
    arr : np.ndarray, pa.Array, or pa.ChunkedArray
        Source array.
    indices : np.ndarray
        Indices to take (-1 means missing).

    Returns
    -------
    Array of the same backend with NaN/None/null for missing values.
    """
    import pyarrow as pa

    if isinstance(arr, (pa.Array, pa.ChunkedArray)):
        import pyarrow.compute as pc

        # -1 indices become nulls via the mask; Arrow take maps null
        # indices to null output values
        pa_indices = pa.array(indices, mask=indices < 0)
        return pc.take(arr, pa_indices)

    mask = indices >= 0
    has_missing = not mask.all()

    if not has_missing:
        # No missing values - simple take, preserve dtype
        return arr[indices]

    # Has missing values - need to handle NaN/None
    if np.issubdtype(arr.dtype, np.floating):
        result = np.empty(len(indices), dtype=arr.dtype)
        result[mask] = arr[indices[mask]]
        result[~mask] = np.nan
    elif np.issubdtype(arr.dtype, np.integer):
        # Convert to float to handle NaN
        result = np.empty(len(indices), dtype=np.float64)
        result[mask] = arr[indices[mask]]
        result[~mask] = np.nan
    else:
        # Object dtype
        result = np.empty(len(indices), dtype=object)
        result[mask] = arr[indices[mask]]
        result[~mask] = None

    return result


@register_kernel("inner_join", "numpy")
def numpy_inner_join(
    left_arrays: dict[str, np.ndarray],
    right_arrays: dict[str, np.ndarray],
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> dict[str, np.ndarray]:
    """
    Perform an inner join between two sets of NumPy arrays.

    Parameters
    ----------
    left_arrays : dict[str, np.ndarray]
        Left table as dict of column name to array.
    right_arrays : dict[str, np.ndarray]
        Right table as dict of column name to array.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    dict[str, np.ndarray]
        Result of the inner join.
    """
    return numpy_hash_join(
        left_arrays,
        right_arrays,
        keys=keys,
        join_type="inner",
        left_suffix=left_suffix,
        right_suffix=right_suffix,
    )


@register_kernel("left_join", "numpy")
def numpy_left_join(
    left_arrays: dict[str, np.ndarray],
    right_arrays: dict[str, np.ndarray],
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> dict[str, np.ndarray]:
    """
    Perform a left outer join between two sets of NumPy arrays.

    Parameters
    ----------
    left_arrays : dict[str, np.ndarray]
        Left table as dict of column name to array.
    right_arrays : dict[str, np.ndarray]
        Right table as dict of column name to array.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    dict[str, np.ndarray]
        Result of the left join.
    """
    return numpy_hash_join(
        left_arrays,
        right_arrays,
        keys=keys,
        join_type="left",
        left_suffix=left_suffix,
        right_suffix=right_suffix,
    )


@register_kernel("right_join", "numpy")
def numpy_right_join(
    left_arrays: dict[str, np.ndarray],
    right_arrays: dict[str, np.ndarray],
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> dict[str, np.ndarray]:
    """
    Perform a right outer join between two sets of NumPy arrays.

    Parameters
    ----------
    left_arrays : dict[str, np.ndarray]
        Left table as dict of column name to array.
    right_arrays : dict[str, np.ndarray]
        Right table as dict of column name to array.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    dict[str, np.ndarray]
        Result of the right join.
    """
    return numpy_hash_join(
        left_arrays,
        right_arrays,
        keys=keys,
        join_type="right",
        left_suffix=left_suffix,
        right_suffix=right_suffix,
    )


@register_kernel("outer_join", "numpy")
def numpy_outer_join(
    left_arrays: dict[str, np.ndarray],
    right_arrays: dict[str, np.ndarray],
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> dict[str, np.ndarray]:
    """
    Perform a full outer join between two sets of NumPy arrays.

    Parameters
    ----------
    left_arrays : dict[str, np.ndarray]
        Left table as dict of column name to array.
    right_arrays : dict[str, np.ndarray]
        Right table as dict of column name to array.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    dict[str, np.ndarray]
        Result of the full outer join.
    """
    return numpy_hash_join(
        left_arrays,
        right_arrays,
        keys=keys,
        join_type="outer",
        left_suffix=left_suffix,
        right_suffix=right_suffix,
    )


# =============================================================================
# Semi and Anti Joins
# =============================================================================


@register_kernel("ismember", "numpy")
def ismember(
    left_arr: np.ndarray,
    right_arr: np.ndarray,
) -> np.ndarray:
    """
    Return boolean mask indicating which left elements exist in right.

    Uses pandas' Cython hashtable.ismember for O(1) lookup per element.
    Falls back to np.isin for unsupported dtypes.

    Parameters
    ----------
    left_arr : np.ndarray
        Values to check membership for.
    right_arr : np.ndarray
        Values to check membership against.

    Returns
    -------
    np.ndarray[bool]
        Boolean mask of same length as left_arr.
    """
    from pandas._libs import hashtable as htable

    if len(left_arr) == 0:
        return np.array([], dtype=np.bool_)

    if len(right_arr) == 0:
        return np.zeros(len(left_arr), dtype=np.bool_)

    # Ensure contiguous arrays
    left_arr = np.ascontiguousarray(left_arr)
    right_arr = np.ascontiguousarray(right_arr)

    # htable.ismember supports: int8/16/32/64, uint8/16/32/64,
    # float32/64, complex64/128, object
    try:
        return htable.ismember(left_arr, right_arr)
    except TypeError:
        # Fallback for unsupported dtypes (e.g., strings in some cases)
        return np.isin(left_arr, right_arr)


# Keep the old name as an alias for internal use
_ismember_mask = ismember


def _semi_join_single_key(
    left_arr: np.ndarray,
    right_arr: np.ndarray,
) -> np.ndarray:
    """
    Semi join on a single key column using ismember.

    Returns indices of left rows that have a match in right.
    """
    mask = _ismember_mask(left_arr, right_arr)
    return np.where(mask)[0].astype(np.intp)


def _anti_join_single_key(
    left_arr: np.ndarray,
    right_arr: np.ndarray,
) -> np.ndarray:
    """
    Anti join on a single key column using ismember.

    Returns indices of left rows that have NO match in right.
    """
    mask = _ismember_mask(left_arr, right_arr)
    return np.where(~mask)[0].astype(np.intp)


def _semi_join_composite_key(
    left_key_arrays: list[np.ndarray],
    right_key_arrays: list[np.ndarray],
) -> np.ndarray:
    """
    Semi join on composite (multi-column) keys.

    Creates composite key codes via factorize, then uses ismember.
    """
    if not left_key_arrays or not right_key_arrays:
        return np.array([], dtype=np.intp)

    n_left = len(left_key_arrays[0])
    if n_left == 0:
        return np.array([], dtype=np.intp)

    n_right = len(right_key_arrays[0])
    if n_right == 0:
        return np.array([], dtype=np.intp)

    # Concatenate and factorize to get consistent codes
    combined_arrays = []
    for left_arr, right_arr in zip(left_key_arrays, right_key_arrays, strict=False):
        combined_arrays.append(np.concatenate([left_arr, right_arr]))

    combined_codes = _create_composite_key_vectorized(combined_arrays)
    left_codes = combined_codes[:n_left]
    right_codes = combined_codes[n_left:]

    # Use ismember on the codes (int64)
    mask = _ismember_mask(left_codes, right_codes)
    return np.where(mask)[0].astype(np.intp)


def _anti_join_composite_key(
    left_key_arrays: list[np.ndarray],
    right_key_arrays: list[np.ndarray],
) -> np.ndarray:
    """
    Anti join on composite (multi-column) keys.

    Creates composite key codes via factorize, then uses ismember.
    """
    if not left_key_arrays:
        return np.array([], dtype=np.intp)

    n_left = len(left_key_arrays[0])
    if n_left == 0:
        return np.array([], dtype=np.intp)

    if not right_key_arrays or len(right_key_arrays[0]) == 0:
        # No right keys - all left rows don't match
        return np.arange(n_left, dtype=np.intp)

    # Concatenate and factorize to get consistent codes
    combined_arrays = []
    for left_arr, right_arr in zip(left_key_arrays, right_key_arrays, strict=False):
        combined_arrays.append(np.concatenate([left_arr, right_arr]))

    combined_codes = _create_composite_key_vectorized(combined_arrays)
    left_codes = combined_codes[:n_left]
    right_codes = combined_codes[len(left_key_arrays[0]) :]

    # Use ismember on the codes (int64)
    mask = _ismember_mask(left_codes, right_codes)
    return np.where(~mask)[0].astype(np.intp)


@register_kernel("semi_join", "numpy")
def numpy_semi_join(
    left_arrays: dict[str, np.ndarray],
    right_arrays: dict[str, np.ndarray],
    *,
    keys: list[str] | None = None,
    left_keys: list[str] | None = None,
    right_keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Perform a left semi join - return left rows that exist in right.

    Semi join is useful for filtering (EXISTS subquery equivalent).
    Only returns columns from the left table.

    Uses pandas' Cython hashtable for O(n+m) performance.

    Parameters
    ----------
    left_arrays : dict[str, np.ndarray]
        Left table as dict of column name to array.
    right_arrays : dict[str, np.ndarray]
        Right table as dict of column name to array.
    keys : list[str] or None
        Column names to join on (used for both tables).
    left_keys : list[str] or None
        Column names from left table to join on.
    right_keys : list[str] or None
        Column names from right table to join on.

    Returns
    -------
    dict[str, np.ndarray]
        Left rows that have a match in right.
    """
    # Determine keys
    if keys is not None:
        left_key_cols = keys
        right_key_cols = keys
    else:
        left_key_cols = left_keys or []
        right_key_cols = right_keys or []

    # Get key arrays
    left_key_arrays = [left_arrays[k] for k in left_key_cols]
    right_key_arrays = [right_arrays[k] for k in right_key_cols]

    # Choose optimal path based on number of keys
    if len(left_key_cols) == 1:
        # Single key - use direct ismember (fastest path)
        left_indices = _semi_join_single_key(left_key_arrays[0], right_key_arrays[0])
    else:
        # Multiple keys - use composite key approach
        left_indices = _semi_join_composite_key(left_key_arrays, right_key_arrays)

    # Build result - only left columns
    result = {}
    for col, arr in left_arrays.items():
        result[col] = arr[left_indices]

    return result


@register_kernel("anti_join", "numpy")
def numpy_anti_join(
    left_arrays: dict[str, np.ndarray],
    right_arrays: dict[str, np.ndarray],
    *,
    keys: list[str] | None = None,
    left_keys: list[str] | None = None,
    right_keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Perform a left anti join - return left rows that do NOT exist in right.

    Anti join is useful for filtering (NOT EXISTS subquery equivalent).
    Only returns columns from the left table.

    Uses pandas' Cython hashtable for O(n+m) performance.

    Parameters
    ----------
    left_arrays : dict[str, np.ndarray]
        Left table as dict of column name to array.
    right_arrays : dict[str, np.ndarray]
        Right table as dict of column name to array.
    keys : list[str] or None
        Column names to join on (used for both tables).
    left_keys : list[str] or None
        Column names from left table to join on.
    right_keys : list[str] or None
        Column names from right table to join on.

    Returns
    -------
    dict[str, np.ndarray]
        Left rows that have no match in right.
    """
    # Determine keys
    if keys is not None:
        left_key_cols = keys
        right_key_cols = keys
    else:
        left_key_cols = left_keys or []
        right_key_cols = right_keys or []

    # Get key arrays
    left_key_arrays = [left_arrays[k] for k in left_key_cols]
    right_key_arrays = [right_arrays[k] for k in right_key_cols]

    # Choose optimal path based on number of keys
    if len(left_key_cols) == 1:
        # Single key - use direct ismember (fastest path)
        left_indices = _anti_join_single_key(left_key_arrays[0], right_key_arrays[0])
    else:
        # Multiple keys - use composite key approach
        left_indices = _anti_join_composite_key(left_key_arrays, right_key_arrays)

    # Build result - only left columns
    result = {}
    for col, arr in left_arrays.items():
        result[col] = arr[left_indices]

    return result


# =============================================================================


# =============================================================================
# Cython single-pass hash join driver (pandas._libs.lazy_join)
# =============================================================================

# Probe rows below this run single-threaded (thread-pool overhead dominates).
PARALLEL_JOIN_MIN_ROWS = 500_000


def inner_join_indexers_i8(
    left_keys: np.ndarray,
    right_keys: np.ndarray,
    max_hit_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(left_idx, right_idx)`` for an inner join on int64 keys.

    Exactly ``pd.merge``'s inner row order: the Cython kernel builds a
    CSR-grouped hash table on the RIGHT side and probes the LEFT in row
    order, so the output is left-row-major with right matches in right
    order by construction (no reordering pass).

    The probe (count + fill) is embarrassingly parallel: ``nogil`` chunk
    kernels driven by a thread pool above ``PARALLEL_JOIN_MIN_ROWS`` probe
    rows — the same no-OpenMP pattern as the radix sort driver in core.py.
    """
    from pandas._libs.lazy_join import (
        build_join_table_i8,
        probe_count_chunk,
        probe_fill_chunk,
    )

    n_left = len(left_keys)
    if n_left == 0 or len(right_keys) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    slot_key, slot_gid, counts, offsets, group_rows = build_join_table_i8(right_keys)

    if max_hit_fraction is not None and n_left > 200_000:
        # Sampled selectivity pre-screen: the caller only wants this path for
        # *selective* joins (high-hit shapes lose to pd.merge's consolidated
        # block gather on wide intermediates). A strided sample of the probe
        # estimates output/probe cheaply; bail before paying the full probe.
        step = max(1, n_left // 65_536)
        samp = np.ascontiguousarray(left_keys[::step])
        est = probe_count_chunk(samp, 0, len(samp), slot_key, slot_gid, counts) / len(
            samp
        )
        if est > max_hit_fraction:
            return None

    import os

    n_threads = min(8, os.cpu_count() or 1)
    if n_left < PARALLEL_JOIN_MIN_ROWS or n_threads == 1:
        total = probe_count_chunk(left_keys, 0, n_left, slot_key, slot_gid, counts)
        out_left = np.empty(total, dtype=np.int64)
        out_right = np.empty(total, dtype=np.int64)
        probe_fill_chunk(
            left_keys,
            0,
            n_left,
            slot_key,
            slot_gid,
            counts,
            offsets,
            group_rows,
            out_left,
            out_right,
            0,
        )
        return out_left, out_right

    from concurrent.futures import ThreadPoolExecutor

    bounds = np.linspace(0, n_left, n_threads + 1).astype(np.int64)
    spans = [
        (int(bounds[i]), int(bounds[i + 1]))
        for i in range(n_threads)
        if bounds[i] < bounds[i + 1]
    ]
    with ThreadPoolExecutor(len(spans)) as ex:
        chunk_counts = list(
            ex.map(
                lambda s: probe_count_chunk(
                    left_keys, s[0], s[1], slot_key, slot_gid, counts
                ),
                spans,
            )
        )
        starts = np.concatenate(([0], np.cumsum(chunk_counts)))
        out_left = np.empty(int(starts[-1]), dtype=np.int64)
        out_right = np.empty(int(starts[-1]), dtype=np.int64)
        list(
            ex.map(
                lambda sa: probe_fill_chunk(
                    left_keys,
                    sa[0][0],
                    sa[0][1],
                    slot_key,
                    slot_gid,
                    counts,
                    offsets,
                    group_rows,
                    out_left,
                    out_right,
                    sa[1],
                ),
                zip(spans, (int(s) for s in starts[:-1]), strict=False),
            )
        )
    return out_left, out_right


def partitioned_join_gather(
    left_keys: np.ndarray,
    right_keys: np.ndarray,
    right_payload_rm: np.ndarray,
    n_buckets: int = 16,
    preserve_order: bool = True,
):
    """Fused partitioned parallel inner join + row-major payload gather.

    For an inner join on a single ``int64`` key with a **unique build (right)
    side** (TPC-H fact→dim joins): radix-partition both sides by key hash, then
    per partition build a cache-resident hash table on the right keys, probe the
    left keys, and ``memcpy`` the matched right row's payload directly into the
    output (fused — no intermediate index array). Beats Polars on wide payloads
    at SF-3 scale in the unordered mode (see ``JOIN_KERNEL_REBUILD_PROBE.md``).

    ``right_payload_rm`` is the right (dim) payload as a **row-major** block of
    shape ``(n_right, P)`` float64 (so each match is one contiguous copy).
    Returns ``(out_payload_rm, out_left_rows)``: the gathered payload as a
    row-major ``(K, P)`` block and the matched left row indices.

    ``preserve_order=True`` returns rows in exact ``pd.merge`` (left-row) order
    via an O(N) position map (valid under the unique-build assumption); the
    ``False`` fast core returns partition order.

    The unique-build assumption is NOT checked here (the engine gates on it /
    falls back); a non-unique build produces wrong counts.
    """
    from concurrent.futures import ThreadPoolExecutor

    from pandas._libs.lazy_join import (
        join_gather_bucket_rm,
        partition_by_bucket,
    )

    n_left = len(left_keys)
    p_cols = right_payload_rm.shape[1]
    if n_left == 0 or len(right_keys) == 0:
        return np.empty((0, p_cols), dtype=np.float64), np.empty(0, dtype=np.int64)

    lks, lrs, lb = partition_by_bucket(left_keys, n_buckets)
    rks, rrs, rb = partition_by_bucket(right_keys, n_buckets)

    def _bucket(b):
        cap = int(lb[b + 1] - lb[b])
        if cap == 0 or rb[b + 1] == rb[b]:
            return None
        out = np.empty((cap, p_cols), dtype=np.float64)
        out_lrow = np.empty(cap, dtype=np.int64)
        cnt = join_gather_bucket_rm(
            lks,
            lrs,
            int(lb[b]),
            int(lb[b + 1]),
            rks,
            rrs,
            int(rb[b]),
            int(rb[b + 1]),
            right_payload_rm,
            out,
            out_lrow,
        )
        if cnt == 0:
            return None
        return out[:cnt], out_lrow[:cnt]

    if n_left < PARALLEL_JOIN_MIN_ROWS:
        parts = [r for r in map(_bucket, range(n_buckets)) if r is not None]
    else:
        n_workers = min(8, n_buckets)
        with ThreadPoolExecutor(n_workers) as ex:
            parts = [r for r in ex.map(_bucket, range(n_buckets)) if r is not None]

    if not parts:
        return np.empty((0, p_cols), dtype=np.float64), np.empty(0, dtype=np.int64)
    out = np.concatenate([p[0] for p in parts], axis=0)
    out_lrow = np.concatenate([p[1] for p in parts])
    if preserve_order:
        # O(N) stable position map: unique build ⇒ left rows unique in the
        # output, so ordering by left row reproduces pd.merge inner order.
        pos = np.full(n_left, -1, dtype=np.int64)
        pos[out_lrow] = np.arange(len(out_lrow))
        order = pos[pos >= 0]
        out = out[order]
        out_lrow = out_lrow[order]
    return out, out_lrow
