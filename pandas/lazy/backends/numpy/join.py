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


def _build_right_index_vectorized(
    right_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build index structures for right table using vectorized operations.

    Returns
    -------
    tuple
        (unique_codes, code_starts, code_counts) for O(1) lookup
    """
    if len(right_codes) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    # Sort right codes to group identical keys together
    sort_indices = np.argsort(right_codes)
    sorted_codes = right_codes[sort_indices]

    # Find unique codes and their positions
    unique_mask = np.concatenate([[True], sorted_codes[1:] != sorted_codes[:-1]])
    unique_codes = sorted_codes[unique_mask]
    unique_positions = np.where(unique_mask)[0]

    # Count occurrences of each unique code
    counts = np.diff(np.concatenate([unique_positions, [len(sorted_codes)]]))

    return unique_codes, sort_indices, unique_positions, counts


def _vectorized_inner_join(
    left_codes: np.ndarray,
    right_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform inner join matching using vectorized operations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (left_indices, right_indices) for matched rows
    """
    n_left = len(left_codes)
    n_right = len(right_codes)

    if n_left == 0 or n_right == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    # Build index on right table
    unique_codes, right_sort_idx, unique_pos, counts = _build_right_index_vectorized(
        right_codes
    )

    if len(unique_codes) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    # For each left code, find matching right codes using searchsorted
    # searchsorted gives us the position in unique_codes where left_code would go
    insert_pos = np.searchsorted(unique_codes, left_codes)

    # Check which left codes actually match
    valid_match = (insert_pos < len(unique_codes)) & (
        unique_codes[np.minimum(insert_pos, len(unique_codes) - 1)] == left_codes
    )

    # For matched codes, we need to expand based on counts
    matched_left_idx = np.where(valid_match)[0]
    if len(matched_left_idx) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    matched_insert_pos = insert_pos[valid_match]
    matched_counts = counts[matched_insert_pos]
    matched_starts = unique_pos[matched_insert_pos]

    # Total result size
    total_matches = matched_counts.sum()

    # Build result indices
    left_indices = np.repeat(matched_left_idx, matched_counts)

    # Build right indices - need to expand ranges
    right_indices = np.empty(total_matches, dtype=np.intp)
    pos = 0
    for i, (start, count) in enumerate(
        zip(matched_starts, matched_counts, strict=False)
    ):
        right_indices[pos : pos + count] = right_sort_idx[start : start + count]
        pos += count

    return left_indices, right_indices


def _vectorized_left_join(
    left_codes: np.ndarray,
    right_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform left join matching using vectorized operations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (left_indices, right_indices) where right_indices=-1 means no match
    """
    n_left = len(left_codes)

    if n_left == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    if len(right_codes) == 0:
        # All left rows, no matches
        return np.arange(n_left, dtype=np.intp), np.full(n_left, -1, dtype=np.intp)

    # Build index on right table
    unique_codes, right_sort_idx, unique_pos, counts = _build_right_index_vectorized(
        right_codes
    )

    # Find matches
    insert_pos = np.searchsorted(unique_codes, left_codes)
    valid_match = (insert_pos < len(unique_codes)) & (
        unique_codes[np.minimum(insert_pos, len(unique_codes) - 1)] == left_codes
    )

    # Separate matched and unmatched
    matched_mask = valid_match
    unmatched_mask = ~valid_match

    # For matched: expand based on counts
    matched_left_idx = np.where(matched_mask)[0]
    unmatched_left_idx = np.where(unmatched_mask)[0]

    if len(matched_left_idx) == 0:
        # No matches - all left rows with -1 right
        return np.arange(n_left, dtype=np.intp), np.full(n_left, -1, dtype=np.intp)

    matched_insert_pos = insert_pos[matched_mask]
    matched_counts = counts[matched_insert_pos]
    matched_starts = unique_pos[matched_insert_pos]

    # Build matched indices
    total_matched = matched_counts.sum()
    matched_left = np.repeat(matched_left_idx, matched_counts)
    matched_right = np.empty(total_matched, dtype=np.intp)
    pos = 0
    for start, count in zip(matched_starts, matched_counts, strict=False):
        matched_right[pos : pos + count] = right_sort_idx[start : start + count]
        pos += count

    # Combine matched and unmatched
    n_unmatched = len(unmatched_left_idx)
    total_rows = total_matched + n_unmatched

    left_indices = np.empty(total_rows, dtype=np.intp)
    right_indices = np.empty(total_rows, dtype=np.intp)

    left_indices[:total_matched] = matched_left
    right_indices[:total_matched] = matched_right
    left_indices[total_matched:] = unmatched_left_idx
    right_indices[total_matched:] = -1

    # Sort by left index to maintain order
    sort_order = np.argsort(left_indices)
    return left_indices[sort_order], right_indices[sort_order]


def _vectorized_outer_join(
    left_codes: np.ndarray,
    right_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform full outer join matching using vectorized operations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (left_indices, right_indices) where -1 means no match
    """
    n_left = len(left_codes)
    n_right = len(right_codes)

    if n_left == 0 and n_right == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    if n_left == 0:
        return np.full(n_right, -1, dtype=np.intp), np.arange(n_right, dtype=np.intp)

    if n_right == 0:
        return np.arange(n_left, dtype=np.intp), np.full(n_left, -1, dtype=np.intp)

    # Get left join results
    left_indices, right_indices = _vectorized_left_join(left_codes, right_codes)

    # Find right rows that weren't matched
    matched_right = set(right_indices[right_indices >= 0])
    unmatched_right = np.array(
        [i for i in range(n_right) if i not in matched_right], dtype=np.intp
    )

    if len(unmatched_right) == 0:
        return left_indices, right_indices

    # Add unmatched right rows
    n_unmatched = len(unmatched_right)
    full_left = np.concatenate([left_indices, np.full(n_unmatched, -1, dtype=np.intp)])
    full_right = np.concatenate([right_indices, unmatched_right])

    return full_left, full_right


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

    # Create composite keys using vectorized factorize
    left_key_arrays = [left_arrays[k] for k in left_key_cols]
    right_key_arrays = [right_arrays[k] for k in right_key_cols]

    # For join, we need to factorize both sides together to get consistent codes
    # Concatenate and factorize, then split
    if len(left_key_arrays) > 0 and len(right_key_arrays) > 0:
        n_left = len(left_key_arrays[0]) if left_key_arrays else 0

        # Concatenate corresponding key columns
        combined_arrays = []
        for left_arr, right_arr in zip(left_key_arrays, right_key_arrays, strict=False):
            combined_arrays.append(np.concatenate([left_arr, right_arr]))

        # Create composite codes for combined data
        combined_codes = _create_composite_key_vectorized(combined_arrays)

        # Split back
        left_codes = combined_codes[:n_left]
        right_codes = combined_codes[n_left:]
    else:
        left_codes = _create_composite_key_vectorized(left_key_arrays)
        right_codes = _create_composite_key_vectorized(right_key_arrays)

    # Perform join using vectorized matching
    if join_type == "inner":
        left_indices, right_indices = _vectorized_inner_join(left_codes, right_codes)
    elif join_type == "left":
        left_indices, right_indices = _vectorized_left_join(left_codes, right_codes)
    elif join_type == "right":
        # Right join = swap and do left join
        right_indices, left_indices = _vectorized_left_join(right_codes, left_codes)
    elif join_type == "outer":
        left_indices, right_indices = _vectorized_outer_join(left_codes, right_codes)
    else:
        raise ValueError(f"Unsupported join type: {join_type}")

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


def _take_with_missing(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Take elements from array, handling -1 as missing.

    Parameters
    ----------
    arr : np.ndarray
        Source array.
    indices : np.ndarray
        Indices to take (-1 means missing).

    Returns
    -------
    np.ndarray
        Result array with NaN/None for missing values.
    """
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
