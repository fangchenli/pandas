"""
NumPy Join kernel implementations.

This module contains join operations (inner, left, right, outer, cross).
"""

import numpy as np

from pandas.lazy.backends import register_kernel

# Join Operations
# =============================================================================


def _build_hash_index(keys: np.ndarray) -> dict:
    """
    Build a hash index mapping key values to row indices.

    Parameters
    ----------
    keys : np.ndarray
        Array of keys (can be object array for composite keys).

    Returns
    -------
    dict
        Mapping from key value to list of row indices.
    """
    index = {}
    for i, key in enumerate(keys):
        # Make hashable
        if isinstance(key, np.ndarray):
            key = tuple(key)
        if key not in index:
            index[key] = []
        index[key].append(i)
    return index


def _create_composite_key(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Create a composite key from multiple arrays.

    Parameters
    ----------
    arrays : list[np.ndarray]
        Arrays to combine into composite keys.

    Returns
    -------
    np.ndarray
        Object array of tuples representing composite keys.
    """
    if len(arrays) == 1:
        return arrays[0]

    n = len(arrays[0])
    composite = np.empty(n, dtype=object)
    for i in range(n):
        composite[i] = tuple(arr[i] for arr in arrays)
    return composite


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

    # Create composite keys
    left_key_arrays = [left_arrays[k] for k in left_key_cols]
    right_key_arrays = [right_arrays[k] for k in right_key_cols]

    left_keys_composite = _create_composite_key(left_key_arrays)
    right_keys_composite = _create_composite_key(right_key_arrays)

    # Build hash index on right table
    right_index = _build_hash_index(right_keys_composite)

    # Perform join
    left_indices = []
    right_indices = []

    if join_type in ("inner", "left"):
        # Iterate through left, find matches in right
        for i, key in enumerate(left_keys_composite):
            if isinstance(key, np.ndarray):
                key = tuple(key)
            if key in right_index:
                for j in right_index[key]:
                    left_indices.append(i)
                    right_indices.append(j)
            elif join_type == "left":
                left_indices.append(i)
                right_indices.append(-1)  # No match

    elif join_type == "right":
        # Build left index instead
        left_index = _build_hash_index(left_keys_composite)
        for j, key in enumerate(right_keys_composite):
            if isinstance(key, np.ndarray):
                key = tuple(key)
            if key in left_index:
                for i in left_index[key]:
                    left_indices.append(i)
                    right_indices.append(j)
            else:
                left_indices.append(-1)
                right_indices.append(j)

    elif join_type == "outer":
        # Full outer join
        left_matched = set()
        for i, key in enumerate(left_keys_composite):
            if isinstance(key, np.ndarray):
                key = tuple(key)
            if key in right_index:
                for j in right_index[key]:
                    left_indices.append(i)
                    right_indices.append(j)
                    left_matched.add(i)
            else:
                left_indices.append(i)
                right_indices.append(-1)
                left_matched.add(i)

        # Add unmatched from right
        right_matched = set()
        for j, key in enumerate(right_keys_composite):
            if isinstance(key, np.ndarray):
                key = tuple(key)
            found = False
            for i, lkey in enumerate(left_keys_composite):
                if isinstance(lkey, np.ndarray):
                    lkey = tuple(lkey)
                if lkey == key:
                    right_matched.add(j)
                    found = True
                    break
            if not found and j not in right_matched:
                left_indices.append(-1)
                right_indices.append(j)

    else:
        raise ValueError(f"Unsupported join type: {join_type}")

    left_indices = np.array(left_indices, dtype=np.intp)
    right_indices = np.array(right_indices, dtype=np.intp)

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
            # Key column - take from left (or coalesce for outer)
            result_arr = np.empty(len(left_indices), dtype=arr.dtype)
            indices_iter = zip(left_indices, right_indices, strict=True)
            for idx, (li, ri) in enumerate(indices_iter):
                if li >= 0:
                    result_arr[idx] = arr[li]
                elif ri >= 0 and col in right_key_cols:
                    # Get from right for outer join
                    right_col_idx = (
                        right_key_cols.index(col) if col in right_key_cols else -1
                    )
                    if right_col_idx >= 0:
                        rcol = right_key_cols[right_col_idx]
                        result_arr[idx] = right_arrays[rcol][ri]
                    else:
                        result_arr[idx] = (
                            np.nan if np.issubdtype(arr.dtype, np.floating) else 0
                        )
                else:
                    result_arr[idx] = (
                        np.nan if np.issubdtype(arr.dtype, np.floating) else 0
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
    # Determine output dtype
    if np.issubdtype(arr.dtype, np.floating):
        result = np.empty(len(indices), dtype=arr.dtype)
        mask = indices >= 0
        result[mask] = arr[indices[mask]]
        result[~mask] = np.nan
    elif np.issubdtype(arr.dtype, np.integer):
        # Convert to float to handle NaN
        result = np.empty(len(indices), dtype=np.float64)
        mask = indices >= 0
        result[mask] = arr[indices[mask]]
        result[~mask] = np.nan
    else:
        # Object dtype
        result = np.empty(len(indices), dtype=object)
        mask = indices >= 0
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
