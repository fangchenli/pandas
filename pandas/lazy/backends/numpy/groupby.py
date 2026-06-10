"""
NumPy GroupBy kernel implementations.

This module contains groupby aggregation operations.
"""

from __future__ import annotations

import numpy as np

from pandas.lazy.backends import register_kernel

# Factorization and Grouping Utilities
# =============================================================================


@register_kernel("factorize", "numpy")
def factorize(
    arr: np.ndarray, sort: bool = False
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Factorize a numpy array to integer codes.

    Uses pandas' Cython-based factorize for fast hash table lookups.

    Parameters
    ----------
    arr : np.ndarray
        Input array to factorize.
    sort : bool, default False
        Whether to sort the unique values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        - codes: Integer codes as numpy array
        - uniques: Unique values as numpy array
        - n_uniques: Number of unique values
    """
    from pandas.core.algorithms import factorize as pd_factorize

    codes, uniques = pd_factorize(arr, sort=sort)
    return codes, uniques, len(uniques)


@register_kernel("get_group_index", "numpy")
def get_group_index(
    labels: list[np.ndarray],
    shape: tuple[int, ...],
    sort: bool = False,
    xnull: bool = True,
) -> np.ndarray:
    """
    Create a single group index from multiple label arrays.

    Combines multiple factorized arrays into a single compound index.
    Uses pandas' Cython implementation for speed.

    Parameters
    ----------
    labels : list[np.ndarray]
        List of integer label arrays (from factorize).
    shape : tuple[int, ...]
        Number of unique values in each label array.
    sort : bool, default False
        Whether to sort the resulting index.
    xnull : bool, default True
        Whether to exclude null labels.

    Returns
    -------
    np.ndarray
        Single integer array representing compound group index.
    """
    from pandas.core.sorting import get_group_index as pd_get_group_index

    return pd_get_group_index(labels=labels, shape=shape, sort=sort, xnull=xnull)


@register_kernel("duplicated", "numpy")
def duplicated(values: np.ndarray, keep: str = "first") -> np.ndarray:
    """
    Find duplicate values in an array.

    Uses pandas' Cython hash table for fast duplicate detection.

    Parameters
    ----------
    values : np.ndarray
        Input array to check for duplicates.
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for the first occurrence.
        - 'last': Mark duplicates as True except for the last occurrence.
        - False: Mark all duplicates as True.

    Returns
    -------
    np.ndarray
        Boolean array where True indicates a duplicate value.
    """
    from pandas.core.algorithms import duplicated as pd_duplicated

    return pd_duplicated(values, keep=keep)


# GroupBy / Aggregation Operations
# =============================================================================


def _numpy_groupby_aggregate(
    keys: np.ndarray,
    values: np.ndarray,
    agg_func: str,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal helper for grouped aggregation using numpy.

    Uses factorize to create group indices and then aggregates with numpy
    functions. For better performance on large datasets, consider pandas
    cython-based groupby or other optimized implementations.

    Parameters
    ----------
    keys : np.ndarray
        Array to group by.
    values : np.ndarray
        Array to aggregate.
    agg_func : str
        Aggregation function name.
    **kwargs
        Additional arguments for specific aggregations (e.g., ddof for std/var).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, aggregated_values)
    """
    # Get unique keys and indices using the factorize kernel
    codes, unique_keys, n_groups = factorize(keys)

    # pandas semantics: rows with missing group keys are dropped
    # (groupby dropna=True). factorize encodes missing keys as -1, which
    # would otherwise be interpreted as the last group by np.add.at etc.
    valid = codes >= 0
    if not valid.all():
        codes = codes[valid]
        values = values[valid]

    # pandas skipna semantics: float NaN values are missing and excluded
    # from every aggregation (sum/mean/count/min/max/std/var/first/last).
    if np.issubdtype(values.dtype, np.floating):
        nan_mask = np.isnan(values)
        if nan_mask.any():
            codes = codes[~nan_mask]
            values = values[~nan_mask]

    # Aggregation functions mapping
    if agg_func == "sum":
        # Use bincount for numeric sum (very fast)
        if np.issubdtype(values.dtype, np.floating):
            result = np.bincount(codes, weights=values, minlength=n_groups)
        else:
            # For integers, accumulate
            result = np.zeros(n_groups, dtype=values.dtype)
            np.add.at(result, codes, values)

    elif agg_func == "count":
        result = np.bincount(codes, minlength=n_groups)

    elif agg_func == "mean":
        sums = np.bincount(codes, weights=values.astype(float), minlength=n_groups)
        counts = np.bincount(codes, minlength=n_groups)
        result = sums / counts

    elif agg_func == "min":
        result = np.empty(n_groups, dtype=values.dtype)
        result[:] = (
            np.iinfo(values.dtype).max
            if np.issubdtype(values.dtype, np.integer)
            else np.inf
        )
        np.minimum.at(result, codes, values)

    elif agg_func == "max":
        result = np.empty(n_groups, dtype=values.dtype)
        result[:] = (
            np.iinfo(values.dtype).min
            if np.issubdtype(values.dtype, np.integer)
            else -np.inf
        )
        np.maximum.at(result, codes, values)

    elif agg_func == "first":
        result = np.empty(n_groups, dtype=values.dtype)
        seen = np.zeros(n_groups, dtype=bool)
        for i, (code, val) in enumerate(zip(codes, values, strict=False)):
            if not seen[code]:
                result[code] = val
                seen[code] = True

    elif agg_func == "last":
        result = np.empty(n_groups, dtype=values.dtype)
        for code, val in zip(codes, values, strict=False):
            result[code] = val

    elif agg_func == "std":
        ddof = kwargs.get("ddof", 1)
        # Two-pass algorithm for numerical stability
        sums = np.bincount(codes, weights=values.astype(float), minlength=n_groups)
        counts = np.bincount(codes, minlength=n_groups)
        means = sums / counts
        # Compute squared deviations
        sq_devs = (values - means[codes]) ** 2
        sum_sq_devs = np.bincount(codes, weights=sq_devs, minlength=n_groups)
        result = np.sqrt(sum_sq_devs / (counts - ddof))

    elif agg_func == "var":
        ddof = kwargs.get("ddof", 1)
        sums = np.bincount(codes, weights=values.astype(float), minlength=n_groups)
        counts = np.bincount(codes, minlength=n_groups)
        means = sums / counts
        sq_devs = (values - means[codes]) ** 2
        sum_sq_devs = np.bincount(codes, weights=sq_devs, minlength=n_groups)
        result = sum_sq_devs / (counts - ddof)

    elif agg_func == "median":
        # Exact per-group median. Arrow has no exact hash_median kernel (only
        # approximate t-digest), so this NumPy path computes it via pandas'
        # Cython grouped median; empty/all-NaN groups yield NaN.
        import pandas as pd

        medians = pd.Series(values).groupby(codes, sort=True).median()
        result = medians.reindex(np.arange(n_groups)).to_numpy()

    elif agg_func == "prod":
        result = np.ones(n_groups, dtype=values.dtype)
        np.multiply.at(result, codes, values)

    elif agg_func == "any":
        result = np.zeros(n_groups, dtype=bool)
        np.logical_or.at(result, codes, values.astype(bool))

    elif agg_func == "all":
        result = np.ones(n_groups, dtype=bool)
        np.logical_and.at(result, codes, values.astype(bool))

    elif agg_func in ("nunique", "n_unique"):
        # Vectorized count distinct per group using pandas-style hash approach:
        # 1. Factorize values to get unique value codes
        # 2. Create compound (group, value) index using get_group_index
        # 3. Find duplicates using hash table (duplicated)
        # 4. Count unique values per group with bincount
        #
        # This is ~18x faster than lexsort-based approach for large datasets
        value_codes, _, n_values = factorize(values)

        # Create compound index: unique integer for each (group, value) pair
        group_index = get_group_index([codes, value_codes], (n_groups, n_values))

        # Find duplicates using Cython hash table (much faster than sort)
        mask = duplicated(group_index)

        # Count unique values per group (only count first occurrence of each pair)
        result = np.bincount(codes[~mask], minlength=n_groups).astype(np.int64)

    else:
        raise NotImplementedError(f"Aggregation function '{agg_func}' not implemented")

    return unique_keys, result


@register_kernel("groupby_sum", "numpy")
def numpy_groupby_sum(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, summed_values)
    """
    return _numpy_groupby_aggregate(keys, values, "sum")


@register_kernel("groupby_mean", "numpy")
def numpy_groupby_mean(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean of values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, mean_values)
    """
    return _numpy_groupby_aggregate(keys, values, "mean")


@register_kernel("groupby_min", "numpy")
def numpy_groupby_min(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Min of values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, min_values)
    """
    return _numpy_groupby_aggregate(keys, values, "min")


@register_kernel("groupby_max", "numpy")
def numpy_groupby_max(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Max of values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, max_values)
    """
    return _numpy_groupby_aggregate(keys, values, "max")


@register_kernel("groupby_count", "numpy")
def numpy_groupby_count(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count of values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, count_values)
    """
    return _numpy_groupby_aggregate(keys, values, "count")


@register_kernel("groupby_first", "numpy")
def numpy_groupby_first(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    First value in each group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, first_values)
    """
    return _numpy_groupby_aggregate(keys, values, "first")


@register_kernel("groupby_last", "numpy")
def numpy_groupby_last(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Last value in each group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, last_values)
    """
    return _numpy_groupby_aggregate(keys, values, "last")


@register_kernel("groupby_std", "numpy")
def numpy_groupby_std(
    keys: np.ndarray, values: np.ndarray, *, ddof: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard deviation of values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, std_values)
    """
    return _numpy_groupby_aggregate(keys, values, "std", ddof=ddof)


@register_kernel("groupby_var", "numpy")
def numpy_groupby_var(
    keys: np.ndarray, values: np.ndarray, *, ddof: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Variance of values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, var_values)
    """
    return _numpy_groupby_aggregate(keys, values, "var", ddof=ddof)


@register_kernel("groupby_median", "numpy")
def numpy_groupby_median(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact median of values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, median_values)
    """
    return _numpy_groupby_aggregate(keys, values, "median")


@register_kernel("groupby_prod", "numpy")
def numpy_groupby_prod(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Product of values grouped by keys.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, prod_values)
    """
    return _numpy_groupby_aggregate(keys, values, "prod")


@register_kernel("groupby_any", "numpy")
def numpy_groupby_any(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Any True value in each group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, any_values)
    """
    return _numpy_groupby_aggregate(keys, values, "any")


@register_kernel("groupby_all", "numpy")
def numpy_groupby_all(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    All True values in each group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, all_values)
    """
    return _numpy_groupby_aggregate(keys, values, "all")


@register_kernel("groupby_nunique", "numpy")
@register_kernel("groupby_n_unique", "numpy")
def numpy_groupby_nunique(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count distinct values in each group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (unique_keys, nunique_values)
    """
    # Integer fast path: pack (key, value) into one int64, dedup once, then
    # run-length count per key — 2.3x over the pandas grouped nunique on
    # TPC-H q21's 6M-row shape (428 -> 183 ms). Safe only when the packed
    # product fits well inside int64; otherwise fall through.
    if (
        keys.dtype.kind in "iu"
        and values.dtype.kind in "iu"
        and len(keys)
        and keys.dtype != np.dtype("uint64")
        and values.dtype != np.dtype("uint64")
    ):
        kmin = int(keys.min())
        vmin = int(values.min())
        span = int(values.max()) - vmin + 1
        krange = int(keys.max()) - kmin + 1
        if 0 < span and 0 < krange <= (2**62) // span:
            comb = (keys.astype(np.int64) - kmin) * span + (
                values.astype(np.int64) - vmin
            )
            distinct = np.unique(comb)  # sorted -> group keys ascending
            gk = distinct // span
            # Run-length encode the sorted group keys.
            change = np.flatnonzero(gk[1:] != gk[:-1])
            starts = np.concatenate(([0], change + 1))
            ends = np.concatenate((change + 1, [len(gk)]))
            unique_keys = (gk[starts] + kmin).astype(keys.dtype, copy=False)
            return unique_keys, (ends - starts).astype(np.int64)
    return _numpy_groupby_aggregate(keys, values, "nunique")


def _numpy_multi_key_groupby(
    key_arrays: list[np.ndarray],
    value_arrays: list[np.ndarray],
    agg_funcs: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Multi-key groupby aggregation.

    Uses vectorized composite key creation via structured arrays,
    avoiding Python-level iteration for better performance.

    Parameters
    ----------
    key_arrays : list[np.ndarray]
        Arrays to group by.
    value_arrays : list[np.ndarray]
        Arrays to aggregate.
    agg_funcs : list[str]
        Aggregation function for each value array.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        (unique_key_arrays, aggregated_value_arrays)
    """
    # For single key, use the simpler path
    if len(key_arrays) == 1:
        result_values = []
        unique_keys = None
        for values, agg_func in zip(value_arrays, agg_funcs, strict=False):
            keys, agg_result = _numpy_groupby_aggregate(key_arrays[0], values, agg_func)
            if unique_keys is None:
                unique_keys = keys
            result_values.append(agg_result)
        return [unique_keys], result_values

    # Multi-key: Create composite key codes using factorize on each key,
    # then combine codes into a single composite code
    n_rows = len(key_arrays[0])

    # Factorize each key column to get integer codes
    all_codes = []
    all_uniques = []
    multipliers = []
    current_multiplier = 1

    for key_arr in key_arrays:
        codes, uniques, _ = factorize(key_arr)
        all_codes.append(codes)
        all_uniques.append(uniques)
        multipliers.append(current_multiplier)
        current_multiplier *= len(uniques)

    # Create composite codes: code0 + code1*n0 + code2*n0*n1 + ...
    # This creates a unique integer for each unique combination
    composite_codes = np.zeros(n_rows, dtype=np.int64)
    for codes, mult in zip(all_codes, multipliers, strict=False):
        composite_codes += codes.astype(np.int64) * mult

    # Factorize the composite codes to get contiguous group indices
    group_codes, unique_composites, n_groups = factorize(composite_codes)

    # Aggregate each value array using the group codes
    result_values = []
    for values, agg_func in zip(value_arrays, agg_funcs, strict=False):
        # Use the core aggregation logic with group_codes
        if agg_func == "sum":
            if np.issubdtype(values.dtype, np.floating):
                result = np.bincount(group_codes, weights=values, minlength=n_groups)
            else:
                result = np.zeros(n_groups, dtype=np.float64)
                np.add.at(result, group_codes, values.astype(np.float64))
        elif agg_func == "count":
            result = np.bincount(group_codes, minlength=n_groups)
        elif agg_func == "mean":
            sums = np.bincount(
                group_codes, weights=values.astype(float), minlength=n_groups
            )
            counts = np.bincount(group_codes, minlength=n_groups)
            result = sums / counts
        elif agg_func == "min":
            result = np.empty(n_groups, dtype=values.dtype)
            result[:] = (
                np.iinfo(values.dtype).max
                if np.issubdtype(values.dtype, np.integer)
                else np.inf
            )
            np.minimum.at(result, group_codes, values)
        elif agg_func == "max":
            result = np.empty(n_groups, dtype=values.dtype)
            result[:] = (
                np.iinfo(values.dtype).min
                if np.issubdtype(values.dtype, np.integer)
                else -np.inf
            )
            np.maximum.at(result, group_codes, values)
        else:
            # Fallback for other aggregations
            _, result = _numpy_groupby_aggregate(composite_codes, values, agg_func)
        result_values.append(result)

    # Decompose the unique composite codes back to individual key values
    result_keys = []
    for i, (uniques, mult) in enumerate(zip(all_uniques, multipliers, strict=False)):
        # Extract the code for this key from the composite
        if i == len(key_arrays) - 1:
            # Last key: just divide
            key_codes = unique_composites // mult
        else:
            # Extract using modulo with the next multiplier
            next_mult = multipliers[i + 1]
            key_codes = (unique_composites % next_mult) // mult
        # Map codes back to original values
        key_codes = key_codes.astype(np.intp)
        result_keys.append(uniques[key_codes])

    return result_keys, result_values


@register_kernel("hash_aggregate", "numpy")
def numpy_hash_aggregate(
    keys: list[np.ndarray],
    values: list[np.ndarray],
    agg_funcs: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Multi-key hash-based aggregation on arrays.

    Parameters
    ----------
    keys : list[np.ndarray]
        Arrays to group by.
    values : list[np.ndarray]
        Arrays to aggregate.
    agg_funcs : list[str]
        Aggregation function for each value array.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        (unique_keys, aggregated_values)
    """
    return _numpy_multi_key_groupby(keys, values, agg_funcs)


# =============================================================================
