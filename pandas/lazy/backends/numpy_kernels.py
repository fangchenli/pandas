"""
NumPy kernel implementations for lazy pandas.

This module registers NumPy-based implementations for operations.
These kernels use NumPy functions directly on NumPy arrays.

Future: numexpr Integration
---------------------------
For large arrays (>1M elements), numexpr can accelerate arithmetic:

    _MIN_ELEMENTS = 1_000_000
    _NUMEXPR_DTYPES = {"int64", "int32", "float64", "float32", "bool"}

    def numpy_add(left, right):
        if (len(left) > _MIN_ELEMENTS and
            left.dtype.name in _NUMEXPR_DTYPES):
            import numexpr as ne
            return ne.evaluate("left + right")
        return np.add(left, right)

Supported numexpr ops: +, -, *, /, **, ==, !=, <, <=, >, >=, &, |, ^
NOT supported: //, % (use numpy for these)
"""

from typing import Any

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
        arr_copy = arr.copy()
        if null_placement == "at_end":
            fill_val = np.inf if not descending else -np.inf
        else:  # at_start
            fill_val = -np.inf if not descending else np.inf
        arr_copy[mask] = fill_val
        indices = np.argsort(arr_copy)
    else:
        indices = np.argsort(arr)

    if descending:
        indices = indices[::-1]

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
        (unique_values, indices of first occurrence)
    """
    return np.unique(arr, return_index=True)


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
    # Get unique keys and indices
    # Use pandas factorize for proper handling of various dtypes
    from pandas import factorize

    codes, unique_keys = factorize(keys, sort=False)

    n_groups = len(unique_keys)

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

    elif agg_func == "prod":
        result = np.ones(n_groups, dtype=values.dtype)
        np.multiply.at(result, codes, values)

    elif agg_func == "any":
        result = np.zeros(n_groups, dtype=bool)
        np.logical_or.at(result, codes, values.astype(bool))

    elif agg_func == "all":
        result = np.ones(n_groups, dtype=bool)
        np.logical_and.at(result, codes, values.astype(bool))

    elif agg_func == "nunique":
        # Count distinct per group
        result = np.zeros(n_groups, dtype=np.int64)
        # Use sets per group
        seen = [set() for _ in range(n_groups)]
        for code, val in zip(codes, values, strict=False):
            seen[code].add(val)
        for i, s in enumerate(seen):
            result[i] = len(s)

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
    return _numpy_groupby_aggregate(keys, values, "nunique")


def _numpy_multi_key_groupby(
    key_arrays: list[np.ndarray],
    value_arrays: list[np.ndarray],
    agg_funcs: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Multi-key groupby aggregation.

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
    from pandas import factorize

    # Create composite key via structured array
    if len(key_arrays) == 1:
        composite_keys = key_arrays[0]
    else:
        # Use tuple-based composite key
        n = len(key_arrays[0])
        composite_keys = np.empty(n, dtype=object)
        for i in range(n):
            composite_keys[i] = tuple(arr[i] for arr in key_arrays)

    codes, unique_composite = factorize(composite_keys, sort=False)
    len(unique_composite)

    # Aggregate each value array
    result_values = []
    for values, agg_func in zip(value_arrays, agg_funcs, strict=False):
        _, agg_result = _numpy_groupby_aggregate(composite_keys, values, agg_func)
        result_values.append(agg_result)

    # Decompose unique composite keys back to individual arrays
    if len(key_arrays) == 1:
        result_keys = [unique_composite]
    else:
        result_keys = []
        for i in range(len(key_arrays)):
            key_vals = np.array(
                [k[i] for k in unique_composite], dtype=key_arrays[i].dtype
            )
            result_keys.append(key_vals)

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
# Cumulative / Window Functions
# =============================================================================


@register_kernel("cumulative_sum", "numpy")
def numpy_cumulative_sum(arr: np.ndarray, skip_nulls: bool = True) -> np.ndarray:
    """
    Compute cumulative sum.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    np.ndarray
        Cumulative sum array.
    """
    if skip_nulls and np.issubdtype(arr.dtype, np.floating):
        # Handle NaN values - use pandas-style cumsum that skips NaN
        result = np.empty_like(arr)
        mask = ~np.isnan(arr)
        # Fill with cumsum of non-NaN values
        cum = 0.0
        for i in range(len(arr)):
            if mask[i]:
                cum += arr[i]
            result[i] = cum
        return result
    return np.cumsum(arr)


@register_kernel("cumulative_max", "numpy")
def numpy_cumulative_max(arr: np.ndarray, skip_nulls: bool = True) -> np.ndarray:
    """
    Compute cumulative maximum.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    np.ndarray
        Cumulative max array.
    """
    if skip_nulls and np.issubdtype(arr.dtype, np.floating):
        result = np.empty_like(arr)
        mask = ~np.isnan(arr)
        cur_max = -np.inf
        for i in range(len(arr)):
            if mask[i]:
                cur_max = max(cur_max, arr[i])
            result[i] = cur_max if cur_max != -np.inf else np.nan
        return result
    return np.maximum.accumulate(arr)


@register_kernel("cumulative_min", "numpy")
def numpy_cumulative_min(arr: np.ndarray, skip_nulls: bool = True) -> np.ndarray:
    """
    Compute cumulative minimum.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    np.ndarray
        Cumulative min array.
    """
    if skip_nulls and np.issubdtype(arr.dtype, np.floating):
        result = np.empty_like(arr)
        mask = ~np.isnan(arr)
        cur_min = np.inf
        for i in range(len(arr)):
            if mask[i]:
                cur_min = min(cur_min, arr[i])
            result[i] = cur_min if cur_min != np.inf else np.nan
        return result
    return np.minimum.accumulate(arr)


@register_kernel("cumulative_prod", "numpy")
def numpy_cumulative_prod(arr: np.ndarray, skip_nulls: bool = True) -> np.ndarray:
    """
    Compute cumulative product.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    np.ndarray
        Cumulative product array.
    """
    if skip_nulls and np.issubdtype(arr.dtype, np.floating):
        result = np.empty_like(arr)
        mask = ~np.isnan(arr)
        cum = 1.0
        for i in range(len(arr)):
            if mask[i]:
                cum *= arr[i]
            result[i] = cum
        return result
    return np.cumprod(arr)


@register_kernel("cumulative_mean", "numpy")
def numpy_cumulative_mean(arr: np.ndarray, skip_nulls: bool = True) -> np.ndarray:
    """
    Compute cumulative mean.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    np.ndarray
        Cumulative mean array.
    """
    if skip_nulls and np.issubdtype(arr.dtype, np.floating):
        result = np.empty_like(arr, dtype=np.float64)
        mask = ~np.isnan(arr)
        cum_sum = 0.0
        cum_count = 0
        for i in range(len(arr)):
            if mask[i]:
                cum_sum += arr[i]
                cum_count += 1
            result[i] = cum_sum / cum_count if cum_count > 0 else np.nan
        return result
    # For non-floating point or no skip_nulls
    result = np.empty(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        result[i] = np.mean(arr[: i + 1])
    return result


# =============================================================================
# Datetime Extraction Functions
# =============================================================================


@register_kernel("dt_year", "numpy")
def numpy_dt_year(arr: np.ndarray) -> np.ndarray:
    """
    Extract year from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of years (int64).
    """
    # Convert to datetime64[Y] and extract year
    return arr.astype("datetime64[Y]").astype(int) + 1970


@register_kernel("dt_month", "numpy")
def numpy_dt_month(arr: np.ndarray) -> np.ndarray:
    """
    Extract month from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of months (1-12).
    """
    import pandas as pd

    # Use pandas for reliable datetime component extraction
    return pd.DatetimeIndex(arr).month.to_numpy()


@register_kernel("dt_day", "numpy")
def numpy_dt_day(arr: np.ndarray) -> np.ndarray:
    """
    Extract day of month from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of days (1-31).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).day.to_numpy()


@register_kernel("dt_hour", "numpy")
def numpy_dt_hour(arr: np.ndarray) -> np.ndarray:
    """
    Extract hour from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of hours (0-23).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).hour.to_numpy()


@register_kernel("dt_minute", "numpy")
def numpy_dt_minute(arr: np.ndarray) -> np.ndarray:
    """
    Extract minute from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of minutes (0-59).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).minute.to_numpy()


@register_kernel("dt_second", "numpy")
def numpy_dt_second(arr: np.ndarray) -> np.ndarray:
    """
    Extract second from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of seconds (0-59).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).second.to_numpy()


@register_kernel("dt_microsecond", "numpy")
def numpy_dt_microsecond(arr: np.ndarray) -> np.ndarray:
    """
    Extract microsecond from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of microseconds (0-999999).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).microsecond.to_numpy()


@register_kernel("dt_nanosecond", "numpy")
def numpy_dt_nanosecond(arr: np.ndarray) -> np.ndarray:
    """
    Extract nanosecond from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of nanoseconds (0-999999999).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).nanosecond.to_numpy()


@register_kernel("dt_day_of_week", "numpy")
def numpy_dt_day_of_week(arr: np.ndarray, count_from_zero: bool = True) -> np.ndarray:
    """
    Extract day of week from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.
    count_from_zero : bool, default True
        If True, returns 0-6 (Monday-Sunday).
        If False, returns 1-7 (Monday-Sunday).

    Returns
    -------
    np.ndarray
        Array of day of week values.
    """
    import pandas as pd

    dow = pd.DatetimeIndex(arr).dayofweek.to_numpy()
    if not count_from_zero:
        dow = dow + 1
    return dow


@register_kernel("dt_day_of_year", "numpy")
def numpy_dt_day_of_year(arr: np.ndarray) -> np.ndarray:
    """
    Extract day of year from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of day of year values (1-366).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).dayofyear.to_numpy()


@register_kernel("dt_week", "numpy")
def numpy_dt_week(arr: np.ndarray) -> np.ndarray:
    """
    Extract ISO week number from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of week numbers (1-53).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).isocalendar().week.to_numpy()


@register_kernel("dt_quarter", "numpy")
def numpy_dt_quarter(arr: np.ndarray) -> np.ndarray:
    """
    Extract quarter from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of quarters (1-4).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).quarter.to_numpy()


@register_kernel("dt_is_leap_year", "numpy")
def numpy_dt_is_leap_year(arr: np.ndarray) -> np.ndarray:
    """
    Check if datetime values fall in a leap year.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Boolean array indicating leap year.
    """
    import pandas as pd

    result = pd.DatetimeIndex(arr).is_leap_year
    # is_leap_year returns ndarray directly
    if hasattr(result, "to_numpy"):
        return result.to_numpy()
    return np.asarray(result)


@register_kernel("dt_iso_year", "numpy")
def numpy_dt_iso_year(arr: np.ndarray) -> np.ndarray:
    """
    Extract ISO year from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of ISO years.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).isocalendar().year.to_numpy()


@register_kernel("dt_strftime", "numpy")
def numpy_dt_strftime(arr: np.ndarray, format: str) -> np.ndarray:
    """
    Format datetime array to strings.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.
    format : str
        strftime format string.

    Returns
    -------
    np.ndarray
        String array of formatted datetimes.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).strftime(format).to_numpy()


# =============================================================================
# Additional String Functions
# =============================================================================


@register_kernel("str_capitalize", "numpy")
def numpy_str_capitalize(arr: np.ndarray) -> np.ndarray:
    """Capitalize strings (first char uppercase, rest lowercase)."""
    return np.char.capitalize(arr)


@register_kernel("str_title", "numpy")
def numpy_str_title(arr: np.ndarray) -> np.ndarray:
    """Title-case strings (first char of each word uppercase)."""
    return np.char.title(arr)


@register_kernel("str_swapcase", "numpy")
def numpy_str_swapcase(arr: np.ndarray) -> np.ndarray:
    """Swap case of strings (upper to lower, lower to upper)."""
    return np.char.swapcase(arr)


@register_kernel("str_is_alnum", "numpy")
def numpy_str_is_alnum(arr: np.ndarray) -> np.ndarray:
    """Check if strings contain only alphanumeric characters."""
    return np.char.isalnum(arr)


@register_kernel("str_is_alpha", "numpy")
def numpy_str_is_alpha(arr: np.ndarray) -> np.ndarray:
    """Check if strings contain only alphabetic characters."""
    return np.char.isalpha(arr)


@register_kernel("str_is_decimal", "numpy")
def numpy_str_is_decimal(arr: np.ndarray) -> np.ndarray:
    """Check if strings contain only decimal characters."""
    return np.char.isdecimal(arr)


@register_kernel("str_is_digit", "numpy")
def numpy_str_is_digit(arr: np.ndarray) -> np.ndarray:
    """Check if strings contain only digit characters."""
    return np.char.isdigit(arr)


@register_kernel("str_is_lower", "numpy")
def numpy_str_is_lower(arr: np.ndarray) -> np.ndarray:
    """Check if strings are all lowercase."""
    return np.char.islower(arr)


@register_kernel("str_is_upper", "numpy")
def numpy_str_is_upper(arr: np.ndarray) -> np.ndarray:
    """Check if strings are all uppercase."""
    return np.char.isupper(arr)


@register_kernel("str_is_numeric", "numpy")
def numpy_str_is_numeric(arr: np.ndarray) -> np.ndarray:
    """Check if strings contain only numeric characters."""
    return np.char.isnumeric(arr)


@register_kernel("str_is_space", "numpy")
def numpy_str_is_space(arr: np.ndarray) -> np.ndarray:
    """Check if strings contain only whitespace."""
    return np.char.isspace(arr)


@register_kernel("str_is_title", "numpy")
def numpy_str_is_title(arr: np.ndarray) -> np.ndarray:
    """Check if strings are title-cased."""
    return np.char.istitle(arr)


@register_kernel("str_count", "numpy")
def numpy_str_count(arr: np.ndarray, pattern: str, regex: bool = False) -> np.ndarray:
    """
    Count occurrences of a pattern in strings.

    Parameters
    ----------
    arr : np.ndarray
        Input string array.
    pattern : str
        Pattern to count.
    regex : bool, default False
        Whether to treat pattern as regex.

    Returns
    -------
    np.ndarray
        Integer array of counts.
    """
    if regex:
        import re

        compiled = re.compile(pattern)
        return np.array([len(compiled.findall(s)) for s in arr])
    return np.char.count(arr, pattern)


@register_kernel("str_find", "numpy")
def numpy_str_find(arr: np.ndarray, pattern: str, regex: bool = False) -> np.ndarray:
    """
    Find first occurrence of pattern in strings.

    Parameters
    ----------
    arr : np.ndarray
        Input string array.
    pattern : str
        Pattern to find.
    regex : bool, default False
        Whether to treat pattern as regex.

    Returns
    -------
    np.ndarray
        Integer array of positions (-1 if not found).
    """
    if regex:
        import re

        compiled = re.compile(pattern)
        result = np.empty(len(arr), dtype=np.int64)
        for i, s in enumerate(arr):
            match = compiled.search(s)
            result[i] = match.start() if match else -1
        return result
    return np.char.find(arr, pattern)


@register_kernel("str_pad", "numpy")
def numpy_str_pad(
    arr: np.ndarray, width: int, side: str = "left", fillchar: str = " "
) -> np.ndarray:
    """
    Pad strings to a specified width.

    Parameters
    ----------
    arr : np.ndarray
        Input string array.
    width : int
        Target width.
    side : str, default "left"
        Side to pad ("left", "right", or "both").
    fillchar : str, default " "
        Character to use for padding.

    Returns
    -------
    np.ndarray
        Padded string array.
    """
    if side == "left":
        return np.char.rjust(arr, width, fillchar)
    elif side == "right":
        return np.char.ljust(arr, width, fillchar)
    else:  # "both" - center
        return np.char.center(arr, width, fillchar)


@register_kernel("str_center", "numpy")
def numpy_str_center(arr: np.ndarray, width: int, fillchar: str = " ") -> np.ndarray:
    """
    Center strings in a field of given width.

    Parameters
    ----------
    arr : np.ndarray
        Input string array.
    width : int
        Target width.
    fillchar : str, default " "
        Character to use for padding.

    Returns
    -------
    np.ndarray
        Centered string array.
    """
    return np.char.center(arr, width, fillchar)


@register_kernel("str_zfill", "numpy")
def numpy_str_zfill(arr: np.ndarray, width: int) -> np.ndarray:
    """
    Pad strings with zeros on the left.

    Parameters
    ----------
    arr : np.ndarray
        Input string array.
    width : int
        Target width.

    Returns
    -------
    np.ndarray
        Zero-padded string array.
    """
    return np.char.zfill(arr, width)


@register_kernel("str_match", "numpy")
def numpy_str_match(arr: np.ndarray, pattern: str, regex: bool = True) -> np.ndarray:
    """
    Check if strings match a pattern.

    Parameters
    ----------
    arr : np.ndarray
        Input string array.
    pattern : str
        Pattern to match.
    regex : bool, default True
        Whether to treat pattern as regex.

    Returns
    -------
    np.ndarray
        Boolean array indicating matches.
    """
    if regex:
        import re

        compiled = re.compile(pattern)
        return np.array([bool(compiled.search(s)) for s in arr])
    # Simple substring match
    result = np.char.find(arr, pattern)
    return result >= 0


# =============================================================================
# Rolling Window Kernels
# =============================================================================


@register_kernel("rolling_sum", "numpy")
def numpy_rolling_sum(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling sum over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    np.ndarray
        Rolling sum values.
    """
    if min_periods is None:
        min_periods = window

    arr = arr.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start : i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        if valid_count >= min_periods:
            result[i] = np.nansum(window_data)

    return result


@register_kernel("rolling_mean", "numpy")
def numpy_rolling_mean(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling mean over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    np.ndarray
        Rolling mean values.
    """
    if min_periods is None:
        min_periods = window

    arr = arr.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start : i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        if valid_count >= min_periods:
            result[i] = np.nanmean(window_data)

    return result


@register_kernel("rolling_min", "numpy")
def numpy_rolling_min(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling minimum over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    np.ndarray
        Rolling minimum values.
    """
    if min_periods is None:
        min_periods = window

    arr = arr.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start : i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        if valid_count >= min_periods:
            result[i] = np.nanmin(window_data)

    return result


@register_kernel("rolling_max", "numpy")
def numpy_rolling_max(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling maximum over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    np.ndarray
        Rolling maximum values.
    """
    if min_periods is None:
        min_periods = window

    arr = arr.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start : i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        if valid_count >= min_periods:
            result[i] = np.nanmax(window_data)

    return result


@register_kernel("rolling_std", "numpy")
def numpy_rolling_std(
    arr: np.ndarray, window: int, min_periods: int | None = None, ddof: int = 1
) -> np.ndarray:
    """
    Calculate rolling standard deviation over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.
    ddof : int, default 1
        Delta degrees of freedom.

    Returns
    -------
    np.ndarray
        Rolling standard deviation values.
    """
    if min_periods is None:
        min_periods = window

    arr = arr.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start : i + 1]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) >= min_periods and len(valid_data) > ddof:
            result[i] = np.std(valid_data, ddof=ddof)

    return result


@register_kernel("rolling_var", "numpy")
def numpy_rolling_var(
    arr: np.ndarray, window: int, min_periods: int | None = None, ddof: int = 1
) -> np.ndarray:
    """
    Calculate rolling variance over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.
    ddof : int, default 1
        Delta degrees of freedom.

    Returns
    -------
    np.ndarray
        Rolling variance values.
    """
    if min_periods is None:
        min_periods = window

    arr = arr.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start : i + 1]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) >= min_periods and len(valid_data) > ddof:
            result[i] = np.var(valid_data, ddof=ddof)

    return result


# =============================================================================
# Shift/Lag/Lead Kernels
# =============================================================================


@register_kernel("shift", "numpy")
def numpy_shift(
    arr: np.ndarray, periods: int = 1, fill_value: Any = None
) -> np.ndarray:
    """
    Shift array values by specified number of periods.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    periods : int, default 1
        Number of periods to shift. Positive shifts forward, negative shifts backward.
    fill_value : Any, default None
        Value to use for filling gaps.

    Returns
    -------
    np.ndarray
        Shifted array.
    """
    n = len(arr)
    dtype = arr.dtype

    # Determine fill value
    if fill_value is None:
        if np.issubdtype(dtype, np.floating):
            fill_val = np.nan
        elif np.issubdtype(dtype, np.integer):
            # Convert to float to support NaN
            arr = arr.astype(float)
            fill_val = np.nan
        else:
            fill_val = None
    else:
        fill_val = fill_value

    result = np.empty(n, dtype=arr.dtype if fill_val is not None else object)

    if periods > 0:
        result[:periods] = fill_val
        if periods < n:
            result[periods:] = arr[:-periods]
    elif periods < 0:
        result[periods:] = fill_val
        if -periods < n:
            result[:periods] = arr[-periods:]
    else:
        result = arr.copy()

    return result


@register_kernel("lag", "numpy")
def numpy_lag(arr: np.ndarray, periods: int = 1, fill_value: Any = None) -> np.ndarray:
    """
    Get lagged values (shift forward).

    This is an alias for shift with positive periods.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    periods : int, default 1
        Number of periods to lag.
    fill_value : Any, default None
        Value to use for filling gaps.

    Returns
    -------
    np.ndarray
        Lagged array.
    """
    return numpy_shift(arr, periods=abs(periods), fill_value=fill_value)


@register_kernel("lead", "numpy")
def numpy_lead(arr: np.ndarray, periods: int = 1, fill_value: Any = None) -> np.ndarray:
    """
    Get lead values (shift backward).

    This is an alias for shift with negative periods.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    periods : int, default 1
        Number of periods to lead.
    fill_value : Any, default None
        Value to use for filling gaps.

    Returns
    -------
    np.ndarray
        Lead array.
    """
    return numpy_shift(arr, periods=-abs(periods), fill_value=fill_value)


# =============================================================================
# Fill NA Variant Kernels
# =============================================================================


@register_kernel("ffill", "numpy")
def numpy_ffill(arr: np.ndarray, limit: int | None = None) -> np.ndarray:
    """
    Forward fill missing values.

    Parameters
    ----------
    arr : np.ndarray
        Input array with potential missing values.
    limit : int or None
        Maximum number of consecutive NaN values to forward fill.

    Returns
    -------
    np.ndarray
        Array with forward-filled values.
    """
    result = arr.copy().astype(float)
    n = len(result)

    last_valid = None
    consecutive_nulls = 0

    for i in range(n):
        if np.isnan(result[i]):
            consecutive_nulls += 1
            if last_valid is not None and (limit is None or consecutive_nulls <= limit):
                result[i] = last_valid
        else:
            last_valid = result[i]
            consecutive_nulls = 0

    return result


@register_kernel("bfill", "numpy")
def numpy_bfill(arr: np.ndarray, limit: int | None = None) -> np.ndarray:
    """
    Backward fill missing values.

    Parameters
    ----------
    arr : np.ndarray
        Input array with potential missing values.
    limit : int or None
        Maximum number of consecutive NaN values to backward fill.

    Returns
    -------
    np.ndarray
        Array with backward-filled values.
    """
    result = arr.copy().astype(float)
    n = len(result)

    last_valid = None
    consecutive_nulls = 0

    for i in range(n - 1, -1, -1):
        if np.isnan(result[i]):
            consecutive_nulls += 1
            if last_valid is not None and (limit is None or consecutive_nulls <= limit):
                result[i] = last_valid
        else:
            last_valid = result[i]
            consecutive_nulls = 0

    return result


@register_kernel("interpolate_linear", "numpy")
def numpy_interpolate_linear(arr: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate missing values.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array with potential missing values.

    Returns
    -------
    np.ndarray
        Array with linearly interpolated values.
    """
    result = arr.copy().astype(float)
    n = len(result)

    # Find indices of valid values
    mask = ~np.isnan(result)
    if not mask.any():
        return result

    valid_indices = np.where(mask)[0]
    valid_values = result[valid_indices]

    # Interpolate
    all_indices = np.arange(n)
    result = np.interp(all_indices, valid_indices, valid_values)

    # Keep leading/trailing NaN if original had them
    if valid_indices[0] > 0:
        result[: valid_indices[0]] = np.nan
    if valid_indices[-1] < n - 1:
        result[valid_indices[-1] + 1 :] = np.nan

    return result
