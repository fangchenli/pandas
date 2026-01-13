"""
Arrow compute kernel implementations for lazy pandas.

This module registers Arrow-based implementations for operations.
These kernels use PyArrow compute functions directly on Arrow arrays.
"""

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import pandas as pd
from pandas.lazy.backends import register_kernel
from pandas.lazy.backends.types import PyArrowArray

# =============================================================================
# String Operations (Major performance wins over pandas)
# =============================================================================


@register_kernel("str_lower", "arrow")
def arrow_str_lower(arr: PyArrowArray) -> PyArrowArray:
    """Convert strings to lowercase."""
    return pc.utf8_lower(arr)


@register_kernel("str_upper", "arrow")
def arrow_str_upper(arr: PyArrowArray) -> PyArrowArray:
    """Convert strings to uppercase."""
    return pc.utf8_upper(arr)


@register_kernel("str_len", "arrow")
def arrow_str_len(arr: PyArrowArray) -> PyArrowArray:
    """Get string lengths."""
    return pc.utf8_length(arr)


@register_kernel("str_strip", "arrow")
def arrow_str_strip(arr: PyArrowArray) -> PyArrowArray:
    """Strip whitespace from both ends."""
    return pc.utf8_trim_whitespace(arr)


@register_kernel("str_lstrip", "arrow")
def arrow_str_lstrip(arr: PyArrowArray) -> PyArrowArray:
    """Strip whitespace from left."""
    return pc.utf8_ltrim_whitespace(arr)


@register_kernel("str_rstrip", "arrow")
def arrow_str_rstrip(arr: PyArrowArray) -> PyArrowArray:
    """Strip whitespace from right."""
    return pc.utf8_rtrim_whitespace(arr)


@register_kernel("str_contains", "arrow")
def arrow_str_contains(
    arr: PyArrowArray, pattern: str, *, regex: bool = True
) -> PyArrowArray:
    """Check if strings contain pattern."""
    if regex:
        return pc.match_substring_regex(arr, pattern)
    return pc.match_substring(arr, pattern)


@register_kernel("str_startswith", "arrow")
def arrow_str_startswith(arr: PyArrowArray, pattern: str) -> PyArrowArray:
    """Check if strings start with pattern."""
    return pc.starts_with(arr, pattern)


@register_kernel("str_endswith", "arrow")
def arrow_str_endswith(arr: PyArrowArray, pattern: str) -> PyArrowArray:
    """Check if strings end with pattern."""
    return pc.ends_with(arr, pattern)


@register_kernel("str_replace", "arrow")
def arrow_str_replace(
    arr: PyArrowArray,
    pattern: str,
    replacement: str,
    *,
    regex: bool = True,
) -> PyArrowArray:
    """Replace pattern in strings."""
    if regex:
        return pc.replace_substring_regex(arr, pattern, replacement)
    return pc.replace_substring(arr, pattern, replacement)


@register_kernel("str_slice", "arrow")
def arrow_str_slice(
    arr: PyArrowArray,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> PyArrowArray:
    """Slice strings."""
    start = start or 0
    if stop is None:
        return pc.utf8_slice_codeunits(arr, start)
    return pc.utf8_slice_codeunits(arr, start, stop - start)


# =============================================================================
# Null Operations (Native Arrow support)
# =============================================================================


@register_kernel("is_null", "arrow")
def arrow_is_null(arr: PyArrowArray) -> PyArrowArray:
    """Check for null values."""
    return pc.is_null(arr)


@register_kernel("is_not_null", "arrow")
def arrow_is_not_null(arr: PyArrowArray) -> PyArrowArray:
    """Check for non-null values."""
    return pc.is_valid(arr)


@register_kernel("fill_null", "arrow")
def arrow_fill_null(arr: PyArrowArray, fill_value) -> PyArrowArray:
    """Fill null values with a scalar or array."""
    return pc.fill_null(arr, fill_value)


@register_kernel("coalesce", "arrow")
def arrow_coalesce(*arrays: PyArrowArray) -> PyArrowArray:
    """Return first non-null value from arrays."""
    return pc.coalesce(*arrays)


# =============================================================================
# Arithmetic Operations
# =============================================================================


@register_kernel("add", "arrow")
def arrow_add(left: PyArrowArray, right: PyArrowArray | float | int) -> PyArrowArray:
    """Add two arrays or array and scalar."""
    return pc.add(left, right)


@register_kernel("subtract", "arrow")
def arrow_subtract(
    left: PyArrowArray, right: PyArrowArray | float | int
) -> PyArrowArray:
    """Subtract two arrays or array and scalar."""
    return pc.subtract(left, right)


@register_kernel("multiply", "arrow")
def arrow_multiply(
    left: PyArrowArray, right: PyArrowArray | float | int
) -> PyArrowArray:
    """Multiply two arrays or array and scalar."""
    return pc.multiply(left, right)


@register_kernel("divide", "arrow")
def arrow_divide(left: PyArrowArray, right: PyArrowArray | float | int) -> PyArrowArray:
    """Divide two arrays or array and scalar."""
    return pc.divide(left, right)


@register_kernel("floor_divide", "arrow")
def arrow_floor_divide(
    left: PyArrowArray, right: PyArrowArray | float | int
) -> PyArrowArray:
    """Floor divide two arrays or array and scalar."""
    # PyArrow doesn't have direct floor_divide, use divide + floor
    result = pc.divide(left, right)
    return pc.floor(result)


@register_kernel("modulo", "arrow")
def arrow_modulo(left: PyArrowArray, right: PyArrowArray | float | int) -> PyArrowArray:
    """Modulo of two arrays or array and scalar."""
    # PyArrow uses different semantics; compute manually
    # a % b = a - floor(a/b) * b
    quotient = pc.floor(pc.divide(left, right))
    return pc.subtract(left, pc.multiply(quotient, right))


@register_kernel("power", "arrow")
def arrow_power(left: PyArrowArray, right: PyArrowArray | float | int) -> PyArrowArray:
    """Raise array to power."""
    return pc.power(left, right)


@register_kernel("negate", "arrow")
def arrow_negate(arr: PyArrowArray) -> PyArrowArray:
    """Negate array."""
    return pc.negate(arr)


@register_kernel("abs", "arrow")
def arrow_abs(arr: PyArrowArray) -> PyArrowArray:
    """Absolute value."""
    return pc.abs(arr)


# =============================================================================
# Comparison Operations
# =============================================================================


@register_kernel("equal", "arrow")
def arrow_equal(
    left: PyArrowArray, right: PyArrowArray | float | int | str
) -> PyArrowArray:
    """Check equality."""
    return pc.equal(left, right)


@register_kernel("not_equal", "arrow")
def arrow_not_equal(
    left: PyArrowArray, right: PyArrowArray | float | int | str
) -> PyArrowArray:
    """Check inequality."""
    return pc.not_equal(left, right)


@register_kernel("less", "arrow")
def arrow_less(left: PyArrowArray, right: PyArrowArray | float | int) -> PyArrowArray:
    """Check less than."""
    return pc.less(left, right)


@register_kernel("less_equal", "arrow")
def arrow_less_equal(
    left: PyArrowArray, right: PyArrowArray | float | int
) -> PyArrowArray:
    """Check less than or equal."""
    return pc.less_equal(left, right)


@register_kernel("greater", "arrow")
def arrow_greater(
    left: PyArrowArray, right: PyArrowArray | float | int
) -> PyArrowArray:
    """Check greater than."""
    return pc.greater(left, right)


@register_kernel("greater_equal", "arrow")
def arrow_greater_equal(
    left: PyArrowArray, right: PyArrowArray | float | int
) -> PyArrowArray:
    """Check greater than or equal."""
    return pc.greater_equal(left, right)


# =============================================================================
# Logical Operations
# =============================================================================


@register_kernel("and_", "arrow")
def arrow_and(left: PyArrowArray, right: PyArrowArray) -> PyArrowArray:
    """Logical AND."""
    return pc.and_(left, right)


@register_kernel("or_", "arrow")
def arrow_or(left: PyArrowArray, right: PyArrowArray) -> PyArrowArray:
    """Logical OR."""
    return pc.or_(left, right)


@register_kernel("invert", "arrow")
def arrow_invert(arr: PyArrowArray) -> PyArrowArray:
    """Logical NOT."""
    return pc.invert(arr)


# =============================================================================
# Aggregation Operations
# =============================================================================


@register_kernel("sum", "arrow")
def arrow_sum(arr: PyArrowArray):
    """Sum of array elements."""
    result = pc.sum(arr)
    return result.as_py()


@register_kernel("mean", "arrow")
def arrow_mean(arr: PyArrowArray):
    """Mean of array elements."""
    result = pc.mean(arr)
    return result.as_py()


@register_kernel("min", "arrow")
def arrow_min(arr: PyArrowArray):
    """Minimum of array elements."""
    result = pc.min(arr)
    return result.as_py()


@register_kernel("max", "arrow")
def arrow_max(arr: PyArrowArray):
    """Maximum of array elements."""
    result = pc.max(arr)
    return result.as_py()


@register_kernel("count", "arrow")
def arrow_count(arr: PyArrowArray) -> int:
    """Count non-null elements."""
    result = pc.count(arr)
    return result.as_py()


@register_kernel("std", "arrow")
def arrow_std(arr: PyArrowArray, *, ddof: int = 1):
    """Standard deviation."""
    result = pc.stddev(arr, ddof=ddof)
    return result.as_py()


@register_kernel("var", "arrow")
def arrow_var(arr: PyArrowArray, *, ddof: int = 1):
    """Variance."""
    result = pc.variance(arr, ddof=ddof)
    return result.as_py()


@register_kernel("n_unique", "arrow")
def arrow_n_unique(arr: PyArrowArray) -> int:
    """Count unique values."""
    unique = pc.unique(arr)
    return len(unique)


# =============================================================================
# Filter Operation
# =============================================================================


@register_kernel("filter", "arrow")
def arrow_filter(arr: PyArrowArray, mask: PyArrowArray) -> PyArrowArray:
    """Filter array by boolean mask."""
    return pc.filter(arr, mask)


# =============================================================================
# Sort Operations
# =============================================================================


@register_kernel("sort_indices", "arrow")
def arrow_sort_indices(
    arr: PyArrowArray,
    *,
    descending: bool = False,
    null_placement: str = "at_end",
) -> PyArrowArray:
    """
    Return indices that would sort the array.

    Parameters
    ----------
    arr : PyArrowArray
        Array to sort.
    descending : bool, default False
        Sort in descending order.
    null_placement : str, default "at_end"
        Where to place nulls: "at_start" or "at_end".

    Returns
    -------
    PyArrowArray
        Indices that would sort the array.
    """
    return pc.sort_indices(
        arr,
        sort_keys=[("", "descending" if descending else "ascending")],
        null_placement=null_placement,
    )


@register_kernel("array_sort_indices", "arrow")
def arrow_array_sort_indices(
    arr: PyArrowArray,
    *,
    order: str = "ascending",
    null_placement: str = "at_end",
) -> PyArrowArray:
    """
    Return indices that would sort the array (simpler API).

    Parameters
    ----------
    arr : PyArrowArray
        Array to sort.
    order : str, default "ascending"
        Sort order: "ascending" or "descending".
    null_placement : str, default "at_end"
        Where to place nulls: "at_start" or "at_end".

    Returns
    -------
    PyArrowArray
        Indices that would sort the array.
    """
    return pc.array_sort_indices(arr, order=order, null_placement=null_placement)


# =============================================================================
# Take/Gather Operations
# =============================================================================


@register_kernel("take", "arrow")
def arrow_take(
    arr: PyArrowArray,
    indices: PyArrowArray,
) -> PyArrowArray:
    """
    Select elements from array by indices.

    Parameters
    ----------
    arr : PyArrowArray
        Source array.
    indices : PyArrowArray
        Indices to select.

    Returns
    -------
    PyArrowArray
        Elements at the specified indices.
    """
    return pc.take(arr, indices)


# =============================================================================
# Unique/Distinct Operations
# =============================================================================


@register_kernel("unique", "arrow")
def arrow_unique(arr: PyArrowArray) -> PyArrowArray:
    """
    Return unique values in the array.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.

    Returns
    -------
    PyArrowArray
        Unique values.
    """
    return pc.unique(arr)


@register_kernel("dictionary_encode", "arrow")
def arrow_dictionary_encode(arr: PyArrowArray) -> PyArrowArray:
    """
    Dictionary-encode an array (useful for deduplication).

    Parameters
    ----------
    arr : PyArrowArray
        Input array.

    Returns
    -------
    PyArrowArray
        Dictionary-encoded array.
    """
    return pc.dictionary_encode(arr)


# =============================================================================
# Set Membership Operations
# =============================================================================


@register_kernel("is_in", "arrow")
def arrow_is_in(
    arr: PyArrowArray,
    value_set: PyArrowArray,
) -> PyArrowArray:
    """
    Check if elements are in a set of values.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    value_set : PyArrowArray
        Set of values to check membership against.

    Returns
    -------
    PyArrowArray
        Boolean array indicating membership.
    """
    return pc.is_in(arr, value_set)


@register_kernel("index_in", "arrow")
def arrow_index_in(
    arr: PyArrowArray,
    value_set: PyArrowArray,
) -> PyArrowArray:
    """
    Return the index of each element in value_set, or null if not found.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    value_set : PyArrowArray
        Set of values to find indices for.

    Returns
    -------
    PyArrowArray
        Index of each element in value_set, or null.
    """
    return pc.index_in(arr, value_set)


# =============================================================================
# Ranking Operations (for TopK)
# =============================================================================


@register_kernel("rank", "arrow")
def arrow_rank(
    arr: PyArrowArray,
    *,
    sort_keys: str = "ascending",
    null_placement: str = "at_end",
    tiebreaker: str = "min",
) -> PyArrowArray:
    """
    Compute numerical rank of each element.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    sort_keys : str, default "ascending"
        Sort order for ranking.
    null_placement : str, default "at_end"
        Where to place nulls.
    tiebreaker : str, default "min"
        How to handle ties: "min", "max", "first", "dense".

    Returns
    -------
    PyArrowArray
        Ranks as uint64 array.
    """
    return pc.rank(
        arr, sort_keys=sort_keys, null_placement=null_placement, tiebreaker=tiebreaker
    )


@register_kernel("select_k_unstable", "arrow")
def arrow_select_k_unstable(
    arr: PyArrowArray,
    k: int,
    *,
    sort_keys: list[tuple[str, str]] | None = None,
) -> PyArrowArray:
    """
    Select k smallest or largest elements (indices).

    Note: This operates on a table/struct, so for single arrays,
    use sort_indices + take with slicing.

    Parameters
    ----------
    arr : PyArrowArray
        Input array (will be wrapped in a table).
    k : int
        Number of elements to select.
    sort_keys : list of tuples, optional
        List of (name, order) tuples for sorting.

    Returns
    -------
    PyArrowArray
        Indices of selected elements.
    """
    # For a single array, we can use sort_indices and slice
    order = "ascending"
    if sort_keys and len(sort_keys) > 0:
        order = sort_keys[0][1]
    indices = pc.array_sort_indices(arr, order=order)
    return indices[:k]


# =============================================================================
# Cast/Type Conversion Operations
# =============================================================================


@register_kernel("cast", "arrow")
def arrow_cast(
    arr: PyArrowArray,
    target_type: pa.DataType,
    *,
    safe: bool = True,
) -> PyArrowArray:
    """
    Cast array to a different type.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    target_type : pa.DataType
        Target Arrow data type.
    safe : bool, default True
        If True, check for overflows and invalid casts.

    Returns
    -------
    PyArrowArray
        Cast array.
    """
    return pc.cast(arr, target_type, safe=safe)


# =============================================================================
# If-Else / Case-When Operations
# =============================================================================


@register_kernel("if_else", "arrow")
def arrow_if_else(
    condition: PyArrowArray,
    true_value: PyArrowArray,
    false_value: PyArrowArray,
) -> PyArrowArray:
    """
    Choose elements based on condition.

    Parameters
    ----------
    condition : PyArrowArray
        Boolean condition array.
    true_value : PyArrowArray
        Values to use where condition is True.
    false_value : PyArrowArray
        Values to use where condition is False.

    Returns
    -------
    PyArrowArray
        Result array.
    """
    return pc.if_else(condition, true_value, false_value)


@register_kernel("case_when", "arrow")
def arrow_case_when(
    *args,
) -> PyArrowArray:
    """
    Case-when expression (multiple conditions).

    Parameters
    ----------
    *args : PyArrowArray
        Alternating (condition, value) pairs, ending with a default value.
        E.g., (cond1, val1, cond2, val2, default)

    Returns
    -------
    PyArrowArray
        Result array.
    """
    return pc.case_when(*args)


# =============================================================================
# GroupBy / Aggregation Operations
# =============================================================================


@register_kernel("group_by", "arrow")
def arrow_group_by(
    table: pa.Table,
    group_keys: list[str],
    aggregations: list[tuple[str, str, str]],
) -> pa.Table:
    """
    Perform grouped aggregation on an Arrow table.

    Parameters
    ----------
    table : pa.Table
        Input table to aggregate.
    group_keys : list[str]
        Column names to group by.
    aggregations : list[tuple[str, str, str]]
        List of (output_name, input_column, agg_function) tuples.
        Supported agg_function values: "sum", "mean", "min", "max", "count",
        "first", "last", "std", "var", "any", "all", "count_distinct".

    Returns
    -------
    pa.Table
        Aggregated table with group keys and aggregated values.
    """
    # Map pandas aggregation names to PyArrow hash aggregation functions
    # Note: PyArrow strips "hash_" prefix in output column names
    agg_map = {
        "sum": ("hash_sum", "sum"),
        "mean": ("hash_mean", "mean"),
        "min": ("hash_min", "min"),
        "max": ("hash_max", "max"),
        "count": ("hash_count", "count"),
        "first": ("hash_first", "first"),
        "last": ("hash_last", "last"),
        "std": ("hash_stddev", "stddev"),
        "var": ("hash_variance", "variance"),
        "any": ("hash_any", "any"),
        "all": ("hash_all", "all"),
        "count_distinct": ("hash_count_distinct", "count_distinct"),
        "nunique": ("hash_count_distinct", "count_distinct"),
        "prod": ("hash_product", "product"),
        "list": ("hash_list", "list"),
    }

    # Check if we need single-threaded execution (for first/last)
    needs_single_thread = any(
        agg_func in ("first", "last") for _, _, agg_func in aggregations
    )

    # Build aggregation specs for pyarrow
    agg_specs = []
    for output_name, input_col, agg_func in aggregations:
        pa_func, _ = agg_map.get(agg_func, (f"hash_{agg_func}", agg_func))
        agg_specs.append((input_col, pa_func, pc.ScalarAggregateOptions()))

    # Use PyArrow's group_by functionality
    grouped = table.group_by(group_keys, use_threads=not needs_single_thread)

    # Apply aggregations
    result = grouped.aggregate(agg_specs)

    # Rename columns to match expected output names
    new_names = list(group_keys)
    for output_name, input_col, agg_func in aggregations:
        new_names.append(output_name)

    # Rename columns if needed
    if list(result.column_names) != new_names:
        result = result.rename_columns(new_names)

    return result


@register_kernel("hash_aggregate", "arrow")
def arrow_hash_aggregate(
    keys: list[PyArrowArray],
    values: list[PyArrowArray],
    agg_funcs: list[str],
) -> tuple[list[PyArrowArray], list[PyArrowArray]]:
    """
    Low-level hash-based aggregation on arrays.

    Parameters
    ----------
    keys : list[PyArrowArray]
        Arrays to group by.
    values : list[PyArrowArray]
        Arrays to aggregate.
    agg_funcs : list[str]
        Aggregation function for each value array.

    Returns
    -------
    tuple[list[PyArrowArray], list[PyArrowArray]]
        (unique_keys, aggregated_values)
    """
    # Build a table from the arrays
    columns = {}
    key_names = [f"__key_{i}__" for i in range(len(keys))]
    val_names = [f"__val_{i}__" for i in range(len(values))]

    columns = dict(zip(key_names, keys, strict=True))
    columns.update(dict(zip(val_names, values, strict=True)))

    table = pa.table(columns)

    # Build aggregation specs
    # Map: agg_name -> (pa_function, output_suffix)
    # PyArrow strips "hash_" prefix in output column names
    agg_map = {
        "sum": ("hash_sum", "sum"),
        "mean": ("hash_mean", "mean"),
        "min": ("hash_min", "min"),
        "max": ("hash_max", "max"),
        "count": ("hash_count", "count"),
        "first": ("hash_first", "first"),
        "last": ("hash_last", "last"),
        "std": ("hash_stddev", "stddev"),
        "var": ("hash_variance", "variance"),
        "any": ("hash_any", "any"),
        "all": ("hash_all", "all"),
        "count_distinct": ("hash_count_distinct", "count_distinct"),
        "nunique": ("hash_count_distinct", "count_distinct"),
        "prod": ("hash_product", "product"),
    }

    # Check if we need single-threaded execution
    needs_single_thread = any(f in ("first", "last") for f in agg_funcs)

    agg_specs = [
        (
            val_name,
            agg_map.get(agg_func, (f"hash_{agg_func}", agg_func))[0],
            pc.ScalarAggregateOptions(),
        )
        for val_name, agg_func in zip(val_names, agg_funcs, strict=True)
    ]

    # Execute grouped aggregation
    grouped = table.group_by(key_names, use_threads=not needs_single_thread)
    result = grouped.aggregate(agg_specs)

    # Extract results
    result_keys = [result.column(name) for name in key_names]
    result_vals = [
        result.column(
            f"{val_name}_{agg_map.get(agg_func, (f'hash_{agg_func}', agg_func))[1]}"
        )
        for val_name, agg_func in zip(val_names, agg_funcs, strict=True)
    ]

    return result_keys, result_vals


# Single-column groupby aggregations (simpler API)
# Note: PyArrow strips "hash_" prefix from column names, e.g. "hash_sum" -> "value_sum"


@register_kernel("groupby_sum", "arrow")
def arrow_groupby_sum(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Sum values grouped by keys.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, summed_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_sum")])
    return result.column("key"), result.column("value_sum")


@register_kernel("groupby_mean", "arrow")
def arrow_groupby_mean(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Mean of values grouped by keys.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, mean_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_mean")])
    return result.column("key"), result.column("value_mean")


@register_kernel("groupby_min", "arrow")
def arrow_groupby_min(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Min of values grouped by keys.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, min_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_min")])
    return result.column("key"), result.column("value_min")


@register_kernel("groupby_max", "arrow")
def arrow_groupby_max(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Max of values grouped by keys.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, max_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_max")])
    return result.column("key"), result.column("value_max")


@register_kernel("groupby_count", "arrow")
def arrow_groupby_count(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Count of values grouped by keys.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, count_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_count")])
    return result.column("key"), result.column("value_count")


@register_kernel("groupby_first", "arrow")
def arrow_groupby_first(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    First value in each group.

    Note: Uses single-threaded execution as hash_first requires ordered execution.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, first_values)
    """
    table = pa.table({"key": keys, "value": values})
    # hash_first requires single-threaded execution
    result = table.group_by("key", use_threads=False).aggregate(
        [("value", "hash_first")]
    )
    return result.column("key"), result.column("value_first")


@register_kernel("groupby_last", "arrow")
def arrow_groupby_last(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Last value in each group.

    Note: Uses single-threaded execution as hash_last requires ordered execution.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, last_values)
    """
    table = pa.table({"key": keys, "value": values})
    # hash_last requires single-threaded execution
    result = table.group_by("key", use_threads=False).aggregate(
        [("value", "hash_last")]
    )
    return result.column("key"), result.column("value_last")


@register_kernel("groupby_std", "arrow")
def arrow_groupby_std(
    keys: PyArrowArray, values: PyArrowArray, *, ddof: int = 1
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Standard deviation of values grouped by keys.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, std_values)
    """
    table = pa.table({"key": keys, "value": values})
    options = pc.VarianceOptions(ddof=ddof)
    result = table.group_by("key").aggregate([("value", "hash_stddev", options)])
    return result.column("key"), result.column("value_stddev")


@register_kernel("groupby_var", "arrow")
def arrow_groupby_var(
    keys: PyArrowArray, values: PyArrowArray, *, ddof: int = 1
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Variance of values grouped by keys.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, var_values)
    """
    table = pa.table({"key": keys, "value": values})
    options = pc.VarianceOptions(ddof=ddof)
    result = table.group_by("key").aggregate([("value", "hash_variance", options)])
    return result.column("key"), result.column("value_variance")


@register_kernel("groupby_nunique", "arrow")
def arrow_groupby_nunique(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Count distinct values in each group.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, nunique_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_count_distinct")])
    return result.column("key"), result.column("value_count_distinct")


@register_kernel("groupby_any", "arrow")
def arrow_groupby_any(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Any True value in each group.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, any_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_any")])
    return result.column("key"), result.column("value_hash_any")


@register_kernel("groupby_all", "arrow")
def arrow_groupby_all(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    All True values in each group.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, all_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_all")])
    return result.column("key"), result.column("value_hash_all")


@register_kernel("groupby_prod", "arrow")
def arrow_groupby_prod(
    keys: PyArrowArray, values: PyArrowArray
) -> tuple[PyArrowArray, PyArrowArray]:
    """
    Product of values grouped by keys.

    Returns
    -------
    tuple[PyArrowArray, PyArrowArray]
        (unique_keys, prod_values)
    """
    table = pa.table({"key": keys, "value": values})
    result = table.group_by("key").aggregate([("value", "hash_product")])
    return result.column("key"), result.column("value_hash_product")


# =============================================================================
# Join Operations
# =============================================================================


@register_kernel("hash_join", "arrow")
def arrow_hash_join(
    left_table: pa.Table,
    right_table: pa.Table,
    *,
    keys: list[str] | None = None,
    left_keys: list[str] | None = None,
    right_keys: list[str] | None = None,
    join_type: str = "inner",
    left_suffix: str = "",
    right_suffix: str = "_right",
    coalesce_keys: bool = True,
) -> pa.Table:
    """
    Perform a hash join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str] or None
        Column names to join on (used for both tables).
    left_keys : list[str] or None
        Column names from left table to join on.
    right_keys : list[str] or None
        Column names from right table to join on.
    join_type : str, default "inner"
        Type of join: "inner", "left", "right", "outer", "left semi",
        "right semi", "left anti", "right anti".
    left_suffix : str, default ""
        Suffix to add to left column names for disambiguation.
    right_suffix : str, default "_right"
        Suffix to add to right column names for disambiguation.
    coalesce_keys : bool, default True
        Whether to coalesce duplicate key columns.

    Returns
    -------
    pa.Table
        Result of the join operation.
    """
    # Map pandas join types to PyArrow join types
    join_type_map = {
        "inner": "inner",
        "left": "left outer",
        "right": "right outer",
        "outer": "full outer",
        "cross": None,  # Special case - handled separately
        "left_semi": "left semi",
        "right_semi": "right semi",
        "left_anti": "left anti",
        "right_anti": "right anti",
    }

    pa_join_type = join_type_map.get(join_type, join_type)

    if pa_join_type is None:
        raise ValueError(f"Join type '{join_type}' not supported by Arrow kernel")

    # Determine keys
    if keys is not None:
        join_keys = keys if isinstance(keys, list) else [keys]
        right_join_keys = None
    else:
        join_keys = left_keys if isinstance(left_keys, list) else [left_keys]
        right_join_keys = right_keys if isinstance(right_keys, list) else [right_keys]

    # Perform join
    result = left_table.join(
        right_table,
        keys=join_keys,
        right_keys=right_join_keys,
        join_type=pa_join_type,
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=coalesce_keys,
    )

    return result


@register_kernel("inner_join", "arrow")
def arrow_inner_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> pa.Table:
    """
    Perform an inner join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    pa.Table
        Result of the inner join.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="inner",
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=True,
    )


@register_kernel("left_join", "arrow")
def arrow_left_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> pa.Table:
    """
    Perform a left outer join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    pa.Table
        Result of the left join.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="left outer",
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=True,
    )


@register_kernel("right_join", "arrow")
def arrow_right_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> pa.Table:
    """
    Perform a right outer join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    pa.Table
        Result of the right join.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="right outer",
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=True,
    )


@register_kernel("outer_join", "arrow")
def arrow_outer_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> pa.Table:
    """
    Perform a full outer join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    pa.Table
        Result of the full outer join.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="full outer",
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=True,
    )


@register_kernel("semi_join", "arrow")
def arrow_semi_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
) -> pa.Table:
    """
    Perform a left semi join (filter left by existence in right).

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.

    Returns
    -------
    pa.Table
        Rows from left table that have matches in right table.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="left semi",
        coalesce_keys=True,
    )


@register_kernel("anti_join", "arrow")
def arrow_anti_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
) -> pa.Table:
    """
    Perform a left anti join (filter left by non-existence in right).

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.

    Returns
    -------
    pa.Table
        Rows from left table that have no matches in right table.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="left anti",
        coalesce_keys=True,
    )


# =============================================================================
# Cumulative / Window Functions
# =============================================================================


@register_kernel("cumulative_sum", "arrow")
def arrow_cumulative_sum(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative sum.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative sum array.
    """
    return pc.cumulative_sum(arr, skip_nulls=skip_nulls)


@register_kernel("cumulative_max", "arrow")
def arrow_cumulative_max(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative maximum.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative max array.
    """
    return pc.cumulative_max(arr, skip_nulls=skip_nulls)


@register_kernel("cumulative_min", "arrow")
def arrow_cumulative_min(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative minimum.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative min array.
    """
    return pc.cumulative_min(arr, skip_nulls=skip_nulls)


@register_kernel("cumulative_prod", "arrow")
def arrow_cumulative_prod(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative product.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative product array.
    """
    return pc.cumulative_prod(arr, skip_nulls=skip_nulls)


@register_kernel("cumulative_mean", "arrow")
def arrow_cumulative_mean(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative mean.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative mean array.
    """
    return pc.cumulative_mean(arr, skip_nulls=skip_nulls)


# =============================================================================
# Datetime Extraction Functions
# =============================================================================


@register_kernel("dt_year", "arrow")
def arrow_dt_year(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract year from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of years (int64).
    """
    return pc.year(arr)


@register_kernel("dt_month", "arrow")
def arrow_dt_month(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract month from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of months (1-12).
    """
    return pc.month(arr)


@register_kernel("dt_day", "arrow")
def arrow_dt_day(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract day of month from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of days (1-31).
    """
    return pc.day(arr)


@register_kernel("dt_hour", "arrow")
def arrow_dt_hour(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract hour from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of hours (0-23).
    """
    return pc.hour(arr)


@register_kernel("dt_minute", "arrow")
def arrow_dt_minute(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract minute from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of minutes (0-59).
    """
    return pc.minute(arr)


@register_kernel("dt_second", "arrow")
def arrow_dt_second(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract second from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of seconds (0-59).
    """
    return pc.second(arr)


@register_kernel("dt_microsecond", "arrow")
def arrow_dt_microsecond(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract microsecond from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of microseconds (0-999999).
    """
    return pc.microsecond(arr)


@register_kernel("dt_nanosecond", "arrow")
def arrow_dt_nanosecond(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract nanosecond from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of nanoseconds (0-999999999).
    """
    return pc.nanosecond(arr)


@register_kernel("dt_day_of_week", "arrow")
def arrow_dt_day_of_week(
    arr: PyArrowArray, count_from_zero: bool = True
) -> PyArrowArray:
    """
    Extract day of week from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.
    count_from_zero : bool, default True
        If True, returns 0-6 (Monday-Sunday).
        If False, returns 1-7 (Monday-Sunday).

    Returns
    -------
    PyArrowArray
        Array of day of week values.
    """
    return pc.day_of_week(arr, count_from_zero=count_from_zero)


@register_kernel("dt_day_of_year", "arrow")
def arrow_dt_day_of_year(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract day of year from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of day of year values (1-366).
    """
    return pc.day_of_year(arr)


@register_kernel("dt_week", "arrow")
def arrow_dt_week(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract ISO week number from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of week numbers (1-53).
    """
    return pc.iso_week(arr)


@register_kernel("dt_quarter", "arrow")
def arrow_dt_quarter(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract quarter from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of quarters (1-4).
    """
    return pc.quarter(arr)


@register_kernel("dt_is_leap_year", "arrow")
def arrow_dt_is_leap_year(arr: PyArrowArray) -> PyArrowArray:
    """
    Check if datetime values fall in a leap year.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Boolean array indicating leap year.
    """
    return pc.is_leap_year(arr)


@register_kernel("dt_iso_year", "arrow")
def arrow_dt_iso_year(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract ISO year from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of ISO years.
    """
    return pc.iso_year(arr)


@register_kernel("dt_strftime", "arrow")
def arrow_dt_strftime(arr: PyArrowArray, format: str) -> PyArrowArray:
    """
    Format datetime array to strings.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.
    format : str
        strftime format string.

    Returns
    -------
    PyArrowArray
        String array of formatted datetimes.
    """
    return pc.strftime(arr, format=format)


# =============================================================================
# Additional String Functions
# =============================================================================


@register_kernel("str_capitalize", "arrow")
def arrow_str_capitalize(arr: PyArrowArray) -> PyArrowArray:
    """Capitalize strings (first char uppercase, rest lowercase)."""
    return pc.utf8_capitalize(arr)


@register_kernel("str_title", "arrow")
def arrow_str_title(arr: PyArrowArray) -> PyArrowArray:
    """Title-case strings (first char of each word uppercase)."""
    return pc.utf8_title(arr)


@register_kernel("str_swapcase", "arrow")
def arrow_str_swapcase(arr: PyArrowArray) -> PyArrowArray:
    """Swap case of strings (upper to lower, lower to upper)."""
    return pc.utf8_swapcase(arr)


@register_kernel("str_reverse", "arrow")
def arrow_str_reverse(arr: PyArrowArray) -> PyArrowArray:
    """Reverse strings."""
    return pc.utf8_reverse(arr)


@register_kernel("str_is_alnum", "arrow")
def arrow_str_is_alnum(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings contain only alphanumeric characters."""
    return pc.utf8_is_alnum(arr)


@register_kernel("str_is_alpha", "arrow")
def arrow_str_is_alpha(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings contain only alphabetic characters."""
    return pc.utf8_is_alpha(arr)


@register_kernel("str_is_decimal", "arrow")
def arrow_str_is_decimal(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings contain only decimal characters."""
    return pc.utf8_is_decimal(arr)


@register_kernel("str_is_digit", "arrow")
def arrow_str_is_digit(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings contain only digit characters."""
    return pc.utf8_is_digit(arr)


@register_kernel("str_is_lower", "arrow")
def arrow_str_is_lower(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings are all lowercase."""
    return pc.utf8_is_lower(arr)


@register_kernel("str_is_upper", "arrow")
def arrow_str_is_upper(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings are all uppercase."""
    return pc.utf8_is_upper(arr)


@register_kernel("str_is_numeric", "arrow")
def arrow_str_is_numeric(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings contain only numeric characters."""
    return pc.utf8_is_numeric(arr)


@register_kernel("str_is_space", "arrow")
def arrow_str_is_space(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings contain only whitespace."""
    return pc.utf8_is_space(arr)


@register_kernel("str_is_title", "arrow")
def arrow_str_is_title(arr: PyArrowArray) -> PyArrowArray:
    """Check if strings are title-cased."""
    return pc.utf8_is_title(arr)


@register_kernel("str_count", "arrow")
def arrow_str_count(
    arr: PyArrowArray, pattern: str, regex: bool = False
) -> PyArrowArray:
    """
    Count occurrences of a pattern in strings.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    pattern : str
        Pattern to count.
    regex : bool, default False
        Whether to treat pattern as regex.

    Returns
    -------
    PyArrowArray
        Integer array of counts.
    """
    if regex:
        return pc.count_substring_regex(arr, pattern=pattern)
    return pc.count_substring(arr, pattern=pattern)


@register_kernel("str_find", "arrow")
def arrow_str_find(
    arr: PyArrowArray, pattern: str, regex: bool = False
) -> PyArrowArray:
    """
    Find first occurrence of pattern in strings.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    pattern : str
        Pattern to find.
    regex : bool, default False
        Whether to treat pattern as regex.

    Returns
    -------
    PyArrowArray
        Integer array of positions (-1 if not found).
    """
    if regex:
        return pc.find_substring_regex(arr, pattern=pattern)
    return pc.find_substring(arr, pattern=pattern)


@register_kernel("str_pad", "arrow")
def arrow_str_pad(
    arr: PyArrowArray, width: int, side: str = "left", fillchar: str = " "
) -> PyArrowArray:
    """
    Pad strings to a specified width.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    width : int
        Target width.
    side : str, default "left"
        Side to pad ("left", "right", or "both").
    fillchar : str, default " "
        Character to use for padding.

    Returns
    -------
    PyArrowArray
        Padded string array.
    """
    if side == "left":
        return pc.utf8_lpad(arr, width, padding=fillchar)
    elif side == "right":
        return pc.utf8_rpad(arr, width, padding=fillchar)
    else:
        # "both" - center
        return pc.utf8_center(arr, width, padding=fillchar)


@register_kernel("str_center", "arrow")
def arrow_str_center(
    arr: PyArrowArray, width: int, fillchar: str = " "
) -> PyArrowArray:
    """
    Center strings in a field of given width.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    width : int
        Target width.
    fillchar : str, default " "
        Character to use for padding.

    Returns
    -------
    PyArrowArray
        Centered string array.
    """
    return pc.utf8_center(arr, width, padding=fillchar)


@register_kernel("str_zfill", "arrow")
def arrow_str_zfill(arr: PyArrowArray, width: int) -> PyArrowArray:
    """
    Pad strings with zeros on the left.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    width : int
        Target width.

    Returns
    -------
    PyArrowArray
        Zero-padded string array.
    """
    return pc.utf8_zfill(arr, width)


@register_kernel("str_match", "arrow")
def arrow_str_match(
    arr: PyArrowArray, pattern: str, regex: bool = True
) -> PyArrowArray:
    """
    Check if strings match a pattern.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    pattern : str
        Pattern to match.
    regex : bool, default True
        Whether to treat pattern as regex.

    Returns
    -------
    PyArrowArray
        Boolean array indicating matches.
    """
    if regex:
        return pc.match_substring_regex(arr, pattern=pattern)
    return pc.match_substring(arr, pattern=pattern)


# =============================================================================
# Rolling Window Kernels
# =============================================================================


@register_kernel("rolling_sum", "arrow")
def arrow_rolling_sum(
    arr: PyArrowArray, window: int, min_periods: int | None = None
) -> PyArrowArray:
    """
    Calculate rolling sum over a window.

    Parameters
    ----------
    arr : PyArrowArray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    PyArrowArray
        Rolling sum values.
    """
    if min_periods is None:
        min_periods = window

    # Convert to numpy for rolling calculation
    np_arr = arr.to_numpy(zero_copy_only=False).astype(float)
    n = len(np_arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = np_arr[start : i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        if valid_count >= min_periods:
            result[i] = np.nansum(window_data)

    return pa.array(result)


@register_kernel("rolling_mean", "arrow")
def arrow_rolling_mean(
    arr: PyArrowArray, window: int, min_periods: int | None = None
) -> PyArrowArray:
    """
    Calculate rolling mean over a window.

    Parameters
    ----------
    arr : PyArrowArray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    PyArrowArray
        Rolling mean values.
    """
    if min_periods is None:
        min_periods = window

    np_arr = arr.to_numpy(zero_copy_only=False).astype(float)
    n = len(np_arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = np_arr[start : i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        if valid_count >= min_periods:
            result[i] = np.nanmean(window_data)

    return pa.array(result)


@register_kernel("rolling_min", "arrow")
def arrow_rolling_min(
    arr: PyArrowArray, window: int, min_periods: int | None = None
) -> PyArrowArray:
    """
    Calculate rolling minimum over a window.

    Parameters
    ----------
    arr : PyArrowArray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    PyArrowArray
        Rolling minimum values.
    """
    if min_periods is None:
        min_periods = window

    np_arr = arr.to_numpy(zero_copy_only=False).astype(float)
    n = len(np_arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = np_arr[start : i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        if valid_count >= min_periods:
            result[i] = np.nanmin(window_data)

    return pa.array(result)


@register_kernel("rolling_max", "arrow")
def arrow_rolling_max(
    arr: PyArrowArray, window: int, min_periods: int | None = None
) -> PyArrowArray:
    """
    Calculate rolling maximum over a window.

    Parameters
    ----------
    arr : PyArrowArray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    PyArrowArray
        Rolling maximum values.
    """
    if min_periods is None:
        min_periods = window

    np_arr = arr.to_numpy(zero_copy_only=False).astype(float)
    n = len(np_arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = np_arr[start : i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        if valid_count >= min_periods:
            result[i] = np.nanmax(window_data)

    return pa.array(result)


@register_kernel("rolling_std", "arrow")
def arrow_rolling_std(
    arr: PyArrowArray, window: int, min_periods: int | None = None, ddof: int = 1
) -> PyArrowArray:
    """
    Calculate rolling standard deviation over a window.

    Parameters
    ----------
    arr : PyArrowArray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.
    ddof : int, default 1
        Delta degrees of freedom.

    Returns
    -------
    PyArrowArray
        Rolling standard deviation values.
    """
    if min_periods is None:
        min_periods = window

    np_arr = arr.to_numpy(zero_copy_only=False).astype(float)
    n = len(np_arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = np_arr[start : i + 1]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) >= min_periods and len(valid_data) > ddof:
            result[i] = np.std(valid_data, ddof=ddof)

    return pa.array(result)


@register_kernel("rolling_var", "arrow")
def arrow_rolling_var(
    arr: PyArrowArray, window: int, min_periods: int | None = None, ddof: int = 1
) -> PyArrowArray:
    """
    Calculate rolling variance over a window.

    Parameters
    ----------
    arr : PyArrowArray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.
    ddof : int, default 1
        Delta degrees of freedom.

    Returns
    -------
    PyArrowArray
        Rolling variance values.
    """
    if min_periods is None:
        min_periods = window

    np_arr = arr.to_numpy(zero_copy_only=False).astype(float)
    n = len(np_arr)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = np_arr[start : i + 1]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) >= min_periods and len(valid_data) > ddof:
            result[i] = np.var(valid_data, ddof=ddof)

    return pa.array(result)


# =============================================================================
# Shift/Lag/Lead Kernels
# =============================================================================


@register_kernel("shift", "arrow")
def arrow_shift(
    arr: PyArrowArray, periods: int = 1, fill_value: Any = None
) -> PyArrowArray:
    """
    Shift array values by specified number of periods.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    periods : int, default 1
        Number of periods to shift. Positive shifts forward, negative shifts backward.
    fill_value : Any, default None
        Value to use for filling gaps.

    Returns
    -------
    PyArrowArray
        Shifted array.
    """
    np_arr = arr.to_numpy(zero_copy_only=False)
    n = len(np_arr)
    dtype = np_arr.dtype

    # Determine fill value
    if fill_value is None:
        if np.issubdtype(dtype, np.floating):
            fill_val = np.nan
        elif np.issubdtype(dtype, np.integer):
            # Convert to float to support NaN
            np_arr = np_arr.astype(float)
            fill_val = np.nan
        else:
            fill_val = None
    else:
        fill_val = fill_value

    result = np.empty(n, dtype=np_arr.dtype if fill_val is not None else object)

    if periods > 0:
        result[:periods] = fill_val
        result[periods:] = np_arr[:-periods] if periods < n else []
    elif periods < 0:
        result[periods:] = fill_val
        result[:periods] = np_arr[-periods:] if -periods < n else []
    else:
        result = np_arr.copy()

    return pa.array(result)


@register_kernel("lag", "arrow")
def arrow_lag(
    arr: PyArrowArray, periods: int = 1, fill_value: Any = None
) -> PyArrowArray:
    """
    Get lagged values (shift forward).

    This is an alias for shift with positive periods.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    periods : int, default 1
        Number of periods to lag.
    fill_value : Any, default None
        Value to use for filling gaps.

    Returns
    -------
    PyArrowArray
        Lagged array.
    """
    return arrow_shift(arr, periods=abs(periods), fill_value=fill_value)


@register_kernel("lead", "arrow")
def arrow_lead(
    arr: PyArrowArray, periods: int = 1, fill_value: Any = None
) -> PyArrowArray:
    """
    Get lead values (shift backward).

    This is an alias for shift with negative periods.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    periods : int, default 1
        Number of periods to lead.
    fill_value : Any, default None
        Value to use for filling gaps.

    Returns
    -------
    PyArrowArray
        Lead array.
    """
    return arrow_shift(arr, periods=-abs(periods), fill_value=fill_value)


# =============================================================================
# Fill NA Variant Kernels
# =============================================================================


@register_kernel("ffill", "arrow")
def arrow_ffill(arr: PyArrowArray, limit: int | None = None) -> PyArrowArray:
    """
    Forward fill missing values.

    Parameters
    ----------
    arr : PyArrowArray
        Input array with potential missing values.
    limit : int or None
        Maximum number of consecutive NaN values to forward fill.

    Returns
    -------
    PyArrowArray
        Array with forward-filled values.
    """
    # PyArrow has fill_null_forward
    if limit is None:
        return pc.fill_null_forward(arr)

    # With limit, need manual implementation
    np_arr = arr.to_numpy(zero_copy_only=False)
    result = np_arr.copy()
    n = len(result)

    last_valid = None
    consecutive_nulls = 0

    for i in range(n):
        if pd.isna(result[i]):
            consecutive_nulls += 1
            if last_valid is not None and consecutive_nulls <= limit:
                result[i] = last_valid
        else:
            last_valid = result[i]
            consecutive_nulls = 0

    return pa.array(result)


@register_kernel("bfill", "arrow")
def arrow_bfill(arr: PyArrowArray, limit: int | None = None) -> PyArrowArray:
    """
    Backward fill missing values.

    Parameters
    ----------
    arr : PyArrowArray
        Input array with potential missing values.
    limit : int or None
        Maximum number of consecutive NaN values to backward fill.

    Returns
    -------
    PyArrowArray
        Array with backward-filled values.
    """
    # PyArrow has fill_null_backward
    if limit is None:
        return pc.fill_null_backward(arr)

    # With limit, need manual implementation
    np_arr = arr.to_numpy(zero_copy_only=False)
    result = np_arr.copy()
    n = len(result)

    last_valid = None
    consecutive_nulls = 0

    for i in range(n - 1, -1, -1):
        if pd.isna(result[i]):
            consecutive_nulls += 1
            if last_valid is not None and consecutive_nulls <= limit:
                result[i] = last_valid
        else:
            last_valid = result[i]
            consecutive_nulls = 0

    return pa.array(result)


@register_kernel("interpolate_linear", "arrow")
def arrow_interpolate_linear(arr: PyArrowArray) -> PyArrowArray:
    """
    Linearly interpolate missing values.

    Parameters
    ----------
    arr : PyArrowArray
        Input numeric array with potential missing values.

    Returns
    -------
    PyArrowArray
        Array with linearly interpolated values.
    """
    np_arr = arr.to_numpy(zero_copy_only=False).astype(float)
    result = np_arr.copy()
    n = len(result)

    # Find indices of valid values
    mask = ~np.isnan(result)
    if not mask.any():
        return pa.array(result)

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

    return pa.array(result)
