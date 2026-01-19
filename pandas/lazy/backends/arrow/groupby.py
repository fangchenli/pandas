"""
Arrow GroupBy kernel implementations.

This module contains groupby aggregation operations using PyArrow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from pandas.lazy.backends import register_kernel

if TYPE_CHECKING:
    from pandas.lazy.backends.types import PyArrowArray

# Factorization
# =============================================================================


@register_kernel("factorize", "arrow")
def factorize(
    arr: pa.Array | pa.ChunkedArray,
) -> tuple[pa.Array, pa.Array, int]:
    """
    Factorize an Arrow array using dictionary encoding.

    Uses PyArrow's dictionary_encode which is implemented as a C++ hash table.
    This is faster than converting to pandas and using pandas' factorize.

    Parameters
    ----------
    arr : pa.Array or pa.ChunkedArray
        Input array to factorize.

    Returns
    -------
    tuple[pa.Array, pa.Array, int]
        - codes: Integer codes as Arrow array
        - uniques: Unique values as PyArrow array
        - n_uniques: Number of unique values
    """
    # Use PyArrow's dictionary_encode (C++ hash table)
    # Works directly on both Array and ChunkedArray
    dict_arr = pc.dictionary_encode(arr)

    # For ChunkedArray result, combine to get contiguous indices
    if isinstance(dict_arr, pa.ChunkedArray):
        dict_arr = dict_arr.combine_chunks()

    codes = dict_arr.indices
    uniques = dict_arr.dictionary
    return codes, uniques, len(uniques)


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
        "n_unique": ("hash_count_distinct", "count_distinct"),
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
        # Use appropriate options for each aggregation type
        # Note: hash_count_distinct should not receive ScalarAggregateOptions
        if agg_func == "count":
            # Count non-null values (matches pandas behavior)
            agg_specs.append((input_col, pa_func, pc.CountOptions(mode="only_valid")))
        elif agg_func in ("count_distinct", "nunique", "n_unique"):
            # count_distinct doesn't use ScalarAggregateOptions
            agg_specs.append((input_col, pa_func))
        else:
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
        "n_unique": ("hash_count_distinct", "count_distinct"),
        "prod": ("hash_product", "product"),
    }

    # Check if we need single-threaded execution
    needs_single_thread = any(f in ("first", "last") for f in agg_funcs)

    # Build aggregation specs with appropriate options
    agg_specs = []
    for val_name, agg_func in zip(val_names, agg_funcs, strict=True):
        pa_func = agg_map.get(agg_func, (f"hash_{agg_func}", agg_func))[0]
        # Use appropriate options for each aggregation type
        # Note: hash_count_distinct should not receive ScalarAggregateOptions
        if agg_func == "count":
            # Count non-null values (matches pandas behavior)
            agg_specs.append((val_name, pa_func, pc.CountOptions(mode="only_valid")))
        elif agg_func in ("count_distinct", "nunique", "n_unique"):
            # count_distinct doesn't use ScalarAggregateOptions
            agg_specs.append((val_name, pa_func))
        else:
            agg_specs.append((val_name, pa_func, pc.ScalarAggregateOptions()))

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
@register_kernel("groupby_n_unique", "arrow")
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
