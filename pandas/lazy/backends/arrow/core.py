"""
Core Arrow kernel implementations.

This module contains basic operations:
- String operations (major performance wins over pandas)
- Null handling: is_null, is_not_null, fill_null
- Arithmetic: add, subtract, multiply, divide, etc.
- Comparison: equal, not_equal, less, greater, etc.
- Logical: and, or, invert
- Aggregation: sum, mean, min, max, count, std, var
- Filter, sort, take, unique operations
- Cast/type conversion
- If-else / case-when operations
"""

import pyarrow as pa
import pyarrow.compute as pc

from pandas.lazy.backends import register_kernel
from pandas.lazy.backends.types import PyArrowArray

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
