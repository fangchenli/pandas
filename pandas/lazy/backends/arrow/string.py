"""
Arrow additional String kernel implementations.

This module contains additional string operations beyond the core.
"""

import pyarrow.compute as pc

from pandas.lazy.backends import register_kernel
from pandas.lazy.backends.types import PyArrowArray

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
