"""
Arrow additional String kernel implementations.

This module contains additional string operations beyond the core.
"""

import pyarrow.compute as pc

from pandas.lazy.backends import register_kernel
from pandas.lazy.backends.types import PyArrowArray

# Additional String Functions
# =============================================================================


@register_kernel("str_split", "arrow")
def arrow_str_split(
    arr: PyArrowArray, pattern: str = " ", n: int = -1, regex: bool = False
) -> PyArrowArray:
    """
    Split strings by pattern.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    pattern : str, default " "
        String or regex pattern to split on.
    n : int, default -1
        Maximum number of splits. -1 means no limit.
    regex : bool, default False
        Whether to treat pattern as regex.

    Returns
    -------
    PyArrowArray
        List array of split strings.
    """
    if regex:
        return pc.split_pattern_regex(
            arr, pattern=pattern, max_splits=n if n >= 0 else None
        )
    return pc.split_pattern(arr, pattern=pattern, max_splits=n if n >= 0 else None)


@register_kernel("str_rsplit", "arrow")
def arrow_str_rsplit(
    arr: PyArrowArray, pattern: str = " ", n: int = -1
) -> PyArrowArray:
    """
    Split strings by pattern from the right.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    pattern : str, default " "
        String pattern to split on.
    n : int, default -1
        Maximum number of splits. -1 means no limit.

    Returns
    -------
    PyArrowArray
        List array of split strings.

    Notes
    -----
    PyArrow doesn't have native rsplit; this falls back to forward split.
    """
    # PyArrow doesn't have rsplit directly - use forward split
    # TODO: Implement proper rsplit with reverse + split + reverse logic
    return pc.split_pattern(arr, pattern=pattern, max_splits=n if n >= 0 else None)


@register_kernel("str_repeat", "arrow")
def arrow_str_repeat(arr: PyArrowArray, repeats: int) -> PyArrowArray:
    """
    Repeat strings a specified number of times.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    repeats : int
        Number of times to repeat each string.

    Returns
    -------
    PyArrowArray
        Array with repeated strings.
    """
    return pc.binary_repeat(arr, repeats)


@register_kernel("str_get", "arrow")
def arrow_str_get(arr: PyArrowArray, index: int) -> PyArrowArray:
    """
    Extract character at specified position.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    index : int
        Position of character to extract (supports negative indexing).

    Returns
    -------
    PyArrowArray
        Array of single characters.
    """
    if index >= 0:
        return pc.utf8_slice_codeunits(arr, index, 1)
    else:
        # For negative indices, we need length - abs(index)
        lengths = pc.utf8_length(arr)
        # Use if_else to handle strings shorter than abs(index)
        start = pc.subtract(lengths, -index)
        # Clamp to 0 if negative
        start = pc.if_else(pc.less(start, 0), 0, start)
        return pc.utf8_slice_codeunits(arr, start, 1)


@register_kernel("str_ljust", "arrow")
def arrow_str_ljust(arr: PyArrowArray, width: int, fillchar: str = " ") -> PyArrowArray:
    """Left-justify strings in a field of given width."""
    return pc.utf8_rpad(arr, width, padding=fillchar)


@register_kernel("str_rjust", "arrow")
def arrow_str_rjust(arr: PyArrowArray, width: int, fillchar: str = " ") -> PyArrowArray:
    """Right-justify strings in a field of given width."""
    return pc.utf8_lpad(arr, width, padding=fillchar)


@register_kernel("str_normalize", "arrow")
def arrow_str_normalize(arr: PyArrowArray, form: str = "NFC") -> PyArrowArray:
    """
    Normalize Unicode strings.

    Parameters
    ----------
    arr : PyArrowArray
        Input string array.
    form : str, default "NFC"
        Unicode normalization form: "NFC", "NFKC", "NFD", or "NFKD".

    Returns
    -------
    PyArrowArray
        Normalized string array.
    """
    return pc.utf8_normalize(arr, form=form)


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
