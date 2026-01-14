"""
NumPy String kernel implementations.

This module contains additional string operations.
Note: Arrow is generally preferred for string ops - see router.py.
"""

import numpy as np

from pandas.lazy.backends import register_kernel

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
