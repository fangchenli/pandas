"""
NumPy Window/Cumulative kernel implementations.

This module contains cumulative and window operations.
"""

import numpy as np

from pandas.lazy.backends import register_kernel

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
