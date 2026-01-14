"""
NumPy Window/Cumulative kernel implementations.

This module contains cumulative and window operations.

All functions are vectorized for performance.
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
        # Vectorized: replace NaN with 0 for cumsum, then forward-fill result
        filled = np.where(np.isnan(arr), 0.0, arr)
        return np.cumsum(filled)
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
        # Vectorized: replace NaN with -inf, then accumulate max
        filled = np.where(np.isnan(arr), -np.inf, arr)
        result = np.maximum.accumulate(filled)
        # Replace -inf (meaning no valid values seen yet) with NaN
        result = np.where(result == -np.inf, np.nan, result)
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
        # Vectorized: replace NaN with inf, then accumulate min
        filled = np.where(np.isnan(arr), np.inf, arr)
        result = np.minimum.accumulate(filled)
        # Replace inf (meaning no valid values seen yet) with NaN
        result = np.where(result == np.inf, np.nan, result)
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
        # Vectorized: replace NaN with 1 for cumprod (identity for multiplication)
        filled = np.where(np.isnan(arr), 1.0, arr)
        return np.cumprod(filled)
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
        # Vectorized: cumsum of values / cumsum of counts
        mask = ~np.isnan(arr)
        filled = np.where(mask, arr, 0.0)
        cum_sum = np.cumsum(filled)
        cum_count = np.cumsum(mask.astype(np.float64))
        # Avoid division by zero
        result = np.where(cum_count > 0, cum_sum / cum_count, np.nan)
        return result
    # For non-floating point or no skip_nulls: cumsum / position
    cum_sum = np.cumsum(arr.astype(np.float64))
    positions = np.arange(1, len(arr) + 1, dtype=np.float64)
    return cum_sum / positions


# =============================================================================
