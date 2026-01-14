"""
NumPy miscellaneous kernel implementations.

This module contains:
- Rolling window operations (vectorized using sliding_window_view)
- Shift/Lag/Lead operations
- Fill NA variants (vectorized)
"""

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pandas.lazy.backends import register_kernel

# =============================================================================
# Rolling Window Helper Functions (Vectorized)
# =============================================================================


def _rolling_apply_vectorized(
    arr: np.ndarray,
    window: int,
    min_periods: int,
    agg_func,
    handle_nan: bool = True,
) -> np.ndarray:
    """
    Vectorized rolling window aggregation.

    Uses sliding_window_view for O(n) performance instead of O(n*window).
    """
    n = len(arr)
    arr = arr.astype(float)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    # For the "ramp-up" period (first window-1 elements), we need special handling
    # Handle the initial ramp-up period where window is not full
    for i in range(min(window - 1, n)):
        window_data = arr[: i + 1]
        if handle_nan:
            valid_count = np.sum(~np.isnan(window_data))
            if valid_count >= min_periods:
                result[i] = agg_func(window_data)
        elif len(window_data) >= min_periods:
            result[i] = agg_func(window_data)

    # For the main portion, use sliding_window_view for vectorization
    if n >= window:
        windows = sliding_window_view(arr, window)
        # windows.shape is (n - window + 1, window)

        if handle_nan:
            # Count valid values per window
            valid_counts = np.sum(~np.isnan(windows), axis=1)
            # Compute aggregation for all windows
            agg_values = agg_func(windows, axis=1)
            # Apply min_periods mask
            mask = valid_counts >= min_periods
            result[window - 1 :] = np.where(mask, agg_values, np.nan)
        else:
            result[window - 1 :] = agg_func(windows, axis=1)

    return result


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
    return _rolling_apply_vectorized(arr, window, min_periods, np.nansum)


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
    return _rolling_apply_vectorized(arr, window, min_periods, np.nanmean)


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
    return _rolling_apply_vectorized(arr, window, min_periods, np.nanmin)


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
    return _rolling_apply_vectorized(arr, window, min_periods, np.nanmax)


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

    n = len(arr)
    arr = arr.astype(float)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    # Handle ramp-up period
    for i in range(min(window - 1, n)):
        window_data = arr[: i + 1]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) >= min_periods and len(valid_data) > ddof:
            result[i] = np.std(valid_data, ddof=ddof)

    # Main vectorized portion using sliding_window_view
    if n >= window:
        windows = sliding_window_view(arr, window)
        # For std, we need to handle ddof properly
        # Count valid values per window
        valid_counts = np.sum(~np.isnan(windows), axis=1)
        # Compute std with ddof (nanstd doesn't support axis+ddof well, so use formula)
        # std = sqrt(var) where var = sum((x - mean)^2) / (n - ddof)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(windows, axis=1, keepdims=True)
            squared_diff = (windows - means) ** 2
            sum_sq = np.nansum(squared_diff, axis=1)
            # variance = sum_sq / (valid_counts - ddof)
            variance = sum_sq / (valid_counts - ddof)
            std_values = np.sqrt(variance)

        # Apply min_periods mask and ddof requirement
        mask = (valid_counts >= min_periods) & (valid_counts > ddof)
        result[window - 1 :] = np.where(mask, std_values, np.nan)

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

    n = len(arr)
    arr = arr.astype(float)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    # Handle ramp-up period
    for i in range(min(window - 1, n)):
        window_data = arr[: i + 1]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) >= min_periods and len(valid_data) > ddof:
            result[i] = np.var(valid_data, ddof=ddof)

    # Main vectorized portion using sliding_window_view
    if n >= window:
        windows = sliding_window_view(arr, window)
        valid_counts = np.sum(~np.isnan(windows), axis=1)

        with np.errstate(invalid="ignore"):
            means = np.nanmean(windows, axis=1, keepdims=True)
            squared_diff = (windows - means) ** 2
            sum_sq = np.nansum(squared_diff, axis=1)
            variance = sum_sq / (valid_counts - ddof)

        mask = (valid_counts >= min_periods) & (valid_counts > ddof)
        result[window - 1 :] = np.where(mask, variance, np.nan)

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

    if n == 0:
        return result

    # Vectorized forward fill using maximum.accumulate trick
    mask = np.isnan(result)
    if not mask.any():
        return result

    # Create index array where valid values have their own index, NaN have 0
    idx = np.where(~mask, np.arange(n), 0)
    # Use maximum.accumulate to propagate valid indices forward
    idx = np.maximum.accumulate(idx)
    # Use the propagated indices to fill values
    result = result[idx]

    # Handle limit if specified
    if limit is not None:
        # Compute cumulative count of consecutive NaNs
        cumsum_nan = np.cumsum(mask)
        # At each position, count consecutive NaNs = cumsum_nan - cumsum_nan[last_valid]
        # Find the cumsum at last valid index for each position
        last_valid_cumsum = np.where(~mask, cumsum_nan, 0)
        last_valid_cumsum = np.maximum.accumulate(last_valid_cumsum)
        consecutive_nans = cumsum_nan - last_valid_cumsum
        # Reset values that exceed limit
        result = np.where(consecutive_nans > limit, np.nan, result)

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

    if n == 0:
        return result

    # Vectorized backward fill: reverse, ffill, reverse back
    mask = np.isnan(result)
    if not mask.any():
        return result

    # Reverse the array
    result_rev = result[::-1]
    mask_rev = mask[::-1]

    # Create index array for reversed data
    idx = np.where(~mask_rev, np.arange(n), 0)
    idx = np.maximum.accumulate(idx)
    result_rev = result_rev[idx]

    # Handle limit if specified
    if limit is not None:
        cumsum_nan = np.cumsum(mask_rev)
        last_valid_cumsum = np.where(~mask_rev, cumsum_nan, 0)
        last_valid_cumsum = np.maximum.accumulate(last_valid_cumsum)
        consecutive_nans = cumsum_nan - last_valid_cumsum
        result_rev = np.where(consecutive_nans > limit, np.nan, result_rev)

    # Reverse back
    return result_rev[::-1]


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
