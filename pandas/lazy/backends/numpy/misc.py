"""
NumPy miscellaneous kernel implementations.

This module contains:
- Rolling window operations (with optional Bottleneck acceleration)
- Shift/Lag/Lead operations
- Fill NA variants (vectorized)

When Bottleneck is installed and enabled via pd.set_option("compute.use_bottleneck"),
rolling window and fill operations use Bottleneck's C-optimized implementations
for significantly better performance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pandas.lazy.backends import register_kernel
from pandas.lazy.backends._bottleneck import (
    bn,
    use_bottleneck,
)

# =============================================================================
# Rolling Window Helper Functions (Vectorized)
# =============================================================================


def _rolling_sum_cumsum(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Rolling sum using cumsum difference - O(n) time complexity.

    This is much faster than sliding_window_view + nansum approach.
    """
    n = len(arr)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    # Replace NaN with 0 for cumsum, track valid counts separately
    is_nan = np.isnan(arr)
    arr_filled = np.where(is_nan, 0.0, arr)

    # Compute cumsum of values and valid counts
    cumsum = np.cumsum(arr_filled)
    valid_cumsum = np.cumsum(~is_nan)

    # Rolling sum = cumsum[i] - cumsum[i - window]
    # For indices >= window: sum = cumsum[i] - cumsum[i - window]
    # For indices < window: sum = cumsum[i] (partial window)

    # Full windows (indices >= window - 1)
    if n >= window:
        # Prepend 0 to handle the subtraction cleanly
        cumsum_padded = np.concatenate([[0], cumsum])
        valid_padded = np.concatenate([[0], valid_cumsum])

        rolling_sums = cumsum_padded[window:] - cumsum_padded[:-window]
        valid_counts = valid_padded[window:] - valid_padded[:-window]

        # Apply min_periods mask
        mask = valid_counts >= min_periods
        result[window - 1 :] = np.where(mask, rolling_sums, np.nan)

    # Partial windows (ramp-up period)
    for i in range(min(window - 1, n)):
        valid_count = valid_cumsum[i]
        if valid_count >= min_periods:
            result[i] = cumsum[i]

    return result


def _rolling_mean_cumsum(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Rolling mean using cumsum difference - O(n) time complexity.
    """
    n = len(arr)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    is_nan = np.isnan(arr)
    arr_filled = np.where(is_nan, 0.0, arr)

    cumsum = np.cumsum(arr_filled)
    valid_cumsum = np.cumsum(~is_nan)

    if n >= window:
        cumsum_padded = np.concatenate([[0], cumsum])
        valid_padded = np.concatenate([[0], valid_cumsum])

        rolling_sums = cumsum_padded[window:] - cumsum_padded[:-window]
        valid_counts = valid_padded[window:] - valid_padded[:-window]

        # Mean = sum / count (avoiding division by zero)
        with np.errstate(invalid="ignore", divide="ignore"):
            rolling_means = rolling_sums / valid_counts

        mask = valid_counts >= min_periods
        result[window - 1 :] = np.where(mask, rolling_means, np.nan)

    # Partial windows
    for i in range(min(window - 1, n)):
        valid_count = valid_cumsum[i]
        if valid_count >= min_periods:
            result[i] = cumsum[i] / valid_count

    return result


def _get_rolling_valid_counts(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling count of valid (non-NaN) values using O(n) cumsum.

    Returns array of length n with valid counts for each position.
    """
    n = len(arr)
    is_valid = ~np.isnan(arr)
    valid_cumsum = np.cumsum(is_valid)

    # For positions < window, the count is just cumsum up to that point
    # For positions >= window, count = cumsum[i] - cumsum[i - window]
    counts = np.zeros(n, dtype=np.int64)

    # Ramp-up: counts[i] = valid_cumsum[i]
    ramp_up_end = min(window - 1, n)
    counts[:ramp_up_end] = valid_cumsum[:ramp_up_end]

    # Main: counts[i] = valid_cumsum[i] - valid_cumsum[i - window]
    if n >= window:
        counts[window - 1 :] = valid_cumsum[window - 1 :] - np.concatenate(
            [[0], valid_cumsum[: n - window]]
        )

    return counts


def _rolling_apply_vectorized(
    arr: np.ndarray,
    window: int,
    min_periods: int,
    agg_func,
    handle_nan: bool = True,
) -> np.ndarray:
    """
    Vectorized rolling window aggregation using sliding_window_view.

    Used for min/max where cumsum trick doesn't apply.
    """
    n = len(arr)
    arr = arr.astype(float)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    # Compute valid counts using O(n) cumsum method
    if handle_nan:
        valid_counts = _get_rolling_valid_counts(arr, window)

    # For the "ramp-up" period (first window-1 elements), we need special handling
    for i in range(min(window - 1, n)):
        window_data = arr[: i + 1]
        if handle_nan:
            if valid_counts[i] >= min_periods:
                result[i] = agg_func(window_data)
        elif len(window_data) >= min_periods:
            result[i] = agg_func(window_data)

    # For the main portion, use sliding_window_view
    if n >= window:
        windows = sliding_window_view(arr, window)
        agg_values = agg_func(windows, axis=1)

        if handle_nan:
            mask = valid_counts[window - 1 :] >= min_periods
            result[window - 1 :] = np.where(mask, agg_values, np.nan)
        else:
            result[window - 1 :] = agg_values

    return result


# Rolling Window Kernels
# =============================================================================


@register_kernel("rolling_sum", "numpy")
def numpy_rolling_sum(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling sum over a window.

    Uses Bottleneck's move_sum when available, otherwise O(n) cumsum-based algorithm.

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

    # Try Bottleneck fast path
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_sum(arr_float, window, min_count=min_periods)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback
    arr = arr.astype(float)
    return _rolling_sum_cumsum(arr, window, min_periods)


@register_kernel("rolling_mean", "numpy")
def numpy_rolling_mean(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling mean over a window.

    Uses Bottleneck's move_mean when available, otherwise O(n) cumsum-based algorithm.

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

    # Try Bottleneck fast path
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_mean(arr_float, window, min_count=min_periods)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback
    arr = arr.astype(float)
    return _rolling_mean_cumsum(arr, window, min_periods)


@register_kernel("rolling_min", "numpy")
def numpy_rolling_min(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling minimum over a window.

    Uses Bottleneck's move_min when available.

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

    # Try Bottleneck fast path
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_min(arr_float, window, min_count=min_periods)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback
    return _rolling_apply_vectorized(arr, window, min_periods, np.nanmin)


@register_kernel("rolling_max", "numpy")
def numpy_rolling_max(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling maximum over a window.

    Uses Bottleneck's move_max when available.

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

    # Try Bottleneck fast path
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_max(arr_float, window, min_count=min_periods)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback
    return _rolling_apply_vectorized(arr, window, min_periods, np.nanmax)


def _rolling_var_cumsum(
    arr: np.ndarray, window: int, min_periods: int, ddof: int = 1
) -> np.ndarray:
    """
    Rolling variance using cumsum - O(n) time complexity.

    Uses the formula: Var(X) = E[X²] - E[X]²
    Adjusted for sample variance with ddof.
    """
    n = len(arr)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    # Replace NaN with 0 for cumsum, track valid counts separately
    is_nan = np.isnan(arr)
    arr_filled = np.where(is_nan, 0.0, arr)
    arr_sq_filled = np.where(is_nan, 0.0, arr**2)

    # Cumulative sums
    cumsum = np.cumsum(arr_filled)
    cumsum_sq = np.cumsum(arr_sq_filled)
    valid_cumsum = np.cumsum(~is_nan)

    # Full windows (indices >= window - 1)
    if n >= window:
        # Prepend 0 to handle the subtraction cleanly
        cumsum_padded = np.concatenate([[0], cumsum])
        cumsum_sq_padded = np.concatenate([[0], cumsum_sq])
        valid_padded = np.concatenate([[0], valid_cumsum])

        rolling_sums = cumsum_padded[window:] - cumsum_padded[:-window]
        rolling_sums_sq = cumsum_sq_padded[window:] - cumsum_sq_padded[:-window]
        valid_counts = valid_padded[window:] - valid_padded[:-window]

        # Variance = (sum_sq - sum² / n) / (n - ddof)
        # This is the sample variance formula
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_sq = (rolling_sums / valid_counts) ** 2
            mean_of_sq = rolling_sums_sq / valid_counts
            # Var = E[X²] - E[X]² adjusted for sample variance
            # sample_var = n / (n - ddof) * (E[X²] - E[X]²)
            variance = (valid_counts / (valid_counts - ddof)) * (mean_of_sq - mean_sq)
            # Ensure non-negative (numerical precision issues)
            variance = np.maximum(variance, 0.0)

        # Apply min_periods mask and ddof requirement
        mask = (valid_counts >= min_periods) & (valid_counts > ddof)
        result[window - 1 :] = np.where(mask, variance, np.nan)

    # Partial windows (ramp-up period)
    for i in range(min(window - 1, n)):
        valid_count = valid_cumsum[i]
        if valid_count >= min_periods and valid_count > ddof:
            sum_val = cumsum[i]
            sum_sq_val = cumsum_sq[i]
            mean_sq = (sum_val / valid_count) ** 2
            mean_of_sq = sum_sq_val / valid_count
            var = (valid_count / (valid_count - ddof)) * (mean_of_sq - mean_sq)
            result[i] = max(var, 0.0)

    return result


@register_kernel("rolling_std", "numpy")
def numpy_rolling_std(
    arr: np.ndarray, window: int, min_periods: int | None = None, ddof: int = 1
) -> np.ndarray:
    """
    Calculate rolling standard deviation over a window.

    Uses Bottleneck's move_std when available, otherwise O(n) cumsum-based algorithm.

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

    # Try Bottleneck fast path
    # Note: Bottleneck defaults to ddof=0, so we must pass ddof explicitly
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_std(arr_float, window, min_count=min_periods, ddof=ddof)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback
    arr = arr.astype(float)
    variance = _rolling_var_cumsum(arr, window, min_periods, ddof)
    return np.sqrt(variance)


@register_kernel("rolling_var", "numpy")
def numpy_rolling_var(
    arr: np.ndarray, window: int, min_periods: int | None = None, ddof: int = 1
) -> np.ndarray:
    """
    Calculate rolling variance over a window.

    Uses Bottleneck's move_var when available, otherwise O(n) cumsum-based algorithm.

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

    # Try Bottleneck fast path
    # Note: Bottleneck defaults to ddof=0, so we must pass ddof explicitly
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_var(arr_float, window, min_count=min_periods, ddof=ddof)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback
    arr = arr.astype(float)
    return _rolling_var_cumsum(arr, window, min_periods, ddof)


@register_kernel("rolling_median", "numpy")
def numpy_rolling_median(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling median over a window.

    Uses Bottleneck's move_median when available (much faster).

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
        Rolling median values.
    """
    if min_periods is None:
        min_periods = window

    # Try Bottleneck fast path
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_median(arr_float, window, min_count=min_periods)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback using sliding_window_view + np.nanmedian
    return _rolling_apply_vectorized(arr, window, min_periods, np.nanmedian)


@register_kernel("rolling_argmin", "numpy")
def numpy_rolling_argmin(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling argmin - index of minimum value in each window.

    Uses Bottleneck's move_argmin when available.

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
        Rolling argmin values (index within window).
    """
    if min_periods is None:
        min_periods = window

    # Try Bottleneck fast path
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_argmin(arr_float, window, min_count=min_periods)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback
    return _rolling_apply_vectorized(arr, window, min_periods, np.nanargmin)


@register_kernel("rolling_argmax", "numpy")
def numpy_rolling_argmax(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling argmax - index of maximum value in each window.

    Uses Bottleneck's move_argmax when available.

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
        Rolling argmax values (index within window).
    """
    if min_periods is None:
        min_periods = window

    # Try Bottleneck fast path
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_argmax(arr_float, window, min_count=min_periods)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # NumPy fallback
    return _rolling_apply_vectorized(arr, window, min_periods, np.nanargmax)


@register_kernel("rolling_rank", "numpy")
def numpy_rolling_rank(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling rank of each value within its window.

    Requires Bottleneck's move_rank for efficient implementation.

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
        Rolling rank values.

    Raises
    ------
    NotImplementedError
        If Bottleneck is not installed.
    """
    if min_periods is None:
        min_periods = window

    # Try Bottleneck fast path
    if use_bottleneck() and arr.dtype != object:
        try:
            arr_float = arr.astype(np.float64, copy=False)
            return bn.move_rank(arr_float, window, min_count=min_periods)
        except (TypeError, ValueError):
            pass

    # No pure NumPy fallback - require Bottleneck
    raise NotImplementedError(
        "rolling_rank requires Bottleneck. Install with: pip install bottleneck"
    )


@register_kernel("rolling_count", "numpy")
def numpy_rolling_count(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Count non-null values in a rolling window.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to 1.

    Returns
    -------
    np.ndarray
        Rolling count of non-null values.
    """
    if min_periods is None:
        min_periods = 1

    n = len(arr)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    # Use O(n) cumsum-based approach
    valid_counts = _get_rolling_valid_counts(arr.astype(float), window)

    # Apply min_periods
    result = valid_counts.astype(float)
    result[valid_counts < min_periods] = np.nan

    return result


@register_kernel("rolling_quantile", "numpy")
def numpy_rolling_quantile(
    arr: np.ndarray, window: int, quantile: float, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling quantile over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    quantile : float
        Quantile to compute (0.0 to 1.0).
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    np.ndarray
        Rolling quantile values.
    """
    if min_periods is None:
        min_periods = window

    def nanquantile_func(x, axis=None):
        if axis is None:
            return np.nanquantile(x, quantile)
        return np.nanquantile(x, quantile, axis=axis)

    return _rolling_apply_vectorized(arr, window, min_periods, nanquantile_func)


@register_kernel("rolling_skew", "numpy")
def numpy_rolling_skew(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling skewness over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to 3.

    Returns
    -------
    np.ndarray
        Rolling skewness values.
    """
    if min_periods is None:
        min_periods = 3  # Need at least 3 values for skewness

    n = len(arr)
    arr = arr.astype(float)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    valid_counts = _get_rolling_valid_counts(arr, window)

    # Ramp-up period
    for i in range(min(window - 1, n)):
        if valid_counts[i] >= min_periods:
            window_data = arr[: i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) >= 3:
                result[i] = _compute_skewness(valid_data)

    # Main portion using sliding_window_view
    if n >= window:
        windows = sliding_window_view(arr, window)
        for j, w in enumerate(windows):
            i = j + window - 1
            if valid_counts[i] >= min_periods:
                valid_data = w[~np.isnan(w)]
                if len(valid_data) >= 3:
                    result[i] = _compute_skewness(valid_data)

    return result


def _compute_skewness(x: np.ndarray) -> float:
    """Compute sample skewness (Fisher's definition)."""
    n = len(x)
    if n < 3:
        return np.nan
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / n
    m3 = np.sum((x - mean) ** 3) / n
    if m2 == 0:
        return 0.0
    # Fisher's skewness with bias correction
    g1 = m3 / (m2**1.5)
    # Bias correction factor
    return g1 * np.sqrt(n * (n - 1)) / (n - 2)


@register_kernel("rolling_kurt", "numpy")
def numpy_rolling_kurt(
    arr: np.ndarray, window: int, min_periods: int | None = None
) -> np.ndarray:
    """
    Calculate rolling kurtosis over a window.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to 4.

    Returns
    -------
    np.ndarray
        Rolling kurtosis values (excess kurtosis, Fisher's definition).
    """
    if min_periods is None:
        min_periods = 4  # Need at least 4 values for kurtosis

    n = len(arr)
    arr = arr.astype(float)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    valid_counts = _get_rolling_valid_counts(arr, window)

    # Ramp-up period
    for i in range(min(window - 1, n)):
        if valid_counts[i] >= min_periods:
            window_data = arr[: i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) >= 4:
                result[i] = _compute_kurtosis(valid_data)

    # Main portion using sliding_window_view
    if n >= window:
        windows = sliding_window_view(arr, window)
        for j, w in enumerate(windows):
            i = j + window - 1
            if valid_counts[i] >= min_periods:
                valid_data = w[~np.isnan(w)]
                if len(valid_data) >= 4:
                    result[i] = _compute_kurtosis(valid_data)

    return result


def _compute_kurtosis(x: np.ndarray) -> float:
    """Compute sample excess kurtosis (Fisher's definition)."""
    n = len(x)
    if n < 4:
        return np.nan
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / n
    m4 = np.sum((x - mean) ** 4) / n
    if m2 == 0:
        return 0.0
    # Fisher's kurtosis (excess kurtosis)
    g2 = m4 / (m2**2) - 3.0
    # Bias correction factor
    return ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6)


@register_kernel("rolling_apply", "numpy")
def numpy_rolling_apply(
    arr: np.ndarray,
    window: int,
    func,
    min_periods: int | None = None,
    raw: bool = True,
) -> np.ndarray:
    """
    Apply a custom function to a rolling window.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    window : int
        Size of the rolling window.
    func : callable
        Function to apply to each window. Should accept an array and return a scalar.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.
    raw : bool, default True
        If True, pass raw numpy array to function.
        If False, pass a pandas Series (not supported here).

    Returns
    -------
    np.ndarray
        Array of function results for each window.
    """
    if min_periods is None:
        min_periods = window

    n = len(arr)
    arr = arr.astype(float)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0:
        return result

    valid_counts = _get_rolling_valid_counts(arr, window)

    # Ramp-up period
    for i in range(min(window - 1, n)):
        if valid_counts[i] >= min_periods:
            window_data = arr[: i + 1]
            try:
                result[i] = func(window_data)
            except Exception:
                result[i] = np.nan

    # Main portion
    if n >= window:
        windows = sliding_window_view(arr, window)
        for j, w in enumerate(windows):
            i = j + window - 1
            if valid_counts[i] >= min_periods:
                try:
                    result[i] = func(w)
                except Exception:
                    result[i] = np.nan

    return result


def _rolling_cov_corr_cumsum(
    arr1: np.ndarray,
    arr2: np.ndarray,
    window: int,
    min_periods: int,
    compute_corr: bool = False,
) -> np.ndarray:
    """
    Rolling covariance/correlation using cumsum - O(n) time complexity.

    Uses the formula: Cov(X,Y) = E[XY] - E[X]E[Y]
    And for correlation: Corr = Cov(X,Y) / (std(X) * std(Y))
    """
    n = len(arr1)
    result = np.full(n, np.nan)

    if n == 0 or window <= 0 or len(arr2) != n:
        return result

    # Both must be non-NaN for a valid pair
    is_nan = np.isnan(arr1) | np.isnan(arr2)
    arr1_filled = np.where(is_nan, 0.0, arr1)
    arr2_filled = np.where(is_nan, 0.0, arr2)
    prod_filled = arr1_filled * arr2_filled
    arr1_sq_filled = arr1_filled**2
    arr2_sq_filled = arr2_filled**2

    # Cumulative sums
    cumsum1 = np.cumsum(arr1_filled)
    cumsum2 = np.cumsum(arr2_filled)
    cumsum_prod = np.cumsum(prod_filled)
    cumsum1_sq = np.cumsum(arr1_sq_filled)
    cumsum2_sq = np.cumsum(arr2_sq_filled)
    valid_cumsum = np.cumsum(~is_nan)

    ddof = 1

    if n >= window:
        # Prepend 0 for subtraction
        c1_padded = np.concatenate([[0], cumsum1])
        c2_padded = np.concatenate([[0], cumsum2])
        cp_padded = np.concatenate([[0], cumsum_prod])
        c1sq_padded = np.concatenate([[0], cumsum1_sq])
        c2sq_padded = np.concatenate([[0], cumsum2_sq])
        valid_padded = np.concatenate([[0], valid_cumsum])

        sum1 = c1_padded[window:] - c1_padded[:-window]
        sum2 = c2_padded[window:] - c2_padded[:-window]
        sum_prod = cp_padded[window:] - cp_padded[:-window]
        sum1_sq = c1sq_padded[window:] - c1sq_padded[:-window]
        sum2_sq = c2sq_padded[window:] - c2sq_padded[:-window]
        valid_counts = valid_padded[window:] - valid_padded[:-window]

        with np.errstate(invalid="ignore", divide="ignore"):
            mean1 = sum1 / valid_counts
            mean2 = sum2 / valid_counts
            mean_prod = sum_prod / valid_counts

            # Cov = E[XY] - E[X]E[Y], adjusted for sample
            cov = (valid_counts / (valid_counts - ddof)) * (mean_prod - mean1 * mean2)

            if compute_corr:
                var1 = (valid_counts / (valid_counts - ddof)) * (
                    sum1_sq / valid_counts - mean1**2
                )
                var2 = (valid_counts / (valid_counts - ddof)) * (
                    sum2_sq / valid_counts - mean2**2
                )
                var1 = np.maximum(var1, 0.0)
                var2 = np.maximum(var2, 0.0)
                std1 = np.sqrt(var1)
                std2 = np.sqrt(var2)
                corr = cov / (std1 * std2)
                values = corr
            else:
                values = cov

        mask = (valid_counts >= min_periods) & (valid_counts > ddof)
        result[window - 1 :] = np.where(mask, values, np.nan)

    # Ramp-up period
    for i in range(min(window - 1, n)):
        valid_count = valid_cumsum[i]
        if valid_count >= min_periods and valid_count > ddof:
            s1 = cumsum1[i]
            s2 = cumsum2[i]
            sp = cumsum_prod[i]
            s1sq = cumsum1_sq[i]
            s2sq = cumsum2_sq[i]
            m1 = s1 / valid_count
            m2 = s2 / valid_count
            mp = sp / valid_count
            cov = (valid_count / (valid_count - ddof)) * (mp - m1 * m2)

            if compute_corr:
                var1 = (valid_count / (valid_count - ddof)) * (
                    s1sq / valid_count - m1**2
                )
                var2 = (valid_count / (valid_count - ddof)) * (
                    s2sq / valid_count - m2**2
                )
                var1 = max(var1, 0.0)
                var2 = max(var2, 0.0)
                if var1 > 0 and var2 > 0:
                    result[i] = cov / (np.sqrt(var1) * np.sqrt(var2))
            else:
                result[i] = cov

    return result


@register_kernel("rolling_cov", "numpy")
def numpy_rolling_cov(
    arr1: np.ndarray,
    arr2: np.ndarray,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> np.ndarray:
    """
    Calculate rolling covariance between two arrays.

    Parameters
    ----------
    arr1 : np.ndarray
        First input array.
    arr2 : np.ndarray
        Second input array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.
    ddof : int, default 1
        Delta degrees of freedom.

    Returns
    -------
    np.ndarray
        Rolling covariance values.
    """
    if min_periods is None:
        min_periods = window

    arr1 = arr1.astype(float)
    arr2 = arr2.astype(float)

    return _rolling_cov_corr_cumsum(arr1, arr2, window, min_periods, compute_corr=False)


@register_kernel("rolling_corr", "numpy")
def numpy_rolling_corr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    """
    Calculate rolling Pearson correlation between two arrays.

    Parameters
    ----------
    arr1 : np.ndarray
        First input array.
    arr2 : np.ndarray
        Second input array.
    window : int
        Size of the rolling window.
    min_periods : int or None
        Minimum number of observations required. Defaults to window size.

    Returns
    -------
    np.ndarray
        Rolling correlation values.
    """
    if min_periods is None:
        min_periods = window

    arr1 = arr1.astype(float)
    arr2 = arr2.astype(float)

    return _rolling_cov_corr_cumsum(arr1, arr2, window, min_periods, compute_corr=True)


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

    Uses Bottleneck's push when available, otherwise O(n) vectorized algorithm.

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
    # Convert to float for NaN support
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)

    n = len(arr)
    if n == 0:
        return arr.copy()

    # Try Bottleneck fast path
    # bn.push: n parameter is the fill limit (None = fill all)
    if use_bottleneck():
        try:
            return bn.push(arr, n=limit)
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # Check for NaN - if none, return copy
    mask = np.isnan(arr)
    if not mask.any():
        return arr.copy()

    # Vectorized forward fill using maximum.accumulate trick
    # Create index array where valid values have their own index, NaN have 0
    idx = np.where(~mask, np.arange(n), 0)
    # Use maximum.accumulate to propagate valid indices forward
    np.maximum.accumulate(idx, out=idx)
    # Use the propagated indices to fill values
    result = arr[idx]

    # Handle limit if specified
    if limit is not None:
        # Compute cumulative count of consecutive NaNs
        cumsum_nan = np.cumsum(mask)
        # At each position, count consecutive NaNs = cumsum_nan - cumsum_nan[last_valid]
        # Find the cumsum at last valid index for each position
        last_valid_cumsum = np.where(~mask, cumsum_nan, 0)
        np.maximum.accumulate(last_valid_cumsum, out=last_valid_cumsum)
        consecutive_nans = cumsum_nan - last_valid_cumsum
        # Reset values that exceed limit
        exceed_mask = consecutive_nans > limit
        if exceed_mask.any():
            result = result.copy()  # Only copy if we need to modify
            result[exceed_mask] = np.nan

    return result


@register_kernel("bfill", "numpy")
def numpy_bfill(arr: np.ndarray, limit: int | None = None) -> np.ndarray:
    """
    Backward fill missing values.

    Uses Bottleneck's push on reversed array when available.

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
    # Convert to float for NaN support
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)

    n = len(arr)
    if n == 0:
        return arr.copy()

    # Try Bottleneck fast path (push on reversed, then reverse back)
    if use_bottleneck():
        try:
            result = bn.push(arr[::-1], n=limit)[::-1].copy()
            return result
        except (TypeError, ValueError):
            pass  # Fall through to NumPy implementation

    # Check for NaN - if none, return copy
    mask = np.isnan(arr)
    if not mask.any():
        return arr.copy()

    # Vectorized backward fill: work on reversed views
    # Create reversed view (no copy)
    arr_rev = arr[::-1]
    mask_rev = mask[::-1]

    # Create index array for reversed data
    idx = np.where(~mask_rev, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    result_rev = arr_rev[idx]

    # Handle limit if specified
    if limit is not None:
        cumsum_nan = np.cumsum(mask_rev)
        last_valid_cumsum = np.where(~mask_rev, cumsum_nan, 0)
        np.maximum.accumulate(last_valid_cumsum, out=last_valid_cumsum)
        consecutive_nans = cumsum_nan - last_valid_cumsum
        exceed_mask = consecutive_nans > limit
        if exceed_mask.any():
            result_rev = result_rev.copy()
            result_rev[exceed_mask] = np.nan

    # Reverse back (copy to ensure contiguous memory)
    return result_rev[::-1].copy()


# =============================================================================
# Clip/Between Kernels
# =============================================================================


@register_kernel("clip", "numpy")
def numpy_clip(arr: np.ndarray, lower=None, upper=None) -> np.ndarray:
    """
    Clip (limit) array values to a given range.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    lower : scalar, optional
        Minimum value. Values below this will be set to lower.
    upper : scalar, optional
        Maximum value. Values above this will be set to upper.

    Returns
    -------
    np.ndarray
        Clipped array.
    """
    return np.clip(arr, lower, upper)


@register_kernel("between", "numpy")
def numpy_between(arr: np.ndarray, left, right, inclusive: str = "both") -> np.ndarray:
    """
    Check if values are between left and right bounds.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    left : scalar
        Left bound.
    right : scalar
        Right bound.
    inclusive : str, default "both"
        Include boundaries: "both", "neither", "left", or "right".

    Returns
    -------
    np.ndarray
        Boolean array indicating if values are in range.
    """
    if inclusive == "both":
        return (arr >= left) & (arr <= right)
    elif inclusive == "neither":
        return (arr > left) & (arr < right)
    elif inclusive == "left":
        return (arr >= left) & (arr < right)
    elif inclusive == "right":
        return (arr > left) & (arr <= right)
    else:
        raise ValueError(f"Invalid inclusive value: {inclusive}")


# =============================================================================
# Diff/Pct_change Kernels
# =============================================================================


@register_kernel("diff", "numpy")
def numpy_diff(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """
    Calculate the difference between consecutive values.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    periods : int, default 1
        Number of periods to shift for calculating difference.

    Returns
    -------
    np.ndarray
        Array of differences.
    """
    arr = arr.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)

    if periods > 0 and periods < n:
        result[periods:] = arr[periods:] - arr[:-periods]
    elif periods < 0 and -periods < n:
        result[:periods] = arr[:periods] - arr[-periods:]

    return result


@register_kernel("pct_change", "numpy")
def numpy_pct_change(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """
    Calculate percentage change between consecutive values.

    Parameters
    ----------
    arr : np.ndarray
        Input numeric array.
    periods : int, default 1
        Number of periods to shift for calculating change.

    Returns
    -------
    np.ndarray
        Array of percentage changes.
    """
    arr = arr.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)

    if periods > 0 and periods < n:
        with np.errstate(divide="ignore", invalid="ignore"):
            result[periods:] = (arr[periods:] - arr[:-periods]) / arr[:-periods]
    elif periods < 0 and -periods < n:
        with np.errstate(divide="ignore", invalid="ignore"):
            result[:periods] = (arr[:periods] - arr[-periods:]) / arr[-periods:]

    return result


# =============================================================================
# Fill NA Variant Kernels
# =============================================================================


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
