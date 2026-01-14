"""
Arrow miscellaneous kernel implementations.

This module contains:
- Shift/Lag/Lead operations
- Fill NA variants

NOTE: Rolling window operations (rolling_sum, rolling_mean, rolling_min, rolling_max,
rolling_std, rolling_var, rolling_median, rolling_argmin, rolling_argmax, rolling_rank)
are NOT implemented for Arrow because PyArrow doesn't have native rolling window
functions. The kernel dispatcher should route these to the NumPy backend instead,
which has optimized O(n) implementations with optional Bottleneck acceleration.
See pandas/lazy/backends/numpy/misc.py for the implementations.
"""

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import pandas as pd
from pandas.lazy.backends import register_kernel
from pandas.lazy.backends.types import PyArrowArray

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
