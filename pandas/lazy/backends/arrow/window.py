"""
Arrow Window/Cumulative kernel implementations.

This module contains cumulative and window operations.
"""

import pyarrow.compute as pc

from pandas.lazy.backends import register_kernel
from pandas.lazy.backends.types import PyArrowArray

# Cumulative / Window Functions
# =============================================================================


@register_kernel("cumulative_sum", "arrow")
def arrow_cumulative_sum(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative sum.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative sum array.
    """
    return pc.cumulative_sum(arr, skip_nulls=skip_nulls)


@register_kernel("cumulative_max", "arrow")
def arrow_cumulative_max(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative maximum.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative max array.
    """
    return pc.cumulative_max(arr, skip_nulls=skip_nulls)


@register_kernel("cumulative_min", "arrow")
def arrow_cumulative_min(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative minimum.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative min array.
    """
    return pc.cumulative_min(arr, skip_nulls=skip_nulls)


@register_kernel("cumulative_prod", "arrow")
def arrow_cumulative_prod(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative product.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative product array.
    """
    return pc.cumulative_prod(arr, skip_nulls=skip_nulls)


@register_kernel("cumulative_mean", "arrow")
def arrow_cumulative_mean(arr: PyArrowArray, skip_nulls: bool = True) -> PyArrowArray:
    """
    Compute cumulative mean.

    Parameters
    ----------
    arr : PyArrowArray
        Input array.
    skip_nulls : bool, default True
        Whether to skip null values.

    Returns
    -------
    PyArrowArray
        Cumulative mean array.
    """
    return pc.cumulative_mean(arr, skip_nulls=skip_nulls)


# =============================================================================
