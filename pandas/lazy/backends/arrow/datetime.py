"""
Arrow Datetime kernel implementations.

This module contains datetime extraction and manipulation operations.
"""

import pyarrow.compute as pc

from pandas.lazy.backends import register_kernel
from pandas.lazy.backends.types import PyArrowArray

# Datetime Extraction Functions
# =============================================================================


@register_kernel("dt_year", "arrow")
def arrow_dt_year(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract year from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of years (int64).
    """
    return pc.year(arr)


@register_kernel("dt_month", "arrow")
def arrow_dt_month(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract month from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of months (1-12).
    """
    return pc.month(arr)


@register_kernel("dt_day", "arrow")
def arrow_dt_day(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract day of month from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of days (1-31).
    """
    return pc.day(arr)


@register_kernel("dt_hour", "arrow")
def arrow_dt_hour(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract hour from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of hours (0-23).
    """
    return pc.hour(arr)


@register_kernel("dt_minute", "arrow")
def arrow_dt_minute(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract minute from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of minutes (0-59).
    """
    return pc.minute(arr)


@register_kernel("dt_second", "arrow")
def arrow_dt_second(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract second from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of seconds (0-59).
    """
    return pc.second(arr)


@register_kernel("dt_microsecond", "arrow")
def arrow_dt_microsecond(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract microsecond from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of microseconds (0-999999).
    """
    return pc.microsecond(arr)


@register_kernel("dt_nanosecond", "arrow")
def arrow_dt_nanosecond(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract nanosecond from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of nanoseconds (0-999999999).
    """
    return pc.nanosecond(arr)


@register_kernel("dt_day_of_week", "arrow")
def arrow_dt_day_of_week(
    arr: PyArrowArray, count_from_zero: bool = True
) -> PyArrowArray:
    """
    Extract day of week from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.
    count_from_zero : bool, default True
        If True, returns 0-6 (Monday-Sunday).
        If False, returns 1-7 (Monday-Sunday).

    Returns
    -------
    PyArrowArray
        Array of day of week values.
    """
    return pc.day_of_week(arr, count_from_zero=count_from_zero)


@register_kernel("dt_day_of_year", "arrow")
def arrow_dt_day_of_year(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract day of year from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of day of year values (1-366).
    """
    return pc.day_of_year(arr)


@register_kernel("dt_week", "arrow")
def arrow_dt_week(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract ISO week number from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of week numbers (1-53).
    """
    return pc.iso_week(arr)


@register_kernel("dt_quarter", "arrow")
def arrow_dt_quarter(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract quarter from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of quarters (1-4).
    """
    return pc.quarter(arr)


@register_kernel("dt_is_leap_year", "arrow")
def arrow_dt_is_leap_year(arr: PyArrowArray) -> PyArrowArray:
    """
    Check if datetime values fall in a leap year.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Boolean array indicating leap year.
    """
    return pc.is_leap_year(arr)


@register_kernel("dt_iso_year", "arrow")
def arrow_dt_iso_year(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract ISO year from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of ISO years.
    """
    return pc.iso_year(arr)


@register_kernel("dt_strftime", "arrow")
def arrow_dt_strftime(arr: PyArrowArray, format: str) -> PyArrowArray:
    """
    Format datetime array to strings.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.
    format : str
        strftime format string.

    Returns
    -------
    PyArrowArray
        String array of formatted datetimes.
    """
    return pc.strftime(arr, format=format)


# =============================================================================
