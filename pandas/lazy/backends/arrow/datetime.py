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
# Datetime Rounding/Truncation Functions
# =============================================================================


@register_kernel("dt_floor", "arrow")
def arrow_dt_floor(arr: PyArrowArray, unit: str) -> PyArrowArray:
    """
    Floor datetime to specified unit.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.
    unit : str
        Unit to floor to (e.g., "day", "hour", "minute", "second").

    Returns
    -------
    PyArrowArray
        Floored datetime array.
    """
    return pc.floor_temporal(arr, unit=unit)


@register_kernel("dt_ceil", "arrow")
def arrow_dt_ceil(arr: PyArrowArray, unit: str) -> PyArrowArray:
    """
    Ceil datetime to specified unit.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.
    unit : str
        Unit to ceil to.

    Returns
    -------
    PyArrowArray
        Ceiled datetime array.
    """
    return pc.ceil_temporal(arr, unit=unit)


@register_kernel("dt_round", "arrow")
def arrow_dt_round(arr: PyArrowArray, unit: str) -> PyArrowArray:
    """
    Round datetime to specified unit.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.
    unit : str
        Unit to round to.

    Returns
    -------
    PyArrowArray
        Rounded datetime array.
    """
    return pc.round_temporal(arr, unit=unit)


@register_kernel("dt_date", "arrow")
def arrow_dt_date(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract date component from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Date array (date32 type).
    """
    import pyarrow as pa

    return pc.cast(arr, pa.date32())


@register_kernel("dt_time", "arrow")
def arrow_dt_time(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract time component from datetime array.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Time array (time64 type).
    """
    import pyarrow as pa

    return pc.cast(arr, pa.time64("ns"))


@register_kernel("dt_normalize", "arrow")
def arrow_dt_normalize(arr: PyArrowArray) -> PyArrowArray:
    """
    Normalize datetime to midnight (remove time component).

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Normalized datetime array.
    """
    return pc.floor_temporal(arr, unit="day")


@register_kernel("dt_weekday", "arrow")
def arrow_dt_weekday(arr: PyArrowArray) -> PyArrowArray:
    """
    Extract weekday as integer (0=Monday, 6=Sunday).

    Alias for dt_day_of_week with count_from_zero=True.

    Parameters
    ----------
    arr : PyArrowArray
        Input datetime array.

    Returns
    -------
    PyArrowArray
        Array of weekday values (0-6).
    """
    return pc.day_of_week(arr, count_from_zero=True)


@register_kernel("dt_is_month_start", "arrow")
def arrow_dt_is_month_start(arr: PyArrowArray) -> PyArrowArray:
    """Check if datetime is first day of month."""
    return pc.equal(pc.day(arr), 1)


@register_kernel("dt_is_month_end", "arrow")
def arrow_dt_is_month_end(arr: PyArrowArray) -> PyArrowArray:
    """Check if datetime is last day of month."""
    return pc.equal(
        pc.day(arr), pc.days_between(arr, pc.ceil_temporal(arr, unit="month"))
    )


@register_kernel("dt_is_year_start", "arrow")
def arrow_dt_is_year_start(arr: PyArrowArray) -> PyArrowArray:
    """Check if datetime is first day of year."""
    return pc.equal(pc.day_of_year(arr), 1)


@register_kernel("dt_is_year_end", "arrow")
def arrow_dt_is_year_end(arr: PyArrowArray) -> PyArrowArray:
    """Check if datetime is last day of year."""
    return pc.equal(pc.day_of_year(arr), pc.if_else(pc.is_leap_year(arr), 366, 365))


@register_kernel("dt_days_in_month", "arrow")
def arrow_dt_days_in_month(arr: PyArrowArray) -> PyArrowArray:
    """Get number of days in the month."""
    # Get first of current month and first of next month, compute difference
    first_of_month = pc.floor_temporal(arr, unit="month")
    first_of_next = pc.ceil_temporal(arr, unit="month")
    return pc.days_between(first_of_month, first_of_next)


# =============================================================================
