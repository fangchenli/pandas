"""
NumPy Datetime kernel implementations.

This module contains datetime extraction and manipulation operations.
"""

import numpy as np

from pandas.lazy.backends import register_kernel

# Datetime Extraction Functions
# =============================================================================


@register_kernel("dt_year", "numpy")
def numpy_dt_year(arr: np.ndarray) -> np.ndarray:
    """
    Extract year from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of years (int64).
    """
    # Convert to datetime64[Y] and extract year
    return arr.astype("datetime64[Y]").astype(int) + 1970


@register_kernel("dt_month", "numpy")
def numpy_dt_month(arr: np.ndarray) -> np.ndarray:
    """
    Extract month from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of months (1-12).
    """
    import pandas as pd

    # Use pandas for reliable datetime component extraction
    return pd.DatetimeIndex(arr).month.to_numpy()


@register_kernel("dt_day", "numpy")
def numpy_dt_day(arr: np.ndarray) -> np.ndarray:
    """
    Extract day of month from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of days (1-31).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).day.to_numpy()


@register_kernel("dt_hour", "numpy")
def numpy_dt_hour(arr: np.ndarray) -> np.ndarray:
    """
    Extract hour from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of hours (0-23).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).hour.to_numpy()


@register_kernel("dt_minute", "numpy")
def numpy_dt_minute(arr: np.ndarray) -> np.ndarray:
    """
    Extract minute from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of minutes (0-59).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).minute.to_numpy()


@register_kernel("dt_second", "numpy")
def numpy_dt_second(arr: np.ndarray) -> np.ndarray:
    """
    Extract second from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of seconds (0-59).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).second.to_numpy()


@register_kernel("dt_microsecond", "numpy")
def numpy_dt_microsecond(arr: np.ndarray) -> np.ndarray:
    """
    Extract microsecond from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of microseconds (0-999999).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).microsecond.to_numpy()


@register_kernel("dt_nanosecond", "numpy")
def numpy_dt_nanosecond(arr: np.ndarray) -> np.ndarray:
    """
    Extract nanosecond from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of nanoseconds (0-999999999).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).nanosecond.to_numpy()


@register_kernel("dt_day_of_week", "numpy")
def numpy_dt_day_of_week(arr: np.ndarray, count_from_zero: bool = True) -> np.ndarray:
    """
    Extract day of week from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.
    count_from_zero : bool, default True
        If True, returns 0-6 (Monday-Sunday).
        If False, returns 1-7 (Monday-Sunday).

    Returns
    -------
    np.ndarray
        Array of day of week values.
    """
    import pandas as pd

    dow = pd.DatetimeIndex(arr).day_of_week.to_numpy()
    if not count_from_zero:
        dow = dow + 1
    return dow


@register_kernel("dt_day_of_year", "numpy")
def numpy_dt_day_of_year(arr: np.ndarray) -> np.ndarray:
    """
    Extract day of year from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of day of year values (1-366).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).day_of_year.to_numpy()


@register_kernel("dt_week", "numpy")
def numpy_dt_week(arr: np.ndarray) -> np.ndarray:
    """
    Extract ISO week number from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of week numbers (1-53).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).isocalendar().week.to_numpy()


@register_kernel("dt_quarter", "numpy")
def numpy_dt_quarter(arr: np.ndarray) -> np.ndarray:
    """
    Extract quarter from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of quarters (1-4).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).quarter.to_numpy()


@register_kernel("dt_is_leap_year", "numpy")
def numpy_dt_is_leap_year(arr: np.ndarray) -> np.ndarray:
    """
    Check if datetime values fall in a leap year.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Boolean array indicating leap year.
    """
    import pandas as pd

    result = pd.DatetimeIndex(arr).is_leap_year
    # is_leap_year returns ndarray directly
    if hasattr(result, "to_numpy"):
        return result.to_numpy()
    return np.asarray(result)


@register_kernel("dt_iso_year", "numpy")
def numpy_dt_iso_year(arr: np.ndarray) -> np.ndarray:
    """
    Extract ISO year from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of ISO years.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).isocalendar().year.to_numpy()


@register_kernel("dt_strftime", "numpy")
def numpy_dt_strftime(arr: np.ndarray, format: str) -> np.ndarray:
    """
    Format datetime array to strings.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.
    format : str
        strftime format string.

    Returns
    -------
    np.ndarray
        String array of formatted datetimes.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).strftime(format).to_numpy()


# =============================================================================
# Datetime Rounding/Truncation Functions
# =============================================================================


@register_kernel("dt_floor", "numpy")
def numpy_dt_floor(arr: np.ndarray, unit: str) -> np.ndarray:
    """
    Floor datetime to specified unit.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.
    unit : str
        Unit to floor to.

    Returns
    -------
    np.ndarray
        Floored datetime array.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).floor(unit).to_numpy()


@register_kernel("dt_ceil", "numpy")
def numpy_dt_ceil(arr: np.ndarray, unit: str) -> np.ndarray:
    """
    Ceil datetime to specified unit.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.
    unit : str
        Unit to ceil to.

    Returns
    -------
    np.ndarray
        Ceiled datetime array.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).ceil(unit).to_numpy()


@register_kernel("dt_round", "numpy")
def numpy_dt_round(arr: np.ndarray, unit: str) -> np.ndarray:
    """
    Round datetime to specified unit.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.
    unit : str
        Unit to round to.

    Returns
    -------
    np.ndarray
        Rounded datetime array.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).round(unit).to_numpy()


@register_kernel("dt_date", "numpy")
def numpy_dt_date(arr: np.ndarray) -> np.ndarray:
    """
    Extract date component from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of date objects.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).date


@register_kernel("dt_time", "numpy")
def numpy_dt_time(arr: np.ndarray) -> np.ndarray:
    """
    Extract time component from datetime array.

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of time objects.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).time


@register_kernel("dt_normalize", "numpy")
def numpy_dt_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize datetime to midnight (remove time component).

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Normalized datetime array.
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).normalize().to_numpy()


@register_kernel("dt_weekday", "numpy")
def numpy_dt_weekday(arr: np.ndarray) -> np.ndarray:
    """
    Extract weekday as integer (0=Monday, 6=Sunday).

    Parameters
    ----------
    arr : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of weekday values (0-6).
    """
    import pandas as pd

    return pd.DatetimeIndex(arr).day_of_week.to_numpy()


@register_kernel("dt_is_month_start", "numpy")
def numpy_dt_is_month_start(arr: np.ndarray) -> np.ndarray:
    """Check if datetime is first day of month."""
    import pandas as pd

    return pd.DatetimeIndex(arr).is_month_start.to_numpy()


@register_kernel("dt_is_month_end", "numpy")
def numpy_dt_is_month_end(arr: np.ndarray) -> np.ndarray:
    """Check if datetime is last day of month."""
    import pandas as pd

    return pd.DatetimeIndex(arr).is_month_end.to_numpy()


@register_kernel("dt_is_year_start", "numpy")
def numpy_dt_is_year_start(arr: np.ndarray) -> np.ndarray:
    """Check if datetime is first day of year."""
    import pandas as pd

    return pd.DatetimeIndex(arr).is_year_start.to_numpy()


@register_kernel("dt_is_year_end", "numpy")
def numpy_dt_is_year_end(arr: np.ndarray) -> np.ndarray:
    """Check if datetime is last day of year."""
    import pandas as pd

    return pd.DatetimeIndex(arr).is_year_end.to_numpy()


@register_kernel("dt_days_in_month", "numpy")
def numpy_dt_days_in_month(arr: np.ndarray) -> np.ndarray:
    """Get number of days in the month."""
    import pandas as pd

    return pd.DatetimeIndex(arr).days_in_month.to_numpy()


# =============================================================================
