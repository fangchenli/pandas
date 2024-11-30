from typing import (
    Any,
    Literal,
    TypeAlias,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    DtypeArg,
    DtypeBackend,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.api import (
    DataFrame,
    DatetimeIndex,
    Series,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import (
    DatetimeScalar,
    DatetimeScalarOrArrayConvertible,
    DatetimeTZDtype,
    DictConvertible,
    to_datetime,
)

DatesParsingArgs: TypeAlias = (
    list[str] | dict[str, DatetimeScalarOrArrayConvertible | DictConvertible] | None
)


def _process_parse_dates_argument(
    parse_dates: DatesParsingArgs = None,
) -> list[str]:
    """Process parse_dates argument for read_sql functions"""
    # handle non-list entries for parse_dates gracefully
    if parse_dates is None:
        parse_dates = []

    elif not hasattr(parse_dates, "__iter__"):
        parse_dates = [parse_dates]
    return parse_dates


def _handle_date_column(
    col: Series, utc: bool = False, format: str | dict[str, Any] | None = None
) -> DatetimeIndex | Series | DatetimeScalar | NaTType | None:
    if isinstance(format, dict):
        # GH35185 Allow custom error values in parse_dates argument of
        # read_sql like functions.
        # Format can take on custom to_datetime argument values such as
        # {"errors": "coerce"} or {"dayfirst": True}
        return to_datetime(col, **format)
    else:
        # Allow passing of formatting string for integers
        # GH17855
        if format is None and (
            issubclass(col.dtype.type, np.floating)
            or issubclass(col.dtype.type, np.integer)
        ):
            format = "s"
        if format in ["D", "d", "h", "m", "s", "ms", "us", "ns"]:
            return to_datetime(col, errors="coerce", unit=format, utc=utc)
        elif isinstance(col.dtype, DatetimeTZDtype):
            # coerce to UTC timezone
            # GH11216
            return to_datetime(col, utc=True)
        else:
            return to_datetime(col, errors="coerce", format=format, utc=utc)


def _parse_date_columns(
    data_frame: DataFrame,
    parse_dates: DatesParsingArgs = None,
) -> DataFrame:
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
    parse_dates = _process_parse_dates_argument(parse_dates)

    # we want to coerce datetime64_tz dtypes for now to UTC
    # we could in theory do a 'nice' conversion from a FixedOffset tz
    # GH11216
    for i, (col_name, df_col) in enumerate(data_frame.items()):
        if isinstance(df_col.dtype, DatetimeTZDtype) or col_name in parse_dates:
            try:
                fmt = parse_dates[col_name]
            except (KeyError, TypeError):
                fmt = None
            data_frame.isetitem(i, _handle_date_column(df_col, format=fmt))

    return data_frame


def _convert_arrays_to_dataframe(
    data: list,
    columns: list[str],
    coerce_float: bool = True,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    content = lib.to_object_array_tuples(data)
    idx_len = content.shape[0]
    arrays = convert_object_array(
        list(content.T),
        dtype=None,
        coerce_float=coerce_float,
        dtype_backend=dtype_backend,
    )
    if dtype_backend == "pyarrow":
        pa = import_optional_dependency("pyarrow")

        result_arrays = []
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == "string":
                # TODO: Arrow still infers strings arrays as regular strings instead
                # of large_string, which is what we preserver everywhere else for
                # dtype_backend="pyarrow". We may want to reconsider this
                pa_array = pa_array.cast(pa.string())
            result_arrays.append(ArrowExtensionArray(pa_array))
        arrays = result_arrays  # type: ignore[assignment]
    if arrays:
        return DataFrame._from_arrays(
            arrays, columns=columns, index=range(idx_len), verify_integrity=False
        )
    else:
        return DataFrame(columns=columns)


def wrap_result(
    data: list,
    columns: list[str],
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    parse_dates: DatesParsingArgs = None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    frame = _convert_arrays_to_dataframe(data, columns, coerce_float, dtype_backend)

    if dtype:
        frame = frame.astype(dtype)

    frame = _parse_date_columns(frame, parse_dates)

    if index_col is not None:
        frame = frame.set_index(index_col)

    return frame
