"""
Expression evaluation for lazy pandas.

This module provides the Evaluator class that executes IR nodes
against pandas DataFrames.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.lazy.ir import IRNode


def _convert_series_to_nullable(series: Series, original_dtype=None) -> Series:
    """
    Convert a Series with NaN to nullable dtype (using pd.NA).

    This ensures operations like lag/lead that introduce null values
    use pd.NA instead of np.nan, preserving the original dtype category
    (integer stays integer, float stays float).

    Parameters
    ----------
    series : Series
        The input Series, typically from a shift operation.
    original_dtype : dtype, optional
        The original dtype before the operation that introduced NaN.
        If provided, uses this to determine the appropriate nullable type.

    Returns
    -------
    Series
        Series with nullable dtype if it contains NaN values.
    """
    import pandas as pd

    if not series.isna().any():
        return series

    # Determine the appropriate nullable dtype
    if original_dtype is not None:
        # Use the original dtype to pick the right nullable variant
        if original_dtype.kind == "i":  # signed integer
            # Map to appropriate nullable int dtype
            if original_dtype == np.dtype("int8"):
                return pd.array(series, dtype=pd.Int8Dtype())
            elif original_dtype == np.dtype("int16"):
                return pd.array(series, dtype=pd.Int16Dtype())
            elif original_dtype == np.dtype("int32"):
                return pd.array(series, dtype=pd.Int32Dtype())
            else:  # int64 or default
                return pd.array(series, dtype=pd.Int64Dtype())
        elif original_dtype.kind == "u":  # unsigned integer
            if original_dtype == np.dtype("uint8"):
                return pd.array(series, dtype=pd.UInt8Dtype())
            elif original_dtype == np.dtype("uint16"):
                return pd.array(series, dtype=pd.UInt16Dtype())
            elif original_dtype == np.dtype("uint32"):
                return pd.array(series, dtype=pd.UInt32Dtype())
            else:  # uint64 or default
                return pd.array(series, dtype=pd.UInt64Dtype())
        elif original_dtype.kind == "f":  # float
            if original_dtype == np.dtype("float32"):
                return pd.array(series, dtype=pd.Float32Dtype())
            else:  # float64 or default
                return pd.array(series, dtype=pd.Float64Dtype())

    # Fallback: if series is float (due to NaN promotion), use Float64
    if series.dtype.kind == "f":
        return pd.array(series, dtype=pd.Float64Dtype())

    return series


class Evaluator:
    """
    Evaluates IR expressions against a DataFrame context.

    The evaluator takes an IR node and a DataFrame, and produces
    the result (Series or scalar) by recursively evaluating the
    expression tree.
    """

    def __init__(self, df: DataFrame) -> None:
        """
        Initialize evaluator with a DataFrame context.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to evaluate expressions against.
        """
        self._df = df

    def evaluate(self, node: IRNode) -> Series | Any:
        """
        Evaluate an IR node and return the result.

        Parameters
        ----------
        node : IRNode
            The IR node to evaluate.

        Returns
        -------
        Series or scalar
            The result of the evaluation.
        """
        from pandas.lazy.ir import (
            Alias,
            Call,
            Cast,
            FieldRef,
            Literal,
        )

        if isinstance(node, FieldRef):
            return self._df[node.name]

        elif isinstance(node, Literal):
            return node.value

        elif isinstance(node, Alias):
            # Alias just wraps an expression, evaluate the inner
            return self.evaluate(node.arg)

        elif isinstance(node, Call):
            return self._evaluate_call(node)

        elif isinstance(node, Cast):
            result = self.evaluate(node.arg)
            return result.astype(node.target_dtype)

        else:
            raise NotImplementedError(f"Unknown IR node type: {type(node)}")

    def _evaluate_call(self, node) -> Any:
        """Evaluate a Call node."""

        func = node.function
        args = [self.evaluate(arg) for arg in node.args]

        # Arithmetic operations
        if func == "add":
            return args[0] + args[1]
        elif func == "subtract":
            return args[0] - args[1]
        elif func == "multiply":
            return args[0] * args[1]
        elif func == "divide":
            return args[0] / args[1]
        elif func == "floor_divide":
            return args[0] // args[1]
        elif func == "modulo":
            return args[0] % args[1]
        elif func == "power":
            return args[0] ** args[1]

        # Unary operations
        elif func == "negate":
            return -args[0]
        elif func == "abs":
            return np.abs(args[0])

        # Comparison operations
        elif func == "equal":
            return args[0] == args[1]
        elif func == "not_equal":
            return args[0] != args[1]
        elif func == "less":
            return args[0] < args[1]
        elif func == "less_equal":
            return args[0] <= args[1]
        elif func == "greater":
            return args[0] > args[1]
        elif func == "greater_equal":
            return args[0] >= args[1]

        # Logical operations
        elif func == "and_":
            return args[0] & args[1]
        elif func == "or_":
            return args[0] | args[1]
        elif func == "invert":
            return ~args[0]

        # Null checks
        elif func == "is_null":
            import pandas as pd

            return pd.isna(args[0])
        elif func == "is_not_null":
            import pandas as pd

            return pd.notna(args[0])

        elif func == "isin":
            values = node.kwargs.get("values", [])
            return args[0].isin(values)

        # String operations
        elif func == "str_lower":
            return args[0].str.lower()
        elif func == "str_upper":
            return args[0].str.upper()
        elif func == "str_len":
            return args[0].str.len()
        elif func == "str_strip":
            return args[0].str.strip()
        elif func == "str_lstrip":
            return args[0].str.lstrip()
        elif func == "str_rstrip":
            return args[0].str.rstrip()
        elif func == "str_contains":
            pattern = args[1]
            regex = node.kwargs.get("regex", True)
            return args[0].str.contains(pattern, regex=regex)
        elif func == "str_startswith":
            pattern = args[1]
            return args[0].str.startswith(pattern)
        elif func == "str_endswith":
            pattern = args[1]
            return args[0].str.endswith(pattern)
        elif func == "str_replace":
            pat = args[1]
            repl = args[2]
            regex = node.kwargs.get("regex", True)
            return args[0].str.replace(pat, repl, regex=regex)
        elif func == "str_slice":
            start = node.kwargs.get("start")
            stop = node.kwargs.get("stop")
            return args[0].str.slice(start=start, stop=stop)
        elif func == "str_capitalize":
            return args[0].str.capitalize()
        elif func == "str_title":
            return args[0].str.title()
        elif func == "str_swapcase":
            return args[0].str.swapcase()
        elif func == "str_center":
            width = args[1]
            fillchar = args[2] if len(args) > 2 else " "
            return args[0].str.center(width, fillchar)
        elif func == "str_pad":
            width = args[1]
            side = node.kwargs.get("side", "left")
            fillchar = node.kwargs.get("fillchar", " ")
            return args[0].str.pad(width, side=side, fillchar=fillchar)
        elif func == "str_zfill":
            width = args[1]
            return args[0].str.zfill(width)
        elif func == "str_count":
            pattern = args[1]
            return args[0].str.count(pattern)
        elif func == "str_find":
            sub = args[1]
            start = node.kwargs.get("start", 0)
            end = node.kwargs.get("end")
            return args[0].str.find(sub, start=start, end=end)
        elif func == "str_match":
            pattern = args[1]
            return args[0].str.match(pattern)
        elif func == "str_reverse":
            return args[0].str[::-1]
        elif func == "str_is_alnum":
            return args[0].str.isalnum()
        elif func == "str_is_alpha":
            return args[0].str.isalpha()
        elif func == "str_is_digit":
            return args[0].str.isdigit()
        elif func == "str_is_decimal":
            return args[0].str.isdecimal()
        elif func == "str_is_space":
            return args[0].str.isspace()
        elif func == "str_is_lower":
            return args[0].str.islower()
        elif func == "str_is_upper":
            return args[0].str.isupper()
        elif func == "str_is_title":
            return args[0].str.istitle()
        elif func == "str_is_numeric":
            return args[0].str.isnumeric()
        elif func == "str_split":
            pattern = args[1]
            n = args[2] if len(args) > 2 else -1
            regex = node.kwargs.get("regex", False)
            return args[0].str.split(pattern, n=n, regex=regex)
        elif func == "str_rsplit":
            pattern = args[1]
            n = args[2] if len(args) > 2 else -1
            return args[0].str.rsplit(pattern, n=n)

        # Aggregation operations
        elif func == "sum":
            return args[0].sum()
        elif func == "mean":
            return args[0].mean()
        elif func == "min":
            return args[0].min()
        elif func == "max":
            return args[0].max()
        elif func == "count":
            return args[0].count()
        elif func == "std":
            ddof = node.kwargs.get("ddof", 1)
            return args[0].std(ddof=ddof)
        elif func == "var":
            ddof = node.kwargs.get("ddof", 1)
            return args[0].var(ddof=ddof)
        elif func == "first":
            return args[0].iloc[0] if len(args[0]) > 0 else None
        elif func == "last":
            return args[0].iloc[-1] if len(args[0]) > 0 else None
        elif func == "n_unique":
            return args[0].nunique()

        # Datetime operations
        elif func == "dt_year":
            return args[0].dt.year
        elif func == "dt_month":
            return args[0].dt.month
        elif func == "dt_day":
            return args[0].dt.day
        elif func == "dt_hour":
            return args[0].dt.hour
        elif func == "dt_minute":
            return args[0].dt.minute
        elif func == "dt_second":
            return args[0].dt.second
        elif func == "dt_weekday":
            return args[0].dt.day_of_week
        elif func == "dt_dayofyear":
            return args[0].dt.day_of_year
        elif func == "dt_quarter":
            return args[0].dt.quarter
        elif func == "dt_is_month_start":
            return args[0].dt.is_month_start
        elif func == "dt_is_month_end":
            return args[0].dt.is_month_end
        elif func == "dt_is_year_start":
            return args[0].dt.is_year_start
        elif func == "dt_is_year_end":
            return args[0].dt.is_year_end
        elif func == "dt_date":
            return args[0].dt.date

        # Conditional operations
        elif func == "case_when":
            return self._evaluate_case_when(node)

        # Null handling
        elif func == "fill_null":
            return self._evaluate_fill_null(args[0], args[1])

        elif func == "coalesce":
            return self._evaluate_coalesce(args)

        # Window functions
        elif func == "window":
            return self._evaluate_window(node)

        elif func == "rank":
            return args[0].rank(method="min")

        elif func == "dense_rank":
            return args[0].rank(method="dense")

        elif func == "row_number":
            # Row number is 1-based
            from pandas import Series

            return Series(range(1, len(args[0]) + 1), index=args[0].index)

        elif func == "row_index":
            # Row index is 0-based by default, with optional offset.
            # Align to the input index so downstream DataFrame construction
            # does not misalign rows for non-RangeIndex inputs.
            from pandas import Series

            offset = node.kwargs.get("offset", 0)
            return Series(range(offset, offset + len(self._df)), index=self._df.index)

        elif func == "lag":
            n = node.kwargs.get("n", 1)
            default_node = node.kwargs.get("default")
            original_dtype = args[0].dtype
            result = args[0].shift(n)
            if default_node is not None:
                default_val = self.evaluate(default_node)
                result = result.fillna(default_val)
            else:
                # Convert to nullable dtype, preserving original type
                result = _convert_series_to_nullable(result, original_dtype)
            return result

        elif func == "lead":
            n = node.kwargs.get("n", 1)
            default_node = node.kwargs.get("default")
            original_dtype = args[0].dtype
            result = args[0].shift(-n)
            if default_node is not None:
                default_val = self.evaluate(default_node)
                result = result.fillna(default_val)
            else:
                # Convert to nullable dtype, preserving original type
                result = _convert_series_to_nullable(result, original_dtype)
            return result

        elif func == "shift":
            # Parity with the physical evaluator's shift (was unimplemented in
            # the eager path). pandas Series.shift handles positive/negative
            # periods and the optional fill value directly.
            periods = node.kwargs.get("periods", 1)
            fill_value = node.kwargs.get("fill_value")
            result = args[0].shift(periods)
            if fill_value is not None:
                result = result.fillna(fill_value)
            return result

        elif func == "clip":
            # Parity with the physical evaluator (was eager-unimplemented).
            return args[0].clip(node.kwargs.get("lower"), node.kwargs.get("upper"))

        elif func == "diff":
            # Parity with the physical evaluator (was eager-unimplemented).
            return args[0].diff(node.kwargs.get("periods", 1))

        elif func == "cum_sum":
            return args[0].cumsum()

        elif func == "cum_min":
            return args[0].cummin()

        elif func == "cum_max":
            return args[0].cummax()

        elif func == "cum_mean":
            # pandas doesn't have cummean, compute manually
            from pandas import Series

            cumsum = args[0].cumsum()
            counts = Series(range(1, len(args[0]) + 1), index=args[0].index)
            return cumsum / counts

        elif func == "cum_prod":
            return args[0].cumprod()

        # Rolling window operations - convert to nullable dtype if NaN introduced
        # Operations that preserve dtype: sum, min, max
        # Operations that produce floats: mean, std, var, median
        elif func == "rolling_sum":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            original_dtype = args[0].dtype
            result = args[0].rolling(window, min_periods=min_periods).sum()
            return _convert_series_to_nullable(result, original_dtype)

        elif func == "rolling_mean":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            result = args[0].rolling(window, min_periods=min_periods).mean()
            # mean always produces floats
            return _convert_series_to_nullable(result)

        elif func == "rolling_min":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            original_dtype = args[0].dtype
            result = args[0].rolling(window, min_periods=min_periods).min()
            return _convert_series_to_nullable(result, original_dtype)

        elif func == "rolling_max":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            original_dtype = args[0].dtype
            result = args[0].rolling(window, min_periods=min_periods).max()
            return _convert_series_to_nullable(result, original_dtype)

        elif func == "rolling_std":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            ddof = node.kwargs.get("ddof", 1)
            result = args[0].rolling(window, min_periods=min_periods).std(ddof=ddof)
            # std always produces floats
            return _convert_series_to_nullable(result)

        elif func == "rolling_var":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            ddof = node.kwargs.get("ddof", 1)
            result = args[0].rolling(window, min_periods=min_periods).var(ddof=ddof)
            # var always produces floats
            return _convert_series_to_nullable(result)

        elif func == "rolling_median":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            result = args[0].rolling(window, min_periods=min_periods).median()
            # median always produces floats
            return _convert_series_to_nullable(result)

        elif func == "rolling_argmax":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")

            def rolling_argmax_func(x):
                if len(x.dropna()) < (min_periods or window):
                    return np.nan
                return x.values.argmax()

            result = (
                args[0]
                .rolling(window, min_periods=min_periods)
                .apply(rolling_argmax_func, raw=False)
            )
            return _convert_series_to_nullable(result)

        elif func == "rolling_argmin":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")

            def rolling_argmin_func(x):
                if len(x.dropna()) < (min_periods or window):
                    return np.nan
                return x.values.argmin()

            result = (
                args[0]
                .rolling(window, min_periods=min_periods)
                .apply(rolling_argmin_func, raw=False)
            )
            return _convert_series_to_nullable(result)

        elif func == "rolling_rank":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            pct = node.kwargs.get("pct", False)

            def rolling_rank_func(x):
                if len(x.dropna()) < (min_periods or window):
                    return np.nan
                # Rank of the last element within the window
                rank = (x.values < x.values[-1]).sum() + 1
                if pct:
                    return rank / len(x)
                return rank

            result = (
                args[0]
                .rolling(window, min_periods=min_periods)
                .apply(rolling_rank_func, raw=False)
            )
            return _convert_series_to_nullable(result)

        else:
            raise NotImplementedError(f"Function not implemented: {func}")

    def _evaluate_case_when(self, node) -> Any:
        """Evaluate a case_when (when/then/otherwise) expression."""

        cases = node.kwargs.get("cases", ())
        otherwise_node = node.kwargs.get("otherwise")

        # Evaluate the otherwise value first to get the base result
        otherwise_value = self.evaluate(otherwise_node)

        # If otherwise is a scalar, we need to broadcast it
        from pandas import Series

        if not hasattr(otherwise_value, "__len__") or isinstance(otherwise_value, str):
            result = Series([otherwise_value] * len(self._df), index=self._df.index)
        else:
            result = otherwise_value.copy()

        # Apply cases in reverse order (last case has lowest priority)
        for condition_node, value_node in reversed(cases):
            condition = self.evaluate(condition_node)
            value = self.evaluate(value_node)

            # Use np.where or Series.where to apply the condition
            if hasattr(value, "__len__") and not isinstance(value, str):
                # value is a Series
                result = result.where(~condition, value)
            else:
                # value is a scalar
                result = result.mask(condition, value)

        return result

    def _evaluate_window(self, node) -> Any:
        """Evaluate a window function (aggregation with .over())."""
        from pandas import Series
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        inner_node = node.args[0]
        partition_by = node.kwargs.get("partition_by", ())

        # Get the inner function to apply
        if not isinstance(inner_node, Call):
            raise ValueError("Window function requires an aggregation or window op")

        func_name = inner_node.function

        # Evaluate the column we're aggregating
        if inner_node.args:
            col_series = self.evaluate(inner_node.args[0])
        else:
            # For row_number, we don't have a column
            col_series = None

        # No partition - global window
        if not partition_by:
            if func_name in {"sum", "mean", "min", "max", "count", "std", "var"}:
                agg_value = getattr(col_series, func_name)()
                return Series([agg_value] * len(self._df), index=self._df.index)
            elif func_name == "rank":
                return col_series.rank(method="min")
            elif func_name == "dense_rank":
                return col_series.rank(method="dense")
            elif func_name == "row_number":
                return Series(range(1, len(self._df) + 1), index=self._df.index)
            elif func_name == "lag":
                n = inner_node.kwargs.get("n", 1)
                default_node = inner_node.kwargs.get("default")
                original_dtype = col_series.dtype
                result = col_series.shift(n)
                if default_node is not None:
                    default_val = self.evaluate(default_node)
                    result = result.fillna(default_val)
                else:
                    result = _convert_series_to_nullable(result, original_dtype)
                return result
            elif func_name == "lead":
                n = inner_node.kwargs.get("n", 1)
                default_node = inner_node.kwargs.get("default")
                original_dtype = col_series.dtype
                result = col_series.shift(-n)
                if default_node is not None:
                    default_val = self.evaluate(default_node)
                    result = result.fillna(default_val)
                else:
                    result = _convert_series_to_nullable(result, original_dtype)
                return result
            elif func_name == "cum_sum":
                return col_series.cumsum()
            elif func_name == "cum_min":
                return col_series.cummin()
            elif func_name == "cum_max":
                return col_series.cummax()
            elif func_name == "cum_mean":
                from pandas import Series

                cumsum = col_series.cumsum()
                counts = Series(range(1, len(col_series) + 1), index=col_series.index)
                return cumsum / counts
            elif func_name == "cum_prod":
                return col_series.cumprod()
            else:
                raise NotImplementedError(f"Window function not supported: {func_name}")

        # With partitions - group by then transform
        partition_cols = []
        for p in partition_by:
            if isinstance(p, FieldRef):
                partition_cols.append(p.name)
            else:
                # Evaluate the expression
                partition_cols.append(self.evaluate(p))

        grouped = self._df.groupby(partition_cols, sort=False)

        if func_name in {"sum", "mean", "min", "max", "count", "std", "var"}:
            # Use transform to broadcast aggregate back to each row
            col_name = None
            if isinstance(inner_node.args[0], FieldRef):
                col_name = inner_node.args[0].name

            if col_name:
                return grouped[col_name].transform(func_name)
            else:
                # If it's a computed column, we need a different approach
                return col_series.groupby(
                    [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                    sort=False,
                ).transform(func_name)

        elif func_name == "rank":
            col_name = (
                inner_node.args[0].name
                if isinstance(inner_node.args[0], FieldRef)
                else None
            )
            if col_name:
                return grouped[col_name].rank(method="min")
            return col_series.groupby(
                [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                sort=False,
            ).rank(method="min")

        elif func_name == "dense_rank":
            col_name = (
                inner_node.args[0].name
                if isinstance(inner_node.args[0], FieldRef)
                else None
            )
            if col_name:
                return grouped[col_name].rank(method="dense")
            return col_series.groupby(
                [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                sort=False,
            ).rank(method="dense")

        elif func_name == "row_number":
            # Row number within each group
            return grouped.cumcount() + 1

        elif func_name in {"lag", "lead"}:
            n = inner_node.kwargs.get("n", 1)
            if func_name == "lead":
                n = -n
            default_node = inner_node.kwargs.get("default")
            original_dtype = col_series.dtype

            col_name = (
                inner_node.args[0].name
                if isinstance(inner_node.args[0], FieldRef)
                else None
            )
            if col_name:
                result = grouped[col_name].shift(n)
            else:
                result = col_series.groupby(
                    [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                    sort=False,
                ).shift(n)

            if default_node is not None:
                default_val = self.evaluate(default_node)
                result = result.fillna(default_val)
            else:
                result = _convert_series_to_nullable(result, original_dtype)
            return result

        elif func_name == "cum_sum":
            col_name = (
                inner_node.args[0].name
                if isinstance(inner_node.args[0], FieldRef)
                else None
            )
            if col_name:
                return grouped[col_name].cumsum()
            return col_series.groupby(
                [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                sort=False,
            ).cumsum()

        elif func_name == "cum_min":
            col_name = (
                inner_node.args[0].name
                if isinstance(inner_node.args[0], FieldRef)
                else None
            )
            if col_name:
                return grouped[col_name].cummin()
            return col_series.groupby(
                [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                sort=False,
            ).cummin()

        elif func_name == "cum_max":
            col_name = (
                inner_node.args[0].name
                if isinstance(inner_node.args[0], FieldRef)
                else None
            )
            if col_name:
                return grouped[col_name].cummax()
            return col_series.groupby(
                [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                sort=False,
            ).cummax()

        elif func_name == "cum_mean":
            col_name = (
                inner_node.args[0].name
                if isinstance(inner_node.args[0], FieldRef)
                else None
            )
            if col_name:
                # GroupBy cumulative mean - compute from cumsum and cumcount
                cumsum = grouped[col_name].cumsum()
                cumcount = grouped[col_name].cumcount() + 1
                return cumsum / cumcount
            series_grouped = col_series.groupby(
                [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                sort=False,
            )
            cumsum = series_grouped.cumsum()
            cumcount = series_grouped.cumcount() + 1
            return cumsum / cumcount

        elif func_name == "cum_prod":
            col_name = (
                inner_node.args[0].name
                if isinstance(inner_node.args[0], FieldRef)
                else None
            )
            if col_name:
                return grouped[col_name].cumprod()
            return col_series.groupby(
                [self._df[c] if isinstance(c, str) else c for c in partition_cols],
                sort=False,
            ).cumprod()

        else:
            raise NotImplementedError(f"Window function not supported: {func_name}")

    def _evaluate_fill_null(self, series, fill_value) -> Any:
        """Evaluate fill_null operation."""
        import pandas as pd

        if hasattr(fill_value, "__len__") and not isinstance(fill_value, str):
            # fill_value is a Series - use combine_first or where
            return series.where(pd.notna(series), fill_value)
        else:
            # fill_value is a scalar
            return series.fillna(fill_value)

    def _evaluate_coalesce(self, args: list) -> Any:
        """Evaluate coalesce operation - return first non-null value."""
        import pandas as pd
        from pandas import Series

        if not args:
            raise ValueError("coalesce requires at least one argument")

        # Start with the first argument
        result = args[0]
        if not hasattr(result, "__len__") or isinstance(result, str):
            # First arg is a scalar, broadcast it
            result = Series([result] * len(self._df), index=self._df.index)
        else:
            result = result.copy()

        # Fill nulls from subsequent arguments
        for arg in args[1:]:
            if hasattr(arg, "__len__") and not isinstance(arg, str):
                # arg is a Series
                result = result.where(pd.notna(result), arg)
            else:
                # arg is a scalar
                result = result.fillna(arg)

        return result
