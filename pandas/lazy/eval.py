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
            return args[0].dt.weekday
        elif func == "dt_dayofyear":
            return args[0].dt.dayofyear
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

        elif func == "lag":
            n = node.kwargs.get("n", 1)
            default_node = node.kwargs.get("default")
            result = args[0].shift(n)
            if default_node is not None:
                default_val = self.evaluate(default_node)
                result = result.fillna(default_val)
            return result

        elif func == "lead":
            n = node.kwargs.get("n", 1)
            default_node = node.kwargs.get("default")
            result = args[0].shift(-n)
            if default_node is not None:
                default_val = self.evaluate(default_node)
                result = result.fillna(default_val)
            return result

        elif func == "cum_sum":
            return args[0].cumsum()

        elif func == "cum_min":
            return args[0].cummin()

        elif func == "cum_max":
            return args[0].cummax()

        # Rolling window operations
        elif func == "rolling_sum":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            return args[0].rolling(window, min_periods=min_periods).sum()

        elif func == "rolling_mean":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            return args[0].rolling(window, min_periods=min_periods).mean()

        elif func == "rolling_min":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            return args[0].rolling(window, min_periods=min_periods).min()

        elif func == "rolling_max":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            return args[0].rolling(window, min_periods=min_periods).max()

        elif func == "rolling_std":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            ddof = node.kwargs.get("ddof", 1)
            return args[0].rolling(window, min_periods=min_periods).std(ddof=ddof)

        elif func == "rolling_var":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            ddof = node.kwargs.get("ddof", 1)
            return args[0].rolling(window, min_periods=min_periods).var(ddof=ddof)

        elif func == "rolling_median":
            window = node.kwargs.get("window")
            min_periods = node.kwargs.get("min_periods")
            return args[0].rolling(window, min_periods=min_periods).median()

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
                result = col_series.shift(n)
                if default_node is not None:
                    default_val = self.evaluate(default_node)
                    result = result.fillna(default_val)
                return result
            elif func_name == "lead":
                n = inner_node.kwargs.get("n", 1)
                default_node = inner_node.kwargs.get("default")
                result = col_series.shift(-n)
                if default_node is not None:
                    default_val = self.evaluate(default_node)
                    result = result.fillna(default_val)
                return result
            elif func_name == "cum_sum":
                return col_series.cumsum()
            elif func_name == "cum_min":
                return col_series.cummin()
            elif func_name == "cum_max":
                return col_series.cummax()
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
