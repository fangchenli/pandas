"""
Tests for pandas.lazy datetime operations.
"""

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.ir import Call


class TestExprDatetimeAccessor:
    """Tests for Expr.dt accessor methods."""

    def test_dt_accessor_returns_accessor(self):
        from pandas.lazy.expr import ExprDatetimeAccessor

        expr = col("ts")
        assert isinstance(expr.dt, ExprDatetimeAccessor)

    def test_dt_year_ir(self):
        expr = col("ts").dt.year
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_year"

    def test_dt_month_ir(self):
        expr = col("ts").dt.month
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_month"

    def test_dt_day_ir(self):
        expr = col("ts").dt.day
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_day"

    def test_dt_hour_ir(self):
        expr = col("ts").dt.hour
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_hour"

    def test_dt_minute_ir(self):
        expr = col("ts").dt.minute
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_minute"

    def test_dt_second_ir(self):
        expr = col("ts").dt.second
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_second"

    def test_dt_weekday_ir(self):
        expr = col("ts").dt.weekday
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_weekday"

    def test_dt_dayofweek_ir(self):
        expr = col("ts").dt.dayofweek
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_weekday"  # alias

    def test_dt_dayofyear_ir(self):
        expr = col("ts").dt.dayofyear
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_dayofyear"

    def test_dt_quarter_ir(self):
        expr = col("ts").dt.quarter
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_quarter"

    def test_dt_is_month_start_ir(self):
        expr = col("ts").dt.is_month_start
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_is_month_start"

    def test_dt_is_month_end_ir(self):
        expr = col("ts").dt.is_month_end
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_is_month_end"

    def test_dt_date_ir(self):
        expr = col("ts").dt.date
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dt_date"


class TestEvaluatorDatetimeOps:
    """Tests for datetime operations in the evaluator."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    [
                        "2023-01-15 10:30:45",
                        "2023-06-20 14:15:30",
                        "2023-12-31 23:59:59",
                        "2024-01-01 00:00:00",
                        "2024-03-31 12:00:00",
                    ]
                )
            }
        )

    def test_dt_year(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_year", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.year
        tm.assert_series_equal(result, expected)

    def test_dt_month(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_month", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.month
        tm.assert_series_equal(result, expected)

    def test_dt_day(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_day", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.day
        tm.assert_series_equal(result, expected)

    def test_dt_hour(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_hour", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.hour
        tm.assert_series_equal(result, expected)

    def test_dt_minute(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_minute", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.minute
        tm.assert_series_equal(result, expected)

    def test_dt_second(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_second", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.second
        tm.assert_series_equal(result, expected)

    def test_dt_weekday(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_weekday", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.day_of_week
        tm.assert_series_equal(result, expected)

    def test_dt_dayofyear(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_dayofyear", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.day_of_year
        tm.assert_series_equal(result, expected)

    def test_dt_quarter(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_quarter", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.quarter
        tm.assert_series_equal(result, expected)

    def test_dt_is_month_start(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_is_month_start", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.is_month_start
        tm.assert_series_equal(result, expected)

    def test_dt_is_month_end(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_is_month_end", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.is_month_end
        tm.assert_series_equal(result, expected)

    def test_dt_is_year_start(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_is_year_start", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.is_year_start
        tm.assert_series_equal(result, expected)

    def test_dt_is_year_end(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_is_year_end", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.is_year_end
        tm.assert_series_equal(result, expected)

    def test_dt_date(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("dt_date", (FieldRef("ts"),))
        result = evaluator.evaluate(node)
        expected = sample_df["ts"].dt.date
        tm.assert_series_equal(result, expected)


class TestLazyFrameDatetimeOps:
    """Tests for datetime operations in LazyDataFrame workflows."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    [
                        "2023-01-15 10:30:45",
                        "2023-06-20 14:15:30",
                        "2023-12-31 23:59:59",
                        "2024-01-01 00:00:00",
                    ]
                ),
                "value": [100, 200, 300, 400],
            }
        )

    def test_with_columns_dt_year(self, sample_df):
        result = (
            sample_df.select().with_columns(col("ts").dt.year.alias("year")).collect()
        )
        expected = sample_df.copy()
        expected["year"] = sample_df["ts"].dt.year
        tm.assert_frame_equal(result, expected)

    def test_with_columns_dt_month(self, sample_df):
        result = (
            sample_df.select().with_columns(col("ts").dt.month.alias("month")).collect()
        )
        expected = sample_df.copy()
        expected["month"] = sample_df["ts"].dt.month
        tm.assert_frame_equal(result, expected)

    def test_with_columns_multiple_dt_components(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(
                col("ts").dt.year.alias("year"),
                col("ts").dt.month.alias("month"),
                col("ts").dt.day.alias("day"),
            )
            .collect()
        )
        expected = sample_df.copy()
        expected["year"] = sample_df["ts"].dt.year
        expected["month"] = sample_df["ts"].dt.month
        expected["day"] = sample_df["ts"].dt.day
        tm.assert_frame_equal(result, expected)

    def test_filter_by_dt_year(self, sample_df):
        result = sample_df.select().filter(col("ts").dt.year == 2023).collect()
        expected = sample_df[sample_df["ts"].dt.year == 2023].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_filter_by_dt_month(self, sample_df):
        result = sample_df.select().filter(col("ts").dt.month == 1).collect()
        expected = sample_df[sample_df["ts"].dt.month == 1].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_filter_dt_is_year_start(self, sample_df):
        result = sample_df.select().filter(col("ts").dt.is_year_start).collect()
        expected = sample_df[sample_df["ts"].dt.is_year_start].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_group_by_dt_year(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("ts").dt.year.alias("year"))
            .group_by("year")
            .agg(col("value").sum().alias("total"))
            .sort("year")
            .collect()
        )
        expected = sample_df.copy()
        expected["year"] = sample_df["ts"].dt.year
        expected = expected.groupby("year")["value"].sum().reset_index()
        expected.columns = ["year", "total"]
        expected = expected.sort_values("year").reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_group_by_dt_quarter(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("ts").dt.quarter.alias("quarter"))
            .group_by("quarter")
            .agg(col("value").mean().alias("avg"))
            .sort("quarter")
            .collect()
        )
        expected = sample_df.copy()
        expected["quarter"] = sample_df["ts"].dt.quarter
        expected = expected.groupby("quarter")["value"].mean().reset_index()
        expected.columns = ["quarter", "avg"]
        expected = expected.sort_values("quarter").reset_index(drop=True)
        tm.assert_frame_equal(result, expected)


class TestDatetimeTypeInference:
    """Tests for datetime type inference."""

    def test_dt_year_infers_numeric(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"ts": LazyDtype("datetime", None, None, False)})
        node = Call("dt_year", (FieldRef("ts"),))
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "numeric"

    def test_dt_is_month_start_infers_boolean(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"ts": LazyDtype("datetime", None, None, False)})
        node = Call("dt_is_month_start", (FieldRef("ts"),))
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "boolean"

    def test_dt_date_infers_datetime(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"ts": LazyDtype("datetime", None, None, False)})
        node = Call("dt_date", (FieldRef("ts"),))
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "datetime"
