"""
Tests for pandas.lazy window functions.
"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.ir import Call


class TestWindowAPI:
    """Tests for window function expression API."""

    def test_over_no_partition(self):
        expr = col("a").sum().over()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "window"

    def test_over_single_partition(self):
        expr = col("a").sum().over("group")
        ir = expr._ir
        assert isinstance(ir, Call)
        assert ir.function == "window"
        assert len(ir.kwargs["partition_by"]) == 1

    def test_over_multiple_partitions(self):
        expr = col("a").sum().over(["group1", "group2"])
        ir = expr._ir
        assert isinstance(ir, Call)
        assert len(ir.kwargs["partition_by"]) == 2

    def test_rank_ir(self):
        expr = col("a").rank()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "rank"

    def test_dense_rank_ir(self):
        expr = col("a").dense_rank()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "dense_rank"

    def test_row_number_ir(self):
        expr = col("a").row_number()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "row_number"

    def test_lag_ir(self):
        expr = col("a").lag(2)
        ir = expr._ir
        assert isinstance(ir, Call)
        assert ir.function == "lag"
        assert ir.kwargs["n"] == 2

    def test_lead_ir(self):
        expr = col("a").lead(3)
        ir = expr._ir
        assert isinstance(ir, Call)
        assert ir.function == "lead"
        assert ir.kwargs["n"] == 3

    def test_cum_sum_ir(self):
        expr = col("a").cum_sum()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "cum_sum"

    def test_cum_min_ir(self):
        expr = col("a").cum_min()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "cum_min"

    def test_cum_max_ir(self):
        expr = col("a").cum_max()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "cum_max"


class TestWindowAggregations:
    """Tests for window aggregations with .over()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [1, 2, 3, 10, 20, 30],
            }
        )

    def test_sum_over_global(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").sum().over().alias("total"))
            .collect()
        )
        expected = sample_df.copy()
        expected["total"] = 66  # sum of all values
        tm.assert_frame_equal(result, expected)

    def test_sum_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").sum().over("group").alias("group_total"))
            .collect()
        )
        expected = sample_df.copy()
        expected["group_total"] = [6, 6, 6, 60, 60, 60]
        tm.assert_frame_equal(result, expected)

    def test_mean_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").mean().over("group").alias("group_mean"))
            .collect()
        )
        expected = sample_df.copy()
        expected["group_mean"] = [2.0, 2.0, 2.0, 20.0, 20.0, 20.0]
        tm.assert_frame_equal(result, expected)

    def test_min_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").min().over("group").alias("group_min"))
            .collect()
        )
        expected = sample_df.copy()
        expected["group_min"] = [1, 1, 1, 10, 10, 10]
        tm.assert_frame_equal(result, expected)

    def test_max_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").max().over("group").alias("group_max"))
            .collect()
        )
        expected = sample_df.copy()
        expected["group_max"] = [3, 3, 3, 30, 30, 30]
        tm.assert_frame_equal(result, expected)

    def test_count_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").count().over("group").alias("group_count"))
            .collect()
        )
        expected = sample_df.copy()
        expected["group_count"] = [3, 3, 3, 3, 3, 3]
        tm.assert_frame_equal(result, expected)


class TestWindowRankFunctions:
    """Tests for ranking window functions."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [3, 1, 2, 30, 10, 20],
            }
        )

    def test_rank_global(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").rank().over().alias("rank"))
            .collect()
        )
        expected = sample_df.copy()
        expected["rank"] = sample_df["value"].rank(method="min")
        tm.assert_frame_equal(result, expected)

    def test_rank_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").rank().over("group").alias("rank"))
            .collect()
        )
        expected = sample_df.copy()
        # Within A: 3->3, 1->1, 2->2
        # Within B: 30->3, 10->1, 20->2
        expected["rank"] = [3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
        tm.assert_frame_equal(result, expected)

    def test_dense_rank_over_partition(self, sample_df):
        # Add ties for dense rank testing
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "A"],
                "value": [1, 2, 2, 3],
            }
        )
        result = (
            df.select()
            .with_columns(col("value").dense_rank().over("group").alias("dense_rank"))
            .collect()
        )
        expected = df.copy()
        expected["dense_rank"] = [1.0, 2.0, 2.0, 3.0]
        tm.assert_frame_equal(result, expected)

    def test_row_number_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").row_number().over("group").alias("row_num"))
            .collect()
        )
        expected = sample_df.copy()
        # Row numbers within each group (0-indexed cumcount + 1)
        expected["row_num"] = [1, 2, 3, 1, 2, 3]
        tm.assert_frame_equal(result, expected)


class TestWindowLagLead:
    """Tests for lag/lead window functions."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [1, 2, 3, 10, 20, 30],
            }
        )

    def test_lag_global(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").lag(1).over().alias("prev"))
            .collect()
        )
        expected = sample_df.copy()
        expected["prev"] = [np.nan, 1.0, 2.0, 3.0, 10.0, 20.0]
        tm.assert_frame_equal(result, expected)

    def test_lag_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").lag(1).over("group").alias("prev"))
            .collect()
        )
        expected = sample_df.copy()
        expected["prev"] = [np.nan, 1.0, 2.0, np.nan, 10.0, 20.0]
        tm.assert_frame_equal(result, expected)

    def test_lag_with_default(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").lag(1, default=0).over("group").alias("prev"))
            .collect()
        )
        expected = sample_df.copy()
        expected["prev"] = [0.0, 1.0, 2.0, 0.0, 10.0, 20.0]
        tm.assert_frame_equal(result, expected)

    def test_lead_global(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").lead(1).over().alias("next"))
            .collect()
        )
        expected = sample_df.copy()
        expected["next"] = [2.0, 3.0, 10.0, 20.0, 30.0, np.nan]
        tm.assert_frame_equal(result, expected)

    def test_lead_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").lead(1).over("group").alias("next"))
            .collect()
        )
        expected = sample_df.copy()
        expected["next"] = [2.0, 3.0, np.nan, 20.0, 30.0, np.nan]
        tm.assert_frame_equal(result, expected)

    def test_lead_with_default(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").lead(1, default=-1).over("group").alias("next"))
            .collect()
        )
        expected = sample_df.copy()
        expected["next"] = [2.0, 3.0, -1.0, 20.0, 30.0, -1.0]
        tm.assert_frame_equal(result, expected)

    def test_lag_n2(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").lag(2).over("group").alias("prev2"))
            .collect()
        )
        expected = sample_df.copy()
        expected["prev2"] = [np.nan, np.nan, 1.0, np.nan, np.nan, 10.0]
        tm.assert_frame_equal(result, expected)


class TestWindowCumulative:
    """Tests for cumulative window functions."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [1, 2, 3, 10, 20, 30],
            }
        )

    def test_cum_sum_global(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").cum_sum().over().alias("running_total"))
            .collect()
        )
        expected = sample_df.copy()
        expected["running_total"] = [1, 3, 6, 16, 36, 66]
        tm.assert_frame_equal(result, expected)

    def test_cum_sum_over_partition(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("value").cum_sum().over("group").alias("group_running"))
            .collect()
        )
        expected = sample_df.copy()
        expected["group_running"] = [1, 3, 6, 10, 30, 60]
        tm.assert_frame_equal(result, expected)

    def test_cum_min_over_partition(self, sample_df):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [3, 1, 2, 30, 10, 20],
            }
        )
        result = (
            df.select()
            .with_columns(col("value").cum_min().over("group").alias("running_min"))
            .collect()
        )
        expected = df.copy()
        expected["running_min"] = [3, 1, 1, 30, 10, 10]
        tm.assert_frame_equal(result, expected)

    def test_cum_max_over_partition(self, sample_df):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [1, 3, 2, 10, 30, 20],
            }
        )
        result = (
            df.select()
            .with_columns(col("value").cum_max().over("group").alias("running_max"))
            .collect()
        )
        expected = df.copy()
        expected["running_max"] = [1, 3, 3, 10, 30, 30]
        tm.assert_frame_equal(result, expected)


class TestWindowMultiplePartitions:
    """Tests for window functions with multiple partition columns."""

    def test_sum_over_two_partitions(self):
        df = pd.DataFrame(
            {
                "region": ["East", "East", "East", "West", "West", "West"],
                "year": [2020, 2020, 2021, 2020, 2020, 2021],
                "sales": [100, 200, 300, 400, 500, 600],
            }
        )
        result = (
            df.select()
            .with_columns(
                col("sales").sum().over(["region", "year"]).alias("region_year_total")
            )
            .collect()
        )
        expected = df.copy()
        # East 2020: 100+200=300, East 2021: 300, West 2020: 400+500=900, West 2021: 600
        expected["region_year_total"] = [300, 300, 300, 900, 900, 600]
        tm.assert_frame_equal(result, expected)


class TestWindowTypeInference:
    """Tests for type inference of window functions."""

    def test_window_sum_type(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"a": LazyDtype("numeric", np.dtype("int64"), None, False)})
        # col("a").sum().over()
        inner = Call("sum", (FieldRef("a"),), is_aggregate=True)
        node = Call("window", (inner,), {"partition_by": ()})
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "numeric"

    def test_rank_type(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"a": LazyDtype("numeric", np.dtype("int64"), None, False)})
        node = Call("rank", (FieldRef("a"),))
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "numeric"

    def test_lag_preserves_type(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"a": LazyDtype("string", None, None, False)})
        node = Call("lag", (FieldRef("a"),), {"n": 1, "default": None})
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "string"

    def test_cum_sum_type(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"a": LazyDtype("numeric", np.dtype("float64"), None, False)})
        node = Call("cum_sum", (FieldRef("a"),))
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "numeric"
