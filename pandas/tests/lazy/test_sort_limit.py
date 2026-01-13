"""
Tests for pandas.lazy sort and limit operations.
"""

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col


class TestSortPlanNode:
    """Tests for the Sort plan node."""

    def test_sort_repr(self):
        from pandas.lazy.plan import (
            DataFrameSource,
            Sort,
        )

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        source = DataFrameSource(df)
        sort = Sort(source, (col("a"),), (False,))
        result = repr(sort)
        assert "Sort" in result
        assert "a:asc" in result

    def test_sort_repr_descending(self):
        from pandas.lazy.plan import (
            DataFrameSource,
            Sort,
        )

        df = pd.DataFrame({"a": [1, 2]})
        source = DataFrameSource(df)
        sort = Sort(source, (col("a"),), (True,))
        result = repr(sort)
        assert "a:desc" in result

    def test_sort_schema_unchanged(self):
        from pandas.lazy.plan import (
            DataFrameSource,
            Sort,
        )

        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        source = DataFrameSource(df)
        sort = Sort(source, (col("a"),), (False,))
        schema = sort.resolve_schema()
        assert "a" in schema
        assert "b" in schema

    def test_sort_children(self):
        from pandas.lazy.plan import (
            DataFrameSource,
            Sort,
        )

        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        sort = Sort(source, (col("a"),), (False,))
        assert sort.children() == [source]


class TestLimitPlanNode:
    """Tests for the Limit plan node."""

    def test_limit_repr(self):
        from pandas.lazy.plan import (
            DataFrameSource,
            Limit,
        )

        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        limit = Limit(source, 5)
        assert repr(limit) == "Limit(n=5)"

    def test_limit_repr_with_offset(self):
        from pandas.lazy.plan import (
            DataFrameSource,
            Limit,
        )

        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        limit = Limit(source, 5, offset=10)
        assert repr(limit) == "Limit(n=5, offset=10)"

    def test_limit_schema_unchanged(self):
        from pandas.lazy.plan import (
            DataFrameSource,
            Limit,
        )

        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        source = DataFrameSource(df)
        limit = Limit(source, 1)
        schema = limit.resolve_schema()
        assert "a" in schema
        assert "b" in schema


class TestLazyDataFrameSort:
    """Tests for LazyDataFrame.sort()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [3, 1, 4, 1, 5],
                "b": ["c", "a", "d", "b", "e"],
                "c": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

    def test_sort_single_column_asc(self, sample_df):
        result = sample_df.select().sort("a").collect()
        expected = sample_df.sort_values("a").reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_sort_single_column_desc(self, sample_df):
        result = sample_df.select().sort("a", descending=True).collect()
        expected = sample_df.sort_values("a", ascending=False).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_sort_multiple_columns(self, sample_df):
        result = sample_df.select().sort("a", "b").collect()
        expected = sample_df.sort_values(["a", "b"]).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_sort_multiple_columns_mixed_order(self, sample_df):
        result = sample_df.select().sort("a", "b", descending=[True, False]).collect()
        expected = sample_df.sort_values(
            ["a", "b"], ascending=[False, True]
        ).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_sort_with_expr(self, sample_df):
        result = sample_df.select().sort(col("a")).collect()
        expected = sample_df.sort_values("a").reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_sort_empty_raises(self, sample_df):
        with pytest.raises(ValueError, match="requires at least one column"):
            sample_df.select().sort()

    def test_sort_descending_length_mismatch_raises(self, sample_df):
        with pytest.raises(ValueError, match="Length of descending"):
            sample_df.select().sort("a", "b", descending=[True])


class TestLazyDataFrameHead:
    """Tests for LazyDataFrame.head()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": list(range(10)),
                "b": list("abcdefghij"),
            }
        )

    def test_head_default(self, sample_df):
        result = sample_df.select().head().collect()
        expected = sample_df.head(5).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_head_n(self, sample_df):
        result = sample_df.select().head(3).collect()
        expected = sample_df.head(3).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_head_zero(self, sample_df):
        result = sample_df.select().head(0).collect()
        expected = sample_df.head(0).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_head_larger_than_df(self, sample_df):
        result = sample_df.select().head(100).collect()
        expected = sample_df.head(100).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_head_negative_raises(self, sample_df):
        with pytest.raises(ValueError, match="must be non-negative"):
            sample_df.select().head(-1)


class TestLazyDataFrameTail:
    """Tests for LazyDataFrame.tail()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": list(range(10)),
                "b": list("abcdefghij"),
            }
        )

    def test_tail_default(self, sample_df):
        result = sample_df.select().tail().collect()
        expected = sample_df.tail(5).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_tail_n(self, sample_df):
        result = sample_df.select().tail(3).collect()
        expected = sample_df.tail(3).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_tail_zero(self, sample_df):
        result = sample_df.select().tail(0).collect()
        expected = sample_df.tail(0).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_tail_larger_than_df(self, sample_df):
        result = sample_df.select().tail(100).collect()
        expected = sample_df.tail(100).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_tail_negative_raises(self, sample_df):
        with pytest.raises(ValueError, match="must be non-negative"):
            sample_df.select().tail(-1)


class TestLazyDataFrameLimit:
    """Tests for LazyDataFrame.limit()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": list(range(10)),
            }
        )

    def test_limit_is_alias_for_head(self, sample_df):
        result1 = sample_df.select().limit(3).collect()
        result2 = sample_df.select().head(3).collect()
        tm.assert_frame_equal(result1, result2)


class TestSortWithOtherOperations:
    """Tests for combining sort with filter, select, etc."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [30, 10, 20, 40, 50],
            }
        )

    def test_filter_then_sort(self, sample_df):
        result = (
            sample_df.select().filter(col("category") == "A").sort("value").collect()
        )
        filtered = sample_df[sample_df["category"] == "A"]
        expected = filtered.sort_values("value").reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_sort_then_head(self, sample_df):
        result = sample_df.select().sort("value").head(3).collect()
        expected = sample_df.sort_values("value").head(3).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_sort_descending_then_head(self, sample_df):
        result = sample_df.select().sort("value", descending=True).head(2).collect()
        expected = (
            sample_df.sort_values("value", ascending=False)
            .head(2)
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(result, expected)

    def test_filter_sort_head(self, sample_df):
        result = (
            sample_df.select()
            .filter(col("value") > 15)
            .sort("value", descending=True)
            .head(2)
            .collect()
        )
        filtered = sample_df[sample_df["value"] > 15]
        expected = (
            filtered.sort_values("value", ascending=False)
            .head(2)
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(result, expected)

    def test_with_columns_then_sort(self, sample_df):
        result = (
            sample_df.select()
            .with_columns((col("value") * 2).alias("double_value"))
            .sort("double_value")
            .collect()
        )
        expected = sample_df.copy()
        expected["double_value"] = expected["value"] * 2
        expected = expected.sort_values("double_value").reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_group_by_then_sort(self, sample_df):
        result = (
            sample_df.select()
            .group_by("category")
            .agg(col("value").sum().alias("total"))
            .sort("total", descending=True)
            .collect()
        )
        grouped = sample_df.groupby("category")["value"].sum().reset_index()
        grouped.columns = ["category", "total"]
        expected = grouped.sort_values("total", ascending=False).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)


class TestSortByExpression:
    """Tests for sorting by computed expressions."""

    def test_sort_by_arithmetic(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [3, 1, 2],
            }
        )
        result = df.select().sort(col("a") + col("b")).collect()
        # a+b = [4, 3, 5], sorted order: row1, row0, row2
        expected = df.iloc[[1, 0, 2]].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_sort_by_negated_column(self):
        df = pd.DataFrame({"a": [1, 3, 2]})
        result = df.select().sort(-col("a")).collect()
        # -a = [-1, -3, -2], sorted: row1, row2, row0
        expected = df.iloc[[1, 2, 0]].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)
