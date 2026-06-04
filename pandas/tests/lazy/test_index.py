"""
Tests for pandas.lazy index operations.

Tests cover:
- Index preservation with preserve_index=True
- Default RangeIndex behavior
- .with_row_index() method
- MultiIndex handling
"""

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col


class TestIndexPreservation:
    """Tests for index preservation via preserve_index parameter."""

    def test_collect_default_rangeindex(self):
        """Default collect() with physical planner returns RangeIndex."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx"))
        # Physical planner resets index by default
        result = df.select().collect(use_physical_planner=True)
        assert isinstance(result.index, pd.RangeIndex)
        assert list(result.index) == [0, 1, 2]

    def test_collect_preserve_index_single(self):
        """preserve_index=True restores original single index."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx"))
        result = df.select().collect(use_physical_planner=True, preserve_index=True)
        tm.assert_index_equal(result.index, df.index)

    def test_collect_preserve_index_unnamed(self):
        """preserve_index=True works with unnamed index."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([100, 200, 300]))
        result = df.select().collect(use_physical_planner=True, preserve_index=True)
        tm.assert_index_equal(result.index, df.index)

    def test_collect_preserve_multiindex(self):
        """preserve_index=True works with MultiIndex."""
        idx = pd.MultiIndex.from_tuples(
            [(1, "a"), (1, "b"), (2, "a")], names=["num", "letter"]
        )
        df = pd.DataFrame({"val": [10, 20, 30]}, index=idx)
        result = df.select().collect(use_physical_planner=True, preserve_index=True)
        tm.assert_index_equal(result.index, df.index)

    def test_preserve_index_through_filter(self):
        """Index preserved through filter operation."""
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4]}, index=pd.Index([10, 20, 30, 40], name="idx")
        )
        result = (
            df.select()
            .filter(col("a") > 2)
            .collect(use_physical_planner=True, preserve_index=True)
        )
        expected_index = pd.Index([30, 40], name="idx")
        tm.assert_index_equal(result.index, expected_index)

    def test_preserve_index_default_rangeindex(self):
        """Preserving RangeIndex (the default) works."""
        df = pd.DataFrame({"a": [1, 2, 3]})  # Default RangeIndex
        result = df.select().collect(use_physical_planner=True, preserve_index=True)
        # Original RangeIndex should be preserved (materialized as int64 Index)
        tm.assert_index_equal(result.index, df.index, exact="equiv")

    def test_preserve_index_streaming(self):
        """Index preservation works in streaming mode."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx"))
        batches = list(
            df.select().collect(
                use_physical_planner=True,
                streaming=True,
                batch_size=2,
                preserve_index=True,
            )
        )
        # Concatenate batches
        result = pd.concat(batches, ignore_index=False)
        tm.assert_index_equal(result.index, df.index)


class TestWithRowIndex:
    """Tests for .with_row_index() method."""

    def test_with_row_index_default_name(self):
        """with_row_index() adds 'index' column by default."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().with_row_index().collect()
        assert "index" in result.columns
        assert list(result["index"]) == [0, 1, 2]

    def test_with_row_index_custom_name(self):
        """with_row_index() respects custom column name."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().with_row_index("row_num").collect()
        assert "row_num" in result.columns
        assert "index" not in result.columns
        assert list(result["row_num"]) == [0, 1, 2]

    def test_with_row_index_offset(self):
        """with_row_index() respects offset parameter."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().with_row_index(offset=100).collect()
        assert list(result["index"]) == [100, 101, 102]

    def test_with_row_index_offset_negative(self):
        """with_row_index() works with negative offset."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().with_row_index(offset=-1).collect()
        assert list(result["index"]) == [-1, 0, 1]

    def test_with_row_index_preserves_columns(self):
        """with_row_index() preserves existing columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.select().with_row_index().collect()
        assert list(result.columns) == ["a", "b", "index"]
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == [4, 5, 6]

    def test_with_row_index_after_filter(self):
        """with_row_index() after filter gives correct row numbers."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = df.select().filter(col("a") > 2).with_row_index().collect()
        # After filter, row numbers should be 0, 1, 2 (not 2, 3, 4)
        assert list(result["index"]) == [0, 1, 2]
        assert list(result["a"]) == [3, 4, 5]

    def test_with_row_index_before_filter(self):
        """with_row_index() before filter gives post-filter row numbers.

        Note: Unlike Polars where with_row_index() materializes row numbers
        at that point in the query, lazy pandas evaluates row_index at
        execution time on the current data. This means calling with_row_index()
        before or after filter produces the same result - row numbers are
        assigned to the final filtered data.

        To get original row indices through a filter, use preserve_index=True
        in collect() with a DataFrame that has the desired index.
        """
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = df.select().with_row_index().filter(col("a") > 2).collect()
        # Row numbers are assigned at execution time (post-filter)
        assert list(result["index"]) == [0, 1, 2]
        assert list(result["a"]) == [3, 4, 5]

    def test_with_row_index_empty_dataframe(self):
        """with_row_index() works with empty DataFrame."""
        df = pd.DataFrame({"a": []})
        result = df.select().with_row_index().collect()
        assert "index" in result.columns
        assert len(result) == 0


class TestIndexEdgeCases:
    """Tests for edge cases in index handling."""

    def test_empty_dataframe_preserve_index(self):
        """Empty DataFrame with preserve_index works."""
        df = pd.DataFrame({"a": []}, index=pd.Index([], dtype="int64", name="idx"))
        result = df.select().collect(use_physical_planner=True, preserve_index=True)
        assert len(result) == 0
        assert result.index.name == "idx"

    def test_single_row_preserve_index(self):
        """Single row DataFrame with preserve_index works."""
        df = pd.DataFrame({"a": [1]}, index=pd.Index([42], name="idx"))
        result = df.select().collect(use_physical_planner=True, preserve_index=True)
        assert list(result.index) == [42]
        assert result.index.name == "idx"

    def test_string_index_preserve(self):
        """String index preserved correctly."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index(["x", "y", "z"], name="key"))
        result = df.select().collect(use_physical_planner=True, preserve_index=True)
        assert list(result.index) == ["x", "y", "z"]
        assert result.index.name == "key"

    def test_datetime_index_preserve(self):
        """DatetimeIndex preserved correctly."""
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
        result = df.select().collect(use_physical_planner=True, preserve_index=True)
        tm.assert_index_equal(result.index, dates)


class TestJoinIndexPreservation:
    """Tests for index preservation in join operations."""

    def test_inner_join_preserves_left_index(self):
        """Inner join preserves left side index with preserve_index=True."""
        left = pd.DataFrame(
            {"id": [1, 2, 3], "val": [10, 20, 30]},
            index=pd.Index([100, 200, 300], name="left_idx"),
        )
        right = pd.DataFrame({"id": [2, 3, 4], "other": ["a", "b", "c"]})
        result = (
            left.select()
            .join(right.select(), on="id")
            .collect(use_physical_planner=True, preserve_index=True)
        )
        # Inner join matches id=2,3 which have left indices 200,300
        expected_index = pd.Index([200, 300], name="left_idx")
        tm.assert_index_equal(result.index, expected_index)
        assert list(result["id"]) == [2, 3]
        assert list(result["val"]) == [20, 30]

    def test_left_join_preserves_left_index(self):
        """Left join preserves all left side indices."""
        left = pd.DataFrame(
            {"id": [1, 2, 3], "val": [10, 20, 30]},
            index=pd.Index([100, 200, 300], name="left_idx"),
        )
        right = pd.DataFrame({"id": [2, 3, 4], "other": ["a", "b", "c"]})
        result = (
            left.select()
            .join(right.select(), on="id", how="left")
            .collect(use_physical_planner=True, preserve_index=True)
        )
        # Left join keeps all left rows
        expected_index = pd.Index([100, 200, 300], name="left_idx")
        tm.assert_index_equal(result.index, expected_index)

    def test_join_default_rangeindex(self):
        """Join returns RangeIndex by default (without preserve_index)."""
        left = pd.DataFrame(
            {"id": [1, 2, 3], "val": [10, 20, 30]},
            index=pd.Index([100, 200, 300], name="left_idx"),
        )
        right = pd.DataFrame({"id": [2, 3, 4], "other": ["a", "b", "c"]})
        result = (
            left.select()
            .join(right.select(), on="id")
            .collect(use_physical_planner=True)
        )
        assert isinstance(result.index, pd.RangeIndex)

    def test_join_multiindex_preserved(self):
        """Join preserves MultiIndex from left side."""
        idx = pd.MultiIndex.from_tuples(
            [(1, "a"), (2, "b"), (3, "c")], names=["num", "letter"]
        )
        left = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]}, index=idx)
        right = pd.DataFrame({"id": [2, 3, 4], "other": ["x", "y", "z"]})
        result = (
            left.select()
            .join(right.select(), on="id")
            .collect(use_physical_planner=True, preserve_index=True)
        )
        # Inner join matches id=2,3
        expected_idx = pd.MultiIndex.from_tuples(
            [(2, "b"), (3, "c")], names=["num", "letter"]
        )
        tm.assert_index_equal(result.index, expected_idx)


class TestSetIndex:
    """Tests for .set_index() method."""

    def test_set_index_single_column(self):
        """set_index() sets column as index."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.select().set_index("a").collect()
        assert result.index.name == "a"
        assert list(result.index) == [1, 2, 3]
        assert "a" not in result.columns
        assert list(result["b"]) == [4, 5, 6]

    def test_set_index_drop_false(self):
        """set_index(drop=False) keeps column."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.select().set_index("a", drop=False).collect()
        assert result.index.name == "a"
        assert list(result.index) == [1, 2, 3]
        assert "a" in result.columns
        assert list(result["a"]) == [1, 2, 3]

    def test_set_index_multiple_columns(self):
        """set_index() with multiple columns creates MultiIndex."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"], "c": [10, 20, 30]})
        result = df.select().set_index(["a", "b"]).collect()
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["a", "b"]
        assert "a" not in result.columns
        assert "b" not in result.columns
        assert list(result["c"]) == [10, 20, 30]

    def test_set_index_physical_planner(self):
        """set_index() works with physical planner."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = (
            df.select()
            .set_index("a")
            .collect(use_physical_planner=True, preserve_index=True)
        )
        assert result.index.name == "a"
        assert list(result.index) == [1, 2, 3]
        assert "a" not in result.columns

    def test_set_index_after_filter(self):
        """set_index() after filter works correctly."""
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        result = df.select().filter(col("b") > 15).set_index("a").collect()
        assert result.index.name == "a"
        assert list(result.index) == [2, 3, 4]
        assert list(result["b"]) == [20, 30, 40]

    def test_set_index_string_column(self):
        """set_index() works with string column."""
        df = pd.DataFrame({"key": ["x", "y", "z"], "val": [1, 2, 3]})
        result = df.select().set_index("key").collect()
        assert result.index.name == "key"
        assert list(result.index) == ["x", "y", "z"]


class TestResetIndex:
    """Tests for .reset_index() method."""

    def test_reset_index_to_column(self):
        """reset_index() converts index to column."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx"))
        result = df.select().reset_index().collect()
        assert "idx" in result.columns
        assert isinstance(result.index, pd.RangeIndex)
        assert list(result["idx"]) == [10, 20, 30]
        assert list(result["a"]) == [1, 2, 3]

    def test_reset_index_drop(self):
        """reset_index(drop=True) discards index."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx"))
        result = df.select().reset_index(drop=True).collect()
        assert "idx" not in result.columns
        assert isinstance(result.index, pd.RangeIndex)
        assert list(result["a"]) == [1, 2, 3]

    def test_reset_index_unnamed(self):
        """reset_index() with unnamed index creates 'index' column."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30]))
        result = df.select().reset_index().collect()
        assert "index" in result.columns
        assert list(result["index"]) == [10, 20, 30]

    def test_reset_index_physical_planner(self):
        """reset_index() works with physical planner."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx"))
        result = df.select().reset_index().collect(use_physical_planner=True)
        assert "idx" in result.columns
        assert isinstance(result.index, pd.RangeIndex)

    def test_reset_index_after_set_index(self):
        """reset_index() after set_index() round-trips correctly."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.select().set_index("a").reset_index().collect()
        # Should have both columns back
        assert "a" in result.columns
        assert "b" in result.columns
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == [4, 5, 6]

    def test_reset_multiindex(self):
        """reset_index() works with MultiIndex."""
        idx = pd.MultiIndex.from_tuples(
            [(1, "a"), (1, "b"), (2, "a")], names=["num", "letter"]
        )
        df = pd.DataFrame({"val": [10, 20, 30]}, index=idx)
        result = df.select().reset_index().collect()
        assert "num" in result.columns
        assert "letter" in result.columns
        assert list(result["num"]) == [1, 1, 2]
        assert list(result["letter"]) == ["a", "b", "a"]
        assert isinstance(result.index, pd.RangeIndex)


class TestGroupByIndexPreservation:
    """Tests for index preservation in groupby operations."""

    def test_groupby_default_rangeindex(self):
        """GroupBy returns RangeIndex by default (group keys as columns)."""
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
        result = (
            df.select()
            .group_by("group")
            .agg(col("val").sum().alias("val"))
            .collect(use_physical_planner=True)
        )
        # Default: group key is a regular column, index is RangeIndex
        assert isinstance(result.index, pd.RangeIndex)
        assert "group" in result.columns

    def test_groupby_preserve_index_single_key(self):
        """GroupBy with preserve_index=True puts group key as index."""
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
        result = (
            df.select()
            .group_by("group")
            .agg(col("val").sum().alias("val"))
            .collect(use_physical_planner=True, preserve_index=True)
        )
        # With preserve_index: group key becomes index
        assert result.index.name == "group"
        assert set(result.index) == {"A", "B"}
        assert "group" not in result.columns
        # A: 1+2=3, B: 3+4=7
        expected_sums = {"A": 3, "B": 7}
        for idx_val in result.index:
            assert result.loc[idx_val, "val"] == expected_sums[idx_val]

    def test_groupby_preserve_index_multi_key(self):
        """GroupBy with multiple keys creates MultiIndex when preserve_index=True."""
        df = pd.DataFrame(
            {
                "region": ["East", "East", "West", "West"],
                "year": [2020, 2021, 2020, 2021],
                "sales": [100, 200, 300, 400],
            }
        )
        result = (
            df.select()
            .group_by("region", "year")
            .agg(col("sales").sum().alias("sales"))
            .collect(use_physical_planner=True, preserve_index=True)
        )
        # With preserve_index: group keys become MultiIndex
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["region", "year"]
        assert "region" not in result.columns
        assert "year" not in result.columns
        # Check values
        assert result.loc[("East", 2020), "sales"] == 100
        assert result.loc[("East", 2021), "sales"] == 200
        assert result.loc[("West", 2020), "sales"] == 300
        assert result.loc[("West", 2021), "sales"] == 400

    def test_groupby_multi_key_default(self):
        """GroupBy with multiple keys returns group keys as columns by default."""
        df = pd.DataFrame(
            {
                "region": ["East", "East", "West", "West"],
                "year": [2020, 2021, 2020, 2021],
                "sales": [100, 200, 300, 400],
            }
        )
        result = (
            df.select()
            .group_by("region", "year")
            .agg(col("sales").sum().alias("sales"))
            .collect(use_physical_planner=True)
        )
        # Default: group keys are regular columns
        assert isinstance(result.index, pd.RangeIndex)
        assert "region" in result.columns
        assert "year" in result.columns

    def test_groupby_multiple_aggregations_preserve_index(self):
        """GroupBy with multiple aggregations preserves index correctly."""
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
        result = (
            df.select()
            .group_by("group")
            .agg(col("val").sum().alias("total"), col("val").mean().alias("avg"))
            .collect(use_physical_planner=True, preserve_index=True)
        )
        assert result.index.name == "group"
        assert set(result.index) == {"A", "B"}
        assert "group" not in result.columns
        assert "total" in result.columns
        assert "avg" in result.columns
        # A: sum=3, mean=1.5; B: sum=7, mean=3.5
        assert result.loc["A", "total"] == 3
        assert result.loc["A", "avg"] == 1.5
        assert result.loc["B", "total"] == 7
        assert result.loc["B", "avg"] == 3.5

    def test_groupby_string_group_key_preserve_index(self):
        """GroupBy with string group key creates string index."""
        df = pd.DataFrame(
            {"category": ["cat", "dog", "cat", "dog"], "count": [1, 2, 3, 4]}
        )
        result = (
            df.select()
            .group_by("category")
            .agg(col("count").sum().alias("count"))
            .collect(use_physical_planner=True, preserve_index=True)
        )
        assert result.index.name == "category"
        # cat: 1+3=4, dog: 2+4=6
        assert result.loc["cat", "count"] == 4
        assert result.loc["dog", "count"] == 6
