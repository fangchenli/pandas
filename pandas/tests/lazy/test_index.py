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
        # Original RangeIndex should be preserved
        tm.assert_index_equal(result.index, df.index)

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
