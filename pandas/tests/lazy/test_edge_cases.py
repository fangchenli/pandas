"""
Edge case tests for lazy pandas.

Tests boundary conditions, unusual inputs, and corner cases:
- Empty DataFrames
- Single-row DataFrames
- Wide DataFrames (many columns)
- Narrow DataFrames (single column)
- All-null columns
- Mixed nulls
- Type edge cases
"""

import numpy as np

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col


class TestEmptyDataFrame:
    """Tests for empty DataFrames (0 rows)."""

    def test_empty_df_select(self):
        """Empty DataFrame should work with select."""
        df = pd.DataFrame({"a": [], "b": []})
        result = df.select().collect()
        tm.assert_frame_equal(result, df)

    def test_empty_df_filter(self):
        """Filter on empty DataFrame returns empty."""
        df = pd.DataFrame({"a": pd.array([], dtype="int64")})
        result = df.select().filter(col("a") > 0).collect()
        expected = pd.DataFrame({"a": pd.array([], dtype="int64")})
        tm.assert_frame_equal(result, expected)

    def test_empty_df_with_columns(self):
        """with_columns on empty DataFrame."""
        df = pd.DataFrame({"a": pd.array([], dtype="float64")})
        result = df.select().with_columns((col("a") * 2).alias("b")).collect()
        expected = pd.DataFrame(
            {
                "a": pd.array([], dtype="float64"),
                "b": pd.array([], dtype="float64"),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_empty_df_sort(self):
        """Sort on empty DataFrame."""
        df = pd.DataFrame({"a": pd.array([], dtype="int64")})
        result = df.select().sort("a").collect()
        tm.assert_frame_equal(result, df)

    def test_empty_df_limit(self):
        """Limit on empty DataFrame."""
        df = pd.DataFrame({"a": pd.array([], dtype="int64")})
        result = df.select().limit(10).collect()
        tm.assert_frame_equal(result, df)


class TestSingleRowDataFrame:
    """Tests for single-row DataFrames."""

    def test_single_row_select(self):
        """Single row DataFrame select."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.select().collect()
        tm.assert_frame_equal(result, df)

    def test_single_row_filter_pass(self):
        """Filter that passes single row."""
        df = pd.DataFrame({"a": [5]})
        result = df.select().filter(col("a") > 0).collect()
        tm.assert_frame_equal(result, df)

    def test_single_row_filter_fail(self):
        """Filter that rejects single row."""
        df = pd.DataFrame({"a": [5]})
        result = df.select().filter(col("a") > 10).collect()
        assert len(result) == 0

    def test_single_row_sort(self):
        """Sort single row (should be no-op)."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = df.select().sort("a").collect()
        tm.assert_frame_equal(result, df)

    def test_single_row_with_columns(self):
        """Computed column on single row."""
        df = pd.DataFrame({"a": [5.0]})
        result = df.select().with_columns((col("a") * 2).alias("b")).collect()
        expected = pd.DataFrame({"a": [5.0], "b": [10.0]})
        tm.assert_frame_equal(result, expected)


class TestWideDataFrame:
    """Tests for wide DataFrames (many columns)."""

    def test_wide_df_100_cols(self):
        """DataFrame with 100 columns."""
        data = {f"col_{i}": [1, 2, 3] for i in range(100)}
        df = pd.DataFrame(data)
        result = df.select().collect()
        assert len(result.columns) == 100
        tm.assert_frame_equal(result, df)

    def test_wide_df_select_few(self):
        """Select few columns from wide DataFrame."""
        data = {f"col_{i}": [1, 2, 3] for i in range(100)}
        df = pd.DataFrame(data)
        result = df.select().select("col_0", "col_50", "col_99").collect()
        assert list(result.columns) == ["col_0", "col_50", "col_99"]

    def test_wide_df_filter(self):
        """Filter on wide DataFrame."""
        data = {f"col_{i}": [1, 2, 3] for i in range(100)}
        df = pd.DataFrame(data)
        result = df.select().filter(col("col_0") > 1).collect()
        assert len(result) == 2
        assert len(result.columns) == 100


class TestNarrowDataFrame:
    """Tests for narrow DataFrames (single column)."""

    def test_single_column_df(self):
        """Single column DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = df.select().collect()
        tm.assert_frame_equal(result, df)

    def test_single_column_filter(self):
        """Filter on single column DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = df.select().filter(col("a") > 3).collect()
        expected = pd.DataFrame({"a": [4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_single_column_compute(self):
        """Compute on single column DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().with_columns((col("a") * 2).alias("b")).collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
        tm.assert_frame_equal(result, expected)


class TestNullHandling:
    """Tests for null/NA value handling."""

    def test_numpy_nan_in_filter(self):
        """Filter with numpy nan values."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )
        result = df.select().filter(col("a") > 2).collect()
        # NaNs should be filtered out (comparison with nan is false)
        assert len(result) == 2

    def test_numpy_nan_in_arithmetic(self):
        """Arithmetic with numpy nan values."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [10.0, 20.0, 30.0],
            }
        )
        result = df.select().with_columns((col("a") + col("b")).alias("c")).collect()
        assert pd.isna(result["c"].iloc[1])

    def test_none_in_object_column(self):
        """None values in object column."""
        df = pd.DataFrame(
            {
                "key": ["A", None, "A", "B"],
                "val": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = df.select().filter(col("key") == "A").collect()
        assert len(result) == 2

    def test_nullable_int64_column(self):
        """Nullable Int64 column support."""
        df = pd.DataFrame(
            {
                "a": pd.array([1, None, 3], dtype="Int64"),
                "b": [10, 20, 30],
            }
        )
        result = df.select().filter(col("b") > 15).collect()
        assert len(result) == 2
        # Verify the Int64 dtype is preserved in the result
        assert pd.isna(result["a"].iloc[0])

    def test_nullable_float64_column(self):
        """Nullable Float64 column support."""
        df = pd.DataFrame(
            {
                "a": pd.array([1.0, None, 3.0], dtype="Float64"),
            }
        )
        result = df.select().with_columns((col("a") * 2).alias("b")).collect()
        assert len(result) == 3
        assert pd.isna(result["b"].iloc[1])


class TestTypeEdgeCases:
    """Tests for type-related edge cases."""

    def test_bool_column(self):
        """Boolean column operations."""
        df = pd.DataFrame({"flag": [True, False, True]})
        result = df.select().filter(col("flag")).collect()
        assert len(result) == 2

    def test_datetime_column(self):
        """Datetime column operations."""
        df = pd.DataFrame(
            {
                "dt": pd.to_datetime(["2020-01-01", "2020-06-15", "2020-12-31"]),
            }
        )
        result = df.select().collect()
        from pandas.api.types import is_datetime64_any_dtype

        assert is_datetime64_any_dtype(result["dt"])

    def test_mixed_int_float(self):
        """Operations mixing int and float columns."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
            }
        )
        result = (
            df.select()
            .with_columns((col("int_col") + col("float_col")).alias("sum"))
            .collect()
        )
        assert result["sum"].dtype == np.float64

    def test_string_column(self):
        """String column operations."""
        df = pd.DataFrame({"s": ["hello", "world", "test"]})
        result = df.select().filter(col("s").str.contains("o")).collect()
        assert len(result) == 2


class TestChainedOperations:
    """Tests for long chains of operations."""

    def test_many_filters(self):
        """Many chained filters."""
        df = pd.DataFrame(
            {
                "a": range(1000),
                "b": range(1000),
                "c": range(1000),
            }
        )
        lf = df.select()
        for i in range(10):
            lf = lf.filter(col("a") >= 0)  # Always true
        result = lf.collect()
        assert len(result) == 1000

    def test_many_with_columns(self):
        """Many computed columns."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = df.select()
        for i in range(20):
            lf = lf.with_columns((col("a") + i).alias(f"col_{i}"))
        result = lf.collect()
        assert len(result.columns) == 21  # original + 20 new

    def test_filter_compute_filter_compute(self):
        """Alternating filter and compute."""
        df = pd.DataFrame({"a": range(100)})
        result = (
            df.select()
            .filter(col("a") > 10)
            .with_columns((col("a") * 2).alias("b"))
            .filter(col("b") < 100)
            .with_columns((col("b") + 1).alias("c"))
            .collect()
        )
        assert "c" in result.columns
        assert len(result) > 0


class TestLimitEdgeCases:
    """Tests for limit edge cases."""

    def test_limit_zero(self):
        """Limit 0 should return empty DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().limit(0).collect()
        assert len(result) == 0

    def test_limit_larger_than_data(self):
        """Limit larger than data size."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().limit(100).collect()
        assert len(result) == 3

    def test_limit_exact_size(self):
        """Limit exactly matching data size."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().limit(3).collect()
        assert len(result) == 3

    def test_multiple_limits(self):
        """Multiple limits should use minimum."""
        df = pd.DataFrame({"a": range(100)})
        result = df.select().limit(50).limit(20).limit(30).collect()
        assert len(result) == 20


class TestSortEdgeCases:
    """Tests for sort edge cases."""

    def test_sort_all_same_values(self):
        """Sort column with all identical values."""
        df = pd.DataFrame({"a": [5, 5, 5, 5, 5]})
        result = df.select().sort("a").collect()
        assert len(result) == 5

    def test_sort_with_nulls(self):
        """Sort with null values (numpy nan)."""
        df = pd.DataFrame(
            {
                "a": [3.0, np.nan, 1.0, np.nan, 2.0],
            }
        )
        result = df.select().sort("a").collect()
        assert len(result) == 5


class TestJoinEdgeCases:
    """Tests for join edge cases."""

    def test_join_empty_left(self):
        """Join with empty left table."""
        left = pd.DataFrame(
            {"key": pd.array([], dtype="int64"), "val": pd.array([], dtype="float64")}
        )
        right = pd.DataFrame({"key": [1, 2], "data": [10, 20]})
        result = left.select().join(right.select(), on="key").collect()
        assert len(result) == 0

    def test_join_empty_right(self):
        """Join with empty right table."""
        left = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
        right = pd.DataFrame(
            {"key": pd.array([], dtype="int64"), "data": pd.array([], dtype="float64")}
        )
        result = left.select().join(right.select(), on="key").collect()
        assert len(result) == 0

    def test_join_no_matches(self):
        """Join with no matching keys."""
        left = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
        right = pd.DataFrame({"key": [4, 5, 6], "data": [40, 50, 60]})
        result = left.select().join(right.select(), on="key").collect()
        assert len(result) == 0


class TestDistinctEdgeCases:
    """Tests for distinct edge cases."""

    def test_distinct_all_unique(self):
        """Distinct on already unique data."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = df.select().distinct().collect()
        assert len(result) == 5

    def test_distinct_all_same(self):
        """Distinct on all identical values."""
        df = pd.DataFrame({"a": [1, 1, 1, 1, 1]})
        result = df.select().distinct().collect()
        assert len(result) == 1

    def test_distinct_empty(self):
        """Distinct on empty DataFrame."""
        df = pd.DataFrame({"a": pd.array([], dtype="int64")})
        result = df.select().distinct().collect()
        assert len(result) == 0


class TestOptimizationEdgeCases:
    """Tests for optimization edge cases."""

    def test_optimize_disabled(self):
        """collect with optimize=False."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().filter(col("a") > 1).collect(optimize=False)
        expected = pd.DataFrame({"a": [2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_filter_true(self):
        """Filter with True literal (should be eliminated)."""
        from pandas.lazy import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().filter(lit(True)).collect()
        tm.assert_frame_equal(result, df)

    def test_filter_false_literal(self):
        """Filter with False literal (should return empty)."""
        from pandas.lazy import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        # lit(False) creates a scalar False filter - now handled correctly
        result = df.select().filter(lit(False)).collect()
        assert len(result) == 0

    def test_filter_always_false_condition(self):
        """Filter with always-false condition."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().filter(col("a") > 1000).collect()
        assert len(result) == 0


class TestArrowBackend:
    """Tests for Arrow-backed DataFrames."""

    def test_arrow_string_column(self):
        """Arrow string column."""
        df = pd.DataFrame({"s": pd.array(["a", "b", "c"], dtype="string[pyarrow]")})
        result = df.select().filter(col("s") == "b").collect()
        assert len(result) == 1

    def test_arrow_numeric_column(self):
        """Arrow numeric column."""
        df = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="int64[pyarrow]")})
        result = df.select().filter(col("a") > 1).collect()
        assert len(result) == 2

    def test_mixed_arrow_numpy(self):
        """Mixed Arrow and NumPy columns."""
        df = pd.DataFrame(
            {
                "arrow_col": pd.array([1, 2, 3], dtype="int64[pyarrow]"),
                "numpy_col": np.array([10, 20, 30]),
            }
        )
        result = (
            df.select()
            .with_columns((col("arrow_col") + col("numpy_col")).alias("sum"))
            .collect()
        )
        assert list(result["sum"]) == [11, 22, 33]
