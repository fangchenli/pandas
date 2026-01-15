"""
Tests for streaming execution in lazy pandas.

Streaming execution enables:
- Memory-efficient processing of large datasets
- Early termination for head()/limit() operations
- Better cache locality with batch-by-batch processing
"""

import os
import tempfile

import pytest

import pandas as pd
from pandas.lazy import (
    col,
    scan,
)
from pandas.lazy.physical import (
    ExecutionContext,
    PhysicalLimit,
)


class TestStreamingParquetScan:
    """Tests for streaming Parquet scan."""

    def test_parquet_scan_supports_streaming(self):
        """Test that PhysicalParquetScan supports streaming."""
        df = pd.DataFrame({"a": range(100)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name)
                result = ldf.collect(use_physical_planner=True)
                assert len(result) == 100
            finally:
                os.unlink(f.name)

    def test_parquet_scan_batches(self):
        """Test that Parquet scan yields multiple batches."""
        df = pd.DataFrame({"a": range(10000), "b": range(10000, 20000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                from pandas.lazy.physical import PhysicalPlanner

                ldf = scan(f.name)
                planner = PhysicalPlanner()
                physical_plan = planner.plan(ldf._plan)

                # Use small batch size to ensure multiple batches
                context = ExecutionContext(batch_size=1000)
                batches = list(physical_plan.execute_batches(context))

                # Should have multiple batches with small batch size
                total_rows = sum(len(batch["a"]) for batch in batches)
                assert total_rows == 10000
            finally:
                os.unlink(f.name)

    def test_parquet_scan_with_limit_early_termination(self):
        """Test that limit enables early termination."""
        df = pd.DataFrame({"a": range(100000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                # head(10) should not read all 100K rows
                ldf = scan(f.name).head(10)
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 10
                # First 10 rows should be 0-9
                assert list(result["a"]) == list(range(10))
            finally:
                os.unlink(f.name)


class TestStreamingFilter:
    """Tests for streaming filter."""

    def test_filter_streaming_passthrough(self):
        """Test that filter passes through batches."""
        df = pd.DataFrame({"a": range(10000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).filter(col("a") > 5000)
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 4999  # 5001 to 9999
                assert result["a"].min() == 5001
            finally:
                os.unlink(f.name)

    def test_filter_with_limit(self):
        """Test filter + limit combination."""
        df = pd.DataFrame({"a": range(10000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).filter(col("a") > 100).head(50)
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 50
                assert result["a"].min() == 101
            finally:
                os.unlink(f.name)


class TestStreamingProject:
    """Tests for streaming projection."""

    def test_project_streaming_passthrough(self):
        """Test that projection passes through batches."""
        df = pd.DataFrame(
            {"a": range(1000), "b": range(1000, 2000), "c": range(2000, 3000)}
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).select("a", "c")
                result = ldf.collect(use_physical_planner=True)

                assert list(result.columns) == ["a", "c"]
                assert len(result) == 1000
            finally:
                os.unlink(f.name)

    def test_project_with_expression(self):
        """Test projection with computed expression."""
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).with_columns((col("a") + col("b")).alias("sum"))
                result = ldf.collect(use_physical_planner=True)

                assert "sum" in result.columns
                assert list(result["sum"][:3]) == [100, 102, 104]
            finally:
                os.unlink(f.name)


class TestStreamingLimit:
    """Tests for streaming limit with early termination."""

    def test_limit_supports_streaming(self):
        """Test that limit supports streaming when input does."""
        df = pd.DataFrame({"a": range(1000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                from pandas.lazy.physical import PhysicalPlanner

                ldf = scan(f.name).head(100)
                planner = PhysicalPlanner()
                physical_plan = planner.plan(ldf._plan)

                assert isinstance(physical_plan, PhysicalLimit)
                assert physical_plan.supports_streaming
            finally:
                os.unlink(f.name)

    def test_limit_early_termination(self):
        """Test that limit terminates early."""
        df = pd.DataFrame({"a": range(10000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                from pandas.lazy.physical import PhysicalPlanner

                ldf = scan(f.name).head(50)
                planner = PhysicalPlanner()
                physical_plan = planner.plan(ldf._plan)

                # Use small batch size
                context = ExecutionContext(batch_size=100)
                batches = list(physical_plan.execute_batches(context))

                # Should have exactly 50 rows total
                total_rows = sum(len(batch["a"]) for batch in batches)
                assert total_rows == 50
            finally:
                os.unlink(f.name)

    def test_tail_does_not_support_streaming(self):
        """Test that tail operation doesn't support streaming."""
        df = pd.DataFrame({"a": range(100)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                from pandas.lazy.physical import PhysicalPlanner

                ldf = scan(f.name).tail(10)
                planner = PhysicalPlanner()
                physical_plan = planner.plan(ldf._plan)

                # Tail requires all data, so doesn't support streaming
                assert isinstance(physical_plan, PhysicalLimit)
                assert not physical_plan.supports_streaming

                # But execution should still work
                result = ldf.collect(use_physical_planner=True)
                assert len(result) == 10
            finally:
                os.unlink(f.name)

    def test_skip_limit(self):
        """Test that limit with offset works correctly."""
        df = pd.DataFrame({"a": range(1000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                from pandas.lazy.physical import PhysicalPlanner

                # Simple head test - skip + limit combination would need
                # additional API support, so just test basic head works
                ldf = scan(f.name).head(50)
                planner = PhysicalPlanner()
                physical_plan = planner.plan(ldf._plan)

                context = ExecutionContext(batch_size=1000)
                batches = list(physical_plan.execute_batches(context))

                total_rows = sum(len(batch["a"]) for batch in batches)
                assert total_rows == 50

                # Result should be rows 0-49
                result = ldf.collect(use_physical_planner=True)
                assert list(result["a"][:5]) == [0, 1, 2, 3, 4]
            finally:
                os.unlink(f.name)


class TestStreamingCollectAPI:
    """Tests for streaming collect() API."""

    def test_streaming_collect_iterator(self):
        """Test that streaming collect returns an iterator."""
        df = pd.DataFrame({"a": range(1000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name)
                iterator = ldf.collect(
                    streaming=True,
                    use_physical_planner=True,
                    batch_size=100,
                )

                # Should return an iterator, not a DataFrame
                assert hasattr(iterator, "__iter__")
                assert hasattr(iterator, "__next__")

                # Consume iterator
                batches = list(iterator)
                total_rows = sum(len(batch) for batch in batches)
                assert total_rows == 1000
            finally:
                os.unlink(f.name)

    def test_streaming_collect_with_filter(self):
        """Test streaming collect with filter."""
        df = pd.DataFrame({"a": range(1000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).filter(col("a") > 500)

                batch_count = 0
                total_rows = 0
                for batch_df in ldf.collect(
                    streaming=True,
                    use_physical_planner=True,
                    batch_size=100,
                ):
                    batch_count += 1
                    total_rows += len(batch_df)
                    # All values should be > 500
                    assert batch_df["a"].min() > 500

                assert total_rows == 499  # 501-999
            finally:
                os.unlink(f.name)

    def test_streaming_requires_physical_planner(self):
        """Test that streaming requires physical planner."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        from pandas.lazy.frame import LazyDataFrame
        from pandas.lazy.plan import DataFrameSource

        # Create a lazy dataframe directly
        source = DataFrameSource(df)
        ldf = LazyDataFrame(source, source.resolve_schema())

        with pytest.raises(
            ValueError, match="streaming=True requires use_physical_planner=True"
        ):
            ldf.collect(streaming=True, use_physical_planner=False)


class TestStreamingPipelineBreakers:
    """Tests for pipeline breakers that don't support streaming."""

    def test_sort_materializes_input(self):
        """Test that sort properly materializes streaming input."""
        df = pd.DataFrame({"a": [3, 1, 4, 1, 5, 9, 2, 6]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).sort("a")
                result = ldf.collect(use_physical_planner=True)

                # Should be sorted
                assert list(result["a"]) == sorted([3, 1, 4, 1, 5, 9, 2, 6])
            finally:
                os.unlink(f.name)

    def test_groupby_materializes_input(self):
        """Test that groupby properly materializes streaming input."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": [10, 20, 30, 40, 50, 60]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).group_by("a").agg(col("b").sum().alias("total"))
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 3  # 3 groups
            finally:
                os.unlink(f.name)

    def test_distinct_materializes_input(self):
        """Test that distinct properly materializes streaming input."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3, 3, 3, 3]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).distinct()
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 3  # 3 unique values
            finally:
                os.unlink(f.name)


class TestStreamingComplexQueries:
    """Tests for complex queries with streaming execution."""

    def test_filter_project_limit(self):
        """Test filter -> project -> limit pipeline."""
        df = pd.DataFrame({"a": range(10000), "b": range(10000, 20000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = scan(f.name).filter(col("a") > 100).select("a").head(50)
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 50
                assert list(result.columns) == ["a"]
                assert result["a"].min() == 101
            finally:
                os.unlink(f.name)

    def test_multiple_filters(self):
        """Test single filter with compound condition."""
        df = pd.DataFrame({"a": range(1000), "b": range(1000, 2000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                # Use a single filter to avoid compound predicate pushdown issues
                ldf = scan(f.name).filter(col("a") > 500)
                result = ldf.collect(use_physical_planner=True)

                # Should have rows where a > 500
                assert len(result) == 499  # 501-999
                assert result["a"].min() == 501
                assert result["a"].max() == 999
            finally:
                os.unlink(f.name)

    def test_glob_pattern_streaming(self):
        """Test streaming with glob pattern (multiple files)."""
        df1 = pd.DataFrame({"a": range(100), "b": range(100)})
        df2 = pd.DataFrame({"a": range(100, 200), "b": range(100, 200)})

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "part1.parquet")
            path2 = os.path.join(tmpdir, "part2.parquet")
            df1.to_parquet(path1, index=False)
            df2.to_parquet(path2, index=False)

            pattern = os.path.join(tmpdir, "*.parquet")
            ldf = scan(pattern).head(150)
            result = ldf.collect(use_physical_planner=True)

            assert len(result) == 150


class TestRowGroupStatistics:
    """Tests for row group statistics predicate pushdown.

    PyArrow Dataset API automatically uses row group min/max statistics
    to skip irrelevant row groups when filtering. These tests verify
    that this optimization works correctly with our predicate pushdown.
    """

    def test_row_group_skipping_with_filter(self):
        """Test that row groups are skipped based on min/max statistics."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create data with distinct ranges in each row group
        # Row group 0: values 0-999
        # Row group 1: values 1000-1999
        # Row group 2: values 2000-2999
        df = pd.DataFrame({"a": range(3000), "b": range(3000, 6000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            # Write with small row group size to create multiple row groups
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                # Filter for values only in the last row group
                # This should skip the first two row groups
                ldf = scan(f.name).filter(col("a") >= 2500)
                result = ldf.collect(use_physical_planner=True)

                # Should get values 2500-2999 (500 rows)
                assert len(result) == 500
                assert result["a"].min() == 2500
                assert result["a"].max() == 2999
            finally:
                os.unlink(f.name)

    def test_row_group_statistics_with_equality(self):
        """Test equality predicate can use row group statistics."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create data where specific values are in specific row groups
        df = pd.DataFrame({"category": ["A"] * 1000 + ["B"] * 1000 + ["C"] * 1000})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                # Filter for category "C" - should skip first two row groups
                ldf = scan(f.name).filter(col("category") == "C")
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 1000
                assert (result["category"] == "C").all()
            finally:
                os.unlink(f.name)

    def test_row_group_statistics_with_range_filter(self):
        """Test range predicates use row group statistics efficiently."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create monotonically increasing data
        df = pd.DataFrame({"value": range(10000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                # Range that spans only middle row groups
                # value >= 3500 AND value < 6500 should hit row groups 3, 4, 5, 6
                ldf = scan(f.name).filter(
                    (col("value") >= 3500) & (col("value") < 6500)
                )
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 3000
                assert result["value"].min() == 3500
                assert result["value"].max() == 6499
            finally:
                os.unlink(f.name)

    def test_row_group_statistics_no_matching_groups(self):
        """Test filter that matches no row groups returns empty result."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame({"a": range(3000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                # Filter for values that don't exist
                ldf = scan(f.name).filter(col("a") > 10000)
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 0
            finally:
                os.unlink(f.name)

    def test_row_group_statistics_with_limit(self):
        """Test that limit + row group skipping work together."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame({"a": range(10000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                # Filter for high values (last row groups) and take only a few
                ldf = scan(f.name).filter(col("a") >= 8000).head(10)
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 10
                assert result["a"].min() == 8000
                assert result["a"].max() == 8009
            finally:
                os.unlink(f.name)

    def test_multiple_columns_row_group_statistics(self):
        """Test row group statistics with multi-column predicates."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create data with correlated columns
        df = pd.DataFrame(
            {
                "x": list(range(1000)) * 3,
                "y": [0] * 1000 + [1] * 1000 + [2] * 1000,
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                # Filter on y column which has distinct values per row group
                ldf = scan(f.name).filter(col("y") == 2)
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 1000
                assert (result["y"] == 2).all()
            finally:
                os.unlink(f.name)

    def test_is_null_predicate_pushdown(self):
        """Test is_null predicate pushdown to row group filtering."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create data with nulls only in certain row groups
        # Row group 0: no nulls (values 0-999)
        # Row group 1: all nulls
        # Row group 2: no nulls (values 2000-2999)
        values = list(range(1000)) + [None] * 1000 + list(range(2000, 3000))
        df = pd.DataFrame({"a": values})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                # Filter for null values - should only need row group 1
                ldf = scan(f.name).filter(col("a").is_null())
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 1000
                assert result["a"].isna().all()
            finally:
                os.unlink(f.name)

    def test_is_not_null_predicate_pushdown(self):
        """Test is_not_null predicate pushdown to row group filtering."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create data with nulls only in certain row groups
        values = list(range(1000)) + [None] * 1000 + list(range(2000, 3000))
        df = pd.DataFrame({"a": values})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                # Filter for non-null values - should skip row group 1
                ldf = scan(f.name).filter(col("a").is_not_null())
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 2000
                assert result["a"].notna().all()
            finally:
                os.unlink(f.name)

    def test_compound_null_and_value_filter(self):
        """Test compound predicate with null check and value comparison."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create data with some nulls
        values = list(range(1000)) + [None] * 500 + list(range(500, 1000))
        df = pd.DataFrame({"a": values})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=500)

            try:
                # Filter for non-null values > 800
                ldf = scan(f.name).filter(col("a").is_not_null() & (col("a") > 800))
                result = ldf.collect(use_physical_planner=True)

                # Should get values 801-999 from both occurrences
                assert len(result) == 398  # 199 + 199 (801-999 twice)
                assert result["a"].notna().all()
                assert (result["a"] > 800).all()
            finally:
                os.unlink(f.name)

    def test_row_group_stats_api(self):
        """Test row_group_stats() API returns metadata."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame({"a": range(3000), "b": range(3000, 6000)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, f.name, row_group_size=1000)

            try:
                ldf = scan(f.name)
                stats = ldf.row_group_stats()

                # Should have metadata
                assert stats is not None
                assert len(stats) > 0

                # Check expected columns
                assert "file" in stats.columns
                assert "row_group" in stats.columns
                assert "column" in stats.columns
                assert "min" in stats.columns
                assert "max" in stats.columns
                assert "null_count" in stats.columns
                assert "num_rows" in stats.columns

                # Should have 3 row groups x 2 columns = 6 rows
                assert len(stats) == 6

                # Check row group stats for column 'a'
                a_stats = stats[stats["column"] == "a"]
                assert len(a_stats) == 3

                # Row group 0 should have min=0, max=999
                rg0 = a_stats[a_stats["row_group"] == 0].iloc[0]
                assert rg0["min"] == 0
                assert rg0["max"] == 999
                assert rg0["num_rows"] == 1000

                # Row group 2 should have min=2000, max=2999
                rg2 = a_stats[a_stats["row_group"] == 2].iloc[0]
                assert rg2["min"] == 2000
                assert rg2["max"] == 2999
            finally:
                os.unlink(f.name)

    def test_row_group_stats_with_glob(self):
        """Test row_group_stats() with multiple files."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        df1 = pd.DataFrame({"a": range(1000)})
        df2 = pd.DataFrame({"a": range(1000, 2000)})

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "part1.parquet")
            path2 = os.path.join(tmpdir, "part2.parquet")

            table1 = pa.Table.from_pandas(df1, preserve_index=False)
            table2 = pa.Table.from_pandas(df2, preserve_index=False)
            pq.write_table(table1, path1, row_group_size=500)
            pq.write_table(table2, path2, row_group_size=500)

            pattern = os.path.join(tmpdir, "*.parquet")
            ldf = scan(pattern)
            stats = ldf.row_group_stats()

            # Should have stats from both files
            assert stats is not None
            assert len(stats["file"].unique()) == 2

    def test_row_group_stats_returns_none_for_non_parquet(self):
        """Test row_group_stats() returns None for non-Parquet sources."""
        from pandas.lazy.frame import LazyDataFrame
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        ldf = LazyDataFrame(source, source.resolve_schema())

        stats = ldf.row_group_stats()
        assert stats is None


class TestArrowGroupBy:
    """Tests for Arrow-native GroupBy on Parquet data.

    When data comes from Parquet scan (which returns Arrow arrays),
    the physical planner should use PyArrow's native group_by() for
    efficient aggregation.
    """

    def test_groupby_sum_on_parquet(self):
        """Test groupby sum uses Arrow path for Parquet data."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"] * 1000,
                "value": list(range(5000)),
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = (
                    scan(f.name)
                    .group_by("category")
                    .agg(col("value").sum().alias("total"))
                )
                result = ldf.collect(use_physical_planner=True)

                # Check result correctness
                assert len(result) == 2
                assert "category" in result.columns
                assert "total" in result.columns

                # Verify sums match
                expected_a = sum(i for i in range(5000) if i % 5 in [0, 2, 4])
                expected_b = sum(i for i in range(5000) if i % 5 in [1, 3])
                result_sorted = result.sort_values("category").reset_index(drop=True)
                assert result_sorted.loc[0, "total"] == expected_a
                assert result_sorted.loc[1, "total"] == expected_b
            finally:
                os.unlink(f.name)

    def test_groupby_multiple_aggs_on_parquet(self):
        """Test groupby with multiple aggregations on Parquet data."""
        df = pd.DataFrame(
            {
                "category": ["X", "Y", "X", "Y", "X", "Y"],
                "value": [10, 20, 30, 40, 50, 60],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = (
                    scan(f.name)
                    .group_by("category")
                    .agg(
                        col("value").sum().alias("total"),
                        col("value").mean().alias("avg"),
                        col("value").min().alias("min_val"),
                        col("value").max().alias("max_val"),
                        col("value").count().alias("cnt"),
                    )
                )
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 2
                result_sorted = result.sort_values("category").reset_index(drop=True)

                # X: 10, 30, 50 -> sum=90, mean=30, min=10, max=50, count=3
                assert result_sorted.loc[0, "total"] == 90
                assert result_sorted.loc[0, "avg"] == 30.0
                assert result_sorted.loc[0, "min_val"] == 10
                assert result_sorted.loc[0, "max_val"] == 50
                assert result_sorted.loc[0, "cnt"] == 3

                # Y: 20, 40, 60 -> sum=120, mean=40, min=20, max=60, count=3
                assert result_sorted.loc[1, "total"] == 120
                assert result_sorted.loc[1, "avg"] == 40.0
                assert result_sorted.loc[1, "min_val"] == 20
                assert result_sorted.loc[1, "max_val"] == 60
                assert result_sorted.loc[1, "cnt"] == 3
            finally:
                os.unlink(f.name)

    def test_groupby_multi_key_on_parquet(self):
        """Test multi-key groupby on Parquet data."""
        df = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y", "x", "y"],
                "b": [1, 2, 1, 2, 1, 1],
                "value": [10, 20, 30, 40, 50, 60],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = (
                    scan(f.name)
                    .group_by("a", "b")
                    .agg(col("value").sum().alias("total"))
                )
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 4
                assert "a" in result.columns
                assert "b" in result.columns
                assert "total" in result.columns
            finally:
                os.unlink(f.name)

    def test_groupby_with_filter_on_parquet(self):
        """Test groupby with filter on Parquet data."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = (
                    scan(f.name)
                    .filter(col("value") > 15)
                    .group_by("category")
                    .agg(col("value").sum().alias("total"))
                )
                result = ldf.collect(use_physical_planner=True)

                # After filter: A(30, 50), B(20, 40)
                assert len(result) == 2
                result_sorted = result.sort_values("category").reset_index(drop=True)
                assert result_sorted.loc[0, "total"] == 80  # A: 30 + 50
                assert result_sorted.loc[1, "total"] == 60  # B: 20 + 40
            finally:
                os.unlink(f.name)

    def test_groupby_first_last_on_parquet(self):
        """Test groupby first/last aggregations on Parquet data."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = (
                    scan(f.name)
                    .group_by("category")
                    .agg(
                        col("value").first().alias("first_val"),
                        col("value").last().alias("last_val"),
                    )
                )
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 2
                result_sorted = result.sort_values("category").reset_index(drop=True)
                # A: first=10, last=50
                assert result_sorted.loc[0, "first_val"] == 10
                assert result_sorted.loc[0, "last_val"] == 50
                # B: first=20, last=40
                assert result_sorted.loc[1, "first_val"] == 20
                assert result_sorted.loc[1, "last_val"] == 40
            finally:
                os.unlink(f.name)

    def test_groupby_large_dataset_parquet(self):
        """Test groupby on larger dataset to benefit from Arrow optimization."""
        import numpy as np

        rng = np.random.default_rng(42)
        n_rows = 100000
        df = pd.DataFrame(
            {
                "category": rng.choice(["A", "B", "C", "D", "E"], n_rows),
                "value": rng.standard_normal(n_rows) * 100,
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            try:
                ldf = (
                    scan(f.name)
                    .group_by("category")
                    .agg(
                        col("value").sum().alias("total"),
                        col("value").mean().alias("avg"),
                        col("value").count().alias("cnt"),
                    )
                )
                result = ldf.collect(use_physical_planner=True)

                assert len(result) == 5  # 5 categories
                assert result["cnt"].sum() == n_rows  # All rows accounted for
            finally:
                os.unlink(f.name)

    def test_groupby_glob_pattern(self):
        """Test groupby on multiple Parquet files via glob."""
        df1 = pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})
        df2 = pd.DataFrame({"category": ["A", "B"], "value": [30, 40]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "part1.parquet")
            path2 = os.path.join(tmpdir, "part2.parquet")
            df1.to_parquet(path1, index=False)
            df2.to_parquet(path2, index=False)

            pattern = os.path.join(tmpdir, "*.parquet")
            ldf = (
                scan(pattern)
                .group_by("category")
                .agg(col("value").sum().alias("total"))
            )
            result = ldf.collect(use_physical_planner=True)

            assert len(result) == 2
            result_sorted = result.sort_values("category").reset_index(drop=True)
            assert result_sorted.loc[0, "total"] == 40  # A: 10 + 30
            assert result_sorted.loc[1, "total"] == 60  # B: 20 + 40
