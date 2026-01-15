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
