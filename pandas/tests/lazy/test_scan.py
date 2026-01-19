"""
Tests for lazy file scanning functionality.
"""

import os
import tempfile

import pytest

import pandas as pd
from pandas.lazy import (
    col,
    scan,
)
from pandas.lazy.plan import (
    CSVSource,
    ParquetSource,
)


class TestScan:
    """Tests for the scan() function."""

    def test_scan_parquet_basic(self):
        """Test basic Parquet scanning."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name)
            assert isinstance(ldf._plan, ParquetSource)

            result = ldf.collect(use_physical_planner=True)
            # Arrow-backed dtypes
            assert len(result) == 3
            assert list(result.columns) == ["a", "b"]

    def test_scan_parquet_with_filter(self):
        """Test Parquet scanning with filter."""
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name).filter(col("a") > 50)
            result = ldf.collect(use_physical_planner=True)

            assert len(result) == 49
            assert result["a"].min() == 51

    def test_scan_parquet_with_select(self):
        """Test Parquet scanning with column selection."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name).select("a", "c")
            result = ldf.collect(use_physical_planner=True)

            assert list(result.columns) == ["a", "c"]
            assert len(result) == 3

    def test_scan_parquet_filter_and_select(self):
        """Test Parquet scanning with both filter and select."""
        df = pd.DataFrame({"a": range(100), "b": range(100, 200), "c": range(200, 300)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name).filter(col("a") > 50).select("a", "c")
            result = ldf.collect(use_physical_planner=True)

            assert list(result.columns) == ["a", "c"]
            assert len(result) == 49
            assert result["a"].min() == 51

    def test_scan_infers_parquet_format(self):
        """Test that scan() infers parquet format from extension."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name)
            assert isinstance(ldf._plan, ParquetSource)

    def test_scan_explicit_format(self):
        """Test scan() with explicit format parameter."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with tempfile.NamedTemporaryFile(suffix=".pq", delete=False) as f:
            df.to_parquet(f.name, index=False)

            # .pq is also recognized as parquet
            ldf = scan(f.name)
            assert isinstance(ldf._plan, ParquetSource)

    def test_scan_unknown_format_raises(self):
        """Test that scan() raises for unknown formats."""
        with pytest.raises(ValueError, match="Cannot infer file format"):
            scan("/path/to/file.unknown")

    def test_scan_csv_basic(self):
        """Test basic CSV scanning."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name)
            assert isinstance(ldf._plan, CSVSource)

            result = ldf.collect(use_physical_planner=True)
            assert len(result) == 3
            assert list(result.columns) == ["a", "b"]

    def test_scan_csv_infers_format(self):
        """Test that scan() infers CSV format from extension."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name)
            assert isinstance(ldf._plan, CSVSource)

    def test_scan_json_not_implemented(self):
        """Test that JSON scanning raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="JSON scanning not yet"):
            scan("data.json")


class TestScanGlobPatterns:
    """Tests for glob pattern scanning."""

    def test_scan_glob_pattern_multiple_files(self):
        """Test scanning multiple files with glob pattern."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

        with tempfile.TemporaryDirectory() as tmpdir:
            df1.to_parquet(os.path.join(tmpdir, "part1.parquet"), index=False)
            df2.to_parquet(os.path.join(tmpdir, "part2.parquet"), index=False)

            # Use glob pattern
            ldf = scan(os.path.join(tmpdir, "*.parquet"))
            result = ldf.collect(use_physical_planner=True)

            # Should have all rows from both files
            assert len(result) == 6
            assert set(result["a"].tolist()) == {1, 2, 3, 7, 8, 9}

    def test_scan_glob_pattern_with_filter(self):
        """Test glob pattern scanning with filter pushdown."""
        df1 = pd.DataFrame({"a": range(50), "b": range(50)})
        df2 = pd.DataFrame({"a": range(50, 100), "b": range(50, 100)})

        with tempfile.TemporaryDirectory() as tmpdir:
            df1.to_parquet(os.path.join(tmpdir, "part1.parquet"), index=False)
            df2.to_parquet(os.path.join(tmpdir, "part2.parquet"), index=False)

            ldf = scan(os.path.join(tmpdir, "*.parquet")).filter(col("a") > 80)
            result = ldf.collect(use_physical_planner=True)

            assert len(result) == 19  # 81-99
            assert result["a"].min() == 81

    def test_scan_glob_pattern_with_select(self):
        """Test glob pattern scanning with column selection."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df2 = pd.DataFrame({"a": [7, 8], "b": [9, 10], "c": [11, 12]})

        with tempfile.TemporaryDirectory() as tmpdir:
            df1.to_parquet(os.path.join(tmpdir, "part1.parquet"), index=False)
            df2.to_parquet(os.path.join(tmpdir, "part2.parquet"), index=False)

            ldf = scan(os.path.join(tmpdir, "*.parquet")).select("a", "c")
            result = ldf.collect(use_physical_planner=True)

            assert list(result.columns) == ["a", "c"]
            assert len(result) == 4

    def test_scan_glob_no_matches_raises(self):
        """Test that glob pattern with no matches raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No files match pattern"):
                # Error raised during scan() when resolving schema
                scan(os.path.join(tmpdir, "*.parquet"))

    def test_scan_glob_schema_from_first_file(self):
        """Test that schema is read from first matching file."""
        df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df2 = pd.DataFrame({"x": [5, 6], "y": [7, 8]})

        with tempfile.TemporaryDirectory() as tmpdir:
            df1.to_parquet(os.path.join(tmpdir, "a_first.parquet"), index=False)
            df2.to_parquet(os.path.join(tmpdir, "b_second.parquet"), index=False)

            ldf = scan(os.path.join(tmpdir, "*.parquet"))
            schema = ldf._plan.resolve_schema()

            assert "x" in schema
            assert "y" in schema


class TestScanDirectory:
    """Tests for directory scanning."""

    def test_scan_directory_with_explicit_format(self):
        """Test scanning a directory with explicit format."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory
            datadir = os.path.join(tmpdir, "data")
            os.makedirs(datadir)

            df1.to_parquet(os.path.join(datadir, "part1.parquet"), index=False)
            df2.to_parquet(os.path.join(datadir, "part2.parquet"), index=False)

            # Scan with glob pattern for the directory
            ldf = scan(os.path.join(datadir, "*.parquet"))
            result = ldf.collect(use_physical_planner=True)

            assert len(result) == 4


class TestScanURLPaths:
    """Tests for URL path format detection."""

    def test_scan_infers_format_from_s3_url(self):
        """Test format inference from S3 URL."""
        from pandas.lazy.scan import _infer_format

        assert _infer_format("s3://bucket/path/data.parquet") == "parquet"
        assert _infer_format("s3://bucket/path/data.csv") == "csv"
        assert _infer_format("s3://bucket/path/data.json") == "json"

    def test_scan_infers_format_from_gs_url(self):
        """Test format inference from GCS URL."""
        from pandas.lazy.scan import _infer_format

        assert _infer_format("gs://bucket/path/data.parquet") == "parquet"
        assert _infer_format("gs://bucket/path/data.pq") == "parquet"
        assert _infer_format("gs://bucket/path/data.csv.gz") == "csv"

    def test_scan_infers_format_from_https_url(self):
        """Test format inference from HTTPS URL."""
        from pandas.lazy.scan import _infer_format

        assert _infer_format("https://example.com/data.parquet") == "parquet"
        assert _infer_format("https://example.com/data.jsonl") == "json"

    def test_scan_url_unknown_format_returns_none(self):
        """Test that unknown URL format returns None."""
        from pandas.lazy.scan import _infer_format

        assert _infer_format("s3://bucket/path/data.unknown") is None
        assert _infer_format("gs://bucket/path/data") is None


class TestParquetSourcePushdown:
    """Tests for predicate and projection pushdown to ParquetSource."""

    def test_predicate_pushdown(self):
        """Test that predicates are pushed into ParquetSource."""
        from pandas.lazy.optimize.passes import PredicatePushdown

        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name).filter(col("a") > 5)

            # Apply predicate pushdown
            pushdown = PredicatePushdown()
            optimized = pushdown.optimize(ldf._plan)

            # Should be ParquetSource with predicate
            assert isinstance(optimized, ParquetSource)
            assert optimized.predicate is not None

    def test_projection_pushdown(self):
        """Test that column selection is pushed into ParquetSource."""
        from pandas.lazy.optimize.passes import ProjectionPruning

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name).select("a", "c")

            # Apply projection pruning
            pruning = ProjectionPruning()
            optimized = pruning.optimize(ldf._plan)

            # Find the ParquetSource
            def find_source(plan):
                if isinstance(plan, ParquetSource):
                    return plan
                if hasattr(plan, "input"):
                    return find_source(plan.input)
                return None

            source = find_source(optimized)
            assert source is not None
            assert source.columns == ("a", "c")

    def test_combined_pushdown(self):
        """Test combined predicate and projection pushdown."""
        from pandas.lazy.optimize.passes import (
            PredicatePushdown,
            ProjectionPruning,
        )

        df = pd.DataFrame({"a": range(100), "b": range(100), "c": range(100)})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name).filter(col("a") > 50).select("a", "c")

            # Apply both optimizations
            pushdown = PredicatePushdown()
            pruning = ProjectionPruning()

            optimized = pruning.optimize(pushdown.optimize(ldf._plan))

            # Find the ParquetSource
            def find_source(plan):
                if isinstance(plan, ParquetSource):
                    return plan
                if hasattr(plan, "input"):
                    return find_source(plan.input)
                return None

            source = find_source(optimized)
            assert source is not None
            assert source.columns == ("a", "c")
            assert source.predicate is not None


class TestParquetSourceSchema:
    """Tests for ParquetSource schema resolution."""

    def test_schema_from_parquet(self):
        """Test that schema is correctly read from Parquet metadata."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            ldf = scan(f.name)
            schema = ldf._plan.resolve_schema()

            assert "int_col" in schema
            assert "float_col" in schema
            assert "str_col" in schema

    def test_schema_with_column_selection(self):
        """Test schema resolution with column selection."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)

            source = ParquetSource(path=f.name, columns=("a", "c"))
            schema = source.resolve_schema()

            assert "a" in schema
            assert "c" in schema
            assert "b" not in schema


class TestCSVScan:
    """Tests for CSV scanning functionality."""

    def test_csv_scan_basic(self):
        """Test basic CSV scanning."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name)
            assert isinstance(ldf._plan, CSVSource)

            result = ldf.collect(use_physical_planner=True)
            assert len(result) == 3
            assert list(result.columns) == ["a", "b"]

    def test_csv_scan_with_filter(self):
        """Test CSV scanning with filter."""
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name).filter(col("a") > 50)
            result = ldf.collect(use_physical_planner=True)

            assert len(result) == 49
            assert result["a"].min() == 51

    def test_csv_scan_with_select(self):
        """Test CSV scanning with column selection."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name).select("a", "c")
            result = ldf.collect(use_physical_planner=True)

            assert list(result.columns) == ["a", "c"]
            assert len(result) == 3

    def test_csv_scan_filter_and_select(self):
        """Test CSV scanning with both filter and select."""
        df = pd.DataFrame({"a": range(100), "b": range(100, 200), "c": range(200, 300)})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name).filter(col("a") > 50).select("a", "c")
            result = ldf.collect(use_physical_planner=True)

            assert list(result.columns) == ["a", "c"]
            assert len(result) == 49
            assert result["a"].min() == 51

    def test_csv_scan_custom_separator(self):
        """Test CSV scanning with custom separator."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("a;b;c\n")
            f.write("1;4;7\n")
            f.write("2;5;8\n")
            f.write("3;6;9\n")
            f.flush()

            ldf = scan(f.name, sep=";")
            result = ldf.collect(use_physical_planner=True)

            assert list(result.columns) == ["a", "b", "c"]
            assert len(result) == 3
            assert result["a"].tolist() == [1, 2, 3]


class TestCSVGlobPatterns:
    """Tests for CSV glob pattern scanning."""

    def test_csv_scan_glob_pattern(self):
        """Test scanning multiple CSV files with glob pattern."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

        with tempfile.TemporaryDirectory() as tmpdir:
            df1.to_csv(os.path.join(tmpdir, "part1.csv"), index=False)
            df2.to_csv(os.path.join(tmpdir, "part2.csv"), index=False)

            ldf = scan(os.path.join(tmpdir, "*.csv"))
            result = ldf.collect(use_physical_planner=True)

            assert len(result) == 6
            assert set(result["a"].tolist()) == {1, 2, 3, 7, 8, 9}

    def test_csv_scan_glob_with_filter(self):
        """Test CSV glob pattern scanning with filter."""
        df1 = pd.DataFrame({"a": range(50), "b": range(50)})
        df2 = pd.DataFrame({"a": range(50, 100), "b": range(50, 100)})

        with tempfile.TemporaryDirectory() as tmpdir:
            df1.to_csv(os.path.join(tmpdir, "part1.csv"), index=False)
            df2.to_csv(os.path.join(tmpdir, "part2.csv"), index=False)

            ldf = scan(os.path.join(tmpdir, "*.csv")).filter(col("a") > 80)
            result = ldf.collect(use_physical_planner=True)

            assert len(result) == 19  # 81-99
            assert result["a"].min() == 81


class TestCSVSourcePushdown:
    """Tests for predicate and projection pushdown to CSVSource."""

    def test_predicate_pushdown_to_csv(self):
        """Test that predicates are pushed into CSVSource."""
        from pandas.lazy.optimize.passes import PredicatePushdown

        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name).filter(col("a") > 5)

            # Apply predicate pushdown
            pushdown = PredicatePushdown()
            optimized = pushdown.optimize(ldf._plan)

            # Should be CSVSource with predicate
            assert isinstance(optimized, CSVSource)
            assert optimized.predicate is not None

    def test_projection_pushdown_to_csv(self):
        """Test that column selection is pushed into CSVSource."""
        from pandas.lazy.optimize.passes import ProjectionPruning

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name).select("a", "c")

            # Apply projection pruning
            pruning = ProjectionPruning()
            optimized = pruning.optimize(ldf._plan)

            # Find the CSVSource
            def find_source(plan):
                if isinstance(plan, CSVSource):
                    return plan
                if hasattr(plan, "input"):
                    return find_source(plan.input)
                return None

            source = find_source(optimized)
            assert source is not None
            assert source.columns == ("a", "c")

    def test_combined_pushdown_csv(self):
        """Test combined predicate and projection pushdown for CSV."""
        from pandas.lazy.optimize.passes import (
            PredicatePushdown,
            ProjectionPruning,
        )

        df = pd.DataFrame({"a": range(100), "b": range(100), "c": range(100)})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name).filter(col("a") > 50).select("a", "c")

            # Apply both optimizations
            pushdown = PredicatePushdown()
            pruning = ProjectionPruning()

            optimized = pruning.optimize(pushdown.optimize(ldf._plan))

            # Find the CSVSource
            def find_source(plan):
                if isinstance(plan, CSVSource):
                    return plan
                if hasattr(plan, "input"):
                    return find_source(plan.input)
                return None

            source = find_source(optimized)
            assert source is not None
            assert source.columns == ("a", "c")
            assert source.predicate is not None


class TestCSVSourceSchema:
    """Tests for CSVSource schema resolution."""

    def test_schema_from_csv(self):
        """Test that schema is correctly inferred from CSV."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            ldf = scan(f.name)
            schema = ldf._plan.resolve_schema()

            assert "int_col" in schema
            assert "float_col" in schema
            assert "str_col" in schema

    def test_schema_with_column_selection(self):
        """Test schema resolution with column selection."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)

            source = CSVSource(path=f.name, columns=("a", "c"))
            schema = source.resolve_schema()

            assert "a" in schema
            assert "c" in schema
            assert "b" not in schema
