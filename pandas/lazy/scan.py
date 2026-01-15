"""
Lazy file scanning for pandas.

This module provides the `scan` function for lazily reading files
without immediately loading data into memory.
"""

from __future__ import annotations

from typing import Literal

from pandas.lazy.frame import LazyDataFrame


def scan(
    path: str,
    *,
    format: Literal["parquet", "csv", "json"] | None = None,
) -> LazyDataFrame:
    """
    Create a lazy query from a file or files.

    This function creates a lazy query plan that will read from the
    specified file(s) only when `collect()` is called. It supports
    predicate and projection pushdown to minimize I/O.

    Parameters
    ----------
    path : str
        Path to file(s). Supports:
        - Local paths: "/path/to/data.parquet"
        - Glob patterns: "/path/to/*.parquet"
        - URLs: "s3://bucket/data.parquet", "gs://bucket/data.parquet"

        For directories or glob patterns, all matching files must have
        the same schema.

    format : {{"parquet", "csv", "json"}}, optional
        File format. If not specified, inferred from file extension.
        Required for directories without extension or ambiguous paths.

    Returns
    -------
    LazyDataFrame
        A lazy query that will read from the file(s) when executed.

    Raises
    ------
    ValueError
        If format cannot be inferred and is not specified.
    FileNotFoundError
        If no files match the path pattern.

    Examples
    --------
    >>> from pandas.lazy import scan, col
    >>> # Single file
    >>> ldf = scan("data.parquet")
    >>> result = ldf.filter(col("a") > 10).collect()

    >>> # Glob pattern
    >>> ldf = scan("data/*.parquet")
    >>> result = ldf.select("a", "b").collect()

    >>> # URL (requires appropriate filesystem support)
    >>> ldf = scan("s3://bucket/data.parquet")
    >>> result = ldf.collect()

    Notes
    -----
    **Predicate Pushdown**: When using the physical planner
    (`collect(use_physical_planner=True)`), filter predicates are
    pushed down to the file reader. For Parquet files, this means
    entire row groups can be skipped based on column statistics.

    **Projection Pushdown**: Only columns that are actually used in
    the query will be read from the file, reducing I/O.

    See Also
    --------
    LazyDataFrame.collect : Execute the query and return a DataFrame.
    LazyDataFrame.filter : Filter rows by predicate.
    LazyDataFrame.select : Select columns.
    """
    # Infer format from path if not specified
    if format is None:
        format = _infer_format(path)

    if format is None:
        raise ValueError(
            f"Cannot infer file format from path: {path!r}. "
            "Please specify format='parquet', 'csv', or 'json'."
        )

    # Create appropriate source node
    if format == "parquet":
        return _scan_parquet(path)
    elif format == "csv":
        raise NotImplementedError("CSV scanning not yet implemented")
    elif format == "json":
        raise NotImplementedError("JSON scanning not yet implemented")
    else:
        raise ValueError(f"Unsupported format: {format!r}")


def _infer_format(path: str) -> Literal["parquet", "csv", "json"] | None:
    """Infer file format from path."""
    # Remove URL scheme if present
    path_lower = path.lower()

    # Handle URLs
    if "://" in path_lower:
        # Extract path part after scheme
        path_lower = path_lower.split("://", 1)[1]

    # Handle glob patterns - use the pattern itself
    # e.g., "*.parquet" -> ".parquet"

    # Check extensions
    if path_lower.endswith((".parquet", ".pq")):
        return "parquet"
    elif path_lower.endswith((".csv", ".csv.gz")):
        return "csv"
    elif path_lower.endswith((".json", ".jsonl")):
        return "json"

    return None


def _scan_parquet(path: str) -> LazyDataFrame:
    """Create a lazy query from Parquet file(s)."""
    from pandas.lazy.plan import ParquetSource

    source = ParquetSource(path=path)
    schema = source.resolve_schema()
    return LazyDataFrame(source, schema)
