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
    # CSV-specific options
    sep: str = ",",
    header: bool = True,
    skip_rows: int = 0,
    n_rows: int | None = None,
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

    sep : str, default ","
        Delimiter/separator character. Only used for CSV files.

    header : bool, default True
        Whether the CSV file has a header row. Only used for CSV files.

    skip_rows : int, default 0
        Number of rows to skip at the start of the file. Only used for CSV files.

    n_rows : int, optional
        Maximum number of rows to read. Only used for CSV files.

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
    >>> # Single Parquet file
    >>> ldf = scan("data.parquet")
    >>> result = ldf.filter(col("a") > 10).collect()

    >>> # CSV file with custom separator
    >>> ldf = scan("data.csv", sep=";")
    >>> result = ldf.select("a", "b").collect()

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
    For CSV files, predicates are applied after reading.

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
        import os

    path = os.fspath(path)
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
        return _scan_csv(
            path, sep=sep, header=header, skip_rows=skip_rows, n_rows=n_rows
        )
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


def _scan_csv(
    path: str,
    sep: str = ",",
    header: bool = True,
    skip_rows: int = 0,
    n_rows: int | None = None,
) -> LazyDataFrame:
    """Create a lazy query from CSV file(s)."""
    from pandas.lazy.plan import CSVSource

    source = CSVSource(
        path=path,
        sep=sep,
        header=header,
        skip_rows=skip_rows,
        n_rows=n_rows,
    )
    schema = source.resolve_schema()
    return LazyDataFrame(source, schema)
