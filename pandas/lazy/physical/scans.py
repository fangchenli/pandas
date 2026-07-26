"""Physical plan — scans operators (split from physical.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pandas.lazy.backends.convert import extract_array
from pandas.lazy.backends.types import (
    INDEX_COL_NAME,
    ArrayDict,
    index_col_name,
)
from pandas.lazy.ir import (
    Call,
    FieldRef,
    Literal as IRLiteral,
)
from pandas.lazy.physical.base import (
    ExecutionContext,
    PhysicalPlan,
    get_ordered_columns,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas import DataFrame
    from pandas.lazy.expr import Expr
    from pandas.lazy.types import Schema

# =============================================================================
# Scan Nodes (Data Sources)
# =============================================================================


@dataclass
class PhysicalScan(PhysicalPlan):
    """
    Physical scan of a DataFrame source.

    This is a leaf node that reads from an in-memory DataFrame.
    Extracts arrays from DataFrame and stores index as special columns.
    """

    df: DataFrame
    schema: Schema

    def execute(self, context: ExecutionContext) -> ArrayDict:
        import pandas as pd

        arrays: ArrayDict = {}

        # Extract data columns
        for col in self.df.columns:
            arrays[col] = extract_array(self.df[col])

        # Always extract source index as special column(s)
        # This is needed for:
        # 1. reset_index() to work (adds index as column)
        # 2. Filters/joins to track row correspondence
        # Whether the index is reconstructed at the end depends on preserve_index
        # OR whether there's an explicit set_index() in the plan (which marks
        # the context as having a user-specified index).
        if isinstance(self.df.index, pd.MultiIndex):
            context.index_is_multi = True
            context.index_names = list(self.df.index.names)
            for i in range(self.df.index.nlevels):
                col_name = index_col_name(i)
                arrays[col_name] = self.df.index.get_level_values(i).to_numpy()
        else:
            context.index_is_multi = False
            context.index_names = [self.df.index.name]
            arrays[index_col_name()] = self.df.index.to_numpy()

        return arrays

    def children(self) -> list[PhysicalPlan]:
        return []

    @property
    def output_schema(self) -> Schema:
        return self.schema


@dataclass
class PhysicalParquetScan(PhysicalPlan):
    """
    Physical scan of Parquet file(s).

    Supports predicate and projection pushdown to minimize I/O.
    Uses PyArrow for efficient Parquet reading.

    Streaming Execution
    -------------------
    This operator supports streaming execution via `execute_batches()`.
    When a `limit` is set, it enables early termination - only reading
    enough row groups to satisfy the limit. This can provide significant
    speedups for `head()` operations on large files.

    Parameters
    ----------
    path : str
        Path to Parquet file(s). Supports local paths, globs, and URLs.
    schema : Schema
        Output schema (after column pruning if applicable).
    columns : tuple[str, ...] | None
        Columns to read. None means all columns.
    predicate : Expr | None
        Filter predicate to push down to Parquet reader.
    limit : int | None
        Maximum number of rows to return. Enables early termination
        during streaming execution.
    """

    path: str
    schema: Schema
    columns: tuple[str, ...] | None = None
    predicate: Expr | None = None
    limit: int | None = None

    @property
    def supports_streaming(self) -> bool:
        return True

    def _resolve_paths(self) -> str | list[str]:
        """Resolve path, expanding glob patterns if needed."""
        path = self.path
        if "*" in path and "://" not in path:
            import glob as glob_module

            files = sorted(glob_module.glob(path))
            if not files:
                raise FileNotFoundError(f"No files found matching pattern: {path}")
            return files
        return path

    def _execute_small_limit_batches(
        self, paths: str | list[str], context: ExecutionContext
    ) -> Iterator[ArrayDict]:
        """Direct ParquetFile streaming for small unfiltered limits.

        Reads files in order with iter_batches, stopping as soon as the
        limit is satisfied - no Dataset scanner, no fragment readahead.
        """

        file_list = [paths] if isinstance(paths, str) else list(paths)
        columns = list(self.columns) if self.columns else None
        remaining = self.limit
        row_offset = 0

        for path in file_list:
            if remaining <= 0:
                return
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=max(remaining, 1024), columns=columns
            ):
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                arrays: ArrayDict = {
                    name: batch.column(name) for name in batch.schema.names
                }
                arrays[INDEX_COL_NAME] = pa.array(
                    np.arange(row_offset, row_offset + batch.num_rows, dtype=np.int64)
                )
                context.index_is_multi = False
                context.index_names = [None]
                yield arrays
                row_offset += batch.num_rows
                remaining -= batch.num_rows
                if remaining <= 0:
                    return

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Stream batches from Parquet file(s).

        Uses PyArrow's Dataset API for efficient batch iteration with
        predicate pushdown. Supports early termination when a limit
        is set.

        Yields
        ------
        ArrayDict
            Batches of data from the Parquet file(s).
        """

        # Resolve file paths
        paths = self._resolve_paths()

        # Create dataset (handles single file, list of files, or directory)
        dataset = ds.dataset(paths, format="parquet")

        # Build scanner with pushdown optimizations
        filter_expr = None
        if self.predicate is not None:
            filter_expr = self._build_arrow_filters(self.predicate)

        # Small unfiltered limits bypass the Dataset scanner entirely:
        # the scanner pays ~50 ms of fixed startup on a multi-file
        # dataset (fragment readahead opens and pre-decodes across all
        # files) regardless of batch size, while ParquetFile.iter_batches
        # reads the first 1,000 rows of the first file in ~6 ms. Streams
        # files in order, stopping at the limit.
        if (
            self.predicate is None
            and self.limit is not None
            and self.limit <= context.batch_size
        ):
            yield from self._execute_small_limit_batches(paths, context)
            return

        scanner = dataset.scanner(
            columns=list(self.columns) if self.columns else None,
            filter=filter_expr,
            batch_size=context.batch_size,
        )

        # Pushdown is BEST-EFFORT: _build_arrow_filters returns None for
        # predicates Arrow dataset expressions can't represent (e.g. regex
        # contains). The Filter node was already removed from the plan by
        # the optimizer, so when conversion fails the predicate MUST be
        # applied here per batch — silently dropping it returned every row
        # (TPC-H q16/q13 through scans gave wrong results).
        fallback_predicate = (
            self.predicate._ir
            if self.predicate is not None and filter_expr is None
            else None
        )

        rows_yielded = 0
        row_offset = 0  # Track row offset for index generation

        for batch in scanner.to_batches():
            batch_len = batch.num_rows
            if batch_len == 0:
                continue

            if fallback_predicate is not None:
                from pandas.lazy.backends.array_eval import ArrayEvaluator

                cols0 = {n: batch.column(n) for n in batch.schema.names}
                mask = ArrayEvaluator(cols0, preferred_backend="auto").evaluate(
                    fallback_predicate
                )
                if isinstance(mask, (pa.Array, pa.ChunkedArray)):
                    mask = mask.to_numpy(zero_copy_only=False)
                mask = np.asarray(mask, dtype=bool)
                batch = batch.filter(pa.array(mask))
                batch_len = batch.num_rows
                if batch_len == 0:
                    continue

            # Check limit for early termination
            if self.limit is not None:
                remaining = self.limit - rows_yielded
                if remaining <= 0:
                    return
                if batch_len > remaining:
                    # Slice the batch to exact limit
                    batch = batch.slice(0, remaining)
                    batch_len = remaining

            # Convert RecordBatch to ArrayDict
            arrays: ArrayDict = {}
            for col_name in batch.schema.names:
                arrays[col_name] = batch.column(col_name)

            # Generate index column for this batch. np.arange, not a
            # Python range: pa.array(range(...)) iterates one Python int
            # per row - it dominated full-scan time (104 ms for a
            # 2.9M-row file whose raw threaded read costs 9 ms).

            arrays[INDEX_COL_NAME] = pa.array(
                np.arange(row_offset, row_offset + batch_len, dtype=np.int64)
            )
            context.index_is_multi = False
            context.index_names = [None]

            yield arrays

            rows_yielded += batch_len
            row_offset += batch_len

            # Early termination check
            if self.limit is not None and rows_yielded >= self.limit:
                return

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute and return all data at once.

        For non-streaming execution or when downstream operators need
        all data. Materializes batches from execute_batches() into
        a single ArrayDict.
        """

        # Collect all batches
        batches = list(self.execute_batches(context))

        if not batches:
            # Return empty ArrayDict with correct schema
            arrays: ArrayDict = {}
            for col_name in self.schema.fields:
                arrays[col_name] = pa.array([])
            arrays[INDEX_COL_NAME] = pa.array([], type=pa.int64())
            return arrays

        if len(batches) == 1:
            return batches[0]

        # Concatenate all batches preserving column order
        result: ArrayDict = {}
        for col_name in get_ordered_columns(batches):
            chunks = [batch[col_name] for batch in batches if col_name in batch]
            if chunks:
                # Combine into single contiguous array
                chunked = pa.chunked_array(chunks)
                result[col_name] = chunked.combine_chunks()

        return result

    def _build_arrow_filters(self, predicate: Expr) -> list | None:
        """
        Convert lazy predicate to PyArrow filter expression.

        PyArrow supports filter pushdown in the format:
        - Simple: [("col", "op", value)]
        - Compound: [[("col1", ">", 5)], [("col2", "<", 10)]]  # OR of ANDs

        Returns None if predicate cannot be pushed down.
        """

        ir = predicate._ir

        # Try to convert to PyArrow compute expression for row group filtering
        arrow_expr = self._ir_to_arrow_expr(ir)
        if arrow_expr is not None:
            # Return as PyArrow compute expression
            return arrow_expr

        return None

    def _ir_to_arrow_expr(self, ir):
        """Convert IR node to PyArrow compute expression."""

        if isinstance(ir, FieldRef):
            return pc.field(ir.name)

        if isinstance(ir, IRLiteral):
            return ir.value

        if isinstance(ir, Call):
            # Binary comparison operators
            if ir.function == "greater" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.greater(left, right)

            elif ir.function == "greater_equal" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.greater_equal(left, right)

            elif ir.function == "less" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.less(left, right)

            elif ir.function == "less_equal" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.less_equal(left, right)

            elif ir.function == "equal" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.equal(left, right)

            elif ir.function == "not_equal" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.not_equal(left, right)

            # Logical operators
            # Use and_kleene/or_kleene which handle expression types correctly
            elif ir.function == "and_" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.and_kleene(left, right)

            elif ir.function == "or_" and len(ir.args) == 2:
                left = self._ir_to_arrow_expr(ir.args[0])
                right = self._ir_to_arrow_expr(ir.args[1])
                if left is not None and right is not None:
                    return pc.or_kleene(left, right)

            elif ir.function == "invert" and len(ir.args) == 1:
                arg = self._ir_to_arrow_expr(ir.args[0])
                if arg is not None:
                    return pc.invert(arg)

            # Null-checking operators - can use row group null_count statistics
            elif ir.function == "is_null" and len(ir.args) == 1:
                arg = self._ir_to_arrow_expr(ir.args[0])
                if arg is not None:
                    return pc.is_null(arg)

            elif ir.function == "is_not_null" and len(ir.args) == 1:
                arg = self._ir_to_arrow_expr(ir.args[0])
                if arg is not None:
                    return pc.is_valid(arg)

            # isin operator - enables row group filtering on dict/categorical
            elif ir.function == "isin" and len(ir.args) >= 1:
                col = self._ir_to_arrow_expr(ir.args[0])
                # Values can be in args[1] or kwargs["values"]
                values = ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("values")
                if col is not None and values is not None:
                    # Convert values to PyArrow array for is_in
                    try:
                        if isinstance(values, (list, tuple)):
                            value_set = pa.array(values)
                        else:
                            value_set = pa.array(list(values))
                        return pc.is_in(col, value_set=value_set)
                    except (TypeError, pa.ArrowInvalid):
                        # Cannot convert values to Arrow, fall back
                        pass

            # String operations - enable row group filtering on string columns
            elif ir.function == "str_startswith" and len(ir.args) >= 1:
                col = self._ir_to_arrow_expr(ir.args[0])
                prefix_arg = ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("prefix")
                # Extract value if it's a Literal
                if isinstance(prefix_arg, IRLiteral):
                    prefix = prefix_arg.value
                else:
                    prefix = prefix_arg
                if col is not None and isinstance(prefix, str):
                    return pc.starts_with(col, prefix)

            elif ir.function == "str_endswith" and len(ir.args) >= 1:
                col = self._ir_to_arrow_expr(ir.args[0])
                suffix_arg = ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("suffix")
                # Extract value if it's a Literal
                if isinstance(suffix_arg, IRLiteral):
                    suffix = suffix_arg.value
                else:
                    suffix = suffix_arg
                if col is not None and isinstance(suffix, str):
                    return pc.ends_with(col, suffix)

            elif ir.function == "str_contains" and len(ir.args) >= 1:
                col = self._ir_to_arrow_expr(ir.args[0])
                pattern_arg = (
                    ir.args[1] if len(ir.args) > 1 else ir.kwargs.get("pattern")
                )
                # Extract value if it's a Literal
                if isinstance(pattern_arg, IRLiteral):
                    pattern = pattern_arg.value
                else:
                    pattern = pattern_arg
                if col is not None and isinstance(pattern, str):
                    # Expr.str.contains defaults to REGEX semantics (pandas
                    # str.contains), but pc.match_substring is LITERAL — a
                    # regex pattern pushed as a literal silently filters
                    # everything out (TPC-H q16's 'Customer.*Complaints'
                    # matched 0 rows through a scan). Push down only when
                    # the match is literal-safe; otherwise leave the filter
                    # in-engine where the regex kernel evaluates it.
                    regex = ir.kwargs.get("regex", True)
                    if not regex or not (set(".^$*+?{}[]\\|()") & set(pattern)):
                        return pc.match_substring(col, pattern)
                    return None

        # Cannot convert this expression
        return None

    def children(self) -> list[PhysicalPlan]:
        return []

    @property
    def output_schema(self) -> Schema:
        return self.schema


@dataclass
class PhysicalCSVScan(PhysicalPlan):
    """
    Physical scan of CSV file(s).

    Supports projection pushdown to minimize I/O. Predicates are applied
    after reading since CSV doesn't support native predicate pushdown.

    Uses PyArrow CSV reader for efficient batch processing.

    Parameters
    ----------
    path : str
        Path to CSV file(s). Supports local paths and globs.
    schema : Schema
        Output schema (after column pruning if applicable).
    columns : tuple[str, ...] | None
        Columns to read. None means all columns.
    predicate : Expr | None
        Filter predicate to apply after reading.
    limit : int | None
        Maximum number of rows to return. Enables early termination.
    sep : str
        Delimiter/separator character.
    header : bool
        Whether the file has a header row.
    skip_rows : int
        Number of rows to skip at the start.
    n_rows : int | None
        Maximum number of rows to read from source.
    """

    path: str
    schema: Schema
    columns: tuple[str, ...] | None = None
    predicate: Expr | None = None
    limit: int | None = None
    sep: str = ","
    header: bool = True
    skip_rows: int = 0
    n_rows: int | None = None

    @property
    def supports_streaming(self) -> bool:
        return True

    def _resolve_paths(self) -> str | list[str]:
        """Resolve path, expanding glob patterns if needed."""
        path = self.path
        if "*" in path and "://" not in path:
            import glob as glob_module

            files = sorted(glob_module.glob(path))
            if not files:
                raise FileNotFoundError(f"No files found matching pattern: {path}")
            return files
        return path

    def execute_batches(self, context: ExecutionContext) -> Iterator[ArrayDict]:
        """
        Stream batches from CSV file(s).

        Uses PyArrow's CSV streaming reader for efficient batch iteration.
        Supports early termination when a limit is set.

        Yields
        ------
        ArrayDict
            Batches of data from the CSV file(s).
        """
        from pyarrow import csv

        # Resolve file paths
        paths = self._resolve_paths()
        if isinstance(paths, str):
            paths = [paths]

        # Configure CSV read options
        read_options = csv.ReadOptions(
            skip_rows=self.skip_rows,
            autogenerate_column_names=not self.header,
            block_size=context.batch_size * 1024,  # Approximate bytes per batch
        )
        parse_options = csv.ParseOptions(delimiter=self.sep)

        rows_yielded = 0
        row_offset = 0

        for file_path in paths:
            # Create streaming reader
            reader = csv.open_csv(
                file_path,
                read_options=read_options,
                parse_options=parse_options,
            )

            for batch in reader:
                batch_len = batch.num_rows
                if batch_len == 0:
                    continue

                # Apply n_rows limit from source
                if self.n_rows is not None:
                    remaining = self.n_rows - rows_yielded
                    if remaining <= 0:
                        return
                    if batch_len > remaining:
                        batch = batch.slice(0, remaining)
                        batch_len = remaining

                # Check limit for early termination
                if self.limit is not None:
                    remaining = self.limit - rows_yielded
                    if remaining <= 0:
                        return
                    if batch_len > remaining:
                        batch = batch.slice(0, remaining)
                        batch_len = remaining

                # Convert RecordBatch to ArrayDict
                arrays: ArrayDict = {}

                # If columns specified, only include those
                columns_to_read = (
                    list(self.columns) if self.columns else batch.schema.names
                )
                for col_name in columns_to_read:
                    if col_name in batch.schema.names:
                        arrays[col_name] = batch.column(col_name)

                # Generate index column for this batch
                arrays[INDEX_COL_NAME] = pa.array(
                    range(row_offset, row_offset + batch_len)
                )
                context.index_is_multi = False
                context.index_names = [None]

                yield arrays

                rows_yielded += batch_len
                row_offset += batch_len

                # Early termination check
                if self.limit is not None and rows_yielded >= self.limit:
                    return
                if self.n_rows is not None and rows_yielded >= self.n_rows:
                    return

    def execute(self, context: ExecutionContext) -> ArrayDict:
        """
        Execute and return all data at once.

        For non-streaming execution or when downstream operators need
        all data. Materializes batches from execute_batches() into
        a single ArrayDict.
        """

        # Collect all batches
        batches = list(self.execute_batches(context))

        if not batches:
            # Return empty ArrayDict with correct schema
            arrays: ArrayDict = {}
            for col_name in self.schema.fields:
                arrays[col_name] = pa.array([])
            arrays[INDEX_COL_NAME] = pa.array([], type=pa.int64())
            return arrays

        if len(batches) == 1:
            return batches[0]

        # Concatenate all batches preserving column order
        result: ArrayDict = {}
        for col_name in get_ordered_columns(batches):
            chunks = [batch[col_name] for batch in batches if col_name in batch]
            if chunks:
                # Combine into single contiguous array
                chunked = pa.chunked_array(chunks)
                result[col_name] = chunked.combine_chunks()

        return result

    def children(self) -> list[PhysicalPlan]:
        return []

    @property
    def output_schema(self) -> Schema:
        return self.schema
