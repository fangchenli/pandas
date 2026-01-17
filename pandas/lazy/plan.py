"""
Logical plan nodes for lazy pandas.

These nodes represent the structure of a query before optimization
and physical planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.lazy.expr import Expr
    from pandas.lazy.types import Schema


class LogicalPlan:
    """
    Base class for logical plan nodes.

    Note: This is not a dataclass to avoid inheritance issues with
    the schema cache field.
    """

    # Cache for resolved schema
    __slots__ = ("_cached_schema",)

    def __init__(self) -> None:
        self._cached_schema: Schema | None = None

    def resolve_schema(self) -> Schema:
        """
        Resolve output schema for this plan node.

        Results are cached to avoid repeated computation during
        physical planning.
        """
        if self._cached_schema is not None:
            return self._cached_schema
        self._cached_schema = self._resolve_schema_impl()
        return self._cached_schema

    def _resolve_schema_impl(self) -> Schema:
        """Actual schema resolution - override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _resolve_schema_impl()"
        )

    def children(self) -> list[LogicalPlan]:
        """Return child plan nodes."""
        raise NotImplementedError(f"{type(self).__name__} must implement children()")

    def estimate_row_count(self) -> int | None:
        """
        Estimate the number of rows this plan node will produce.

        Used by the physical planner for optimization decisions such as:
        - Join build/probe side selection (build on smaller side)
        - Choosing between algorithms (hash vs sort-merge join)
        - Parallel execution decisions

        Returns
        -------
        int or None
            Estimated row count, or None if unknown.

        Notes
        -----
        These are rough estimates for optimization purposes.
        Actual row counts may differ significantly, especially
        after filters with unknown selectivity.
        """
        return self._estimate_row_count_impl()

    def _estimate_row_count_impl(self) -> int | None:
        """Override in subclasses to provide row count estimates."""
        # Default: unknown
        return None


@dataclass
class DataFrameSource(LogicalPlan):
    """Source node wrapping an eager DataFrame."""

    df: DataFrame

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        from pandas.lazy.types import Schema

        return Schema.from_dataframe(self.df)

    def children(self) -> list[LogicalPlan]:
        return []

    def _estimate_row_count_impl(self) -> int | None:
        return len(self.df)

    def __repr__(self) -> str:
        cols = list(self.df.columns)
        return f"DataFrameSource(columns={cols}, rows={len(self.df)})"


@dataclass
class ParquetSource(LogicalPlan):
    """
    Source node for reading Parquet files lazily.

    Supports predicate and projection pushdown to minimize I/O.
    The actual reading happens during physical execution.

    Parameters
    ----------
    path : str
        Path to Parquet file or directory. Supports:
        - Local paths: "/path/to/data.parquet"
        - Glob patterns: "/path/to/*.parquet"
        - URLs: "s3://bucket/data.parquet", "gs://bucket/data.parquet"
    columns : tuple[str, ...] | None
        Columns to read. None means all columns.
        Set by ProjectionPruning optimization pass.
    predicate : Expr | None
        Filter predicate to push down to Parquet reader.
        Set by PredicatePushdown optimization pass.
    """

    path: str
    columns: tuple[str, ...] | None = None
    predicate: Expr | None = None

    # Cached schema from Parquet metadata
    _parquet_schema: Schema | None = None

    def __post_init__(self) -> None:
        self._cached_schema = None
        # Don't set _parquet_schema here - it will be resolved lazily

    def _resolve_schema_impl(self) -> Schema:
        from pandas.lazy.types import Schema

        if self._parquet_schema is not None:
            schema = self._parquet_schema
        else:
            # Read schema from Parquet metadata (no data read)
            schema = self._read_parquet_schema()
            # Cache it for future use
            object.__setattr__(self, "_parquet_schema", schema)

        # If columns are specified, filter schema
        if self.columns is not None:
            return Schema({c: schema[c] for c in self.columns if c in schema})
        return schema

    def _read_parquet_schema(self) -> Schema:
        """Read schema from Parquet file metadata without reading data."""
        import pyarrow.parquet as pq

        from pandas.lazy.types import (
            LazyDtype,
            Schema,
        )

        # Handle glob patterns
        path = self.path
        if "*" in path:
            import glob

            files = glob.glob(path)
            if not files:
                raise FileNotFoundError(f"No files match pattern: {path}")
            path = files[0]  # Use first file for schema

        # Read schema from metadata only
        parquet_file = pq.ParquetFile(path)
        arrow_schema = parquet_file.schema_arrow

        # Convert Arrow schema to lazy Schema
        columns = {}
        for field in arrow_schema:
            columns[field.name] = LazyDtype.from_arrow_type(field.type)

        return Schema(columns)

    def children(self) -> list[LogicalPlan]:
        return []

    def _estimate_row_count_impl(self) -> int | None:
        """Estimate row count from Parquet metadata."""
        try:
            import pyarrow.parquet as pq

            path = self.path
            if "*" in path:
                import glob

                files = glob.glob(path)
                if not files:
                    return None
                # Sum row counts from all files
                total = 0
                for f in files:
                    pf = pq.ParquetFile(f)
                    total += pf.metadata.num_rows
                return total

            pf = pq.ParquetFile(path)
            row_count = pf.metadata.num_rows

            # Apply selectivity estimate if predicate is present
            # Use a conservative estimate of 30% selectivity for unknown predicates
            if self.predicate is not None:
                row_count = int(row_count * 0.3)

            return row_count
        except Exception:
            return None

    def __repr__(self) -> str:
        parts = [f"path={self.path!r}"]
        if self.columns is not None:
            parts.append(f"columns={list(self.columns)}")
        if self.predicate is not None:
            parts.append(f"predicate={self.predicate!r}")
        return f"ParquetSource({', '.join(parts)})"


@dataclass
class Project(LogicalPlan):
    """Projection (column selection/computation)."""

    input: LogicalPlan
    exprs: tuple[Expr, ...]

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        from pandas.lazy.types import Schema

        input_schema = self.input.resolve_schema()
        return Schema.from_exprs(self.exprs, input_schema)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # Projection doesn't change row count
        return self.input.estimate_row_count()

    def __repr__(self) -> str:
        from pandas.lazy.expr import extract_output_name

        try:
            names = [extract_output_name(e) for e in self.exprs]
        except ValueError:
            names = [repr(e) for e in self.exprs]
        return f"Project({names})"


@dataclass
class Filter(LogicalPlan):
    """Row filtering by predicate."""

    input: LogicalPlan
    predicate: Expr

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        return self.input.resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # Use a default selectivity estimate of 30% for unknown predicates
        # This is a conservative estimate that works reasonably well in practice
        input_count = self.input.estimate_row_count()
        if input_count is None:
            return None
        return max(1, int(input_count * 0.3))

    def __repr__(self) -> str:
        return f"Filter({self.predicate!r})"


@dataclass
class Aggregate(LogicalPlan):
    """Aggregation with optional grouping."""

    input: LogicalPlan
    group_by: tuple[Expr, ...]  # Grouping columns (empty for global agg)
    agg_exprs: tuple[Expr, ...]  # Aggregation expressions

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        from pandas.lazy.expr import extract_output_name
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        input_schema = self.input.resolve_schema()
        columns: dict[str, LazyDtype] = {}

        # Add group-by columns first
        for expr in self.group_by:
            name = extract_output_name(expr)
            columns[name] = infer_expr_dtype(expr._ir, input_schema)

        # Add aggregation result columns
        for expr in self.agg_exprs:
            name = extract_output_name(expr)
            columns[name] = infer_expr_dtype(expr._ir, input_schema)

        return Schema(columns)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # For global aggregation, result is 1 row
        if len(self.group_by) == 0:
            return 1

        # For grouped aggregation, estimate based on number of groups
        # Heuristic: assume cardinality reduces by sqrt(n)
        input_count = self.input.estimate_row_count()
        if input_count is None:
            return None

        # Conservative estimate: at most input_count groups,
        # but typically much fewer (use sqrt as heuristic)
        import math

        return max(1, min(input_count, int(math.sqrt(input_count) * 10)))

    def __repr__(self) -> str:
        from pandas.lazy.expr import extract_output_name

        try:
            group_names = [extract_output_name(e) for e in self.group_by]
            agg_names = [extract_output_name(e) for e in self.agg_exprs]
        except ValueError:
            group_names = [repr(e) for e in self.group_by]
            agg_names = [repr(e) for e in self.agg_exprs]
        return f"Aggregate(group_by={group_names}, aggs={agg_names})"


@dataclass
class Sort(LogicalPlan):
    """Sort rows by expressions."""

    input: LogicalPlan
    by: tuple[Expr, ...]  # Sort keys
    descending: tuple[bool, ...]  # Descending flags for each key

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        return self.input.resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # Sort doesn't change row count
        return self.input.estimate_row_count()

    def __repr__(self) -> str:
        from pandas.lazy.expr import extract_output_name

        try:
            names = [extract_output_name(e) for e in self.by]
        except ValueError:
            names = [repr(e) for e in self.by]
        dirs = ["desc" if d else "asc" for d in self.descending]
        parts = [f"{n}:{d}" for n, d in zip(names, dirs, strict=False)]
        return f"Sort({parts})"


@dataclass
class Limit(LogicalPlan):
    """Limit the number of rows."""

    input: LogicalPlan
    n: int
    offset: int = 0  # For skip/offset functionality

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        return self.input.resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # Limit returns at most n rows (minus offset)
        input_count = self.input.estimate_row_count()
        if input_count is None:
            return self.n  # Best estimate is the limit itself
        return min(self.n, max(0, input_count - self.offset))

    def __repr__(self) -> str:
        if self.offset > 0:
            return f"Limit(n={self.n}, offset={self.offset})"
        return f"Limit(n={self.n})"


@dataclass
class Distinct(LogicalPlan):
    """Remove duplicate rows."""

    input: LogicalPlan
    subset: tuple[str, ...] | None = None  # Columns to consider for uniqueness

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        return self.input.resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # Distinct reduces rows, use sqrt heuristic similar to Aggregate
        input_count = self.input.estimate_row_count()
        if input_count is None:
            return None
        import math

        return max(1, int(math.sqrt(input_count) * 10))

    def __repr__(self) -> str:
        if self.subset:
            return f"Distinct(subset={list(self.subset)})"
        return "Distinct()"


@dataclass
class Convert(LogicalPlan):
    """
    Explicit conversion between backends.

    Inserted by the EngineSelection optimization pass to make conversions
    visible in the plan. This enables:
    - explain() to show conversion points
    - Optimizer to minimize/eliminate conversions
    - Cost estimation for backend choices
    """

    input: LogicalPlan
    target_backend: str  # "arrow" | "numpy"

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        # Schema unchanged, only backend representation changes
        return self.input.resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # Conversion doesn't change row count
        return self.input.estimate_row_count()

    def __repr__(self) -> str:
        return f"Convert(to={self.target_backend!r})"


@dataclass
class TopK(LogicalPlan):
    """
    Get top K rows by sort criteria.

    This is an optimization of Sort followed by Limit that can use
    more efficient algorithms (e.g., heap-based selection) instead of
    fully sorting all data.
    """

    input: LogicalPlan
    k: int  # Number of rows to return
    by: tuple[Expr, ...]  # Sort keys
    descending: tuple[bool, ...]  # Descending flags for each key

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        return self.input.resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # TopK returns exactly k rows (or fewer if input is smaller)
        input_count = self.input.estimate_row_count()
        if input_count is None:
            return self.k
        return min(self.k, input_count)

    def __repr__(self) -> str:
        from pandas.lazy.expr import extract_output_name

        try:
            names = [extract_output_name(e) for e in self.by]
        except ValueError:
            names = [repr(e) for e in self.by]
        dirs = ["desc" if d else "asc" for d in self.descending]
        parts = [f"{n}:{d}" for n, d in zip(names, dirs, strict=False)]
        return f"TopK(k={self.k}, by={parts})"


@dataclass
class Join(LogicalPlan):
    """Join two DataFrames."""

    left: LogicalPlan
    right: LogicalPlan
    on: tuple[str, ...] | None = None  # Columns to join on (same name in both)
    left_on: tuple[str, ...] | None = None  # Left join columns
    right_on: tuple[str, ...] | None = None  # Right join columns
    how: str = "inner"  # inner, left, right, outer, cross
    suffix: tuple[str, str] = ("_x", "_y")  # Suffixes for overlapping columns

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
        )

        left_schema = self.left.resolve_schema()
        right_schema = self.right.resolve_schema()
        columns: dict[str, LazyDtype] = {}

        # Determine join columns
        if self.on is not None:
            join_cols = set(self.on)
        elif self.left_on is not None and self.right_on is not None:
            join_cols = set(self.left_on)
        else:
            join_cols = set()

        # Add all left columns
        for name in left_schema.names:
            dtype = left_schema[name]
            if name in right_schema and name not in join_cols:
                # Overlapping non-join column gets suffix
                columns[name + self.suffix[0]] = dtype
            else:
                columns[name] = dtype

        # Add right columns (excluding join columns if using 'on')
        for name in right_schema.names:
            if self.on is not None and name in join_cols:
                continue  # Skip, already added from left
            dtype = right_schema[name]
            if name in left_schema and name not in join_cols:
                # Overlapping non-join column gets suffix
                columns[name + self.suffix[1]] = dtype
            elif self.left_on is not None and name in set(self.right_on or ()):
                # Right join column with different name
                columns[name] = dtype
            elif name not in columns:
                columns[name] = dtype

        return Schema(columns)

    def children(self) -> list[LogicalPlan]:
        return [self.left, self.right]

    def _estimate_row_count_impl(self) -> int | None:
        """
        Estimate join cardinality based on join type.

        These are rough estimates for optimization purposes:
        - Inner: min(left, right) * selectivity factor
        - Left/Right: preserves one side, may expand for many-to-many
        - Outer: max(left, right) * expansion factor
        - Cross: left * right
        - Semi/Anti: subset of left
        """
        left_count = self.left.estimate_row_count()
        right_count = self.right.estimate_row_count()

        if left_count is None and right_count is None:
            return None

        # Use available estimates, default to 10000 for unknown
        left_count = left_count or 10000
        right_count = right_count or 10000

        if self.how == "cross":
            # Cross join: Cartesian product
            return left_count * right_count

        if self.how == "inner":
            # Inner join: typically reduces result size
            # Heuristic: selectivity based on smaller table
            return min(left_count, right_count)

        if self.how == "left":
            # Left join: at least left_count rows
            # May have more if one-to-many relationship
            return left_count

        if self.how == "right":
            # Right join: at least right_count rows
            return right_count

        if self.how == "outer":
            # Full outer: at most sum of both sides (no matches)
            # More typically: max of both sides
            return max(left_count, right_count)

        if self.how == "semi":
            # Semi join: at most left_count rows
            # Typically filters down to matched rows
            return min(left_count, right_count)

        if self.how == "anti":
            # Anti join: at most left_count rows (unmatched)
            # Heuristic: assume most left rows don't match
            return max(1, int(left_count * 0.7))

        # Default: use left count
        return left_count

    def __repr__(self) -> str:
        if self.on:
            return f"Join(on={list(self.on)}, how={self.how!r})"
        elif self.left_on and self.right_on:
            left = list(self.left_on)
            right = list(self.right_on)
            return f"Join(left_on={left}, right_on={right}, how={self.how!r})"
        return f"Join(how={self.how!r})"


@dataclass
class SetIndex(LogicalPlan):
    """
    Set column(s) as the DataFrame index.

    This is a lazy operation - the index is set when .collect() is called
    with preserve_index=True.

    Parameters
    ----------
    input : LogicalPlan
        Input plan node.
    keys : tuple[str, ...]
        Column name(s) to use as index.
    drop : bool, default True
        If True, delete columns to be used as the new index.
    """

    input: LogicalPlan
    keys: tuple[str, ...]
    drop: bool = True

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        from pandas.lazy.types import Schema

        input_schema = self.input.resolve_schema()

        if self.drop:
            # Remove key columns from schema
            new_fields = {
                k: v for k, v in input_schema.fields.items() if k not in self.keys
            }
            return Schema(new_fields)
        # Keep all columns
        return input_schema

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # SetIndex doesn't change row count
        return self.input.estimate_row_count()

    def __repr__(self) -> str:
        return f"SetIndex(keys={list(self.keys)}, drop={self.drop})"


@dataclass
class ResetIndex(LogicalPlan):
    """
    Reset the index to default RangeIndex.

    Parameters
    ----------
    input : LogicalPlan
        Input plan node.
    drop : bool, default False
        If False, insert index into DataFrame columns.
        If True, discard the index entirely.
    """

    input: LogicalPlan
    drop: bool = False

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        from pandas.lazy.types import Schema

        input_schema = self.input.resolve_schema()

        if self.drop:
            # Just return the same schema - index is discarded
            return input_schema

        # Add index columns back to schema
        # Get index names from input schema
        new_fields = dict(input_schema.fields)

        # Add index columns at the beginning
        # Note: The actual index column names depend on the source DataFrame
        # This will be handled at execution time
        return Schema(new_fields)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # ResetIndex doesn't change row count
        return self.input.estimate_row_count()

    def __repr__(self) -> str:
        return f"ResetIndex(drop={self.drop})"


@dataclass
class Concat(LogicalPlan):
    """
    Concatenate multiple DataFrames vertically (union all).

    All inputs must have compatible schemas (same column names and types).
    This is a lazy operation - the concatenation happens at execution time.

    Parameters
    ----------
    inputs : tuple[LogicalPlan, ...]
        Tuple of plan nodes to concatenate.

    Examples
    --------
    >>> from pandas.lazy import concat, scan
    >>> lf1 = scan("data/part1.parquet")
    >>> lf2 = scan("data/part2.parquet")
    >>> combined = concat([lf1, lf2])
    >>> result = combined.filter(col("value") > 100).collect()
    """

    inputs: tuple[LogicalPlan, ...]

    def __post_init__(self) -> None:
        self._cached_schema = None
        if len(self.inputs) == 0:
            raise ValueError("Concat requires at least one input")

    def _resolve_schema_impl(self) -> Schema:
        # Use schema from first input (all should be compatible)
        return self.inputs[0].resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return list(self.inputs)

    def _estimate_row_count_impl(self) -> int | None:
        total = 0
        for inp in self.inputs:
            count = inp.estimate_row_count()
            if count is None:
                return None
            total += count
        return total

    def __repr__(self) -> str:
        return f"Concat(inputs={len(self.inputs)})"
