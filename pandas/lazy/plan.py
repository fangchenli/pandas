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

    def __repr__(self) -> str:
        cols = list(self.df.columns)
        return f"DataFrameSource(columns={cols}, rows={len(self.df)})"


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

    def __repr__(self) -> str:
        if self.on:
            return f"Join(on={list(self.on)}, how={self.how!r})"
        elif self.left_on and self.right_on:
            left = list(self.left_on)
            right = list(self.right_on)
            return f"Join(left_on={left}, right_on={right}, how={self.how!r})"
        return f"Join(how={self.how!r})"
