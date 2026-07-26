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


from pandas.lazy.expr import (
    Expr,
    extract_output_name,
)
from pandas.lazy.ir import (
    Alias,
    FieldRef,
)
from pandas.lazy.type_inference import infer_expr_dtype
from pandas.lazy.types import (
    LazyDtype,
    Schema,
)


def _fresh_range_index_fields(data_names) -> dict[str, LazyDtype]:
    """Index metadata for an operator that emits a fresh default RangeIndex.

    Aggregate and Join produce a brand-new positional index, which a later
    ``reset_index()`` materializes as a leading integer column (matching
    eager pandas). The column is named ``"index"`` unless that name is already
    taken by a data column, in which case pandas falls back to ``"level_0"``.
    """
    import numpy as np

    name = "index" if "index" not in data_names else "level_0"
    return {name: LazyDtype("numeric", np.dtype("int64"), None, False)}


def _unify_concat_dtypes(dtypes: list[LazyDtype]) -> LazyDtype:
    """Combine the dtypes of one column across the inputs of a vertical concat.

    The result is nullable if any input is nullable, and widens across mixed
    numeric types the way pandas does (int + float -> float64). This keeps the
    reported dtype/nullability truthful so optimizer rewrites that depend on
    non-nullability do not fire on a column that concat can fill with NaN.
    """
    import numpy as np

    first = dtypes[0]
    nullable = any(d.nullable for d in dtypes)

    def _npdt(d: LazyDtype):
        # ``numpy_dtype`` may be a numpy scalar *type* (from Arrow's
        # to_pandas_dtype) rather than a np.dtype; normalize so ``.kind`` /
        # ``.itemsize`` are always available.
        return np.dtype(d.numpy_dtype) if d.numpy_dtype is not None else None

    # Identical concrete dtype across all inputs: only nullability can change.
    if all(
        d.category == first.category
        and _npdt(d) == _npdt(first)
        and d.arrow_type == first.arrow_type
        for d in dtypes[1:]
    ):
        return LazyDtype(first.category, first.numpy_dtype, first.arrow_type, nullable)

    # Mixed numeric widths/kinds: pandas upcasts int+float to float64. A float
    # participant means the output can carry NaN, so force nullable.
    if all(d.is_numeric() for d in dtypes):
        np_dts = [dt for dt in (_npdt(d) for d in dtypes) if dt is not None]
        has_float = any(dt.kind == "f" for dt in np_dts)
        if all(d.arrow_type is not None for d in dtypes):
            # Keep the column Arrow-backed and let Arrow's promotion rules pick
            # the common type (matching the physical concat's promote path).
            import pyarrow as pa

            promoted = (
                pa.float64()
                if has_float
                else max((d.arrow_type for d in dtypes), key=lambda t: t.bit_width)
            )
            return LazyDtype.from_arrow_type(promoted)
        if has_float:
            return LazyDtype("numeric", np.dtype("float64"), None, True)
        # All-integer but differing widths: no NaN introduced, keep the widest.
        widest = max(np_dts, key=lambda dt: dt.itemsize) if np_dts else None
        return LazyDtype("numeric", widest, None, nullable)

    # Incompatible categories (e.g. numeric + string): fall back to object.
    return LazyDtype("object", np.dtype("object"), None, True)


class LogicalPlan:
    """
    Base class for logical plan nodes.

    Note: This is not a dataclass to avoid inheritance issues with
    the schema cache field.
    """

    # Intentionally NOT slotted. The @dataclass subclasses are dict-backed
    # (they'd each need their own slots=True to benefit, and the mutable
    # _cached_schema memo rules out the frozen+slots combo the IR nodes use).
    # Plan nodes are allocated in the single digits per query and hold no bulk
    # data, so slotting them saves ~1 KB against MB-GB row payloads — not worth
    # the per-subclass friction. Slots stay where they pay off: ir.py / types.py.

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

    def invalidate_schema_cache(self) -> None:
        """Clear cached schemas on this node and all descendants.

        A ``DataFrameSource`` schema is derived from a live, user-held
        DataFrame whose dtypes can change after the plan is built (an int
        column gains a NaN and becomes float). Clearing the cache before
        optimization guarantees nullability-dependent rewrites see the current
        dtypes rather than a stale snapshot captured at construction time.
        """
        self._cached_schema = None
        for child in self.children():
            child.invalidate_schema_cache()

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

    def column_statistics(self, name: str):
        """Best-effort `ColumnStats` for a source column, or None.

        Sources compute statistics (sampled for in-memory frames, metadata for
        Parquet); row-preserving operators that pass a column through unchanged
        delegate to their input, so a `Filter`'s selectivity estimate can size
        a predicate by the underlying column's real distribution. Operators
        that transform or aggregate a column return None for it. Used only by
        the cardinality model, which always has a constant fallback.
        """
        children = self.children()
        if len(children) == 1:
            return children[0].column_statistics(name)
        return None


@dataclass
class DataFrameSource(LogicalPlan):
    """Source node wrapping an eager DataFrame."""

    df: DataFrame

    def __post_init__(self) -> None:
        # Duplicate labels make column lookup return a DataFrame instead of a
        # Series, which otherwise crashes deep in execution with a cryptic
        # AttributeError. Fail clearly at plan construction instead.
        if self.df.columns.has_duplicates:
            dups = self.df.columns[self.df.columns.duplicated()].unique().tolist()
            raise NotImplementedError(
                "lazy pandas does not support DataFrames with duplicate column "
                f"labels: {dups}. Rename or drop the duplicates before going lazy."
            )
        self._cached_schema = None
        self._stats_cache: dict[str, object] = {}

    def _resolve_schema_impl(self) -> Schema:

        return Schema.from_dataframe(self.df)

    def children(self) -> list[LogicalPlan]:
        return []

    def _estimate_row_count_impl(self) -> int | None:
        return len(self.df)

    def column_statistics(self, name: str):
        if name not in self.df.columns:
            return None
        if name not in self._stats_cache:
            from pandas.lazy.optimize.cardinality import compute_column_stats

            self._stats_cache[name] = compute_column_stats(self.df[name])
        return self._stats_cache[name]

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
    # Row limit pushed into the scan (head over scan). Enables early
    # read termination and the direct small-limit ParquetFile path.
    # Set by the LimitPushdown optimization pass.
    limit: int | None = None

    # Cached schema from Parquet metadata
    _parquet_schema: Schema | None = None

    def __post_init__(self) -> None:
        self._cached_schema = None
        # Don't set _parquet_schema here - it will be resolved lazily

    def _resolve_schema_impl(self) -> Schema:

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

            # Apply predicate-aware selectivity to the pushed-down filter,
            # refined by row-group statistics where available (min/max for
            # ranges, null fraction for is_null); see optimize/cardinality.py.
            if self.predicate is not None:
                from pandas.lazy.optimize.cardinality import estimate_selectivity

                sel = estimate_selectivity(self.predicate._ir, self.column_statistics)
                row_count = max(1, int(row_count * sel))

            return row_count
        except Exception:
            return None

    def column_statistics(self, name: str):
        """`ColumnStats` from Parquet row-group metadata, or None.

        Min/max and null counts are aggregated across all row groups (and
        files for a glob) straight from the footer — no data is read. Parquet
        does not carry a reliable cross-group distinct count, so ``ndv`` stays
        None and equality predicates fall back to the constant model.
        """
        cache = getattr(self, "_col_stats_cache", None)
        if cache is None:
            cache = self._read_column_statistics()
            object.__setattr__(self, "_col_stats_cache", cache)
        return cache.get(name)

    def _read_column_statistics(self) -> dict:
        import pyarrow.parquet as pq

        from pandas.lazy.optimize.cardinality import ColumnStats

        path = self.path
        if "*" in path:
            import glob

            files = glob.glob(path)
        else:
            files = [path]
        if not files:
            return {}

        # Per column: running min/max (numeric only), summed null_count, and
        # flags for whether every chunk supplied each statistic.
        acc: dict[str, dict] = {}
        total_rows = 0
        try:
            for fp in files:
                md = pq.ParquetFile(fp).metadata
                total_rows += md.num_rows
                for rg_i in range(md.num_row_groups):
                    rg = md.row_group(rg_i)
                    for j in range(md.num_columns):
                        cc = rg.column(j)
                        st = cc.statistics
                        a = acc.setdefault(
                            cc.path_in_schema,
                            {"min": None, "max": None, "nulls": 0, "has_nulls": True},
                        )
                        if st is not None and st.has_min_max:
                            lo, hi = st.min, st.max
                            if isinstance(lo, (int, float)) and not isinstance(
                                lo, bool
                            ):
                                a["min"] = lo if a["min"] is None else min(a["min"], lo)
                                a["max"] = hi if a["max"] is None else max(a["max"], hi)
                        if st is not None and st.has_null_count:
                            a["nulls"] += st.null_count
                        else:
                            a["has_nulls"] = False
        except Exception:
            return {}

        result: dict[str, ColumnStats] = {}
        for nm, a in acc.items():
            result[nm] = ColumnStats(
                row_count=total_rows,
                min_val=(float(a["min"]) if a["min"] is not None else None),
                max_val=(float(a["max"]) if a["max"] is not None else None),
                null_count=(a["nulls"] if a["has_nulls"] else None),
                approximate=False,
            )
        return result

    def __repr__(self) -> str:
        parts = [f"path={self.path!r}"]
        if self.columns is not None:
            parts.append(f"columns={list(self.columns)}")
        if self.predicate is not None:
            parts.append(f"predicate={self.predicate!r}")
        return f"ParquetSource({', '.join(parts)})"


@dataclass
class CSVSource(LogicalPlan):
    """
    Source node for reading CSV files lazily.

    Supports projection pushdown to minimize I/O. Predicate pushdown
    is also supported but filtering happens after reading (CSV doesn't
    have native pushdown like Parquet row group filtering).

    Parameters
    ----------
    path : str
        Path to CSV file or directory. Supports:
        - Local paths: "/path/to/data.csv"
        - Glob patterns: "/path/to/*.csv"
        - Compressed files: "/path/to/data.csv.gz"
    columns : tuple[str, ...] | None
        Columns to read. None means all columns.
        Set by ProjectionPruning optimization pass.
    predicate : Expr | None
        Filter predicate to apply after reading.
        Set by PredicatePushdown optimization pass.
    sep : str
        Delimiter/separator character.
    header : bool
        Whether the file has a header row.
    skip_rows : int
        Number of rows to skip at the start.
    n_rows : int | None
        Maximum number of rows to read. None means all rows.
    """

    path: str
    columns: tuple[str, ...] | None = None
    predicate: Expr | None = None
    sep: str = ","
    header: bool = True
    skip_rows: int = 0
    n_rows: int | None = None

    # Cached schema from CSV header/inference
    _csv_schema: Schema | None = None

    def __post_init__(self) -> None:
        self._cached_schema = None
        # Don't set _csv_schema here - it will be resolved lazily

    def _resolve_schema_impl(self) -> Schema:

        if self._csv_schema is not None:
            schema = self._csv_schema
        else:
            # Read schema from CSV header/first rows
            schema = self._read_csv_schema()
            # Cache it for future use
            object.__setattr__(self, "_csv_schema", schema)

        # If columns are specified, filter schema
        if self.columns is not None:
            return Schema({c: schema[c] for c in self.columns if c in schema})
        return schema

    def _read_csv_schema(self) -> Schema:
        """Read schema from CSV file by scanning first rows."""
        import pyarrow as pa
        from pyarrow import csv

        # Handle glob patterns
        path = self.path
        if "*" in path:
            import glob

            files = glob.glob(path)
            if not files:
                raise FileNotFoundError(f"No files match pattern: {path}")
            path = files[0]  # Use first file for schema

        # Configure CSV read options
        read_options = csv.ReadOptions(
            skip_rows=self.skip_rows,
            autogenerate_column_names=not self.header,
        )
        parse_options = csv.ParseOptions(delimiter=self.sep)

        # Read a small sample to infer schema (just read first batch)
        # Using block_size to limit how much data we read
        try:
            reader = csv.open_csv(
                path,
                read_options=read_options,
                parse_options=parse_options,
            )
            # Get schema from reader
            arrow_schema = reader.schema
        except pa.ArrowInvalid as e:
            raise ValueError(f"Cannot read CSV schema from {path}: {e}") from e

        # Convert Arrow schema to lazy Schema
        columns = {}
        for field in arrow_schema:
            columns[field.name] = LazyDtype.from_arrow_type(field.type)

        return Schema(columns)

    def children(self) -> list[LogicalPlan]:
        return []

    def _estimate_row_count_impl(self) -> int | None:
        """
        Estimate row count from CSV file.

        CSV files don't have row count metadata like Parquet, so we estimate
        based on file size and average row length from a sample.
        """
        try:
            path = self.path
            if "*" in path:
                import glob

                files = glob.glob(path)
                if not files:
                    return None
                # Sum estimated row counts from all files
                total = 0
                for f in files:
                    count = self._estimate_single_file_rows(f)
                    if count is not None:
                        total += count
                return total if total > 0 else None

            return self._estimate_single_file_rows(path)
        except Exception:
            return None

    def _estimate_single_file_rows(self, path: str) -> int | None:
        """Estimate row count for a single CSV file."""
        import os

        try:
            file_size = os.path.getsize(path)

            # For compressed files, estimate is less accurate
            if path.endswith(".gz"):
                # Assume ~5x compression ratio
                file_size *= 5

            # Sample first 64KB to estimate average row length
            sample_size = min(65536, file_size)
            with open(path, "rb") as f:
                sample = f.read(sample_size)

            # Handle gzip
            if path.endswith(".gz"):
                import gzip

                with gzip.open(path, "rb") as f:
                    sample = f.read(sample_size)
                    # For gzip, use decompressed sample size ratio
                    decompressed_size = len(sample)
                    if decompressed_size > 0:
                        file_size = int(
                            file_size * (decompressed_size / sample_size) * 5
                        )
                    sample_size = decompressed_size

            # Count newlines in sample
            newlines = sample.count(b"\n")
            if newlines == 0:
                return None

            # Estimate average row length
            avg_row_len = sample_size / newlines

            # Estimate total rows
            estimated_rows = int(file_size / avg_row_len)

            # Subtract header row if present
            if self.header:
                estimated_rows -= 1

            # Apply selectivity estimate if predicate is present
            if self.predicate is not None:
                estimated_rows = int(estimated_rows * 0.3)

            return max(1, estimated_rows)
        except Exception:
            return None

    def __repr__(self) -> str:
        parts = [f"path={self.path!r}"]
        if self.columns is not None:
            parts.append(f"columns={list(self.columns)}")
        if self.predicate is not None:
            parts.append(f"predicate={self.predicate!r}")
        if self.sep != ",":
            parts.append(f"sep={self.sep!r}")
        return f"CSVSource({', '.join(parts)})"


@dataclass
class Project(LogicalPlan):
    """Projection (column selection/computation)."""

    input: LogicalPlan
    exprs: tuple[Expr, ...]

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:

        input_schema = self.input.resolve_schema()
        return Schema.from_exprs(self.exprs, input_schema)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # Projection doesn't change row count
        return self.input.estimate_row_count()

    def column_statistics(self, name: str):
        # A column reaches the output unchanged only if it is projected by a
        # plain column reference; computed columns have an unknown
        # distribution. Delegate pass-throughs to the input.

        for expr in self.exprs:
            try:
                out_name = extract_output_name(expr)
            except ValueError:
                continue
            if out_name != name:
                continue
            ir = expr._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if isinstance(ir, FieldRef):
                return self.input.column_statistics(ir.name)
            return None  # computed column: distribution unknown
        return None

    def __repr__(self) -> str:

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
        # Predicate-aware selectivity: equality is sized by 1/NDV, ranges by
        # min/max interpolation (from the underlying column's statistics when
        # available), AND/OR/NOT compose, with System R constants as the
        # fallback. Feeds join build-side selection downstream, so a filtered
        # side is sized by its predicate and the column's real distribution
        # rather than a flat 30%.
        input_count = self.input.estimate_row_count()
        if input_count is None:
            return None
        from pandas.lazy.optimize.cardinality import estimate_selectivity

        sel = estimate_selectivity(self.predicate._ir, self.input.column_statistics)
        return max(1, int(input_count * sel))

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

        input_schema = self.input.resolve_schema()
        columns: dict[str, LazyDtype] = {}

        def _add(name: str, dtype: LazyDtype) -> None:
            # Group keys and aggregate outputs share one namespace. A plain
            # dict would let a duplicate silently overwrite an earlier column
            # (the logical schema then under-reports, eager emits duplicate
            # labels, and physical execution raises a cryptic Arrow duplicate-
            # field error). Reject it up front instead.
            if name in columns:
                raise ValueError(
                    f"Duplicate output column name {name!r} in aggregation; "
                    "group keys and aggregate outputs must be uniquely named "
                    "(use .alias() to rename one)."
                )
            columns[name] = dtype

        # Add group-by columns first
        for expr in self.group_by:
            name = extract_output_name(expr)
            _add(name, infer_expr_dtype(expr._ir, input_schema))

        # Add aggregation result columns
        for expr in self.agg_exprs:
            name = extract_output_name(expr)
            _add(name, infer_expr_dtype(expr._ir, input_schema))

        # The aggregate output carries a fresh RangeIndex; model it so a
        # later reset_index() reports the leading column like eager pandas.
        return Schema(columns, index_fields=_fresh_range_index_fields(set(columns)))

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # For global aggregation, result is 1 row
        if len(self.group_by) == 0:
            return 1

        input_count = self.input.estimate_row_count()
        if input_count is None:
            return None

        # The result has one row per distinct group key. With a single
        # column group-by we can use that column's NDV directly instead of
        # the sqrt heuristic.
        if len(self.group_by) == 1:
            ir = self.group_by[0]._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if isinstance(ir, FieldRef):
                st = self.input.column_statistics(ir.name)
                if st is not None and st.ndv:
                    return max(1, min(input_count, st.ndv))

        # Fallback heuristic: cardinality reduces by ~sqrt(n).
        import math

        return max(1, min(input_count, int(math.sqrt(input_count) * 10)))

    def column_statistics(self, name: str):
        # Output columns are group keys and aggregates whose distributions
        # differ from the input; do not propagate stats.
        return None

    def __repr__(self) -> str:

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
class GroupByHead(LogicalPlan):
    """Keep the first ``n`` rows of each group (group-wise head).

    Combined with a preceding ``sort`` this expresses "top-k rows per group"
    (e.g. the largest two values per key) as plain rows, without list columns
    or an explode operator.
    """

    input: LogicalPlan
    group_by: tuple[Expr, ...]
    n: int

    def __post_init__(self) -> None:
        self._cached_schema = None

    def _resolve_schema_impl(self) -> Schema:
        return self.input.resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def _estimate_row_count_impl(self) -> int | None:
        # At most n rows per group; bounded by the input row count.
        return self.input.estimate_row_count()

    def __repr__(self) -> str:
        return f"GroupByHead(n={self.n})"


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

        from pandas.errors import MergeError

        left_schema = self.left.resolve_schema()
        right_schema = self.right.resolve_schema()
        left_names = left_schema.names
        right_names = right_schema.names
        left_set = set(left_names)
        right_set = set(right_names)
        columns: dict[str, LazyDtype] = {}

        # Only an ``on`` join collapses a key to a single output column. With
        # ``left_on``/``right_on`` both key columns are kept and are suffixed
        # like any other overlap (matching pandas.merge).
        merged_keys = set(self.on) if self.on is not None else set()

        # pandas suffixes every column present on both sides except the
        # collapsed ``on`` keys. Deriving the overlap from the real column
        # sets (not from left_on alone) fixes the asymmetric
        # left_on/right_on case where a right data column shares a name with
        # a left key.
        overlap = (left_set & right_set) - merged_keys

        # An outer-family join fills the non-preserved side with NaN/NA for
        # unmatched rows, so those columns become nullable. Keeping this
        # truthful stops the optimizer from rewriting ``right_col -
        # right_col -> 0`` on a left join and dropping the introduced nulls.
        left_becomes_nullable = self.how in ("right", "outer")
        right_becomes_nullable = self.how in ("left", "outer")

        def _add(name: str, dtype: LazyDtype) -> None:
            # A suffixed name that collides with an existing column silently
            # overwrote it before, so projection pruning could drop the
            # original and optimized execution would lose user data. pandas
            # raises MergeError here; mirror that.
            if name in columns:
                raise MergeError(
                    f"Passing 'suffixes' which cause duplicate columns "
                    f"{{'{name}'}} is not allowed."
                )
            columns[name] = dtype

        # Add all left columns
        for name in left_names:
            dtype = left_schema[name]
            if name in merged_keys:
                # Collapsed key: only a full outer join can leave it unmatched.
                if self.how == "outer":
                    dtype = dtype.after_introducing_nulls()
            elif left_becomes_nullable:
                dtype = dtype.after_introducing_nulls()
            _add(name + self.suffix[0] if name in overlap else name, dtype)

        # Add right columns (the collapsed ``on`` keys already came from left)
        for name in right_names:
            if name in merged_keys:
                continue
            dtype = right_schema[name]
            if right_becomes_nullable:
                dtype = dtype.after_introducing_nulls()
            _add(name + self.suffix[1] if name in overlap else name, dtype)

        # A join produces a fresh RangeIndex; model it so a later
        # reset_index() reports the leading column like eager pandas.
        return Schema(columns, index_fields=_fresh_range_index_fields(set(columns)))

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

    This operation always sets the specified column(s) as the index when
    executed. The ``preserve_index`` parameter in ``.collect()`` controls
    whether the *source DataFrame's original index* is preserved through
    other operations (like Filter, Join, etc.), not whether SetIndex takes
    effect.

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

        input_schema = self.input.resolve_schema()

        # Reject unknown keys instead of silently ignoring them: eager
        # execution raises KeyError, so a filtering comprehension here would
        # make physical execution return the frame unchanged and diverge.
        missing = [k for k in self.keys if k not in input_schema.fields]
        if missing:
            raise KeyError(f"None of {missing} are in the columns")

        # The key columns become the (new) index, replacing any prior index.
        index_fields = {k: input_schema.fields[k] for k in self.keys}
        if self.drop:
            # Key columns move out of the visible columns into the index.
            new_fields = {
                k: v for k, v in input_schema.fields.items() if k not in self.keys
            }
            return Schema(new_fields, index_fields=index_fields)
        # drop=False: keys stay as columns AND become the index.
        return Schema(dict(input_schema.fields), index_fields=index_fields)

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

    This operation always resets the index when executed. The ``preserve_index``
    parameter in ``.collect()`` controls whether the *source DataFrame's original
    index* is preserved through other operations, not whether ResetIndex takes
    effect.

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

        input_schema = self.input.resolve_schema()

        if self.drop:
            # Index is discarded; visible columns unchanged, no index remains.
            return Schema(dict(input_schema.fields))

        # drop=False: materialize the index as leading columns, then clear it.
        # An index name that collides with an existing data column cannot be
        # inserted; pandas raises here, and silently coalescing them (the old
        # setdefault) let physical execution overwrite the index array with the
        # data column and lose the index values. Match pandas.
        new_fields: dict[str, LazyDtype] = {}
        for name, dtype in input_schema.index_fields.items():
            if name in input_schema.fields:
                raise ValueError(f"cannot insert {name}, already exists")
            new_fields[name] = dtype
        new_fields.update(input_schema.fields)
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
        # Validate that all inputs have compatible schemas (same column
        # names). Without this check, eager and physical execution produce
        # different (and both wrong) results for mismatched inputs.
        first_names = set(self.inputs[0].resolve_schema().names)
        for i, inp in enumerate(self.inputs[1:], start=1):
            names = set(inp.resolve_schema().names)
            if names != first_names:
                missing = sorted(first_names - names)
                extra = sorted(names - first_names)
                parts = []
                if missing:
                    parts.append(f"missing {missing}")
                if extra:
                    parts.append(f"unexpected {extra}")
                raise ValueError(
                    f"concat input {i} has incompatible columns "
                    f"({', '.join(parts)}); all inputs must have the same "
                    f"column names as input 0: {sorted(first_names)}"
                )

    def _resolve_schema_impl(self) -> Schema:
        # Union the per-column dtypes across all inputs rather than trusting
        # input 0. Concatenating an int64 column with a float64 column yields a
        # float column that can hold NaN; reporting input 0's non-nullable int
        # would let the optimizer rewrite ``x - x -> 0`` and lose those NaNs.
        schemas = [inp.resolve_schema() for inp in self.inputs]
        base = schemas[0]
        fields: dict[str, LazyDtype] = {}
        for name in base.names:
            fields[name] = _unify_concat_dtypes([s[name] for s in schemas])
        # All inputs share column names (validated in __post_init__); the row
        # index metadata follows input 0, matching physical execution.
        return Schema(fields, index_fields=dict(base.index_fields))

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
