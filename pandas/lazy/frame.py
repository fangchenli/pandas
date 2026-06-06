"""
LazyDataFrame - the core lazy query object.

This module provides the LazyDataFrame class which represents a lazy
query that can be optimized and executed.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas import DataFrame
    from pandas.lazy.backends.spill import SpillConfig
    from pandas.lazy.expr import Expr
    from pandas.lazy.plan import (
        Aggregate,
        Limit,
        LogicalPlan,
        Sort,
    )
    from pandas.lazy.types import Schema


class LazyDataFrame:
    """
    Lazy representation of a DataFrame query.

    This object builds an expression graph that is optimized and executed
    only when collect() is called. It supports method chaining for building
    complex queries.

    LazyDataFrame should not be instantiated directly. Use DataFrame.select()
    to enter lazy mode.

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas.lazy import col
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> ldf = df.select()  # Enter lazy mode
    >>> ldf.columns
    ['a', 'b']
    >>> result = ldf.collect()  # Execute and return DataFrame
    """

    __slots__ = ("_optimized_plan", "_plan", "_schema")

    def __init__(self, plan: LogicalPlan, schema: Schema) -> None:
        """
        Initialize a LazyDataFrame.

        This constructor is internal. Use DataFrame.select() to create
        LazyDataFrame instances.

        Parameters
        ----------
        plan : LogicalPlan
            The logical plan representing the query.
        schema : Schema
            The output schema of the query.
        """
        self._plan = plan
        self._schema = schema
        self._optimized_plan: LogicalPlan | None = None

    def _get_optimized_plan(self) -> LogicalPlan:
        """
        Get the optimized plan, caching the result.

        Returns
        -------
        LogicalPlan
            The optimized logical plan.
        """
        if self._optimized_plan is None:
            from pandas.lazy.optimize import Optimizer

            self._optimized_plan = Optimizer().optimize(self._plan)
        return self._optimized_plan

    @classmethod
    def _from_dataframe(
        cls,
        df: DataFrame,
        exprs: tuple[Expr, ...],
    ) -> LazyDataFrame:
        """
        Create LazyDataFrame from eager DataFrame with projection.

        Parameters
        ----------
        df : DataFrame
            The source DataFrame.
        exprs : tuple of Expr
            The projection expressions.

        Returns
        -------
        LazyDataFrame
            A new lazy query object.
        """
        from pandas.lazy.plan import (
            DataFrameSource,
            Project,
        )

        source = DataFrameSource(df)
        plan = Project(source, exprs)
        schema = plan.resolve_schema()
        return cls(plan, schema)

    # --- Properties ---

    @property
    def schema(self) -> Schema:
        """
        Return the output schema (column names and types).

        Returns
        -------
        Schema
            The schema describing output columns and their types.
        """
        return self._schema

    @property
    def columns(self) -> list[str]:
        """
        Return column names in output.

        Returns
        -------
        list of str
            The names of columns that will be in the output.
        """
        return self._schema.names

    # --- Chainable Operations ---

    def select(self, *exprs: Expr | str) -> LazyDataFrame:
        """
        Add projection of columns/expressions.

        Parameters
        ----------
        *exprs : Expr or str
            Column references or expressions to select.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with the projection applied.

        Examples
        --------
        >>> ldf.select("a", "b")
        >>> ldf.select(col("a"), col("b"))
        """
        from pandas.lazy.expr import normalize_exprs
        from pandas.lazy.plan import Project

        if not exprs:
            raise ValueError("select() requires at least one expression")

        exprs = normalize_exprs(exprs)
        new_plan = Project(self._plan, exprs)
        return LazyDataFrame(new_plan, new_plan.resolve_schema())

    def filter(self, predicate: Expr) -> LazyDataFrame:
        """
        Filter rows by boolean predicate.

        Parameters
        ----------
        predicate : Expr
            A boolean expression to filter rows.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with the filter applied.

        Examples
        --------
        >>> from pandas.lazy import col
        >>> ldf.filter(col("a") > 0)
        """
        from pandas.lazy.expr import Expr
        from pandas.lazy.plan import Filter

        if not isinstance(predicate, Expr):
            raise TypeError(
                f"predicate must be an Expr, got {type(predicate).__name__}"
            )

        new_plan = Filter(self._plan, predicate)
        return LazyDataFrame(new_plan, self._schema)

    def with_columns(self, *exprs: Expr) -> LazyDataFrame:
        """
        Add or replace columns with computed expressions.

        Parameters
        ----------
        *exprs : Expr
            Expressions to add as columns. Each expression must have
            an alias (use .alias("name")) to specify the output column name.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with the new columns added.

        Examples
        --------
        >>> from pandas.lazy import col
        >>> ldf.with_columns((col("a") + col("b")).alias("sum"))
        >>> ldf.with_columns(
        ...     (col("price") * col("qty")).alias("total"),
        ...     (col("a") > 0).alias("is_positive"),
        ... )
        """
        from pandas.lazy.expr import (
            Expr,
            col,
            extract_output_name,
        )
        from pandas.lazy.plan import Project

        if not exprs:
            raise ValueError("with_columns() requires at least one expression")

        for e in exprs:
            if not isinstance(e, Expr):
                raise TypeError(
                    f"Expected Expr, got {type(e).__name__}. "
                    f"Use col('name') or (expr).alias('name')."
                )

        # Build expression list: all existing columns + new columns
        # New columns can replace existing ones with the same name
        existing_names = self._schema.names
        new_names = set()
        for e in exprs:
            name = extract_output_name(e)
            new_names.add(name)

        # Keep existing columns that aren't being replaced
        all_exprs = [col(name) for name in existing_names if name not in new_names]

        # Add new columns
        all_exprs.extend(exprs)

        new_plan = Project(self._plan, tuple(all_exprs))
        return LazyDataFrame(new_plan, new_plan.resolve_schema())

    def with_row_index(self, name: str = "index", offset: int = 0) -> LazyDataFrame:
        """
        Add a row index column.

        This method adds a column containing row numbers (starting from offset).
        The row numbers are computed based on the position of rows in the
        DataFrame at execution time.

        Parameters
        ----------
        name : str, default "index"
            Name for the new row index column.
        offset : int, default 0
            Start counting from this value.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with the row index column added.

        Examples
        --------
        >>> df.select().with_row_index().collect()
           index  a  b
        0      0  1  4
        1      1  2  5
        2      2  3  6

        >>> df.select().with_row_index("row_num", offset=1).collect()
           row_num  a  b
        0        1  1  4
        1        2  2  5
        2        3  3  6

        See Also
        --------
        LazyDataFrame.with_columns : Add computed columns.
        """
        from pandas.lazy.expr import Expr
        from pandas.lazy.ir import Call

        # Create row_index expression
        row_index_ir = Call("row_index", (), {"offset": offset})
        row_index_expr = Expr(row_index_ir).alias(name)

        return self.with_columns(row_index_expr)

    def set_index(self, keys: str | list[str], drop: bool = True) -> LazyDataFrame:
        """
        Set column(s) as the DataFrame index.

        This is a lazy operation that always takes effect when .collect() is
        called. If drop=True (default), the columns used as index are removed
        from the result.

        Note: The ``preserve_index`` parameter in ``.collect()`` controls
        whether the *source DataFrame's original index* is preserved through
        operations, not whether ``set_index()`` takes effect. An explicit
        ``set_index()`` call always sets the index.

        Parameters
        ----------
        keys : str or list of str
            Column name(s) to use as index.
        drop : bool, default True
            If True, delete columns to be used as the new index.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with the index set.

        Examples
        --------
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> df.select().set_index("a").collect()
           b
        a
        1  4
        2  5
        3  6

        See Also
        --------
        LazyDataFrame.reset_index : Reset index to default RangeIndex.
        """
        from pandas.lazy.plan import SetIndex

        if isinstance(keys, str):
            keys = [keys]

        new_plan = SetIndex(input=self._plan, keys=tuple(keys), drop=drop)
        return LazyDataFrame(new_plan, new_plan.resolve_schema())

    def reset_index(self, drop: bool = False) -> LazyDataFrame:
        """
        Reset the index to default RangeIndex.

        This is a lazy operation. When collected, the result will have
        a fresh RangeIndex. If drop=False (default), the current index
        is inserted as a column in the result.

        Parameters
        ----------
        drop : bool, default False
            If False, insert index into DataFrame columns.
            If True, discard the index entirely.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with the index reset.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx")
        ... )
        >>> df.select().reset_index().collect()
           idx  a
        0   10  1
        1   20  2
        2   30  3

        >>> df.select().reset_index(drop=True).collect()
           a
        0  1
        1  2
        2  3

        See Also
        --------
        LazyDataFrame.set_index : Set column(s) as index.
        """
        from pandas.lazy.plan import ResetIndex

        new_plan = ResetIndex(input=self._plan, drop=drop)
        return LazyDataFrame(new_plan, new_plan.resolve_schema())

    def group_by(self, *by: Expr | str) -> LazyGroupBy:
        """
        Group by one or more columns for aggregation.

        Parameters
        ----------
        *by : Expr or str
            Columns to group by. Can be column names or expressions.

        Returns
        -------
        LazyGroupBy
            A grouped lazy dataframe for aggregation operations.

        Examples
        --------
        >>> from pandas.lazy import col
        >>> ldf.group_by("category").agg(col("value").sum().alias("total"))
        >>> ldf.group_by(col("a"), col("b")).agg(
        ...     col("x").mean().alias("avg_x"),
        ...     col("y").max().alias("max_y"),
        ... )
        """
        from pandas.lazy.expr import normalize_exprs

        if not by:
            raise ValueError("group_by() requires at least one column")

        group_exprs = normalize_exprs(by)
        return LazyGroupBy(self._plan, group_exprs, self._schema)

    def sort(
        self,
        *by: Expr | str,
        descending: bool | list[bool] = False,
    ) -> LazyDataFrame:
        """
        Sort by one or more columns.

        Parameters
        ----------
        *by : Expr or str
            Columns to sort by. Can be column names or expressions.
        descending : bool or list of bool, default False
            Sort descending. If a list, must match the number of sort keys.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with sort applied.

        Examples
        --------
        >>> from pandas.lazy import col
        >>> ldf.sort("a")
        >>> ldf.sort("a", descending=True)
        >>> ldf.sort("a", "b", descending=[True, False])
        >>> ldf.sort(col("a") + col("b"))
        """
        from pandas.lazy.expr import normalize_exprs
        from pandas.lazy.plan import Sort

        if not by:
            raise ValueError("sort() requires at least one column")

        sort_exprs = normalize_exprs(by)

        # Normalize descending to tuple
        if isinstance(descending, bool):
            desc_tuple = tuple(descending for _ in sort_exprs)
        else:
            if len(descending) != len(sort_exprs):
                raise ValueError(
                    f"Length of descending ({len(descending)}) must match "
                    f"number of sort keys ({len(sort_exprs)})"
                )
            desc_tuple = tuple(descending)

        new_plan = Sort(self._plan, sort_exprs, desc_tuple)
        return LazyDataFrame(new_plan, self._schema)

    def head(self, n: int = 5) -> LazyDataFrame:
        """
        Get the first n rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame limited to first n rows.

        Examples
        --------
        >>> ldf.head()
        >>> ldf.head(10)
        """
        from pandas.lazy.plan import Limit

        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")

        new_plan = Limit(self._plan, n)
        return LazyDataFrame(new_plan, self._schema)

    def tail(self, n: int = 5) -> LazyDataFrame:
        """
        Get the last n rows.

        Note: This operation requires materializing the input to determine
        the total row count, which may be expensive for large datasets.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame limited to last n rows.

        Examples
        --------
        >>> ldf.tail()
        >>> ldf.tail(10)
        """
        from pandas.lazy.plan import Limit

        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")

        # Use negative offset to indicate "from end"
        # The evaluator will handle this specially
        new_plan = Limit(self._plan, n, offset=-1)
        return LazyDataFrame(new_plan, self._schema)

    def limit(self, n: int) -> LazyDataFrame:
        """
        Limit to at most n rows.

        Alias for head(n).

        Parameters
        ----------
        n : int
            Maximum number of rows to return.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame limited to at most n rows.

        Examples
        --------
        >>> ldf.limit(100)
        """
        return self.head(n)

    def distinct(self, *subset: str) -> LazyDataFrame:
        """
        Remove duplicate rows.

        Parameters
        ----------
        *subset : str
            Column names to consider for identifying duplicates.
            If not provided, all columns are used.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with duplicate rows removed.

        Examples
        --------
        >>> ldf.distinct()  # All columns
        >>> ldf.distinct("a", "b")  # Only consider columns a and b
        """
        from pandas.lazy.plan import Distinct

        subset_tuple = subset if subset else None
        new_plan = Distinct(self._plan, subset_tuple)
        return LazyDataFrame(new_plan, self._schema)

    def join(
        self,
        other: LazyDataFrame | DataFrame,
        on: str | list[str] | None = None,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        how: Literal["inner", "left", "right", "outer", "cross"] = "inner",
        suffix: tuple[str, str] = ("_x", "_y"),
    ) -> LazyDataFrame:
        """
        Join with another LazyDataFrame or DataFrame.

        Parameters
        ----------
        other : LazyDataFrame or DataFrame
            The right DataFrame to join with. If a regular DataFrame is
            provided, it will be automatically converted to a LazyDataFrame.
        on : str or list of str, optional
            Column(s) to join on. Must exist in both DataFrames.
        left_on : str or list of str, optional
            Column(s) from the left DataFrame to join on.
        right_on : str or list of str, optional
            Column(s) from the right DataFrame to join on.
        how : {"inner", "left", "right", "outer", "cross"}, default "inner"
            Type of join to perform.
        suffix : tuple of str, default ("_x", "_y")
            Suffixes to apply to overlapping column names.

        Returns
        -------
        LazyDataFrame
            A new LazyDataFrame with the join result.

        Examples
        --------
        >>> ldf1.join(ldf2, on="id")
        >>> ldf1.join(df2, on="id")  # Also accepts regular DataFrame
        >>> ldf1.join(ldf2, left_on="id1", right_on="id2")
        >>> ldf1.join(ldf2, on="id", how="left")
        """
        from pandas import DataFrame
        from pandas.lazy.plan import (
            DataFrameSource,
            Join,
        )

        # Convert DataFrame to LazyDataFrame if needed
        if isinstance(other, DataFrame):
            from pandas.lazy.expr import col

            # Create a simple projection that selects all columns
            other_exprs = tuple(col(c) for c in other.columns)
            other_source = DataFrameSource(other)
            from pandas.lazy.plan import Project

            other_plan = Project(other_source, other_exprs)
            other_schema = other_plan.resolve_schema()
            other = LazyDataFrame(other_plan, other_schema)

        # Normalize on/left_on/right_on to tuples
        if on is not None:
            on_tuple = (on,) if isinstance(on, str) else tuple(on)
            left_on_tuple = None
            right_on_tuple = None
        elif left_on is not None and right_on is not None:
            on_tuple = None
            left_on_tuple = (left_on,) if isinstance(left_on, str) else tuple(left_on)
            right_on_tuple = (
                (right_on,) if isinstance(right_on, str) else tuple(right_on)
            )
            if len(left_on_tuple) != len(right_on_tuple):
                raise ValueError(
                    f"left_on and right_on must have same length, "
                    f"got {len(left_on_tuple)} and {len(right_on_tuple)}"
                )
        elif how == "cross":
            on_tuple = None
            left_on_tuple = None
            right_on_tuple = None
        else:
            raise ValueError(
                "Must provide either 'on' or both 'left_on' and 'right_on' "
                "(unless how='cross')"
            )

        new_plan = Join(
            self._plan,
            other._plan,
            on=on_tuple,
            left_on=left_on_tuple,
            right_on=right_on_tuple,
            how=how,
            suffix=suffix,
        )
        new_schema = new_plan.resolve_schema()
        return LazyDataFrame(new_plan, new_schema)

    # --- Terminal Operations ---

    def collect(
        self,
        *,
        optimize: bool = True,
        strict: bool = False,
        engine: Literal["auto", "arrow", "numpy"] = "auto",
        use_physical_planner: bool = False,
        streaming: bool = False,
        batch_size: int = 65536,
        preserve_index: bool = False,
        spill_config: SpillConfig | None = None,
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Execute the query and return a pandas DataFrame.

        Parameters
        ----------
        optimize : bool, default True
            If True, apply query optimization passes before execution.
            Set to False for debugging or when optimization causes issues.
        strict : bool, default False
            If True, error if any engine fallback or implicit conversion
            is required. All operations must run in their preferred engine
            without crossing boundaries.
        engine : {"auto", "arrow", "numpy"}, default "auto"
            Preferred execution engine. "auto" selects based on data types
            and operation support.
        use_physical_planner : bool, default False
            If True, use the physical planner for execution. The physical
            planner converts the logical plan to a physical plan with
            concrete execution strategies (algorithm selection, backend
            choice). If False, use direct eager evaluation with pandas
            native operations.
        streaming : bool, default False
            If True, return an iterator of DataFrames (one per batch).
            This enables memory-efficient processing of large datasets.
            Requires use_physical_planner=True.

            Streaming execution is useful for:
            - Processing data larger than available RAM
            - Early termination with head()/limit() operations
            - Better cache locality with batch-by-batch processing
        batch_size : int, default 65536
            Batch size for streaming execution. Default is 64K rows
            which is typically L3 cache-friendly.
        preserve_index : bool, default False
            If True, reconstruct the original DataFrame index from the
            source data. If False (default), return DataFrame with
            RangeIndex. The default matches Polars-style behavior where
            lazy execution returns positional indexes.
        spill_config : SpillConfig or None, default None
            Configuration for disk spilling when memory pressure is detected.
            When enabled, intermediate results from memory-intensive operations
            (sort, join, groupby) can be spilled to disk. This allows processing
            datasets larger than available RAM. Requires use_physical_planner=True.

            Example::

                from pandas.lazy.backends.spill import SpillConfig

                config = SpillConfig(
                    enabled=True,
                    threshold_mb=2048,  # Spill when >2GB tracked
                    operator_budget_mb=512,  # Per-operator budget
                    spill_dir="/tmp/spill",  # Optional custom directory
                )
                result = ldf.collect(
                    use_physical_planner=True,
                    spill_config=config,
                )

        Returns
        -------
        DataFrame or Iterator[DataFrame]
            If streaming=False, returns the materialized result as a
            single DataFrame. If streaming=True, returns an iterator
            yielding DataFrames (one per batch).

        Raises
        ------
        StrictModeError
            If strict=True and fallback/conversion would be required.
        ValueError
            If streaming=True but use_physical_planner=False.
            If spill_config is provided but use_physical_planner=False.

        Examples
        --------
        >>> result = ldf.collect()
        >>> result = ldf.collect(optimize=False)  # Skip optimization
        >>> result = ldf.collect(use_physical_planner=True)  # Use physical planner

        >>> # Streaming execution for large datasets
        >>> for batch_df in ldf.collect(streaming=True, use_physical_planner=True):
        ...     process(batch_df)

        >>> # Early termination with head()
        >>> result = ldf.head(100).collect(use_physical_planner=True)

        >>> # Preserve original index
        >>> result = ldf.collect(preserve_index=True)

        >>> # Out-of-core processing with disk spilling
        >>> from pandas.lazy.backends.spill import SpillConfig
        >>> config = SpillConfig(enabled=True, threshold_mb=2048)
        >>> result = ldf.collect(use_physical_planner=True, spill_config=config)
        """
        # Validate spill_config requires physical planner
        if spill_config is not None and not use_physical_planner:
            raise ValueError("spill_config requires use_physical_planner=True")

        if streaming:
            if not use_physical_planner:
                raise ValueError("streaming=True requires use_physical_planner=True")
            return self._collect_streaming(
                optimize=optimize,
                strict=strict,
                engine=engine,
                batch_size=batch_size,
                preserve_index=preserve_index,
                spill_config=spill_config,
            )

        # Trivial-plan fast path: head()/tail()/slice over pure column
        # selections of an in-memory DataFrame slices the source directly
        # (microseconds) instead of running the planner and engine
        # (milliseconds). Polars answers head(10) in ~10us; without this
        # short-circuit we paid full pipeline cost for it.
        fast = self._try_trivial_slice(preserve_index)
        if fast is not None:
            return fast

        if use_physical_planner:
            return self._collect_physical(
                optimize=optimize,
                strict=strict,
                engine=engine,
                preserve_index=preserve_index,
                spill_config=spill_config,
            )
        else:
            return self._collect_eager(optimize=optimize, preserve_index=preserve_index)

    def _try_trivial_slice(self, preserve_index: bool) -> DataFrame | None:
        """
        Fast path for ``Limit`` over pure column selections over an
        in-memory ``DataFrameSource``: slice the source directly.

        Returns None when the plan is anything more than that (filters,
        computed expressions, file scans, joins, ...), in which case the
        normal execution paths run. Safe under Copy-on-Write: returned
        slices share buffers with the source and copy lazily on mutation.
        """
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )
        from pandas.lazy.plan import (
            DataFrameSource,
            Limit,
            Project,
        )

        plan = self._plan
        if not isinstance(plan, Limit):
            return None

        # Walk down through projections that only reference/rename
        # columns, composing output-name -> source-column mapping.
        # None mapping means identity (all source columns, source order).
        mapping: dict[str, str] | None = None
        node = plan.input
        while isinstance(node, Project):
            this_map: dict[str, str] = {}
            for expr in node.exprs:
                ir = expr._ir
                if isinstance(ir, Alias) and isinstance(ir.arg, FieldRef):
                    this_map[ir.name] = ir.arg.name
                elif isinstance(ir, FieldRef):
                    this_map[ir.name] = ir.name
                else:
                    return None  # computed expression - not trivial
            if mapping is None:
                mapping = this_map
            else:
                try:
                    mapping = {out: this_map[src] for out, src in mapping.items()}
                except KeyError:
                    return None
            node = node.input

        if not isinstance(node, DataFrameSource):
            return None

        df = node.df
        n, offset = plan.n, plan.offset
        if offset == -1:
            sliced = df.tail(n)
        elif offset > 0:
            sliced = df.iloc[offset : offset + n]
        else:
            sliced = df.head(n)

        if mapping is not None:
            sliced = sliced[list(mapping.values())]
            if list(mapping.keys()) != list(mapping.values()):
                sliced = sliced.set_axis(list(mapping.keys()), axis=1)

        if not preserve_index:
            sliced = sliced.reset_index(drop=True)
        return sliced

    def _collect_eager(
        self, optimize: bool = True, preserve_index: bool = False
    ) -> DataFrame:
        """
        Eager execution that evaluates the plan.

        Parameters
        ----------
        optimize : bool, default True
            If True, apply query optimization passes before execution.
        preserve_index : bool, default False
            If True, preserve the original DataFrame index.
        """
        import numpy as np

        from pandas import (
            DataFrame as PdDataFrame,
            Series,
        )
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.expr import extract_output_name
        from pandas.lazy.plan import (
            Aggregate,
            Concat,
            Convert,
            DataFrameSource,
            Distinct,
            Filter,
            Join,
            Limit,
            Project,
            ResetIndex,
            SetIndex,
            Sort,
            TopK,
        )

        # Apply optimization if requested
        plan = self._get_optimized_plan() if optimize else self._plan

        # Whether the user explicitly set an index via set_index(). An
        # explicit index is always kept in the result, regardless of
        # preserve_index (mirrors ExecutionContext.user_set_index in the
        # physical planner).
        user_set_index = False

        def evaluate(plan: LogicalPlan) -> DataFrame:
            """Evaluate a plan node."""
            nonlocal user_set_index

            if isinstance(plan, DataFrameSource):
                return plan.df.copy()

            elif isinstance(plan, Project):
                input_df = evaluate(plan.input)
                evaluator = Evaluator(input_df)
                result = {}
                for expr in plan.exprs:
                    name = extract_output_name(expr)
                    value = evaluator.evaluate(expr._ir)
                    if isinstance(value, Series) and not value.index.equals(
                        input_df.index
                    ):
                        # Expression results are positional; realign to the
                        # input index so DataFrame construction does not
                        # misalign rows (e.g. row_index on non-RangeIndex).
                        value = value.set_axis(input_df.index)
                    result[name] = value
                # Pass the input index explicitly so all-scalar projections
                # broadcast to the input length instead of raising.
                return PdDataFrame(result, index=input_df.index)

            elif isinstance(plan, Filter):
                input_df = evaluate(plan.input)
                evaluator = Evaluator(input_df)
                mask = evaluator.evaluate(plan.predicate._ir)
                # Handle scalar boolean masks (e.g., lit(True) or lit(False))
                if isinstance(mask, (bool, np.bool_)):
                    if mask:
                        return input_df.copy()
                    else:
                        return input_df.iloc[:0].copy()
                # Keep the input index labels; the final index policy
                # (RangeIndex vs preserved) is applied once at the end.
                return input_df.loc[mask]

            elif isinstance(plan, Aggregate):
                input_df = evaluate(plan.input)
                return _evaluate_aggregate(
                    input_df, plan, preserve_index=preserve_index
                )

            elif isinstance(plan, Sort):
                input_df = evaluate(plan.input)
                return _evaluate_sort(input_df, plan)

            elif isinstance(plan, Limit):
                input_df = evaluate(plan.input)
                return _evaluate_limit(input_df, plan)

            elif isinstance(plan, Distinct):
                input_df = evaluate(plan.input)
                return _evaluate_distinct(input_df, plan)

            elif isinstance(plan, Join):
                left_df = evaluate(plan.left)
                right_df = evaluate(plan.right)
                return _evaluate_join(
                    left_df, right_df, plan, preserve_index=preserve_index
                )

            elif isinstance(plan, Concat):
                # Evaluate all inputs and concatenate
                import pandas as pd

                dfs = [evaluate(inp) for inp in plan.inputs]
                return pd.concat(dfs, ignore_index=not preserve_index)

            elif isinstance(plan, Convert):
                # Convert node handles backend conversion (Arrow <-> NumPy)
                # For now, this is a no-op pass-through since pandas handles
                # conversion automatically. In the future, this could be used
                # to optimize backend-specific operations.
                return evaluate(plan.input)

            elif isinstance(plan, TopK):
                input_df = evaluate(plan.input)
                return _evaluate_topk(input_df, plan)

            elif isinstance(plan, SetIndex):
                input_df = evaluate(plan.input)
                user_set_index = True
                keys = list(plan.keys)
                if len(keys) == 1:
                    result = input_df.set_index(keys[0], drop=plan.drop)
                else:
                    result = input_df.set_index(keys, drop=plan.drop)
                return result

            elif isinstance(plan, ResetIndex):
                input_df = evaluate(plan.input)
                user_set_index = False
                return input_df.reset_index(drop=plan.drop)

            else:
                raise NotImplementedError(f"Unknown plan type: {type(plan)}")

        result = evaluate(plan)

        # Apply the index contract: an explicit set_index() is always kept;
        # otherwise preserve_index=True keeps source index labels and the
        # default returns a positional RangeIndex (matching the physical
        # planner).
        if not user_set_index and not preserve_index:
            result = result.reset_index(drop=True)
        return result

    def _collect_physical(
        self,
        optimize: bool = True,
        strict: bool = False,
        engine: Literal["auto", "arrow", "numpy"] = "auto",
        preserve_index: bool = False,
        spill_config: SpillConfig | None = None,
    ) -> DataFrame:
        """
        Execute the query using the physical planner.

        This method converts the logical plan to a physical plan and
        executes it. The physical planner makes execution decisions
        based on data characteristics and user preferences.

        Parameters
        ----------
        optimize : bool, default True
            If True, apply query optimization passes before execution.
        strict : bool, default False
            If True, fail on backend fallbacks.
        engine : {"auto", "arrow", "numpy"}, default "auto"
            Preferred execution backend.
        preserve_index : bool, default False
            If True, preserve the original DataFrame index.
        spill_config : SpillConfig or None, default None
            Configuration for disk spilling under memory pressure.

        Returns
        -------
        DataFrame
            The execution result.
        """
        from pandas.lazy.physical import (
            PhysicalPlanner,
            execute_physical_plan,
        )

        # Apply optimization if requested
        plan = self._get_optimized_plan() if optimize else self._plan

        # Convert logical plan to physical plan
        planner = PhysicalPlanner(preferred_backend=engine)
        physical_plan = planner.plan(plan)

        # Execute physical plan
        return execute_physical_plan(
            physical_plan,
            preferred_backend=engine,
            strict=strict,
            preserve_index=preserve_index,
            spill_config=spill_config,
        )

    def _collect_streaming(
        self,
        optimize: bool = True,
        strict: bool = False,
        engine: Literal["auto", "arrow", "numpy"] = "auto",
        batch_size: int = 65536,
        preserve_index: bool = False,
        spill_config: SpillConfig | None = None,
    ) -> Iterator[DataFrame]:
        """
        Execute the query and stream results as batches of DataFrames.

        This method yields DataFrames one batch at a time, enabling
        memory-efficient processing of large datasets. The streaming
        execution also supports early termination for head()/limit()
        operations.

        Parameters
        ----------
        optimize : bool, default True
            If True, apply query optimization passes before execution.
        strict : bool, default False
            If True, fail on backend fallbacks.
        engine : {"auto", "arrow", "numpy"}, default "auto"
            Preferred execution backend.
        batch_size : int, default 65536
            Number of rows per batch.
        preserve_index : bool, default False
            If True, preserve the original DataFrame index.
        spill_config : SpillConfig or None, default None
            Configuration for disk spilling under memory pressure.

        Yields
        ------
        DataFrame
            DataFrames, one per batch from the execution.
        """
        import pandas as pd
        from pandas.lazy.backends.convert import arrays_to_dataframe
        from pandas.lazy.physical import (
            ExecutionContext,
            PhysicalPlanner,
        )

        # Apply optimization if requested
        plan = self._get_optimized_plan() if optimize else self._plan

        # Convert logical plan to physical plan
        planner = PhysicalPlanner(preferred_backend=engine)
        physical_plan = planner.plan(plan)

        # Check if adaptive thresholds are enabled
        adaptive_enabled = pd.get_option("compute.lazy.adaptive_thresholds")

        # Create execution context with streaming configuration
        context = ExecutionContext(
            preferred_backend=engine,
            strict=strict,
            batch_size=batch_size,
            streaming_enabled=True,
            adaptive_thresholds=adaptive_enabled,
            _spill_config=spill_config,
        )

        # Stream batches
        for batch in physical_plan.execute_batches(context):
            yield arrays_to_dataframe(
                batch,
                index_names=context.index_names,
                index_is_multi=context.index_is_multi,
                preserve_index=preserve_index,
            )

    def explain(
        self,
        *,
        optimized: bool = True,
        physical: bool = False,
        format: Literal["text", "tree", "json"] = "text",
    ) -> str:
        """
        Show the query plan.

        Parameters
        ----------
        optimized : bool, default True
            Show optimized plan. If False, shows raw logical plan.
        physical : bool, default False
            Show physical execution plan instead of logical plan.
            Physical plans show explicit materialization boundaries,
            backend choices, and algorithm selection.
        format : {"text", "tree", "json"}, default "text"
            Output format.

        Returns
        -------
        str
            Human-readable plan representation.

        Examples
        --------
        >>> print(ldf.explain())  # Show optimized logical plan
        >>> print(ldf.explain(optimized=False))  # Show unoptimized plan
        >>> print(ldf.explain(physical=True))  # Show physical plan
        """
        if physical:
            return self._explain_physical(format=format)
        elif format == "text":
            return self._explain_text(optimized=optimized)
        elif format == "tree":
            return self._explain_tree(optimized=optimized)
        elif format == "json":
            return self._explain_json(optimized=optimized)
        else:
            raise ValueError(
                f"Unknown format: {format!r}. Use 'text', 'tree', or 'json'."
            )

    def _get_plan_for_explain(
        self, optimized: bool
    ) -> tuple[LogicalPlan, Literal["optimized", "unoptimized"]]:
        """Get the plan to explain, optionally optimized."""
        if optimized:
            plan = self._get_optimized_plan()
            plan_type: Literal["optimized", "unoptimized"] = "optimized"
        else:
            plan = self._plan
            plan_type = "unoptimized"
        return plan, plan_type

    def _explain_text(self, optimized: bool = True) -> str:
        """Text format explain output with simple indentation."""
        plan, plan_type = self._get_plan_for_explain(optimized)

        lines = []
        lines.append("=" * 50)
        lines.append(f"LAZY PANDAS QUERY PLAN ({plan_type})")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Output columns: {self.columns}")
        lines.append("")
        lines.append("Plan tree:")
        lines.append("-" * 30)

        def format_plan(plan_node: LogicalPlan, indent: int = 0) -> None:
            prefix = "  " * indent
            lines.append(f"{prefix}{plan_node!r}")
            for child in plan_node.children():
                format_plan(child, indent + 1)

        format_plan(plan)

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def _explain_tree(self, optimized: bool = True) -> str:
        """Tree format explain output with box-drawing characters."""
        plan, plan_type = self._get_plan_for_explain(optimized)

        lines = []
        lines.append(f"LAZY PANDAS QUERY PLAN ({plan_type})")
        lines.append(f"Output columns: {self.columns}")
        lines.append("")

        def format_tree(
            plan_node: LogicalPlan, prefix: str = "", is_last: bool = True
        ) -> None:
            # Current node connector
            connector = "└─ " if is_last else "├─ "
            lines.append(f"{prefix}{connector}{plan_node!r}")

            # Prepare prefix for children
            child_prefix = prefix + ("   " if is_last else "│  ")

            children = plan_node.children()
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                format_tree(child, child_prefix, is_last_child)

        # Start with root (no prefix, always "last" since it's root)
        lines.append(f"┌─ {plan!r}")
        children = plan.children()
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            format_tree(child, "", is_last_child)

        return "\n".join(lines)

    def _explain_json(self, optimized: bool = True) -> str:
        """JSON format explain output for programmatic use."""
        import json

        plan, plan_type = self._get_plan_for_explain(optimized)

        def plan_to_dict(plan_node: LogicalPlan) -> dict:
            node_dict: dict = {
                "type": type(plan_node).__name__,
            }

            # Add node-specific attributes based on type
            if hasattr(plan_node, "exprs") and plan_node.exprs:
                node_dict["expressions"] = [repr(e) for e in plan_node.exprs]
            if hasattr(plan_node, "predicate") and plan_node.predicate is not None:
                node_dict["predicate"] = repr(plan_node.predicate)
            if hasattr(plan_node, "keys") and plan_node.keys:
                node_dict["keys"] = list(plan_node.keys)
            if hasattr(plan_node, "group_by") and plan_node.group_by:
                node_dict["group_by"] = [repr(g) for g in plan_node.group_by]
            if hasattr(plan_node, "agg_exprs") and plan_node.agg_exprs:
                node_dict["aggregations"] = [repr(a) for a in plan_node.agg_exprs]
            if hasattr(plan_node, "by") and plan_node.by:
                node_dict["sort_by"] = [repr(b) for b in plan_node.by]
            if hasattr(plan_node, "source"):
                node_dict["source"] = repr(plan_node.source)
            if hasattr(plan_node, "left") and hasattr(plan_node, "right"):
                node_dict["left"] = plan_to_dict(plan_node.left)
                node_dict["right"] = plan_to_dict(plan_node.right)
                if hasattr(plan_node, "on"):
                    node_dict["on"] = list(plan_node.on) if plan_node.on else None
                if hasattr(plan_node, "how"):
                    node_dict["how"] = plan_node.how

            # Process children (except join which has left/right)
            children = plan_node.children()
            if children and not (
                hasattr(plan_node, "left") and hasattr(plan_node, "right")
            ):
                if len(children) == 1:
                    node_dict["input"] = plan_to_dict(children[0])
                else:
                    node_dict["inputs"] = [plan_to_dict(c) for c in children]

            return node_dict

        result = {
            "plan_type": plan_type,
            "output_columns": list(self.columns),
            "plan": plan_to_dict(plan),
        }

        return json.dumps(result, indent=2)

    def _explain_physical(
        self, format: Literal["text", "tree", "json"] = "text"
    ) -> str:
        """
        Explain the physical execution plan.

        Shows the physical plan with explicit materialization boundaries,
        backend choices, and algorithm selection.
        """
        from pandas.lazy.physical import (
            PhysicalMaterialize,
            PhysicalPlanner,
        )

        # Get optimized logical plan and convert to physical
        plan = self._get_optimized_plan()
        planner = PhysicalPlanner()
        physical_plan = planner.plan(plan)

        lines = []
        lines.append("=" * 60)
        lines.append("LAZY PANDAS PHYSICAL PLAN")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Output columns: {self.columns}")
        lines.append("")
        lines.append("Physical plan tree:")
        lines.append("  [BREAKER] = pipeline breaker (materializes input)")
        lines.append("-" * 40)

        def format_node(node, indent: int = 0) -> None:
            prefix = "  " * indent

            # Build node description
            node_name = type(node).__name__

            # Special handling for PhysicalMaterialize to show boundary
            if isinstance(node, PhysicalMaterialize):
                lines.append(f"{prefix}[BREAKER] Materialize (reason={node.reason})")
            elif node.is_pipeline_breaker:
                lines.append(f"{prefix}[BREAKER] {node_name}")
            else:
                # Add streaming indicator for nodes that support it
                streaming = " [streaming]" if node.supports_streaming else ""
                lines.append(f"{prefix}{node_name}{streaming}")

            # Recurse to children
            for child in node.children():
                format_node(child, indent + 1)

        format_node(physical_plan)

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def row_group_stats(self) -> DataFrame | None:
        """
        Get row group statistics for the underlying Parquet file(s).

        This method returns metadata about row groups in Parquet files,
        including min/max values for each column. This information helps
        understand how predicate pushdown will filter row groups.

        Returns
        -------
        DataFrame or None
            DataFrame with columns: file, row_group, column, min, max,
            null_count, num_rows. Returns None if the source is not a
            Parquet file or glob pattern.

        Notes
        -----
        Row group statistics are used by PyArrow to skip reading row
        groups that cannot match filter predicates. Understanding these
        statistics helps write efficient queries.

        Examples
        --------
        >>> ldf = pd.lazy.scan("data.parquet")
        >>> stats = ldf.row_group_stats()
        >>> print(stats)  # Shows min/max for each column per row group
        """

        from pandas.lazy.plan import ParquetSource

        # Find the source node
        source = self._find_source_node(self._plan)
        if source is None or not isinstance(source, ParquetSource):
            return None

        return _get_parquet_row_group_stats(source.path)

    def _find_source_node(self, plan: LogicalPlan):
        """Recursively find the source node in the plan tree."""
        from pandas.lazy.plan import ParquetSource

        if isinstance(plan, ParquetSource):
            return plan

        children = plan.children()
        if children:
            return self._find_source_node(children[0])
        return None

    def __repr__(self) -> str:
        return f"LazyDataFrame(columns={self.columns})"


def _get_parquet_row_group_stats(path: str) -> DataFrame:
    """
    Extract row group statistics from a Parquet file.

    Parameters
    ----------
    path : str
        Path to Parquet file or glob pattern.

    Returns
    -------
    DataFrame
        Statistics for each row group and column.
    """
    import glob

    import pyarrow.parquet as pq

    from pandas import DataFrame as PdDataFrame

    records: list[dict] = []

    # Resolve glob patterns
    if "*" in path or "?" in path:
        files = sorted(glob.glob(path))
    else:
        files = [path]

    for file_path in files:
        pf = pq.ParquetFile(file_path)
        metadata = pf.metadata

        for rg_idx in range(metadata.num_row_groups):
            rg = metadata.row_group(rg_idx)

            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                col_path = col.path_in_schema

                stats = col.statistics
                if stats is not None:
                    records.append(
                        {
                            "file": file_path,
                            "row_group": rg_idx,
                            "column": col_path,
                            "min": stats.min if stats.has_min_max else None,
                            "max": stats.max if stats.has_min_max else None,
                            "null_count": stats.null_count,
                            "num_rows": rg.num_rows,
                        }
                    )
                else:
                    records.append(
                        {
                            "file": file_path,
                            "row_group": rg_idx,
                            "column": col_path,
                            "min": None,
                            "max": None,
                            "null_count": None,
                            "num_rows": rg.num_rows,
                        }
                    )

    return PdDataFrame(records)


class LazyGroupBy:
    """
    Grouped LazyDataFrame for aggregation operations.

    This class is returned by LazyDataFrame.group_by() and provides
    the agg() method for specifying aggregation expressions.

    Examples
    --------
    >>> from pandas.lazy import col
    >>> ldf.group_by("category").agg(col("value").sum().alias("total"))
    """

    __slots__ = ("_group_by", "_plan", "_schema")

    def __init__(
        self,
        plan: LogicalPlan,
        group_by: tuple[Expr, ...],
        schema: Schema,
    ) -> None:
        self._plan = plan
        self._group_by = group_by
        self._schema = schema

    def agg(self, *exprs: Expr) -> LazyDataFrame:
        """
        Specify aggregation expressions.

        Parameters
        ----------
        *exprs : Expr
            Aggregation expressions. Each expression should be an
            aggregation (sum, mean, etc.) with an alias for the output
            column name.

        Returns
        -------
        LazyDataFrame
            A LazyDataFrame with the aggregation results.

        Examples
        --------
        >>> from pandas.lazy import col
        >>> ldf.group_by("category").agg(
        ...     col("value").sum().alias("total"),
        ...     col("value").mean().alias("average"),
        ... )
        """
        from pandas.lazy.expr import Expr
        from pandas.lazy.plan import Aggregate

        if not exprs:
            raise ValueError("agg() requires at least one expression")

        for e in exprs:
            if not isinstance(e, Expr):
                raise TypeError(
                    f"Expected Expr, got {type(e).__name__}. "
                    f"Use col('name').sum().alias('result_name')."
                )

        new_plan = Aggregate(self._plan, self._group_by, exprs)
        return LazyDataFrame(new_plan, new_plan.resolve_schema())

    def __repr__(self) -> str:
        from pandas.lazy.expr import extract_output_name

        try:
            group_names = [extract_output_name(e) for e in self._group_by]
        except ValueError:
            group_names = [repr(e) for e in self._group_by]
        return f"LazyGroupBy(by={group_names})"


def _evaluate_aggregate(
    df: DataFrame, plan: Aggregate, *, preserve_index: bool = False
) -> DataFrame:
    """
    Evaluate an Aggregate plan node.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    plan : Aggregate
        The aggregation plan.
    preserve_index : bool, default False
        If True, group keys become the result index instead of columns.

    Returns
    -------
    DataFrame
        The aggregated result.
    """
    from pandas import DataFrame as PdDataFrame
    from pandas.lazy.eval import Evaluator
    from pandas.lazy.expr import extract_output_name
    from pandas.lazy.ir import (
        Alias,
        Call,
        FieldRef,
    )

    # Get group-by column names
    group_names = [extract_output_name(e) for e in plan.group_by]

    if not group_names:
        # Global aggregation (no group-by)
        evaluator = Evaluator(df)
        result = {}
        for expr in plan.agg_exprs:
            name = extract_output_name(expr)
            value = evaluator.evaluate(expr._ir)
            result[name] = [value]  # Wrap scalar in list for DataFrame
        return PdDataFrame(result)

    # Group-by aggregation using pandas native groupby for performance
    # Build aggregation spec: {input_col: [(pandas_func, output_name), ...]}
    agg_spec: dict[str, list[tuple[str | callable, str]]] = {}
    result_names: list[tuple[str, str]] = []  # (input_col, output_name)

    for expr in plan.agg_exprs:
        output_name = extract_output_name(expr)
        ir = expr._ir

        # Unwrap Alias if present
        if isinstance(ir, Alias):
            ir = ir.arg

        if isinstance(ir, Call) and ir.is_aggregate:
            # Get the column being aggregated
            if ir.args and isinstance(ir.args[0], FieldRef):
                col_name = ir.args[0].name
            else:
                raise ValueError(f"Aggregation must be on a column, got: {ir.args[0]}")

            # Map function name to pandas agg function
            func_map = {
                "sum": "sum",
                "mean": "mean",
                "min": "min",
                "max": "max",
                "count": "count",
                "std": "std",
                "var": "var",
                "first": "first",
                "last": "last",
                "n_unique": "nunique",
            }

            if ir.function not in func_map:
                raise NotImplementedError(
                    f"Aggregation function not supported: {ir.function}"
                )

            pandas_func: str | callable = func_map[ir.function]

            # Handle ddof for std/var
            if ir.function in ("std", "var") and ir.kwargs.get("ddof", 1) != 1:
                ddof = ir.kwargs["ddof"]
                # Use lambda to capture ddof
                if ir.function == "std":
                    pandas_func = lambda x, d=ddof: x.std(ddof=d)
                else:
                    pandas_func = lambda x, d=ddof: x.var(ddof=d)

            if col_name not in agg_spec:
                agg_spec[col_name] = []
            agg_spec[col_name].append((pandas_func, output_name))
            result_names.append((col_name, output_name))
        else:
            raise ValueError(f"Expected aggregation expression, got: {ir}")

    # Use pandas native groupby for efficient hash-based grouping
    grouped = df.groupby(group_names, sort=False)

    # Use pandas named aggregation for clean output names
    # Format: output_name=(col, func)
    named_agg = {}
    for col_name, aggs in agg_spec.items():
        for pandas_func, output_name in aggs:
            named_agg[output_name] = (col_name, pandas_func)

    # Perform the aggregation with named output columns
    result = grouped.agg(**named_agg)

    if preserve_index:
        # Group keys become the index (pandas-style, matches physical planner)
        return result[[name for _, name in result_names]]

    # Reset index to get group columns as regular columns
    result = result.reset_index()

    # Ensure correct column order
    ordered_cols = group_names + [name for _, name in result_names]
    return result[ordered_cols]


def _evaluate_sort(df: DataFrame, plan: Sort) -> DataFrame:
    """
    Evaluate a Sort plan node.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    plan : Sort
        The sort plan.

    Returns
    -------
    DataFrame
        The sorted result.
    """
    from pandas.lazy.eval import Evaluator
    from pandas.lazy.ir import FieldRef

    # For simple column references, use column names directly
    # For complex expressions, evaluate them first
    sort_keys = []
    temp_cols = []

    evaluator = Evaluator(df)

    for i, expr in enumerate(plan.by):
        ir = expr._ir
        if isinstance(ir, FieldRef):
            # Simple column reference
            sort_keys.append(ir.name)
        else:
            # Complex expression - evaluate and add as temp column
            temp_name = f"__sort_key_{i}__"
            df = df.copy()
            df[temp_name] = evaluator.evaluate(ir)
            sort_keys.append(temp_name)
            temp_cols.append(temp_name)

    # Perform sort. Index labels are kept; the final index policy is
    # applied once at the end of _collect_eager.
    ascending = [not d for d in plan.descending]
    # kind="stable" so tie order matches the physical engine, whose
    # NumPy and Arrow sort paths are both stable
    result = df.sort_values(by=sort_keys, ascending=ascending, kind="stable")

    # Remove temporary columns
    if temp_cols:
        result = result.drop(columns=temp_cols)

    return result


def _evaluate_limit(df: DataFrame, plan: Limit) -> DataFrame:
    """
    Evaluate a Limit plan node.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    plan : Limit
        The limit plan.

    Returns
    -------
    DataFrame
        The limited result.
    """
    if plan.offset == -1:
        # Tail operation
        return df.tail(plan.n)
    elif plan.offset > 0:
        # Skip + limit
        return df.iloc[plan.offset : plan.offset + plan.n]
    else:
        # Simple head/limit
        return df.head(plan.n)


def _evaluate_distinct(df: DataFrame, plan) -> DataFrame:
    """
    Evaluate a Distinct plan node.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    plan : Distinct
        The distinct plan.

    Returns
    -------
    DataFrame
        The deduplicated result.
    """
    subset = list(plan.subset) if plan.subset else None
    return df.drop_duplicates(subset=subset)


def _evaluate_topk(df: DataFrame, plan) -> DataFrame:
    """
    Evaluate a TopK plan node.

    TopK is an optimization of Sort + Limit that uses nsmallest/nlargest
    for efficient partial sorting when k is small relative to n.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    plan : TopK
        The TopK plan.

    Returns
    -------
    DataFrame
        The top K rows by sort criteria.
    """
    from pandas.lazy.eval import Evaluator
    from pandas.lazy.ir import FieldRef

    # For simple column references, use column names directly
    # For complex expressions, evaluate them first
    sort_keys = []
    temp_cols = []

    evaluator = Evaluator(df)

    for i, expr in enumerate(plan.by):
        ir = expr._ir
        if isinstance(ir, FieldRef):
            # Simple column reference
            sort_keys.append(ir.name)
        else:
            # Complex expression - evaluate and add as temp column
            temp_name = f"__sort_key_{i}__"
            df = df.copy()
            df[temp_name] = evaluator.evaluate(ir)
            sort_keys.append(temp_name)
            temp_cols.append(temp_name)

    # Use nlargest/nsmallest for efficiency when possible
    # This is more efficient than full sort + head for small k
    k = min(plan.k, len(df))

    if k == 0:
        # Return empty DataFrame with same columns
        result = df.head(0)
    elif len(sort_keys) == 1 and len(plan.descending) == 1:
        # Single sort key - can use nlargest/nsmallest
        key = sort_keys[0]
        if plan.descending[0]:
            result = df.nlargest(k, key)
        else:
            result = df.nsmallest(k, key)
    else:
        # Multiple keys or mixed directions - fall back to sort + head
        ascending = [not d for d in plan.descending]
        result = df.sort_values(by=sort_keys, ascending=ascending, kind="stable").head(
            k
        )

    # Remove temporary columns
    if temp_cols:
        result = result.drop(columns=temp_cols)

    return result


def _evaluate_join(
    left_df: DataFrame, right_df: DataFrame, plan, *, preserve_index: bool = False
) -> DataFrame:
    """
    Evaluate a Join plan node.

    Parameters
    ----------
    left_df : DataFrame
        The left input DataFrame.
    right_df : DataFrame
        The right input DataFrame.
    plan : Join
        The join plan.
    preserve_index : bool, default False
        If True, the left side's index is carried through the join
        (matching the physical planner).

    Returns
    -------
    DataFrame
        The joined result.
    """
    index_cols: list[str] = []
    index_names: list = []
    if preserve_index:
        # pd.merge discards the index, so carry the left index through
        # the merge as temporary columns and restore it afterwards.
        index_names = list(left_df.index.names)
        index_cols = [f"__left_index_{i}__" for i in range(left_df.index.nlevels)]
        left_df = left_df.copy()
        for i, tmp_name in enumerate(index_cols):
            left_df[tmp_name] = left_df.index.get_level_values(i)

    if plan.on is not None:
        # Same column name(s) in both DataFrames
        result = left_df.merge(
            right_df,
            on=list(plan.on),
            how=plan.how,
            suffixes=plan.suffix,
        )
    elif plan.left_on is not None and plan.right_on is not None:
        # Different column names
        result = left_df.merge(
            right_df,
            left_on=list(plan.left_on),
            right_on=list(plan.right_on),
            how=plan.how,
            suffixes=plan.suffix,
        )
    elif plan.how == "cross":
        # Cross join
        result = left_df.merge(right_df, how="cross", suffixes=plan.suffix)
    else:
        raise ValueError("Invalid join parameters")

    if preserve_index:
        from pandas import (
            Index,
            MultiIndex,
        )

        if len(index_cols) == 1:
            result.index = Index(result[index_cols[0]], name=index_names[0])
        else:
            result.index = MultiIndex.from_arrays(
                [result[c] for c in index_cols], names=index_names
            )
        result = result.drop(columns=index_cols)
        return result

    return result.reset_index(drop=True)


def concat(dfs: list[LazyDataFrame] | tuple[LazyDataFrame, ...]) -> LazyDataFrame:
    """
    Concatenate multiple LazyDataFrames vertically (union all).

    This creates a lazy Concat plan node that will stream batches
    from each input sequentially during execution.

    Parameters
    ----------
    dfs : list of LazyDataFrame or tuple of LazyDataFrame
        The LazyDataFrames to concatenate. All must have compatible schemas
        (same column names).

    Returns
    -------
    LazyDataFrame
        A new LazyDataFrame representing the concatenation of all inputs.

    Raises
    ------
    ValueError
        If the inputs do not all have the same column names, or if
        ``dfs`` is empty.
    TypeError
        If any input is not a LazyDataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    >>> from pandas.lazy import concat
    >>> lf = concat([df1.select(), df2.select()])
    >>> result = lf.collect()
    >>> result
       a  b
    0  1  3
    1  2  4
    2  5  7
    3  6  8

    Works with Parquet scans for multi-file processing:

    >>> from pandas.lazy import scan, concat
    >>> lf1 = scan("data/part1.parquet")
    >>> lf2 = scan("data/part2.parquet")
    >>> combined = concat([lf1, lf2])
    >>> result = combined.filter(col("value") > 100).collect()
    """
    from pandas.lazy.plan import Concat as ConcatPlan

    if not dfs:
        raise ValueError("concat requires at least one LazyDataFrame")

    if not all(isinstance(df, LazyDataFrame) for df in dfs):
        raise TypeError("All inputs to concat must be LazyDataFrame instances")

    plans = tuple(df._plan for df in dfs)
    concat_plan = ConcatPlan(plans)
    # Use schema from first input (all should be compatible)
    schema = concat_plan.resolve_schema()
    return LazyDataFrame(concat_plan, schema)
