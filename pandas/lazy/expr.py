"""
Expression building API for lazy pandas.

Provides col(), lit(), and the Expr class for building expression trees.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.lazy.ir import (
    Alias,
    Call,
    FieldRef,
    IRNode,
    Literal,
)

if TYPE_CHECKING:
    from pandas.lazy.types import LazyDtype


def _to_expr(value: Any) -> Expr:
    """Convert a value to an Expr, wrapping non-Expr values as Literal."""
    if isinstance(value, Expr):
        return value
    return Expr(Literal(value))


def col(name: str) -> Expr:
    """
    Reference a column by name.

    Parameters
    ----------
    name : str
        The name of the column to reference.

    Returns
    -------
    Expr
        An expression referencing the column.

    Examples
    --------
    >>> from pandas.lazy import col
    >>> expr = col("a")
    >>> expr
    Expr(FieldRef('a'))
    """
    if not isinstance(name, str):
        raise TypeError(f"Column name must be a string, got {type(name).__name__}")
    return Expr(FieldRef(name))


def lit(value: Any, dtype: LazyDtype | None = None) -> Expr:
    """
    Create a literal value expression.

    Parameters
    ----------
    value : Any
        The literal value.
    dtype : LazyDtype, optional
        The type of the literal. If not provided, it will be inferred.

    Returns
    -------
    Expr
        An expression representing the literal value.

    Examples
    --------
    >>> from pandas.lazy import lit
    >>> expr = lit(42)
    >>> expr
    Expr(Literal(42))
    """
    return Expr(Literal(value, dtype))


class Expr:
    """
    Expression builder with operator overloading.

    Wraps an IR node and provides methods/operators for building
    expression trees. This class is immutable - all operations return
    new Expr instances.

    Examples
    --------
    >>> from pandas.lazy import col, lit
    >>> expr = col("a") + col("b")
    >>> expr = col("a") > 0
    >>> expr = col("a").alias("new_name")
    """

    __slots__ = ("_node",)

    def __init__(self, node: IRNode) -> None:
        self._node = node

    @property
    def _ir(self) -> IRNode:
        """Return the underlying IR node."""
        return self._node

    def alias(self, name: str) -> Expr:
        """
        Give this expression an output name.

        Parameters
        ----------
        name : str
            The name for the output column.

        Returns
        -------
        Expr
            A new expression with the alias.

        Examples
        --------
        >>> from pandas.lazy import col
        >>> expr = col("a").alias("new_a")
        """
        if not isinstance(name, str):
            raise TypeError(f"Alias name must be a string, got {type(name).__name__}")
        return Expr(Alias(self._node, name))

    def __repr__(self) -> str:
        return f"Expr({self._node!r})"

    # -------------------------------------------------------------------------
    # Arithmetic operators
    # -------------------------------------------------------------------------

    def __add__(self, other: Any) -> Expr:
        """Add two expressions: col("a") + col("b") or col("a") + 1"""
        return Expr(Call("add", (self._node, _to_expr(other)._node)))

    def __radd__(self, other: Any) -> Expr:
        """Reverse add: 1 + col("a")"""
        return Expr(Call("add", (_to_expr(other)._node, self._node)))

    def __sub__(self, other: Any) -> Expr:
        """Subtract: col("a") - col("b") or col("a") - 1"""
        return Expr(Call("subtract", (self._node, _to_expr(other)._node)))

    def __rsub__(self, other: Any) -> Expr:
        """Reverse subtract: 1 - col("a")"""
        return Expr(Call("subtract", (_to_expr(other)._node, self._node)))

    def __mul__(self, other: Any) -> Expr:
        """Multiply: col("a") * col("b") or col("a") * 2"""
        return Expr(Call("multiply", (self._node, _to_expr(other)._node)))

    def __rmul__(self, other: Any) -> Expr:
        """Reverse multiply: 2 * col("a")"""
        return Expr(Call("multiply", (_to_expr(other)._node, self._node)))

    def __truediv__(self, other: Any) -> Expr:
        """Divide: col("a") / col("b") or col("a") / 2"""
        return Expr(Call("divide", (self._node, _to_expr(other)._node)))

    def __rtruediv__(self, other: Any) -> Expr:
        """Reverse divide: 1 / col("a")"""
        return Expr(Call("divide", (_to_expr(other)._node, self._node)))

    def __floordiv__(self, other: Any) -> Expr:
        """Floor divide: col("a") // col("b") or col("a") // 2"""
        return Expr(Call("floor_divide", (self._node, _to_expr(other)._node)))

    def __rfloordiv__(self, other: Any) -> Expr:
        """Reverse floor divide: 10 // col("a")"""
        return Expr(Call("floor_divide", (_to_expr(other)._node, self._node)))

    def __mod__(self, other: Any) -> Expr:
        """Modulo: col("a") % col("b") or col("a") % 2"""
        return Expr(Call("modulo", (self._node, _to_expr(other)._node)))

    def __rmod__(self, other: Any) -> Expr:
        """Reverse modulo: 10 % col("a")"""
        return Expr(Call("modulo", (_to_expr(other)._node, self._node)))

    def __pow__(self, other: Any) -> Expr:
        """Power: col("a") ** 2"""
        return Expr(Call("power", (self._node, _to_expr(other)._node)))

    def __rpow__(self, other: Any) -> Expr:
        """Reverse power: 2 ** col("a")"""
        return Expr(Call("power", (_to_expr(other)._node, self._node)))

    # -------------------------------------------------------------------------
    # Unary operators
    # -------------------------------------------------------------------------

    def __neg__(self) -> Expr:
        """Negate: -col("a")"""
        return Expr(Call("negate", (self._node,)))

    def __pos__(self) -> Expr:
        """Positive (identity): +col("a")"""
        return self  # No-op, return self

    def __abs__(self) -> Expr:
        """Absolute value: abs(col("a"))"""
        return Expr(Call("abs", (self._node,)))

    # -------------------------------------------------------------------------
    # Comparison operators
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        """Equal: col("a") == col("b") or col("a") == 1"""
        return Expr(Call("equal", (self._node, _to_expr(other)._node)))

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        """Not equal: col("a") != col("b") or col("a") != 1"""
        return Expr(Call("not_equal", (self._node, _to_expr(other)._node)))

    def __lt__(self, other: Any) -> Expr:
        """Less than: col("a") < col("b") or col("a") < 1"""
        return Expr(Call("less", (self._node, _to_expr(other)._node)))

    def __le__(self, other: Any) -> Expr:
        """Less than or equal: col("a") <= col("b") or col("a") <= 1"""
        return Expr(Call("less_equal", (self._node, _to_expr(other)._node)))

    def __gt__(self, other: Any) -> Expr:
        """Greater than: col("a") > col("b") or col("a") > 1"""
        return Expr(Call("greater", (self._node, _to_expr(other)._node)))

    def __ge__(self, other: Any) -> Expr:
        """Greater than or equal: col("a") >= col("b") or col("a") >= 1"""
        return Expr(Call("greater_equal", (self._node, _to_expr(other)._node)))

    # -------------------------------------------------------------------------
    # Logical operators
    # -------------------------------------------------------------------------

    def __and__(self, other: Any) -> Expr:
        """Logical AND: (col("a") > 0) & (col("b") < 10)"""
        return Expr(Call("and_", (self._node, _to_expr(other)._node)))

    def __rand__(self, other: Any) -> Expr:
        """Reverse AND"""
        return Expr(Call("and_", (_to_expr(other)._node, self._node)))

    def __or__(self, other: Any) -> Expr:
        """Logical OR: (col("a") > 0) | (col("b") < 10)"""
        return Expr(Call("or_", (self._node, _to_expr(other)._node)))

    def __ror__(self, other: Any) -> Expr:
        """Reverse OR"""
        return Expr(Call("or_", (_to_expr(other)._node, self._node)))

    def __invert__(self) -> Expr:
        """Logical NOT: ~(col("a") > 0)"""
        return Expr(Call("invert", (self._node,)))

    # -------------------------------------------------------------------------
    # Null checks
    # -------------------------------------------------------------------------

    def is_null(self) -> Expr:
        """Check if values are null/NA: col("a").is_null()"""
        return Expr(Call("is_null", (self._node,)))

    def is_not_null(self) -> Expr:
        """Check if values are not null/NA: col("a").is_not_null()"""
        return Expr(Call("is_not_null", (self._node,)))

    def fill_null(self, value: Any) -> Expr:
        """
        Fill null values with a replacement value.

        Parameters
        ----------
        value : Any
            The value to use for filling nulls. Can be a scalar or Expr.

        Returns
        -------
        Expr
            Expression with nulls filled.

        Examples
        --------
        >>> col("a").fill_null(0)
        >>> col("a").fill_null(col("b"))
        """
        return Expr(Call("fill_null", (self._node, _to_expr(value)._node)))

    def cast(self, dtype) -> Expr:
        """
        Cast expression to a different dtype.

        Parameters
        ----------
        dtype : dtype
            The target dtype. Can be a numpy dtype, pandas dtype string,
            or Python type (int, float, str, bool).

        Returns
        -------
        Expr
            Expression cast to the target dtype.

        Examples
        --------
        >>> col("a").cast("float64")
        >>> col("a").cast(int)
        >>> col("a").cast("string")
        """
        from pandas.lazy.ir import Cast

        return Expr(Cast(self._node, dtype))

    # -------------------------------------------------------------------------
    # Aggregation methods
    # -------------------------------------------------------------------------

    def sum(self) -> Expr:
        """
        Compute the sum of values.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").sum()
        """
        return Expr(Call("sum", (self._node,), is_aggregate=True))

    def mean(self) -> Expr:
        """
        Compute the mean of values.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").mean()
        """
        return Expr(Call("mean", (self._node,), is_aggregate=True))

    def min(self) -> Expr:
        """
        Compute the minimum value.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").min()
        """
        return Expr(Call("min", (self._node,), is_aggregate=True))

    def max(self) -> Expr:
        """
        Compute the maximum value.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").max()
        """
        return Expr(Call("max", (self._node,), is_aggregate=True))

    def count(self) -> Expr:
        """
        Count non-null values.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").count()
        """
        return Expr(Call("count", (self._node,), is_aggregate=True))

    def std(self, ddof: int = 1) -> Expr:
        """
        Compute the standard deviation.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").std()
        """
        return Expr(Call("std", (self._node,), {"ddof": ddof}, is_aggregate=True))

    def var(self, ddof: int = 1) -> Expr:
        """
        Compute the variance.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").var()
        """
        return Expr(Call("var", (self._node,), {"ddof": ddof}, is_aggregate=True))

    def first(self) -> Expr:
        """
        Get the first value.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").first()
        """
        return Expr(Call("first", (self._node,), is_aggregate=True))

    def last(self) -> Expr:
        """
        Get the last value.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").last()
        """
        return Expr(Call("last", (self._node,), is_aggregate=True))

    def n_unique(self) -> Expr:
        """
        Count unique values.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").n_unique()
        """
        return Expr(Call("n_unique", (self._node,), is_aggregate=True))

    # -------------------------------------------------------------------------
    # Window functions
    # -------------------------------------------------------------------------

    def over(self, partition_by: str | Expr | list[str | Expr] | None = None) -> Expr:
        """
        Apply aggregation as a window function over partitions.

        This turns an aggregation into a window function that computes
        the aggregation over each partition and broadcasts the result
        back to each row.

        Parameters
        ----------
        partition_by : str, Expr, or list, optional
            Column(s) to partition by. If None, computes over the entire
            DataFrame (equivalent to a global aggregation broadcasted).

        Returns
        -------
        Expr
            Window expression.

        Examples
        --------
        >>> # Running sum over entire DataFrame
        >>> col("value").sum().over()
        >>> # Sum per group
        >>> col("value").sum().over("group")
        >>> # Sum per multiple groups
        >>> col("value").sum().over(["region", "year"])
        """
        # Normalize partition_by to a tuple of IR nodes
        if partition_by is None:
            partition_nodes: tuple = ()
        elif isinstance(partition_by, str):
            partition_nodes = (FieldRef(partition_by),)
        elif isinstance(partition_by, Expr):
            partition_nodes = (partition_by._node,)
        elif isinstance(partition_by, list):
            partition_nodes = tuple(
                FieldRef(p) if isinstance(p, str) else p._node for p in partition_by
            )
        else:
            type_name = type(partition_by).__name__
            raise TypeError(f"partition_by must be str, Expr, or list, got {type_name}")

        return Expr(
            Call(
                "window",
                (self._node,),
                {"partition_by": partition_nodes},
            )
        )

    def rank(self) -> Expr:
        """
        Compute rank of values within each partition.

        Returns
        -------
        Expr
            Expression with rank values (1-based).

        Examples
        --------
        >>> col("value").rank().over("group")
        """
        return Expr(Call("rank", (self._node,)))

    def dense_rank(self) -> Expr:
        """
        Compute dense rank of values (no gaps in ranking).

        Returns
        -------
        Expr
            Expression with dense rank values (1-based).

        Examples
        --------
        >>> col("value").dense_rank().over("group")
        """
        return Expr(Call("dense_rank", (self._node,)))

    def row_number(self) -> Expr:
        """
        Compute row number within each partition.

        Returns
        -------
        Expr
            Expression with row numbers (1-based).

        Examples
        --------
        >>> col("value").row_number().over("group")
        """
        return Expr(Call("row_number", (self._node,)))

    def lag(self, n: int = 1, default: Any = None) -> Expr:
        """
        Access value from previous row.

        Parameters
        ----------
        n : int, default 1
            Number of rows to look back.
        default : Any, optional
            Value to use for rows without a previous value.

        Returns
        -------
        Expr
            Expression with lagged values.

        Examples
        --------
        >>> col("value").lag(1).over("group")
        >>> col("value").lag(2, default=0).over("group")
        """
        default_node = _to_expr(default)._node if default is not None else None
        return Expr(Call("lag", (self._node,), {"n": n, "default": default_node}))

    def lead(self, n: int = 1, default: Any = None) -> Expr:
        """
        Access value from next row.

        Parameters
        ----------
        n : int, default 1
            Number of rows to look forward.
        default : Any, optional
            Value to use for rows without a next value.

        Returns
        -------
        Expr
            Expression with lead values.

        Examples
        --------
        >>> col("value").lead(1).over("group")
        >>> col("value").lead(2, default=0).over("group")
        """
        default_node = _to_expr(default)._node if default is not None else None
        return Expr(Call("lead", (self._node,), {"n": n, "default": default_node}))

    def cum_sum(self) -> Expr:
        """
        Compute cumulative sum.

        Returns
        -------
        Expr
            Expression with cumulative sums.

        Examples
        --------
        >>> col("value").cum_sum().over("group")
        """
        return Expr(Call("cum_sum", (self._node,)))

    def cum_min(self) -> Expr:
        """
        Compute cumulative minimum.

        Returns
        -------
        Expr
            Expression with cumulative minimums.

        Examples
        --------
        >>> col("value").cum_min().over("group")
        """
        return Expr(Call("cum_min", (self._node,)))

    def cum_max(self) -> Expr:
        """
        Compute cumulative maximum.

        Returns
        -------
        Expr
            Expression with cumulative maximums.

        Examples
        --------
        >>> col("value").cum_max().over("group")
        """
        return Expr(Call("cum_max", (self._node,)))

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    @property
    def str(self) -> ExprStringAccessor:
        """Access string methods on this expression."""
        return ExprStringAccessor(self)

    @property
    def dt(self) -> ExprDatetimeAccessor:
        """Access datetime methods on this expression."""
        return ExprDatetimeAccessor(self)

    # -------------------------------------------------------------------------
    # Note: Expr is intentionally unhashable since __eq__ returns Expr
    # Use expr._ir for identity comparison if needed
    # -------------------------------------------------------------------------

    __hash__ = None  # type: ignore[assignment]


class ExprStringAccessor:
    """
    String accessor for Expr, providing string operations.

    This class is accessed via `Expr.str` and provides methods similar
    to pandas Series.str accessor, but for lazy expressions.

    Examples
    --------
    >>> from pandas.lazy import col
    >>> expr = col("name").str.lower()
    >>> expr = col("text").str.contains("pattern")
    """

    __slots__ = ("_expr",)

    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def _make_call(self, func: str, *args: Any, **kwargs: Any) -> Expr:
        """Helper to create a Call node with this expression as first arg."""
        ir_args = (self._expr._node,) + tuple(_to_expr(a)._node for a in args)
        return Expr(Call(func, ir_args, kwargs))

    def lower(self) -> Expr:
        """
        Convert strings to lowercase.

        Returns
        -------
        Expr
            Expression with lowercase strings.

        Examples
        --------
        >>> col("name").str.lower()
        """
        return self._make_call("str_lower")

    def upper(self) -> Expr:
        """
        Convert strings to uppercase.

        Returns
        -------
        Expr
            Expression with uppercase strings.

        Examples
        --------
        >>> col("name").str.upper()
        """
        return self._make_call("str_upper")

    def len(self) -> Expr:
        """
        Compute the length of each string.

        Returns
        -------
        Expr
            Expression with string lengths.

        Examples
        --------
        >>> col("name").str.len()
        """
        return self._make_call("str_len")

    def strip(self) -> Expr:
        """
        Strip whitespace from both ends of strings.

        Returns
        -------
        Expr
            Expression with stripped strings.

        Examples
        --------
        >>> col("name").str.strip()
        """
        return self._make_call("str_strip")

    def lstrip(self) -> Expr:
        """
        Strip whitespace from the left side of strings.

        Returns
        -------
        Expr
            Expression with left-stripped strings.

        Examples
        --------
        >>> col("name").str.lstrip()
        """
        return self._make_call("str_lstrip")

    def rstrip(self) -> Expr:
        """
        Strip whitespace from the right side of strings.

        Returns
        -------
        Expr
            Expression with right-stripped strings.

        Examples
        --------
        >>> col("name").str.rstrip()
        """
        return self._make_call("str_rstrip")

    def contains(self, pattern: str, *, regex: bool = True) -> Expr:
        """
        Test if pattern is contained within each string.

        Parameters
        ----------
        pattern : str
            Character sequence or regular expression.
        regex : bool, default True
            If True, assumes pattern is a regular expression.
            If False, treats pattern as a literal string.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.contains("foo")
        >>> col("text").str.contains("foo", regex=False)
        """
        return self._make_call("str_contains", pattern, regex=regex)

    def startswith(self, pattern: str) -> Expr:
        """
        Test if each string starts with pattern.

        Parameters
        ----------
        pattern : str
            Character sequence to search for.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("name").str.startswith("Dr.")
        """
        return self._make_call("str_startswith", pattern)

    def endswith(self, pattern: str) -> Expr:
        """
        Test if each string ends with pattern.

        Parameters
        ----------
        pattern : str
            Character sequence to search for.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("name").str.endswith(".txt")
        """
        return self._make_call("str_endswith", pattern)

    def replace(self, pat: str, repl: str, *, regex: bool = True) -> Expr:
        """
        Replace occurrences of pattern with replacement.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        repl : str
            Replacement string.
        regex : bool, default True
            If True, assumes pattern is a regular expression.
            If False, treats pattern as a literal string.

        Returns
        -------
        Expr
            Expression with replaced strings.

        Examples
        --------
        >>> col("text").str.replace("old", "new")
        """
        return self._make_call("str_replace", pat, repl, regex=regex)

    def slice(self, start: int | None = None, stop: int | None = None) -> Expr:
        """
        Slice substrings from each string.

        Parameters
        ----------
        start : int, optional
            Start position for slice.
        stop : int, optional
            Stop position for slice.

        Returns
        -------
        Expr
            Expression with sliced strings.

        Examples
        --------
        >>> col("name").str.slice(0, 3)
        """
        return self._make_call("str_slice", start=start, stop=stop)


class ExprDatetimeAccessor:
    """
    Datetime accessor for Expr, providing datetime component extraction.

    This class is accessed via `Expr.dt` and provides methods similar
    to pandas Series.dt accessor, but for lazy expressions.

    Examples
    --------
    >>> from pandas.lazy import col
    >>> expr = col("timestamp").dt.year
    >>> expr = col("date").dt.month
    """

    __slots__ = ("_expr",)

    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def _make_call(self, func: str) -> Expr:
        """Helper to create a Call node with this expression as first arg."""
        return Expr(Call(func, (self._expr._node,)))

    @property
    def year(self) -> Expr:
        """
        Extract the year component.

        Returns
        -------
        Expr
            Expression with year values.

        Examples
        --------
        >>> col("date").dt.year
        """
        return self._make_call("dt_year")

    @property
    def month(self) -> Expr:
        """
        Extract the month component (1-12).

        Returns
        -------
        Expr
            Expression with month values.

        Examples
        --------
        >>> col("date").dt.month
        """
        return self._make_call("dt_month")

    @property
    def day(self) -> Expr:
        """
        Extract the day component (1-31).

        Returns
        -------
        Expr
            Expression with day values.

        Examples
        --------
        >>> col("date").dt.day
        """
        return self._make_call("dt_day")

    @property
    def hour(self) -> Expr:
        """
        Extract the hour component (0-23).

        Returns
        -------
        Expr
            Expression with hour values.

        Examples
        --------
        >>> col("timestamp").dt.hour
        """
        return self._make_call("dt_hour")

    @property
    def minute(self) -> Expr:
        """
        Extract the minute component (0-59).

        Returns
        -------
        Expr
            Expression with minute values.

        Examples
        --------
        >>> col("timestamp").dt.minute
        """
        return self._make_call("dt_minute")

    @property
    def second(self) -> Expr:
        """
        Extract the second component (0-59).

        Returns
        -------
        Expr
            Expression with second values.

        Examples
        --------
        >>> col("timestamp").dt.second
        """
        return self._make_call("dt_second")

    @property
    def weekday(self) -> Expr:
        """
        Extract the day of the week (Monday=0, Sunday=6).

        Returns
        -------
        Expr
            Expression with weekday values.

        Examples
        --------
        >>> col("date").dt.weekday
        """
        return self._make_call("dt_weekday")

    @property
    def dayofweek(self) -> Expr:
        """Alias for weekday."""
        return self._make_call("dt_weekday")

    @property
    def dayofyear(self) -> Expr:
        """
        Extract the day of the year (1-366).

        Returns
        -------
        Expr
            Expression with day of year values.

        Examples
        --------
        >>> col("date").dt.dayofyear
        """
        return self._make_call("dt_dayofyear")

    @property
    def quarter(self) -> Expr:
        """
        Extract the quarter (1-4).

        Returns
        -------
        Expr
            Expression with quarter values.

        Examples
        --------
        >>> col("date").dt.quarter
        """
        return self._make_call("dt_quarter")

    @property
    def is_month_start(self) -> Expr:
        """
        Check if date is first day of month.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("date").dt.is_month_start
        """
        return self._make_call("dt_is_month_start")

    @property
    def is_month_end(self) -> Expr:
        """
        Check if date is last day of month.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("date").dt.is_month_end
        """
        return self._make_call("dt_is_month_end")

    @property
    def is_year_start(self) -> Expr:
        """
        Check if date is first day of year.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("date").dt.is_year_start
        """
        return self._make_call("dt_is_year_start")

    @property
    def is_year_end(self) -> Expr:
        """
        Check if date is last day of year.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("date").dt.is_year_end
        """
        return self._make_call("dt_is_year_end")

    @property
    def date(self) -> Expr:
        """
        Extract the date part (without time).

        Returns
        -------
        Expr
            Expression with date values.

        Examples
        --------
        >>> col("timestamp").dt.date
        """
        return self._make_call("dt_date")


# =============================================================================
# Conditional Expressions (when/then/otherwise)
# =============================================================================


def when(condition: Expr) -> When:
    """
    Start a conditional expression with when/then/otherwise.

    Parameters
    ----------
    condition : Expr
        A boolean expression that determines which values to select.

    Returns
    -------
    When
        A When object that can be chained with .then()

    Examples
    --------
    >>> from pandas.lazy import col, when
    >>> # Simple conditional
    >>> expr = when(col("a") > 0).then(1).otherwise(0)
    >>> # Multiple conditions can be chained
    >>> expr = (
    ...     when(col("a") > 100)
    ...     .then("high")
    ...     .when(col("a") > 50)
    ...     .then("medium")
    ...     .otherwise("low")
    ... )

    See Also
    --------
    When : The When builder class.
    Then : The Then builder class.
    """
    if not isinstance(condition, Expr):
        raise TypeError(
            f"condition must be an Expr, got {type(condition).__name__}. "
            f"Use col('name') for column references."
        )
    return When(((condition, None),))


class When:
    """
    Builder for conditional expressions (when/then/otherwise).

    This class is created by the when() function and allows building
    conditional expressions through method chaining.

    Do not instantiate directly - use the when() function.

    Examples
    --------
    >>> from pandas.lazy import col, when
    >>> expr = when(col("a") > 0).then(1).otherwise(0)
    """

    __slots__ = ("_cases",)

    def __init__(self, cases: tuple[tuple[Expr, Expr | None], ...]) -> None:
        # cases is a tuple of (condition, value) pairs
        # The last value may be None if .then() hasn't been called yet
        self._cases = cases

    def then(self, value: Any) -> Then:
        """
        Specify the value to use when the condition is true.

        Parameters
        ----------
        value : Any
            The value to use when the condition is true.
            Can be a scalar, Expr, or column reference.

        Returns
        -------
        Then
            A Then object for chaining .when() or .otherwise().

        Examples
        --------
        >>> when(col("a") > 0).then(1)
        >>> when(col("a") > 0).then(col("b"))
        """
        value_expr = _to_expr(value)
        # Update the last case with the value
        updated_cases = self._cases[:-1] + ((self._cases[-1][0], value_expr),)
        return Then(updated_cases)


class Then:
    """
    Builder for conditional expressions after .then().

    This class allows chaining additional .when() conditions or
    completing the expression with .otherwise().

    Do not instantiate directly - use when().then().

    Examples
    --------
    >>> from pandas.lazy import col, when
    >>> # Chain another condition
    >>> expr = when(col("a") > 0).then(1).when(col("a") < 0).then(-1).otherwise(0)
    >>> # Complete with otherwise
    >>> expr = when(col("a") > 0).then(1).otherwise(0)
    """

    __slots__ = ("_cases",)

    def __init__(self, cases: tuple[tuple[Expr, Expr], ...]) -> None:
        self._cases = cases

    def when(self, condition: Expr) -> When:
        """
        Add another condition to the chain.

        Parameters
        ----------
        condition : Expr
            A boolean expression for this condition.

        Returns
        -------
        When
            A When object that can be chained with .then().

        Examples
        --------
        >>> (
        ...     when(col("a") > 100)
        ...     .then("high")
        ...     .when(col("a") > 50)
        ...     .then("medium")
        ...     .otherwise("low")
        ... )
        """
        if not isinstance(condition, Expr):
            raise TypeError(
                f"condition must be an Expr, got {type(condition).__name__}. "
                f"Use col('name') for column references."
            )
        return When(self._cases + ((condition, None),))

    def otherwise(self, value: Any) -> Expr:
        """
        Specify the default value when no conditions match.

        Parameters
        ----------
        value : Any
            The default value to use when no conditions are true.
            Can be a scalar, Expr, or column reference.

        Returns
        -------
        Expr
            The complete conditional expression.

        Examples
        --------
        >>> when(col("a") > 0).then(1).otherwise(0)
        >>> when(col("a") > 0).then(col("b")).otherwise(col("c"))
        """
        value_expr = _to_expr(value)

        # Build the IR node for the conditional
        # cases: tuple of (condition, value) pairs
        # otherwise_value: the default value
        cases_ir = tuple((c._node, v._node) for c, v in self._cases)
        return Expr(
            Call(
                "case_when",
                (),  # No positional args, all in kwargs
                {"cases": cases_ir, "otherwise": value_expr._node},
            )
        )


def coalesce(*exprs: Expr | Any) -> Expr:
    """
    Return the first non-null value from the given expressions.

    Parameters
    ----------
    *exprs : Expr or scalar
        Expressions to coalesce. At least one must be provided.

    Returns
    -------
    Expr
        Expression that evaluates to the first non-null value.

    Examples
    --------
    >>> from pandas.lazy import col, coalesce
    >>> # Use col b if col a is null
    >>> expr = coalesce(col("a"), col("b"))
    >>> # Use col b, then col c, then default 0
    >>> expr = coalesce(col("a"), col("b"), 0)
    """
    if not exprs:
        raise ValueError("coalesce requires at least one expression")

    expr_nodes = tuple(_to_expr(e)._node for e in exprs)
    return Expr(Call("coalesce", expr_nodes))


def extract_output_name(expr: Expr) -> str:
    """
    Extract output column name from expression.

    Parameters
    ----------
    expr : Expr
        The expression to extract the name from.

    Returns
    -------
    str
        The output column name.

    Raises
    ------
    ValueError
        If the expression doesn't have a determinable output name.
    """
    node = expr._ir
    if isinstance(node, Alias):
        return node.name
    elif isinstance(node, FieldRef):
        return node.name
    elif isinstance(node, Call):
        # For computed expressions, we require an explicit alias
        raise ValueError(
            f"Computed expression requires .alias(): {node.function}(...). "
            f"Use .alias('name') to give this expression an output name."
        )
    else:
        raise ValueError(f"Cannot determine output name for {type(node).__name__}")


def normalize_exprs(exprs: tuple[Expr | str, ...]) -> tuple[Expr, ...]:
    """
    Normalize mixed str/Expr inputs to Expr.

    Parameters
    ----------
    exprs : tuple of Expr or str
        The expressions to normalize.

    Returns
    -------
    tuple of Expr
        All inputs converted to Expr.
    """
    result = []
    for e in exprs:
        if isinstance(e, str):
            result.append(col(e))
        elif isinstance(e, Expr):
            result.append(e)
        else:
            raise TypeError(
                f"Expected Expr or str, got {type(e).__name__}. "
                f"Use col('name') for column references or lit(value) for literals."
            )
    return tuple(result)
