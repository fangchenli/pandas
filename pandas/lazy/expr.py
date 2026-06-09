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

    def isin(self, values: list | tuple | set) -> Expr:
        """
        Check if values are contained in a set of values.

        Parameters
        ----------
        values : list, tuple, or set
            The set of values to check membership against.

        Returns
        -------
        Expr
            Boolean expression indicating membership.

        Examples
        --------
        >>> col("payment_type").isin([1, 2])
        >>> col("region").isin(["North", "South", "East"])
        """
        # Convert to list for consistent handling
        if isinstance(values, (set, tuple)):
            values = list(values)
        if not isinstance(values, list):
            raise TypeError(
                f"values must be a list, tuple, or set, got {type(values).__name__}"
            )
        # Pass values as a kwarg since it's a set, not a column reference
        return Expr(Call("isin", (self._node,), {"values": values}))

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

    def corr(self, other: Expr) -> Expr:
        """
        Pearson correlation coefficient between this expression and ``other``.

        Composed from sums so it works inside ``group_by().agg(...)`` (and as
        a global aggregate) without a dedicated kernel: the engine computes the
        six leaf sums - n, Sx, Sy, Sxx, Syy, Sxy - and evaluates the closed
        form ``(n*Sxy - Sx*Sy) / sqrt((n*Sxx - Sx^2)(n*Syy - Sy^2))`` as a
        post-aggregation projection.

        Examples
        --------
        >>> ldf.group_by("g").agg(col("x").corr(col("y")).alias("r"))
        """
        n = self.count()
        sx, sy = self.sum(), other.sum()
        sxx, syy, sxy = (self * self).sum(), (other * other).sum(), (self * other).sum()
        numerator = n * sxy - sx * sy
        denominator = ((n * sxx - sx * sx) * (n * syy - sy * sy)) ** 0.5
        return numerator / denominator

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

    def median(self) -> Expr:
        """
        Compute the median value.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").median()
        """
        return Expr(Call("median", (self._node,), is_aggregate=True))

    def quantile(self, q: float = 0.5, interpolation: str = "linear") -> Expr:
        """
        Compute the quantile value.

        Parameters
        ----------
        q : float, default 0.5
            Quantile to compute (0.0 to 1.0).
        interpolation : str, default "linear"
            Interpolation method.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").quantile(0.75)
        """
        return Expr(
            Call(
                "quantile",
                (self._node,),
                {"q": q, "interpolation": interpolation},
                is_aggregate=True,
            )
        )

    def any(self) -> Expr:
        """
        Check if any value is True.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("flag").any()
        """
        return Expr(Call("any", (self._node,), is_aggregate=True))

    def all(self) -> Expr:
        """
        Check if all values are True.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("flag").all()
        """
        return Expr(Call("all", (self._node,), is_aggregate=True))

    def product(self) -> Expr:
        """
        Compute the product of values.

        Returns
        -------
        Expr
            Aggregation expression.

        Examples
        --------
        >>> col("a").product()
        """
        return Expr(Call("product", (self._node,), is_aggregate=True))

    # -------------------------------------------------------------------------
    # General transforms
    # -------------------------------------------------------------------------

    def clip(self, lower=None, upper=None) -> Expr:
        """
        Clip (limit) values to a given range.

        Parameters
        ----------
        lower : scalar, optional
            Minimum value. Values below this will be set to lower.
        upper : scalar, optional
            Maximum value. Values above this will be set to upper.

        Returns
        -------
        Expr
            Expression with clipped values.

        Examples
        --------
        >>> col("a").clip(lower=0, upper=100)
        """
        return Expr(Call("clip", (self._node,), {"lower": lower, "upper": upper}))

    def diff(self, periods: int = 1) -> Expr:
        """
        Calculate the difference between consecutive values.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift for calculating difference.

        Returns
        -------
        Expr
            Expression with differences.

        Examples
        --------
        >>> col("a").diff()
        >>> col("a").diff(periods=2)
        """
        return Expr(Call("diff", (self._node,), {"periods": periods}))

    def pct_change(self, periods: int = 1) -> Expr:
        """
        Calculate percentage change between consecutive values.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift for calculating change.

        Returns
        -------
        Expr
            Expression with percentage changes.

        Examples
        --------
        >>> col("price").pct_change()
        """
        return Expr(Call("pct_change", (self._node,), {"periods": periods}))

    def shift(self, periods: int = 1, fill_value: Any = None) -> Expr:
        """
        Shift values by specified number of periods.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift. Positive shifts forward.
        fill_value : Any, optional
            Value to use for filling gaps.

        Returns
        -------
        Expr
            Expression with shifted values.

        Examples
        --------
        >>> col("a").shift(1)
        >>> col("a").shift(-1, fill_value=0)
        """
        return Expr(
            Call("shift", (self._node,), {"periods": periods, "fill_value": fill_value})
        )

    def between(self, left, right, inclusive: str = "both") -> Expr:
        """
        Check if values are between left and right bounds.

        Parameters
        ----------
        left : scalar
            Left bound.
        right : scalar
            Right bound.
        inclusive : str, default "both"
            Include boundaries: "both", "neither", "left", or "right".

        Returns
        -------
        Expr
            Boolean expression indicating if values are in range.

        Examples
        --------
        >>> col("a").between(0, 100)
        >>> col("a").between(0, 100, inclusive="left")
        """
        return Expr(
            Call(
                "between",
                (self._node,),
                {"left": left, "right": right, "inclusive": inclusive},
            )
        )

    def ffill(self, limit: int | None = None) -> Expr:
        """
        Forward fill missing values.

        Parameters
        ----------
        limit : int, optional
            Maximum number of consecutive NaN values to forward fill.

        Returns
        -------
        Expr
            Expression with forward-filled values.

        Examples
        --------
        >>> col("a").ffill()
        >>> col("a").ffill(limit=2)
        """
        return Expr(Call("ffill", (self._node,), {"limit": limit}))

    def bfill(self, limit: int | None = None) -> Expr:
        """
        Backward fill missing values.

        Parameters
        ----------
        limit : int, optional
            Maximum number of consecutive NaN values to backward fill.

        Returns
        -------
        Expr
            Expression with backward-filled values.

        Examples
        --------
        >>> col("a").bfill()
        >>> col("a").bfill(limit=2)
        """
        return Expr(Call("bfill", (self._node,), {"limit": limit}))

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

    def cum_mean(self) -> Expr:
        """
        Compute cumulative mean.

        Returns
        -------
        Expr
            Expression with cumulative means.

        Examples
        --------
        >>> col("value").cum_mean().over("group")
        """
        return Expr(Call("cum_mean", (self._node,)))

    def cum_prod(self) -> Expr:
        """
        Compute cumulative product.

        Returns
        -------
        Expr
            Expression with cumulative products.

        Examples
        --------
        >>> col("value").cum_prod().over("group")
        """
        return Expr(Call("cum_prod", (self._node,)))

    # -------------------------------------------------------------------------
    # Rolling Window Operations
    # -------------------------------------------------------------------------

    def rolling_sum(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling sum over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with rolling sums.

        Examples
        --------
        >>> col("value").rolling_sum(window=5)
        """
        return Expr(
            Call(
                "rolling_sum",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_mean(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling mean over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with rolling means.

        Examples
        --------
        >>> col("value").rolling_mean(window=5)
        """
        return Expr(
            Call(
                "rolling_mean",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_min(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling minimum over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with rolling minimums.

        Examples
        --------
        >>> col("value").rolling_min(window=5)
        """
        return Expr(
            Call(
                "rolling_min",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_max(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling maximum over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with rolling maximums.

        Examples
        --------
        >>> col("value").rolling_max(window=5)
        """
        return Expr(
            Call(
                "rolling_max",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_std(
        self, window: int, min_periods: int | None = None, ddof: int = 1
    ) -> Expr:
        """
        Compute rolling standard deviation over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.
        ddof : int, default 1
            Delta degrees of freedom.

        Returns
        -------
        Expr
            Expression with rolling standard deviations.

        Examples
        --------
        >>> col("value").rolling_std(window=5)
        """
        return Expr(
            Call(
                "rolling_std",
                (self._node,),
                {"window": window, "min_periods": min_periods, "ddof": ddof},
            )
        )

    def rolling_var(
        self, window: int, min_periods: int | None = None, ddof: int = 1
    ) -> Expr:
        """
        Compute rolling variance over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.
        ddof : int, default 1
            Delta degrees of freedom.

        Returns
        -------
        Expr
            Expression with rolling variances.

        Examples
        --------
        >>> col("value").rolling_var(window=5)
        """
        return Expr(
            Call(
                "rolling_var",
                (self._node,),
                {"window": window, "min_periods": min_periods, "ddof": ddof},
            )
        )

    def rolling_median(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling median over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with rolling medians.

        Examples
        --------
        >>> col("value").rolling_median(window=5)
        """
        return Expr(
            Call(
                "rolling_median",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_argmax(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling argmax (index of maximum) over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with indices of rolling maximums.

        Examples
        --------
        >>> col("value").rolling_argmax(window=5)
        """
        return Expr(
            Call(
                "rolling_argmax",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_argmin(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling argmin (index of minimum) over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with indices of rolling minimums.

        Examples
        --------
        >>> col("value").rolling_argmin(window=5)
        """
        return Expr(
            Call(
                "rolling_argmin",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_rank(
        self, window: int, min_periods: int | None = None, pct: bool = False
    ) -> Expr:
        """
        Compute rolling rank over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.
        pct : bool, default False
            If True, compute percentage rank.

        Returns
        -------
        Expr
            Expression with rolling ranks.

        Examples
        --------
        >>> col("value").rolling_rank(window=5)
        >>> col("value").rolling_rank(window=5, pct=True)
        """
        return Expr(
            Call(
                "rolling_rank",
                (self._node,),
                {"window": window, "min_periods": min_periods, "pct": pct},
            )
        )

    def rolling_count(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Count non-null values in a rolling window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to 1.

        Returns
        -------
        Expr
            Expression with rolling counts.

        Examples
        --------
        >>> col("value").rolling_count(window=5)
        """
        return Expr(
            Call(
                "rolling_count",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_quantile(
        self, window: int, quantile: float, min_periods: int | None = None
    ) -> Expr:
        """
        Compute rolling quantile over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        quantile : float
            Quantile to compute (0.0 to 1.0).
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with rolling quantiles.

        Examples
        --------
        >>> col("value").rolling_quantile(window=5, quantile=0.5)
        >>> col("value").rolling_quantile(window=10, quantile=0.75)
        """
        return Expr(
            Call(
                "rolling_quantile",
                (self._node,),
                {"window": window, "quantile": quantile, "min_periods": min_periods},
            )
        )

    def rolling_skew(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling skewness over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to 3.

        Returns
        -------
        Expr
            Expression with rolling skewness values.

        Examples
        --------
        >>> col("value").rolling_skew(window=10)
        """
        return Expr(
            Call(
                "rolling_skew",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_kurt(self, window: int, min_periods: int | None = None) -> Expr:
        """
        Compute rolling kurtosis over a window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to 4.

        Returns
        -------
        Expr
            Expression with rolling kurtosis values (excess kurtosis).

        Examples
        --------
        >>> col("value").rolling_kurt(window=10)
        """
        return Expr(
            Call(
                "rolling_kurt",
                (self._node,),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_apply(
        self, window: int, func, min_periods: int | None = None, raw: bool = True
    ) -> Expr:
        """
        Apply a custom function to a rolling window.

        Parameters
        ----------
        window : int
            Size of the rolling window.
        func : callable
            Function to apply to each window. Should accept an array
            and return a scalar.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.
        raw : bool, default True
            If True, pass raw numpy array to function.

        Returns
        -------
        Expr
            Expression with custom rolling function results.

        Examples
        --------
        >>> col("value").rolling_apply(window=5, func=np.nanmean)
        """
        return Expr(
            Call(
                "rolling_apply",
                (self._node,),
                {
                    "window": window,
                    "func": func,
                    "min_periods": min_periods,
                    "raw": raw,
                },
            )
        )

    def rolling_cov(
        self, other: Expr | str, window: int, min_periods: int | None = None
    ) -> Expr:
        """
        Compute rolling covariance with another column.

        Parameters
        ----------
        other : Expr or str
            The other column to compute covariance with.
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with rolling covariance values.

        Examples
        --------
        >>> col("x").rolling_cov(col("y"), window=10)
        >>> col("x").rolling_cov("y", window=10)
        """
        # Treat string as column reference for convenience
        other_expr = col(other) if isinstance(other, str) else other
        return Expr(
            Call(
                "rolling_cov",
                (self._node, other_expr._node),
                {"window": window, "min_periods": min_periods},
            )
        )

    def rolling_corr(
        self, other: Expr | str, window: int, min_periods: int | None = None
    ) -> Expr:
        """
        Compute rolling Pearson correlation with another column.

        Parameters
        ----------
        other : Expr or str
            The other column to compute correlation with.
        window : int
            Size of the rolling window.
        min_periods : int or None, default None
            Minimum number of observations required. Defaults to window size.

        Returns
        -------
        Expr
            Expression with rolling correlation values.

        Examples
        --------
        >>> col("x").rolling_corr(col("y"), window=10)
        >>> col("x").rolling_corr("y", window=10)
        """
        # Treat string as column reference for convenience
        other_expr = col(other) if isinstance(other, str) else other
        return Expr(
            Call(
                "rolling_corr",
                (self._node, other_expr._node),
                {"window": window, "min_periods": min_periods},
            )
        )

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
        ir_args = (self._expr._node, *(_to_expr(a)._node for a in args))
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

    def capitalize(self) -> Expr:
        """
        Convert strings to have only the first character capitalized.

        Returns
        -------
        Expr
            Expression with capitalized strings.

        Examples
        --------
        >>> col("name").str.capitalize()
        """
        return self._make_call("str_capitalize")

    def title(self) -> Expr:
        """
        Convert strings to titlecase.

        Returns
        -------
        Expr
            Expression with titlecased strings.

        Examples
        --------
        >>> col("name").str.title()
        """
        return self._make_call("str_title")

    def swapcase(self) -> Expr:
        """
        Swap case of strings (upper becomes lower and vice versa).

        Returns
        -------
        Expr
            Expression with swapped case strings.

        Examples
        --------
        >>> col("name").str.swapcase()
        """
        return self._make_call("str_swapcase")

    def center(self, width: int, fillchar: str = " ") -> Expr:
        """
        Center strings in the specified width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string.
        fillchar : str, default " "
            Character to use for padding.

        Returns
        -------
        Expr
            Expression with centered strings.

        Examples
        --------
        >>> col("name").str.center(10)
        >>> col("name").str.center(10, "-")
        """
        return self._make_call("str_center", width, fillchar)

    def pad(self, width: int, side: str = "left", fillchar: str = " ") -> Expr:
        """
        Pad strings to specified width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string.
        side : {"left", "right", "both"}, default "left"
            Side to pad on.
        fillchar : str, default " "
            Character to use for padding.

        Returns
        -------
        Expr
            Expression with padded strings.

        Examples
        --------
        >>> col("name").str.pad(10, side="left")
        """
        return self._make_call("str_pad", width, side=side, fillchar=fillchar)

    def zfill(self, width: int) -> Expr:
        """
        Pad strings with zeros on the left.

        Parameters
        ----------
        width : int
            Minimum width of resulting string.

        Returns
        -------
        Expr
            Expression with zero-padded strings.

        Examples
        --------
        >>> col("id").str.zfill(5)
        """
        return self._make_call("str_zfill", width)

    def count(self, pattern: str) -> Expr:
        """
        Count occurrences of pattern in each string.

        Parameters
        ----------
        pattern : str
            Character sequence or regular expression.

        Returns
        -------
        Expr
            Expression with occurrence counts.

        Examples
        --------
        >>> col("text").str.count("a")
        """
        return self._make_call("str_count", pattern)

    def find(self, sub: str, start: int = 0, end: int | None = None) -> Expr:
        """
        Return lowest index of substring.

        Parameters
        ----------
        sub : str
            Substring to search for.
        start : int, default 0
            Start position for search.
        end : int, optional
            End position for search.

        Returns
        -------
        Expr
            Expression with index positions (-1 if not found).

        Examples
        --------
        >>> col("text").str.find("needle")
        """
        return self._make_call("str_find", sub, start=start, end=end)

    def match(self, pattern: str) -> Expr:
        """
        Determine if each string matches a regular expression.

        Parameters
        ----------
        pattern : str
            Regular expression pattern.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.match(r"^[A-Z]")
        """
        return self._make_call("str_match", pattern)

    def reverse(self) -> Expr:
        """
        Reverse each string.

        Returns
        -------
        Expr
            Expression with reversed strings.

        Examples
        --------
        >>> col("name").str.reverse()
        """
        return self._make_call("str_reverse")

    def isalnum(self) -> Expr:
        """
        Check whether all characters are alphanumeric.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.isalnum()
        """
        return self._make_call("str_is_alnum")

    def isalpha(self) -> Expr:
        """
        Check whether all characters are alphabetic.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.isalpha()
        """
        return self._make_call("str_is_alpha")

    def isdigit(self) -> Expr:
        """
        Check whether all characters are digits.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.isdigit()
        """
        return self._make_call("str_is_digit")

    def isdecimal(self) -> Expr:
        """
        Check whether all characters are decimal.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.isdecimal()
        """
        return self._make_call("str_is_decimal")

    def isspace(self) -> Expr:
        """
        Check whether all characters are whitespace.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.isspace()
        """
        return self._make_call("str_is_space")

    def islower(self) -> Expr:
        """
        Check whether all characters are lowercase.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.islower()
        """
        return self._make_call("str_is_lower")

    def isupper(self) -> Expr:
        """
        Check whether all characters are uppercase.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.isupper()
        """
        return self._make_call("str_is_upper")

    def istitle(self) -> Expr:
        """
        Check whether all characters are titlecase.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.istitle()
        """
        return self._make_call("str_is_title")

    def isnumeric(self) -> Expr:
        """
        Check whether all characters are numeric.

        Returns
        -------
        Expr
            Boolean expression.

        Examples
        --------
        >>> col("text").str.isnumeric()
        """
        return self._make_call("str_is_numeric")

    def split(self, pattern: str = " ", n: int = -1, *, regex: bool = False) -> Expr:
        """
        Split strings by pattern.

        Parameters
        ----------
        pattern : str, default " "
            String or regex pattern to split on.
        n : int, default -1
            Maximum number of splits. -1 means no limit.
        regex : bool, default False
            Whether to treat pattern as regex.

        Returns
        -------
        Expr
            Expression with list of split strings.

        Examples
        --------
        >>> col("text").str.split(" ")
        >>> col("text").str.split(",", n=2)
        """
        return self._make_call("str_split", pattern, n, regex=regex)

    def rsplit(self, pattern: str = " ", n: int = -1) -> Expr:
        """
        Split strings by pattern from the right.

        Parameters
        ----------
        pattern : str, default " "
            String pattern to split on.
        n : int, default -1
            Maximum number of splits. -1 means no limit.

        Returns
        -------
        Expr
            Expression with list of split strings.

        Examples
        --------
        >>> col("text").str.rsplit(" ", n=2)
        """
        return self._make_call("str_rsplit", pattern, n)

    def repeat(self, repeats: int) -> Expr:
        """
        Repeat strings a specified number of times.

        Parameters
        ----------
        repeats : int
            Number of times to repeat each string.

        Returns
        -------
        Expr
            Expression with repeated strings.

        Examples
        --------
        >>> col("text").str.repeat(3)
        """
        return self._make_call("str_repeat", repeats)

    def get(self, index: int) -> Expr:
        """
        Extract character at specified position.

        Parameters
        ----------
        index : int
            Position of character to extract (supports negative indexing).

        Returns
        -------
        Expr
            Expression with single characters.

        Examples
        --------
        >>> col("text").str.get(0)  # First character
        >>> col("text").str.get(-1)  # Last character
        """
        return self._make_call("str_get", index)

    def ljust(self, width: int, fillchar: str = " ") -> Expr:
        """
        Left-justify strings in a field of given width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string.
        fillchar : str, default " "
            Character to use for padding.

        Returns
        -------
        Expr
            Expression with left-justified strings.

        Examples
        --------
        >>> col("text").str.ljust(10)
        """
        return self._make_call("str_ljust", width, fillchar)

    def rjust(self, width: int, fillchar: str = " ") -> Expr:
        """
        Right-justify strings in a field of given width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string.
        fillchar : str, default " "
            Character to use for padding.

        Returns
        -------
        Expr
            Expression with right-justified strings.

        Examples
        --------
        >>> col("text").str.rjust(10)
        """
        return self._make_call("str_rjust", width, fillchar)

    def normalize(self, form: str = "NFC") -> Expr:
        """
        Normalize Unicode strings.

        Parameters
        ----------
        form : str, default "NFC"
            Unicode normalization form: "NFC", "NFKC", "NFD", or "NFKD".

        Returns
        -------
        Expr
            Expression with normalized strings.

        Examples
        --------
        >>> col("text").str.normalize("NFC")
        """
        return self._make_call("str_normalize", form)


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

    @property
    def time(self) -> Expr:
        """
        Extract the time part (without date).

        Returns
        -------
        Expr
            Expression with time values.

        Examples
        --------
        >>> col("timestamp").dt.time
        """
        return self._make_call("dt_time")

    @property
    def days_in_month(self) -> Expr:
        """
        Get the number of days in the month.

        Returns
        -------
        Expr
            Expression with days in month values.

        Examples
        --------
        >>> col("date").dt.days_in_month
        """
        return self._make_call("dt_days_in_month")

    def floor(self, unit: str) -> Expr:
        """
        Floor datetime to specified unit.

        Parameters
        ----------
        unit : str
            Unit to floor to (e.g., "D" for day, "h" for hour).

        Returns
        -------
        Expr
            Expression with floored datetimes.

        Examples
        --------
        >>> col("timestamp").dt.floor("D")
        >>> col("timestamp").dt.floor("h")
        """
        return Expr(Call("dt_floor", (self._expr._node,), {"unit": unit}))

    def ceil(self, unit: str) -> Expr:
        """
        Ceil datetime to specified unit.

        Parameters
        ----------
        unit : str
            Unit to ceil to (e.g., "D" for day, "h" for hour).

        Returns
        -------
        Expr
            Expression with ceiled datetimes.

        Examples
        --------
        >>> col("timestamp").dt.ceil("D")
        >>> col("timestamp").dt.ceil("h")
        """
        return Expr(Call("dt_ceil", (self._expr._node,), {"unit": unit}))

    def round(self, unit: str) -> Expr:
        """
        Round datetime to specified unit.

        Parameters
        ----------
        unit : str
            Unit to round to (e.g., "D" for day, "h" for hour).

        Returns
        -------
        Expr
            Expression with rounded datetimes.

        Examples
        --------
        >>> col("timestamp").dt.round("D")
        >>> col("timestamp").dt.round("h")
        """
        return Expr(Call("dt_round", (self._expr._node,), {"unit": unit}))

    def normalize(self) -> Expr:
        """
        Normalize datetime to midnight (remove time component).

        Returns
        -------
        Expr
            Expression with normalized datetimes.

        Examples
        --------
        >>> col("timestamp").dt.normalize()
        """
        return self._make_call("dt_normalize")

    def strftime(self, format: str) -> Expr:
        """
        Format datetime as string.

        Parameters
        ----------
        format : str
            strftime format string.

        Returns
        -------
        Expr
            Expression with formatted strings.

        Examples
        --------
        >>> col("timestamp").dt.strftime("%Y-%m-%d")
        """
        return Expr(Call("dt_strftime", (self._expr._node,), {"format": format}))


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
        updated_cases = (*self._cases[:-1], (self._cases[-1][0], value_expr))
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
        return When((*self._cases, (condition, None)))

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
