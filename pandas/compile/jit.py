"""
jit.py — JIT compiler for pandas operations.

Provides TracedDataFrame / TracedSeries proxy objects that intercept
pandas operations and build an intermediate representation (IR) graph.
Graph breaks are handled transparently via materialization (Dynamo-style):
compile the IR built so far, execute it on a backend, return the real
value, and resume tracing.

Usage:
    @compile
    def process(df):
        filtered = df[df["price"] > 100]
        if len(filtered) > 10:        # graph break — handled!
            return filtered.head(10)
        return filtered

    result = process(my_dataframe)
    print(process.explain(my_dataframe))
"""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import re as _re
from typing import (
    TYPE_CHECKING,
    Any,
    overload,
)

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

import pandas as pd
from pandas.compile.compiler import (
    Backend,
    CompiledSegment,
    CompiledStage,
    ConnectedPlan,
    EagerSegment,
    ExecutionPlan,
    GraphBreakStage,
    PandasBackend,
    SchemaGuard,
    StageSchema,
    SubstraitCompiler,
    default_backend,
    infer_schema,
)
from pandas.compile.ir import (
    NUMERIC_DTYPES,
    AddColumn,
    Aggregate,
    BinOp,
    CastExpr,
    ColRef,
    Distinct,
    DType,
    Expr,
    Filter,
    FunctionCall,
    IfThenExpr,
    IRNode,
    Join,
    Limit,
    Literal,
    Project,
    ReadTable,
    RenameColumns,
    ScalarSubquery,
    Schema,
    SingularOrList,
    Sort,
    UnaryOp,
    Union,
    Window,
    WindowSpec,
    explain_expr,
    explain_ir,
    pandas_dtype_to_ir,
)

log = logging.getLogger("pandas.compile")

# Save the real pd.concat so we can restore it after tracing.
_original_concat = pd.concat

# Active tracing context for the pd.concat interceptor.
_active_trace_ctx: TraceContext | None = None


def _traced_concat(objs, *args, **kwargs):
    """Interceptor for pd.concat that builds a Union IR when tracing."""
    ctx = _active_trace_ctx
    objs_list = list(objs)
    has_traced = any(isinstance(o, TracedDataFrame) for o in objs_list)
    if ctx is None or not has_traced:
        return _original_concat(objs_list, *args, **kwargs)

    # Normalize all inputs to TracedDataFrame.
    ir_inputs = []
    for obj in objs_list:
        if isinstance(obj, TracedDataFrame):
            ir_inputs.append(obj._ir)
        elif isinstance(obj, pd.DataFrame):
            name = ctx.next_materialized_name()
            ctx.register_table(name, obj)
            ir_inputs.append(ReadTable(name, infer_schema(obj)))
        else:
            raise TypeError(f"Cannot concat {type(obj)} in traced context")

    return TracedDataFrame(ctx, Union(ir_inputs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_expr(value) -> Expr:
    """Convert a Python value or TracedSeries to an Expr."""
    if isinstance(value, DeferredScalar):
        return value._expr
    elif isinstance(value, TracedSeries):
        return value._expr
    elif isinstance(value, Expr):
        return value
    elif isinstance(value, (int, float, str, bool)):
        return Literal(value)
    raise TypeError(f"Cannot convert {type(value)} to expression")


_SENTINEL = object()


class DeferredScalar:
    """
    A lazy scalar that wraps a deferred aggregation.

    When used in arithmetic with TracedSeries (e.g. ``series / total``),
    it produces a ``ScalarSubquery`` expression in the IR, avoiding a
    graph break. When Python needs the actual value (bool, comparison,
    control flow), it materializes.
    """

    def __init__(
        self,
        ctx: TraceContext,
        source_ir: IRNode,
        column_name: str,
        agg_func: str,
        dtype: DType,
    ):
        self._ctx = ctx
        self._source_ir = source_ir
        self._column_name = column_name
        self._agg_func = agg_func
        self._dtype = dtype
        self._materialized = _SENTINEL

    @property
    def _expr(self) -> ScalarSubquery:
        agg_node = Aggregate(
            self._source_ir,
            [],
            [(self._column_name, self._column_name, self._agg_func)],
        )
        return ScalarSubquery(agg_node, self._dtype)

    def _materialize(self):
        if self._materialized is not _SENTINEL:
            return self._materialized
        _, df = self._ctx.materialize(self._source_ir)
        agg_map = {
            "sum": "sum",
            "avg": "mean",
            "mean": "mean",
            "min": "min",
            "max": "max",
            "count": "count",
            "std": "std",
            "var": "var",
        }
        self._materialized = getattr(df[self._column_name], agg_map[self._agg_func])()
        return self._materialized

    # Python protocol: force materialization
    def __bool__(self):
        return bool(self._materialize())

    def __int__(self):
        return int(self._materialize())

    def __float__(self):
        return float(self._materialize())

    def __gt__(self, other):
        return self._materialize() > other

    def __ge__(self, other):
        return self._materialize() >= other

    def __lt__(self, other):
        return self._materialize() < other

    def __le__(self, other):
        return self._materialize() <= other

    def __eq__(self, other):
        return self._materialize() == other

    def __ne__(self, other):
        return self._materialize() != other

    def item(self):
        return self._materialize()

    def __repr__(self):
        return (
            f"DeferredScalar({self._agg_func}({self._column_name}), "
            f"dtype={self._dtype.name})"
        )


# ---------------------------------------------------------------------------
# Query string parser (for df.query())
# ---------------------------------------------------------------------------


def _parse_query_string(expr_str: str, schema: Schema) -> Expr:
    parts_and_ops = _re.split(r"\s+(and|or)\s+", expr_str)
    if len(parts_and_ops) == 1:
        tree = ast.parse(expr_str.strip(), mode="eval")
        return _ast_to_expr(tree.body, schema)
    exprs, ops = [], []
    for i, part in enumerate(parts_and_ops):
        if i % 2 == 0:
            tree = ast.parse(part.strip(), mode="eval")
            exprs.append(_ast_to_expr(tree.body, schema))
        else:
            ops.append(part)
    result = exprs[0]
    for i, op in enumerate(ops):
        result = BinOp(op, result, exprs[i + 1])
    return result


def _ast_to_expr(node, schema: Schema) -> Expr:
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ValueError("Chained comparisons not supported")
        left = _ast_to_expr(node.left, schema)
        right = _ast_to_expr(node.comparators[0], schema)
        op_map = {
            ast.Gt: "gt",
            ast.GtE: "gte",
            ast.Lt: "lt",
            ast.LtE: "lte",
            ast.Eq: "eq",
            ast.NotEq: "ne",
        }
        return BinOp(op_map[type(node.ops[0])], left, right)
    elif isinstance(node, ast.BinOp):
        left = _ast_to_expr(node.left, schema)
        right = _ast_to_expr(node.right, schema)
        op_map = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.BitAnd: "and",
            ast.BitOr: "or",
        }
        return BinOp(op_map[type(node.op)], left, right)
    elif isinstance(node, ast.Name):
        if node.id in schema.columns:
            return ColRef(node.id)
        raise ValueError(f"Unknown column: {node.id}")
    elif isinstance(node, ast.Constant):
        return Literal(node.value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return UnaryOp("not", _ast_to_expr(node.operand, schema))
    raise ValueError(f"Unsupported: {type(node).__name__}")


# ---------------------------------------------------------------------------
# TraceContext — the tracing runtime
# ---------------------------------------------------------------------------


class TraceContext:
    """
    The runtime context during tracing. Manages the current IR graph,
    table registry (for materialization), and segment accumulation.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.tables: dict[str, pd.DataFrame] = {}
        self.scalars: dict[str, Any] = {}
        self.segments: list[CompiledSegment | EagerSegment] = []
        self._mat_counter = 0
        self._segment_counter = 0

    def register_table(self, name: str, df: pd.DataFrame):
        self.tables[name] = df

    def next_materialized_name(self) -> str:
        self._mat_counter += 1
        return f"__mat_{self._mat_counter}"

    def next_segment_id(self) -> int:
        self._segment_counter += 1
        return self._segment_counter

    def materialize(
        self, ir_node: IRNode, reason: str = ""
    ) -> tuple[str, pd.DataFrame]:
        """
        Compile and execute the IR up to this point.
        This is the GRAPH BREAK mechanism.
        """
        input_tables = self._collect_read_tables(ir_node)
        out_name = self.next_materialized_name()

        seg = CompiledSegment(
            ir_node=ir_node,
            input_tables=input_tables,
            output_table=out_name,
            description=f"segment_{self.next_segment_id()}",
        )
        self.segments.append(seg)

        result_df = self.backend.execute(ir_node, self.tables)
        self.tables[out_name] = result_df

        log.info(
            "MATERIALIZED %s: %d rows, columns=%s",
            out_name,
            len(result_df),
            list(result_df.columns),
        )
        return out_name, result_df

    def add_eager_segment(
        self,
        operation: str,
        fn: Callable,
        input_tables: list[str],
        output_names: list[str],
        reason: str = "",
    ):
        seg = EagerSegment(
            operation=operation,
            fn=fn,
            input_tables=input_tables,
            output_names=output_names,
            reason=reason,
        )
        self.segments.append(seg)

    def build_plan(self, final_output: str) -> ExecutionPlan:
        return ExecutionPlan(
            segments=list(self.segments),
            final_output=final_output,
        )

    def _collect_read_tables(self, node: IRNode) -> list[str]:
        names: list[str] = []
        self._walk_reads(node, names)
        return list(dict.fromkeys(names))

    def _walk_reads(self, node: IRNode, acc: list[str]):
        if isinstance(node, ReadTable):
            acc.append(node.name)
        elif hasattr(node, "input"):
            self._walk_reads(node.input, acc)
        elif hasattr(node, "inputs") and isinstance(node.inputs, list):
            for inp in node.inputs:
                self._walk_reads(inp, acc)
        elif hasattr(node, "left") and hasattr(node, "right"):
            self._walk_reads(node.left, acc)
            self._walk_reads(node.right, acc)
        # Walk expressions for embedded ScalarSubquery references
        if isinstance(node, (Filter, AddColumn)):
            expr = node.predicate if isinstance(node, Filter) else node.expr
            self._walk_expr_reads(expr, acc)

    def _walk_expr_reads(self, expr, acc: list[str]):
        if isinstance(expr, ScalarSubquery):
            self._walk_reads(expr.agg_node, acc)
        elif isinstance(expr, BinOp):
            self._walk_expr_reads(expr.left, acc)
            self._walk_expr_reads(expr.right, acc)
        elif isinstance(expr, UnaryOp):
            self._walk_expr_reads(expr.operand, acc)
        elif isinstance(expr, IfThenExpr):
            self._walk_expr_reads(expr.condition, acc)
            self._walk_expr_reads(expr.then_expr, acc)
            self._walk_expr_reads(expr.else_expr, acc)
        elif isinstance(expr, CastExpr):
            self._walk_expr_reads(expr.input, acc)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                self._walk_expr_reads(arg, acc)


# ---------------------------------------------------------------------------
# TracedSeries — proxy for pd.Series
# ---------------------------------------------------------------------------


class TracedSeries:
    """
    Proxy for pd.Series. Records expressions in the IR graph.
    Materializing operations (bool, item, aggregations) trigger graph breaks.
    """

    def __init__(
        self,
        ctx: TraceContext,
        source_ir: IRNode,
        column_name: str | None = None,
        expr: Expr | None = None,
        dtype: DType = DType.FLOAT64,
    ):
        self._ctx = ctx
        self._source_ir = source_ir
        self._column_name = column_name
        self._expr = expr or (ColRef(column_name) if column_name else None)
        self._dtype = dtype

    # -- Comparison operators --
    def __gt__(self, other):
        return self._binop("gt", other, DType.BOOL)

    def __ge__(self, other):
        return self._binop("gte", other, DType.BOOL)

    def __lt__(self, other):
        return self._binop("lt", other, DType.BOOL)

    def __le__(self, other):
        return self._binop("lte", other, DType.BOOL)

    def __eq__(self, other):
        return self._binop("eq", other, DType.BOOL)

    def __ne__(self, other):
        return self._binop("ne", other, DType.BOOL)

    # -- Arithmetic --
    def __add__(self, other):
        return self._binop("add", other, self._dtype)

    def __radd__(self, other):
        return self._binop("add", other, self._dtype)

    def __sub__(self, other):
        return self._binop("sub", other, self._dtype)

    def __rsub__(self, other):
        expr = BinOp("sub", _to_expr(other), self._expr)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=self._dtype)

    def __mul__(self, other):
        return self._binop("mul", other, self._dtype)

    def __rmul__(self, other):
        return self._binop("mul", other, self._dtype)

    def __truediv__(self, other):
        return self._binop("div", other, DType.FLOAT64)

    # -- Boolean --
    def __and__(self, other):
        return self._binop("and", other, DType.BOOL)

    def __or__(self, other):
        return self._binop("or", other, DType.BOOL)

    def __invert__(self):
        return TracedSeries(
            self._ctx,
            self._source_ir,
            expr=UnaryOp("not", self._expr),
            dtype=DType.BOOL,
        )

    # -- Materializing operations (GRAPH BREAKS) --

    def __bool__(self):
        return bool(self._materialize_scalar())

    def item(self):
        return self._materialize_scalar()

    def sum(self):
        return self._deferred_agg("sum", self._dtype)

    def mean(self):
        return self._deferred_agg("avg", DType.FLOAT64)

    def min(self):
        return self._deferred_agg("min", self._dtype)

    def max(self):
        return self._deferred_agg("max", self._dtype)

    def count(self):
        return self._deferred_agg("count", DType.INT64)

    def std(self):
        return self._deferred_agg("std", DType.FLOAT64)

    def var(self):
        return self._deferred_agg("var", DType.FLOAT64)

    # -- Non-materializing pandas methods --

    def isin(self, values) -> TracedSeries:
        if not values:
            return TracedSeries(
                self._ctx,
                self._source_ir,
                expr=Literal(False, DType.BOOL),
                dtype=DType.BOOL,
            )
        expr = SingularOrList(self._expr, [Literal(v) for v in values])
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=DType.BOOL)

    def isna(self) -> TracedSeries:
        return TracedSeries(
            self._ctx,
            self._source_ir,
            expr=UnaryOp("is_null", self._expr),
            dtype=DType.BOOL,
        )

    def notna(self) -> TracedSeries:
        return TracedSeries(
            self._ctx,
            self._source_ir,
            expr=UnaryOp("not", UnaryOp("is_null", self._expr)),
            dtype=DType.BOOL,
        )

    def fillna(self, value) -> TracedSeries:
        return TracedSeries(
            self._ctx,
            self._source_ir,
            column_name=self._column_name,
            expr=BinOp("coalesce", self._expr, _to_expr(value)),
            dtype=self._dtype,
        )

    def abs(self) -> TracedSeries:
        return TracedSeries(
            self._ctx,
            self._source_ir,
            expr=UnaryOp("abs", self._expr),
            dtype=self._dtype,
        )

    def __neg__(self) -> TracedSeries:
        return TracedSeries(
            self._ctx,
            self._source_ir,
            expr=UnaryOp("negate", self._expr),
            dtype=self._dtype,
        )

    def __abs__(self) -> TracedSeries:
        return self.abs()

    def between(self, left, right, inclusive="both") -> TracedSeries:
        if inclusive == "both":
            lo = BinOp("gte", self._expr, _to_expr(left))
            hi = BinOp("lte", self._expr, _to_expr(right))
        elif inclusive == "neither":
            lo = BinOp("gt", self._expr, _to_expr(left))
            hi = BinOp("lt", self._expr, _to_expr(right))
        elif inclusive == "left":
            lo = BinOp("gte", self._expr, _to_expr(left))
            hi = BinOp("lt", self._expr, _to_expr(right))
        else:  # "right"
            lo = BinOp("gt", self._expr, _to_expr(left))
            hi = BinOp("lte", self._expr, _to_expr(right))
        return TracedSeries(
            self._ctx,
            self._source_ir,
            expr=BinOp("and", lo, hi),
            dtype=DType.BOOL,
        )

    # -- Accessor properties --

    @property
    def dt(self) -> TracedDatetimeAccessor:
        return TracedDatetimeAccessor(self)

    @property
    def str(self) -> TracedStringAccessor:
        return TracedStringAccessor(self)

    # -- Graph-breaking Series methods --

    def apply(self, func, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].apply(func, **kwargs)

    def map(self, func, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].map(func, **kwargs)

    def unique(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].unique()

    def nunique(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].nunique()

    def value_counts(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].value_counts(**kwargs)

    def __len__(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return len(df[col])

    # -- Internal --

    def _binop(self, op, other, result_dtype):
        return TracedSeries(
            self._ctx,
            self._source_ir,
            expr=BinOp(op, self._expr, _to_expr(other)),
            dtype=result_dtype,
        )

    def _deferred_agg(self, func, result_dtype):
        col = self._column_name or next(iter(self._source_ir.output_schema().columns))
        return DeferredScalar(self._ctx, self._source_ir, col, func, result_dtype)

    def _materialize_scalar(self):
        _, df = self._ctx.materialize(self._source_ir)
        backend = PandasBackend()
        if self._column_name:
            if len(df) == 1:
                return df[self._column_name].iloc[0]
            return df[self._column_name]
        return backend._eval_expr(self._expr, df)

    def _materialize_agg(self, func):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        agg_map = {
            "sum": "sum",
            "avg": "mean",
            "mean": "mean",
            "min": "min",
            "max": "max",
            "count": "count",
            "std": "std",
            "var": "var",
        }
        return getattr(df[col], agg_map[func])()

    def __repr__(self):
        if self._column_name:
            return f"TracedSeries({self._column_name})"
        return f"TracedSeries({explain_expr(self._expr)})"


# ---------------------------------------------------------------------------
# Accessors — .dt and .str on TracedSeries
# ---------------------------------------------------------------------------


class TracedDatetimeAccessor:
    """Proxy for Series.dt — maps to extract() IR expressions."""

    def __init__(self, series: TracedSeries):
        self._series = series

    def _extract(self, component: str) -> TracedSeries:
        expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": component},
            return_dtype=DType.INT64,
        )
        return TracedSeries(
            self._series._ctx,
            self._series._source_ir,
            expr=expr,
            dtype=DType.INT64,
        )

    @property
    def year(self) -> TracedSeries:
        return self._extract("YEAR")

    @property
    def month(self) -> TracedSeries:
        return self._extract("MONTH")

    @property
    def day(self) -> TracedSeries:
        return self._extract("DAY")

    @property
    def hour(self) -> TracedSeries:
        return self._extract("HOUR")

    @property
    def minute(self) -> TracedSeries:
        return self._extract("MINUTE")

    @property
    def second(self) -> TracedSeries:
        return self._extract("SECOND")

    @property
    def quarter(self) -> TracedSeries:
        return self._extract("QUARTER")

    @property
    def dayofweek(self) -> TracedSeries:
        return self._extract("MONDAY_DAY_OF_WEEK")

    @property
    def day_of_week(self) -> TracedSeries:
        return self._extract("MONDAY_DAY_OF_WEEK")

    @property
    def dayofyear(self) -> TracedSeries:
        return self._extract("DAY_OF_YEAR")

    @property
    def day_of_year(self) -> TracedSeries:
        return self._extract("DAY_OF_YEAR")


class TracedStringAccessor:
    """Proxy for Series.str — maps to str_* IR expressions."""

    def __init__(self, series: TracedSeries):
        self._series = series

    def _str_func(
        self,
        method: str,
        *extra_args,
        return_dtype: DType = DType.STRING,
        **options,
    ) -> TracedSeries:
        args = [self._series._expr] + [_to_expr(a) for a in extra_args]
        expr = FunctionCall(
            f"str_{method}", args, options=options, return_dtype=return_dtype
        )
        return TracedSeries(
            self._series._ctx,
            self._series._source_ir,
            expr=expr,
            dtype=return_dtype,
        )

    def upper(self) -> TracedSeries:
        return self._str_func("upper")

    def lower(self) -> TracedSeries:
        return self._str_func("lower")

    def strip(self) -> TracedSeries:
        return self._str_func("strip")

    def lstrip(self) -> TracedSeries:
        return self._str_func("lstrip")

    def rstrip(self) -> TracedSeries:
        return self._str_func("rstrip")

    def len(self) -> TracedSeries:
        return self._str_func("len", return_dtype=DType.INT64)

    def contains(self, pat, regex=True) -> TracedSeries:
        return self._str_func(
            "contains", pat, return_dtype=DType.BOOL, regex=str(regex).lower()
        )

    def startswith(self, pat) -> TracedSeries:
        return self._str_func("startswith", pat, return_dtype=DType.BOOL)

    def endswith(self, pat) -> TracedSeries:
        return self._str_func("endswith", pat, return_dtype=DType.BOOL)

    def replace(self, pat, repl, regex=True) -> TracedSeries:
        return self._str_func("replace", pat, repl, regex=str(regex).lower())

    def slice(self, start=None, stop=None) -> TracedSeries:
        args = []
        if start is not None:
            args.append(start)
        if stop is not None:
            args.append(stop)
        return self._str_func("slice", *args)


# ---------------------------------------------------------------------------
# TracedRolling / TracedExpanding — window proxies (graph break)
# ---------------------------------------------------------------------------


class TracedRolling:
    """Proxy for DataFrame.rolling() — builds Window IR node."""

    def __init__(self, ctx, source_ir, window, **kwargs):
        self._ctx = ctx
        self._source_ir = source_ir
        self._window = window
        self._kwargs = kwargs

    def _apply(self, method, **method_kwargs):
        schema = self._source_ir.output_schema()
        numeric_cols = [
            col for col, dtype in schema.columns.items() if dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            # No numeric columns — fall back to graph break
            _, df = self._ctx.materialize(self._source_ir)
            result = getattr(df.rolling(self._window, **self._kwargs), method)(
                **method_kwargs
            )
            if isinstance(result, pd.DataFrame):
                name = self._ctx.next_materialized_name()
                self._ctx.register_table(name, result)
                return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
            return result

        func = {"mean": "avg"}.get(method, method)
        window_funcs = [(col, col, func) for col in numeric_cols]
        spec = WindowSpec(kind="rows", lower_offset=self._window - 1, upper_offset=0)
        node = Window(self._source_ir, window_funcs, spec)
        return TracedDataFrame(self._ctx, node)

    def mean(self, **kwargs):
        return self._apply("mean", **kwargs)

    def sum(self, **kwargs):
        return self._apply("sum", **kwargs)

    def std(self, **kwargs):
        return self._apply("std", **kwargs)

    def var(self, **kwargs):
        return self._apply("var", **kwargs)

    def min(self, **kwargs):
        return self._apply("min", **kwargs)

    def max(self, **kwargs):
        return self._apply("max", **kwargs)

    def count(self, **kwargs):
        return self._apply("count", **kwargs)


class TracedExpanding:
    """Proxy for DataFrame.expanding() — builds Window IR node."""

    def __init__(self, ctx, source_ir, **kwargs):
        self._ctx = ctx
        self._source_ir = source_ir
        self._kwargs = kwargs

    def _apply(self, method, **method_kwargs):
        schema = self._source_ir.output_schema()
        numeric_cols = [
            col for col, dtype in schema.columns.items() if dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            _, df = self._ctx.materialize(self._source_ir)
            result = getattr(df.expanding(**self._kwargs), method)(**method_kwargs)
            if isinstance(result, pd.DataFrame):
                name = self._ctx.next_materialized_name()
                self._ctx.register_table(name, result)
                return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
            return result

        func = {"mean": "avg"}.get(method, method)
        window_funcs = [(col, col, func) for col in numeric_cols]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(self._source_ir, window_funcs, spec)
        return TracedDataFrame(self._ctx, node)

    def mean(self, **kwargs):
        return self._apply("mean", **kwargs)

    def sum(self, **kwargs):
        return self._apply("sum", **kwargs)

    def std(self, **kwargs):
        return self._apply("std", **kwargs)

    def var(self, **kwargs):
        return self._apply("var", **kwargs)

    def min(self, **kwargs):
        return self._apply("min", **kwargs)

    def max(self, **kwargs):
        return self._apply("max", **kwargs)


# ---------------------------------------------------------------------------
# TracedGroupBy — proxy for DataFrameGroupBy
# ---------------------------------------------------------------------------


class TracedGroupBy:
    def __init__(
        self,
        ctx: TraceContext,
        source_ir: IRNode,
        keys: list[str],
        source_schema: Schema,
    ):
        self._ctx = ctx
        self._source_ir = source_ir
        self._keys = keys
        self._source_schema = source_schema

    def __getitem__(self, key):
        if isinstance(key, str):
            return TracedGroupBySeries(
                self._ctx,
                self._source_ir,
                self._keys,
                key,
                self._source_schema,
            )
        elif isinstance(key, list):
            return TracedGroupByMulti(
                self._ctx,
                self._source_ir,
                self._keys,
                key,
                self._source_schema,
            )
        raise TypeError(f"GroupBy key must be str or list, got {type(key)}")

    def agg(self, agg_dict=None, **kwargs) -> TracedDataFrame:
        agg_specs = []
        if agg_dict and isinstance(agg_dict, dict):
            for col, func in agg_dict.items():
                if isinstance(func, str):
                    agg_specs.append((col, col, func))
                elif isinstance(func, list):
                    agg_specs.extend((f"{col}_{f}", col, f) for f in func)
        for out_name, (src_col, func) in kwargs.items():
            agg_specs.append((out_name, src_col, func))
        node = Aggregate(self._source_ir, self._keys, agg_specs)
        return TracedDataFrame(self._ctx, node)

    def sum(self):
        return self._simple_agg("sum")

    def mean(self):
        return self._simple_agg("avg")

    def count(self):
        return self._simple_agg("count")

    def min(self):
        return self._simple_agg("min")

    def max(self):
        return self._simple_agg("max")

    def std(self):
        return self._simple_agg("std")

    def var(self):
        return self._simple_agg("var")

    def first(self):
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys, as_index=False).first()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def last(self):
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys, as_index=False).last()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def _simple_agg(self, func):
        agg_specs = []
        for col, dtype in self._source_schema.columns.items():
            if col not in self._keys:
                if func == "count" or dtype in NUMERIC_DTYPES:
                    agg_specs.append((col, col, func))
        return TracedDataFrame(
            self._ctx, Aggregate(self._source_ir, self._keys, agg_specs)
        )

    def size(self):
        first_col = next(iter(self._source_schema.columns.keys()))
        return TracedDataFrame(
            self._ctx,
            Aggregate(
                self._source_ir,
                self._keys,
                [("size", first_col, "count")],
            ),
        )


class TracedGroupBySeries:
    def __init__(self, ctx, source_ir, keys, column, source_schema):
        self._ctx = ctx
        self._source_ir = source_ir
        self._keys = keys
        self._column = column
        self._source_schema = source_schema

    def sum(self):
        return self._agg("sum")

    def mean(self):
        return self._agg("avg")

    def count(self):
        return self._agg("count")

    def min(self):
        return self._agg("min")

    def max(self):
        return self._agg("max")

    def std(self):
        return self._agg("std")

    def var(self):
        return self._agg("var")

    def _agg(self, func):
        return TracedDataFrame(
            self._ctx,
            Aggregate(
                self._source_ir,
                self._keys,
                [(self._column, self._column, func)],
            ),
        )


class TracedGroupByMulti:
    def __init__(self, ctx, source_ir, keys, columns, source_schema):
        self._ctx = ctx
        self._source_ir = source_ir
        self._keys = keys
        self._columns = columns
        self._source_schema = source_schema

    def sum(self):
        return self._agg("sum")

    def mean(self):
        return self._agg("avg")

    def std(self):
        return self._agg("std")

    def var(self):
        return self._agg("var")

    def _agg(self, func):
        agg_specs = [(c, c, func) for c in self._columns]
        return TracedDataFrame(
            self._ctx, Aggregate(self._source_ir, self._keys, agg_specs)
        )


# ---------------------------------------------------------------------------
# TracedDataFrame — the main proxy
# ---------------------------------------------------------------------------


class TracedDataFrame:
    """
    Proxy for pd.DataFrame. Records operations as IR nodes.

    When a graph-breaking operation is called (len, bool, shape, values,
    to_csv, apply, etc.), we MATERIALIZE the computation so far on the
    backend, run the breaking op in eager Python, then resume tracing
    with a fresh ReadTable from the materialized result.
    """

    def __init__(self, ctx: TraceContext, ir_node: IRNode):
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_ir", ir_node)

    # -- Helpers --

    def _materialize(self) -> pd.DataFrame:
        name, df = self._ctx.materialize(self._ir)
        schema = infer_schema(df)
        new_ir = ReadTable(name, schema)
        object.__setattr__(self, "_ir", new_ir)
        return df

    def _ensure_materialized(self) -> pd.DataFrame:
        if isinstance(self._ir, ReadTable) and self._ir.name in self._ctx.tables:
            return self._ctx.tables[self._ir.name]
        return self._materialize()

    # -- Column access --

    def __getitem__(self, key):
        if isinstance(key, str):
            schema = self._ir.output_schema()
            dtype = schema.columns.get(key, DType.STRING)
            return TracedSeries(self._ctx, self._ir, column_name=key, dtype=dtype)
        elif isinstance(key, list):
            return TracedDataFrame(self._ctx, Project(self._ir, key))
        elif isinstance(key, TracedSeries):
            if key._dtype != DType.BOOL:
                raise TypeError("Boolean indexing requires boolean series")
            return TracedDataFrame(self._ctx, Filter(self._ir, key._expr))
        elif isinstance(key, pd.Series):
            df = self._materialize()
            result = df[key]
            result_name = self._ctx.next_materialized_name()
            self._ctx.register_table(result_name, result)
            return TracedDataFrame(
                self._ctx, ReadTable(result_name, infer_schema(result))
            )
        else:
            df = self._materialize()
            result = df[key]
            if isinstance(result, pd.DataFrame):
                result_name = self._ctx.next_materialized_name()
                self._ctx.register_table(result_name, result)
                return TracedDataFrame(
                    self._ctx,
                    ReadTable(result_name, infer_schema(result)),
                )
            return result

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if isinstance(value, TracedSeries):
                expr, dtype = value._expr, value._dtype
                node = AddColumn(self._ir, key, expr, dtype)
                object.__setattr__(self, "_ir", node)
            elif isinstance(value, (int, float, str, bool)):
                lit = Literal(value)
                node = AddColumn(self._ir, key, lit, lit.dtype)
                object.__setattr__(self, "_ir", node)
            else:
                df = self._ensure_materialized().copy()
                df[key] = value
                new_name = self._ctx.next_materialized_name()
                self._ctx.register_table(new_name, df)
                object.__setattr__(self, "_ir", ReadTable(new_name, infer_schema(df)))
        else:
            df = self._ensure_materialized().copy()
            df[key] = value
            new_name = self._ctx.next_materialized_name()
            self._ctx.register_table(new_name, df)
            object.__setattr__(self, "_ir", ReadTable(new_name, infer_schema(df)))

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Use df['{name}'] = ... instead of df.{name} = ...")

    def __getattr__(self, name):
        try:
            ir = object.__getattribute__(self, "_ir")
            ctx = object.__getattribute__(self, "_ctx")
        except AttributeError:
            raise AttributeError(name) from None
        schema = ir.output_schema()
        if name in schema.columns:
            return TracedSeries(ctx, ir, column_name=name, dtype=schema.columns[name])
        raise AttributeError(
            f"No column '{name}'. Available: {list(schema.columns.keys())}"
        )

    # -- Pandas methods (traced) --

    def assign(self, **kwargs) -> TracedDataFrame:
        result = self
        for name, value in kwargs.items():
            if isinstance(value, TracedSeries):
                node = AddColumn(result._ir, name, value._expr, value._dtype)
            elif callable(value):
                traced_result = value(result)
                if isinstance(traced_result, TracedSeries):
                    node = AddColumn(
                        result._ir,
                        name,
                        traced_result._expr,
                        traced_result._dtype,
                    )
                else:
                    df = result._materialize()
                    df[name] = value(df)
                    result_name = result._ctx.next_materialized_name()
                    result._ctx.register_table(result_name, df)
                    node = ReadTable(result_name, infer_schema(df))
            elif isinstance(value, (int, float, str, bool)):
                lit = Literal(value)
                node = AddColumn(result._ir, name, lit, lit.dtype)
            else:
                df = result._materialize()
                df[name] = value
                result_name = result._ctx.next_materialized_name()
                result._ctx.register_table(result_name, df)
                node = ReadTable(result_name, infer_schema(df))
            result = TracedDataFrame(result._ctx, node)
        return result

    def drop(self, columns=None, axis=None, **kwargs) -> TracedDataFrame:
        if columns is None and axis == 1:
            columns = kwargs.get("labels", [])
        if columns is None:
            df = self._materialize()
            return TracedDataFrame(
                self._ctx,
                ReadTable(
                    self._ctx.next_materialized_name(),
                    infer_schema(df.drop(**kwargs)),
                ),
            )
        if isinstance(columns, str):
            columns = [columns]
        schema = self._ir.output_schema()
        remaining = [c for c in schema.column_names() if c not in columns]
        return TracedDataFrame(self._ctx, Project(self._ir, remaining))

    def rename(self, columns=None, **kwargs) -> TracedDataFrame:
        if columns is None:
            df = self._materialize()
            result = df.rename(**kwargs)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return TracedDataFrame(self._ctx, RenameColumns(self._ir, columns))

    def groupby(self, by, **kwargs) -> TracedGroupBy:
        if isinstance(by, str):
            keys = [by]
        elif isinstance(by, list):
            keys = by
        else:
            raise TypeError(f"groupby key must be str or list, got {type(by)}")
        return TracedGroupBy(self._ctx, self._ir, keys, self._ir.output_schema())

    def merge(
        self,
        right,
        on=None,
        left_on=None,
        right_on=None,
        how="inner",
        **kwargs,
    ) -> TracedDataFrame:
        if not isinstance(right, TracedDataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, right)
            right = TracedDataFrame(self._ctx, ReadTable(name, infer_schema(right)))
        if on is not None:
            left_on = [on] if isinstance(on, str) else list(on)
            right_on = left_on
        else:
            if isinstance(left_on, str):
                left_on = [left_on]
            if isinstance(right_on, str):
                right_on = [right_on]
        return TracedDataFrame(
            self._ctx,
            Join(self._ir, right._ir, left_on, right_on, how),
        )

    def sort_values(self, by, ascending=True, **kwargs) -> TracedDataFrame:
        if isinstance(by, str):
            by = [by]
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        return TracedDataFrame(
            self._ctx, Sort(self._ir, list(zip(by, ascending, strict=True)))
        )

    def head(self, n=5) -> TracedDataFrame:
        return TracedDataFrame(self._ctx, Limit(self._ir, n))

    def nlargest(self, n, columns) -> TracedDataFrame:
        if isinstance(columns, str):
            columns = [columns]
        return TracedDataFrame(
            self._ctx,
            Limit(Sort(self._ir, [(c, False) for c in columns]), n),
        )

    def nsmallest(self, n, columns) -> TracedDataFrame:
        if isinstance(columns, str):
            columns = [columns]
        return TracedDataFrame(
            self._ctx,
            Limit(Sort(self._ir, [(c, True) for c in columns]), n),
        )

    def query(self, expr_str: str) -> TracedDataFrame:
        predicate = _parse_query_string(expr_str, self._ir.output_schema())
        return TracedDataFrame(self._ctx, Filter(self._ir, predicate))

    def dropna(self, subset=None, **kwargs) -> TracedDataFrame:
        schema = self._ir.output_schema()
        cols = subset if subset else list(schema.columns.keys())
        if isinstance(cols, str):
            cols = [cols]
        predicates = [UnaryOp("not", UnaryOp("is_null", ColRef(c))) for c in cols]
        combined = predicates[0]
        for p in predicates[1:]:
            combined = BinOp("and", combined, p)
        return TracedDataFrame(self._ctx, Filter(self._ir, combined))

    def fillna(self, value) -> TracedDataFrame:
        schema = self._ir.output_schema()
        if isinstance(value, dict):
            fill_map = value
        else:
            fill_map = dict.fromkeys(schema.column_names(), value)
        result = self
        for col_name, fill_val in fill_map.items():
            if col_name in schema.columns:
                expr = BinOp("coalesce", ColRef(col_name), Literal(fill_val))
                dtype = schema.columns[col_name]
                result = TracedDataFrame(
                    result._ctx,
                    AddColumn(result._ir, col_name, expr, dtype),
                )
        return result

    def drop_duplicates(self, subset=None, keep="first", **kwargs) -> TracedDataFrame:
        schema = self._ir.output_schema()
        all_cols = list(schema.columns.keys())
        cols = (
            all_cols
            if subset is None
            else ([subset] if isinstance(subset, str) else list(subset))
        )
        # Trace when keep="first" and subset covers all columns
        if keep == "first" and set(cols) == set(all_cols):
            return TracedDataFrame(self._ctx, Distinct(self._ir, cols))
        # Graph break for partial subsets or non-first keep
        df = self._materialize()
        result = df.drop_duplicates(subset=subset, keep=keep, **kwargs).reset_index(
            drop=True
        )
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def rolling(self, window, **kwargs) -> TracedRolling:
        return TracedRolling(self._ctx, self._ir, window, **kwargs)

    def expanding(self, **kwargs) -> TracedExpanding:
        return TracedExpanding(self._ctx, self._ir, **kwargs)

    def pivot_table(self, **kwargs) -> TracedDataFrame:
        df = self._materialize()
        result = df.pivot_table(**kwargs).reset_index()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def melt(self, **kwargs) -> TracedDataFrame:
        df = self._materialize()
        result = df.melt(**kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def stack(self, *args, **kwargs):
        df = self._materialize()
        result = df.stack(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return result

    def unstack(self, *args, **kwargs):
        df = self._materialize()
        result = df.unstack(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return result

    def astype(self, dtype) -> TracedDataFrame:
        schema = self._ir.output_schema()
        if isinstance(dtype, dict):
            cast_map = dtype
        elif isinstance(dtype, str):
            cast_map = dict.fromkeys(schema.column_names(), dtype)
        else:
            # Unsupported dtype form — fall back to graph break
            df = self._materialize()
            result = df.astype(dtype)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

        ir = self._ir
        for col_name, target_str in cast_map.items():
            if col_name not in schema.columns:
                continue
            target_dtype = pandas_dtype_to_ir(str(target_str))
            # If the dtype maps to STRING (unknown), fall back to graph break
            if target_dtype == DType.STRING and str(target_str) not in (
                "object",
                "string",
                "str",
                "string[python]",
                "string[pyarrow]",
            ):
                df = self._materialize()
                result = df.astype(dtype)
                name = self._ctx.next_materialized_name()
                self._ctx.register_table(name, result)
                return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
            ir = AddColumn(
                ir, col_name, CastExpr(ColRef(col_name), target_dtype), target_dtype
            )
        return TracedDataFrame(self._ctx, ir)

    def where(self, cond, other=None, **kwargs) -> TracedDataFrame:
        if isinstance(cond, TracedSeries) and isinstance(
            other, (int, float, str, bool, type(None))
        ):
            other_val = other if other is not None else float("nan")
            schema = self._ir.output_schema()
            ir = self._ir
            for col_name, dtype in schema.columns.items():
                ir = AddColumn(
                    ir,
                    col_name,
                    IfThenExpr(cond._expr, ColRef(col_name), Literal(other_val)),
                    dtype,
                )
            return TracedDataFrame(self._ctx, ir)
        # Fall back to graph break for complex conditions
        df = self._materialize()
        if isinstance(cond, TracedDataFrame):
            cond = cond._materialize()
        result = df.where(cond, other, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def mask(self, cond, other=None, **kwargs) -> TracedDataFrame:
        if isinstance(cond, TracedSeries) and isinstance(
            other, (int, float, str, bool, type(None))
        ):
            other_val = other if other is not None else float("nan")
            inverted = UnaryOp("not", cond._expr)
            schema = self._ir.output_schema()
            ir = self._ir
            for col_name, dtype in schema.columns.items():
                ir = AddColumn(
                    ir,
                    col_name,
                    IfThenExpr(inverted, ColRef(col_name), Literal(other_val)),
                    dtype,
                )
            return TracedDataFrame(self._ctx, ir)
        # Fall back to graph break for complex conditions
        df = self._materialize()
        if isinstance(cond, TracedDataFrame):
            cond = cond._materialize()
        result = df.mask(cond, other, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def reset_index(self, drop=True, **kwargs):
        return self  # No-op in relational algebra

    def copy(self):
        return self

    # -- GRAPH BREAK: materializing operations --

    def __len__(self) -> int:
        df = self._ensure_materialized()
        return len(df)

    def __bool__(self) -> bool:
        df = self._ensure_materialized()
        return len(df) > 0

    @property
    def shape(self) -> tuple[int, int]:
        df = self._ensure_materialized()
        return df.shape

    @property
    def values(self) -> np.ndarray:
        df = self._ensure_materialized()
        return df.values

    def to_numpy(self) -> np.ndarray:
        return self.values

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def iloc(self):
        return _IlocProxy(self._ctx, self)

    @property
    def loc(self):
        df = self._ensure_materialized()
        return _LocProxy(self._ctx, df)

    def iterrows(self):
        df = self._ensure_materialized()
        return df.iterrows()

    def itertuples(self, **kwargs):
        df = self._ensure_materialized()
        return df.itertuples(**kwargs)

    def apply(self, func, axis=0, **kwargs):
        df = self._ensure_materialized()
        result = df.apply(func, axis=axis, **kwargs)
        if isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return result

    def pipe(self, func, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception:
            df = self._ensure_materialized()
            result = func(df, *args, **kwargs)
            if isinstance(result, pd.DataFrame):
                name = self._ctx.next_materialized_name()
                self._ctx.register_table(name, result)
                return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
            return result

    def to_csv(self, *args, **kwargs):
        df = self._ensure_materialized()
        return df.to_csv(*args, **kwargs)

    def to_parquet(self, *args, **kwargs):
        df = self._ensure_materialized()
        return df.to_parquet(*args, **kwargs)

    def describe(self, **kwargs):
        df = self._ensure_materialized()
        return df.describe(**kwargs)

    def value_counts(self, subset=None, **kwargs):
        df = self._ensure_materialized()
        return df.value_counts(subset=subset, **kwargs)

    def info(self, **kwargs):
        df = self._ensure_materialized()
        return df.info(**kwargs)

    def __iter__(self):
        df = self._ensure_materialized()
        return iter(df)

    # -- Schema inspection (no materialization needed) --

    @property
    def columns(self):
        return pd.Index(self._ir.output_schema().column_names())

    @property
    def dtypes(self):
        schema = self._ir.output_schema()
        dtype_map = {
            DType.INT8: np.dtype("int8"),
            DType.INT16: np.dtype("int16"),
            DType.INT32: np.dtype("int32"),
            DType.INT64: np.dtype("int64"),
            DType.UINT8: np.dtype("uint8"),
            DType.UINT16: np.dtype("uint16"),
            DType.UINT32: np.dtype("uint32"),
            DType.UINT64: np.dtype("uint64"),
            DType.FLOAT32: np.dtype("float32"),
            DType.FLOAT64: np.dtype("float64"),
            DType.STRING: np.dtype("object"),
            DType.BINARY: np.dtype("object"),
            DType.BOOL: np.dtype("bool"),
            DType.DATE: np.dtype("datetime64[ns]"),
            DType.TIME: np.dtype("object"),
            DType.TIMESTAMP: np.dtype("datetime64[ns]"),
            DType.TIMESTAMP_TZ: np.dtype("datetime64[ns]"),
            DType.TIMEDELTA: np.dtype("timedelta64[ns]"),
            DType.DECIMAL: np.dtype("float64"),
        }
        return pd.Series(
            {c: dtype_map.get(d, np.dtype("object")) for c, d in schema.columns.items()}
        )

    def __repr__(self):
        schema = self._ir.output_schema()
        cols = ", ".join(f"{n}: {t.name}" for n, t in schema.columns.items())
        return f"TracedDataFrame[{cols}]"


class _IlocProxy:
    def __init__(self, ctx, traced_df):
        self._ctx = ctx
        self._traced_df = traced_df

    def __getitem__(self, key):
        # Try IR-based slicing for simple slice patterns
        if isinstance(key, slice) and key.step is None:
            start = key.start
            stop = key.stop
            # df.iloc[:n] → Limit(n)
            if start is None and isinstance(stop, int) and stop > 0:
                return TracedDataFrame(self._ctx, Limit(self._traced_df._ir, stop))
            # df.iloc[start:stop] → Limit(stop-start, offset=start)
            if (
                isinstance(start, int)
                and isinstance(stop, int)
                and start >= 0
                and stop > start
            ):
                return TracedDataFrame(
                    self._ctx,
                    Limit(self._traced_df._ir, stop - start, offset=start),
                )

        # Fall back to materialization for unsupported patterns
        df = self._traced_df._ensure_materialized()
        result = df.iloc[key]
        if isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return result


class _LocProxy:
    def __init__(self, ctx, df):
        self._ctx = ctx
        self._df = df

    def __getitem__(self, key):
        result = self._df.loc[key]
        if isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return result


# ---------------------------------------------------------------------------
# Substrait export helpers
# ---------------------------------------------------------------------------


def _extract_substrait_plans(ctx: TraceContext) -> list[Any]:
    """Extract Substrait Plan protobufs from a TraceContext."""
    plans = []
    for seg in ctx.segments:
        if isinstance(seg, CompiledSegment):
            compiler = SubstraitCompiler()
            plans.append(compiler.compile(seg.ir_node))
    return plans


def _build_connected_plan(ctx: TraceContext) -> ConnectedPlan:
    """Build a ConnectedPlan from a TraceContext with full metadata."""
    stages: list[CompiledStage | GraphBreakStage] = []
    for i, seg in enumerate(ctx.segments):
        if isinstance(seg, CompiledSegment):
            compiler = SubstraitCompiler()
            plan = compiler.compile(seg.ir_node)
            plan_bytes = plan.SerializeToString()
            out_schema = seg.ir_node.output_schema()
            stage_schema = StageSchema(
                table_name=seg.output_table,
                columns={c: d.name for c, d in out_schema.columns.items()},
            )
            stages.append(
                CompiledStage(
                    index=i,
                    plan=plan,
                    plan_bytes=plan_bytes,
                    input_tables=seg.input_tables,
                    output_table=seg.output_table,
                    output_schema=stage_schema,
                    description=seg.description,
                )
            )
        elif isinstance(seg, EagerSegment):
            stages.append(
                GraphBreakStage(
                    index=i,
                    reason=seg.reason or seg.operation,
                    input_tables=seg.input_tables,
                    output_tables=seg.output_names,
                )
            )
    final = f"__mat_{ctx._mat_counter}" if ctx._mat_counter > 0 else ""
    return ConnectedPlan(stages=stages, final_output=final)


def _plans_to_json(plans: list[Any]) -> str:
    """Serialize a list of Substrait Plan protobufs to a JSON string."""
    from google.protobuf.json_format import MessageToJson

    if len(plans) == 1:
        return MessageToJson(plans[0])
    parts = [MessageToJson(p) for p in plans]
    return "[\n" + ",\n".join(parts) + "\n]"


# ---------------------------------------------------------------------------
# @compile decorator
# ---------------------------------------------------------------------------


class CompiledFunction:
    """
    Wraps a user function with JIT compilation.

    First call: traces the function with TracedDataFrame proxies,
    building an ExecutionPlan. Subsequent calls: if guards pass
    (same schema), reuse the plan.
    """

    def __init__(self, fn: Callable, backend: Backend | None = None):
        self._fn = fn
        self._backend = backend
        self._cached_plans: list[tuple[SchemaGuard, ExecutionPlan]] = []
        self._last_ctx: TraceContext | None = None
        functools.update_wrapper(self, fn)

    def _get_backend(self) -> Backend:
        if self._backend is None:
            self._backend = default_backend()
        return self._backend

    def __call__(self, *args: Any, **kwargs: Any) -> pd.DataFrame | Any:
        backend = self._get_backend()

        sig = inspect.signature(self._fn)
        param_names = list(sig.parameters.keys())
        df_args: dict[str, pd.DataFrame] = {
            (param_names[i] if i < len(param_names) else f"input_{i}"): arg
            for i, arg in enumerate(args)
            if isinstance(arg, pd.DataFrame)
        }
        df_args.update(
            {name: arg for name, arg in kwargs.items() if isinstance(arg, pd.DataFrame)}
        )

        # Capture non-DataFrame arguments for guard comparison
        scalar_args = tuple(a for a in args if not isinstance(a, pd.DataFrame))
        scalar_kwargs = {
            k: v for k, v in kwargs.items() if not isinstance(v, pd.DataFrame)
        }

        for guard, plan in self._cached_plans:
            if guard.check(df_args, scalar_args, scalar_kwargs):
                has_eager = any(isinstance(s, EagerSegment) for s in plan.segments)
                if not has_eager:
                    log.info(
                        "Cache HIT: reusing compiled plan (%d segments)",
                        len(plan.segments),
                    )
                    return plan.execute(df_args, backend)
                else:
                    log.info(
                        "Cache HIT (schema match) but plan has eager "
                        "segments — re-tracing"
                    )
                    break

        log.info("Tracing function")
        ctx, result = self._trace(args, kwargs, backend)
        self._last_ctx = ctx

        if isinstance(result, TracedDataFrame):
            final_name, result_df = ctx.materialize(result._ir)
            plan = ctx.build_plan(final_output=final_name)
        elif isinstance(result, DeferredScalar):
            return result._materialize()
        elif isinstance(result, pd.DataFrame):
            result_df = result
            final_name = ctx.next_materialized_name()
            ctx.register_table(final_name, result_df)
            plan = ctx.build_plan(final_output=final_name)
        else:
            return result

        guard_schemas = {name: infer_schema(df) for name, df in df_args.items()}
        guard = SchemaGuard(guard_schemas, scalar_args, scalar_kwargs)
        self._cached_plans.append((guard, plan))

        return result_df

    def _trace(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], backend: Backend
    ) -> tuple[TraceContext, Any]:
        ctx = TraceContext(backend)

        sig = inspect.signature(self._fn)
        param_names = list(sig.parameters.keys())

        traced_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame):
                name = param_names[i] if i < len(param_names) else f"input_{i}"
                schema = infer_schema(arg)
                ctx.register_table(name, arg)
                ir_node = ReadTable(name, schema)
                traced_args.append(TracedDataFrame(ctx, ir_node))
            else:
                traced_args.append(arg)

        traced_kwargs = {}
        for name, arg in kwargs.items():
            if isinstance(arg, pd.DataFrame):
                schema = infer_schema(arg)
                ctx.register_table(name, arg)
                ir_node = ReadTable(name, schema)
                traced_kwargs[name] = TracedDataFrame(ctx, ir_node)
            else:
                traced_kwargs[name] = arg

        global _active_trace_ctx
        _active_trace_ctx = ctx
        pd.concat = _traced_concat
        try:
            result = self._fn(*traced_args, **traced_kwargs)
        finally:
            pd.concat = _original_concat
            _active_trace_ctx = None
        return ctx, result

    def trace(self, *sample_args: Any, **kwargs: Any) -> TraceContext:
        backend = self._get_backend()
        ctx, result = self._trace(sample_args, kwargs, backend)
        self._last_ctx = ctx
        if isinstance(result, TracedDataFrame):
            ctx.materialize(result._ir)
        return ctx

    def explain(self, *sample_args, **kwargs) -> str:
        ctx = self.trace(*sample_args, **kwargs)
        plan = ctx.build_plan(final_output=f"__mat_{ctx._mat_counter}")
        lines = [f"ExecutionPlan for {self._fn.__name__}():"]
        lines.append(f"  Backend: {self._get_backend().name}")
        lines.append(f"  Segments: {len(plan.segments)}")
        lines.append("")
        for i, seg in enumerate(plan.segments):
            if isinstance(seg, CompiledSegment):
                lines.append(f"  [{i}] COMPILED -> {seg.output_table}")
                ir_lines = explain_ir(seg.ir_node, indent=2)
                lines.extend(f"      {line}" for line in ir_lines.split("\n"))
            elif isinstance(seg, EagerSegment):
                lines.append(f"  [{i}] EAGER: {seg.operation} -> {seg.output_names}")
        lines.append(f"\n  Output: {plan.final_output}")
        return "\n".join(lines)

    @property
    def last_context(self) -> TraceContext | None:
        return self._last_ctx

    def to_substrait(self, *sample_args: Any, **kwargs: Any) -> list[Any]:
        """
        Export Substrait plans for the given sample inputs.

        Traces the function with the provided arguments and returns
        a list of ``substrait.plan_pb2.Plan`` protobuf objects, one
        per compiled segment.

        Parameters
        ----------
        *sample_args : positional arguments
            Sample inputs (DataFrames and scalars) for tracing.
        **kwargs : keyword arguments
            Additional keyword arguments passed to the function.

        Returns
        -------
        list of substrait.plan_pb2.Plan
            Substrait plans that can be serialized via
            ``.SerializeToString()`` or inspected directly.
        """
        ctx = self.trace(*sample_args, **kwargs)
        return _extract_substrait_plans(ctx)

    def to_substrait_json(self, *sample_args: Any, **kwargs: Any) -> str:
        """
        Export Substrait plans as a JSON string.

        Parameters
        ----------
        *sample_args : positional arguments
            Sample inputs (DataFrames and scalars) for tracing.
        **kwargs : keyword arguments
            Additional keyword arguments passed to the function.

        Returns
        -------
        str
            JSON representation of all Substrait plans.
        """
        plans = self.to_substrait(*sample_args, **kwargs)
        return _plans_to_json(plans)

    def to_connected_plan(self, *sample_args: Any, **kwargs: Any) -> ConnectedPlan:
        """
        Export a ConnectedPlan with full metadata.

        Returns a linked DAG of compiled stages and graph break stages
        with schemas, connections, and break reasons.

        Parameters
        ----------
        *sample_args : positional arguments
            Sample inputs (DataFrames and scalars) for tracing.
        **kwargs : keyword arguments
            Additional keyword arguments passed to the function.

        Returns
        -------
        ConnectedPlan
        """
        ctx = self.trace(*sample_args, **kwargs)
        return _build_connected_plan(ctx)


@overload
def compile(fn: Callable) -> CompiledFunction: ...


@overload
def compile(
    fn: None = ..., *, backend: Backend | None = ...
) -> Callable[[Callable], CompiledFunction]: ...


def compile(
    fn: Callable | None = None, *, backend: Backend | None = None
) -> CompiledFunction | Callable[[Callable], CompiledFunction]:
    """
    Decorator that JIT-compiles a pandas function to an IR graph.

    Graph breaks are handled transparently via materialization.

    Parameters
    ----------
    fn : callable, optional
        The function to compile. When used as ``@compile`` (without
        parentheses), this is the decorated function.
    backend : Backend or None, default None
        Execution backend. ``None`` selects ``AceroBackend`` when pyarrow
        is available, otherwise ``PandasBackend``.

    Returns
    -------
    CompiledFunction or callable
        When called as ``@compile``, returns a ``CompiledFunction``.
        When called as ``@compile(backend=...)``, returns a decorator
        that produces a ``CompiledFunction``.

    Examples
    --------
    >>> @compile
    ... def process(df):
    ...     return df[df["price"] > 100]

    >>> @compile(backend=PandasBackend())
    ... def process(df):
    ...     return df[df["price"] > 100]
    """
    if fn is not None:
        return CompiledFunction(fn, backend=backend)

    def decorator(f: Callable) -> CompiledFunction:
        return CompiledFunction(f, backend=backend)

    return decorator


# ---------------------------------------------------------------------------
# Context manager API
# ---------------------------------------------------------------------------


class Tracer:
    """
    Context manager for tracing a block of pandas operations.

    Usage:
        with Tracer() as t:
            df = t.input(my_df, "sales")
            df["total"] = df["price"] * df["qty"]
            if len(df) > 0:
                result = df.groupby("region").sum()
            t.output(result)
        plan = t.explain()
    """

    def __init__(self, backend: Backend | None = None):
        self._backend = backend if backend is not None else default_backend()
        self._ctx = TraceContext(self._backend)
        self._output_name: str | None = None

    def __enter__(self):
        global _active_trace_ctx
        _active_trace_ctx = self._ctx
        pd.concat = _traced_concat
        return self

    def __exit__(self, *exc_info):
        global _active_trace_ctx
        pd.concat = _original_concat
        _active_trace_ctx = None
        return False

    def input(self, df: pd.DataFrame, name: str = "input") -> TracedDataFrame:
        schema = infer_schema(df)
        self._ctx.register_table(name, df)
        return TracedDataFrame(self._ctx, ReadTable(name, schema))

    def output(self, result):
        if isinstance(result, TracedDataFrame):
            name, _ = self._ctx.materialize(result._ir)
            self._output_name = name
        elif isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            self._output_name = name

    def result(self) -> pd.DataFrame:
        if self._output_name:
            return self._ctx.tables[self._output_name]
        raise RuntimeError("No output set")

    def explain(self) -> str:
        plan = self._ctx.build_plan(self._output_name)
        return plan.explain()

    def to_substrait(self) -> list[Any]:
        """
        Export Substrait plans from traced operations.

        Returns
        -------
        list of substrait.plan_pb2.Plan
            Substrait plans that can be serialized via
            ``.SerializeToString()`` or inspected directly.
        """
        return _extract_substrait_plans(self._ctx)

    # Keep old name as alias for backward compat
    to_substrait_plans = to_substrait

    def to_substrait_json(self) -> str:
        """
        Export Substrait plans as a JSON string.

        Returns
        -------
        str
            JSON representation of all Substrait plans.
        """
        return _plans_to_json(self.to_substrait())

    def to_connected_plan(self) -> ConnectedPlan:
        """
        Export a ConnectedPlan with full metadata.

        Returns
        -------
        ConnectedPlan
        """
        return _build_connected_plan(self._ctx)
