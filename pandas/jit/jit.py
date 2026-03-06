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
from pandas.jit.compiler import (
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
from pandas.jit.ir import (
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

log = logging.getLogger("pandas.jit")

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


def _ir_contains(ir_node, target) -> bool:
    """Return True if *target* is *ir_node* itself or reachable via its input chain."""
    if ir_node is target:
        return True
    if hasattr(ir_node, "input") and isinstance(ir_node.input, IRNode):
        if _ir_contains(ir_node.input, target):
            return True
    if isinstance(ir_node, Join):
        if _ir_contains(ir_node.left, target) or _ir_contains(ir_node.right, target):
            return True
    if isinstance(ir_node, Union):
        for inp in ir_node.inputs:
            if _ir_contains(inp, target):
                return True
    return False


def _rewrite_col_refs(expr: Expr, mapping: dict[str, str]) -> Expr:
    """Recursively rewrite ColRef names in an expression tree."""
    if isinstance(expr, ColRef):
        return ColRef(mapping.get(expr.name, expr.name))
    elif isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _rewrite_col_refs(expr.left, mapping),
            _rewrite_col_refs(expr.right, mapping),
        )
    elif isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _rewrite_col_refs(expr.operand, mapping))
    elif isinstance(expr, IfThenExpr):
        return IfThenExpr(
            _rewrite_col_refs(expr.condition, mapping),
            _rewrite_col_refs(expr.then_expr, mapping),
            _rewrite_col_refs(expr.else_expr, mapping),
        )
    elif isinstance(expr, CastExpr):
        return CastExpr(_rewrite_col_refs(expr.input, mapping), expr.target_dtype)
    elif isinstance(expr, SingularOrList):
        return SingularOrList(
            _rewrite_col_refs(expr.value, mapping),
            [_rewrite_col_refs(o, mapping) for o in expr.options],
        )
    elif isinstance(expr, FunctionCall):
        return FunctionCall(
            expr.func_name,
            [_rewrite_col_refs(a, mapping) for a in expr.args],
            expr.options,
            expr.return_dtype,
        )
    # Literal, ScalarSubquery — no column refs to rewrite
    return expr


def _composable_window(series_ir, target_ir):
    """Return the Window node if series_ir is a composable Window on target_ir."""
    if isinstance(series_ir, Window) and _ir_contains(target_ir, series_ir.input):
        return series_ir
    return None


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
            ast.FloorDiv: "floordiv",
            ast.Mod: "mod",
            ast.Pow: "pow",
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
        # Short-circuit: ReadTable for an already-registered table.
        # Avoids re-executing through the backend, which would lose
        # pandas index info in Arrow roundtrip (Acero/DataFusion).
        if isinstance(ir_node, ReadTable) and ir_node.name in self.tables:
            return ir_node.name, self.tables[ir_node.name]

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

    # -- Comparison method forms --
    def eq(self, other) -> TracedSeries:
        return self._binop("eq", other, DType.BOOL)

    def ne(self, other) -> TracedSeries:
        return self._binop("ne", other, DType.BOOL)

    def lt(self, other) -> TracedSeries:
        return self._binop("lt", other, DType.BOOL)

    def le(self, other) -> TracedSeries:
        return self._binop("lte", other, DType.BOOL)

    def gt(self, other) -> TracedSeries:
        return self._binop("gt", other, DType.BOOL)

    def ge(self, other) -> TracedSeries:
        return self._binop("gte", other, DType.BOOL)

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

    def __rtruediv__(self, other):
        expr = BinOp("div", _to_expr(other), self._expr)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=DType.FLOAT64)

    def __floordiv__(self, other):
        return self._binop("floordiv", other, self._dtype)

    def __rfloordiv__(self, other):
        expr = BinOp("floordiv", _to_expr(other), self._expr)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=self._dtype)

    def __mod__(self, other):
        return self._binop("mod", other, self._dtype)

    def __rmod__(self, other):
        expr = BinOp("mod", _to_expr(other), self._expr)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=self._dtype)

    def __pow__(self, other):
        return self._binop("pow", other, DType.FLOAT64)

    def __rpow__(self, other):
        expr = BinOp("pow", _to_expr(other), self._expr)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=DType.FLOAT64)

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

    def any(self, **kwargs):
        return self._deferred_agg("max", DType.BOOL)

    def all(self, **kwargs):
        return self._deferred_agg("min", DType.BOOL)

    def median(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].median(**kwargs)

    def quantile(self, q=0.5, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].quantile(q=q, **kwargs)

    def prod(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].prod(**kwargs)

    product = prod

    def sem(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].sem(**kwargs)

    def skew(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].skew(**kwargs)

    def kurt(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].kurt(**kwargs)

    kurtosis = kurt

    @property
    def name(self):
        return self._column_name

    @property
    def dtype(self):
        _DTYPE_MAP = {
            DType.INT8: "int8",
            DType.INT16: "int16",
            DType.INT32: "int32",
            DType.INT64: "int64",
            DType.UINT8: "uint8",
            DType.UINT16: "uint16",
            DType.UINT32: "uint32",
            DType.UINT64: "uint64",
            DType.FLOAT32: "float32",
            DType.FLOAT64: "float64",
            DType.STRING: "object",
            DType.BOOL: "bool",
            DType.TIMESTAMP: "datetime64[ns]",
            DType.TIMESTAMP_TZ: "datetime64[ns]",
            DType.DATE: "datetime64[ns]",
            DType.TIMEDELTA: "timedelta64[ns]",
            DType.DECIMAL: "float64",
            DType.TIME: "object",
            DType.BINARY: "object",
        }
        pd_dtype = _DTYPE_MAP.get(self._dtype, "object")
        return pd.api.types.pandas_dtype(pd_dtype)

    @property
    def ndim(self):
        return 1

    @property
    def hasnans(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].hasnans

    @property
    def is_unique(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].is_unique

    @property
    def is_monotonic_increasing(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].is_monotonic_decreasing

    @property
    def empty(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].empty

    def duplicated(self, keep="first") -> TracedSeries:
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        result = df[col].duplicated(keep=keep)
        result_df = pd.DataFrame({col: result})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(
            self._ctx, ReadTable(name, infer_schema(result_df)), col, dtype=DType.BOOL
        )

    def factorize(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].factorize(**kwargs)

    def explode(self, **kwargs) -> TracedSeries:
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        result = df[col].explode(**kwargs)
        result_df = pd.DataFrame({col: result})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(self._ctx, ReadTable(name, infer_schema(result_df)), col)

    def searchsorted(self, value, side="left", sorter=None):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].searchsorted(value, side=side, sorter=sorter)

    def copy(self, deep=True):
        return TracedSeries(
            self._ctx,
            self._source_ir,
            column_name=self._column_name,
            expr=self._expr,
            dtype=self._dtype,
        )

    def tolist(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].tolist()

    to_list = tolist

    def __array__(self, dtype=None, copy=None):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return np.array(df[col], dtype=dtype)

    def to_numpy(self, dtype=None, copy=False, na_value=None, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].to_numpy(dtype=dtype, copy=copy, na_value=na_value, **kwargs)

    def to_dict(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].to_dict(**kwargs)

    def to_csv(self, *args, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].to_csv(*args, **kwargs)

    def to_json(self, *args, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].to_json(*args, **kwargs)

    def items(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].items()

    def __iter__(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return iter(df[col])

    def __contains__(self, value):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return value in df[col].values

    @property
    def values(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].values

    @property
    def shape(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].shape

    @property
    def index(self):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].index

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

    def fillna(self, value=None, **kwargs) -> TracedSeries:
        return TracedSeries(
            self._ctx,
            self._source_ir,
            column_name=self._column_name,
            expr=BinOp("coalesce", self._expr, _to_expr(value)),
            dtype=self._dtype,
        )

    def ffill(self, **kwargs) -> TracedSeries:
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        result = df[col].ffill(**kwargs)
        result_df = pd.DataFrame({col: result})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(self._ctx, ReadTable(name, infer_schema(result_df)), col)

    def bfill(self, **kwargs) -> TracedSeries:
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        result = df[col].bfill(**kwargs)
        result_df = pd.DataFrame({col: result})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(self._ctx, ReadTable(name, infer_schema(result_df)), col)

    def dropna(self) -> TracedSeries:
        col = self._column_name
        if col is None:
            _, df = self._ctx.materialize(self._source_ir)
            col = next(iter(df.columns))
            result = df[[col]].dropna()
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedSeries(self._ctx, ReadTable(name, infer_schema(result)), col)
        filter_node = Filter(
            self._source_ir,
            UnaryOp("not", UnaryOp("is_null", ColRef(col))),
        )
        return TracedSeries(self._ctx, filter_node, col, dtype=self._dtype)

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

    def clip(self, lower=None, upper=None, **kwargs) -> TracedSeries:
        if isinstance(lower, (TracedSeries, pd.Series)) or isinstance(
            upper, (TracedSeries, pd.Series)
        ):
            _, df = self._ctx.materialize(self._source_ir)
            col = self._column_name or next(iter(df.columns))
            return df[col].clip(lower=lower, upper=upper)
        if lower is None and upper is None:
            return self
        expr = self._expr
        if lower is not None:
            expr = IfThenExpr(
                BinOp("lt", self._expr, Literal(lower)),
                Literal(lower),
                expr,
            )
        if upper is not None:
            inner = expr
            expr = IfThenExpr(
                BinOp("gt", self._expr, Literal(upper)),
                Literal(upper),
                inner,
            )
        return TracedSeries(
            self._ctx,
            self._source_ir,
            expr=expr,
            dtype=self._dtype,
        )

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

    def to_frame(self, name=None) -> TracedDataFrame:
        col = self._column_name
        if col is None:
            # Computed expression — add as column, then project
            out_name = name or "__to_frame__"
            add_node = AddColumn(self._source_ir, out_name, self._expr, self._dtype)
            return TracedDataFrame(self._ctx, Project(add_node, [out_name]))
        proj = Project(self._source_ir, [col])
        if name is not None and name != col:
            proj = RenameColumns(proj, {col: name})
        return TracedDataFrame(self._ctx, proj)

    # -- Accessor properties --

    @property
    def dt(self):
        if self._dtype == DType.TIMEDELTA:
            return TracedTimedeltaAccessor(self)
        return TracedDatetimeAccessor(self)

    @property
    def str(self) -> TracedStringAccessor:
        return TracedStringAccessor(self)

    def where(self, cond, other=None, **kwargs) -> TracedSeries:
        if isinstance(cond, TracedSeries) and isinstance(
            other, (int, float, str, bool, type(None))
        ):
            other_val = other if other is not None else float("nan")
            expr = IfThenExpr(cond._expr, self._expr, Literal(other_val))
            return TracedSeries(
                self._ctx, self._source_ir, expr=expr, dtype=self._dtype
            )
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        if isinstance(cond, TracedSeries):
            cond = self._ctx.materialize(cond._source_ir)[1][cond._column_name]
        return df[col].where(cond, other)

    def mask(self, cond, other=None, **kwargs) -> TracedSeries:
        if isinstance(cond, TracedSeries) and isinstance(
            other, (int, float, str, bool, type(None))
        ):
            other_val = other if other is not None else float("nan")
            inverted = UnaryOp("not", cond._expr)
            expr = IfThenExpr(inverted, self._expr, Literal(other_val))
            return TracedSeries(
                self._ctx, self._source_ir, expr=expr, dtype=self._dtype
            )
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        if isinstance(cond, TracedSeries):
            cond = self._ctx.materialize(cond._source_ir)[1][cond._column_name]
        return df[col].mask(cond, other)

    def astype(self, dtype) -> TracedSeries:
        target_dtype = pandas_dtype_to_ir(str(dtype))
        known_strings = {"object", "string", "str", "string[python]", "string[pyarrow]"}
        if target_dtype == DType.STRING and str(dtype) not in known_strings:
            _, df = self._ctx.materialize(self._source_ir)
            col = self._column_name or next(iter(df.columns))
            return df[col].astype(dtype)
        expr = CastExpr(self._expr, target_dtype)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=target_dtype)

    def round(self, decimals=0) -> TracedSeries:
        expr = FunctionCall(
            "round",
            args=[self._expr, Literal(int(decimals))],
            return_dtype=self._dtype,
        )
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=self._dtype)

    def sqrt(self) -> TracedSeries:
        expr = FunctionCall("sqrt", args=[self._expr], return_dtype=DType.FLOAT64)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=DType.FLOAT64)

    def log(self) -> TracedSeries:
        expr = FunctionCall("log", args=[self._expr], return_dtype=DType.FLOAT64)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=DType.FLOAT64)

    def log10(self) -> TracedSeries:
        expr = FunctionCall("log10", args=[self._expr], return_dtype=DType.FLOAT64)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=DType.FLOAT64)

    def exp(self) -> TracedSeries:
        expr = FunctionCall("exp", args=[self._expr], return_dtype=DType.FLOAT64)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=DType.FLOAT64)

    def ceil(self) -> TracedSeries:
        expr = FunctionCall("ceil", args=[self._expr], return_dtype=self._dtype)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=self._dtype)

    def floor(self) -> TracedSeries:
        expr = FunctionCall("floor", args=[self._expr], return_dtype=self._dtype)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=self._dtype)

    def sign(self) -> TracedSeries:
        expr = FunctionCall("sign", args=[self._expr], return_dtype=self._dtype)
        return TracedSeries(self._ctx, self._source_ir, expr=expr, dtype=self._dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import numpy as np

        if method != "__call__":
            return NotImplemented
        _UFUNC_MAP = {
            np.sqrt: "sqrt",
            np.log: "log",
            np.log10: "log10",
            np.exp: "exp",
            np.ceil: "ceil",
            np.floor: "floor",
            np.sign: "sign",
            np.abs: "abs",
            np.negative: "negate",
        }
        if ufunc in _UFUNC_MAP:
            name = _UFUNC_MAP[ufunc]
            if name == "abs":
                return self.abs()
            if name == "negate":
                return -self
            expr = FunctionCall(name, args=[self._expr], return_dtype=DType.FLOAT64)
            return TracedSeries(
                self._ctx, self._source_ir, expr=expr, dtype=DType.FLOAT64
            )
        return NotImplemented

    # -- Graph-breaking Series methods --

    def apply(self, func, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].apply(func, **kwargs)

    def map(self, func, na_action=None, **kwargs):
        # Dict mapping: traced via chained IfThenExpr (unmapped -> NaN)
        if isinstance(func, dict) and na_action is None:
            col = self._column_name
            expr: Expr = Literal(float("nan"))
            for old_val, new_val in reversed(func.items()):
                expr = IfThenExpr(
                    BinOp("eq", self._expr, Literal(old_val)),
                    Literal(new_val),
                    expr,
                )
            # Materialize the expression as a named column so downstream
            # operations (to_frame, etc.) preserve the column name.
            add_node = AddColumn(
                self._source_ir, col or "__mapped__", expr, self._dtype
            )
            return TracedSeries(
                self._ctx, add_node, col or "__mapped__", dtype=self._dtype
            )
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].map(func, na_action=na_action, **kwargs)

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

    def describe(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].describe(**kwargs)

    def mode(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].mode(**kwargs)

    def corr(self, other, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        if isinstance(other, TracedSeries):
            _, other_df = self._ctx.materialize(other._source_ir)
            other_col = other._column_name or next(iter(other_df.columns))
            return df[col].corr(other_df[other_col], **kwargs)
        return df[col].corr(other, **kwargs)

    def cov(self, other, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        if isinstance(other, TracedSeries):
            _, other_df = self._ctx.materialize(other._source_ir)
            other_col = other._column_name or next(iter(other_df.columns))
            return df[col].cov(other_df[other_col], **kwargs)
        return df[col].cov(other, **kwargs)

    def idxmin(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].idxmin(**kwargs)

    def idxmax(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        return df[col].idxmax(**kwargs)

    def replace(self, to_replace, value=_SENTINEL, **kwargs) -> TracedSeries:
        # Simple scalar-to-scalar: x == old ? new : x
        if (
            isinstance(to_replace, (int, float, str, bool))
            and value is not _SENTINEL
            and isinstance(value, (int, float, str, bool))
        ):
            expr = IfThenExpr(
                BinOp("eq", self._expr, Literal(to_replace)),
                Literal(value),
                self._expr,
            )
            return TracedSeries(
                self._ctx, self._source_ir, expr=expr, dtype=self._dtype
            )
        # Dict mapping: {old1: new1, old2: new2}
        if isinstance(to_replace, dict) and value is _SENTINEL:
            expr = self._expr
            for old_val, new_val in to_replace.items():
                expr = IfThenExpr(
                    BinOp("eq", self._expr, Literal(old_val)),
                    Literal(new_val),
                    expr,
                )
            return TracedSeries(
                self._ctx, self._source_ir, expr=expr, dtype=self._dtype
            )
        # Complex cases: regex, list, etc. → graph break
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        result = df[col].replace(to_replace, value, **kwargs)
        result_df = pd.DataFrame({col: result})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(self._ctx, ReadTable(name, infer_schema(result_df)), col)

    def nlargest(self, n=5, keep="first"):
        col = self._column_name
        if col is None or keep != "first":
            _, df = self._ctx.materialize(self._source_ir)
            col = col or next(iter(df.columns))
            return df[col].nlargest(n, keep=keep)
        node = Limit(Sort(self._source_ir, [(col, False)]), n)
        return TracedSeries(self._ctx, node, col, dtype=self._dtype)

    def nsmallest(self, n=5, keep="first"):
        col = self._column_name
        if col is None or keep != "first":
            _, df = self._ctx.materialize(self._source_ir)
            col = col or next(iter(df.columns))
            return df[col].nsmallest(n, keep=keep)
        node = Limit(Sort(self._source_ir, [(col, True)]), n)
        return TracedSeries(self._ctx, node, col, dtype=self._dtype)

    # --- cumulative (traced via Window IR with expanding frame) ---

    def _apply_series_cumulative(self, func):
        col = self._column_name
        window_funcs = [(col, col, func, 0)]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(self._source_ir, window_funcs, spec)
        return TracedSeries(self._ctx, node, col)

    def cumsum(self, **kwargs):
        return self._apply_series_cumulative("sum")

    def cummax(self, **kwargs):
        return self._apply_series_cumulative("max")

    def cummin(self, **kwargs):
        return self._apply_series_cumulative("min")

    def cumprod(self, **kwargs):
        return self._apply_series_cumulative("product")

    def shift(self, periods=1, **kwargs):
        col = self._column_name
        if periods >= 0:
            window_funcs = [(col, col, "lag", periods)]
        else:
            window_funcs = [(col, col, "lead", -periods)]
        node = Window(self._source_ir, window_funcs, WindowSpec())
        return TracedSeries(self._ctx, node, col)

    def diff(self, periods=1, **kwargs):
        col = self._column_name
        lag_col = f"__jit_lag_{col}"
        if periods >= 0:
            wfuncs = [(lag_col, col, "lag", periods)]
        else:
            wfuncs = [(lag_col, col, "lead", -periods)]
        window_node = Window(self._source_ir, wfuncs, WindowSpec())
        diff_expr = BinOp("sub", ColRef(col), ColRef(lag_col))
        add_node = AddColumn(window_node, col, diff_expr, self._dtype)
        parent_cols = list(self._source_ir.output_schema().columns.keys())
        proj_node = Project(add_node, parent_cols)
        return TracedSeries(self._ctx, proj_node, col)

    def pct_change(self, periods=1, **kwargs):
        col = self._column_name
        lag_col = f"__jit_lag_{col}"
        if periods >= 0:
            wfuncs = [(lag_col, col, "lag", periods)]
        else:
            wfuncs = [(lag_col, col, "lead", -periods)]
        window_node = Window(self._source_ir, wfuncs, WindowSpec())
        diff_expr = BinOp("sub", ColRef(col), ColRef(lag_col))
        pct_expr = BinOp("div", diff_expr, ColRef(lag_col))
        add_node = AddColumn(window_node, col, pct_expr, DType.FLOAT64)
        parent_cols = list(self._source_ir.output_schema().columns.keys())
        proj_node = Project(add_node, parent_cols)
        return TracedSeries(self._ctx, proj_node, col, dtype=DType.FLOAT64)

    def rank(
        self, method="average", ascending=True, na_option="keep", pct=False, **kwargs
    ):
        _RANK_FUNC_MAP = {"min": "rank", "dense": "dense_rank", "first": "row_number"}
        if method not in _RANK_FUNC_MAP or pct or na_option != "keep":
            _, df = self._ctx.materialize(self._source_ir)
            col = self._column_name or next(iter(df.columns))
            return df[col].rank(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
            )
        col = self._column_name
        func = _RANK_FUNC_MAP[method]
        window_funcs = [(col, col, func, 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            order_by=[(col, ascending)],
        )
        return TracedSeries(self._ctx, node, col, dtype=DType.FLOAT64)

    def reset_index(self, drop=True, **kwargs):
        if drop:
            return self
        _, df = self._ctx.materialize(self._source_ir)
        col = self._column_name or next(iter(df.columns))
        result = df[col].reset_index(drop=False, **kwargs)
        # Series.reset_index(drop=False) returns a DataFrame
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

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

    @property
    def weekday(self) -> TracedSeries:
        return self._extract("MONDAY_DAY_OF_WEEK")

    @property
    def microsecond(self) -> TracedSeries:
        return self._extract("MICROSECOND")

    @property
    def nanosecond(self) -> TracedSeries:
        return self._extract("NANOSECOND")

    @property
    def week(self) -> TracedSeries:
        return self._extract("WEEK")

    @property
    def weekofyear(self) -> TracedSeries:
        return self._extract("WEEK")

    @property
    def date(self) -> TracedSeries:
        expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": "DATE"},
            return_dtype=DType.DATE,
        )
        return TracedSeries(
            self._series._ctx,
            self._series._source_ir,
            expr=expr,
            dtype=DType.DATE,
        )

    @property
    def is_month_start(self) -> TracedSeries:
        day_expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": "DAY"},
            return_dtype=DType.INT64,
        )
        expr = BinOp("eq", day_expr, Literal(1))
        return TracedSeries(
            self._series._ctx,
            self._series._source_ir,
            expr=expr,
            dtype=DType.BOOL,
        )

    @property
    def is_year_start(self) -> TracedSeries:
        month_expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": "MONTH"},
            return_dtype=DType.INT64,
        )
        day_expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": "DAY"},
            return_dtype=DType.INT64,
        )
        expr = BinOp(
            "and",
            BinOp("eq", month_expr, Literal(1)),
            BinOp("eq", day_expr, Literal(1)),
        )
        return TracedSeries(
            self._series._ctx,
            self._series._source_ir,
            expr=expr,
            dtype=DType.BOOL,
        )

    @property
    def is_quarter_start(self) -> TracedSeries:
        month_expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": "MONTH"},
            return_dtype=DType.INT64,
        )
        day_expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": "DAY"},
            return_dtype=DType.INT64,
        )
        month_check = BinOp(
            "or",
            BinOp(
                "or",
                BinOp("eq", month_expr, Literal(1)),
                BinOp("eq", month_expr, Literal(4)),
            ),
            BinOp(
                "or",
                BinOp("eq", month_expr, Literal(7)),
                BinOp("eq", month_expr, Literal(10)),
            ),
        )
        expr = BinOp(
            "and",
            month_check,
            BinOp("eq", day_expr, Literal(1)),
        )
        return TracedSeries(
            self._series._ctx,
            self._series._source_ir,
            expr=expr,
            dtype=DType.BOOL,
        )

    @property
    def is_year_end(self) -> TracedSeries:
        month_expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": "MONTH"},
            return_dtype=DType.INT64,
        )
        day_expr = FunctionCall(
            "extract",
            args=[self._series._expr],
            options={"component": "DAY"},
            return_dtype=DType.INT64,
        )
        expr = BinOp(
            "and",
            BinOp("eq", month_expr, Literal(12)),
            BinOp("eq", day_expr, Literal(31)),
        )
        return TracedSeries(
            self._series._ctx,
            self._series._source_ir,
            expr=expr,
            dtype=DType.BOOL,
        )

    @property
    def is_month_end(self) -> TracedSeries:
        _, df = self._series._ctx.materialize(self._series._source_ir)
        col = self._series._column_name
        result_series = df[col].dt.is_month_end
        result_df = pd.DataFrame({col: result_series})
        name = self._series._ctx.next_materialized_name()
        self._series._ctx.register_table(name, result_df)
        return TracedSeries(
            self._series._ctx, ReadTable(name, infer_schema(result_df)), col
        )

    @property
    def is_quarter_end(self) -> TracedSeries:
        _, df = self._series._ctx.materialize(self._series._source_ir)
        col = self._series._column_name
        result_series = df[col].dt.is_quarter_end
        result_df = pd.DataFrame({col: result_series})
        name = self._series._ctx.next_materialized_name()
        self._series._ctx.register_table(name, result_df)
        return TracedSeries(
            self._series._ctx, ReadTable(name, infer_schema(result_df)), col
        )

    def month_name(self) -> TracedSeries:
        _, df = self._series._ctx.materialize(self._series._source_ir)
        col = self._series._column_name
        result_series = df[col].dt.month_name()
        result_df = pd.DataFrame({col: result_series})
        name = self._series._ctx.next_materialized_name()
        self._series._ctx.register_table(name, result_df)
        return TracedSeries(
            self._series._ctx, ReadTable(name, infer_schema(result_df)), col
        )

    def day_name(self) -> TracedSeries:
        _, df = self._series._ctx.materialize(self._series._source_ir)
        col = self._series._column_name
        result_series = df[col].dt.day_name()
        result_df = pd.DataFrame({col: result_series})
        name = self._series._ctx.next_materialized_name()
        self._series._ctx.register_table(name, result_df)
        return TracedSeries(
            self._series._ctx, ReadTable(name, infer_schema(result_df)), col
        )

    def _dt_graph_break(self, method, *args, **kwargs) -> TracedSeries:
        _, df = self._series._ctx.materialize(self._series._source_ir)
        col = self._series._column_name
        result_series = getattr(df[col].dt, method)(*args, **kwargs)
        result_df = pd.DataFrame({col: result_series})
        name = self._series._ctx.next_materialized_name()
        self._series._ctx.register_table(name, result_df)
        return TracedSeries(
            self._series._ctx, ReadTable(name, infer_schema(result_df)), col
        )

    def strftime(self, date_format) -> TracedSeries:
        return self._dt_graph_break("strftime", date_format)

    def tz_localize(self, tz, **kwargs) -> TracedSeries:
        return self._dt_graph_break("tz_localize", tz, **kwargs)

    def tz_convert(self, tz) -> TracedSeries:
        return self._dt_graph_break("tz_convert", tz)

    def normalize(self) -> TracedSeries:
        return self._dt_graph_break("normalize")

    def ceil(self, freq, **kwargs) -> TracedSeries:
        return self._dt_graph_break("ceil", freq, **kwargs)

    def floor(self, freq, **kwargs) -> TracedSeries:
        return self._dt_graph_break("floor", freq, **kwargs)

    def round(self, freq, **kwargs) -> TracedSeries:
        return self._dt_graph_break("round", freq, **kwargs)


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

    # Case operations
    def capitalize(self) -> TracedSeries:
        return self._str_func("capitalize")

    def title(self) -> TracedSeries:
        return self._str_func("title")

    def swapcase(self) -> TracedSeries:
        return self._str_func("swapcase")

    # Predicates (return bool)
    def isdigit(self) -> TracedSeries:
        return self._str_func("isdigit", return_dtype=DType.BOOL)

    def isalpha(self) -> TracedSeries:
        return self._str_func("isalpha", return_dtype=DType.BOOL)

    def isnumeric(self) -> TracedSeries:
        return self._str_func("isnumeric", return_dtype=DType.BOOL)

    def isspace(self) -> TracedSeries:
        return self._str_func("isspace", return_dtype=DType.BOOL)

    def islower(self) -> TracedSeries:
        return self._str_func("islower", return_dtype=DType.BOOL)

    def isupper(self) -> TracedSeries:
        return self._str_func("isupper", return_dtype=DType.BOOL)

    # Search
    def count(self, sub) -> TracedSeries:
        return self._str_func("count", sub, return_dtype=DType.INT64)

    def find(self, sub) -> TracedSeries:
        return self._str_func("find", sub, return_dtype=DType.INT64)

    def cat(self, sep=None) -> TracedSeries:
        if sep is not None:
            return self._str_func("cat", sep)
        return self._str_func("cat")

    # Padding
    def pad(self, width, side="left", fillchar=" ") -> TracedSeries:
        return self._str_func("pad", width, side, fillchar)

    def zfill(self, width) -> TracedSeries:
        return self._str_func("zfill", width)

    def center(self, width, fillchar=" ") -> TracedSeries:
        return self._str_func("center", width, fillchar)

    def repeat(self, repeats) -> TracedSeries:
        return self._str_func("repeat", repeats)

    # Graph-break methods that return DataFrames or complex results

    def _graph_break(self, method, *args, **kwargs):
        """Materialize and call a str method, re-register result."""
        _, df = self._series._ctx.materialize(self._series._source_ir)
        col = self._series._column_name
        result = getattr(df[col].str, method)(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            name = self._series._ctx.next_materialized_name()
            self._series._ctx.register_table(name, result)
            return TracedDataFrame(
                self._series._ctx, ReadTable(name, infer_schema(result))
            )
        result_df = pd.DataFrame({col: result})
        name = self._series._ctx.next_materialized_name()
        self._series._ctx.register_table(name, result_df)
        return TracedSeries(
            self._series._ctx, ReadTable(name, infer_schema(result_df)), col
        )

    def split(self, pat=None, n=-1, expand=False):
        return self._graph_break("split", pat=pat, n=n, expand=expand)

    def get(self, i):
        return self._graph_break("get", i)

    def extract(self, pat, flags=0, expand=True):
        return self._graph_break("extract", pat, flags=flags, expand=expand)

    def match(self, pat, case=True, flags=0, na=None):
        return self._graph_break("match", pat, case=case, flags=flags, na=na)

    def fullmatch(self, pat, case=True, flags=0, na=None):
        return self._graph_break("fullmatch", pat, case=case, flags=flags, na=na)

    def rsplit(self, pat=None, n=-1, expand=False):
        return self._graph_break("rsplit", pat=pat, n=n, expand=expand)

    def wrap(self, width, **kwargs):
        return self._graph_break("wrap", width, **kwargs)

    def ljust(self, width, fillchar=" "):
        return self._graph_break("ljust", width, fillchar=fillchar)

    def rjust(self, width, fillchar=" "):
        return self._graph_break("rjust", width, fillchar=fillchar)


# ---------------------------------------------------------------------------
# TracedTimedeltaAccessor — .dt for timedelta Series
# ---------------------------------------------------------------------------


class TracedTimedeltaAccessor:
    """Proxy for Series.dt on timedelta Series — extracts components."""

    def __init__(self, series: TracedSeries):
        self._series = series

    def _extract_td(self, component: str, return_dtype: DType = DType.INT64):
        expr = FunctionCall(
            "extract_td",
            args=[self._series._expr],
            options={"component": component},
            return_dtype=return_dtype,
        )
        return TracedSeries(
            self._series._ctx,
            self._series._source_ir,
            expr=expr,
            dtype=return_dtype,
        )

    @property
    def days(self) -> TracedSeries:
        return self._extract_td("days")

    @property
    def seconds(self) -> TracedSeries:
        return self._extract_td("seconds")

    @property
    def microseconds(self) -> TracedSeries:
        return self._extract_td("microseconds")

    def total_seconds(self) -> TracedSeries:
        return self._extract_td("total_seconds", return_dtype=DType.FLOAT64)


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
        window_funcs = [(col, col, func, 0) for col in numeric_cols]
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

    def _graph_break(self, method, **method_kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        result = getattr(df.rolling(self._window, **self._kwargs), method)(
            **method_kwargs
        )
        if isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return result

    def median(self, **kwargs):
        return self._graph_break("median", **kwargs)

    def quantile(self, q, **kwargs):
        return self._graph_break("quantile", q=q, **kwargs)

    def skew(self, **kwargs):
        return self._graph_break("skew", **kwargs)

    def kurt(self, **kwargs):
        return self._graph_break("kurt", **kwargs)

    def sem(self, **kwargs):
        return self._graph_break("sem", **kwargs)

    def rank(self, **kwargs):
        return self._graph_break("rank", **kwargs)

    def apply(self, func, raw=False, **kwargs):
        return self._graph_break("apply", func=func, raw=raw, **kwargs)

    def corr(self, other=None, **kwargs):
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        return self._graph_break("corr", other=other, **kwargs)

    def cov(self, other=None, **kwargs):
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        return self._graph_break("cov", other=other, **kwargs)

    def agg(self, func, *args, **kwargs):
        _DISPATCH = {"mean", "sum", "std", "var", "min", "max", "count"}
        if isinstance(func, str) and func in _DISPATCH:
            return self._apply(func, **kwargs)
        return self._graph_break("agg", *args, func=func, **kwargs)

    aggregate = agg


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
        window_funcs = [(col, col, func, 0) for col in numeric_cols]
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

    def count(self, **kwargs):
        return self._apply("count", **kwargs)

    def _graph_break(self, method, **method_kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        result = getattr(df.expanding(**self._kwargs), method)(**method_kwargs)
        if isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return result

    def median(self, **kwargs):
        return self._graph_break("median", **kwargs)

    def quantile(self, q, **kwargs):
        return self._graph_break("quantile", q=q, **kwargs)

    def skew(self, **kwargs):
        return self._graph_break("skew", **kwargs)

    def kurt(self, **kwargs):
        return self._graph_break("kurt", **kwargs)

    def sem(self, **kwargs):
        return self._graph_break("sem", **kwargs)

    def rank(self, **kwargs):
        return self._graph_break("rank", **kwargs)

    def apply(self, func, raw=False, **kwargs):
        return self._graph_break("apply", func=func, raw=raw, **kwargs)

    def corr(self, other=None, **kwargs):
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        return self._graph_break("corr", other=other, **kwargs)

    def cov(self, other=None, **kwargs):
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        return self._graph_break("cov", other=other, **kwargs)

    def agg(self, func, *args, **kwargs):
        _DISPATCH = {"mean", "sum", "std", "var", "min", "max", "count"}
        if isinstance(func, str) and func in _DISPATCH:
            return self._apply(func, **kwargs)
        return self._graph_break("agg", *args, func=func, **kwargs)

    aggregate = agg


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
        # Traced: Window(row_number, partition_by=keys) + Filter(eq, 1)
        all_cols = list(self._source_schema.columns.keys())
        rank_col = "__jit_grp_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("eq", ColRef(rank_col), Literal(1.0)))
        return TracedDataFrame(self._ctx, Project(filter_node, all_cols))

    def last(self):
        # Graph break — no easy way to get last row_number without total count
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

    def nunique(self):
        agg_specs = [
            (col, col, "count_distinct")
            for col in self._source_schema.columns
            if col not in self._keys
        ]
        return TracedDataFrame(
            self._ctx, Aggregate(self._source_ir, self._keys, agg_specs)
        )

    def head(self, n=5) -> TracedDataFrame:
        all_cols = list(self._source_schema.columns.keys())
        rank_col = "__jit_grp_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("lte", ColRef(rank_col), Literal(float(n))))
        return TracedDataFrame(self._ctx, Project(filter_node, all_cols))

    def nth(self, n) -> TracedDataFrame:
        if n < 0:
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys).nth(n).reset_index(drop=True)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        all_cols = list(self._source_schema.columns.keys())
        rank_col = "__jit_grp_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("eq", ColRef(rank_col), Literal(float(n + 1))))
        return TracedDataFrame(self._ctx, Project(filter_node, all_cols))

    def _apply_groupby_cumulative(self, func):
        numeric_cols = [
            col
            for col, dtype in self._source_schema.columns.items()
            if col not in self._keys and dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            _, df = self._ctx.materialize(self._source_ir)
            pandas_func = {"product": "prod"}.get(func, func)
            result = getattr(df.groupby(self._keys), f"cum{pandas_func}")()
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        window_funcs = [(col, col, func, 0) for col in numeric_cols]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(
            self._source_ir,
            window_funcs,
            spec,
            partition_by=self._keys,
        )
        return TracedDataFrame(self._ctx, node)

    def cumsum(self):
        return self._apply_groupby_cumulative("sum")

    def cummax(self):
        return self._apply_groupby_cumulative("max")

    def cummin(self):
        return self._apply_groupby_cumulative("min")

    def cumprod(self):
        return self._apply_groupby_cumulative("product")

    def transform(self, func) -> TracedDataFrame:
        _SUPPORTED = {"sum", "mean", "min", "max", "count", "std", "var"}
        if isinstance(func, str) and func in _SUPPORTED:
            numeric_cols = [
                col
                for col, dtype in self._source_schema.columns.items()
                if col not in self._keys and dtype in NUMERIC_DTYPES
            ]
            if not numeric_cols:
                _, df = self._ctx.materialize(self._source_ir)
                result = df.groupby(self._keys).transform(func)
                name = self._ctx.next_materialized_name()
                self._ctx.register_table(name, result)
                return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
            window_func = {"mean": "avg"}.get(func, func)
            window_funcs = [(col, col, window_func, 0) for col in numeric_cols]
            spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
            node = Window(self._source_ir, window_funcs, spec, partition_by=self._keys)
            return TracedDataFrame(self._ctx, node)
        # Non-string or unsupported → graph break
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).transform(func)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def apply(self, func, *args, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).apply(func, *args, **kwargs)
        if isinstance(result, pd.Series):
            result = result.reset_index(drop=True).to_frame()
        elif isinstance(result.index, pd.MultiIndex):
            result = result.reset_index(drop=True)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def median(self, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).median(**kwargs)
        result = result.reset_index()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def quantile(self, q=0.5, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).quantile(q=q, **kwargs)
        result = result.reset_index()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def shift(self, periods=1, **kwargs) -> TracedDataFrame:
        non_key_cols = [
            col for col in self._source_schema.columns if col not in self._keys
        ]
        if periods >= 0:
            wfuncs = [(col, col, "lag", periods) for col in non_key_cols]
        else:
            wfuncs = [(col, col, "lead", -periods) for col in non_key_cols]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
        node = Window(self._source_ir, wfuncs, spec, partition_by=self._keys)
        return TracedDataFrame(self._ctx, node)

    def diff(self, periods=1, **kwargs) -> TracedDataFrame:
        numeric_cols = [
            col
            for col, dtype in self._source_schema.columns.items()
            if col not in self._keys and dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys).diff(periods=periods)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        lag_funcs = [
            (
                f"__jit_lag_{col}",
                col,
                "lag" if periods >= 0 else "lead",
                abs(periods),
            )
            for col in numeric_cols
        ]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
        window_node = Window(self._source_ir, lag_funcs, spec, partition_by=self._keys)
        result_ir = window_node
        for col in numeric_cols:
            lag_col = f"__jit_lag_{col}"
            diff_expr = BinOp("sub", ColRef(col), ColRef(lag_col))
            dtype = self._source_schema.columns[col]
            result_ir = AddColumn(result_ir, col, diff_expr, dtype)
        all_cols = list(self._source_schema.columns.keys())
        proj_node = Project(result_ir, all_cols)
        return TracedDataFrame(self._ctx, proj_node)

    def pct_change(self, periods=1, **kwargs) -> TracedDataFrame:
        numeric_cols = [
            col
            for col, dtype in self._source_schema.columns.items()
            if col not in self._keys and dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys).pct_change(periods=periods)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        lag_funcs = [
            (
                f"__jit_lag_{col}",
                col,
                "lag" if periods >= 0 else "lead",
                abs(periods),
            )
            for col in numeric_cols
        ]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
        window_node = Window(self._source_ir, lag_funcs, spec, partition_by=self._keys)
        result_ir = window_node
        for col in numeric_cols:
            lag_col = f"__jit_lag_{col}"
            diff_expr = BinOp("sub", ColRef(col), ColRef(lag_col))
            pct_expr = BinOp("div", diff_expr, ColRef(lag_col))
            result_ir = AddColumn(result_ir, col, pct_expr, DType.FLOAT64)
        all_cols = list(self._source_schema.columns.keys())
        proj_node = Project(result_ir, all_cols)
        return TracedDataFrame(self._ctx, proj_node)

    def ffill(self, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).ffill(**kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def bfill(self, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).bfill(**kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def describe(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys).describe(**kwargs)

    @property
    def ngroups(self):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys).ngroups

    def get_group(self, name, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        # pandas requires tuple for single-key groupby in newer versions
        if not isinstance(name, tuple):
            name = (name,)
        result = df.groupby(self._keys).get_group(name, **kwargs)
        nm = self._ctx.next_materialized_name()
        self._ctx.register_table(nm, result)
        return TracedDataFrame(self._ctx, ReadTable(nm, infer_schema(result)))

    def prod(self, **kwargs) -> TracedDataFrame:
        return self._simple_agg("prod")

    product = prod

    def sem(self, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).sem(**kwargs).reset_index()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def skew(self, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).skew(**kwargs).reset_index()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def kurt(self, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = (
            df.groupby(self._keys)
            .apply(
                lambda g: g.select_dtypes(include="number").kurt(), include_groups=False
            )
            .reset_index()
        )
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def idxmin(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys).idxmin(**kwargs)

    def idxmax(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys).idxmax(**kwargs)

    def cumcount(self, ascending=True) -> TracedSeries:
        rank_col = "__jit_grp_cc"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        # row_number is 1-based; cumcount is 0-based
        minus_one = AddColumn(
            node,
            rank_col,
            BinOp("sub", ColRef(rank_col), Literal(1.0)),
            DType.FLOAT64,
        )
        if not ascending:
            # For descending: count - 1 - cumcount = count - row_number
            count_col = "__jit_grp_cnt"
            count_funcs = [(count_col, self._keys[0], "count", 0)]
            count_node = Window(
                minus_one,
                count_funcs,
                WindowSpec(kind="rows", lower_offset=None, upper_offset=None),
                partition_by=self._keys,
            )
            desc_expr = BinOp(
                "sub",
                BinOp("sub", ColRef(count_col), Literal(1.0)),
                ColRef(rank_col),
            )
            minus_one = AddColumn(count_node, rank_col, desc_expr, DType.FLOAT64)
        return TracedSeries(self._ctx, minus_one, rank_col, dtype=DType.FLOAT64)

    def filter(self, func, *args, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).filter(func, *args, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def value_counts(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys).value_counts(**kwargs)

    @property
    def groups(self):
        _, df = self._ctx.materialize(self._source_ir)
        by = self._keys[0] if len(self._keys) == 1 else self._keys
        return df.groupby(by).groups

    @property
    def indices(self):
        _, df = self._ctx.materialize(self._source_ir)
        by = self._keys[0] if len(self._keys) == 1 else self._keys
        return df.groupby(by).indices

    def rank(
        self, method="average", ascending=True, na_option="keep", pct=False, **kwargs
    ):
        _RANK_MAP = {"min": "rank", "dense": "dense_rank", "first": "row_number"}
        if method not in _RANK_MAP or pct or na_option != "keep":
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys).rank(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
            )
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        numeric_cols = [
            col
            for col, dtype in self._source_schema.columns.items()
            if col not in self._keys and dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys).rank(
                method=method,
                ascending=ascending,
            )
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        func = _RANK_MAP[method]
        result_ir = self._source_ir
        for col in numeric_cols:
            window_funcs = [(col, col, func, 0)]
            result_ir = Window(
                result_ir,
                window_funcs,
                WindowSpec(),
                partition_by=self._keys,
                order_by=[(col, ascending)],
            )
        return TracedDataFrame(self._ctx, result_ir)


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

    def nunique(self):
        return self._agg("count_distinct")

    def prod(self):
        return self._agg("prod")

    product = prod

    def rank(
        self, method="average", ascending=True, na_option="keep", pct=False, **kwargs
    ):
        _RANK_FUNC_MAP = {"min": "rank", "dense": "dense_rank", "first": "row_number"}
        if method not in _RANK_FUNC_MAP or pct or na_option != "keep":
            _, df = self._ctx.materialize(self._source_ir)
            return df.groupby(self._keys)[self._column].rank(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
            )
        col = self._column
        func = _RANK_FUNC_MAP[method]
        window_funcs = [(col, col, func, 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
            order_by=[(col, ascending)],
        )
        return TracedSeries(self._ctx, node, col, dtype=DType.FLOAT64)

    def _agg(self, func):
        return TracedDataFrame(
            self._ctx,
            Aggregate(
                self._source_ir,
                self._keys,
                [(self._column, self._column, func)],
            ),
        )

    def _apply_groupby_cumulative(self, func):
        col = self._column
        window_funcs = [(col, col, func, 0)]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(
            self._source_ir,
            window_funcs,
            spec,
            partition_by=self._keys,
        )
        return TracedSeries(self._ctx, node, col)

    def cumsum(self, **kwargs):
        return self._apply_groupby_cumulative("sum")

    def cummax(self, **kwargs):
        return self._apply_groupby_cumulative("max")

    def cummin(self, **kwargs):
        return self._apply_groupby_cumulative("min")

    def cumprod(self, **kwargs):
        return self._apply_groupby_cumulative("product")

    def transform(self, func) -> TracedSeries:
        _SUPPORTED = {"sum", "mean", "min", "max", "count", "std", "var"}
        if isinstance(func, str) and func in _SUPPORTED:
            col = self._column
            window_func = {"mean": "avg"}.get(func, func)
            window_funcs = [(col, col, window_func, 0)]
            spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
            node = Window(self._source_ir, window_funcs, spec, partition_by=self._keys)
            return TracedSeries(self._ctx, node, col)
        # Graph break
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys)[self._column].transform(func)
        result_df = pd.DataFrame({self._column: result})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(
            self._ctx, ReadTable(name, infer_schema(result_df)), self._column
        )

    def first(self) -> TracedSeries:
        rank_col = "__jit_gbs_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("eq", ColRef(rank_col), Literal(1.0)))
        proj = Project(filter_node, [*self._keys, self._column])
        return TracedSeries(self._ctx, proj, self._column)

    def last(self) -> TracedSeries:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys, as_index=False)[self._column].last()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedSeries(
            self._ctx, ReadTable(name, infer_schema(result)), self._column
        )

    def shift(self, periods=1, **kwargs) -> TracedSeries:
        col = self._column
        if periods >= 0:
            wfuncs = [(col, col, "lag", periods)]
        else:
            wfuncs = [(col, col, "lead", -periods)]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
        node = Window(self._source_ir, wfuncs, spec, partition_by=self._keys)
        return TracedSeries(self._ctx, node, col)

    def diff(self, periods=1, **kwargs) -> TracedSeries:
        col = self._column
        lag_col = f"__jit_gbs_lag_{col}"
        lag_funcs = [(lag_col, col, "lag" if periods >= 0 else "lead", abs(periods))]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
        window_node = Window(self._source_ir, lag_funcs, spec, partition_by=self._keys)
        diff_expr = BinOp("sub", ColRef(col), ColRef(lag_col))
        dtype = self._source_schema.columns.get(col, DType.FLOAT64)
        result_ir = AddColumn(window_node, col, diff_expr, dtype)
        return TracedSeries(self._ctx, result_ir, col)

    def head(self, n=5) -> TracedDataFrame:
        all_cols = list(self._source_schema.columns.keys())
        rank_col = "__jit_gbs_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("lte", ColRef(rank_col), Literal(float(n))))
        proj = Project(filter_node, all_cols)
        return TracedDataFrame(self._ctx, proj)

    def nth(self, n) -> TracedSeries:
        if n < 0:
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys)[self._column].nth(n)
            result_df = result.reset_index(drop=True).to_frame()
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result_df)
            return TracedSeries(
                self._ctx, ReadTable(name, infer_schema(result_df)), self._column
            )
        rank_col = "__jit_gbs_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("eq", ColRef(rank_col), Literal(float(n + 1))))
        proj = Project(filter_node, [*self._keys, self._column])
        return TracedSeries(self._ctx, proj, self._column)

    def apply(self, func, *args, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys)[self._column].apply(func, *args, **kwargs)
        if isinstance(result, pd.Series):
            result_df = result.reset_index(drop=True).to_frame(name=self._column)
        else:
            result_df = pd.DataFrame({self._column: [result]})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(
            self._ctx, ReadTable(name, infer_schema(result_df)), self._column
        )

    def describe(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys)[self._column].describe(**kwargs)

    def ffill(self, **kwargs) -> TracedSeries:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys)[self._column].ffill(**kwargs)
        result_df = pd.DataFrame({self._column: result})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(
            self._ctx, ReadTable(name, infer_schema(result_df)), self._column
        )

    def bfill(self, **kwargs) -> TracedSeries:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys)[self._column].bfill(**kwargs)
        result_df = pd.DataFrame({self._column: result})
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result_df)
        return TracedSeries(
            self._ctx, ReadTable(name, infer_schema(result_df)), self._column
        )

    def median(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys)[self._column].median(**kwargs)

    def quantile(self, q=0.5, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys)[self._column].quantile(q=q, **kwargs)

    def value_counts(self, **kwargs):
        _, df = self._ctx.materialize(self._source_ir)
        return df.groupby(self._keys)[self._column].value_counts(**kwargs)


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

    def count(self):
        return self._agg("count")

    def min(self):
        return self._agg("min")

    def max(self):
        return self._agg("max")

    def prod(self):
        return self._agg("prod")

    product = prod

    def nunique(self):
        agg_specs = [(c, c, "count_distinct") for c in self._columns]
        return TracedDataFrame(
            self._ctx, Aggregate(self._source_ir, self._keys, agg_specs)
        )

    def _agg(self, func):
        agg_specs = [(c, c, func) for c in self._columns]
        return TracedDataFrame(
            self._ctx, Aggregate(self._source_ir, self._keys, agg_specs)
        )

    def first(self):
        all_cols = list(self._source_schema.columns.keys())
        rank_col = "__jit_gbm_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("eq", ColRef(rank_col), Literal(1.0)))
        return TracedDataFrame(self._ctx, Project(filter_node, all_cols))

    def last(self):
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys, as_index=False)[self._columns].last()
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def head(self, n=5) -> TracedDataFrame:
        all_cols = list(self._source_schema.columns.keys())
        rank_col = "__jit_gbm_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("lte", ColRef(rank_col), Literal(float(n))))
        return TracedDataFrame(self._ctx, Project(filter_node, all_cols))

    def nth(self, n) -> TracedDataFrame:
        if n < 0:
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys)[self._columns].nth(n).reset_index(drop=True)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        all_cols = list(self._source_schema.columns.keys())
        rank_col = "__jit_gbm_rn"
        window_funcs = [(rank_col, self._keys[0], "row_number", 0)]
        node = Window(
            self._source_ir,
            window_funcs,
            WindowSpec(),
            partition_by=self._keys,
        )
        filter_node = Filter(node, BinOp("eq", ColRef(rank_col), Literal(float(n + 1))))
        return TracedDataFrame(self._ctx, Project(filter_node, all_cols))

    def shift(self, periods=1, **kwargs) -> TracedDataFrame:
        if periods >= 0:
            wfuncs = [(col, col, "lag", periods) for col in self._columns]
        else:
            wfuncs = [(col, col, "lead", -periods) for col in self._columns]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
        node = Window(self._source_ir, wfuncs, spec, partition_by=self._keys)
        return TracedDataFrame(self._ctx, node)

    def diff(self, periods=1, **kwargs) -> TracedDataFrame:
        numeric_cols = [
            c
            for c in self._columns
            if self._source_schema.columns.get(c) in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys)[self._columns].diff(periods=periods)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        lag_funcs = [
            (f"__jit_lag_{c}", c, "lag" if periods >= 0 else "lead", abs(periods))
            for c in numeric_cols
        ]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
        window_node = Window(self._source_ir, lag_funcs, spec, partition_by=self._keys)
        result_ir = window_node
        for col in numeric_cols:
            diff_expr = BinOp("sub", ColRef(col), ColRef(f"__jit_lag_{col}"))
            dtype = self._source_schema.columns[col]
            result_ir = AddColumn(result_ir, col, diff_expr, dtype)
        proj_cols = self._keys + list(self._columns)
        return TracedDataFrame(self._ctx, Project(result_ir, proj_cols))

    def _apply_multi_cumulative(self, func):
        target_cols = [
            c
            for c in self._columns
            if self._source_schema.columns.get(c) in NUMERIC_DTYPES
        ]
        if not target_cols:
            _, df = self._ctx.materialize(self._source_ir)
            pandas_func = {"product": "prod"}.get(func, func)
            result = getattr(
                df.groupby(self._keys)[self._columns], f"cum{pandas_func}"
            )()
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        window_funcs = [(col, col, func, 0) for col in target_cols]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(self._source_ir, window_funcs, spec, partition_by=self._keys)
        return TracedDataFrame(self._ctx, node)

    def cumsum(self):
        return self._apply_multi_cumulative("sum")

    def cummax(self):
        return self._apply_multi_cumulative("max")

    def cummin(self):
        return self._apply_multi_cumulative("min")

    def cumprod(self):
        return self._apply_multi_cumulative("product")

    def apply(self, func, *args, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys)[self._columns].apply(func, *args, **kwargs)
        if isinstance(result, pd.Series):
            result = result.reset_index(drop=True).to_frame()
        elif isinstance(result.index, pd.MultiIndex):
            result = result.reset_index(drop=True)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def filter(self, func, *args, **kwargs) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys).filter(func, *args, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def rank(
        self, method="average", ascending=True, na_option="keep", pct=False, **kwargs
    ):
        _RANK_MAP = {"min": "rank", "dense": "dense_rank", "first": "row_number"}
        if method not in _RANK_MAP or pct or na_option != "keep":
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys)[self._columns].rank(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
            )
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        func = _RANK_MAP[method]
        numeric_cols = [
            c
            for c in self._columns
            if self._source_schema.columns.get(c) in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            _, df = self._ctx.materialize(self._source_ir)
            result = df.groupby(self._keys)[self._columns].rank(
                method=method,
                ascending=ascending,
            )
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        result_ir = self._source_ir
        for col in numeric_cols:
            window_funcs = [(col, col, func, 0)]
            result_ir = Window(
                result_ir,
                window_funcs,
                WindowSpec(),
                partition_by=self._keys,
                order_by=[(col, ascending)],
            )
        return TracedDataFrame(self._ctx, result_ir)

    def transform(self, func) -> TracedDataFrame:
        _SUPPORTED = {"sum", "mean", "min", "max", "count", "std", "var"}
        if isinstance(func, str) and func in _SUPPORTED:
            target_cols = [
                c
                for c in self._columns
                if self._source_schema.columns.get(c) in NUMERIC_DTYPES
            ]
            if not target_cols:
                _, df = self._ctx.materialize(self._source_ir)
                result = df.groupby(self._keys)[self._columns].transform(func)
                name = self._ctx.next_materialized_name()
                self._ctx.register_table(name, result)
                return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
            window_func = {"mean": "avg"}.get(func, func)
            window_funcs = [(col, col, window_func, 0) for col in target_cols]
            spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=None)
            node = Window(self._source_ir, window_funcs, spec, partition_by=self._keys)
            return TracedDataFrame(self._ctx, node)
        # Graph break
        _, df = self._ctx.materialize(self._source_ir)
        result = df.groupby(self._keys)[self._columns].transform(func)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))


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

    # -- Arithmetic --

    def _arith_scalar(self, op, other):
        schema = self._ir.output_schema()
        result_ir = self._ir
        for col, dtype in schema.columns.items():
            if dtype in NUMERIC_DTYPES:
                expr = BinOp(op, ColRef(col), _to_expr(other))
                result_ir = AddColumn(result_ir, col, expr, dtype)
        return TracedDataFrame(self._ctx, result_ir)

    def _arith_df(self, op, other):
        df = self._materialize()
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        ops = {
            "add": "__add__",
            "sub": "__sub__",
            "mul": "__mul__",
            "div": "__truediv__",
            "floordiv": "__floordiv__",
            "mod": "__mod__",
        }
        result = getattr(df, ops.get(op, f"__{op}__"))(other)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def __add__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            return self._arith_scalar("add", other)
        return self._arith_df("add", other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            return self._arith_scalar("sub", other)
        return self._arith_df("sub", other)

    def __rsub__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            schema = self._ir.output_schema()
            result_ir = self._ir
            for col, dtype in schema.columns.items():
                if dtype in NUMERIC_DTYPES:
                    expr = BinOp("sub", _to_expr(other), ColRef(col))
                    result_ir = AddColumn(result_ir, col, expr, dtype)
            return TracedDataFrame(self._ctx, result_ir)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            return self._arith_scalar("mul", other)
        return self._arith_df("mul", other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            return self._arith_scalar("div", other)
        return self._arith_df("div", other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            schema = self._ir.output_schema()
            result_ir = self._ir
            for col, dtype in schema.columns.items():
                if dtype in NUMERIC_DTYPES:
                    expr = BinOp("div", _to_expr(other), ColRef(col))
                    result_ir = AddColumn(result_ir, col, expr, DType.FLOAT64)
            return TracedDataFrame(self._ctx, result_ir)
        return NotImplemented

    def __neg__(self):
        schema = self._ir.output_schema()
        result_ir = self._ir
        for col, dtype in schema.columns.items():
            if dtype in NUMERIC_DTYPES:
                expr = UnaryOp("negate", ColRef(col))
                result_ir = AddColumn(result_ir, col, expr, dtype)
        return TracedDataFrame(self._ctx, result_ir)

    def any(self, **kwargs):
        df = self._ensure_materialized()
        return df.any(**kwargs)

    def all(self, **kwargs):
        df = self._ensure_materialized()
        return df.all(**kwargs)

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
            if key._source_ir is self._ir:
                return TracedDataFrame(self._ctx, Filter(self._ir, key._expr))
            w = _composable_window(key._source_ir, self._ir)
            if w is not None:
                # Build Window with temp names, filter, then project away temps
                parent_cols = list(self._ir.output_schema().columns.keys())
                parent_set = set(parent_cols)
                mapping: dict[str, str] = {}
                new_wfuncs = []
                for out, src, func, off in w.window_funcs:
                    if out in parent_set:
                        temp = f"__jit_tmp_{out}"
                        mapping[out] = temp
                        new_wfuncs.append((temp, src, func, off))
                    else:
                        new_wfuncs.append((out, src, func, off))
                window_node = Window(
                    self._ir,
                    new_wfuncs,
                    w.window_spec,
                    w.partition_by,
                    w.order_by,
                )
                rewritten = _rewrite_col_refs(key._expr, mapping)
                filter_node = Filter(window_node, rewritten)
                project_node = Project(filter_node, parent_cols)
                return TracedDataFrame(self._ctx, project_node)
            # Incompatible IR — materialize mask
            _, mat = self._ctx.materialize(key._source_ir)
            col = key._column_name or next(iter(mat.columns))
            df = self._materialize()
            result = df[mat[col].astype(bool)]
            result_name = self._ctx.next_materialized_name()
            self._ctx.register_table(result_name, result)
            return TracedDataFrame(
                self._ctx, ReadTable(result_name, infer_schema(result))
            )
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
                node = self._assign_series(self, key, value)
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

    def __contains__(self, key) -> bool:
        return key in self._ir.output_schema().columns

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
                node = self._assign_series(result, name, value)
            elif callable(value):
                traced_result = value(result)
                if isinstance(traced_result, TracedSeries):
                    node = self._assign_series(result, name, traced_result)
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

    @staticmethod
    def _assign_series(target, col_name, series):
        """Build IR node for assigning a TracedSeries to a column.

        Handles same-IR (AddColumn), composable Window (transplant),
        and incompatible IR (materialize fallback).
        """
        if series._source_ir is target._ir:
            return AddColumn(target._ir, col_name, series._expr, series._dtype)
        w = _composable_window(series._source_ir, target._ir)
        if w is not None:
            new_wfuncs = [
                (col_name, src, func, off) for (_, src, func, off) in w.window_funcs
            ]
            return Window(
                target._ir,
                new_wfuncs,
                w.window_spec,
                w.partition_by,
                w.order_by,
            )
        # Incompatible IR — materialize both sides
        _, mat = target._ctx.materialize(series._source_ir)
        col = series._column_name or next(iter(mat.columns))
        df = target._materialize().copy()
        df[col_name] = mat[col].values
        nm = target._ctx.next_materialized_name()
        target._ctx.register_table(nm, df)
        return ReadTable(nm, infer_schema(df))

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

    def tail(self, n=5) -> TracedDataFrame:
        _, df = self._ctx.materialize(self._ir)
        result = df.tail(n).reset_index(drop=True)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

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

    def fillna(self, value=None, **kwargs) -> TracedDataFrame:
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

    def ffill(self, **kwargs) -> TracedDataFrame:
        df = self._materialize()
        result = df.ffill(**kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def bfill(self, **kwargs) -> TracedDataFrame:
        df = self._materialize()
        result = df.bfill(**kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def abs(self) -> TracedDataFrame:
        schema = self._ir.output_schema()
        result_ir = self._ir
        for col_name, dtype in schema.columns.items():
            if dtype in NUMERIC_DTYPES:
                result_ir = AddColumn(
                    result_ir,
                    col_name,
                    UnaryOp("abs", ColRef(col_name)),
                    dtype,
                )
        return TracedDataFrame(self._ctx, result_ir)

    def clip(self, lower=None, upper=None, **kwargs) -> TracedDataFrame:
        if isinstance(lower, (TracedSeries, pd.Series, dict)) or isinstance(
            upper, (TracedSeries, pd.Series, dict)
        ):
            df = self._materialize()
            result = df.clip(lower=lower, upper=upper)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        if lower is None and upper is None:
            return self
        schema = self._ir.output_schema()
        result_ir = self._ir
        for col_name, dtype in schema.columns.items():
            if dtype not in NUMERIC_DTYPES:
                continue
            expr: Expr = ColRef(col_name)
            if lower is not None:
                expr = IfThenExpr(
                    BinOp("lt", ColRef(col_name), Literal(lower)),
                    Literal(lower),
                    expr,
                )
            if upper is not None:
                inner = expr
                expr = IfThenExpr(
                    BinOp("gt", ColRef(col_name), Literal(upper)),
                    Literal(upper),
                    inner,
                )
            result_ir = AddColumn(result_ir, col_name, expr, dtype)
        return TracedDataFrame(self._ctx, result_ir)

    def drop_duplicates(self, subset=None, keep="first", **kwargs) -> TracedDataFrame:
        schema = self._ir.output_schema()
        all_cols = list(schema.columns.keys())
        cols = (
            all_cols
            if subset is None
            else ([subset] if isinstance(subset, str) else list(subset))
        )
        if keep == "first":
            # Full-column distinct
            if set(cols) == set(all_cols):
                return TracedDataFrame(self._ctx, Distinct(self._ir, cols))
            # Partial subset: Window(row_number) partitioned by subset,
            # filter rank == 1, project original columns
            rank_col = "__jit_dedup_rank"
            window_funcs = [(rank_col, cols[0], "row_number", 0)]
            node = Window(
                self._ir,
                window_funcs,
                WindowSpec(),
                partition_by=cols,
            )
            filter_node = Filter(node, BinOp("eq", ColRef(rank_col), Literal(1.0)))
            proj_node = Project(filter_node, all_cols)
            return TracedDataFrame(self._ctx, proj_node)
        # keep="last" or keep=False → graph break
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

    # --- cumulative window helpers ---

    def _apply_cumulative(self, func):
        schema = self._ir.output_schema()
        numeric_cols = [
            col for col, dtype in schema.columns.items() if dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            df = self._materialize()
            pandas_func = {"product": "prod"}.get(func, func)
            result = getattr(df, f"cum{pandas_func}")()
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        window_funcs = [(col, col, func, 0) for col in numeric_cols]
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(self._ir, window_funcs, spec)
        return TracedDataFrame(self._ctx, node)

    def cumsum(self, **kwargs):
        return self._apply_cumulative("sum")

    def cummax(self, **kwargs):
        return self._apply_cumulative("max")

    def cummin(self, **kwargs):
        return self._apply_cumulative("min")

    def cumprod(self, **kwargs):
        return self._apply_cumulative("product")

    def shift(self, periods=1, **kwargs):
        schema = self._ir.output_schema()
        all_cols = list(schema.columns.keys())
        if periods >= 0:
            window_funcs = [(col, col, "lag", periods) for col in all_cols]
        else:
            window_funcs = [(col, col, "lead", -periods) for col in all_cols]
        spec = WindowSpec()
        node = Window(self._ir, window_funcs, spec)
        return TracedDataFrame(self._ctx, node)

    def diff(self, periods=1, **kwargs):
        schema = self._ir.output_schema()
        numeric_cols = [
            col for col, dtype in schema.columns.items() if dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            df = self._materialize()
            result = df.diff(periods=periods)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        lag_funcs = [
            (
                f"__jit_lag_{col}",
                col,
                "lag" if periods >= 0 else "lead",
                abs(periods),
            )
            for col in numeric_cols
        ]
        window_node = Window(self._ir, lag_funcs, WindowSpec())
        result_ir = window_node
        for col in numeric_cols:
            lag_col = f"__jit_lag_{col}"
            diff_expr = BinOp("sub", ColRef(col), ColRef(lag_col))
            dtype = schema.columns[col]
            result_ir = AddColumn(result_ir, col, diff_expr, dtype)
        proj_node = Project(result_ir, list(schema.columns.keys()))
        return TracedDataFrame(self._ctx, proj_node)

    def pct_change(self, periods=1, **kwargs):
        schema = self._ir.output_schema()
        numeric_cols = [
            col for col, dtype in schema.columns.items() if dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            df = self._materialize()
            result = df.pct_change(periods=periods)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        lag_funcs = [
            (
                f"__jit_lag_{col}",
                col,
                "lag" if periods >= 0 else "lead",
                abs(periods),
            )
            for col in numeric_cols
        ]
        window_node = Window(self._ir, lag_funcs, WindowSpec())
        result_ir = window_node
        for col in numeric_cols:
            lag_col = f"__jit_lag_{col}"
            diff_expr = BinOp("sub", ColRef(col), ColRef(lag_col))
            pct_expr = BinOp("div", diff_expr, ColRef(lag_col))
            result_ir = AddColumn(result_ir, col, pct_expr, DType.FLOAT64)
        proj_node = Project(result_ir, list(schema.columns.keys()))
        return TracedDataFrame(self._ctx, proj_node)

    def duplicated(self, subset=None, keep="first", **kwargs):
        schema = self._ir.output_schema()
        all_cols = list(schema.columns.keys())
        cols = (
            all_cols
            if subset is None
            else ([subset] if isinstance(subset, str) else list(subset))
        )
        if keep == "first":
            rank_col = "__jit_dup_rank"
            window_funcs = [(rank_col, cols[0], "row_number", 0)]
            node = Window(
                self._ir,
                window_funcs,
                WindowSpec(),
                partition_by=cols,
            )
            expr = BinOp("gt", ColRef(rank_col), Literal(1.0))
            return TracedSeries(self._ctx, node, expr=expr, dtype=DType.BOOL)
        # keep="last" or keep=False → graph break
        df = self._materialize()
        return df.duplicated(subset=subset, keep=keep, **kwargs)

    def rank(
        self, method="average", ascending=True, na_option="keep", pct=False, **kwargs
    ):
        _RANK_MAP = {"min": "rank", "dense": "dense_rank", "first": "row_number"}
        if method not in _RANK_MAP or pct or na_option != "keep":
            df = self._materialize()
            result = df.rank(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
            )
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        schema = self._ir.output_schema()
        numeric_cols = [
            col for col, dtype in schema.columns.items() if dtype in NUMERIC_DTYPES
        ]
        if not numeric_cols:
            df = self._materialize()
            result = df.rank(method=method, ascending=ascending)
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        func = _RANK_MAP[method]
        result_ir = self._ir
        for col in numeric_cols:
            window_funcs = [(col, col, func, 0)]
            result_ir = Window(
                result_ir,
                window_funcs,
                WindowSpec(),
                order_by=[(col, ascending)],
            )
        return TracedDataFrame(self._ctx, result_ir)

    def round(self, decimals=0, **kwargs) -> TracedDataFrame:
        schema = self._ir.output_schema()
        result_ir = self._ir
        for col_name, dtype in schema.columns.items():
            if dtype in NUMERIC_DTYPES:
                expr = FunctionCall(
                    "round",
                    args=[ColRef(col_name), Literal(int(decimals))],
                    return_dtype=dtype,
                )
                result_ir = AddColumn(result_ir, col_name, expr, dtype)
        return TracedDataFrame(self._ctx, result_ir)

    # -- DataFrame-level aggregation (graph breaks) --

    def sum(self, **kwargs):
        df = self._ensure_materialized()
        return df.sum(**kwargs)

    def mean(self, **kwargs):
        df = self._ensure_materialized()
        return df.mean(**kwargs)

    def min(self, **kwargs):
        df = self._ensure_materialized()
        return df.min(**kwargs)

    def max(self, **kwargs):
        df = self._ensure_materialized()
        return df.max(**kwargs)

    def count(self, **kwargs):
        df = self._ensure_materialized()
        return df.count(**kwargs)

    def std(self, **kwargs):
        df = self._ensure_materialized()
        return df.std(**kwargs)

    def var(self, **kwargs):
        df = self._ensure_materialized()
        return df.var(**kwargs)

    def median(self, **kwargs):
        df = self._ensure_materialized()
        return df.median(**kwargs)

    def quantile(self, q=0.5, **kwargs):
        df = self._ensure_materialized()
        return df.quantile(q=q, **kwargs)

    def corr(self, method="pearson", **kwargs):
        df = self._ensure_materialized()
        return df.corr(method=method, **kwargs)

    def cov(self, **kwargs):
        df = self._ensure_materialized()
        return df.cov(**kwargs)

    def nunique(self, **kwargs):
        df = self._ensure_materialized()
        return df.nunique(**kwargs)

    def prod(self, **kwargs):
        df = self._ensure_materialized()
        return df.prod(**kwargs)

    product = prod

    def isin(self, values) -> TracedDataFrame:
        df = self._materialize()
        result = df.isin(values)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def idxmin(self, **kwargs):
        df = self._ensure_materialized()
        return df.idxmin(**kwargs)

    def idxmax(self, **kwargs):
        df = self._ensure_materialized()
        return df.idxmax(**kwargs)

    def join(self, other, on=None, how="left", **kwargs):
        df = self._materialize()
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        result = df.join(other, on=on, how=how, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def combine_first(self, other):
        df = self._materialize()
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        result = df.combine_first(other)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def update(self, other, **kwargs):
        df = self._materialize()
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        df.update(other, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, df)
        new_ir = ReadTable(name, infer_schema(df))
        object.__setattr__(self, "_ir", new_ir)

    def filter(self, items=None, like=None, regex=None, axis=1):
        schema = self._ir.output_schema()
        cols = list(schema.columns.keys())
        if items is not None:
            selected = [c for c in items if c in cols]
        elif like is not None:
            selected = [c for c in cols if like in c]
        elif regex is not None:
            selected = [c for c in cols if _re.search(regex, c)]
        else:
            selected = cols
        return TracedDataFrame(self._ctx, Project(self._ir, selected))

    def reindex(self, *args, columns=None, **kwargs):
        if columns is not None and not args and not kwargs.get("index"):
            schema = self._ir.output_schema()
            existing = set(schema.columns.keys())
            selected = [c for c in columns if c in existing]
            return TracedDataFrame(self._ctx, Project(self._ir, selected))
        df = self._ensure_materialized()
        return df.reindex(*args, columns=columns, **kwargs)

    def equals(self, other):
        df = self._ensure_materialized()
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        return df.equals(other)

    def compare(self, other, **kwargs):
        df = self._ensure_materialized()
        if isinstance(other, TracedDataFrame):
            other = other._ensure_materialized()
        return df.compare(other, **kwargs)

    def explode(self, column, **kwargs):
        df = self._materialize()
        result = df.explode(column, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def take(self, indices, axis=0, **kwargs):
        df = self._ensure_materialized()
        return df.take(indices, axis=axis, **kwargs)

    def get(self, key, default=None):
        schema = self._ir.output_schema()
        if key in schema.columns:
            return self[key]
        return default

    def pop(self, item):
        df = self._materialize()
        result = df.pop(item)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, df)
        object.__setattr__(self, "_ir", ReadTable(name, infer_schema(df)))
        return result

    def xs(self, key, axis=0, level=None, drop_level=True):
        df = self._ensure_materialized()
        return df.xs(key, axis=axis, level=level, drop_level=drop_level)

    def memory_usage(self, **kwargs):
        df = self._ensure_materialized()
        return df.memory_usage(**kwargs)

    def select_dtypes(self, include=None, exclude=None) -> TracedDataFrame:
        schema = self._ir.output_schema()
        _INT_DTYPES = frozenset(
            {
                DType.INT8,
                DType.INT16,
                DType.INT32,
                DType.INT64,
                DType.UINT8,
                DType.UINT16,
                DType.UINT32,
                DType.UINT64,
            }
        )
        _FLOAT_DTYPES = frozenset({DType.FLOAT32, DType.FLOAT64})
        _DATETIME_DTYPES = frozenset({DType.TIMESTAMP, DType.TIMESTAMP_TZ, DType.DATE})
        _DTYPE_GROUPS = {
            "number": NUMERIC_DTYPES,
            "numeric": NUMERIC_DTYPES,
            "int": _INT_DTYPES,
            "integer": _INT_DTYPES,
            "float": _FLOAT_DTYPES,
            "bool": frozenset({DType.BOOL}),
            "object": frozenset({DType.STRING}),
            "string": frozenset({DType.STRING}),
            "datetime": _DATETIME_DTYPES,
            "datetimetz": frozenset({DType.TIMESTAMP_TZ}),
            "timedelta": frozenset({DType.TIMEDELTA}),
        }

        def _resolve(types):
            if types is None:
                return None
            if isinstance(types, str):
                types = [types]
            result: set[DType] = set()
            for t in types:
                if isinstance(t, str) and t in _DTYPE_GROUPS:
                    result |= _DTYPE_GROUPS[t]
                elif isinstance(t, str):
                    result.add(pandas_dtype_to_ir(t))
                elif isinstance(t, type) and issubclass(t, np.integer):
                    result |= _INT_DTYPES
                elif isinstance(t, type) and issubclass(t, np.floating):
                    result |= _FLOAT_DTYPES
            return result

        inc = _resolve(include)
        exc = _resolve(exclude)
        keep = []
        for col, dtype in schema.columns.items():
            if inc is not None and dtype not in inc:
                continue
            if exc is not None and dtype in exc:
                continue
            keep.append(col)
        return TracedDataFrame(self._ctx, Project(self._ir, keep))

    def sample(self, n=None, frac=None, **kwargs) -> TracedDataFrame:
        df = self._ensure_materialized()
        result = df.sample(n=n, frac=frac, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def to_dict(self, *args, **kwargs):
        df = self._ensure_materialized()
        return df.to_dict(*args, **kwargs)

    def to_json(self, *args, **kwargs):
        df = self._ensure_materialized()
        return df.to_json(*args, **kwargs)

    def replace(self, to_replace, value=_SENTINEL, **kwargs) -> TracedDataFrame:
        schema = self._ir.output_schema()
        # Column-specific mapping: {col: {old: new}}
        if isinstance(to_replace, dict) and value is _SENTINEL:
            ir = self._ir
            for col_name, mapping in to_replace.items():
                if col_name in schema.columns and isinstance(mapping, dict):
                    expr: Expr = ColRef(col_name)
                    for old_val, new_val in mapping.items():
                        expr = IfThenExpr(
                            BinOp("eq", ColRef(col_name), Literal(old_val)),
                            Literal(new_val),
                            expr,
                        )
                    ir = AddColumn(ir, col_name, expr, schema.columns[col_name])
            return TracedDataFrame(self._ctx, ir)
        # Simple scalar replace across all columns
        if (
            isinstance(to_replace, (int, float, str, bool))
            and value is not _SENTINEL
            and isinstance(value, (int, float, str, bool))
        ):
            ir = self._ir
            for col_name, dtype in schema.columns.items():
                expr = IfThenExpr(
                    BinOp("eq", ColRef(col_name), Literal(to_replace)),
                    Literal(value),
                    ColRef(col_name),
                )
                ir = AddColumn(ir, col_name, expr, dtype)
            return TracedDataFrame(self._ctx, ir)
        # Complex: graph break
        df = self._materialize()
        if value is _SENTINEL:
            result = df.replace(to_replace, **kwargs)
        else:
            result = df.replace(to_replace, value, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def eval(self, expr_str, inplace=False, **kwargs):
        # Handle "col_name = expression" form
        if "=" in expr_str and not any(
            op in expr_str for op in ["==", "!=", "<=", ">="]
        ):
            lhs, rhs = expr_str.split("=", 1)
            col_name = lhs.strip()
            rhs_str = rhs.strip()
            try:
                schema = self._ir.output_schema()
                parsed_expr = _parse_query_string(rhs_str, schema)
                node = AddColumn(self._ir, col_name, parsed_expr, DType.FLOAT64)
                if inplace:
                    object.__setattr__(self, "_ir", node)
                    return self
                return TracedDataFrame(self._ctx, node)
            except (ValueError, KeyError):
                pass
        # Fallback: graph break
        df = self._materialize()
        result = df.eval(expr_str, inplace=False, **kwargs)
        if isinstance(result, pd.DataFrame):
            name = self._ctx.next_materialized_name()
            self._ctx.register_table(name, result)
            return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))
        return result

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

    def reset_index(self, drop=False, level=None, **kwargs):
        if drop:
            return self  # No-op in relational algebra
        # drop=False: index → column, graph-break only if meaningful index
        df = self._materialize()
        if isinstance(df.index, pd.RangeIndex) and df.index.name is None:
            # Default RangeIndex — nothing to reset
            return self
        result = df.reset_index(drop=False, level=level, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

    def set_index(self, keys, drop=True, append=False, **kwargs):
        df = self._materialize()
        result = df.set_index(keys, drop=drop, append=append, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

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

    def items(self):
        schema = self._ir.output_schema()
        for col in schema.columns:
            yield col, self[col]

    def keys(self):
        return self.columns

    def agg(self, func, *args, **kwargs):
        df = self._ensure_materialized()
        return df.agg(func, *args, **kwargs)

    aggregate = agg

    def map(self, func, na_action=None, **kwargs):
        df = self._materialize()
        result = df.map(func, na_action=na_action, **kwargs)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, result)
        return TracedDataFrame(self._ctx, ReadTable(name, infer_schema(result)))

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

    def insert(self, loc, column, value, allow_duplicates=False):
        df = self._materialize()
        if isinstance(value, TracedSeries):
            _, val_df = self._ctx.materialize(value._source_ir)
            col = value._column_name or next(iter(val_df.columns))
            value = val_df[col]
        df.insert(loc, column, value, allow_duplicates=allow_duplicates)
        name = self._ctx.next_materialized_name()
        self._ctx.register_table(name, df)
        new_ir = ReadTable(name, infer_schema(df))
        object.__setattr__(self, "_ir", new_ir)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, *args, **kwargs):
        df = self._ensure_materialized()
        return df.transpose(*args, **kwargs)

    def __iter__(self):
        schema = self._ir.output_schema()
        return iter(schema.columns)

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

    @property
    def ndim(self):
        return 2

    @property
    def index(self):
        df = self._ensure_materialized()
        return df.index

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
        self.__signature__ = inspect.signature(fn)

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

    def reset_cache(self):
        """Clear the cached execution plans."""
        self._cached_plans.clear()

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
                reason = (
                    f" ({seg.reason})"
                    if seg.reason and seg.reason != seg.operation
                    else ""
                )
                lines.append(
                    f"  [{i}] EAGER: {seg.operation}{reason} -> {seg.output_names}"
                )
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
def compilable(fn: Callable) -> CompiledFunction: ...


@overload
def compilable(
    fn: None = ..., *, backend: Backend | None = ...
) -> Callable[[Callable], CompiledFunction]: ...


def compilable(
    fn: Callable | None = None, *, backend: Backend | None = None
) -> CompiledFunction | Callable[[Callable], CompiledFunction]:
    """
    Decorator that JIT-compiles a pandas function to an IR graph.

    Graph breaks are handled transparently via materialization.

    Parameters
    ----------
    fn : callable, optional
        The function to compile. When used as ``@compilable`` (without
        parentheses), this is the decorated function.
    backend : Backend or None, default None
        Execution backend. ``None`` selects ``AceroBackend`` when pyarrow
        is available, otherwise ``PandasBackend``.

    Returns
    -------
    CompiledFunction or callable
        When called as ``@compilable``, returns a ``CompiledFunction``.
        When called as ``@compilable(backend=...)``, returns a decorator
        that produces a ``CompiledFunction``.

    Examples
    --------
    >>> @compilable
    ... def process(df):
    ...     return df[df["price"] > 100]

    >>> @compilable(backend=PandasBackend())
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
