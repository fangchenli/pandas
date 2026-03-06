"""
ir.py — Intermediate Representation for pandas JIT compilation.

Defines the relational algebra IR that sits between traced pandas operations
and the Substrait output. The IR is a tree of relation nodes (ReadTable,
Filter, Project, ...) with typed scalar expressions (ColRef, Literal, BinOp,
UnaryOp).

This module is standalone — no internal dependencies.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from enum import (
    Enum,
    auto,
)
from typing import Any

# ---------------------------------------------------------------------------
# 1. Type system
# ---------------------------------------------------------------------------


class DType(Enum):
    # Signed integers
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    # Unsigned integers
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()
    # Floating point
    FLOAT32 = auto()
    FLOAT64 = auto()
    # String / binary
    STRING = auto()
    BINARY = auto()
    # Boolean
    BOOL = auto()
    # Temporal
    DATE = auto()
    TIME = auto()
    TIMESTAMP = auto()
    TIMESTAMP_TZ = auto()
    TIMEDELTA = auto()
    # Decimal
    DECIMAL = auto()


_PANDAS_DTYPE_MAP: dict[str, DType] = {
    # Signed integers — numpy
    "int8": DType.INT8,
    "int16": DType.INT16,
    "int32": DType.INT32,
    "int64": DType.INT64,
    # Signed integers — pandas nullable
    "Int8": DType.INT8,
    "Int16": DType.INT16,
    "Int32": DType.INT32,
    "Int64": DType.INT64,
    # Unsigned integers — numpy
    "uint8": DType.UINT8,
    "uint16": DType.UINT16,
    "uint32": DType.UINT32,
    "uint64": DType.UINT64,
    # Unsigned integers — pandas nullable
    "UInt8": DType.UINT8,
    "UInt16": DType.UINT16,
    "UInt32": DType.UINT32,
    "UInt64": DType.UINT64,
    # Floating point
    "float32": DType.FLOAT32,
    "float64": DType.FLOAT64,
    "Float32": DType.FLOAT32,
    "Float64": DType.FLOAT64,
    # String
    "object": DType.STRING,
    "string": DType.STRING,
    "str": DType.STRING,
    "string[python]": DType.STRING,
    "string[pyarrow]": DType.STRING,
    # Binary
    "bytes": DType.BINARY,
    # Boolean
    "bool": DType.BOOL,
    "boolean": DType.BOOL,
    # Temporal
    "datetime64[ns]": DType.TIMESTAMP,
    "datetime64[us]": DType.TIMESTAMP,
    "datetime64[ms]": DType.TIMESTAMP,
    "datetime64[s]": DType.TIMESTAMP,
    "date": DType.DATE,
    "timedelta64[ns]": DType.TIMEDELTA,
    "timedelta64[us]": DType.TIMEDELTA,
    "timedelta64[ms]": DType.TIMEDELTA,
    "timedelta64[s]": DType.TIMEDELTA,
}

# Numeric DTypes — used by aggregation logic to filter numeric columns
NUMERIC_DTYPES: frozenset[DType] = frozenset(
    {
        DType.INT8,
        DType.INT16,
        DType.INT32,
        DType.INT64,
        DType.UINT8,
        DType.UINT16,
        DType.UINT32,
        DType.UINT64,
        DType.FLOAT32,
        DType.FLOAT64,
        DType.DECIMAL,
    }
)


def pandas_dtype_to_ir(dtype_str: str) -> DType:
    """
    Map a pandas/numpy dtype string to a DType.

    Handles exact matches from ``_PANDAS_DTYPE_MAP`` and falls back to
    prefix matching for parameterised types like ``datetime64[ns, UTC]``.
    """
    if dtype_str in _PANDAS_DTYPE_MAP:
        return _PANDAS_DTYPE_MAP[dtype_str]
    # datetime64 with timezone  →  TIMESTAMP_TZ
    if dtype_str.startswith("datetime64[") and "," in dtype_str:
        return DType.TIMESTAMP_TZ
    # bare datetime64 without resolution
    if dtype_str.startswith("datetime64"):
        return DType.TIMESTAMP
    # bare timedelta64 without resolution
    if dtype_str.startswith("timedelta64"):
        return DType.TIMEDELTA
    return DType.STRING


@dataclass
class Schema:
    """Column names and their types."""

    columns: dict[str, DType] = field(default_factory=dict)

    def column_index(self, name: str) -> int:
        cols = list(self.columns.keys())
        return cols.index(name)

    def column_names(self) -> list[str]:
        return list(self.columns.keys())


# ---------------------------------------------------------------------------
# 2. IR nodes — relational algebra
# ---------------------------------------------------------------------------


class IRNode:
    """Base class for all IR nodes."""

    def output_schema(self) -> Schema:
        raise NotImplementedError


class ReadTable(IRNode):
    """Scan a named table."""

    def __init__(self, name: str, schema: Schema):
        self.name = name
        self.schema = schema

    def output_schema(self) -> Schema:
        return self.schema


class Filter(IRNode):
    """Filter rows by a predicate expression."""

    def __init__(self, input: IRNode, predicate: Expr):
        self.input = input
        self.predicate = predicate

    def output_schema(self) -> Schema:
        return self.input.output_schema()


class Project(IRNode):
    """Select a subset of columns."""

    def __init__(self, input: IRNode, columns: list[str]):
        self.input = input
        self.columns = columns

    def output_schema(self) -> Schema:
        parent = self.input.output_schema()
        return Schema({c: parent.columns[c] for c in self.columns})


class AddColumn(IRNode):
    """Add a computed column."""

    def __init__(self, input: IRNode, name: str, expr: Expr, dtype: DType):
        self.input = input
        self.name = name
        self.expr = expr
        self.dtype = dtype

    def output_schema(self) -> Schema:
        parent = self.input.output_schema()
        new_cols = dict(parent.columns)
        new_cols[self.name] = self.dtype
        return Schema(new_cols)


class Sort(IRNode):
    """Sort by one or more columns."""

    def __init__(self, input: IRNode, keys: list[tuple[str, bool]]):
        # keys: list of (column_name, ascending)
        self.input = input
        self.keys = keys

    def output_schema(self) -> Schema:
        return self.input.output_schema()


class Limit(IRNode):
    """Take N rows, optionally skipping the first *offset* rows."""

    def __init__(self, input: IRNode, n: int, offset: int = 0):
        self.input = input
        self.n = n
        self.offset = offset

    def output_schema(self) -> Schema:
        return self.input.output_schema()


class Aggregate(IRNode):
    """Group-by aggregation."""

    def __init__(
        self,
        input: IRNode,
        group_keys: list[str],
        agg_specs: list[tuple[str, str, str]],
    ):
        # agg_specs: list of (output_name, source_col, agg_func)
        self.input = input
        self.group_keys = group_keys
        self.agg_specs = agg_specs

    def output_schema(self) -> Schema:
        parent = self.input.output_schema()
        cols: dict[str, DType] = {}
        for k in self.group_keys:
            cols[k] = parent.columns[k]
        for out_name, src_col, func in self.agg_specs:
            if func in ("count", "count_distinct"):
                cols[out_name] = DType.INT64
            elif func in ("avg", "mean", "std", "var"):
                cols[out_name] = DType.FLOAT64
            else:
                cols[out_name] = parent.columns[src_col]
        return Schema(cols)


class Join(IRNode):
    """Join two relations on one or more key columns."""

    def __init__(
        self,
        left: IRNode,
        right: IRNode,
        left_on: str | list[str],
        right_on: str | list[str],
        how: str = "inner",
    ):
        self.left = left
        self.right = right
        self.left_on = [left_on] if isinstance(left_on, str) else list(left_on)
        self.right_on = [right_on] if isinstance(right_on, str) else list(right_on)
        self.how = how

    def output_schema(self) -> Schema:
        left_schema = self.left.output_schema()
        right_schema = self.right.output_schema()
        right_keys = set(self.right_on)
        cols = dict(left_schema.columns)
        for k, v in right_schema.columns.items():
            if k not in cols and k not in right_keys:
                cols[k] = v
        return Schema(cols)


class Union(IRNode):
    """Concatenate multiple relations (UNION ALL)."""

    def __init__(self, inputs: list[IRNode]):
        self.inputs = list(inputs)

    def output_schema(self) -> Schema:
        return self.inputs[0].output_schema()


class Distinct(IRNode):
    """Remove duplicate rows."""

    def __init__(self, input: IRNode, columns: list[str]):
        self.input = input
        self.columns = list(columns)

    def output_schema(self) -> Schema:
        return self.input.output_schema()


@dataclass
class WindowSpec:
    """Window frame specification."""

    kind: str = "rows"  # "rows" or "range"
    lower_offset: int | None = None  # None = unbounded preceding
    upper_offset: int | None = 0  # 0 = current row, None = unbounded following


class Window(IRNode):
    """Window function computation (rolling / expanding / lag / lead)."""

    def __init__(
        self,
        input: IRNode,
        window_funcs: list[tuple[str, str, str, int]],
        window_spec: WindowSpec,
        partition_by: list[str] | None = None,
        order_by: list[tuple[str, bool]] | None = None,
    ):
        # window_funcs: list of (out_name, src_col, func, offset)
        # offset is 0 for aggregate window funcs, >0 for lag/lead
        self.input = input
        self.window_funcs = window_funcs
        self.window_spec = window_spec
        self.partition_by = partition_by or []
        self.order_by = order_by or []

    def output_schema(self) -> Schema:
        parent = self.input.output_schema()
        cols: dict[str, DType] = {}
        func_output_cols = {
            out: (src, func) for out, src, func, _offset in self.window_funcs
        }
        for col_name, dtype in parent.columns.items():
            if col_name in func_output_cols:
                _, func = func_output_cols[col_name]
                if func in (
                    "avg",
                    "mean",
                    "std",
                    "var",
                    "rank",
                    "dense_rank",
                    "row_number",
                ):
                    cols[col_name] = DType.FLOAT64
                elif func == "count":
                    cols[col_name] = DType.INT64
                else:
                    cols[col_name] = dtype
            else:
                cols[col_name] = dtype
        # Add any new output columns not already in parent
        for out_name, src_col, func, _offset in self.window_funcs:
            if out_name not in cols:
                if func in (
                    "avg",
                    "mean",
                    "std",
                    "var",
                    "rank",
                    "dense_rank",
                    "row_number",
                ):
                    cols[out_name] = DType.FLOAT64
                elif func == "count":
                    cols[out_name] = DType.INT64
                else:
                    cols[out_name] = parent.columns.get(src_col, DType.FLOAT64)
        return Schema(cols)


class RenameColumns(IRNode):
    """Rename columns (maps to Project with renamed outputs in Substrait)."""

    def __init__(self, input: IRNode, mapping: dict[str, str]):
        self.input = input
        self.mapping = mapping  # {old_name: new_name}

    def output_schema(self) -> Schema:
        parent = self.input.output_schema()
        return Schema({self.mapping.get(n, n): d for n, d in parent.columns.items()})


# ---------------------------------------------------------------------------
# 3. Expressions — predicate & scalar expression trees
# ---------------------------------------------------------------------------


class Expr:
    """Base expression node."""


class ColRef(Expr):
    """Reference to a column by name."""

    def __init__(self, name: str):
        self.name = name

    def __gt__(self, other):
        return BinOp("gt", self, _wrap_literal(other))

    def __ge__(self, other):
        return BinOp("gte", self, _wrap_literal(other))

    def __lt__(self, other):
        return BinOp("lt", self, _wrap_literal(other))

    def __le__(self, other):
        return BinOp("lte", self, _wrap_literal(other))

    def __eq__(self, other):
        return BinOp("eq", self, _wrap_literal(other))

    def __ne__(self, other):
        return BinOp("ne", self, _wrap_literal(other))

    def __add__(self, other):
        return BinOp("add", self, _wrap_literal(other))

    def __sub__(self, other):
        return BinOp("sub", self, _wrap_literal(other))

    def __mul__(self, other):
        return BinOp("mul", self, _wrap_literal(other))

    def __truediv__(self, other):
        return BinOp("div", self, _wrap_literal(other))


class Literal(Expr):
    """Literal scalar value."""

    def __init__(self, value: Any, dtype: DType | None = None):
        self.value = value
        if dtype is None:
            if isinstance(value, bool):
                self.dtype = DType.BOOL
            elif isinstance(value, int):
                self.dtype = DType.INT64
            elif isinstance(value, float):
                self.dtype = DType.FLOAT64
            elif isinstance(value, str):
                self.dtype = DType.STRING
            elif isinstance(value, bytes):
                self.dtype = DType.BINARY
            else:
                self.dtype = DType.STRING
        else:
            self.dtype = dtype


class BinOp(Expr):
    """Binary operation."""

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def __and__(self, other):
        return BinOp("and", self, other)

    def __or__(self, other):
        return BinOp("or", self, other)


class UnaryOp(Expr):
    """Unary operation (e.g., NOT, IS NULL)."""

    def __init__(self, op: str, operand: Expr):
        self.op = op
        self.operand = operand


class IfThenExpr(Expr):
    """Conditional expression: IF condition THEN then_expr ELSE else_expr."""

    def __init__(self, condition: Expr, then_expr: Expr, else_expr: Expr):
        self.condition = condition
        self.then_expr = then_expr
        self.else_expr = else_expr


class CastExpr(Expr):
    """Cast an expression to a target DType."""

    def __init__(self, input: Expr, target_dtype: DType):
        self.input = input
        self.target_dtype = target_dtype


class SingularOrList(Expr):
    """IN-list expression: value IN (option1, option2, ...).

    More efficient than OR(eq, eq, ...) chains — maps directly to
    Substrait's SingularOrList expression.
    """

    def __init__(self, value: Expr, options: list[Expr]):
        self.value = value
        self.options = options


class FunctionCall(Expr):
    """Named function call with arguments and options.

    Used for functions like ``extract(component, timestamp)`` or
    ``str_upper(col)`` that don't fit the BinOp/UnaryOp pattern.
    """

    def __init__(
        self,
        func_name: str,
        args: list[Expr],
        options: dict[str, str] | None = None,
        return_dtype: DType = DType.INT64,
    ):
        self.func_name = func_name
        self.args = args
        self.options = options or {}
        self.return_dtype = return_dtype


@dataclass
class ScalarSubquery(Expr):
    """A single-row, single-column subquery used as a scalar expression.

    Wraps an Aggregate IR node (with no group_keys) so that the
    aggregation result can be used inline in arithmetic expressions
    without triggering a graph break.

    Maps to Substrait's ``Expression.Subquery.Scalar``.
    """

    agg_node: IRNode  # Aggregate with no group_keys
    dtype: DType


def _wrap_literal(v):
    if isinstance(v, Expr):
        return v
    return Literal(v)


# ---------------------------------------------------------------------------
# 4. Explain — pretty-print IR trees
# ---------------------------------------------------------------------------

_OP_SYMBOLS = {
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "eq": "==",
    "ne": "!=",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "and": "AND",
    "or": "OR",
}


def explain_ir(ir: IRNode, indent: int = 0) -> str:
    prefix = "  " * indent
    match ir:
        case ReadTable(name=name, schema=schema):
            cols = ", ".join(f"{n}:{t.name}" for n, t in schema.columns.items())
            return f"{prefix}ReadTable({name}) [{cols}]"
        case Filter(input=inp, predicate=pred):
            child = explain_ir(inp, indent + 1)
            return f"{prefix}Filter({explain_expr(pred)})\n{child}"
        case Project(input=inp, columns=columns):
            child = explain_ir(inp, indent + 1)
            return f"{prefix}Project({', '.join(columns)})\n{child}"
        case AddColumn(input=inp, name=name, expr=expr):
            child = explain_ir(inp, indent + 1)
            return f"{prefix}AddColumn({name} = {explain_expr(expr)})\n{child}"
        case Sort(input=inp, keys=keys):
            child = explain_ir(inp, indent + 1)
            key_str = ", ".join(f"{k} {'ASC' if a else 'DESC'}" for k, a in keys)
            return f"{prefix}Sort({key_str})\n{child}"
        case Limit(input=inp, n=n, offset=off):
            child = explain_ir(inp, indent + 1)
            off_str = f", offset={off}" if off else ""
            return f"{prefix}Limit({n}{off_str})\n{child}"
        case Aggregate(input=inp, group_keys=gk, agg_specs=specs):
            child = explain_ir(inp, indent + 1)
            groups = ", ".join(gk)
            aggs = ", ".join(f"{o}={f}({s})" for o, s, f in specs)
            return f"{prefix}Aggregate(by=[{groups}], {aggs})\n{child}"
        case Join(left=left, right=right, left_on=lo, right_on=ro, how=how):
            left_str = explain_ir(left, indent + 1)
            right_str = explain_ir(right, indent + 1)
            keys = ", ".join(f"{lk}={rk}" for lk, rk in zip(lo, ro, strict=True))
            return f"{prefix}Join({how}, {keys})\n{left_str}\n{right_str}"
        case Union(inputs=inputs):
            children = "\n".join(explain_ir(inp, indent + 1) for inp in inputs)
            return f"{prefix}Union({len(inputs)} inputs)\n{children}"
        case Distinct(input=inp, columns=cols):
            child = explain_ir(inp, indent + 1)
            col_str = ", ".join(cols)
            return f"{prefix}Distinct({col_str})\n{child}"
        case Window(
            input=inp,
            window_funcs=funcs,
            window_spec=spec,
            partition_by=part,
            order_by=order,
        ):
            child = explain_ir(inp, indent + 1)
            funcs_str = ", ".join(
                f"{o}={f}({s}, {off})" if off else f"{o}={f}({s})"
                for o, s, f, off in funcs
            )
            frame = "UNBOUNDED" if spec.lower_offset is None else str(spec.lower_offset)
            parts = [f"funcs=[{funcs_str}]", f"frame={frame}..{spec.upper_offset}"]
            if part:
                parts.append(f"partition=[{', '.join(part)}]")
            if order:
                order_str = ", ".join(f"{k} {'ASC' if a else 'DESC'}" for k, a in order)
                parts.append(f"order=[{order_str}]")
            return f"{prefix}Window({', '.join(parts)})\n{child}"
        case RenameColumns(input=inp, mapping=mapping):
            child = explain_ir(inp, indent + 1)
            map_str = ", ".join(f"{k}->{v}" for k, v in mapping.items())
            return f"{prefix}Rename({map_str})\n{child}"
        case _:
            return f"{prefix}Unknown({type(ir).__name__})"


def explain_expr(expr: Expr) -> str:
    match expr:
        case ColRef(name=name):
            return f"${name}"
        case Literal(value=value):
            return repr(value)
        case BinOp(op=op, left=left, right=right):
            sym = _OP_SYMBOLS.get(op, op)
            return f"({explain_expr(left)} {sym} {explain_expr(right)})"
        case UnaryOp(op=op, operand=operand):
            return f"{op}({explain_expr(operand)})"
        case IfThenExpr(condition=cond, then_expr=then, else_expr=els):
            return (
                f"IF({explain_expr(cond)}, {explain_expr(then)}, {explain_expr(els)})"
            )
        case CastExpr(input=inp, target_dtype=dtype):
            return f"CAST({explain_expr(inp)} AS {dtype.name})"
        case SingularOrList(value=value, options=opts):
            val_str = explain_expr(value)
            opts_str = ", ".join(explain_expr(o) for o in opts)
            return f"{val_str} IN ({opts_str})"
        case FunctionCall(func_name=name, args=args, options=opts):
            args_str = ", ".join(explain_expr(a) for a in args)
            if opts:
                opts_str = ", ".join(f"{k}={v}" for k, v in opts.items())
                return f"{name}({args_str}, {opts_str})"
            return f"{name}({args_str})"
        case ScalarSubquery(agg_node=agg, dtype=dtype):
            return f"SCALAR_SUBQUERY({explain_ir(agg)}, {dtype.name})"
        case _:
            return "?"


# ---------------------------------------------------------------------------
# 6. IR optimization — projection pruning
# ---------------------------------------------------------------------------


def _collect_used_columns(expr: Expr) -> set[str]:
    """Collect all column names referenced by an expression."""
    if isinstance(expr, ColRef):
        return {expr.name}
    elif isinstance(expr, BinOp):
        return _collect_used_columns(expr.left) | _collect_used_columns(expr.right)
    elif isinstance(expr, UnaryOp):
        return _collect_used_columns(expr.operand)
    elif isinstance(expr, IfThenExpr):
        return (
            _collect_used_columns(expr.condition)
            | _collect_used_columns(expr.then_expr)
            | _collect_used_columns(expr.else_expr)
        )
    elif isinstance(expr, CastExpr):
        return _collect_used_columns(expr.input)
    elif isinstance(expr, FunctionCall):
        result: set[str] = set()
        for arg in expr.args:
            result |= _collect_used_columns(arg)
        return result
    elif isinstance(expr, SingularOrList):
        result = _collect_used_columns(expr.value)
        for opt in expr.options:
            result |= _collect_used_columns(opt)
        return result
    elif isinstance(expr, ScalarSubquery):
        return set()  # subquery is self-contained
    return set()  # Literal, etc.


def optimize_projections(ir: IRNode, needed: set[str] | None = None) -> IRNode:
    """
    Projection pruning optimization pass.

    Walks the IR tree top-down and prunes unused columns at each level
    by recursing with a narrowed "needed" set.
    """
    if needed is None:
        needed = set(ir.output_schema().column_names())

    if isinstance(ir, ReadTable):
        available = set(ir.schema.columns.keys())
        keep = needed & available
        if keep < available:
            ordered = [c for c in ir.schema.columns if c in keep]
            return Project(ir, ordered)
        return ir

    elif isinstance(ir, Filter):
        expr_cols = _collect_used_columns(ir.predicate)
        child_needed = needed | expr_cols
        new_input = optimize_projections(ir.input, child_needed)
        result = Filter(new_input, ir.predicate)
        available = set(result.output_schema().column_names())
        if needed < available:
            ordered = [c for c in result.output_schema().column_names() if c in needed]
            return Project(result, ordered)
        return result

    elif isinstance(ir, Project):
        child_needed = set(ir.columns)
        new_input = optimize_projections(ir.input, child_needed)
        return Project(new_input, ir.columns)

    elif isinstance(ir, AddColumn):
        expr_cols = _collect_used_columns(ir.expr)
        child_needed = (needed - {ir.name}) | expr_cols
        new_input = optimize_projections(ir.input, child_needed)
        return AddColumn(new_input, ir.name, ir.expr, ir.dtype)

    elif isinstance(ir, Sort):
        sort_cols = {k for k, _ in ir.keys}
        child_needed = needed | sort_cols
        new_input = optimize_projections(ir.input, child_needed)
        return Sort(new_input, ir.keys)

    elif isinstance(ir, Limit):
        new_input = optimize_projections(ir.input, needed)
        return Limit(new_input, ir.n, ir.offset)

    elif isinstance(ir, Aggregate):
        group_cols = set(ir.group_keys)
        src_cols = {src for _, src, _ in ir.agg_specs}
        child_needed = group_cols | src_cols
        new_input = optimize_projections(ir.input, child_needed)
        return Aggregate(new_input, ir.group_keys, ir.agg_specs)

    elif isinstance(ir, Join):
        left_cols = set(ir.left.output_schema().column_names())
        right_cols = set(ir.right.output_schema().column_names())
        left_needed: set[str] = set()
        right_needed: set[str] = set()
        for col in needed:
            if col in left_cols:
                left_needed.add(col)
            elif col in right_cols:
                right_needed.add(col)
        for k in ir.left_on:
            left_needed.add(k)
        for k in ir.right_on:
            right_needed.add(k)
        new_left = optimize_projections(ir.left, left_needed)
        new_right = optimize_projections(ir.right, right_needed)
        return Join(new_left, new_right, ir.left_on, ir.right_on, ir.how)

    elif isinstance(ir, Window):
        child_needed = set(needed)
        for _, src, _, _ in ir.window_funcs:
            child_needed.add(src)
        for col in ir.partition_by:
            child_needed.add(col)
        for col, _ in ir.order_by:
            child_needed.add(col)
        new_input = optimize_projections(ir.input, child_needed)
        return Window(
            new_input,
            ir.window_funcs,
            ir.window_spec,
            ir.partition_by,
            ir.order_by,
        )

    elif isinstance(ir, Union):
        return Union([optimize_projections(inp, needed) for inp in ir.inputs])

    elif isinstance(ir, Distinct):
        child_needed = needed | set(ir.columns)
        new_input = optimize_projections(ir.input, child_needed)
        return Distinct(new_input, ir.columns)

    elif isinstance(ir, RenameColumns):
        inv_map = {v: k for k, v in ir.mapping.items()}
        child_needed = {inv_map.get(c, c) for c in needed}
        new_input = optimize_projections(ir.input, child_needed)
        return RenameColumns(new_input, ir.mapping)

    return ir


# ---------------------------------------------------------------------------
# IR optimization — predicate pushdown
# ---------------------------------------------------------------------------


def _expr_uses_only(expr: Expr, available: set[str]) -> bool:
    """Check if all ColRefs in expr are within the available set."""
    return _collect_used_columns(expr).issubset(available)


def push_predicates(ir: IRNode) -> IRNode:
    """
    Push Filter nodes closer to data sources.

    Recursively transforms the IR tree so that filters are applied as early
    as possible, reducing the amount of data flowing through the plan.
    """
    if isinstance(ir, ReadTable):
        return ir

    elif isinstance(ir, Filter):
        # First, recursively push predicates in the child
        child = push_predicates(ir.input)
        pred = ir.predicate

        # Merge adjacent filters: Filter(Filter(x, p1), p2) → Filter(x, AND(p1, p2))
        if isinstance(child, Filter):
            merged = BinOp("and", child.predicate, pred)
            return Filter(child.input, merged)

        # Push below AddColumn if pred doesn't use the added/overwritten column
        if isinstance(child, AddColumn):
            pred_cols = _collect_used_columns(pred)
            if child.name not in pred_cols:
                return AddColumn(
                    Filter(child.input, pred),
                    child.name,
                    child.expr,
                    child.dtype,
                )
            return Filter(child, pred)

        # Push below Sort (doesn't change semantics)
        if isinstance(child, Sort):
            return Sort(Filter(child.input, pred), child.keys)

        # Do NOT push below Limit (changes semantics)
        if isinstance(child, Limit):
            return Filter(child, pred)

        # Push into Union branches
        if isinstance(child, Union):
            return Union([Filter(inp, pred) for inp in child.inputs])

        # Push below Project if pred columns exist in child
        if isinstance(child, Project):
            child_cols = set(child.input.output_schema().column_names())
            if _expr_uses_only(pred, child_cols):
                return Project(Filter(child.input, pred), child.columns)
            return Filter(child, pred)

        # Push below Window if pred doesn't reference window output columns
        if isinstance(child, Window):
            window_out_cols = {out for out, _, _, _ in child.window_funcs}
            if not _collect_used_columns(pred) & window_out_cols:
                return Window(
                    Filter(child.input, pred),
                    child.window_funcs,
                    child.window_spec,
                    partition_by=child.partition_by,
                    order_by=child.order_by,
                )
            return Filter(child, pred)

        # Push into Join sides
        if isinstance(child, Join):
            left_cols = set(child.left.output_schema().column_names())
            right_cols = set(child.right.output_schema().column_names())
            pred_cols = _collect_used_columns(pred)
            if pred_cols.issubset(left_cols) and not pred_cols.intersection(right_cols):
                return Join(
                    Filter(child.left, pred),
                    child.right,
                    child.left_on,
                    child.right_on,
                    child.how,
                )
            elif pred_cols.issubset(right_cols) and not pred_cols.intersection(
                left_cols
            ):
                return Join(
                    child.left,
                    Filter(child.right, pred),
                    child.left_on,
                    child.right_on,
                    child.how,
                )
            return Filter(child, pred)

        # Push below RenameColumns if we can map pred columns back
        if isinstance(child, RenameColumns):
            inv_map = {v: k for k, v in child.mapping.items()}
            child_cols = set(child.input.output_schema().column_names())
            pred_cols = _collect_used_columns(pred)
            mapped_cols = {inv_map.get(c, c) for c in pred_cols}
            if mapped_cols.issubset(child_cols):
                mapped_pred = _remap_expr(pred, inv_map)
                return RenameColumns(Filter(child.input, mapped_pred), child.mapping)
            return Filter(child, pred)

        return Filter(child, pred)

    elif isinstance(ir, Project):
        return Project(push_predicates(ir.input), ir.columns)

    elif isinstance(ir, AddColumn):
        return AddColumn(push_predicates(ir.input), ir.name, ir.expr, ir.dtype)

    elif isinstance(ir, Sort):
        return Sort(push_predicates(ir.input), ir.keys)

    elif isinstance(ir, Limit):
        return Limit(push_predicates(ir.input), ir.n, ir.offset)

    elif isinstance(ir, Aggregate):
        return Aggregate(push_predicates(ir.input), ir.group_keys, ir.agg_specs)

    elif isinstance(ir, Join):
        return Join(
            push_predicates(ir.left),
            push_predicates(ir.right),
            ir.left_on,
            ir.right_on,
            ir.how,
        )

    elif isinstance(ir, Window):
        return Window(
            push_predicates(ir.input),
            ir.window_funcs,
            ir.window_spec,
            partition_by=ir.partition_by,
            order_by=ir.order_by,
        )

    elif isinstance(ir, Union):
        return Union([push_predicates(inp) for inp in ir.inputs])

    elif isinstance(ir, Distinct):
        return Distinct(push_predicates(ir.input), ir.columns)

    elif isinstance(ir, RenameColumns):
        return RenameColumns(push_predicates(ir.input), ir.mapping)

    return ir


def _remap_expr(expr: Expr, mapping: dict[str, str]) -> Expr:
    """Remap ColRef names through a column rename mapping."""
    if isinstance(expr, ColRef):
        return ColRef(mapping.get(expr.name, expr.name))
    elif isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _remap_expr(expr.left, mapping),
            _remap_expr(expr.right, mapping),
        )
    elif isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _remap_expr(expr.operand, mapping))
    elif isinstance(expr, FunctionCall):
        return FunctionCall(
            expr.func_name,
            [_remap_expr(a, mapping) for a in expr.args],
            options=expr.options,
            return_dtype=expr.return_dtype,
        )
    elif isinstance(expr, Literal):
        return expr
    return expr


# ---------------------------------------------------------------------------
# IR optimization — constant folding
# ---------------------------------------------------------------------------

_BINOP_EVAL: dict[str, object] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if b != 0 else None,
    "floordiv": lambda a, b: a // b if b != 0 else None,
    "mod": lambda a, b: a % b if b != 0 else None,
    "pow": lambda a, b: a**b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
}

_UNARYOP_EVAL: dict[str, object] = {
    "not": lambda a: not a,
    "negate": lambda a: -a,
    "abs": lambda a: abs(a),
}


def fold_constants(expr: Expr) -> Expr:
    """
    Simplify expressions with constant operands.

    Folds literal-only operations and applies identity simplifications.
    """
    if isinstance(expr, BinOp):
        left = fold_constants(expr.left)
        right = fold_constants(expr.right)

        # Full constant fold: Literal op Literal → Literal
        if isinstance(left, Literal) and isinstance(right, Literal):
            fn = _BINOP_EVAL.get(expr.op)
            if fn is not None:
                try:
                    result = fn(left.value, right.value)
                    if result is not None:
                        return Literal(result)
                except (TypeError, ArithmeticError):
                    pass

        # Identity simplifications
        if isinstance(right, Literal):
            if expr.op == "add" and right.value == 0:
                return left
            if expr.op == "sub" and right.value == 0:
                return left
            if expr.op == "mul" and right.value == 1:
                return left
            if expr.op == "mul" and right.value == 0:
                return Literal(0)
            if expr.op == "and" and right.value is True:
                return left
            if expr.op == "or" and right.value is False:
                return left

        if isinstance(left, Literal):
            if expr.op == "add" and left.value == 0:
                return right
            if expr.op == "mul" and left.value == 1:
                return right
            if expr.op == "mul" and left.value == 0:
                return Literal(0)
            if expr.op == "and" and left.value is True:
                return right
            if expr.op == "or" and left.value is False:
                return right

        return BinOp(expr.op, left, right)

    elif isinstance(expr, UnaryOp):
        operand = fold_constants(expr.operand)

        # Full constant fold
        if isinstance(operand, Literal):
            fn = _UNARYOP_EVAL.get(expr.op)
            if fn is not None:
                try:
                    return Literal(fn(operand.value))
                except (TypeError, ArithmeticError):
                    pass

        # NOT(NOT(x)) = x
        if expr.op == "not" and isinstance(operand, UnaryOp) and operand.op == "not":
            return operand.operand

        return UnaryOp(expr.op, operand)

    elif isinstance(expr, FunctionCall):
        return FunctionCall(
            expr.func_name,
            [fold_constants(a) for a in expr.args],
            options=expr.options,
            return_dtype=expr.return_dtype,
        )

    elif isinstance(expr, IfThenExpr):
        return IfThenExpr(
            fold_constants(expr.condition),
            fold_constants(expr.then_expr),
            fold_constants(expr.else_expr),
        )

    elif isinstance(expr, CastExpr):
        return CastExpr(fold_constants(expr.input), expr.target_dtype)

    return expr


def optimize_expressions(ir: IRNode) -> IRNode:
    """Walk IR tree, apply fold_constants to all expressions."""
    if isinstance(ir, ReadTable):
        return ir

    elif isinstance(ir, Filter):
        new_input = optimize_expressions(ir.input)
        new_pred = fold_constants(ir.predicate)
        return Filter(new_input, new_pred)

    elif isinstance(ir, AddColumn):
        new_input = optimize_expressions(ir.input)
        new_expr = fold_constants(ir.expr)
        return AddColumn(new_input, ir.name, new_expr, ir.dtype)

    elif isinstance(ir, Project):
        return Project(optimize_expressions(ir.input), ir.columns)

    elif isinstance(ir, Sort):
        return Sort(optimize_expressions(ir.input), ir.keys)

    elif isinstance(ir, Limit):
        return Limit(optimize_expressions(ir.input), ir.n, ir.offset)

    elif isinstance(ir, Aggregate):
        return Aggregate(optimize_expressions(ir.input), ir.group_keys, ir.agg_specs)

    elif isinstance(ir, Join):
        return Join(
            optimize_expressions(ir.left),
            optimize_expressions(ir.right),
            ir.left_on,
            ir.right_on,
            ir.how,
        )

    elif isinstance(ir, Window):
        return Window(
            optimize_expressions(ir.input),
            ir.window_funcs,
            ir.window_spec,
            partition_by=ir.partition_by,
            order_by=ir.order_by,
        )

    elif isinstance(ir, Union):
        return Union([optimize_expressions(inp) for inp in ir.inputs])

    elif isinstance(ir, Distinct):
        return Distinct(optimize_expressions(ir.input), ir.columns)

    elif isinstance(ir, RenameColumns):
        return RenameColumns(optimize_expressions(ir.input), ir.mapping)

    return ir
