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
            if func == "count":
                cols[out_name] = DType.INT64
            elif func in ("avg", "std", "var"):
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
    lower_offset: int | None = None  # None = unbounded
    upper_offset: int = 0  # 0 = current row


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
