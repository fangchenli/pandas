"""
compiler.py — Substrait compiler and execution backends.

Compiles the relational IR (from ir.py) into Substrait protobuf plans,
and provides backends to execute those plans (PandasBackend interpreter,
or custom user-provided backends).
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
import logging
import re
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import Callable

import substrait.algebra_pb2 as stalg
from substrait.extensions.extensions_pb2 import (
    SimpleExtensionDeclaration,
    SimpleExtensionURI,
)
import substrait.plan_pb2 as stp
import substrait.type_pb2 as stt
from substrait.version import substrait_version

import pandas as pd
from pandas.compile.ir import (
    AddColumn,
    Aggregate,
    BinOp,
    ColRef,
    DType,
    Expr,
    Filter,
    IRNode,
    Join,
    Limit,
    Literal,
    Project,
    ReadTable,
    RenameColumns,
    Schema,
    Sort,
    UnaryOp,
    explain_ir,
    pandas_dtype_to_ir,
)

log = logging.getLogger("pandas.compile")


# ---------------------------------------------------------------------------
# 1. DType → Substrait type mapping (moved from ir.py to keep IR engine-agnostic)
# ---------------------------------------------------------------------------

_DTYPE_TO_SUBSTRAIT: dict[DType, Any] = {
    DType.INT8: lambda: stt.Type(
        i8=stt.Type.I8(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.INT16: lambda: stt.Type(
        i16=stt.Type.I16(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.INT32: lambda: stt.Type(
        i32=stt.Type.I32(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.INT64: lambda: stt.Type(
        i64=stt.Type.I64(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.UINT8: lambda: stt.Type(
        i16=stt.Type.I16(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.UINT16: lambda: stt.Type(
        i32=stt.Type.I32(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.UINT32: lambda: stt.Type(
        i64=stt.Type.I64(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.UINT64: lambda: stt.Type(
        i64=stt.Type.I64(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.FLOAT32: lambda: stt.Type(
        fp32=stt.Type.FP32(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.FLOAT64: lambda: stt.Type(
        fp64=stt.Type.FP64(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.STRING: lambda: stt.Type(
        string=stt.Type.String(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.BINARY: lambda: stt.Type(
        binary=stt.Type.Binary(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.BOOL: lambda: stt.Type(
        bool=stt.Type.Boolean(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.DATE: lambda: stt.Type(
        date=stt.Type.Date(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.TIME: lambda: stt.Type(
        time=stt.Type.Time(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.TIMESTAMP: lambda: stt.Type(
        timestamp=stt.Type.Timestamp(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.TIMESTAMP_TZ: lambda: stt.Type(
        timestamp_tz=stt.Type.TimestampTZ(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.TIMEDELTA: lambda: stt.Type(
        interval_day=stt.Type.IntervalDay(nullability=stt.Type.NULLABILITY_NULLABLE)
    ),
    DType.DECIMAL: lambda: stt.Type(
        decimal=stt.Type.Decimal(
            scale=0,
            precision=38,
            nullability=stt.Type.NULLABILITY_NULLABLE,
        )
    ),
}


def _schema_to_named_struct(schema: Schema) -> stt.NamedStruct:
    """Convert an IR Schema to a Substrait NamedStruct protobuf."""
    return stt.NamedStruct(
        names=list(schema.columns.keys()),
        struct=stt.Type.Struct(
            types=[_DTYPE_TO_SUBSTRAIT[dt]() for dt in schema.columns.values()],
            nullability=stt.Type.NULLABILITY_REQUIRED,
        ),
    )


# ---------------------------------------------------------------------------
# 2. Substrait function registry
# ---------------------------------------------------------------------------

SUBSTRAIT_FUNC_URI = "https://github.com/substrait-io/substrait/blob/main/extensions/"
COMPARISON_URI = SUBSTRAIT_FUNC_URI + "functions_comparison.yaml"
ARITHMETIC_URI = SUBSTRAIT_FUNC_URI + "functions_arithmetic.yaml"
BOOLEAN_URI = SUBSTRAIT_FUNC_URI + "functions_boolean.yaml"
AGGREGATE_URI = SUBSTRAIT_FUNC_URI + "functions_aggregate_generic.yaml"

FUNC_REGISTRY: dict[str, tuple[str, str]] = {
    # Comparison
    "gt": ("gt:any_any", COMPARISON_URI),
    "gte": ("gte:any_any", COMPARISON_URI),
    "lt": ("lt:any_any", COMPARISON_URI),
    "lte": ("lte:any_any", COMPARISON_URI),
    "eq": ("equal:any_any", COMPARISON_URI),
    "ne": ("not_equal:any_any", COMPARISON_URI),
    # Null handling
    "is_null": ("is_null:any", COMPARISON_URI),
    "coalesce": ("coalesce:any", COMPARISON_URI),
    # Arithmetic (scalar)
    "add": ("add:i64_i64", ARITHMETIC_URI),
    "sub": ("subtract:i64_i64", ARITHMETIC_URI),
    "mul": ("multiply:i64_i64", ARITHMETIC_URI),
    "div": ("divide:i64_i64", ARITHMETIC_URI),
    "abs": ("abs:fp64", ARITHMETIC_URI),
    "negate": ("negate:i64", ARITHMETIC_URI),
    # Boolean
    "and": ("and:bool", BOOLEAN_URI),
    "or": ("or:bool", BOOLEAN_URI),
    "not": ("not:bool", BOOLEAN_URI),
    # Aggregation — sum/avg/min/max live under ARITHMETIC_URI per Acero/Substrait spec
    "sum": ("sum:i64", ARITHMETIC_URI),
    "avg": ("avg:i64", ARITHMETIC_URI),
    "mean": ("avg:i64", ARITHMETIC_URI),
    "min": ("min:i64", ARITHMETIC_URI),
    "max": ("max:i64", ARITHMETIC_URI),
    "count": ("count:any", AGGREGATE_URI),
}


# ---------------------------------------------------------------------------
# 2. SubstraitCompiler — IR -> Substrait protobuf
# ---------------------------------------------------------------------------


class SubstraitCompiler:
    """Compiles an IR tree into a Substrait Plan protobuf."""

    def __init__(self):
        self._next_func_id = 1
        self._next_uri_id = 1
        self._func_anchors: dict[str, int] = {}
        self._uri_anchors: dict[str, int] = {}
        self._extension_uris: list = []
        self._extensions: list = []

    def _get_uri_anchor(self, uri: str) -> int:
        if uri not in self._uri_anchors:
            anchor = self._next_uri_id
            self._next_uri_id += 1
            self._uri_anchors[uri] = anchor
            self._extension_uris.append(
                SimpleExtensionURI(extension_uri_anchor=anchor, uri=uri)
            )
        return self._uri_anchors[uri]

    def _get_func_anchor(self, func_key: str) -> int:
        if func_key not in self._func_anchors:
            func_name, uri = FUNC_REGISTRY[func_key]
            uri_anchor = self._get_uri_anchor(uri)
            anchor = self._next_func_id
            self._next_func_id += 1
            self._func_anchors[func_key] = anchor
            self._extensions.append(
                SimpleExtensionDeclaration(
                    extension_function=SimpleExtensionDeclaration.ExtensionFunction(
                        extension_uri_reference=uri_anchor,
                        function_anchor=anchor,
                        name=func_name,
                    )
                )
            )
        return self._func_anchors[func_key]

    def compile(self, ir: IRNode) -> stp.Plan:
        rel = self._compile_rel(ir)
        output_schema = ir.output_schema()

        p = re.compile(r"(\d+)\.(\d+)\.(\d+)")
        m = p.match(substrait_version)
        version = stp.Version(
            major_number=int(m.group(1)),
            minor_number=int(m.group(2)),
            patch_number=int(m.group(3)),
        )

        return stp.Plan(
            version=version,
            extension_uris=self._extension_uris,
            extensions=self._extensions,
            relations=[
                stp.PlanRel(
                    root=stalg.RelRoot(
                        input=rel,
                        names=output_schema.column_names(),
                    )
                )
            ],
        )

    def _compile_rel(self, ir: IRNode) -> stalg.Rel:
        if isinstance(ir, ReadTable):
            return self._compile_read(ir)
        elif isinstance(ir, Filter):
            return self._compile_filter(ir)
        elif isinstance(ir, Project):
            return self._compile_project(ir)
        elif isinstance(ir, AddColumn):
            return self._compile_add_column(ir)
        elif isinstance(ir, Sort):
            return self._compile_sort(ir)
        elif isinstance(ir, Limit):
            return self._compile_limit(ir)
        elif isinstance(ir, Aggregate):
            return self._compile_aggregate(ir)
        elif isinstance(ir, Join):
            return self._compile_join(ir)
        elif isinstance(ir, RenameColumns):
            # Rename is a pass-through — Substrait RelRoot handles naming
            return self._compile_rel(ir.input)
        else:
            raise TypeError(f"Unknown IR node: {type(ir)}")

    def _compile_read(self, ir: ReadTable) -> stalg.Rel:
        return stalg.Rel(
            read=stalg.ReadRel(
                common=stalg.RelCommon(direct=stalg.RelCommon.Direct()),
                base_schema=_schema_to_named_struct(ir.schema),
                named_table=stalg.ReadRel.NamedTable(names=[ir.name]),
            )
        )

    def _compile_filter(self, ir: Filter) -> stalg.Rel:
        input_rel = self._compile_rel(ir.input)
        input_schema = ir.input.output_schema()
        condition = self._compile_expr(ir.predicate, input_schema)
        return stalg.Rel(
            filter=stalg.FilterRel(
                common=stalg.RelCommon(direct=stalg.RelCommon.Direct()),
                input=input_rel,
                condition=condition,
            )
        )

    def _compile_project(self, ir: Project) -> stalg.Rel:
        input_rel = self._compile_rel(ir.input)
        input_schema = ir.input.output_schema()
        indices = [input_schema.column_index(c) for c in ir.columns]
        return stalg.Rel(
            project=stalg.ProjectRel(
                common=stalg.RelCommon(
                    emit=stalg.RelCommon.Emit(output_mapping=indices)
                ),
                input=input_rel,
            )
        )

    def _compile_add_column(self, ir: AddColumn) -> stalg.Rel:
        input_rel = self._compile_rel(ir.input)
        input_schema = ir.input.output_schema()
        expr = self._compile_expr(ir.expr, input_schema)
        n_input_cols = len(input_schema.columns)
        new_expr_idx = n_input_cols  # expression is appended after input cols

        col_names = list(input_schema.columns.keys())
        if ir.name in col_names:
            # Replace existing column: swap old index with expression index
            old_idx = col_names.index(ir.name)
            output_mapping = [
                new_expr_idx if i == old_idx else i for i in range(n_input_cols)
            ]
        else:
            # Append new column
            output_mapping = [*range(n_input_cols), new_expr_idx]

        return stalg.Rel(
            project=stalg.ProjectRel(
                common=stalg.RelCommon(
                    emit=stalg.RelCommon.Emit(output_mapping=output_mapping)
                ),
                input=input_rel,
                expressions=[expr],
            )
        )

    def _compile_sort(self, ir: Sort) -> stalg.Rel:
        input_rel = self._compile_rel(ir.input)
        input_schema = ir.input.output_schema()
        sort_fields = []
        for col_name, ascending in ir.keys:
            idx = input_schema.column_index(col_name)
            direction = (
                stalg.SortField.SORT_DIRECTION_ASC_NULLS_LAST
                if ascending
                else stalg.SortField.SORT_DIRECTION_DESC_NULLS_LAST
            )
            sort_fields.append(
                stalg.SortField(
                    expr=stalg.Expression(
                        selection=stalg.Expression.FieldReference(
                            direct_reference=stalg.Expression.ReferenceSegment(
                                struct_field=stalg.Expression.ReferenceSegment.StructField(
                                    field=idx
                                )
                            ),
                            root_reference=stalg.Expression.FieldReference.RootReference(),
                        )
                    ),
                    direction=direction,
                )
            )
        return stalg.Rel(
            sort=stalg.SortRel(
                common=stalg.RelCommon(direct=stalg.RelCommon.Direct()),
                input=input_rel,
                sorts=sort_fields,
            )
        )

    def _compile_limit(self, ir: Limit) -> stalg.Rel:
        input_rel = self._compile_rel(ir.input)
        return stalg.Rel(
            fetch=stalg.FetchRel(
                common=stalg.RelCommon(direct=stalg.RelCommon.Direct()),
                input=input_rel,
                offset=0,
                count=ir.n,
            )
        )

    def _compile_aggregate(self, ir: Aggregate) -> stalg.Rel:
        input_rel = self._compile_rel(ir.input)
        input_schema = ir.input.output_schema()

        grouping_exprs = []
        for key in ir.group_keys:
            idx = input_schema.column_index(key)
            grouping_exprs.append(
                stalg.Expression(
                    selection=stalg.Expression.FieldReference(
                        direct_reference=stalg.Expression.ReferenceSegment(
                            struct_field=stalg.Expression.ReferenceSegment.StructField(
                                field=idx
                            )
                        ),
                        root_reference=stalg.Expression.FieldReference.RootReference(),
                    )
                )
            )

        measures = []
        for out_name, src_col, func in ir.agg_specs:
            func_anchor = self._get_func_anchor(func)
            idx = input_schema.column_index(src_col)

            agg_func = stalg.AggregateFunction(
                function_reference=func_anchor,
                arguments=[
                    stalg.FunctionArgument(
                        value=stalg.Expression(
                            selection=stalg.Expression.FieldReference(
                                direct_reference=stalg.Expression.ReferenceSegment(
                                    struct_field=stalg.Expression.ReferenceSegment.StructField(
                                        field=idx
                                    )
                                ),
                                root_reference=stalg.Expression.FieldReference.RootReference(),
                            )
                        )
                    )
                ],
                phase=stalg.AggregationPhase.AGGREGATION_PHASE_INITIAL_TO_RESULT,
                output_type=_DTYPE_TO_SUBSTRAIT.get(
                    DType.INT64
                    if func == "count"
                    else input_schema.columns.get(src_col, DType.FLOAT64),
                    _DTYPE_TO_SUBSTRAIT[DType.FLOAT64],
                )(),
            )
            measures.append(stalg.AggregateRel.Measure(measure=agg_func))

        return stalg.Rel(
            aggregate=stalg.AggregateRel(
                common=stalg.RelCommon(direct=stalg.RelCommon.Direct()),
                input=input_rel,
                groupings=[
                    stalg.AggregateRel.Grouping(
                        grouping_expressions=grouping_exprs,
                    )
                ],
                measures=measures,
            )
        )

    def _compile_join(self, ir: Join) -> stalg.Rel:
        left_rel = self._compile_rel(ir.left)
        right_rel = self._compile_rel(ir.right)
        left_schema = ir.left.output_schema()
        right_schema = ir.right.output_schema()

        left_idx = left_schema.column_index(ir.left_on)
        right_idx = right_schema.column_index(ir.right_on)
        n_left = len(left_schema.columns)

        eq_anchor = self._get_func_anchor("eq")
        join_expr = stalg.Expression(
            scalar_function=stalg.Expression.ScalarFunction(
                function_reference=eq_anchor,
                arguments=[
                    stalg.FunctionArgument(
                        value=stalg.Expression(
                            selection=stalg.Expression.FieldReference(
                                direct_reference=stalg.Expression.ReferenceSegment(
                                    struct_field=stalg.Expression.ReferenceSegment.StructField(
                                        field=left_idx
                                    )
                                ),
                                root_reference=stalg.Expression.FieldReference.RootReference(),
                            )
                        )
                    ),
                    stalg.FunctionArgument(
                        value=stalg.Expression(
                            selection=stalg.Expression.FieldReference(
                                direct_reference=stalg.Expression.ReferenceSegment(
                                    struct_field=stalg.Expression.ReferenceSegment.StructField(
                                        field=n_left + right_idx
                                    )
                                ),
                                root_reference=stalg.Expression.FieldReference.RootReference(),
                            )
                        )
                    ),
                ],
                output_type=stt.Type(
                    bool=stt.Type.Boolean(nullability=stt.Type.NULLABILITY_NULLABLE)
                ),
            )
        )

        join_type_map = {
            "inner": stalg.JoinRel.JOIN_TYPE_INNER,
            "left": stalg.JoinRel.JOIN_TYPE_LEFT,
            "right": stalg.JoinRel.JOIN_TYPE_RIGHT,
            "outer": stalg.JoinRel.JOIN_TYPE_OUTER,
        }

        # Build emit mapping that drops the duplicate right join-key column.
        # JoinRel produces [left_col_0..left_col_n, right_col_0..right_col_m].
        # We keep all left columns, then right columns except the join key.
        n_right = len(right_schema.columns)
        output_mapping = [
            *range(n_left),
            *(n_left + i for i in range(n_right) if i != right_idx),
        ]

        return stalg.Rel(
            join=stalg.JoinRel(
                common=stalg.RelCommon(
                    emit=stalg.RelCommon.Emit(output_mapping=output_mapping)
                ),
                left=left_rel,
                right=right_rel,
                expression=join_expr,
                type=join_type_map.get(ir.how, stalg.JoinRel.JOIN_TYPE_INNER),
            )
        )

    def _compile_expr(self, expr: Expr, schema: Schema) -> stalg.Expression:
        if isinstance(expr, ColRef):
            idx = schema.column_index(expr.name)
            return stalg.Expression(
                selection=stalg.Expression.FieldReference(
                    direct_reference=stalg.Expression.ReferenceSegment(
                        struct_field=stalg.Expression.ReferenceSegment.StructField(
                            field=idx
                        )
                    ),
                    root_reference=stalg.Expression.FieldReference.RootReference(),
                )
            )
        elif isinstance(expr, Literal):
            return self._compile_literal(expr)
        elif isinstance(expr, BinOp):
            return self._compile_binop(expr, schema)
        elif isinstance(expr, UnaryOp):
            return self._compile_unaryop(expr, schema)
        else:
            raise TypeError(f"Unknown expression type: {type(expr)}")

    def _compile_literal(self, lit: Literal) -> stalg.Expression:
        match lit.dtype:
            case DType.INT8:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(i8=int(lit.value))
                )
            case DType.INT16:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(i16=int(lit.value))
                )
            case DType.INT32:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(i32=int(lit.value))
                )
            case DType.INT64 | DType.UINT8 | DType.UINT16 | DType.UINT32 | DType.UINT64:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(i64=int(lit.value))
                )
            case DType.FLOAT32:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(fp32=float(lit.value))
                )
            case DType.FLOAT64 | DType.DECIMAL:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(fp64=float(lit.value))
                )
            case DType.STRING:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(string=str(lit.value))
                )
            case DType.BINARY:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(binary=bytes(lit.value))
                )
            case DType.BOOL:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(boolean=bool(lit.value))
                )
            case _:
                return stalg.Expression(
                    literal=stalg.Expression.Literal(string=str(lit.value))
                )

    def _compile_binop(self, expr: BinOp, schema: Schema) -> stalg.Expression:
        func_anchor = self._get_func_anchor(expr.op)
        left = self._compile_expr(expr.left, schema)
        right = self._compile_expr(expr.right, schema)

        if expr.op in ("gt", "gte", "lt", "lte", "eq", "ne", "and", "or"):
            output_type = stt.Type(
                bool=stt.Type.Boolean(nullability=stt.Type.NULLABILITY_NULLABLE)
            )
        else:
            output_type = stt.Type(
                fp64=stt.Type.FP64(nullability=stt.Type.NULLABILITY_NULLABLE)
            )

        return stalg.Expression(
            scalar_function=stalg.Expression.ScalarFunction(
                function_reference=func_anchor,
                arguments=[
                    stalg.FunctionArgument(value=left),
                    stalg.FunctionArgument(value=right),
                ],
                output_type=output_type,
            )
        )

    def _compile_unaryop(self, expr: UnaryOp, schema: Schema) -> stalg.Expression:
        func_anchor = self._get_func_anchor(expr.op)
        operand = self._compile_expr(expr.operand, schema)
        return stalg.Expression(
            scalar_function=stalg.Expression.ScalarFunction(
                function_reference=func_anchor,
                arguments=[stalg.FunctionArgument(value=operand)],
                output_type=stt.Type(
                    bool=stt.Type.Boolean(nullability=stt.Type.NULLABILITY_NULLABLE)
                ),
            )
        )


# ---------------------------------------------------------------------------
# 3. Execution backends
# ---------------------------------------------------------------------------


class Backend:
    """Abstract execution backend."""

    def execute(self, ir_node: IRNode, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class PandasBackend(Backend):
    """
    Execute IR directly on pandas. No Substrait involved.
    This is the fallback — it interprets the IR tree using pandas operations.
    """

    @property
    def name(self) -> str:
        return "pandas"

    def execute(self, ir_node: IRNode, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return self._exec(ir_node, tables)

    def _exec(self, node: IRNode, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(node, ReadTable):
            if node.name not in tables:
                raise KeyError(
                    f"Table '{node.name}' not registered. "
                    f"Available: {list(tables.keys())}"
                )
            return tables[node.name]

        elif isinstance(node, Filter):
            df = self._exec(node.input, tables)
            mask = self._eval_expr(node.predicate, df)
            return df[mask].reset_index(drop=True)

        elif isinstance(node, Project):
            df = self._exec(node.input, tables)
            return df[node.columns]

        elif isinstance(node, AddColumn):
            df = self._exec(node.input, tables)
            df = df.copy()
            df[node.name] = self._eval_expr(node.expr, df)
            return df

        elif isinstance(node, Sort):
            df = self._exec(node.input, tables)
            cols = [k for k, _ in node.keys]
            ascending = [a for _, a in node.keys]
            return df.sort_values(cols, ascending=ascending).reset_index(drop=True)

        elif isinstance(node, Limit):
            df = self._exec(node.input, tables)
            return df.head(node.n)

        elif isinstance(node, Aggregate):
            df = self._exec(node.input, tables)
            if not node.group_keys:
                result = {}
                for out_name, src_col, func in node.agg_specs:
                    result[out_name] = [self._agg_fn(func)(df[src_col])]
                return pd.DataFrame(result)
            agg_dict = {}
            rename_map = {}
            for out_name, src_col, func in node.agg_specs:
                pandas_func = func if func != "avg" else "mean"
                agg_dict[src_col] = pandas_func
                rename_map[src_col] = out_name
            result = df.groupby(node.group_keys, as_index=False).agg(agg_dict)
            non_key_cols = [c for c in result.columns if c not in node.group_keys]
            for col in non_key_cols:
                if col in rename_map and rename_map[col] != col:
                    result = result.rename(columns={col: rename_map[col]})
            return result

        elif isinstance(node, Join):
            left = self._exec(node.left, tables)
            right = self._exec(node.right, tables)
            return left.merge(
                right,
                left_on=node.left_on,
                right_on=node.right_on,
                how=node.how,
            )

        elif isinstance(node, RenameColumns):
            df = self._exec(node.input, tables)
            return df.rename(columns=node.mapping)

        else:
            raise TypeError(f"PandasBackend: unknown IR node {type(node).__name__}")

    def _eval_expr(self, expr: Expr, df: pd.DataFrame):
        if isinstance(expr, ColRef):
            return df[expr.name]
        elif isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, BinOp):
            left = self._eval_expr(expr.left, df)
            right = self._eval_expr(expr.right, df)
            ops = {
                "gt": lambda a, b: a > b,
                "gte": lambda a, b: a >= b,
                "lt": lambda a, b: a < b,
                "lte": lambda a, b: a <= b,
                "eq": lambda a, b: a == b,
                "ne": lambda a, b: a != b,
                "add": lambda a, b: a + b,
                "sub": lambda a, b: a - b,
                "mul": lambda a, b: a * b,
                "div": lambda a, b: a / b,
                "and": lambda a, b: a & b,
                "or": lambda a, b: a | b,
                "coalesce": lambda a, b: (
                    a.fillna(b) if hasattr(a, "fillna") else (b if a is None else a)
                ),
            }
            return ops[expr.op](left, right)
        elif isinstance(expr, UnaryOp):
            val = self._eval_expr(expr.operand, df)
            if expr.op == "not":
                return ~val
            elif expr.op == "is_null":
                return val.isna() if hasattr(val, "isna") else pd.isna(val)
            elif expr.op == "abs":
                return val.abs() if hasattr(val, "abs") else abs(val)
            elif expr.op == "negate":
                return -val
            else:
                raise TypeError(f"Unknown unary op: {expr.op}")
        else:
            raise TypeError(f"Unknown expr: {type(expr)}")

    def _agg_fn(self, name):
        return {
            "sum": pd.Series.sum,
            "avg": pd.Series.mean,
            "mean": pd.Series.mean,
            "min": pd.Series.min,
            "max": pd.Series.max,
            "count": pd.Series.count,
        }[name]


class AceroBackend(Backend):
    """
    Execute IR via PyArrow's Acero engine.

    Compiles the IR tree to a Substrait plan and executes it using
    pyarrow.substrait.run_query(). Requires pyarrow to be installed.
    """

    @property
    def name(self) -> str:
        return "acero"

    def execute(self, ir_node: IRNode, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        from pandas.compat._optional import import_optional_dependency

        pa = import_optional_dependency("pyarrow")
        pas = import_optional_dependency("pyarrow.substrait")

        compiler = SubstraitCompiler()
        plan_bytes = compiler.compile(ir_node).SerializeToString()

        pa_tables = {name: pa.Table.from_pandas(df) for name, df in tables.items()}

        def table_provider(names, _schema):
            return pa_tables[names[0]]

        reader = pas.run_query(plan_bytes, table_provider=table_provider)
        return reader.read_all().to_pandas()


def default_backend() -> Backend:
    """
    Return the default execution backend.

    Returns AceroBackend if pyarrow (with Substrait support) is available,
    otherwise PandasBackend.
    """
    from pandas.compat._optional import import_optional_dependency

    pa = import_optional_dependency("pyarrow", errors="ignore")
    if pa is not None:
        pas = import_optional_dependency("pyarrow.substrait", errors="ignore")
        if pas is not None:
            return AceroBackend()
    return PandasBackend()


# ---------------------------------------------------------------------------
# 4. Execution plan — chain of compiled + eager segments
# ---------------------------------------------------------------------------


@dataclass
class CompiledSegment:
    """A segment of the computation that was captured as IR."""

    ir_node: IRNode
    input_tables: list[str]
    output_table: str
    description: str = ""

    def explain(self, indent=0) -> str:
        prefix = "  " * indent
        ir_str = explain_ir(self.ir_node, indent + 1)
        return f"{prefix}CompiledSegment -> {self.output_table}\n{ir_str}"


@dataclass
class EagerSegment:
    """A segment that runs in eager Python (the graph break)."""

    operation: str
    fn: Callable
    input_tables: list[str]
    output_names: list[str]

    def explain(self, indent=0) -> str:
        prefix = "  " * indent
        return f"{prefix}EagerSegment({self.operation}) -> {self.output_names}"


Segment = CompiledSegment | EagerSegment


@dataclass
class ExecutionPlan:
    """
    A chain of CompiledSegments and EagerSegments that together
    reproduce the user's computation.
    """

    segments: list[Segment] = field(default_factory=list)
    final_output: str | None = None

    def explain(self) -> str:
        lines = ["ExecutionPlan:"]
        for i, seg in enumerate(self.segments):
            lines.append(f"  [{i}] {seg.explain(indent=2)}")
        if self.final_output:
            lines.append(f"  Output: {self.final_output}")
        return "\n".join(lines)

    def execute(
        self,
        initial_tables: dict[str, pd.DataFrame],
        backend: Backend,
    ) -> pd.DataFrame:
        tables = dict(initial_tables)
        scalars: dict[str, Any] = {}

        for seg in self.segments:
            if isinstance(seg, CompiledSegment):
                result = backend.execute(seg.ir_node, tables)
                tables[seg.output_table] = result
            elif isinstance(seg, EagerSegment):
                outputs = seg.fn(tables, scalars)
                for name in seg.output_names:
                    if name in outputs:
                        if isinstance(outputs[name], pd.DataFrame):
                            tables[name] = outputs[name]
                        else:
                            scalars[name] = outputs[name]

        if self.final_output and self.final_output in tables:
            return tables[self.final_output]
        raise RuntimeError(f"Final output '{self.final_output}' not found in tables")


# ---------------------------------------------------------------------------
# 5. Guards — determine if a cached plan can be reused
# ---------------------------------------------------------------------------


@dataclass
class SchemaGuard:
    """
    A guard that checks if input DataFrames match expected schemas
    and non-DataFrame arguments match their traced values.
    """

    expected: dict[str, Schema]
    scalar_args: tuple[Any, ...] = ()
    scalar_kwargs: dict[str, Any] | None = None

    def check(
        self,
        df_args: dict[str, pd.DataFrame],
        scalar_args: tuple[Any, ...] = (),
        scalar_kwargs: dict[str, Any] | None = None,
    ) -> bool:
        if scalar_kwargs is None:
            scalar_kwargs = {}
        # Check non-DataFrame arguments first (cheap)
        if scalar_args != self.scalar_args:
            return False
        if (scalar_kwargs or {}) != (self.scalar_kwargs or {}):
            return False
        # Check DataFrame schemas
        for name, expected_schema in self.expected.items():
            if name not in df_args:
                return False
            actual = infer_schema(df_args[name])
            if actual.column_names() != expected_schema.column_names():
                return False
            if list(actual.columns.values()) != list(expected_schema.columns.values()):
                return False
        return True

    def __repr__(self):
        parts = []
        for name, schema in self.expected.items():
            cols = ", ".join(f"{c}:{t.name}" for c, t in schema.columns.items())
            parts.append(f"{name}[{cols}]")
        extra = ""
        if self.scalar_args:
            extra += f", scalar_args={self.scalar_args!r}"
        if self.scalar_kwargs:
            extra += f", scalar_kwargs={self.scalar_kwargs!r}"
        return f"SchemaGuard({', '.join(parts)}{extra})"


def infer_schema(df: pd.DataFrame) -> Schema:
    """Infer our Schema from a real pandas DataFrame."""
    columns: dict[str, DType] = {}
    for col in df.columns:
        columns[col] = pandas_dtype_to_ir(str(df[col].dtype))
    return Schema(columns)
