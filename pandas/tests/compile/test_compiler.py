"""Tests for pandas.compile.compiler — SubstraitCompiler, backends, infer_schema."""

from __future__ import annotations

import numpy as np
import pytest
import substrait.algebra_pb2 as stalg
import substrait.type_pb2 as stt

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.compile.compiler import (
    _DTYPE_TO_SUBSTRAIT,
    AceroBackend,
    Backend,
    DataFusionBackend,
    PandasBackend,
    SchemaGuard,
    SubstraitCompiler,
    _schema_to_named_struct,
    default_backend,
    infer_schema,
)
from pandas.compile.ir import (
    AddColumn,
    Aggregate,
    BinOp,
    CastExpr,
    ColRef,
    Distinct,
    DType,
    Filter,
    FunctionCall,
    IfThenExpr,
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
)

# ---------------------------------------------------------------------------
# _DTYPE_TO_SUBSTRAIT / _schema_to_named_struct
# ---------------------------------------------------------------------------


class TestDTypeToSubstrait:
    def test_every_dtype_has_substrait_mapping(self):
        for dt in DType:
            assert dt in _DTYPE_TO_SUBSTRAIT, f"Missing Substrait mapping for {dt}"

    @pytest.mark.parametrize(
        "dt, field_name",
        [
            (DType.INT8, "i8"),
            (DType.INT16, "i16"),
            (DType.INT32, "i32"),
            (DType.INT64, "i64"),
            (DType.FLOAT32, "fp32"),
            (DType.FLOAT64, "fp64"),
            (DType.STRING, "string"),
            (DType.BINARY, "binary"),
            (DType.BOOL, "bool"),
            (DType.DATE, "date"),
            (DType.TIME, "time"),
            (DType.TIMESTAMP, "timestamp"),
            (DType.TIMESTAMP_TZ, "timestamp_tz"),
            (DType.TIMEDELTA, "interval_day"),
            (DType.DECIMAL, "decimal"),
        ],
    )
    def test_substrait_type_field(self, dt, field_name):
        result = _DTYPE_TO_SUBSTRAIT[dt]()
        assert isinstance(result, stt.Type)
        assert result.HasField(field_name)

    def test_unsigned_widen_to_signed(self):
        """Unsigned types widen to the next signed Substrait type."""
        assert _DTYPE_TO_SUBSTRAIT[DType.UINT8]().HasField("i16")
        assert _DTYPE_TO_SUBSTRAIT[DType.UINT16]().HasField("i32")
        assert _DTYPE_TO_SUBSTRAIT[DType.UINT32]().HasField("i64")
        assert _DTYPE_TO_SUBSTRAIT[DType.UINT64]().HasField("i64")

    def test_nullable(self):
        for dt in DType:
            stype = _DTYPE_TO_SUBSTRAIT[dt]()
            field_name = stype.WhichOneof("kind")
            inner = getattr(stype, field_name)
            assert inner.nullability == stt.Type.NULLABILITY_NULLABLE


class TestSchemaToNamedStruct:
    def test_basic(self):
        schema = Schema({"id": DType.INT64, "name": DType.STRING})
        ns = _schema_to_named_struct(schema)
        assert isinstance(ns, stt.NamedStruct)
        assert list(ns.names) == ["id", "name"]
        assert len(ns.struct.types) == 2


# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------


class TestInferSchema:
    def test_basic_dtypes(self):
        df = DataFrame(
            {
                "i": np.array([1, 2], dtype="int64"),
                "f": np.array([1.0, 2.0], dtype="float64"),
                "s": ["a", "b"],
                "b": [True, False],
            }
        )
        schema = infer_schema(df)
        assert schema.columns["i"] is DType.INT64
        assert schema.columns["f"] is DType.FLOAT64
        assert schema.columns["s"] is DType.STRING
        assert schema.columns["b"] is DType.BOOL

    def test_int_variants(self):
        df = DataFrame(
            {
                "i8": np.array([1], dtype="int8"),
                "i16": np.array([1], dtype="int16"),
                "i32": np.array([1], dtype="int32"),
                "i64": np.array([1], dtype="int64"),
            }
        )
        schema = infer_schema(df)
        assert schema.columns["i8"] is DType.INT8
        assert schema.columns["i16"] is DType.INT16
        assert schema.columns["i32"] is DType.INT32
        assert schema.columns["i64"] is DType.INT64

    def test_uint_variants(self):
        df = DataFrame(
            {
                "u8": np.array([1], dtype="uint8"),
                "u16": np.array([1], dtype="uint16"),
                "u32": np.array([1], dtype="uint32"),
                "u64": np.array([1], dtype="uint64"),
            }
        )
        schema = infer_schema(df)
        assert schema.columns["u8"] is DType.UINT8
        assert schema.columns["u16"] is DType.UINT16
        assert schema.columns["u32"] is DType.UINT32
        assert schema.columns["u64"] is DType.UINT64

    def test_float_variants(self):
        df = DataFrame(
            {
                "f32": np.array([1.0], dtype="float32"),
                "f64": np.array([1.0], dtype="float64"),
            }
        )
        schema = infer_schema(df)
        assert schema.columns["f32"] is DType.FLOAT32
        assert schema.columns["f64"] is DType.FLOAT64

    def test_datetime(self):
        df = DataFrame({"ts": pd.to_datetime(["2024-01-01"])})
        schema = infer_schema(df)
        assert schema.columns["ts"] is DType.TIMESTAMP

    def test_datetime_with_tz(self):
        df = DataFrame({"ts": pd.to_datetime(["2024-01-01"]).tz_localize("UTC")})
        schema = infer_schema(df)
        assert schema.columns["ts"] is DType.TIMESTAMP_TZ

    def test_timedelta(self):
        df = DataFrame({"td": pd.to_timedelta(["1 days"])})
        schema = infer_schema(df)
        assert schema.columns["td"] is DType.TIMEDELTA

    def test_nullable_integer(self):
        df = DataFrame({"x": pd.array([1, 2, None], dtype="Int32")})
        schema = infer_schema(df)
        assert schema.columns["x"] is DType.INT32

    def test_nullable_boolean(self):
        df = DataFrame({"x": pd.array([True, False, None], dtype="boolean")})
        schema = infer_schema(df)
        assert schema.columns["x"] is DType.BOOL

    def test_string_dtype(self):
        df = DataFrame({"x": pd.array(["a", "b"], dtype="string")})
        schema = infer_schema(df)
        assert schema.columns["x"] is DType.STRING


# ---------------------------------------------------------------------------
# SubstraitCompiler
# ---------------------------------------------------------------------------


class TestSubstraitCompiler:
    def _make_base(self):
        return ReadTable(
            "t",
            Schema({"id": DType.INT64, "name": DType.STRING, "price": DType.FLOAT64}),
        )

    def test_compile_read(self):
        compiler = SubstraitCompiler()
        plan = compiler.compile(self._make_base())
        assert len(plan.relations) == 1
        root = plan.relations[0].root
        assert list(root.names) == ["id", "name", "price"]

    def test_compile_filter(self):
        node = Filter(self._make_base(), BinOp("gt", ColRef("price"), Literal(100)))
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("filter")

    def test_compile_project(self):
        node = Project(self._make_base(), ["id", "price"])
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("project")

    def test_compile_add_column(self):
        expr = BinOp("mul", ColRef("price"), Literal(2))
        node = AddColumn(self._make_base(), "double", expr, DType.FLOAT64)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("project")

    def test_compile_sort(self):
        node = Sort(self._make_base(), [("price", False)])
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("sort")

    def test_compile_limit(self):
        node = Limit(self._make_base(), 10)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("fetch")

    def test_compile_limit_with_offset(self):
        node = Limit(self._make_base(), 5, offset=3)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("fetch")
        assert rel.fetch.offset == 3
        assert rel.fetch.count == 5

    def test_compile_aggregate(self):
        node = Aggregate(self._make_base(), ["name"], [("total", "price", "sum")])
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("aggregate")

    def test_compile_join(self):
        left = ReadTable("l", Schema({"id": DType.INT64, "x": DType.FLOAT64}))
        right = ReadTable("r", Schema({"id": DType.INT64, "y": DType.STRING}))
        node = Join(left, right, "id", "id")
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("join")

    def test_compile_composite_join(self):
        left = ReadTable(
            "l",
            Schema({"a": DType.INT64, "b": DType.STRING, "x": DType.FLOAT64}),
        )
        right = ReadTable(
            "r",
            Schema({"a": DType.INT64, "b": DType.STRING, "y": DType.FLOAT64}),
        )
        node = Join(left, right, ["a", "b"], ["a", "b"])
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("join")
        # The join expression should be an AND of two equality checks.
        expr = rel.join.expression
        assert expr.HasField("scalar_function")
        # Top-level should be the "and" function wrapping two eq calls.
        assert len(expr.scalar_function.arguments) == 2

    def test_compile_union(self):
        a = ReadTable("a", Schema({"x": DType.INT64, "y": DType.STRING}))
        b = ReadTable("b", Schema({"x": DType.INT64, "y": DType.STRING}))
        node = Union([a, b])
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("set")
        assert rel.set.op == stalg.SetRel.SET_OP_UNION_ALL
        assert len(rel.set.inputs) == 2

    def test_compile_distinct(self):
        base = self._make_base()
        node = Distinct(base, ["id", "name", "price"])
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("aggregate")
        assert len(rel.aggregate.groupings) == 1
        assert len(rel.aggregate.groupings[0].grouping_expressions) == 3
        assert len(rel.aggregate.measures) == 0

    def test_compile_rename_is_passthrough(self):
        node = RenameColumns(self._make_base(), {"id": "row_id"})
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        # RenameColumns compiles as the inner node (ReadTable)
        assert rel.HasField("read")

    def test_compile_literal_types(self):
        compiler = SubstraitCompiler()

        for lit, field in [
            (Literal(1, DType.INT8), "i8"),
            (Literal(2, DType.INT16), "i16"),
            (Literal(3, DType.INT32), "i32"),
            (Literal(4, DType.INT64), "i64"),
            (Literal(5, DType.UINT8), "i64"),
            (Literal(1.0, DType.FLOAT32), "fp32"),
            (Literal(2.0, DType.FLOAT64), "fp64"),
            (Literal("hi", DType.STRING), "string"),
            (Literal(b"raw", DType.BINARY), "binary"),
            (Literal(True, DType.BOOL), "boolean"),
        ]:
            result = compiler._compile_literal(lit)
            assert result.HasField("literal")
            assert result.literal.HasField(field), f"Expected {field} for {lit.dtype}"

    def test_compile_window_rolling(self):
        base = self._make_base()
        spec = WindowSpec(kind="rows", lower_offset=2, upper_offset=0)
        node = Window(base, [("price", "price", "sum")], spec)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("window")
        assert len(rel.window.window_functions) == 1

    def test_compile_window_expanding(self):
        base = self._make_base()
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(base, [("price", "price", "avg")], spec)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("window")
        wf = rel.window.window_functions[0]
        assert wf.lower_bound.HasField("unbounded")

    def test_compile_if_then(self):
        expr = IfThenExpr(
            BinOp("gt", ColRef("price"), Literal(200.0)),
            ColRef("price"),
            Literal(0.0),
        )
        node = AddColumn(self._make_base(), "clamped", expr, DType.FLOAT64)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("project")
        proj_expr = rel.project.expressions[0]
        assert proj_expr.HasField("if_then")
        assert len(proj_expr.if_then.ifs) == 1

    def test_compile_cast(self):
        expr = CastExpr(ColRef("price"), DType.INT64)
        node = AddColumn(self._make_base(), "price_int", expr, DType.INT64)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("project")
        # The expression in the project should be a cast
        proj_expr = rel.project.expressions[0]
        assert proj_expr.HasField("cast")
        assert proj_expr.cast.type.HasField("i64")

    def test_compile_singular_or_list(self):
        expr = SingularOrList(ColRef("name"), [Literal("Alice"), Literal("Bob")])
        node = Filter(self._make_base(), expr)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("filter")
        # The condition should be a singular_or_list expression
        condition = rel.filter.condition
        assert condition.HasField("singular_or_list")
        assert len(condition.singular_or_list.options) == 2

    def test_compile_unaryop_abs_output_type(self):
        """abs should produce fp64, not bool."""
        expr = UnaryOp("abs", BinOp("sub", ColRef("price"), Literal(200.0)))
        node = AddColumn(self._make_base(), "dist", expr, DType.FLOAT64)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        proj_expr = rel.project.expressions[0]
        assert proj_expr.HasField("scalar_function")
        assert proj_expr.scalar_function.output_type.HasField("fp64")

    def test_compile_unaryop_negate_output_type(self):
        """negate should produce fp64, not bool."""
        expr = UnaryOp("negate", ColRef("price"))
        node = AddColumn(self._make_base(), "neg", expr, DType.FLOAT64)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        proj_expr = rel.project.expressions[0]
        assert proj_expr.HasField("scalar_function")
        assert proj_expr.scalar_function.output_type.HasField("fp64")

    def test_compile_unaryop_not_output_type(self):
        """not should produce bool."""
        expr = UnaryOp("not", BinOp("gt", ColRef("price"), Literal(100.0)))
        node = Filter(self._make_base(), expr)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        condition = rel.filter.condition
        assert condition.HasField("scalar_function")
        assert condition.scalar_function.output_type.HasField("bool")

    def test_compile_window_has_invocation(self):
        """WindowRelFunction should have AGGREGATION_INVOCATION_ALL."""
        base = self._make_base()
        spec = WindowSpec(kind="rows", lower_offset=2, upper_offset=0)
        node = Window(base, [("price", "price", "sum")], spec)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        rel = plan.relations[0].root.input
        assert rel.HasField("window")
        wf = rel.window.window_functions[0]
        assert wf.invocation == stalg.AggregateFunction.AGGREGATION_INVOCATION_ALL

    def test_unknown_ir_node_raises(self):
        class BadNode:
            pass

        compiler = SubstraitCompiler()
        with pytest.raises(TypeError, match="Unknown IR node"):
            compiler._compile_rel(BadNode())


# ---------------------------------------------------------------------------
# String function Substrait names
# ---------------------------------------------------------------------------


class TestStringSubstraitNames:
    """Verify string functions compile to proper Substrait function names."""

    def _make_base(self):
        return ReadTable("t", Schema({"name": DType.STRING, "val": DType.INT64}))

    @pytest.mark.parametrize(
        "ir_name, substrait_name",
        [
            ("str_upper", "upper:str"),
            ("str_lower", "lower:str"),
            ("str_strip", "trim:str"),
            ("str_lstrip", "ltrim:str"),
            ("str_rstrip", "rtrim:str"),
            ("str_len", "char_length:str"),
            ("str_contains", "contains:str_str"),
            ("str_startswith", "starts_with:str_str"),
            ("str_endswith", "ends_with:str_str"),
            ("str_replace", "replace:str_str_str"),
            ("str_slice", "substring:str_i32_i32"),
        ],
    )
    def test_string_func_substrait_name(self, ir_name, substrait_name):
        from pandas.compile.compiler import (
            FUNC_REGISTRY,
            STRING_URI,
        )

        assert ir_name in FUNC_REGISTRY
        name, uri = FUNC_REGISTRY[ir_name]
        assert name == substrait_name
        assert uri == STRING_URI

    def test_string_func_compiles_to_substrait(self):
        """A string function call compiles to a Substrait scalar function."""
        expr = FunctionCall("str_upper", [ColRef("name")], return_dtype=DType.STRING)
        node = AddColumn(self._make_base(), "upper_name", expr, DType.STRING)
        compiler = SubstraitCompiler()
        plan = compiler.compile(node)
        # Should have STRING_URI in extension URIs
        from pandas.compile.compiler import STRING_URI

        uri_strs = [u.uri for u in plan.extension_uris]
        assert STRING_URI in uri_strs


# ---------------------------------------------------------------------------
# PandasBackend
# ---------------------------------------------------------------------------


class TestPandasBackend:
    @pytest.fixture
    def backend(self):
        return PandasBackend()

    @pytest.fixture
    def sample_df(self):
        return DataFrame(
            {
                "id": [1, 2, 3, 4],
                "region": ["E", "W", "E", "W"],
                "price": [100, 250, 150, 300],
            }
        )

    @pytest.fixture
    def tables(self, sample_df):
        return {"t": sample_df}

    @pytest.fixture
    def base_ir(self, sample_df):
        return ReadTable("t", infer_schema(sample_df))

    def test_read(self, backend, tables, base_ir):
        result = backend.execute(base_ir, tables)
        assert len(result) == 4

    def test_filter(self, backend, tables, base_ir):
        node = Filter(base_ir, BinOp("gt", ColRef("price"), Literal(100)))
        result = backend.execute(node, tables)
        assert len(result) == 3
        assert all(result["price"] > 100)

    def test_project(self, backend, tables, base_ir):
        node = Project(base_ir, ["id", "price"])
        result = backend.execute(node, tables)
        assert list(result.columns) == ["id", "price"]

    def test_add_column(self, backend, tables, base_ir):
        expr = BinOp("mul", ColRef("price"), Literal(2))
        node = AddColumn(base_ir, "double", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert "double" in result.columns
        assert list(result["double"]) == [200, 500, 300, 600]

    def test_sort(self, backend, tables, base_ir):
        node = Sort(base_ir, [("price", False)])
        result = backend.execute(node, tables)
        assert list(result["price"]) == [300, 250, 150, 100]

    def test_limit(self, backend, tables, base_ir):
        node = Limit(base_ir, 2)
        result = backend.execute(node, tables)
        assert len(result) == 2

    def test_limit_with_offset(self, backend, tables, base_ir):
        node = Limit(base_ir, 2, offset=1)
        result = backend.execute(node, tables)
        assert len(result) == 2
        # Should skip first row and take next 2
        assert list(result["id"]) == [2, 3]

    def test_aggregate(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir, ["region"], [("total", "price", "sum"), ("cnt", "id", "count")]
        )
        result = backend.execute(node, tables)
        assert set(result.columns) == {"region", "total", "cnt"}
        assert len(result) == 2

    def test_join(self, backend, sample_df):
        regions = DataFrame({"region": ["E", "W"], "mgr": ["Alice", "Bob"]})
        tables = {"l": sample_df, "r": regions}
        left_ir = ReadTable("l", infer_schema(sample_df))
        right_ir = ReadTable("r", infer_schema(regions))
        node = Join(left_ir, right_ir, "region", "region")
        result = backend.execute(node, tables)
        assert "mgr" in result.columns

    def test_composite_join(self, backend):
        left = DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"], "val": [10, 20, 30]})
        right = DataFrame({"a": [1, 2], "b": ["x", "x"], "score": [100, 200]})
        tables = {"l": left, "r": right}
        left_ir = ReadTable("l", infer_schema(left))
        right_ir = ReadTable("r", infer_schema(right))
        node = Join(left_ir, right_ir, ["a", "b"], ["a", "b"])
        result = backend.execute(node, tables)
        assert "score" in result.columns
        assert len(result) == 2
        assert set(result["val"]) == {10, 30}

    def test_union(self, backend):
        a = DataFrame({"x": [1, 2], "y": ["a", "b"]})
        b = DataFrame({"x": [3, 4], "y": ["c", "d"]})
        tables = {"a": a, "b": b}
        a_ir = ReadTable("a", infer_schema(a))
        b_ir = ReadTable("b", infer_schema(b))
        node = Union([a_ir, b_ir])
        result = backend.execute(node, tables)
        assert len(result) == 4
        assert list(result["x"]) == [1, 2, 3, 4]

    def test_union_three_inputs(self, backend):
        a = DataFrame({"v": [1]})
        b = DataFrame({"v": [2]})
        c = DataFrame({"v": [3]})
        tables = {"a": a, "b": b, "c": c}
        node = Union(
            [
                ReadTable("a", infer_schema(a)),
                ReadTable("b", infer_schema(b)),
                ReadTable("c", infer_schema(c)),
            ]
        )
        result = backend.execute(node, tables)
        assert len(result) == 3
        assert set(result["v"]) == {1, 2, 3}

    def test_distinct(self, backend):
        df = DataFrame({"x": [1, 2, 1, 2], "y": ["a", "b", "a", "b"]})
        tables = {"t": df}
        ir = ReadTable("t", infer_schema(df))
        node = Distinct(ir, ["x", "y"])
        result = backend.execute(node, tables)
        assert len(result) == 2

    def test_rename(self, backend, tables, base_ir):
        node = RenameColumns(base_ir, {"id": "row_id"})
        result = backend.execute(node, tables)
        assert "row_id" in result.columns
        assert "id" not in result.columns

    def test_singular_or_list(self, backend, tables, base_ir):
        expr = SingularOrList(ColRef("region"), [Literal("E")])
        node = Filter(base_ir, expr)
        result = backend.execute(node, tables)
        assert all(result["region"] == "E")

    def test_cast_expr(self, backend, tables, base_ir):
        expr = CastExpr(ColRef("price"), DType.FLOAT64)
        node = AddColumn(base_ir, "price_f64", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert "price_f64" in result.columns
        assert result["price_f64"].dtype == np.float64

    def test_if_then_expr(self, backend, tables, base_ir):
        expr = IfThenExpr(
            BinOp("gt", ColRef("price"), Literal(200)),
            ColRef("price"),
            Literal(0),
        )
        node = AddColumn(base_ir, "clamped", expr, DType.INT64)
        result = backend.execute(node, tables)
        assert "clamped" in result.columns
        # price > 200: keep price, else 0
        for _, row in result.iterrows():
            if row["price"] > 200:
                assert row["clamped"] == row["price"]
            else:
                assert row["clamped"] == 0

    def test_window_rolling(self, backend, tables, base_ir):
        spec = WindowSpec(kind="rows", lower_offset=1, upper_offset=0)
        node = Window(base_ir, [("price", "price", "sum")], spec)
        result = backend.execute(node, tables)
        assert "price" in result.columns
        assert len(result) == 4

    def test_window_expanding(self, backend, tables, base_ir):
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(base_ir, [("price", "price", "avg")], spec)
        result = backend.execute(node, tables)
        assert "price" in result.columns
        assert len(result) == 4

    def test_missing_table_raises(self, backend, base_ir):
        with pytest.raises(KeyError, match="Table 't' not registered"):
            backend.execute(base_ir, {})


# ---------------------------------------------------------------------------
# SchemaGuard
# ---------------------------------------------------------------------------


class TestSchemaGuard:
    def test_match(self):
        df = DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        schema = infer_schema(df)
        guard = SchemaGuard({"input": schema})
        assert guard.check({"input": df}) is True

    def test_mismatch_columns(self):
        df1 = DataFrame({"a": [1], "b": [2.0]})
        df2 = DataFrame({"a": [1], "c": [2.0]})
        guard = SchemaGuard({"input": infer_schema(df1)})
        assert guard.check({"input": df2}) is False

    def test_mismatch_dtypes(self):
        df1 = DataFrame({"a": np.array([1], dtype="int64")})
        df2 = DataFrame({"a": np.array([1.0], dtype="float64")})
        guard = SchemaGuard({"input": infer_schema(df1)})
        assert guard.check({"input": df2}) is False

    def test_missing_table(self):
        df = DataFrame({"a": [1]})
        guard = SchemaGuard({"input": infer_schema(df)})
        assert guard.check({}) is False

    def test_repr(self):
        df = DataFrame({"a": [1]})
        guard = SchemaGuard({"t": infer_schema(df)})
        r = repr(guard)
        assert "SchemaGuard" in r
        assert "INT64" in r


# ---------------------------------------------------------------------------
# AceroBackend
# ---------------------------------------------------------------------------

pa = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.substrait")


class TestAceroBackend:
    @pytest.fixture
    def backend(self):
        return AceroBackend()

    @pytest.fixture
    def sample_df(self):
        return DataFrame(
            {
                "id": [1, 2, 3, 4],
                "region": ["E", "W", "E", "W"],
                "price": [100.0, 250.0, 150.0, 300.0],
            }
        )

    @pytest.fixture
    def tables(self, sample_df):
        return {"t": sample_df}

    @pytest.fixture
    def base_ir(self, sample_df):
        return ReadTable("t", infer_schema(sample_df))

    def test_name(self, backend):
        assert backend.name == "acero"

    def test_read(self, backend, tables, base_ir):
        result = backend.execute(base_ir, tables)
        assert len(result) == 4

    def test_filter(self, backend, tables, base_ir):
        node = Filter(base_ir, BinOp("gt", ColRef("price"), Literal(100.0)))
        result = backend.execute(node, tables)
        assert all(result["price"] > 100)

    def test_project(self, backend, tables, base_ir):
        node = Project(base_ir, ["id", "price"])
        result = backend.execute(node, tables)
        assert list(result.columns) == ["id", "price"]

    def test_add_column(self, backend, tables, base_ir):
        expr = BinOp("mul", ColRef("price"), Literal(2.0))
        node = AddColumn(base_ir, "double", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert "double" in result.columns

    def test_sort(self, backend, tables, base_ir):
        node = Sort(base_ir, [("price", False)])
        result = backend.execute(node, tables)
        assert list(result["price"]) == [300.0, 250.0, 150.0, 100.0]

    def test_limit(self, backend, tables, base_ir):
        node = Limit(base_ir, 2)
        result = backend.execute(node, tables)
        assert len(result) == 2

    def test_limit_with_offset(self, backend, tables, base_ir):
        node = Limit(base_ir, 2, offset=1)
        result = backend.execute(node, tables)
        assert len(result) == 2
        assert list(result["id"]) == [2, 3]

    def test_aggregate(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("total", "price", "sum"), ("cnt", "id", "count")],
        )
        result = backend.execute(node, tables)
        assert set(result.columns) == {"region", "total", "cnt"}
        assert len(result) == 2

    def test_join(self, backend, sample_df):
        regions = DataFrame({"region": ["E", "W"], "mgr": ["Alice", "Bob"]})
        tables = {"l": sample_df, "r": regions}
        left_ir = ReadTable("l", infer_schema(sample_df))
        right_ir = ReadTable("r", infer_schema(regions))
        node = Join(left_ir, right_ir, "region", "region")
        result = backend.execute(node, tables)
        assert "mgr" in result.columns

    def test_composite_join(self, backend):
        left = DataFrame(
            {"a": [1, 1, 2], "b": ["x", "y", "x"], "val": [10.0, 20.0, 30.0]}
        )
        right = DataFrame({"a": [1, 2], "b": ["x", "x"], "score": [100.0, 200.0]})
        tables = {"l": left, "r": right}
        left_ir = ReadTable("l", infer_schema(left))
        right_ir = ReadTable("r", infer_schema(right))
        node = Join(left_ir, right_ir, ["a", "b"], ["a", "b"])
        result = backend.execute(node, tables)
        assert "score" in result.columns
        assert len(result) == 2
        assert set(result["val"]) == {10.0, 30.0}

    def test_union(self, backend):
        a = DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        b = DataFrame({"x": [3.0, 4.0], "y": ["c", "d"]})
        tables = {"a": a, "b": b}
        a_ir = ReadTable("a", infer_schema(a))
        b_ir = ReadTable("b", infer_schema(b))
        node = Union([a_ir, b_ir])
        result = backend.execute(node, tables)
        assert len(result) == 4
        assert set(result["x"]) == {1.0, 2.0, 3.0, 4.0}

    def test_distinct(self, backend):
        df = DataFrame({"x": [1.0, 2.0, 1.0, 2.0], "y": ["a", "b", "a", "b"]})
        tables = {"t": df}
        ir = ReadTable("t", infer_schema(df))
        node = Distinct(ir, ["x", "y"])
        result = backend.execute(node, tables)
        assert len(result) == 2

    def test_rename(self, backend, tables, base_ir):
        node = RenameColumns(base_ir, {"price": "cost"})
        result = backend.execute(node, tables)
        assert "cost" in result.columns
        assert "price" not in result.columns

    def test_filter_is_null(self, backend):
        df = DataFrame(
            {
                "x": [1.0, None, 3.0, None],
                "y": ["a", "b", "c", "d"],
            }
        )
        schema = infer_schema(df)
        tables = {"t": df}
        base = ReadTable("t", schema)
        # Filter for NOT NULL rows: NOT(IS_NULL(x))
        pred = UnaryOp("not", UnaryOp("is_null", ColRef("x")))
        node = Filter(base, pred)
        result = backend.execute(node, tables)
        assert len(result) == 2
        assert result["x"].notna().all()

    def test_add_column_coalesce(self, backend):
        df = DataFrame({"x": [1.0, None, 3.0, None]})
        schema = infer_schema(df)
        tables = {"t": df}
        base = ReadTable("t", schema)
        expr = BinOp("coalesce", ColRef("x"), Literal(0.0))
        node = AddColumn(base, "x_filled", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert result["x_filled"].notna().all()
        assert list(result["x_filled"]) == [1.0, 0.0, 3.0, 0.0]

    def test_add_column_abs(self, backend, tables, base_ir):
        expr = UnaryOp("abs", BinOp("sub", ColRef("price"), Literal(200.0)))
        node = AddColumn(base_ir, "dist", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert "dist" in result.columns
        assert all(result["dist"] >= 0)

    def test_add_column_negate(self, backend, tables, base_ir):
        expr = UnaryOp("negate", ColRef("price"))
        node = AddColumn(base_ir, "neg_price", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert all(result["neg_price"] < 0)

    def test_compound_filter(self, backend, tables, base_ir):
        # price > 100 AND region == "E"
        pred = BinOp(
            "and",
            BinOp("gt", ColRef("price"), Literal(100.0)),
            BinOp("eq", ColRef("region"), Literal("E")),
        )
        node = Filter(base_ir, pred)
        result = backend.execute(node, tables)
        assert all(result["price"] > 100)
        assert all(result["region"] == "E")

    def test_filter_or(self, backend, tables, base_ir):
        # price < 150 OR price > 250
        pred = BinOp(
            "or",
            BinOp("lt", ColRef("price"), Literal(150.0)),
            BinOp("gt", ColRef("price"), Literal(250.0)),
        )
        node = Filter(base_ir, pred)
        result = backend.execute(node, tables)
        assert all((result["price"] < 150) | (result["price"] > 250))

    def test_filter_isin_pattern(self, backend, tables, base_ir):
        # eq(region, "E") OR eq(region, "W") — isin pattern
        pred = BinOp(
            "or",
            BinOp("eq", ColRef("region"), Literal("E")),
            BinOp("eq", ColRef("region"), Literal("W")),
        )
        node = Filter(base_ir, pred)
        result = backend.execute(node, tables)
        assert len(result) == 4

    def test_singular_or_list(self, backend, tables, base_ir):
        expr = SingularOrList(ColRef("region"), [Literal("E"), Literal("W")])
        node = Filter(base_ir, expr)
        result = backend.execute(node, tables)
        assert len(result) == 4

    def test_sort_multiple_keys(self, backend, tables, base_ir):
        node = Sort(base_ir, [("region", True), ("price", False)])
        result = backend.execute(node, tables)
        # Regions should be sorted ascending
        assert list(result["region"]) == ["E", "E", "W", "W"]
        # Within each region, price should be descending
        e_prices = list(result[result["region"] == "E"]["price"])
        assert e_prices == sorted(e_prices, reverse=True)

    def test_aggregate_mean(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("avg_price", "price", "avg")],
        )
        result = backend.execute(node, tables)
        assert "avg_price" in result.columns
        assert len(result) == 2

    def test_aggregate_min_max(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("lo", "price", "min"), ("hi", "price", "max")],
        )
        result = backend.execute(node, tables)
        assert set(result.columns) == {"region", "lo", "hi"}
        for _, row in result.iterrows():
            assert row["lo"] <= row["hi"]

    def test_aggregate_std(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("std_price", "price", "std")],
        )
        result = backend.execute(node, tables)
        assert "std_price" in result.columns
        assert len(result) == 2
        assert all(result["std_price"] >= 0)

    def test_aggregate_var(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("var_price", "price", "var")],
        )
        result = backend.execute(node, tables)
        assert "var_price" in result.columns
        assert len(result) == 2
        assert all(result["var_price"] >= 0)


# ---------------------------------------------------------------------------
# DataFusionBackend
# ---------------------------------------------------------------------------

datafusion = pytest.importorskip("datafusion")
pytest.importorskip("datafusion.substrait")


class TestDataFusionBackend:
    @pytest.fixture
    def backend(self):
        return DataFusionBackend()

    @pytest.fixture
    def sample_df(self):
        return DataFrame(
            {
                "id": [1, 2, 3, 4],
                "region": ["E", "W", "E", "W"],
                "price": [100.0, 250.0, 150.0, 300.0],
            }
        )

    @pytest.fixture
    def tables(self, sample_df):
        return {"t": sample_df}

    @pytest.fixture
    def base_ir(self, sample_df):
        return ReadTable("t", infer_schema(sample_df))

    def test_name(self, backend):
        assert backend.name == "datafusion"

    def test_read(self, backend, tables, base_ir):
        result = backend.execute(base_ir, tables)
        assert len(result) == 4

    def test_filter(self, backend, tables, base_ir):
        node = Filter(base_ir, BinOp("gt", ColRef("price"), Literal(100.0)))
        result = backend.execute(node, tables)
        assert all(result["price"] > 100)

    def test_project(self, backend, tables, base_ir):
        node = Project(base_ir, ["id", "price"])
        result = backend.execute(node, tables)
        assert list(result.columns) == ["id", "price"]

    def test_add_column(self, backend, tables, base_ir):
        expr = BinOp("mul", ColRef("price"), Literal(2.0))
        node = AddColumn(base_ir, "double", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert "double" in result.columns

    def test_sort(self, backend, tables, base_ir):
        node = Sort(base_ir, [("price", False)])
        result = backend.execute(node, tables)
        assert list(result["price"]) == [300.0, 250.0, 150.0, 100.0]

    def test_limit(self, backend, tables, base_ir):
        node = Limit(base_ir, 2)
        result = backend.execute(node, tables)
        assert len(result) == 2

    def test_limit_with_offset(self, backend, tables, base_ir):
        node = Limit(base_ir, 2, offset=1)
        result = backend.execute(node, tables)
        assert len(result) == 2
        assert list(result["id"]) == [2, 3]

    def test_aggregate(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("total", "price", "sum"), ("cnt", "id", "count")],
        )
        result = backend.execute(node, tables)
        assert set(result.columns) == {"region", "total", "cnt"}
        assert len(result) == 2

    def test_join(self, backend, sample_df):
        regions = DataFrame({"region": ["E", "W"], "mgr": ["Alice", "Bob"]})
        tables = {"l": sample_df, "r": regions}
        left_ir = ReadTable("l", infer_schema(sample_df))
        right_ir = ReadTable("r", infer_schema(regions))
        node = Join(left_ir, right_ir, "region", "region")
        result = backend.execute(node, tables)
        assert "mgr" in result.columns

    def test_composite_join(self, backend):
        left = DataFrame(
            {"a": [1, 1, 2], "b": ["x", "y", "x"], "val": [10.0, 20.0, 30.0]}
        )
        right = DataFrame({"a": [1, 2], "b": ["x", "x"], "score": [100.0, 200.0]})
        tables = {"l": left, "r": right}
        left_ir = ReadTable("l", infer_schema(left))
        right_ir = ReadTable("r", infer_schema(right))
        node = Join(left_ir, right_ir, ["a", "b"], ["a", "b"])
        result = backend.execute(node, tables)
        assert "score" in result.columns
        assert len(result) == 2
        assert set(result["val"]) == {10.0, 30.0}

    def test_union(self, backend):
        a = DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        b = DataFrame({"x": [3.0, 4.0], "y": ["c", "d"]})
        tables = {"a": a, "b": b}
        a_ir = ReadTable("a", infer_schema(a))
        b_ir = ReadTable("b", infer_schema(b))
        node = Union([a_ir, b_ir])
        result = backend.execute(node, tables)
        assert len(result) == 4
        assert set(result["x"]) == {1.0, 2.0, 3.0, 4.0}

    def test_distinct(self, backend):
        df = DataFrame({"x": [1.0, 2.0, 1.0, 2.0], "y": ["a", "b", "a", "b"]})
        tables = {"t": df}
        ir = ReadTable("t", infer_schema(df))
        node = Distinct(ir, ["x", "y"])
        result = backend.execute(node, tables)
        assert len(result) == 2

    def test_rename(self, backend, tables, base_ir):
        node = RenameColumns(base_ir, {"price": "cost"})
        result = backend.execute(node, tables)
        assert "cost" in result.columns
        assert "price" not in result.columns

    def test_filter_is_null(self, backend):
        df = DataFrame(
            {
                "x": [1.0, None, 3.0, None],
                "y": ["a", "b", "c", "d"],
            }
        )
        schema = infer_schema(df)
        tables = {"t": df}
        base = ReadTable("t", schema)
        pred = UnaryOp("not", UnaryOp("is_null", ColRef("x")))
        node = Filter(base, pred)
        result = backend.execute(node, tables)
        assert len(result) == 2
        assert result["x"].notna().all()

    def test_add_column_coalesce(self, backend):
        df = DataFrame({"x": [1.0, None, 3.0, None]})
        schema = infer_schema(df)
        tables = {"t": df}
        base = ReadTable("t", schema)
        expr = BinOp("coalesce", ColRef("x"), Literal(0.0))
        node = AddColumn(base, "x_filled", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert result["x_filled"].notna().all()
        assert list(result["x_filled"]) == [1.0, 0.0, 3.0, 0.0]

    def test_add_column_abs(self, backend, tables, base_ir):
        expr = UnaryOp("abs", BinOp("sub", ColRef("price"), Literal(200.0)))
        node = AddColumn(base_ir, "dist", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert "dist" in result.columns
        assert all(result["dist"] >= 0)

    def test_add_column_negate(self, backend, tables, base_ir):
        expr = UnaryOp("negate", ColRef("price"))
        node = AddColumn(base_ir, "neg_price", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert all(result["neg_price"] < 0)

    def test_compound_filter(self, backend, tables, base_ir):
        pred = BinOp(
            "and",
            BinOp("gt", ColRef("price"), Literal(100.0)),
            BinOp("eq", ColRef("region"), Literal("E")),
        )
        node = Filter(base_ir, pred)
        result = backend.execute(node, tables)
        assert all(result["price"] > 100)
        assert all(result["region"] == "E")

    def test_filter_or(self, backend, tables, base_ir):
        pred = BinOp(
            "or",
            BinOp("lt", ColRef("price"), Literal(150.0)),
            BinOp("gt", ColRef("price"), Literal(250.0)),
        )
        node = Filter(base_ir, pred)
        result = backend.execute(node, tables)
        assert all((result["price"] < 150) | (result["price"] > 250))

    def test_singular_or_list(self, backend, tables, base_ir):
        expr = SingularOrList(ColRef("region"), [Literal("E"), Literal("W")])
        node = Filter(base_ir, expr)
        result = backend.execute(node, tables)
        assert len(result) == 4

    def test_sort_multiple_keys(self, backend, tables, base_ir):
        node = Sort(base_ir, [("region", True), ("price", False)])
        result = backend.execute(node, tables)
        assert list(result["region"]) == ["E", "E", "W", "W"]
        e_prices = list(result[result["region"] == "E"]["price"])
        assert e_prices == sorted(e_prices, reverse=True)

    def test_aggregate_mean(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("avg_price", "price", "avg")],
        )
        result = backend.execute(node, tables)
        assert "avg_price" in result.columns
        assert len(result) == 2

    def test_aggregate_min_max(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("lo", "price", "min"), ("hi", "price", "max")],
        )
        result = backend.execute(node, tables)
        assert set(result.columns) == {"region", "lo", "hi"}
        for _, row in result.iterrows():
            assert row["lo"] <= row["hi"]

    def test_aggregate_std(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("std_price", "price", "std")],
        )
        result = backend.execute(node, tables)
        assert "std_price" in result.columns
        assert len(result) == 2
        assert all(result["std_price"] >= 0)

    def test_aggregate_var(self, backend, tables, base_ir):
        node = Aggregate(
            base_ir,
            ["region"],
            [("var_price", "price", "var")],
        )
        result = backend.execute(node, tables)
        assert "var_price" in result.columns
        assert len(result) == 2
        assert all(result["var_price"] >= 0)

    def test_if_then_expr(self, backend, tables, base_ir):
        expr = IfThenExpr(
            BinOp("gt", ColRef("price"), Literal(200.0)),
            ColRef("price"),
            Literal(0.0),
        )
        node = AddColumn(base_ir, "clamped", expr, DType.FLOAT64)
        result = backend.execute(node, tables)
        assert "clamped" in result.columns
        for _, row in result.iterrows():
            if row["price"] > 200:
                assert row["clamped"] == row["price"]
            else:
                assert row["clamped"] == 0.0

    def test_cast_expr(self, backend, tables, base_ir):
        expr = CastExpr(ColRef("price"), DType.INT64)
        node = AddColumn(base_ir, "price_int", expr, DType.INT64)
        result = backend.execute(node, tables)
        assert "price_int" in result.columns

    def test_window_rolling(self, backend, tables, base_ir):
        spec = WindowSpec(kind="rows", lower_offset=1, upper_offset=0)
        node = Window(base_ir, [("price", "price", "sum")], spec)
        result = backend.execute(node, tables)
        assert "price" in result.columns
        assert len(result) == 4

    def test_window_expanding(self, backend, tables, base_ir):
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(base_ir, [("price", "price", "avg")], spec)
        result = backend.execute(node, tables)
        assert "price" in result.columns
        assert len(result) == 4


# ---------------------------------------------------------------------------
# default_backend
# ---------------------------------------------------------------------------


class TestDefaultBackendSelection:
    def test_returns_backend_instance(self):
        backend = default_backend()
        assert isinstance(backend, Backend)

    def test_default_backend_is_datafusion(self):
        pytest.importorskip("datafusion")
        pytest.importorskip("datafusion.substrait")
        backend = default_backend()
        assert isinstance(backend, DataFusionBackend)


# ---------------------------------------------------------------------------
# ScalarSubquery — Substrait compilation + backend evaluation
# ---------------------------------------------------------------------------


class TestScalarSubquery:
    @pytest.fixture
    def schema(self):
        return Schema(
            {
                "price": DType.INT64,
                "region": DType.STRING,
            }
        )

    @pytest.fixture
    def base_ir(self, schema):
        return ReadTable("t", schema)

    @pytest.fixture
    def tables(self, schema):
        return {
            "t": DataFrame(
                {
                    "price": [100, 250, 150, 300],
                    "region": ["E", "W", "E", "W"],
                }
            )
        }

    def test_compile_scalar_subquery_in_expression(self, schema, base_ir):
        """Verify ScalarSubquery compiles to Substrait Expression.Subquery.Scalar."""
        agg_node = Aggregate(base_ir, [], [("price", "price", "sum")])
        subquery = ScalarSubquery(agg_node, DType.INT64)
        expr = BinOp("div", ColRef("price"), subquery)
        node = AddColumn(base_ir, "pct", expr, DType.FLOAT64)

        compiler = SubstraitCompiler()
        plan = compiler.compile(node)

        # The plan should have been built successfully
        assert plan.SerializeToString()
        # Check that we have extension functions registered (div, sum)
        assert len(plan.extensions) >= 2

    def test_pandas_backend_scalar_subquery(self, schema, base_ir, tables):
        """PandasBackend correctly evaluates ScalarSubquery."""
        agg_node = Aggregate(base_ir, [], [("price", "price", "sum")])
        subquery = ScalarSubquery(agg_node, DType.INT64)
        expr = BinOp("div", ColRef("price"), subquery)
        node = AddColumn(base_ir, "pct", expr, DType.FLOAT64)

        backend = PandasBackend()
        result = backend.execute(node, tables)
        expected = tables["t"]["price"] / tables["t"]["price"].sum()
        tm.assert_series_equal(result["pct"], expected, check_names=False)

    def test_datafusion_scalar_subquery(self):
        """DataFusion executes scalar subquery plan correctly."""
        pytest.importorskip("datafusion")
        pytest.importorskip("datafusion.substrait")

        # Use float data to avoid integer division
        float_schema = Schema({"price": DType.FLOAT64, "region": DType.STRING})
        float_ir = ReadTable("t", float_schema)
        float_tables = {
            "t": DataFrame(
                {
                    "price": [100.0, 250.0, 150.0, 300.0],
                    "region": ["E", "W", "E", "W"],
                }
            )
        }

        agg_node = Aggregate(float_ir, [], [("price", "price", "sum")])
        subquery = ScalarSubquery(agg_node, DType.FLOAT64)
        expr = BinOp("div", ColRef("price"), subquery)
        node = AddColumn(float_ir, "pct", expr, DType.FLOAT64)

        backend = DataFusionBackend()
        result = backend.execute(node, float_tables)
        expected = float_tables["t"]["price"] / float_tables["t"]["price"].sum()
        tm.assert_series_equal(result["pct"], expected, check_names=False)

    def test_scalar_subquery_with_avg(self, schema, base_ir, tables):
        """ScalarSubquery with avg aggregation."""
        agg_node = Aggregate(base_ir, [], [("price", "price", "avg")])
        subquery = ScalarSubquery(agg_node, DType.FLOAT64)
        expr = BinOp("sub", ColRef("price"), subquery)
        node = AddColumn(base_ir, "diff", expr, DType.FLOAT64)

        backend = PandasBackend()
        result = backend.execute(node, tables)
        expected = tables["t"]["price"] - tables["t"]["price"].mean()
        tm.assert_series_equal(result["diff"], expected, check_names=False)

    def test_scalar_subquery_in_filter(self, schema, base_ir, tables):
        """ScalarSubquery used in a filter predicate."""
        agg_node = Aggregate(base_ir, [], [("price", "price", "avg")])
        subquery = ScalarSubquery(agg_node, DType.FLOAT64)
        pred = BinOp("gt", ColRef("price"), subquery)
        node = Filter(base_ir, pred)

        backend = PandasBackend()
        result = backend.execute(node, tables)
        avg = tables["t"]["price"].mean()
        expected = tables["t"][tables["t"]["price"] > avg].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)
