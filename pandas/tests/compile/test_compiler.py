"""Tests for pandas.compile.compiler — SubstraitCompiler, backends, infer_schema."""

from __future__ import annotations

import numpy as np
import pytest
import substrait.type_pb2 as stt

import pandas as pd
from pandas import DataFrame
from pandas.compile.compiler import (
    _DTYPE_TO_SUBSTRAIT,
    AceroBackend,
    Backend,
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
    ColRef,
    DType,
    Filter,
    Join,
    Limit,
    Literal,
    Project,
    ReadTable,
    RenameColumns,
    Schema,
    Sort,
    UnaryOp,
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

    def test_unknown_ir_node_raises(self):
        class BadNode:
            pass

        compiler = SubstraitCompiler()
        with pytest.raises(TypeError, match="Unknown IR node"):
            compiler._compile_rel(BadNode())


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

    def test_rename(self, backend, tables, base_ir):
        node = RenameColumns(base_ir, {"id": "row_id"})
        result = backend.execute(node, tables)
        assert "row_id" in result.columns
        assert "id" not in result.columns

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
# default_backend
# ---------------------------------------------------------------------------


class TestDefaultBackend:
    def test_returns_backend_instance(self):
        backend = default_backend()
        assert isinstance(backend, Backend)

    def test_returns_acero_when_pyarrow_available(self):
        pytest.importorskip("pyarrow")
        pytest.importorskip("pyarrow.substrait")
        backend = default_backend()
        assert isinstance(backend, AceroBackend)
