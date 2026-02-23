"""Tests for pandas.jit.ir — DType enum, type mappings, Schema,
IR nodes, expressions."""

from __future__ import annotations

import pytest

from pandas.jit.ir import (
    _PANDAS_DTYPE_MAP,
    NUMERIC_DTYPES,
    AddColumn,
    Aggregate,
    BinOp,
    CastExpr,
    ColRef,
    DType,
    Filter,
    IfThenExpr,
    IRNode,
    Join,
    Limit,
    Literal,
    Project,
    ReadTable,
    RenameColumns,
    Schema,
    SingularOrList,
    Sort,
    UnaryOp,
    Window,
    WindowSpec,
    _wrap_literal,
    explain_expr,
    explain_ir,
    pandas_dtype_to_ir,
)

# ---------------------------------------------------------------------------
# DType enum
# ---------------------------------------------------------------------------


class TestDType:
    def test_all_members(self):
        expected = {
            "INT8",
            "INT16",
            "INT32",
            "INT64",
            "UINT8",
            "UINT16",
            "UINT32",
            "UINT64",
            "FLOAT32",
            "FLOAT64",
            "STRING",
            "BINARY",
            "BOOL",
            "DATE",
            "TIME",
            "TIMESTAMP",
            "TIMESTAMP_TZ",
            "TIMEDELTA",
            "DECIMAL",
        }
        assert {m.name for m in DType} == expected

    def test_unique_values(self):
        values = [m.value for m in DType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# _PANDAS_DTYPE_MAP / pandas_dtype_to_ir
# ---------------------------------------------------------------------------


class TestPandasDtypeMap:
    @pytest.mark.parametrize(
        "dtype_str, expected",
        [
            # numpy signed
            ("int8", DType.INT8),
            ("int16", DType.INT16),
            ("int32", DType.INT32),
            ("int64", DType.INT64),
            # pandas nullable signed
            ("Int8", DType.INT8),
            ("Int16", DType.INT16),
            ("Int32", DType.INT32),
            ("Int64", DType.INT64),
            # numpy unsigned
            ("uint8", DType.UINT8),
            ("uint16", DType.UINT16),
            ("uint32", DType.UINT32),
            ("uint64", DType.UINT64),
            # pandas nullable unsigned
            ("UInt8", DType.UINT8),
            ("UInt16", DType.UINT16),
            ("UInt32", DType.UINT32),
            ("UInt64", DType.UINT64),
            # float
            ("float32", DType.FLOAT32),
            ("float64", DType.FLOAT64),
            ("Float32", DType.FLOAT32),
            ("Float64", DType.FLOAT64),
            # string
            ("object", DType.STRING),
            ("string", DType.STRING),
            ("str", DType.STRING),
            ("string[python]", DType.STRING),
            ("string[pyarrow]", DType.STRING),
            # binary
            ("bytes", DType.BINARY),
            # boolean
            ("bool", DType.BOOL),
            ("boolean", DType.BOOL),
            # temporal
            ("datetime64[ns]", DType.TIMESTAMP),
            ("datetime64[us]", DType.TIMESTAMP),
            ("datetime64[ms]", DType.TIMESTAMP),
            ("datetime64[s]", DType.TIMESTAMP),
            ("date", DType.DATE),
            ("timedelta64[ns]", DType.TIMEDELTA),
            ("timedelta64[us]", DType.TIMEDELTA),
            ("timedelta64[ms]", DType.TIMEDELTA),
            ("timedelta64[s]", DType.TIMEDELTA),
        ],
    )
    def test_exact_match(self, dtype_str, expected):
        assert _PANDAS_DTYPE_MAP[dtype_str] is expected

    @pytest.mark.parametrize(
        "dtype_str, expected",
        [
            ("datetime64[ns, UTC]", DType.TIMESTAMP_TZ),
            ("datetime64[ns, US/Eastern]", DType.TIMESTAMP_TZ),
            ("datetime64[us, Europe/London]", DType.TIMESTAMP_TZ),
        ],
    )
    def test_timestamp_tz_fallback(self, dtype_str, expected):
        assert pandas_dtype_to_ir(dtype_str) is expected

    def test_unknown_dtype_returns_string(self):
        assert pandas_dtype_to_ir("category") is DType.STRING
        assert pandas_dtype_to_ir("complex128") is DType.STRING


# ---------------------------------------------------------------------------
# NUMERIC_DTYPES
# ---------------------------------------------------------------------------


class TestNumericDtypes:
    def test_contains_all_integer_types(self):
        for dt in (
            DType.INT8,
            DType.INT16,
            DType.INT32,
            DType.INT64,
            DType.UINT8,
            DType.UINT16,
            DType.UINT32,
            DType.UINT64,
        ):
            assert dt in NUMERIC_DTYPES

    def test_contains_float_types(self):
        assert DType.FLOAT32 in NUMERIC_DTYPES
        assert DType.FLOAT64 in NUMERIC_DTYPES

    def test_contains_decimal(self):
        assert DType.DECIMAL in NUMERIC_DTYPES

    def test_excludes_non_numeric(self):
        for dt in (
            DType.STRING,
            DType.BINARY,
            DType.BOOL,
            DType.DATE,
            DType.TIMESTAMP,
            DType.TIMESTAMP_TZ,
            DType.TIMEDELTA,
            DType.TIME,
        ):
            assert dt not in NUMERIC_DTYPES


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_column_index(self):
        schema = Schema({"a": DType.INT64, "b": DType.STRING, "c": DType.BOOL})
        assert schema.column_index("a") == 0
        assert schema.column_index("b") == 1
        assert schema.column_index("c") == 2

    def test_column_index_missing_raises(self):
        schema = Schema({"a": DType.INT64})
        with pytest.raises(ValueError, match="x"):
            schema.column_index("x")

    def test_column_names(self):
        schema = Schema({"x": DType.INT32, "y": DType.FLOAT64})
        assert schema.column_names() == ["x", "y"]

    def test_empty_schema(self):
        schema = Schema()
        assert schema.column_names() == []


# ---------------------------------------------------------------------------
# IR nodes — output_schema
# ---------------------------------------------------------------------------


def _sample_schema():
    return Schema({"id": DType.INT64, "name": DType.STRING, "price": DType.FLOAT64})


class TestReadTable:
    def test_output_schema(self):
        schema = _sample_schema()
        node = ReadTable("t", schema)
        assert node.output_schema() is schema
        assert node.name == "t"


class TestFilter:
    def test_output_schema_passthrough(self):
        base = ReadTable("t", _sample_schema())
        pred = BinOp("gt", ColRef("price"), Literal(100))
        node = Filter(base, pred)
        assert node.output_schema().column_names() == ["id", "name", "price"]


class TestProject:
    def test_selects_subset(self):
        base = ReadTable("t", _sample_schema())
        node = Project(base, ["id", "price"])
        schema = node.output_schema()
        assert schema.column_names() == ["id", "price"]
        assert schema.columns["id"] is DType.INT64
        assert schema.columns["price"] is DType.FLOAT64


class TestAddColumn:
    def test_adds_column(self):
        base = ReadTable("t", _sample_schema())
        expr = BinOp("mul", ColRef("price"), Literal(2))
        node = AddColumn(base, "double_price", expr, DType.FLOAT64)
        schema = node.output_schema()
        assert "double_price" in schema.columns
        assert schema.column_names() == ["id", "name", "price", "double_price"]


class TestSort:
    def test_preserves_schema(self):
        base = ReadTable("t", _sample_schema())
        node = Sort(base, [("price", True)])
        assert node.output_schema().column_names() == ["id", "name", "price"]


class TestLimit:
    def test_preserves_schema(self):
        base = ReadTable("t", _sample_schema())
        node = Limit(base, 5)
        assert node.output_schema().column_names() == ["id", "name", "price"]


class TestAggregate:
    def test_output_schema_with_count(self):
        base = ReadTable("t", _sample_schema())
        node = Aggregate(
            base,
            group_keys=["name"],
            agg_specs=[("total", "price", "sum"), ("cnt", "id", "count")],
        )
        schema = node.output_schema()
        assert schema.column_names() == ["name", "total", "cnt"]
        assert schema.columns["total"] is DType.FLOAT64
        assert schema.columns["cnt"] is DType.INT64


class TestJoin:
    def test_merges_schemas(self):
        left = ReadTable("l", Schema({"id": DType.INT64, "val": DType.FLOAT64}))
        right = ReadTable("r", Schema({"id": DType.INT64, "label": DType.STRING}))
        node = Join(left, right, "id", "id")
        schema = node.output_schema()
        assert set(schema.columns.keys()) == {"id", "val", "label"}


class TestWindow:
    def test_rolling_output_schema(self):
        base = ReadTable("t", _sample_schema())
        spec = WindowSpec(kind="rows", lower_offset=2, upper_offset=0)
        node = Window(base, [("price", "price", "sum", 0)], spec)
        schema = node.output_schema()
        # sum preserves source dtype (FLOAT64)
        assert schema.columns["price"] is DType.FLOAT64

    def test_expanding_output_schema(self):
        base = ReadTable("t", _sample_schema())
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(base, [("price", "price", "avg", 0)], spec)
        schema = node.output_schema()
        assert schema.columns["price"] is DType.FLOAT64

    def test_count_output_schema(self):
        base = ReadTable("t", _sample_schema())
        spec = WindowSpec(kind="rows", lower_offset=1, upper_offset=0)
        node = Window(base, [("price", "price", "count", 0)], spec)
        schema = node.output_schema()
        assert schema.columns["price"] is DType.INT64

    def test_lag_output_schema(self):
        base = ReadTable("t", _sample_schema())
        spec = WindowSpec()
        node = Window(base, [("price", "price", "lag", 1)], spec)
        schema = node.output_schema()
        # lag preserves source dtype
        assert schema.columns["price"] is DType.FLOAT64

    def test_explain_rolling(self):
        base = ReadTable("t", Schema({"x": DType.INT64}))
        spec = WindowSpec(kind="rows", lower_offset=2, upper_offset=0)
        node = Window(base, [("x", "x", "sum", 0)], spec)
        result = explain_ir(node)
        assert "Window" in result
        assert "frame=2..0" in result

    def test_explain_expanding(self):
        base = ReadTable("t", Schema({"x": DType.INT64}))
        spec = WindowSpec(kind="rows", lower_offset=None, upper_offset=0)
        node = Window(base, [("x", "x", "sum", 0)], spec)
        result = explain_ir(node)
        assert "UNBOUNDED" in result


class TestRenameColumns:
    def test_renames(self):
        base = ReadTable("t", Schema({"a": DType.INT64, "b": DType.STRING}))
        node = RenameColumns(base, {"a": "x"})
        schema = node.output_schema()
        assert schema.column_names() == ["x", "b"]
        assert schema.columns["x"] is DType.INT64


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


class TestLiteral:
    @pytest.mark.parametrize(
        "value, expected_dtype",
        [
            (True, DType.BOOL),
            (False, DType.BOOL),
            (42, DType.INT64),
            (3.14, DType.FLOAT64),
            ("hello", DType.STRING),
            (b"raw", DType.BINARY),
        ],
    )
    def test_auto_dtype(self, value, expected_dtype):
        lit = Literal(value)
        assert lit.dtype is expected_dtype

    def test_explicit_dtype(self):
        lit = Literal(10, DType.INT32)
        assert lit.dtype is DType.INT32


class TestColRef:
    def test_comparison_ops(self):
        col = ColRef("x")
        for op_name, method in [
            ("gt", "__gt__"),
            ("gte", "__ge__"),
            ("lt", "__lt__"),
            ("lte", "__le__"),
            ("eq", "__eq__"),
            ("ne", "__ne__"),
        ]:
            result = getattr(col, method)(10)
            assert isinstance(result, BinOp)
            assert result.op == op_name

    def test_arithmetic_ops(self):
        col = ColRef("x")
        for op_name, method in [
            ("add", "__add__"),
            ("sub", "__sub__"),
            ("mul", "__mul__"),
            ("div", "__truediv__"),
        ]:
            result = getattr(col, method)(2)
            assert isinstance(result, BinOp)
            assert result.op == op_name


class TestBinOp:
    def test_and_or(self):
        left = BinOp("gt", ColRef("x"), Literal(0))
        right = BinOp("lt", ColRef("x"), Literal(100))
        combined = left & right
        assert isinstance(combined, BinOp)
        assert combined.op == "and"
        combined_or = left | right
        assert combined_or.op == "or"


class TestIfThenExpr:
    def test_creation(self):
        expr = IfThenExpr(
            BinOp("gt", ColRef("x"), Literal(0)),
            ColRef("x"),
            Literal(-1),
        )
        assert isinstance(expr, IfThenExpr)
        assert isinstance(expr.condition, BinOp)

    def test_explain(self):
        expr = IfThenExpr(
            BinOp("gt", ColRef("x"), Literal(0)),
            ColRef("x"),
            Literal(-1),
        )
        result = explain_expr(expr)
        assert result == "IF(($x > 0), $x, -1)"


class TestCastExpr:
    def test_creation(self):
        expr = CastExpr(ColRef("x"), DType.FLOAT64)
        assert isinstance(expr, CastExpr)
        assert expr.target_dtype is DType.FLOAT64

    def test_explain(self):
        expr = CastExpr(ColRef("x"), DType.INT32)
        result = explain_expr(expr)
        assert result == "CAST($x AS INT32)"


class TestSingularOrList:
    def test_creation(self):
        expr = SingularOrList(ColRef("x"), [Literal(1), Literal(2), Literal(3)])
        assert isinstance(expr, SingularOrList)
        assert isinstance(expr.value, ColRef)
        assert len(expr.options) == 3

    def test_explain(self):
        expr = SingularOrList(ColRef("x"), [Literal(1), Literal(2)])
        result = explain_expr(expr)
        assert result == "$x IN (1, 2)"


class TestWrapLiteral:
    def test_expr_passthrough(self):
        expr = ColRef("x")
        assert _wrap_literal(expr) is expr

    def test_wraps_scalar(self):
        result = _wrap_literal(42)
        assert isinstance(result, Literal)
        assert result.value == 42


# ---------------------------------------------------------------------------
# explain_ir / explain_expr
# ---------------------------------------------------------------------------


class TestExplainExpr:
    def test_colref(self):
        assert explain_expr(ColRef("x")) == "$x"

    def test_literal(self):
        assert explain_expr(Literal(42)) == "42"
        assert explain_expr(Literal("hello")) == "'hello'"

    def test_binop(self):
        expr = BinOp("gt", ColRef("x"), Literal(10))
        assert explain_expr(expr) == "($x > 10)"

    def test_unaryop(self):
        expr = UnaryOp("not", ColRef("flag"))
        assert explain_expr(expr) == "not($flag)"


class TestExplainIR:
    def test_read_table(self):
        node = ReadTable("t", Schema({"a": DType.INT64}))
        result = explain_ir(node)
        assert "ReadTable(t)" in result
        assert "a:INT64" in result

    def test_filter(self):
        base = ReadTable("t", Schema({"x": DType.INT64}))
        pred = BinOp("gt", ColRef("x"), Literal(0))
        node = Filter(base, pred)
        result = explain_ir(node)
        assert "Filter" in result
        assert "ReadTable" in result

    def test_nested_indentation(self):
        base = ReadTable("t", Schema({"x": DType.INT64}))
        node = Limit(Sort(base, [("x", True)]), 10)
        result = explain_ir(node)
        lines = result.split("\n")
        assert len(lines) == 3
        # Each nested level should have more indentation
        assert lines[1].startswith("  ")
        assert lines[2].startswith("    ")

    def test_unknown_node(self):
        class CustomNode(IRNode):
            def output_schema(self):
                return Schema()

        result = explain_ir(CustomNode())
        assert "Unknown(CustomNode)" in result
