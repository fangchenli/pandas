"""Tests for pandas.compile.jit — tracing proxies, @compile decorator, Tracer."""

from __future__ import annotations

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
from pandas.compile.compiler import (
    PandasBackend,
    infer_schema,
)
from pandas.compile.ir import (
    AddColumn,
    BinOp,
    DType,
    Filter,
    Project,
    ReadTable,
    Sort,
)
from pandas.compile.jit import (
    TraceContext,
    TracedDataFrame,
    TracedSeries,
    Tracer,
    _pandas_type_to_dtype,
    compile,
)

# ---------------------------------------------------------------------------
# _pandas_type_to_dtype
# ---------------------------------------------------------------------------


class TestPandasTypeToDtype:
    @pytest.mark.parametrize(
        "t, expected",
        [
            (bool, DType.BOOL),
            (np.bool_, DType.BOOL),
            ("bool", DType.BOOL),
            ("boolean", DType.BOOL),
            (int, DType.INT64),
            (np.int64, DType.INT64),
            ("int64", DType.INT64),
            (np.int32, DType.INT32),
            ("int32", DType.INT32),
            (np.int16, DType.INT16),
            (np.int8, DType.INT8),
            (np.uint8, DType.UINT8),
            (np.uint16, DType.UINT16),
            (np.uint32, DType.UINT32),
            (np.uint64, DType.UINT64),
            (float, DType.FLOAT64),
            (np.float64, DType.FLOAT64),
            (np.float32, DType.FLOAT32),
            (str, DType.STRING),
            ("string", DType.STRING),
            (object, DType.STRING),
            (bytes, DType.BINARY),
        ],
    )
    def test_mapping(self, t, expected):
        assert _pandas_type_to_dtype(t) is expected

    def test_unknown_returns_string(self):
        assert _pandas_type_to_dtype(complex) is DType.STRING


# ---------------------------------------------------------------------------
# TracedDataFrame — basic IR construction
# ---------------------------------------------------------------------------


class TestTracedDataFrame:
    @pytest.fixture
    def ctx(self):
        return TraceContext(PandasBackend())

    @pytest.fixture
    def sample_df(self):
        return DataFrame(
            {
                "id": [1, 2, 3, 4],
                "region": ["E", "W", "E", "W"],
                "price": [100, 250, 150, 300],
                "quantity": [10, 5, 8, 3],
            }
        )

    @pytest.fixture
    def traced(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("input", sample_df)
        return TracedDataFrame(ctx, ReadTable("input", schema))

    def test_column_access_returns_traced_series(self, traced):
        series = traced["price"]
        assert isinstance(series, TracedSeries)

    def test_column_list_returns_project(self, traced):
        result = traced[["id", "price"]]
        assert isinstance(result, TracedDataFrame)
        assert isinstance(result._ir, Project)

    def test_boolean_filter(self, traced):
        mask = traced["price"] > 100
        result = traced[mask]
        assert isinstance(result, TracedDataFrame)
        assert isinstance(result._ir, Filter)

    def test_setitem_traced_series(self, traced):
        traced["total"] = traced["price"] * traced["quantity"]
        assert isinstance(traced._ir, AddColumn)
        schema = traced._ir.output_schema()
        assert "total" in schema.columns

    def test_setitem_literal(self, traced):
        traced["flag"] = 1
        assert isinstance(traced._ir, AddColumn)

    def test_sort_values(self, traced):
        result = traced.sort_values("price", ascending=False)
        assert isinstance(result._ir, Sort)

    def test_head(self, traced):
        result = traced.head(2)
        schema = result._ir.output_schema()
        assert schema.column_names() == ["id", "region", "price", "quantity"]

    def test_columns_property(self, traced):
        assert list(traced.columns) == ["id", "region", "price", "quantity"]

    def test_dtypes_property(self, traced):
        dtypes = traced.dtypes
        assert isinstance(dtypes, pd.Series)
        assert dtypes["id"] == np.dtype("int64")

    def test_repr(self, traced):
        r = repr(traced)
        assert "TracedDataFrame" in r
        assert "id" in r


# ---------------------------------------------------------------------------
# TracedDataFrame — graph breaks (materialisation)
# ---------------------------------------------------------------------------


class TestTracedDataFrameGraphBreaks:
    @pytest.fixture
    def ctx(self):
        return TraceContext(PandasBackend())

    @pytest.fixture
    def sample_df(self):
        return DataFrame(
            {
                "id": [1, 2, 3],
                "price": [100, 250, 150],
            }
        )

    @pytest.fixture
    def traced(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("input", sample_df)
        return TracedDataFrame(ctx, ReadTable("input", schema))

    def test_len(self, traced):
        assert len(traced) == 3

    def test_shape(self, traced):
        assert traced.shape == (3, 2)

    def test_bool(self, traced):
        assert bool(traced) is True

    def test_empty(self):
        ctx = TraceContext(PandasBackend())
        df = DataFrame({"x": pd.Series([], dtype="int64")})
        ctx.register_table("input", df)
        traced = TracedDataFrame(ctx, ReadTable("input", infer_schema(df)))
        assert traced.empty is True

    def test_iterrows(self, traced):
        rows = list(traced.iterrows())
        assert len(rows) == 3

    def test_iloc(self, traced):
        result = traced.iloc[0]
        assert result["id"] == 1


# ---------------------------------------------------------------------------
# TracedSeries — operators
# ---------------------------------------------------------------------------


class TestTracedSeries:
    @pytest.fixture
    def ctx(self):
        return TraceContext(PandasBackend())

    @pytest.fixture
    def traced_df(self, ctx):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        ctx.register_table("t", df)
        return TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))

    def test_comparison_returns_bool_series(self, traced_df):
        series = traced_df["x"] > 1
        assert isinstance(series, TracedSeries)
        assert series._dtype is DType.BOOL

    def test_arithmetic(self, traced_df):
        series = traced_df["x"] + traced_df["y"]
        assert isinstance(series, TracedSeries)
        assert isinstance(series._expr, BinOp)
        assert series._expr.op == "add"

    def test_and_or(self, traced_df):
        left = traced_df["x"] > 0
        right = traced_df["y"] < 10
        combined = left & right
        assert isinstance(combined, TracedSeries)
        assert combined._dtype is DType.BOOL

    def test_invert(self, traced_df):
        series = ~(traced_df["x"] > 1)
        assert isinstance(series, TracedSeries)
        assert series._dtype is DType.BOOL

    def test_materializing_aggregations(self, ctx, traced_df):
        series = traced_df["x"]
        assert series.sum() == 6
        assert series.min() == 1
        assert series.max() == 3
        assert series.count() == 3


# ---------------------------------------------------------------------------
# @compile decorator
# ---------------------------------------------------------------------------


class TestCompile:
    @pytest.fixture
    def sample_df(self):
        return DataFrame(
            {
                "id": range(1, 5),
                "region": ["E", "W", "E", "W"],
                "price": [100, 250, 150, 300],
            }
        )

    def test_basic_filter(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        result = f(sample_df)
        assert isinstance(result, DataFrame)
        assert len(result) == 3
        assert all(result["price"] > 100)

    def test_graph_break_with_len(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            filtered = df[df["price"] > 100]
            if len(filtered) > 0:
                return filtered
            return df

        result = f(sample_df)
        assert len(result) == 3

    def test_add_column(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            df["double"] = df["price"] * 2
            return df

        result = f(sample_df)
        assert "double" in result.columns
        assert list(result["double"]) == [200, 500, 300, 600]

    def test_explain(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        plan_str = f.explain(sample_df)
        assert "ExecutionPlan" in plan_str
        assert "COMPILED" in plan_str

    def test_cache_hit(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100].groupby("region").sum()

        r1 = f(sample_df)
        r2 = f(sample_df)
        assert len(f._cached_plans) == 1
        # Results should be identical
        assert r1.equals(r2)

    def test_no_args_decorator(self, sample_df):
        @compile
        def f(df):
            return df.head(2)

        result = f(sample_df)
        assert isinstance(result, DataFrame)
        assert len(result) == 2

    def test_with_various_dtypes(self):
        df = DataFrame(
            {
                "i8": np.array([1, 2], dtype="int8"),
                "u16": np.array([3, 4], dtype="uint16"),
                "f32": np.array([1.0, 2.0], dtype="float32"),
            }
        )

        @compile(backend=PandasBackend())
        def f(df):
            return df[["i8", "f32"]]

        result = f(df)
        assert list(result.columns) == ["i8", "f32"]


# ---------------------------------------------------------------------------
# Tracer context manager
# ---------------------------------------------------------------------------


class TestTracer:
    def test_basic(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with Tracer(backend=PandasBackend()) as t:
            traced = t.input(df, "input")
            filtered = traced[traced["x"] > 1]
            t.output(filtered)

        result = t.result()
        assert isinstance(result, DataFrame)
        assert len(result) == 2

    def test_explain(self):
        df = DataFrame({"x": [1, 2, 3]})
        with Tracer(backend=PandasBackend()) as t:
            traced = t.input(df, "input")
            t.output(traced.head(2))

        plan = t.explain()
        assert "CompiledSegment" in plan

    def test_graph_break_in_context(self):
        df = DataFrame({"x": [10, 20, 30]})
        with Tracer(backend=PandasBackend()) as t:
            traced = t.input(df, "input")
            n = len(traced)
            assert n == 3
            t.output(traced.head(n - 1))

        result = t.result()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Top-level compile access
# ---------------------------------------------------------------------------


class TestPdCompile:
    def test_pd_compile_decorator(self):
        @compile
        def f(df):
            return df.head(2)

        result = f(DataFrame({"x": [1, 2, 3]}))
        assert isinstance(result, DataFrame)
        assert len(result) == 2

    def test_pd_compile_with_backend(self):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["x"] > 1]

        result = f(DataFrame({"x": [1, 2, 3]}))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Default backend selection
# ---------------------------------------------------------------------------


class TestDefaultBackendSelection:
    def test_default_backend_is_acero(self):
        pytest.importorskip("pyarrow")
        pytest.importorskip("pyarrow.substrait")
        from pandas.compile.compiler import AceroBackend

        @compile
        def f(df):
            return df[df["price"] > 100]

        df = DataFrame({"price": [100, 250, 150, 300]})
        result = f(df)
        assert isinstance(result, DataFrame)
        assert isinstance(f._backend, AceroBackend)
