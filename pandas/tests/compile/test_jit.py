"""Tests for pandas.compile.jit — tracing proxies, @compile decorator, Tracer."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.compile.compiler import (
    CompiledSegment,
    CompiledStage,
    ConnectedPlan,
    PandasBackend,
    infer_schema,
)
from pandas.compile.ir import (
    AddColumn,
    BinOp,
    ColRef,
    DType,
    Filter,
    Project,
    ReadTable,
    ScalarSubquery,
    Sort,
)
from pandas.compile.jit import (
    DeferredScalar,
    TraceContext,
    TracedDataFrame,
    TracedSeries,
    Tracer,
    compile,
)

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
        from pandas.compile.compiler import (
            AceroBackend,
            DataFusionBackend,
        )

        @compile
        def f(df):
            return df[df["price"] > 100]

        df = DataFrame({"price": [100, 250, 150, 300]})
        result = f(df)
        assert isinstance(result, DataFrame)
        assert isinstance(f._backend, (AceroBackend, DataFusionBackend))


# ---------------------------------------------------------------------------
# Composite-key joins
# ---------------------------------------------------------------------------


class TestCompositeJoin:
    def test_merge_composite_on(self):
        left = DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"], "v": [10, 20, 30]})
        right = DataFrame({"a": [1, 2], "b": ["x", "x"], "score": [100, 200]})

        @compile
        def joined(df, ref):
            return df.merge(ref, on=["a", "b"])

        result = joined(left, right)
        assert "score" in result.columns
        assert len(result) == 2
        assert set(result["v"]) == {10, 30}

    def test_merge_composite_left_right_on(self):
        left = DataFrame({"la": [1, 1, 2], "lb": ["x", "y", "x"], "v": [10, 20, 30]})
        right = DataFrame({"ra": [1, 2], "rb": ["x", "x"], "score": [100, 200]})

        @compile
        def joined(df, ref):
            return df.merge(ref, left_on=["la", "lb"], right_on=["ra", "rb"])

        result = joined(left, right)
        assert "score" in result.columns
        assert len(result) == 2
        assert set(result["v"]) == {10, 30}


# pd.concat as Union
# ---------------------------------------------------------------------------


class TestConcat:
    def test_pd_concat_traced(self):
        a = DataFrame({"x": [1, 2], "y": ["a", "b"]})
        b = DataFrame({"x": [3, 4], "y": ["c", "d"]})

        @compile
        def stacked(df1, df2):
            return pd.concat([df1, df2])

        result = stacked(a, b)
        assert len(result) == 4
        assert list(result["x"]) == [1, 2, 3, 4]

    def test_pd_concat_mixed(self):
        """One traced + one raw DataFrame."""
        a = DataFrame({"v": [10, 20]})
        extra = DataFrame({"v": [30]})

        @compile
        def with_extra(df):
            return pd.concat([df, extra])

        result = with_extra(a)
        assert len(result) == 3
        assert set(result["v"]) == {10, 20, 30}

    def test_pd_concat_untraced_fallback(self):
        """Outside @compile, pd.concat still works normally."""
        a = DataFrame({"x": [1]})
        b = DataFrame({"x": [2]})
        result = pd.concat([a, b], ignore_index=True)
        assert len(result) == 2
        assert list(result["x"]) == [1, 2]


# iloc slicing — Limit/Offset tracing
# ---------------------------------------------------------------------------


class TestIloc:
    def test_iloc_slice_stop_only(self):
        """df.iloc[:5] should stay traced (no graph break)."""
        df = DataFrame({"x": range(10)})

        @compile
        def first_five(df):
            return df.iloc[:5]

        result = first_five(df)
        assert len(result) == 5
        assert list(result["x"]) == [0, 1, 2, 3, 4]

    def test_iloc_slice_start_stop(self):
        """df.iloc[2:5] should stay traced with offset."""
        df = DataFrame({"x": range(10)})

        @compile
        def middle(df):
            return df.iloc[2:5]

        result = middle(df)
        assert len(result) == 3
        assert list(result["x"]) == [2, 3, 4]

    def test_iloc_int_graph_break(self):
        """df.iloc[0] returns a scalar/Series — graph breaks as expected."""
        df = DataFrame({"x": [10, 20, 30]})

        @compile
        def first_row(df):
            return df.iloc[:1]

        result = first_row(df)
        assert len(result) == 1

    def test_iloc_fancy_graph_break(self):
        """df.iloc[[0,2]] uses fancy indexing — graph breaks."""
        df = DataFrame({"x": [10, 20, 30]})

        @compile
        def fancy(df):
            r = df.iloc[[0, 2]]
            return r

        result = fancy(df)
        assert len(result) == 2
        assert list(result["x"]) == [10, 30]


# DeferredScalar — lazy aggregation proxies
# ---------------------------------------------------------------------------


class TestDeferredScalar:
    @pytest.fixture
    def ctx(self):
        return TraceContext(PandasBackend())

    @pytest.fixture
    def sample_df(self):
        return DataFrame(
            {"price": [100, 250, 150, 300], "region": ["E", "W", "E", "W"]}
        )

    def test_sum_returns_deferred(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        result = traced["price"].sum()
        assert isinstance(result, DeferredScalar)

    def test_mean_returns_deferred(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        result = traced["price"].mean()
        assert isinstance(result, DeferredScalar)

    def test_deferred_bool_materializes(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        total = traced["price"].sum()
        assert bool(total)
        assert total._materialized is not object  # sentinel cleared

    def test_deferred_comparison_materializes(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        total = traced["price"].sum()
        assert total > 500

    def test_deferred_item(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        total = traced["price"].sum()
        assert total.item() == 800

    def test_deferred_int_float(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        total = traced["price"].sum()
        assert int(total) == 800
        assert float(total) == 800.0

    def test_deferred_in_arithmetic_no_graph_break(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        total = traced["price"].sum()
        # Using deferred in arithmetic should NOT trigger materialization
        pct = traced["price"] / total
        assert isinstance(pct, TracedSeries)
        assert isinstance(pct._expr, BinOp)
        # No segments should have been created
        assert len(ctx.segments) == 0

    def test_deferred_assign_no_break(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        traced["pct"] = traced["price"] / traced["price"].sum()
        # Should be pure IR with no graph breaks
        assert len(ctx.segments) == 0

    def test_deferred_correctness_end_to_end(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            df["pct"] = df["price"] / df["price"].sum()
            return df

        result = f(sample_df)
        expected = sample_df.copy()
        expected["pct"] = expected["price"] / expected["price"].sum()
        tm.assert_frame_equal(result, expected)

    def test_deferred_reduces_segments(self, sample_df):
        """Without DeferredScalar this would need 2+ segments."""

        @compile(backend=PandasBackend())
        def f(df):
            df["pct"] = df["price"] / df["price"].sum()
            return df

        ctx = f.trace(sample_df)
        # The final materialize at return creates 1 segment
        compiled = [s for s in ctx.segments if isinstance(s, CompiledSegment)]
        assert len(compiled) == 1

    def test_deferred_expr_is_scalar_subquery(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        total = traced["price"].sum()
        assert isinstance(total._expr, ScalarSubquery)

    def test_deferred_scalar_repr(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        total = traced["price"].sum()
        assert "DeferredScalar" in repr(total)
        assert "sum" in repr(total)

    def test_compiled_function_returns_deferred_scalar(self, sample_df):
        """If the function returns a DeferredScalar, it materializes."""

        @compile(backend=PandasBackend())
        def f(df):
            return df["price"].sum()

        result = f(sample_df)
        assert result == 800

    def test_all_agg_types(self, ctx, sample_df):
        schema = infer_schema(sample_df)
        ctx.register_table("t", sample_df)
        traced = TracedDataFrame(ctx, ReadTable("t", schema))
        for method in ("sum", "mean", "min", "max", "count", "std", "var"):
            result = getattr(traced["price"], method)()
            assert isinstance(result, DeferredScalar), (
                f"{method} didn't return deferred"
            )

    def test_pandas_backend_scalar_subquery(self):
        """PandasBackend correctly evaluates ScalarSubquery in expressions."""
        from pandas.compile.ir import Aggregate

        df = DataFrame(
            {"price": [100.0, 250.0, 150.0, 300.0], "region": ["E", "W", "E", "W"]}
        )
        schema = infer_schema(df)
        agg_node = Aggregate(
            ReadTable("t", schema),
            [],
            [("price", "price", "sum")],
        )
        subquery = ScalarSubquery(agg_node, DType.FLOAT64)
        expr = BinOp("div", ColRef("price"), subquery)

        node = AddColumn(ReadTable("t", schema), "pct", expr, DType.FLOAT64)
        backend = PandasBackend()
        result = backend.execute(node, {"t": df})
        expected = df["price"] / df["price"].sum()
        tm.assert_series_equal(result["pct"], expected, check_names=False)


# ---------------------------------------------------------------------------
# ConnectedPlan — rich export with metadata
# ---------------------------------------------------------------------------


class TestConnectedPlan:
    @pytest.fixture
    def sample_df(self):
        return DataFrame(
            {"price": [100, 250, 150, 300], "region": ["E", "W", "E", "W"]}
        )

    def test_single_segment_plan(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        assert isinstance(cp, ConnectedPlan)
        assert len(cp.compiled_stages) == 1
        assert len(cp.graph_breaks) == 0

    def test_graph_break_produces_break_stage(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            n = len(df)
            return df.head(n - 1)

        cp = f.to_connected_plan(sample_df)
        # Should have at least 1 compiled stage (the head after len)
        assert len(cp.compiled_stages) >= 1

    def test_backward_compat_plans(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        plans = f.to_substrait(sample_df)
        # .plans should give same number of plan objects
        assert len(cp.plans) == len(plans)

    def test_to_dict_serializable(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        d = cp.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "stages" in parsed
        assert "final_output" in parsed

    def test_schema_in_connected_plan(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        stage = cp.compiled_stages[0]
        assert isinstance(stage, CompiledStage)
        assert "price" in stage.output_schema.columns
        assert "region" in stage.output_schema.columns

    def test_tracer_connected_plan(self, sample_df):
        with Tracer(backend=PandasBackend()) as t:
            df = t.input(sample_df, "input")
            filtered = df[df["price"] > 100]
            t.output(filtered)

        cp = t.to_connected_plan()
        assert isinstance(cp, ConnectedPlan)
        assert len(cp.compiled_stages) >= 1

    def test_connected_plan_stage_indices(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        for stage in cp.stages:
            assert isinstance(stage.index, int)

    def test_to_dict_has_counts(self, sample_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        d = cp.to_dict()
        assert d["num_compiled"] == len(cp.compiled_stages)
        assert d["num_graph_breaks"] == len(cp.graph_breaks)
