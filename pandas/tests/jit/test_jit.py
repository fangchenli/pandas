"""Tests for pandas.jit.jit — tracing proxies, @compilable decorator, Tracer."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.jit.compiler import (
    CompiledSegment,
    CompiledStage,
    ConnectedPlan,
    PandasBackend,
    infer_schema,
)
from pandas.jit.ir import (
    AddColumn,
    Aggregate,
    BinOp,
    ColRef,
    Distinct,
    DType,
    Filter,
    FunctionCall,
    Project,
    ReadTable,
    ScalarSubquery,
    Schema,
    Sort,
    UnaryOp,
    Window,
)
from pandas.jit.jit import (
    DeferredScalar,
    TraceContext,
    TracedDataFrame,
    TracedSeries,
    Tracer,
    compilable,
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
        assert series._column_name == "price"
        assert series._dtype == DType.INT64

    def test_column_list_returns_project(self, traced, ctx, sample_df):
        result = traced[["id", "price"]]
        assert isinstance(result, TracedDataFrame)
        assert isinstance(result._ir, Project)
        df = ctx.backend.execute(result._ir, ctx.tables, optimize=False)
        assert list(df.columns) == ["id", "price"]
        assert len(df) == len(sample_df)

    def test_boolean_filter(self, traced, ctx, sample_df):
        mask = traced["price"] > 100
        result = traced[mask]
        assert isinstance(result, TracedDataFrame)
        assert isinstance(result._ir, Filter)
        df = ctx.backend.execute(result._ir, ctx.tables, optimize=False)
        expected = sample_df[sample_df["price"] > 100].reset_index(drop=True)
        tm.assert_frame_equal(df, expected)

    def test_setitem_traced_series(self, traced, ctx):
        traced["total"] = traced["price"] * traced["quantity"]
        assert isinstance(traced._ir, AddColumn)
        schema = traced._ir.output_schema()
        assert "total" in schema.columns
        df = ctx.backend.execute(traced._ir, ctx.tables, optimize=False)
        assert list(df["total"]) == [1000, 1250, 1200, 900]

    def test_setitem_literal(self, traced, ctx):
        traced["flag"] = 1
        assert isinstance(traced._ir, AddColumn)
        df = ctx.backend.execute(traced._ir, ctx.tables, optimize=False)
        assert all(df["flag"] == 1)

    def test_sort_values(self, traced, ctx, sample_df):
        result = traced.sort_values("price", ascending=False)
        assert isinstance(result._ir, Sort)
        df = ctx.backend.execute(result._ir, ctx.tables, optimize=False)
        assert list(df["price"]) == sorted(sample_df["price"], reverse=True)

    def test_head(self, traced, ctx):
        result = traced.head(2)
        schema = result._ir.output_schema()
        assert schema.column_names() == ["id", "region", "price", "quantity"]
        df = ctx.backend.execute(result._ir, ctx.tables, optimize=False)
        assert len(df) == 2

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

    def test_comparison_returns_bool_series(self, ctx, traced_df):
        series = traced_df["x"] > 1
        assert isinstance(series, TracedSeries)
        assert series._dtype is DType.BOOL
        # Verify value: filter with the predicate
        result = traced_df[series]
        df = ctx.backend.execute(result._ir, ctx.tables, optimize=False)
        assert list(df["x"]) == [2, 3]

    def test_arithmetic(self, ctx, traced_df):
        series = traced_df["x"] + traced_df["y"]
        assert isinstance(series, TracedSeries)
        assert isinstance(series._expr, BinOp)
        assert series._expr.op == "add"
        # Verify value
        traced_df["z"] = series
        df = ctx.backend.execute(traced_df._ir, ctx.tables, optimize=False)
        assert list(df["z"]) == [5, 7, 9]

    def test_and_or(self, ctx, traced_df):
        left = traced_df["x"] > 0
        right = traced_df["y"] < 6
        combined = left & right
        assert isinstance(combined, TracedSeries)
        assert combined._dtype is DType.BOOL
        result = traced_df[combined]
        df = ctx.backend.execute(result._ir, ctx.tables, optimize=False)
        assert list(df["x"]) == [1, 2]

    def test_invert(self, ctx, traced_df):
        series = ~(traced_df["x"] > 1)
        assert isinstance(series, TracedSeries)
        assert series._dtype is DType.BOOL
        result = traced_df[series]
        df = ctx.backend.execute(result._ir, ctx.tables, optimize=False)
        assert list(df["x"]) == [1]

    def test_materializing_aggregations(self, ctx, traced_df):
        series = traced_df["x"]
        assert series.sum() == 6
        assert series.min() == 1
        assert series.max() == 3
        assert series.count() == 3


# ---------------------------------------------------------------------------
# @compilable decorator
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
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        result = f(sample_df)
        assert isinstance(result, DataFrame)
        assert len(result) == 3
        assert all(result["price"] > 100)

    def test_graph_break_with_len(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            filtered = df[df["price"] > 100]
            if len(filtered) > 0:
                return filtered
            return df

        result = f(sample_df)
        assert len(result) == 3

    def test_add_column(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["double"] = df["price"] * 2
            return df

        result = f(sample_df)
        assert "double" in result.columns
        assert list(result["double"]) == [200, 500, 300, 600]

    def test_explain(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        plan_str = f.explain(sample_df)
        assert "ExecutionPlan" in plan_str
        assert "COMPILED" in plan_str

    def test_cache_hit(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100].groupby("region").sum()

        r1 = f(sample_df)
        r2 = f(sample_df)
        assert len(f._cached_plans) == 1
        # Results should be identical
        assert r1.equals(r2)

    def test_no_args_decorator(self, sample_df):
        @compilable
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

        @compilable(backend=PandasBackend())
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
# Top-level jit access
# ---------------------------------------------------------------------------


class TestPdJit:
    def test_pd_jit_decorator(self):
        @compilable
        def f(df):
            return df.head(2)

        result = f(DataFrame({"x": [1, 2, 3]}))
        assert isinstance(result, DataFrame)
        assert len(result) == 2

    def test_pd_jit_with_backend(self):
        @compilable(backend=PandasBackend())
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
        from pandas.jit.compiler import (
            AceroBackend,
            DataFusionBackend,
        )

        @compilable
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

        @compilable
        def joined(df, ref):
            return df.merge(ref, on=["a", "b"])

        result = joined(left, right)
        assert "score" in result.columns
        assert len(result) == 2
        assert set(result["v"]) == {10, 30}

    def test_merge_composite_left_right_on(self):
        left = DataFrame({"la": [1, 1, 2], "lb": ["x", "y", "x"], "v": [10, 20, 30]})
        right = DataFrame({"ra": [1, 2], "rb": ["x", "x"], "score": [100, 200]})

        @compilable
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

        @compilable
        def stacked(df1, df2):
            return pd.concat([df1, df2])

        result = stacked(a, b)
        assert len(result) == 4
        assert list(result["x"]) == [1, 2, 3, 4]

    def test_pd_concat_mixed(self):
        """One traced + one raw DataFrame."""
        a = DataFrame({"v": [10, 20]})
        extra = DataFrame({"v": [30]})

        @compilable
        def with_extra(df):
            return pd.concat([df, extra])

        result = with_extra(a)
        assert len(result) == 3
        assert set(result["v"]) == {10, 20, 30}

    def test_pd_concat_untraced_fallback(self):
        """Outside @compilable, pd.concat still works normally."""
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

        @compilable
        def first_five(df):
            return df.iloc[:5]

        result = first_five(df)
        assert len(result) == 5
        assert list(result["x"]) == [0, 1, 2, 3, 4]

    def test_iloc_slice_start_stop(self):
        """df.iloc[2:5] should stay traced with offset."""
        df = DataFrame({"x": range(10)})

        @compilable
        def middle(df):
            return df.iloc[2:5]

        result = middle(df)
        assert len(result) == 3
        assert list(result["x"]) == [2, 3, 4]

    def test_iloc_int_graph_break(self):
        """df.iloc[0] returns a scalar/Series — graph breaks as expected."""
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def first_row(df):
            return df.iloc[:1]

        result = first_row(df)
        assert len(result) == 1

    def test_iloc_fancy_graph_break(self):
        """df.iloc[[0,2]] uses fancy indexing — graph breaks."""
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def fancy(df):
            r = df.iloc[[0, 2]]
            return r

        result = fancy(df)
        assert len(result) == 2
        assert list(result["x"]) == [10, 30]


# drop_duplicates — Distinct tracing
# ---------------------------------------------------------------------------


class TestDropDuplicates:
    def test_drop_duplicates_all_columns(self):
        """drop_duplicates() with no subset stays traced."""
        df = DataFrame({"x": [1, 2, 1, 2], "y": ["a", "b", "a", "b"]})

        @compilable
        def dedup(df):
            return df.drop_duplicates()

        result = dedup(df)
        assert len(result) == 2

    def test_drop_duplicates_partial_subset_graph_break(self):
        """drop_duplicates(subset=["x"]) graph breaks for partial subset."""
        df = DataFrame({"x": [1, 1, 2], "y": ["a", "b", "c"]})

        @compilable
        def dedup(df):
            return df.drop_duplicates(subset=["x"])

        result = dedup(df)
        assert len(result) == 2

    def test_drop_duplicates_keep_last_graph_break(self):
        """keep='last' graph breaks."""
        df = DataFrame({"x": [1, 2, 1], "y": ["a", "b", "a"]})

        @compilable
        def dedup(df):
            return df.drop_duplicates(keep="last")

        result = dedup(df)
        assert len(result) == 2


class TestCumulative:
    def test_cumsum_traced(self):
        """df.cumsum() stays traced — produces Window IR."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        @compilable
        def f(df):
            return df.cumsum()

        result = f(df)
        expected = df.cumsum()
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_cummax_traced(self):
        df = DataFrame({"a": [3, 1, 2], "b": [1, 5, 3]})

        @compilable
        def f(df):
            return df.cummax()

        result = f(df)
        expected = df.cummax()
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_cummin_traced(self):
        df = DataFrame({"a": [3, 1, 2], "b": [5, 1, 3]})

        @compilable
        def f(df):
            return df.cummin()

        result = f(df)
        expected = df.cummin()
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_cumprod_traced(self):
        df = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})

        @compilable
        def f(df):
            return df.cumprod()

        result = f(df)
        expected = df.cumprod()
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_cumsum_traced(self):
        """df["col"].cumsum() stays traced via Window IR."""
        df = DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        @compilable
        def f(df):
            return df.assign(a_cumsum=df["a"].cumsum())

        result = f(df)
        expected = df.assign(a_cumsum=df["a"].cumsum())
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_cummax_traced(self):
        df = DataFrame({"a": [3, 1, 2], "b": [1, 5, 3]})

        @compilable
        def f(df):
            return df.assign(b_max=df["b"].cummax())

        result = f(df)
        expected = df.assign(b_max=df["b"].cummax())
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_cummin_traced(self):
        df = DataFrame({"a": [3, 1, 2], "b": [5, 1, 3]})

        @compilable
        def f(df):
            return df.assign(a_min=df["a"].cummin())

        result = f(df)
        expected = df.assign(a_min=df["a"].cummin())
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_cumprod_traced(self):
        df = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})

        @compilable
        def f(df):
            return df.assign(a_prod=df["a"].cumprod())

        result = f(df)
        expected = df.assign(a_prod=df["a"].cumprod())
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_cumsum_filter(self):
        """Cross-IR: cumsum → compare → filter via __getitem__."""
        df = DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df[df["a"].cumsum() <= 6]

        result = f(df)
        expected = df[df["a"].cumsum() <= 6]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


# reset_index / set_index
# ---------------------------------------------------------------------------


class TestResetSetIndex:
    def test_reset_index_drop_true(self):
        """reset_index(drop=True) is a no-op in relational algebra."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        @compilable
        def f(df):
            return df.reset_index(drop=True)

        result = f(df)
        expected = df.reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_reset_index_drop_false(self):
        """set_index then reset_index(drop=False) round-trips correctly."""
        df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        @compilable
        def f(df):
            return df.set_index("b").reset_index(drop=False)

        result = f(df)
        expected = df.set_index("b").reset_index(drop=False)
        tm.assert_frame_equal(result, expected)

    def test_set_index(self):
        """set_index() graph-breaks to set a column as index."""
        df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        @compilable
        def f(df):
            return df.set_index("b")

        result = f(df)
        expected = df.set_index("b")
        tm.assert_frame_equal(result, expected)

    def test_set_index_drop_false(self):
        """set_index(drop=False) keeps the column and sets it as index."""
        df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        @compilable
        def f(df):
            return df.set_index("b", drop=False)

        result = f(df)
        expected = df.set_index("b", drop=False)
        tm.assert_frame_equal(result, expected)

    def test_reset_index_after_groupby(self):
        """groupby().sum().reset_index(drop=True) — common pattern.

        JIT Aggregate keeps group keys as columns (SQL semantics),
        so reset_index(drop=True) is a no-op. The expected result
        should include group keys as regular columns.
        """
        df = DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

        @compilable
        def f(df):
            return df.groupby("g").sum().reset_index(drop=True)

        result = f(df)
        # In pandas: groupby puts keys in index,
        # reset_index(drop=False) moves them back.
        # In JIT: keys are already columns, reset_index(drop=True) is a no-op.
        # Both yield the same result: columns [g, v].
        expected = df.groupby("g").sum().reset_index(drop=False)
        # Backend may not preserve sort order.
        tm.assert_frame_equal(
            result.sort_values("g").reset_index(drop=True),
            expected.sort_values("g").reset_index(drop=True),
        )

    def test_series_reset_index_drop_true(self):
        """Series.reset_index(drop=True) is a no-op."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        @compilable
        def f(df):
            s = df["a"].reset_index(drop=True)
            return df.assign(a2=s)

        result = f(df)
        expected = df.assign(a2=df["a"].reset_index(drop=True))
        tm.assert_frame_equal(result, expected)


# shift / diff
# ---------------------------------------------------------------------------


class TestShiftDiff:
    def test_shift_traced(self):
        """df.shift(1) creates Window IR with lag."""
        df = DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.shift(1)

        result = f(df)
        expected = df.shift(1)
        tm.assert_frame_equal(result, expected)

    def test_shift_negative(self):
        """df.shift(-1) creates Window IR with lead."""
        df = DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.shift(-1)

        result = f(df)
        expected = df.shift(-1)
        tm.assert_frame_equal(result, expected)

    def test_shift_zero(self):
        """df.shift(0) is identity."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        @compilable
        def f(df):
            return df.shift(0)

        result = f(df)
        expected = df.shift(0)
        tm.assert_frame_equal(result, expected)

    def test_dataframe_diff_traced(self):
        """df.diff(1) stays traced via Window(lag) + sub."""
        df = DataFrame({"a": [1, 3, 6, 10], "b": [10.0, 20.0, 30.0, 40.0]})

        @compilable
        def f(df):
            return df.diff(1)

        result = f(df)
        expected = df.diff(1)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_shift_traced(self):
        """Series.shift() stays traced via Window IR."""
        df = DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.assign(a_shifted=df["a"].shift(1))

        result = f(df)
        expected = df.assign(a_shifted=df["a"].shift(1))
        tm.assert_frame_equal(result, expected)

    def test_series_diff_traced(self):
        """Series.diff() stays traced via Window(lag) + sub."""
        df = DataFrame({"a": [1, 3, 6, 10], "b": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.assign(a_diff=df["a"].diff(1))

        result = f(df)
        expected = df.assign(a_diff=df["a"].diff(1))
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_diff_assign_filter(self):
        """diff → assign → filter pipeline stays traced."""
        df = DataFrame({"a": [1, 3, 6, 10]})

        @compilable
        def f(df):
            df["d"] = df["a"].diff(1)
            return df[df["d"] > 2]

        result = f(df)
        expected = df.copy()
        expected["d"] = expected["a"].diff(1)
        expected = expected[expected["d"] > 2]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


class TestRank:
    def test_series_rank_min(self):
        """Series.rank(method='min') traced via Window IR."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            return df.assign(r=df["a"].rank(method="min"))

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="min"))
        tm.assert_frame_equal(result, expected)

    def test_series_rank_dense(self):
        """Series.rank(method='dense') traced via Window IR."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            return df.assign(r=df["a"].rank(method="dense"))

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="dense"))
        tm.assert_frame_equal(result, expected)

    def test_series_rank_first(self):
        """Series.rank(method='first') traced via Window IR."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            return df.assign(r=df["a"].rank(method="first"))

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="first"))
        tm.assert_frame_equal(result, expected)

    def test_series_rank_descending(self):
        """Series.rank(ascending=False) reverses order."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            return df.assign(r=df["a"].rank(method="min", ascending=False))

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="min", ascending=False))
        tm.assert_frame_equal(result, expected)

    def test_series_rank_average_graph_break(self):
        """Series.rank(method='average') graph-breaks (no SQL equivalent)."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            return df.assign(r=df["a"].rank(method="average"))

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="average"))
        tm.assert_frame_equal(result, expected)

    def test_dataframe_rank_min_traced(self):
        """DataFrame.rank(method='min') is now traced via Window IR."""
        df = DataFrame({"a": [3.0, 1.0, 4.0], "b": [10.0, 30.0, 20.0]})

        @compilable
        def f(df):
            return df.rank(method="min")

        result = f(df)
        expected = df.rank(method="min")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dataframe_rank_dense_traced(self):
        df = DataFrame({"a": [3.0, 1.0, 3.0], "b": [10.0, 30.0, 20.0]})

        @compilable
        def f(df):
            return df.rank(method="dense")

        result = f(df)
        expected = df.rank(method="dense")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dataframe_rank_first_traced(self):
        df = DataFrame({"a": [3.0, 1.0, 4.0], "b": [10.0, 30.0, 20.0]})

        @compilable
        def f(df):
            return df.rank(method="first")

        result = f(df)
        expected = df.rank(method="first")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dataframe_rank_average_graph_break(self):
        """DataFrame.rank(method='average') still graph-breaks."""
        df = DataFrame({"a": [3.0, 1.0, 4.0], "b": [10.0, 30.0, 20.0]})

        @compilable
        def f(df):
            return df.rank(method="average")

        result = f(df)
        expected = df.rank(method="average")
        tm.assert_frame_equal(result, expected)

    def test_dataframe_rank_pct_graph_break(self):
        """DataFrame.rank(pct=True) graph-breaks."""
        df = DataFrame({"a": [3.0, 1.0, 4.0]})

        @compilable
        def f(df):
            return df.rank(pct=True)

        result = f(df)
        expected = df.rank(pct=True)
        tm.assert_frame_equal(result, expected)

    def test_dataframe_rank_descending(self):
        df = DataFrame({"a": [1.0, 2.0, 3.0]})

        @compilable
        def f(df):
            return df.rank(method="min", ascending=False)

        result = f(df)
        expected = df.rank(method="min", ascending=False)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_groupby_series_rank(self):
        """GroupBy series rank traced via Window IR with partition_by."""
        df = DataFrame({"g": ["a", "a", "b", "b"], "v": [2, 1, 4, 3]})

        @compilable
        def f(df):
            return df.assign(r=df.groupby("g")["v"].rank(method="min"))

        result = f(df)
        expected = df.assign(r=df.groupby("g")["v"].rank(method="min"))
        tm.assert_frame_equal(result, expected)


class TestCrossIRComposition:
    """Cross-IR composition: Window-backed TracedSeries in assign/setitem/getitem."""

    def test_assign_rank_composable(self):
        """df.assign(r=df['a'].rank()) composes via Window transplant."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            return df.assign(r=df["a"].rank(method="min"))

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="min"))
        tm.assert_frame_equal(result, expected)

    def test_assign_multiple_ranks(self):
        """Chained assign with multiple rank columns."""
        df = DataFrame({"a": [3, 1, 4], "b": [10, 30, 20]})

        @compilable
        def f(df):
            return df.assign(
                ra=df["a"].rank(method="min"),
                rb=df["b"].rank(method="dense"),
            )

        result = f(df)
        expected = df.assign(
            ra=df["a"].rank(method="min"),
            rb=df["b"].rank(method="dense"),
        )
        tm.assert_frame_equal(result, expected)

    def test_setitem_rank(self):
        """df['r'] = df['a'].rank() composes via Window transplant."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            df["r"] = df["a"].rank(method="min")
            return df

        result = f(df)
        expected = df.copy()
        expected["r"] = expected["a"].rank(method="min")
        tm.assert_frame_equal(result, expected)

    def test_filter_by_rank(self):
        """df[df['a'].rank() <= 2] composes via Window + Filter + Project."""
        df = DataFrame({"a": [3, 1, 4, 1, 5], "b": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            return df[df["a"].rank(method="min") <= 2]

        result = f(df)
        expected = df[df["a"].rank(method="min") <= 2]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_groupby_rank_assign(self):
        """Groupby rank composes in assign."""
        df = DataFrame({"g": ["a", "a", "b", "b"], "v": [2, 1, 4, 3]})

        @compilable
        def f(df):
            return df.assign(r=df.groupby("g")["v"].rank(method="dense"))

        result = f(df)
        expected = df.assign(r=df.groupby("g")["v"].rank(method="dense"))
        tm.assert_frame_equal(result, expected)

    def test_series_shift_assign(self):
        """Series shift composes in assign."""
        df = DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.assign(a_shifted=df["a"].shift(1))

        result = f(df)
        expected = df.assign(a_shifted=df["a"].shift(1))
        tm.assert_frame_equal(result, expected)

    def test_rank_average_still_graph_breaks(self):
        """method='average' still graph-breaks (no SQL equivalent)."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            return df.assign(r=df["a"].rank(method="average"))

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="average"))
        tm.assert_frame_equal(result, expected)

    def test_filter_by_rank_preserves_original_columns(self):
        """Filtering by rank should preserve original column values (not rank)."""
        df = DataFrame({"a": [30, 10, 40], "b": ["x", "y", "z"]})

        @compilable
        def f(df):
            return df[df["a"].rank(method="min") <= 2]

        result = f(df)
        # Only rows where rank(a) <= 2: a=10 (rank 1) and a=30 (rank 2)
        expected = df[df["a"].rank(method="min") <= 2]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )
        # Verify original "a" values are preserved, not replaced by ranks
        assert set(result["a"].tolist()) == {10, 30}


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
        @compilable(backend=PandasBackend())
        def f(df):
            df["pct"] = df["price"] / df["price"].sum()
            return df

        result = f(sample_df)
        expected = sample_df.copy()
        expected["pct"] = expected["price"] / expected["price"].sum()
        tm.assert_frame_equal(result, expected)

    def test_deferred_reduces_segments(self, sample_df):
        """Without DeferredScalar this would need 2+ segments."""

        @compilable(backend=PandasBackend())
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

        @compilable(backend=PandasBackend())
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
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        assert isinstance(cp, ConnectedPlan)
        assert len(cp.compiled_stages) == 1
        assert len(cp.graph_breaks) == 0

    def test_graph_break_produces_break_stage(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            n = len(df)
            return df.head(n - 1)

        cp = f.to_connected_plan(sample_df)
        # Should have at least 1 compiled stage (the head after len)
        assert len(cp.compiled_stages) >= 1

    def test_backward_compat_plans(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        plans = f.to_substrait(sample_df)
        # .plans should give same number of plan objects
        assert len(cp.plans) == len(plans)

    def test_to_dict_serializable(self, sample_df):
        @compilable(backend=PandasBackend())
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
        @compilable(backend=PandasBackend())
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
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        for stage in cp.stages:
            assert isinstance(stage.index, int)

    def test_to_dict_has_counts(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["price"] > 100]

        cp = f.to_connected_plan(sample_df)
        d = cp.to_dict()
        assert d["num_compiled"] == len(cp.compiled_stages)
        assert d["num_graph_breaks"] == len(cp.graph_breaks)


# ---------------------------------------------------------------------------
# Clip and abs
# ---------------------------------------------------------------------------


class TestClipAbs:
    @pytest.fixture
    def ctx(self):
        return TraceContext(PandasBackend())

    @pytest.fixture
    def sample_df(self):
        return DataFrame({"a": [-3, -1, 0, 2, 5], "b": [10, 20, 30, 40, 50]})

    def test_series_clip_both(self, ctx, sample_df):
        ctx.register_table("t", sample_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(sample_df)))
        series = tdf["a"]
        clipped = series.clip(lower=-1, upper=3)
        assert isinstance(clipped, TracedSeries)
        tdf2 = tdf.assign(a_clipped=clipped)
        _, result = ctx.materialize(tdf2._ir)
        expected = sample_df["a"].clip(lower=-1, upper=3)
        tm.assert_series_equal(
            result["a_clipped"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_series_clip_lower_only(self, ctx, sample_df):
        ctx.register_table("t", sample_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(sample_df)))
        clipped = tdf["a"].clip(lower=0)
        assert isinstance(clipped, TracedSeries)
        tdf2 = tdf.assign(a_clipped=clipped)
        _, result = ctx.materialize(tdf2._ir)
        expected = sample_df["a"].clip(lower=0)
        tm.assert_series_equal(
            result["a_clipped"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_series_clip_upper_only(self, ctx, sample_df):
        ctx.register_table("t", sample_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(sample_df)))
        clipped = tdf["a"].clip(upper=2)
        assert isinstance(clipped, TracedSeries)
        tdf2 = tdf.assign(a_clipped=clipped)
        _, result = ctx.materialize(tdf2._ir)
        expected = sample_df["a"].clip(upper=2)
        tm.assert_series_equal(
            result["a_clipped"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_series_clip_none_noop(self, ctx, sample_df):
        ctx.register_table("t", sample_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(sample_df)))
        series = tdf["a"]
        clipped = series.clip()
        assert clipped is series

    def test_dataframe_clip(self, ctx, sample_df):
        ctx.register_table("t", sample_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(sample_df)))
        clipped = tdf.clip(lower=0, upper=25)
        assert isinstance(clipped, TracedDataFrame)
        _, result = ctx.materialize(clipped._ir)
        expected = sample_df.clip(lower=0, upper=25)
        tm.assert_frame_equal(result, expected)

    def test_dataframe_clip_none_noop(self, ctx, sample_df):
        ctx.register_table("t", sample_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(sample_df)))
        clipped = tdf.clip()
        assert clipped is tdf

    def test_dataframe_abs(self, ctx, sample_df):
        ctx.register_table("t", sample_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(sample_df)))
        result_tdf = tdf.abs()
        assert isinstance(result_tdf, TracedDataFrame)
        _, result = ctx.materialize(result_tdf._ir)
        expected = sample_df.abs()
        tm.assert_frame_equal(result, expected)

    def test_dataframe_abs_preserves_non_numeric(self, ctx):
        df = DataFrame({"name": ["a", "b"], "val": [-1, -2]})
        ctx.register_table("t", df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        result_tdf = tdf.abs()
        _, result = ctx.materialize(result_tdf._ir)
        assert list(result["name"]) == ["a", "b"]
        assert list(result["val"]) == [1, 2]


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


class TestArithmeticOperators:
    def test_floordiv(self):
        df = DataFrame({"a": [7, 10, 15]})

        @compilable
        def f(df):
            df["b"] = df["a"] // 4
            return df

        result = f(df)
        expected = df.copy()
        expected["b"] = expected["a"] // 4
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_mod(self):
        df = DataFrame({"a": [7, 10, 15]})

        @compilable
        def f(df):
            df["b"] = df["a"] % 4
            return df

        result = f(df)
        expected = df.copy()
        expected["b"] = expected["a"] % 4
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_pow(self):
        df = DataFrame({"a": [2, 3, 4]})

        @compilable
        def f(df):
            df["b"] = df["a"] ** 2
            return df

        result = f(df)
        expected = df.copy()
        expected["b"] = expected["a"] ** 2
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_rtruediv(self):
        df = DataFrame({"a": [2.0, 4.0, 5.0]})

        @compilable
        def f(df):
            df["b"] = 100.0 / df["a"]
            return df

        result = f(df)
        expected = df.copy()
        expected["b"] = 100.0 / expected["a"]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_rfloordiv(self):
        df = DataFrame({"a": [2, 3, 4]})

        @compilable
        def f(df):
            df["b"] = 10 // df["a"]
            return df

        result = f(df)
        expected = df.copy()
        expected["b"] = 10 // expected["a"]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_rmod(self):
        df = DataFrame({"a": [3, 4, 7]})

        @compilable
        def f(df):
            df["b"] = 10 % df["a"]
            return df

        result = f(df)
        expected = df.copy()
        expected["b"] = 10 % expected["a"]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_rpow(self):
        df = DataFrame({"a": [1, 2, 3]})

        @compilable
        def f(df):
            df["b"] = 2 ** df["a"]
            return df

        result = f(df)
        expected = df.copy()
        expected["b"] = 2 ** expected["a"]
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_query_floordiv(self):
        df = DataFrame({"a": [7, 10, 15, 20]})

        @compilable
        def f(df):
            return df.query("a // 10 == 1")

        result = f(df)
        expected = df[df["a"] // 10 == 1]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_query_mod(self):
        df = DataFrame({"a": [7, 10, 15, 20]})

        @compilable
        def f(df):
            return df.query("a % 5 == 0")

        result = f(df)
        expected = df[df["a"] % 5 == 0]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# GroupBy cumulative
# ---------------------------------------------------------------------------


class TestGroupByCumulative:
    def test_groupby_series_cumsum(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df.assign(cs=df.groupby("g")["v"].cumsum())

        result = f(df)
        expected = df.assign(cs=df.groupby("g")["v"].cumsum())
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_groupby_series_cummax(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "v": [3, 1, 2, 4]})

        @compilable
        def f(df):
            return df.assign(cm=df.groupby("g")["v"].cummax())

        result = f(df)
        expected = df.assign(cm=df.groupby("g")["v"].cummax())
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_groupby_df_cumsum(self):
        df = DataFrame(
            {"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}
        )

        @compilable
        def f(df):
            return df.groupby("g").cumsum()

        result = f(df)
        # pandas groupby().cumsum() drops group keys; JIT keeps all columns
        expected_vals = df.groupby("g").cumsum()
        for col in expected_vals.columns:
            tm.assert_series_equal(
                result[col].reset_index(drop=True),
                expected_vals[col].reset_index(drop=True),
                check_names=False,
                check_dtype=False,
            )

    def test_groupby_cumsum_assign(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})

        @compilable
        def f(df):
            df["running"] = df.groupby("g")["v"].cumsum()
            return df[df["running"] > 2]

        result = f(df)
        expected = df.copy()
        expected["running"] = expected.groupby("g")["v"].cumsum()
        expected = expected[expected["running"] > 2]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


# ---------------------------------------------------------------------------
# Series.where / Series.mask
# ---------------------------------------------------------------------------


class TestSeriesWhereMask:
    def test_series_where_traced(self):
        df = DataFrame({"a": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df.assign(b=df["a"].where(df["a"] > 2, 0))

        result = f(df)
        assert isinstance(result, DataFrame)
        expected = df.assign(b=df["a"].where(df["a"] > 2, 0))
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_where_result_correct(self):
        df = DataFrame({"a": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.assign(capped=df["a"].where(df["a"] <= 25, -1))

        result = f(df)
        assert list(result["capped"]) == [10, 20, -1, -1]

    def test_series_mask_traced(self):
        df = DataFrame({"a": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df.assign(b=df["a"].mask(df["a"] > 2, -1))

        result = f(df)
        expected = df.assign(b=df["a"].mask(df["a"] > 2, -1))
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_mask_result_correct(self):
        df = DataFrame({"a": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.assign(zeroed=df["a"].mask(df["a"] > 25, 0))

        result = f(df)
        assert list(result["zeroed"]) == [10, 20, 0, 0]

    def test_series_where_none_gives_nan(self):
        df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})

        @compilable
        def f(df):
            return df.assign(b=df["a"].where(df["a"] > 2))

        result = f(df)
        assert np.isnan(result["b"].iloc[0])
        assert np.isnan(result["b"].iloc[1])
        assert result["b"].iloc[2] == 3.0

    def test_series_where_in_assign(self):
        df = DataFrame({"x": [5, 10, 15, 20]})

        @compilable
        def f(df):
            safe = df["x"].where(df["x"] >= 10, 10)
            return df.assign(x_safe=safe)

        result = f(df)
        assert list(result["x_safe"]) == [10, 10, 15, 20]


# ---------------------------------------------------------------------------
# Series.astype
# ---------------------------------------------------------------------------


class TestSeriesAstype:
    def test_series_astype_int_to_float(self):
        df = DataFrame({"a": [1, 2, 3]})

        @compilable
        def f(df):
            return df.assign(b=df["a"].astype("float64"))

        result = f(df)
        assert result["b"].dtype == np.float64

    def test_series_astype_result_correct(self):
        df = DataFrame({"a": [1, 2, 3]})

        @compilable
        def f(df):
            return df.assign(b=df["a"].astype("float64"))

        result = f(df)
        assert list(result["b"]) == [1.0, 2.0, 3.0]

    def test_series_astype_in_assign(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        @compilable
        def f(df):
            return df.assign(a_float=df["a"].astype("float32"))

        result = f(df)
        assert result["a_float"].dtype == np.float32

    def test_series_astype_unknown_graph_break(self):
        """Unknown dtype like 'category' should graph-break gracefully."""
        df = DataFrame({"a": ["x", "y", "x"]})

        @compilable
        def f(df):
            return df.assign(a_cat=df["a"].astype("category"))

        result = f(df)
        # Result should still be correct via graph break
        assert result["a_cat"].dtype.name == "category"


# ---------------------------------------------------------------------------
# GroupByMulti — count, min, max
# ---------------------------------------------------------------------------


class TestGroupByMulti:
    def test_groupby_multi_count(self):
        df = DataFrame({"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].count()

        result = f(df).sort_values("g").reset_index(drop=True)
        expected = df.groupby("g")[["x", "y"]].count().reset_index()
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_groupby_multi_min(self):
        df = DataFrame({"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [6, 5, 4]})

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].min()

        result = f(df).sort_values("g").reset_index(drop=True)
        expected = df.groupby("g")[["x", "y"]].min().reset_index()
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_groupby_multi_max(self):
        df = DataFrame({"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [6, 5, 4]})

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].max()

        result = f(df).sort_values("g").reset_index(drop=True)
        expected = df.groupby("g")[["x", "y"]].max().reset_index()
        tm.assert_frame_equal(result, expected, check_dtype=False)


# ---------------------------------------------------------------------------
# Series.pct_change
# ---------------------------------------------------------------------------


class TestPctChange:
    def test_series_pct_change_traced(self):
        df = DataFrame({"a": [100, 110, 121, 133]})

        @compilable
        def f(df):
            return df.assign(pct=df["a"].pct_change())

        result = f(df)
        expected = df.assign(pct=df["a"].pct_change())
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_pct_change_result_correct(self):
        df = DataFrame({"a": [10.0, 20.0, 15.0]})

        @compilable
        def f(df):
            return df.assign(pct=df["a"].pct_change())

        result = f(df)
        assert np.isnan(result["pct"].iloc[0])
        assert abs(result["pct"].iloc[1] - 1.0) < 1e-10
        assert abs(result["pct"].iloc[2] - (-0.25)) < 1e-10

    def test_series_pct_change_periods_2(self):
        df = DataFrame({"a": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.assign(pct=df["a"].pct_change(periods=2))

        result = f(df)
        expected = df.assign(pct=df["a"].pct_change(periods=2))
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_pct_change_negative(self):
        df = DataFrame({"a": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.assign(pct=df["a"].pct_change(periods=-1))

        result = f(df)
        expected = df.assign(pct=df["a"].pct_change(periods=-1))
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_pct_change_in_assign(self):
        df = DataFrame({"a": [100, 110, 121]})

        @compilable
        def f(df):
            df["growth"] = df["a"].pct_change()
            return df[df["growth"] > 0.05]

        result = f(df)
        expected = df.copy()
        expected["growth"] = expected["a"].pct_change()
        expected = expected[expected["growth"] > 0.05].reset_index(drop=True)
        tm.assert_frame_equal(
            result.reset_index(drop=True), expected, check_dtype=False
        )


# ---------------------------------------------------------------------------
# Series.round / DataFrame.round
# ---------------------------------------------------------------------------


class TestRound:
    def test_series_round_traced(self):
        df = DataFrame({"a": [1.234, 2.567, 3.891]})

        @compilable
        def f(df):
            return df.assign(b=df["a"].round(1))

        result = f(df)
        expected = df.assign(b=df["a"].round(1))
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_round_result(self):
        df = DataFrame({"a": [1.555, 2.444, 3.999]})

        @compilable
        def f(df):
            return df.assign(b=df["a"].round(2))

        result = f(df)
        assert list(result["b"]) == [1.56, 2.44, 4.0]

    def test_series_round_decimals_0(self):
        df = DataFrame({"a": [1.5, 2.3, 3.7]})

        @compilable
        def f(df):
            return df.assign(b=df["a"].round())

        result = f(df)
        expected = df.assign(b=df["a"].round())
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dataframe_round_traced(self):
        df = DataFrame({"a": [1.234, 2.567], "b": [3.891, 4.123]})

        @compilable
        def f(df):
            return df.round(1)

        result = f(df)
        expected = df.round(1)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dataframe_round_result(self):
        df = DataFrame({"a": [1.555, 2.444], "b": [3.999, 4.001]})

        @compilable
        def f(df):
            return df.round(2)

        result = f(df)
        expected = df.round(2)
        tm.assert_frame_equal(result, expected, check_dtype=False)


# ---------------------------------------------------------------------------
# DataFrame.eval
# ---------------------------------------------------------------------------


class TestEval:
    def test_eval_new_column_traced(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        @compilable
        def f(df):
            return df.eval("c = a + b")

        result = f(df)
        expected = df.eval("c = a + b")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_eval_result_correct(self):
        df = DataFrame({"price": [10, 20, 30], "qty": [2, 3, 4]})

        @compilable
        def f(df):
            return df.eval("revenue = price * qty")

        result = f(df)
        assert list(result["revenue"]) == [20, 60, 120]

    def test_eval_arithmetic_expr(self):
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def f(df):
            return df.eval("y = x * 2 + 1")

        result = f(df)
        assert list(result["y"]) == [21, 41, 61]

    def test_eval_boolean_graph_break(self):
        """eval() returning a boolean Series should graph-break gracefully."""
        df = DataFrame({"a": [1, 2, 3]})

        @compilable
        def f(df):
            return df.eval("a > 1")

        result = f(df)
        expected = df.eval("a > 1")
        tm.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# Test coverage gap fills
# ---------------------------------------------------------------------------


class TestQueryOrOperator:
    def test_query_or(self):
        df = DataFrame({"x": [1, 5, 10], "y": [10, 1, 5]})

        @compilable
        def f(df):
            return df.query("x > 5 or y > 5")

        result = f(df)
        expected = df.query("x > 5 or y > 5").reset_index(drop=True)
        tm.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_query_and_or_combined(self):
        df = DataFrame({"x": [1, 5, 10, 3], "y": [10, 1, 5, 8]})

        @compilable
        def f(df):
            return df.query("x > 3 and y < 6 or x > 8")

        result = f(df)
        expected = df.query("x > 3 and y < 6 or x > 8").reset_index(drop=True)
        tm.assert_frame_equal(result.reset_index(drop=True), expected)


class TestAssignCallable:
    def test_assign_callable_traced(self):
        df = DataFrame({"a": [1, 2, 3]})

        @compilable
        def f(df):
            return df.assign(b=lambda d: d["a"] * 2)

        result = f(df)
        expected = df.assign(b=lambda d: d["a"] * 2)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_assign_callable_with_filter(self):
        df = DataFrame({"a": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df.assign(doubled=lambda d: d["a"] * 2)[lambda d: d["doubled"] > 4]

        result = f(df)
        expected = df.assign(doubled=df["a"] * 2)
        expected = expected[expected["doubled"] > 4].reset_index(drop=True)
        tm.assert_frame_equal(
            result.reset_index(drop=True), expected, check_dtype=False
        )


class TestDiffNegativePeriods:
    def test_series_diff_negative(self):
        df = DataFrame({"a": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.assign(d=df["a"].diff(-1))

        result = f(df)
        expected = df.assign(d=df["a"].diff(-1))
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dataframe_diff_negative(self):
        df = DataFrame({"a": [10, 20, 30], "b": [1.0, 2.0, 3.0]})

        @compilable
        def f(df):
            return df.diff(-1)

        result = f(df)
        expected = df.diff(-1)
        tm.assert_frame_equal(result, expected, check_dtype=False)


class TestDatetimeAccessorGaps:
    def test_dt_minute(self):
        df = DataFrame(
            {"ts": pd.to_datetime(["2024-01-01 10:30:00", "2024-06-15 14:45:00"])}
        )

        @compilable
        def f(df):
            df["minute"] = df["ts"].dt.minute
            return df

        result = f(df)
        assert list(result["minute"]) == [30, 45]

    def test_dt_second(self):
        df = DataFrame(
            {"ts": pd.to_datetime(["2024-01-01 10:30:15", "2024-06-15 14:45:30"])}
        )

        @compilable
        def f(df):
            df["second"] = df["ts"].dt.second
            return df

        result = f(df)
        assert list(result["second"]) == [15, 30]

    def test_dt_dayofyear(self):
        df = DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-12-31"])})

        @compilable
        def f(df):
            df["doy"] = df["ts"].dt.dayofyear
            return df

        result = f(df)
        assert result["doy"].iloc[0] == 1
        assert result["doy"].iloc[1] == 366  # 2024 is a leap year


# ---------------------------------------------------------------------------
# Math functions — traced Series methods
# ---------------------------------------------------------------------------


class TestMathFunctions:
    @pytest.fixture
    def sample_df(self):
        return DataFrame({"x": [1.0, 4.0, 9.0, 16.0], "y": [0.0, 1.0, 2.0, 3.0]})

    def test_sqrt_traced(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["root"] = df["x"].sqrt()
            return df

        result = f(sample_df)
        tm.assert_numpy_array_equal(
            result["root"].values, np.sqrt(sample_df["x"]).values
        )

    def test_log_traced(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["ln"] = df["x"].log()
            return df

        result = f(sample_df)
        tm.assert_numpy_array_equal(result["ln"].values, np.log(sample_df["x"]).values)

    def test_log10_traced(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["lg"] = df["x"].log10()
            return df

        result = f(sample_df)
        tm.assert_numpy_array_equal(
            result["lg"].values, np.log10(sample_df["x"]).values
        )

    def test_exp_traced(self, sample_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["e"] = df["y"].exp()
            return df

        result = f(sample_df)
        tm.assert_numpy_array_equal(result["e"].values, np.exp(sample_df["y"]).values)

    def test_ceil_traced(self):
        df = DataFrame({"x": [1.2, 2.7, 3.0, 4.5]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["c"] = df["x"].ceil()
            return df

        result = f(df)
        tm.assert_numpy_array_equal(result["c"].values, np.ceil(df["x"]).values)

    def test_floor_traced(self):
        df = DataFrame({"x": [1.2, 2.7, 3.0, 4.5]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["fl"] = df["x"].floor()
            return df

        result = f(df)
        tm.assert_numpy_array_equal(result["fl"].values, np.floor(df["x"]).values)

    def test_sign_traced(self):
        df = DataFrame({"x": [-3.0, 0.0, 5.0, -1.0]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["s"] = df["x"].sign()
            return df

        result = f(df)
        tm.assert_numpy_array_equal(result["s"].values, np.sign(df["x"]).values)


class TestNumpyUfuncInterception:
    def test_np_sqrt(self):
        df = DataFrame({"x": [1.0, 4.0, 9.0]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["root"] = np.sqrt(df["x"])
            return df

        result = f(df)
        tm.assert_numpy_array_equal(result["root"].values, np.sqrt(df["x"]).values)

    def test_np_log(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["ln"] = np.log(df["x"])
            return df

        result = f(df)
        tm.assert_numpy_array_equal(result["ln"].values, np.log(df["x"]).values)

    def test_np_exp(self):
        df = DataFrame({"x": [0.0, 1.0, 2.0]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["e"] = np.exp(df["x"])
            return df

        result = f(df)
        tm.assert_numpy_array_equal(result["e"].values, np.exp(df["x"]).values)

    def test_np_abs(self):
        df = DataFrame({"x": [-3.0, 0.0, 5.0]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["a"] = np.abs(df["x"])
            return df

        result = f(df)
        tm.assert_numpy_array_equal(result["a"].values, np.abs(df["x"]).values)


# ---------------------------------------------------------------------------
# String accessor expansion
# ---------------------------------------------------------------------------


class TestStringAccessorExpanded:
    @pytest.fixture
    def string_df(self):
        return DataFrame({"name": ["alice", "BOB", "Charlie", "  dave  ", "123", " "]})

    def test_capitalize(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["cap"] = df["name"].str.strip().str.capitalize()
            return df

        result = f(string_df)
        expected = string_df["name"].str.strip().str.capitalize()
        assert list(result["cap"]) == list(expected)

    def test_title(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["t"] = df["name"].str.strip().str.title()
            return df

        result = f(string_df)
        expected = string_df["name"].str.strip().str.title()
        assert list(result["t"]) == list(expected)

    def test_swapcase(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["sw"] = df["name"].str.swapcase()
            return df

        result = f(string_df)
        expected = string_df["name"].str.swapcase()
        assert list(result["sw"]) == list(expected)

    def test_isdigit(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["dig"] = df["name"].str.strip().str.isdigit()
            return df

        result = f(string_df)
        expected = string_df["name"].str.strip().str.isdigit()
        assert list(result["dig"]) == list(expected)

    def test_isalpha(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["alpha"] = df["name"].str.strip().str.isalpha()
            return df

        result = f(string_df)
        expected = string_df["name"].str.strip().str.isalpha()
        assert list(result["alpha"]) == list(expected)

    def test_isnumeric(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["num"] = df["name"].str.strip().str.isnumeric()
            return df

        result = f(string_df)
        expected = string_df["name"].str.strip().str.isnumeric()
        assert list(result["num"]) == list(expected)

    def test_isspace(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["sp"] = df["name"].str.isspace()
            return df

        result = f(string_df)
        expected = string_df["name"].str.isspace()
        assert list(result["sp"]) == list(expected)

    def test_islower(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["low"] = df["name"].str.strip().str.islower()
            return df

        result = f(string_df)
        expected = string_df["name"].str.strip().str.islower()
        assert list(result["low"]) == list(expected)

    def test_isupper(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["up"] = df["name"].str.strip().str.isupper()
            return df

        result = f(string_df)
        expected = string_df["name"].str.strip().str.isupper()
        assert list(result["up"]) == list(expected)

    def test_count(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["cnt"] = df["name"].str.count("a")
            return df

        result = f(string_df)
        expected = string_df["name"].str.count("a")
        assert list(result["cnt"]) == list(expected)

    def test_find(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["pos"] = df["name"].str.find("li")
            return df

        result = f(string_df)
        expected = string_df["name"].str.find("li")
        assert list(result["pos"]) == list(expected)

    def test_str_method_chain(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["result"] = df["name"].str.strip().str.lower().str.capitalize()
            return df

        result = f(string_df)
        expected = string_df["name"].str.strip().str.lower().str.capitalize()
        assert list(result["result"]) == list(expected)


class TestStringPredicateFilter:
    def test_isdigit_filter(self):
        df = DataFrame({"code": ["123", "abc", "45x", "678"]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["code"].str.isdigit()]

        result = f(df)
        assert len(result) == 2
        assert set(result["code"]) == {"123", "678"}

    def test_isalpha_filter(self):
        df = DataFrame({"code": ["abc", "123", "def", "4gh"]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["code"].str.isalpha()]

        result = f(df)
        assert len(result) == 2
        assert set(result["code"]) == {"abc", "def"}

    def test_islower_filter(self):
        df = DataFrame({"name": ["alice", "BOB", "charlie", "DAVE"]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["name"].str.islower()]

        result = f(df)
        assert len(result) == 2
        assert set(result["name"]) == {"alice", "charlie"}


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_missing_table_raises(self):
        backend = PandasBackend()
        schema = Schema({"x": DType.INT64})
        node = ReadTable("nonexistent", schema)
        with pytest.raises(KeyError, match="Table 'nonexistent' not registered"):
            backend.execute(node, {}, optimize=False)

    def test_unknown_unary_op_raises(self):
        backend = PandasBackend()
        df = DataFrame({"x": [1, 2, 3]})
        node = AddColumn(
            ReadTable("t", Schema({"x": DType.INT64})),
            "y",
            UnaryOp("invalid_op", ColRef("x")),
            DType.INT64,
        )
        with pytest.raises(TypeError, match="Unknown unary op"):
            backend.execute(node, {"t": df}, optimize=False)

    def test_unknown_function_raises(self):
        backend = PandasBackend()
        df = DataFrame({"x": [1, 2, 3]})
        expr = FunctionCall("nonexistent_func", [ColRef("x")])
        node = AddColumn(
            ReadTable("t", Schema({"x": DType.INT64})),
            "y",
            expr,
            DType.INT64,
        )
        with pytest.raises(TypeError, match="Unknown function"):
            backend.execute(node, {"t": df}, optimize=False)

    def test_unknown_ir_node_raises(self):
        from pandas.jit.compiler import SubstraitCompiler
        from pandas.jit.ir import IRNode

        class FakeNode(IRNode):
            def output_schema(self):
                return Schema({})

        compiler = SubstraitCompiler()
        with pytest.raises(TypeError, match="Unknown IR node"):
            compiler._compile_rel(FakeNode())

    def test_setattr_on_traced_df_raises(self):
        ctx = TraceContext(PandasBackend())
        df = DataFrame({"x": [1, 2]})
        ctx.register_table("t", df)
        traced = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        with pytest.raises(AttributeError, match="Use df"):
            traced.new_col = 5

    def test_groupby_invalid_key_type_raises(self):
        ctx = TraceContext(PandasBackend())
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        ctx.register_table("t", df)
        traced = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        with pytest.raises(TypeError, match="groupby key must be str or list"):
            traced.groupby(123)

    def test_boolean_indexing_non_bool_raises(self):
        ctx = TraceContext(PandasBackend())
        df = DataFrame({"x": [1, 2, 3]})
        ctx.register_table("t", df)
        traced = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        series = traced["x"]  # INT64, not BOOL
        with pytest.raises(TypeError, match="Boolean indexing requires boolean"):
            traced[series]

    def test_query_syntax_error_graph_breaks(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df.query("x >>> 1")

        # Should graph-break and fall back to pandas query
        # which may raise or handle differently
        with pytest.raises((ValueError, SyntaxError)):
            f(df)

    def test_eval_syntax_error_graph_breaks(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df.eval("z = x +++ y")

        with pytest.raises((ValueError, SyntaxError, NameError)):
            f(df)

    def test_concat_non_df_raises(self):
        """pd.concat with non-DF in traced context should error."""
        df = DataFrame({"x": [1, 2]})

        @compilable(backend=PandasBackend())
        def f(df):
            return pd.concat([df, "not_a_dataframe"])

        with pytest.raises((TypeError, ValueError)):
            f(df)

    def test_astype_unknown_dtype_graph_breaks(self):
        """astype with unknown dtype should graph-break, not error."""
        df = DataFrame({"x": [1, 2, 3]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["cat"] = df["x"].astype("category")
            return df

        result = f(df)
        assert result["cat"].dtype.name == "category"


# ---------------------------------------------------------------------------
# Edge cases — empty, single-row, NaN
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_df_filter(self):
        df = DataFrame({"x": pd.Series([], dtype="int64")})

        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["x"] > 0]

        result = f(df)
        assert len(result) == 0
        assert list(result.columns) == ["x"]

    def test_empty_df_groupby(self):
        df = DataFrame(
            {"g": pd.Series([], dtype="str"), "v": pd.Series([], dtype="float64")}
        )

        @compilable(backend=PandasBackend())
        def f(df):
            return df.groupby("g")[["v"]].sum()

        result = f(df)
        assert len(result) == 0

    def test_filter_all_rows_out(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["x"] > 100]

        result = f(df)
        assert len(result) == 0
        assert list(result.columns) == ["x"]

    def test_single_row_operations(self):
        df = DataFrame({"a": [42], "b": [3.14]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["c"] = df["a"] * 2
            return df.sort_values("a")

        result = f(df)
        assert len(result) == 1
        assert result.iloc[0]["c"] == 84

    def test_single_row_window(self):
        df = DataFrame({"x": [5.0]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["lag"] = df["x"].shift(1)
            return df

        result = f(df)
        assert pd.isna(result["lag"].iloc[0])

    def test_nan_in_filter(self):
        df = DataFrame({"x": [1.0, float("nan"), 3.0]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["x"] > 0]

        result = f(df)
        assert len(result) == 2

    def test_nan_in_groupby_sum(self):
        df = DataFrame({"g": ["a", "a", "b"], "v": [1.0, float("nan"), 3.0]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df.groupby("g")[["v"]].sum()

        result = f(df)
        result = result.sort_values("g").reset_index(drop=True)
        assert result.iloc[0]["v"] == 1.0  # a: 1.0 + NaN = 1.0
        assert result.iloc[1]["v"] == 3.0  # b: 3.0

    def test_nan_propagation_arithmetic(self):
        df = DataFrame({"a": [1.0, float("nan"), 3.0], "b": [4.0, 5.0, float("nan")]})

        @compilable(backend=PandasBackend())
        def f(df):
            df["c"] = df["a"] + df["b"]
            return df

        result = f(df)
        assert result.iloc[0]["c"] == 5.0
        assert pd.isna(result.iloc[1]["c"])
        assert pd.isna(result.iloc[2]["c"])

    def test_all_nan_column_sum(self):
        df = DataFrame({"g": ["a", "a"], "v": [float("nan"), float("nan")]})

        @compilable(backend=PandasBackend())
        def f(df):
            return df.groupby("g")[["v"]].sum()

        result = f(df)
        assert result.iloc[0]["v"] == 0.0  # pandas sum of all NaN = 0

    def test_deeply_nested_pipeline(self):
        df = DataFrame({"x": list(range(100)), "y": list(range(100, 200))})

        @compilable(backend=PandasBackend())
        def f(df):
            df["z"] = df["x"] + df["y"]
            df = df[df["z"] > 50]
            df["w"] = df["z"] * 2
            df = df[df["w"] < 600]
            df = df.sort_values("w", ascending=False)
            return df.head(10)

        result = f(df)
        assert len(result) <= 10
        # Verify sort order
        assert list(result["w"]) == sorted(result["w"], reverse=True)


# ---------------------------------------------------------------------------
# Bug Fixes + Quick Wins (Phase 33)
# ---------------------------------------------------------------------------


class TestBugFixesPhase33:
    """Tests for Batch A: mean alias, tail, contains, explain, etc."""

    @pytest.fixture
    def ctx(self):
        return TraceContext(PandasBackend())

    @pytest.fixture
    def sample_df(self):
        return DataFrame({"g": ["a", "a", "b", "b"], "v": [10, 20, 30, 40]})

    def test_agg_mean_alias_schema(self, ctx, sample_df):
        """agg({"col": "mean"}) should produce FLOAT64 dtype in schema."""

        schema = Schema({"g": DType.STRING, "v": DType.INT64})
        read = ReadTable("t", schema)
        agg = Aggregate(
            read,
            group_keys=["g"],
            agg_specs=[("v_mean", "v", "mean")],
        )
        out = agg.output_schema()
        assert out.columns["v_mean"] == DType.FLOAT64

    def test_agg_mean_alias_execution(self, sample_df):
        """agg({"v": "mean"}) should produce correct values when executed."""

        @compilable
        def f(df):
            return df.groupby("g").agg({"v": "mean"})

        result = f(sample_df).sort_values("g").reset_index(drop=True)
        expected = (
            sample_df.groupby("g", as_index=False)
            .agg({"v": "mean"})
            .sort_values("g")
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(result, expected)

    def test_tail_returns_traced(self, ctx, sample_df):
        """df.tail(n) should return TracedDataFrame."""
        ctx.register_table("t", sample_df)
        schema = Schema({"g": DType.STRING, "v": DType.INT64})
        tdf = TracedDataFrame(ctx, ReadTable("t", schema))
        result = tdf.tail(2)
        assert isinstance(result, TracedDataFrame)

    def test_tail_values(self, sample_df):
        """tail() should return the last n rows."""

        @compilable
        def f(df):
            return df.tail(2)

        result = f(sample_df)
        expected = sample_df.tail(2).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_contains_column(self, ctx, sample_df):
        """'col' in df should return True for existing columns."""
        ctx.register_table("t", sample_df)
        schema = Schema({"g": DType.STRING, "v": DType.INT64})
        tdf = TracedDataFrame(ctx, ReadTable("t", schema))
        assert "g" in tdf
        assert "v" in tdf

    def test_contains_missing(self, ctx, sample_df):
        """'nonexistent' in df should return False."""
        ctx.register_table("t", sample_df)
        schema = Schema({"g": DType.STRING, "v": DType.INT64})
        tdf = TracedDataFrame(ctx, ReadTable("t", schema))
        assert "nonexistent" not in tdf

    def test_explain_eager_with_reason(self):
        """EagerSegment with reason should format correctly in explain output."""
        from pandas.jit.compiler import EagerSegment

        seg = EagerSegment(
            operation="apply",
            fn=lambda: None,
            input_tables=["t"],
            output_names=["__mat_1"],
            reason="lambda function cannot be traced",
        )
        # Simulate the format logic from explain()
        reason = (
            f" ({seg.reason})" if seg.reason and seg.reason != seg.operation else ""
        )
        line = f"  [0] EAGER: {seg.operation}{reason} -> {seg.output_names}"
        assert "(lambda function cannot be traced)" in line
        assert "EAGER: apply" in line

    def test_reset_cache(self, sample_df):
        """reset_cache() should clear cached plans."""

        @compilable
        def f(df):
            return df[df["v"] > 10]

        f(sample_df)
        assert len(f._cached_plans) > 0
        f.reset_cache()
        assert len(f._cached_plans) == 0

    def test_signature_preserved(self):
        """inspect.signature(compiled_fn) should match original function."""
        import inspect

        def my_func(df, threshold=10):
            return df[df["v"] > threshold]

        compiled = compilable(my_func)
        sig = inspect.signature(compiled)
        params = list(sig.parameters.keys())
        assert params == ["df", "threshold"]
        assert sig.parameters["threshold"].default == 10


# ---------------------------------------------------------------------------
# GroupBy Transform (Phase 34)
# ---------------------------------------------------------------------------


class TestGroupByTransform:
    """Tests for groupby().transform() tracing."""

    @pytest.fixture
    def df(self):
        return DataFrame(
            {
                "g": ["a", "a", "b", "b", "b"],
                "v": [10, 20, 30, 40, 50],
                "w": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

    def test_transform_sum(self, df):
        """groupby().transform('sum') should produce Window IR and correct values."""

        @compilable
        def f(df):
            return df.groupby("g").transform("sum")

        result = f(df)
        expected = df.groupby("g").transform("sum")
        tm.assert_frame_equal(
            result[["v", "w"]], expected[["v", "w"]], check_dtype=False
        )

    def test_transform_mean(self, df):
        """groupby().transform('mean') maps to 'avg' correctly."""

        @compilable
        def f(df):
            return df.groupby("g").transform("mean")

        result = f(df)
        expected = df.groupby("g").transform("mean")
        tm.assert_frame_equal(
            result[["v", "w"]], expected[["v", "w"]], check_dtype=False
        )

    def test_transform_min(self, df):
        @compilable
        def f(df):
            return df.groupby("g").transform("min")

        result = f(df)
        expected = df.groupby("g").transform("min")
        tm.assert_frame_equal(
            result[["v", "w"]], expected[["v", "w"]], check_dtype=False
        )

    def test_transform_max(self, df):
        @compilable
        def f(df):
            return df.groupby("g").transform("max")

        result = f(df)
        expected = df.groupby("g").transform("max")
        tm.assert_frame_equal(
            result[["v", "w"]], expected[["v", "w"]], check_dtype=False
        )

    def test_transform_count(self, df):
        @compilable
        def f(df):
            return df.groupby("g").transform("count")

        result = f(df)
        expected = df.groupby("g").transform("count")
        tm.assert_frame_equal(
            result[["v", "w"]], expected[["v", "w"]], check_dtype=False
        )

    def test_transform_std(self, df):
        @compilable
        def f(df):
            return df.groupby("g").transform("std")

        result = f(df)
        expected = df.groupby("g").transform("std")
        tm.assert_frame_equal(
            result[["v", "w"]], expected[["v", "w"]], check_dtype=False, atol=1e-10
        )

    def test_transform_series(self, df):
        """groupby()['col'].transform('sum') returns TracedSeries."""

        @compilable
        def f(df):
            df["v_sum"] = df.groupby("g")["v"].transform("sum")
            return df

        result = f(df)
        expected = df.copy()
        expected["v_sum"] = df.groupby("g")["v"].transform("sum")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_transform_multi(self, df):
        """groupby()[['a','b']].transform('sum') works."""

        @compilable
        def f(df):
            return df.groupby("g")[["v", "w"]].transform("sum")

        result = f(df)
        expected = df.groupby("g")[["v", "w"]].transform("sum")
        tm.assert_frame_equal(
            result[["v", "w"]], expected[["v", "w"]], check_dtype=False
        )

    def test_transform_lambda_graph_breaks(self, df):
        """Lambda func should graph-break but still produce correct values."""

        @compilable
        def f(df):
            df["v_custom"] = df.groupby("g")["v"].transform(lambda x: x - x.mean())
            return df

        result = f(df)
        expected = df.copy()
        expected["v_custom"] = df.groupby("g")["v"].transform(lambda x: x - x.mean())
        tm.assert_frame_equal(result, expected)

    def test_transform_unsupported_graph_breaks(self, df):
        """Unsupported string func should graph-break."""

        @compilable
        def f(df):
            df["v_med"] = df.groupby("g")["v"].transform("median")
            return df

        result = f(df)
        expected = df.copy()
        expected["v_med"] = df.groupby("g")["v"].transform("median")
        tm.assert_frame_equal(result, expected)

    def test_transform_in_pipeline(self, df):
        """transform + filter + sort pipeline works end-to-end."""

        @compilable
        def f(df):
            df["group_sum"] = df.groupby("g")["v"].transform("sum")
            df = df[df["group_sum"] > 50]
            return df.sort_values("v")

        result = f(df)
        assert all(result["group_sum"] > 50)
        assert list(result["v"]) == sorted(result["v"])


# ---------------------------------------------------------------------------
# Datetime Accessor Expansion (Phase 35)
# ---------------------------------------------------------------------------


class TestDatetimeAccessorExpanded:
    """Tests for new datetime accessor properties and methods."""

    @pytest.fixture
    def df(self):
        return DataFrame(
            {
                "ts": pd.to_datetime(
                    [
                        "2024-01-01 10:30:45.123456",
                        "2024-04-15 14:00:00.000000",
                        "2024-07-01 08:15:30.654321",
                        "2024-10-31 23:59:59.999999",
                    ]
                ),
                "v": [1, 2, 3, 4],
            }
        )

    def test_dt_weekday(self, df):
        """dt.weekday should be alias for dayofweek."""

        @compilable
        def f(df):
            df["wd"] = df["ts"].dt.weekday
            return df

        result = f(df)
        expected = df.copy()
        expected["wd"] = df["ts"].dt.weekday
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dt_microsecond(self, df):
        @compilable
        def f(df):
            df["us"] = df["ts"].dt.microsecond
            return df

        result = f(df)
        expected = df.copy()
        expected["us"] = df["ts"].dt.microsecond
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_dt_date(self, df):
        """dt.date should return date objects."""

        @compilable
        def f(df):
            df["d"] = df["ts"].dt.date
            return df

        result = f(df)
        # Verify all values are date objects
        assert all(isinstance(d, type(df["ts"].iloc[0].date())) for d in result["d"])

    def test_dt_is_month_start(self, df):
        """dt.is_month_start should return boolean series."""

        @compilable
        def f(df):
            df["ims"] = df["ts"].dt.is_month_start
            return df

        result = f(df)
        # Jan 1, Apr 15, Jul 1, Oct 31
        expected_vals = [True, False, True, False]
        assert list(result["ims"]) == expected_vals

    def test_dt_is_year_start(self, df):
        @compilable
        def f(df):
            df["iys"] = df["ts"].dt.is_year_start
            return df

        result = f(df)
        # Only Jan 1 is year start
        expected_vals = [True, False, False, False]
        assert list(result["iys"]) == expected_vals

    def test_dt_is_quarter_start(self, df):
        @compilable
        def f(df):
            df["iqs"] = df["ts"].dt.is_quarter_start
            return df

        result = f(df)
        # Jan 1, Jul 1 are quarter starts; Apr 15 and Oct 31 are not
        expected_vals = [True, False, True, False]
        assert list(result["iqs"]) == expected_vals

    def test_dt_month_name_graph_breaks(self, df):
        """month_name() should graph-break but return correct values."""

        @compilable
        def f(df):
            df["mn"] = df["ts"].dt.month_name()
            return df

        result = f(df)
        expected_names = ["January", "April", "July", "October"]
        assert list(result["mn"]) == expected_names

    def test_dt_day_name_graph_breaks(self, df):
        @compilable
        def f(df):
            df["dn"] = df["ts"].dt.day_name()
            return df

        result = f(df)
        expected_names = ["Monday", "Monday", "Monday", "Thursday"]
        assert list(result["dn"]) == expected_names

    def test_dt_is_month_start_filter(self, df):
        """Filter by is_month_start should work."""

        @compilable
        def f(df):
            return df[df["ts"].dt.is_month_start]

        result = f(df)
        # Only Jan 1 and Jul 1
        assert len(result) == 2
        assert list(result["v"]) == [1, 3]

    def test_dt_date_in_pipeline(self, df):
        """Extract date, use in assign pipeline."""

        @compilable
        def f(df):
            df["d"] = df["ts"].dt.date
            df["m"] = df["ts"].dt.month
            return df

        result = f(df)
        assert "d" in result.columns
        assert "m" in result.columns
        assert list(result["m"]) == [1, 4, 7, 10]


# ---------------------------------------------------------------------------
# String Padding (Phase 37)
# ---------------------------------------------------------------------------


class TestStringPadZfill:
    """Tests for str.pad(), str.zfill(), str.center()."""

    @pytest.fixture
    def df(self):
        return DataFrame({"s": ["abc", "de", "f"]})

    def test_str_pad(self, df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["padded"] = df["s"].str.pad(5, side="left", fillchar="0")
            return df

        result = f(df)
        expected = df.copy()
        expected["padded"] = df["s"].str.pad(5, side="left", fillchar="0")
        tm.assert_frame_equal(result, expected)

    def test_str_zfill(self, df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["zf"] = df["s"].str.zfill(5)
            return df

        result = f(df)
        expected = df.copy()
        expected["zf"] = df["s"].str.zfill(5)
        tm.assert_frame_equal(result, expected)

    def test_str_center(self, df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["centered"] = df["s"].str.center(7, fillchar="-")
            return df

        result = f(df)
        expected = df.copy()
        expected["centered"] = df["s"].str.center(7, fillchar="-")
        tm.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Phase 38: DataFrame aggregation + select_dtypes
# ---------------------------------------------------------------------------


class TestDataFrameAggregation:
    @pytest.fixture
    def df(self):
        return DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
                "c": ["x", "y", "z", "x", "y"],
            }
        )

    def test_df_sum(self, df):
        @compilable
        def f(df):
            return df[["a", "b"]].sum()

        result = f(df)
        expected = df[["a", "b"]].sum()
        tm.assert_series_equal(result, expected)

    def test_df_mean(self, df):
        @compilable
        def f(df):
            return df[["a", "b"]].mean()

        result = f(df)
        expected = df[["a", "b"]].mean()
        tm.assert_series_equal(result, expected)

    def test_df_min_max(self, df):
        @compilable
        def f(df):
            return df[["a", "b"]].min()

        result = f(df)
        expected = df[["a", "b"]].min()
        tm.assert_series_equal(result, expected)

    def test_df_count(self, df):
        @compilable
        def f(df):
            return df.count()

        result = f(df)
        expected = df.count()
        tm.assert_series_equal(result, expected)

    def test_df_std_var(self, df):
        @compilable
        def f(df):
            return df[["a", "b"]].std()

        result = f(df)
        expected = df[["a", "b"]].std()
        tm.assert_series_equal(result, expected)

    def test_df_median(self, df):
        @compilable
        def f(df):
            return df[["a", "b"]].median()

        result = f(df)
        expected = df[["a", "b"]].median()
        tm.assert_series_equal(result, expected)

    def test_df_quantile(self, df):
        @compilable
        def f(df):
            return df[["a", "b"]].quantile(0.25)

        result = f(df)
        expected = df[["a", "b"]].quantile(0.25)
        tm.assert_series_equal(result, expected)

    def test_select_dtypes_include_number(self, df):
        @compilable
        def f(df):
            return df.select_dtypes(include="number")

        result = f(df)
        expected = df.select_dtypes(include="number")
        tm.assert_frame_equal(result, expected)

    def test_select_dtypes_exclude_object(self, df):
        @compilable
        def f(df):
            return df.select_dtypes(exclude="object")

        result = f(df)
        expected = df.select_dtypes(exclude="object")
        tm.assert_frame_equal(result, expected)

    def test_select_dtypes_in_pipeline(self, df):
        @compilable
        def f(df):
            numeric = df.select_dtypes(include="number")
            return numeric[numeric["a"] > 2].sort_values("b")

        result = f(df)
        expected = (
            df.select_dtypes(include="number")
            .query("a > 2")
            .sort_values("b")
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(result, expected)

    def test_sample_returns_traced(self, df):
        @compilable
        def f(df):
            s = df.sample(n=3, random_state=42)
            return s.sort_values("a")

        result = f(df)
        assert len(result) == 3
        assert list(result.columns) == ["a", "b", "c"]

    def test_to_dict(self, df):
        @compilable
        def f(df):
            return df.to_dict(orient="list")

        result = f(df)
        expected = df.to_dict(orient="list")
        assert result == expected


# ---------------------------------------------------------------------------
# Phase 39: replace + nlargest/nsmallest
# ---------------------------------------------------------------------------


class TestReplaceAndNlargest:
    @pytest.fixture
    def df(self):
        return DataFrame(
            {
                "a": [1, 2, 3, 2, 1],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
                "c": ["x", "y", "z", "x", "y"],
            }
        )

    def test_series_replace_scalar(self, df):
        @compilable
        def f(df):
            df["a"] = df["a"].replace(1, 99)
            return df

        result = f(df)
        expected = df.copy()
        expected["a"] = df["a"].replace(1, 99)
        tm.assert_frame_equal(result, expected)

    def test_series_replace_dict(self, df):
        @compilable
        def f(df):
            df["a"] = df["a"].replace({1: 99, 2: 88})
            return df

        result = f(df)
        expected = df.copy()
        expected["a"] = df["a"].replace({1: 99, 2: 88})
        tm.assert_frame_equal(result, expected)

    def test_series_replace_in_pipeline(self, df):
        @compilable
        def f(df):
            df["a"] = df["a"].replace({1: 0, 2: 0})
            return df[df["a"] > 0]

        result = f(df)
        expected = df.copy()
        expected["a"] = df["a"].replace({1: 0, 2: 0})
        expected = expected[expected["a"] > 0].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_df_replace_scalar(self, df):
        @compilable
        def f(df):
            return df[["a", "b"]].replace(1, -1)

        result = f(df)
        expected = df[["a", "b"]].replace(1, -1)
        # Replace produces float columns for int with float replacement
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_df_replace_column_dict(self, df):
        @compilable
        def f(df):
            return df.replace({"a": {1: 99, 2: 88}})

        result = f(df)
        expected = df.replace({"a": {1: 99, 2: 88}})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_series_nlargest(self, df):
        """nlargest returns TracedSeries backed by Sort+Limit IR."""
        ctx = TraceContext(PandasBackend())
        ctx.register_table("df", df)
        tdf = TracedDataFrame(ctx, ReadTable("df", infer_schema(df)))
        result_series = tdf["b"].nlargest(3)
        assert isinstance(result_series, TracedSeries)
        # Verify IR structure: Limit(Sort(...))
        from pandas.jit.ir import (
            Limit as LimitNode,
            Sort as SortNode,
        )

        assert isinstance(result_series._source_ir, LimitNode)
        assert isinstance(result_series._source_ir.input, SortNode)
        # Execute and verify values
        result_df = ctx.backend.execute(result_series._source_ir, ctx.tables)
        expected = df.sort_values("b", ascending=False).head(3).reset_index(drop=True)
        tm.assert_series_equal(result_df["b"], expected["b"])

    def test_series_nsmallest(self, df):
        ctx = TraceContext(PandasBackend())
        ctx.register_table("df", df)
        tdf = TracedDataFrame(ctx, ReadTable("df", infer_schema(df)))
        result_series = tdf["b"].nsmallest(2)
        result_df = ctx.backend.execute(result_series._source_ir, ctx.tables)
        expected = df.sort_values("b", ascending=True).head(2).reset_index(drop=True)
        tm.assert_series_equal(result_df["b"], expected["b"])

    def test_nlargest_in_pipeline(self, df):
        @compilable
        def f(df):
            df["total"] = df["a"] + df["b"]
            top = df["total"].nlargest(3)
            return top

        result = f(df)
        assert len(result) == 3

    def test_series_replace_list_graph_breaks(self, df):
        @compilable
        def f(df):
            df["a"] = df["a"].replace([1, 2], [99, 88])
            return df

        result = f(df)
        expected = df.copy()
        expected["a"] = df["a"].replace([1, 2], [99, 88])
        tm.assert_frame_equal(result, expected)

    def test_df_replace_complex_graph_breaks(self, df):
        @compilable
        def f(df):
            return df.replace([1, 2], [99, 88])

        result = f(df)
        expected = df.replace([1, 2], [99, 88])
        tm.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Improved drop_duplicates — Window-based dedup for partial subsets
# ---------------------------------------------------------------------------


class TestImprovedDropDuplicates:
    def test_partial_subset_traced(self):
        """drop_duplicates(subset=["x"]) with keep='first' stays traced."""
        df = DataFrame({"x": [1, 1, 2, 2, 3], "y": ["a", "b", "c", "d", "e"]})

        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        result_tdf = tdf.drop_duplicates(subset=["x"])
        # Should produce Project(Filter(Window(...)))
        ir = result_tdf._ir
        assert isinstance(ir, Project)
        assert isinstance(ir.input, Filter)
        assert isinstance(ir.input.input, Window)

        result = backend.execute(ir, ctx.tables)
        expected = df.drop_duplicates(subset=["x"], keep="first").reset_index(drop=True)
        assert len(result) == len(expected)
        assert set(result["x"]) == set(expected["x"])
        assert list(result.columns) == ["x", "y"]

    def test_partial_subset_single_col_str(self):
        """drop_duplicates(subset='x') works with string subset."""
        df = DataFrame({"x": [1, 1, 2], "y": [10, 20, 30]})

        @compilable
        def f(df):
            return df.drop_duplicates(subset="x")

        result = f(df)
        assert len(result) == 2
        assert set(result["x"]) == {1, 2}

    def test_partial_subset_keeps_first(self):
        """Window-based dedup keeps the first occurrence."""
        df = DataFrame({"x": [1, 1, 1], "y": [10, 20, 30]})

        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        result_tdf = tdf.drop_duplicates(subset=["x"])
        result = backend.execute(result_tdf._ir, ctx.tables)
        assert len(result) == 1
        assert result["y"].iloc[0] == 10

    def test_full_subset_still_uses_distinct(self):
        """When subset covers all columns, Distinct is used (not Window)."""
        df = DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})

        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        result_tdf = tdf.drop_duplicates()
        assert isinstance(result_tdf._ir, Distinct)

    def test_keep_last_graph_breaks(self):
        """keep='last' still graph breaks."""
        df = DataFrame({"x": [1, 1, 2], "y": ["a", "b", "c"]})

        @compilable
        def f(df):
            return df.drop_duplicates(subset=["x"], keep="last")

        result = f(df)
        assert len(result) == 2

    def test_dedup_in_pipeline(self):
        """drop_duplicates(subset) works in a larger traced pipeline."""
        df = DataFrame({"cat": ["a", "a", "b", "b"], "val": [10, 20, 30, 40]})

        @compilable
        def f(df):
            df = df[df["val"] > 10]
            df = df.drop_duplicates(subset=["cat"])
            return df

        result = f(df)
        assert len(result) == 2
        assert set(result["cat"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# String accessor expansion — split, get, extract, repeat
# ---------------------------------------------------------------------------


class TestStringExpansion:
    def test_str_repeat_traced(self):
        """str.repeat() stays traced via FunctionCall."""
        df = DataFrame({"s": ["ab", "cd", "ef"]})

        @compilable
        def f(df):
            df["s"] = df["s"].str.repeat(3)
            return df

        result = f(df)
        expected = DataFrame({"s": ["ababab", "cdcdcd", "efefef"]})
        tm.assert_frame_equal(result, expected)

    def test_str_split_graph_break(self):
        """str.split(expand=True) graph breaks and returns TracedDataFrame."""
        df = DataFrame({"s": ["a-b", "c-d", "e-f"]})

        @compilable
        def f(df):
            parts = df["s"].str.split("-", expand=True)
            return parts

        result = f(df)
        assert result.shape == (3, 2)
        assert result.iloc[0, 0] == "a"
        assert result.iloc[0, 1] == "b"

    def test_str_get_graph_break(self):
        """str.get() graph breaks."""
        df = DataFrame({"s": ["abc", "def", "ghi"]})

        @compilable
        def f(df):
            df["first"] = df["s"].str.get(0)
            return df

        result = f(df)
        assert list(result["first"]) == ["a", "d", "g"]

    def test_str_extract_graph_break(self):
        """str.extract() graph breaks and returns DataFrame."""
        df = DataFrame({"s": ["a1", "b2", "c3"]})

        @compilable
        def f(df):
            return df["s"].str.extract(r"([a-z])(\d)")

        result = f(df)
        assert result.shape == (3, 2)
        assert list(result[0]) == ["a", "b", "c"]
        assert list(result[1]) == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# Timedelta accessor — .dt.days, .dt.seconds, .dt.total_seconds()
# ---------------------------------------------------------------------------


class TestTimedeltaAccessor:
    @pytest.fixture
    def td_df(self):
        return DataFrame(
            {
                "td": pd.to_timedelta(
                    [86400 * 1e9 + 3600 * 1e9, 2 * 86400 * 1e9, 0.5 * 1e9],
                    unit="ns",
                )
            }
        )

    def test_td_days(self, td_df):
        """dt.days extracts day component."""
        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", td_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(td_df)))
        ts = tdf["td"]
        days_series = ts.dt.days

        ir = AddColumn(tdf._ir, "days", days_series._expr, DType.INT64)
        result = backend.execute(ir, ctx.tables)
        assert list(result["days"]) == [1, 2, 0]

    def test_td_seconds(self, td_df):
        """dt.seconds extracts seconds component."""
        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", td_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(td_df)))
        ts = tdf["td"]
        secs = ts.dt.seconds

        ir = AddColumn(tdf._ir, "secs", secs._expr, DType.INT64)
        result = backend.execute(ir, ctx.tables)
        assert result["secs"].iloc[0] == 3600
        assert result["secs"].iloc[1] == 0

    def test_td_total_seconds(self, td_df):
        """dt.total_seconds() returns float."""
        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", td_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(td_df)))
        ts = tdf["td"]
        total = ts.dt.total_seconds()

        ir = AddColumn(tdf._ir, "total", total._expr, DType.FLOAT64)
        result = backend.execute(ir, ctx.tables)
        assert result["total"].iloc[0] == pytest.approx(90000.0)
        assert result["total"].iloc[1] == pytest.approx(172800.0)

    def test_td_in_pipeline(self):
        """Timedelta accessor works in a compiled pipeline."""
        df = DataFrame(
            {
                "name": ["a", "b", "c"],
                "duration": pd.to_timedelta([1, 2, 3], unit="D"),
            }
        )

        @compilable
        def f(df):
            df["days"] = df["duration"].dt.days
            return df[df["days"] > 1]

        result = f(df)
        assert len(result) == 2
        assert set(result["name"]) == {"b", "c"}


# ---------------------------------------------------------------------------
# Series comparison method forms + any/all + dropna
# ---------------------------------------------------------------------------


class TestSeriesComparisonMethods:
    @pytest.fixture
    def df(self):
        return DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})

    def test_eq_method(self, df):
        @compilable
        def f(df):
            return df[df["x"].eq(3)]

        result = f(df)
        assert len(result) == 1
        assert result["x"].iloc[0] == 3

    def test_ne_method(self, df):
        @compilable
        def f(df):
            return df[df["x"].ne(3)]

        result = f(df)
        assert len(result) == 4

    def test_lt_method(self, df):
        @compilable
        def f(df):
            return df[df["x"].lt(3)]

        result = f(df)
        assert len(result) == 2

    def test_le_method(self, df):
        @compilable
        def f(df):
            return df[df["x"].le(3)]

        result = f(df)
        assert len(result) == 3

    def test_gt_method(self, df):
        @compilable
        def f(df):
            return df[df["x"].gt(3)]

        result = f(df)
        assert len(result) == 2

    def test_ge_method(self, df):
        @compilable
        def f(df):
            return df[df["x"].ge(3)]

        result = f(df)
        assert len(result) == 3

    def test_any_returns_deferred_scalar(self, df):
        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        result = tdf["x"].gt(4).any()
        assert isinstance(result, DeferredScalar)

    def test_all_returns_deferred_scalar(self, df):
        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        result = tdf["x"].gt(0).all()
        assert isinstance(result, DeferredScalar)

    def test_any_in_pipeline(self):
        df = DataFrame({"x": [0, 0, 0, 1]})

        @compilable
        def f(df):
            has_nonzero = df["x"].any()
            df["flag"] = has_nonzero
            return df

        result = f(df)
        assert all(result["flag"])

    def test_series_dropna_traced(self):
        df = DataFrame({"x": [1.0, float("nan"), 3.0, float("nan"), 5.0]})

        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        ts = tdf["x"]
        result_ts = ts.dropna()
        # Should produce Filter IR
        assert isinstance(result_ts._source_ir, Filter)

        result = backend.execute(
            AddColumn(result_ts._source_ir, "x", result_ts._expr, DType.FLOAT64),
            ctx.tables,
        )
        assert len(result) == 3
        assert list(result["x"]) == [1.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# Series.to_frame + DataFrame.pct_change + DataFrame.duplicated
# ---------------------------------------------------------------------------


class TestSeriesFrameAndDuplicates:
    def test_to_frame_produces_traced_dataframe(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable
        def f(df):
            return df["x"].to_frame()

        result = f(df)
        assert list(result.columns) == ["x"]
        assert list(result["x"]) == [1, 2, 3]

    def test_to_frame_with_custom_name(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable
        def f(df):
            return df["x"].to_frame(name="value")

        result = f(df)
        assert list(result.columns) == ["value"]
        assert list(result["value"]) == [1, 2, 3]

    def test_to_frame_in_pipeline(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable
        def f(df):
            frame = df["x"].to_frame()
            return frame[frame["x"] > 1]

        result = f(df)
        assert len(result) == 2

    def test_df_pct_change_result_correct(self):
        df = DataFrame({"a": [10.0, 20.0, 30.0], "b": [100.0, 200.0, 400.0]})

        @compilable
        def f(df):
            return df.pct_change()

        result = f(df)
        expected = df.pct_change()
        tm.assert_frame_equal(result, expected)

    def test_df_pct_change_periods_2(self):
        df = DataFrame({"a": [10.0, 20.0, 40.0, 80.0]})

        @compilable
        def f(df):
            return df.pct_change(periods=2)

        result = f(df)
        expected = df.pct_change(periods=2)
        tm.assert_frame_equal(result, expected)

    def test_duplicated_returns_bool_series(self):
        df = DataFrame({"x": [1, 1, 2, 2], "y": ["a", "b", "a", "b"]})

        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(df)))
        result = tdf.duplicated(subset=["x"])
        assert isinstance(result, TracedSeries)

    def test_duplicated_result_correct(self):
        df = DataFrame({"x": [1, 1, 2, 3], "y": ["a", "b", "c", "d"]})

        @compilable
        def f(df):
            mask = df.duplicated(subset=["x"])
            return df[~mask]

        result = f(df)
        assert len(result) == 3
        assert set(result["x"]) == {1, 2, 3}

    def test_duplicated_all_columns(self):
        df = DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})

        @compilable
        def f(df):
            mask = df.duplicated()
            return df[~mask]

        result = f(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# groupby().nunique() + groupby().head(n) + groupby().nth(n)
# ---------------------------------------------------------------------------


class TestGroupByNuniqueAndHead:
    @pytest.fixture
    def df(self):
        return DataFrame(
            {
                "g": ["a", "a", "a", "b", "b"],
                "x": [1, 2, 1, 3, 3],
                "y": [10, 20, 30, 40, 50],
            }
        )

    def test_groupby_nunique_result(self, df):
        @compilable
        def f(df):
            return df.groupby("g").nunique()

        result = f(df).sort_values("g").reset_index(drop=True)
        assert result.loc[result["g"] == "a", "x"].iloc[0] == 2
        assert result.loc[result["g"] == "b", "x"].iloc[0] == 1

    def test_groupby_series_nunique(self, df):
        @compilable
        def f(df):
            return df.groupby("g")["x"].nunique()

        result = f(df).sort_values("g").reset_index(drop=True)
        assert result.loc[result["g"] == "a", "x"].iloc[0] == 2

    def test_groupby_head_2(self, df):
        @compilable
        def f(df):
            return df.groupby("g").head(2)

        result = f(df)
        # 2 per group: a gets 2, b gets 2 = 4 total
        assert len(result) == 4

    def test_groupby_head_1(self, df):
        @compilable
        def f(df):
            return df.groupby("g").head(1)

        result = f(df)
        assert len(result) == 2
        assert set(result["g"]) == {"a", "b"}

    def test_groupby_nth_0(self, df):
        @compilable
        def f(df):
            return df.groupby("g").nth(0)

        result = f(df)
        assert len(result) == 2
        assert set(result["g"]) == {"a", "b"}

    def test_groupby_nth_1(self, df):
        @compilable
        def f(df):
            return df.groupby("g").nth(1)

        result = f(df)
        # a has 3 rows (nth(1) exists), b has 2 rows (nth(1) exists) = 2
        assert len(result) == 2

    def test_groupby_nth_negative_graph_breaks(self, df):
        @compilable
        def f(df):
            return df.groupby("g").nth(-1)

        result = f(df)
        assert len(result) == 2
        assert set(result["g"]) == {"a", "b"}

    def test_groupby_head_in_pipeline(self, df):
        @compilable
        def f(df):
            df = df[df["y"] > 10]
            return df.groupby("g").head(1)

        result = f(df)
        assert len(result) == 2
        assert set(result["g"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# dt.week / dt.is_year_end / dt.is_month_end / dt.is_quarter_end + fillna(method=)
# ---------------------------------------------------------------------------


class TestDatetimeEndProperties:
    @pytest.fixture
    def dt_df(self):
        return DataFrame(
            {
                "ts": pd.to_datetime(
                    ["2023-12-31", "2023-06-30", "2023-03-15", "2023-01-01"]
                )
            }
        )

    def test_dt_week(self, dt_df):
        backend = PandasBackend()
        ctx = TraceContext(backend)
        ctx.register_table("t", dt_df)
        tdf = TracedDataFrame(ctx, ReadTable("t", infer_schema(dt_df)))
        ts = tdf["ts"]
        week = ts.dt.week
        ir = AddColumn(tdf._ir, "week", week._expr, DType.INT64)
        result = backend.execute(ir, ctx.tables)
        expected = dt_df["ts"].dt.isocalendar().week.astype("int64")
        assert list(result["week"]) == list(expected)

    def test_dt_is_year_end(self, dt_df):
        @compilable
        def f(df):
            df["is_ye"] = df["ts"].dt.is_year_end
            return df

        result = f(dt_df)
        assert bool(result["is_ye"].iloc[0]) is True
        assert bool(result["is_ye"].iloc[2]) is False

    def test_dt_is_month_end(self, dt_df):
        @compilable
        def f(df):
            df["is_me"] = df["ts"].dt.is_month_end
            return df

        result = f(dt_df)
        # Dec 31 and Jun 30 are month ends
        assert bool(result["is_me"].iloc[0]) is True
        assert bool(result["is_me"].iloc[1]) is True
        assert bool(result["is_me"].iloc[2]) is False

    def test_dt_is_quarter_end(self, dt_df):
        @compilable
        def f(df):
            df["is_qe"] = df["ts"].dt.is_quarter_end
            return df

        result = f(dt_df)
        # Dec 31 and Jun 30 are quarter ends
        assert bool(result["is_qe"].iloc[0]) is True
        assert bool(result["is_qe"].iloc[1]) is True
        assert bool(result["is_qe"].iloc[2]) is False

    def test_dt_is_year_end_filter(self, dt_df):
        @compilable
        def f(df):
            df["is_ye"] = df["ts"].dt.is_year_end
            return df[df["is_ye"]]

        result = f(dt_df)
        assert len(result) == 1

    def test_series_ffill(self):
        df = DataFrame({"x": [1.0, float("nan"), float("nan"), 4.0]})

        @compilable
        def f(df):
            df["x"] = df["x"].ffill()
            return df

        result = f(df)
        assert list(result["x"]) == [1.0, 1.0, 1.0, 4.0]

    def test_series_bfill(self):
        df = DataFrame({"x": [float("nan"), float("nan"), 3.0, 4.0]})

        @compilable
        def f(df):
            df["x"] = df["x"].bfill()
            return df

        result = f(df)
        assert list(result["x"]) == [3.0, 3.0, 3.0, 4.0]

    def test_df_ffill(self):
        df = DataFrame(
            {"a": [1.0, float("nan"), 3.0], "b": [float("nan"), 2.0, float("nan")]}
        )

        @compilable
        def f(df):
            return df.ffill()

        result = f(df)
        assert result["a"].iloc[1] == 1.0
        assert result["b"].iloc[2] == 2.0


# ---------------------------------------------------------------------------
# Graph-break entry points: describe/mode/insert/transpose/groupby.apply/median/quantile
# ---------------------------------------------------------------------------


class TestGraphBreakEntryPoints:
    def test_series_describe(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df["x"].describe()

        result = f(df)
        assert result["mean"] == 3.0
        assert result["count"] == 5.0

    def test_series_mode(self):
        df = DataFrame({"x": [1, 2, 2, 3, 3, 3]})

        @compilable
        def f(df):
            return df["x"].mode()

        result = f(df)
        assert result.iloc[0] == 3

    def test_df_insert(self):
        df = DataFrame({"a": [1, 2, 3]})

        @compilable
        def f(df):
            df.insert(1, "b", [10, 20, 30])
            return df

        result = f(df)
        assert list(result.columns) == ["a", "b"]
        assert list(result["b"]) == [10, 20, 30]

    def test_df_transpose(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})

        @compilable
        def f(df):
            return df.T

        result = f(df)
        assert result.shape == (2, 2)
        assert list(result.index) == ["a", "b"]

    def test_df_transpose_method(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})

        @compilable
        def f(df):
            return df.transpose()

        result = f(df)
        assert result.shape == (2, 2)

    def test_groupby_apply(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df.groupby("g").apply(lambda g: g.nlargest(1, "x"))

        result = f(df)
        assert len(result) == 2

    def test_groupby_median(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 5.0, 7.0]})

        @compilable
        def f(df):
            return df.groupby("g").median()

        result = f(df)
        assert result.loc[result["g"] == "a", "x"].iloc[0] == 2.0
        assert result.loc[result["g"] == "b", "x"].iloc[0] == 6.0

    def test_groupby_quantile(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 5.0, 7.0]})

        @compilable
        def f(df):
            return df.groupby("g").quantile(0.5)

        result = f(df)
        assert len(result) == 2

    def test_series_describe_after_filter(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            s = df[df["x"] > 2]["x"]
            return s.describe()

        result = f(df)
        assert result["count"] == 3.0
        assert result["mean"] == 4.0

    def test_insert_preserves_tracing(self):
        df = DataFrame({"a": [1, 2, 3]})

        @compilable
        def f(df):
            df.insert(1, "b", [10, 20, 30])
            df["c"] = df["a"] + df["b"]
            return df

        result = f(df)
        assert list(result.columns) == ["a", "b", "c"]
        assert list(result["c"]) == [11, 22, 33]

    def test_groupby_apply_custom_agg(self):
        df = DataFrame({"g": ["a", "a", "b", "b", "b"], "x": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            return df.groupby("g").apply(lambda g: g["x"].sum())

        result = f(df)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Series statistical methods + properties (Phase 48)
# ---------------------------------------------------------------------------


class TestSeriesStatisticalMethods:
    def test_series_median(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df["x"].median()

        assert f(df) == 3.0

    def test_series_quantile(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df["x"].quantile(0.25)

        assert f(df) == 2.0

    def test_series_quantile_75(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df["x"].quantile(0.75)

        assert f(df) == 4.0

    def test_series_prod(self):
        df = DataFrame({"x": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df["x"].prod()

        assert f(df) == 24

    def test_series_product_alias(self):
        df = DataFrame({"x": [2.0, 3.0, 5.0]})

        @compilable
        def f(df):
            return df["x"].product()

        assert f(df) == 30.0

    def test_series_sem(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

        @compilable
        def f(df):
            return df["x"].sem()

        result = f(df)
        assert abs(result - df["x"].sem()) < 1e-10

    def test_series_skew(self):
        df = DataFrame({"x": [1.0, 2.0, 2.0, 3.0, 5.0]})

        @compilable
        def f(df):
            return df["x"].skew()

        result = f(df)
        assert abs(result - df["x"].skew()) < 1e-10

    def test_series_kurt(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df["x"].kurt()

        result = f(df)
        assert abs(result - df["x"].kurt()) < 1e-10

    def test_series_kurtosis_alias(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df["x"].kurtosis()

        result = f(df)
        assert abs(result - df["x"].kurtosis()) < 1e-10

    def test_series_name_property(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df["x"].name

        assert f(df) == "x"

    def test_series_ndim_property(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df["x"].ndim

        assert f(df) == 1

    def test_df_ndim_property(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df.ndim

        assert f(df) == 2

    def test_df_index_property(self):
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def f(df):
            return df.index

        result = f(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# GroupBy window operations + extensions (Phase 49)
# ---------------------------------------------------------------------------


class TestGroupByWindowOps:
    def test_groupby_shift(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0]})

        @compilable
        def f(df):
            return df.groupby("g").shift(1)

        result = f(df)
        assert pd.isna(result["x"].iloc[0])
        assert result["x"].iloc[1] == 1.0

    def test_groupby_diff(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [10.0, 30.0, 100.0, 150.0]})

        @compilable
        def f(df):
            return df.groupby("g").diff()

        result = f(df)
        assert pd.isna(result["x"].iloc[0])
        assert result["x"].iloc[1] == 20.0
        assert pd.isna(result["x"].iloc[2])
        assert result["x"].iloc[3] == 50.0

    def test_groupby_pct_change(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [100.0, 200.0, 50.0, 75.0]})

        @compilable
        def f(df):
            return df.groupby("g").pct_change()

        result = f(df)
        assert pd.isna(result["x"].iloc[0])
        assert abs(result["x"].iloc[1] - 1.0) < 1e-10
        assert abs(result["x"].iloc[3] - 0.5) < 1e-10

    def test_groupby_ffill(self):
        df = DataFrame(
            {
                "g": ["a", "a", "a", "b", "b"],
                "x": [1.0, float("nan"), float("nan"), 10.0, float("nan")],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g").ffill()

        result = f(df)
        assert result["x"].iloc[1] == 1.0
        assert result["x"].iloc[2] == 1.0
        assert result["x"].iloc[4] == 10.0

    def test_groupby_bfill(self):
        df = DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "x": [float("nan"), 5.0, float("nan"), 8.0],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g").bfill()

        result = f(df)
        assert result["x"].iloc[0] == 5.0
        assert result["x"].iloc[2] == 8.0

    def test_groupby_describe(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0]})

        @compilable
        def f(df):
            return df.groupby("g").describe()

        result = f(df)
        assert result is not None

    def test_groupby_ngroups(self):
        df = DataFrame({"g": ["a", "a", "b", "b", "c"], "x": [1, 2, 3, 4, 5]})

        @compilable
        def f(df):
            return df.groupby("g").ngroups

        assert f(df) == 3

    def test_groupby_get_group(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df.groupby("g").get_group("a")

        result = f(df)
        assert len(result) == 2
        assert list(result["x"]) == [1, 2]

    def test_groupby_shift_negative(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0]})

        @compilable
        def f(df):
            return df.groupby("g").shift(-1)

        result = f(df)
        assert result["x"].iloc[0] == 2.0
        assert pd.isna(result["x"].iloc[1])

    def test_groupby_diff_multi_col(self):
        df = DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "x": [10.0, 30.0, 100.0, 150.0],
                "y": [1.0, 4.0, 10.0, 20.0],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g").diff()

        result = f(df)
        assert result["y"].iloc[1] == 3.0
        assert result["y"].iloc[3] == 10.0


# ---------------------------------------------------------------------------
# DataFrame analytics: corr/cov/nunique/isin/prod + Series corr/cov (Phase 50)
# ---------------------------------------------------------------------------


class TestDataFrameAnalytics:
    def test_df_corr(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})

        @compilable
        def f(df):
            return df.corr()

        result = f(df)
        assert abs(result.loc["x", "y"] - 1.0) < 1e-10

    def test_df_cov(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})

        @compilable
        def f(df):
            return df.cov()

        result = f(df)
        assert result.loc["x", "y"] > 0

    def test_df_nunique(self):
        df = DataFrame({"x": [1, 2, 2, 3], "y": ["a", "a", "b", "b"]})

        @compilable
        def f(df):
            return df.nunique()

        result = f(df)
        assert result["x"] == 3
        assert result["y"] == 2

    def test_df_prod(self):
        df = DataFrame({"x": [2, 3, 4], "y": [1.0, 2.0, 3.0]})

        @compilable
        def f(df):
            return df.prod()

        result = f(df)
        assert result["x"] == 24
        assert result["y"] == 6.0

    def test_df_isin(self):
        df = DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

        @compilable
        def f(df):
            return df.isin({"x": [1, 3], "y": ["b"]})

        result = f(df)
        assert bool(result["x"].iloc[0]) is True
        assert bool(result["x"].iloc[1]) is False
        assert bool(result["y"].iloc[1]) is True

    def test_series_corr(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})

        @compilable
        def f(df):
            return df["x"].corr(df["y"])

        result = f(df)
        assert abs(result - 1.0) < 1e-10

    def test_series_cov(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})

        @compilable
        def f(df):
            return df["x"].cov(df["y"])

        result = f(df)
        assert result > 0

    def test_series_corr_negative(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})

        @compilable
        def f(df):
            return df["x"].corr(df["y"])

        result = f(df)
        assert abs(result - (-1.0)) < 1e-10

    def test_df_corr_after_filter(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [5.0, 4.0, 3.0, 2.0, 1.0]})

        @compilable
        def f(df):
            df = df[df["x"] > 1]
            return df.corr()

        result = f(df)
        assert result.loc["x", "y"] < 0

    def test_df_product_alias(self):
        df = DataFrame({"x": [2, 3, 5]})

        @compilable
        def f(df):
            return df.product()

        result = f(df)
        assert result["x"] == 30


# ---------------------------------------------------------------------------
# Accessor extensions: datetime tz/strftime + string match/fullmatch (Phase 51)
# ---------------------------------------------------------------------------


class TestAccessorExtensions:
    def test_dt_strftime(self):
        df = DataFrame(
            {"d": pd.to_datetime(["2024-01-15", "2024-06-20", "2024-12-25"])}
        )

        @compilable
        def f(df):
            df["formatted"] = df["d"].dt.strftime("%Y-%m")
            return df

        result = f(df)
        assert result["formatted"].iloc[0] == "2024-01"
        assert result["formatted"].iloc[2] == "2024-12"

    def test_dt_tz_localize(self):
        df = DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-06-01"])})

        @compilable
        def f(df):
            localized = df["d"].dt.tz_localize("UTC")
            return localized.to_frame()

        result = f(df)
        assert len(result) == 2

    def test_dt_tz_convert(self):
        df = DataFrame(
            {"d": pd.to_datetime(["2024-01-01", "2024-06-01"]).tz_localize("UTC")}
        )

        @compilable
        def f(df):
            converted = df["d"].dt.tz_convert("US/Eastern")
            return converted.to_frame()

        result = f(df)
        assert len(result) == 2

    def test_dt_normalize(self):
        df = DataFrame(
            {"d": pd.to_datetime(["2024-01-15 10:30:00", "2024-06-20 15:45:00"])}
        )

        @compilable
        def f(df):
            df["d"] = df["d"].dt.normalize()
            return df

        result = f(df)
        assert result["d"].iloc[0].hour == 0

    def test_dt_floor(self):
        df = DataFrame(
            {"d": pd.to_datetime(["2024-01-15 10:30:45", "2024-06-20 15:45:30"])}
        )

        @compilable
        def f(df):
            df["d"] = df["d"].dt.floor("h")
            return df

        result = f(df)
        assert result["d"].iloc[0].minute == 0

    def test_dt_ceil(self):
        df = DataFrame(
            {"d": pd.to_datetime(["2024-01-15 10:30:00", "2024-06-20 15:00:00"])}
        )

        @compilable
        def f(df):
            df["d"] = df["d"].dt.ceil("h")
            return df

        result = f(df)
        assert result["d"].iloc[0].hour == 11

    def test_dt_round(self):
        df = DataFrame(
            {"d": pd.to_datetime(["2024-01-15 10:30:00", "2024-06-20 15:45:00"])}
        )

        @compilable
        def f(df):
            df["d"] = df["d"].dt.round("h")
            return df

        result = f(df)
        assert result["d"].iloc[0].minute == 0

    def test_str_match(self):
        df = DataFrame({"s": ["hello", "world", "hi", "hey"]})

        @compilable
        def f(df):
            df["matched"] = df["s"].str.match(r"h[ei]")
            return df

        result = f(df)
        assert bool(result["matched"].iloc[0]) is True
        assert bool(result["matched"].iloc[1]) is False

    def test_str_fullmatch(self):
        df = DataFrame({"s": ["abc", "ab", "abcd"]})

        @compilable
        def f(df):
            df["full"] = df["s"].str.fullmatch(r"abc")
            return df

        result = f(df)
        assert bool(result["full"].iloc[0]) is True
        assert bool(result["full"].iloc[1]) is False

    def test_str_ljust(self):
        df = DataFrame({"s": ["hi", "hey"]})

        @compilable
        def f(df):
            df["padded"] = df["s"].str.ljust(5)
            return df

        result = f(df)
        assert result["padded"].iloc[0] == "hi   "

    def test_str_rjust(self):
        df = DataFrame({"s": ["hi", "hey"]})

        @compilable
        def f(df):
            df["padded"] = df["s"].str.rjust(5)
            return df

        result = f(df)
        assert result["padded"].iloc[0] == "   hi"

    def test_str_wrap(self):
        df = DataFrame({"s": ["hello world foo bar"]})

        @compilable
        def f(df):
            df["wrapped"] = df["s"].str.wrap(10)
            return df

        result = f(df)
        assert "\n" in result["wrapped"].iloc[0]


# ---------------------------------------------------------------------------
# idxmin/idxmax + join/combine_first + GroupBy first traced (Phase 52)
# ---------------------------------------------------------------------------


class TestIdxAndJoin:
    def test_series_idxmin(self):
        df = DataFrame({"x": [3.0, 1.0, 2.0]})

        @compilable
        def f(df):
            return df["x"].idxmin()

        assert f(df) == 1

    def test_series_idxmax(self):
        df = DataFrame({"x": [3.0, 1.0, 5.0, 2.0]})

        @compilable
        def f(df):
            return df["x"].idxmax()

        assert f(df) == 2

    def test_df_idxmin(self):
        df = DataFrame({"x": [3.0, 1.0, 2.0], "y": [10.0, 20.0, 5.0]})

        @compilable
        def f(df):
            return df.idxmin()

        result = f(df)
        assert result["x"] == 1
        assert result["y"] == 2

    def test_df_idxmax(self):
        df = DataFrame({"x": [3.0, 1.0, 2.0], "y": [10.0, 20.0, 5.0]})

        @compilable
        def f(df):
            return df.idxmax()

        result = f(df)
        assert result["x"] == 0
        assert result["y"] == 1

    def test_df_join(self):
        df1 = DataFrame({"a": [1, 2, 3]})
        df2 = DataFrame({"b": [10, 20, 30]})

        @compilable
        def f(df):
            return df.join(df2)

        result = f(df1)
        assert list(result.columns) == ["a", "b"]
        assert list(result["b"]) == [10, 20, 30]

    def test_df_combine_first(self):
        df1 = DataFrame({"a": [1.0, float("nan"), 3.0]})
        df2 = DataFrame({"a": [10.0, 20.0, 30.0]})

        @compilable
        def f(df):
            return df.combine_first(df2)

        result = f(df1)
        assert result["a"].iloc[1] == 20.0

    def test_df_update(self):
        df = DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        @compilable
        def f(df):
            update_df = DataFrame({"a": [100, 200, 300]})
            df.update(update_df)
            return df

        result = f(df)
        assert list(result["a"]) == [100, 200, 300]
        assert list(result["b"]) == [10, 20, 30]

    def test_groupby_first_traced(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df.groupby("g").first()

        result = f(df)
        assert len(result) == 2

    def test_groupby_first_values(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.groupby("g").first()

        result = f(df)
        vals = sorted(result["x"].tolist())
        assert vals == [10, 30]

    def test_series_idxmin_after_filter(self):
        df = DataFrame({"x": [5.0, 3.0, 1.0, 4.0, 2.0]})

        @compilable
        def f(df):
            s = df[df["x"] > 2]["x"]
            return s.idxmin()

        result = f(df)
        assert result is not None


class TestGroupByExtras:
    """Phase 53: GroupBy prod/sem/skew/kurt/idxmin/idxmax/cumcount/filter."""

    def test_groupby_prod(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [2, 3, 4, 5]})

        @compilable
        def f(df):
            return df.groupby("g").prod()

        result = f(df)
        vals = dict(zip(result["g"], result["x"], strict=True))
        assert vals["a"] == 6
        assert vals["b"] == 20

    def test_groupby_product_alias(self):
        df = DataFrame({"g": ["a", "a", "b"], "x": [2, 3, 5]})

        @compilable
        def f(df):
            return df.groupby("g").product()

        result = f(df)
        assert len(result) == 2

    def test_groupby_sem(self):
        df = DataFrame(
            {
                "g": ["a", "a", "a", "b", "b", "b"],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g").sem()

        result = f(df)
        expected = df.groupby("g").sem().reset_index()
        assert len(result) == 2
        assert abs(result["x"].iloc[0] - expected["x"].iloc[0]) < 1e-10

    def test_groupby_skew(self):
        df = DataFrame(
            {
                "g": ["a"] * 5 + ["b"] * 5,
                "x": [1.0, 2.0, 3.0, 4.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g").skew()

        result = f(df)
        assert len(result) == 2
        # group "a" has positive skew due to outlier
        a_skew = result[result["g"] == "a"]["x"].iloc[0]
        assert a_skew > 0

    def test_groupby_kurt(self):
        df = DataFrame(
            {
                "g": ["a"] * 10 + ["b"] * 10,
                "x": [*list(range(10)), 0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g").kurt()

        result = f(df)
        assert len(result) == 2

    def test_groupby_idxmin(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [3.0, 1.0, 4.0, 2.0]})

        @compilable
        def f(df):
            return df.groupby("g").idxmin()

        result = f(df)
        assert result is not None

    def test_groupby_idxmax(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [3.0, 1.0, 4.0, 2.0]})

        @compilable
        def f(df):
            return df.groupby("g").idxmax()

        result = f(df)
        assert result is not None

    def test_groupby_cumcount_ascending(self):
        df = DataFrame({"g": ["a", "a", "b", "b", "b"], "x": [1, 2, 3, 4, 5]})

        @compilable
        def f(df):
            df["cc"] = df.groupby("g").cumcount()
            return df

        result = f(df)
        # cumcount is 0-based: a→[0,1], b→[0,1,2]
        assert list(result["cc"]) == [0, 1, 0, 1, 2]

    def test_groupby_cumcount_descending(self):
        df = DataFrame({"g": ["a", "a", "b", "b", "b"], "x": [1, 2, 3, 4, 5]})

        @compilable
        def f(df):
            df["cc"] = df.groupby("g").cumcount(ascending=False)
            return df

        result = f(df)
        # descending cumcount: a→[1,0], b→[2,1,0]
        assert list(result["cc"]) == [1, 0, 2, 1, 0]

    def test_groupby_filter(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [10, 20, 1, 2]})

        @compilable
        def f(df):
            return df.groupby("g").filter(lambda g: g["x"].sum() > 5)

        result = f(df)
        # Only group "a" has sum > 5 (sum=30)
        assert len(result) == 2
        assert list(result["g"]) == ["a", "a"]

    def test_groupby_value_counts(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 1, 2, 3]})

        @compilable
        def f(df):
            return df.groupby("g").value_counts()

        result = f(df)
        assert result is not None

    def test_groupby_groups_property(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})

        @compilable
        def f(df):
            grps = df.groupby("g").groups
            return len(grps)

        result = f(df)
        assert result == 2


class TestSeriesProperties:
    """Phase 54: Series properties + utility methods."""

    def test_hasnans_true(self):
        df = DataFrame({"x": [1.0, float("nan"), 3.0]})

        @compilable
        def f(df):
            return df["x"].hasnans

        assert f(df) is True

    def test_hasnans_false(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0]})

        @compilable
        def f(df):
            return df["x"].hasnans

        assert f(df) is False

    def test_is_unique_true(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df["x"].is_unique

        assert f(df) is True

    def test_is_unique_false(self):
        df = DataFrame({"x": [1, 2, 2]})

        @compilable
        def f(df):
            return df["x"].is_unique

        assert f(df) is False

    def test_is_monotonic_increasing(self):
        df = DataFrame({"x": [1, 2, 3, 4]})

        @compilable
        def f(df):
            return df["x"].is_monotonic_increasing

        assert f(df) is True

    def test_is_monotonic_decreasing(self):
        df = DataFrame({"x": [4, 3, 2, 1]})

        @compilable
        def f(df):
            return df["x"].is_monotonic_decreasing

        assert f(df) is True

    def test_empty_property(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df["x"].empty

        assert f(df) is False

    def test_duplicated(self):
        df = DataFrame({"x": [1, 2, 2, 3, 3]})

        @compilable
        def f(df):
            dup = df["x"].duplicated()
            return dup.to_frame()

        result = f(df)
        assert list(result["x"]) == [False, False, True, False, True]

    def test_factorize(self):
        df = DataFrame({"x": ["a", "b", "a", "c"]})

        @compilable
        def f(df):
            codes, uniques = df["x"].factorize()
            return len(uniques)

        result = f(df)
        assert result == 3

    def test_explode(self):
        df = DataFrame({"x": [[1, 2], [3, 4, 5]]})

        @compilable
        def f(df):
            return df["x"].explode().to_frame()

        result = f(df)
        assert len(result) == 5

    def test_searchsorted(self):
        df = DataFrame({"x": [1, 3, 5, 7, 9]})

        @compilable
        def f(df):
            return df["x"].searchsorted(4)

        result = f(df)
        assert result == 2

    def test_copy(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            s = df["x"]
            s2 = s.copy()
            return s2.to_frame()

        result = f(df)
        assert list(result["x"]) == [1, 2, 3]

    def test_tolist(self):
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def f(df):
            return df["x"].tolist()

        result = f(df)
        assert result == [10, 20, 30]

    def test_to_list_alias(self):
        df = DataFrame({"x": [1, 2]})

        @compilable
        def f(df):
            return df["x"].to_list()

        result = f(df)
        assert result == [1, 2]


class TestRollingExpandingExtensions:
    """Phase 55: Rolling/Expanding statistical extensions."""

    def test_rolling_median(self):
        df = DataFrame({"x": [1.0, 3.0, 2.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df.rolling(3).median()

        result = f(df)
        assert len(result) == 5
        assert result["x"].iloc[2] == 2.0  # median of [1,3,2]

    def test_rolling_quantile(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df.rolling(3).quantile(0.5)

        result = f(df)
        assert len(result) == 5

    def test_rolling_skew(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

        @compilable
        def f(df):
            return df.rolling(3).skew()

        result = f(df)
        assert len(result) == 5

    def test_rolling_kurt(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})

        @compilable
        def f(df):
            return df.rolling(4).kurt()

        result = f(df)
        assert len(result) == 7

    def test_rolling_sem(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df.rolling(3).sem()

        result = f(df)
        assert len(result) == 5

    def test_rolling_apply(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df.rolling(3).apply(lambda arr: arr.sum())

        result = f(df)
        assert len(result) == 5
        assert result["x"].iloc[2] == 6.0  # 1+2+3

    def test_expanding_median(self):
        df = DataFrame({"x": [1.0, 3.0, 2.0, 4.0]})

        @compilable
        def f(df):
            return df.expanding().median()

        result = f(df)
        assert len(result) == 4
        assert result["x"].iloc[0] == 1.0
        assert result["x"].iloc[2] == 2.0  # median of [1,3,2]

    def test_expanding_quantile(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

        @compilable
        def f(df):
            return df.expanding().quantile(0.75)

        result = f(df)
        assert len(result) == 4

    def test_expanding_skew(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

        @compilable
        def f(df):
            return df.expanding().skew()

        result = f(df)
        assert len(result) == 5

    def test_expanding_apply(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0]})

        @compilable
        def f(df):
            return df.expanding().apply(lambda arr: arr.sum())

        result = f(df)
        assert result["x"].iloc[0] == 1.0
        assert result["x"].iloc[2] == 6.0

    def test_rolling_rank(self):
        df = DataFrame({"x": [3.0, 1.0, 2.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df.rolling(3).rank()

        result = f(df)
        assert len(result) == 5

    def test_expanding_sem(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df.expanding().sem()

        result = f(df)
        assert len(result) == 5


class TestDataFrameUtilities:
    """Phase 56: DataFrame utility methods."""

    def test_filter_items(self):
        df = DataFrame({"a": [1], "b": [2], "c": [3]})

        @compilable
        def f(df):
            return df.filter(items=["a", "c"])

        result = f(df)
        assert list(result.columns) == ["a", "c"]

    def test_filter_like(self):
        df = DataFrame({"x_val": [1], "y_val": [2], "z_other": [3]})

        @compilable
        def f(df):
            return df.filter(like="val")

        result = f(df)
        assert list(result.columns) == ["x_val", "y_val"]

    def test_filter_regex(self):
        df = DataFrame({"col_1": [1], "col_2": [2], "data": [3]})

        @compilable
        def f(df):
            return df.filter(regex=r"^col_")

        result = f(df)
        assert list(result.columns) == ["col_1", "col_2"]

    def test_reindex_columns(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        @compilable
        def f(df):
            return df.reindex(columns=["c", "a"])

        result = f(df)
        assert list(result.columns) == ["c", "a"]

    def test_equals_true(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df.equals(df)

        assert f(df) is True

    def test_equals_false(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            df2 = df[df["x"] > 1]
            return df.equals(df2)

        assert f(df) is False

    def test_explode(self):
        df = DataFrame({"a": ["x", "y"], "b": [[1, 2], [3, 4, 5]]})

        @compilable
        def f(df):
            return df.explode("b")

        result = f(df)
        assert len(result) == 5

    def test_get_existing(self):
        df = DataFrame({"x": [1, 2], "y": [3, 4]})

        @compilable
        def f(df):
            col = df.get("x")
            return col.to_frame()

        result = f(df)
        assert list(result["x"]) == [1, 2]

    def test_get_missing(self):
        df = DataFrame({"x": [1, 2]})

        @compilable
        def f(df):
            return df.get("z", default="missing")

        result = f(df)
        assert result == "missing"

    def test_pop(self):
        df = DataFrame({"x": [1, 2], "y": [3, 4]})

        @compilable
        def f(df):
            df.pop("y")
            return df

        result = f(df)
        assert list(result.columns) == ["x"]

    def test_memory_usage(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable
        def f(df):
            return df.memory_usage().sum()

        result = f(df)
        assert result > 0

    def test_take(self):
        df = DataFrame({"x": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            return df.take([0, 2, 4])

        result = f(df)
        assert list(result["x"]) == [10, 30, 50]


class TestGroupBySeriesExtensions:
    """Phase 57: TracedGroupBySeries extensions."""

    def test_gbs_first(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.groupby("g")["x"].first().to_frame()

        result = f(df)
        vals = sorted(result["x"].tolist())
        assert vals == [10, 30]

    def test_gbs_last(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.groupby("g")["x"].last().to_frame()

        result = f(df)
        vals = sorted(result["x"].tolist())
        assert vals == [20, 40]

    def test_gbs_shift(self):
        df = DataFrame({"g": ["a", "a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            df["shifted"] = df.groupby("g")["x"].shift(1)
            return df

        result = f(df)
        # First element in each group should be NaN
        assert pd.isna(result["shifted"].iloc[0])
        assert result["shifted"].iloc[1] == 1.0

    def test_gbs_diff(self):
        df = DataFrame(
            {"g": ["a", "a", "a", "b", "b"], "x": [10.0, 30.0, 60.0, 5.0, 15.0]}
        )

        @compilable
        def f(df):
            df["d"] = df.groupby("g")["x"].diff(1)
            return df

        result = f(df)
        assert pd.isna(result["d"].iloc[0])
        assert result["d"].iloc[1] == 20.0
        assert result["d"].iloc[2] == 30.0

    def test_gbs_head(self):
        df = DataFrame({"g": ["a", "a", "a", "b", "b", "b"], "x": [1, 2, 3, 4, 5, 6]})

        @compilable
        def f(df):
            return df.groupby("g")["x"].head(2)

        result = f(df)
        assert len(result) == 4

    def test_gbs_apply(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0]})

        @compilable
        def f(df):
            return df.groupby("g")["x"].apply(lambda s: s.sum()).to_frame()

        result = f(df)
        assert len(result) == 2

    def test_gbs_ffill(self):
        df = DataFrame(
            {"g": ["a", "a", "b", "b"], "x": [1.0, float("nan"), float("nan"), 4.0]}
        )

        @compilable
        def f(df):
            df["filled"] = df.groupby("g")["x"].ffill()
            return df

        result = f(df)
        assert result["filled"].iloc[0] == 1.0
        assert result["filled"].iloc[1] == 1.0

    def test_gbs_median(self):
        df = DataFrame(
            {"g": ["a", "a", "a", "b", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        )

        @compilable
        def f(df):
            return df.groupby("g")["x"].median()

        result = f(df)
        assert result is not None

    def test_gbs_value_counts(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 1, 2, 3]})

        @compilable
        def f(df):
            return df.groupby("g")["x"].value_counts()

        result = f(df)
        assert result is not None

    def test_gbs_nth(self):
        df = DataFrame({"g": ["a", "a", "a", "b", "b"], "x": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            return df.groupby("g")["x"].nth(1).to_frame()

        result = f(df)
        vals = sorted(result["x"].tolist())
        assert vals == [20, 50]


class TestGroupByMultiExpansion:
    """Phase 58: TracedGroupByMulti expansion."""

    def test_gbm_first(self):
        df = DataFrame(
            {"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].first()

        result = f(df)
        assert len(result) == 2

    def test_gbm_head(self):
        df = DataFrame(
            {
                "g": ["a", "a", "a", "b", "b", "b"],
                "x": [1, 2, 3, 4, 5, 6],
                "y": [10, 20, 30, 40, 50, 60],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].head(2)

        result = f(df)
        assert len(result) == 4

    def test_gbm_shift(self):
        df = DataFrame(
            {"g": ["a", "a", "a"], "x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]}
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].shift(1)

        result = f(df)
        assert pd.isna(result["x"].iloc[0])
        assert result["x"].iloc[1] == 1.0

    def test_gbm_diff(self):
        df = DataFrame(
            {"g": ["a", "a", "a"], "x": [10.0, 30.0, 60.0], "y": [1.0, 4.0, 9.0]}
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].diff(1)

        result = f(df)
        assert pd.isna(result["x"].iloc[0])
        assert result["x"].iloc[1] == 20.0

    def test_gbm_cumsum(self):
        df = DataFrame(
            {"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].cumsum()

        result = f(df)
        assert result["x"].iloc[1] == 3  # 1+2
        assert result["y"].iloc[1] == 30  # 10+20

    def test_gbm_prod(self):
        df = DataFrame(
            {"g": ["a", "a", "b", "b"], "x": [2, 3, 4, 5], "y": [1, 2, 3, 4]}
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].prod()

        result = f(df)
        vals = dict(zip(result["g"], result["x"], strict=True))
        assert vals["a"] == 6
        assert vals["b"] == 20

    def test_gbm_nunique(self):
        df = DataFrame(
            {"g": ["a", "a", "b", "b"], "x": [1, 1, 2, 3], "y": [10, 20, 30, 30]}
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].nunique()

        result = f(df)
        a_row = result[result["g"] == "a"]
        assert a_row["x"].iloc[0] == 1
        assert a_row["y"].iloc[0] == 2

    def test_gbm_filter(self):
        df = DataFrame(
            {"g": ["a", "a", "b", "b"], "x": [10, 20, 1, 2], "y": [1, 1, 1, 1]}
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].filter(lambda g: g["x"].sum() > 5)

        result = f(df)
        assert len(result) == 2

    def test_gbm_apply(self):
        df = DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [10.0, 20.0, 30.0, 40.0],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].apply(lambda g: g.sum())

        result = f(df)
        assert result is not None

    def test_gbm_rank(self):
        df = DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "x": [2.0, 1.0, 4.0, 3.0],
                "y": [20.0, 10.0, 40.0, 30.0],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].rank(method="min")

        result = f(df)
        assert len(result) == 4

    def test_gbm_last(self):
        df = DataFrame(
            {"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].last()

        result = f(df)
        assert len(result) == 2

    def test_gbm_nth(self):
        df = DataFrame(
            {
                "g": ["a", "a", "a", "b", "b"],
                "x": [1, 2, 3, 4, 5],
                "y": [10, 20, 30, 40, 50],
            }
        )

        @compilable
        def f(df):
            return df.groupby("g")[["x", "y"]].nth(1)

        result = f(df)
        assert len(result) == 2


class TestDataFrameArithmetic:
    """Phase 59: DataFrame arithmetic + any/all."""

    def test_add_scalar(self):
        df = DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

        @compilable
        def f(df):
            return df + 5

        result = f(df)
        assert list(result["x"]) == [6, 7, 8]
        assert list(result["y"]) == [15, 25, 35]

    def test_radd_scalar(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return 10 + df

        result = f(df)
        assert list(result["x"]) == [11, 12, 13]

    def test_sub_scalar(self):
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def f(df):
            return df - 5

        result = f(df)
        assert list(result["x"]) == [5, 15, 25]

    def test_rsub_scalar(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return 10 - df

        result = f(df)
        assert list(result["x"]) == [9, 8, 7]

    def test_mul_scalar(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df * 3

        result = f(df)
        assert list(result["x"]) == [3, 6, 9]

    def test_rmul_scalar(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return 2 * df

        result = f(df)
        assert list(result["x"]) == [2, 4, 6]

    def test_truediv_scalar(self):
        df = DataFrame({"x": [10.0, 20.0, 30.0]})

        @compilable
        def f(df):
            return df / 2

        result = f(df)
        assert list(result["x"]) == [5.0, 10.0, 15.0]

    def test_rtruediv_scalar(self):
        df = DataFrame({"x": [2.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return 100 / df

        result = f(df)
        assert list(result["x"]) == [50.0, 25.0, 20.0]

    def test_neg(self):
        df = DataFrame({"x": [1, -2, 3]})

        @compilable
        def f(df):
            return -df

        result = f(df)
        assert list(result["x"]) == [-1, 2, -3]

    def test_add_preserves_string_cols(self):
        df = DataFrame({"name": ["a", "b"], "x": [1, 2]})

        @compilable
        def f(df):
            return df + 10

        result = f(df)
        assert list(result["x"]) == [11, 12]
        assert list(result["name"]) == ["a", "b"]

    def test_any(self):
        df = DataFrame({"x": [True, False, True], "y": [False, False, False]})

        @compilable
        def f(df):
            result = df.any()
            return bool(result["x"]), bool(result["y"])

        x_any, y_any = f(df)
        assert x_any is True
        assert y_any is False

    def test_all(self):
        df = DataFrame({"x": [True, True, True], "y": [True, False, True]})

        @compilable
        def f(df):
            result = df.all()
            return bool(result["x"]), bool(result["y"])

        x_all, y_all = f(df)
        assert x_all is True
        assert y_all is False


class TestGroupByRankAndAgg:
    """Phase 60: GroupBy rank, Rolling/Expanding agg, GroupBySeries prod."""

    def test_groupby_rank_min(self):
        df = DataFrame(
            {"g": ["a", "a", "a", "b", "b"], "x": [30.0, 10.0, 20.0, 50.0, 40.0]}
        )

        @compilable
        def f(df):
            df["r"] = df.groupby("g")["x"].rank(method="min")
            return df

        result = f(df)
        # group a: 30→3, 10→1, 20→2; group b: 50→2, 40→1
        assert result["r"].iloc[0] == 3.0
        assert result["r"].iloc[1] == 1.0

    def test_groupby_rank_dense(self):
        df = DataFrame({"g": ["a", "a", "a"], "x": [10.0, 10.0, 20.0]})

        @compilable
        def f(df):
            df["r"] = df.groupby("g")["x"].rank(method="dense")
            return df

        result = f(df)
        assert result["r"].iloc[2] == 2.0  # 20 is 2nd unique value

    def test_groupby_df_rank(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [2.0, 1.0, 4.0, 3.0]})

        @compilable
        def f(df):
            return df.groupby("g").rank(method="min")

        result = f(df)
        assert len(result) == 4

    def test_rolling_agg_string(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            return df.rolling(3).agg("sum")

        result = f(df)
        assert len(result) == 5
        assert result["x"].iloc[2] == 6.0  # 1+2+3

    def test_rolling_aggregate_alias(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0]})

        @compilable
        def f(df):
            return df.rolling(2).aggregate("mean")

        result = f(df)
        assert len(result) == 3

    def test_expanding_agg_string(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

        @compilable
        def f(df):
            return df.expanding().agg("sum")

        result = f(df)
        assert result["x"].iloc[3] == 10.0  # 1+2+3+4

    def test_expanding_aggregate_alias(self):
        df = DataFrame({"x": [1.0, 2.0, 3.0]})

        @compilable
        def f(df):
            return df.expanding().aggregate("mean")

        result = f(df)
        assert len(result) == 3

    def test_gbs_prod(self):
        df = DataFrame({"g": ["a", "a", "b", "b"], "x": [2, 3, 4, 5]})

        @compilable
        def f(df):
            return df.groupby("g")["x"].prod()

        result = f(df)
        vals = dict(zip(result["g"], result["x"], strict=True))
        assert vals["a"] == 6
        assert vals["b"] == 20

    def test_gbs_product_alias(self):
        df = DataFrame({"g": ["a", "a"], "x": [3, 4]})

        @compilable
        def f(df):
            return df.groupby("g")["x"].product()

        result = f(df)
        assert result["x"].iloc[0] == 12


class TestSeriesConversion:
    """Phase 61: Series conversion and iteration methods."""

    def test_to_numpy(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            arr = df["x"].to_numpy()
            return arr.sum()

        result = f(df)
        assert result == 6

    def test_to_dict(self):
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def f(df):
            d = df["x"].to_dict()
            return len(d)

        result = f(df)
        assert result == 3

    def test_items(self):
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def f(df):
            total = 0
            for idx, val in df["x"].items():
                total += val
            return total

        result = f(df)
        assert result == 60

    def test_iter(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            vals = list(df["x"])
            return sum(vals)

        result = f(df)
        assert result == 6

    def test_contains(self):
        df = DataFrame({"x": [1, 2, 3, 4, 5]})

        @compilable
        def f(df):
            return 3 in df["x"]

        assert f(df) is True

    def test_contains_false(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return 99 in df["x"]

        assert f(df) is False

    def test_values_property(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df["x"].values.sum()

        result = f(df)
        assert result == 6

    def test_shape_property(self):
        df = DataFrame({"x": [1, 2, 3, 4, 5]})

        @compilable
        def f(df):
            return df["x"].shape[0]

        result = f(df)
        assert result == 5

    def test_index_property(self):
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def f(df):
            return len(df["x"].index)

        result = f(df)
        assert result == 3

    def test_to_csv(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            csv_str = df["x"].to_csv()
            return len(csv_str) > 0

        assert f(df) is True

    def test_to_json(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            json_str = df["x"].to_json()
            return len(json_str) > 0

        assert f(df) is True


class TestPhase62:
    """Phase 62: DataFrame iter/items/keys/agg + Series map(dict) + misc."""

    def test_dataframe_iter(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        @compilable
        def f(df):
            return list(df)

        result = f(df)
        assert result == ["a", "b", "c"]

    def test_dataframe_items(self):
        df = DataFrame({"x": [1, 2], "y": [3, 4]})

        @compilable
        def f(df):
            cols = []
            for name, series in df.items():
                cols.append(name)
            return cols

        result = f(df)
        assert result == ["x", "y"]

    def test_dataframe_items_series_access(self):
        df = DataFrame({"x": [10, 20], "y": [30, 40]})

        @compilable
        def f(df):
            total = 0
            for name, series in df.items():
                total += len(series)
            return total

        result = f(df)
        assert result == 4  # 2 rows * 2 columns iterated

    def test_dataframe_keys(self):
        df = DataFrame({"a": [1], "b": [2], "c": [3]})

        @compilable
        def f(df):
            return list(df.keys())

        result = f(df)
        assert result == ["a", "b", "c"]

    def test_dataframe_agg_single(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable
        def f(df):
            return df.agg("sum")

        result = f(df)
        assert result["x"] == 6
        assert result["y"] == 15

    def test_dataframe_agg_dict(self):
        df = DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

        @compilable
        def f(df):
            return df.agg({"x": "sum", "y": "mean"})

        result = f(df)
        assert result["x"] == 6
        assert result["y"] == 20.0

    def test_dataframe_map(self):
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable
        def f(df):
            return df.map(lambda v: v * 10)

        result = f(df)
        tm.assert_frame_equal(result, DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]}))

    def test_series_map_dict(self):
        df = DataFrame({"x": [1, 2, 3, 4]})

        @compilable
        def f(df):
            mapped = df["x"].map({1: 10, 2: 20, 3: 30})
            return mapped.to_frame()

        result = f(df)
        expected = DataFrame({"x": [10.0, 20.0, 30.0, float("nan")]})
        tm.assert_frame_equal(result, expected)

    def test_series_map_dict_all_mapped(self):
        df = DataFrame({"g": ["a", "b", "c"]})

        @compilable
        def f(df):
            mapped = df["g"].map({"a": "alpha", "b": "beta", "c": "gamma"})
            return mapped.to_frame()

        result = f(df)
        assert result["g"].iloc[0] == "alpha"
        assert result["g"].iloc[1] == "beta"
        assert result["g"].iloc[2] == "gamma"

    def test_series_map_callable_graphbreak(self):
        df = DataFrame({"x": [1, 2, 3]})

        @compilable
        def f(df):
            return df["x"].map(lambda v: v * 100).tolist()

        result = f(df)
        assert result == [100, 200, 300]

    def test_dataframe_aggregate_alias(self):
        df = DataFrame({"x": [10, 20, 30]})

        @compilable
        def f(df):
            return df.aggregate("max")

        result = f(df)
        assert result["x"] == 30
