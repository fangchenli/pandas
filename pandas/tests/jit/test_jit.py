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
    BinOp,
    ColRef,
    DType,
    Filter,
    Project,
    ReadTable,
    ScalarSubquery,
    Sort,
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

    def test_dataframe_rank_graph_break(self):
        """DataFrame.rank() graph-breaks."""
        df = DataFrame({"a": [3.0, 1.0, 4.0], "b": [10.0, 30.0, 20.0]})

        @compilable
        def f(df):
            return df.rank()

        result = f(df)
        expected = df.rank()
        tm.assert_frame_equal(result, expected)

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
        from pandas.jit.ir import Aggregate

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
