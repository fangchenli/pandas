"""
Tests for pandas.lazy aggregation operations.
"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.ir import Call


class TestExprAggregationMethods:
    """Tests for aggregation method IR generation."""

    def test_sum_ir(self):
        expr = col("a").sum()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "sum"
        assert expr._ir.is_aggregate is True

    def test_mean_ir(self):
        expr = col("a").mean()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "mean"
        assert expr._ir.is_aggregate is True

    def test_min_ir(self):
        expr = col("a").min()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "min"
        assert expr._ir.is_aggregate is True

    def test_max_ir(self):
        expr = col("a").max()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "max"
        assert expr._ir.is_aggregate is True

    def test_count_ir(self):
        expr = col("a").count()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "count"
        assert expr._ir.is_aggregate is True

    def test_std_ir(self):
        expr = col("a").std()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "std"
        assert expr._ir.is_aggregate is True
        assert expr._ir.kwargs == {"ddof": 1}

    def test_std_ddof_ir(self):
        expr = col("a").std(ddof=0)
        assert expr._ir.kwargs == {"ddof": 0}

    def test_var_ir(self):
        expr = col("a").var()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "var"
        assert expr._ir.is_aggregate is True

    def test_first_ir(self):
        expr = col("a").first()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "first"
        assert expr._ir.is_aggregate is True

    def test_last_ir(self):
        expr = col("a").last()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "last"
        assert expr._ir.is_aggregate is True

    def test_n_unique_ir(self):
        expr = col("a").n_unique()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "n_unique"
        assert expr._ir.is_aggregate is True


class TestEvaluatorAggregation:
    """Tests for aggregation in the evaluator (non-grouped)."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

    def test_sum(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("sum", (FieldRef("a"),), is_aggregate=True)
        result = evaluator.evaluate(node)
        assert result == 15

    def test_mean(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("mean", (FieldRef("b"),), is_aggregate=True)
        result = evaluator.evaluate(node)
        assert result == 30.0

    def test_min(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("min", (FieldRef("a"),), is_aggregate=True)
        result = evaluator.evaluate(node)
        assert result == 1

    def test_max(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("max", (FieldRef("a"),), is_aggregate=True)
        result = evaluator.evaluate(node)
        assert result == 5

    def test_count(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("count", (FieldRef("a"),), is_aggregate=True)
        result = evaluator.evaluate(node)
        assert result == 5

    def test_std(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("std", (FieldRef("a"),), {"ddof": 1}, is_aggregate=True)
        result = evaluator.evaluate(node)
        expected = sample_df["a"].std(ddof=1)
        assert np.isclose(result, expected)

    def test_var(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("var", (FieldRef("a"),), {"ddof": 1}, is_aggregate=True)
        result = evaluator.evaluate(node)
        expected = sample_df["a"].var(ddof=1)
        assert np.isclose(result, expected)

    def test_first(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("first", (FieldRef("a"),), is_aggregate=True)
        result = evaluator.evaluate(node)
        assert result == 1

    def test_last(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        evaluator = Evaluator(sample_df)
        node = Call("last", (FieldRef("a"),), is_aggregate=True)
        result = evaluator.evaluate(node)
        assert result == 5

    def test_n_unique(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import FieldRef

        df = pd.DataFrame({"x": ["a", "b", "a", "c", "b"]})
        evaluator = Evaluator(df)
        node = Call("n_unique", (FieldRef("x"),), is_aggregate=True)
        result = evaluator.evaluate(node)
        assert result == 3


class TestLazyGroupBy:
    """Tests for LazyDataFrame.group_by()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
                "count": [1, 2, 3, 4, 5],
            }
        )

    def test_group_by_returns_lazy_groupby(self, sample_df):
        from pandas.lazy.frame import LazyGroupBy

        result = sample_df.select().group_by("category")
        assert isinstance(result, LazyGroupBy)

    def test_group_by_repr(self, sample_df):
        result = sample_df.select().group_by("category")
        assert "LazyGroupBy" in repr(result)
        assert "category" in repr(result)

    def test_group_by_empty_raises(self, sample_df):
        with pytest.raises(ValueError, match="requires at least one column"):
            sample_df.select().group_by()

    def test_agg_empty_raises(self, sample_df):
        with pytest.raises(ValueError, match="requires at least one expression"):
            sample_df.select().group_by("category").agg()

    def test_agg_invalid_type_raises(self, sample_df):
        with pytest.raises(TypeError, match="Expected Expr"):
            sample_df.select().group_by("category").agg("not_an_expr")


class TestGroupByAggregation:
    """Tests for group-by aggregation execution."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        )

    def test_group_by_sum(self, sample_df):
        result = (
            sample_df.select()
            .group_by("category")
            .agg(col("value").sum().alias("total"))
            .collect()
        )
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "total": [90, 60],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_group_by_mean(self, sample_df):
        result = (
            sample_df.select()
            .group_by("category")
            .agg(col("value").mean().alias("avg"))
            .collect()
        )
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "avg": [30.0, 30.0],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_group_by_min_max(self, sample_df):
        result = (
            sample_df.select()
            .group_by("category")
            .agg(
                col("value").min().alias("min_val"),
                col("value").max().alias("max_val"),
            )
            .collect()
        )
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "min_val": [10, 20],
                "max_val": [50, 40],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_group_by_count(self, sample_df):
        result = (
            sample_df.select()
            .group_by("category")
            .agg(col("value").count().alias("n"))
            .collect()
        )
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "n": [3, 2],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_group_by_multiple_keys(self):
        df = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": [1, 2, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )
        result = (
            df.select()
            .group_by("a", "b")
            .agg(col("value").sum().alias("total"))
            .collect()
        )
        expected = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": [1, 2, 1, 2],
                "total": [10, 20, 30, 40],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_group_by_multiple_aggs_same_column(self, sample_df):
        result = (
            sample_df.select()
            .group_by("category")
            .agg(
                col("value").sum().alias("sum"),
                col("value").mean().alias("mean"),
                col("value").count().alias("count"),
            )
            .collect()
        )
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "sum": [90, 60],
                "mean": [30.0, 30.0],
                "count": [3, 2],
            }
        )
        tm.assert_frame_equal(result, expected)


class TestGroupByWithFilterSelect:
    """Tests for combining group_by with filter and select."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A", "C"],
                "value": [10, 20, 30, 40, 50, 60],
                "flag": [True, True, False, True, True, False],
            }
        )

    def test_filter_then_group_by(self, sample_df):
        result = (
            sample_df.select()
            .filter(col("flag") == True)  # noqa: E712
            .group_by("category")
            .agg(col("value").sum().alias("total"))
            .collect()
        )
        # After filter: A(10, 50), B(20, 40)
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "total": [60, 60],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_group_by_then_select(self, sample_df):
        result = (
            sample_df.select()
            .group_by("category")
            .agg(
                col("value").sum().alias("total"),
                col("value").count().alias("n"),
            )
            .select("category", "total")
            .collect()
        )
        expected = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "total": [90, 60, 60],
            }
        )
        tm.assert_frame_equal(result, expected)


class TestAggregatePlanNode:
    """Tests for the Aggregate plan node."""

    def test_aggregate_schema(self):
        from pandas.lazy.plan import (
            Aggregate,
            DataFrameSource,
        )

        df = pd.DataFrame(
            {
                "category": ["A", "B"],
                "value": [10, 20],
            }
        )
        source = DataFrameSource(df)
        agg = Aggregate(
            source,
            (col("category"),),
            (col("value").sum().alias("total"),),
        )
        schema = agg.resolve_schema()
        assert "category" in schema
        assert "total" in schema

    def test_aggregate_repr(self):
        from pandas.lazy.plan import (
            Aggregate,
            DataFrameSource,
        )

        df = pd.DataFrame({"a": [1], "b": [2]})
        source = DataFrameSource(df)
        agg = Aggregate(
            source,
            (col("a"),),
            (col("b").sum().alias("sum_b"),),
        )
        result = repr(agg)
        assert "Aggregate" in result
        assert "a" in result
        assert "sum_b" in result

    def test_aggregate_children(self):
        from pandas.lazy.plan import (
            Aggregate,
            DataFrameSource,
        )

        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        agg = Aggregate(source, (), (col("a").sum().alias("s"),))
        assert agg.children() == [source]


class TestPostAggregationProjection:
    """Aggregation expressions that wrap arithmetic over aggregates.

    These decompose into pre-project (computed aggregate inputs) -> aggregate
    -> post-project, reusing existing nodes. Validated against eager pandas.
    """

    def _c(self, ldf):
        return ldf.collect(use_physical_planner=True, order="relaxed")

    def test_agg_arithmetic_over_aggregates(self):
        df = pd.DataFrame(
            {"g": [1, 1, 2, 2, 3], "v1": [5, 3, 8, 1, 9], "v2": [2, 7, 4, 6, 1]}
        )
        out = (
            self._c(
                df.select()
                .group_by("g")
                .agg((col("v1").max() - col("v2").min()).alias("rng"))
            )
            .sort_values("g")
            .reset_index(drop=True)
        )
        expected = (
            df.groupby("g")
            .apply(lambda x: x["v1"].max() - x["v2"].min(), include_groups=False)
            .reset_index(name="rng")
        )
        tm.assert_numpy_array_equal(out["rng"].to_numpy(), expected["rng"].to_numpy())

    def test_mixed_simple_and_compound_agg(self):
        df = pd.DataFrame({"g": [1, 1, 2], "v1": [4, 6, 8], "v2": [1, 2, 3]})
        out = self._c(
            df.select()
            .group_by("g")
            .agg(
                col("v1").sum().alias("s"),
                (col("v1").max() - col("v2").min()).alias("rng"),
            )
        )
        assert set(out.columns) == {"g", "s", "rng"}
        assert "__agg_0__" not in out.columns  # temp columns dropped

    def test_aggregate_of_computed_expression(self):
        # sum(v1 * v2) per group - aggregate over a computed input
        df = pd.DataFrame({"g": [1, 1, 2], "v1": [2, 3, 4], "v2": [5, 6, 7]})
        out = (
            self._c(
                df.select().group_by("g").agg((col("v1") * col("v2")).sum().alias("p"))
            )
            .sort_values("g")
            .reset_index(drop=True)
        )
        expected = (df["v1"] * df["v2"]).groupby(df["g"]).sum().reset_index(name="p")
        tm.assert_numpy_array_equal(out["p"].to_numpy(), expected["p"].to_numpy())

    def test_grouped_corr_matches_eager(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "g": rng.integers(0, 5, 500),
                "x": rng.normal(0, 1, 500),
                "y": rng.normal(0, 1, 500),
            }
        )
        out = (
            self._c(df.select().group_by("g").agg(col("x").corr(col("y")).alias("r")))
            .sort_values("g")
            .reset_index(drop=True)
        )
        expected = (
            df.groupby("g")
            .apply(lambda d: d["x"].corr(d["y"]), include_groups=False)
            .reset_index(name="r")
        )
        assert np.allclose(out["r"].to_numpy(), expected["r"].to_numpy(), atol=1e-9)


class TestGroupedMedian:
    """Exact grouped median (NumPy backend; no Arrow hash_median kernel)."""

    def _c(self, ldf):
        return ldf.collect(use_physical_planner=True, order="relaxed")

    def test_single_key_median(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"g": rng.integers(0, 6, 2000), "v": rng.uniform(0, 100, 2000)}
        )
        out = (
            self._c(df.select().group_by("g").agg(col("v").median().alias("m")))
            .sort_values("g")
            .reset_index(drop=True)
        )
        expected = df.groupby("g")["v"].median()
        assert np.allclose(out["m"].to_numpy(), expected.to_numpy(), atol=1e-9)

    def test_multi_key_median_and_std(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "a": rng.integers(0, 5, 3000),
                "b": rng.integers(0, 5, 3000),
                "v": rng.uniform(0, 100, 3000),
            }
        )
        out = (
            self._c(
                df.select()
                .group_by("a", "b")
                .agg(col("v").median().alias("m"), col("v").std().alias("s"))
            )
            .sort_values(["a", "b"])
            .reset_index(drop=True)
        )
        expected = (
            df.groupby(["a", "b"]).agg(m=("v", "median"), s=("v", "std")).reset_index()
        )
        assert np.allclose(out["m"].to_numpy(), expected["m"].to_numpy(), atol=1e-9)
        assert np.allclose(out["s"].to_numpy(), expected["s"].to_numpy(), atol=1e-9)


class TestGroupByHead:
    """Group-wise head, and its sort+head composition for top-k per group."""

    def _c(self, ldf):
        return ldf.collect(use_physical_planner=True, order="relaxed")

    def test_head_per_group_preserves_input_order(self):
        df = pd.DataFrame({"g": [1, 1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50, 60]})
        out = self._c(df.select().group_by("g").head(2))
        # first 2 rows of each group in input order
        got = set(map(tuple, out[["g", "v"]].to_numpy()))
        assert got == {(1, 10), (1, 20), (2, 40), (2, 50), (3, 60)}

    def test_top_k_per_group_via_sort_then_head(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"g": rng.integers(0, 50, 5000), "v": rng.uniform(0, 100, 5000)}
        )
        out = self._c(df.select().sort("v", descending=True).group_by("g").head(2))
        expected = df.sort_values("v", ascending=False).groupby("g", sort=False).head(2)
        got = set(map(tuple, np.round(out[["g", "v"]].to_numpy(), 9)))
        want = set(map(tuple, np.round(expected[["g", "v"]].to_numpy(), 9)))
        assert got == want
        assert len(out) == len(expected)

    def test_head_default_eager_collect(self):
        # The default collect() is eager; group_by().head() must not raise
        # NotImplementedError there.
        df = pd.DataFrame({"g": [1, 1, 1, 2, 2], "v": [10, 20, 30, 40, 50]})
        out = df.select().group_by("g").head(2).collect()
        assert out["v"].tolist() == [10, 20, 40, 50]

    def test_head_computed_key_both_engines(self):
        # A computed group key must be materialized so neither engine looks up
        # an unmaterialized column name.
        df = pd.DataFrame({"g": [1, 1, 2, 2], "v": [10, 20, 30, 40]})
        ldf = df.select().group_by((col("g") * 10).alias("gk")).head(1)
        eager = ldf.collect().sort_values("gk").reset_index(drop=True)
        phys = (
            ldf.collect(use_physical_planner=True)
            .sort_values("gk")
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(eager, phys, check_dtype=False)
        assert eager["gk"].tolist() == [10, 20]


class TestCorrPairwiseComplete:
    """Regression: corr must use pairwise-complete pairs (drop a row when
    either input is null), matching pandas.Series.corr, in both engines."""

    def test_corr_nulls_in_different_rows(self):
        df = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, None], "y": [1.0, 2.0, 3.0, None, 5.0]}
        )
        for phys in (False, True):
            got = float(
                df.select(col("x").corr(col("y")).alias("r"))
                .collect(use_physical_planner=phys)["r"]
                .iloc[0]
            )
            assert np.isclose(got, df["x"].corr(df["y"]), rtol=1e-9)

    def test_corr_no_nulls_matches(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = 0.5 * x + rng.standard_normal(200)
        df = pd.DataFrame({"x": x, "y": y})
        for phys in (False, True):
            got = float(
                df.select(col("x").corr(col("y")).alias("r"))
                .collect(use_physical_planner=phys)["r"]
                .iloc[0]
            )
            assert np.isclose(got, df["x"].corr(df["y"]), rtol=1e-9)


class TestComputedGroupKeys:
    """Regression: a computed group key must be materialized before grouping."""

    def test_group_by_computed_key_both_engines(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "v": [10, 20, 30, 40, 50]})
        q = (
            df.select()
            .group_by((col("a") % 2).alias("parity"))
            .agg(col("v").sum().alias("t"))
        )
        expected = (
            df.assign(parity=df["a"] % 2)
            .groupby("parity")["v"]
            .sum()
            .reset_index()
            .rename(columns={"v": "t"})
        )
        for phys in (False, True):
            out = (
                q.collect(use_physical_planner=phys)
                .sort_values("parity")
                .reset_index(drop=True)
            )
            assert out["t"].tolist() == expected["t"].tolist()

    def test_group_by_computed_key_with_composed_agg(self):
        # Computed key + a non-trivial (composed) aggregate exercises the
        # pre-project decomposition path too.
        df = pd.DataFrame({"a": [1, 2, 3, 4], "x": [1.0, 2, 3, 4], "y": [4.0, 3, 2, 1]})
        q = (
            df.select()
            .group_by((col("a") % 2).alias("p"))
            .agg((col("x").sum() - col("y").sum()).alias("d"))
        )
        out = q.collect().sort_values("p").reset_index(drop=True)
        exp = (
            df.assign(p=df["a"] % 2)
            .groupby("p")
            .apply(lambda g: g["x"].sum() - g["y"].sum())
        )
        assert out["d"].tolist() == [exp.loc[0], exp.loc[1]]


class TestAggregateNaNSkipping:
    """Regression: physical reducing aggregates must skip NaN (not just null)
    to match pandas, deterministically — the cached-Arrow path counted/summed
    a float NaN as present, hash-routing-dependently."""

    def test_count_over_nan_column(self):
        from pandas.lazy import (
            lit,
            when,
        )

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None], "y": [1.0, 2, 3, None, 5]})
        xp = when(col("y").is_null()).then(lit(float("nan"))).otherwise(col("x"))
        for phys in (False, True):
            n = df.select(xp.count().alias("n")).collect(use_physical_planner=phys)
            assert int(n["n"].iloc[0]) == 3

    def test_scalar_aggregate_arithmetic_over_computed_column(self):
        from pandas.lazy import (
            lit,
            when,
        )

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None], "y": [1.0, 2, 3, None, 5]})
        xp = when(col("y").is_null()).then(lit(float("nan"))).otherwise(col("x"))
        r = df.select((xp.count() * xp.sum()).alias("r")).collect(
            use_physical_planner=True
        )
        assert float(r["r"].iloc[0]) == 3 * 6.0  # count=3, sum=1+2+3
