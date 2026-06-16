"""Fused filter+aggregate kernel path (PhysicalFusedFilterAgg)."""

import numpy as np

import pandas as pd
from pandas.lazy import col


class TestFusedFilterAgg:
    def _frame(self, n=600_000):
        rng = np.random.default_rng(11)
        return pd.DataFrame(
            {
                "dt": pd.to_datetime("2020-01-01")
                + pd.to_timedelta(rng.integers(0, 720, n), unit="D"),
                "disc": rng.uniform(0, 0.1, n),
                "qty": rng.uniform(0, 50, n),
                "px": rng.uniform(900, 10_000, n),
            }
        )

    def _plan(self, df):
        # The q6 shape: filter -> project(product) -> frame-level sum.
        return (
            df.select()
            .filter(
                (col("dt") >= pd.Timestamp("2020-06-01"))
                & (col("dt") < pd.Timestamp("2021-01-01"))
                & (col("disc") >= 0.05)
                & (col("disc") <= 0.07)
                & (col("qty") < 24)
            )
            .select((col("px") * col("disc")).alias("rev"))
            .sum()
        )

    def test_takes_fused_path_and_matches_eager(self):
        df = self._frame()
        plan = self._plan(df)
        assert "FusedFilterAgg" in plan.explain(physical=True)
        phys = float(plan.collect(use_physical_planner=True)["rev"].iloc[0])
        eager = float(plan.collect(use_physical_planner=False)["rev"].iloc[0])
        assert np.isclose(phys, eager, rtol=1e-12)

    def test_int_literal_on_float_column_is_exact(self):
        # `qty < 24` with an int literal over a float64 column must use
        # float range semantics (hi = nextafter(24)), not hi = 23.
        df = self._frame(100_000)
        df.loc[:999, "qty"] = 23.5  # rows that int semantics would drop
        plan = df.select().filter(col("qty") < 24).select(col("px").alias("s")).sum()
        assert "FusedFilterAgg" in plan.explain(physical=True)
        phys = float(plan.collect(use_physical_planner=True)["s"].iloc[0])
        expected = float(df.loc[df["qty"] < 24, "px"].sum())
        assert np.isclose(phys, expected, rtol=1e-12)

    def test_nan_in_agg_column_falls_back(self):
        df = self._frame(50_000)
        df.loc[100, "px"] = np.nan
        plan = df.select().filter(col("qty") < 24).select(col("px").alias("s")).sum()
        phys = float(plan.collect(use_physical_planner=True)["s"].iloc[0])
        eager = float(plan.collect(use_physical_planner=False)["s"].iloc[0])
        assert np.isclose(phys, eager, rtol=1e-12)


class TestScalarAggSelectFusion:
    """Shape B: ungrouped `filter().select(col.agg())` (a reducing project at
    the tail of the fused pipeline, not a HashAggregate) must reach the fused
    kernel rather than the row-compacting generic pipeline.
    See docs/MATERIALIZATION_EXPERIMENT.md.
    """

    def _frame(self, n=300_000):
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "a": rng.standard_normal(n),
                "b": rng.standard_normal(n),
                "i": rng.integers(0, 1000, n).astype(np.int64),
            }
        )

    def test_select_sum_routes_to_kernel_and_matches(self):
        df = self._frame()
        a = df["a"].to_numpy()
        plan = df.select().filter(col("a") > 0).select(col("a").sum().alias("s"))
        assert "FusedFilterAgg" in plan.explain(physical=True)
        got = float(plan.collect(use_physical_planner=True)["s"].iloc[0])
        assert np.isclose(got, float(a[a > 0].sum()), rtol=1e-12)

    def test_select_count_multi_predicate(self):
        df = self._frame()
        a, b, i = (df[c].to_numpy() for c in ("a", "b", "i"))
        plan = (
            df.select()
            .filter((col("a") > 0) & (col("i") < 500) & (col("b") < 0.5))
            .select(col("a").count().alias("n"))
        )
        assert "FusedFilterAgg" in plan.explain(physical=True)
        got = int(plan.collect(use_physical_planner=True)["n"].iloc[0])
        assert got == int(((a > 0) & (i < 500) & (b < 0.5)).sum())

    def test_select_mean_and_product_sum(self):
        df = self._frame()
        a, b = df["a"].to_numpy(), df["b"].to_numpy()
        mean_plan = df.select().filter(col("a") > 0).select(col("a").mean().alias("m"))
        prod_plan = (
            df.select()
            .filter(col("a") > 0)
            .select((col("a") * col("b")).sum().alias("p"))
        )
        assert "FusedFilterAgg" in mean_plan.explain(physical=True)
        assert "FusedFilterAgg" in prod_plan.explain(physical=True)
        m = float(mean_plan.collect(use_physical_planner=True)["m"].iloc[0])
        p = float(prod_plan.collect(use_physical_planner=True)["p"].iloc[0])
        assert np.isclose(m, float(a[a > 0].mean()), rtol=1e-12)
        assert np.isclose(p, float((a * b)[a > 0].sum()), rtol=1e-9)

    def test_select_agg_no_filter_stays_correct(self):
        # No filter -> no compaction to eliminate, so this is intentionally
        # left on the reducing-project path (not rerouted); still correct.
        df = self._frame()
        a = df["a"].to_numpy()
        plan = df.select(col("a").sum().alias("s"))
        got = float(plan.collect(use_physical_planner=True)["s"].iloc[0])
        assert np.isclose(got, float(a.sum()), rtol=1e-12)

    def test_mixed_agg_and_passthrough_not_fused(self):
        # A select mixing an aggregate with a bare column is not an ungrouped
        # reduction; it must not be rerouted to the scalar-agg kernel.
        df = self._frame(1000)
        plan = (
            df.select().filter(col("a") > 0).select(col("a").sum().alias("s"), col("b"))
        )
        assert "FusedFilterAgg" not in plan.explain(physical=True)
