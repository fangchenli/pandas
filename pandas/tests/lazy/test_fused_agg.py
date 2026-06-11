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
