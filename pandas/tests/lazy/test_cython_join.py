"""
Tests for the Cython single-pass hash join (pandas._libs.lazy_join) and its
engine integration (PhysicalHashJoin._try_cython_join).

The contract under test: for eligible joins (inner, single int key) the
physical engine's output is EXACTLY pd.merge's, row order included — whether
the kernel runs in its natural direction (build right, probe left) or swapped
(build on a much smaller left, with the stable regroup restoring pd.merge's
left-row-major order).
"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestKernelIndexers:
    """Kernel-level: indexers match pd.merge's join result exactly."""

    @pytest.mark.parametrize(
        "n_left,n_right,key_range",
        [
            (1000, 800, 50),  # dup x dup
            (5000, 1000, 1000),  # right unique-ish
            (500, 20000, 800),  # left small
            (3000, 3000, 1000),  # negatives mixed in below
        ],
    )
    def test_indexers_match_pd_merge(self, rng, n_left, n_right, key_range):
        from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

        lk = rng.integers(-key_range // 2, key_range, n_left).astype(np.int64)
        rk = rng.integers(-key_range // 2, key_range, n_right).astype(np.int64)
        li, ri = inner_join_indexers_i8(lk, rk)
        left = pd.DataFrame({"k": lk, "li": np.arange(n_left)})
        right = pd.DataFrame({"k": rk, "ri": np.arange(n_right)})
        ref = left.merge(right, on="k", how="inner")
        assert len(ref) == len(li)
        assert (ref["li"].to_numpy() == li).all()
        assert (ref["ri"].to_numpy() == ri).all()

    def test_empty_sides(self):
        from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

        a = np.arange(10, dtype=np.int64)
        e = np.empty(0, dtype=np.int64)
        for lk, rk in [(a, e), (e, a), (e, e)]:
            li, ri = inner_join_indexers_i8(lk, rk)
            assert len(li) == 0 and len(ri) == 0

    def test_parallel_path_matches_serial(self, rng):
        from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

        lk = rng.integers(0, 100_000, 1_200_000).astype(np.int64)
        rk = np.arange(100_000, dtype=np.int64)
        li, ri = inner_join_indexers_i8(lk, rk)
        assert len(li) == len(lk)  # every probe hits
        assert (lk == rk[ri]).all()
        assert (li == np.arange(len(lk))).all()  # left order preserved

    def test_selectivity_bail(self, rng):
        from pandas.lazy.backends.numpy.join import inner_join_indexers_i8

        lk = rng.integers(0, 1000, 300_000).astype(np.int64)
        rk = np.arange(1000, dtype=np.int64)  # 100% hit rate
        assert inner_join_indexers_i8(lk, rk, max_hit_fraction=0.5) is None


class TestEngineParity:
    """Engine-level: physical output equals eager (pd.merge) exactly."""

    def _check(self, left, right, **jkw):
        plan_l = left.select().join(right.select(), **jkw)
        phys = plan_l.collect(use_physical_planner=True)
        eager = plan_l.collect(use_physical_planner=False)
        tm.assert_frame_equal(phys, eager)

    def test_right_unique(self, rng):
        left = pd.DataFrame(
            {"k": rng.integers(0, 1000, 5000), "v": rng.standard_normal(5000)}
        )
        right = pd.DataFrame({"rk": np.arange(1000), "w": rng.standard_normal(1000)})
        self._check(left, right, left_on="k", right_on="rk")

    def test_swap_path_left_small(self, rng):
        left = pd.DataFrame({"lk": np.arange(500) + 100, "v": rng.standard_normal(500)})
        right = pd.DataFrame(
            {"k2": rng.integers(0, 800, 20000), "w": rng.standard_normal(20000)}
        )
        self._check(left, right, left_on="lk", right_on="k2")

    def test_dup_dup_shared_key(self, rng):
        a = pd.DataFrame(
            {"k": rng.integers(0, 50, 3000), "v": rng.standard_normal(3000)}
        )
        b = pd.DataFrame(
            {"k": rng.integers(0, 50, 2500), "w": rng.standard_normal(2500)}
        )
        self._check(a, b, on="k")

    def test_misses_and_negative_keys(self, rng):
        left = pd.DataFrame(
            {"k": rng.integers(-50, 2000, 3000), "v": rng.standard_normal(3000)}
        )
        right = pd.DataFrame({"rk": np.arange(1000), "w": rng.standard_normal(1000)})
        self._check(left, right, left_on="k", right_on="rk")

    def test_string_payload_rides_through(self, rng):
        left = pd.DataFrame(
            {
                "k": rng.integers(0, 1000, 5000),
                "s": rng.choice(["aa", "bb", "cc"], 5000),
            }
        )
        right = pd.DataFrame({"rk": np.arange(1000), "w": rng.standard_normal(1000)})
        self._check(left, right, left_on="k", right_on="rk")

    def test_int32_keys(self, rng):
        left = pd.DataFrame(
            {
                "k": rng.integers(0, 100, 4000).astype("int32"),
                "v": np.ones(4000),
            }
        )
        right = pd.DataFrame({"rk": np.arange(100, dtype="int32"), "w": np.ones(100)})
        self._check(left, right, left_on="k", right_on="rk")

    def test_empty_result(self):
        left = pd.DataFrame({"k": np.arange(10) + 5000, "v": np.ones(10)})
        right = pd.DataFrame({"rk": np.arange(1000), "w": np.ones(1000)})
        self._check(left, right, left_on="k", right_on="rk")


class TestJoinChainComposition:
    """Late-materialization join chains (PhysicalHashJoin._execute_composite).

    Chains of inner int-key joins compose kernel indexers instead of
    gathering intermediate payloads; output must equal the eager cascade
    exactly (order included), and ineligible steps must degrade gracefully.
    """

    def _check(self, plan):
        phys = plan.collect(use_physical_planner=True)
        eager = plan.collect(use_physical_planner=False)
        tm.assert_frame_equal(phys, eager)

    def test_three_table_chain(self, rng):
        a = pd.DataFrame(
            {"k1": rng.integers(0, 300, 4000), "v": rng.standard_normal(4000)}
        )
        b = pd.DataFrame(
            {
                "k1": np.arange(300),
                "k2": rng.integers(0, 100, 300),
                "w": rng.standard_normal(300),
            }
        )
        c = pd.DataFrame({"k2": np.arange(100), "x": rng.standard_normal(100)})
        self._check(a.select().join(b.select(), on="k1").join(c.select(), on="k2"))

    def test_chain_with_duplicate_keys(self, rng):
        a = pd.DataFrame(
            {"k1": rng.integers(0, 300, 4000), "v": rng.standard_normal(4000)}
        )
        b = pd.DataFrame(
            {
                "k1": np.arange(300),
                "k2": rng.integers(0, 50, 300),
                "w": rng.standard_normal(300),
            }
        )
        d = pd.DataFrame(
            {"k3": rng.integers(0, 50, 5000), "y": rng.standard_normal(5000)}
        )
        self._check(
            a.select()
            .join(b.select(), on="k1")
            .join(d.select(), left_on="k2", right_on="k3")
        )

    def test_ineligible_step_degrades_gracefully(self, rng):
        a = pd.DataFrame(
            {"k1": rng.integers(0, 300, 4000), "v": rng.standard_normal(4000)}
        )
        e = pd.DataFrame(
            {
                "s": rng.choice(["p", "q", "r"], 300),
                "k1": np.arange(300),
                "z": np.ones(300),
            }
        )
        f = pd.DataFrame({"s": ["p", "q"], "t": [1.0, 2.0]})
        self._check(a.select().join(e.select(), on="k1").join(f.select(), on="s"))

    def test_chain_into_groupby(self, rng):
        a = pd.DataFrame(
            {"k1": rng.integers(0, 300, 4000), "v": rng.standard_normal(4000)}
        )
        b = pd.DataFrame(
            {
                "k1": np.arange(300),
                "k2": rng.integers(0, 100, 300),
                "w": rng.standard_normal(300),
            }
        )
        c = pd.DataFrame({"k2": np.arange(100), "x": rng.standard_normal(100)})
        plan = (
            a.select()
            .join(b.select(), on="k1")
            .join(c.select(), on="k2")
            .group_by("k2")
            .agg(col("v").sum().alias("sv"), col("x").mean().alias("mx"))
        )
        phys = plan.collect(use_physical_planner=True).sort_values("k2")
        eager = plan.collect(use_physical_planner=False).sort_values("k2")
        tm.assert_frame_equal(phys.reset_index(drop=True), eager.reset_index(drop=True))


class TestTwoKeyPackedJoin:
    """Composite (two-int-key) joins pack into one int64 key and use the
    Cython kernel + chains (G2). Output must equal pd.merge exactly."""

    def _check(self, left, right, **kw):
        plan = left.select().join(right.select(), **kw)
        tm.assert_frame_equal(
            plan.collect(use_physical_planner=True),
            plan.collect(use_physical_planner=False),
        )

    def test_two_key_on_dup_dup(self, rng):
        left = pd.DataFrame(
            {
                "a": rng.integers(0, 50, 4000),
                "b": rng.integers(0, 30, 4000),
                "v": rng.standard_normal(4000),
            }
        )
        right = pd.DataFrame(
            {
                "a": rng.integers(0, 50, 3000),
                "b": rng.integers(0, 30, 3000),
                "w": rng.standard_normal(3000),
            }
        )
        self._check(left, right, on=["a", "b"])

    def test_two_key_left_right_names_and_negatives(self, rng):
        left = pd.DataFrame(
            {
                "a": rng.integers(-25, 25, 4000),
                "b": rng.integers(0, 30, 4000),
                "v": rng.standard_normal(4000),
            }
        )
        right = pd.DataFrame(
            {
                "x": rng.integers(-25, 25, 3000),
                "y": rng.integers(0, 30, 3000),
                "w": rng.standard_normal(3000),
            }
        )
        self._check(left, right, left_on=["a", "b"], right_on=["x", "y"])

    def test_two_key_chain_middle_step(self, rng):
        left = pd.DataFrame(
            {
                "a": rng.integers(0, 50, 4000),
                "b": rng.integers(0, 30, 4000),
                "v": rng.standard_normal(4000),
            }
        )
        mid = pd.DataFrame(
            {
                "x": rng.integers(0, 50, 3000),
                "y": rng.integers(0, 30, 3000),
                "w": rng.standard_normal(3000),
            }
        )
        last = pd.DataFrame(
            {
                "a": rng.integers(0, 50, 2000),
                "b": rng.integers(0, 30, 2000),
                "z": np.ones(2000),
            }
        )
        plan = (
            left.select()
            .join(mid.select(), left_on=["a", "b"], right_on=["x", "y"])
            .join(last.select(), on=["a", "b"])
        )
        tm.assert_frame_equal(
            plan.collect(use_physical_planner=True),
            plan.collect(use_physical_planner=False),
        )

    def test_huge_range_falls_back(self, rng):
        # Packed span would exceed int63 -> falls back to pd.merge, still exact.
        left = pd.DataFrame(
            {
                "a": rng.integers(0, 2**40, 1000),
                "b": rng.integers(0, 2**40, 1000),
                "v": np.ones(1000),
            }
        )
        right = pd.DataFrame(
            {
                "a": rng.integers(0, 2**40, 1000),
                "b": rng.integers(0, 2**40, 1000),
                "w": np.ones(1000),
            }
        )
        self._check(left, right, on=["a", "b"])
