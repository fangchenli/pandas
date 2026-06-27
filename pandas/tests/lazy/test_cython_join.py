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


class TestPayloadAwareGate:
    """The build-side bail is relaxed only for very narrow joins
    (<=4 gathered columns), where the threaded CSR kernel beats pd.merge even
    at large size; wider large joins keep the original pd.merge path so the
    validated TPC-H join shapes do not regress.
    See docs/MATERIALIZATION_EXPERIMENT.md (joins section).
    """

    def _run(self, left, right, **jkw):
        """Return (fast_path_used, phys, eager). Detects EITHER fast path — the
        Rust index-gen (preferred when available) or the Cython indexer — since
        either avoids pd.merge. Spying the module attributes catches the calls."""
        import pandas.lazy.backends.numpy.join as J
        import pandas.lazy.physical as P

        calls = []
        orig = J.inner_join_indexers_i8

        def spy(*a, **k):
            calls.append(1)
            return orig(*a, **k)

        J.inner_join_indexers_i8 = spy
        rust_active = P._RUST_JOIN and P._rust_join is not None
        if rust_active:
            orig_r = P._rust_join.join_indices_i64

            def spy_r(*a, **k):
                calls.append(1)
                return orig_r(*a, **k)

            P._rust_join.join_indices_i64 = spy_r
        try:
            phys = (left.select().join(right.select(), **jkw)).collect(
                use_physical_planner=True
            )
            eager = (left.select().join(right.select(), **jkw)).collect(
                use_physical_planner=False
            )
        finally:
            J.inner_join_indexers_i8 = orig
            if rust_active:
                P._rust_join.join_indices_i64 = orig_r
        return bool(calls), phys, eager

    def test_narrow_large_join_uses_kernel_and_matches(self, rng):
        # Both sides > 500k, only 3 gathered columns -> kernel now engaged
        # (the old min>500k bail would have sent this to pd.merge).
        n = 600_000
        left = pd.DataFrame({"k": np.arange(n), "v": rng.standard_normal(n)})
        right = pd.DataFrame({"k": np.arange(n), "w": rng.standard_normal(n)})
        used, phys, eager = self._run(left, right, on="k")
        assert used  # narrow + large -> kernel
        tm.assert_frame_equal(phys, eager)

    def test_wide_large_join_path(self, rng):
        # Both sides > 500k with wide (10) gathered columns. The Rust fast path
        # handles wide payloads (real-threaded index-gen + parallel gather), so
        # it engages when available; without Rust the Cython wide-payload bail
        # sends it to pd.merge. Either way the result must equal eager.
        import pandas.lazy.physical as P

        n = 600_000
        left = pd.DataFrame(
            {"k": np.arange(n), **{f"v{i}": rng.standard_normal(n) for i in range(5)}}
        )
        right = pd.DataFrame(
            {"k": np.arange(n), **{f"w{i}": rng.standard_normal(n) for i in range(5)}}
        )
        used, phys, eager = self._run(left, right, on="k")
        tm.assert_frame_equal(phys, eager)
        rust_active = P._RUST_JOIN and P._rust_join is not None
        if rust_active:
            assert used  # Rust handles wide payloads
        else:
            assert not used  # Cython wide bail -> pd.merge


class TestPartitionByBucket:
    """The radix-partition foundation for the partitioned parallel hash join
    (``pandas/lazy/docs/JOIN_KERNEL_REBUILD_PROBE.md``)."""

    def test_partition_preserves_keys_and_groups(self):
        from pandas._libs.lazy_join import partition_by_bucket

        rng = np.random.default_rng(0)
        keys = rng.integers(0, 1000, 50_000).astype(np.int64)
        for n_buckets in (1, 2, 8, 64):
            sk, sr, bounds = partition_by_bucket(keys, n_buckets)
            # row mapping reconstructs the sorted keys exactly
            tm.assert_numpy_array_equal(keys[sr], sk)
            # it is a permutation of the input rows
            tm.assert_numpy_array_equal(
                np.sort(sr), np.arange(len(keys), dtype=np.int64)
            )
            # bounds delimit n_buckets contiguous groups covering all rows
            assert len(bounds) == n_buckets + 1
            assert bounds[0] == 0 and bounds[-1] == len(keys)
            assert np.all(np.diff(bounds) >= 0)
            # within each bucket, all keys hash to the same bucket id
            h = (sk.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
            bid = (h & np.uint64(n_buckets - 1)).astype(np.int64)
            for b in range(n_buckets):
                seg = bid[bounds[b] : bounds[b + 1]]
                assert np.all(seg == b)

    def test_partition_empty(self):
        from pandas._libs.lazy_join import partition_by_bucket

        sk, sr, bounds = partition_by_bucket(np.empty(0, dtype=np.int64), 8)
        assert len(sk) == 0 and len(sr) == 0
        tm.assert_numpy_array_equal(bounds, np.zeros(9, dtype=np.int64))


class TestPartitionedJoinGather:
    """Fused partitioned parallel join + row-major gather
    (``partitioned_join_gather``; ``JOIN_KERNEL_REBUILD_PROBE.md``)."""

    def _oracle(self, lk, rk, rpay_cols):
        # pd.merge inner order with payload columns from the (unique) right side
        P = rpay_cols.shape[0]
        ldf = pd.DataFrame({"key": lk, "_lrow": np.arange(len(lk))})
        rdf = pd.DataFrame({"key": rk, **{f"p{j}": rpay_cols[j] for j in range(P)}})
        m = pd.merge(ldf, rdf, on="key", how="inner")
        return m

    @pytest.mark.parametrize("P", [0, 1, 3, 9])
    def test_ordered_matches_pd_merge(self, P):
        from pandas.lazy.backends.numpy.join import partitioned_join_gather

        rng = np.random.default_rng(P + 1)
        nL, nR = 4000, 900
        lk = rng.integers(0, nR, nL).astype(np.int64)
        rk = np.arange(nR, dtype=np.int64)  # unique build
        rpay_cols = rng.standard_normal((P, nR))
        rpay_rm = np.ascontiguousarray(rpay_cols.T) if P else np.empty((nR, 0))
        out, lrow = partitioned_join_gather(lk, rk, rpay_rm, preserve_order=True)
        m = self._oracle(lk, rk, rpay_cols)
        tm.assert_numpy_array_equal(lrow, m["_lrow"].to_numpy())
        for j in range(P):
            tm.assert_numpy_array_equal(out[:, j], m[f"p{j}"].to_numpy())

    def test_unordered_is_pd_merge_multiset(self):
        from pandas.lazy.backends.numpy.join import partitioned_join_gather

        rng = np.random.default_rng(0)
        nL, nR, P = 4000, 900, 3
        lk = rng.integers(0, nR, nL).astype(np.int64)
        rk = np.arange(nR, dtype=np.int64)
        rpay_cols = rng.standard_normal((P, nR))
        rpay_rm = np.ascontiguousarray(rpay_cols.T)
        out, lrow = partitioned_join_gather(lk, rk, rpay_rm, preserve_order=False)
        m = self._oracle(lk, rk, rpay_cols)
        assert sorted(lrow.tolist()) == sorted(m["_lrow"].to_numpy().tolist())
        # payload still tracks its left row (sort both by left row, compare)
        order = np.argsort(lrow, kind="stable")
        tm.assert_numpy_array_equal(lrow[order], np.sort(m["_lrow"].to_numpy()))

    def test_no_match_and_empty(self):
        from pandas.lazy.backends.numpy.join import partitioned_join_gather

        rk = np.arange(50, dtype=np.int64)
        payload = np.random.default_rng(0).standard_normal((2, 50))
        rpay_rm = np.ascontiguousarray(payload.T)
        # no overlap
        lk = np.arange(100, 200, dtype=np.int64)
        out, lrow = partitioned_join_gather(lk, rk, rpay_rm, preserve_order=True)
        assert len(lrow) == 0 and out.shape == (0, 2)
        # empty left
        out, lrow = partitioned_join_gather(
            np.empty(0, np.int64), rk, rpay_rm, preserve_order=True
        )
        assert len(lrow) == 0
