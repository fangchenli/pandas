"""
Tests for the parallel driver behind :meth:`Index._inner_indexer`.

The threshold is lowered in these tests so the parallel path actually runs;
with the shipped threshold it would take a multi-million-row index to reach.
"""

import numpy as np
import pytest

from pandas._libs import join as libjoin
from pandas.compat import WASM

import pandas as pd
from pandas import Index
import pandas._testing as tm
from pandas.core.array_algos import join as array_algos_join
from pandas.core.util.threading import max_workers


@pytest.fixture
def assert_parallel_ran(monkeypatch):
    """
    Force the parallel path on and fail if it did not actually run.

    _n_workers() is patched directly rather than _MAX_DEFAULT_WORKERS: the
    latter is combined with os.cpu_count(), so on a one- or two-core CI runner
    the worker count would collapse to 1, _can_parallelize would decline, and
    every test here would compare the serial kernel against itself and pass
    without exercising any chunking.
    """
    if WASM:
        pytest.skip("WASM runtimes cannot start threads")

    # disarmed until use() is called: a test that skips before reaching it --
    # importorskip on a missing optional dependency, say -- would otherwise
    # fail this assertion during teardown and be reported as an error
    state = {"calls": 0, "expect_parallel": False}
    real = libjoin.inner_join_count_range

    def spy(*args):
        state["calls"] += 1
        return real(*args)

    monkeypatch.setattr(array_algos_join, "_PARALLEL_JOIN_MIN_ROWS", 4)
    # the shipped dispatch floor is tens of thousands of iterations, which
    # would swamp these fixtures the way the size threshold would; 1 keeps the
    # scoring comparison intact without letting the constant dominate it
    monkeypatch.setattr(array_algos_join, "_DISPATCH_COST_ITERS", 1)
    monkeypatch.setattr(libjoin, "inner_join_count_range", spy)

    def use(n_workers, *, expect_parallel: bool = True):
        monkeypatch.setattr(array_algos_join, "max_workers", lambda cap: n_workers)
        # _n_chunks scales with the input, so these small fixtures would all
        # split in two; pin it so the parametrised counts are really exercised
        monkeypatch.setattr(
            array_algos_join, "_n_chunks", lambda n, workers: min(n_workers, workers)
        )
        state["expect_parallel"] = expect_parallel

    yield use
    if state["expect_parallel"]:
        assert state["calls"], "the parallel path never ran; this test proved nothing"
    else:
        assert not state["calls"], "expected the serial fallback, but chunks ran"


@pytest.mark.parametrize("dtype", ["int64", "int32", "uint64", "float64"])
@pytest.mark.parametrize("n_workers", [2, 3, 4, 8])
@pytest.mark.parametrize(
    "left_vals, right_vals",
    [
        (range(64), range(64)),
        (range(64), range(32, 96)),
        (range(0, 64, 2), range(1, 64, 2)),  # no matches, overlapping ranges
        (range(32), range(100, 132)),  # disjoint ranges
        ([1] * 40 + [2] * 40, [1] * 20 + [2] * 60),  # long runs of equal keys
        ([1] * 40 + [3] * 40, [1] * 40 + [2] * 40 + [3] * 40),  # gap on the right
    ],
)
def test_parallel_matches_serial(
    left_vals, right_vals, dtype, n_workers, assert_parallel_ran
):
    assert_parallel_ran(n_workers)
    left = np.array(list(left_vals), dtype=dtype)
    right = np.array(list(right_vals), dtype=dtype)

    expected = libjoin.inner_join_indexer(left, right)
    result = array_algos_join.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


def _strided_pair(layout: str, dtype: str):
    """Two ascending join targets in the requested memory layout."""
    base = np.arange(256, dtype=dtype)
    if layout == "step":
        return base[::2], base[::3]
    if layout == "mixed_steps":
        return base[::4], base[::6]
    if layout == "offset":
        return base[7::3], base[11::5]
    if layout == "negative_stride":
        # stored descending, viewed ascending: strides[0] is negative
        desc = np.arange(255, -1, -1, dtype=dtype)
        return desc[::-1], desc[::-2]
    if layout == "column_of_2d":
        return np.arange(256, dtype=dtype).reshape(16, 16)[:, 0], base[::8]
    raise AssertionError(layout)


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("n_workers", [2, 4])
@pytest.mark.parametrize(
    "layout", ["step", "mixed_steps", "offset", "negative_stride", "column_of_2d"]
)
def test_strided_inputs_match_serial(layout, n_workers, dtype, assert_parallel_ran):
    assert_parallel_ran(n_workers)
    left, right = _strided_pair(layout, dtype)
    assert not (left.flags.c_contiguous and right.flags.c_contiguous)

    # the oracle takes contiguous copies, so a stride bug cannot cancel out
    expected = libjoin.inner_join_indexer(
        np.ascontiguousarray(left), np.ascontiguousarray(right)
    )
    result = array_algos_join.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


@pytest.mark.parametrize("n_workers", [2, 4, 8])
def test_single_key_falls_back_to_serial(n_workers, assert_parallel_ran):
    # one run of equal keys spanning the whole of left cannot be split without
    # cutting the run, so the driver must hand off to the serial indexer
    assert_parallel_ran(n_workers, expect_parallel=False)
    values = np.array([1] * 80, dtype=np.int64)

    result = array_algos_join.inner_join_indexer(values, values)
    expected = libjoin.inner_join_indexer(values, values)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_index_intersection_parallel(dtype, assert_parallel_ran):
    assert_parallel_ran(2)
    # the user-facing path: Index.intersection goes through _inner_indexer
    left = Index(np.arange(0, 200, 2, dtype=dtype))
    right = Index(np.arange(0, 200, 3, dtype=dtype))

    result = left.intersection(right)
    expected = Index(np.intersect1d(left.to_numpy(), right.to_numpy()), dtype=dtype)
    tm.assert_index_equal(result, expected)


def test_index_join_inner_parallel(assert_parallel_ran):
    left = Index(np.arange(0, 200, 2))
    right = Index(np.arange(0, 200, 3))

    # The fixture lowers the threshold during setup, so a plain call here would
    # already take the parallel path; go through the serial kernel explicitly.
    serial_values, serial_lidx, serial_ridx = libjoin.inner_join_indexer(
        left._values, right._values
    )

    assert_parallel_ran(2)
    result = left.join(right, how="inner", return_indexers=True)
    expected = (Index(serial_values), serial_lidx, serial_ridx)

    tm.assert_index_equal(result[0], expected[0])
    tm.assert_numpy_array_equal(result[1], expected[1])
    tm.assert_numpy_array_equal(result[2], expected[2])


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "float64[pyarrow]", "Int64"])
def test_extension_backed_index_parallel(dtype, assert_parallel_ran):
    # Index._get_join_target hands libjoin the backing buffer directly, so
    # these reach the chunked kernels like any numpy index.  Arrow's buffer is
    # read-only, which is the realistic way a non-writable array gets here: a
    # kernel that asks for a writable buffer raises instead of silently
    # copying, and that has regressed once already.
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    assert_parallel_ran(2)
    lvals = np.arange(0, 200, 2)
    rvals = np.arange(0, 200, 3)
    left = Index(lvals, dtype=dtype)
    right = Index(rvals, dtype=dtype)
    if "pyarrow" in dtype:
        assert not left._get_join_target().flags.writeable

    result = left.intersection(right)
    expected = Index(np.intersect1d(lvals, rvals), dtype=dtype)
    tm.assert_index_equal(result, expected)


def test_datetime_index_intersection_parallel(assert_parallel_ran):
    assert_parallel_ran(2)
    # datetimelike indexes reach libjoin as i8, so they use the parallel path
    # too -- but only without a freq, since DatetimeIndex._intersection has a
    # freq-aware fastpath that never touches libjoin
    left = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=200, freq="h").values)
    right = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=200, freq="2h").values)
    assert left.freq is None and right.freq is None

    result = left.intersection(right)
    tm.assert_index_equal(result, right[:100], check_freq=False)


class TestEligibility:
    @pytest.fixture
    def low_threshold(self, monkeypatch):
        monkeypatch.setattr(array_algos_join, "_PARALLEL_JOIN_MIN_ROWS", 4)

    def test_object_dtype_is_serial(self, low_threshold):
        # the kernel holds the GIL for object dtype, so threads cannot help
        values = np.arange(64).astype(object)
        assert not array_algos_join._can_parallelize(values, values)

    def test_below_threshold_is_serial(self):
        values = np.arange(64, dtype=np.int64)
        assert not array_algos_join._can_parallelize(values, values)

    def test_mismatched_dtype_is_serial(self, low_threshold):
        left = np.arange(64, dtype=np.int64)
        right = np.arange(64, dtype=np.int32)
        assert not array_algos_join._can_parallelize(left, right)

    def test_non_contiguous_is_eligible(self, low_threshold):
        # a plain slice of an index is non-contiguous; the kernel indexes
        # through strides, so there is no reason to exclude it
        values = np.arange(256, dtype=np.int64)[::2]
        assert not values.flags.c_contiguous
        assert array_algos_join._can_parallelize(values, values)

    def test_much_smaller_right_is_eligible(self, low_threshold):
        # only left is divided, and each chunk clips right to its own key span,
        # so a right side orders of magnitude smaller still splits usefully --
        # 20M against 100k measured 3.7x.  The planner has the final say.
        # right must be shorter than the threshold itself, or gating on
        # min(left, right) would accept this too and the test prove nothing
        left = np.arange(64, dtype=np.int64)
        right = np.array([0, 32], dtype=np.int64)
        assert right.shape[0] < array_algos_join._PARALLEL_JOIN_MIN_ROWS
        assert array_algos_join._can_parallelize(left, right)

    def test_empty_right_is_serial(self, low_threshold):
        left = np.arange(64, dtype=np.int64)
        right = np.array([], dtype=np.int64)
        assert not array_algos_join._can_parallelize(left, right)

    def test_two_dimensional_is_serial(self, low_threshold):
        values = np.arange(64, dtype=np.int64).reshape(8, 8)
        assert not array_algos_join._can_parallelize(values, values)

    def test_eligible_input_is_accepted(self, low_threshold):
        values = np.arange(64, dtype=np.int64)
        assert array_algos_join._can_parallelize(values, values)

    def test_single_worker_stays_serial(self, monkeypatch):
        # a single worker must not reach the chunking code
        monkeypatch.setattr(array_algos_join, "_PARALLEL_JOIN_MIN_ROWS", 4)
        monkeypatch.setattr(array_algos_join, "max_workers", lambda cap: 1)
        values = np.arange(64, dtype=np.int64)
        result = array_algos_join.inner_join_indexer(values, values)
        expected = libjoin.inner_join_indexer(values, values)
        for res, exp in zip(result, expected, strict=True):
            tm.assert_numpy_array_equal(res, exp)

    def test_max_threads_one_disables(self):
        with pd.option_context("mode.max_threads", 1):
            assert max_workers(8) == 1

    @pytest.mark.skipif(WASM, reason="WASM is always single-threaded")
    def test_max_threads_respected(self):
        with pd.option_context("mode.max_threads", 3):
            assert max_workers(8) == 3


def test_dominant_run_falls_back_to_serial(assert_parallel_ran):
    # A run of equal keys cannot be split, so a dominant run leaves one chunk
    # doing nearly the whole walk.  Splitting then costs dispatch without
    # buying concurrency, so the driver must decline.
    assert_parallel_ran(4, expect_parallel=False)
    left = np.concatenate([np.ones(400, dtype=np.int64), np.array([2], dtype=np.int64)])
    right = np.arange(400, dtype=np.int64)

    # each scheme produces a legal split -- it is the balance that rules them out
    assert array_algos_join._even_boundaries(left, 4).shape[0] > 2
    assert array_algos_join._plan_chunks(left, right, 4) is None

    result = array_algos_join.inner_join_indexer(left, right)
    expected = libjoin.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


def test_dominant_run_on_right_falls_back_to_serial(assert_parallel_ran):
    # The mirror of the case above, and the one a left-only guard misses: left
    # splits perfectly evenly, but every key in right is the same, so
    # _chunk_ranges hands the whole of right to one chunk and the others get
    # nothing.  Measured ~15% slower than serial at a million rows.
    assert_parallel_ran(4, expect_parallel=False)
    left = np.arange(400, dtype=np.int64)
    right = np.zeros(400, dtype=np.int64)

    # left on its own divides into four near-equal chunks; only right rules it out
    even = array_algos_join._even_boundaries(left, 4)
    assert even.shape[0] == 5
    assert int(np.diff(even).max()) <= left.shape[0] // 4 + 1
    rstart, rstop = array_algos_join._chunk_ranges(left, right, even)
    assert int((rstop - rstart).max()) == right.shape[0]
    assert array_algos_join._plan_chunks(left, right, 4) is None

    result = array_algos_join.inner_join_indexer(left, right)
    expected = libjoin.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


def test_skew_on_the_smaller_side_still_parallelizes(assert_parallel_ran):
    # Right is 96% one value, so it alone looks hopelessly lopsided -- but it
    # is a tenth the size of left, which divides evenly and carries the work.
    # Judging the sides separately rejects this; at 10M against 1M the split
    # measured 2.4x faster than serial.
    assert_parallel_ran(4)
    left = np.arange(4000, dtype=np.int64)
    right = np.concatenate(
        [np.zeros(384, dtype=np.int64), np.arange(16, dtype=np.int64) * 250 + 125]
    )

    # dividing left evenly gives one chunk almost the whole of right; Merge
    # Path spreads it, and the planner is expected to prefer that
    even = array_algos_join._even_boundaries(left, 4)
    mp = array_algos_join._merge_path_boundaries(left, right, 4)

    def peak(bounds):
        rstart, rstop = array_algos_join._chunk_ranges(left, right, bounds)
        return int(
            array_algos_join._walk_cost(
                left, right, bounds[:-1], bounds[1:], rstart, rstop
            ).max()
        )

    assert peak(mp) < peak(even)

    plan = array_algos_join._plan_chunks(left, right, 4)
    assert plan is not None
    tm.assert_numpy_array_equal(plan[0], mp)

    result = array_algos_join.inner_join_indexer(left, right)
    expected = libjoin.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


def test_clipped_right_still_parallelizes(assert_parallel_ran):
    # Almost every right row sits below every left key, so each chunk clips
    # them away and the one chunk with anything to do runs a couple of
    # iterations where the serial walk scans the whole prefix.  The split is
    # about as unbalanced as one can be -- a single chunk holds all the work --
    # so judging balance rejects it, and judging it against serial does not.
    assert_parallel_ran(2)
    left = np.arange(1000, 2000, dtype=np.int64)
    right = np.array([0] * 999 + [1000], dtype=np.int64)

    bounds = array_algos_join._even_boundaries(left, 2)
    rstart, rstop = array_algos_join._chunk_ranges(left, right, bounds)
    work = array_algos_join._walk_cost(
        left, right, bounds[:-1], bounds[1:], rstart, rstop
    )
    # balance is 1.0 -- one chunk, everything -- yet it beats serial by far
    assert int(work.sum()) == int(work.max())
    assert int(work.max()) * 100 < left.shape[0] + right.shape[0]

    result = array_algos_join.inner_join_indexer(left, right)
    expected = libjoin.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


def test_tiny_right_below_left_falls_back_to_serial(assert_parallel_ran):
    # Right lies entirely below left, so the serial walk stops the moment right
    # runs out -- one comparison -- while every chunk clips right away to
    # nothing.  Costing empty chunks at zero makes that look infinitely good
    # and buys a thread pool for a join already finished: 61x slower when
    # forced.  It is the mirror of the disjoint case, where right lies *above*
    # left and serial has to scan all of left, which is worth splitting.
    assert_parallel_ran(2, expect_parallel=False)
    left = np.arange(1000, 2000, dtype=np.int64)
    right = np.array([0], dtype=np.int64)

    bounds = array_algos_join._even_boundaries(left, 2)
    rstart, rstop = array_algos_join._chunk_ranges(left, right, bounds)
    work = array_algos_join._walk_cost(
        left, right, bounds[:-1], bounds[1:], rstart, rstop
    )
    assert int(work.max()) == 0, "every chunk should clip away to nothing"
    assert array_algos_join._plan_chunks(left, right, 2) is None

    result = array_algos_join.inner_join_indexer(left, right)
    expected = libjoin.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


def test_dispatch_floor_declines_a_free_join():
    # Deliberately does NOT use assert_parallel_ran, which scales
    # _DISPATCH_COST_ITERS down to keep the small fixtures scoreable; the point
    # here is the shipped value.  Every chunk clips right away to nothing, so
    # without a floor the peak is zero and the split scores unboundedly well,
    # winning a pool for a join the serial walk finishes in one comparison.
    left = np.arange(1000, 2000, dtype=np.int64)
    right = np.zeros(100, dtype=np.int64)

    bounds = array_algos_join._even_boundaries(left, 2)
    rstart, rstop = array_algos_join._chunk_ranges(left, right, bounds)
    peak = int(
        array_algos_join._walk_cost(
            left, right, bounds[:-1], bounds[1:], rstart, rstop
        ).max()
    )
    serial = int(
        array_algos_join._walk_cost(
            left,
            right,
            np.array([0], dtype=np.intp),
            np.array([left.shape[0]], dtype=np.intp),
            np.array([0], dtype=np.intp),
            np.array([right.shape[0]], dtype=np.intp),
        )[0]
    )
    assert peak == 0, "every chunk should clip away to nothing"
    # unfloored this scores serial/0 -- unboundedly good -- and would be taken;
    # floored it is serial over tens of thousands, which is not worth a pool
    assert serial > array_algos_join._MIN_USEFUL_SPEEDUP
    assert serial < array_algos_join._DISPATCH_COST_ITERS

    assert array_algos_join._plan_chunks(left, right, 2) is None


def test_balanced_chunks_still_parallelize(assert_parallel_ran):
    # the guard above must not reject an ordinary balanced split
    assert_parallel_ran(4)
    left = np.arange(400, dtype=np.int64)
    right = np.arange(0, 800, 2, dtype=np.int64)

    result = array_algos_join.inner_join_indexer(left, right)
    expected = libjoin.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


class TestKeyBoundaries:
    @pytest.mark.parametrize("n_chunks", [2, 3, 4, 8])
    @pytest.mark.parametrize(
        "values",
        [
            list(range(64)),
            [1] * 64,
            [1] * 32 + [2] * 32,
            sorted([1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 5, 6] * 4),
        ],
    )
    def test_never_splits_a_run(self, values, n_chunks):
        left = np.array(values, dtype=np.int64)
        # both schemes must respect the snapping rule
        for bounds in (
            array_algos_join._even_boundaries(left, n_chunks),
            array_algos_join._merge_path_boundaries(left, left, n_chunks),
        ):
            self._check_bounds(left, bounds)

    @staticmethod
    def _check_bounds(left, bounds):
        assert bounds[0] == 0
        assert bounds[-1] == len(left)
        assert (np.diff(bounds) > 0).all(), "bounds must be strictly increasing"
        # the key before each interior split differs from the key at the split
        for b in bounds[1:-1]:
            assert left[b - 1] != left[b]
