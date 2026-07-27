"""
Tests for the parallel driver behind :meth:`Index._inner_indexer`.

The threshold is lowered in these tests so the parallel path actually runs;
with the shipped threshold it would take a multi-million-row index to reach.
"""

import numpy as np
import pytest

from pandas._libs import join as libjoin

import pandas as pd
from pandas import Index
import pandas._testing as tm
from pandas.core.array_algos import join as array_algos_join


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
    state = {"calls": 0, "expect_parallel": True}
    real = libjoin.inner_join_count_range

    def spy(*args):
        state["calls"] += 1
        return real(*args)

    monkeypatch.setattr(array_algos_join, "_PARALLEL_JOIN_MIN_ROWS", 4)
    monkeypatch.setattr(libjoin, "inner_join_count_range", spy)

    def use(n_workers, *, expect_parallel: bool = True):
        monkeypatch.setattr(array_algos_join, "_n_workers", lambda: n_workers)
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

    # serial first, before the fixture lowers the threshold
    expected = left.join(right, how="inner", return_indexers=True)

    assert_parallel_ran(2)
    result = left.join(right, how="inner", return_indexers=True)

    tm.assert_index_equal(result[0], expected[0])
    tm.assert_numpy_array_equal(result[1], expected[1])
    tm.assert_numpy_array_equal(result[2], expected[2])


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

    def test_non_contiguous_is_serial(self, low_threshold):
        values = np.arange(256, dtype=np.int64)[::2]
        assert not values.flags.c_contiguous
        assert not array_algos_join._can_parallelize(values, values)

    def test_eligible_input_is_accepted(self, low_threshold):
        values = np.arange(64, dtype=np.int64)
        assert array_algos_join._can_parallelize(values, values)

    def test_single_worker_stays_serial(self, monkeypatch):
        # _n_workers() == 1 must not reach the chunking code
        monkeypatch.setattr(array_algos_join, "_PARALLEL_JOIN_MIN_ROWS", 4)
        monkeypatch.setattr(array_algos_join, "_n_workers", lambda: 1)
        values = np.arange(64, dtype=np.int64)
        result = array_algos_join.inner_join_indexer(values, values)
        expected = libjoin.inner_join_indexer(values, values)
        for res, exp in zip(result, expected, strict=True):
            tm.assert_numpy_array_equal(res, exp)

    def test_max_threads_one_disables(self):
        with pd.option_context("mode.max_threads", 1):
            assert array_algos_join._n_workers() == 1

    def test_max_threads_respected(self):
        with pd.option_context("mode.max_threads", 3):
            assert array_algos_join._n_workers() == 3


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
        bounds = array_algos_join._key_boundaries(left, n_chunks)

        assert bounds[0] == 0
        assert bounds[-1] == len(left)
        assert (np.diff(bounds) > 0).all(), "bounds must be strictly increasing"
        # the key before each interior split differs from the key at the split
        for b in bounds[1:-1]:
            assert left[b - 1] != left[b]
