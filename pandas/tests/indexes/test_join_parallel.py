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
def force_parallel(monkeypatch):
    """Make every eligible join take the parallel path."""
    monkeypatch.setattr(array_algos_join, "_PARALLEL_JOIN_MIN_ROWS", 4)
    return array_algos_join


@pytest.fixture
def two_workers(monkeypatch):
    monkeypatch.setattr(array_algos_join, "_MAX_DEFAULT_WORKERS", 2)


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
        ([1] * 80, [1] * 80),  # a single key -- cannot be split
        ([1] * 40 + [3] * 40, [1] * 40 + [2] * 40 + [3] * 40),  # gap on the right
    ],
)
def test_parallel_matches_serial(
    left_vals, right_vals, dtype, n_workers, force_parallel, monkeypatch
):
    monkeypatch.setattr(array_algos_join, "_MAX_DEFAULT_WORKERS", n_workers)
    left = np.array(list(left_vals), dtype=dtype)
    right = np.array(list(right_vals), dtype=dtype)

    expected = libjoin.inner_join_indexer(left, right)
    result = array_algos_join.inner_join_indexer(left, right)

    for res, exp in zip(result, expected, strict=True):
        tm.assert_numpy_array_equal(res, exp)


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_index_intersection_parallel(dtype, force_parallel, two_workers):
    # the user-facing path: Index.intersection goes through _inner_indexer
    left = Index(np.arange(0, 200, 2, dtype=dtype))
    right = Index(np.arange(0, 200, 3, dtype=dtype))

    result = left.intersection(right)
    expected = Index(np.intersect1d(left.to_numpy(), right.to_numpy()), dtype=dtype)
    tm.assert_index_equal(result, expected)


def test_index_join_inner_parallel(monkeypatch, two_workers):
    left = Index(np.arange(0, 200, 2))
    right = Index(np.arange(0, 200, 3))

    # serial first, at the shipped threshold
    expected = left.join(right, how="inner", return_indexers=True)

    monkeypatch.setattr(array_algos_join, "_PARALLEL_JOIN_MIN_ROWS", 4)
    result = left.join(right, how="inner", return_indexers=True)

    tm.assert_index_equal(result[0], expected[0])
    tm.assert_numpy_array_equal(result[1], expected[1])
    tm.assert_numpy_array_equal(result[2], expected[2])


def test_datetime_index_intersection_parallel(force_parallel, two_workers):
    # datetimelike indexes reach libjoin as i8, so they use the parallel path too
    left = pd.date_range("2020-01-01", periods=200, freq="h")
    right = pd.date_range("2020-01-01 00:00", periods=200, freq="2h")

    result = left.intersection(right)
    tm.assert_index_equal(result, right[:100])


class TestEligibility:
    def test_object_dtype_is_serial(self, force_parallel):
        # the kernel holds the GIL for object dtype, so threads cannot help
        values = np.arange(64).astype(object)
        assert not array_algos_join._can_parallelize(values, values, 4)

    def test_below_threshold_is_serial(self):
        values = np.arange(64, dtype=np.int64)
        assert not array_algos_join._can_parallelize(values, values, 4)

    def test_single_worker_is_serial(self, force_parallel):
        values = np.arange(64, dtype=np.int64)
        assert not array_algos_join._can_parallelize(values, values, 1)

    def test_mismatched_dtype_is_serial(self, force_parallel):
        left = np.arange(64, dtype=np.int64)
        right = np.arange(64, dtype=np.int32)
        assert not array_algos_join._can_parallelize(left, right, 4)

    def test_non_contiguous_is_serial(self, force_parallel):
        values = np.arange(256, dtype=np.int64)[::2]
        assert not values.flags.c_contiguous
        assert not array_algos_join._can_parallelize(values, values, 4)

    def test_max_threads_one_disables(self, force_parallel):
        with pd.option_context("mode.max_threads", 1):
            assert array_algos_join._n_workers() == 1

    def test_max_threads_respected(self, force_parallel):
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
