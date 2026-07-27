"""
Tests for the threaded payload gather in ``pandas.core.array_algos.take``.

The threshold is lowered here so the parallel path actually runs; with the
shipped one it would take a million-element result to reach.
"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.array_algos import take as libtake


@pytest.fixture
def force_parallel_take(monkeypatch):
    """Take the parallel path for every eligible gather."""
    monkeypatch.setattr(libtake, "_PARALLEL_TAKE_MIN_ELEMENTS", 8)
    monkeypatch.setattr(libtake, "_take_n_workers", lambda: 2)


def _serial(monkeypatch, arr, indexer, **kwargs):
    monkeypatch.setattr(libtake, "_PARALLEL_TAKE_MIN_ELEMENTS", np.iinfo(np.int64).max)
    return libtake.take_nd(arr, indexer, **kwargs)


@pytest.mark.parametrize(
    "dtype",
    ["int8", "int64", "uint32", "float32", "float64", "bool", "datetime64[ns]"],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("n_indexer", [0, 1, 7, 64])
@pytest.mark.parametrize("allow_fill", [False, True])
def test_parallel_take_2d_matches_serial(
    dtype, axis, n_indexer, allow_fill, monkeypatch
):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 50, size=(6, 40))
    arr = (base % 2).astype(bool) if dtype == "bool" else base.astype(dtype)

    indexer = rng.integers(0, arr.shape[axis], size=n_indexer).astype(np.intp)
    if allow_fill and n_indexer:
        indexer = indexer.copy()
        indexer[::3] = -1

    expected = _serial(monkeypatch, arr, indexer, axis=axis, allow_fill=allow_fill)
    monkeypatch.setattr(libtake, "_PARALLEL_TAKE_MIN_ELEMENTS", 8)
    monkeypatch.setattr(libtake, "_take_n_workers", lambda: 2)
    result = libtake.take_nd(arr, indexer, axis=axis, allow_fill=allow_fill)

    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("n_indexer", [0, 1, 64])
def test_parallel_take_1d_matches_serial(dtype, n_indexer, monkeypatch):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 50, size=200).astype(dtype)
    indexer = rng.integers(0, arr.shape[0], size=n_indexer).astype(np.intp)

    expected = _serial(monkeypatch, arr, indexer)
    monkeypatch.setattr(libtake, "_PARALLEL_TAKE_MIN_ELEMENTS", 8)
    monkeypatch.setattr(libtake, "_take_n_workers", lambda: 2)
    result = libtake.take_nd(arr, indexer)

    tm.assert_numpy_array_equal(result, expected)


def test_parallel_take_f_contiguous(monkeypatch):
    # F-contiguous input goes through the flip_order branch, which transposes
    # both the input and the output around the take
    rng = np.random.default_rng(2)
    arr = np.asfortranarray(rng.random((5, 60)))
    indexer = rng.integers(0, 60, size=60).astype(np.intp)

    expected = _serial(monkeypatch, arr, indexer, axis=1)
    monkeypatch.setattr(libtake, "_PARALLEL_TAKE_MIN_ELEMENTS", 8)
    monkeypatch.setattr(libtake, "_take_n_workers", lambda: 2)
    result = libtake.take_nd(arr, indexer, axis=1)

    tm.assert_numpy_array_equal(result, expected)


def test_dataframe_join_parallel_take(force_parallel_take):
    left = pd.DataFrame({"a": np.arange(200.0)}, index=pd.Index(np.arange(200)))
    right = pd.DataFrame({"b": np.arange(200.0)}, index=pd.Index(np.arange(0, 400, 2)))

    result = left.join(right, how="inner")

    expected = pd.DataFrame(
        {"a": np.arange(0, 200, 2.0), "b": np.arange(100.0)},
        index=pd.Index(np.arange(0, 200, 2)),
    )
    tm.assert_frame_equal(result, expected)


class TestTakeEligibility:
    def test_object_dtype_stays_serial(self, monkeypatch):
        # the take kernel holds the GIL for object dtype
        monkeypatch.setattr(libtake, "_PARALLEL_TAKE_MIN_ELEMENTS", 1)
        calls = []
        monkeypatch.setattr(
            libtake,
            "_take_n_workers",
            lambda: (calls.append(1), 2)[1],
        )
        arr = np.array([["a", "b"], ["c", "d"]], dtype=object)

        libtake.take_nd(arr, np.array([1, 0], dtype=np.intp), axis=1)

        assert not calls, "object dtype must not reach the worker-count lookup"

    def test_small_take_stays_serial(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            libtake,
            "_take_n_workers",
            lambda: (calls.append(1), 2)[1],
        )
        arr = np.arange(100.0).reshape(10, 10)

        libtake.take_nd(arr, np.array([1, 0], dtype=np.intp), axis=1)

        assert not calls, "the size gate must be checked before the worker count"

    def test_max_threads_one_disables(self):
        with pd.option_context("mode.max_threads", 1):
            assert libtake._take_n_workers() == 1

    def test_workers_capped(self):
        # a gather saturates at two threads; asking for more is wasted
        with pd.option_context("mode.max_threads", 16):
            assert libtake._take_n_workers() == libtake._MAX_TAKE_WORKERS
