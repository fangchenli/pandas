import numpy as np
import pytest

from pandas._libs import join as libjoin
from pandas._libs.join import (
    inner_join,
    left_outer_join,
)

import pandas._testing as tm


class TestIndexer:
    @pytest.mark.parametrize(
        "dtype", ["int32", "int64", "float32", "float64", "object"]
    )
    def test_outer_join_indexer(self, dtype):
        indexer = libjoin.outer_join_indexer

        left = np.arange(3, dtype=dtype)
        right = np.arange(2, 5, dtype=dtype)
        empty = np.array([], dtype=dtype)

        result, lindexer, rindexer = indexer(left, right)
        assert isinstance(result, np.ndarray)
        assert isinstance(lindexer, np.ndarray)
        assert isinstance(rindexer, np.ndarray)
        tm.assert_numpy_array_equal(result, np.arange(5, dtype=dtype))
        exp = np.array([0, 1, 2, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([-1, -1, 0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

        result, lindexer, rindexer = indexer(empty, right)
        tm.assert_numpy_array_equal(result, right)
        exp = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

        result, lindexer, rindexer = indexer(left, empty)
        tm.assert_numpy_array_equal(result, left)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

    def test_cython_left_outer_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1], dtype=np.intp)
        max_group = 5

        ls, rs = left_outer_join(left, right, max_group)

        exp_ls = left.argsort(kind="mergesort")
        exp_rs = right.argsort(kind="mergesort")

        exp_li = np.array([0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10])
        exp_ri = np.array(
            [0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5, -1, -1]
        )

        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1

        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1

        tm.assert_numpy_array_equal(ls, exp_ls, check_dtype=False)
        tm.assert_numpy_array_equal(rs, exp_rs, check_dtype=False)

    def test_cython_right_outer_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1], dtype=np.intp)
        max_group = 5

        rs, ls = left_outer_join(right, left, max_group)

        exp_ls = left.argsort(kind="mergesort")
        exp_rs = right.argsort(kind="mergesort")

        #            0        1        1        1
        exp_li = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                3,
                4,
                5,
                3,
                4,
                5,
                #            2        2        4
                6,
                7,
                8,
                6,
                7,
                8,
                -1,
            ]
        )
        exp_ri = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6])

        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1

        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1

        tm.assert_numpy_array_equal(ls, exp_ls)
        tm.assert_numpy_array_equal(rs, exp_rs)

    def test_cython_inner_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1, 4], dtype=np.intp)
        max_group = 5

        ls, rs = inner_join(left, right, max_group)

        exp_ls = left.argsort(kind="mergesort")
        exp_rs = right.argsort(kind="mergesort")

        exp_li = np.array([0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8])
        exp_ri = np.array([0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5])

        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1

        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1

        tm.assert_numpy_array_equal(ls, exp_ls)
        tm.assert_numpy_array_equal(rs, exp_rs)


def test_left_join_indexer_unique(writable):
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([2, 2, 3, 4, 4], dtype=np.int64)
    # GH#37312, GH#37264
    a.setflags(write=writable)
    b.setflags(write=writable)

    result = libjoin.left_join_indexer_unique(b, a)
    expected = np.array([1, 1, 2, 3, 3], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


def test_left_outer_join_bug():
    left = np.array(
        [
            0,
            1,
            0,
            1,
            1,
            2,
            3,
            1,
            0,
            2,
            1,
            2,
            0,
            1,
            1,
            2,
            3,
            2,
            3,
            2,
            1,
            1,
            3,
            0,
            3,
            2,
            3,
            0,
            0,
            2,
            3,
            2,
            0,
            3,
            1,
            3,
            0,
            1,
            3,
            0,
            0,
            1,
            0,
            3,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            2,
            2,
            2,
            2,
            2,
            0,
            3,
            1,
            2,
            0,
            0,
            3,
            1,
            3,
            2,
            2,
            0,
            1,
            3,
            0,
            2,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            1,
            3,
            2,
            0,
            0,
            3,
            1,
            1,
            1,
            0,
            2,
            3,
            3,
            1,
            2,
            0,
            3,
            1,
            2,
            0,
            2,
        ],
        dtype=np.intp,
    )

    right = np.array([3, 1], dtype=np.intp)
    max_groups = 4

    lidx, ridx = libjoin.left_outer_join(left, right, max_groups, sort=False)

    exp_lidx = np.arange(len(left), dtype=np.intp)
    exp_ridx = -np.ones(len(left), dtype=np.intp)

    exp_ridx[left == 1] = 1
    exp_ridx[left == 3] = 0

    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_inner_join_indexer():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    index, ares, bres = libjoin.inner_join_indexer(a, b)

    index_exp = np.array([3, 5], dtype=np.int64)
    tm.assert_almost_equal(index, index_exp)

    aexp = np.array([2, 4], dtype=np.intp)
    bexp = np.array([1, 2], dtype=np.intp)
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)

    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)

    index, ares, bres = libjoin.inner_join_indexer(a, b)
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))


def test_outer_join_indexer():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    index, ares, bres = libjoin.outer_join_indexer(a, b)

    index_exp = np.array([0, 1, 2, 3, 4, 5, 7, 9], dtype=np.int64)
    tm.assert_almost_equal(index, index_exp)

    aexp = np.array([-1, 0, 1, 2, 3, 4, -1, -1], dtype=np.intp)
    bexp = np.array([0, -1, -1, 1, -1, 2, 3, 4], dtype=np.intp)
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)

    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)

    index, ares, bres = libjoin.outer_join_indexer(a, b)
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))


def test_left_join_indexer():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    index, ares, bres = libjoin.left_join_indexer(a, b)

    tm.assert_almost_equal(index, a)

    aexp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
    bexp = np.array([-1, -1, 1, -1, 2], dtype=np.intp)
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)

    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)

    index, ares, bres = libjoin.left_join_indexer(a, b)
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))


def test_left_join_indexer2():
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    res, lidx, ridx = libjoin.left_join_indexer(idx2, idx)

    exp_res = np.array([1, 1, 2, 5, 7, 9], dtype=np.int64)
    tm.assert_almost_equal(res, exp_res)

    exp_lidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
    tm.assert_almost_equal(lidx, exp_lidx)

    exp_ridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
    tm.assert_almost_equal(ridx, exp_ridx)


def test_outer_join_indexer2():
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    res, lidx, ridx = libjoin.outer_join_indexer(idx2, idx)

    exp_res = np.array([1, 1, 2, 5, 7, 9], dtype=np.int64)
    tm.assert_almost_equal(res, exp_res)

    exp_lidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
    tm.assert_almost_equal(lidx, exp_lidx)

    exp_ridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
    tm.assert_almost_equal(ridx, exp_ridx)


def test_inner_join_indexer2():
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    res, lidx, ridx = libjoin.inner_join_indexer(idx2, idx)

    exp_res = np.array([1, 1, 2, 5], dtype=np.int64)
    tm.assert_almost_equal(res, exp_res)

    exp_lidx = np.array([0, 0, 1, 2], dtype=np.intp)
    tm.assert_almost_equal(lidx, exp_lidx)

    exp_ridx = np.array([0, 1, 2, 3], dtype=np.intp)
    tm.assert_almost_equal(ridx, exp_ridx)


class TestInnerJoinFillRange:
    def _views(self):
        left = np.arange(10, dtype=np.int64)
        right = np.arange(5, 15, dtype=np.int64)
        return left, right

    def test_returns_pairs_written(self):
        left, right = self._views()
        n = libjoin.inner_join_count_range(left, right)
        result = np.empty(n, dtype=np.int64)
        lidx = np.empty(n, dtype=np.intp)
        ridx = np.empty(n, dtype=np.intp)

        written = libjoin.inner_join_fill_range(left, right, result, lidx, ridx, 0, 0)
        assert written == n

    def test_short_output_truncates_instead_of_overflowing(self):
        # GH#51364 the fill loop runs with bounds checking off, so a caller that
        # under-sizes the output must be truncated, not allowed to write past
        # the end; the short count is how the caller finds out
        left, right = self._views()
        n = libjoin.inner_join_count_range(left, right)
        assert n > 2

        result = np.zeros(2, dtype=np.int64)
        lidx = np.zeros(2, dtype=np.intp)
        ridx = np.zeros(2, dtype=np.intp)

        written = libjoin.inner_join_fill_range(left, right, result, lidx, ridx, 0, 0)
        assert written == 2

    def test_mismatched_output_lengths_raise(self):
        left, right = self._views()
        msg = "must be equal length"
        with pytest.raises(ValueError, match=msg):
            libjoin.inner_join_fill_range(
                left,
                right,
                np.empty(5, dtype=np.int64),
                np.empty(4, dtype=np.intp),
                np.empty(5, dtype=np.intp),
                0,
                0,
            )


@pytest.mark.parametrize("dtype", ["int64", "float64", "uint32"])
def test_inner_join_range_read_only_input(dtype):
    # GH#51364 the join target of an Arrow-backed Index is a zero-copy,
    # read-only view, so neither pass may ask for a writable buffer on the
    # inputs.  Freshly allocated test arrays are writable and hide this.
    left = np.arange(10, dtype=dtype)
    right = np.arange(5, 15, dtype=dtype)
    left.setflags(write=False)
    right.setflags(write=False)

    count = libjoin.inner_join_count_range(left, right)

    result = np.empty(count, dtype=dtype)
    lidx = np.empty(count, dtype=np.intp)
    ridx = np.empty(count, dtype=np.intp)
    written = libjoin.inner_join_fill_range(left, right, result, lidx, ridx, 0, 0)

    assert written == count
    expected = libjoin.inner_join_indexer(left, right)
    tm.assert_numpy_array_equal(result, expected[0])
    tm.assert_numpy_array_equal(lidx, expected[1])
    tm.assert_numpy_array_equal(ridx, expected[2])
