"""
Stable LSD radix argsort over order-preserving ``uint64`` keys.

Used by the lazy execution engine's sort (``pandas/lazy``). The caller builds
the keys vectorized in NumPy (sign-bit / float bit transforms) and replaces
NaN beforehand, so this kernel is a single dtype-agnostic radix that returns a
stable ascending permutation.

Keys are carried alongside the indices and both are scattered every pass, so
the digit reads stay sequential. Index indirection (``keys[idx[i]]``) is the
cache-killer that makes a naive radix *arg*sort slower than a comparison sort;
moving the keys is what makes radix win here. 16-bit digits (4 passes) over
moved key/index pairs measured ~215 ms at 10M float64 vs the NumPy k-way
merge's ~320-410 ms, matching Polars' arg_sort (~205 ms).
"""

cimport cython
from cython cimport Py_ssize_t
from libc.stdint cimport (
    int64_t,
    uint64_t,
)

import numpy as np

cimport numpy as cnp

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def radix_argsort_u64(const uint64_t[::1] keys_in):
    """
    Stable ascending argsort of order-preserving ``uint64`` keys.

    Parameters
    ----------
    keys_in : 1-D contiguous ``uint64`` memoryview
        Order-preserving keys (see module docstring); index ``i`` sorts
        before ``j`` iff ``keys_in[i] <= keys_in[j]``, ties broken by ``i``.

    Returns
    -------
    numpy.ndarray
        ``int64`` permutation that stably sorts the keys ascending.
    """
    cdef:
        Py_ssize_t n = keys_in.shape[0]
        Py_ssize_t i, p
        int RADIX = 16
        int BUCKETS = 1 << 16
        uint64_t MASK = (<uint64_t>1 << 16) - 1
        int shift = 0
        int b, d
        int passes = 0

    cdef:
        cnp.ndarray[int64_t, ndim=1] idx_arr = np.arange(n, dtype=np.int64)
        cnp.ndarray[uint64_t, ndim=1] keys_arr = np.array(keys_in, dtype=np.uint64)
        cnp.ndarray[uint64_t, ndim=1] kb_arr = np.empty(n, dtype=np.uint64)
        cnp.ndarray[int64_t, ndim=1] ib_arr = np.empty(n, dtype=np.int64)
        cnp.ndarray[int64_t, ndim=1] count_arr = np.empty(BUCKETS + 1, dtype=np.int64)

    cdef:
        uint64_t[::1] keys = keys_arr
        uint64_t[::1] kb = kb_arr
        int64_t[::1] idx = idx_arr
        int64_t[::1] ib = ib_arr
        int64_t[::1] count = count_arr
        uint64_t[::1] tmpk
        int64_t[::1] tmpi

    if n <= 1:
        return idx_arr

    while shift < 64:
        for b in range(BUCKETS + 1):
            count[b] = 0
        for i in range(n):
            count[<int>((keys[i] >> shift) & MASK) + 1] += 1
        for b in range(BUCKETS):
            count[b + 1] += count[b]
        for i in range(n):
            d = <int>((keys[i] >> shift) & MASK)
            p = count[d]
            count[d] = p + 1
            kb[p] = keys[i]
            ib[p] = idx[i]
        tmpk = keys
        keys = kb
        kb = tmpk
        tmpi = idx
        idx = ib
        ib = tmpi
        shift += RADIX
        passes += 1

    # Each pass swaps the active buffer; the final indices live in idx_arr
    # when an even number of passes ran, in ib_arr otherwise.
    if passes % 2 == 0:
        return idx_arr
    return ib_arr
