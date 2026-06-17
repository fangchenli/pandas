"""
Hash-partition kernel for the lazy engine's parallel grouped aggregate.

Arrow's ``Table.group_by`` is single-threaded but has an excellent serial
hash-aggregate. Polars beats it on high-cardinality group-bys purely by running
across cores. This kernel supplies the missing piece: a cheap partition of the
rows by a hash of the (combined) group key into ``n_buckets`` buckets, such that
**every group lands wholly in one bucket**. The Python driver then runs Arrow's
group_by on each bucket on its own thread (pyarrow releases the GIL) and simply
concatenates the per-bucket results — no cross-bucket merge, because groups do
not span buckets. See docs/PARALLEL_GROUPBY_SCOPE.md.

The work is a counting sort by bucket (histogram + scatter), two nogil passes
over the row indices — replacing the ``np.argsort``-based partition that
dominated the naive prototype (~190 ms vs ~20 ms here at 2.7M rows).

Classic algorithm (counting sort / distribution sort): Knuth TAOCP Vol. 3
§5.2 / Cormen et al. §8.2. The multiplicative key hash (Fibonacci hashing)
is Knuth §6.4.
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

# Fibonacci hashing multiplier (2^64 / golden ratio), then keep the high bits:
# high bits of a multiplicative hash are the well-mixed ones, so the bucket is
# ``(key * GOLDEN) >> (64 - log2(n_buckets))``. Mixing matters because raw group
# keys (e.g. dense partkeys) modulo a power of two would cluster.
cdef uint64_t GOLDEN = <uint64_t>0x9E3779B97F4A7C15


@cython.boundscheck(False)
@cython.wraparound(False)
def partition_by_key(const uint64_t[::1] keys, int n_buckets):
    """
    Partition rows by a hash of ``keys`` into ``n_buckets`` buckets.

    All rows sharing a key value map to the same bucket (so a group never
    spans buckets). ``n_buckets`` must be a power of two in [1, 1<<20].

    Parameters
    ----------
    keys : 1-D contiguous ``uint64`` memoryview
        The (combined) group key per row.
    n_buckets : int
        Number of buckets; power of two.

    Returns
    -------
    perm : numpy.ndarray[int64]
        Row indices grouped by bucket: ``perm[offsets[b]:offsets[b+1]]`` are the
        rows of bucket ``b`` (in input order within a bucket).
    offsets : numpy.ndarray[int64]
        Length ``n_buckets + 1`` prefix-sum of bucket sizes.
    """
    cdef:
        Py_ssize_t n = keys.shape[0]
        Py_ssize_t i
        int b
        int shift
        uint64_t bk

    if n_buckets < 1 or (n_buckets & (n_buckets - 1)) != 0 or n_buckets > (1 << 20):
        raise ValueError("n_buckets must be a power of two in [1, 1<<20]")

    if n_buckets == 1:
        # Single bucket: identity order, no hashing (a >>64 shift is UB).
        return np.arange(n, dtype=np.int64), np.array([0, n], dtype=np.int64)

    # shift to keep the top log2(n_buckets) bits of the mixed key
    shift = 64
    b = n_buckets
    while b > 1:
        shift -= 1
        b >>= 1

    cdef:
        cnp.ndarray[int64_t, ndim=1] perm_arr = np.empty(n, dtype=np.int64)
        cnp.ndarray[int64_t, ndim=1] off_arr = np.zeros(n_buckets + 1, dtype=np.int64)
        cnp.ndarray[int64_t, ndim=1] cur_arr = np.empty(n_buckets, dtype=np.int64)
        cnp.ndarray[uint64_t, ndim=1] bkt_arr = np.empty(n, dtype=np.uint64)
        int64_t[::1] perm = perm_arr
        int64_t[::1] off = off_arr
        int64_t[::1] cur = cur_arr
        uint64_t[::1] bkt = bkt_arr

    with nogil:
        # Pass 1: hash each key to a bucket, histogram into off[bucket+1].
        for i in range(n):
            bk = (keys[i] * GOLDEN) >> shift
            bkt[i] = bk
            off[<Py_ssize_t>bk + 1] += 1
        # Prefix sum -> bucket start offsets.
        for b in range(n_buckets):
            off[b + 1] += off[b]
            cur[b] = off[b]
        # Pass 2: scatter row indices into their bucket slots.
        for i in range(n):
            bk = bkt[i]
            perm[cur[<Py_ssize_t>bk]] = i
            cur[<Py_ssize_t>bk] += 1

    return perm_arr, off_arr
