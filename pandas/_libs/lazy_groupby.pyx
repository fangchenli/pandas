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
    uint8_t,
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

# FNV-1a constants for hashing variable-length string key bytes.
cdef uint64_t FNV_OFFSET = <uint64_t>1469598103934665603
cdef uint64_t FNV_PRIME = <uint64_t>1099511628211
# Column-mix multiplier (a large odd prime) to combine per-column hashes into a
# single row hash: acc = acc * MIX + col_hash.
cdef uint64_t MIX = <uint64_t>1000000007


@cython.boundscheck(False)
@cython.wraparound(False)
def hash_string_col(
    const int64_t[::1] offsets,
    const uint8_t[::1] data,
    uint64_t[::1] acc,
    bint first,
    Py_ssize_t lo,
    Py_ssize_t hi,
):
    """Fold one Arrow string column's per-row FNV-1a hash into ``acc[lo:hi]``.

    ``offsets`` are int64 (length n+1) into the ``data`` byte buffer. If
    ``first``, ``acc[i]`` is set to the string hash; otherwise mixed in. The
    ``[lo, hi)`` row range lets the driver hash disjoint ranges on separate
    threads (the body is ``nogil``).
    """
    cdef:
        Py_ssize_t i, j, start, end
        uint64_t h
    with nogil:
        for i in range(lo, hi):
            start = offsets[i]
            end = offsets[i + 1]
            h = FNV_OFFSET
            for j in range(start, end):
                h ^= data[j]
                h *= FNV_PRIME
            if first:
                acc[i] = h
            else:
                acc[i] = acc[i] * MIX + h


@cython.boundscheck(False)
@cython.wraparound(False)
def hash_int_col(
    const int64_t[::1] vals,
    uint64_t[::1] acc,
    bint first,
    Py_ssize_t lo,
    Py_ssize_t hi,
):
    """Fold one int64 key column's mixed hash into ``acc[lo:hi]``."""
    cdef:
        Py_ssize_t i
        uint64_t h
    with nogil:
        for i in range(lo, hi):
            h = (<uint64_t>vals[i]) * GOLDEN
            if first:
                acc[i] = h
            else:
                acc[i] = acc[i] * MIX + h


@cython.boundscheck(False)
@cython.wraparound(False)
def bucket_hash_sum(
    const int64_t[::1] perm,
    Py_ssize_t lo,
    Py_ssize_t hi,
    const uint64_t[::1] keyhash,
    const double[::1] values,
):
    """Aggregate ``sum(values)`` per distinct ``keyhash`` over rows
    ``perm[lo:hi]`` using an open-addressing table.

    Returns (rep_rows int64[], sums float64[]) — one entry per distinct group in
    this bucket, ``rep_rows`` being the first row seen for the group (used by the
    caller to gather the actual key columns).

    PROTOTYPE: groups by the 64-bit ``keyhash`` directly (no key-equality
    verification). Collisions among distinct keys are astronomically unlikely at
    these scales but NOT impossible; the production version must verify keys.
    """
    cdef:
        Py_ssize_t m = hi - lo
        Py_ssize_t i, row, cap, mask, slot, ng
        uint64_t hv
        cnp.ndarray[int64_t, ndim=1] rep_arr
        cnp.ndarray[double, ndim=1] sum_arr
        int64_t[::1] rep
        double[::1] sums
        # open-addressing slots: -1 = empty
        cnp.ndarray[int64_t, ndim=1] slot_rep
        int64_t[::1] sr

    if m <= 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))

    # capacity = next power of two >= 2*m (load factor 0.5)
    cap = 1
    while cap < 2 * m:
        cap <<= 1
    mask = cap - 1

    rep_arr = np.empty(m, dtype=np.int64)
    sum_arr = np.zeros(m, dtype=np.float64)
    slot_rep = np.full(cap, -1, dtype=np.int64)  # slot -> group index (or -1)
    rep = rep_arr
    sums = sum_arr
    sr = slot_rep
    ng = 0

    with nogil:
        for i in range(lo, hi):
            row = perm[i]
            hv = keyhash[row]
            slot = <Py_ssize_t>(hv & <uint64_t>mask)
            while sr[slot] != -1:
                if keyhash[rep[sr[slot]]] == hv:
                    break
                slot = (slot + 1) & mask
            if sr[slot] == -1:
                sr[slot] = ng
                rep[ng] = row
                sums[ng] = values[row]
                ng += 1
            else:
                sums[sr[slot]] += values[row]

    return (rep_arr[:ng], sum_arr[:ng])


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
