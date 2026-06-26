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

# ---------------------------------------------------------------------------
# Parallel string-key factorize (docs/STRING_HASH_AGGREGATE_KERNEL.md).
#
# Arrow's group_by on raw string keys is single-threaded and ~2x Polars on
# high-cardinality string groups; dict-encoding doesn't help when keys are
# near-unique. This factorizes a mixed (string + int/temporal) group key into
# dense int64 codes IN PARALLEL, so the caller can then aggregate on the cheap
# integer codes via the existing (Arrow / parallel-partitioned) machinery and
# attach the real key columns at each group's representative row. The genuinely
# parallel piece — and the whole measured win — is hashing the string bytes
# across threads (the row-range params below).
#
# Keys are compared by a 128-bit hash (two independent 64-bit hashes). Collision
# probability is ~n^2 / 2^128 (e.g. ~3e-23 for 1e8 distinct keys) — far below
# the engine's existing float-sum nondeterminism, so groups are exact in
# practice. partition_by_key uses only the first hash (collisions there are
# harmless: they only co-locate distinct groups in a bucket, where the 128-bit
# compare separates them).
# ---------------------------------------------------------------------------

# FNV-1a constants for hashing variable-length string key bytes (two seeds for
# the independent second hash).
cdef uint64_t FNV_OFFSET = <uint64_t>1469598103934665603
cdef uint64_t FNV_PRIME = <uint64_t>1099511628211
cdef uint64_t FNV_OFFSET2 = <uint64_t>14695981039346656037ULL
cdef uint64_t FNV_PRIME2 = <uint64_t>0x100000001B3
# Column-mix multipliers (large odd constants) to combine per-column hashes into
# a single row hash: acc = acc * MIX + col_hash.
cdef uint64_t MIX = <uint64_t>1000000007
cdef uint64_t MIX2 = <uint64_t>0x9E3779B185EBCA87ULL
cdef uint64_t GOLDEN2 = <uint64_t>0xD6E8FEB86659FD93ULL


@cython.boundscheck(False)
@cython.wraparound(False)
def hash_string_col(
    const int64_t[::1] offsets,
    const uint8_t[::1] data,
    uint64_t[::1] acc,
    uint64_t[::1] acc2,
    bint first,
    Py_ssize_t lo,
    Py_ssize_t hi,
):
    """Fold one Arrow string column's per-row 128-bit FNV-1a hash into
    ``(acc, acc2)[lo:hi]``.

    ``offsets`` are int64 (length n+1) into the ``data`` byte buffer; both
    hashes are computed in one byte pass. If ``first``, the slots are set;
    otherwise mixed in. The ``[lo, hi)`` row range lets the driver hash disjoint
    ranges on separate threads (the body is ``nogil``).
    """
    cdef:
        Py_ssize_t i, j, start, end
        uint64_t h, h2
        uint8_t b
    with nogil:
        for i in range(lo, hi):
            start = offsets[i]
            end = offsets[i + 1]
            h = FNV_OFFSET
            h2 = FNV_OFFSET2
            for j in range(start, end):
                b = data[j]
                h ^= b
                h *= FNV_PRIME
                h2 ^= b
                h2 *= FNV_PRIME2
            if first:
                acc[i] = h
                acc2[i] = h2
            else:
                acc[i] = acc[i] * MIX + h
                acc2[i] = acc2[i] * MIX2 + h2


@cython.boundscheck(False)
@cython.wraparound(False)
def hash_int_col(
    const int64_t[::1] vals,
    uint64_t[::1] acc,
    uint64_t[::1] acc2,
    bint first,
    Py_ssize_t lo,
    Py_ssize_t hi,
):
    """Fold one int64 key column's 128-bit mixed hash into ``(acc, acc2)[lo:hi]``."""
    cdef:
        Py_ssize_t i
        uint64_t h, h2, v
    with nogil:
        for i in range(lo, hi):
            v = <uint64_t>vals[i]
            h = v * GOLDEN
            h2 = v * GOLDEN2
            if first:
                acc[i] = h
                acc2[i] = h2
            else:
                acc[i] = acc[i] * MIX + h
                acc2[i] = acc2[i] * MIX2 + h2


@cython.boundscheck(False)
@cython.wraparound(False)
def bucket_factorize(
    const int64_t[::1] perm,
    Py_ssize_t lo,
    Py_ssize_t hi,
    const uint64_t[::1] h1,
    const uint64_t[::1] h2,
    int64_t[::1] codes,
):
    """Factorize the rows ``perm[lo:hi]`` (one partition bucket) by their
    128-bit key hash ``(h1, h2)`` using an open-addressing table.

    Writes a LOCAL dense code (0..ng-1) into ``codes[row]`` for each row in the
    bucket; the caller offsets these by the bucket's global base. Returns
    (ng, rep_rows int64[ng]) where ``rep_rows[c]`` is the first row seen for
    local code ``c`` (the caller gathers the real key columns there). Buckets
    are disjoint by key, so concatenating per-bucket results yields a global
    factorization with no merge.
    """
    cdef:
        Py_ssize_t m = hi - lo
        Py_ssize_t i, row, cap, mask, slot, ng, g
        uint64_t a, b
        cnp.ndarray[int64_t, ndim=1] rep_arr
        int64_t[::1] rep
        cnp.ndarray[int64_t, ndim=1] slot_grp
        int64_t[::1] sg

    if m <= 0:
        return (0, np.empty(0, dtype=np.int64))

    # capacity = next power of two >= 2*m (load factor 0.5)
    cap = 1
    while cap < 2 * m:
        cap <<= 1
    mask = cap - 1

    rep_arr = np.empty(m, dtype=np.int64)
    slot_grp = np.full(cap, -1, dtype=np.int64)  # slot -> local group id (or -1)
    rep = rep_arr
    sg = slot_grp
    ng = 0

    with nogil:
        for i in range(lo, hi):
            row = perm[i]
            a = h1[row]
            b = h2[row]
            slot = <Py_ssize_t>(a & <uint64_t>mask)
            while sg[slot] != -1:
                g = sg[slot]
                if h1[rep[g]] == a and h2[rep[g]] == b:
                    break
                slot = (slot + 1) & mask
            if sg[slot] == -1:
                sg[slot] = ng
                rep[ng] = row
                codes[row] = ng
                ng += 1
            else:
                codes[row] = sg[slot]

    return (ng, rep_arr[:ng])


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
