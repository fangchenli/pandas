"""
Single-pass hash inner join over ``int64`` keys.

Used by the lazy execution engine's join (``pandas/lazy``) as a fast path for
inner equi-joins on a single integer key. It produces the ``(left_indices,
right_indices)`` gather indexers; payload columns are gathered by the caller
with its existing parallel take.

Row-order contract — exactly ``pd.merge(how="inner")``: the hash table is
built on the RIGHT side with each key's right-row list kept in right-row
order (CSR layout), and the LEFT side is probed in row order, emitting every
(left, right) pair as encountered. Output is therefore left-row-major with
right matches in right order, which is pandas' documented inner-merge order.

Why this beats ``pd.merge``'s factorize-then-join: probe and emit happen in
one tight loop per row (no intermediate factorization arrays, no DataFrame
round-trip), and the probe is embarrassingly parallel — the count and fill
passes are ``nogil`` chunk kernels driven by a Python thread pool, same
pattern as ``lazy_radix.pyx`` (pandas builds without OpenMP).

The hash table is open-addressing with linear probing at load factor <= 0.5
(power-of-two size), using a Fibonacci/avalanche hash of the key. Classic
textbook structure; see Knuth TAOCP Vol. 3 §6.4.
"""

cimport cython
from cython cimport Py_ssize_t
from libc.stdint cimport (
    int64_t,
    uint64_t,
)
from libc.stdlib cimport (
    free,
    malloc,
)
from libc.string cimport memcpy

import numpy as np

cimport numpy as cnp

cnp.import_array()


@cython.cdivision(True)
cdef inline uint64_t _hash(int64_t key) noexcept nogil:
    cdef uint64_t h = <uint64_t>key
    h *= <uint64_t>0x9E3779B97F4A7C15
    h ^= h >> 32
    return h


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _build(
    const int64_t[::1] keys,
    int64_t[::1] slot_key,
    int64_t[::1] slot_gid,
    int64_t[::1] gids,
    int64_t[::1] counts,
    int64_t* n_groups_out,
) noexcept nogil:
    """Insert all right keys; record each row's group id and group sizes."""
    cdef Py_ssize_t n = keys.shape[0]
    cdef uint64_t mask = <uint64_t>(slot_key.shape[0] - 1)
    cdef Py_ssize_t i, j
    cdef int64_t key, gid, n_groups = 0
    for i in range(n):
        key = keys[i]
        j = <Py_ssize_t>(_hash(key) & mask)
        while True:
            gid = slot_gid[j]
            if gid == -1:
                slot_key[j] = key
                slot_gid[j] = n_groups
                gids[i] = n_groups
                counts[n_groups] = 1
                n_groups += 1
                break
            if slot_key[j] == key:
                gids[i] = gid
                counts[gid] += 1
                break
            j = <Py_ssize_t>((<uint64_t>j + 1) & mask)
    n_groups_out[0] = n_groups


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_groups(
    const int64_t[::1] gids,
    const int64_t[::1] offsets,
    int64_t[::1] cursor,
    int64_t[::1] group_rows,
) noexcept nogil:
    """Scatter right-row numbers into their group's slice, in right order."""
    cdef Py_ssize_t i, n = gids.shape[0]
    cdef int64_t g
    for i in range(n):
        g = gids[i]
        group_rows[offsets[g] + cursor[g]] = i
        cursor[g] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int64_t _probe_count(
    const int64_t[::1] keys,
    Py_ssize_t lo,
    Py_ssize_t hi,
    const int64_t[::1] slot_key,
    const int64_t[::1] slot_gid,
    const int64_t[::1] counts,
) noexcept nogil:
    """Output rows produced by probe rows [lo, hi)."""
    cdef uint64_t mask = <uint64_t>(slot_key.shape[0] - 1)
    cdef Py_ssize_t i, j
    cdef int64_t key, gid, total = 0
    for i in range(lo, hi):
        key = keys[i]
        j = <Py_ssize_t>(_hash(key) & mask)
        while True:
            gid = slot_gid[j]
            if gid == -1:
                break
            if slot_key[j] == key:
                total += counts[gid]
                break
            j = <Py_ssize_t>((<uint64_t>j + 1) & mask)
    return total


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _probe_fill(
    const int64_t[::1] keys,
    Py_ssize_t lo,
    Py_ssize_t hi,
    const int64_t[::1] slot_key,
    const int64_t[::1] slot_gid,
    const int64_t[::1] counts,
    const int64_t[::1] offsets,
    const int64_t[::1] group_rows,
    int64_t[::1] out_left,
    int64_t[::1] out_right,
    Py_ssize_t out_start,
) noexcept nogil:
    """Emit (left, right) index pairs for probe rows [lo, hi)."""
    cdef uint64_t mask = <uint64_t>(slot_key.shape[0] - 1)
    cdef Py_ssize_t i, j, pos = out_start
    cdef int64_t key, gid, k, off, cnt
    for i in range(lo, hi):
        key = keys[i]
        j = <Py_ssize_t>(_hash(key) & mask)
        while True:
            gid = slot_gid[j]
            if gid == -1:
                break
            if slot_key[j] == key:
                off = offsets[gid]
                cnt = counts[gid]
                for k in range(cnt):
                    out_left[pos] = i
                    out_right[pos] = group_rows[off + k]
                    pos += 1
                break
            j = <Py_ssize_t>((<uint64_t>j + 1) & mask)


def probe_count_chunk(
    const int64_t[::1] keys,
    Py_ssize_t lo,
    Py_ssize_t hi,
    const int64_t[::1] slot_key,
    const int64_t[::1] slot_gid,
    const int64_t[::1] counts,
):
    """Thread-friendly wrapper: count output rows for probe rows [lo, hi)."""
    with nogil:
        total = _probe_count(keys, lo, hi, slot_key, slot_gid, counts)
    return total


def probe_fill_chunk(
    const int64_t[::1] keys,
    Py_ssize_t lo,
    Py_ssize_t hi,
    const int64_t[::1] slot_key,
    const int64_t[::1] slot_gid,
    const int64_t[::1] counts,
    const int64_t[::1] offsets,
    const int64_t[::1] group_rows,
    int64_t[::1] out_left,
    int64_t[::1] out_right,
    Py_ssize_t out_start,
):
    """Thread-friendly wrapper: fill output pairs for probe rows [lo, hi)."""
    with nogil:
        _probe_fill(
            keys, lo, hi, slot_key, slot_gid, counts, offsets, group_rows,
            out_left, out_right, out_start,
        )


def build_join_table_i8(const int64_t[::1] keys):
    """Build the right-side hash table.

    Returns ``(slot_key, slot_gid, counts, offsets, group_rows)`` — the CSR
    group structure consumed by the probe kernels. ``counts``/``offsets`` are
    sized to the number of groups.
    """
    cdef Py_ssize_t n = keys.shape[0]
    cdef Py_ssize_t m = 8
    while m < 2 * n:
        m <<= 1
    slot_key = np.empty(m, dtype=np.int64)
    slot_gid = np.full(m, -1, dtype=np.int64)
    gids = np.empty(n, dtype=np.int64)
    counts_full = np.empty(n, dtype=np.int64)
    cdef int64_t n_groups = 0
    cdef int64_t[::1] slot_key_v = slot_key
    cdef int64_t[::1] slot_gid_v = slot_gid
    cdef int64_t[::1] gids_v = gids
    cdef int64_t[::1] counts_v = counts_full
    with nogil:
        _build(keys, slot_key_v, slot_gid_v, gids_v, counts_v, &n_groups)
    counts = counts_full[:n_groups]
    offsets = np.empty(n_groups + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    group_rows = np.empty(n, dtype=np.int64)
    cursor = np.zeros(n_groups, dtype=np.int64)
    cdef const int64_t[::1] offs_v = offsets[:n_groups]
    cdef int64_t[::1] cursor_v = cursor
    cdef int64_t[::1] rows_v = group_rows
    with nogil:
        _fill_groups(gids_v, offs_v, cursor_v, rows_v)
    return slot_key, slot_gid, np.asarray(counts), offsets, group_rows


# ---------------------------------------------------------------------------
# Radix partition for the partitioned parallel hash join.
#
# A monolithic hash join on a large key set is single-threaded on the build and
# does two full probe passes (count, fill). Partitioning both sides by key-hash
# into B cache-resident buckets lets the per-bucket joins run in parallel and
# keeps each hash table in cache. This is an O(N) counting sort (two passes), the
# same no-OpenMP pattern as lazy_radix.pyx; the per-bucket joins are driven by a
# Python thread pool over ``build_join_table_i8`` + ``probe_*_chunk``.
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def partition_by_bucket(const int64_t[::1] keys, int64_t n_buckets):
    """Counting-sort ``keys`` into ``n_buckets`` (power-of-two) hash buckets.

    Returns ``(sorted_keys, sorted_rows, bounds)`` where each bucket ``b`` is the
    contiguous slice ``[bounds[b], bounds[b+1])`` of ``sorted_keys`` (and the
    original row index in ``sorted_rows``). ``nogil`` two-pass O(N) counting sort.
    """
    cdef Py_ssize_t n = keys.shape[0]
    bucket_of = np.empty(n, dtype=np.int64)
    counts = np.zeros(n_buckets, dtype=np.int64)
    cdef int64_t[::1] bucket_v = bucket_of
    cdef int64_t[::1] counts_v = counts
    cdef Py_ssize_t i
    cdef int64_t b
    cdef uint64_t mask = <uint64_t>(n_buckets - 1)
    cdef uint64_t h
    # histogram first (need offsets before scatter)
    with nogil:
        for i in range(n):
            h = (<uint64_t>keys[i]) * <uint64_t>0x9E3779B97F4A7C15
            b = <int64_t>((h >> 40) & mask)
            bucket_v[i] = b
            counts_v[b] += 1
    bounds = np.empty(n_buckets + 1, dtype=np.int64)
    bounds[0] = 0
    np.cumsum(counts, out=bounds[1:])
    out_keys = np.empty(n, dtype=np.int64)
    out_rows = np.empty(n, dtype=np.int64)
    cursor = np.empty(n_buckets, dtype=np.int64)
    cdef int64_t[::1] outk_v = out_keys
    cdef int64_t[::1] outr_v = out_rows
    cdef int64_t[::1] cursor_v = cursor
    cdef const int64_t[::1] bounds_v = bounds[:n_buckets]
    cdef int64_t p
    with nogil:
        for i in range(n_buckets):
            cursor_v[i] = bounds_v[i]
        for i in range(n):
            b = bucket_v[i]
            p = cursor_v[b]
            cursor_v[b] = p + 1
            outk_v[p] = keys[i]
            outr_v[p] = i
    return out_keys, out_rows, bounds


# ---------------------------------------------------------------------------
# Fused per-partition build + probe + ROW-MAJOR payload gather.
#
# Variant of join_gather_bucket_f64 that gathers a row-major payload block
# ``rpay`` of shape (n_right, P): on each match it memcpy's the matched right
# row's P contiguous float64s into the output row. Row-major makes the gather
# one cache-line-friendly copy per match (measured ~2.2x faster than a
# column-major scalar/np.take gather at high P). The build side is small (dim
# table), so transposing its columns to a row-major block once is cheap; the
# output is itself a row-major block (the natural consolidated-block form a
# pandas-returning engine consumes, as pd.merge does). Unique build assumed.
# ---------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def join_gather_bucket_rm(
    const int64_t[::1] lkeys,
    const int64_t[::1] lrows,
    Py_ssize_t llo,
    Py_ssize_t lhi,
    const int64_t[::1] rkeys,
    const int64_t[::1] rrows,
    Py_ssize_t rlo,
    Py_ssize_t rhi,
    const double[:, ::1] rpay,
    double[:, ::1] out,
    int64_t[::1] out_lrow,
):
    """Fused inner-join + row-major gather for one partition. Returns count."""
    cdef Py_ssize_t nr = rhi - rlo
    cdef Py_ssize_t P = rpay.shape[1]
    cdef Py_ssize_t cursor = 0
    cdef Py_ssize_t i, j
    cdef int64_t key, lrow, rrow
    cdef uint64_t slot, mask
    cdef Py_ssize_t m = 8
    cdef size_t rowbytes = <size_t>P * sizeof(double)
    cdef int64_t* slot_key
    cdef int64_t* slot_row
    if nr == 0 or lhi <= llo:
        return 0
    while m < 2 * nr:
        m <<= 1
    with nogil:
        slot_key = <int64_t*>malloc(m * sizeof(int64_t))
        slot_row = <int64_t*>malloc(m * sizeof(int64_t))
        if slot_key == NULL or slot_row == NULL:
            if slot_key != NULL:
                free(slot_key)
            if slot_row != NULL:
                free(slot_row)
            with gil:
                raise MemoryError()
        mask = <uint64_t>(m - 1)
        for i in range(m):
            slot_row[i] = -1
        for j in range(rlo, rhi):
            key = rkeys[j]
            slot = _hash(key) & mask
            while slot_row[slot] != -1:
                slot = (slot + 1) & mask
            slot_key[slot] = key
            slot_row[slot] = rrows[j]
        for i in range(llo, lhi):
            key = lkeys[i]
            slot = _hash(key) & mask
            while slot_row[slot] != -1:
                if slot_key[slot] == key:
                    lrow = lrows[i]
                    rrow = slot_row[slot]
                    if P > 0:
                        memcpy(&out[cursor, 0], &rpay[rrow, 0], rowbytes)
                    out_lrow[cursor] = lrow
                    cursor += 1
                    break
                slot = (slot + 1) & mask
        free(slot_key)
        free(slot_row)
    return cursor
