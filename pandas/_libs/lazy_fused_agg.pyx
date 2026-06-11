"""Fused filter+aggregate kernel for the lazy engine's hot path.

Single pass over the input: range predicates evaluate into an L1-resident
chunk mask (one tight vectorizable loop per predicate), then accumulators
consume the masked chunk — the filtered intermediate never materializes.
Probe that motivated this (q6 @ SF-3, 18M rows): fused C loop 10.8 ms on
8 threads vs 70 ms through materializing operators and ~26 ms for Polars.

Threading is driven from Python over row slices (the lazy_join pattern);
this kernel releases the GIL. Predicates are closed ranges [lo, hi] over
int64 (datetimes) or float64 columns; open bounds pass INT64_MIN/MAX or
±inf. Aggregations: 0 = sum(a), 1 = sum(a*b), 2 = count, 3 = min(a),
4 = max(a).
"""

cimport cython
from libc.stdlib cimport (
    free,
    malloc,
)

import numpy as np

cimport numpy as cnp

cnp.import_array()

DEF CHUNK = 4096


@cython.boundscheck(False)
@cython.wraparound(False)
def fused_filter_aggs(
    list i64_cols,
    cnp.ndarray[cnp.int64_t, ndim=1] i64_lo,
    cnp.ndarray[cnp.int64_t, ndim=1] i64_hi,
    list f64_cols,
    cnp.ndarray[cnp.float64_t, ndim=1] f64_lo,
    cnp.ndarray[cnp.float64_t, ndim=1] f64_hi,
    cnp.ndarray[cnp.int64_t, ndim=1] agg_kinds,
    list agg_a,
    list agg_b,
    Py_ssize_t start,
    Py_ssize_t end,
):
    """Return (accumulators: float64[n_aggs], rows_kept: int64) over [start, end)."""
    cdef Py_ssize_t n_i64 = len(i64_cols)
    cdef Py_ssize_t n_f64 = len(f64_cols)
    cdef Py_ssize_t n_aggs = agg_kinds.shape[0]

    # Flatten column pointers into typed memoryview arrays of pointers.
    cdef cnp.int64_t** ip = NULL
    cdef double** fp = NULL
    cdef double** ap = NULL
    cdef double** bp = NULL
    cdef cnp.ndarray arr
    cdef Py_ssize_t k
    cdef cnp.int64_t kept = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_v

    ip = <cnp.int64_t**>malloc(max(n_i64, 1) * sizeof(cnp.int64_t*))
    fp = <double**>malloc(max(n_f64, 1) * sizeof(double*))
    ap = <double**>malloc(max(n_aggs, 1) * sizeof(double*))
    bp = <double**>malloc(max(n_aggs, 1) * sizeof(double*))
    if ip == NULL or fp == NULL or ap == NULL or bp == NULL:
        free(ip)
        free(fp)
        free(ap)
        free(bp)
        raise MemoryError()

    try:
        for k in range(n_i64):
            arr = i64_cols[k]
            ip[k] = <cnp.int64_t*>arr.data
        for k in range(n_f64):
            arr = f64_cols[k]
            fp[k] = <double*>arr.data
        for k in range(n_aggs):
            arr = agg_a[k]
            ap[k] = <double*>arr.data
            if agg_b[k] is not None:
                arr = agg_b[k]
                bp[k] = <double*>arr.data
            else:
                bp[k] = NULL

        out = np.zeros(n_aggs, dtype=np.float64)
        # min/max slots start "unset": NaN sentinel checked via x != x.
        out[np.isin(np.asarray(agg_kinds), (3, 4))] = np.nan
        out_v = out
        with nogil:
            kept = _run(
                ip, <cnp.int64_t*>i64_lo.data, <cnp.int64_t*>i64_hi.data, n_i64,
                fp, <double*>f64_lo.data, <double*>f64_hi.data, n_f64,
                <cnp.int64_t*>agg_kinds.data, ap, bp, n_aggs,
                start, end, <double*>out_v.data,
            )
        return out, kept
    finally:
        free(ip)
        free(fp)
        free(ap)
        free(bp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.int64_t _run(
    cnp.int64_t** ip, cnp.int64_t* ilo, cnp.int64_t* ihi, Py_ssize_t n_i64,
    double** fp, double* flo, double* fhi, Py_ssize_t n_f64,
    cnp.int64_t* kinds, double** ap, double** bp, Py_ssize_t n_aggs,
    Py_ssize_t start, Py_ssize_t end, double* out,
) noexcept nogil:
    cdef cnp.uint8_t mask[CHUNK]
    cdef Py_ssize_t c, i, m, p, a
    cdef cnp.int64_t kept = 0
    cdef cnp.int64_t* icol
    cdef double* fcol
    cdef double* acol
    cdef double* bcol
    cdef double acc, mn, mx, v
    cdef cnp.int64_t lo_i, hi_i
    cdef double lo_f, hi_f
    cdef bint first

    for c in range(start, end, CHUNK):
        m = end - c
        if m > CHUNK:
            m = CHUNK
        for i in range(m):
            mask[i] = 1
        for p in range(n_i64):
            icol = ip[p] + c
            lo_i = ilo[p]
            hi_i = ihi[p]
            for i in range(m):
                mask[i] &= (icol[i] >= lo_i) & (icol[i] <= hi_i)
        for p in range(n_f64):
            fcol = fp[p] + c
            lo_f = flo[p]
            hi_f = fhi[p]
            for i in range(m):
                mask[i] &= (fcol[i] >= lo_f) & (fcol[i] <= hi_f)
        for a in range(n_aggs):
            acol = ap[a] + c if ap[a] != NULL else NULL
            if kinds[a] == 0:  # sum(a)
                acc = 0.0
                for i in range(m):
                    acc += acol[i] if mask[i] else 0.0
                out[a] += acc
            elif kinds[a] == 1:  # sum(a*b)
                bcol = bp[a] + c
                acc = 0.0
                for i in range(m):
                    acc += acol[i] * bcol[i] if mask[i] else 0.0
                out[a] += acc
            elif kinds[a] == 2:  # count
                pass  # counted once below
            elif kinds[a] == 3:  # min(a)
                first = out[a] != out[a]  # NaN sentinel means "unset"
                mn = out[a]
                for i in range(m):
                    if mask[i]:
                        v = acol[i]
                        if first or v < mn:
                            mn = v
                            first = 0
                out[a] = mn
            elif kinds[a] == 4:  # max(a)
                first = out[a] != out[a]
                mx = out[a]
                for i in range(m):
                    if mask[i]:
                        v = acol[i]
                        if first or v > mx:
                            mx = v
                            first = 0
                out[a] = mx
        for i in range(m):
            kept += mask[i]
    for a in range(n_aggs):
        if kinds[a] == 2:
            out[a] = <double>kept
    return kept
