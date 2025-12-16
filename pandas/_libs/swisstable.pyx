# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Cython wrapper for Swiss Table hash maps

This provides Python-accessible interfaces to the C Swiss table implementation
for all pandas numeric types.

Available types:
- SwissInt64Map: int64_t -> size_t
- SwissUInt64Map: uint64_t -> size_t
- SwissInt32Map: int32_t -> size_t
- SwissUInt32Map: uint32_t -> size_t
- SwissInt16Map: int16_t -> size_t
- SwissUInt16Map: uint16_t -> size_t
- SwissInt8Map: int8_t -> size_t
- SwissUInt8Map: uint8_t -> size_t
- SwissFloat64Map: double -> size_t (with NaN handling)
- SwissFloat32Map: float -> size_t (with NaN handling)
- SwissComplex64Map: complex64 -> size_t (with NaN handling)
- SwissComplex128Map: complex128 -> size_t (with NaN handling)

Integration Methods (matching khash HashTable API):
- map_locations(): Build table mapping values to their array positions
- lookup(): Look up array values, returning positions or -1 for misses
- ismember_*(): Standalone functions for membership testing (isin)
- unique(): Get unique values with optional inverse mapping
- factorize(): Get unique values and labels (for categorical encoding)
- value_count_*(): Standalone functions for counting value occurrences
- duplicated_*(): Standalone functions for finding duplicated values
"""

cimport cython
from libc.math cimport isnan
from libc.stddef cimport size_t
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

import numpy as np

cimport numpy as cnp
from numpy cimport (
    intp_t,
    ndarray,
    uint8_t as np_uint8_t,
)

# intp_t matches ssize_t in the C header
ctypedef intp_t c_ssize_t

cnp.import_array()

# Core functions from swisstable_core.h for inlining
cdef extern from "swisstable/swisstable_core.h" nogil:
    int8_t swiss_h2(uint64_t hash)

# Hash functions for int64
cdef extern from "swisstable/swisstable_types.h" nogil:
    uint64_t hash_int64(int64_t key)

# External declarations for all Swiss Table types
cdef extern from "swisstable/swisstable_types.h" nogil:
    # Int64
    ctypedef struct SwissTable_int64:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask
        int8_t *ctrl
        int64_t *keys
        size_t *vals

    SwissTable_int64* swiss_int64_init()
    SwissTable_int64* swiss_int64_init_with_capacity(size_t capacity)
    void swiss_int64_destroy(SwissTable_int64 *table)
    void swiss_int64_clear(SwissTable_int64 *table)
    size_t swiss_int64_find(const SwissTable_int64 *table, int64_t key)
    int swiss_int64_insert(SwissTable_int64 *table, int64_t key, size_t val)
    bint swiss_int64_get(const SwissTable_int64 *table, int64_t key, size_t *val_out)
    bint swiss_int64_contains(const SwissTable_int64 *table, int64_t key)
    bint swiss_int64_delete(SwissTable_int64 *table, int64_t key)
    size_t swiss_int64_size(const SwissTable_int64 *table)
    size_t swiss_int64_capacity(const SwissTable_int64 *table)
    size_t swiss_int64_iter_next(const SwissTable_int64 *table, size_t index)
    int64_t swiss_int64_key_at(const SwissTable_int64 *table, size_t index)
    size_t swiss_int64_val_at(const SwissTable_int64 *table, size_t index)
    int swiss_int64_insert_key_only(SwissTable_int64 *table, int64_t key)
    int swiss_int64_build_set(SwissTable_int64 *table, const int64_t *keys, size_t n)
    int swiss_int64_insert_if_absent(SwissTable_int64 *table, int64_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_int64_increment(SwissTable_int64 *table, int64_t key)
    bint swiss_int64_reserve(SwissTable_int64 *table, size_t want)
    # Batch operations for reduced function call overhead
    int swiss_int64_map_locations(SwissTable_int64 *table, const int64_t *keys, size_t n)  # noqa: E501
    void swiss_int64_lookup_batch(const SwissTable_int64 *table, const int64_t *keys, size_t n, int64_t *locs)  # noqa: E501
    void swiss_int64_contains_batch(const SwissTable_int64 *table, const int64_t *keys, size_t n, uint8_t *result)  # noqa: E501
    int64_t swiss_int64_unique(SwissTable_int64 *table, const int64_t *keys, size_t n, int64_t *uniques_out)  # noqa: E501
    int64_t swiss_int64_unique_with_inverse(SwissTable_int64 *table, const int64_t *keys, size_t n, int64_t *uniques_out, int64_t *labels_out)  # noqa: E501
    # Batch factorize with native intp_t (ssize_t) labels - no conversion needed
    c_ssize_t swiss_int64_factorize_batch(SwissTable_int64 *table, const int64_t *keys, size_t n, int64_t *uniques_out, c_ssize_t *labels_out)  # noqa: E501
    # Batch value_count - stores indices for direct count access
    c_ssize_t swiss_int64_value_count_batch(SwissTable_int64 *table, const int64_t *keys, size_t n, int64_t *keys_out, size_t *indices_out)  # noqa: E501

    # UInt64
    ctypedef struct SwissTable_uint64:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_uint64* swiss_uint64_init()
    SwissTable_uint64* swiss_uint64_init_with_capacity(size_t capacity)
    void swiss_uint64_destroy(SwissTable_uint64 *table)
    void swiss_uint64_clear(SwissTable_uint64 *table)
    int swiss_uint64_insert(SwissTable_uint64 *table, uint64_t key, size_t val)
    bint swiss_uint64_get(const SwissTable_uint64 *table, uint64_t key, size_t *val_out)
    bint swiss_uint64_contains(const SwissTable_uint64 *table, uint64_t key)
    bint swiss_uint64_delete(SwissTable_uint64 *table, uint64_t key)
    size_t swiss_uint64_size(const SwissTable_uint64 *table)
    size_t swiss_uint64_capacity(const SwissTable_uint64 *table)
    size_t swiss_uint64_iter_next(const SwissTable_uint64 *table, size_t index)
    uint64_t swiss_uint64_key_at(const SwissTable_uint64 *table, size_t index)
    size_t swiss_uint64_val_at(const SwissTable_uint64 *table, size_t index)
    int swiss_uint64_insert_key_only(SwissTable_uint64 *table, uint64_t key)
    int swiss_uint64_build_set(SwissTable_uint64 *table, const uint64_t *keys, size_t n)
    int swiss_uint64_insert_if_absent(SwissTable_uint64 *table, uint64_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_uint64_increment(SwissTable_uint64 *table, uint64_t key)
    bint swiss_uint64_reserve(SwissTable_uint64 *table, size_t want)

    # Int32
    ctypedef struct SwissTable_int32:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_int32* swiss_int32_init()
    SwissTable_int32* swiss_int32_init_with_capacity(size_t capacity)
    void swiss_int32_destroy(SwissTable_int32 *table)
    void swiss_int32_clear(SwissTable_int32 *table)
    int swiss_int32_insert(SwissTable_int32 *table, int32_t key, size_t val)
    bint swiss_int32_get(const SwissTable_int32 *table, int32_t key, size_t *val_out)
    bint swiss_int32_contains(const SwissTable_int32 *table, int32_t key)
    bint swiss_int32_delete(SwissTable_int32 *table, int32_t key)
    size_t swiss_int32_size(const SwissTable_int32 *table)
    size_t swiss_int32_capacity(const SwissTable_int32 *table)
    size_t swiss_int32_iter_next(const SwissTable_int32 *table, size_t index)
    int32_t swiss_int32_key_at(const SwissTable_int32 *table, size_t index)
    size_t swiss_int32_val_at(const SwissTable_int32 *table, size_t index)
    int swiss_int32_insert_key_only(SwissTable_int32 *table, int32_t key)
    int swiss_int32_build_set(SwissTable_int32 *table, const int32_t *keys, size_t n)
    int swiss_int32_insert_if_absent(SwissTable_int32 *table, int32_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_int32_increment(SwissTable_int32 *table, int32_t key)
    bint swiss_int32_reserve(SwissTable_int32 *table, size_t want)

    # UInt32
    ctypedef struct SwissTable_uint32:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_uint32* swiss_uint32_init()
    SwissTable_uint32* swiss_uint32_init_with_capacity(size_t capacity)
    void swiss_uint32_destroy(SwissTable_uint32 *table)
    void swiss_uint32_clear(SwissTable_uint32 *table)
    int swiss_uint32_insert(SwissTable_uint32 *table, uint32_t key, size_t val)
    bint swiss_uint32_get(const SwissTable_uint32 *table, uint32_t key, size_t *val_out)
    bint swiss_uint32_contains(const SwissTable_uint32 *table, uint32_t key)
    bint swiss_uint32_delete(SwissTable_uint32 *table, uint32_t key)
    size_t swiss_uint32_size(const SwissTable_uint32 *table)
    size_t swiss_uint32_capacity(const SwissTable_uint32 *table)
    size_t swiss_uint32_iter_next(const SwissTable_uint32 *table, size_t index)
    uint32_t swiss_uint32_key_at(const SwissTable_uint32 *table, size_t index)
    size_t swiss_uint32_val_at(const SwissTable_uint32 *table, size_t index)
    int swiss_uint32_insert_key_only(SwissTable_uint32 *table, uint32_t key)
    int swiss_uint32_build_set(SwissTable_uint32 *table, const uint32_t *keys, size_t n)
    int swiss_uint32_insert_if_absent(SwissTable_uint32 *table, uint32_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_uint32_increment(SwissTable_uint32 *table, uint32_t key)
    bint swiss_uint32_reserve(SwissTable_uint32 *table, size_t want)

    # Int16
    ctypedef struct SwissTable_int16:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_int16* swiss_int16_init()
    SwissTable_int16* swiss_int16_init_with_capacity(size_t capacity)
    void swiss_int16_destroy(SwissTable_int16 *table)
    void swiss_int16_clear(SwissTable_int16 *table)
    int swiss_int16_insert(SwissTable_int16 *table, int16_t key, size_t val)
    bint swiss_int16_get(const SwissTable_int16 *table, int16_t key, size_t *val_out)
    bint swiss_int16_contains(const SwissTable_int16 *table, int16_t key)
    bint swiss_int16_delete(SwissTable_int16 *table, int16_t key)
    size_t swiss_int16_size(const SwissTable_int16 *table)
    size_t swiss_int16_capacity(const SwissTable_int16 *table)
    size_t swiss_int16_iter_next(const SwissTable_int16 *table, size_t index)
    int16_t swiss_int16_key_at(const SwissTable_int16 *table, size_t index)
    size_t swiss_int16_val_at(const SwissTable_int16 *table, size_t index)
    int swiss_int16_insert_key_only(SwissTable_int16 *table, int16_t key)
    int swiss_int16_build_set(SwissTable_int16 *table, const int16_t *keys, size_t n)
    int swiss_int16_insert_if_absent(SwissTable_int16 *table, int16_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_int16_increment(SwissTable_int16 *table, int16_t key)
    bint swiss_int16_reserve(SwissTable_int16 *table, size_t want)

    # UInt16
    ctypedef struct SwissTable_uint16:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_uint16* swiss_uint16_init()
    SwissTable_uint16* swiss_uint16_init_with_capacity(size_t capacity)
    void swiss_uint16_destroy(SwissTable_uint16 *table)
    void swiss_uint16_clear(SwissTable_uint16 *table)
    int swiss_uint16_insert(SwissTable_uint16 *table, uint16_t key, size_t val)
    bint swiss_uint16_get(const SwissTable_uint16 *table, uint16_t key, size_t *val_out)
    bint swiss_uint16_contains(const SwissTable_uint16 *table, uint16_t key)
    bint swiss_uint16_delete(SwissTable_uint16 *table, uint16_t key)
    size_t swiss_uint16_size(const SwissTable_uint16 *table)
    size_t swiss_uint16_capacity(const SwissTable_uint16 *table)
    size_t swiss_uint16_iter_next(const SwissTable_uint16 *table, size_t index)
    uint16_t swiss_uint16_key_at(const SwissTable_uint16 *table, size_t index)
    size_t swiss_uint16_val_at(const SwissTable_uint16 *table, size_t index)
    int swiss_uint16_insert_key_only(SwissTable_uint16 *table, uint16_t key)
    int swiss_uint16_build_set(SwissTable_uint16 *table, const uint16_t *keys, size_t n)
    int swiss_uint16_insert_if_absent(SwissTable_uint16 *table, uint16_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_uint16_increment(SwissTable_uint16 *table, uint16_t key)
    bint swiss_uint16_reserve(SwissTable_uint16 *table, size_t want)

    # Int8
    ctypedef struct SwissTable_int8:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_int8* swiss_int8_init()
    SwissTable_int8* swiss_int8_init_with_capacity(size_t capacity)
    void swiss_int8_destroy(SwissTable_int8 *table)
    void swiss_int8_clear(SwissTable_int8 *table)
    int swiss_int8_insert(SwissTable_int8 *table, int8_t key, size_t val)
    bint swiss_int8_get(const SwissTable_int8 *table, int8_t key, size_t *val_out)
    bint swiss_int8_contains(const SwissTable_int8 *table, int8_t key)
    bint swiss_int8_delete(SwissTable_int8 *table, int8_t key)
    size_t swiss_int8_size(const SwissTable_int8 *table)
    size_t swiss_int8_capacity(const SwissTable_int8 *table)
    size_t swiss_int8_iter_next(const SwissTable_int8 *table, size_t index)
    int8_t swiss_int8_key_at(const SwissTable_int8 *table, size_t index)
    size_t swiss_int8_val_at(const SwissTable_int8 *table, size_t index)
    int swiss_int8_insert_key_only(SwissTable_int8 *table, int8_t key)
    int swiss_int8_build_set(SwissTable_int8 *table, const int8_t *keys, size_t n)
    int swiss_int8_insert_if_absent(SwissTable_int8 *table, int8_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_int8_increment(SwissTable_int8 *table, int8_t key)
    bint swiss_int8_reserve(SwissTable_int8 *table, size_t want)

    # UInt8
    ctypedef struct SwissTable_uint8:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_uint8* swiss_uint8_init()
    SwissTable_uint8* swiss_uint8_init_with_capacity(size_t capacity)
    void swiss_uint8_destroy(SwissTable_uint8 *table)
    void swiss_uint8_clear(SwissTable_uint8 *table)
    int swiss_uint8_insert(SwissTable_uint8 *table, uint8_t key, size_t val)
    bint swiss_uint8_get(const SwissTable_uint8 *table, uint8_t key, size_t *val_out)
    bint swiss_uint8_contains(const SwissTable_uint8 *table, uint8_t key)
    bint swiss_uint8_delete(SwissTable_uint8 *table, uint8_t key)
    size_t swiss_uint8_size(const SwissTable_uint8 *table)
    size_t swiss_uint8_capacity(const SwissTable_uint8 *table)
    size_t swiss_uint8_iter_next(const SwissTable_uint8 *table, size_t index)
    uint8_t swiss_uint8_key_at(const SwissTable_uint8 *table, size_t index)
    size_t swiss_uint8_val_at(const SwissTable_uint8 *table, size_t index)
    int swiss_uint8_insert_key_only(SwissTable_uint8 *table, uint8_t key)
    int swiss_uint8_build_set(SwissTable_uint8 *table, const uint8_t *keys, size_t n)
    int swiss_uint8_insert_if_absent(SwissTable_uint8 *table, uint8_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_uint8_increment(SwissTable_uint8 *table, uint8_t key)
    bint swiss_uint8_reserve(SwissTable_uint8 *table, size_t want)

    # Float64
    ctypedef struct SwissTable_float64:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_float64* swiss_float64_init()
    SwissTable_float64* swiss_float64_init_with_capacity(size_t capacity)
    void swiss_float64_destroy(SwissTable_float64 *table)
    void swiss_float64_clear(SwissTable_float64 *table)
    int swiss_float64_insert(SwissTable_float64 *table, double key, size_t val)
    bint swiss_float64_get(const SwissTable_float64 *table, double key, size_t *val_out)
    bint swiss_float64_contains(const SwissTable_float64 *table, double key)
    bint swiss_float64_delete(SwissTable_float64 *table, double key)
    size_t swiss_float64_size(const SwissTable_float64 *table)
    size_t swiss_float64_capacity(const SwissTable_float64 *table)
    size_t swiss_float64_iter_next(const SwissTable_float64 *table, size_t index)
    double swiss_float64_key_at(const SwissTable_float64 *table, size_t index)
    size_t swiss_float64_val_at(const SwissTable_float64 *table, size_t index)
    int swiss_float64_insert_key_only(SwissTable_float64 *table, double key)
    int swiss_float64_build_set(SwissTable_float64 *table, const double *keys, size_t n)
    int swiss_float64_insert_if_absent(SwissTable_float64 *table, double key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_float64_increment(SwissTable_float64 *table, double key)
    bint swiss_float64_reserve(SwissTable_float64 *table, size_t want)

    # Float32
    ctypedef struct SwissTable_float32:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_float32* swiss_float32_init()
    SwissTable_float32* swiss_float32_init_with_capacity(size_t capacity)
    void swiss_float32_destroy(SwissTable_float32 *table)
    void swiss_float32_clear(SwissTable_float32 *table)
    int swiss_float32_insert(SwissTable_float32 *table, float key, size_t val)
    bint swiss_float32_get(const SwissTable_float32 *table, float key, size_t *val_out)
    bint swiss_float32_contains(const SwissTable_float32 *table, float key)
    bint swiss_float32_delete(SwissTable_float32 *table, float key)
    size_t swiss_float32_size(const SwissTable_float32 *table)
    size_t swiss_float32_capacity(const SwissTable_float32 *table)
    size_t swiss_float32_iter_next(const SwissTable_float32 *table, size_t index)
    float swiss_float32_key_at(const SwissTable_float32 *table, size_t index)
    size_t swiss_float32_val_at(const SwissTable_float32 *table, size_t index)
    int swiss_float32_insert_key_only(SwissTable_float32 *table, float key)
    int swiss_float32_build_set(SwissTable_float32 *table, const float *keys, size_t n)
    int swiss_float32_insert_if_absent(SwissTable_float32 *table, float key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_float32_increment(SwissTable_float32 *table, float key)
    bint swiss_float32_reserve(SwissTable_float32 *table, size_t want)

    # Complex types
    ctypedef struct swiss_complex64_t:
        float real
        float imag

    ctypedef struct swiss_complex128_t:
        double real
        double imag

    # Complex64
    ctypedef struct SwissTable_complex64:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_complex64* swiss_complex64_init()
    SwissTable_complex64* swiss_complex64_init_with_capacity(size_t capacity)
    void swiss_complex64_destroy(SwissTable_complex64 *table)
    void swiss_complex64_clear(SwissTable_complex64 *table)
    int swiss_complex64_insert(SwissTable_complex64 *table, swiss_complex64_t key, size_t val)  # noqa: E501
    int swiss_complex64_insert_if_absent(SwissTable_complex64 *table, swiss_complex64_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_complex64_insert_key_only(SwissTable_complex64 *table, swiss_complex64_t key)  # noqa: E501
    int swiss_complex64_build_set(SwissTable_complex64 *table, const swiss_complex64_t *keys, size_t n)  # noqa: E501
    bint swiss_complex64_get(const SwissTable_complex64 *table, swiss_complex64_t key, size_t *val_out)  # noqa: E501
    bint swiss_complex64_contains(const SwissTable_complex64 *table, swiss_complex64_t key)  # noqa: E501
    bint swiss_complex64_delete(SwissTable_complex64 *table, swiss_complex64_t key)
    size_t swiss_complex64_size(const SwissTable_complex64 *table)
    size_t swiss_complex64_capacity(const SwissTable_complex64 *table)
    size_t swiss_complex64_iter_next(const SwissTable_complex64 *table, size_t index)
    swiss_complex64_t swiss_complex64_key_at(const SwissTable_complex64 *table, size_t index)  # noqa: E501
    size_t swiss_complex64_val_at(const SwissTable_complex64 *table, size_t index)
    bint swiss_complex64_reserve(SwissTable_complex64 *table, size_t want)

    # Complex128
    ctypedef struct SwissTable_complex128:
        size_t capacity
        size_t size
        size_t growth_left
        size_t mask

    SwissTable_complex128* swiss_complex128_init()
    SwissTable_complex128* swiss_complex128_init_with_capacity(size_t capacity)
    void swiss_complex128_destroy(SwissTable_complex128 *table)
    void swiss_complex128_clear(SwissTable_complex128 *table)
    int swiss_complex128_insert(SwissTable_complex128 *table, swiss_complex128_t key, size_t val)  # noqa: E501
    int swiss_complex128_insert_if_absent(SwissTable_complex128 *table, swiss_complex128_t key, size_t new_val, size_t *val_out)  # noqa: E501
    int swiss_complex128_insert_key_only(SwissTable_complex128 *table, swiss_complex128_t key)  # noqa: E501
    int swiss_complex128_build_set(SwissTable_complex128 *table, const swiss_complex128_t *keys, size_t n)  # noqa: E501
    bint swiss_complex128_get(const SwissTable_complex128 *table, swiss_complex128_t key, size_t *val_out)  # noqa: E501
    bint swiss_complex128_contains(const SwissTable_complex128 *table, swiss_complex128_t key)  # noqa: E501
    bint swiss_complex128_delete(SwissTable_complex128 *table, swiss_complex128_t key)
    size_t swiss_complex128_size(const SwissTable_complex128 *table)
    size_t swiss_complex128_capacity(const SwissTable_complex128 *table)
    size_t swiss_complex128_iter_next(const SwissTable_complex128 *table, size_t index)
    swiss_complex128_t swiss_complex128_key_at(const SwissTable_complex128 *table, size_t index)  # noqa: E501
    size_t swiss_complex128_val_at(const SwissTable_complex128 *table, size_t index)
    bint swiss_complex128_reserve(SwissTable_complex128 *table, size_t want)


# =============================================================================
# Inline helper functions for hot paths
# These avoid C function call overhead by directly accessing table internals
# =============================================================================

# Control byte constants
DEF CTRL_EMPTY = -128  # 0x80
DEF CTRL_DELETED = -2  # 0xFE


@cython.cdivision(True)
cdef inline bint _is_full(int8_t ctrl) noexcept nogil:
    """Check if control byte indicates occupied slot (0x00-0x7F)"""
    return ctrl >= 0


# =============================================================================
# Int64 Map
# =============================================================================
cdef class SwissInt64Map:
    """
    Swiss Table hash map: int64 -> size_t

    High-performance hash table using SIMD-accelerated probing.
    """
    cdef SwissTable_int64 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_int64_init_with_capacity(size_hint)
        else:
            self.table = swiss_int64_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_int64_destroy(self.table)

    def insert(self, int64_t key, size_t value):
        cdef int ret = swiss_int64_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, int64_t key, default=None):
        cdef size_t val
        if swiss_int64_get(self.table, key, &val):
            return val
        return default

    def delete(self, int64_t key):
        return swiss_int64_delete(self.table, key)

    def clear(self):
        swiss_int64_clear(self.table)

    def __getitem__(self, int64_t key):
        cdef size_t val
        if swiss_int64_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, int64_t key, size_t value):
        cdef int ret = swiss_int64_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, int64_t key):
        if not swiss_int64_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, int64_t key):
        return swiss_int64_contains(self.table, key)

    def __len__(self):
        return swiss_int64_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_int64_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_int64_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_int64_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_int64_key_at(self.table, index),
                       swiss_int64_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_int64_size(self.table)

    @property
    def capacity(self):
        return swiss_int64_capacity(self.table)

    @property
    def load_factor(self):
        cdef size_t cap = swiss_int64_capacity(self.table)
        if cap == 0:
            return 0.0
        return <double>swiss_int64_size(self.table) / <double>cap

    def __repr__(self):
        return f"SwissInt64Map(size={self.size}, capacity={self.capacity})"

    # =========================================================================
    # Integration Methods (khash HashTable API compatibility)
    # =========================================================================

    def map_locations(self, const int64_t[:] values) -> None:
        """
        Build table mapping values to their array positions.

        For each value in the input array, stores (value -> index) in the table.
        If a value appears multiple times, the last occurrence wins.

        Parameters
        ----------
        values : ndarray[int64]
            Array of values to map

        Notes
        -----
        This matches the khash HashTable.map_locations() API.
        Used in libindex, safe_sort.
        """
        cdef:
            Py_ssize_t n = len(values)
            int ret = 0

        with nogil:
            ret = swiss_int64_map_locations(self.table, &values[0], <size_t>n)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const int64_t[:] values) -> ndarray:
        """
        Look up array values in table, returning positions.

        Parameters
        ----------
        values : ndarray[int64]
            Array of values to look up

        Returns
        -------
        ndarray[intp_t]
            Array of positions. -1 indicates value not found.

        Notes
        -----
        This matches the khash HashTable.lookup() API.
        Used in safe_sort, IndexEngine.get_indexer.
        """
        cdef:
            Py_ssize_t n = len(values)
            int64_t[::1] locs = np.empty(n, dtype=np.int64)

        with nogil:
            swiss_int64_lookup_batch(self.table, &values[0], <size_t>n, &locs[0])

        return np.asarray(locs)

    def unique(self, const int64_t[:] values, bint return_inverse=False,
               object mask=None):
        """
        Calculate unique values and optionally their inverse mapping.

        Parameters
        ----------
        values : ndarray[int64]
            Array of values to get uniques from
        return_inverse : bool, default False
            Whether to return the inverse mapping (labels)
        mask : ndarray[uint8], optional
            If provided, mask[i] == 1 indicates values[i] is NA/missing.
            When mask is provided, returns (uniques, result_mask).

        Returns
        -------
        uniques : ndarray[int64]
            Unique values in order of first appearance
        labels : ndarray[intp_t] (only if return_inverse=True)
            Mapping from values to position in uniques
        result_mask : ndarray[uint8] (only if mask is provided)
            Mask indicating which uniques are NA (1=NA, 0=valid)
        """
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            int64_t val
            size_t idx
            int ret = 0
            int64_t[::1] uniques_arr = np.empty(n, dtype=np.int64)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        # Masked value - treat as NA
                        if not seen_na:
                            # First NA - add to uniques
                            uniques_arr[count] = values[i]  # Value irrelevant for NA
                            result_mask_arr[count] = 1  # Mark as NA
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            # Subsequent NA - find previous NA index
                            # Since NA is always first in the unique order among NAs
                            # we need to find it by scanning result_mask
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_int64_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0  # Not NA
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            # No inverse needed
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        # Masked value - treat as NA
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1  # Mark as NA
                            seen_na = True
                            count += 1
                        # Subsequent NAs are duplicates, skip
                    else:
                        val = values[i]
                        ret = swiss_int64_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0  # Not NA
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const int64_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """
        Calculate unique values and labels for categorical encoding.

        Missing values handling depends on ignore_na:
        - ignore_na=True (default): NAs excluded from uniques, labels=na_sentinel
        - ignore_na=False: NA values are included as a unique value

        Parameters
        ----------
        values : ndarray[int64]
            Array of values to factorize
        na_sentinel : Py_ssize_t, default -1
            Value to use for missing data in labels (when ignore_na=True)
        na_value : object, default None
            Additional value to treat as missing (not applicable for int64)
        mask : ndarray[uint8], optional
            If provided, mask[i] == 1 indicates values[i] is NA/missing.
        ignore_na : bool, default True
            If True, NA values get na_sentinel label and are excluded from uniques.
            If False, NA values are treated as a unique value.

        Returns
        -------
        uniques : ndarray[int64]
            Unique values (excluding missing if ignore_na=True)
        labels : ndarray[intp_t]
            Labels mapping values to uniques (na_sentinel for missing if ignore_na=True)
        """
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            int64_t val
            size_t found_idx
            int ret = 0
            int64_t[::1] uniques_arr = np.empty(n, dtype=np.int64)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_int64_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        # Fast path: no mask - use C batch function for maximum performance
        if not uses_mask:
            with nogil:
                count = swiss_int64_factorize_batch(
                    self.table, &values[0], <size_t>n,
                    &uniques_arr[0], &labels[0]
                )
            if count == -1:
                raise MemoryError("Failed to insert into Swiss table")
            uniques = np.asarray(uniques_arr)[:count].copy()
            return uniques, np.asarray(labels)

        # Slow path: handle masked values
        mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        with nogil:
            for i in range(n):
                if mask_view[i]:
                    # Masked value - treat as NA
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        # Include NA as unique value
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                else:
                    val = values[i]
                    # Find-first approach: check if key exists before insert
                    found_idx = swiss_int64_find(self.table, val)
                    if found_idx != self.table.capacity:
                        # Key exists - common case for duplicates
                        labels[i] = <intp_t>swiss_int64_val_at(self.table, found_idx)
                    else:
                        # Key doesn't exist - insert it
                        ret = swiss_int64_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        # Trim to actual size
        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


# =============================================================================
# Standalone Functions for Integration
# =============================================================================

def ismember_int64(const int64_t[:] arr, const int64_t[:] values) -> ndarray:
    """
    Return boolean array indicating membership of arr elements in values.

    This is the Swiss Table version of isin() for int64 arrays.
    Swiss Tables excel at lookup-miss-heavy operations like this.

    Parameters
    ----------
    arr : ndarray[int64]
        Array to check for membership
    values : ndarray[int64]
        Set of values to check against

    Returns
    -------
    ndarray[bool]
        Boolean array of same length as arr
    """
    cdef:
        Py_ssize_t n_arr = len(arr), n_values = len(values)
        SwissTable_int64 *table
        int ret = 0
        np_uint8_t[::1] result_view

    # Build the lookup table from values
    table = swiss_int64_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    # Pre-allocate result array
    result = np.empty(n_arr, dtype=np.uint8)
    result_view = result

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_int64_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        # Check membership using batch contains
        with nogil:
            swiss_int64_contains_batch(table, &arr[0], <size_t>n_arr, &result_view[0])

        return result.view(np.bool_)

    finally:
        swiss_int64_destroy(table)


def value_count_int64(const int64_t[:] values, bint dropna=True):
    """
    Count occurrences of each unique value.

    Parameters
    ----------
    values : ndarray[int64]
        Array to count values in
    dropna : bool, default True
        Whether to exclude NA values from count (not applicable for int64)

    Returns
    -------
    keys : ndarray[int64]
        Unique values in order of first appearance
    counts : ndarray[int64]
        Count for each unique value
    """
    cdef:
        Py_ssize_t i, n = len(values)
        c_ssize_t n_unique
        SwissTable_int64 *table
        int64_t[::1] keys_arr = np.empty(n, dtype=np.int64)
        size_t[::1] indices_arr = np.empty(n, dtype=np.uintp)
        int64_t[::1] counts_arr

    table = swiss_int64_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Use batch value_count which stores indices for direct access
        with nogil:
            n_unique = swiss_int64_value_count_batch(
                table, &values[0], <size_t>n,
                &keys_arr[0], &indices_arr[0]
            )

        if n_unique == -1:
            raise MemoryError("Failed to insert into Swiss table")

        # Extract counts using stored indices (direct access, no lookup)
        counts_arr = np.empty(n_unique, dtype=np.int64)
        with nogil:
            for i in range(n_unique):
                counts_arr[i] = <int64_t>table.vals[indices_arr[i]]

        keys = np.asarray(keys_arr)[:n_unique].copy()
        counts = np.asarray(counts_arr)
        return keys, counts

    finally:
        swiss_int64_destroy(table)


def duplicated_int64(const int64_t[:] values, object keep="first",
                     const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values.

    Parameters
    ----------
    values : ndarray[int64]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.
        NA values are treated as a single group for duplicate detection.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_int64 *table
        int64_t val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint _keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if keep not in ("first", "last", False):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_int64_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            # Use insert_key_only: no value storage needed
            # ret == 1: newly inserted (not duplicate)
            # ret == 0: already exists (duplicate)
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA
                    else:
                        val = values[i]
                        ret = swiss_int64_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA (from the end)
                    else:
                        val = values[i]
                        ret = swiss_int64_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            # Need insert_if_absent: must store index to mark first occurrence
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values - mark all as duplicates if multiple
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_int64_insert_if_absent(table, val, <size_t>i, &idx)
                        if ret == 0:
                            # Already exists - mark both first occurrence and current
                            result[idx] = 1
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_int64_destroy(table)


# =============================================================================
# UInt64 Map
# =============================================================================
cdef class SwissUInt64Map:
    """Swiss Table hash map: uint64 -> size_t"""
    cdef SwissTable_uint64 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_uint64_init_with_capacity(size_hint)
        else:
            self.table = swiss_uint64_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_uint64_destroy(self.table)

    def insert(self, uint64_t key, size_t value):
        cdef int ret = swiss_uint64_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, uint64_t key, default=None):
        cdef size_t val
        if swiss_uint64_get(self.table, key, &val):
            return val
        return default

    def delete(self, uint64_t key):
        return swiss_uint64_delete(self.table, key)

    def clear(self):
        swiss_uint64_clear(self.table)

    def __getitem__(self, uint64_t key):
        cdef size_t val
        if swiss_uint64_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, uint64_t key, size_t value):
        cdef int ret = swiss_uint64_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, uint64_t key):
        if not swiss_uint64_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, uint64_t key):
        return swiss_uint64_contains(self.table, key)

    def __len__(self):
        return swiss_uint64_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_uint64_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_uint64_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_uint64_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_uint64_key_at(self.table, index),
                       swiss_uint64_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_uint64_size(self.table)

    @property
    def capacity(self):
        return swiss_uint64_capacity(self.table)

    def __repr__(self):
        return f"SwissUInt64Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const uint64_t[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            uint64_t val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_uint64_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const uint64_t[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            uint64_t val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_uint64_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const uint64_t[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            uint64_t val
            size_t idx
            int ret = 0
            uint64_t[::1] uniques_arr = np.empty(n, dtype=np.uint64)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_uint64_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_uint64_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const uint64_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            uint64_t val
            size_t idx
            int ret = 0
            uint64_t[::1] uniques_arr = np.empty(n, dtype=np.uint64)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_uint64_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                else:
                    val = values[i]
                    if not swiss_uint64_get(self.table, val, &idx):
                        ret = swiss_uint64_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_uint64(const uint64_t[:] arr, const uint64_t[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_uint64 *table
        uint64_t val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_uint64_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_uint64_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_uint64_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_uint64_destroy(table)


def value_count_uint64(const uint64_t[:] values, bint dropna=True):
    """Count occurrences of each unique value."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_uint64 *table
        SwissTable_uint64 *order_table
        uint64_t val
        size_t count_val
        int ret = 0
        uint64_t[::1] keys_arr = np.empty(n, dtype=np.uint64)
        int64_t[::1] counts_arr

    table = swiss_uint64_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_uint64_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_uint64_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_uint64_get(table, val, &count_val):
                    swiss_uint64_insert(table, val, count_val + 1)
                else:
                    ret = swiss_uint64_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_uint64_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        counts_arr = np.empty(n_unique, dtype=np.int64)
        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_uint64_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        keys = np.asarray(keys_arr)[:n_unique].copy()
        counts = np.asarray(counts_arr)
        return keys, counts

    finally:
        swiss_uint64_destroy(table)
        swiss_uint64_destroy(order_table)


def duplicated_uint64(const uint64_t[:] values, object keep="first",
                      const uint8_t[:] mask=None):
    """Return boolean array indicating duplicated values."""
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_uint64 *table
        uint64_t val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if keep not in ("first", "last", False):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_uint64_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        val = values[i]
                        ret = swiss_uint64_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        val = values[i]
                        ret = swiss_uint64_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_uint64_insert_if_absent(table, val, <size_t>i, &idx)
                        if ret == 0:
                            result[idx] = 1
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_uint64_destroy(table)


# =============================================================================
# Int32 Map
# =============================================================================
cdef class SwissInt32Map:
    """Swiss Table hash map: int32 -> size_t"""
    cdef SwissTable_int32 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_int32_init_with_capacity(size_hint)
        else:
            self.table = swiss_int32_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_int32_destroy(self.table)

    def insert(self, int32_t key, size_t value):
        cdef int ret = swiss_int32_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, int32_t key, default=None):
        cdef size_t val
        if swiss_int32_get(self.table, key, &val):
            return val
        return default

    def delete(self, int32_t key):
        return swiss_int32_delete(self.table, key)

    def clear(self):
        swiss_int32_clear(self.table)

    def __getitem__(self, int32_t key):
        cdef size_t val
        if swiss_int32_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, int32_t key, size_t value):
        cdef int ret = swiss_int32_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, int32_t key):
        if not swiss_int32_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, int32_t key):
        return swiss_int32_contains(self.table, key)

    def __len__(self):
        return swiss_int32_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_int32_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_int32_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_int32_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_int32_key_at(self.table, index),
                       swiss_int32_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_int32_size(self.table)

    @property
    def capacity(self):
        return swiss_int32_capacity(self.table)

    def __repr__(self):
        return f"SwissInt32Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const int32_t[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            int32_t val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_int32_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const int32_t[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            int32_t val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_int32_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const int32_t[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            int32_t val
            size_t idx
            int ret = 0
            int32_t[::1] uniques_arr = np.empty(n, dtype=np.int32)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_int32_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_int32_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const int32_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            int32_t val
            size_t idx
            int ret = 0
            int32_t[::1] uniques_arr = np.empty(n, dtype=np.int32)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_int32_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                else:
                    val = values[i]
                    if not swiss_int32_get(self.table, val, &idx):
                        ret = swiss_int32_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_int32(const int32_t[:] arr, const int32_t[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_int32 *table
        int32_t val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_int32_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_int32_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_int32_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_int32_destroy(table)


def value_count_int32(const int32_t[:] values, bint dropna=True):
    """Count occurrences of each unique value."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_int32 *table
        SwissTable_int32 *order_table
        int32_t val
        size_t count_val
        int ret = 0
        int32_t[::1] keys_arr = np.empty(n, dtype=np.int32)
        int64_t[::1] counts_arr

    table = swiss_int32_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_int32_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_int32_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_int32_get(table, val, &count_val):
                    swiss_int32_insert(table, val, count_val + 1)
                else:
                    ret = swiss_int32_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_int32_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        counts_arr = np.empty(n_unique, dtype=np.int64)
        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_int32_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        keys = np.asarray(keys_arr)[:n_unique].copy()
        counts = np.asarray(counts_arr)
        return keys, counts

    finally:
        swiss_int32_destroy(table)
        swiss_int32_destroy(order_table)


def duplicated_int32(const int32_t[:] values, object keep="first",
                     const uint8_t[:] mask=None):
    """Return boolean array indicating duplicated values."""
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_int32 *table
        int32_t val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if keep not in ("first", "last", False):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_int32_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        val = values[i]
                        ret = swiss_int32_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        val = values[i]
                        ret = swiss_int32_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_int32_insert_if_absent(table, val, <size_t>i, &idx)
                        if ret == 0:
                            result[idx] = 1
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_int32_destroy(table)


# =============================================================================
# UInt32 Map
# =============================================================================
cdef class SwissUInt32Map:
    """Swiss Table hash map: uint32 -> size_t"""
    cdef SwissTable_uint32 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_uint32_init_with_capacity(size_hint)
        else:
            self.table = swiss_uint32_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_uint32_destroy(self.table)

    def insert(self, uint32_t key, size_t value):
        cdef int ret = swiss_uint32_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, uint32_t key, default=None):
        cdef size_t val
        if swiss_uint32_get(self.table, key, &val):
            return val
        return default

    def delete(self, uint32_t key):
        return swiss_uint32_delete(self.table, key)

    def clear(self):
        swiss_uint32_clear(self.table)

    def __getitem__(self, uint32_t key):
        cdef size_t val
        if swiss_uint32_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, uint32_t key, size_t value):
        cdef int ret = swiss_uint32_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, uint32_t key):
        if not swiss_uint32_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, uint32_t key):
        return swiss_uint32_contains(self.table, key)

    def __len__(self):
        return swiss_uint32_size(self.table)

    @property
    def size(self):
        return swiss_uint32_size(self.table)

    @property
    def capacity(self):
        return swiss_uint32_capacity(self.table)

    def __repr__(self):
        return f"SwissUInt32Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const uint32_t[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            uint32_t val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_uint32_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const uint32_t[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            uint32_t val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_uint32_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const uint32_t[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            uint32_t val
            size_t idx
            int ret = 0
            uint32_t[::1] uniques_arr = np.empty(n, dtype=np.uint32)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_uint32_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_uint32_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const uint32_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            uint32_t val
            size_t idx
            int ret = 0
            uint32_t[::1] uniques_arr = np.empty(n, dtype=np.uint32)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_uint32_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                else:
                    val = values[i]
                    if not swiss_uint32_get(self.table, val, &idx):
                        ret = swiss_uint32_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_uint32(const uint32_t[:] arr, const uint32_t[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_uint32 *table
        uint32_t val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_uint32_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_uint32_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_uint32_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_uint32_destroy(table)


def value_count_uint32(const uint32_t[:] values, bint dropna=True):
    """Count occurrences of each unique value."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_uint32 *table
        SwissTable_uint32 *order_table
        uint32_t val
        size_t count_val
        int ret = 0
        uint32_t[::1] keys_arr = np.empty(n, dtype=np.uint32)
        int64_t[::1] counts_arr

    table = swiss_uint32_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_uint32_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_uint32_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_uint32_get(table, val, &count_val):
                    swiss_uint32_insert(table, val, count_val + 1)
                else:
                    ret = swiss_uint32_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_uint32_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        counts_arr = np.empty(n_unique, dtype=np.int64)
        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_uint32_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        keys = np.asarray(keys_arr)[:n_unique].copy()
        counts = np.asarray(counts_arr)
        return keys, counts

    finally:
        swiss_uint32_destroy(table)
        swiss_uint32_destroy(order_table)


def duplicated_uint32(const uint32_t[:] values, object keep="first",
                      const uint8_t[:] mask=None):
    """Return boolean array indicating duplicated values."""
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_uint32 *table
        uint32_t val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if keep not in ("first", "last", False):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_uint32_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        val = values[i]
                        ret = swiss_uint32_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        val = values[i]
                        ret = swiss_uint32_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_uint32_insert_if_absent(table, val, <size_t>i, &idx)
                        if ret == 0:
                            result[idx] = 1
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_uint32_destroy(table)


# =============================================================================
# Float64 Map (with NaN handling)
# =============================================================================
cdef class SwissFloat64Map:
    """
    Swiss Table hash map: float64 -> size_t

    NaN handling: All NaN values are considered equal.
    Zero handling: +0.0 and -0.0 are considered equal.
    """
    cdef SwissTable_float64 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_float64_init_with_capacity(size_hint)
        else:
            self.table = swiss_float64_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_float64_destroy(self.table)

    def insert(self, double key, size_t value):
        cdef int ret = swiss_float64_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, double key, default=None):
        cdef size_t val
        if swiss_float64_get(self.table, key, &val):
            return val
        return default

    def delete(self, double key):
        return swiss_float64_delete(self.table, key)

    def clear(self):
        swiss_float64_clear(self.table)

    def __getitem__(self, double key):
        cdef size_t val
        if swiss_float64_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, double key, size_t value):
        cdef int ret = swiss_float64_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, double key):
        if not swiss_float64_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, double key):
        return swiss_float64_contains(self.table, key)

    def __len__(self):
        return swiss_float64_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_float64_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_float64_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_float64_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_float64_key_at(self.table, index),
                       swiss_float64_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_float64_size(self.table)

    @property
    def capacity(self):
        return swiss_float64_capacity(self.table)

    def __repr__(self):
        return f"SwissFloat64Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const double[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            double val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_float64_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const double[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            double val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_float64_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const double[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            double val
            size_t idx
            int ret = 0
            double[::1] uniques_arr = np.empty(n, dtype=np.float64)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_float64_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_float64_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const double[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding (NaN-aware)."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            double val
            size_t idx
            int ret = 0
            double[::1] uniques_arr = np.empty(n, dtype=np.float64)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_float64_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                val = values[i]
                # Check for NA: mask takes precedence, then NaN
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = val
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                elif ignore_na and isnan(val):
                    labels[i] = na_sentinel
                else:
                    if not swiss_float64_get(self.table, val, &idx):
                        ret = swiss_float64_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_float64(const double[:] arr, const double[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_float64 *table
        double val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_float64_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_float64_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_float64_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_float64_destroy(table)


def value_count_float64(const double[:] values, bint dropna=True):
    """Count occurrences of each unique value (NaN-aware)."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_float64 *table
        SwissTable_float64 *order_table
        double val
        size_t count_val
        int ret = 0
        double[::1] keys_arr = np.empty(n, dtype=np.float64)
        int64_t[::1] counts_arr
        Py_ssize_t na_count = 0

    table = swiss_float64_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_float64_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_float64_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                # Count NaN separately if not dropping
                if isnan(val):
                    if not dropna:
                        na_count += 1
                    continue
                if swiss_float64_get(table, val, &count_val):
                    swiss_float64_insert(table, val, count_val + 1)
                else:
                    ret = swiss_float64_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_float64_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        # Add NaN slot if needed
        extra = 1 if na_count > 0 else 0
        counts_arr = np.empty(n_unique + extra, dtype=np.int64)
        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_float64_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        keys = np.asarray(keys_arr)[:n_unique].copy()
        counts = np.asarray(counts_arr)[:n_unique].copy()

        # Add NaN to output if not dropping
        if na_count > 0:
            keys = np.append(keys, np.nan)
            counts = np.append(counts, na_count)

        return keys, counts

    finally:
        swiss_float64_destroy(table)
        swiss_float64_destroy(order_table)


def duplicated_float64(const double[:] values, object keep="first",
                       const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values (NaN-aware).

    Parameters
    ----------
    values : ndarray[float64]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.
        NA values are treated as a single group for duplicate detection.
        Mask takes precedence over NaN check in values.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_float64 *table
        double val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint _keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if keep not in ("first", "last", False):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_float64_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n):
                    # When mask is provided, use mask only. Otherwise check NaN.
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA
                    else:
                        val = values[i]
                        ret = swiss_float64_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate (NaN handled by hash table)
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n - 1, -1, -1):
                    # When mask is provided, use mask only. Otherwise check NaN.
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA (from end)
                    else:
                        val = values[i]
                        ret = swiss_float64_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate (NaN handled by hash table)
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            # Need insert_if_absent: must store index to mark first occurrence
            with nogil:
                for i in range(n):
                    # When mask is provided, use mask only. Otherwise check NaN.
                    if uses_mask and mask[i]:
                        # Handle NA values - mark all as duplicates if multiple
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_float64_insert_if_absent(table, val, <size_t>i, &idx)  # noqa: E501
                        if ret == 0:
                            result[idx] = 1
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_float64_destroy(table)


# =============================================================================
# Float32 Map (with NaN handling)
# =============================================================================
cdef class SwissFloat32Map:
    """
    Swiss Table hash map: float32 -> size_t

    NaN handling: All NaN values are considered equal.
    Zero handling: +0.0f and -0.0f are considered equal.
    """
    cdef SwissTable_float32 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_float32_init_with_capacity(size_hint)
        else:
            self.table = swiss_float32_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_float32_destroy(self.table)

    def insert(self, float key, size_t value):
        cdef int ret = swiss_float32_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, float key, default=None):
        cdef size_t val
        if swiss_float32_get(self.table, key, &val):
            return val
        return default

    def delete(self, float key):
        return swiss_float32_delete(self.table, key)

    def clear(self):
        swiss_float32_clear(self.table)

    def __getitem__(self, float key):
        cdef size_t val
        if swiss_float32_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, float key, size_t value):
        cdef int ret = swiss_float32_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, float key):
        if not swiss_float32_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, float key):
        return swiss_float32_contains(self.table, key)

    def __len__(self):
        return swiss_float32_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_float32_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_float32_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_float32_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_float32_key_at(self.table, index),
                       swiss_float32_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_float32_size(self.table)

    @property
    def capacity(self):
        return swiss_float32_capacity(self.table)

    def __repr__(self):
        return f"SwissFloat32Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const float[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            float val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_float32_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const float[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            float val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_float32_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const float[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            float val
            size_t idx
            int ret = 0
            float[::1] uniques_arr = np.empty(n, dtype=np.float32)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_float32_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_float32_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const float[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding (NaN-aware)."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            float val
            size_t idx
            int ret = 0
            float[::1] uniques_arr = np.empty(n, dtype=np.float32)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_float32_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                val = values[i]
                # Check for NA: mask takes precedence, then NaN
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = val
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                elif ignore_na and isnan(val):
                    labels[i] = na_sentinel
                else:
                    if not swiss_float32_get(self.table, val, &idx):
                        ret = swiss_float32_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_float32(const float[:] arr, const float[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_float32 *table
        float val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_float32_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_float32_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_float32_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_float32_destroy(table)


def value_count_float32(const float[:] values, bint dropna=True):
    """Count occurrences of each unique value (NaN-aware)."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_float32 *table
        SwissTable_float32 *order_table
        float val
        size_t count_val
        int ret = 0
        float[::1] keys_arr = np.empty(n, dtype=np.float32)
        int64_t[::1] counts_arr
        Py_ssize_t na_count = 0
        bint skip_na = dropna

    table = swiss_float32_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_float32_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_float32_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                if isnan(val):
                    if not skip_na:
                        na_count += 1
                    continue
                if swiss_float32_get(table, val, &count_val):
                    swiss_float32_insert(table, val, count_val + 1)
                else:
                    ret = swiss_float32_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_float32_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        counts_arr = np.empty(n_unique + (1 if na_count > 0 else 0), dtype=np.int64)

        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_float32_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        if na_count > 0:
            keys = np.empty(n_unique + 1, dtype=np.float32)
            keys[:n_unique] = np.asarray(keys_arr)[:n_unique]
            keys[n_unique] = float("nan")
            counts_arr[n_unique] = na_count
            return keys, np.asarray(counts_arr)

        return np.asarray(keys_arr)[:n_unique].copy(), np.asarray(counts_arr)

    finally:
        swiss_float32_destroy(table)
        swiss_float32_destroy(order_table)


def duplicated_float32(const float[:] values, object keep="first",
                       const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values (NaN-aware).

    Parameters
    ----------
    values : ndarray[float32]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.
        NA values are treated as a single group for duplicate detection.
        Mask takes precedence over NaN check in values.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_float32 *table
        float val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if not (keep_first or keep_last or keep_none):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_float32_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n):
                    # When mask is provided, use mask only. Otherwise check NaN.
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA
                    else:
                        val = values[i]
                        ret = swiss_float32_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate (NaN handled by hash table)
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n - 1, -1, -1):
                    # When mask is provided, use mask only. Otherwise check NaN.
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA (from end)
                    else:
                        val = values[i]
                        ret = swiss_float32_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate (NaN handled by hash table)
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            # Need insert_if_absent: must store index to mark first occurrence
            with nogil:
                for i in range(n):
                    # When mask is provided, use mask only. Otherwise check NaN.
                    if uses_mask and mask[i]:
                        # Handle NA values - mark all as duplicates if multiple
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_float32_insert_if_absent(table, val, <size_t>i, &idx)  # noqa: E501
                        if ret == 0:
                            result[idx] = 1
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_float32_destroy(table)


# =============================================================================
# Int16 Map
# =============================================================================
cdef class SwissInt16Map:
    """Swiss Table hash map: int16 -> size_t"""
    cdef SwissTable_int16 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_int16_init_with_capacity(size_hint)
        else:
            self.table = swiss_int16_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_int16_destroy(self.table)

    def insert(self, int16_t key, size_t value):
        cdef int ret = swiss_int16_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, int16_t key, default=None):
        cdef size_t val
        if swiss_int16_get(self.table, key, &val):
            return val
        return default

    def delete(self, int16_t key):
        return swiss_int16_delete(self.table, key)

    def clear(self):
        swiss_int16_clear(self.table)

    def __getitem__(self, int16_t key):
        cdef size_t val
        if swiss_int16_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, int16_t key, size_t value):
        cdef int ret = swiss_int16_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, int16_t key):
        if not swiss_int16_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, int16_t key):
        return swiss_int16_contains(self.table, key)

    def __len__(self):
        return swiss_int16_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_int16_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_int16_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_int16_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_int16_key_at(self.table, index),
                       swiss_int16_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_int16_size(self.table)

    @property
    def capacity(self):
        return swiss_int16_capacity(self.table)

    def __repr__(self):
        return f"SwissInt16Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const int16_t[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            int16_t val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_int16_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const int16_t[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            int16_t val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_int16_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const int16_t[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            int16_t val
            size_t idx
            int ret = 0
            int16_t[::1] uniques_arr = np.empty(n, dtype=np.int16)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_int16_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_int16_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const int16_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            int16_t val
            size_t idx
            int ret = 0
            int16_t[::1] uniques_arr = np.empty(n, dtype=np.int16)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_int16_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                else:
                    val = values[i]
                    if not swiss_int16_get(self.table, val, &idx):
                        ret = swiss_int16_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_int16(const int16_t[:] arr, const int16_t[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_int16 *table
        int16_t val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_int16_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_int16_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_int16_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_int16_destroy(table)


def value_count_int16(const int16_t[:] values, bint dropna=True):
    """Count occurrences of each unique value."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_int16 *table
        SwissTable_int16 *order_table
        int16_t val
        size_t count_val
        int ret = 0
        int16_t[::1] keys_arr = np.empty(n, dtype=np.int16)
        int64_t[::1] counts_arr

    table = swiss_int16_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_int16_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_int16_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_int16_get(table, val, &count_val):
                    swiss_int16_insert(table, val, count_val + 1)
                else:
                    ret = swiss_int16_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_int16_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        counts_arr = np.empty(n_unique, dtype=np.int64)

        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_int16_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        return np.asarray(keys_arr)[:n_unique].copy(), np.asarray(counts_arr)

    finally:
        swiss_int16_destroy(table)
        swiss_int16_destroy(order_table)


def duplicated_int16(const int16_t[:] values, object keep="first",
                     const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values.

    Parameters
    ----------
    values : ndarray[int16]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.
        NA values are treated as a single group for duplicate detection.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_int16 *table
        int16_t val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if not (keep_first or keep_last or keep_none):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_int16_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA
                    else:
                        val = values[i]
                        ret = swiss_int16_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA (from end)
                    else:
                        val = values[i]
                        ret = swiss_int16_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            # Need insert_if_absent: must store index to mark first occurrence
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values - mark all as duplicates if multiple
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_int16_insert_if_absent(table, val, <size_t>i, &idx)
                        if ret == 0:
                            result[idx] = 1  # Mark first occurrence
                            result[i] = 1    # Mark this occurrence
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_int16_destroy(table)


# =============================================================================
# UInt16 Map
# =============================================================================
cdef class SwissUInt16Map:
    """Swiss Table hash map: uint16 -> size_t"""
    cdef SwissTable_uint16 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_uint16_init_with_capacity(size_hint)
        else:
            self.table = swiss_uint16_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_uint16_destroy(self.table)

    def insert(self, uint16_t key, size_t value):
        cdef int ret = swiss_uint16_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, uint16_t key, default=None):
        cdef size_t val
        if swiss_uint16_get(self.table, key, &val):
            return val
        return default

    def delete(self, uint16_t key):
        return swiss_uint16_delete(self.table, key)

    def clear(self):
        swiss_uint16_clear(self.table)

    def __getitem__(self, uint16_t key):
        cdef size_t val
        if swiss_uint16_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, uint16_t key, size_t value):
        cdef int ret = swiss_uint16_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, uint16_t key):
        if not swiss_uint16_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, uint16_t key):
        return swiss_uint16_contains(self.table, key)

    def __len__(self):
        return swiss_uint16_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_uint16_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_uint16_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_uint16_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_uint16_key_at(self.table, index),
                       swiss_uint16_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_uint16_size(self.table)

    @property
    def capacity(self):
        return swiss_uint16_capacity(self.table)

    def __repr__(self):
        return f"SwissUInt16Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const uint16_t[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            uint16_t val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_uint16_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const uint16_t[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            uint16_t val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_uint16_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const uint16_t[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            uint16_t val
            size_t idx
            int ret = 0
            uint16_t[::1] uniques_arr = np.empty(n, dtype=np.uint16)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_uint16_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_uint16_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const uint16_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            uint16_t val
            size_t idx
            int ret = 0
            uint16_t[::1] uniques_arr = np.empty(n, dtype=np.uint16)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_uint16_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                else:
                    val = values[i]
                    if not swiss_uint16_get(self.table, val, &idx):
                        ret = swiss_uint16_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_uint16(const uint16_t[:] arr, const uint16_t[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_uint16 *table
        uint16_t val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_uint16_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_uint16_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_uint16_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_uint16_destroy(table)


def value_count_uint16(const uint16_t[:] values, bint dropna=True):
    """Count occurrences of each unique value."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_uint16 *table
        SwissTable_uint16 *order_table
        uint16_t val
        size_t count_val
        int ret = 0
        uint16_t[::1] keys_arr = np.empty(n, dtype=np.uint16)
        int64_t[::1] counts_arr

    table = swiss_uint16_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_uint16_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_uint16_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_uint16_get(table, val, &count_val):
                    swiss_uint16_insert(table, val, count_val + 1)
                else:
                    ret = swiss_uint16_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_uint16_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        counts_arr = np.empty(n_unique, dtype=np.int64)

        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_uint16_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        return np.asarray(keys_arr)[:n_unique].copy(), np.asarray(counts_arr)

    finally:
        swiss_uint16_destroy(table)
        swiss_uint16_destroy(order_table)


def duplicated_uint16(const uint16_t[:] values, object keep="first",
                      const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values.

    Parameters
    ----------
    values : ndarray[uint16]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.
        NA values are treated as a single group for duplicate detection.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_uint16 *table
        uint16_t val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if not (keep_first or keep_last or keep_none):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_uint16_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA
                    else:
                        val = values[i]
                        ret = swiss_uint16_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA (from end)
                    else:
                        val = values[i]
                        ret = swiss_uint16_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            # Need insert_if_absent: must store index to mark first occurrence
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values - mark all as duplicates if multiple
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_uint16_insert_if_absent(table, val, <size_t>i, &idx)
                        if ret == 0:
                            result[idx] = 1  # Mark first occurrence
                            result[i] = 1    # Mark this occurrence
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_uint16_destroy(table)


# =============================================================================
# Int8 Map
# =============================================================================
cdef class SwissInt8Map:
    """Swiss Table hash map: int8 -> size_t"""
    cdef SwissTable_int8 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_int8_init_with_capacity(size_hint)
        else:
            self.table = swiss_int8_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_int8_destroy(self.table)

    def insert(self, int8_t key, size_t value):
        cdef int ret = swiss_int8_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, int8_t key, default=None):
        cdef size_t val
        if swiss_int8_get(self.table, key, &val):
            return val
        return default

    def delete(self, int8_t key):
        return swiss_int8_delete(self.table, key)

    def clear(self):
        swiss_int8_clear(self.table)

    def __getitem__(self, int8_t key):
        cdef size_t val
        if swiss_int8_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, int8_t key, size_t value):
        cdef int ret = swiss_int8_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, int8_t key):
        if not swiss_int8_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, int8_t key):
        return swiss_int8_contains(self.table, key)

    def __len__(self):
        return swiss_int8_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_int8_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_int8_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_int8_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_int8_key_at(self.table, index),
                       swiss_int8_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_int8_size(self.table)

    @property
    def capacity(self):
        return swiss_int8_capacity(self.table)

    def __repr__(self):
        return f"SwissInt8Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const int8_t[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            int8_t val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_int8_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const int8_t[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            int8_t val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_int8_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const int8_t[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            int8_t val
            size_t idx
            int ret = 0
            int8_t[::1] uniques_arr = np.empty(n, dtype=np.int8)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_int8_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_int8_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const int8_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            int8_t val
            size_t idx
            int ret = 0
            int8_t[::1] uniques_arr = np.empty(n, dtype=np.int8)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_int8_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                else:
                    val = values[i]
                    if not swiss_int8_get(self.table, val, &idx):
                        ret = swiss_int8_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_int8(const int8_t[:] arr, const int8_t[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_int8 *table
        int8_t val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_int8_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_int8_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_int8_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_int8_destroy(table)


def value_count_int8(const int8_t[:] values, bint dropna=True):
    """Count occurrences of each unique value."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_int8 *table
        SwissTable_int8 *order_table
        int8_t val
        size_t count_val
        int ret = 0
        int8_t[::1] keys_arr = np.empty(n, dtype=np.int8)
        int64_t[::1] counts_arr

    table = swiss_int8_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_int8_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_int8_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_int8_get(table, val, &count_val):
                    swiss_int8_insert(table, val, count_val + 1)
                else:
                    ret = swiss_int8_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_int8_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        counts_arr = np.empty(n_unique, dtype=np.int64)

        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_int8_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        return np.asarray(keys_arr)[:n_unique].copy(), np.asarray(counts_arr)

    finally:
        swiss_int8_destroy(table)
        swiss_int8_destroy(order_table)


def duplicated_int8(const int8_t[:] values, object keep="first",
                    const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values.

    Parameters
    ----------
    values : ndarray[int8]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.
        NA values are treated as a single group for duplicate detection.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_int8 *table
        int8_t val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if not (keep_first or keep_last or keep_none):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_int8_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA
                    else:
                        val = values[i]
                        ret = swiss_int8_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA (from end)
                    else:
                        val = values[i]
                        ret = swiss_int8_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            # Need insert_if_absent: must store index to mark first occurrence
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values - mark all as duplicates if multiple
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_int8_insert_if_absent(table, val, <size_t>i, &idx)
                        if ret == 0:
                            result[idx] = 1  # Mark first occurrence
                            result[i] = 1    # Mark this occurrence
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_int8_destroy(table)


# =============================================================================
# UInt8 Map
# =============================================================================
cdef class SwissUInt8Map:
    """Swiss Table hash map: uint8 -> size_t"""
    cdef SwissTable_uint8 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_uint8_init_with_capacity(size_hint)
        else:
            self.table = swiss_uint8_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_uint8_destroy(self.table)

    def insert(self, uint8_t key, size_t value):
        cdef int ret = swiss_uint8_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, uint8_t key, default=None):
        cdef size_t val
        if swiss_uint8_get(self.table, key, &val):
            return val
        return default

    def delete(self, uint8_t key):
        return swiss_uint8_delete(self.table, key)

    def clear(self):
        swiss_uint8_clear(self.table)

    def __getitem__(self, uint8_t key):
        cdef size_t val
        if swiss_uint8_get(self.table, key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, uint8_t key, size_t value):
        cdef int ret = swiss_uint8_insert(self.table, key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, uint8_t key):
        if not swiss_uint8_delete(self.table, key):
            raise KeyError(key)

    def __contains__(self, uint8_t key):
        return swiss_uint8_contains(self.table, key)

    def __len__(self):
        return swiss_uint8_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_uint8_iter_next(self.table, index)
            if index < self.table.capacity:
                yield swiss_uint8_key_at(self.table, index)
                index += 1

    def items(self):
        cdef size_t index = 0
        while index < self.table.capacity:
            index = swiss_uint8_iter_next(self.table, index)
            if index < self.table.capacity:
                yield (swiss_uint8_key_at(self.table, index),
                       swiss_uint8_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_uint8_size(self.table)

    @property
    def capacity(self):
        return swiss_uint8_capacity(self.table)

    def __repr__(self):
        return f"SwissUInt8Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const uint8_t[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            uint8_t val
            int ret = 0

        with nogil:
            for i in range(n):
                val = values[i]
                ret = swiss_uint8_insert(self.table, val, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const uint8_t[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            uint8_t val
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_uint8_get(self.table, val, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const uint8_t[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            uint8_t val
            size_t idx
            int ret = 0
            uint8_t[::1] uniques_arr = np.empty(n, dtype=np.uint8)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        val = values[i]
                        ret = swiss_uint8_insert_if_absent(self.table, val, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        val = values[i]
                        ret = swiss_uint8_insert_key_only(self.table, val)
                        if ret == 1:
                            uniques_arr[count] = val
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const uint8_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            uint8_t val
            size_t idx
            int ret = 0
            uint8_t[::1] uniques_arr = np.empty(n, dtype=np.uint8)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_uint8_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                else:
                    val = values[i]
                    if not swiss_uint8_get(self.table, val, &idx):
                        ret = swiss_uint8_insert(self.table, val, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = val
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_uint8(const uint8_t[:] arr, const uint8_t[:] values) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_uint8 *table
        uint8_t val
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_uint8_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values (key-only, no value storage needed)
        with nogil:
            ret = swiss_uint8_build_set(table, &values[0], <size_t>n_values)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                val = arr[i]
                result[i] = swiss_uint8_contains(table, val)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_uint8_destroy(table)


def value_count_uint8(const uint8_t[:] values, bint dropna=True):
    """Count occurrences of each unique value."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_uint8 *table
        SwissTable_uint8 *order_table
        uint8_t val
        size_t count_val
        int ret = 0
        uint8_t[::1] keys_arr = np.empty(n, dtype=np.uint8)
        int64_t[::1] counts_arr

    table = swiss_uint8_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_uint8_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_uint8_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                val = values[i]
                if swiss_uint8_get(table, val, &count_val):
                    swiss_uint8_insert(table, val, count_val + 1)
                else:
                    ret = swiss_uint8_insert(table, val, 1)
                    if ret == -1:
                        break
                    ret = swiss_uint8_insert(order_table, val, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = val
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        counts_arr = np.empty(n_unique, dtype=np.int64)

        with nogil:
            for i in range(n_unique):
                val = keys_arr[i]
                swiss_uint8_get(table, val, &count_val)
                counts_arr[i] = <int64_t>count_val

        return np.asarray(keys_arr)[:n_unique].copy(), np.asarray(counts_arr)

    finally:
        swiss_uint8_destroy(table)
        swiss_uint8_destroy(order_table)


def duplicated_uint8(const uint8_t[:] values, object keep="first",
                     const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values.

    Parameters
    ----------
    values : ndarray[uint8]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.
        NA values are treated as a single group for duplicate detection.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_uint8 *table
        uint8_t val
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if not (keep_first or keep_last or keep_none):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_uint8_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA
                    else:
                        val = values[i]
                        ret = swiss_uint8_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            # Use insert_key_only: no value storage needed
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        # Handle NA values
                        if seen_na:
                            result[i] = 1  # Duplicate NA
                        else:
                            seen_na = True
                            result[i] = 0  # First NA (from end)
                    else:
                        val = values[i]
                        ret = swiss_uint8_insert_key_only(table, val)
                        if ret == 0:
                            result[i] = 1  # Duplicate
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            # Need insert_if_absent: must store index to mark first occurrence
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        # Handle NA values - mark all as duplicates if multiple
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        val = values[i]
                        ret = swiss_uint8_insert_if_absent(table, val, <size_t>i, &idx)
                        if ret == 0:
                            result[idx] = 1  # Mark first occurrence
                            result[i] = 1    # Mark this occurrence
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_uint8_destroy(table)


# =============================================================================
# Complex64 Map
# =============================================================================
cdef class SwissComplex64Map:
    """
    Swiss Table hash map: complex64 -> size_t

    NaN handling: Complex numbers with NaN components are considered equal.
    """
    cdef SwissTable_complex64 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_complex64_init_with_capacity(size_hint)
        else:
            self.table = swiss_complex64_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_complex64_destroy(self.table)

    cdef swiss_complex64_t _to_c_complex(self, object key):
        cdef swiss_complex64_t c_key
        c_key.real = key.real
        c_key.imag = key.imag
        return c_key

    def insert(self, key, size_t value):
        cdef swiss_complex64_t c_key = self._to_c_complex(key)
        cdef int ret = swiss_complex64_insert(self.table, c_key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, key, default=None):
        cdef swiss_complex64_t c_key = self._to_c_complex(key)
        cdef size_t val
        if swiss_complex64_get(self.table, c_key, &val):
            return val
        return default

    def delete(self, key):
        cdef swiss_complex64_t c_key = self._to_c_complex(key)
        return swiss_complex64_delete(self.table, c_key)

    def clear(self):
        swiss_complex64_clear(self.table)

    def __getitem__(self, key):
        cdef swiss_complex64_t c_key = self._to_c_complex(key)
        cdef size_t val
        if swiss_complex64_get(self.table, c_key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, key, size_t value):
        cdef swiss_complex64_t c_key = self._to_c_complex(key)
        cdef int ret = swiss_complex64_insert(self.table, c_key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, key):
        cdef swiss_complex64_t c_key = self._to_c_complex(key)
        if not swiss_complex64_delete(self.table, c_key):
            raise KeyError(key)

    def __contains__(self, key):
        cdef swiss_complex64_t c_key = self._to_c_complex(key)
        return swiss_complex64_contains(self.table, c_key)

    def __len__(self):
        return swiss_complex64_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        cdef swiss_complex64_t c_key
        while index < self.table.capacity:
            index = swiss_complex64_iter_next(self.table, index)
            if index < self.table.capacity:
                c_key = swiss_complex64_key_at(self.table, index)
                yield complex(c_key.real, c_key.imag)
                index += 1

    def items(self):
        cdef size_t index = 0
        cdef swiss_complex64_t c_key
        while index < self.table.capacity:
            index = swiss_complex64_iter_next(self.table, index)
            if index < self.table.capacity:
                c_key = swiss_complex64_key_at(self.table, index)
                yield (complex(c_key.real, c_key.imag),
                       swiss_complex64_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_complex64_size(self.table)

    @property
    def capacity(self):
        return swiss_complex64_capacity(self.table)

    def __repr__(self):
        return f"SwissComplex64Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const float complex[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            swiss_complex64_t c_key
            int ret = 0

        with nogil:
            for i in range(n):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                ret = swiss_complex64_insert(self.table, c_key, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const float complex[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            swiss_complex64_t c_key
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                if swiss_complex64_get(self.table, c_key, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const float complex[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            swiss_complex64_t c_key
            size_t idx
            int ret = 0
            float complex[::1] uniques_arr = np.empty(n, dtype=np.complex64)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex64_insert_if_absent(self.table, c_key, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = values[i]
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex64_insert_key_only(self.table, c_key)
                        if ret == 1:
                            uniques_arr[count] = values[i]
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const float complex[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding (NaN-aware)."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            swiss_complex64_t c_key
            size_t idx
            int ret = 0
            float complex[::1] uniques_arr = np.empty(n, dtype=np.complex64)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            float real_part, imag_part

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_complex64_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                real_part = c_key.real
                imag_part = c_key.imag
                # Check for NA: mask takes precedence, then NaN in components
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                elif ignore_na and (isnan(real_part) or isnan(imag_part)):
                    labels[i] = na_sentinel
                else:
                    if not swiss_complex64_get(self.table, c_key, &idx):
                        ret = swiss_complex64_insert(self.table, c_key, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = values[i]
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_complex64(
    const float complex[:] arr, const float complex[:] values
) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_complex64 *table
        swiss_complex64_t c_key
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_complex64_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values
        with nogil:
            for i in range(n_values):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                ret = swiss_complex64_insert_key_only(table, c_key)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                c_key.real = arr[i].real
                c_key.imag = arr[i].imag
                result[i] = swiss_complex64_contains(table, c_key)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_complex64_destroy(table)


def value_count_complex64(const float complex[:] values, bint dropna=True):
    """Count occurrences of each unique value (NaN-aware)."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_complex64 *table
        SwissTable_complex64 *order_table
        swiss_complex64_t c_key
        size_t count_val
        int ret = 0
        float complex[::1] keys_arr = np.empty(n, dtype=np.complex64)
        int64_t[::1] counts_arr
        Py_ssize_t na_count = 0
        float real_part, imag_part

    table = swiss_complex64_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_complex64_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_complex64_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                real_part = c_key.real
                imag_part = c_key.imag
                # Count NaN separately if not dropping
                if isnan(real_part) or isnan(imag_part):
                    if not dropna:
                        na_count += 1
                    continue
                if swiss_complex64_get(table, c_key, &count_val):
                    swiss_complex64_insert(table, c_key, count_val + 1)
                else:
                    ret = swiss_complex64_insert(table, c_key, 1)
                    if ret == -1:
                        break
                    ret = swiss_complex64_insert(order_table, c_key, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = values[i]
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        # Add NaN slot if needed
        extra = 1 if na_count > 0 else 0
        counts_arr = np.empty(n_unique + extra, dtype=np.int64)
        with nogil:
            for i in range(n_unique):
                c_key.real = keys_arr[i].real
                c_key.imag = keys_arr[i].imag
                swiss_complex64_get(table, c_key, &count_val)
                counts_arr[i] = <int64_t>count_val

        keys = np.asarray(keys_arr)[:n_unique].copy()
        counts = np.asarray(counts_arr)[:n_unique].copy()

        # Add NaN to output if not dropping
        if na_count > 0:
            keys = np.append(keys, complex(np.nan, np.nan))
            counts = np.append(counts, na_count)

        return keys, counts

    finally:
        swiss_complex64_destroy(table)
        swiss_complex64_destroy(order_table)


def duplicated_complex64(const float complex[:] values, object keep="first",
                         const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values (NaN-aware).

    Parameters
    ----------
    values : ndarray[complex64]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_complex64 *table
        swiss_complex64_t c_key
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint _keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if keep not in ("first", "last", False):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_complex64_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex64_insert_key_only(table, c_key)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex64_insert_key_only(table, c_key)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex64_insert_if_absent(table, c_key, <size_t>i, &idx)  # noqa: E501
                        if ret == 0:
                            result[idx] = 1
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_complex64_destroy(table)


# =============================================================================
# Complex128 Map
# =============================================================================
cdef class SwissComplex128Map:
    """
    Swiss Table hash map: complex128 -> size_t

    NaN handling: Complex numbers with NaN components are considered equal.
    """
    cdef SwissTable_complex128 *table

    def __cinit__(self, size_t size_hint=0):
        if size_hint > 0:
            self.table = swiss_complex128_init_with_capacity(size_hint)
        else:
            self.table = swiss_complex128_init()
        if self.table == NULL:
            raise MemoryError("Failed to allocate Swiss table")

    def __dealloc__(self):
        if self.table != NULL:
            swiss_complex128_destroy(self.table)

    cdef swiss_complex128_t _to_c_complex(self, object key):
        cdef swiss_complex128_t c_key
        c_key.real = key.real
        c_key.imag = key.imag
        return c_key

    def insert(self, key, size_t value):
        cdef swiss_complex128_t c_key = self._to_c_complex(key)
        cdef int ret = swiss_complex128_insert(self.table, c_key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")
        return ret == 1

    def get(self, key, default=None):
        cdef swiss_complex128_t c_key = self._to_c_complex(key)
        cdef size_t val
        if swiss_complex128_get(self.table, c_key, &val):
            return val
        return default

    def delete(self, key):
        cdef swiss_complex128_t c_key = self._to_c_complex(key)
        return swiss_complex128_delete(self.table, c_key)

    def clear(self):
        swiss_complex128_clear(self.table)

    def __getitem__(self, key):
        cdef swiss_complex128_t c_key = self._to_c_complex(key)
        cdef size_t val
        if swiss_complex128_get(self.table, c_key, &val):
            return val
        raise KeyError(key)

    def __setitem__(self, key, size_t value):
        cdef swiss_complex128_t c_key = self._to_c_complex(key)
        cdef int ret = swiss_complex128_insert(self.table, c_key, value)
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def __delitem__(self, key):
        cdef swiss_complex128_t c_key = self._to_c_complex(key)
        if not swiss_complex128_delete(self.table, c_key):
            raise KeyError(key)

    def __contains__(self, key):
        cdef swiss_complex128_t c_key = self._to_c_complex(key)
        return swiss_complex128_contains(self.table, c_key)

    def __len__(self):
        return swiss_complex128_size(self.table)

    def __iter__(self):
        cdef size_t index = 0
        cdef swiss_complex128_t c_key
        while index < self.table.capacity:
            index = swiss_complex128_iter_next(self.table, index)
            if index < self.table.capacity:
                c_key = swiss_complex128_key_at(self.table, index)
                yield complex(c_key.real, c_key.imag)
                index += 1

    def items(self):
        cdef size_t index = 0
        cdef swiss_complex128_t c_key
        while index < self.table.capacity:
            index = swiss_complex128_iter_next(self.table, index)
            if index < self.table.capacity:
                c_key = swiss_complex128_key_at(self.table, index)
                yield (complex(c_key.real, c_key.imag),
                       swiss_complex128_val_at(self.table, index))
                index += 1

    @property
    def size(self):
        return swiss_complex128_size(self.table)

    @property
    def capacity(self):
        return swiss_complex128_capacity(self.table)

    def __repr__(self):
        return f"SwissComplex128Map(size={self.size}, capacity={self.capacity})"

    def map_locations(self, const double complex[:] values) -> None:
        """Build table mapping values to their array positions."""
        cdef:
            Py_ssize_t i, n = len(values)
            swiss_complex128_t c_key
            int ret = 0

        with nogil:
            for i in range(n):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                ret = swiss_complex128_insert(self.table, c_key, <size_t>i)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

    def lookup(self, const double complex[:] values) -> ndarray:
        """Look up array values in table, returning positions (-1 if not found)."""
        cdef:
            Py_ssize_t i, n = len(values)
            swiss_complex128_t c_key
            size_t loc
            intp_t[::1] locs = np.empty(n, dtype=np.intp)

        with nogil:
            for i in range(n):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                if swiss_complex128_get(self.table, c_key, &loc):
                    locs[i] = <intp_t>loc
                else:
                    locs[i] = -1

        return np.asarray(locs)

    def unique(self, const double complex[:] values, bint return_inverse=False,
               object mask=None):
        """Calculate unique values and optionally their inverse mapping."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0
            swiss_complex128_t c_key
            size_t idx
            int ret = 0
            double complex[::1] uniques_arr = np.empty(n, dtype=np.complex128)
            intp_t[::1] labels
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            np_uint8_t[::1] result_mask_arr

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)
            result_mask_arr = np.empty(n, dtype=np.uint8)

        if return_inverse:
            labels = np.empty(n, dtype=np.intp)
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            for idx in range(<size_t>count):
                                if result_mask_arr[idx] == 1:
                                    labels[i] = <intp_t>idx
                                    break
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex128_insert_if_absent(self.table, c_key, <size_t>count, &idx)  # noqa: E501
                        if ret == 1:
                            uniques_arr[count] = values[i]
                            if uses_mask:
                                result_mask_arr[count] = 0
                            labels[i] = count
                            count += 1
                        elif ret == 0:
                            labels[i] = <intp_t>idx
                        else:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                result_mask = np.asarray(result_mask_arr)[:count].copy()
                return uniques, np.asarray(labels), result_mask
            return uniques, np.asarray(labels)
        else:
            with nogil:
                for i in range(n):
                    if uses_mask and mask_view[i]:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            result_mask_arr[count] = 1
                            seen_na = True
                            count += 1
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex128_insert_key_only(self.table, c_key)
                        if ret == 1:
                            uniques_arr[count] = values[i]
                            if uses_mask:
                                result_mask_arr[count] = 0
                            count += 1
                        elif ret == -1:
                            break

            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

            uniques = np.asarray(uniques_arr)[:count].copy()
            if uses_mask:
                return uniques, np.asarray(result_mask_arr)[:count].copy()
            return uniques

    def factorize(self, const double complex[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None, object mask=None, bint ignore_na=True):
        """Calculate unique values and labels for categorical encoding (NaN-aware)."""
        cdef:
            Py_ssize_t i, n = len(values), count = 0, na_idx = -1
            swiss_complex128_t c_key
            size_t idx
            int ret = 0
            double complex[::1] uniques_arr = np.empty(n, dtype=np.complex128)
            intp_t[::1] labels = np.empty(n, dtype=np.intp)
            bint uses_mask = mask is not None
            bint seen_na = False
            const uint8_t[:] mask_view
            double real_part, imag_part

        if uses_mask:
            mask_view = mask if mask.dtype == np.uint8 else mask.view(np.uint8)

        # Pre-reserve capacity to avoid resize checks in hot loop
        if not swiss_complex128_reserve(self.table, <size_t>n):
            raise MemoryError("Failed to reserve capacity in Swiss table")

        with nogil:
            for i in range(n):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                real_part = c_key.real
                imag_part = c_key.imag
                # Check for NA: mask takes precedence, then NaN in components
                if uses_mask and mask_view[i]:
                    if ignore_na:
                        labels[i] = na_sentinel
                    else:
                        if not seen_na:
                            uniques_arr[count] = values[i]
                            na_idx = count
                            labels[i] = count
                            seen_na = True
                            count += 1
                        else:
                            labels[i] = na_idx
                elif ignore_na and (isnan(real_part) or isnan(imag_part)):
                    labels[i] = na_sentinel
                else:
                    if not swiss_complex128_get(self.table, c_key, &idx):
                        ret = swiss_complex128_insert(self.table, c_key, <size_t>count)
                        if ret == -1:
                            break
                        uniques_arr[count] = values[i]
                        labels[i] = count
                        count += 1
                    else:
                        labels[i] = <intp_t>idx

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        uniques = np.asarray(uniques_arr)[:count].copy()
        return uniques, np.asarray(labels)


def ismember_complex128(
    const double complex[:] arr, const double complex[:] values
) -> ndarray:
    """Return boolean array indicating membership of arr elements in values."""
    cdef:
        Py_ssize_t i, n_arr = len(arr), n_values = len(values)
        SwissTable_complex128 *table
        swiss_complex128_t c_key
        int ret = 0
        np_uint8_t[::1] result

    table = swiss_complex128_init_with_capacity(<size_t>n_values)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    try:
        # Build set from values
        with nogil:
            for i in range(n_values):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                ret = swiss_complex128_insert_key_only(table, c_key)
                if ret == -1:
                    break
        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        result = np.empty(n_arr, dtype=np.uint8)
        with nogil:
            for i in range(n_arr):
                c_key.real = arr[i].real
                c_key.imag = arr[i].imag
                result[i] = swiss_complex128_contains(table, c_key)

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_complex128_destroy(table)


def value_count_complex128(const double complex[:] values, bint dropna=True):
    """Count occurrences of each unique value (NaN-aware)."""
    cdef:
        Py_ssize_t i, n = len(values), n_unique = 0
        SwissTable_complex128 *table
        SwissTable_complex128 *order_table
        swiss_complex128_t c_key
        size_t count_val
        int ret = 0
        double complex[::1] keys_arr = np.empty(n, dtype=np.complex128)
        int64_t[::1] counts_arr
        Py_ssize_t na_count = 0
        double real_part, imag_part

    table = swiss_complex128_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    order_table = swiss_complex128_init_with_capacity(<size_t>n)
    if order_table == NULL:
        swiss_complex128_destroy(table)
        raise MemoryError("Failed to allocate Swiss table")

    try:
        with nogil:
            for i in range(n):
                c_key.real = values[i].real
                c_key.imag = values[i].imag
                real_part = c_key.real
                imag_part = c_key.imag
                # Count NaN separately if not dropping
                if isnan(real_part) or isnan(imag_part):
                    if not dropna:
                        na_count += 1
                    continue
                if swiss_complex128_get(table, c_key, &count_val):
                    swiss_complex128_insert(table, c_key, count_val + 1)
                else:
                    ret = swiss_complex128_insert(table, c_key, 1)
                    if ret == -1:
                        break
                    ret = swiss_complex128_insert(order_table, c_key, <size_t>n_unique)
                    if ret == -1:
                        break
                    keys_arr[n_unique] = values[i]
                    n_unique += 1

        if ret == -1:
            raise MemoryError("Failed to insert into Swiss table")

        # Add NaN slot if needed
        extra = 1 if na_count > 0 else 0
        counts_arr = np.empty(n_unique + extra, dtype=np.int64)
        with nogil:
            for i in range(n_unique):
                c_key.real = keys_arr[i].real
                c_key.imag = keys_arr[i].imag
                swiss_complex128_get(table, c_key, &count_val)
                counts_arr[i] = <int64_t>count_val

        keys = np.asarray(keys_arr)[:n_unique].copy()
        counts = np.asarray(counts_arr)[:n_unique].copy()

        # Add NaN to output if not dropping
        if na_count > 0:
            keys = np.append(keys, complex(np.nan, np.nan))
            counts = np.append(counts, na_count)

        return keys, counts

    finally:
        swiss_complex128_destroy(table)
        swiss_complex128_destroy(order_table)


def duplicated_complex128(const double complex[:] values, object keep="first",
                          const uint8_t[:] mask=None):
    """
    Return boolean array indicating duplicated values (NaN-aware).

    Parameters
    ----------
    values : ndarray[complex128]
        Array to check for duplicates
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates as True except for first occurrence
        - 'last': Mark duplicates as True except for last occurrence
        - False: Mark all duplicates as True
    mask : ndarray[uint8], optional
        If provided, mask[i] == 1 indicates that values[i] is NA/missing.

    Returns
    -------
    ndarray[bool]
        Boolean array indicating duplicates
    """
    cdef:
        Py_ssize_t i, n = len(values), first_na = -1
        SwissTable_complex128 *table
        swiss_complex128_t c_key
        size_t idx
        int ret = 0
        np_uint8_t[::1] result
        bint keep_first = keep == "first"
        bint keep_last = keep == "last"
        bint _keep_none = keep is False
        bint uses_mask = mask is not None
        bint seen_na = False
        bint seen_multiple_na = False

    if keep not in ("first", "last", False):
        raise ValueError('keep must be either "first", "last" or False')

    table = swiss_complex128_init_with_capacity(<size_t>n)
    if table == NULL:
        raise MemoryError("Failed to allocate Swiss table")

    result = np.zeros(n, dtype=np.uint8)

    try:
        if keep_first:
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex128_insert_key_only(table, c_key)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        elif keep_last:
            with nogil:
                for i in range(n - 1, -1, -1):
                    if uses_mask and mask[i]:
                        if seen_na:
                            result[i] = 1
                        else:
                            seen_na = True
                            result[i] = 0
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex128_insert_key_only(table, c_key)
                        if ret == 0:
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        else:  # keep == False
            with nogil:
                for i in range(n):
                    if uses_mask and mask[i]:
                        if not seen_na:
                            first_na = i
                            seen_na = True
                            result[i] = 0
                        elif not seen_multiple_na:
                            result[i] = 1
                            result[first_na] = 1
                            seen_multiple_na = True
                        else:
                            result[i] = 1
                    else:
                        c_key.real = values[i].real
                        c_key.imag = values[i].imag
                        ret = swiss_complex128_insert_if_absent(table, c_key, <size_t>i, &idx)  # noqa: E501
                        if ret == 0:
                            result[idx] = 1
                            result[i] = 1
                        elif ret == -1:
                            break
            if ret == -1:
                raise MemoryError("Failed to insert into Swiss table")

        return np.asarray(result).view(np.bool_)

    finally:
        swiss_complex128_destroy(table)
