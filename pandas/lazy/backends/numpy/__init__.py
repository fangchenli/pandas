"""
NumPy kernel implementations for lazy pandas.

This package contains NumPy-based implementations for operations.
These kernels use NumPy functions directly on NumPy arrays.

Submodules:
- core: Arithmetic, comparison, logical, null, aggregation, sort, unique, cast
- groupby: GroupBy aggregation operations
- join: Join operations (inner, left, right, outer, cross)
- window: Cumulative and window functions
- datetime: Datetime extraction and manipulation
- string: String operations
- misc: Rolling, shift, fill operations

Future: numexpr Integration
---------------------------
For large arrays (>1M elements), numexpr can accelerate arithmetic:

    _MIN_ELEMENTS = 1_000_000
    _NUMEXPR_DTYPES = {"int64", "int32", "float64", "float32", "bool"}

    def numpy_add(left, right):
        if (len(left) > _MIN_ELEMENTS and
            left.dtype.name in _NUMEXPR_DTYPES):
            import numexpr as ne
            return ne.evaluate("left + right")
        return np.add(left, right)

Supported numexpr ops: +, -, *, /, **, ==, !=, <, <=, >, >=, &, |, ^
NOT supported: //, % (use numpy for these)
"""

# Import all submodules to trigger kernel registration
from pandas.lazy.backends.numpy import (
    core,
    datetime,
    groupby,
    join,
    misc,
    string,
    window,
)

__all__ = [
    "core",
    "datetime",
    "groupby",
    "join",
    "misc",
    "string",
    "window",
]
