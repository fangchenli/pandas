"""
Arrow compute kernel implementations for lazy pandas.

This package contains Arrow-based implementations for operations.
These kernels use PyArrow compute functions directly on Arrow arrays.

Submodules:
- core: String, null, arithmetic, comparison, logical, aggregation, sort, unique, cast
- groupby: GroupBy aggregation operations
- join: Join operations (inner, left, right, outer, cross)
- window: Cumulative and window functions
- datetime: Datetime extraction and manipulation
- string: Additional string operations
- misc: Rolling, shift, fill operations
"""

# Import all submodules to trigger kernel registration
from pandas.lazy.backends.arrow import (
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
