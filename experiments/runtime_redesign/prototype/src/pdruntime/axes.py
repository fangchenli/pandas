"""Owned flat pandas axes; label matching does not touch payload storage."""

import numpy as np
import pandas as pd


def axis(values, size):
    if values is None:
        return pd.RangeIndex(size)
    result = pd.Index(values, copy=True)
    if isinstance(result, pd.MultiIndex) or result.hasnans:
        raise TypeError("only non-null flat integer/string labels are supported")
    if len(result) != size:
        raise ValueError("axis length mismatch")
    if len(result) and result.inferred_type not in ("integer", "string"):
        raise TypeError("axis must contain homogeneous integer or string labels")
    return result


def align(left, right, maps):
    if left.equals(right):
        return left.copy(deep=True), maps.identity(len(left)), maps.identity(len(right))
    if not left.is_unique or not right.is_unique:
        raise ValueError("unequal axes must be unique for alignment")
    if len(left) and len(right) and left.inferred_type != right.inferred_type:
        raise TypeError("aligned axes must have the same label type")
    labels, li, ri = left.join(right, how="outer", return_indexers=True, sort=False)
    li = np.arange(len(left)) if li is None else li
    ri = np.arange(len(right)) if ri is None else ri
    return (
        labels,
        maps.positions(len(left), li, li >= 0, negative=False),
        maps.positions(len(right), ri, ri >= 0, negative=False),
    )
