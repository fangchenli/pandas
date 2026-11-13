"""Owned flat pandas axes; label matching does not touch payload storage.

pandas Index objects are immutable, so handles share them internally. Public
accessors return deep copies so callers cannot reach the underlying arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pdruntime.maps import Affine, Identity, Positions

if TYPE_CHECKING:
    from pdruntime.maps import NumpyMaps, RowMap


LABEL_TYPES = ("integer", "string")


def axis(values: Iterable[object] | None, size: int) -> pd.Index:
    if values is None:
        return pd.RangeIndex(size)
    return _validated(pd.Index(values, copy=True), size)


def labels(values: Iterable[object]) -> pd.Index:
    """An owned, validated copy of flat integer or string labels."""
    return _validated(pd.Index(values, copy=True))


def _validated(result: pd.Index, size: int | None = None) -> pd.Index:
    if isinstance(result, pd.MultiIndex) or result.hasnans:
        raise TypeError("only non-null flat integer/string labels are supported")
    if size is not None and len(result) != size:
        raise ValueError("axis length mismatch")
    if len(result) and result.inferred_type not in LABEL_TYPES:
        raise TypeError("axis must contain homogeneous integer or string labels")
    return result


def take(index: pd.Index, row_map: RowMap, maps: NumpyMaps) -> pd.Index:
    """Labels for a public selection, which never introduces missing rows."""
    if row_map.present is not None:
        raise ValueError("public label selection requires present positions")
    match row_map:
        case Identity():
            return index
        case Affine():
            return index[maps.slice_object(row_map)]
        case Positions(positions=positions):
            return index.take(positions)
    raise TypeError(f"not a row map: {row_map!r}")


def align(
    left: pd.Index, right: pd.Index, maps: NumpyMaps
) -> tuple[pd.Index, RowMap, RowMap]:
    """Outer-join labels as pandas binary operations do; return both row maps."""
    if left.equals(right):
        return left, maps.identity(len(left)), maps.identity(len(right))
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


def reindexer(source: pd.Index, target: pd.Index, maps: NumpyMaps) -> RowMap:
    """Map ``target`` labels onto ``source`` positions; absent labels are missing."""
    if source.equals(target):
        return maps.identity(len(source))
    if not source.is_unique:
        raise ValueError("cannot reindex from an axis with duplicate labels")
    indexer = source.get_indexer(target)
    return maps.positions(len(source), indexer, indexer >= 0, negative=False)


def concat(indexes: Sequence[pd.Index]) -> pd.Index:
    """Labels end to end, duplicates kept, as ``pandas.concat`` does."""
    kinds = {index.inferred_type for index in indexes if len(index)}
    result = indexes[0].append(list(indexes[1:]))
    # Equal kinds can still promote: int64 with uint64 labels becomes float64.
    if len(kinds) > 1 or (len(result) and result.inferred_type not in LABEL_TYPES):
        raise TypeError("concatenated axes must have the same label type")
    return result
