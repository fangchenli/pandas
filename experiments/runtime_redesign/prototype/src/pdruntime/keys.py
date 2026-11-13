"""Key computation for structural operations: orders and match positions.

Keys arrive as materialized, read-only NumPy values plus validity. Payload
columns and labels never enter. Matching reuses the axis adapter's pandas
reindexer; it is isolated and replaceable, not part of the representation model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pdruntime import axes

if TYPE_CHECKING:
    from pdruntime.maps import BoolArray, IntArray, NumpyMaps, RowMap


def sort_order(values: np.ndarray, valid: BoolArray | None) -> IntArray:
    """Stable ascending order, with missing keys last in their input order."""
    if valid is None:
        return np.argsort(values, kind="stable")
    present = np.flatnonzero(valid)
    order = present[np.argsort(values[present], kind="stable")]
    return np.concatenate([order, np.flatnonzero(~valid)])


def match_unique(left: np.ndarray, right: np.ndarray, maps: NumpyMaps) -> RowMap:
    """Map each left key to the equal right key; unmatched keys are missing."""
    source = pd.Index(right)
    if not source.is_unique:
        raise ValueError("join requires unique right keys")
    return axes.reindexer(source, pd.Index(left), maps)
