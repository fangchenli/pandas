"""Key computation for structural operations: orders and match positions.

Keys arrive as materialized, read-only NumPy values plus validity. Payload
columns and labels never enter. Matching uses the public pandas
``Index.join``, the same machinery the axis adapter uses for labels. It is
isolated and replaceable, not part of the representation model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pdruntime.maps import BoolArray, IntArray, NumpyMaps, RowMap


def sort_order(values: np.ndarray, valid: BoolArray | None) -> IntArray:
    """Stable ascending order, with missing keys last in their input order."""
    if valid is None:
        return np.argsort(values, kind="stable")
    present = np.flatnonzero(valid)
    order = present[np.argsort(values[present], kind="stable")]
    return np.concatenate([order, np.flatnonzero(~valid)])


def join_maps(
    left: np.ndarray, right: np.ndarray, how: str, maps: NumpyMaps
) -> tuple[RowMap, RowMap]:
    """Left and right row maps for an inner or left equijoin.

    Rows follow the left keys' order and, within one left key, the right
    keys' order, as ``pandas.merge`` orders them (pandas >= 2.2). Right keys
    may repeat. A left row with no match (``how="left"``) maps to a missing
    right row. One public ``Index.join`` call covers unique and duplicate keys.
    """
    _, rows, matches = pd.Index(left).join(
        pd.Index(right), how=how, return_indexers=True, sort=False
    )
    size = len(left) if rows is None else len(rows)
    # A left join with one output row per left row keeps every row in order.
    if rows is None or (
        size == len(left) and (how == "left" or np.array_equal(rows, np.arange(size)))
    ):
        left_map = maps.identity(len(left))
    else:
        left_map = maps.positions(len(left), rows, negative=False)
    if matches is None:
        right_map = maps.identity(len(right))
    else:
        right_map = maps.positions(len(right), matches, matches >= 0, negative=False)
    return left_map, right_map
