"""Validated row maps. NumPy is the first internal map adapter."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def owned(array, dtype=None):
    result = np.array(array, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def length(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("length must be an integer")
    if value < 0:
        raise ValueError("length must be nonnegative")
    return int(value)


@dataclass(frozen=True, eq=False)
class Identity:
    source_length: int

    @property
    def output_length(self):
        return self.source_length


@dataclass(frozen=True, eq=False)
class Affine:
    source_length: int
    start: int
    step: int
    output_length: int


@dataclass(frozen=True, eq=False)
class Positions:
    source_length: int
    positions: np.ndarray
    present: np.ndarray | None = None

    @property
    def output_length(self):
        return len(self.positions)


class NumpyMaps:
    namespace = "numpy"
    device = "cpu"

    def identity(self, n):
        return Identity(length(n))

    def positions(self, n, indices, present=None, *, negative=True):
        n = length(n)
        # Reject Python mixtures such as [True, 1] before NumPy erases bool identity.
        if isinstance(indices, (list, tuple)):
            if any(isinstance(x, (bool, np.bool_)) for x in indices):
                raise TypeError("boolean values are not positions")
        a = np.asarray(indices)
        if a.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        if a.dtype.kind not in "iu":
            if len(a) or not isinstance(indices, (list, tuple)):
                raise TypeError("positions must be integers")
        if present is None:
            valid = np.ones(len(a), dtype=bool)
        else:
            valid = np.asarray(present)
            if isinstance(present, (list, tuple)) and not present:
                valid = np.empty(0, dtype=bool)
            if valid.dtype.kind != "b":
                raise TypeError("map presence must be boolean")
            if valid.shape != a.shape:
                raise ValueError("map presence length mismatch")
        selected = a[valid]
        lower = -n if negative else 0
        if np.any(selected < lower) or np.any(selected >= n):
            raise IndexError("position out of bounds")
        result = np.zeros(len(a), dtype=np.int64)
        result[valid] = selected.astype(np.int64)
        if negative:
            result[result < 0] += n
        return Positions(n, owned(result), None if valid.all() else owned(valid))

    def slice(self, n, value):
        n = length(n)
        if not isinstance(value, slice):
            raise TypeError("expected a slice")
        for component in (value.start, value.stop, value.step):
            if component is not None and (
                isinstance(component, (bool, np.bool_))
                or not isinstance(component, Integral)
            ):
                raise TypeError("slice components must be integers")
        start, stop, step = value.indices(n)
        size = len(range(start, stop, step))
        if start == 0 and step == 1 and size == n:
            return Identity(n)
        return Affine(n, start, step, size)

    def dense(self, row_map):
        if isinstance(row_map, Positions):
            return row_map.positions, row_map.present
        if isinstance(row_map, Identity):
            return np.arange(row_map.output_length, dtype=np.int64), None
        return (
            row_map.start
            + row_map.step * np.arange(row_map.output_length, dtype=np.int64),
            None,
        )

    def at(self, row_map, index):
        if isinstance(row_map, Identity):
            return index
        if isinstance(row_map, Affine):
            return row_map.start + row_map.step * index
        if row_map.present is not None and not row_map.present[index]:
            return None
        return int(row_map.positions[index])

    def compose(self, inner, outer, context=None):
        if outer.source_length != inner.output_length:
            raise ValueError("map coordinate lengths differ")
        if isinstance(inner, Identity):
            return outer
        if isinstance(outer, Identity):
            return inner
        key = (inner, outer)
        if context is not None and key in context:
            return context[key]
        if isinstance(inner, Affine) and isinstance(outer, Affine):
            result = Affine(
                inner.source_length,
                inner.start + inner.step * outer.start,
                inner.step * outer.step,
                outer.output_length,
            )
        else:
            indices, present = self.dense(outer)
            valid = (
                np.ones(len(indices), dtype=bool) if present is None else present.copy()
            )
            positions = np.zeros(len(indices), dtype=np.int64)
            if isinstance(inner, Affine):
                positions[valid] = inner.start + inner.step * indices[valid]
            else:
                slots = np.flatnonzero(valid)
                parent_indices = indices[slots]
                if inner.present is not None:
                    valid[slots] &= inner.present[parent_indices]
                positions[valid] = inner.positions[indices[valid]]
            result = Positions(
                inner.source_length,
                owned(positions),
                None if valid.all() else owned(valid),
            )
        if context is not None:
            context[key] = result
        return result

    def slice_object(self, row_map):
        stop = row_map.start + row_map.step * row_map.output_length
        return slice(
            row_map.start,
            None if row_map.step < 0 and stop < 0 else stop,
            row_map.step,
        )
