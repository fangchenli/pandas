"""Validated row maps. NumPy is the first internal map adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import numpy as np
import numpy.typing as npt

from pdruntime.types import is_integer

IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]


def owned(array: npt.ArrayLike, dtype: npt.DTypeLike = None) -> np.ndarray:
    """Copy into a runtime-owned, read-only array."""
    result = np.array(array, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def frozen(array: np.ndarray) -> np.ndarray:
    """Mark a freshly allocated array read-only without copying it."""
    array.flags.writeable = False
    return array


def check_length(value: object) -> int:
    if not is_integer(value):
        raise TypeError("length must be an integer")
    if value < 0:
        raise ValueError("length must be nonnegative")
    return int(value)


# Maps compare by identity: sharing is keyed on the map object, never on equal
# contents, because equal positions need not denote the same row ordering.
@dataclass(frozen=True, eq=False)
class Identity:
    source_length: int
    present: ClassVar[None] = None

    @property
    def output_length(self) -> int:
        return self.source_length


@dataclass(frozen=True, eq=False)
class Affine:
    source_length: int
    start: int
    step: int
    output_length: int
    present: ClassVar[None] = None


@dataclass(frozen=True, eq=False)
class Positions:
    """Dense positions; ``present`` marks introduced missing rows when not None."""

    source_length: int
    positions: IntArray
    present: BoolArray | None = None

    @property
    def output_length(self) -> int:
        return len(self.positions)


RowMap: TypeAlias = Identity | Affine | Positions


class NumpyMaps:
    namespace = "numpy"
    device = "cpu"

    def identity(self, n: int) -> Identity:
        return Identity(check_length(n))

    def positions(
        self,
        n: int,
        indices: npt.ArrayLike,
        present: npt.ArrayLike | None = None,
        *,
        negative: bool = True,
    ) -> Positions:
        n = check_length(n)
        # Reject Python mixtures such as [True, 1] before NumPy erases bool identity.
        if isinstance(indices, (list, tuple)) and any(
            isinstance(x, (bool, np.bool_)) for x in indices
        ):
            raise TypeError("boolean values are not positions")
        a = np.asarray(indices)
        if a.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        if isinstance(indices, (list, tuple)) and not len(a):
            a = a.astype(np.int64)  # np.asarray([]) is float64
        if a.dtype.kind not in "iu":
            raise TypeError("positions must be integers")
        valid = self._presence(present, len(a))
        selected = a[valid]
        lower = -n if negative else 0
        if np.any(selected < lower) or np.any(selected >= n):
            raise IndexError("position out of bounds")
        result = np.zeros(len(a), dtype=np.int64)
        result[valid] = selected.astype(np.int64)
        if negative:
            result[result < 0] += n
        return Positions(n, frozen(result), None if valid.all() else owned(valid))

    @staticmethod
    def _presence(present: npt.ArrayLike | None, size: int) -> BoolArray:
        if present is None:
            return np.ones(size, dtype=bool)
        if isinstance(present, Sequence) and not present:
            valid = np.empty(0, dtype=bool)
        else:
            valid = np.asarray(present)
        if valid.dtype.kind != "b":
            raise TypeError("map presence must be boolean")
        if valid.shape != (size,):
            raise ValueError("map presence length mismatch")
        return valid

    def slice(self, n: int, value: slice) -> Identity | Affine:
        n = check_length(n)
        if not isinstance(value, slice):
            raise TypeError("expected a slice")
        components = (value.start, value.stop, value.step)
        if any(c is not None and not is_integer(c) for c in components):
            raise TypeError("slice components must be integers")
        start, stop, step = value.indices(n)
        size = len(range(start, stop, step))
        if start == 0 and step == 1 and size == n:
            return Identity(n)
        return Affine(n, start, step, size)

    def dense(self, row_map: RowMap) -> tuple[IntArray, BoolArray | None]:
        match row_map:
            case Positions(positions=positions, present=present):
                return positions, present
            case Identity(output_length=size):
                return np.arange(size, dtype=np.int64), None
            case Affine(start=start, step=step, output_length=size):
                return start + step * np.arange(size, dtype=np.int64), None
        raise TypeError(f"not a row map: {row_map!r}")

    def at(self, row_map: RowMap, index: int) -> int | None:
        """Source position for one output position, or None if it is missing."""
        match row_map:
            case Identity():
                return index
            case Affine(start=start, step=step):
                return start + step * index
            case Positions(positions=positions, present=present):
                if present is not None and not present[index]:
                    return None
                return int(positions[index])
        raise TypeError(f"not a row map: {row_map!r}")

    def compose(self, inner: RowMap, outer: RowMap) -> RowMap:
        """The map equivalent to ``inner[outer]``; missing entries propagate."""
        if outer.source_length != inner.output_length:
            raise ValueError("map coordinate lengths differ")
        match inner, outer:
            case Identity(), _:
                return outer
            case _, Identity():
                return inner
            case Affine(), Affine():
                return Affine(
                    inner.source_length,
                    inner.start + inner.step * outer.start,
                    inner.step * outer.step,
                    outer.output_length,
                )
            case Positions(present=None), Affine():
                # a strided view of the inner positions: no gather, no copy
                view = inner.positions[self.slice_object(outer)]
                return Positions(inner.source_length, frozen(view))
            case Positions(present=None), Positions(present=None):
                # nothing missing on either side: one gather
                gathered = inner.positions[outer.positions]
                return Positions(inner.source_length, frozen(gathered))
        indices, present = self.dense(outer)
        valid = np.ones(len(indices), dtype=bool) if present is None else present.copy()
        slots = np.flatnonzero(valid)
        parents = indices[slots]
        positions = np.zeros(len(indices), dtype=np.int64)
        if isinstance(inner, Affine):
            positions[slots] = inner.start + inner.step * parents
        else:
            if inner.present is not None:
                # Test presence before dereferencing: absent slots hold no position.
                parent_present = inner.present[parents]
                valid[slots] = parent_present
                slots, parents = slots[parent_present], parents[parent_present]
            positions[slots] = inner.positions[parents]
        return Positions(
            inner.source_length,
            frozen(positions),
            None if valid.all() else frozen(valid),
        )

    def slice_object(self, row_map: Affine) -> slice:
        stop = row_map.start + row_map.step * row_map.output_length
        return slice(
            row_map.start,
            None if row_map.step < 0 and stop < 0 else stop,
            row_map.step,
        )
