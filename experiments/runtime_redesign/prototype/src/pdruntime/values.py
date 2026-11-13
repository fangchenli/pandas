"""Immutable values with replaceable physical representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from pdruntime.backends import Storage
    from pdruntime.maps import IntArray, RowMap
    from pdruntime.types import LogicalType, Scalar


@dataclass(frozen=True)
class Physical:
    storage: Storage


@dataclass(frozen=True)
class Select:
    source: ValueCell
    row_map: RowMap


@dataclass(frozen=True)
class Constant:
    scalar: Scalar


# Holds an array, so it compares by identity, as the row maps do.
@dataclass(frozen=True, eq=False)
class Segments:
    """Children laid end to end; ``offsets`` holds cumulative starts and the total."""

    children: tuple[ValueCell, ...]
    offsets: IntArray


State: TypeAlias = "Physical | Select | Constant | Segments"


@dataclass(eq=False)
class ValueCell:
    """One immutable logical sequence. Only its representation may change."""

    dtype: LogicalType
    length: int
    backend: str
    state: State
    lock: RLock = field(default_factory=RLock, repr=False)

    def snapshot(self) -> State:
        with self.lock:
            return self.state
