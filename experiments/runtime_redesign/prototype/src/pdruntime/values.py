"""Immutable values with replaceable physical representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from pdruntime.backends import Storage
    from pdruntime.maps import RowMap
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


State: TypeAlias = "Physical | Select | Constant"


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
