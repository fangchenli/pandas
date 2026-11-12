"""Immutable values with replaceable physical representations."""

from dataclasses import dataclass, field
from threading import RLock


@dataclass(frozen=True)
class Physical:
    storage: object


@dataclass(frozen=True)
class Select:
    source: object
    row_map: object


@dataclass(frozen=True)
class Constant:
    scalar: object


@dataclass(eq=False)
class ValueCell:
    dtype: object
    length: int
    backend: str
    state: object
    lock: object = field(default_factory=RLock, repr=False)

    def snapshot(self):
        with self.lock:
            return self.state
