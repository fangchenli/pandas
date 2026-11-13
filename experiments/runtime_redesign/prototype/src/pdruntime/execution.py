"""Selection policies and subset reads share all backend semantics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, TypeAlias, get_args

from pdruntime.maps import Identity, Positions
from pdruntime.types import INT64, NA, UTF8, Scalar
from pdruntime.values import Constant, Physical, Select, ValueCell

if TYPE_CHECKING:
    from pdruntime.backends import Backend, Storage
    from pdruntime.maps import NumpyMaps, RowMap
    from pdruntime.metrics import Metrics

Strategy: TypeAlias = Literal["eager", "compact", "tree"]
# Memoizes composition by map identity for the duration of one frame operation.
CompositionCache: TypeAlias = "dict[tuple[RowMap, RowMap], RowMap]"


class Execution:
    def __init__(
        self,
        strategy: Strategy,
        maps: NumpyMaps,
        backends: dict[str, Backend],
        metrics: Metrics,
    ) -> None:
        if strategy not in get_args(Strategy):
            raise ValueError("strategy must be eager, compact, or tree")
        self.strategy = strategy
        self.maps = maps
        self.backends = backends
        self.metrics = metrics

    def adapter(self, cell: ValueCell) -> Backend:
        return self.backends[cell.backend]

    def compose(self, inner: RowMap, outer: RowMap, cache: CompositionCache) -> RowMap:
        key = (inner, outer)
        if (result := cache.get(key)) is None:
            result = cache[key] = self.maps.compose(inner, outer)
            if result is not inner and result is not outer:
                self.metrics.record("map_composition", rows=result.output_length)
        return result

    def physical_cell(self, storage: Storage) -> ValueCell:
        return ValueCell(
            storage.dtype, len(storage), storage.backend, Physical(storage)
        )

    def select(
        self, cell: ValueCell, row_map: RowMap, cache: CompositionCache | None = None
    ) -> ValueCell:
        if row_map.source_length != cell.length:
            raise ValueError("selection source length mismatch")
        if row_map.output_length == 0:
            return self.physical_cell(self.adapter(cell).empty(cell.dtype))
        if isinstance(row_map, Identity):
            return cell
        cache = {} if cache is None else cache
        state = cell.snapshot()
        if isinstance(state, Constant) and row_map.present is None:
            return ValueCell(cell.dtype, row_map.output_length, cell.backend, state)
        if self.strategy == "eager":
            return self.physical_cell(self.read(cell, row_map, cache))
        if self.strategy == "compact":
            while isinstance(state, Select):
                row_map = self.compose(state.row_map, row_map, cache)
                cell = state.source
                state = cell.snapshot()
        return ValueCell(
            cell.dtype, row_map.output_length, cell.backend, Select(cell, row_map)
        )

    def read(
        self, cell: ValueCell, row_map: RowMap, cache: CompositionCache | None = None
    ) -> Storage:
        """The requested positions of ``cell``, gathering payload only at the leaf."""
        cache = {} if cache is None else cache
        while True:
            match cell.snapshot():
                case Physical(storage):
                    return self.adapter(cell).select(storage, row_map)
                case Constant(scalar):
                    return self.adapter(cell).full(
                        cell.dtype, scalar, row_map.output_length, row_map.present
                    )
                case Select(source, inner):
                    row_map = self.compose(inner, row_map, cache)
                    cell = source

    def ensure_physical(
        self, cell: ValueCell, cache: CompositionCache | None = None
    ) -> Storage:
        with cell.lock:
            if isinstance(cell.state, Physical):
                self.metrics.record("cache_hit")
                return cell.state.storage
            storage = self.read(cell, self.maps.identity(cell.length), cache)
            self.adapter(cell).wait(storage)
            # Replace the descriptor so the cached result releases its sources.
            cell.state = Physical(storage)
            self.metrics.record("materialization", rows=cell.length)
            return storage

    def scalar_at(self, cell: ValueCell, position: int) -> Scalar:
        while True:
            match cell.snapshot():
                case Physical(storage):
                    return self.adapter(cell).scalar_at(storage, position)
                case Constant(scalar):
                    return scalar
                case Select(source, row_map):
                    position = self.maps.at(row_map, position)
                    if position is None:
                        return NA
                    cell = source

    def describe(self, cell: ValueCell) -> dict[str, object]:
        depth = 0
        state = cell.snapshot()
        while isinstance(state, Select):
            depth += 1
            state = state.source.snapshot()
        return {
            "backend": cell.backend,
            "length": cell.length,
            "selection_depth": depth,
            "leaf": type(state).__name__,
        }

    def memory_report(self, cells: Iterable[ValueCell]) -> dict[str, object]:
        cells = list(cells)
        buffers: dict[tuple[str, int], int] = {}
        map_buffers: dict[int, int] = {}
        visited: set[int] = set()
        stack = list(cells)
        while stack:
            cell = stack.pop()
            if id(cell) in visited:
                continue
            visited.add(id(cell))
            match cell.snapshot():
                case Physical(storage):
                    buffers.update(self.adapter(cell).buffers(storage))
                case Select(source, Positions(positions=positions, present=present)):
                    stack.append(source)
                    for array in (positions, present):
                        if array is not None:
                            map_buffers[id(array)] = array.nbytes
                case Select(source, _):
                    stack.append(source)
        return {
            "known_retained_payload_bytes": sum(buffers.values()),
            "map_bytes": sum(map_buffers.values()),
            "value_cells": len(visited),
            "max_selection_depth": max(
                (self.describe(c)["selection_depth"] for c in cells), default=0
            ),
            "logical_fixed_value_bytes": sum(
                c.length * (8 if c.dtype == INT64 else 1)
                for c in cells
                if c.dtype != UTF8
            ),
            "coverage": (
                "known buffer owners only; excludes variable logical widths, "
                "Python metadata, allocator overhead and axes"
            ),
        }
