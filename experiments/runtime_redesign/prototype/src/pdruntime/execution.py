"""Selection policies and subset reads share all backend semantics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, get_args

import numpy as np

from pdruntime.maps import Identity, Positions, frozen
from pdruntime.types import INT64, NA, UTF8, Scalar
from pdruntime.values import Constant, Physical, Segments, Select, ValueCell

if TYPE_CHECKING:
    from pdruntime.backends import Backend, Storage
    from pdruntime.maps import IntArray, NumpyMaps, RowMap
    from pdruntime.metrics import Metrics

Strategy: TypeAlias = Literal["eager", "compact", "tree"]
# Memoizes map composition and segment routing for one frame operation.
CompositionCache: TypeAlias = "dict[tuple[object, ...], object]"


@dataclass(frozen=True, eq=False)
class Routing:
    """How a map's present rows split across segments; columns share it."""

    parts: tuple[tuple[int, Positions], ...]  # (child index, child-local map)
    output: Positions | None  # orders the concatenated parts; None if already


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
                case Segments() as segments:
                    return self._read_segments(cell, segments, row_map, cache)

    def _read_segments(
        self,
        cell: ValueCell,
        segments: Segments,
        row_map: RowMap,
        cache: CompositionCache,
    ) -> Storage:
        """Read each touched child once, then place its rows in output order."""
        adapter = self.adapter(cell)
        children, offsets = self._leaves(segments)
        if isinstance(row_map, Identity):
            parts = [
                self.read(child, self.maps.identity(child.length), cache)
                for child in children
            ]
            return adapter.concat(cell.dtype, parts)
        # Routing depends only on offset values and the map, so columns with the
        # same layout share it, and the shared child maps compose once below.
        key = ("route", offsets.tobytes(), row_map)
        if (routing := cache.get(key)) is None:
            routing = cache[key] = self._route(offsets, row_map)
        parts = [self.read(children[k], local, cache) for k, local in routing.parts]
        combined = adapter.concat(cell.dtype, parts)
        if routing.output is None:
            return combined
        return adapter.select(combined, routing.output)

    @staticmethod
    def _leaves(segments: Segments) -> tuple[tuple[ValueCell, ...], IntArray]:
        """Flatten nested segments iteratively; tree nesting may be deep."""
        children = segments.children
        if not any(isinstance(child.snapshot(), Segments) for child in children):
            return children, segments.offsets
        leaves, stack = [], list(reversed(children))
        while stack:
            child = stack.pop()
            state = child.snapshot()
            if isinstance(state, Segments):
                stack.extend(reversed(state.children))
            else:
                leaves.append(child)
        lengths = [0, *(leaf.length for leaf in leaves)]
        return tuple(leaves), frozen(np.cumsum(lengths, dtype=np.int64))

    def _route(self, offsets: IntArray, row_map: RowMap) -> Routing:
        indices, present = self.maps.dense(row_map)
        slots = np.arange(len(indices)) if present is None else np.flatnonzero(present)
        sources = indices[slots]
        owners = np.searchsorted(offsets, sources, side="right") - 1
        in_order = bool(np.all(owners[1:] >= owners[:-1]))
        order = (
            np.arange(len(owners)) if in_order else np.argsort(owners, kind="stable")
        )
        bounds = np.searchsorted(owners[order], np.arange(len(offsets)))
        parts = []
        for k in range(len(offsets) - 1):
            group = order[bounds[k] : bounds[k + 1]]
            if len(group):
                size = int(offsets[k + 1] - offsets[k])
                parts.append((k, Positions(size, frozen(sources[group] - offsets[k]))))
        self.metrics.record("segment_routing", rows=len(indices))
        if in_order and present is None:
            return Routing(tuple(parts), None)
        # Row j of the concatenated parts belongs at output slot ``slots[order[j]]``.
        positions = np.zeros(len(indices), dtype=np.int64)
        positions[slots[order]] = np.arange(len(order))
        return Routing(tuple(parts), Positions(len(order), frozen(positions), present))

    def concat(
        self,
        cells: Sequence[ValueCell],
        layouts: dict[tuple[int, ...], IntArray] | None = None,
    ) -> ValueCell:
        """``cells`` end to end; the caller checks that dtypes and backends agree.

        ``layouts`` lets the columns of one frame operation share offsets.
        """
        first = cells[0]
        adapter = self.adapter(first)
        if self.strategy == "eager":
            parts = [self.read(cell, self.maps.identity(cell.length)) for cell in cells]
            return self.physical_cell(adapter.concat(first.dtype, parts))
        children = []
        for cell in cells:
            state = cell.snapshot()
            # Compact flattens nested segments now, as it composes maps; tree nests.
            if self.strategy == "compact" and isinstance(state, Segments):
                children.extend(state.children)
            elif cell.length:
                children.append(cell)
        if not children:
            return self.physical_cell(adapter.empty(first.dtype))
        if len(children) == 1:
            return children[0]
        layouts = {} if layouts is None else layouts
        lengths = tuple(child.length for child in children)
        if (offsets := layouts.get(lengths)) is None:
            offsets = layouts[lengths] = frozen(
                np.cumsum((0, *lengths), dtype=np.int64)
            )
        self.metrics.record("segments", count=len(children))
        segments = Segments(tuple(children), offsets)
        return ValueCell(first.dtype, int(offsets[-1]), first.backend, segments)

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
                case Segments(children, offsets):
                    k = int(np.searchsorted(offsets, position, side="right")) - 1
                    position -= int(offsets[k])
                    cell = children[k]

    def describe(self, cell: ValueCell) -> dict[str, object]:
        depth = 0
        state = cell.snapshot()
        while isinstance(state, Select):
            depth += 1
            state = state.source.snapshot()
        result = {
            "backend": cell.backend,
            "length": cell.length,
            "selection_depth": depth,
            "leaf": type(state).__name__,
        }
        if isinstance(state, Segments):
            result["segments"] = len(state.children)
        return result

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
                case Segments(children, offsets):
                    stack.extend(children)
                    map_buffers[id(offsets)] = offsets.nbytes
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
