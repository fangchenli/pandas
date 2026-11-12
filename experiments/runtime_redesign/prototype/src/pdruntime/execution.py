"""Selection policies and subset reads share all backend semantics."""

import numpy as np
import pyarrow as pa

from pdruntime.backends import ARROW_TYPES, NUMPY_TYPES, Storage, as_chunks
from pdruntime.maps import Identity, Positions, owned
from pdruntime.types import INT64, NA
from pdruntime.values import Constant, Physical, Select, ValueCell


class Execution:
    def __init__(self, strategy, maps, backends, metrics):
        if strategy not in ("eager", "compact", "tree"):
            raise ValueError("strategy must be eager, compact, or tree")
        self.strategy = strategy
        self.maps = maps
        self.backends = backends
        self.metrics = metrics

    def compose(self, inner, outer, context):
        size = len(context)
        result = self.maps.compose(inner, outer, context)
        if len(context) > size:
            self.metrics.record("map_composition", rows=result.output_length)
        return result

    def physical_cell(self, storage):
        return ValueCell(
            storage.dtype, len(storage), storage.backend, Physical(storage)
        )

    def select(self, cell, row_map, context=None):
        if row_map.source_length != cell.length:
            raise ValueError("selection source length mismatch")
        if row_map.output_length == 0:
            return self.physical_cell(self.backends[cell.backend].empty(cell.dtype))
        if isinstance(row_map, Identity):
            return cell
        context = {} if context is None else context
        state = cell.snapshot()
        if isinstance(state, Constant) and not (
            isinstance(row_map, Positions) and row_map.present is not None
        ):
            return ValueCell(cell.dtype, row_map.output_length, cell.backend, state)
        if self.strategy == "eager":
            return self.physical_cell(self.read(cell, row_map, context))
        if self.strategy == "compact":
            while isinstance(state, Select):
                row_map = self.compose(state.row_map, row_map, context)
                cell = state.source
                state = cell.snapshot()
        return ValueCell(
            cell.dtype, row_map.output_length, cell.backend, Select(cell, row_map)
        )

    def constant_storage(self, cell, scalar, row_map):
        n = row_map.output_length
        present = row_map.present if isinstance(row_map, Positions) else None
        if cell.backend == "numpy":
            data = np.full(
                n, 0 if scalar is NA else scalar, dtype=NUMPY_TYPES[cell.dtype]
            )
            valid = np.zeros(n, dtype=bool) if scalar is NA else present
            return Storage(
                cell.dtype,
                "numpy",
                owned(data),
                None if valid is None else owned(valid),
            )
        scalar = pa.scalar(
            None if scalar is NA else scalar, type=ARROW_TYPES[cell.dtype]
        )
        data = pa.repeat(scalar, n)
        if present is not None:
            import pyarrow.compute as pc

            data = pc.if_else(pa.array(present), data, pa.scalar(None, type=data.type))
        return Storage(cell.dtype, "arrow", as_chunks(data))

    def read(self, cell, row_map, context=None):
        context = {} if context is None else context
        while True:
            state = cell.snapshot()
            if isinstance(state, Physical):
                return self.backends[cell.backend].select(state.storage, row_map)
            if isinstance(state, Constant):
                return self.constant_storage(cell, state.scalar, row_map)
            row_map = self.compose(state.row_map, row_map, context)
            cell = state.source

    def ensure_physical(self, cell, context=None):
        with cell.lock:
            if isinstance(cell.state, Physical):
                self.metrics.record("cache_hit")
                return cell.state.storage
            storage = self.read(cell, self.maps.identity(cell.length), context)
            self.backends[cell.backend].wait(storage)
            cell.state = Physical(storage)
            self.metrics.record("materialization", rows=cell.length)
            return storage

    def scalar_at(self, cell, position):
        while True:
            state = cell.snapshot()
            if isinstance(state, Physical):
                return self.backends[cell.backend].scalar_at(state.storage, position)
            if isinstance(state, Constant):
                return state.scalar
            position = self.maps.at(state.row_map, position)
            if position is None:
                return NA
            cell = state.source

    def describe(self, cell):
        depth = 0
        current = cell
        while True:
            state = current.snapshot()
            if not isinstance(state, Select):
                break
            depth += 1
            current = state.source
        return {
            "backend": cell.backend,
            "length": cell.length,
            "selection_depth": depth,
            "leaf": type(state).__name__,
        }

    def memory_report(self, cells):
        buffers, map_buffers = {}, {}
        visited = set()
        stack = list(cells)
        while stack:
            cell = stack.pop()
            if id(cell) in visited:
                continue
            visited.add(id(cell))
            state = cell.snapshot()
            if isinstance(state, Physical):
                buffers.update(self.backends[cell.backend].buffers(state.storage))
            elif isinstance(state, Select):
                stack.append(state.source)
                row_map = state.row_map
                if isinstance(row_map, Positions):
                    for array in (row_map.positions, row_map.present):
                        if array is not None:
                            map_buffers[id(array)] = array.nbytes
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
                if c.dtype.kind != "utf8"
            ),
            "coverage": (
                "known buffer owners only; excludes variable logical widths, "
                "Python metadata, allocator overhead and axes"
            ),
        }
