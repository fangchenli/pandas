"""A deliberately small research API, independent of pandas internals.

Handles (Array, Series, Frame) are mutable references to immutable value cells.
Methods prefixed with an underscore, cells, and backend objects are internal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pdruntime import axes
from pdruntime.backends import Storage, arrow_copy, make_backends, python_scalar
from pdruntime.execution import Execution, Strategy
from pdruntime.maps import NumpyMaps, check_length
from pdruntime.metrics import Metrics
from pdruntime.type_rules import TypeRules
from pdruntime.types import (
    BOOLEAN,
    INT64,
    NA,
    UTF8,
    LogicalType,
    Scalar,
    is_scalar,
    normalize_dtype,
    validate_scalar,
)
from pdruntime.values import Constant, ValueCell

if TYPE_CHECKING:
    import pyarrow as pa

    from pdruntime.backends import Backend
    from pdruntime.maps import BoolArray, RowMap


class Runtime:
    def __init__(self, *, strategy: Strategy = "compact", metrics: bool = False):
        self.metrics = Metrics(metrics)
        self.maps = NumpyMaps()
        self.types = TypeRules()
        self.backends = make_backends(self.maps, self.metrics)
        self.execution = Execution(strategy, self.maps, self.backends, self.metrics)

    def __repr__(self) -> str:
        return f"Runtime(strategy={self.execution.strategy!r})"

    def backend(self, name: str) -> Backend:
        try:
            return self.backends[name]
        except KeyError:
            raise TypeError(f"unsupported backend: {name}") from None

    def array(
        self,
        values: Iterable[object],
        *,
        dtype: LogicalType | str,
        backend: str,
        validity: Sequence[bool] | BoolArray | None = None,
        device: str | None = None,
    ) -> Array:
        dtype = normalize_dtype(dtype)
        adapter = self.backend(backend)
        storage = adapter.import_owned(values, dtype, validity, device)
        adapter.wait(storage)
        return Array(self, self.execution.physical_cell(storage))

    def full(
        self, size: int, value: object, *, dtype: LogicalType | str, backend="numpy"
    ) -> Array:
        dtype = normalize_dtype(dtype)
        if not self.backend(backend).supports(dtype):
            raise TypeError(f"{backend} does not support {dtype}")
        scalar = validate_scalar(python_scalar(value), dtype)
        cell = ValueCell(dtype, check_length(size), backend, Constant(scalar))
        return Array(self, cell)

    def frame(
        self, columns: Mapping[str, Array], *, index: Iterable[object] | None = None
    ) -> Frame:
        for name, array in columns.items():
            if not isinstance(name, str):
                raise TypeError("column names must be strings")
            self._check_owned(array, "columns must be Arrays from this runtime")
        names = tuple(columns)  # Mapping keys are unique.
        cells = tuple(array.cell for array in columns.values())
        if cells:
            size = cells[0].length
        else:
            index = None if index is None else list(index)
            size = 0 if index is None else len(index)
        if any(cell.length != size for cell in cells):
            raise ValueError("column lengths differ")
        return Frame(self, names, cells, axes.axis(index, size))

    def _check_owned(self, value: object, message: str) -> None:
        if not isinstance(value, Array) or value.runtime is not self:
            raise TypeError(message)

    def _convert(
        self, storage: Storage, backend: str, device: str | None = None
    ) -> Storage:
        adapter = self.backend(backend)
        # There is no implicit device fallback. CPU conversion is explicit here.
        values = self.backend(storage.backend).to_arrow(storage)
        result = adapter.import_owned(values, storage.dtype, device=device)
        adapter.wait(result)
        self.metrics.record(
            "backend_conversion", rows=len(storage), bytes=adapter.nbytes(result)
        )
        return result

    def _true_positions(self, values: object, size: int) -> np.ndarray:
        """Positions where a Boolean predicate is true; missing counts as false."""
        if isinstance(values, Series):
            raise TypeError("filtering is positional; pass an Array or sequence")
        if isinstance(values, Array):
            if values.runtime is not self or values.dtype != BOOLEAN:
                raise TypeError("filter requires Boolean from the same runtime")
            predicate = values
        else:
            predicate = self.array(values, dtype=BOOLEAN, backend="numpy")
        if len(predicate) != size:
            raise ValueError("filter length mismatch")
        storage = self.execution.ensure_physical(predicate.cell)
        return np.flatnonzero(self.backend(storage.backend).mask(storage))


class Array:
    """An unlabeled, positionally indexed logical array."""

    def __init__(self, runtime: Runtime, cell: ValueCell):
        self.runtime = runtime
        self.cell = cell

    def __len__(self) -> int:
        return self.cell.length

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dtype={self.dtype}, length={len(self)}, "
            f"backend={self.cell.backend})"
        )

    @property
    def dtype(self) -> LogicalType:
        return self.cell.dtype

    def _wrap(self, cell: ValueCell) -> Array:
        """A handle of the same kind, with the same labels, around ``cell``."""
        return Array(self.runtime, cell)

    def _select_map(self, row_map: RowMap) -> Array:
        return self._wrap(self.runtime.execution.select(self.cell, row_map))

    def _physical(self) -> tuple[Storage, Backend]:
        storage = self.runtime.execution.ensure_physical(self.cell)
        return storage, self.runtime.backend(storage.backend)

    # Structural selection

    def take(self, indices: Sequence[int] | np.ndarray) -> Array:
        return self._select_map(self.runtime.maps.positions(len(self), indices))

    def slice(self, value: slice) -> Array:
        return self._select_map(self.runtime.maps.slice(len(self), value))

    def filter(self, values: object) -> Array:
        positions = self.runtime._true_positions(values, len(self))
        return self._select_map(
            self.runtime.maps.positions(len(self), positions, negative=False)
        )

    def scalar_at(self, position: int) -> Scalar:
        row_map = self.runtime.maps.positions(len(self), [position])
        return self.runtime.execution.scalar_at(self.cell, int(row_map.positions[0]))

    # Representation and export

    def materialize(self) -> Array:
        self.runtime.execution.ensure_physical(self.cell)
        return self

    def compact(self) -> Array:
        source, adapter = self._physical()
        result = adapter.copy_owned(source)
        adapter.wait(result)
        self.runtime.metrics.record("compaction", bytes=adapter.nbytes(result))
        return self._wrap(self.runtime.execution.physical_cell(result))

    def to_backend(self, backend: str, *, device: str | None = None) -> Array:
        source, _ = self._physical()
        result = self.runtime._convert(source, backend, device)
        return self._wrap(self.runtime.execution.physical_cell(result))

    def to_list(self) -> list[Scalar]:
        # Boxing produces fresh Python objects, so no detached copy is needed.
        source, adapter = self._physical()
        values = adapter.to_arrow(source).to_pylist()
        return [NA if value is None else value for value in values]

    def to_arrow(self) -> pa.ChunkedArray:
        source, adapter = self._physical()
        result = arrow_copy(adapter.to_arrow(source))
        self.runtime.metrics.record("export_copy", bytes=result.nbytes)
        return result

    def to_numpy(self) -> tuple[np.ndarray, BoolArray | None]:
        """Detached ``(values, validity)``; missing slots hold a fill value."""
        source, adapter = self._physical()
        data, valid = adapter.to_numpy(source)
        self.runtime.metrics.record(
            "export_copy", bytes=data.nbytes + (0 if valid is None else valid.nbytes)
        )
        return data, valid

    def _to_pandas_array(self) -> pd.api.extensions.ExtensionArray:
        if self.dtype == UTF8:
            return pd.arrays.ArrowStringArray(self.to_arrow())
        values, valid = self.to_numpy()
        mask = np.zeros(len(values), dtype=bool) if valid is None else ~valid
        if self.dtype == INT64:
            return pd.arrays.IntegerArray(values, mask)
        return pd.arrays.BooleanArray(values, mask)

    # Computation

    def _binary(self, operation: str, other: object) -> Array:
        runtime = self.runtime
        execution = runtime.execution
        cache = {}
        if isinstance(other, Array):
            if other.runtime is not runtime:
                raise TypeError("operands must belong to the same runtime")
            if type(other) is not type(self):
                raise TypeError("cannot mix labeled and positional operands")
            if len(other) != len(self):
                raise ValueError("operand lengths differ")
            signature = runtime.types.bind(operation, (self.dtype, other.dtype))
            right = execution.ensure_physical(other.cell, cache)
        else:
            signature = runtime.types.bind(operation, (self.dtype, self.dtype))
            right = validate_scalar(python_scalar(other), self.dtype)
        left = execution.ensure_physical(self.cell, cache)
        backend = left.backend
        if isinstance(right, Storage) and right.backend != backend:
            if self.dtype == UTF8:
                raise TypeError("strings require Arrow")
            backend = "numpy"  # The initial mixed CPU numeric policy.
            if left.backend != backend:
                left = runtime._convert(left, backend)
            if right.backend != backend:
                right = runtime._convert(right, backend)
        adapter = runtime.backend(backend)
        if not adapter.capabilities(signature):
            raise TypeError("backend does not support the bound operation")
        result = adapter.binary(signature, left, right)
        adapter.wait(result)
        if result.dtype != signature.result_type:
            raise TypeError("backend violated the logical result type")
        runtime.metrics.record("kernel", operation=operation, rows=len(result))
        return self._wrap(execution.physical_cell(result))

    def add(self, other: object) -> Array:
        return self._binary("add", other)

    def eq(self, other: object) -> Array:
        return self._binary("eq", other)

    def lt(self, other: object) -> Array:
        return self._binary("lt", other)

    def sum(self, *, skipna: bool = True, min_count: int = 0) -> Scalar:
        self.runtime.types.bind("sum", (self.dtype,))
        if not isinstance(skipna, bool):
            raise TypeError("skipna must be boolean")
        min_count = check_length(min_count)
        source, adapter = self._physical()
        result = adapter.reduce(source, skipna=skipna, min_count=min_count)
        self.runtime.metrics.record("kernel", operation="sum", rows=len(self))
        return result

    # Mutation: each write detaches the column and replaces this handle's cell.

    def set_values(self, rows: Sequence[int] | np.ndarray, values: object) -> None:
        runtime = self.runtime
        adapter = runtime.backend(self.cell.backend)
        # Validate everything before materializing or changing the target.
        positions = runtime.maps.positions(len(self), rows).positions
        values = python_scalar(values)
        if is_scalar(values):
            scalar = validate_scalar(values, self.dtype)
            update = adapter.full(self.dtype, scalar, len(positions), None)
        elif not isinstance(values, Iterable):
            raise TypeError(f"value {values!r} is neither a scalar nor a sequence")
        else:
            if len(np.unique(positions)) != len(positions):
                raise ValueError("vector assignment requires unique destinations")
            update = adapter.import_owned(values, self.dtype)
            if len(update) != len(positions):
                raise ValueError("assignment length mismatch")
        source = runtime.execution.ensure_physical(self.cell)
        result = adapter.replace(source, positions, update)
        adapter.wait(result)
        runtime.metrics.record("write_detach", bytes=adapter.nbytes(result))
        self.cell = runtime.execution.physical_cell(result)

    def set_at(self, position: int, value: object) -> None:
        value = python_scalar(value)
        if not is_scalar(value):
            raise TypeError("set_at requires a scalar value")
        self.set_values([position], value)

    # Diagnostics

    def explain_representation(self) -> dict[str, object]:
        return self.runtime.execution.describe(self.cell)

    storage_info = explain_representation  # the name used in logical_types.md

    def memory_report(self) -> dict[str, object]:
        return self.runtime.execution.memory_report([self.cell])


class Series(Array):
    """A labeled array; binary operations align on labels first."""

    def __init__(
        self, runtime: Runtime, cell: ValueCell, index: pd.Index, name: str | None
    ):
        super().__init__(runtime, cell)
        self._index = index
        self.name = name

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]}, name={self.name!r})"

    @property
    def index(self) -> pd.Index:
        return self._index.copy(deep=True)

    def _wrap(self, cell: ValueCell) -> Series:
        return Series(self.runtime, cell, self._index, self.name)

    def _select_map(self, row_map: RowMap) -> Series:
        index = axes.take(self._index, row_map, self.runtime.maps)
        cell = self.runtime.execution.select(self.cell, row_map)
        return Series(self.runtime, cell, index, self.name)

    def _binary(self, operation: str, other: object) -> Series:
        if isinstance(other, Series) and other.runtime is self.runtime:
            # Align labels first, then compute positionally on aligned handles.
            index, left_map, right_map = axes.align(
                self._index, other._index, self.runtime.maps
            )
            execution, cache = self.runtime.execution, {}
            left_cell = execution.select(self.cell, left_map, cache)
            right_cell = execution.select(other.cell, right_map, cache)
            left = Series(self.runtime, left_cell, index, self.name)
            right = Series(other.runtime, right_cell, index, other.name)
            return Array._binary(left, operation, right)
        return super()._binary(operation, other)

    def to_pandas(self) -> pd.Series:
        return pd.Series(self._to_pandas_array(), index=self.index, name=self.name)


class Frame:
    """Ordered axes plus ordered column slots, each referencing a value cell."""

    def __init__(
        self,
        runtime: Runtime,
        names: tuple[str, ...],
        cells: tuple[ValueCell, ...],
        index: pd.Index,
    ):
        self.runtime = runtime
        self.names = names
        self.cells = cells
        self._index = index

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"Frame(rows={len(self)}, columns={list(self.names)})"

    @property
    def index(self) -> pd.Index:
        return self._index.copy(deep=True)

    @property
    def columns(self) -> tuple[str, ...]:
        return self.names

    def _slot(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(name) from None

    def _set_cell(self, slot: int, cell: ValueCell) -> None:
        cells = list(self.cells)
        cells[slot] = cell
        self.cells = tuple(cells)

    def column(self, name: str) -> Series:
        return Series(self.runtime, self.cells[self._slot(name)], self._index, name)

    def project(self, names: Iterable[str]) -> Frame:
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError("project requires distinct names")
        cells = tuple(self.cells[self._slot(n)] for n in names)
        return Frame(self.runtime, names, cells, self._index)

    # Structural selection: one shared map and composition cache for all columns.

    def _select_map(self, row_map: RowMap) -> Frame:
        index = axes.take(self._index, row_map, self.runtime.maps)
        cache = {}
        cells = tuple(
            self.runtime.execution.select(cell, row_map, cache) for cell in self.cells
        )
        return Frame(self.runtime, self.names, cells, index)

    def take(self, indices: Sequence[int] | np.ndarray) -> Frame:
        return self._select_map(self.runtime.maps.positions(len(self), indices))

    def slice(self, value: slice) -> Frame:
        return self._select_map(self.runtime.maps.slice(len(self), value))

    def filter(self, values: object) -> Frame:
        positions = self.runtime._true_positions(values, len(self))
        return self._select_map(
            self.runtime.maps.positions(len(self), positions, negative=False)
        )

    # Mutation: only this frame's slot changes.

    def replace_column(self, name: str, array: Array) -> None:
        """Install ``array`` as a column. A Series aligns on labels, as in pandas."""
        slot = self._slot(name)
        self.runtime._check_owned(array, "replacement must belong to this runtime")
        if isinstance(array, Series):
            row_map = axes.reindexer(array._index, self._index, self.runtime.maps)
            cell = self.runtime.execution.select(array.cell, row_map)
        elif len(array) != len(self):
            raise ValueError("replacement length mismatch")
        else:
            cell = array.cell
        self._set_cell(slot, cell)

    def set_at(self, *, row: int, column: str, value: object) -> None:
        series = self.column(column)
        series.set_at(row, value)
        self._set_cell(self._slot(column), series.cell)

    def set_values(
        self, *, rows: Sequence[int] | np.ndarray, column: str, values: object
    ) -> None:
        series = self.column(column)
        series.set_values(rows, values)
        self._set_cell(self._slot(column), series.cell)

    # Representation and export

    def materialize(self) -> Frame:
        cache = {}
        for cell in self.cells:
            self.runtime.execution.ensure_physical(cell, cache)
        return self

    def compact(self) -> Frame:
        self.materialize()
        cells = tuple(self.column(n).compact().cell for n in self.names)
        return Frame(self.runtime, self.names, cells, self._index.copy(deep=True))

    def to_pandas(self) -> pd.DataFrame:
        self.materialize()
        arrays = {n: self.column(n)._to_pandas_array() for n in self.names}
        return pd.DataFrame(arrays, index=self.index)

    def explain_representation(self) -> dict[str, dict[str, object]]:
        return {
            name: self.runtime.execution.describe(cell)
            for name, cell in zip(self.names, self.cells, strict=True)
        }

    def memory_report(self) -> dict[str, object]:
        result = self.runtime.execution.memory_report(self.cells)
        result["axis_bytes"] = int(self._index.memory_usage(deep=True))
        return result
