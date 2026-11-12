"""A deliberately small research API, independent of pandas internals."""

from numbers import Integral

import numpy as np
import pandas as pd

from pdruntime import axes
from pdruntime.backends import ArrowBackend, NumpyBackend, arrow_copy
from pdruntime.execution import Execution
from pdruntime.maps import NumpyMaps, length
from pdruntime.metrics import Metrics
from pdruntime.type_rules import TypeRules
from pdruntime.types import BOOLEAN, INT64, NA, UTF8, normalize_dtype, validate_scalar
from pdruntime.values import Constant, ValueCell


class Runtime:
    def __init__(self, *, strategy="compact", metrics=False):
        self.metrics = Metrics(metrics)
        self.maps = NumpyMaps()
        self.types = TypeRules()
        self.backends = {
            "numpy": NumpyBackend(self.maps, self.metrics),
            "arrow": ArrowBackend(self.maps, self.metrics),
        }
        self.execution = Execution(strategy, self.maps, self.backends, self.metrics)

    def backend(self, name):
        if name not in self.backends:
            raise TypeError(f"unsupported backend: {name}")
        return self.backends[name]

    def array(self, values, *, dtype, backend, validity=None, device=None):
        dtype = normalize_dtype(dtype)
        adapter = self.backend(backend)
        storage = adapter.import_owned(values, dtype, validity, device)
        adapter.wait(storage)
        return Array(self, self.execution.physical_cell(storage))

    def full(self, size, value, *, dtype, backend="numpy"):
        dtype = normalize_dtype(dtype)
        adapter = self.backend(backend)
        adapter.empty(dtype)  # Validate backend/type capability before creating a cell.
        value = validate_scalar(value, dtype)
        return Array(self, ValueCell(dtype, length(size), backend, Constant(value)))

    def frame(self, columns, *, index=None):
        names, cells = [], []
        for name, array in columns.items():
            if not isinstance(name, str):
                raise TypeError("column names must be strings")
            if not isinstance(array, Array) or array.runtime is not self:
                raise TypeError("columns must be Arrays from this runtime")
            names.append(name)
            cells.append(array.cell)
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        size = cells[0].length if cells else (0 if index is None else len(index))
        if any(cell.length != size for cell in cells):
            raise ValueError("column lengths differ")
        return Frame(self, tuple(names), tuple(cells), axes.axis(index, size))

    def convert(self, storage, backend, device=None):
        adapter = self.backend(backend)
        # There is no implicit device fallback. CPU conversion is explicit here.
        values = self.backend(storage.backend).to_arrow(storage)
        result = adapter.import_owned(values, storage.dtype, device=device)
        adapter.wait(result)
        self.metrics.record(
            "backend_conversion", rows=len(storage), bytes=adapter.nbytes(result)
        )
        return result

    def predicate(self, values, size):
        if isinstance(values, Series):
            raise TypeError("filtering is positional; pass an Array or sequence")
        if isinstance(values, Array):
            if values.runtime is not self or values.dtype != BOOLEAN:
                raise TypeError("filter requires Boolean from the same runtime")
            data, valid = values.to_numpy()
        else:
            array = self.array(values, dtype=BOOLEAN, backend="numpy")
            data, valid = array.to_numpy()
        if len(data) != size:
            raise ValueError("filter length mismatch")
        return np.flatnonzero(data if valid is None else data & valid)


class Array:
    def __init__(self, runtime, cell):
        self.runtime = runtime
        self.cell = cell

    def __len__(self):
        return self.cell.length

    @property
    def dtype(self):
        return self.cell.dtype

    def wrap(self, cell, index=None):
        return Array(self.runtime, cell)

    def select_map(self, row_map):
        result = self.runtime.execution.select(self.cell, row_map)
        if isinstance(self, Series):
            indices, present = self.runtime.maps.dense(row_map)
            if present is not None:
                raise ValueError("public label selection requires present positions")
            return self.wrap(result, self._index.take(indices))
        return self.wrap(result)

    def take(self, indices):
        return self.select_map(self.runtime.maps.positions(len(self), indices))

    def slice(self, value):
        return self.select_map(self.runtime.maps.slice(len(self), value))

    def filter(self, values):
        return self.take(self.runtime.predicate(values, len(self)))

    def scalar_at(self, position):
        row_map = self.runtime.maps.positions(len(self), [position])
        return self.runtime.execution.scalar_at(self.cell, int(row_map.positions[0]))

    def materialize(self):
        self.runtime.execution.ensure_physical(self.cell)
        return self

    def compact(self):
        source = self.runtime.execution.ensure_physical(self.cell)
        adapter = self.runtime.backend(source.backend)
        result = adapter.copy_owned(source)
        adapter.wait(result)
        self.runtime.metrics.record("compaction", bytes=adapter.nbytes(result))
        return self.wrap(self.runtime.execution.physical_cell(result))

    def to_backend(self, backend, *, device=None):
        source = self.runtime.execution.ensure_physical(self.cell)
        result = self.runtime.convert(source, backend, device)
        return self.wrap(self.runtime.execution.physical_cell(result))

    def to_list(self):
        values = self.to_arrow().to_pylist()
        return [NA if value is None else value for value in values]

    def to_arrow(self):
        source = self.runtime.execution.ensure_physical(self.cell)
        adapter = self.runtime.backend(source.backend)
        result = arrow_copy(adapter.to_arrow(source))
        self.runtime.metrics.record("export_copy", bytes=result.nbytes)
        return result

    def to_numpy(self):
        source = self.runtime.execution.ensure_physical(self.cell)
        if source.backend == "numpy":
            data = source.data.copy()
            valid = None if source.validity is None else source.validity.copy()
        elif self.dtype == UTF8:
            data = np.array(source.data.fill_null("").to_pylist(), dtype=object)
            valid = source.data.is_valid().to_numpy(zero_copy_only=False).copy()
        else:
            converted = self.runtime.convert(source, "numpy")
            data = converted.data.copy()
            valid = None if converted.validity is None else converted.validity.copy()
        if valid is not None and valid.all():
            valid = None
        self.runtime.metrics.record(
            "export_copy", bytes=data.nbytes + (0 if valid is None else valid.nbytes)
        )
        return data, valid

    def binary(self, operation, other):
        runtime = self.runtime
        left = self.cell
        index = self._index if isinstance(self, Series) else None
        if isinstance(other, Array):
            if other.runtime is not runtime:
                raise TypeError("operands must belong to the same runtime")
            if isinstance(self, Series) != isinstance(other, Series):
                raise TypeError("cannot mix labeled and positional operands")
            signature = runtime.types.bind(operation, (self.dtype, other.dtype))
            right = other.cell
            if isinstance(self, Series):
                index, lm, rm = axes.align(self._index, other._index, runtime.maps)
                context = {}
                left = runtime.execution.select(left, lm, context)
                right = runtime.execution.select(right, rm, context)
            elif len(self) != len(other):
                raise ValueError("operand lengths differ")
        else:
            scalar = validate_scalar(other, self.dtype)
            signature = runtime.types.bind(operation, (self.dtype, self.dtype))
            right = ValueCell(
                self.dtype, len(self), self.cell.backend, Constant(scalar)
            )
        context = {}
        left_storage = runtime.execution.ensure_physical(left, context)
        right_storage = runtime.execution.ensure_physical(right, context)
        backend = left_storage.backend
        if left_storage.backend != right_storage.backend:
            if self.dtype == UTF8:
                raise TypeError("strings require Arrow")
            backend = "numpy"
            if left_storage.backend != backend:
                left_storage = runtime.convert(left_storage, backend)
            if right_storage.backend != backend:
                right_storage = runtime.convert(right_storage, backend)
        adapter = runtime.backend(backend)
        if not adapter.capabilities(signature):
            raise TypeError("backend does not support the bound operation")
        result = adapter.binary(signature, left_storage, right_storage)
        adapter.wait(result)
        if result.dtype != signature.result_type:
            raise TypeError("backend violated the logical result type")
        runtime.metrics.record("kernel", operation=operation, rows=len(result))
        return self.wrap(runtime.execution.physical_cell(result), index)

    def add(self, other):
        return self.binary("add", other)

    def eq(self, other):
        return self.binary("eq", other)

    def lt(self, other):
        return self.binary("lt", other)

    def sum(self, *, skipna=True, min_count=0):
        self.runtime.types.bind("sum", (self.dtype,))
        if not isinstance(skipna, bool):
            raise TypeError("skipna must be boolean")
        min_count = length(min_count)
        source = self.runtime.execution.ensure_physical(self.cell)
        result = self.runtime.backend(source.backend).reduce(
            source, skipna=skipna, min_count=min_count
        )
        self.runtime.metrics.record("kernel", operation="sum", rows=len(self))
        return result

    def set_values(self, rows, values):
        runtime = self.runtime
        positions = runtime.maps.positions(len(self), rows).positions
        scalar = (
            values is NA
            or values is None
            or values is pd.NA
            or isinstance(values, (Integral, str, np.bool_))
        )
        if scalar:
            checked = validate_scalar(values, self.dtype)
            update_cell = ValueCell(
                self.dtype, len(positions), self.cell.backend, Constant(checked)
            )
            update = runtime.execution.ensure_physical(update_cell)
        else:
            if len(np.unique(positions)) != len(positions):
                raise ValueError("vector assignment requires unique destinations")
            update = runtime.backend(self.cell.backend).import_owned(values, self.dtype)
            if len(update) != len(positions):
                raise ValueError("assignment length mismatch")
        source = runtime.execution.ensure_physical(self.cell)
        adapter = runtime.backend(source.backend)
        result = adapter.replace(source, positions, update)
        adapter.wait(result)
        runtime.metrics.record("write_detach", bytes=adapter.nbytes(result))
        self.cell = runtime.execution.physical_cell(result)

    def set_at(self, position, value):
        validate_scalar(value, self.dtype)
        self.set_values([position], value)

    def explain_representation(self):
        return self.runtime.execution.describe(self.cell)

    def memory_report(self):
        return self.runtime.execution.memory_report([self.cell])

    def storage_info(self):
        return self.explain_representation()


class Series(Array):
    def __init__(self, runtime, cell, index, name=None):
        super().__init__(runtime, cell)
        self._index = index
        self.name = name

    @property
    def index(self):
        return self._index.copy(deep=True)

    def wrap(self, cell, index=None):
        return Series(
            self.runtime,
            cell,
            self._index.copy(deep=True) if index is None else index,
            self.name,
        )

    def to_pandas(self):
        if self.dtype == UTF8:
            data = pd.array(self.to_arrow().to_pylist(), dtype="string")
        else:
            values, valid = self.to_numpy()
            mask = np.zeros(len(values), dtype=bool) if valid is None else ~valid
            cls = (
                pd.arrays.IntegerArray
                if self.dtype == INT64
                else pd.arrays.BooleanArray
            )
            data = cls(values, mask)
        return pd.Series(data, index=self.index, name=self.name)


class Frame:
    def __init__(self, runtime, names, cells, index):
        self.runtime, self.names, self.cells, self._index = runtime, names, cells, index

    def __len__(self):
        return len(self._index)

    @property
    def index(self):
        return self._index.copy(deep=True)

    @property
    def columns(self):
        return self.names

    def slot(self, name):
        try:
            return self.names.index(name)
        except ValueError as err:
            raise KeyError(name) from err

    def column(self, name):
        return Series(self.runtime, self.cells[self.slot(name)], self._index, name)

    def project(self, names):
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError("project requires distinct names")
        return Frame(
            self.runtime,
            names,
            tuple(self.cells[self.slot(n)] for n in names),
            self._index,
        )

    def select_map(self, row_map):
        context = {}
        cells = tuple(
            self.runtime.execution.select(c, row_map, context) for c in self.cells
        )
        indices, present = self.runtime.maps.dense(row_map)
        if present is not None:
            raise ValueError("public frame selection requires present positions")
        return Frame(self.runtime, self.names, cells, self._index.take(indices))

    def take(self, indices):
        return self.select_map(self.runtime.maps.positions(len(self), indices))

    def slice(self, value):
        return self.select_map(self.runtime.maps.slice(len(self), value))

    def filter(self, values):
        return self.take(self.runtime.predicate(values, len(self)))

    def replace_column(self, name, array):
        slot = self.slot(name)
        if not isinstance(array, Array) or array.runtime is not self.runtime:
            raise TypeError("replacement must belong to this runtime")
        if len(array) != len(self):
            raise ValueError("replacement length mismatch")
        cells = list(self.cells)
        cells[slot] = array.cell
        self.cells = tuple(cells)

    def set_at(self, *, row, column, value):
        array = self.column(column)
        array.set_at(row, value)
        self.replace_column(column, array)

    def set_values(self, *, rows, column, values):
        array = self.column(column)
        array.set_values(rows, values)
        self.replace_column(column, array)

    def materialize(self):
        context = {}
        for cell in self.cells:
            self.runtime.execution.ensure_physical(cell, context)
        return self

    def compact(self):
        self.materialize()
        cells = tuple(self.column(n).compact().cell for n in self.names)
        return Frame(self.runtime, self.names, cells, self._index.copy(deep=True))

    def to_pandas(self):
        self.materialize()
        return pd.DataFrame(
            {n: self.column(n).to_pandas() for n in self.names}, index=self.index
        )

    def explain_representation(self):
        return {
            name: self.runtime.execution.describe(cell)
            for name, cell in zip(self.names, self.cells, strict=True)
        }

    def memory_report(self):
        result = self.runtime.execution.memory_report(self.cells)
        result["axis_bytes"] = int(self._index.memory_usage(deep=True))
        return result
