"""CPU adapters. Array API/JAX can be added behind the same storage boundary."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pdruntime.maps import Affine, Identity, owned
from pdruntime.types import BOOLEAN, INT64, NA, UTF8, validate_scalar

ARROW_TYPES = {INT64: pa.int64(), BOOLEAN: pa.bool_(), UTF8: pa.string()}
NUMPY_TYPES = {INT64: np.dtype("int64"), BOOLEAN: np.dtype("bool")}


@dataclass(frozen=True, eq=False)
class Storage:
    dtype: object
    backend: str
    data: object
    validity: object = None
    device: str = "cpu"

    def __len__(self):
        return len(self.data)


def as_chunks(array):
    return array if isinstance(array, pa.ChunkedArray) else pa.chunked_array([array])


def arrow_copy(array):
    """IPC is a conservative, verifiable detached-copy boundary, not a fast path."""
    table = pa.table({"values": as_chunks(array)})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return pa.ipc.open_stream(sink.getvalue()).read_all().column(0)


def arrow_input(values, dtype, validity=None):
    target = ARROW_TYPES[dtype]
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        actual = values.type
        compatible = (
            pa.types.is_null(actual)
            or dtype == INT64
            and pa.types.is_integer(actual)
            or dtype == BOOLEAN
            and pa.types.is_boolean(actual)
            or dtype == UTF8
            and (pa.types.is_string(actual) or pa.types.is_large_string(actual))
        )
        if not compatible:
            raise TypeError("Arrow input has an incompatible logical type")
        result = pc.cast(values, target, safe=True)
    elif isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError("input must be one-dimensional")
        compatible = (
            dtype == INT64
            and values.dtype.kind in "iu"
            or dtype == BOOLEAN
            and values.dtype.kind == "b"
            or dtype == UTF8
            and values.dtype.kind == "U"
        )
        if not compatible:
            raise TypeError("NumPy input has an incompatible logical type")
        result = pa.array(values, type=target, safe=True)
    else:
        # Python iterable ingress necessarily validates/boxes scalars. Native
        # arrays and all computation/selection paths use bulk kernels.
        checked = [validate_scalar(value, dtype) for value in values]
        result = pa.array(
            [None if value is NA else value for value in checked], type=target
        )
    if validity is not None:
        mask = np.asarray(validity)
        if mask.dtype.kind != "b":
            raise TypeError("validity must be boolean")
        if mask.shape != (len(result),):
            raise ValueError("validity length mismatch")
        result = pc.if_else(pa.array(mask), result, pa.scalar(None, type=target))
    return as_chunks(result)


class NumpyBackend:
    name = "numpy"

    def __init__(self, maps, metrics):
        self.maps = maps
        self.metrics = metrics

    def capabilities(self, signature, input_layouts=(), device="cpu"):
        return device == "cpu" and all(
            t in NUMPY_TYPES for t in signature.operand_types
        )

    def import_owned(self, values, dtype, validity=None, device=None):
        if device not in (None, "cpu") or dtype not in NUMPY_TYPES:
            raise TypeError("NumPy adapter supports CPU Int64 and Boolean only")
        try:
            source = arrow_input(values, dtype, validity)
        except (pa.ArrowException, OverflowError) as err:
            raise TypeError("input does not fit the logical dtype") from err
        valid = source.is_valid().to_numpy(zero_copy_only=False)
        data = pc.fill_null(source, False if dtype == BOOLEAN else 0).to_numpy(
            zero_copy_only=False
        )
        result = Storage(
            dtype, self.name, owned(data), None if valid.all() else owned(valid)
        )
        self.metrics.record("ingress_copy", bytes=self.nbytes(result))
        return result

    def empty(self, dtype):
        if dtype not in NUMPY_TYPES:
            raise TypeError("NumPy adapter supports Int64 and Boolean only")
        return Storage(dtype, self.name, owned([], NUMPY_TYPES[dtype]))

    def scalar_at(self, storage, position):
        if storage.validity is not None and not storage.validity[position]:
            return NA
        return storage.data[position].item()

    def select(self, storage, row_map):
        if isinstance(row_map, Identity):
            return storage
        if row_map.output_length == 0:
            return self.empty(storage.dtype)
        if isinstance(row_map, Affine):
            index = self.maps.slice_object(row_map)
            return Storage(
                storage.dtype,
                self.name,
                storage.data[index],
                None if storage.validity is None else storage.validity[index],
            )
        indices, present = self.maps.dense(row_map)
        valid = np.ones(len(indices), dtype=bool) if present is None else present.copy()
        data = np.zeros(len(indices), dtype=storage.data.dtype)
        data[valid] = storage.data[indices[valid]]
        if storage.validity is not None:
            valid[valid] &= storage.validity[indices[valid]]
        data.flags.writeable = False
        result = Storage(
            storage.dtype, self.name, data, None if valid.all() else owned(valid)
        )
        self.metrics.record("payload_gather", rows=len(data), bytes=self.nbytes(result))
        return result

    def copy_owned(self, storage):
        return Storage(
            storage.dtype,
            self.name,
            owned(storage.data),
            None if storage.validity is None else owned(storage.validity),
        )

    def binary(self, signature, left, right):
        if not self.capabilities(signature):
            raise TypeError("unsupported NumPy kernel")
        operation = {"add": np.add, "eq": np.equal, "lt": np.less}[signature.operation]
        data = operation(left.data, right.data)
        valid = np.ones(len(data), dtype=bool)
        for operand in (left, right):
            if operand.validity is not None:
                valid &= operand.validity
        data.flags.writeable = False
        return Storage(
            signature.result_type,
            self.name,
            data,
            None if valid.all() else owned(valid),
        )

    def reduce(self, storage, *, skipna, min_count):
        valid = storage.validity
        count = len(storage) if valid is None else int(valid.sum())
        if count < min_count or not skipna and count < len(storage):
            return NA
        values = storage.data if valid is None else storage.data[valid]
        return int(np.sum(values, dtype=np.int64))

    def replace(self, storage, positions, values):
        data = storage.data.copy()
        valid = (
            np.ones(len(storage), dtype=bool)
            if storage.validity is None
            else storage.validity.copy()
        )
        data[positions] = values.data
        valid[positions] = True if values.validity is None else values.validity
        data.flags.writeable = False
        return Storage(
            storage.dtype, self.name, data, None if valid.all() else owned(valid)
        )

    def to_arrow(self, storage):
        return pa.chunked_array(
            [
                pa.array(
                    storage.data,
                    mask=None if storage.validity is None else ~storage.validity,
                    type=ARROW_TYPES[storage.dtype],
                )
            ]
        )

    def nbytes(self, storage):
        return storage.data.nbytes + (
            0 if storage.validity is None else storage.validity.nbytes
        )

    def buffers(self, storage):
        result = {}
        for array in (storage.data, storage.validity):
            if array is None:
                continue
            while isinstance(array.base, np.ndarray):
                array = array.base
            result[("numpy", array.__array_interface__["data"][0])] = array.nbytes
        return result

    def wait(self, storage):
        return None


class ArrowBackend:
    name = "arrow"

    def __init__(self, maps, metrics):
        self.maps = maps
        self.metrics = metrics

    def capabilities(self, signature, input_layouts=(), device="cpu"):
        return device == "cpu" and all(
            t in ARROW_TYPES for t in signature.operand_types
        )

    def import_owned(self, values, dtype, validity=None, device=None):
        if device not in (None, "cpu"):
            raise TypeError("initial Arrow adapter supports CPU only")
        try:
            result = Storage(
                dtype, self.name, arrow_copy(arrow_input(values, dtype, validity))
            )
        except (pa.ArrowException, OverflowError) as err:
            raise TypeError("input does not fit the logical dtype") from err
        self.metrics.record("ingress_copy", bytes=self.nbytes(result))
        return result

    def empty(self, dtype):
        return Storage(dtype, self.name, pa.chunked_array([], type=ARROW_TYPES[dtype]))

    def scalar_at(self, storage, position):
        value = storage.data[position]
        return value.as_py() if value.is_valid else NA

    def select(self, storage, row_map):
        if isinstance(row_map, Identity):
            return storage
        if row_map.output_length == 0:
            return self.empty(storage.dtype)
        if isinstance(row_map, Affine) and row_map.step == 1:
            return Storage(
                storage.dtype,
                self.name,
                storage.data.slice(row_map.start, row_map.output_length),
            )
        indices, present = self.maps.dense(row_map)
        if len(storage) == 0:
            result = pa.nulls(len(indices), type=ARROW_TYPES[storage.dtype])
        else:
            index = pa.array(indices, mask=None if present is None else ~present)
            result = pc.take(storage.data, index)
        output = Storage(storage.dtype, self.name, as_chunks(result))
        self.metrics.record(
            "payload_gather", rows=len(output), bytes=self.nbytes(output)
        )
        return output

    def copy_owned(self, storage):
        return Storage(storage.dtype, self.name, arrow_copy(storage.data))

    def binary(self, signature, left, right):
        operation = {"add": pc.add, "eq": pc.equal, "lt": pc.less}[signature.operation]
        return Storage(
            signature.result_type,
            self.name,
            as_chunks(operation(left.data, right.data)),
        )

    def reduce(self, storage, *, skipna, min_count):
        result = pc.sum(storage.data, skip_nulls=skipna, min_count=min_count)
        return result.as_py() if result.is_valid else NA

    def replace(self, storage, positions, values):
        combined = pa.chunked_array(
            list(storage.data.chunks) + list(values.data.chunks),
            type=ARROW_TYPES[storage.dtype],
        )
        index = np.arange(len(storage), dtype=np.int64)
        index[positions] = len(storage) + np.arange(len(positions))
        result = Storage(
            storage.dtype, self.name, as_chunks(pc.take(combined, pa.array(index)))
        )
        return self.copy_owned(result)

    def to_arrow(self, storage):
        return storage.data

    def nbytes(self, storage):
        return storage.data.nbytes

    def buffers(self, storage):
        result = {}
        for chunk in storage.data.chunks:
            for buffer in chunk.buffers():
                if buffer is None:
                    continue
                while buffer.parent is not None:
                    buffer = buffer.parent
                result[("arrow", buffer.address)] = buffer.size
        return result

    def wait(self, storage):
        return None
