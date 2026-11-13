"""CPU adapters. Array API/JAX can be added behind the same storage boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from pdruntime.maps import Affine, Identity, Positions, frozen, owned
from pdruntime.types import (
    BOOLEAN,
    INT64,
    NA,
    UTF8,
    LogicalType,
    Scalar,
    is_scalar,
    validate_scalar,
)

if TYPE_CHECKING:
    from pdruntime.maps import BoolArray, NumpyMaps, RowMap
    from pdruntime.metrics import Metrics
    from pdruntime.type_rules import OperationSignature

# UTF-8 uses 64-bit offsets, as pandas' StringDtype('pyarrow') does.
ARROW_TYPES: Final = {INT64: pa.int64(), BOOLEAN: pa.bool_(), UTF8: pa.large_string()}
NUMPY_TYPES: Final = {INT64: np.dtype("int64"), BOOLEAN: np.dtype("bool")}
FILL_VALUES: Final[dict[LogicalType, Scalar]] = {INT64: 0, BOOLEAN: False, UTF8: ""}

NUMPY_KERNELS: Final[dict[str, Callable[..., np.ndarray]]] = {
    "add": np.add,
    "eq": np.equal,
    "lt": np.less,
}
ARROW_KERNELS: Final[dict[str, Callable[..., pa.ChunkedArray]]] = {
    "add": pc.add,  # wrapping, matching the Int64 contract
    "eq": pc.equal,
    "lt": pc.less,
}

_ARROW_INPUT_TYPES: Final[dict[LogicalType, Callable[[pa.DataType], bool]]] = {
    INT64: pa.types.is_integer,
    BOOLEAN: pa.types.is_boolean,
    UTF8: lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
}
_NUMPY_INPUT_KINDS: Final = {INT64: "iu", BOOLEAN: "b", UTF8: "U"}


@dataclass(frozen=True, eq=False)
class Storage:
    dtype: LogicalType
    backend: str
    data: np.ndarray | pa.ChunkedArray
    validity: BoolArray | None = None
    device: str = "cpu"

    def __len__(self) -> int:
        return len(self.data)


class Backend(Protocol):
    """The private adapter contract shared by every storage backend."""

    name: str
    scalar_types: tuple[type, ...]  # native scalar classes this adapter unboxes

    @staticmethod
    def unbox(value: object) -> Scalar | float: ...

    def supports(self, dtype: LogicalType) -> bool: ...
    def capabilities(
        self, signature: OperationSignature, input_layouts=(), device: str = "cpu"
    ) -> bool: ...
    def import_owned(
        self, values, dtype: LogicalType, validity=None, device: str | None = None
    ) -> Storage: ...
    def empty(self, dtype: LogicalType) -> Storage: ...
    def full(
        self, dtype: LogicalType, scalar: Scalar, size: int, present: BoolArray | None
    ) -> Storage: ...
    def scalar_at(self, storage: Storage, position: int) -> Scalar: ...
    def select(self, storage: Storage, row_map: RowMap) -> Storage: ...
    def concat(self, dtype: LogicalType, parts: Sequence[Storage]) -> Storage: ...
    def key_values(self, storage: Storage) -> tuple[np.ndarray, BoolArray | None]: ...
    def mask(self, storage: Storage) -> BoolArray: ...
    def copy_owned(self, storage: Storage) -> Storage: ...
    def binary(
        self, signature: OperationSignature, left: Storage, right: Storage | Scalar
    ) -> Storage: ...
    def reduce(self, storage: Storage, *, skipna: bool, min_count: int) -> Scalar: ...
    def replace(self, storage: Storage, positions, values: Storage) -> Storage: ...
    def to_arrow(self, storage: Storage) -> pa.ChunkedArray: ...
    def to_numpy(self, storage: Storage) -> tuple[np.ndarray, BoolArray | None]: ...
    def nbytes(self, storage: Storage) -> int: ...
    def buffers(self, storage: Storage) -> dict[tuple[str, int], int]: ...
    def wait(self, storage: Storage) -> None: ...


def as_chunks(array: pa.Array | pa.ChunkedArray) -> pa.ChunkedArray:
    return array if isinstance(array, pa.ChunkedArray) else pa.chunked_array([array])


def arrow_copy(array: pa.Array | pa.ChunkedArray) -> pa.ChunkedArray:
    """IPC is a verifiable detached-copy boundary that runs at memcpy speed."""
    table = pa.table({"values": as_chunks(array)})
    # The result's buffers are slices of the stream buffer, which keeps all of
    # it alive. A growing BufferOutputStream doubles its capacity (40 MB of
    # strings held 64 MB), so measure the stream first and allocate it exactly.
    size = pa.MockOutputStream()
    _write_stream(size, table)
    buffer = pa.allocate_buffer(size.size())
    _write_stream(pa.FixedSizeBufferWriter(buffer), table)
    return pa.ipc.open_stream(buffer).read_all().column(0)


def _write_stream(sink: pa.NativeFile, table: pa.Table) -> None:
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)


def arrow_input(values, dtype: LogicalType, validity=None) -> pa.ChunkedArray:
    """Validate input against the logical type; the result may alias the input."""
    target = ARROW_TYPES[dtype]
    match values:
        case pa.Array() | pa.ChunkedArray():
            if not (
                pa.types.is_null(values.type) or _ARROW_INPUT_TYPES[dtype](values.type)
            ):
                raise TypeError("Arrow input has an incompatible logical type")
            result = pc.cast(values, target, safe=True)
        case np.ndarray():
            if values.ndim != 1:
                raise ValueError("input must be one-dimensional")
            if values.dtype.kind not in _NUMPY_INPUT_KINDS[dtype]:
                raise TypeError("NumPy input has an incompatible logical type")
            result = pa.array(values, type=target, safe=True)
        case _:
            # Python iterable ingress necessarily validates/boxes scalars. Native
            # arrays and all computation/selection paths use bulk kernels.
            checked = (validate_scalar(python_scalar(v), dtype) for v in values)
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


def _validity(valid: BoolArray | None) -> BoolArray | None:
    """Canonical validity: None when every entry is valid."""
    return None if valid is None or valid.all() else frozen(valid)


class NumpyBackend:
    name = "numpy"
    scalar_types = (np.generic,)

    @staticmethod
    def unbox(value: np.generic) -> Scalar | float:
        return value.item()

    def __init__(self, maps: NumpyMaps, metrics: Metrics) -> None:
        self.maps = maps
        self.metrics = metrics

    def supports(self, dtype: LogicalType) -> bool:
        return dtype in NUMPY_TYPES

    def capabilities(self, signature, input_layouts=(), device="cpu") -> bool:
        return device == "cpu" and all(map(self.supports, signature.operand_types))

    def import_owned(self, values, dtype, validity=None, device=None) -> Storage:
        if device not in (None, "cpu") or not self.supports(dtype):
            raise TypeError("NumPy adapter supports CPU Int64 and Boolean only")
        try:
            source = arrow_input(values, dtype, validity)
        except (pa.ArrowException, OverflowError) as err:
            raise TypeError("input does not fit the logical dtype") from err
        valid = source.is_valid().to_numpy(zero_copy_only=False)
        # The filled values may still alias the caller's buffer; owned() copies.
        data = pc.fill_null(source, FILL_VALUES[dtype]).to_numpy(zero_copy_only=False)
        result = Storage(dtype, self.name, owned(data), _validity(valid))
        self.metrics.record("ingress_copy", bytes=self.nbytes(result))
        return result

    def empty(self, dtype) -> Storage:
        if not self.supports(dtype):
            raise TypeError("NumPy adapter supports Int64 and Boolean only")
        return Storage(dtype, self.name, owned([], NUMPY_TYPES[dtype]))

    def full(self, dtype, scalar, size, present) -> Storage:
        fill = FILL_VALUES[dtype] if scalar is NA else scalar
        data = frozen(np.full(size, fill, dtype=NUMPY_TYPES[dtype]))
        valid = np.zeros(size, dtype=bool) if scalar is NA else present
        return Storage(dtype, self.name, data, None if valid is None else frozen(valid))

    def scalar_at(self, storage, position) -> Scalar:
        if storage.validity is not None and not storage.validity[position]:
            return NA
        return storage.data[position].item()

    def select(self, storage, row_map) -> Storage:
        match row_map:
            case Identity():
                return storage
            case _ if row_map.output_length == 0:
                return self.empty(storage.dtype)
            case Affine():
                index = self.maps.slice_object(row_map)
                validity = storage.validity
                return Storage(
                    storage.dtype,
                    self.name,
                    storage.data[index],
                    None if validity is None else validity[index],
                )
            case Positions(positions=indices, present=None):
                # Fancy indexing; np.take is 3-5x slower for int64 gathers.
                data = frozen(storage.data[indices])
                valid = None if storage.validity is None else storage.validity[indices]
            case Positions(positions=indices, present=present):
                # Absent slots must not be dereferenced; they hold no position.
                slots = np.flatnonzero(present)
                sources = indices[slots]
                data = np.zeros(len(indices), dtype=storage.data.dtype)
                data[slots] = storage.data[sources]
                valid = present.copy()
                if storage.validity is not None:
                    valid[slots] = storage.validity[sources]
                data = frozen(data)
        result = Storage(storage.dtype, self.name, data, _validity(valid))
        self.metrics.record("payload_gather", rows=len(data), bytes=self.nbytes(result))
        return result

    def concat(self, dtype, parts) -> Storage:
        if not parts:
            return self.empty(dtype)
        if len(parts) == 1:
            return parts[0]
        data = frozen(np.concatenate([part.data for part in parts]))
        valid = None
        if any(part.validity is not None for part in parts):
            valid = np.concatenate(
                [
                    np.ones(len(part), dtype=bool)
                    if part.validity is None
                    else part.validity
                    for part in parts
                ]
            )
        result = Storage(dtype, self.name, data, _validity(valid))
        self.metrics.record("segment_concat", rows=len(data), bytes=self.nbytes(result))
        return result

    def key_values(self, storage) -> tuple[np.ndarray, BoolArray | None]:
        """Read-only key values; storage is immutable, so no copy is needed."""
        return storage.data, storage.validity

    def mask(self, storage) -> BoolArray:
        if storage.validity is None:
            return storage.data
        return storage.data & storage.validity

    def copy_owned(self, storage) -> Storage:
        validity = storage.validity
        return Storage(
            storage.dtype,
            self.name,
            owned(storage.data),
            None if validity is None else owned(validity),
        )

    def binary(self, signature, left, right) -> Storage:
        kernel = NUMPY_KERNELS[signature.operation]
        result_dtype = NUMPY_TYPES[signature.result_type]
        if isinstance(right, Storage):
            data = kernel(left.data, right.data)
            valid = _and(left.validity, right.validity)
        elif right is NA:
            data = np.zeros(len(left), dtype=result_dtype)
            valid = np.zeros(len(left), dtype=bool)
        else:
            # Typed scalar operands broadcast natively; no constant array is built.
            data = kernel(left.data, NUMPY_TYPES[left.dtype].type(right))
            valid = left.validity
        return Storage(signature.result_type, self.name, frozen(data), _validity(valid))

    def reduce(self, storage, *, skipna, min_count) -> Scalar:
        valid = storage.validity
        count = len(storage) if valid is None else int(valid.sum())
        if count < min_count or (not skipna and count < len(storage)):
            return NA
        values = storage.data if valid is None else storage.data[valid]
        return int(np.sum(values, dtype=np.int64))

    def replace(self, storage, positions, values) -> Storage:
        data = storage.data.copy()
        valid = (
            np.ones(len(storage), dtype=bool)
            if storage.validity is None
            else storage.validity.copy()
        )
        data[positions] = values.data
        valid[positions] = True if values.validity is None else values.validity
        return Storage(storage.dtype, self.name, frozen(data), _validity(valid))

    def to_arrow(self, storage) -> pa.ChunkedArray:
        mask = None if storage.validity is None else ~storage.validity
        array = pa.array(storage.data, mask=mask, type=ARROW_TYPES[storage.dtype])
        return pa.chunked_array([array])

    def to_numpy(self, storage) -> tuple[np.ndarray, BoolArray | None]:
        validity = storage.validity
        return storage.data.copy(), None if validity is None else validity.copy()

    def nbytes(self, storage) -> int:
        validity = storage.validity
        return storage.data.nbytes + (0 if validity is None else validity.nbytes)

    def buffers(self, storage) -> dict[tuple[str, int], int]:
        result = {}
        for array in (storage.data, storage.validity):
            if array is None:
                continue
            while isinstance(array.base, np.ndarray):
                array = array.base
            result["numpy", array.__array_interface__["data"][0]] = array.nbytes
        return result

    def wait(self, storage) -> None:
        return None


def _and(left: BoolArray | None, right: BoolArray | None) -> BoolArray | None:
    if left is None:
        return right
    if right is None:
        return left
    return left & right


class ArrowBackend:
    name = "arrow"
    scalar_types = (pa.Scalar,)

    @staticmethod
    def unbox(value: pa.Scalar) -> Scalar | float:
        return value.as_py() if value.is_valid else NA

    def __init__(self, maps: NumpyMaps, metrics: Metrics) -> None:
        self.maps = maps
        self.metrics = metrics

    def supports(self, dtype: LogicalType) -> bool:
        return dtype in ARROW_TYPES

    def capabilities(self, signature, input_layouts=(), device="cpu") -> bool:
        return device == "cpu" and all(map(self.supports, signature.operand_types))

    def import_owned(self, values, dtype, validity=None, device=None) -> Storage:
        if device not in (None, "cpu"):
            raise TypeError("initial Arrow adapter supports CPU only")
        try:
            data = arrow_copy(arrow_input(values, dtype, validity))
        except (pa.ArrowException, OverflowError) as err:
            raise TypeError("input does not fit the logical dtype") from err
        result = Storage(dtype, self.name, data)
        self.metrics.record("ingress_copy", bytes=self.nbytes(result))
        return result

    def empty(self, dtype) -> Storage:
        return Storage(dtype, self.name, pa.chunked_array([], type=ARROW_TYPES[dtype]))

    def full(self, dtype, scalar, size, present) -> Storage:
        arrow_type = ARROW_TYPES[dtype]
        data = pa.repeat(pa.scalar(None if scalar is NA else scalar, arrow_type), size)
        if present is not None:
            data = pc.if_else(pa.array(present), data, pa.scalar(None, arrow_type))
        return Storage(dtype, self.name, as_chunks(data))

    def scalar_at(self, storage, position) -> Scalar:
        value = storage.data[position]
        return value.as_py() if value.is_valid else NA

    def select(self, storage, row_map) -> Storage:
        match row_map:
            case Identity():
                return storage
            case _ if row_map.output_length == 0:
                return self.empty(storage.dtype)
            case Affine(start=start, step=1, output_length=size):
                return Storage(
                    storage.dtype, self.name, storage.data.slice(start, size)
                )
        indices, present = self.maps.dense(row_map)
        if len(storage) == 0:  # Only all-missing selections reach an empty source.
            result = pa.nulls(len(indices), type=ARROW_TYPES[storage.dtype])
        else:
            mask = None if present is None else ~present
            result = pc.take(storage.data, pa.array(indices, mask=mask))
        output = Storage(storage.dtype, self.name, as_chunks(result))
        self.metrics.record(
            "payload_gather", rows=len(output), bytes=self.nbytes(output)
        )
        return output

    def concat(self, dtype, parts) -> Storage:
        if not parts:
            return self.empty(dtype)
        if len(parts) == 1:
            return parts[0]
        # Chunk layout belongs to storage: concatenation shares the parts' chunks.
        chunks = [chunk for part in parts for chunk in part.data.chunks]
        result = Storage(
            dtype, self.name, pa.chunked_array(chunks, type=ARROW_TYPES[dtype])
        )
        self.metrics.record("segment_concat", rows=len(result), bytes=0)
        return result

    def key_values(self, storage) -> tuple[np.ndarray, BoolArray | None]:
        """Read-only key values: a zero-copy view when one null-free chunk allows."""
        data = storage.data
        if data.num_chunks == 1 and data.null_count == 0:
            return data.chunk(0).to_numpy(), None
        return self.to_numpy(storage)

    def mask(self, storage) -> BoolArray:
        return pc.fill_null(storage.data, False).to_numpy(zero_copy_only=False)

    def copy_owned(self, storage) -> Storage:
        return Storage(storage.dtype, self.name, arrow_copy(storage.data))

    def binary(self, signature, left, right) -> Storage:
        kernel = ARROW_KERNELS[signature.operation]
        if isinstance(right, Storage):
            operand = right.data
        else:
            value = None if right is NA else right
            operand = pa.scalar(value, type=ARROW_TYPES[left.dtype])
        return Storage(
            signature.result_type, self.name, as_chunks(kernel(left.data, operand))
        )

    def reduce(self, storage, *, skipna, min_count) -> Scalar:
        result = pc.sum(storage.data, skip_nulls=skipna, min_count=min_count)
        return result.as_py() if result.is_valid else NA

    def replace(self, storage, positions, values) -> Storage:
        combined = pa.chunked_array(
            [*storage.data.chunks, *values.data.chunks],
            type=ARROW_TYPES[storage.dtype],
        )
        index = np.arange(len(storage), dtype=np.int64)
        index[positions] = len(storage) + np.arange(len(positions))
        # A gather writes every output buffer, so the result is already detached.
        return Storage(
            storage.dtype, self.name, as_chunks(pc.take(combined, pa.array(index)))
        )

    def to_arrow(self, storage) -> pa.ChunkedArray:
        return storage.data

    def to_numpy(self, storage) -> tuple[np.ndarray, BoolArray | None]:
        filled = pc.fill_null(storage.data, FILL_VALUES[storage.dtype])
        values = filled.to_numpy(zero_copy_only=False)
        if storage.dtype != UTF8:
            values = values.copy()  # fixed-width results may view Arrow memory
        valid = storage.data.is_valid().to_numpy(zero_copy_only=False)
        return values, None if valid.all() else valid

    def nbytes(self, storage) -> int:
        return storage.data.nbytes

    def buffers(self, storage) -> dict[tuple[str, int], int]:
        result = {}
        for chunk in storage.data.chunks:
            for buffer in chunk.buffers():
                if buffer is None:
                    continue
                while buffer.parent is not None:
                    buffer = buffer.parent
                result["arrow", buffer.address] = buffer.size
        return result

    def wait(self, storage) -> None:
        return None


ADAPTERS: Final = (NumpyBackend, ArrowBackend)


def python_scalar(value: object) -> object:
    """Unbox a backend-native scalar; other values pass through unchanged.

    The result is then validated against a logical type, so an unboxed value
    of the wrong kind (``np.float64``, ``pa.scalar(True)`` for Int64) still
    raises. Recognition depends only on the value, not on a runtime.
    """
    for adapter in ADAPTERS:
        if isinstance(value, adapter.scalar_types):
            unboxed = adapter.unbox(value)
            # Nested scalars unbox to lists/dicts and must not become vectors.
            if not is_scalar(unboxed):
                raise TypeError(f"{type(value).__name__} is not a supported scalar")
            return unboxed
    return value


def make_backends(maps: NumpyMaps, metrics: Metrics) -> dict[str, Backend]:
    return {adapter.name: adapter(maps, metrics) for adapter in ADAPTERS}
