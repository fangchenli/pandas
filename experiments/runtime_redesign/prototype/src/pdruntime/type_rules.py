"""Bind logical results before selecting a backend kernel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from pdruntime.types import BOOLEAN, INT64, UTF8, LogicalType, normalize_dtype


@dataclass(frozen=True)
class OperationSignature:
    operation: str
    operand_types: tuple[LogicalType, ...]
    result_type: LogicalType


_RESULT_TYPES: Final[dict[tuple[str, tuple[LogicalType, ...]], LogicalType]] = {
    ("add", (INT64, INT64)): INT64,
    ("lt", (INT64, INT64)): BOOLEAN,
    ("sum", (INT64,)): INT64,
    **{("eq", (dtype, dtype)): BOOLEAN for dtype in (INT64, BOOLEAN, UTF8)},
}


class TypeRules:
    """A closed registry of operation signatures for the initial type set."""

    def __init__(self) -> None:
        self._result_types = dict(_RESULT_TYPES)

    def bind(
        self, operation: str, operand_types: Iterable[LogicalType | str]
    ) -> OperationSignature:
        types = tuple(normalize_dtype(t) for t in operand_types)
        try:
            result_type = self._result_types[operation, types]
        except KeyError:
            raise TypeError(
                f"unsupported operation/type combination: {operation}, {types}"
            ) from None
        return OperationSignature(operation, types, result_type)

    def common_type(self, types: Iterable[LogicalType | str]) -> LogicalType:
        normalized = {normalize_dtype(t) for t in types}
        if len(normalized) != 1:
            raise TypeError("only identical logical types have an initial common type")
        return normalized.pop()

    def resolve_cast(
        self, source: LogicalType | str, target: LogicalType | str
    ) -> LogicalType:
        source, target = normalize_dtype(source), normalize_dtype(target)
        if source != target:
            raise TypeError("only identity logical casts are supported")
        return target
