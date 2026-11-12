"""Bind logical results before selecting a backend kernel."""

from dataclasses import dataclass

from pdruntime.types import BOOLEAN, INT64, normalize_dtype


@dataclass(frozen=True)
class OperationSignature:
    operation: str
    operand_types: tuple
    result_type: object


class TypeRules:
    def bind(self, operation, operand_types):
        types = tuple(normalize_dtype(t) for t in operand_types)
        if operation == "sum" and types == (INT64,):
            return OperationSignature(operation, types, INT64)
        if len(types) == 2 and types[0] == types[1]:
            if operation == "eq":
                return OperationSignature(operation, types, BOOLEAN)
            if types[0] == INT64 and operation in ("add", "lt"):
                return OperationSignature(
                    operation, types, INT64 if operation == "add" else BOOLEAN
                )
        raise TypeError(f"unsupported operation/type combination: {operation}, {types}")

    def common_type(self, types):
        normalized = tuple(normalize_dtype(t) for t in types)
        if normalized and all(t == normalized[0] for t in normalized):
            return normalized[0]
        raise TypeError("only identical logical types have an initial common type")

    def resolve_cast(self, source, target):
        source, target = normalize_dtype(source), normalize_dtype(target)
        if source != target:
            raise TypeError("only identity logical casts are supported")
        return target
