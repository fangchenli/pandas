"""Logical identity and scalar validation, independent of physical storage."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Number
from typing import Final, TypeAlias

import numpy as np
import pandas as pd


class Missing:
    """The missing-value singleton. Its truth value is ambiguous."""

    __slots__ = ()
    _instance: Missing | None = None

    def __new__(cls) -> Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NA"

    def __bool__(self) -> bool:
        raise TypeError("the truth value of NA is ambiguous")

    def __reduce__(self) -> str:
        # Copies and unpickled values resolve to the module-level singleton.
        return "NA"


NA: Final = Missing()

Scalar: TypeAlias = int | bool | str | Missing


@dataclass(frozen=True)
class LogicalType:
    kind: str
    parameters: tuple[int, ...] = ()
    missing_policy: str = "NA"


INT64: Final = LogicalType("signed_integer", (64,))
BOOLEAN: Final = LogicalType("boolean")
UTF8: Final = LogicalType("utf8")
ALIASES: Final[dict[str, LogicalType]] = {
    "int64": INT64,
    "boolean": BOOLEAN,
    "utf8": UTF8,
}
_CANONICAL: Final = {dtype: dtype for dtype in ALIASES.values()}
_INT64_RANGE: Final = range(-(2**63), 2**63)


def normalize_dtype(spec: str | LogicalType) -> LogicalType:
    match spec:
        case str() if spec in ALIASES:
            return ALIASES[spec]
        case LogicalType() if spec in _CANONICAL:
            return _CANONICAL[spec]
    raise TypeError(f"unsupported logical dtype: {spec!r}")


def is_integer(value: object) -> bool:
    """True for integral values other than booleans."""
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def is_missing(value: object) -> bool:
    return value is NA or value is None or value is pd.NA


def is_scalar(value: object) -> bool:
    return is_missing(value) or isinstance(value, (Number, str, np.bool_))


def validate_scalar(value: object, dtype: LogicalType) -> Scalar:
    if is_missing(value):
        return NA
    if dtype == INT64 and is_integer(value):
        if int(value) not in _INT64_RANGE:
            raise TypeError("integer value is outside int64 range")
        return int(value)
    if dtype == BOOLEAN and isinstance(value, (bool, np.bool_)):
        return bool(value)
    if dtype == UTF8 and isinstance(value, str):
        return str(value)
    raise TypeError(f"value {value!r} does not belong to {dtype.kind}")


# Constructor functions follow the pyarrow convention (``pa.int64()``).
def int64() -> LogicalType:
    return INT64


def boolean() -> LogicalType:
    return BOOLEAN


def utf8() -> LogicalType:
    return UTF8
