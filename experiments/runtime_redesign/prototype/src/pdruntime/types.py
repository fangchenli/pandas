"""Logical identity and scalar validation, independent of physical storage.

This module imports no backend. Adapters unbox their native scalars (NumPy,
Arrow, ...) into the canonical Python domain before validation here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from numbers import Integral, Number
from typing import ClassVar, Final, TypeAlias


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


class TypeKind(StrEnum):
    """Stable type-family identifiers; serialization uses these, not class names."""

    SIGNED_INTEGER = "signed_integer"
    BOOLEAN = "boolean"
    UTF8 = "utf8"


class MissingPolicy(StrEnum):
    NA = "NA"


@dataclass(frozen=True, kw_only=True)
class LogicalType:
    """Base for logical types. Each family declares its semantic parameters as
    named fields; equality and hashing use the family, those fields, and the
    missing policy. Storage backend and encoding are never part of identity.
    """

    kind: ClassVar[TypeKind]
    missing_policy: MissingPolicy = field(default=MissingPolicy.NA, repr=False)

    def __post_init__(self) -> None:
        if not hasattr(type(self), "kind"):
            raise TypeError("LogicalType is abstract; use a concrete type family")

    @property
    def name(self) -> str:
        return self.kind.value

    @property
    def key(self) -> tuple[str, tuple[tuple[str, object], ...], str]:
        """Canonical structural key, independent of the Python class."""
        parameters = tuple(
            (f.name, getattr(self, f.name))
            for f in fields(self)
            if f.name != "missing_policy"
        )
        return self.kind.value, parameters, self.missing_policy.value

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, kw_only=True)
class SignedInteger(LogicalType):
    kind = TypeKind.SIGNED_INTEGER
    bits: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bits != 64:
            # Do not create a type whose operations would all fail later.
            raise TypeError(f"unsupported integer width: {self.bits}")

    @property
    def name(self) -> str:
        return f"int{self.bits}"

    @property
    def value_range(self) -> range:
        return range(-(2 ** (self.bits - 1)), 2 ** (self.bits - 1))


@dataclass(frozen=True, kw_only=True)
class Boolean(LogicalType):
    kind = TypeKind.BOOLEAN


@dataclass(frozen=True, kw_only=True)
class Utf8(LogicalType):
    kind = TypeKind.UTF8


INT64: Final = SignedInteger(bits=64)
BOOLEAN: Final = Boolean()
UTF8: Final = Utf8()
ALIASES: Final[dict[str, LogicalType]] = {
    "int64": INT64,
    "boolean": BOOLEAN,
    "utf8": UTF8,
}
_CANONICAL: Final = {dtype: dtype for dtype in ALIASES.values()}


def normalize_dtype(spec: str | LogicalType) -> LogicalType:
    match spec:
        case str() if spec in ALIASES:
            return ALIASES[spec]
        case LogicalType() if spec in _CANONICAL:
            return _CANONICAL[spec]
    raise TypeError(f"unsupported logical dtype: {spec!r}")


def is_integer(value: object) -> bool:
    """True for integral values other than booleans.

    ``Integral`` also admits NumPy integers, which register themselves with it.
    """
    return isinstance(value, Integral) and not isinstance(value, bool)


def is_missing(value: object) -> bool:
    """Only the runtime's NA and None. pd.NA belongs to the pandas boundary."""
    return value is NA or value is None


def is_scalar(value: object) -> bool:
    return is_missing(value) or isinstance(value, (Number, str))


def validate_scalar(value: object, dtype: LogicalType) -> Scalar:
    if is_missing(value):
        return NA
    match dtype:
        case SignedInteger() if is_integer(value):
            if int(value) not in dtype.value_range:
                raise TypeError(f"integer value is outside {dtype.name} range")
            return int(value)
        case Boolean() if isinstance(value, bool):
            return bool(value)
        case Utf8() if isinstance(value, str):
            return str(value)
    raise TypeError(f"value {value!r} does not belong to {dtype.name}")


# Constructor functions follow the pyarrow convention (``pa.int64()``).
def int64() -> LogicalType:
    return INT64


def boolean() -> LogicalType:
    return BOOLEAN


def utf8() -> LogicalType:
    return UTF8
