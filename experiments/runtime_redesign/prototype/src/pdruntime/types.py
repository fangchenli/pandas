"""Logical identity and scalar validation, independent of physical storage."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd


class Missing:
    __slots__ = ()

    def __repr__(self):
        return "NA"

    def __bool__(self):
        raise TypeError("the truth value of NA is ambiguous")


NA = Missing()


@dataclass(frozen=True)
class LogicalType:
    kind: str
    parameters: tuple = ()
    missing_policy: str = "NA"


INT64 = LogicalType("signed_integer", (64,))
BOOLEAN = LogicalType("boolean")
UTF8 = LogicalType("utf8")
ALIASES = {"int64": INT64, "boolean": BOOLEAN, "utf8": UTF8}


def normalize_dtype(spec):
    if isinstance(spec, str) and spec in ALIASES:
        return ALIASES[spec]
    if isinstance(spec, LogicalType):
        for dtype in ALIASES.values():
            if spec == dtype:
                return dtype
    raise TypeError(f"unsupported logical dtype: {spec!r}")


def is_missing(value):
    return value is NA or value is None or value is pd.NA


def validate_scalar(value, dtype):
    if is_missing(value):
        return NA
    if dtype == INT64 and isinstance(value, Integral):
        if not isinstance(value, (bool, np.bool_)):
            if -(2**63) <= int(value) < 2**63:
                return int(value)
            raise TypeError("integer value is outside int64 range")
    if dtype == BOOLEAN and isinstance(value, (bool, np.bool_)):
        return bool(value)
    if dtype == UTF8 and isinstance(value, str):
        return str(value)
    raise TypeError(f"value {value!r} does not belong to {dtype.kind}")


def int64():
    return INT64


def boolean():
    return BOOLEAN


def utf8():
    return UTF8
