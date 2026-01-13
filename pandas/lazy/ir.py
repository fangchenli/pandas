"""
Intermediate Representation (IR) nodes for lazy expressions.

These nodes form the expression tree that represents computations
before they are executed.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from pandas.lazy.types import LazyDtype


@dataclass(frozen=True, slots=True)
class IRNode:
    """Base class for all IR nodes."""


@dataclass(frozen=True, slots=True)
class FieldRef(IRNode):
    """Reference to a column by name."""

    name: str

    def __repr__(self) -> str:
        return f"FieldRef({self.name!r})"


@dataclass(frozen=True, slots=True)
class Literal(IRNode):
    """A constant/literal value."""

    value: Any
    dtype: LazyDtype | None = None  # If None, inferred from value

    def __repr__(self) -> str:
        if self.dtype is not None:
            return f"Literal({self.value!r}, dtype={self.dtype})"
        return f"Literal({self.value!r})"


@dataclass(frozen=True, slots=True)
class Alias(IRNode):
    """Named expression for output column naming."""

    arg: IRNode
    name: str

    def __repr__(self) -> str:
        return f"Alias({self.arg!r}, name={self.name!r})"


@dataclass(frozen=True)
class Call(IRNode):
    """
    Function call node.

    Uses canonical function names that map to backend implementations.
    """

    function: str  # Canonical name, e.g., "add", "str_contains"
    args: tuple[IRNode, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    is_aggregate: bool = False

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        if self.kwargs:
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
            args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str
        agg_marker = ", agg=True" if self.is_aggregate else ""
        return f"Call({self.function!r}, ({args_str}){agg_marker})"


@dataclass(frozen=True, slots=True)
class Cast(IRNode):
    """Explicit type cast."""

    arg: IRNode
    target_dtype: Any  # Can be LazyDtype or string

    def __repr__(self) -> str:
        return f"Cast({self.arg!r}, {self.target_dtype!r})"
