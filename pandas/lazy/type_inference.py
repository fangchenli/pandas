"""
Type inference over IR expression nodes.

This module sits *above* both :mod:`pandas.lazy.types` (the type lattice:
``LazyDtype`` / ``Schema``) and :mod:`pandas.lazy.ir` (the expression nodes),
and depends on both. Keeping it separate is what lets ``types`` stay a
dependency-free foundation: ``ir`` refers to ``types`` for annotations, and the
inference dispatch that needs to know about concrete IR node classes lives here
instead of inside ``types`` — so there is no ``types`` <-> ``ir`` import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.lazy.ir import (
    Alias,
    Call,
    Cast,
    FieldRef,
    Literal,
)
from pandas.lazy.types import LazyDtype

if TYPE_CHECKING:
    from pandas.lazy.ir import IRNode
    from pandas.lazy.types import Schema


def infer_expr_dtype(node: IRNode, schema: Schema) -> LazyDtype:
    """Infer output dtype of an IR node."""
    if isinstance(node, FieldRef):
        return schema[node.name]

    elif isinstance(node, Literal):
        if node.dtype:
            return node.dtype
        # Infer from Python value
        if isinstance(node.value, bool):
            return LazyDtype("boolean", np.dtype("bool"), None, False)
        elif isinstance(node.value, int):
            return LazyDtype("numeric", np.dtype("int64"), None, False)
        elif isinstance(node.value, float):
            return LazyDtype("numeric", np.dtype("float64"), None, False)
        elif isinstance(node.value, str):
            return LazyDtype("string", None, None, False)
        else:
            return LazyDtype("object", np.dtype("object"), None, False)

    elif isinstance(node, Call):
        return _infer_call_dtype(node, schema)

    elif isinstance(node, Cast):
        return LazyDtype.from_pandas_dtype(node.target_dtype)

    elif isinstance(node, Alias):
        return infer_expr_dtype(node.arg, schema)

    else:
        return LazyDtype("object", np.dtype("object"), None, True)


def _infer_call_dtype(node: Call, schema: Schema) -> LazyDtype:
    """Infer dtype for function calls."""
    # Comparison/logical -> boolean
    comparison_funcs = {
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
    }
    logical_funcs = {"and_", "or_", "invert"}

    if node.function in comparison_funcs | logical_funcs:
        return LazyDtype("boolean", np.dtype("bool"), None, False)

    # Unary operators
    if node.function == "negate":
        if node.args:
            return infer_expr_dtype(node.args[0], schema)
        return LazyDtype("numeric", np.dtype("float64"), None, True)

    if node.function == "abs":
        if node.args:
            return infer_expr_dtype(node.args[0], schema)
        return LazyDtype("numeric", np.dtype("float64"), None, True)

    # Division always returns float
    if node.function == "divide":
        return LazyDtype("numeric", np.dtype("float64"), None, True)

    # Arithmetic operators that preserve type
    arithmetic_funcs = {
        "add",
        "subtract",
        "multiply",
        "floor_divide",
        "modulo",
        "power",
    }
    if node.function in arithmetic_funcs and node.args:
        # Determine result type from operands
        # For numeric ops, use the "wider" type
        left_dtype = infer_expr_dtype(node.args[0], schema)
        right_dtype = (
            infer_expr_dtype(node.args[1], schema) if len(node.args) > 1 else left_dtype
        )

        # If either is float, result is float
        if left_dtype.numpy_dtype is not None and np.issubdtype(
            left_dtype.numpy_dtype, np.floating
        ):
            return LazyDtype("numeric", np.dtype("float64"), None, True)
        if right_dtype.numpy_dtype is not None and np.issubdtype(
            right_dtype.numpy_dtype, np.floating
        ):
            return LazyDtype("numeric", np.dtype("float64"), None, True)

        # Both are integer-like
        nullable = left_dtype.nullable or right_dtype.nullable
        return LazyDtype("numeric", np.dtype("int64"), None, nullable)

    # is_null/is_not_null -> boolean
    if node.function in {"is_null", "is_not_null"}:
        return LazyDtype("boolean", np.dtype("bool"), None, False)

    # String functions that return strings
    if node.function in {
        "str_lower",
        "str_upper",
        "str_strip",
        "str_lstrip",
        "str_rstrip",
        "str_replace",
        "str_slice",
    }:
        return LazyDtype("string", None, None, True)

    # String functions that return non-strings
    if node.function == "str_len":
        return LazyDtype("numeric", np.dtype("int64"), None, True)
    if node.function in {"str_contains", "str_startswith", "str_endswith"}:
        return LazyDtype("boolean", np.dtype("bool"), None, True)

    # Datetime components -> numeric
    datetime_numeric_funcs = {
        "dt_year",
        "dt_month",
        "dt_day",
        "dt_hour",
        "dt_minute",
        "dt_second",
        "dt_weekday",
        "dt_dayofyear",
        "dt_quarter",
    }
    if node.function in datetime_numeric_funcs:
        return LazyDtype("numeric", np.dtype("int32"), None, True)

    # Datetime boolean functions
    datetime_bool_funcs = {
        "dt_is_month_start",
        "dt_is_month_end",
        "dt_is_year_start",
        "dt_is_year_end",
    }
    if node.function in datetime_bool_funcs:
        return LazyDtype("boolean", np.dtype("bool"), None, True)

    # dt_date returns date (treat as datetime for now)
    if node.function == "dt_date":
        return LazyDtype("datetime", None, None, True)

    # Null handling
    if node.function == "fill_null":
        if node.args:
            # Return type is the type of the first argument
            return infer_expr_dtype(node.args[0], schema)
        return LazyDtype("object", np.dtype("object"), None, True)

    if node.function == "coalesce":
        if node.args:
            # Return type is the type of the first argument
            return infer_expr_dtype(node.args[0], schema)
        return LazyDtype("object", np.dtype("object"), None, True)

    # Window functions
    if node.function == "window":
        # Window function returns the same type as the inner aggregation
        if node.args:
            inner_node = node.args[0]
            return infer_expr_dtype(inner_node, schema)
        return LazyDtype("object", np.dtype("object"), None, True)

    if node.function in {"rank", "dense_rank", "row_number"}:
        return LazyDtype("numeric", np.dtype("float64"), None, True)

    if node.function in {"lag", "lead"}:
        if node.args:
            return infer_expr_dtype(node.args[0], schema)
        return LazyDtype("object", np.dtype("object"), None, True)

    if node.function in {"cum_sum", "cum_min", "cum_max", "cum_mean", "cum_prod"}:
        if node.args:
            return infer_expr_dtype(node.args[0], schema)
        return LazyDtype("numeric", np.dtype("float64"), None, True)

    # Conditional expressions (case_when)
    if node.function == "case_when":
        cases = node.kwargs.get("cases", ())
        otherwise_node = node.kwargs.get("otherwise")

        # Collect all value types from cases and otherwise
        value_types = []
        for _, value_node in cases:
            value_types.append(infer_expr_dtype(value_node, schema))
        if otherwise_node:
            value_types.append(infer_expr_dtype(otherwise_node, schema))

        if not value_types:
            return LazyDtype("object", np.dtype("object"), None, True)

        # Use the first value's type as the result type
        # In a more sophisticated implementation, we might unify types
        result_dtype = value_types[0]
        # Mark as nullable since conditions might not all match
        return LazyDtype(
            result_dtype.category,
            result_dtype.numpy_dtype,
            result_dtype.arrow_type,
            True,
        )

    # Aggregations
    if node.is_aggregate:
        if node.function == "count":
            return LazyDtype("numeric", np.dtype("int64"), None, False)
        elif node.function in {"mean", "std", "var"}:
            return LazyDtype("numeric", np.dtype("float64"), None, True)
        elif node.args:
            # sum/min/max preserve the input's nullability: an aggregate of a
            # NumPy int column stays NumPy int64 (eager semantics), while one of
            # a masked Int64 column stays nullable — keeping ``nullable`` a
            # reliable masked-vs-NumPy signal for the output dtype contract.
            arg_dtype = infer_expr_dtype(node.args[0], schema)
            return LazyDtype(
                arg_dtype.category,
                arg_dtype.numpy_dtype,
                arg_dtype.arrow_type,
                arg_dtype.nullable,
            )

    # Arithmetic: propagate from first arg, mark nullable if any input nullable
    if node.args:
        arg_dtype = infer_expr_dtype(node.args[0], schema)
        nullable = any(
            infer_expr_dtype(a, schema).nullable
            for a in node.args
            if not isinstance(a, Literal)
        )
        return LazyDtype(
            arg_dtype.category, arg_dtype.numpy_dtype, arg_dtype.arrow_type, nullable
        )

    return LazyDtype("object", np.dtype("object"), None, True)
