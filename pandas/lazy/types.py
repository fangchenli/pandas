"""
Type system for lazy pandas.

Provides LazyDtype for type tracking and Schema for column metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

    from pandas import DataFrame
    from pandas.lazy.expr import Expr
    from pandas.lazy.ir import Call


@dataclass(frozen=True, slots=True)
class LazyDtype:
    """
    Lazy type representation with nullability tracking.

    Simplified type system for backend selection and validation.
    """

    category: str  # "numeric", "string", "datetime", "boolean", "object"
    numpy_dtype: np.dtype | None  # Specific dtype if known
    arrow_type: pa.DataType | None  # Arrow type if known
    nullable: bool  # Whether nulls are possible

    @classmethod
    def from_pandas_dtype(cls, dtype, has_nulls: bool = False) -> LazyDtype:
        """
        Create LazyDtype from pandas/numpy dtype.

        Parameters
        ----------
        dtype : dtype
            A pandas or numpy dtype.
        has_nulls : bool, default False
            Whether the data contains null values.

        Returns
        -------
        LazyDtype

        Notes
        -----
        String type handling:
        - ``pd.StringDtype()`` -> category="string"
        - ``pd.ArrowDtype(pa.string())`` -> category="string" (via from_arrow_type)
        - ``np.dtype("object")`` -> category="object" (not string, since object
          can hold mixed types)

        For object dtype columns that are known to contain only strings,
        consider converting to StringDtype first.
        """
        import pandas as pd

        # Handle ArrowDtype first
        if isinstance(dtype, pd.ArrowDtype):
            return cls.from_arrow_type(dtype.pyarrow_dtype)

        # Handle pandas StringDtype (includes string[python], string[pyarrow])
        if isinstance(dtype, pd.StringDtype):
            return cls("string", None, None, has_nulls)

        # Handle numpy/pandas numeric types
        if pd.api.types.is_integer_dtype(dtype):
            return cls("numeric", np.dtype(dtype), None, has_nulls)
        elif pd.api.types.is_float_dtype(dtype):
            return cls("numeric", np.dtype(dtype), None, True)  # floats can have NaN
        elif pd.api.types.is_bool_dtype(dtype):
            return cls("boolean", np.dtype(dtype), None, has_nulls)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return cls("datetime", np.dtype(dtype), None, has_nulls)
        else:
            # This includes np.dtype("object") - we explicitly categorize as "object"
            # not "string", since object dtype can hold mixed Python types
            return cls("object", np.dtype("object"), None, has_nulls)

    @classmethod
    def from_arrow_type(cls, arrow_type) -> LazyDtype:
        """Create LazyDtype from Arrow type."""
        import pyarrow as pa

        if pa.types.is_integer(arrow_type):
            return cls("numeric", arrow_type.to_pandas_dtype(), arrow_type, True)
        elif pa.types.is_floating(arrow_type):
            return cls("numeric", arrow_type.to_pandas_dtype(), arrow_type, True)
        elif pa.types.is_boolean(arrow_type):
            return cls("boolean", np.dtype("bool"), arrow_type, True)
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return cls("string", None, arrow_type, True)
        elif pa.types.is_timestamp(arrow_type):
            return cls("datetime", None, arrow_type, True)
        else:
            return cls("object", np.dtype("object"), arrow_type, True)

    def is_numeric(self) -> bool:
        return self.category == "numeric"

    def is_string(self) -> bool:
        return self.category == "string"

    def is_datetime(self) -> bool:
        return self.category == "datetime"

    def is_boolean(self) -> bool:
        return self.category == "boolean"

    @property
    def storage_backend(self) -> str:
        """
        Return the physical storage backend: 'numpy' or 'arrow'.

        Used by the optimizer for engine selection decisions.
        """
        if self.arrow_type is not None:
            return "arrow"
        return "numpy"

    def __repr__(self) -> str:
        parts = [f"category={self.category!r}"]
        if self.numpy_dtype is not None:
            parts.append(f"numpy={self.numpy_dtype}")
        if self.arrow_type is not None:
            parts.append(f"arrow={self.arrow_type}")
        if self.nullable:
            parts.append("nullable")
        return f"LazyDtype({', '.join(parts)})"


@dataclass
class Schema:
    """Schema for lazy operations - column names and types."""

    fields: dict[str, LazyDtype]

    @property
    def names(self) -> list[str]:
        """Return column names."""
        return list(self.fields.keys())

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> Schema:
        """Build schema from DataFrame."""
        fields = {}
        for col in df.columns:
            dtype = df[col].dtype
            has_nulls = df[col].isna().any()
            fields[str(col)] = LazyDtype.from_pandas_dtype(dtype, has_nulls)
        return cls(fields)

    @classmethod
    def from_exprs(cls, exprs: tuple[Expr, ...], input_schema: Schema) -> Schema:
        """Build schema from expression list."""
        from pandas.lazy.expr import extract_output_name
        from pandas.lazy.types import infer_expr_dtype

        fields = {}
        for expr in exprs:
            name = extract_output_name(expr)
            dtype = infer_expr_dtype(expr._ir, input_schema)
            fields[name] = dtype
        return cls(fields)

    def with_columns(self, exprs: tuple[Expr, ...]) -> Schema:
        """Return schema with additional/replaced columns."""
        from pandas.lazy.expr import extract_output_name
        from pandas.lazy.types import infer_expr_dtype

        fields = dict(self.fields)
        for expr in exprs:
            name = extract_output_name(expr)
            dtype = infer_expr_dtype(expr._ir, self)
            fields[name] = dtype
        return Schema(fields)

    def __getitem__(self, name: str) -> LazyDtype:
        if name not in self.fields:
            raise KeyError(
                f"Column {name!r} not found in schema. "
                f"Available: {list(self.fields.keys())}"
            )
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    def dominant_backend(self) -> str:
        """
        Return the dominant storage backend for this schema.

        Returns 'arrow' if majority of columns are Arrow-backed,
        otherwise 'numpy'.
        """
        if not self.fields:
            return "numpy"
        backends = [dtype.storage_backend for dtype in self.fields.values()]
        arrow_count = sum(1 for b in backends if b == "arrow")
        return "arrow" if arrow_count > len(backends) // 2 else "numpy"

    def __repr__(self) -> str:
        fields_str = ", ".join(f"{k}: {v.category}" for k, v in self.fields.items())
        return f"Schema({{{fields_str}}})"


def infer_expr_dtype(node, schema: Schema) -> LazyDtype:
    """Infer output dtype of an IR node."""
    from pandas.lazy.ir import (
        Alias,
        Call,
        Cast,
        FieldRef,
        Literal,
    )

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
    from pandas.lazy.ir import Literal

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

    if node.function in {"cum_sum", "cum_min", "cum_max"}:
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
            arg_dtype = infer_expr_dtype(node.args[0], schema)
            return LazyDtype(
                arg_dtype.category, arg_dtype.numpy_dtype, arg_dtype.arrow_type, True
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
