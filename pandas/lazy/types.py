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
            # Check if it's Arrow-backed via storage attribute
            storage = getattr(dtype, "storage", None)
            if storage == "pyarrow":
                import pyarrow as pa

                # Arrow-backed string uses large_string internally
                return cls("string", None, pa.large_string(), has_nulls)
            return cls("string", None, None, has_nulls)

        # Categorical -> Arrow dictionary. extract_array wraps Categorical
        # zero-copy as a pa.DictionaryArray, and acero hash-aggregates
        # dictionary keys ~5x faster than raw strings; the schema must
        # report arrow storage so the decision layer routes accordingly
        # (with the old object/numpy mapping, a 10M category groupby ran
        # 313 ms through the NumPy kernel's object path vs ~20 ms here).
        if isinstance(dtype, pd.CategoricalDtype):
            import pyarrow as pa

            cat_dtype = dtype.categories.dtype
            if cat_dtype == np.dtype("object") or str(cat_dtype) in ("str", "string"):
                return cls(
                    "string", None, pa.dictionary(pa.int32(), pa.large_string()), True
                )
            return cls(
                "numeric",
                np.dtype(cat_dtype),
                pa.dictionary(pa.int32(), pa.from_numpy_dtype(cat_dtype)),
                True,
            )

        # Handle nullable extension dtypes (Int64, Float64, boolean, etc.)
        # These must be checked before the standard is_*_dtype checks because
        # is_integer_dtype(Int64Dtype()) returns True but np.dtype() fails on it
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            # Get the underlying numpy dtype if available
            numpy_dtype = getattr(dtype, "numpy_dtype", None)
            if numpy_dtype is not None:
                numpy_dtype = np.dtype(numpy_dtype)

            if pd.api.types.is_integer_dtype(dtype):
                return cls("numeric", numpy_dtype, None, True)  # nullable ints
            elif pd.api.types.is_float_dtype(dtype):
                return cls("numeric", numpy_dtype, None, True)
            elif pd.api.types.is_bool_dtype(dtype):
                return cls("boolean", numpy_dtype, None, True)
            else:
                return cls("object", np.dtype("object"), None, True)

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


def _cheap_has_nulls(ser) -> bool:
    """
    Null presence from metadata only — never scans column data.

    - Arrow-backed arrays: null_count is chunk metadata (O(num_chunks))
    - Masked extension arrays (Int64, Float64, boolean): mask.any() is a
      fast vectorized check on a bool array
    - NumPy int/uint/bool: cannot hold nulls
    - NumPy float/object/datetime: would require an O(n) scan; report
      False and let kernels handle actual NaN values at execution time
    """
    arr = ser.array
    pa_arr = getattr(arr, "_pa_array", None)
    if pa_arr is not None:
        return pa_arr.null_count > 0
    mask = getattr(arr, "_mask", None)
    if mask is not None:
        return bool(mask.any())
    return False


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
        """Build schema from DataFrame.

        Plan construction must be O(1) per column: a previous version
        scanned every column with isna().any() to set the nullability
        flag, adding ~5 ms per df.select() at 10M rows x 3 columns. The
        flag only feeds advisory dtype inference, so it is now derived
        from metadata that is free to read: Arrow null counts (chunk
        metadata), extension-array masks, and the fact that NumPy
        int/uint/bool arrays cannot hold nulls. NumPy float/object
        columns report has_nulls=False without scanning; actual NaN
        handling happens in the kernels, not the schema.
        """
        fields = {}
        for col in df.columns:
            ser = df[col]
            dtype = ser.dtype
            fields[str(col)] = LazyDtype.from_pandas_dtype(dtype, _cheap_has_nulls(ser))
        return cls(fields)

    @classmethod
    def from_exprs(cls, exprs: tuple[Expr, ...], input_schema: Schema) -> Schema:
        """Build schema from expression list."""
        from pandas.lazy.expr import extract_output_name
        from pandas.lazy.type_inference import infer_expr_dtype

        fields = {}
        for expr in exprs:
            name = extract_output_name(expr)
            dtype = infer_expr_dtype(expr._ir, input_schema)
            fields[name] = dtype
        return cls(fields)

    def with_columns(self, exprs: tuple[Expr, ...]) -> Schema:
        """Return schema with additional/replaced columns."""
        from pandas.lazy.expr import extract_output_name
        from pandas.lazy.type_inference import infer_expr_dtype

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

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

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
