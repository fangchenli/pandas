"""
Tests for pandas.lazy.types module - Type system.
"""

import numpy as np
import pytest

import pandas as pd
from pandas.lazy.expr import col
from pandas.lazy.ir import (
    FieldRef,
    Literal,
)
from pandas.lazy.types import (
    LazyDtype,
    Schema,
    infer_expr_dtype,
)


class TestLazyDtype:
    """Tests for LazyDtype class."""

    def test_creation(self):
        dtype = LazyDtype("numeric", np.dtype("int64"), None, False)
        assert dtype.category == "numeric"
        assert dtype.numpy_dtype == np.dtype("int64")
        assert dtype.arrow_type is None
        assert dtype.nullable is False

    def test_is_numeric(self):
        dtype = LazyDtype("numeric", np.dtype("int64"), None, False)
        assert dtype.is_numeric() is True
        assert dtype.is_string() is False
        assert dtype.is_boolean() is False
        assert dtype.is_datetime() is False

    def test_is_string(self):
        dtype = LazyDtype("string", None, None, True)
        assert dtype.is_string() is True
        assert dtype.is_numeric() is False

    def test_is_boolean(self):
        dtype = LazyDtype("boolean", np.dtype("bool"), None, False)
        assert dtype.is_boolean() is True
        assert dtype.is_numeric() is False

    def test_is_datetime(self):
        dtype = LazyDtype("datetime", np.dtype("datetime64[ns]"), None, False)
        assert dtype.is_datetime() is True

    def test_from_pandas_dtype_int(self):
        dtype = LazyDtype.from_pandas_dtype(np.dtype("int64"))
        assert dtype.category == "numeric"
        assert dtype.numpy_dtype == np.dtype("int64")
        assert dtype.nullable is False

    def test_from_pandas_dtype_float(self):
        dtype = LazyDtype.from_pandas_dtype(np.dtype("float64"))
        assert dtype.category == "numeric"
        assert dtype.nullable is True  # Floats can have NaN

    def test_from_pandas_dtype_bool(self):
        dtype = LazyDtype.from_pandas_dtype(np.dtype("bool"))
        assert dtype.category == "boolean"

    def test_from_pandas_dtype_object(self):
        # object dtype is explicitly categorized as "object", not "string"
        # because object dtype can hold mixed Python types
        dtype = LazyDtype.from_pandas_dtype(np.dtype("object"))
        assert dtype.category == "object"

    def test_from_pandas_dtype_string(self):
        # pd.StringDtype is categorized as "string"
        dtype = LazyDtype.from_pandas_dtype(pd.StringDtype())
        assert dtype.category == "string"

    def test_from_pandas_dtype_datetime(self):
        dtype = LazyDtype.from_pandas_dtype(np.dtype("datetime64[ns]"))
        assert dtype.category == "datetime"

    def test_from_pandas_dtype_with_nulls(self):
        dtype = LazyDtype.from_pandas_dtype(np.dtype("int64"), has_nulls=True)
        assert dtype.nullable is True

    def test_repr(self):
        dtype = LazyDtype("numeric", np.dtype("int64"), None, True)
        result = repr(dtype)
        assert "numeric" in result
        assert "int64" in result
        assert "nullable" in result

    def test_frozen(self):
        dtype = LazyDtype("numeric", np.dtype("int64"), None, False)
        with pytest.raises(AttributeError):
            dtype.category = "string"


class TestLazyDtypeArrow:
    """Tests for LazyDtype with Arrow types."""

    @pytest.fixture
    def pyarrow(self):
        pa = pytest.importorskip("pyarrow")
        return pa

    def test_from_arrow_type_int(self, pyarrow):
        dtype = LazyDtype.from_arrow_type(pyarrow.int64())
        assert dtype.category == "numeric"
        assert dtype.arrow_type == pyarrow.int64()

    def test_from_arrow_type_string(self, pyarrow):
        dtype = LazyDtype.from_arrow_type(pyarrow.string())
        assert dtype.category == "string"

    def test_from_arrow_type_bool(self, pyarrow):
        dtype = LazyDtype.from_arrow_type(pyarrow.bool_())
        assert dtype.category == "boolean"

    def test_from_pandas_arrow_dtype(self, pyarrow):
        arrow_dtype = pd.ArrowDtype(pyarrow.int64())
        dtype = LazyDtype.from_pandas_dtype(arrow_dtype)
        assert dtype.category == "numeric"
        assert dtype.arrow_type == pyarrow.int64()


class TestSchema:
    """Tests for Schema class."""

    def test_from_dataframe(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.0, 2.0, 3.0],
                "c": pd.array(["x", "y", "z"], dtype="string"),  # explicit StringDtype
            }
        )
        schema = Schema.from_dataframe(df)
        assert "a" in schema
        assert "b" in schema
        assert "c" in schema
        assert schema["a"].is_numeric()
        assert schema["b"].is_numeric()
        assert schema["c"].is_string()

    def test_from_dataframe_object_dtype(self):
        # Object dtype columns are categorized as "object", not "string"
        # Use mixed types to ensure object dtype (pandas coerces pure strings to StringDtype)
        df = pd.DataFrame(
            {
                "mixed_col": [1, "hello", 3.14],  # mixed types force object dtype
            }
        )
        schema = Schema.from_dataframe(df)
        assert schema["mixed_col"].category == "object"

    def test_names(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        schema = Schema.from_dataframe(df)
        assert schema.names == ["a", "b", "c"]

    def test_getitem(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        schema = Schema.from_dataframe(df)
        dtype = schema["a"]
        assert isinstance(dtype, LazyDtype)
        assert dtype.is_numeric()

    def test_getitem_missing(self):
        df = pd.DataFrame({"a": [1]})
        schema = Schema.from_dataframe(df)
        with pytest.raises(KeyError, match="not found in schema"):
            schema["nonexistent"]

    def test_contains(self):
        df = pd.DataFrame({"a": [1]})
        schema = Schema.from_dataframe(df)
        assert "a" in schema
        assert "b" not in schema

    def test_repr(self):
        df = pd.DataFrame({"a": [1], "b": pd.array(["x"], dtype="string")})
        schema = Schema.from_dataframe(df)
        result = repr(schema)
        assert "Schema" in result
        assert "a" in result
        assert "b" in result

    def test_nullable_detection(self):
        df = pd.DataFrame(
            {
                "with_null": [1.0, None, 3.0],
                "no_null": [1.0, 2.0, 3.0],
            }
        )
        schema = Schema.from_dataframe(df)
        # Float columns are always nullable (can have NaN)
        assert schema["with_null"].nullable is True
        assert schema["no_null"].nullable is True

    def test_from_exprs(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        input_schema = Schema.from_dataframe(df)

        exprs = (col("a"), col("b").alias("new_b"))
        schema = Schema.from_exprs(exprs, input_schema)

        assert schema.names == ["a", "new_b"]
        assert schema["a"].is_numeric()
        assert schema["new_b"].is_numeric()


class TestInferExprDtype:
    """Tests for infer_expr_dtype function."""

    @pytest.fixture
    def sample_schema(self):
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        return Schema.from_dataframe(df)

    def test_field_ref(self, sample_schema):
        node = FieldRef("int_col")
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_numeric()

    def test_literal_int(self, sample_schema):
        node = Literal(42)
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_numeric()
        assert dtype.numpy_dtype == np.dtype("int64")

    def test_literal_float(self, sample_schema):
        node = Literal(3.14)
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_numeric()
        assert dtype.numpy_dtype == np.dtype("float64")

    def test_literal_string(self, sample_schema):
        node = Literal("hello")
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_string()

    def test_literal_bool(self, sample_schema):
        node = Literal(True)
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_boolean()


class TestInferExprDtypeOperators:
    """Tests for type inference with expression operators."""

    @pytest.fixture
    def sample_schema(self):
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        return Schema.from_dataframe(df)

    def test_add_int_int(self, sample_schema):
        from pandas.lazy.ir import Call

        node = Call("add", (FieldRef("int_col"), FieldRef("int_col")))
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_numeric()
        assert dtype.numpy_dtype == np.dtype("int64")

    def test_add_int_float(self, sample_schema):
        from pandas.lazy.ir import Call

        node = Call("add", (FieldRef("int_col"), FieldRef("float_col")))
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_numeric()
        assert dtype.numpy_dtype == np.dtype("float64")

    def test_divide_returns_float(self, sample_schema):
        from pandas.lazy.ir import Call

        node = Call("divide", (FieldRef("int_col"), Literal(2)))
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_numeric()
        assert dtype.numpy_dtype == np.dtype("float64")

    def test_comparison_returns_bool(self, sample_schema):
        from pandas.lazy.ir import Call

        node = Call("greater", (FieldRef("int_col"), Literal(0)))
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_boolean()

    def test_logical_and_returns_bool(self, sample_schema):
        from pandas.lazy.ir import Call

        left = Call("greater", (FieldRef("int_col"), Literal(0)))
        right = Call("less", (FieldRef("int_col"), Literal(10)))
        node = Call("and_", (left, right))
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_boolean()

    def test_negate_preserves_type(self, sample_schema):
        from pandas.lazy.ir import Call

        node = Call("negate", (FieldRef("int_col"),))
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_numeric()

    def test_abs_preserves_type(self, sample_schema):
        from pandas.lazy.ir import Call

        node = Call("abs", (FieldRef("float_col"),))
        dtype = infer_expr_dtype(node, sample_schema)
        assert dtype.is_numeric()
