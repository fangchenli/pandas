"""
Tests for pandas.lazy null handling and cast operations.
"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import (
    coalesce,
    col,
)
from pandas.lazy.ir import (
    Call,
    Cast,
)


class TestFillNullAPI:
    """Tests for the fill_null expression API."""

    def test_fill_null_scalar_ir(self):
        expr = col("a").fill_null(0)
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "fill_null"

    def test_fill_null_column_ir(self):
        expr = col("a").fill_null(col("b"))
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "fill_null"


class TestFillNullEvaluation:
    """Tests for fill_null evaluation."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1.0, None, 3.0, None, 5.0],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

    def test_fill_null_with_scalar(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("a").fill_null(0).alias("filled"))
            .collect()
        )
        expected = sample_df.copy()
        expected["filled"] = [1.0, 0.0, 3.0, 0.0, 5.0]
        tm.assert_frame_equal(result, expected)

    def test_fill_null_with_column(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("a").fill_null(col("b")).alias("filled"))
            .collect()
        )
        expected = sample_df.copy()
        expected["filled"] = [1.0, 20.0, 3.0, 40.0, 5.0]
        tm.assert_frame_equal(result, expected)

    def test_fill_null_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = (
            df.select().with_columns(col("a").fill_null(0).alias("filled")).collect()
        )
        expected = df.copy()
        expected["filled"] = [1, 2, 3, 4, 5]
        tm.assert_frame_equal(result, expected)

    def test_fill_null_all_nulls_float(self):
        # Float dtype with NaN
        df = pd.DataFrame({"a": np.array([np.nan, np.nan, np.nan])})
        result = (
            df.select().with_columns(col("a").fill_null(-1).alias("filled")).collect()
        )
        expected = df.copy()
        expected["filled"] = [-1.0, -1.0, -1.0]
        tm.assert_frame_equal(result, expected)

    def test_fill_null_all_nulls_object(self):
        # Object dtype with Python None - fillna preserves object dtype
        df = pd.DataFrame({"a": [None, None, None]})
        result = (
            df.select().with_columns(col("a").fill_null(-1).alias("filled")).collect()
        )
        # With object dtype, fillna(-1) keeps object dtype with integer values
        assert list(result["filled"]) == [-1, -1, -1]
        # Verify original column preserved
        assert list(result["a"]) == [None, None, None]


class TestCoalesceAPI:
    """Tests for the coalesce expression API."""

    def test_coalesce_two_columns(self):
        expr = coalesce(col("a"), col("b"))
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "coalesce"

    def test_coalesce_with_scalar(self):
        expr = coalesce(col("a"), 0)
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "coalesce"

    def test_coalesce_requires_args(self):
        with pytest.raises(ValueError, match="requires at least one expression"):
            coalesce()

    def test_coalesce_many_args(self):
        expr = coalesce(col("a"), col("b"), col("c"), 0)
        assert isinstance(expr._ir, Call)
        assert len(expr._ir.args) == 4


class TestCoalesceEvaluation:
    """Tests for coalesce evaluation."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1.0, None, None, None, 5.0],
                "b": [None, 2.0, None, None, 6.0],
                "c": [None, None, 3.0, None, 7.0],
            }
        )

    def test_coalesce_two_columns(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(coalesce(col("a"), col("b")).alias("result"))
            .collect()
        )
        expected = sample_df.copy()
        expected["result"] = [1.0, 2.0, None, None, 5.0]
        tm.assert_frame_equal(result, expected)

    def test_coalesce_three_columns(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(coalesce(col("a"), col("b"), col("c")).alias("result"))
            .collect()
        )
        expected = sample_df.copy()
        expected["result"] = [1.0, 2.0, 3.0, None, 5.0]
        tm.assert_frame_equal(result, expected)

    def test_coalesce_with_default(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(coalesce(col("a"), col("b"), col("c"), -1).alias("result"))
            .collect()
        )
        expected = sample_df.copy()
        expected["result"] = [1.0, 2.0, 3.0, -1.0, 5.0]
        tm.assert_frame_equal(result, expected)

    def test_coalesce_single_column(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        result = df.select().with_columns(coalesce(col("a")).alias("result")).collect()
        expected = df.copy()
        expected["result"] = df["a"]
        tm.assert_frame_equal(result, expected)


class TestCastAPI:
    """Tests for the cast expression API."""

    def test_cast_to_float(self):
        expr = col("a").cast("float64")
        assert isinstance(expr._ir, Cast)
        assert expr._ir.target_dtype == "float64"

    def test_cast_to_int(self):
        expr = col("a").cast(int)
        assert isinstance(expr._ir, Cast)
        assert expr._ir.target_dtype == int

    def test_cast_to_string(self):
        expr = col("a").cast("string")
        assert isinstance(expr._ir, Cast)
        assert expr._ir.target_dtype == "string"


class TestCastEvaluation:
    """Tests for cast evaluation."""

    def test_cast_int_to_float(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = (
            df.select()
            .with_columns(col("a").cast("float64").alias("a_float"))
            .collect()
        )
        expected = df.copy()
        expected["a_float"] = df["a"].astype("float64")
        tm.assert_frame_equal(result, expected)

    def test_cast_float_to_int(self):
        df = pd.DataFrame({"a": [1.0, 2.5, 3.9, 4.1, 5.0]})
        result = (
            df.select().with_columns(col("a").cast("int64").alias("a_int")).collect()
        )
        expected = df.copy()
        expected["a_int"] = df["a"].astype("int64")
        tm.assert_frame_equal(result, expected)

    def test_cast_int_to_string(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = (
            df.select().with_columns(col("a").cast("string").alias("a_str")).collect()
        )
        expected = df.copy()
        expected["a_str"] = df["a"].astype("string")
        tm.assert_frame_equal(result, expected)

    def test_cast_string_to_int(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = (
            df.select().with_columns(col("a").cast("int64").alias("a_int")).collect()
        )
        expected = df.copy()
        expected["a_int"] = pd.to_numeric(df["a"]).astype("int64")
        tm.assert_frame_equal(result, expected)

    def test_cast_to_bool(self):
        df = pd.DataFrame({"a": [0, 1, 2, 0, 1]})
        result = (
            df.select().with_columns(col("a").cast("bool").alias("a_bool")).collect()
        )
        expected = df.copy()
        expected["a_bool"] = df["a"].astype("bool")
        tm.assert_frame_equal(result, expected)


class TestNullCastTypeInference:
    """Tests for type inference of null and cast operations."""

    def test_fill_null_preserves_type(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"a": LazyDtype("numeric", np.dtype("float64"), None, True)})
        node = Call("fill_null", (FieldRef("a"), FieldRef("a")))
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "numeric"

    def test_coalesce_type_from_first_arg(self):
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema(
            {
                "a": LazyDtype("string", None, None, True),
                "b": LazyDtype("string", None, None, True),
            }
        )
        node = Call("coalesce", (FieldRef("a"), FieldRef("b")))
        dtype = infer_expr_dtype(node, schema)
        assert dtype.category == "string"


class TestIsNullNaNSemantics:
    """is_null / is_not_null must treat float NaN as null (pandas semantics).

    Regression: an unmatched integer key surfaces as a float NaN after a left
    join. On the Arrow backend (a frame with string columns), both the cached
    fast path and the kernel called bare pc.is_null, which by default does NOT
    treat NaN as null — so the left-join + is_null anti-join idiom silently
    returned nothing (TPC-H Q22).
    """

    @pytest.mark.parametrize("use_physical", [True, False])
    def test_is_null_catches_nan_after_left_join(self, use_physical):
        # string column forces the Arrow backend, where the bug lived
        left = pd.DataFrame({"k": [1, 2, 3, 4], "s": ["a", "b", "c", "d"]})
        right = pd.DataFrame({"ok": [2, 4], "rv": [9.0, 9.0]})
        joined = left.select().join(
            right.select(), left_on="k", right_on="ok", how="left"
        )

        anti = joined.filter(col("ok").is_null()).collect(
            use_physical_planner=use_physical
        )
        assert sorted(anti["k"].tolist()) == [1, 3]

        matched = joined.filter(col("ok").is_not_null()).collect(
            use_physical_planner=use_physical
        )
        assert sorted(matched["k"].tolist()) == [2, 4]

    def test_is_null_on_plain_nan_column(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "s": ["a", "b", "c"]})
        out = df.select().filter(col("x").is_null()).collect(use_physical_planner=True)
        assert out["s"].tolist() == ["b"]
