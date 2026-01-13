"""
Tests for pandas.lazy string operations.
"""

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.eval import Evaluator
from pandas.lazy.ir import (
    Call,
    FieldRef,
)


class TestExprStringAccessor:
    """Tests for Expr.str accessor methods."""

    def test_str_accessor_returns_accessor(self):
        from pandas.lazy.expr import ExprStringAccessor

        expr = col("s")
        assert isinstance(expr.str, ExprStringAccessor)

    def test_str_lower_ir(self):
        expr = col("s").str.lower()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_lower"

    def test_str_upper_ir(self):
        expr = col("s").str.upper()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_upper"

    def test_str_len_ir(self):
        expr = col("s").str.len()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_len"

    def test_str_strip_ir(self):
        expr = col("s").str.strip()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_strip"

    def test_str_contains_ir(self):
        expr = col("s").str.contains("pattern")
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_contains"
        assert expr._ir.kwargs == {"regex": True}

    def test_str_contains_no_regex_ir(self):
        expr = col("s").str.contains("pattern", regex=False)
        assert isinstance(expr._ir, Call)
        assert expr._ir.kwargs == {"regex": False}

    def test_str_startswith_ir(self):
        expr = col("s").str.startswith("Dr.")
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_startswith"

    def test_str_endswith_ir(self):
        expr = col("s").str.endswith(".txt")
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_endswith"

    def test_str_replace_ir(self):
        expr = col("s").str.replace("old", "new")
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_replace"
        assert expr._ir.kwargs == {"regex": True}

    def test_str_slice_ir(self):
        expr = col("s").str.slice(0, 3)
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "str_slice"
        assert expr._ir.kwargs == {"start": 0, "stop": 3}


class TestEvaluatorStringOps:
    """Tests for string operations in the evaluator."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "text": ["hello world", "foo bar", "test test", "abc", "xyz"],
                "spaced": ["  trim  ", "  left", "right  ", " both ", "none"],
            }
        )

    def test_str_lower(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("str_lower", (FieldRef("name"),))
        result = evaluator.evaluate(node)
        expected = sample_df["name"].str.lower()
        tm.assert_series_equal(result, expected)

    def test_str_upper(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("str_upper", (FieldRef("name"),))
        result = evaluator.evaluate(node)
        expected = sample_df["name"].str.upper()
        tm.assert_series_equal(result, expected)

    def test_str_len(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("str_len", (FieldRef("name"),))
        result = evaluator.evaluate(node)
        expected = sample_df["name"].str.len()
        tm.assert_series_equal(result, expected)

    def test_str_strip(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("str_strip", (FieldRef("spaced"),))
        result = evaluator.evaluate(node)
        expected = sample_df["spaced"].str.strip()
        tm.assert_series_equal(result, expected)

    def test_str_lstrip(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("str_lstrip", (FieldRef("spaced"),))
        result = evaluator.evaluate(node)
        expected = sample_df["spaced"].str.lstrip()
        tm.assert_series_equal(result, expected)

    def test_str_rstrip(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("str_rstrip", (FieldRef("spaced"),))
        result = evaluator.evaluate(node)
        expected = sample_df["spaced"].str.rstrip()
        tm.assert_series_equal(result, expected)

    def test_str_contains_regex(self, sample_df):
        from pandas.lazy.ir import Literal

        evaluator = Evaluator(sample_df)
        node = Call("str_contains", (FieldRef("text"), Literal("o+")), {"regex": True})
        result = evaluator.evaluate(node)
        expected = sample_df["text"].str.contains("o+", regex=True)
        tm.assert_series_equal(result, expected)

    def test_str_contains_literal(self, sample_df):
        from pandas.lazy.ir import Literal

        evaluator = Evaluator(sample_df)
        node = Call(
            "str_contains", (FieldRef("text"), Literal("foo")), {"regex": False}
        )
        result = evaluator.evaluate(node)
        expected = sample_df["text"].str.contains("foo", regex=False)
        tm.assert_series_equal(result, expected)

    def test_str_startswith(self, sample_df):
        from pandas.lazy.ir import Literal

        evaluator = Evaluator(sample_df)
        node = Call("str_startswith", (FieldRef("name"), Literal("A")))
        result = evaluator.evaluate(node)
        expected = sample_df["name"].str.startswith("A")
        tm.assert_series_equal(result, expected)

    def test_str_endswith(self, sample_df):
        from pandas.lazy.ir import Literal

        evaluator = Evaluator(sample_df)
        node = Call("str_endswith", (FieldRef("name"), Literal("e")))
        result = evaluator.evaluate(node)
        expected = sample_df["name"].str.endswith("e")
        tm.assert_series_equal(result, expected)

    def test_str_replace_regex(self, sample_df):
        from pandas.lazy.ir import Literal

        evaluator = Evaluator(sample_df)
        node = Call(
            "str_replace",
            (FieldRef("text"), Literal("o+"), Literal("O")),
            {"regex": True},
        )
        result = evaluator.evaluate(node)
        expected = sample_df["text"].str.replace("o+", "O", regex=True)
        tm.assert_series_equal(result, expected)

    def test_str_replace_literal(self, sample_df):
        from pandas.lazy.ir import Literal

        evaluator = Evaluator(sample_df)
        node = Call(
            "str_replace",
            (FieldRef("text"), Literal("test"), Literal("TEST")),
            {"regex": False},
        )
        result = evaluator.evaluate(node)
        expected = sample_df["text"].str.replace("test", "TEST", regex=False)
        tm.assert_series_equal(result, expected)

    def test_str_slice(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("str_slice", (FieldRef("name"),), {"start": 0, "stop": 2})
        result = evaluator.evaluate(node)
        expected = sample_df["name"].str.slice(start=0, stop=2)
        tm.assert_series_equal(result, expected)

    def test_str_slice_start_only(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("str_slice", (FieldRef("name"),), {"start": 2, "stop": None})
        result = evaluator.evaluate(node)
        expected = sample_df["name"].str.slice(start=2, stop=None)
        tm.assert_series_equal(result, expected)


class TestLazyFrameStringOps:
    """Tests for string operations in LazyDataFrame workflows."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "email": ["alice@test.com", "bob@example.com", "charlie@test.com"],
                "notes": ["  hello  ", "world", "  foo  "],
            }
        )

    def test_filter_with_str_contains(self, sample_df):
        result = sample_df.select().filter(col("email").str.contains("test")).collect()
        expected = sample_df[sample_df["email"].str.contains("test")].reset_index(
            drop=True
        )
        tm.assert_frame_equal(result, expected)

    def test_filter_with_str_startswith(self, sample_df):
        result = sample_df.select().filter(col("name").str.startswith("A")).collect()
        expected = sample_df[sample_df["name"].str.startswith("A")].reset_index(
            drop=True
        )
        tm.assert_frame_equal(result, expected)

    def test_with_columns_str_lower(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("name").str.lower().alias("name_lower"))
            .collect()
        )
        expected = sample_df.copy()
        expected["name_lower"] = sample_df["name"].str.lower()
        tm.assert_frame_equal(result, expected)

    def test_with_columns_str_upper(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("name").str.upper().alias("name_upper"))
            .collect()
        )
        expected = sample_df.copy()
        expected["name_upper"] = sample_df["name"].str.upper()
        tm.assert_frame_equal(result, expected)

    def test_with_columns_str_len(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("name").str.len().alias("name_len"))
            .collect()
        )
        expected = sample_df.copy()
        expected["name_len"] = sample_df["name"].str.len()
        tm.assert_frame_equal(result, expected)

    def test_with_columns_str_strip(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("notes").str.strip().alias("notes_stripped"))
            .collect()
        )
        expected = sample_df.copy()
        expected["notes_stripped"] = sample_df["notes"].str.strip()
        tm.assert_frame_equal(result, expected)

    def test_with_columns_str_replace(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(
                col("email").str.replace("@.*", "@company.com").alias("new_email")
            )
            .collect()
        )
        expected = sample_df.copy()
        expected["new_email"] = sample_df["email"].str.replace(
            "@.*", "@company.com", regex=True
        )
        tm.assert_frame_equal(result, expected)

    def test_with_columns_str_slice(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(col("name").str.slice(0, 3).alias("initials"))
            .collect()
        )
        expected = sample_df.copy()
        expected["initials"] = sample_df["name"].str.slice(start=0, stop=3)
        tm.assert_frame_equal(result, expected)

    def test_chained_string_ops(self, sample_df):
        """Test chaining multiple string operations."""
        result = (
            sample_df.select()
            .with_columns(col("notes").str.strip().str.upper().alias("clean_notes"))
            .collect()
        )
        expected = sample_df.copy()
        expected["clean_notes"] = sample_df["notes"].str.strip().str.upper()
        tm.assert_frame_equal(result, expected)

    def test_filter_and_transform_strings(self, sample_df):
        """Test filtering and transforming strings together."""
        result = (
            sample_df.select()
            .filter(col("email").str.endswith(".com"))
            .with_columns(col("name").str.lower().alias("name_lower"))
            .select("name", "email", "name_lower")
            .collect()
        )
        filtered = sample_df[sample_df["email"].str.endswith(".com")].reset_index(
            drop=True
        )
        expected = filtered.copy()
        expected["name_lower"] = filtered["name"].str.lower()
        expected = expected[["name", "email", "name_lower"]]
        tm.assert_frame_equal(result, expected)


class TestNullMethods:
    """Tests for is_null and is_not_null methods on Expr."""

    def test_is_null_ir(self):
        expr = col("a").is_null()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "is_null"

    def test_is_not_null_ir(self):
        expr = col("a").is_not_null()
        assert isinstance(expr._ir, Call)
        assert expr._ir.function == "is_not_null"

    def test_filter_is_null(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        result = df.select().filter(col("a").is_null()).collect()
        expected = df[df["a"].isna()].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_filter_is_not_null(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        result = df.select().filter(col("a").is_not_null()).collect()
        expected = df[df["a"].notna()].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)
