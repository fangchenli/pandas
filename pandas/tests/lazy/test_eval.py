"""
Tests for pandas.lazy.eval module - Expression evaluation.
"""

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy.eval import Evaluator
from pandas.lazy.ir import (
    Alias,
    Call,
    FieldRef,
    Literal,
)


class TestEvaluatorBasics:
    """Tests for basic Evaluator functionality."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": [1.5, 2.5, 3.5, 4.5, 5.5],
                "s": ["x", "y", "z", "w", "v"],
                "flag": [True, False, True, False, True],
            }
        )

    def test_field_ref(self, sample_df):
        evaluator = Evaluator(sample_df)
        result = evaluator.evaluate(FieldRef("a"))
        tm.assert_series_equal(result, sample_df["a"])

    def test_literal_int(self, sample_df):
        evaluator = Evaluator(sample_df)
        result = evaluator.evaluate(Literal(42))
        assert result == 42

    def test_literal_string(self, sample_df):
        evaluator = Evaluator(sample_df)
        result = evaluator.evaluate(Literal("hello"))
        assert result == "hello"

    def test_alias(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Alias(FieldRef("a"), "new_a")
        result = evaluator.evaluate(node)
        tm.assert_series_equal(result, sample_df["a"])


class TestEvaluatorArithmetic:
    """Tests for arithmetic operations."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": [1.5, 2.5, 3.5, 4.5, 5.5],
            }
        )

    def test_add(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("add", (FieldRef("a"), FieldRef("b")))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] + sample_df["b"]
        tm.assert_series_equal(result, expected)

    def test_add_with_literal(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("add", (FieldRef("a"), Literal(10)))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] + 10
        tm.assert_series_equal(result, expected)

    def test_subtract(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("subtract", (FieldRef("b"), FieldRef("a")))
        result = evaluator.evaluate(node)
        expected = sample_df["b"] - sample_df["a"]
        tm.assert_series_equal(result, expected)

    def test_multiply(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("multiply", (FieldRef("a"), FieldRef("b")))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] * sample_df["b"]
        tm.assert_series_equal(result, expected)

    def test_divide(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("divide", (FieldRef("b"), FieldRef("a")))
        result = evaluator.evaluate(node)
        expected = sample_df["b"] / sample_df["a"]
        tm.assert_series_equal(result, expected)

    def test_floor_divide(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("floor_divide", (FieldRef("b"), FieldRef("a")))
        result = evaluator.evaluate(node)
        expected = sample_df["b"] // sample_df["a"]
        tm.assert_series_equal(result, expected)

    def test_modulo(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("modulo", (FieldRef("b"), Literal(3)))
        result = evaluator.evaluate(node)
        expected = sample_df["b"] % 3
        tm.assert_series_equal(result, expected)

    def test_power(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("power", (FieldRef("a"), Literal(2)))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] ** 2
        tm.assert_series_equal(result, expected)

    def test_negate(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("negate", (FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = -sample_df["a"]
        tm.assert_series_equal(result, expected)

    def test_abs(self, sample_df):
        df = pd.DataFrame({"x": [-1, 2, -3, 4, -5]})
        evaluator = Evaluator(df)
        node = Call("abs", (FieldRef("x"),))
        result = evaluator.evaluate(node)
        expected = pd.Series([1, 2, 3, 4, 5], name="x")
        tm.assert_series_equal(result, expected)


class TestEvaluatorComparison:
    """Tests for comparison operations."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [3, 3, 3, 3, 3],
                "s": ["x", "y", "x", "y", "x"],
            }
        )

    def test_equal(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("equal", (FieldRef("a"), FieldRef("b")))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] == sample_df["b"]
        tm.assert_series_equal(result, expected)

    def test_equal_with_literal(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("equal", (FieldRef("a"), Literal(3)))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] == 3
        tm.assert_series_equal(result, expected)

    def test_equal_string(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("equal", (FieldRef("s"), Literal("x")))
        result = evaluator.evaluate(node)
        expected = sample_df["s"] == "x"
        tm.assert_series_equal(result, expected)

    def test_not_equal(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("not_equal", (FieldRef("a"), Literal(3)))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] != 3
        tm.assert_series_equal(result, expected)

    def test_less(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("less", (FieldRef("a"), Literal(3)))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] < 3
        tm.assert_series_equal(result, expected)

    def test_less_equal(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("less_equal", (FieldRef("a"), Literal(3)))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] <= 3
        tm.assert_series_equal(result, expected)

    def test_greater(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("greater", (FieldRef("a"), Literal(3)))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] > 3
        tm.assert_series_equal(result, expected)

    def test_greater_equal(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("greater_equal", (FieldRef("a"), Literal(3)))
        result = evaluator.evaluate(node)
        expected = sample_df["a"] >= 3
        tm.assert_series_equal(result, expected)


class TestEvaluatorLogical:
    """Tests for logical operations."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "flag": [True, False, True, False, True],
            }
        )

    def test_and(self, sample_df):
        evaluator = Evaluator(sample_df)
        left = Call("greater", (FieldRef("a"), Literal(2)))
        right = Call("less", (FieldRef("a"), Literal(5)))
        node = Call("and_", (left, right))
        result = evaluator.evaluate(node)
        expected = (sample_df["a"] > 2) & (sample_df["a"] < 5)
        tm.assert_series_equal(result, expected)

    def test_or(self, sample_df):
        evaluator = Evaluator(sample_df)
        left = Call("less", (FieldRef("a"), Literal(2)))
        right = Call("greater", (FieldRef("a"), Literal(4)))
        node = Call("or_", (left, right))
        result = evaluator.evaluate(node)
        expected = (sample_df["a"] < 2) | (sample_df["a"] > 4)
        tm.assert_series_equal(result, expected)

    def test_invert(self, sample_df):
        evaluator = Evaluator(sample_df)
        node = Call("invert", (FieldRef("flag"),))
        result = evaluator.evaluate(node)
        expected = ~sample_df["flag"]
        tm.assert_series_equal(result, expected)


class TestEvaluatorNullChecks:
    """Tests for null check operations."""

    def test_is_null(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        evaluator = Evaluator(df)
        node = Call("is_null", (FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = pd.isna(df["a"])
        tm.assert_series_equal(result, expected)

    def test_is_not_null(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        evaluator = Evaluator(df)
        node = Call("is_not_null", (FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = pd.notna(df["a"])
        tm.assert_series_equal(result, expected)


class TestEvaluatorComplex:
    """Tests for complex expression evaluation."""

    def test_chained_arithmetic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        evaluator = Evaluator(df)
        # (a + b) * 2
        add = Call("add", (FieldRef("a"), FieldRef("b")))
        node = Call("multiply", (add, Literal(2)))
        result = evaluator.evaluate(node)
        expected = (df["a"] + df["b"]) * 2
        tm.assert_series_equal(result, expected)

    def test_comparison_with_arithmetic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        evaluator = Evaluator(df)
        # (a + b) > 5
        add = Call("add", (FieldRef("a"), FieldRef("b")))
        node = Call("greater", (add, Literal(5)))
        result = evaluator.evaluate(node)
        expected = (df["a"] + df["b"]) > 5
        tm.assert_series_equal(result, expected)

    def test_complex_filter_expression(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        evaluator = Evaluator(df)
        # (a > 2) & (b < 4)
        left = Call("greater", (FieldRef("a"), Literal(2)))
        right = Call("less", (FieldRef("b"), Literal(4)))
        node = Call("and_", (left, right))
        result = evaluator.evaluate(node)
        expected = (df["a"] > 2) & (df["b"] < 4)
        tm.assert_series_equal(result, expected)
