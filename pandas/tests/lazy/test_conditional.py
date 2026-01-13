"""
Tests for pandas.lazy conditional expressions (when/then/otherwise).
"""

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import (
    col,
    when,
)
from pandas.lazy.expr import (
    Then,
    When,
)
from pandas.lazy.ir import Call


class TestWhenThenOtherwiseAPI:
    """Tests for the when/then/otherwise expression API."""

    def test_when_returns_when_object(self):
        result = when(col("a") > 0)
        assert isinstance(result, When)

    def test_when_then_returns_then_object(self):
        result = when(col("a") > 0).then(1)
        assert isinstance(result, Then)

    def test_when_then_otherwise_returns_expr(self):
        from pandas.lazy.expr import Expr

        result = when(col("a") > 0).then(1).otherwise(0)
        assert isinstance(result, Expr)

    def test_when_requires_expr(self):
        with pytest.raises(TypeError, match="condition must be an Expr"):
            when("a > 0")

    def test_chained_when_requires_expr(self):
        with pytest.raises(TypeError, match="condition must be an Expr"):
            when(col("a") > 0).then(1).when("b > 0")

    def test_ir_structure_simple(self):
        expr = when(col("a") > 0).then(1).otherwise(0)
        ir = expr._ir
        assert isinstance(ir, Call)
        assert ir.function == "case_when"
        assert "cases" in ir.kwargs
        assert "otherwise" in ir.kwargs

    def test_ir_structure_multiple_conditions(self):
        expr = (
            when(col("a") > 100)
            .then("high")
            .when(col("a") > 50)
            .then("medium")
            .otherwise("low")
        )
        ir = expr._ir
        assert isinstance(ir, Call)
        assert ir.function == "case_when"
        assert len(ir.kwargs["cases"]) == 2


class TestWhenThenOtherwiseEvaluation:
    """Tests for evaluating when/then/otherwise expressions."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1, -2, 3, -4, 0],
                "b": [10, 20, 30, 40, 50],
                "category": ["x", "y", "x", "y", "x"],
            }
        )

    def test_simple_positive_negative(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(
                when(col("a") > 0)
                .then("positive")
                .otherwise("non-positive")
                .alias("sign")
            )
            .collect()
        )
        expected = sample_df.copy()
        expected["sign"] = [
            "positive",
            "non-positive",
            "positive",
            "non-positive",
            "non-positive",
        ]
        tm.assert_frame_equal(result, expected)

    def test_numeric_result(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(when(col("a") > 0).then(1).otherwise(-1).alias("sign"))
            .collect()
        )
        expected = sample_df.copy()
        expected["sign"] = [1, -1, 1, -1, -1]
        tm.assert_frame_equal(result, expected)

    def test_column_reference_in_then(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(
                when(col("a") > 0).then(col("b")).otherwise(0).alias("result")
            )
            .collect()
        )
        expected = sample_df.copy()
        expected["result"] = [10, 0, 30, 0, 0]
        tm.assert_frame_equal(result, expected)

    def test_column_reference_in_otherwise(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(
                when(col("a") > 0).then(100).otherwise(col("b")).alias("result")
            )
            .collect()
        )
        expected = sample_df.copy()
        expected["result"] = [100, 20, 100, 40, 50]
        tm.assert_frame_equal(result, expected)

    def test_multiple_conditions(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(
                when(col("a") > 0)
                .then("positive")
                .when(col("a") < 0)
                .then("negative")
                .otherwise("zero")
                .alias("sign")
            )
            .collect()
        )
        expected = sample_df.copy()
        expected["sign"] = ["positive", "negative", "positive", "negative", "zero"]
        tm.assert_frame_equal(result, expected)

    def test_three_conditions(self):
        df = pd.DataFrame({"score": [95, 75, 55, 35, 85]})
        result = (
            df.select()
            .with_columns(
                when(col("score") >= 90)
                .then("A")
                .when(col("score") >= 70)
                .then("B")
                .when(col("score") >= 50)
                .then("C")
                .otherwise("F")
                .alias("grade")
            )
            .collect()
        )
        expected = df.copy()
        expected["grade"] = ["A", "B", "C", "F", "B"]
        tm.assert_frame_equal(result, expected)

    def test_with_arithmetic(self, sample_df):
        result = (
            sample_df.select()
            .with_columns(
                when(col("a") > 0)
                .then(col("a") * 2)
                .otherwise(col("a"))
                .alias("doubled_positive")
            )
            .collect()
        )
        expected = sample_df.copy()
        expected["doubled_positive"] = [2, -2, 6, -4, 0]
        tm.assert_frame_equal(result, expected)

    def test_in_filter_context(self, sample_df):
        # Use conditional in filter (compare result to a value)
        result = (
            sample_df.select()
            .with_columns(when(col("a") > 0).then(1).otherwise(0).alias("flag"))
            .filter(col("flag") == 1)
            .collect()
        )
        expected = sample_df[sample_df["a"] > 0].copy().reset_index(drop=True)
        expected["flag"] = 1
        tm.assert_frame_equal(result, expected)


class TestWhenThenOtherwiseTypeInference:
    """Tests for type inference of conditional expressions."""

    def test_numeric_type_inference(self):
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"a": LazyDtype("numeric", None, None, False)})
        expr = when(col("a") > 0).then(1).otherwise(0)
        dtype = infer_expr_dtype(expr._ir, schema)
        assert dtype.category == "numeric"

    def test_string_type_inference(self):
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"a": LazyDtype("numeric", None, None, False)})
        expr = when(col("a") > 0).then("yes").otherwise("no")
        dtype = infer_expr_dtype(expr._ir, schema)
        assert dtype.category == "string"

    def test_nullable_output(self):
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
            infer_expr_dtype,
        )

        schema = Schema({"a": LazyDtype("numeric", None, None, False)})
        expr = when(col("a") > 0).then(1).otherwise(0)
        dtype = infer_expr_dtype(expr._ir, schema)
        # Conditionals are marked nullable
        assert dtype.nullable is True


class TestWhenThenOtherwiseEdgeCases:
    """Tests for edge cases in conditional expressions."""

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        result = (
            df.select()
            .with_columns(when(col("a") > 0).then(1).otherwise(0).alias("flag"))
            .collect()
        )
        # For empty DataFrames, verify structure is correct
        # (dtype may vary for empty series depending on pandas version)
        assert len(result) == 0
        assert "flag" in result.columns
        assert "a" in result.columns

    def test_all_true_condition(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = (
            df.select()
            .with_columns(
                when(col("a") > 0).then("positive").otherwise("negative").alias("sign")
            )
            .collect()
        )
        expected = df.copy()
        expected["sign"] = ["positive"] * 5
        tm.assert_frame_equal(result, expected)

    def test_all_false_condition(self):
        df = pd.DataFrame({"a": [-1, -2, -3, -4, -5]})
        result = (
            df.select()
            .with_columns(
                when(col("a") > 0).then("positive").otherwise("negative").alias("sign")
            )
            .collect()
        )
        expected = df.copy()
        expected["sign"] = ["negative"] * 5
        tm.assert_frame_equal(result, expected)

    def test_with_null_values(self):
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})
        result = (
            df.select()
            .with_columns(
                when(col("a").is_not_null())
                .then(col("a"))
                .otherwise(-1)
                .alias("filled")
            )
            .collect()
        )
        expected = df.copy()
        expected["filled"] = [1.0, -1.0, 3.0, -1.0, 5.0]
        tm.assert_frame_equal(result, expected)

    def test_boolean_result(self):
        df = pd.DataFrame({"a": [1, -2, 3, -4, 0]})
        result = (
            df.select()
            .with_columns(
                when(col("a") > 0).then(True).otherwise(False).alias("is_positive")
            )
            .collect()
        )
        expected = df.copy()
        expected["is_positive"] = [True, False, True, False, False]
        tm.assert_frame_equal(result, expected)


class TestDirectEvaluatorConditional:
    """Tests for conditional operations directly in the evaluator."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1, -2, 3, -4, 0],
                "b": [10, 20, 30, 40, 50],
            }
        )

    def test_evaluator_case_when_simple(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        evaluator = Evaluator(sample_df)

        # Build the IR manually
        condition = Call("greater", (FieldRef("a"), Literal(0)))
        then_value = Literal(1)
        otherwise_value = Literal(0)

        node = Call(
            "case_when",
            (),
            {"cases": ((condition, then_value),), "otherwise": otherwise_value},
        )

        result = evaluator.evaluate(node)
        expected = pd.Series([1, 0, 1, 0, 0], index=sample_df.index)
        tm.assert_series_equal(result, expected)

    def test_evaluator_case_when_with_column(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        evaluator = Evaluator(sample_df)

        condition = Call("greater", (FieldRef("a"), Literal(0)))
        then_value = FieldRef("b")
        otherwise_value = Literal(-1)

        node = Call(
            "case_when",
            (),
            {"cases": ((condition, then_value),), "otherwise": otherwise_value},
        )

        result = evaluator.evaluate(node)
        expected = pd.Series([10, -1, 30, -1, -1], index=sample_df.index)
        tm.assert_series_equal(result, expected)

    def test_evaluator_case_when_multiple_conditions(self, sample_df):
        from pandas.lazy.eval import Evaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        evaluator = Evaluator(sample_df)

        # when a > 0 then "positive"
        # when a < 0 then "negative"
        # otherwise "zero"
        cond1 = Call("greater", (FieldRef("a"), Literal(0)))
        cond2 = Call("less", (FieldRef("a"), Literal(0)))

        node = Call(
            "case_when",
            (),
            {
                "cases": (
                    (cond1, Literal("positive")),
                    (cond2, Literal("negative")),
                ),
                "otherwise": Literal("zero"),
            },
        )

        result = evaluator.evaluate(node)
        expected = pd.Series(
            ["positive", "negative", "positive", "negative", "zero"],
            index=sample_df.index,
        )
        tm.assert_series_equal(result, expected)
