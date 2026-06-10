"""
Tests for pandas.lazy.expr module - Expression building API.
"""

import pytest

from pandas.lazy.expr import (
    Expr,
    col,
    extract_output_name,
    lit,
    normalize_exprs,
)
from pandas.lazy.ir import (
    Alias,
    Call,
    FieldRef,
    Literal,
)


class TestCol:
    """Tests for col() function."""

    def test_col_creates_expr(self):
        expr = col("column_name")
        assert isinstance(expr, Expr)

    def test_col_creates_field_ref(self):
        expr = col("column_name")
        assert isinstance(expr._ir, FieldRef)
        assert expr._ir.name == "column_name"

    def test_col_requires_string(self):
        with pytest.raises(TypeError, match="must be a string"):
            col(123)

    def test_col_empty_string(self):
        # Empty string is technically allowed
        expr = col("")
        assert expr._ir.name == ""


class TestLit:
    """Tests for lit() function."""

    def test_lit_int(self):
        expr = lit(42)
        assert isinstance(expr, Expr)
        assert isinstance(expr._ir, Literal)
        assert expr._ir.value == 42

    def test_lit_float(self):
        expr = lit(3.14)
        assert expr._ir.value == 3.14

    def test_lit_string(self):
        expr = lit("hello")
        assert expr._ir.value == "hello"

    def test_lit_bool(self):
        expr = lit(True)
        assert expr._ir.value is True

    def test_lit_none(self):
        expr = lit(None)
        assert expr._ir.value is None


class TestExpr:
    """Tests for Expr class."""

    def test_ir_property(self):
        expr = col("a")
        assert expr._ir == expr._node

    def test_alias(self):
        expr = col("a").alias("new_name")
        assert isinstance(expr._ir, Alias)
        assert expr._ir.name == "new_name"

    def test_alias_preserves_inner(self):
        expr = col("a").alias("new_name")
        assert isinstance(expr._ir.arg, FieldRef)
        assert expr._ir.arg.name == "a"

    def test_alias_requires_string(self):
        expr = col("a")
        with pytest.raises(TypeError, match="must be a string"):
            expr.alias(123)

    def test_chained_alias(self):
        expr = col("a").alias("b").alias("c")
        # Outer alias should be "c"
        assert expr._ir.name == "c"
        # Inner alias should be "b"
        assert expr._ir.arg.name == "b"

    def test_repr(self):
        expr = col("a")
        result = repr(expr)
        assert "Expr" in result
        assert "FieldRef" in result

    def test_equality_returns_expr(self):
        """__eq__ returns an Expr for expression comparison, not bool."""
        expr1 = col("a")
        expr2 = col("b")
        result = expr1 == expr2
        # __eq__ returns an Expr, not bool
        assert isinstance(result, Expr)
        assert isinstance(result._ir, Call)
        assert result._ir.function == "equal"

    def test_inequality_returns_expr(self):
        """__ne__ returns an Expr for expression comparison."""
        expr1 = col("a")
        result = expr1 != 5
        assert isinstance(result, Expr)
        assert result._ir.function == "not_equal"

    def test_unhashable(self):
        """Expr is unhashable since __eq__ returns Expr."""
        expr = col("a")
        with pytest.raises(TypeError, match="unhashable"):
            hash(expr)

    def test_ir_identity_comparison(self):
        """Use _ir for identity comparison of underlying IR nodes."""
        expr1 = col("a")
        expr2 = col("a")
        expr3 = col("b")
        # IR nodes can be compared directly for identity
        assert expr1._ir == expr2._ir
        assert expr1._ir != expr3._ir


class TestExtractOutputName:
    """Tests for extract_output_name function."""

    def test_field_ref(self):
        expr = col("column_name")
        assert extract_output_name(expr) == "column_name"

    def test_alias(self):
        expr = col("a").alias("new_name")
        assert extract_output_name(expr) == "new_name"

    def test_nested_alias(self):
        expr = col("a").alias("b").alias("c")
        # Should return outermost alias
        assert extract_output_name(expr) == "c"

    def test_literal_raises(self):
        expr = lit(42)
        with pytest.raises(ValueError, match="Cannot determine output name"):
            extract_output_name(expr)


class TestNormalizeExprs:
    """Tests for normalize_exprs function."""

    def test_string_to_col(self):
        result = normalize_exprs(("a", "b"))
        assert len(result) == 2
        assert all(isinstance(e, Expr) for e in result)
        assert result[0]._ir.name == "a"
        assert result[1]._ir.name == "b"

    def test_expr_unchanged(self):
        expr = col("a")
        result = normalize_exprs((expr,))
        assert result[0] is expr

    def test_mixed(self):
        expr = col("a")
        result = normalize_exprs((expr, "b"))
        assert len(result) == 2
        assert result[0] is expr
        assert result[1]._ir.name == "b"

    def test_empty(self):
        result = normalize_exprs(())
        assert result == ()

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Expected Expr or str"):
            normalize_exprs((123,))

    def test_list_invalid(self):
        # Lists should not be accepted (must be individual args)
        with pytest.raises(TypeError, match="Expected Expr or str"):
            normalize_exprs(([1, 2, 3],))


class TestArithmeticOperators:
    """Tests for arithmetic operators on Expr."""

    def test_add_expr_expr(self):
        result = col("a") + col("b")
        assert isinstance(result, Expr)
        assert isinstance(result._ir, Call)
        assert result._ir.function == "add"
        assert len(result._ir.args) == 2

    def test_add_expr_literal(self):
        result = col("a") + 1
        assert result._ir.function == "add"
        # Second arg should be wrapped in Literal
        assert isinstance(result._ir.args[1], Literal)
        assert result._ir.args[1].value == 1

    def test_radd(self):
        result = 1 + col("a")
        assert result._ir.function == "add"
        # First arg should be the literal
        assert isinstance(result._ir.args[0], Literal)
        assert result._ir.args[0].value == 1

    def test_sub(self):
        result = col("a") - col("b")
        assert result._ir.function == "subtract"

    def test_rsub(self):
        result = 10 - col("a")
        assert result._ir.function == "subtract"
        assert result._ir.args[0].value == 10

    def test_mul(self):
        result = col("a") * col("b")
        assert result._ir.function == "multiply"

    def test_rmul(self):
        result = 2 * col("a")
        assert result._ir.function == "multiply"
        assert result._ir.args[0].value == 2

    def test_truediv(self):
        result = col("a") / col("b")
        assert result._ir.function == "divide"

    def test_rtruediv(self):
        result = 1 / col("a")
        assert result._ir.function == "divide"
        assert result._ir.args[0].value == 1

    def test_floordiv(self):
        result = col("a") // col("b")
        assert result._ir.function == "floor_divide"

    def test_rfloordiv(self):
        result = 10 // col("a")
        assert result._ir.function == "floor_divide"
        assert result._ir.args[0].value == 10

    def test_mod(self):
        result = col("a") % col("b")
        assert result._ir.function == "modulo"

    def test_rmod(self):
        result = 10 % col("a")
        assert result._ir.function == "modulo"
        assert result._ir.args[0].value == 10

    def test_pow(self):
        result = col("a") ** 2
        assert result._ir.function == "power"
        assert result._ir.args[1].value == 2

    def test_rpow(self):
        result = 2 ** col("a")
        assert result._ir.function == "power"
        assert result._ir.args[0].value == 2

    def test_chained_arithmetic(self):
        result = (col("a") + col("b")) * 2
        assert result._ir.function == "multiply"
        # First arg is the add call
        inner = result._ir.args[0]
        assert isinstance(inner, Call)
        assert inner.function == "add"


class TestUnaryOperators:
    """Tests for unary operators on Expr."""

    def test_neg(self):
        result = -col("a")
        assert isinstance(result, Expr)
        assert result._ir.function == "negate"
        assert len(result._ir.args) == 1

    def test_pos(self):
        expr = col("a")
        result = +expr
        # Positive is a no-op, returns same expr
        assert result is expr

    def test_abs(self):
        result = abs(col("a"))
        assert result._ir.function == "abs"
        assert len(result._ir.args) == 1


class TestComparisonOperators:
    """Tests for comparison operators on Expr."""

    def test_eq(self):
        result = col("a") == col("b")
        assert isinstance(result, Expr)
        assert result._ir.function == "equal"

    def test_eq_with_literal(self):
        result = col("a") == 5
        assert result._ir.function == "equal"
        assert result._ir.args[1].value == 5

    def test_ne(self):
        result = col("a") != col("b")
        assert result._ir.function == "not_equal"

    def test_lt(self):
        result = col("a") < col("b")
        assert result._ir.function == "less"

    def test_lt_with_literal(self):
        result = col("a") < 10
        assert result._ir.function == "less"
        assert result._ir.args[1].value == 10

    def test_le(self):
        result = col("a") <= col("b")
        assert result._ir.function == "less_equal"

    def test_gt(self):
        result = col("a") > col("b")
        assert result._ir.function == "greater"

    def test_gt_with_literal(self):
        result = col("a") > 0
        assert result._ir.function == "greater"
        assert result._ir.args[1].value == 0

    def test_ge(self):
        result = col("a") >= col("b")
        assert result._ir.function == "greater_equal"


class TestLogicalOperators:
    """Tests for logical operators on Expr."""

    def test_and(self):
        result = (col("a") > 0) & (col("b") < 10)
        assert isinstance(result, Expr)
        assert result._ir.function == "and_"
        # Both args should be comparison calls
        assert result._ir.args[0].function == "greater"
        assert result._ir.args[1].function == "less"

    def test_rand(self):
        # This tests reverse and, though less common
        result = True & (col("a") > 0)
        assert result._ir.function == "and_"

    def test_or(self):
        result = (col("a") > 0) | (col("b") < 10)
        assert result._ir.function == "or_"

    def test_ror(self):
        result = False | (col("a") > 0)
        assert result._ir.function == "or_"

    def test_invert(self):
        result = ~(col("a") > 0)
        assert result._ir.function == "invert"
        assert len(result._ir.args) == 1
        # Inner should be the comparison
        assert result._ir.args[0].function == "greater"

    def test_complex_logical(self):
        # ((a > 0) & (b < 10)) | (c == 5)
        result = ((col("a") > 0) & (col("b") < 10)) | (col("c") == 5)
        assert result._ir.function == "or_"
        # First arg is the and_ call
        assert result._ir.args[0].function == "and_"
        # Second arg is the equal call
        assert result._ir.args[1].function == "equal"


class TestExpressionComposition:
    """Tests for complex expression composition."""

    def test_arithmetic_then_comparison(self):
        # (a + b) > 10
        result = (col("a") + col("b")) > 10
        assert result._ir.function == "greater"
        assert result._ir.args[0].function == "add"
        assert result._ir.args[1].value == 10

    def test_comparison_then_alias(self):
        result = (col("a") > 0).alias("is_positive")
        assert isinstance(result._ir, Alias)
        assert result._ir.name == "is_positive"
        assert result._ir.arg.function == "greater"

    def test_arithmetic_with_alias(self):
        result = (col("a") + col("b")).alias("sum")
        assert isinstance(result._ir, Alias)
        assert result._ir.name == "sum"
        assert result._ir.arg.function == "add"

    def test_filter_expression(self):
        # Typical filter: (a > 0) & (b != "x")
        result = (col("a") > 0) & (col("b") != "x")
        assert result._ir.function == "and_"

    def test_computed_column(self):
        # Typical computed column: (price * quantity).alias("total")
        result = (col("price") * col("quantity")).alias("total")
        assert isinstance(result._ir, Alias)
        assert result._ir.name == "total"
        assert result._ir.arg.function == "multiply"


class TestEngineParityRobustness:
    """Eager↔physical parity for Expr ops + clear errors (robustness pass).

    Regressions: shift/clip/diff were implemented in the physical evaluator but
    raised NotImplementedError in the eager one; rank diverged in dtype (int vs
    pandas' float); duplicate column labels crashed deep in execution with a
    cryptic AttributeError.
    """

    def _df(self):
        import pandas as pd

        return pd.DataFrame({"x": [1.0, -2.0, 3.0, 4.0]})

    @pytest.mark.parametrize(
        "build",
        [
            lambda: col("x").shift(1),
            lambda: col("x").shift(-1, fill_value=0),
            lambda: col("x").clip(0, 3),
            lambda: col("x").clip(0, None),
            lambda: col("x").diff(),
        ],
    )
    def test_eager_physical_parity(self, build):
        import numpy as np

        df = self._df()
        plan = df.select().select(build().alias("r"))
        phys = plan.collect(use_physical_planner=True)["r"]
        eager = plan.collect(use_physical_planner=False)["r"]
        assert str(phys.dtype) == str(eager.dtype)
        assert np.allclose(
            phys.to_numpy(dtype="float64"),
            eager.to_numpy(dtype="float64"),
            equal_nan=True,
        )

    def test_duplicate_labels_raise_clearly(self):
        import pandas as pd

        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        with pytest.raises(NotImplementedError, match="duplicate column"):
            df.select()
