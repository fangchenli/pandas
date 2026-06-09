"""
Tests for pandas.lazy.frame module - LazyDataFrame class.

This is the main test file for the lazy DataFrame functionality.
"""

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import (
    LazyDataFrame,
    col,
)


class TestDataFrameSelect:
    """Tests for DataFrame.select() entry point."""

    def test_select_returns_lazy_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select()
        assert isinstance(ldf, LazyDataFrame)

    def test_select_all_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select()
        assert ldf.columns == ["a", "b"]

    def test_select_specific_columns_str(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        ldf = df.select("a", "c")
        assert ldf.columns == ["a", "c"]

    def test_select_specific_columns_col(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select(col("a"), col("b"))
        assert ldf.columns == ["a", "b"]

    def test_select_mixed_str_and_col(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("a", col("b"))
        assert ldf.columns == ["a", "b"]

    def test_select_with_alias(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select(col("a").alias("new_a"))
        assert ldf.columns == ["new_a"]

    def test_select_reorder_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ldf = df.select("c", "a", "b")
        assert ldf.columns == ["c", "a", "b"]

    def test_select_duplicate_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select("a", col("a").alias("a_copy"))
        assert ldf.columns == ["a", "a_copy"]

    def test_select_invalid_type(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError, match="Expected Expr or str"):
            df.select(123)


class TestLazyDataFrameProperties:
    """Tests for LazyDataFrame properties."""

    def test_schema(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ldf = df.select()
        schema = ldf.schema
        assert "a" in schema
        assert "b" in schema
        assert schema["a"].is_numeric()
        assert schema["b"].is_string()

    def test_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ldf = df.select()
        assert ldf.columns == ["a", "b", "c"]

    def test_repr(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ldf = df.select()
        result = repr(ldf)
        assert "LazyDataFrame" in result
        assert "columns" in result


class TestLazyDataFrameSelect:
    """Tests for LazyDataFrame.select() method."""

    def test_select_chain(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ldf = df.select().select("a", "b")
        assert ldf.columns == ["a", "b"]

    def test_select_reduces_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ldf = df.select("a", "b", "c").select("a")
        assert ldf.columns == ["a"]

    def test_select_empty_raises(self):
        df = pd.DataFrame({"a": [1]})
        ldf = df.select()
        with pytest.raises(ValueError, match="requires at least one expression"):
            ldf.select()

    def test_select_with_alias_chain(self):
        df = pd.DataFrame({"a": [1]})
        ldf = df.select(col("a").alias("b")).select(col("b").alias("c"))
        assert ldf.columns == ["c"]


class TestLazyDataFrameFilter:
    """Tests for LazyDataFrame.filter() method."""

    def test_filter_preserves_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select().filter(col("a"))
        # Filter doesn't change schema
        assert ldf.columns == ["a", "b"]

    def test_filter_requires_expr(self):
        df = pd.DataFrame({"a": [1]})
        ldf = df.select()
        with pytest.raises(TypeError, match="must be an Expr"):
            ldf.filter("a > 0")


class TestLazyDataFrameExplain:
    """Tests for LazyDataFrame.explain() method."""

    def test_explain_returns_string(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select()
        result = ldf.explain()
        assert isinstance(result, str)

    def test_explain_contains_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ldf = df.select("a", "b")
        result = ldf.explain()
        assert "a" in result
        assert "b" in result

    def test_explain_contains_plan(self):
        df = pd.DataFrame({"a": [1]})
        ldf = df.select()
        result = ldf.explain()
        assert "Plan" in result or "plan" in result.lower()

    def test_explain_shows_project(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ldf = df.select("a")
        result = ldf.explain()
        assert "Project" in result

    def test_explain_shows_source(self):
        df = pd.DataFrame({"a": [1]})
        ldf = df.select()
        result = ldf.explain()
        assert "DataFrameSource" in result

    def test_explain_text_format(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ldf = df.select()
        result = ldf.explain(format="text")
        assert "=" * 50 in result
        assert "LAZY PANDAS QUERY PLAN" in result
        assert "Output columns:" in result
        assert "Plan tree:" in result

    def test_explain_tree_format(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ldf = df.select()
        result = ldf.explain(format="tree")
        assert "LAZY PANDAS QUERY PLAN" in result
        assert "Output columns:" in result
        # Box-drawing characters
        assert "┌─" in result

    def test_explain_tree_format_with_children(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ldf = df.select("a")
        result = ldf.explain(format="tree")
        # Should have connector for child node
        assert "└─" in result or "├─" in result

    def test_explain_tree_format_join(self):
        df1 = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
        df2 = pd.DataFrame({"key": [1, 2], "other": [100, 200]})
        ldf = df1.select().join(df2.select(), on="key")
        result = ldf.explain(format="tree")
        # Join should show both children with proper connectors
        assert "├─" in result
        assert "└─" in result

    def test_explain_tree_format_deep_nesting(self):
        from pandas.lazy import col

        df1 = pd.DataFrame({"key": [1, 2, 3], "a": [10, 20, 30]})
        df2 = pd.DataFrame({"key": [1, 2, 3], "b": [100, 200, 300]})

        # Create deeply nested plan: project -> sort -> join -> (filter, source)
        ldf = (
            df1.select()
            .join(df2.select(), on="key")
            .filter(col("a") > 10)
            .sort("b", descending=True)
        )
        result = ldf.explain(format="tree")

        # Should have vertical connector for nested children
        assert "│" in result
        # Should have proper child connectors
        assert "├─" in result
        assert "└─" in result
        # Should show all operations
        assert "Sort" in result
        assert "Join" in result
        assert "Filter" in result
        assert "DataFrameSource" in result

    def test_explain_json_format(self):
        import json

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ldf = df.select()
        result = ldf.explain(format="json")
        # Should be valid JSON
        parsed = json.loads(result)
        assert "plan_type" in parsed
        assert "output_columns" in parsed
        assert "plan" in parsed

    def test_explain_json_format_plan_type(self):
        import json

        df = pd.DataFrame({"a": [1]})
        ldf = df.select()

        result_opt = json.loads(ldf.explain(format="json", optimized=True))
        assert result_opt["plan_type"] == "optimized"

        result_unopt = json.loads(ldf.explain(format="json", optimized=False))
        assert result_unopt["plan_type"] == "unoptimized"

    def test_explain_json_format_has_type(self):
        import json

        df = pd.DataFrame({"a": [1]})
        ldf = df.select()
        result = json.loads(ldf.explain(format="json"))
        assert "type" in result["plan"]

    def test_explain_json_format_filter(self):
        import json

        from pandas.lazy import col

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().filter(col("a") > 1)
        result = json.loads(ldf.explain(format="json"))
        assert result["plan"]["type"] == "Filter"
        assert "predicate" in result["plan"]

    def test_explain_json_format_join(self):
        import json

        df1 = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
        df2 = pd.DataFrame({"key": [1, 2], "other": [100, 200]})
        ldf = df1.select().join(df2.select(), on="key")
        result = json.loads(ldf.explain(format="json"))
        # Find the Join node (may be nested)
        plan = result["plan"]
        if plan["type"] != "Join":
            plan = plan.get("input", plan)
        if plan["type"] == "Join":
            assert "left" in plan
            assert "right" in plan
            assert plan["on"] == ["key"]
            assert plan["how"] == "inner"

    def test_explain_invalid_format(self):
        import pytest

        df = pd.DataFrame({"a": [1]})
        ldf = df.select()
        with pytest.raises(ValueError, match="Unknown format"):
            ldf.explain(format="invalid")

    def test_explain_default_format_is_text(self):
        df = pd.DataFrame({"a": [1]})
        ldf = df.select()
        # Default should be text format
        result = ldf.explain()
        assert "=" * 50 in result


class TestOptimizationCaching:
    """Tests for optimization plan caching."""

    def test_optimized_plan_starts_as_none(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select()
        assert ldf._optimized_plan is None

    def test_explain_caches_optimized_plan(self):
        from pandas.lazy import col

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().filter(col("a") > 1)

        assert ldf._optimized_plan is None
        ldf.explain()
        assert ldf._optimized_plan is not None

    def test_collect_caches_optimized_plan(self):
        from pandas.lazy import col

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().filter(col("a") > 1)

        assert ldf._optimized_plan is None
        ldf.collect()
        assert ldf._optimized_plan is not None

    def test_cached_plan_reused_across_calls(self):
        from pandas.lazy import col

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().filter(col("a") > 1)

        # First call caches
        ldf.explain()
        plan_id_after_explain = id(ldf._optimized_plan)

        # Second call should reuse cached plan
        ldf.collect()
        plan_id_after_collect = id(ldf._optimized_plan)

        assert plan_id_after_explain == plan_id_after_collect

    def test_unoptimized_explain_does_not_cache(self):
        from pandas.lazy import col

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().filter(col("a") > 1)

        ldf.explain(optimized=False)
        # Should not cache when optimized=False
        assert ldf._optimized_plan is None

    def test_chained_operations_have_separate_caches(self):
        from pandas.lazy import col

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf1 = df.select().filter(col("a") > 1)
        ldf2 = ldf1.filter(col("a") > 2)

        # ldf1 and ldf2 are different objects with separate caches
        ldf1.explain()
        assert ldf1._optimized_plan is not None
        assert ldf2._optimized_plan is None

        ldf2.explain()
        assert ldf2._optimized_plan is not None
        assert id(ldf1._optimized_plan) != id(ldf2._optimized_plan)


class TestLazyDataFrameCollect:
    """Tests for LazyDataFrame.collect() method."""

    def test_collect_returns_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select()
        result = ldf.collect()
        assert isinstance(result, pd.DataFrame)

    def test_collect_all_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select()
        result = ldf.collect()
        tm.assert_frame_equal(result, df)

    def test_collect_subset_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        ldf = df.select("a", "c")
        result = ldf.collect()
        expected = df[["a", "c"]]
        tm.assert_frame_equal(result, expected)

    def test_collect_reordered_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        ldf = df.select("c", "a", "b")
        result = ldf.collect()
        expected = df[["c", "a", "b"]]
        tm.assert_frame_equal(result, expected)

    def test_collect_with_alias(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select(col("a").alias("new_a"))
        result = ldf.collect()
        expected = pd.DataFrame({"new_a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_collect_chained_select(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        ldf = df.select("a", "b", "c").select("a", "b")
        result = ldf.collect()
        expected = df[["a", "b"]]
        tm.assert_frame_equal(result, expected)

    def test_collect_preserves_dtypes(self):
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
            }
        )
        ldf = df.select()
        result = ldf.collect()
        assert result["int_col"].dtype == df["int_col"].dtype
        assert result["float_col"].dtype == df["float_col"].dtype
        assert result["str_col"].dtype == df["str_col"].dtype

    def test_collect_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        ldf = df.select()
        result = ldf.collect()
        tm.assert_frame_equal(result, df)

    def test_collect_single_row(self):
        df = pd.DataFrame({"a": [42], "b": ["hello"]})
        ldf = df.select()
        result = ldf.collect()
        tm.assert_frame_equal(result, df)

    def test_collect_does_not_modify_source(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        original = df.copy()
        ldf = df.select()
        _ = ldf.collect()
        tm.assert_frame_equal(df, original)


class TestLazyDataFrameIntegration:
    """Integration tests for complete lazy workflows."""

    def test_full_workflow(self):
        # Create DataFrame
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "SF"],
            }
        )

        # Enter lazy mode, select columns, collect
        result = df.select("name", "age").collect()

        expected = df[["name", "age"]]
        tm.assert_frame_equal(result, expected)

    def test_explain_then_collect(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select("a")

        # explain() shouldn't affect collect()
        explain_result = ldf.explain()
        assert isinstance(explain_result, str)

        collect_result = ldf.collect()
        expected = df[["a"]]
        tm.assert_frame_equal(collect_result, expected)

    def test_multiple_collects(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select()

        # Multiple collects should return same result
        result1 = ldf.collect()
        result2 = ldf.collect()
        tm.assert_frame_equal(result1, result2)

    def test_rename_column_workflow(self):
        df = pd.DataFrame({"old_name": [1, 2, 3]})

        result = df.select(col("old_name").alias("new_name")).collect()

        expected = pd.DataFrame({"new_name": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)


class TestLazyDataFrameArithmetic:
    """Tests for arithmetic expression execution."""

    def test_add_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.select((col("a") + col("b")).alias("sum")).collect()
        expected = pd.DataFrame({"sum": [5, 7, 9]})
        tm.assert_frame_equal(result, expected)

    def test_add_literal(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select((col("a") + 10).alias("plus_ten")).collect()
        expected = pd.DataFrame({"plus_ten": [11, 12, 13]})
        tm.assert_frame_equal(result, expected)

    def test_subtract(self):
        df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
        result = df.select((col("a") - col("b")).alias("diff")).collect()
        expected = pd.DataFrame({"diff": [9, 18, 27]})
        tm.assert_frame_equal(result, expected)

    def test_multiply(self):
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "qty": [2, 3, 4]})
        result = df.select((col("price") * col("qty")).alias("total")).collect()
        expected = pd.DataFrame({"total": [20.0, 60.0, 120.0]})
        tm.assert_frame_equal(result, expected)

    def test_divide(self):
        df = pd.DataFrame({"a": [10, 20, 30], "b": [2, 4, 5]})
        result = df.select((col("a") / col("b")).alias("ratio")).collect()
        expected = pd.DataFrame({"ratio": [5.0, 5.0, 6.0]})
        tm.assert_frame_equal(result, expected)

    def test_negate(self):
        df = pd.DataFrame({"a": [1, -2, 3]})
        result = df.select((-col("a")).alias("neg")).collect()
        expected = pd.DataFrame({"neg": [-1, 2, -3]})
        tm.assert_frame_equal(result, expected)

    def test_chained_arithmetic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # (a + b) * 2
        result = df.select(((col("a") + col("b")) * 2).alias("result")).collect()
        expected = pd.DataFrame({"result": [10, 14, 18]})
        tm.assert_frame_equal(result, expected)


class TestLazyDataFrameFilterExecution:
    """Tests for filter execution."""

    def test_filter_greater(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = df.select().filter(col("a") > 2).collect()
        expected = pd.DataFrame({"a": [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_filter_equal(self):
        df = pd.DataFrame({"s": ["x", "y", "x", "z", "x"]})
        result = df.select().filter(col("s") == "x").collect()
        expected = pd.DataFrame({"s": ["x", "x", "x"]})
        tm.assert_frame_equal(result, expected)

    def test_filter_with_and(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        result = df.select().filter((col("a") > 2) & (col("b") < 4)).collect()
        # a > 2: [F, F, T, T, T], b < 4: [F, F, T, T, T] -> AND: [F, F, T, T, T]
        expected = pd.DataFrame({"a": [3, 4, 5], "b": [3, 2, 1]})
        tm.assert_frame_equal(result, expected)

    def test_filter_with_or(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = df.select().filter((col("a") < 2) | (col("a") > 4)).collect()
        expected = pd.DataFrame({"a": [1, 5]})
        tm.assert_frame_equal(result, expected)

    def test_filter_preserves_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})
        result = df.select().filter(col("a") >= 2).collect()
        expected = pd.DataFrame({"a": [2, 3], "b": ["y", "z"], "c": [2.2, 3.3]})
        tm.assert_frame_equal(result, expected)

    def test_filter_then_select(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        result = df.select().filter(col("a") > 2).select("b").collect()
        expected = pd.DataFrame({"b": [30, 40, 50]})
        tm.assert_frame_equal(result, expected)


class TestLazyDataFrameWithColumns:
    """Tests for with_columns() method."""

    def test_add_computed_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.select().with_columns((col("a") + col("b")).alias("sum")).collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "sum": [5, 7, 9]})
        tm.assert_frame_equal(result, expected)

    def test_add_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = (
            df.select()
            .with_columns(
                (col("a") + col("b")).alias("sum"),
                (col("a") * col("b")).alias("product"),
            )
            .collect()
        )
        expected = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6], "sum": [5, 7, 9], "product": [4, 10, 18]}
        )
        tm.assert_frame_equal(result, expected)

    def test_replace_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = (
            df.select()
            .with_columns((col("a") * 2).alias("a"))  # Replace "a"
            .collect()
        )
        expected = pd.DataFrame({"b": [4, 5, 6], "a": [2, 4, 6]})
        tm.assert_frame_equal(result, expected)

    def test_with_columns_requires_alias(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="requires .alias"):
            df.select().with_columns(col("a") + 1).collect()

    def test_with_columns_empty_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="requires at least one"):
            df.select().with_columns()


class TestLazyDataFrameComplexWorkflows:
    """Tests for complex end-to-end workflows."""

    def test_filter_then_compute(self):
        df = pd.DataFrame(
            {
                "price": [100, 200, 300, 400],
                "qty": [1, 2, 3, 4],
                "discount": [0.1, 0.0, 0.2, 0.1],
            }
        )
        result = (
            df.select()
            .filter(col("qty") >= 2)
            .with_columns((col("price") * col("qty")).alias("subtotal"))
            .select("qty", "subtotal")
            .collect()
        )
        expected = pd.DataFrame({"qty": [2, 3, 4], "subtotal": [400, 900, 1600]})
        tm.assert_frame_equal(result, expected)

    def test_compute_then_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        result = (
            df.select()
            .with_columns((col("a") + col("b")).alias("sum"))
            .filter(col("sum") > 30)
            .collect()
        )
        expected = pd.DataFrame(
            {"a": [3, 4, 5], "b": [30, 40, 50], "sum": [33, 44, 55]}
        )
        tm.assert_frame_equal(result, expected)

    def test_boolean_column_creation(self):
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        result = (
            df.select().with_columns((col("value") > 25).alias("is_high")).collect()
        )
        expected = pd.DataFrame(
            {"value": [10, 20, 30, 40, 50], "is_high": [False, False, True, True, True]}
        )
        tm.assert_frame_equal(result, expected)


class TestScalarProjection:
    """Regression tests: all-scalar projections must broadcast in the
    eager path instead of raising ValueError."""

    def test_all_scalar_projection_eager(self):
        from pandas.lazy import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select(lit(1).alias("one")).collect()
        assert len(result) == 3
        assert list(result["one"]) == [1, 1, 1]

    def test_all_scalar_projection_matches_physical(self):
        from pandas.lazy import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        eager = df.select(lit(1).alias("one")).collect()
        physical = df.select(lit(1).alias("one")).collect(use_physical_planner=True)
        tm.assert_frame_equal(eager, physical, check_dtype=False)

    def test_mixed_scalar_and_column_projection(self):
        from pandas.lazy import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select(col("a"), lit("x").alias("tag")).collect()
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["tag"]) == ["x", "x", "x"]


class TestConcatValidation:
    """Regression tests: concat() must validate schema compatibility."""

    def test_concat_mismatched_columns_raises(self):
        from pandas.lazy import concat

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        with pytest.raises(ValueError, match="incompatible columns"):
            concat([df1.select("a"), df2.select("b")])

    def test_concat_partial_overlap_raises(self):
        from pandas.lazy import concat

        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [3], "c": [4]})
        with pytest.raises(ValueError, match="incompatible columns"):
            concat([df1.select(), df2.select()])

    def test_concat_compatible_schemas_ok(self):
        from pandas.lazy import concat

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        result = concat([df1.select(), df2.select()]).collect()
        assert list(result["a"]) == [1, 2, 3, 4]


class TestTrivialSliceFastPath:
    """head/tail/slice over pure column selections slices the source
    directly (microseconds) instead of running the engine."""

    def test_head_matches_eager(self):
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        result = df.select().head(5).collect()
        tm.assert_frame_equal(result, df.head(5).reset_index(drop=True))

    def test_tail_and_offset(self):
        df = pd.DataFrame({"a": range(100)})
        tm.assert_frame_equal(
            df.select().tail(4).collect(), df.tail(4).reset_index(drop=True)
        )

    def test_column_selection_and_rename(self):
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        result = df.select("b", "a").head(3).collect()
        tm.assert_frame_equal(result, df[["b", "a"]].head(3).reset_index(drop=True))
        renamed = df.select(col("a").alias("x")).head(2).collect()
        assert list(renamed.columns) == ["x"]
        assert list(renamed["x"]) == [0, 1]

    def test_preserve_index_keeps_labels(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="i"))
        result = df.select().head(2).collect(preserve_index=True)
        assert list(result.index) == [10, 20]

    def test_non_trivial_plans_not_fast_pathed(self):
        # Filters change row membership; the fast path must decline
        df = pd.DataFrame({"a": range(100)})
        result = df.select().filter(col("a") > 50).head(5).collect()
        assert list(result["a"]) == [51, 52, 53, 54, 55]

    def test_source_not_mutated_by_result(self):
        # CoW: mutating the sliced result must not touch the source
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.select().head(2).collect()
        result.loc[0, "a"] = 999
        assert df.loc[0, "a"] == 1


class TestManipulationVerbs:
    """Frame-level verbs added to close the Polars API gap."""

    def _df(self):
        return pd.DataFrame(
            {"a": [1, 2, 3, None], "b": [4.0, 5.0, 6.0, 7.0], "s": ["x", "y", "z", "w"]}
        )

    def c(self, ldf):
        return ldf.collect(use_physical_planner=True)

    def test_rename(self):
        out = self.c(self._df().select().rename({"a": "A", "s": "S"}))
        assert list(out.columns) == ["A", "b", "S"]

    def test_rename_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown columns"):
            self._df().select().rename({"zzz": "x"})

    def test_drop(self):
        out = self.c(self._df().select().drop("s"))
        assert list(out.columns) == ["a", "b"]

    def test_drop_all_raises(self):
        with pytest.raises(ValueError, match="cannot drop all"):
            self._df().select().drop("a", "b", "s")

    def test_drop_nulls_all_and_subset(self):
        assert len(self.c(self._df().select().drop_nulls())) == 3
        assert len(self.c(self._df().select().drop_nulls("b"))) == 4

    def test_fill_null(self):
        out = self.c(self._df().select(col("a")).fill_null(0))
        assert out["a"].tolist() == [1.0, 2.0, 3.0, 0.0]

    def test_cast(self):
        out = self.c(self._df().select().cast({"a": "float64"}))
        assert str(out["a"].dtype) == "float64"

    def test_pipe(self):
        out = self.c(self._df().select().pipe(lambda d: d.filter(col("b") > 4)))
        assert len(out) == 3

    def test_frame_aggregations_match_eager(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        for method in ("sum", "mean", "min", "max", "std", "var", "median", "count"):
            out = self.c(getattr(df.select(), method)())
            assert len(out) == 1
            for c in ("a", "b"):
                expected = getattr(df[c], method)()
                assert out[c].iloc[0] == pytest.approx(expected)

    def test_min_max_count_include_strings(self):
        out = self.c(self._df().select().max())
        assert out["s"].iloc[0] == "z"  # lexical max
        assert self.c(self._df().select().count())["a"].iloc[0] == 3  # null excluded

    def test_unpivot(self):
        wide = pd.DataFrame({"id": [1, 2], "q1": [10, 20], "q2": [30, 40]})
        out = (
            self.c(wide.select().unpivot(on=["q1", "q2"], index=["id"]))
            .sort_values(["id", "variable"])
            .reset_index(drop=True)
        )
        assert list(out.columns) == ["id", "variable", "value"]
        assert out.shape == (4, 3)
        assert out["value"].tolist() == [10, 30, 20, 40]
        # melt is an alias.
        assert self._df().select().melt.__func__ is type(self._df().select()).unpivot
