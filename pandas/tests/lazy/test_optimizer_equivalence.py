"""
Tests for optimizer equivalence.

These tests verify that optimizations don't change query semantics by comparing:
1. Unoptimized eager evaluation vs optimized eager evaluation
2. Eager evaluation vs physical planner output

This catches subtle bugs like:
- ExpressionSimplification breaking NaN/NA semantics
- CSE leaking internal __cse_* columns
- Limit/offset interactions breaking tail() behavior
- Schema corruption after pushdowns/pruning
- Join suffixing + pruning mismatches

Note: Some tests are marked xfail to document known limitations or bugs.
"""

from __future__ import annotations

import numpy as np

import pandas as pd
from pandas import testing as tm
from pandas.lazy import col


def assert_values_equal(left, right, check_dtype=False):
    """Assert two DataFrames have equal values, optionally ignoring dtype."""
    tm.assert_frame_equal(left, right, check_dtype=check_dtype)


class TestEagerVsPhysicalEquivalence:
    """Test that eager and physical planner produce equivalent results."""

    def test_simple_filter(self):
        """Filter should produce same results in both modes."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        ldf = df.select().filter(col("a") > 2)

        # Compare results
        eager_result = ldf.collect(use_physical_planner=False)
        physical_result = ldf.collect(use_physical_planner=True)

        tm.assert_frame_equal(
            eager_result.reset_index(drop=True),
            physical_result.reset_index(drop=True),
        )

    def test_filter_with_na_values(self):
        """Filter with NA values should handle nulls consistently."""
        df = pd.DataFrame(
            {"a": [1, None, 3, None, 5], "b": [10, 20, None, 40, 50]},
            dtype="Int64",
        )

        ldf = df.select().filter(col("a") > 2)

        eager_result = ldf.collect(use_physical_planner=False)
        physical_result = ldf.collect(use_physical_planner=True)

        # Both should filter out NA values in the comparison
        # Note: dtypes may differ (Int64 vs int64) between eager and physical
        assert_values_equal(
            eager_result.reset_index(drop=True),
            physical_result.reset_index(drop=True),
            check_dtype=False,
        )

    def test_groupby_aggregation(self):
        """GroupBy aggregation should produce same results."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        ldf = df.select().group_by("group").agg(col("value").sum().alias("total"))

        eager_result = ldf.collect(use_physical_planner=False)
        physical_result = ldf.collect(use_physical_planner=True)

        # Sort for comparison since groupby order may differ
        # Note: dtypes may differ between eager and physical
        assert_values_equal(
            eager_result.sort_values("group").reset_index(drop=True),
            physical_result.sort_values("group").reset_index(drop=True),
            check_dtype=False,
        )

    def test_multiple_columns_projection(self):
        """Projection order should be preserved."""
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}
        )

        # Select in specific order
        ldf = df.select().select("c", "a", "d")

        eager_result = ldf.collect(use_physical_planner=False)
        physical_result = ldf.collect(use_physical_planner=True)

        assert list(eager_result.columns) == ["c", "a", "d"]
        assert list(physical_result.columns) == ["c", "a", "d"]
        tm.assert_frame_equal(eager_result, physical_result)


class TestOptimizedVsUnoptimized:
    """Test that optimizations don't change query semantics."""

    def _run_with_and_without_optimization(
        self, ldf, passes_to_skip: list[str] | None = None
    ):
        """Run a query with and without specific optimizations."""
        # Get optimized result
        optimized_result = ldf.collect(use_physical_planner=True)

        # For comparison, we can't easily disable optimization in collect(),
        # so we compare against eager which doesn't apply physical optimizations
        unoptimized_result = ldf.collect(use_physical_planner=False)

        return unoptimized_result, optimized_result

    def test_constant_folding_preserves_semantics(self):
        """Constant folding should not change results."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        # Expression that can be constant-folded
        ldf = df.select().with_columns((col("a") + (2 + 3)).alias("result"))

        eager, physical = self._run_with_and_without_optimization(ldf)

        tm.assert_frame_equal(
            eager.reset_index(drop=True),
            physical.reset_index(drop=True),
        )
        # Verify the actual values
        expected = [6, 7, 8, 9, 10]  # a + 5
        assert list(physical["result"]) == expected

    def test_filter_fusion_preserves_semantics(self):
        """Filter fusion should not change results."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        # Multiple filters that could be fused
        ldf = (
            df.select().filter(col("a") > 1).filter(col("a") < 5).filter(col("b") > 15)
        )

        eager, physical = self._run_with_and_without_optimization(ldf)

        tm.assert_frame_equal(
            eager.reset_index(drop=True),
            physical.reset_index(drop=True),
        )

    def test_projection_pruning_preserves_semantics(self):
        """Projection pruning should not change visible results."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
                "d": [10, 11, 12],
                "e": [13, 14, 15],
            }
        )

        # Only use columns a and c
        ldf = df.select().filter(col("a") > 1).select("a", "c")

        eager, physical = self._run_with_and_without_optimization(ldf)

        assert list(eager.columns) == ["a", "c"]
        assert list(physical.columns) == ["a", "c"]
        tm.assert_frame_equal(
            eager.reset_index(drop=True),
            physical.reset_index(drop=True),
        )


class TestNaNNASemantics:
    """Test that optimizations correctly handle NaN and pd.NA values."""

    def test_expression_simplification_with_nan_float(self):
        """ExpressionSimplification must preserve NaN propagation in floats."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})

        # x + 0 should NOT be simplified to x if x can be NaN
        # (In standard IEEE 754, NaN + 0 = NaN, so this is actually safe)
        ldf = df.select().with_columns((col("a") + 0).alias("result"))

        result = ldf.collect(use_physical_planner=True)

        # NaN should propagate - convert to float for comparison
        result_values = result["result"].to_numpy(dtype="float64", na_value=np.nan)
        assert np.isnan(result_values[1])
        assert np.isnan(result_values[3])
        assert result_values[0] == 1.0
        assert result_values[2] == 3.0

    def test_expression_simplification_with_nullable_int(self):
        """ExpressionSimplification must preserve pd.NA propagation."""
        df = pd.DataFrame({"a": pd.array([1, None, 3, None, 5], dtype="Int64")})

        # x * 1 should preserve NA values
        ldf = df.select().with_columns((col("a") * 1).alias("result"))

        result = ldf.collect(use_physical_planner=True)

        # Physical planner may use different types, check values
        result_values = result["result"].to_numpy(dtype="float64", na_value=np.nan)
        assert np.isnan(result_values[1])
        assert np.isnan(result_values[3])
        assert result_values[0] == 1
        assert result_values[2] == 3

    def test_filter_with_nan_comparison(self):
        """Filters should correctly handle NaN in comparisons."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})

        # NaN > 2 should be False (not NaN), so NaN rows are filtered out
        ldf = df.select().filter(col("a") > 2)

        result = ldf.collect(use_physical_planner=True)

        # Only rows with a > 2 (excluding NaN)
        assert len(result) == 2
        assert list(result["a"]) == [3.0, 5.0]

    def test_aggregation_with_nan_values(self):
        """Aggregations should handle NaN correctly."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "value": [1.0, np.nan, 3.0, 4.0],
            }
        )

        ldf = df.select().group_by("group").agg(col("value").sum().alias("total"))

        result = ldf.collect(use_physical_planner=True)
        result = result.sort_values("group").reset_index(drop=True)

        # Group A: 1.0 + NaN = NaN (pandas behavior)
        # Group B: 3.0 + 4.0 = 7.0
        assert np.isnan(result[result["group"] == "A"]["total"].iloc[0])
        assert result[result["group"] == "B"]["total"].iloc[0] == 7.0

    def test_all_na_column_in_filter(self):
        """Filter on all-NA column should return empty result."""
        df = pd.DataFrame(
            {
                "a": pd.array(
                    [None, None, None], dtype="Float64"
                ),  # Use Float64 for NA support
                "b": [1, 2, 3],
            }
        )

        ldf = df.select().filter(col("a") > 0)

        result = ldf.collect(use_physical_planner=True)

        # All NA comparisons are False, so result should be empty
        assert len(result) == 0


class TestOuterJoinNullRejection:
    """Test outer joins with post-join null-rejecting filters."""

    def test_left_join_with_null_rejecting_filter(self):
        """
        Left join + filter on right column should NOT be simplified to inner join
        by optimizer (unless specifically implemented).

        This test verifies current behavior is correct.
        """
        left = pd.DataFrame({"key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [2, 3, 4], "right_val": [20, 30, 40]})

        # Left join will have NULL for right_val where key=1
        ldf = (
            left.select()
            .join(right.select(), on="key", how="left")
            .filter(col("right_val") > 0)  # This rejects nulls
        )

        result = ldf.collect(use_physical_planner=True)

        # After null-rejecting filter, should only have rows where right matched
        assert len(result) == 2
        assert set(result["key"]) == {2, 3}

    def test_outer_join_preserves_nulls_without_filter(self):
        """Outer join should preserve NULL values from non-matching rows."""
        left = pd.DataFrame({"key": [1, 2], "left_val": [10, 20]})
        right = pd.DataFrame({"key": [2, 3], "right_val": [200, 300]})

        ldf = left.select().join(right.select(), on="key", how="outer")

        result = ldf.collect(use_physical_planner=True)

        # Should have 3 rows: key 1 (left only), 2 (both), 3 (right only)
        assert len(result) == 3

    def test_right_join_with_filter_on_left_column(self):
        """Right join + filter on left column that rejects nulls."""
        left = pd.DataFrame({"key": [1, 2], "left_val": [10, 20]})
        right = pd.DataFrame({"key": [2, 3, 4], "right_val": [200, 300, 400]})

        ldf = (
            left.select()
            .join(right.select(), on="key", how="right")
            .filter(col("left_val") > 0)
        )

        result = ldf.collect(use_physical_planner=True)

        # Only key=2 has non-null left_val
        assert len(result) == 1
        assert result["key"].iloc[0] == 2


class TestCSEColumnLeakage:
    """Test that CSE doesn't leak internal __cse_* columns to output."""

    def test_cse_columns_not_in_output(self):
        """CSE temporary columns should never appear in final output."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        # Create duplicate expressions that CSE might optimize
        ldf = df.select().with_columns(
            (col("a") * 2).alias("double1"),
            (col("a") * 2).alias("double2"),  # Duplicate expression
            (col("a") * 2 + 1).alias("double_plus_one"),
        )

        result = ldf.collect(use_physical_planner=True)

        # Check no __cse_* columns leaked
        for col_name in result.columns:
            assert not col_name.startswith("__cse_"), (
                f"CSE column '{col_name}' leaked to output"
            )

        # Verify expected columns
        assert set(result.columns) == {"a", "double1", "double2", "double_plus_one"}

    def test_cse_with_nested_expressions(self):
        """CSE with nested common subexpressions."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # (x + y) appears multiple times
        ldf = df.select().with_columns(
            (col("x") + col("y")).alias("sum1"),
            (col("x") + col("y")).alias("sum2"),
            ((col("x") + col("y")) * 2).alias("sum_doubled"),
        )

        result = ldf.collect(use_physical_planner=True)

        # No CSE columns in output
        for col_name in result.columns:
            assert not col_name.startswith("__cse_")

        # Correct values
        assert list(result["sum1"]) == [5, 7, 9]
        assert list(result["sum2"]) == [5, 7, 9]
        assert list(result["sum_doubled"]) == [10, 14, 18]

    def test_cse_after_filter(self):
        """CSE should work correctly after filter operations."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        ldf = (
            df.select()
            .filter(col("a") > 2)
            .with_columns(
                (col("a") + col("b")).alias("sum1"),
                (col("a") + col("b")).alias("sum2"),
            )
        )

        result = ldf.collect(use_physical_planner=True)

        # No CSE columns leaked
        assert "__cse_" not in str(result.columns)
        assert set(result.columns) == {"a", "b", "sum1", "sum2"}


class TestLimitOffsetInteractions:
    """Test that Limit/offset optimizations don't break tail() behavior."""

    def test_head_basic(self):
        """head() should return first N rows."""
        df = pd.DataFrame({"a": list(range(100))})

        ldf = df.select().head(10)
        result = ldf.collect(use_physical_planner=True)

        assert len(result) == 10
        assert list(result["a"]) == list(range(10))

    def test_tail_basic(self):
        """tail() should return last N rows."""
        df = pd.DataFrame({"a": list(range(100))})

        ldf = df.select().tail(10)
        result = ldf.collect(use_physical_planner=True)

        assert len(result) == 10
        assert list(result["a"]) == list(range(90, 100))

    def test_head_after_filter(self):
        """head() after filter should limit filtered results."""
        df = pd.DataFrame({"a": list(range(100))})

        ldf = df.select().filter(col("a") >= 50).head(5)
        result = ldf.collect(use_physical_planner=True)

        assert len(result) == 5
        assert list(result["a"]) == [50, 51, 52, 53, 54]

    def test_tail_after_filter(self):
        """tail() after filter should return last N of filtered results."""
        df = pd.DataFrame({"a": list(range(100))})

        ldf = df.select().filter(col("a") < 50).tail(5)
        result = ldf.collect(use_physical_planner=True)

        assert len(result) == 5
        assert list(result["a"]) == [45, 46, 47, 48, 49]

    def test_head_then_tail(self):
        """head() then tail() should work correctly."""
        df = pd.DataFrame({"a": list(range(100))})

        ldf = df.select().head(50).tail(10)
        result = ldf.collect(use_physical_planner=True)

        assert len(result) == 10
        # Last 10 of first 50 = rows 40-49
        assert list(result["a"]) == list(range(40, 50))

    def test_tail_then_head(self):
        """tail() then head() should work correctly."""
        df = pd.DataFrame({"a": list(range(100))})

        ldf = df.select().tail(50).head(10)
        result = ldf.collect(use_physical_planner=True)

        assert len(result) == 10
        # First 10 of last 50 = rows 50-59
        assert list(result["a"]) == list(range(50, 60))

    def test_limit_pushdown_doesnt_break_tail(self):
        """LimitPushdown optimization shouldn't incorrectly transform tail."""
        df = pd.DataFrame({"a": list(range(20)), "b": list(range(20, 40))})

        # Filter then tail - limit pushdown shouldn't push limit before filter
        ldf = df.select().filter(col("a") > 10).tail(3)
        result = ldf.collect(use_physical_planner=True)

        # Should get last 3 rows where a > 10
        # a > 10 gives: 11,12,13,14,15,16,17,18,19
        # tail(3) gives: 17,18,19
        assert len(result) == 3
        assert list(result["a"]) == [17, 18, 19]


class TestSchemaPushdownCorrectness:
    """Test schema correctness after pushdowns and pruning."""

    def test_projection_pruning_preserves_schema_order(self):
        """Projection pruning should preserve column order."""
        df = pd.DataFrame(
            {"z": [1, 2, 3], "y": [4, 5, 6], "x": [7, 8, 9], "w": [10, 11, 12]}
        )

        # Select in specific order
        ldf = df.select().select("y", "w", "z")
        result = ldf.collect(use_physical_planner=True)

        assert list(result.columns) == ["y", "w", "z"]

    def test_filter_pushdown_preserves_schema(self):
        """Filter pushdown should not alter output schema."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": ["x", "y", "z", "w", "v"],
            }
        )

        ldf = df.select().select("a", "c").filter(col("a") > 2)
        result = ldf.collect(use_physical_planner=True)

        # Schema should be [a, c] not [a, b, c]
        assert list(result.columns) == ["a", "c"]

    def test_join_suffix_with_pruning(self):
        """Join suffixing combined with pruning should maintain correct schema."""
        left = pd.DataFrame({"key": [1, 2, 3], "value": [10, 20, 30]})
        right = pd.DataFrame({"key": [2, 3, 4], "value": [200, 300, 400]})

        # Join creates value_x and value_y, then we select specific columns
        ldf = (
            left.select()
            .join(right.select(), on="key", how="inner")
            .select("key", "value_x")
        )

        result = ldf.collect(use_physical_planner=True)

        # Should only have key and value_x
        assert list(result.columns) == ["key", "value_x"]
        assert len(result) == 2  # keys 2 and 3 match

    def test_aggregation_schema_after_pruning(self):
        """Aggregation output schema should be correct after pruning."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "val1": [1, 2, 3, 4],
                "val2": [10, 20, 30, 40],
            }
        )

        ldf = (
            df.select()
            .group_by("group")
            .agg(
                col("val1").sum().alias("sum1"),
                col("val2").mean().alias("mean2"),
            )
            .select("group", "sum1")  # Prune mean2
        )

        result = ldf.collect(use_physical_planner=True)

        # Schema should be [group, sum1]
        assert list(result.columns) == ["group", "sum1"]

    def test_wide_dataframe_pruning_schema(self):
        """Schema correctness with wide DataFrame and aggressive pruning."""
        # Create wide DataFrame
        data = {f"col_{i}": list(range(10)) for i in range(50)}
        df = pd.DataFrame(data)

        # Select just a few columns
        ldf = df.select().select("col_5", "col_25", "col_45")
        result = ldf.collect(use_physical_planner=True)

        assert list(result.columns) == ["col_5", "col_25", "col_45"]
        assert len(result) == 10


class TestComplexQueryEquivalence:
    """Test equivalence for complex multi-operation queries."""

    def test_full_pipeline_equivalence(self):
        """Complex pipeline should produce same results across execution modes."""
        df = pd.DataFrame(
            {
                "region": ["North", "South", "North", "South", "East"] * 20,
                "product": ["A", "B", "A", "B", "A"] * 20,
                "sales": np.random.default_rng(42).integers(100, 1000, 100),
                "quantity": np.random.default_rng(42).integers(1, 50, 100),
            }
        )

        ldf = (
            df.select()
            .filter(col("sales") > 200)
            .with_columns((col("sales") * col("quantity")).alias("revenue"))
            .group_by("region", "product")
            .agg(
                col("revenue").sum().alias("total_revenue"),
                col("quantity").mean().alias("avg_quantity"),
            )
            .filter(col("total_revenue") > 10000)
            .sort("total_revenue", descending=True)
        )

        eager_result = ldf.collect(use_physical_planner=False)
        physical_result = ldf.collect(use_physical_planner=True)

        # Sort both for comparison
        eager_sorted = eager_result.sort_values(
            ["total_revenue", "region", "product"]
        ).reset_index(drop=True)
        physical_sorted = physical_result.sort_values(
            ["total_revenue", "region", "product"]
        ).reset_index(drop=True)

        # Dtypes may differ between eager and physical planners
        assert_values_equal(eager_sorted, physical_sorted, check_dtype=False)

    def test_join_then_aggregate_equivalence(self):
        """Join followed by aggregation should produce consistent results."""
        orders = pd.DataFrame(
            {
                "order_id": range(100),
                "customer_id": np.random.default_rng(42).integers(1, 20, 100),
                "amount": np.random.default_rng(42).uniform(10, 500, 100),
            }
        )

        customers = pd.DataFrame(
            {
                "customer_id": range(1, 21),
                "region": ["North", "South", "East", "West"] * 5,
            }
        )

        ldf = (
            orders.select()
            .join(customers.select(), on="customer_id", how="inner")
            .group_by("region")
            .agg(
                col("amount").sum().alias("total_sales"),
                col("order_id").count().alias("num_orders"),
            )
        )

        eager_result = ldf.collect(use_physical_planner=False)
        physical_result = ldf.collect(use_physical_planner=True)

        # Sort for comparison
        eager_sorted = eager_result.sort_values("region").reset_index(drop=True)
        physical_sorted = physical_result.sort_values("region").reset_index(drop=True)

        tm.assert_frame_equal(eager_sorted, physical_sorted)
