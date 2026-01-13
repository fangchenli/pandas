"""
Integration tests for the lazy pandas pipeline.

These tests verify the full pipeline:
    Logical Plan → Optimizer → Physical Plan → Execution

Unlike unit tests that test individual components, integration tests ensure
that all components work together correctly and that optimizations don't
break correctness.

Test Categories:
1. Correctness: Optimized == Unoptimized == Eager
2. Plan Structure: Verify optimizer produces expected plan shapes
3. Backend Selection: Verify correct backend choices
4. Edge Cases: Empty data, nulls, type coercions
5. Complex Pipelines: Multi-step real-world queries

Note on Index Handling:
    Lazy execution resets indices to RangeIndex(0, n) while eager pandas
    preserves original indices. Tests use check_index=False or reset_index()
    to account for this design decision.
"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import (
    col,
    lit,
    when,
)
from pandas.lazy.optimize import (
    ConstantFolding,
    FilterFusion,
    LimitPushdown,
    Optimizer,
    PredicatePushdown,
    SortLimitToTopK,
)
from pandas.lazy.physical import (
    PhysicalFilter,
    PhysicalPlanner,
    PhysicalProject,
    PhysicalScan,
    PhysicalTopK,
    execute_physical_plan,
)
from pandas.lazy.plan import (
    Filter,
    Limit,
    Sort,
    TopK,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_df():
    """Simple numeric DataFrame."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def string_df():
    """DataFrame with string columns."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "city": ["NYC", "LA", "NYC", "SF", "LA"],
            "value": [100, 200, 150, 300, 250],
        }
    )


@pytest.fixture
def nullable_df():
    """DataFrame with nullable data."""
    return pd.DataFrame(
        {
            "a": [1, 2, None, 4, 5],
            "b": [10, None, 30, 40, None],
            "c": ["x", "y", None, "z", "w"],
        }
    )


@pytest.fixture
def arrow_df():
    """DataFrame with Arrow-backed dtypes."""
    return pd.DataFrame(
        {
            "a": pd.array([1, 2, 3, 4, 5], dtype="int64[pyarrow]"),
            "b": pd.array([10, 20, 30, 40, 50], dtype="int64[pyarrow]"),
            "c": pd.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="double[pyarrow]"),
        }
    )


@pytest.fixture
def large_df():
    """Larger DataFrame for performance-sensitive tests."""
    rng = np.random.default_rng(42)
    n = 10_000
    return pd.DataFrame(
        {
            "a": rng.integers(0, 100, n),
            "b": rng.random(n),
            "c": rng.choice(["X", "Y", "Z"], n),
        }
    )


# =============================================================================
# Helper Functions for Probing
# =============================================================================


def get_plan_structure(plan):
    """
    Return a string representation of plan structure for debugging.

    Example: "Filter(Project(Scan))"
    """
    name = type(plan).__name__

    if hasattr(plan, "input"):
        child = get_plan_structure(plan.input)
        return f"{name}({child})"
    elif hasattr(plan, "left") and hasattr(plan, "right"):
        left = get_plan_structure(plan.left)
        right = get_plan_structure(plan.right)
        return f"{name}({left}, {right})"
    else:
        return name


def count_plan_nodes(plan, node_type=None):
    """Count nodes of a specific type in a plan tree."""
    count = 0
    if node_type is None or isinstance(plan, node_type):
        count = 1

    if hasattr(plan, "input"):
        count += count_plan_nodes(plan.input, node_type)
    if hasattr(plan, "left"):
        count += count_plan_nodes(plan.left, node_type)
    if hasattr(plan, "right"):
        count += count_plan_nodes(plan.right, node_type)

    return count


def execute_with_and_without_optimization(lf):
    """
    Execute a LazyDataFrame with and without optimization.
    Returns (optimized_result, unoptimized_result).
    """
    optimized = lf.collect(optimize=True)
    unoptimized = lf.collect(optimize=False)
    return optimized, unoptimized


def _compare_lazy_to_eager(lf, eager_result):  # not a test
    """Compare lazy execution to pre-computed eager result."""
    lazy_result = lf.collect()
    tm.assert_frame_equal(lazy_result, eager_result, check_dtype=False)


# =============================================================================
# Test Class: Basic Correctness
# =============================================================================


class TestBasicCorrectness:
    """Verify basic operations produce correct results through full pipeline."""

    def test_select_all_columns(self, simple_df):
        """Select all columns should return identical DataFrame."""
        lf = simple_df.select()
        result = lf.collect()
        tm.assert_frame_equal(result, simple_df)

    def test_select_subset(self, simple_df):
        """Select subset of columns."""
        lf = simple_df.select("a", "b")
        result = lf.collect()
        expected = simple_df[["a", "b"]]
        tm.assert_frame_equal(result, expected)

    def test_filter_simple(self, simple_df):
        """Simple filter operation."""
        lf = simple_df.select().filter(col("a") > 2)
        result = lf.collect()
        expected = simple_df[simple_df["a"] > 2].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_computed_column(self, simple_df):
        """Add computed column."""
        lf = simple_df.select().with_columns((col("a") + col("b")).alias("sum_ab"))
        result = lf.collect()

        expected = simple_df.copy()
        expected["sum_ab"] = expected["a"] + expected["b"]
        tm.assert_frame_equal(result, expected)

    def test_sort(self, simple_df):
        """Sort operation."""
        lf = simple_df.select().sort("a", descending=True)
        result = lf.collect()
        expected = simple_df.sort_values("a", ascending=False).reset_index(drop=True)
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_limit(self, simple_df):
        """Limit operation."""
        lf = simple_df.select().head(3)
        result = lf.collect()
        expected = simple_df.head(3)
        tm.assert_frame_equal(result, expected)


# =============================================================================
# Test Class: Optimizer Correctness
# =============================================================================


class TestOptimizerCorrectness:
    """Verify optimizations don't change results."""

    def test_filter_fusion_correctness(self, simple_df):
        """Fused filters should produce same results as separate filters."""
        lf = (
            simple_df.select()
            .filter(col("a") > 1)
            .filter(col("a") < 5)
            .filter(col("b") > 15)
        )

        opt, unopt = execute_with_and_without_optimization(lf)
        tm.assert_frame_equal(opt, unopt)

        # Verify against eager (reset index since lazy resets indices)
        expected = simple_df[
            (simple_df["a"] > 1) & (simple_df["a"] < 5) & (simple_df["b"] > 15)
        ].reset_index(drop=True)
        tm.assert_frame_equal(opt, expected)

    def test_predicate_pushdown_correctness(self, simple_df):
        """Pushed predicates should produce same results."""
        lf = (
            simple_df.select()
            .with_columns((col("a") * 2).alias("double_a"))
            .filter(col("a") > 2)  # Can be pushed
        )

        opt, unopt = execute_with_and_without_optimization(lf)
        tm.assert_frame_equal(opt, unopt)

    def test_projection_pruning_correctness(self, simple_df):
        """Pruned projections should produce same results."""
        lf = (
            simple_df.select()
            .with_columns(
                (col("a") + col("b")).alias("sum"),
                (col("a") * col("b")).alias("prod"),  # Will be pruned
            )
            .select("a", "sum")
        )

        opt, unopt = execute_with_and_without_optimization(lf)
        tm.assert_frame_equal(opt, unopt)

    def test_limit_pushdown_correctness(self, simple_df):
        """Pushed limits should produce same results."""
        lf = simple_df.select().head(3)

        opt, unopt = execute_with_and_without_optimization(lf)
        tm.assert_frame_equal(opt, unopt)

    def test_sort_limit_to_topk_correctness(self, simple_df):
        """TopK optimization should produce same results as sort+limit."""
        lf = simple_df.select().sort("a", descending=True).head(3)

        opt, unopt = execute_with_and_without_optimization(lf)
        tm.assert_frame_equal(opt, unopt)

        expected = simple_df.nlargest(3, "a").reset_index(drop=True)
        tm.assert_frame_equal(opt, expected, check_names=False)

    def test_constant_folding_correctness(self, simple_df):
        """Constant folding should produce same results."""
        lf = simple_df.select().with_columns(
            (lit(2) + lit(3)).alias("const"),
            (col("a") + lit(10)).alias("plus_ten"),
        )

        opt, unopt = execute_with_and_without_optimization(lf)
        tm.assert_frame_equal(opt, unopt)


# =============================================================================
# Test Class: Plan Structure Verification
# =============================================================================


class TestPlanStructure:
    """Verify optimizer produces expected plan structures."""

    def test_filter_fusion_combines_filters(self, simple_df):
        """Multiple filters should be fused into one."""
        lf = simple_df.select().filter(col("a") > 1).filter(col("b") > 10)

        # Before optimization: multiple Filter nodes
        unopt_plan = lf._plan
        assert count_plan_nodes(unopt_plan, Filter) == 2

        # After optimization: single Filter node
        optimizer = Optimizer(passes=[FilterFusion()])
        opt_plan = optimizer.optimize(unopt_plan)
        assert count_plan_nodes(opt_plan, Filter) == 1

    def test_sort_limit_becomes_topk(self, simple_df):
        """Sort followed by limit should become TopK."""
        lf = simple_df.select().sort("a").head(3)

        optimizer = Optimizer(passes=[SortLimitToTopK()])
        opt_plan = optimizer.optimize(lf._plan)

        assert isinstance(opt_plan, TopK)
        assert count_plan_nodes(opt_plan, Sort) == 0
        assert count_plan_nodes(opt_plan, Limit) == 0

    def test_predicate_pushdown_moves_filter(self, simple_df):
        """Filter on source column should be pushed past projection."""
        lf = (
            simple_df.select()
            .with_columns((col("a") * 2).alias("double_a"))
            .filter(col("b") > 10)  # Filter on source column
        )

        optimizer = Optimizer(passes=[PredicatePushdown()])
        opt_plan = optimizer.optimize(lf._plan)

        # Filter should be below Project (pushed down)
        structure = get_plan_structure(opt_plan)
        # Expected: Project(Project(Filter(...)))
        # Filter pushed from top to inside the projects
        assert "Filter" in structure
        # Filter should not be at the top level anymore
        assert not structure.startswith("Filter(")


# =============================================================================
# Test Class: Physical Plan Verification
# =============================================================================


class TestPhysicalPlan:
    """Verify physical planner produces correct physical plans."""

    def test_physical_scan(self, simple_df):
        """Scan should produce PhysicalScan."""
        lf = simple_df.select()
        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalProject)
        assert isinstance(physical.input, PhysicalScan)

    def test_physical_filter(self, simple_df):
        """Filter should produce PhysicalFilter."""
        lf = simple_df.select().filter(col("a") > 2)
        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalFilter)

    def test_topk_physical_plan(self, simple_df):
        """TopK optimization should produce PhysicalTopK."""
        lf = simple_df.select().sort("a").head(3)

        # First optimize to get TopK logical plan
        optimizer = Optimizer()
        opt_plan = optimizer.optimize(lf._plan)

        # Then physical plan
        planner = PhysicalPlanner()
        physical = planner.plan(opt_plan)

        assert isinstance(physical, PhysicalTopK)

    def test_physical_execution(self, simple_df):
        """Physical plan should execute correctly."""
        lf = simple_df.select().filter(col("a") > 2)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        result_df = execute_physical_plan(physical)

        expected = simple_df[simple_df["a"] > 2].reset_index(drop=True)
        tm.assert_frame_equal(result_df, expected)


# =============================================================================
# Test Class: Backend Selection
# =============================================================================


class TestBackendSelection:
    """Verify correct backend selection for different data types."""

    def test_numpy_data_uses_numpy(self, simple_df):
        """NumPy-backed data should use NumPy kernels."""
        lf = simple_df.select().with_columns((col("a") + col("b")).alias("sum"))

        planner = PhysicalPlanner(preferred_backend="auto")
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalProject)
        # Backend selection is "auto" which should work with NumPy data

    def test_arrow_data_uses_arrow(self, arrow_df):
        """Arrow-backed data should use Arrow kernels."""
        lf = arrow_df.select().with_columns((col("a") + col("b")).alias("sum"))

        planner = PhysicalPlanner(preferred_backend="auto")
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalProject)

    def test_arrow_preferred_backend(self, simple_df):
        """Preferred backend should be respected."""
        lf = simple_df.select()

        planner_arrow = PhysicalPlanner(preferred_backend="arrow")
        planner_numpy = PhysicalPlanner(preferred_backend="numpy")

        physical_arrow = planner_arrow.plan(lf._plan)
        physical_numpy = planner_numpy.plan(lf._plan)

        # Both should work, just with different backend preferences
        assert isinstance(physical_arrow, PhysicalProject)
        assert isinstance(physical_numpy, PhysicalProject)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases that might break the pipeline."""

    def test_empty_dataframe(self):
        """Empty DataFrame should work through pipeline."""
        df = pd.DataFrame({"a": [], "b": []})
        lf = df.select().filter(col("a") > 0)
        result = lf.collect()

        expected = df[df["a"] > 0]
        tm.assert_frame_equal(result, expected)

    def test_single_row(self):
        """Single row DataFrame should work."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        lf = df.select().with_columns((col("a") + col("b")).alias("sum"))
        result = lf.collect()

        expected = df.copy()
        expected["sum"] = 3
        tm.assert_frame_equal(result, expected)

    def test_all_filtered_out(self, simple_df):
        """Filter that removes all rows."""
        lf = simple_df.select().filter(col("a") > 100)
        result = lf.collect()

        expected = simple_df[simple_df["a"] > 100]
        tm.assert_frame_equal(result, expected)

    def test_duplicate_column_names(self):
        """Handling of expressions that produce same-named columns."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = df.select().with_columns(
            (col("a") * 2).alias("result"),
        )
        result = lf.collect()

        expected = df.copy()
        expected["result"] = expected["a"] * 2
        tm.assert_frame_equal(result, expected)

    def test_chain_of_transformations(self, simple_df):
        """Long chain of transformations."""
        lf = (
            simple_df.select()
            .filter(col("a") > 1)
            .with_columns((col("a") * 2).alias("double_a"))
            .filter(col("double_a") > 5)
            .with_columns((col("b") + col("c")).alias("sum_bc"))
            .select("a", "double_a", "sum_bc")
        )

        result = lf.collect()

        # Compute expected manually
        expected = simple_df[simple_df["a"] > 1].copy()
        expected["double_a"] = expected["a"] * 2
        expected = expected[expected["double_a"] > 5]
        expected["sum_bc"] = expected["b"] + expected["c"]
        expected = expected[["a", "double_a", "sum_bc"]].reset_index(drop=True)

        tm.assert_frame_equal(result, expected)


# =============================================================================
# Test Class: Complex Pipelines
# =============================================================================


class TestComplexPipelines:
    """Test complex real-world query patterns."""

    def test_etl_pipeline(self, large_df):
        """Typical ETL pipeline: filter → transform → aggregate-like select."""
        lf = (
            large_df.select()
            .filter(col("a") > 50)
            .filter(col("c") == "X")
            .with_columns(
                (col("b") * 100).alias("scaled_b"),
                (col("a") + 10).alias("shifted_a"),
            )
            .select("a", "scaled_b", "shifted_a")
        )

        opt, unopt = execute_with_and_without_optimization(lf)
        tm.assert_frame_equal(opt, unopt)

        # Verify against eager
        expected = large_df[(large_df["a"] > 50) & (large_df["c"] == "X")].copy()
        expected["scaled_b"] = expected["b"] * 100
        expected["shifted_a"] = expected["a"] + 10
        expected = expected[["a", "scaled_b", "shifted_a"]].reset_index(drop=True)
        tm.assert_frame_equal(opt, expected)

    def test_reporting_query(self, string_df):
        """Typical reporting query: filter → compute → sort → limit."""
        lf = (
            string_df.select()
            .filter(col("city") == "NYC")
            .with_columns((col("value") * 1.1).alias("adjusted_value"))
            .sort("adjusted_value", descending=True)
            .head(5)
        )

        result = lf.collect()

        expected = string_df[string_df["city"] == "NYC"].copy()
        expected["adjusted_value"] = expected["value"] * 1.1
        expected = (
            expected.sort_values("adjusted_value", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_multiple_computed_columns(self, simple_df):
        """Multiple computed columns at once."""
        lf = simple_df.select().with_columns(
            (col("a") + col("b")).alias("sum"),
            (col("a") - col("b")).alias("diff"),
            (col("a") * col("b")).alias("prod"),
            (col("a") / (col("b") + 1)).alias("ratio"),
        )

        opt, unopt = execute_with_and_without_optimization(lf)
        tm.assert_frame_equal(opt, unopt)

    def test_conditional_expression(self, simple_df):
        """Conditional (case-when) expressions."""
        lf = simple_df.select().with_columns(
            when(col("a") > 3)
            .then(lit("high"))
            .when(col("a") > 1)
            .then(lit("medium"))
            .otherwise(lit("low"))
            .alias("category")
        )

        result = lf.collect()

        # Verify structure
        assert "category" in result.columns
        assert set(result["category"].unique()).issubset({"high", "medium", "low"})


# =============================================================================
# Test Class: Regression Tests
# =============================================================================


class TestRegressions:
    """Tests for specific bugs or edge cases encountered."""

    def test_filter_on_computed_column(self, simple_df):
        """Filter on a column that was just computed."""
        lf = (
            simple_df.select()
            .with_columns((col("a") * 2).alias("double_a"))
            .filter(col("double_a") > 5)
        )

        result = lf.collect()

        expected = simple_df.copy()
        expected["double_a"] = expected["a"] * 2
        expected = expected[expected["double_a"] > 5].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_select_after_filter(self, simple_df):
        """Select specific columns after filter."""
        lf = simple_df.select().filter(col("a") > 2).select("a", "b")

        result = lf.collect()

        expected = simple_df[simple_df["a"] > 2][["a", "b"]].reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_multiple_sorts(self, simple_df):
        """Multiple sort operations (last should win)."""
        lf = simple_df.select().sort("a").sort("b", descending=True)

        result = lf.collect()

        expected = simple_df.sort_values("b", ascending=False).reset_index(drop=True)
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_filter_then_limit(self, simple_df):
        """Filter followed by limit."""
        lf = simple_df.select().filter(col("a") > 1).head(2)

        result = lf.collect()

        expected = simple_df[simple_df["a"] > 1].head(2).reset_index(drop=True)
        tm.assert_frame_equal(result, expected)


# =============================================================================
# Test Class: Optimization Pass Isolation
# =============================================================================


class TestPassIsolation:
    """Test each optimization pass in isolation."""

    def test_only_constant_folding(self, simple_df):
        """Test constant folding pass alone."""
        lf = simple_df.select().with_columns(
            (lit(2) + lit(3)).alias("five"),
            (col("a") + lit(0)).alias("a_plus_zero"),
        )

        optimizer = Optimizer(passes=[ConstantFolding()])
        opt_plan = optimizer.optimize(lf._plan)

        # Execute both
        planner = PhysicalPlanner()
        result = execute_physical_plan(planner.plan(opt_plan))

        expected = simple_df.copy()
        expected["five"] = 5
        expected["a_plus_zero"] = expected["a"] + 0
        tm.assert_frame_equal(result, expected)

    def test_only_filter_fusion(self, simple_df):
        """Test filter fusion pass alone."""
        lf = simple_df.select().filter(col("a") > 1).filter(col("a") < 5)

        optimizer = Optimizer(passes=[FilterFusion()])
        opt_plan = optimizer.optimize(lf._plan)

        # Should have one filter
        assert count_plan_nodes(opt_plan, Filter) == 1

        # Results should be same
        planner = PhysicalPlanner()
        result = execute_physical_plan(planner.plan(opt_plan))
        expected = simple_df[(simple_df["a"] > 1) & (simple_df["a"] < 5)].reset_index(
            drop=True
        )
        tm.assert_frame_equal(result, expected)

    def test_only_limit_pushdown(self, simple_df):
        """Test limit pushdown pass alone."""
        lf = simple_df.select().with_columns((col("a") * 2).alias("double")).head(2)

        optimizer = Optimizer(passes=[LimitPushdown()])
        opt_plan = optimizer.optimize(lf._plan)

        planner = PhysicalPlanner()
        result = execute_physical_plan(planner.plan(opt_plan))

        expected = simple_df.head(2).copy()
        expected["double"] = expected["a"] * 2
        tm.assert_frame_equal(result, expected)


# =============================================================================
# Test Class: Debugging Utilities
# =============================================================================


class TestDebuggingUtilities:
    """Test probing and debugging utilities for the lazy pipeline."""

    def test_get_plan_structure(self, simple_df):
        """Plan structure helper should work."""
        lf = simple_df.select().filter(col("a") > 2)
        structure = get_plan_structure(lf._plan)

        assert isinstance(structure, str)
        assert "Filter" in structure
        assert "Project" in structure

    def test_count_plan_nodes(self, simple_df):
        """Node counting helper should work."""
        lf = simple_df.select().filter(col("a") > 1).filter(col("b") > 10)

        # Before fusion
        assert count_plan_nodes(lf._plan, Filter) == 2

        # After fusion
        optimizer = Optimizer(passes=[FilterFusion()])
        opt_plan = optimizer.optimize(lf._plan)
        assert count_plan_nodes(opt_plan, Filter) == 1

    def test_execute_with_and_without_optimization(self, simple_df):
        """Helper should return both results."""
        lf = simple_df.select().filter(col("a") > 2)
        opt, unopt = execute_with_and_without_optimization(lf)

        # Both should be DataFrames
        assert isinstance(opt, pd.DataFrame)
        assert isinstance(unopt, pd.DataFrame)

        # Both should be equal
        tm.assert_frame_equal(opt, unopt)

    def test_compare_plan_before_after_pass(self, simple_df):
        """Demonstrate comparing plans before/after a specific pass."""
        lf = simple_df.select().filter(col("a") > 1).filter(col("a") < 5)

        before_structure = get_plan_structure(lf._plan)
        before_filter_count = count_plan_nodes(lf._plan, Filter)

        optimizer = Optimizer(passes=[FilterFusion()])
        opt_plan = optimizer.optimize(lf._plan)

        after_structure = get_plan_structure(opt_plan)
        after_filter_count = count_plan_nodes(opt_plan, Filter)

        # Verify transformation happened
        assert before_filter_count == 2
        assert after_filter_count == 1
        assert before_structure != after_structure


# =============================================================================
# Test Class: End-to-End with explain()
# =============================================================================


class TestExplainIntegration:
    """Test explain() functionality for debugging."""

    def test_explain_simple(self, simple_df):
        """Explain should return string representation."""
        lf = simple_df.select().filter(col("a") > 2)
        explanation = lf.explain()

        assert isinstance(explanation, str)
        assert "Filter" in explanation or "filter" in explanation.lower()

    def test_explain_optimized(self, simple_df):
        """Explain with optimization should show optimized plan."""
        lf = simple_df.select().filter(col("a") > 1).filter(col("a") < 5)

        # Optimized should have fused filter
        explanation = lf.explain()
        assert isinstance(explanation, str)

    def test_explain_complex(self, simple_df):
        """Explain complex query."""
        lf = (
            simple_df.select()
            .filter(col("a") > 1)
            .with_columns((col("a") * 2).alias("double"))
            .sort("double")
            .head(3)
        )

        explanation = lf.explain()
        assert isinstance(explanation, str)
