"""
Tests for the lazy pandas physical planner.

These tests verify that:
1. Physical planner correctly converts logical plans to physical plans
2. Physical plan execution produces correct results
3. Different physical plan nodes execute correctly
"""

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.physical import (
    ExecutionContext,
    PhysicalDistinct,
    PhysicalFilter,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalLimit,
    PhysicalPlanner,
    PhysicalProject,
    PhysicalScan,
    PhysicalSort,
    execute_physical_plan,
)


class TestPhysicalPlanner:
    """Tests for the PhysicalPlanner class."""

    def test_plan_scan(self):
        """Test planning a simple scan (select all columns)."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select()

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        # select() creates a Project, so we get PhysicalProject
        assert isinstance(physical, PhysicalProject)

    def test_plan_project(self):
        """Test planning a projection."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select("a")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalProject)
        assert isinstance(physical.input, PhysicalScan)

    def test_plan_filter(self):
        """Test planning a filter."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select().filter(col("a") > 1)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalFilter)
        assert isinstance(physical.input, PhysicalProject)

    def test_plan_aggregate(self):
        """Test planning an aggregation."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
        lf = df.select().group_by("a").agg(col("b").sum().alias("sum_b"))

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalHashAggregate)

    def test_plan_sort(self):
        """Test planning a sort."""
        df = pd.DataFrame({"a": [3, 1, 2], "b": [4, 5, 6]})
        lf = df.select().sort("a")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalSort)

    def test_plan_limit(self):
        """Test planning a limit."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select().head(2)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalLimit)

    def test_plan_distinct(self):
        """Test planning a distinct."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 4, 6]})
        lf = df.select().distinct()

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalDistinct)

    def test_plan_join(self):
        """Test planning a join."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 4], "c": [7, 8, 9]})
        lf = df1.select().join(df2.select(), on="a")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        assert isinstance(physical, PhysicalHashJoin)

    def test_plan_complex_query(self):
        """Test planning a complex query with multiple operations."""
        df = pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": [10, 20, 30, 40, 50]})
        lf = (
            df.select()
            .filter(col("a") > 0)
            .group_by("a")
            .agg(col("b").sum().alias("sum_b"))
            .sort("sum_b", descending=True)
            .head(2)
        )

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        # Should be: Limit -> Sort -> Aggregate -> Filter -> Project -> Scan
        assert isinstance(physical, PhysicalLimit)
        assert isinstance(physical.input, PhysicalSort)
        assert isinstance(physical.input.input, PhysicalHashAggregate)


class TestPhysicalPlanExecution:
    """Tests for physical plan execution."""

    def test_execute_scan(self):
        """Test executing a scan (via select)."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select()

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        tm.assert_frame_equal(result, df)

    def test_execute_project(self):
        """Test executing a projection."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select("a")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_execute_project_with_expression(self):
        """Test executing a projection with computed columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select((col("a") + col("b")).alias("c"))

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame({"c": [5, 7, 9]})
        tm.assert_frame_equal(result, expected)

    def test_execute_filter(self):
        """Test executing a filter."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select().filter(col("a") > 1)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame({"a": [2, 3], "b": [5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_execute_aggregate(self):
        """Test executing an aggregation."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
        lf = df.select().group_by("a").agg(col("b").sum().alias("sum_b"))

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        # Sort for comparison
        result = result.sort_values("a").reset_index(drop=True)
        expected = pd.DataFrame({"a": [1, 2], "sum_b": [30, 30]})
        tm.assert_frame_equal(result, expected)

    def test_execute_sort(self):
        """Test executing a sort."""
        df = pd.DataFrame({"a": [3, 1, 2], "b": [4, 5, 6]})
        lf = df.select().sort("a")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame({"a": [1, 2, 3], "b": [5, 6, 4]})
        tm.assert_frame_equal(result, expected)

    def test_execute_sort_descending(self):
        """Test executing a descending sort."""
        df = pd.DataFrame({"a": [3, 1, 2], "b": [4, 5, 6]})
        lf = df.select().sort("a", descending=True)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame({"a": [3, 2, 1], "b": [4, 6, 5]})
        tm.assert_frame_equal(result, expected)

    def test_execute_limit(self):
        """Test executing a limit."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        lf = df.select().head(3)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame({"a": [1, 2, 3], "b": [6, 7, 8]})
        tm.assert_frame_equal(result, expected)

    def test_execute_distinct(self):
        """Test executing a distinct."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 4, 6]})
        lf = df.select().distinct()

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame({"a": [1, 2], "b": [4, 6]})
        tm.assert_frame_equal(result, expected)

    def test_execute_join(self):
        """Test executing a join."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 4], "c": [7, 8, 9]})
        lf = df1.select().join(df2.select(), on="a")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame({"a": [1, 2], "b": [4, 5], "c": [7, 8]})
        tm.assert_frame_equal(result, expected)

    def test_execute_left_join(self):
        """Test executing a left join."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 4], "c": [7, 8, 9]})
        lf = df1.select().join(df2.select(), on="a", how="left")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        expected = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": pd.array([7.0, 8.0, pd.NA], dtype=pd.Float64Dtype()),
            }
        )
        tm.assert_frame_equal(result, expected)


class TestPhysicalTopK:
    """Tests for PhysicalTopK execution."""

    def test_topk_ascending(self):
        """Test TopK with ascending order."""
        df = pd.DataFrame({"a": [5, 3, 1, 4, 2], "b": [10, 20, 30, 40, 50]})
        lf = df.select().sort("a").head(3)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        # Should get the 3 smallest values of 'a'
        expected = pd.DataFrame({"a": [1, 2, 3], "b": [30, 50, 20]})
        tm.assert_frame_equal(result, expected)

    def test_topk_descending(self):
        """Test TopK with descending order."""
        df = pd.DataFrame({"a": [5, 3, 1, 4, 2], "b": [10, 20, 30, 40, 50]})
        lf = df.select().sort("a", descending=True).head(3)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        # Should get the 3 largest values of 'a'
        expected = pd.DataFrame({"a": [5, 4, 3], "b": [10, 40, 20]})
        tm.assert_frame_equal(result, expected)


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_default_context(self):
        """Test default execution context."""
        context = ExecutionContext()
        assert context.preferred_backend == "auto"
        assert context.strict is False
        assert context.cache == {}

    def test_custom_context(self):
        """Test custom execution context."""
        context = ExecutionContext(
            preferred_backend="arrow",
            strict=True,
        )
        assert context.preferred_backend == "arrow"
        assert context.strict is True


class TestPhysicalPlanChildren:
    """Tests for physical plan children() method."""

    def test_project_has_one_child(self):
        """Test that project has one child."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = df.select()

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        children = physical.children()
        assert len(children) == 1
        assert isinstance(children[0], PhysicalScan)

    def test_filter_has_one_child(self):
        """Test that filter has one child."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        lf = df.select().filter(col("a") > 1)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        children = physical.children()
        assert len(children) == 1
        # Filter's child is the project
        assert isinstance(children[0], PhysicalProject)

    def test_join_has_two_children(self):
        """Test that join has two children."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        lf = df1.select().join(df2.select(), on="a")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        children = physical.children()
        assert len(children) == 2


class TestPhysicalPlanOutputSchema:
    """Tests for physical plan output_schema property."""

    def test_project_output_schema(self):
        """Test project output schema."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select()

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        schema = physical.output_schema
        assert "a" in schema
        assert "b" in schema

    def test_project_single_column_schema(self):
        """Test project output schema with single column."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lf = df.select("a")

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        schema = physical.output_schema
        assert "a" in schema
        assert "b" not in schema


class TestEndToEndPhysicalExecution:
    """End-to-end tests comparing physical plan execution with collect()."""

    def test_complex_query_matches_collect(self):
        """Test that physical execution matches collect() result."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        lf = (
            df.select()
            .filter(col("value") > 15)
            .group_by("category")
            .agg(col("value").sum().alias("total"))
            .sort("total", descending=True)
        )

        # Get result via collect()
        expected = lf.collect()

        # Get result via physical planner
        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        # check_dtype=False because physical planner returns Arrow-backed dtypes
        # for near-zero-copy conversion, while collect() returns NumPy dtypes
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_join_query_matches_collect(self):
        """Test that join execution matches collect() result."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        df2 = pd.DataFrame({"id": [1, 2, 4], "score": [100, 200, 300]})

        lf = df1.select().join(df2.select(), on="id").select("name", "score")

        expected = lf.collect()

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_multiple_filters_matches_collect(self):
        """Test that multiple filters execution matches collect() result."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        lf = df.select().filter(col("a") > 1).filter(col("b") < 50)

        expected = lf.collect()

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)
        result = execute_physical_plan(physical)

        tm.assert_frame_equal(result, expected, check_dtype=False)


class TestCollectPhysicalPlannerFlag:
    """Tests for the use_physical_planner flag in collect()."""

    def test_eager_and_physical_produce_same_results(self):
        """Both execution modes should produce identical results."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        lf = (
            df.select()
            .filter(col("value") > 15)
            .group_by("category")
            .agg(col("value").sum().alias("total"))
            .sort("total", descending=True)
        )

        # Default (eager)
        result_eager = lf.collect(use_physical_planner=False)

        # Physical planner
        result_physical = lf.collect(use_physical_planner=True)

        # check_dtype=False because physical planner returns Arrow-backed dtypes
        # for near-zero-copy conversion, while eager returns NumPy dtypes
        tm.assert_frame_equal(result_eager, result_physical, check_dtype=False)

    def test_physical_planner_with_join(self):
        """Physical planner should handle joins correctly."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        df2 = pd.DataFrame({"id": [1, 2, 4], "score": [100, 200, 300]})

        lf = df1.select().join(df2.select(), on="id").select("name", "score")

        result_eager = lf.collect(use_physical_planner=False)
        result_physical = lf.collect(use_physical_planner=True)

        tm.assert_frame_equal(result_eager, result_physical, check_dtype=False)

    def test_physical_planner_with_topk(self):
        """Physical planner should handle TopK correctly."""
        df = pd.DataFrame({"a": [5, 3, 1, 4, 2], "b": [10, 20, 30, 40, 50]})
        lf = df.select().sort("a").head(3)

        result_eager = lf.collect(use_physical_planner=False)
        result_physical = lf.collect(use_physical_planner=True)

        tm.assert_frame_equal(result_eager, result_physical, check_dtype=False)

    def test_physical_planner_with_tail(self):
        """Physical planner should handle tail correctly."""
        df = pd.DataFrame({"a": range(10)})
        lf = df.select().tail(3)

        result_eager = lf.collect(use_physical_planner=False)
        result_physical = lf.collect(use_physical_planner=True)

        tm.assert_frame_equal(result_eager, result_physical, check_dtype=False)

    def test_physical_planner_with_distinct(self):
        """Physical planner should handle distinct correctly."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [10, 10, 20, 20, 30]})
        lf = df.select().distinct()

        result_eager = lf.collect(use_physical_planner=False)
        result_physical = lf.collect(use_physical_planner=True)

        tm.assert_frame_equal(result_eager, result_physical, check_dtype=False)

    def test_physical_planner_respects_optimize_flag(self):
        """Physical planner should respect the optimize flag."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        lf = df.select().filter(col("a") > 2).filter(col("a") < 5)

        # With optimization (filters should be fused)
        result_opt = lf.collect(optimize=True, use_physical_planner=True)

        # Without optimization
        result_no_opt = lf.collect(optimize=False, use_physical_planner=True)

        # Results should be the same regardless of optimization
        tm.assert_frame_equal(result_opt, result_no_opt)
