"""
Tests for the lazy pandas physical planner.

These tests verify that:
1. Physical planner correctly converts logical plans to physical plans
2. Physical plan execution produces correct results
3. Different physical plan nodes execute correctly
"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.physical import (
    ExecutionContext,
    PhysicalDistinct,
    PhysicalFilter,
    PhysicalFusedPipeline,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalLimit,
    PhysicalMaterialize,
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
        # Disable fusion to test individual operator planning
        physical = planner.plan(lf._plan, enable_fusion=False)

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
        # Disable fusion to test individual operator planning
        physical = planner.plan(lf._plan, enable_fusion=False)

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

        # Should be:
        # Limit -> Sort -> Materialize(sort) -> Aggregate -> Materialize(agg) -> ...
        # Each pipeline breaker (Sort, Aggregate) has explicit Materialize input
        assert isinstance(physical, PhysicalLimit)
        assert isinstance(physical.input, PhysicalSort)
        assert physical.input.is_pipeline_breaker  # Sort is a breaker

        # Sort's input should be Materialize (for sort)
        sort_input = physical.input.input
        assert isinstance(sort_input, PhysicalMaterialize)
        assert sort_input.reason == "sort"

        # Materialize's input: the aggregate (or, if grouped fusion is
        # ever re-enabled, the fused kernel node carrying it as fallback).
        from pandas.lazy.physical import PhysicalFusedFilterAgg

        agg_node = sort_input.input
        if isinstance(agg_node, PhysicalFusedFilterAgg):
            agg_node = agg_node.fallback
        assert isinstance(agg_node, PhysicalHashAggregate)


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

        # Output dtype contract: a NumPy int column that acquires nulls via
        # the left join widens to NumPy float64 (np.nan), matching eager
        # pd.merge - not nullable Float64 (see convert.arrays_to_dataframe).
        expected = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": np.array([7.0, 8.0, np.nan]),
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
        # Disable fusion to test individual operator structure
        physical = planner.plan(lf._plan, enable_fusion=False)

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


class TestOperatorFusion:
    """Tests for operator fusion in physical planning."""

    def test_filter_project_fusion(self):
        """Filter + Project should be fused into FusedPipeline."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        lf = df.select().filter(col("a") > 2)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)  # Fusion enabled by default

        # Should be fused: Filter + Project -> FusedPipeline
        assert isinstance(physical, PhysicalFusedPipeline)
        assert len(physical.operations) == 2
        assert physical.operations[0].op_type == "project"
        assert physical.operations[1].op_type == "filter"

    def test_project_limit_fusion(self):
        """Project + Limit should be fused."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        lf = df.select().head(3)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        # Should be fused: Project + Limit -> FusedPipeline
        assert isinstance(physical, PhysicalFusedPipeline)
        assert len(physical.operations) == 2
        assert physical.operations[0].op_type == "project"
        assert physical.operations[1].op_type == "limit"

    def test_filter_filter_fusion(self):
        """Multiple filters should be fused."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        lf = df.select().filter(col("a") > 1).filter(col("a") < 5)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        # Should be fused: Project + Filter + Filter
        assert isinstance(physical, PhysicalFusedPipeline)
        filter_ops = [op for op in physical.operations if op.op_type == "filter"]
        assert len(filter_ops) == 2

    def test_fused_execution_correctness(self):
        """Fused execution should produce same results as unfused."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        lf = df.select().filter(col("a") > 2).head(2)

        planner = PhysicalPlanner()

        # With fusion
        physical_fused = planner.plan(lf._plan, enable_fusion=True)
        result_fused = execute_physical_plan(physical_fused)

        # Without fusion
        physical_unfused = planner.plan(lf._plan, enable_fusion=False)
        result_unfused = execute_physical_plan(physical_unfused)

        tm.assert_frame_equal(result_fused, result_unfused)

    def test_fusion_stops_at_pipeline_breaker(self):
        """Fusion should stop at pipeline breakers (sort, aggregate, etc)."""
        df = pd.DataFrame({"a": [3, 1, 2], "b": [10, 20, 30]})
        lf = df.select().sort("a").head(2)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan)

        # Sort is a pipeline breaker, so structure should be:
        # FusedPipeline(limit) -> Sort -> Materialize -> FusedPipeline(project)
        # or just Limit -> Sort depending on fusion logic
        # The key is that sort should NOT be fused
        assert not isinstance(physical, PhysicalSort)  # Limit is on top

    def test_tail_not_fused(self):
        """Tail operations should not be fused (they need all data)."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        lf = df.select().tail(2)

        planner = PhysicalPlanner()
        physical = planner.plan(lf._plan, enable_fusion=False)

        # Tail should be PhysicalLimit with offset=-1
        assert isinstance(physical, PhysicalLimit)
        assert physical.offset == -1


class TestParallelExecutionPaths:
    """Regression tests for parallel sort/take/concat execution paths.

    Thresholds are monkeypatched down so the parallel code paths run on
    small test data.
    """

    def test_parallel_argsort_is_stable_exact_permutation(self):
        # Lazy sorts are stable by contract: the parallel path must return
        # the EXACT same permutation as np.argsort(kind="stable"), so that
        # tied rows keep their original relative order. Comparing only the
        # sorted key values would miss tie-order divergence.
        from pandas.lazy.backends.numpy.core import _parallel_argsort

        rng = np.random.default_rng(42)
        for arr in (
            rng.standard_normal(10_001),
            np.arange(5_000, dtype=np.float64),
            np.arange(5_000, dtype=np.float64)[::-1].copy(),
            rng.integers(0, 100, 10_000),  # heavy ties across chunks
            rng.integers(0, 3, 10_000),  # extreme ties
            np.zeros(10_000),  # all ties
        ):
            result = _parallel_argsort(arr)
            expected = np.argsort(arr, kind="stable")
            tm.assert_numpy_array_equal(result, expected)

    def test_radix_argsort_matches_numpy_stable_across_dtypes(self):
        # The Cython LSD radix kernel (pandas._libs.lazy_radix) must return
        # the EXACT np.argsort(kind="stable") permutation for every numeric
        # dtype, including the IEEE edge cases (-0.0 == +0.0 ties, +/-inf).
        from pandas.lazy.backends.numpy.core import _radix_argsort

        rng = np.random.default_rng(7)
        arrays = [
            rng.standard_normal(20_000),
            rng.integers(-(10**9), 10**9, 20_000).astype(np.int64),
            (-rng.integers(0, 10**9, 20_000)).astype(np.int64),
            rng.integers(0, 10**12, 20_000).astype(np.uint64),
            rng.integers(-5, 5, 20_000).astype(np.int32),  # heavy ties
            np.arange(20_000, dtype=np.float64)[::-1].copy(),
            rng.choice([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf], 20_000).astype(
                np.float64
            ),
            np.array([], dtype=np.float64),
            np.array([3.0]),
        ]
        for arr in arrays:
            result = _radix_argsort(arr)
            expected = np.argsort(arr, kind="stable")
            tm.assert_numpy_array_equal(result, expected)

    def test_radix_argsort_returns_none_for_unsupported_dtype(self):
        # Non-numeric kinds fall back to the k-way merge / np.argsort.
        from pandas.lazy.backends.numpy.core import _radix_argsort

        assert _radix_argsort(np.array(["b", "a", "c"], dtype=object)) is None

    def test_parallel_radix_matches_serial_and_numpy(self, monkeypatch):
        # The thread-parallel radix (per-chunk histogram + partitioned
        # scatter) must produce the exact same stable permutation as the
        # serial kernel and np.argsort, across dtypes and IEEE edge cases.
        from pandas.lazy.backends.numpy import core

        # Force the parallel path on modest sizes.
        monkeypatch.setattr(core, "RADIX_PARALLEL_MIN_ROWS", 1_000)
        rng = np.random.default_rng(99)
        arrays = [
            rng.standard_normal(50_000),
            rng.integers(-(10**15), 10**15, 50_000).astype(np.int64),
            rng.integers(0, 10**15, 50_000).astype(np.uint64),
            rng.integers(-3, 3, 50_000).astype(np.int64),  # heavy ties
            rng.choice([0.0, -0.0, np.inf, -np.inf, 1.0, -1.0], 50_000).astype(
                np.float64
            ),
        ]
        for arr in arrays:
            assert len(arr) >= core.RADIX_PARALLEL_MIN_ROWS
            tm.assert_numpy_array_equal(
                core._radix_argsort(arr), np.argsort(arr, kind="stable")
            )

    def test_parallel_radix_does_not_mutate_keys(self):
        # The driver copies the keys before scattering; the order-preserving
        # key array a caller might reuse must be left intact.
        from pandas.lazy.backends.numpy.core import _radix_sort_parallel

        keys = np.array([5, 1, 9, 1, 7, 3, 3, 8], dtype=np.uint64)
        snapshot = keys.copy()
        out = _radix_sort_parallel(keys, n_threads=4)
        tm.assert_numpy_array_equal(keys, snapshot)
        # stable sort of the keys themselves
        tm.assert_numpy_array_equal(out, np.argsort(keys, kind="stable"))

    @pytest.mark.parametrize(
        "descending",
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_radix_lexsort_matches_numpy_lexsort(self, descending):
        # The radix lexsort must equal the eager lexicographic order. NumPy's
        # lexsort is all-ascending, so build the reference with negated keys
        # for descending columns (heavy ties exercise the stable composition).
        from pandas.lazy.backends.numpy.core import radix_lexsort

        rng = np.random.default_rng(5)
        g = rng.integers(0, 20, 40_000).astype(np.int64)  # primary, many ties
        v = rng.integers(0, 100, 40_000).astype(np.int64)  # secondary
        keys = [g, v]
        got = radix_lexsort(keys, descending)
        ref_g = -g if descending[0] else g
        ref_v = -v if descending[1] else v
        expected = np.lexsort((ref_v, ref_g))  # primary last in np.lexsort
        # Compare the realized key order (permutations can differ only where
        # both keys tie, and there the rows are identical).
        tm.assert_numpy_array_equal(g[got], g[expected])
        tm.assert_numpy_array_equal(v[got], v[expected])

    def test_multikey_sort_uses_radix_lexsort_path(self, monkeypatch):
        # At/above the multikey threshold with all-numeric keys, the physical
        # sort must match eager exactly (ascending + descending + a float key
        # with NaN, which the per-key NaN handling places last).
        from pandas.lazy import physical

        monkeypatch.setattr(physical, "ARROW_MULTIKEY_SORT_MIN_ROWS", 1_000)
        rng = np.random.default_rng(6)
        n = 5_000
        df = pd.DataFrame(
            {
                "g": rng.integers(0, 30, n).astype("int64"),
                "f": np.where(rng.random(n) < 0.1, np.nan, rng.standard_normal(n)),
                "u": rng.integers(0, 5, n).astype("uint32"),
            }
        )
        for keys, desc in [
            (["g", "f"], [False, False]),
            (["g", "f"], [True, False]),
            (["u", "g", "f"], [False, True, False]),
        ]:
            eager = df.select().sort(*keys, descending=desc).collect()
            physical_res = (
                df.select()
                .sort(*keys, descending=desc)
                .collect(use_physical_planner=True)
            )
            tm.assert_frame_equal(
                eager.reset_index(drop=True),
                physical_res.reset_index(drop=True),
                check_dtype=False,
            )

    def test_multikey_sort_with_string_key_matches_eager(self, monkeypatch):
        # A multi-key sort that includes a string key factorizes it into
        # order-preserving codes and rides the radix lexsort; the result must
        # match eager exactly, including null-bearing string keys (nulls last)
        # and per-key descending.
        from pandas.lazy import physical

        monkeypatch.setattr(physical, "ARROW_MULTIKEY_SORT_MIN_ROWS", 1_000)
        rng = np.random.default_rng(11)
        n = 6_000
        df = pd.DataFrame(
            {
                "s": rng.choice(["alpha", "beta", "gamma", "delta"], n),
                "snull": np.where(
                    rng.random(n) < 0.1, None, rng.choice(["x", "y", "z"], n)
                ),
                "v": rng.standard_normal(n),
                "i": rng.integers(0, 50, n).astype("int64"),
            }
        )
        for keys, desc in [
            (["s", "v"], [False, False]),
            (["s", "v"], [True, False]),
            (["snull", "v"], [False, False]),
            (["snull", "v"], [True, False]),
            (["i", "s", "v"], [False, True, False]),
        ]:
            eager = df.select().sort(*keys, descending=desc).collect()
            physical_res = (
                df.select()
                .sort(*keys, descending=desc)
                .collect(use_physical_planner=True)
            )
            tm.assert_frame_equal(
                eager.reset_index(drop=True),
                physical_res.reset_index(drop=True),
                check_dtype=False,
            )

    def test_keys_for_radix_lexsort_codes_strings_and_rejects_datetime(self):
        from pandas.lazy.physical import _keys_for_radix_lexsort

        # String key -> order-preserving float codes (alpha<beta<gamma).
        keyed = _keys_for_radix_lexsort([np.array(["beta", "alpha", "gamma"])])
        assert keyed is not None
        tm.assert_numpy_array_equal(keyed[0], np.array([1.0, 0.0, 2.0]))
        # null in a string key -> NaN (sorts last).
        keyed = _keys_for_radix_lexsort([np.array(["b", None, "a"], dtype=object)])
        assert keyed is not None and np.isnan(keyed[0][1])
        # datetime key is not coded -> None (caller falls back to Arrow).
        dt = np.array(["2020-01-01", "2019-01-01"], dtype="datetime64[ns]")
        assert _keys_for_radix_lexsort([dt]) is None

    def test_sort_kernel_uses_parallel_path(self, monkeypatch):
        import pandas.lazy.backends.numpy.core as np_core

        monkeypatch.setattr(np_core, "PARALLEL_SORT_MIN_ROWS", 10)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"k": rng.standard_normal(5_000), "v": np.arange(5_000)})

        result = df.select().sort("k").collect(use_physical_planner=True)
        expected = df.sort_values("k").reset_index(drop=True)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("descending", [False, True])
    def test_sort_ties_match_across_engines(self, monkeypatch, descending):
        # Duplicate keys: payload order for tied rows must be identical
        # between the eager evaluator and the physical engine (stable sort
        # contract), including through the parallel argsort path.
        import pandas.lazy.backends.numpy.core as np_core
        import pandas.lazy.physical as phys

        monkeypatch.setattr(np_core, "PARALLEL_SORT_MIN_ROWS", 10)
        monkeypatch.setattr(phys, "PARALLEL_TAKE_MIN_ROWS", 10)
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "k": rng.integers(0, 5, 4_000).astype("float64"),
                "payload": np.arange(4_000),
            }
        )

        ldf = df.select().sort("k", descending=descending)
        eager = ldf.collect()
        physical = ldf.collect(use_physical_planner=True)
        tm.assert_frame_equal(eager, physical, check_dtype=False)
        # Stability: tied rows keep original (ascending payload) order
        for value in np.unique(df["k"]):
            payloads = physical.loc[physical["k"] == value, "payload"].to_numpy()
            assert (np.diff(payloads) > 0).all()

    def test_multikey_sort_ties_match_across_engines(self, monkeypatch):
        import pandas.lazy.physical as phys

        monkeypatch.setattr(phys, "ARROW_MULTIKEY_SORT_MIN_ROWS", 10)
        monkeypatch.setattr(phys, "PARALLEL_TAKE_MIN_ROWS", 10)
        rng = np.random.default_rng(4)
        df = pd.DataFrame(
            {
                "k1": rng.integers(0, 3, 2_000),
                "k2": rng.integers(0, 3, 2_000),
                "payload": np.arange(2_000),
            }
        )

        ldf = df.select().sort("k1", "k2")
        eager = ldf.collect()
        physical = ldf.collect(use_physical_planner=True)
        tm.assert_frame_equal(eager, physical, check_dtype=False)

    def test_arrow_multikey_sort_path(self, monkeypatch):
        import pandas.lazy.physical as phys

        monkeypatch.setattr(phys, "ARROW_MULTIKEY_SORT_MIN_ROWS", 10)
        monkeypatch.setattr(phys, "PARALLEL_TAKE_MIN_ROWS", 10)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "k1": rng.integers(0, 5, 1_000),
                "k2": rng.standard_normal(1_000),
                "v": np.arange(1_000),
            }
        )

        result = df.select().sort("k1", "k2").collect(use_physical_planner=True)
        expected = df.sort_values(["k1", "k2"]).reset_index(drop=True)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_arrow_multikey_sort_mixed_directions(self, monkeypatch):
        import pandas.lazy.physical as phys

        monkeypatch.setattr(phys, "ARROW_MULTIKEY_SORT_MIN_ROWS", 10)
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "k1": rng.integers(0, 5, 500),
                "k2": rng.standard_normal(500),
            }
        )

        result = (
            df.select()
            .sort("k1", "k2", descending=[True, False])
            .collect(use_physical_planner=True)
        )
        expected = df.sort_values(["k1", "k2"], ascending=[False, True]).reset_index(
            drop=True
        )
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_parallel_take_after_single_key_sort(self, monkeypatch):
        import pandas.lazy.physical as phys

        monkeypatch.setattr(phys, "PARALLEL_TAKE_MIN_ROWS", 10)
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "k": rng.standard_normal(1_000),
                "a": np.arange(1_000),
                "b": rng.standard_normal(1_000),
                "s": pd.array(
                    rng.choice(["x", "y", "z"], 1_000), dtype="string[pyarrow]"
                ),
            }
        )

        result = df.select().sort("k").collect(use_physical_planner=True)
        expected = df.sort_values("k").reset_index(drop=True)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_parallel_concat_execute(self):
        from pandas.lazy import concat as lazy_concat

        dfs = [pd.DataFrame({"a": np.arange(i * 100, (i + 1) * 100)}) for i in range(4)]
        result = lazy_concat([d.select() for d in dfs]).collect(
            use_physical_planner=True
        )
        expected = pd.DataFrame({"a": np.arange(400)})
        # Input order must be preserved despite parallel execution
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_parallel_concat_index_metadata_deterministic(self):
        # Concurrent concat inputs must not race on context index
        # metadata: the first input's metadata wins, deterministically
        # (matches Concat.resolve_schema using the first input's schema).
        from pandas.lazy import concat as lazy_concat

        df1 = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="first_idx"))
        df2 = pd.DataFrame({"a": [3, 4]}, index=pd.Index([30, 40], name="second_idx"))

        ldf = lazy_concat([df1.select(), df2.select()])
        for _ in range(5):  # repeated runs: no completion-order dependence
            result = ldf.collect(use_physical_planner=True, preserve_index=True)
            assert result.index.name == "first_idx"

    def test_fused_pipeline_absorption_preserves_order(self):
        # Regression: when _try_fuse absorbed an already-fused inner
        # pipeline, its operations (stored in execution order) were
        # appended to the top-down collection list and wrongly flipped
        # by the final reverse, making later projections drop computed
        # columns that subsequent filters referenced (KeyError) or
        # silently reorder operations.
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "g": rng.choice(["X", "Y", "Z"], 2_000),
                "v1": rng.uniform(0, 100, 2_000),
                "v2": rng.uniform(0, 1_000, 2_000),
            }
        )

        ldf = (
            df.select()
            .filter((col("g") == "X") | (col("g") == "Y"))
            .with_columns((col("v1") + col("v2")).alias("total"))
            .filter(col("total") > 500)
            .with_columns((col("v1") / col("total")).alias("score"))
        )

        eager = ldf.collect()
        physical = ldf.collect(use_physical_planner=True)
        assert len(eager) > 0  # non-trivial result
        tm.assert_frame_equal(eager, physical, check_dtype=False)

    def test_passthrough_arrow_column_not_copied(self):
        # Regression: arrays_to_dataframe round-tripped pass-through
        # Arrow columns through object arrays (to_pandas + re-inference),
        # copying data twice for columns no operation touched. The output
        # must wrap the same Arrow buffers zero-copy.
        df = pd.DataFrame(
            {
                "v1": np.arange(1_000, dtype="float64"),
                "v2": np.arange(1_000, dtype="float64"),
                "text": pd.array([f"item_{i}" for i in range(1_000)]),
            }
        )
        in_addr = df["text"].array._pa_array.chunk(0).buffers()[2].address

        result = (
            df.select()
            .with_columns((col("v1") + col("v2")).alias("s"))
            .collect(use_physical_planner=True)
        )

        # Untouched numpy columns stay numpy-backed
        assert result["v1"].dtype == "float64"
        # Untouched Arrow-backed string column shares the input buffers
        out_arr = result["text"].array
        out_addr = out_arr._pa_array.chunk(0).buffers()[2].address
        assert out_addr == in_addr, "pass-through column was copied"

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_join_values_match_eager_all_types(self, how):
        # Regression for the indexer-based join rewrite: values must match
        # eager pd.merge for every join type (NA representation may differ:
        # physical uses nullable dtypes, eager uses NaN upcast).
        lt = pd.DataFrame(
            {
                "id": [1, 2, 3, 5],
                "v": [10.0, 20.0, 30.0, 50.0],
                "s": pd.array(["a", "b", "c", "e"]),
            }
        )
        rt = pd.DataFrame({"id": [2, 3, 4], "w": [200, 300, 400]})

        phys = (
            lt.select()
            .join(rt.select(), on="id", how=how)
            .collect(use_physical_planner=True)
        )
        eager = lt.merge(rt, on="id", how=how)

        assert len(phys) == len(eager)
        pk = phys.sort_values("id", na_position="last").reset_index(drop=True)
        ek = eager.sort_values("id", na_position="last").reset_index(drop=True)
        for c in ["id", "v", "w"]:
            pv = pk[c].to_numpy(dtype="float64", na_value=np.nan)
            ev = ek[c].to_numpy(dtype="float64", na_value=np.nan)
            tm.assert_numpy_array_equal(pv, ev)
        # string column: compare with nulls normalized
        assert list(pk["s"].fillna("<NA>")) == list(ek["s"].fillna("<NA>"))

    def test_join_passthrough_arrow_column_stays_arrow(self):
        # Payload columns must be gathered in their native backend: the
        # join previously np.asarray'd every column, converting Arrow
        # string payloads to object arrays on input.
        lt = pd.DataFrame(
            {
                "id": np.arange(1_000),
                "text": pd.array([f"item_{i}" for i in range(1_000)]),
            }
        )
        rt = pd.DataFrame({"id": np.arange(0, 1_000, 2), "w": np.arange(500)})

        result = (
            lt.select().join(rt.select(), on="id").collect(use_physical_planner=True)
        )
        # Output dtype contract normalizes all string columns to the default
        # ``str`` dtype; the point here is the payload stays an Arrow-backed
        # string type, never round-tripped through an object array.
        assert str(result["text"].dtype) == "str"
        assert result["id"].dtype.kind in ("i", "u")

    def test_groupby_backend_chosen_from_relevant_columns(self, monkeypatch):
        # Regression: the groupby backend was chosen from the *first* input
        # column, so a wide frame with leading NumPy columns sent
        # Arrow-string-keyed groupbys down the NumPy path (object-array
        # factorize, ~5x slower). The decision must consider only the
        # group keys and aggregation value columns.
        from pandas.lazy.physical import PhysicalHashAggregate

        calls = []
        original = PhysicalHashAggregate._execute_arrow_table_groupby

        def spy(self, *args, **kwargs):
            calls.append(True)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(PhysicalHashAggregate, "_execute_arrow_table_groupby", spy)

        rng = np.random.default_rng(0)
        n = 2_000
        df = pd.DataFrame(
            {
                "id": np.arange(n),  # leading NumPy column
                "value1": rng.standard_normal(n),
                "group": pd.array(rng.choice(["A", "B", "C"], n)),  # Arrow key
            }
        )
        result = (
            df.select()
            .group_by("group")
            .agg(col("value1").sum().alias("total"))
            .collect(use_physical_planner=True)
        )

        assert calls, "Arrow-keyed groupby did not take the Arrow path"
        expected = df.groupby("group")["value1"].sum().sort_index()
        got = result.sort_values("group")["total"].to_numpy(dtype="float64")
        # Approximate: Arrow's multi-threaded sum accumulates in a
        # different order than pandas' sequential sum
        assert np.allclose(got, expected.to_numpy())

    def test_fused_filter_select_prunes_before_masking(self):
        # Regression: the fused pipeline applied the deferred filter mask
        # to ALL scan columns (including string payloads) before the
        # projection dropped them. Correctness check over a wide
        # mixed-dtype frame against the eager path.
        rng = np.random.default_rng(7)
        n = 5_000
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "group": pd.array(rng.choice(["A", "B", "C"], n)),
                "value1": rng.standard_normal(n),
                "value2": rng.standard_normal(n),
                "text": pd.array([f"t{i % 50}" for i in range(n)]),
            }
        )
        ldf = df.select().filter(col("value1") > 0).select("id", "group", "value1")
        eager = ldf.collect()
        physical = ldf.collect(use_physical_planner=True)
        tm.assert_frame_equal(eager, physical, check_dtype=False)

    def test_fused_project_computed_expr_keeps_referenced_inputs(self):
        # The pre-mask pruning must keep columns referenced INSIDE computed
        # expressions, not just projection output names: here 'value2' is
        # consumed by the expression but absent from the output.
        rng = np.random.default_rng(8)
        n = 3_000
        df = pd.DataFrame(
            {
                "a": rng.standard_normal(n),
                "value2": rng.standard_normal(n),
                "junk": pd.array([f"j{i % 10}" for i in range(n)]),
            }
        )
        ldf = df.select().filter(col("a") > 0).select((col("value2") * 2).alias("d"))
        eager = ldf.collect()
        physical = ldf.collect(use_physical_planner=True)
        tm.assert_frame_equal(eager, physical, check_dtype=False)


class TestOutputDtypeContract:
    """The physical engine returns the same dtypes the eager path would.

    NumPy-backed numeric/bool, default ``str`` for strings, and genuine
    ``pd.ArrowDtype`` columns preserved. Internal Arrow kernels (acero
    groupby/join) must not leak Arrow dtypes onto NumPy-sourced columns.
    See convert.arrays_to_dataframe.
    """

    def _frame(self):
        # NumPy numerics + plain-string columns: the common case where the
        # physical output dtype must match eager exactly.
        return pd.DataFrame(
            {
                "i": np.arange(6, dtype="int64"),
                "f": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                "g": ["x", "y", "x", "y", "x", "y"],
                "v": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            }
        )

    @pytest.mark.parametrize(
        "build",
        [
            lambda d: d.select(),
            lambda d: d.select().filter(col("i") > 1),
            lambda d: d.select().sort("f", descending=True),
            lambda d: d.select().with_columns((col("i") * 2).alias("i2")),
            lambda d: d.select().group_by("g").agg(col("v").sum().alias("s")),
            lambda d: d.select().filter(col("i") >= 0).head(3),
        ],
    )
    def test_physical_dtypes_match_eager(self, build):
        ldf = build(self._frame())
        eager = ldf.collect()
        physical = ldf.collect(use_physical_planner=True)
        assert {c: str(physical[c].dtype) for c in physical.columns} == {
            c: str(eager[c].dtype) for c in eager.columns
        }

    def test_groupby_numeric_is_numpy_not_arrow(self):
        # Regression: acero groupby returned double[pyarrow] for a NumPy
        # float source column.
        df = pd.DataFrame({"g": ["a", "b", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
        out = df.select().group_by("g").agg(col("v").sum().alias("s"))
        res = out.collect(use_physical_planner=True)
        assert str(res["s"].dtype) == "float64"

    def test_float_nulls_stay_numpy_nan(self):
        # NumPy float with NaN must not be promoted to nullable Float64.
        df = pd.DataFrame({"f": np.array([1.0, np.nan, 3.0])})
        res = df.select().filter(col("f") >= 0).collect(use_physical_planner=True)
        assert str(res["f"].dtype) == "float64"

    def test_strings_normalized_to_default_str(self):
        # Both plain-list and explicit ``string`` inputs come out ``str``.
        df = pd.DataFrame({"p": ["a", "b", "c"], "s": pd.array(["d", "e", "f"])})
        res = df.select().collect(use_physical_planner=True)
        assert str(res["p"].dtype) == "str"
        assert str(res["s"].dtype) == "str"

    def test_left_join_widened_int_is_numpy_float(self):
        # A NumPy int column gaining nulls via left join widens to NumPy
        # float64 (np.nan), matching eager pd.merge.
        left = pd.DataFrame({"k": [1, 2, 3], "lv": [1, 2, 3]})
        right = pd.DataFrame({"k": [1, 2], "rv": [10, 20]})
        res = (
            left.select()
            .join(right.select(), on="k", how="left")
            .collect(use_physical_planner=True)
        )
        assert str(res["rv"].dtype) == "float64"
        eager = left.merge(right, on="k", how="left")
        assert str(eager["rv"].dtype) == "float64"

    def test_genuine_arrow_dtype_preserved(self):
        # pd.ArrowDtype inputs are detectable (schema arrow_type set) and
        # preserved through the engine, matching eager passthrough.
        import pyarrow as pa

        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype=pd.ArrowDtype(pa.int64()))})
        res = df.select().filter(col("x") > 0).collect(use_physical_planner=True)
        assert isinstance(res["x"].dtype, pd.ArrowDtype)


class TestCollectDoesNotAliasSource:
    """A collected result must never share buffers with the source frame.

    Pipeline breakers assemble output with copy=False (every column is a
    fresh take/aggregate); projections keep the consolidating copy because
    a passthrough column is a view of source data. Both must be isolated.
    """

    def _src(self):
        return pd.DataFrame(
            {
                "a": np.arange(10, dtype="float64"),
                "b": np.arange(10, 20, dtype="float64"),
                "g": np.array([0, 1] * 5),
            }
        )

    @pytest.mark.parametrize(
        "build",
        [
            lambda d: d.select().sort("a", descending=True),
            lambda d: d.select().distinct(),
            lambda d: d.select().sort("a").head(3),
            lambda d: d.select().filter(col("a") >= 0),
            # passthrough projection: 'a' would be a view of the source -
            # the consolidating copy must isolate it.
            lambda d: d.select().with_columns((col("a") + col("b")).alias("s")),
            lambda d: d.select("a", "b"),
        ],
    )
    def test_mutating_result_leaves_source_unchanged(self, build):
        src = self._src()
        res = build(src).collect(use_physical_planner=True)
        before = src["a"].to_numpy().copy()
        if "a" in res.columns:
            res.iloc[:, res.columns.get_loc("a")] = -999.0
        tm.assert_numpy_array_equal(src["a"].to_numpy(), before)

    @pytest.mark.parametrize(
        "build",
        [
            lambda d: d.select().sort("a", descending=True),
            lambda d: d.select().group_by("g").agg(col("a").sum().alias("a")),
            lambda d: d.select().with_columns((col("a") * 2).alias("a2")),
        ],
    )
    def test_mutating_source_leaves_result_unchanged(self, build):
        src = self._src()
        res = build(src).collect(use_physical_planner=True)
        snapshot = res.copy(deep=True)
        src.iloc[:, src.columns.get_loc("a")] = 123.0
        tm.assert_frame_equal(res, snapshot)
