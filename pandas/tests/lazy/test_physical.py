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

        # Materialize's input should be Aggregate
        assert isinstance(sort_input.input, PhysicalHashAggregate)
        assert sort_input.input.is_pipeline_breaker  # Aggregate is a breaker

        # Aggregate's input should be Materialize (for aggregate)
        agg_input = sort_input.input.input
        assert isinstance(agg_input, PhysicalMaterialize)
        assert agg_input.reason == "aggregate"


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
        assert str(result["text"].dtype) == "string"
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
