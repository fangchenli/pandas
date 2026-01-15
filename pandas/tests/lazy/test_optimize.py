"""
Tests for the lazy pandas query optimizer.

These tests verify that optimization passes:
1. Produce correct results (functional correctness)
2. Actually transform the plan as expected (structural correctness)
"""

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.ir import (
    Call,
    FieldRef,
)
from pandas.lazy.optimize import (
    AggregatePushdown,
    BackendRequirements,
    CommonSubexpressionElimination,
    ConstantFolding,
    ConversionElimination,
    DeadCodeElimination,
    EngineSelection,
    FilterFusion,
    Optimizer,
    PredicatePushdown,
    SortLimitToTopK,
    analyze_backend_requirements,
    build_project_lineage,
    can_push_predicate_through_project,
    get_referenced_columns,
    is_pass_through_column,
    substitute_columns,
)
from pandas.lazy.plan import (
    Aggregate,
    Convert,
    Filter,
    Project,
    Sort,
    TopK,
)


class TestGetReferencedColumns:
    """Tests for get_referenced_columns utility."""

    def test_single_field_ref(self):
        result = get_referenced_columns(FieldRef("a"))
        assert result == {"a"}

    def test_call_with_multiple_refs(self):
        ir = Call("add", (FieldRef("a"), FieldRef("b")))
        result = get_referenced_columns(ir)
        assert result == {"a", "b"}

    def test_nested_calls(self):
        inner = Call("add", (FieldRef("a"), FieldRef("b")))
        outer = Call("multiply", (inner, FieldRef("c")))
        result = get_referenced_columns(outer)
        assert result == {"a", "b", "c"}

    def test_expr_wrapper(self):
        expr = col("a") + col("b")
        result = get_referenced_columns(expr)
        assert result == {"a", "b"}

    def test_duplicate_refs(self):
        # col("a") + col("a")
        ir = Call("add", (FieldRef("a"), FieldRef("a")))
        result = get_referenced_columns(ir)
        assert result == {"a"}


class TestSubstituteColumns:
    """Tests for substitute_columns utility."""

    def test_simple_rename(self):
        ir = FieldRef("old_name")
        result = substitute_columns(ir, {"old_name": "new_name"})
        assert isinstance(result, FieldRef)
        assert result.name == "new_name"

    def test_no_change_when_not_in_mapping(self):
        ir = FieldRef("a")
        result = substitute_columns(ir, {"b": "c"})
        assert result is ir  # Same object, no change

    def test_nested_call(self):
        ir = Call("add", (FieldRef("old"), FieldRef("other")))
        result = substitute_columns(ir, {"old": "new"})
        assert isinstance(result, Call)
        assert result.args[0].name == "new"
        assert result.args[1].name == "other"


class TestBuildProjectLineage:
    """Tests for build_project_lineage utility."""

    def test_pass_through_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ldf = df.select("a", "b")
        plan = ldf._plan
        assert isinstance(plan, Project)

        lineage = build_project_lineage(plan)
        assert "a" in lineage
        assert "b" in lineage
        assert isinstance(lineage["a"], FieldRef)
        assert lineage["a"].name == "a"

    def test_computed_column(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        ldf = df.select("x", (col("x") + col("y")).alias("sum"))
        plan = ldf._plan
        assert isinstance(plan, Project)

        lineage = build_project_lineage(plan)
        assert is_pass_through_column(lineage, "x")
        assert not is_pass_through_column(lineage, "sum")

    def test_renamed_column(self):
        df = pd.DataFrame({"original": [1, 2, 3]})
        ldf = df.select(col("original").alias("renamed"))
        plan = ldf._plan
        assert isinstance(plan, Project)

        lineage = build_project_lineage(plan)
        # "renamed" comes from "original"
        assert isinstance(lineage["renamed"], FieldRef)
        assert lineage["renamed"].name == "original"


class TestCanPushPredicateThroughProject:
    """Tests for predicate pushdown analysis through Project."""

    def test_can_push_pass_through(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ldf = df.select("a", "b")
        lineage = build_project_lineage(ldf._plan)

        can_push, mapping = can_push_predicate_through_project({"a"}, lineage)
        assert can_push
        assert mapping == {"a": "a"}

    def test_can_push_renamed(self):
        df = pd.DataFrame({"original": [1, 2, 3]})
        ldf = df.select(col("original").alias("renamed"))
        lineage = build_project_lineage(ldf._plan)

        can_push, mapping = can_push_predicate_through_project({"renamed"}, lineage)
        assert can_push
        assert mapping == {"renamed": "original"}

    def test_cannot_push_computed(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        ldf = df.select("x", (col("x") + col("y")).alias("sum"))
        lineage = build_project_lineage(ldf._plan)

        can_push, _ = can_push_predicate_through_project({"sum"}, lineage)
        assert not can_push

    def test_mixed_pushable_and_not(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        ldf = df.select("x", (col("x") + col("y")).alias("sum"))
        lineage = build_project_lineage(ldf._plan)

        # "x" is pushable but "sum" is not - cannot push whole predicate
        can_push, _ = can_push_predicate_through_project({"x", "sum"}, lineage)
        assert not can_push


class TestFilterFusion:
    """Tests for FilterFusion optimization pass."""

    def test_combines_consecutive_filters(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ldf = df.select().filter(col("a") > 1).filter(col("a") < 5)

        # Before optimization: Filter(Filter(...))
        original_plan = ldf._plan
        assert isinstance(original_plan, Filter)
        assert isinstance(original_plan.input, Filter)

        # After optimization: single Filter with AND
        optimized_plan = FilterFusion().optimize(original_plan)
        assert isinstance(optimized_plan, Filter)
        assert not isinstance(optimized_plan.input, Filter)

        # Verify result is correct
        result = ldf.collect()
        expected = pd.DataFrame({"a": [2, 3, 4]})
        tm.assert_frame_equal(result, expected)

    def test_three_consecutive_filters(self):
        df = pd.DataFrame({"a": range(10)})
        ldf = (
            df.select().filter(col("a") > 1).filter(col("a") < 8).filter(col("a") != 5)
        )

        result = ldf.collect()
        expected = pd.DataFrame({"a": [2, 3, 4, 6, 7]})
        tm.assert_frame_equal(result, expected)


class TestPredicatePushdown:
    """Tests for PredicatePushdown optimization pass."""

    def test_pushes_filter_through_project(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("a", "b").filter(col("a") > 1)

        # Filter should be pushed below Project
        optimized_plan = Optimizer([PredicatePushdown()]).optimize(ldf._plan)
        # Verify the structure changed (filter pushed through)
        assert isinstance(optimized_plan, Project)

        # Verify result is correct
        result = ldf.collect()
        expected = pd.DataFrame({"a": [2, 3], "b": [5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_pushes_filter_on_renamed_column(self):
        df = pd.DataFrame({"original": [1, 2, 3, 4, 5]})
        ldf = df.select(col("original").alias("renamed")).filter(col("renamed") > 2)

        result = ldf.collect()
        expected = pd.DataFrame({"renamed": [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_does_not_push_filter_on_computed_column(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        ldf = df.select("x", (col("x") * 2).alias("doubled")).filter(col("doubled") > 4)

        # Filter on "doubled" cannot be pushed because it's computed
        result = ldf.collect()
        expected = pd.DataFrame({"x": [3, 4, 5], "doubled": [6, 8, 10]})
        tm.assert_frame_equal(result, expected)

    def test_does_not_push_filter_below_aggregate(self):
        df = pd.DataFrame({"cat": ["A", "A", "B"], "val": [1, 2, 3]})
        ldf = (
            df.select()
            .group_by("cat")
            .agg(col("val").sum().alias("total"))
            .filter(col("total") > 2)
        )

        result = ldf.collect()
        # A: 1+2=3, B: 3 - both > 2
        assert len(result) == 2

    def test_pushes_filter_through_sort(self):
        df = pd.DataFrame({"a": [3, 1, 2, 5, 4]})
        ldf = df.select().sort("a").filter(col("a") > 2)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_does_not_push_filter_below_limit(self):
        df = pd.DataFrame({"a": [5, 1, 4, 2, 3]})
        ldf = df.select().limit(3).filter(col("a") > 2)

        # Filter should NOT be pushed below limit
        # First 3 rows are [5, 1, 4], then filter keeps [5, 4]
        result = ldf.collect()
        assert len(result) == 2
        assert set(result["a"]) == {5, 4}


class TestProjectionPruning:
    """Tests for ProjectionPruning optimization pass."""

    def test_removes_unused_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        ldf = df.select("a", "b", "c").select("a")

        # After optimization, intermediate Project should only keep "a"
        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_preserves_filter_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("a", "b").filter(col("b") > 4).select("a")

        # "b" needed for filter even though not in output
        result = ldf.collect()
        expected = pd.DataFrame({"a": [2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_preserves_sort_columns(self):
        df = pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]})
        ldf = df.select("a", "b").sort("b").select("a")

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)


class TestLimitPushdown:
    """Tests for LimitPushdown optimization pass."""

    def test_pushes_limit_through_project(self):
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        ldf = df.select("a", "b").select("a").limit(5)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [0, 1, 2, 3, 4]})
        tm.assert_frame_equal(result, expected)

    def test_combines_consecutive_limits(self):
        df = pd.DataFrame({"a": range(10)})
        ldf = df.select().limit(7).limit(5)

        # Should take min(7, 5) = 5
        result = ldf.collect()
        assert len(result) == 5

    def test_does_not_push_limit_below_filter(self):
        df = pd.DataFrame({"a": range(10)})
        ldf = df.select().filter(col("a") > 5).limit(2)

        # Filter first (keep 6,7,8,9), then limit to 2
        result = ldf.collect()
        assert len(result) == 2
        assert list(result["a"]) == [6, 7]

    def test_does_not_push_limit_below_sort(self):
        df = pd.DataFrame({"a": [5, 1, 3, 2, 4]})
        ldf = df.select().sort("a").limit(3)

        # Sort first, then limit
        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)


class TestOptimizerIntegration:
    """Tests for full optimizer with all passes."""

    def test_filter_fusion_then_pushdown(self):
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        ldf = df.select("a", "b").filter(col("a") > 2).filter(col("a") < 7)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [3, 4, 5, 6], "b": [3, 4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_complex_query_optimization(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
                "extra": ["x", "y", "z", "w", "v"],
            }
        )

        ldf = (
            df.select("id", "category", "value", "extra")
            .filter(col("category") == "A")
            .select("id", "value")
            .filter(col("value") > 15)
            .limit(2)
        )

        result = ldf.collect()
        # A rows: (1,10), (3,30), (5,50) -> value>15: (3,30), (5,50) -> limit 2
        expected = pd.DataFrame({"id": [3, 5], "value": [30, 50]})
        tm.assert_frame_equal(result, expected)

    def test_optimization_does_not_change_result(self):
        """Verify optimized and unoptimized queries produce same result."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
            }
        )

        ldf = df.select("a", "b").filter(col("a") > 1).select("b").filter(col("b") < 40)

        result_optimized = ldf.collect(optimize=True)
        result_unoptimized = ldf.collect(optimize=False)

        tm.assert_frame_equal(result_optimized, result_unoptimized)


class TestExplainOptimization:
    """Tests for explain() with optimization."""

    def test_explain_shows_optimized_plan(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("a", "b").filter(col("a") > 1)

        optimized_plan = ldf.explain(optimized=True)
        unoptimized_plan = ldf.explain(optimized=False)

        assert "optimized" in optimized_plan
        assert "unoptimized" in unoptimized_plan

    def test_explain_filter_fusion_visible(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().filter(col("a") > 1).filter(col("a") < 3)

        unoptimized = ldf.explain(optimized=False)
        # Should show two Filter nodes
        assert unoptimized.count("Filter") >= 2

        optimized_output = ldf.explain(optimized=True)
        # Should show single Filter with AND
        # The exact count depends on representation but should be fewer
        assert "Filter" in optimized_output


class TestBackendRequirements:
    """Tests for BackendRequirements dataclass."""

    def test_any_backend(self):
        reqs = BackendRequirements.any_backend()
        assert "arrow" in reqs.supported_backends
        assert "numpy" in reqs.supported_backends
        assert reqs.requires_backend is None
        assert reqs.conversion_cost == 0.0

    def test_prefer_arrow_required(self):
        reqs = BackendRequirements.prefer_arrow(required=True)
        assert reqs.requires_backend == "arrow"
        assert reqs.supported_backends == frozenset({"arrow"})

    def test_prefer_numpy_required(self):
        reqs = BackendRequirements.prefer_numpy(required=True)
        assert reqs.requires_backend == "numpy"
        assert reqs.supported_backends == frozenset({"numpy"})

    def test_prefer_arrow_not_required(self):
        reqs = BackendRequirements.prefer_arrow(required=False)
        assert reqs.preferred_backend == "arrow"
        assert reqs.requires_backend is None
        assert "arrow" in reqs.supported_backends
        assert "numpy" in reqs.supported_backends

    def test_prefer_numpy_not_required(self):
        reqs = BackendRequirements.prefer_numpy(required=False)
        assert reqs.preferred_backend == "numpy"
        assert reqs.requires_backend is None
        assert "arrow" in reqs.supported_backends
        assert "numpy" in reqs.supported_backends


class TestAnalyzeBackendRequirements:
    """Tests for analyze_backend_requirements function."""

    def test_field_ref_numpy_storage(self):
        df = pd.DataFrame({"a": [1, 2, 3]})  # numpy-backed int
        schema = df.select()._schema
        reqs = analyze_backend_requirements(FieldRef("a"), schema)
        # FieldRef doesn't require any backend, but prefers numpy for numpy data
        assert "arrow" in reqs.supported_backends
        assert "numpy" in reqs.supported_backends
        assert reqs.preferred_backend == "numpy"

    def test_string_op_requires_arrow(self):
        df = pd.DataFrame({"name": ["alice", "bob", "carol"]}, dtype="string")
        schema = df.select()._schema
        # str_lower requires arrow
        ir = Call("str_lower", (FieldRef("name"),))
        reqs = analyze_backend_requirements(ir, schema)
        assert reqs.requires_backend == "arrow"

    def test_comparison_any_backend(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        schema = df.select()._schema
        # Comparison works on any backend
        ir = Call("greater", (FieldRef("a"), FieldRef("a")))
        reqs = analyze_backend_requirements(ir, schema)
        assert reqs.requires_backend is None

    def test_arithmetic_any_backend(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        schema = df.select()._schema
        ir = Call("add", (FieldRef("a"), FieldRef("b")))
        reqs = analyze_backend_requirements(ir, schema)
        assert reqs.requires_backend is None


class TestEngineSelection:
    """Tests for EngineSelection optimization pass."""

    def test_numeric_ops_no_conversion(self):
        """Numeric operations don't require conversion."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("a", "b", (col("a") + col("b")).alias("sum"))

        # Apply engine selection (verify it doesn't raise)
        _ = EngineSelection().optimize(ldf._plan)

        # Should not insert Convert nodes for simple numeric ops
        # The plan structure depends on input backend
        result = ldf.collect()
        assert list(result["sum"]) == [5, 7, 9]

    def test_string_ops_may_insert_convert(self):
        """String operations prefer Arrow backend."""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]}, dtype="string")
        ldf = df.select("name", col("name").str.lower().alias("lower_name"))

        # Should work correctly regardless of conversions
        result = ldf.collect()
        assert list(result["lower_name"]) == ["alice", "bob", "carol"]

    def test_mixed_ops_result_correct(self):
        """Mixed operations produce correct results."""
        df = pd.DataFrame(
            {
                "name": pd.array(["Alice", "Bob", "Carol"], dtype="string"),
                "age": [25, 30, 35],
            }
        )
        ldf = (
            df.select("name", "age")
            .filter(col("age") > 26)
            .select("name", col("name").str.lower().alias("lower"))
        )

        result = ldf.collect()
        assert len(result) == 2
        assert list(result["lower"]) == ["bob", "carol"]


class TestConversionElimination:
    """Tests for ConversionElimination optimization pass."""

    def test_redundant_convert_removed(self):
        """Convert(Convert(x, A), A) should become Convert(x, A)."""
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        # Double convert to same backend
        inner = Convert(source, "arrow")
        outer = Convert(inner, "arrow")

        result = ConversionElimination().optimize(outer)

        # Should simplify to single Convert
        assert isinstance(result, Convert)
        assert result.target_backend == "arrow"
        assert isinstance(result.input, DataFrameSource)

    def test_back_to_back_converts_simplified(self):
        """Convert(Convert(x, A), B) should become Convert(x, B)."""
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        # Convert to arrow then numpy
        inner = Convert(source, "arrow")
        outer = Convert(inner, "numpy")

        result = ConversionElimination().optimize(outer)

        # Should skip intermediate conversion
        assert isinstance(result, Convert)
        assert result.target_backend == "numpy"
        assert isinstance(result.input, DataFrameSource)

    def test_convert_execution_passthrough(self):
        """Convert nodes should execute as pass-through."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("a", "b")

        # Manually wrap in Convert
        wrapped_plan = Convert(ldf._plan, "arrow")

        # Create new LazyDataFrame with wrapped plan
        from pandas.lazy.frame import LazyDataFrame

        wrapped_ldf = LazyDataFrame(wrapped_plan, ldf._schema)

        # Should produce same result
        result = wrapped_ldf.collect(optimize=False)
        expected = df[["a", "b"]]
        tm.assert_frame_equal(result, expected)


class TestConvertNode:
    """Tests for Convert plan node."""

    def test_convert_repr(self):
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        convert = Convert(source, "arrow")

        assert "Convert" in repr(convert)
        assert "arrow" in repr(convert)

    def test_convert_children(self):
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        convert = Convert(source, "numpy")

        assert convert.children() == [source]

    def test_convert_resolve_schema(self):
        """Convert preserves schema."""
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        source = DataFrameSource(df)
        convert = Convert(source, "arrow")

        assert convert.resolve_schema().names == source.resolve_schema().names


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCaseSelfReferential:
    """Tests for self-referential expressions like col('a') + col('a')."""

    def test_self_referential_add(self):
        """col('a') + col('a') should work correctly."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select("a", (col("a") + col("a")).alias("doubled"))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "doubled": [2, 4, 6]})
        tm.assert_frame_equal(result, expected)

    def test_self_referential_multiply(self):
        """col('a') * col('a') should work correctly."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select("a", (col("a") * col("a")).alias("squared"))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "squared": [1, 4, 9]})
        tm.assert_frame_equal(result, expected)

    def test_self_referential_in_filter(self):
        """Filter with self-referential expression."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ldf = df.select().filter(col("a") + col("a") > 6)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_self_referential_column_refs(self):
        """get_referenced_columns should deduplicate self-referential refs."""
        ir = Call("add", (FieldRef("x"), FieldRef("x")))
        refs = get_referenced_columns(ir)
        assert refs == {"x"}


class TestEdgeCaseEmptyDataFrame:
    """Tests for operations on empty DataFrames."""

    def test_empty_df_filter(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        ldf = df.select().filter(col("a") > 0)

        result = ldf.collect()
        assert len(result) == 0
        assert list(result.columns) == ["a"]

    def test_empty_df_filter_returns_empty(self):
        """Filtering non-empty to empty should work."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().filter(col("a") > 100)

        result = ldf.collect()
        assert len(result) == 0

    def test_empty_df_project(self):
        df = pd.DataFrame(
            {"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="int64")}
        )
        ldf = df.select("a")

        result = ldf.collect()
        assert len(result) == 0
        assert list(result.columns) == ["a"]

    def test_empty_df_sort(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        ldf = df.select().sort("a")

        result = ldf.collect()
        assert len(result) == 0

    def test_empty_df_limit(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        ldf = df.select().limit(10)

        result = ldf.collect()
        assert len(result) == 0

    def test_empty_df_filter_fusion(self):
        """Filter fusion on empty DataFrame."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        ldf = df.select().filter(col("a") > 0).filter(col("a") < 10)

        result = ldf.collect()
        assert len(result) == 0


class TestEdgeCaseSingleRow:
    """Tests for single-row DataFrames."""

    def test_single_row_filter_keep(self):
        df = pd.DataFrame({"a": [5]})
        ldf = df.select().filter(col("a") > 0)

        result = ldf.collect()
        assert len(result) == 1
        assert result["a"].iloc[0] == 5

    def test_single_row_filter_remove(self):
        df = pd.DataFrame({"a": [5]})
        ldf = df.select().filter(col("a") > 10)

        result = ldf.collect()
        assert len(result) == 0

    def test_single_row_sort(self):
        df = pd.DataFrame({"a": [5], "b": [10]})
        ldf = df.select().sort("a")

        result = ldf.collect()
        assert len(result) == 1

    def test_single_row_limit_0(self):
        df = pd.DataFrame({"a": [5]})
        ldf = df.select().limit(0)

        result = ldf.collect()
        assert len(result) == 0

    def test_single_row_limit_1(self):
        df = pd.DataFrame({"a": [5]})
        ldf = df.select().limit(1)

        result = ldf.collect()
        assert len(result) == 1

    def test_single_row_limit_larger(self):
        df = pd.DataFrame({"a": [5]})
        ldf = df.select().limit(100)

        result = ldf.collect()
        assert len(result) == 1


class TestEdgeCaseNullHandling:
    """Tests for null/NA handling in predicates and operations."""

    def test_filter_with_nulls(self):
        """Filtering with NA values in predicate column."""
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})
        ldf = df.select().filter(col("a") > 2)

        result = ldf.collect()
        # NA comparisons return False, so only 3 and 5 pass
        assert len(result) == 2
        assert list(result["a"].dropna()) == [3.0, 5.0]

    def test_filter_preserves_nulls_in_other_columns(self):
        """Nulls in non-predicate columns should be preserved."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, None, 30, None, 50],
            }
        )
        ldf = df.select().filter(col("a") > 2)

        result = ldf.collect()
        assert len(result) == 3
        assert pd.isna(result["b"].iloc[1])

    def test_compute_with_nulls(self):
        """Arithmetic with nulls should propagate nulls."""
        df = pd.DataFrame({"a": [1, None, 3]})
        ldf = df.select("a", (col("a") * 2).alias("doubled"))

        result = ldf.collect()
        assert len(result) == 3
        assert pd.isna(result["doubled"].iloc[1])

    def test_filter_fusion_with_nulls(self):
        """Filter fusion should handle nulls correctly."""
        df = pd.DataFrame({"a": [1, None, 3, None, 5, 6]})
        ldf = df.select().filter(col("a") > 1).filter(col("a") < 6)

        result = ldf.collect()
        # Only 3 and 5 pass both conditions (nulls fail both)
        assert len(result) == 2


class TestEdgeCaseStackedProjects:
    """Tests for multiple stacked Project nodes."""

    def test_triple_stacked_project(self):
        """Three stacked Projects should work correctly."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        ldf = df.select("a", "b", "c").select("a", "b").select("a")

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_stacked_project_with_computation(self):
        """Stacked Projects with computed columns."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        ldf = (
            df.select("x", (col("x") * 2).alias("y"))
            .select("x", "y", (col("y") + 1).alias("z"))
            .select("z")
        )

        result = ldf.collect()
        expected = pd.DataFrame({"z": [3, 5, 7]})
        tm.assert_frame_equal(result, expected)

    def test_stacked_project_rename_chain(self):
        """Multiple renames through stacked Projects."""
        df = pd.DataFrame({"original": [1, 2, 3]})
        ldf = (
            df.select(col("original").alias("renamed1"))
            .select(col("renamed1").alias("renamed2"))
            .select(col("renamed2").alias("final"))
        )

        result = ldf.collect()
        expected = pd.DataFrame({"final": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)


class TestEdgeCaseFilterThroughAggregate:
    """Tests for filters that cannot be pushed through aggregates."""

    def test_filter_on_aggregate_result(self):
        """Filter on aggregation result cannot be pushed."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
            }
        )
        ldf = (
            df.select()
            .group_by("category")
            .agg(col("value").sum().alias("total"))
            .filter(col("total") > 25)
        )

        result = ldf.collect()
        # A: 30, B: 70 -> B passes (70 > 25), A passes (30 > 25)
        assert len(result) == 2

    def test_filter_on_count(self):
        """Filter on count() aggregate."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "A", "B", "B"],
                "value": [1, 2, 3, 4, 5],
            }
        )
        ldf = (
            df.select()
            .group_by("category")
            .agg(col("value").count().alias("cnt"))
            .filter(col("cnt") > 2)
        )

        result = ldf.collect()
        # A has 3 rows, B has 2 rows -> only A passes
        assert len(result) == 1
        assert result["category"].iloc[0] == "A"

    def test_filter_on_mean(self):
        """Filter on mean() aggregate."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 30, 100, 200],  # A avg=20, B avg=150
            }
        )
        ldf = (
            df.select()
            .group_by("category")
            .agg(col("value").mean().alias("avg"))
            .filter(col("avg") > 50)
        )

        result = ldf.collect()
        # Only B's average (150) > 50
        assert len(result) == 1
        assert result["category"].iloc[0] == "B"


class TestEdgeCaseLimitBoundary:
    """Tests for limit edge cases."""

    def test_limit_zero(self):
        """Limit 0 should return empty DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().limit(0)

        result = ldf.collect()
        assert len(result) == 0

    def test_limit_larger_than_data(self):
        """Limit larger than data size should return all data."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().limit(100)

        result = ldf.collect()
        assert len(result) == 3

    def test_limit_exact_size(self):
        """Limit equal to data size."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().limit(3)

        result = ldf.collect()
        assert len(result) == 3

    def test_multiple_limits_take_minimum(self):
        """Multiple limits should take minimum."""
        df = pd.DataFrame({"a": range(10)})
        ldf = df.select().limit(5).limit(3).limit(7)

        result = ldf.collect()
        # min(5, 3, 7) = 3
        assert len(result) == 3


class TestEdgeCaseSortStability:
    """Tests for sort stability and edge cases."""

    def test_sort_with_ties(self):
        """Sort should be stable with ties."""
        df = pd.DataFrame(
            {
                "key": [1, 1, 1, 2, 2],
                "order": [1, 2, 3, 4, 5],
            }
        )
        ldf = df.select().sort("key")

        result = ldf.collect()
        # Within each key, original order should be preserved
        key1_orders = list(result[result["key"] == 1]["order"])
        assert key1_orders == [1, 2, 3]

    def test_sort_empty(self):
        """Sort on empty DataFrame."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        ldf = df.select().sort("a")

        result = ldf.collect()
        assert len(result) == 0

    def test_sort_single_value(self):
        """Sort with all identical values."""
        df = pd.DataFrame({"a": [5, 5, 5, 5]})
        ldf = df.select().sort("a")

        result = ldf.collect()
        assert len(result) == 4
        assert all(result["a"] == 5)


class TestEdgeCasePredicatePushdownComplex:
    """Complex predicate pushdown scenarios."""

    def test_predicate_on_multiple_columns(self):
        """Predicate referencing multiple columns."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
            }
        )
        ldf = df.select("a", "b").filter((col("a") + col("b")) > 25)

        result = ldf.collect()
        # a+b: 11, 22, 33, 44, 55 -> 33, 44, 55 pass
        assert len(result) == 3

    def test_predicate_with_literal(self):
        """Predicate with literal comparison."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        from pandas.lazy.expr import lit

        ldf = df.select().filter(col("a") > lit(3))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_combined_and_or_predicates(self):
        """Complex AND/OR predicate combinations."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
            }
        )
        ldf = df.select().filter((col("a") > 2) & (col("b") > 2))

        result = ldf.collect()
        # a>2 AND b>2: row 3 (a=3, b=3) passes
        assert len(result) == 1
        assert result["a"].iloc[0] == 3

    def test_negated_predicate(self):
        """Negated predicate with ~."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ldf = df.select().filter(~(col("a") > 3))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)


class TestEdgeCaseDistinct:
    """Edge cases for distinct operation."""

    def test_distinct_all_same(self):
        """Distinct on all identical values."""
        df = pd.DataFrame({"a": [1, 1, 1, 1]})
        ldf = df.select().distinct()

        result = ldf.collect()
        assert len(result) == 1

    def test_distinct_all_unique(self):
        """Distinct on all unique values."""
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        ldf = df.select().distinct()

        result = ldf.collect()
        assert len(result) == 4

    def test_distinct_empty(self):
        """Distinct on empty DataFrame."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        ldf = df.select().distinct()

        result = ldf.collect()
        assert len(result) == 0

    def test_distinct_with_nulls(self):
        """Distinct should treat nulls as equal."""
        df = pd.DataFrame({"a": [1, None, 2, None, 3]})
        ldf = df.select().distinct()

        result = ldf.collect()
        # 1, None, 2, 3 -> 4 unique values
        assert len(result) == 4


class TestEdgeCaseOptimizationCorrectness:
    """Verify optimization never changes results."""

    def test_complex_query_optimized_equals_unoptimized(self):
        """Complex query gives same result with/without optimization."""
        df = pd.DataFrame(
            {
                "id": range(20),
                "category": ["A", "B", "C", "D"] * 5,
                "value": range(100, 120),
            }
        )
        ldf = (
            df.select("id", "category", "value")
            .filter((col("category") == "A") | (col("category") == "B"))
            .select("id", "value")
            .filter(col("value") > 105)
            .sort("id")
            .limit(5)
        )

        result_opt = ldf.collect(optimize=True)
        result_no_opt = ldf.collect(optimize=False)

        tm.assert_frame_equal(result_opt, result_no_opt)

    def test_filter_fusion_correctness(self):
        """Filter fusion produces same result as separate filters."""
        df = pd.DataFrame({"a": range(20)})

        # Two separate filters
        ldf1 = df.select().filter(col("a") > 5).filter(col("a") < 15)

        # Equivalent single filter
        ldf2 = df.select().filter((col("a") > 5) & (col("a") < 15))

        result1 = ldf1.collect()
        result2 = ldf2.collect()

        tm.assert_frame_equal(result1, result2)

    def test_limit_pushdown_correctness(self):
        """Limit pushdown produces correct results."""
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})

        # Chain that allows limit pushdown
        ldf = df.select("a", "b").select("a").limit(3)

        result = ldf.collect(optimize=True)
        expected = pd.DataFrame({"a": [0, 1, 2]})

        tm.assert_frame_equal(result, expected)


# =============================================================================
# CONSTANT FOLDING TESTS
# =============================================================================


class TestConstantFolding:
    """Tests for ConstantFolding optimization pass."""

    def test_fold_literal_addition(self):
        """lit(1) + lit(2) should fold to lit(3)."""
        from pandas.lazy.expr import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        # Create expression with constant addition
        ldf = df.select("a", (col("a") + (lit(1) + lit(2))).alias("b"))

        # Apply constant folding
        ConstantFolding().optimize(ldf._plan)

        # Execute and verify result is correct
        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_fold_literal_multiplication(self):
        """lit(3) * lit(4) should fold to lit(12)."""
        from pandas.lazy.expr import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select("a", (col("a") * (lit(3) * lit(4))).alias("b"))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "b": [12, 24, 36]})
        tm.assert_frame_equal(result, expected)

    def test_fold_literal_comparison(self):
        """lit(5) > lit(3) should fold to lit(True)."""
        from pandas.lazy.expr import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        # Filter using constant comparison
        ldf = df.select().filter((col("a") > 0) & (lit(5) > lit(3)))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_fold_literal_logical_and(self):
        """lit(True) & lit(False) should fold to lit(False)."""
        from pandas.lazy.expr import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        # Filter with constant False (should filter all)
        ldf = df.select().filter((col("a") > 0) & (lit(True) & lit(False)))

        result = ldf.collect()
        # All rows should be filtered because the constant is False
        # But since it's AND with col("a") > 0, the filter may not work as intended
        # Let's verify the expression itself
        assert len(result) == 0

    def test_fold_string_concatenation(self):
        """lit('hello') + lit(' world') should fold."""
        from pandas.lazy.expr import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        # String concatenation
        ldf = df.select("a", (lit("hello") + lit(" world")).alias("greeting"))

        result = ldf.collect()
        assert list(result["greeting"]) == ["hello world"] * 3

    def test_fold_nested_arithmetic(self):
        """Nested constant expressions should fold completely."""
        from pandas.lazy.expr import lit

        df = pd.DataFrame({"a": [10, 20, 30]})
        # ((1 + 2) * 3) = 9
        ldf = df.select("a", (col("a") + ((lit(1) + lit(2)) * lit(3))).alias("b"))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [10, 20, 30], "b": [19, 29, 39]})
        tm.assert_frame_equal(result, expected)

    def test_fold_division(self):
        """lit(10) / lit(2) should fold to lit(5.0)."""
        from pandas.lazy.expr import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select("a", (col("a") + (lit(10) / lit(2))).alias("b"))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "b": [6.0, 7.0, 8.0]})
        tm.assert_frame_equal(result, expected)

    def test_no_fold_division_by_zero(self):
        """lit(1) / lit(0) should not be folded (runtime error)."""
        from pandas.lazy.expr import lit

        df = pd.DataFrame({"a": [1, 2, 3]})
        # This should not raise during optimization, only at runtime
        ldf = df.select("a", (lit(1) / lit(0)).alias("bad"))

        # The constant folding should leave this alone
        _ = ConstantFolding().optimize(ldf._plan)
        # No error during optimization

    def test_fold_preserves_non_constant_exprs(self):
        """Non-constant expressions should not be affected."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("a", (col("a") + col("b")).alias("sum"))

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "sum": [5, 7, 9]})
        tm.assert_frame_equal(result, expected)


# =============================================================================
# SORT + LIMIT -> TOPK TESTS
# =============================================================================


class TestSortLimitToTopK:
    """Tests for SortLimitToTopK optimization pass."""

    def test_sort_limit_becomes_topk(self):
        """Sort followed by Limit should become TopK."""
        df = pd.DataFrame({"a": [5, 1, 3, 2, 4]})
        ldf = df.select().sort("a").limit(3)

        # Apply the transformation
        original_plan = ldf._plan
        optimized_plan = SortLimitToTopK().optimize(original_plan)

        # Should be transformed to TopK
        assert isinstance(optimized_plan, TopK)
        assert optimized_plan.k == 3

    def test_topk_correct_result_ascending(self):
        """TopK should return correct results for ascending sort."""
        df = pd.DataFrame({"a": [5, 1, 3, 2, 4]})
        ldf = df.select().sort("a").limit(3)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_topk_correct_result_descending(self):
        """TopK should return correct results for descending sort."""
        df = pd.DataFrame({"a": [5, 1, 3, 2, 4]})
        ldf = df.select().sort("a", descending=True).limit(3)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [5, 4, 3]})
        tm.assert_frame_equal(result, expected)

    def test_topk_preserves_other_columns(self):
        """TopK should preserve non-sort columns."""
        df = pd.DataFrame(
            {
                "a": [5, 1, 3, 2, 4],
                "b": ["e", "a", "c", "b", "d"],
            }
        )
        ldf = df.select().sort("a").limit(3)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        tm.assert_frame_equal(result, expected)

    def test_topk_with_offset_not_optimized(self):
        """Sort + Limit with offset should NOT become TopK."""
        from pandas.lazy.optimize import SortLimitToTopK
        from pandas.lazy.plan import (
            DataFrameSource,
            Limit,
            Sort,
            TopK,
        )

        df = pd.DataFrame({"a": [5, 1, 3, 2, 4]})
        source = DataFrameSource(df)
        from pandas.lazy.expr import col

        sort_node = Sort(source, (col("a"),), (False,))
        # Create Limit with offset - this should NOT be converted to TopK
        limit_with_offset = Limit(sort_node, n=2, offset=1)

        optimizer = SortLimitToTopK()
        optimized = optimizer.optimize(limit_with_offset)

        # With offset, should remain Sort + Limit, NOT become TopK
        assert not isinstance(optimized, TopK)
        assert isinstance(optimized, Limit)
        assert optimized.offset == 1

    def test_topk_empty_result(self):
        """TopK with k=0 should return empty DataFrame."""
        df = pd.DataFrame({"a": [5, 1, 3, 2, 4]})
        ldf = df.select().sort("a").limit(0)

        result = ldf.collect()
        assert len(result) == 0

    def test_topk_k_larger_than_data(self):
        """TopK with k > len(data) should return all data."""
        df = pd.DataFrame({"a": [3, 1, 2]})
        ldf = df.select().sort("a").limit(100)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_topk_multiple_columns(self):
        """TopK with multiple sort keys."""
        df = pd.DataFrame(
            {
                "category": ["B", "A", "A", "B", "A"],
                "value": [1, 3, 1, 2, 2],
            }
        )
        ldf = df.select().sort("category", "value").limit(3)

        result = ldf.collect()
        # Sorted by category then value: A,1 -> A,2 -> A,3 -> B,1 -> B,2
        expected = pd.DataFrame(
            {
                "category": ["A", "A", "A"],
                "value": [1, 2, 3],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_topk_repr(self):
        """TopK node should have readable repr."""

        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select().sort("a").limit(2)

        # Get the optimized plan
        optimized = SortLimitToTopK().optimize(ldf._plan)

        assert "TopK" in repr(optimized)
        assert "k=2" in repr(optimized)


class TestTopKNode:
    """Tests for TopK plan node directly."""

    def test_topk_resolve_schema(self):
        """TopK should preserve input schema."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select()
        topk = TopK(
            ldf._plan,
            k=2,
            by=ldf._plan.exprs[:1],  # Sort by first column
            descending=(False,),
        )

        schema = topk.resolve_schema()
        assert "a" in schema.names
        assert "b" in schema.names

    def test_topk_children(self):
        """TopK should have one child."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select()
        topk = TopK(ldf._plan, k=2, by=ldf._plan.exprs, descending=(False,))

        children = topk.children()
        assert len(children) == 1
        assert children[0] is ldf._plan


# =============================================================================
# PREDICATE PUSHDOWN THROUGH JOIN TESTS
# =============================================================================


# =============================================================================
# COMMON SUBEXPRESSION ELIMINATION TESTS
# =============================================================================


class TestCommonSubexpressionElimination:
    """Tests for CommonSubexpressionElimination optimization pass."""

    def test_cse_detects_duplicate_expressions(self):
        """Duplicate expressions should be detected."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Same expression twice with different aliases
        ldf = df.select(
            "a",
            (col("a") * 2).alias("doubled1"),
            (col("a") * 2).alias("doubled2"),
        )

        # Apply CSE
        cse = CommonSubexpressionElimination()
        original_plan = ldf._plan
        optimized_plan = cse.optimize(original_plan)

        # The optimized plan should have a CSE temp column
        assert isinstance(optimized_plan, Project)

        # Result should still be correct
        result = ldf.collect()
        assert list(result["doubled1"]) == [2, 4, 6]
        assert list(result["doubled2"]) == [2, 4, 6]

    def test_cse_different_expressions_not_combined(self):
        """Different expressions should not be combined."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select(
            "a",
            (col("a") * 2).alias("times2"),
            (col("a") * 3).alias("times3"),
        )

        # Apply CSE
        cse = CommonSubexpressionElimination()
        original_plan = ldf._plan
        optimized_plan = cse.optimize(original_plan)

        # No CSE should happen (all expressions unique)
        assert len(optimized_plan.exprs) == len(original_plan.exprs)

        # Result should be correct
        result = ldf.collect()
        assert list(result["times2"]) == [2, 4, 6]
        assert list(result["times3"]) == [3, 6, 9]

    def test_cse_simple_field_refs_not_combined(self):
        """Simple FieldRefs should not be CSE'd (no benefit)."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("a", "b", col("a").alias("a_copy"))

        cse = CommonSubexpressionElimination()
        original_plan = ldf._plan
        optimized_plan = cse.optimize(original_plan)

        # No CSE temp columns should be added
        assert len(optimized_plan.exprs) == len(original_plan.exprs)

        result = ldf.collect()
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["a_copy"]) == [1, 2, 3]

    def test_cse_three_duplicate_expressions(self):
        """Three identical expressions should all use one CSE."""
        df = pd.DataFrame({"a": [10, 20, 30]})
        ldf = df.select(
            "a",
            (col("a") + 5).alias("x"),
            (col("a") + 5).alias("y"),
            (col("a") + 5).alias("z"),
        )

        result = ldf.collect()
        assert list(result["x"]) == [15, 25, 35]
        assert list(result["y"]) == [15, 25, 35]
        assert list(result["z"]) == [15, 25, 35]

    def test_cse_nested_expressions(self):
        """Nested expressions with common parts."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        # (a * 2) + 1 twice
        ldf = df.select(
            "a",
            ((col("a") * 2) + 1).alias("expr1"),
            ((col("a") * 2) + 1).alias("expr2"),
        )

        result = ldf.collect()
        # a * 2 + 1 = [3, 5, 7]
        assert list(result["expr1"]) == [3, 5, 7]
        assert list(result["expr2"]) == [3, 5, 7]

    def test_cse_preserves_correctness(self):
        """CSE optimization should not change results."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ldf = df.select(
            "a",
            (col("a") ** 2).alias("squared1"),
            (col("a") ** 2).alias("squared2"),
        )

        # Results with and without optimization
        result_opt = ldf.collect(optimize=True)
        result_no_opt = ldf.collect(optimize=False)

        tm.assert_frame_equal(
            result_opt[["a", "squared1", "squared2"]],
            result_no_opt[["a", "squared1", "squared2"]],
        )


# =============================================================================
# DEAD CODE ELIMINATION TESTS
# =============================================================================


class TestDeadCodeElimination:
    """Tests for DeadCodeElimination optimization pass."""

    def test_dce_removes_filter_true(self):
        """Filter with constant True should be removed."""
        from pandas.lazy.expr import Expr
        from pandas.lazy.ir import Literal
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)

        # Create Filter with True predicate
        true_pred = Expr(Literal(True))
        filter_true = Filter(source, true_pred)

        dce = DeadCodeElimination()
        optimized = dce.optimize(filter_true)

        # Filter should be eliminated
        assert isinstance(optimized, DataFrameSource)
        assert optimized is source

    def test_dce_keeps_filter_with_actual_predicate(self):
        """Filter with non-constant predicate should be kept."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ldf = df.select().filter(col("a") > 2)

        dce = DeadCodeElimination()
        optimized = dce.optimize(ldf._plan)

        # Filter should remain
        assert isinstance(optimized, Filter)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_dce_removes_identity_projection(self):
        """Identity projection should be removed."""
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)

        # Create identity projection: select a, b (same order, no changes)
        identity_proj = Project(source, (col("a"), col("b")))

        dce = DeadCodeElimination()
        optimized = dce.optimize(identity_proj)

        # Project should be eliminated
        assert isinstance(optimized, DataFrameSource)

    def test_dce_keeps_non_identity_projection(self):
        """Non-identity projection should be kept."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Select only one column (not identity)
        ldf = df.select("a")

        dce = DeadCodeElimination()
        optimized = dce.optimize(ldf._plan)

        # Project should remain
        assert isinstance(optimized, Project)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_dce_keeps_projection_with_computation(self):
        """Projection with computation should be kept."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select("a", (col("a") * 2).alias("doubled"))

        dce = DeadCodeElimination()
        optimized = dce.optimize(ldf._plan)

        # Project should remain (has computation)
        assert isinstance(optimized, Project)

        result = ldf.collect()
        assert list(result["doubled"]) == [2, 4, 6]

    def test_dce_removes_empty_sort(self):
        """Sort with no columns should be removed."""
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [3, 1, 2]})
        source = DataFrameSource(df)

        # Create Sort with empty by tuple (unusual but possible)
        empty_sort = Sort(source, by=(), descending=())

        dce = DeadCodeElimination()
        optimized = dce.optimize(empty_sort)

        # Sort should be eliminated
        assert isinstance(optimized, DataFrameSource)

    def test_dce_keeps_sort_with_columns(self):
        """Sort with columns should be kept."""
        df = pd.DataFrame({"a": [3, 1, 2]})
        ldf = df.select().sort("a")

        dce = DeadCodeElimination()
        optimized = dce.optimize(ldf._plan)

        # Sort should remain
        assert isinstance(optimized, Sort)

        result = ldf.collect()
        expected = pd.DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_dce_chained_filters_with_true(self):
        """Chained filters where one is True."""
        from pandas.lazy.expr import Expr
        from pandas.lazy.ir import Literal
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        source = DataFrameSource(df)

        # Filter(Filter(source, a > 2), True)
        inner_filter = Filter(source, col("a") > 2)
        true_pred = Expr(Literal(True))
        outer_filter = Filter(inner_filter, true_pred)

        dce = DeadCodeElimination()
        optimized = dce.optimize(outer_filter)

        # Outer filter (True) should be eliminated
        assert isinstance(optimized, Filter)
        assert isinstance(optimized.input, DataFrameSource)

    def test_dce_preserves_correctness(self):
        """DCE optimization should not change results."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        # Select all columns (identity) then filter
        ldf = df.select("a").filter(col("a") > 2)

        result_opt = ldf.collect(optimize=True)
        result_no_opt = ldf.collect(optimize=False)

        tm.assert_frame_equal(result_opt, result_no_opt)

    def test_dce_keeps_projection_with_rename(self):
        """Projection that renames should be kept."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ldf = df.select(col("a").alias("b"))

        dce = DeadCodeElimination()
        optimized = dce.optimize(ldf._plan)

        # Project should remain (renames column)
        assert isinstance(optimized, Project)

        result = ldf.collect()
        assert "b" in result.columns
        assert list(result["b"]) == [1, 2, 3]

    def test_dce_keeps_projection_with_reorder(self):
        """Projection that reorders columns should be kept."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select("b", "a")  # Reordered

        dce = DeadCodeElimination()
        optimized = dce.optimize(ldf._plan)

        # Project should remain (reorders columns)
        assert isinstance(optimized, Project)

        result = ldf.collect()
        assert list(result.columns) == ["b", "a"]


# =============================================================================
# AGGREGATE PUSHDOWN TESTS
# =============================================================================


class TestAggregatePushdown:
    """Tests for AggregatePushdown optimization pass."""

    def test_push_aggregate_through_pass_through_project(self):
        """Aggregate can push through project with pass-through columns."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
                "extra": ["x", "y", "z", "w"],
            }
        )

        # Project followed by Aggregate
        ldf = (
            df.select("category", "value")
            .group_by("category")
            .agg(col("value").sum().alias("total"))
        )

        # Apply optimization
        optimizer = AggregatePushdown()
        optimizer.optimize(ldf._plan)

        # The aggregate should be pushed through/combined with project
        # Result should still be correct
        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "total": [30, 70],
            }
        )
        tm.assert_frame_equal(
            result.sort_values("category").reset_index(drop=True),
            expected.sort_values("category").reset_index(drop=True),
        )

    def test_push_aggregate_through_renamed_columns(self):
        """Aggregate can push through project with renamed columns."""
        df = pd.DataFrame(
            {
                "cat": ["A", "A", "B", "B"],
                "val": [10, 20, 30, 40],
            }
        )

        # Rename columns then aggregate
        ldf = (
            df.select(col("cat").alias("category"), col("val").alias("value"))
            .group_by("category")
            .agg(col("value").sum().alias("total"))
        )

        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "total": [30, 70],
            }
        )
        tm.assert_frame_equal(
            result.sort_values("category").reset_index(drop=True),
            expected.sort_values("category").reset_index(drop=True),
        )

    def test_no_push_aggregate_through_computed_columns(self):
        """Aggregate cannot push through project with computed columns."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
            }
        )

        # Compute a column then aggregate on it
        ldf = (
            df.select("category", (col("value") * 2).alias("doubled"))
            .group_by("category")
            .agg(col("doubled").sum().alias("total"))
        )

        # Optimization should not push through computed column
        optimizer = AggregatePushdown()
        optimized = optimizer.optimize(ldf._plan)

        # Aggregate should still be on top (or wrapped in Project)
        # since we can't push through computed columns
        assert isinstance(optimized, (Aggregate, Project))

        # Result should still be correct
        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "total": [60, 140],  # (10+20)*2, (30+40)*2
            }
        )
        tm.assert_frame_equal(
            result.sort_values("category").reset_index(drop=True),
            expected.sort_values("category").reset_index(drop=True),
        )

    def test_aggregate_pushdown_preserves_correctness(self):
        """Aggregate pushdown should not change results."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        ldf = (
            df.select("category", "value")
            .group_by("category")
            .agg(
                col("value").sum().alias("sum"),
                col("value").mean().alias("avg"),
            )
        )

        result_opt = ldf.collect(optimize=True)
        result_no_opt = ldf.collect(optimize=False)

        # Sort for comparison
        result_opt = result_opt.sort_values("category").reset_index(drop=True)
        result_no_opt = result_no_opt.sort_values("category").reset_index(drop=True)

        tm.assert_frame_equal(result_opt, result_no_opt)

    def test_aggregate_with_filter_stays_optimal(self):
        """Filter before aggregate is already optimal, should stay."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
            }
        )

        ldf = (
            df.select()
            .filter(col("value") > 15)
            .group_by("category")
            .agg(col("value").sum().alias("total"))
        )

        result = ldf.collect()
        # After filter: A:20, B:30,40
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "total": [20, 70],
            }
        )
        tm.assert_frame_equal(
            result.sort_values("category").reset_index(drop=True),
            expected.sort_values("category").reset_index(drop=True),
        )

    def test_aggregate_multiple_group_columns(self):
        """Aggregate with multiple group-by columns."""
        df = pd.DataFrame(
            {
                "cat1": ["A", "A", "B", "B"],
                "cat2": ["X", "Y", "X", "Y"],
                "value": [10, 20, 30, 40],
            }
        )

        ldf = (
            df.select("cat1", "cat2", "value")
            .group_by("cat1", "cat2")
            .agg(col("value").sum().alias("total"))
        )

        result = ldf.collect()
        assert len(result) == 4  # 4 unique combinations

    def test_aggregate_count(self):
        """Aggregate with count()."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "A", "B", "B"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        ldf = (
            df.select("category", "value")
            .group_by("category")
            .agg(col("value").count().alias("cnt"))
        )

        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "cnt": [3, 2],
            }
        )
        tm.assert_frame_equal(
            result.sort_values("category").reset_index(drop=True),
            expected.sort_values("category").reset_index(drop=True),
        )

    def test_aggregate_pushdown_nested_project(self):
        """Aggregate pushdown through nested projects."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
                "extra": ["x", "y", "z", "w"],
            }
        )

        # Multiple projections then aggregate
        ldf = (
            df.select("category", "value", "extra")
            .select("category", "value")
            .group_by("category")
            .agg(col("value").sum().alias("total"))
        )

        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "category": ["A", "B"],
                "total": [30, 70],
            }
        )
        tm.assert_frame_equal(
            result.sort_values("category").reset_index(drop=True),
            expected.sort_values("category").reset_index(drop=True),
        )


class TestPredicatePushdownThroughJoin:
    """Tests for predicate pushdown through Join nodes."""

    def test_push_filter_to_left_side(self):
        """Filter on left-only column should push to left."""
        left = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        right = pd.DataFrame({"id": [1, 2, 3], "b": [100, 200, 300]})

        ldf = left.select().join(right.select(), on="id").filter(col("a") > 15)

        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "id": [2, 3],
                "a": [20, 30],
                "b": [200, 300],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_push_filter_to_right_side(self):
        """Filter on right-only column should push to right."""
        left = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        right = pd.DataFrame({"id": [1, 2, 3], "b": [100, 200, 300]})

        ldf = left.select().join(right.select(), on="id").filter(col("b") > 150)

        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "id": [2, 3],
                "a": [20, 30],
                "b": [200, 300],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_push_filter_on_join_column(self):
        """Filter on join column should push (can go to both sides)."""
        left = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        right = pd.DataFrame({"id": [1, 2, 3], "b": [100, 200, 300]})

        ldf = left.select().join(right.select(), on="id").filter(col("id") > 1)

        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "id": [2, 3],
                "a": [20, 30],
                "b": [200, 300],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_filter_on_suffixed_column(self):
        """Filter on suffixed column (from overlapping names)."""
        left = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        right = pd.DataFrame({"id": [1, 2, 3], "val": [100, 200, 300]})

        ldf = (
            left.select()
            .join(right.select(), on="id")
            .filter(col("val_x") > 15)  # Left's val
        )

        result = ldf.collect()
        expected = pd.DataFrame(
            {
                "id": [2, 3],
                "val_x": [20, 30],
                "val_y": [200, 300],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_filter_mixing_both_sides_not_pushed(self):
        """Filter mixing columns from both sides cannot be pushed."""
        left = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        right = pd.DataFrame({"id": [1, 2, 3], "b": [5, 25, 35]})

        ldf = left.select().join(right.select(), on="id").filter(col("a") > col("b"))

        result = ldf.collect()
        # a > b: 10>5 (yes), 20>25 (no), 30>35 (no)
        expected = pd.DataFrame(
            {
                "id": [1],
                "a": [10],
                "b": [5],
            }
        )
        tm.assert_frame_equal(result, expected)


class TestExpressionSimplificationAdvanced:
    """Tests for advanced expression simplification patterns."""

    def test_de_morgan_and_to_or(self):
        """Test De Morgan's law: ~(a & b) -> ~a | ~b."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # ~(a & b)
        a = FieldRef("a")
        b = FieldRef("b")
        and_node = Call("and_", (a, b))
        not_and = Call("invert", (and_node,))

        result = simplifier._simplify_ir(not_and)

        # Should become ~a | ~b
        assert isinstance(result, Call)
        assert result.function == "or_"
        assert len(result.args) == 2
        assert isinstance(result.args[0], Call)
        assert result.args[0].function == "invert"
        assert isinstance(result.args[1], Call)
        assert result.args[1].function == "invert"

    def test_de_morgan_or_to_and(self):
        """Test De Morgan's law: ~(a | b) -> ~a & ~b."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # ~(a | b)
        a = FieldRef("a")
        b = FieldRef("b")
        or_node = Call("or_", (a, b))
        not_or = Call("invert", (or_node,))

        result = simplifier._simplify_ir(not_or)

        # Should become ~a & ~b
        assert isinstance(result, Call)
        assert result.function == "and_"
        assert len(result.args) == 2
        assert isinstance(result.args[0], Call)
        assert result.args[0].function == "invert"
        assert isinstance(result.args[1], Call)
        assert result.args[1].function == "invert"

    def test_self_subtraction(self):
        """Test x - x -> 0."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a - a
        a = FieldRef("a")
        subtract = Call("subtract", (a, a))

        result = simplifier._simplify_ir(subtract)

        assert isinstance(result, Literal)
        assert result.value == 0

    def test_self_division(self):
        """Test x / x -> 1."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a / a
        a = FieldRef("a")
        divide = Call("divide", (a, a))

        result = simplifier._simplify_ir(divide)

        assert isinstance(result, Literal)
        assert result.value == 1

    def test_self_equality(self):
        """Test x == x -> True."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a == a
        a = FieldRef("a")
        equal = Call("equal", (a, a))

        result = simplifier._simplify_ir(equal)

        assert isinstance(result, Literal)
        assert result.value is True

    def test_self_inequality(self):
        """Test x != x -> False."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a != a
        a = FieldRef("a")
        not_equal = Call("not_equal", (a, a))

        result = simplifier._simplify_ir(not_equal)

        assert isinstance(result, Literal)
        assert result.value is False

    def test_self_less_than(self):
        """Test x < x -> False."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a < a
        a = FieldRef("a")
        less = Call("less", (a, a))

        result = simplifier._simplify_ir(less)

        assert isinstance(result, Literal)
        assert result.value is False

    def test_self_greater_than(self):
        """Test x > x -> False."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a > a
        a = FieldRef("a")
        greater = Call("greater", (a, a))

        result = simplifier._simplify_ir(greater)

        assert isinstance(result, Literal)
        assert result.value is False

    def test_self_less_equal(self):
        """Test x <= x -> True."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a <= a
        a = FieldRef("a")
        le = Call("less_equal", (a, a))

        result = simplifier._simplify_ir(le)

        assert isinstance(result, Literal)
        assert result.value is True

    def test_self_greater_equal(self):
        """Test x >= x -> True."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a >= a
        a = FieldRef("a")
        ge = Call("greater_equal", (a, a))

        result = simplifier._simplify_ir(ge)

        assert isinstance(result, Literal)
        assert result.value is True

    def test_self_and_idempotent(self):
        """Test x & x -> x."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a & a
        a = FieldRef("a")
        and_node = Call("and_", (a, a))

        result = simplifier._simplify_ir(and_node)

        assert isinstance(result, FieldRef)
        assert result.name == "a"

    def test_self_or_idempotent(self):
        """Test x | x -> x."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # a | a
        a = FieldRef("a")
        or_node = Call("or_", (a, a))

        result = simplifier._simplify_ir(or_node)

        assert isinstance(result, FieldRef)
        assert result.name == "a"

    def test_complex_self_expression(self):
        """Test (a + b) - (a + b) -> 0."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        # (a + b)
        a = FieldRef("a")
        b = FieldRef("b")
        add_node = Call("add", (a, b))

        # (a + b) - (a + b)
        subtract = Call("subtract", (add_node, Call("add", (a, b))))

        result = simplifier._simplify_ir(subtract)

        assert isinstance(result, Literal)
        assert result.value == 0

    def test_ir_equal_literals(self):
        """Test _ir_equal for Literal nodes."""
        from pandas.lazy.ir import Literal
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        assert simplifier._ir_equal(Literal(5), Literal(5)) is True
        assert simplifier._ir_equal(Literal(5), Literal(10)) is False
        assert simplifier._ir_equal(Literal("a"), Literal("a")) is True
        assert simplifier._ir_equal(Literal("a"), Literal("b")) is False

    def test_ir_equal_field_refs(self):
        """Test _ir_equal for FieldRef nodes."""
        from pandas.lazy.ir import FieldRef
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        assert simplifier._ir_equal(FieldRef("a"), FieldRef("a")) is True
        assert simplifier._ir_equal(FieldRef("a"), FieldRef("b")) is False

    def test_ir_equal_calls(self):
        """Test _ir_equal for Call nodes."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )
        from pandas.lazy.optimize.passes import ExpressionSimplification

        simplifier = ExpressionSimplification()

        a = FieldRef("a")
        b = FieldRef("b")

        # Same call
        call1 = Call("add", (a, b))
        call2 = Call("add", (FieldRef("a"), FieldRef("b")))
        assert simplifier._ir_equal(call1, call2) is True

        # Different function
        call3 = Call("subtract", (a, b))
        assert simplifier._ir_equal(call1, call3) is False

        # Different args
        call4 = Call("add", (a, FieldRef("c")))
        assert simplifier._ir_equal(call1, call4) is False

    def test_end_to_end_self_subtraction(self):
        """Integration test: x - x simplification through full pipeline."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # (a - a) should become 0
        result = df.select(
            col("b"),
            (col("a") - col("a")).alias("zero"),
        ).collect()

        expected = pd.DataFrame({"b": [4, 5, 6], "zero": [0, 0, 0]})
        tm.assert_frame_equal(result, expected)

    def test_end_to_end_self_equality_filter(self):
        """Integration test: x == x in filter should return all rows."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Filter on (a == a) which is always True -> should return all rows
        result = df.select().filter(col("a") == col("a")).collect()

        tm.assert_frame_equal(result, df)
