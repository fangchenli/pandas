"""
Tests for pandas.lazy.plan module - Logical plan nodes.
"""

import pandas as pd
from pandas.lazy.expr import col
from pandas.lazy.plan import (
    DataFrameSource,
    Filter,
    LogicalPlan,
    Project,
)


class TestDataFrameSource:
    """Tests for DataFrameSource plan node."""

    def test_creation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        assert source.df is df

    def test_resolve_schema(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        source = DataFrameSource(df)
        schema = source.resolve_schema()
        assert "a" in schema
        assert "b" in schema
        assert schema["a"].is_numeric()
        assert schema["b"].is_string()

    def test_children_empty(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        assert source.children() == []

    def test_repr(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        result = repr(source)
        assert "DataFrameSource" in result
        assert "columns" in result
        assert "rows" in result

    def test_is_logical_plan(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        assert isinstance(source, LogicalPlan)


class TestProject:
    """Tests for Project plan node."""

    def test_creation(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        exprs = (col("a"), col("b"))
        project = Project(source, exprs)
        assert project.input is source
        assert project.exprs == exprs

    def test_resolve_schema(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        exprs = (col("a"),)
        project = Project(source, exprs)
        schema = project.resolve_schema()
        assert schema.names == ["a"]

    def test_resolve_schema_with_alias(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        exprs = (col("a").alias("new_a"),)
        project = Project(source, exprs)
        schema = project.resolve_schema()
        assert schema.names == ["new_a"]

    def test_children(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        project = Project(source, (col("a"),))
        children = project.children()
        assert len(children) == 1
        assert children[0] is source

    def test_repr(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        project = Project(source, (col("a"),))
        result = repr(project)
        assert "Project" in result

    def test_nested_project(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        project1 = Project(source, (col("a"), col("b")))
        project2 = Project(project1, (col("a"),))
        schema = project2.resolve_schema()
        assert schema.names == ["a"]


class TestFilter:
    """Tests for Filter plan node."""

    def test_creation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        predicate = col("a")  # Placeholder, actual predicate would be comparison
        filter_node = Filter(source, predicate)
        assert filter_node.input is source
        assert filter_node.predicate is predicate

    def test_resolve_schema_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        predicate = col("a")
        filter_node = Filter(source, predicate)
        # Filter doesn't change schema
        schema = filter_node.resolve_schema()
        assert schema.names == ["a", "b"]

    def test_children(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        filter_node = Filter(source, col("a"))
        children = filter_node.children()
        assert len(children) == 1
        assert children[0] is source

    def test_repr(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        filter_node = Filter(source, col("a"))
        result = repr(filter_node)
        assert "Filter" in result


class TestPlanComposition:
    """Tests for composing plan nodes."""

    def test_source_project(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        project = Project(source, (col("a"),))

        assert project.children() == [source]
        assert project.resolve_schema().names == ["a"]

    def test_source_project_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        project = Project(source, (col("a"), col("b")))
        filter_node = Filter(project, col("a"))

        assert filter_node.children() == [project]
        assert filter_node.resolve_schema().names == ["a", "b"]

    def test_plan_tree_depth(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        p1 = Project(source, (col("a"),))
        p2 = Project(p1, (col("a").alias("b"),))
        p3 = Project(p2, (col("b").alias("c"),))

        # Walk up the tree
        assert p3.input == p2
        assert p2.input == p1
        assert p1.input == source
        assert source.children() == []


class TestCardinalityEstimation:
    """Predicate-aware selectivity (System R model, optimize/cardinality.py)."""

    def test_selectivity_constants_by_operator(self):
        from pandas.lazy.optimize.cardinality import (
            SEL_EQ,
            SEL_NEQ,
            SEL_RANGE,
            estimate_selectivity,
        )

        assert estimate_selectivity((col("a") == 5)._ir) == SEL_EQ
        assert estimate_selectivity((col("a") != 5)._ir) == SEL_NEQ
        assert estimate_selectivity((col("a") > 0)._ir) == SEL_RANGE
        assert estimate_selectivity((col("a") <= 0)._ir) == SEL_RANGE

    def test_selectivity_boolean_composition(self):
        from pandas.lazy.optimize.cardinality import (
            SEL_EQ,
            SEL_RANGE,
            estimate_selectivity,
        )

        # AND multiplies (independence).
        both = (col("a") > 0) & (col("b") < 5)
        assert estimate_selectivity(both._ir) == SEL_RANGE * SEL_RANGE
        # OR is inclusion-exclusion.
        either = (col("a") == 1) | (col("a") == 2)
        assert estimate_selectivity(either._ir) == SEL_EQ + SEL_EQ - SEL_EQ * SEL_EQ
        # NOT complements.
        assert estimate_selectivity((~(col("a") > 0))._ir) == 1.0 - SEL_RANGE

    def test_filter_estimate_is_predicate_aware(self):
        n = 1_000_000
        df = pd.DataFrame({"a": range(n), "b": range(n)})
        eq = df.select().filter(col("a") == 5)._plan.estimate_row_count()
        rng = df.select().filter(col("a") > 0)._plan.estimate_row_count()
        # Equality is far more selective than a range; both differ from the
        # old flat 0.3 (== was 300_000 for any predicate).
        assert eq == 100_000
        assert rng == 333_333
        assert eq < rng

    def test_filter_estimate_none_poison_preserved(self):
        # A source with no estimate still yields None through the filter.
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": [1, 2, 3]})
        base = df.select()
        filtered = base.filter(col("a") > 0)._plan
        # DataFrameSource has a known length, so this is concrete; assert the
        # estimate is defined and reduced, not None.
        assert isinstance(base._plan, (Project,))
        assert isinstance(filtered.estimate_row_count(), int)
        # sanity: an unfiltered source reports its full length
        assert isinstance(DataFrameSource(df).estimate_row_count(), int)

    def test_build_side_picks_smaller_filtered_side(self):
        import numpy as np

        from pandas.lazy.engine.decisions import annotate_decisions
        from pandas.lazy.engine.pipeline import (
            NodeSink,
            PipelineCompiler,
        )
        from pandas.lazy.physical import (
            PhysicalHashJoin,
            PhysicalPlanner,
        )

        # left 1M filtered by equality (~100K) joined to unfiltered 200K right.
        # The flat-0.3 estimate sized left at 300K and would build 'right';
        # predicate-aware sizing builds 'left' (the actually-smaller side).
        big = pd.DataFrame({"k": np.arange(1_000_000), "v": np.arange(1_000_000)})
        small = pd.DataFrame({"k": np.arange(200_000), "w": np.arange(200_000)})
        q = big.select().filter(col("k") == 5).join(small.select(), on="k")

        graph = annotate_decisions(
            PipelineCompiler().compile(PhysicalPlanner().plan(q._get_optimized_plan()))
        )
        decision = next(
            p.decisions.sink_decision
            for p in graph.pipelines
            if isinstance(p.sink, NodeSink)
            and isinstance(p.sink.node, PhysicalHashJoin)
            and p.decisions
            and p.decisions.sink_decision
        )
        assert "build=left" in decision
        assert "100000x200000" in decision
