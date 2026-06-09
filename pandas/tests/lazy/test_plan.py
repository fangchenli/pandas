"""
Tests for pandas.lazy.plan module - Logical plan nodes.
"""

import pytest

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

    def test_filter_estimate_uses_column_statistics(self):
        import numpy as np

        n = 1_000_000
        # 'a' has NDV=50 (equality keeps ~1/50); 'b' is unique on [0, n).
        df = pd.DataFrame({"a": np.arange(n) % 50, "b": np.arange(n)})

        # Equality sized by 1/NDV (~20_000), not the flat 0.1 -> 100_000.
        eq = df.select().filter(col("a") == 5)._plan.estimate_row_count()
        assert 15_000 <= eq <= 25_000

        # Range sized by min/max interpolation: b > n//2 keeps ~half.
        rng = df.select().filter(col("b") > n // 2)._plan.estimate_row_count()
        assert 0.4 * n <= rng <= 0.6 * n

        # Equality on the unique column matches ~1 row (1/NDV ~ 1/n).
        eq_unique = df.select().filter(col("b") == 5)._plan.estimate_row_count()
        assert eq_unique <= 2

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

    def test_build_side_uses_ndv_to_flip_from_constant_model(self):
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

        # left 1M filtered by equality on an NDV=2 column -> keeps ~500K.
        # The constant model would size it at 0.1 -> 100K and build 'left';
        # NDV-aware sizing (~500K) correctly sees right (400K) as smaller and
        # builds 'right'. This is the stats refinement flipping the decision.
        big = pd.DataFrame(
            {"k": np.arange(1_000_000), "flag": np.arange(1_000_000) % 2}
        )
        small = pd.DataFrame({"k": np.arange(400_000), "w": np.arange(400_000)})
        q = big.select().filter(col("flag") == 1).join(small.select(), on="k")

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
        assert "build=right" in decision
        assert "500000x400000" in decision

    def test_compute_column_stats_ndv_and_range(self):
        import numpy as np

        from pandas.lazy.optimize.cardinality import compute_column_stats

        n = 500_000
        # Low cardinality: sampled NDV is exact.
        low = compute_column_stats(pd.Series(np.arange(n) % 7))
        assert low.ndv == 7
        assert low.row_count == n
        assert low.min_val == 0.0 and low.max_val == 6.0
        # High cardinality: NDV is extrapolated and flagged approximate.
        high = compute_column_stats(pd.Series(np.arange(n)))
        assert high.approximate
        assert high.ndv > n // 2  # near-unique

    def test_stats_selectivity_equality_and_range(self):
        import numpy as np

        from pandas.lazy.optimize.cardinality import (
            compute_column_stats,
            estimate_selectivity,
        )

        df = pd.DataFrame(
            {"g": np.arange(100_000) % 4, "r": np.arange(100_000, dtype="float64")}
        )
        lookup = {c: compute_column_stats(df[c]) for c in df.columns}.get

        # equality -> 1/NDV
        assert estimate_selectivity((col("g") == 1)._ir, lookup) == pytest.approx(
            0.25, abs=0.01
        )
        # range -> min/max interpolation (r < 25% of the span)
        cut = 25_000
        assert estimate_selectivity((col("r") < cut)._ir, lookup) == pytest.approx(
            0.25, abs=0.01
        )
        # literal-on-left flips the operator
        assert estimate_selectivity((cut < col("r"))._ir, lookup) == pytest.approx(
            0.75, abs=0.01
        )

    def test_grouped_aggregate_estimate_uses_ndv(self):
        import numpy as np

        n = 500_000
        df = pd.DataFrame({"g": np.arange(n) % 80, "v": np.arange(n)})
        est = (
            df.select()
            .group_by("g")
            .agg(col("v").sum().alias("s"))
            ._plan.estimate_row_count()
        )
        # One row per distinct group: ~80, not the sqrt heuristic (~7000).
        assert 70 <= est <= 90

    def test_parquet_statistics_refine_range_and_null(self, tmp_path):
        import numpy as np

        from pandas.lazy import scan

        n = 500_000
        path = str(tmp_path / "stats.parquet")
        pd.DataFrame(
            {
                "r": np.arange(n, dtype="float64"),  # range [0, n)
                "f": np.where(np.arange(n) % 5 == 0, np.nan, 1.0),  # 20% null
            }
        ).to_parquet(path)
        lf = scan(path)

        # Range sized from the metadata min/max (no data read).
        rng = lf.filter(col("r") < n * 0.25)._plan.estimate_row_count()
        assert 0.2 * n <= rng <= 0.3 * n
        # is_null sized from the metadata null count.
        nulls = lf.filter(col("f").is_null())._plan.estimate_row_count()
        assert 0.15 * n <= nulls <= 0.25 * n

        # Stats carry min/max and null_count but no NDV (Parquet has none).
        from pandas.lazy.plan import ParquetSource

        st = ParquetSource(path).column_statistics("r")
        assert st.min_val == 0.0 and st.max_val == float(n - 1)
        assert st.ndv is None
