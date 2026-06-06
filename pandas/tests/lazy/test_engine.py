"""
Tests for the pipeline execution engine (M1, docs/ENGINE_DESIGN.md).

The full lazy suite already exercises execution end-to-end through the
pipeline engine (execute_physical_plan routes through it); these tests
pin the compiler's *shape* guarantees — what compiles into one pipeline,
what becomes a sink, and graph ordering.
"""

import numpy as np

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.engine import (
    CollectSink,
    NodeSink,
    PipelineCompiler,
)
from pandas.lazy.physical import (
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalPlanner,
)


def compile_graph(ldf):
    physical = PhysicalPlanner().plan(ldf._get_optimized_plan())
    return PipelineCompiler().compile(physical)


class TestPipelineCompiler:
    def test_streaming_chain_is_one_pipeline(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select().filter(col("a") > 1).select("a")

        graph = compile_graph(ldf)

        assert len(graph.pipelines) == 1
        (p,) = graph.pipelines
        assert p.source_node is not None
        assert isinstance(p.sink, CollectSink)

    def test_breaker_becomes_sink(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
        ldf = df.select().group_by("a").agg(col("b").sum().alias("s"))

        graph = compile_graph(ldf)

        # P1: scan -> aggregate sink; P2: sink -> collect
        assert len(graph.pipelines) == 2
        p1, p2 = graph.pipelines
        assert isinstance(p1.sink, NodeSink)
        assert isinstance(p1.sink.node, PhysicalHashAggregate)
        assert p2.source_sink is p1.sink
        assert isinstance(p2.sink, CollectSink)

    def test_join_compiles_to_three_pipelines(self):
        left = pd.DataFrame({"k": [1, 2], "v": [10, 20]})
        right = pd.DataFrame({"k": [1, 2], "w": [30, 40]})
        ldf = left.select().join(right.select(), on="k")

        graph = compile_graph(ldf)

        # one pipeline per join input + one from the join sink to collect
        join_sinks = {
            id(p.sink)
            for p in graph.pipelines
            if isinstance(p.sink, NodeSink)
            and isinstance(p.sink.node, PhysicalHashJoin)
        }
        assert len(join_sinks) == 1
        feeders = [
            p
            for p in graph.pipelines
            if isinstance(p.sink, NodeSink)
            and isinstance(p.sink.node, PhysicalHashJoin)
        ]
        assert sorted(p.sink_slot for p in feeders) == [0, 1]
        # graph order is topological: both feeders precede the consumer
        consumer_idx = next(
            i for i, p in enumerate(graph.pipelines) if p.source_sink is feeders[0].sink
        )
        assert all(graph.pipelines.index(p) < consumer_idx for p in feeders)

    def test_complex_query_graph_shape(self):
        df = pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        ldf = (
            df.select()
            .filter(col("a") > 0)
            .group_by("a")
            .agg(col("b").sum().alias("s"))
            .sort("s", descending=True)
            .head(2)
        )

        graph = compile_graph(ldf)

        sink_nodes = [
            type(p.sink.node).__name__
            for p in graph.pipelines
            if isinstance(p.sink, NodeSink)
        ]
        # Aggregate and Sort (or TopK if the optimizer fused sort+limit)
        assert any("Aggregate" in name for name in sink_nodes)
        assert any("Sort" in name or "TopK" in name for name in sink_nodes)
        assert isinstance(graph.pipelines[-1].sink, CollectSink)

    def test_describe_renders_every_pipeline(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        ldf = df.select().sort("a")

        graph = compile_graph(ldf)
        text = graph.describe()

        assert "Pipeline graph:" in text
        for p in graph.pipelines:
            assert f"P{p.pid}:" in text

    def test_explain_includes_pipelines(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        text = df.select().filter(col("a") > 1).explain(physical=True)
        assert "Pipeline graph:" in text


class TestPipelineExecutorEquivalence:
    """Spot-check executor equivalence on the multi-input shapes whose
    metadata handling is order-sensitive (the rest of the suite covers
    everything else end-to-end)."""

    def test_join_preserves_left_index_metadata(self):
        left = pd.DataFrame(
            {"k": [1, 2], "v": [10, 20]},
            index=pd.Index([100, 200], name="left_idx"),
        )
        right = pd.DataFrame(
            {"k": [1, 2], "w": [30, 40]},
            index=pd.Index([7, 8], name="right_idx"),
        )
        result = (
            left.select()
            .join(right.select(), on="k")
            .collect(use_physical_planner=True, preserve_index=True)
        )
        assert result.index.name == "left_idx"

    def test_sort_after_join_values_match_eager(self):
        left = pd.DataFrame({"k": [2, 1, 3], "v": [20.0, 10.0, 30.0]})
        right = pd.DataFrame({"k": [1, 2, 3], "w": [1, 2, 3]})
        ldf = left.select().join(right.select(), on="k").sort("k")

        eager = ldf.collect(use_physical_planner=False)
        physical = ldf.collect(use_physical_planner=True)
        tm.assert_numpy_array_equal(
            physical["v"].to_numpy(dtype="float64"),
            eager["v"].to_numpy(dtype="float64"),
        )
        assert list(physical["k"]) == list(eager["k"])

    def test_concat_first_input_metadata_wins(self):
        from pandas.lazy import concat as lazy_concat

        df1 = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="first_idx"))
        df2 = pd.DataFrame({"a": [3, 4]}, index=pd.Index([30, 40], name="second_idx"))
        result = lazy_concat([df1.select(), df2.select()]).collect(
            use_physical_planner=True, preserve_index=True
        )
        assert result.index.name == "first_idx"

    def test_groupby_after_filter_matches_eager(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "g": rng.choice(["x", "y", "z"], 500),
                "v": rng.standard_normal(500),
            }
        )
        ldf = (
            df.select()
            .filter(col("v") > 0)
            .group_by("g")
            .agg(col("v").sum().alias("s"))
        )
        eager = ldf.collect().sort_values("g").reset_index(drop=True)
        physical = (
            ldf.collect(use_physical_planner=True)
            .sort_values("g")
            .reset_index(drop=True)
        )
        assert np.allclose(
            physical["s"].to_numpy(dtype="float64"),
            eager["s"].to_numpy(dtype="float64"),
        )
