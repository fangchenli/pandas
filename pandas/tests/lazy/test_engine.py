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


class TestDecisionLayer:
    """M2: plan-time decisions annotated on the pipeline graph."""

    def test_groupby_backend_planned_from_schema(self):
        # The original routing-bug shape: leading NumPy column, Arrow
        # string key. The decision layer must plan arrow from the
        # schema alone - before any data is touched.
        df = pd.DataFrame(
            {
                "id": np.arange(100),  # leading numpy column
                "g": pd.array(["a", "b"] * 50),  # arrow string key
                "v": np.arange(100, dtype="float64"),
            }
        )
        ldf = df.select().group_by("g").agg(col("v").sum().alias("s"))

        from pandas.lazy.engine.decisions import annotate_decisions
        from pandas.lazy.physical import PhysicalHashAggregate as Agg

        graph = annotate_decisions(compile_graph(ldf))
        agg_sinks = [
            p.sink
            for p in graph.pipelines
            if isinstance(p.sink, NodeSink) and isinstance(p.sink.node, Agg)
        ]
        assert len(agg_sinks) == 1
        assert agg_sinks[0].node.planned_backend == "arrow"

    def test_numeric_only_groupby_planned_numpy(self):
        df = pd.DataFrame(
            {
                "g": np.array([1, 2] * 50),
                "v": np.arange(100, dtype="float64"),
                "s": pd.array(["x"] * 100),  # irrelevant arrow payload
            }
        )
        ldf = df.select().group_by("g").agg(col("v").sum().alias("t"))

        from pandas.lazy.engine.decisions import annotate_decisions
        from pandas.lazy.physical import PhysicalHashAggregate as Agg

        graph = annotate_decisions(compile_graph(ldf))
        agg = next(
            p.sink.node
            for p in graph.pipelines
            if isinstance(p.sink, NodeSink) and isinstance(p.sink.node, Agg)
        )
        # irrelevant arrow payload must not flip the decision... but note
        # projection pruning may have dropped 's' already; either way the
        # relevant columns (g, v) are numpy
        assert agg.planned_backend == "numpy"

    def test_explain_shows_decisions(self):
        df = pd.DataFrame({"g": pd.array(["a", "b"]), "v": [1.0, 2.0]})
        text = (
            df.select()
            .group_by("g")
            .agg(col("v").sum().alias("s"))
            .explain(physical=True)
        )
        assert "sink: groupby[arrow]" in text
        assert "backends=" in text

    def test_planned_decision_matches_runtime_result(self):
        # The planned and runtime rules must agree on results
        rng = np.random.default_rng(5)
        df = pd.DataFrame(
            {
                "id": np.arange(1000),
                "g": pd.array(rng.choice(["x", "y", "z"], 1000)),
                "v": rng.standard_normal(1000),
            }
        )
        ldf = df.select().group_by("g").agg(col("v").sum().alias("s"))
        physical = (
            ldf.collect(use_physical_planner=True)
            .sort_values("g")
            .reset_index(drop=True)
        )
        eager = ldf.collect().sort_values("g").reset_index(drop=True)
        assert np.allclose(
            physical["s"].to_numpy(dtype="float64"),
            eager["s"].to_numpy(dtype="float64"),
        )

    def test_filter_backend_planned_when_all_arrow(self):
        # All data columns arrow-backed -> filter[arrow] planned from
        # schema; PhysicalFilter.planned_backend set
        df = pd.DataFrame(
            {
                "s": pd.array(["a", "b", "c", "d"]),
                "t": pd.array(["x", "y", "z", "w"]),
            }
        )
        # disable fusion so a bare PhysicalFilter survives planning
        from pandas.lazy.engine.decisions import annotate_decisions
        from pandas.lazy.physical import (
            PhysicalFilter,
            PhysicalPlanner,
        )

        ldf = df.select().filter(col("s") == "a")
        physical = PhysicalPlanner().plan(
            ldf._get_optimized_plan(), enable_fusion=False
        )
        graph = annotate_decisions(PipelineCompiler().compile(physical))

        filters = [
            op
            for p in graph.pipelines
            for op in p.operators
            if isinstance(op, PhysicalFilter)
        ]
        assert filters, "expected an unfused PhysicalFilter"
        assert all(f.planned_backend == "arrow" for f in filters)

    def test_join_build_side_annotated_from_estimates(self):
        left = pd.DataFrame({"k": range(1000), "v": [1.0] * 1000})
        right = pd.DataFrame({"k": range(100), "w": [2.0] * 100})
        text = left.select().join(right.select(), on="k").explain(physical=True)
        assert "join[build=right, est=1000x100]" in text

    def test_sort_sink_annotated(self):
        df = pd.DataFrame({"a": [3.0, 1.0, 2.0]})
        text = df.select().sort("a").explain(physical=True)
        assert "sink: sort[numpy]" in text


class TestMorselParallelism:
    """M3: morsel-parallel execution of stateless pipelines."""

    def _wide_query(self, n=2_000):
        rng = np.random.default_rng(9)
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "v1": rng.standard_normal(n),
                "v2": rng.standard_normal(n),
                "text": pd.array([f"t{i % 50}" for i in range(n)]),
            }
        )
        return (
            df.select()
            .filter(col("v1") > -3)
            .with_columns(
                (col("v1") * 2 + col("v2") * col("v2")).alias("d"),
                col("text").str.upper().alias("T"),
            )
        )

    def test_parallel_matches_sequential_exactly(self, monkeypatch):
        import pandas.lazy.engine.parallel as par

        q = self._wide_query()
        # shrink morsels so the parallel path runs on test-sized data
        monkeypatch.setattr(par, "MORSEL_SIZE", 256)
        monkeypatch.setattr(par, "MIN_PARALLEL_ROWS", 512)

        result_par = q.collect(use_physical_planner=True)

        monkeypatch.setattr(par, "pipeline_is_morsel_parallel", lambda p: False)
        result_seq = q.collect(use_physical_planner=True)

        # exact equality including row order: the stable-order contract
        tm.assert_frame_equal(result_par, result_seq, check_dtype=False)

    def test_low_compute_chains_stay_sequential(self):
        from pandas.lazy.engine.decisions import annotate_decisions
        from pandas.lazy.engine.parallel import (
            pipeline_is_morsel_parallel,
        )

        df = pd.DataFrame({"a": np.arange(1000), "b": np.arange(1000)})
        ldf = df.select().filter(col("a") > 1).select("a")
        graph = annotate_decisions(compile_graph(ldf))
        assert not any(pipeline_is_morsel_parallel(p) for p in graph.pipelines)

    def test_string_chains_are_parallel_eligible(self):
        from pandas.lazy.engine.parallel import (
            pipeline_is_morsel_parallel,
        )

        df = pd.DataFrame(
            {"v": np.arange(1000, dtype="float64"), "s": pd.array(["x"] * 1000)}
        )
        ldf = (
            df.select()
            .filter(col("v") > 0)
            .with_columns(col("s").str.upper().alias("S"))
        )
        graph = compile_graph(ldf)
        assert any(pipeline_is_morsel_parallel(p) for p in graph.pipelines)

    def test_aggregate_and_window_exprs_excluded(self):
        from pandas.lazy.engine.parallel import (
            pipeline_is_morsel_parallel,
        )

        df = pd.DataFrame({"v": np.arange(1000, dtype="float64")})
        # scalar aggregate broadcast in a projection: not morsel-safe
        ldf1 = df.select().with_columns(col("v").sum().alias("total"))
        graph1 = compile_graph(ldf1)
        assert not any(pipeline_is_morsel_parallel(p) for p in graph1.pipelines)
        # cumulative: order-dependent, not morsel-safe
        ldf2 = df.select().with_columns(col("v").cum_sum().alias("c"))
        graph2 = compile_graph(ldf2)
        assert not any(pipeline_is_morsel_parallel(p) for p in graph2.pipelines)

    def test_tail_morsel_backend_normalized(self, monkeypatch):
        # uneven tail morsel + mixed result backends must concat cleanly
        import pandas.lazy.engine.parallel as par

        monkeypatch.setattr(par, "MORSEL_SIZE", 300)  # 1000 rows -> 4 morsels
        monkeypatch.setattr(par, "MIN_PARALLEL_ROWS", 600)
        df = pd.DataFrame(
            {
                "v": np.arange(1000, dtype="float64"),
                "s": pd.array([f"x{i}" for i in range(1000)]),
            }
        )
        q = (
            df.select()
            .filter(col("v") >= 0)
            .with_columns(col("s").str.upper().alias("S"))
        )
        result = q.collect(use_physical_planner=True)
        assert len(result) == 1000
        assert result["S"].iloc[299] == "X299"
        assert result["S"].iloc[999] == "X999"
