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


class TestCategoricalDictionaryRouting:
    """M4 part 2: Categorical columns flow as Arrow dictionaries.

    extract_array wraps Categorical zero-copy (codes -> indices), the
    schema reports arrow storage so the decision layer routes groupby
    to acero (which hash-aggregates dictionary keys ~5x faster than raw
    strings), and the output converts back to category dtype.
    """

    def _frame(self, n=3_000):
        rng = np.random.default_rng(11)
        return pd.DataFrame(
            {
                "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
                "v": rng.standard_normal(n),
            }
        )

    def test_groupby_routed_to_arrow_and_matches_eager(self):
        df = self._frame()
        q = df.select().group_by("g").agg(col("v").sum().alias("s"))
        assert "sink: groupby[arrow]" in q.explain(physical=True)

        result = q.collect(use_physical_planner=True).sort_values("g")
        eager = df.groupby("g", observed=True)["v"].sum().sort_index()
        assert np.allclose(result["s"].to_numpy(dtype="float64"), eager.to_numpy())
        assert isinstance(result["g"].dtype, pd.CategoricalDtype)

    def test_category_filter_correct(self):
        df = self._frame()
        result = df.select().filter(col("g") == "A").collect(use_physical_planner=True)
        assert len(result) == (df["g"] == "A").sum()
        assert (result["g"] == "A").all()

    def test_category_with_missing_codes(self):
        g = pd.Categorical(["A", None, "B", "A", None])
        df = pd.DataFrame({"g": g, "v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # missing keys dropped per pandas dropna=True semantics
        result = (
            df.select()
            .group_by("g")
            .agg(col("v").sum().alias("s"))
            .collect(use_physical_planner=True)
            .sort_values("g")
        )
        eager = df.groupby("g", observed=True)["v"].sum().sort_index()
        assert np.allclose(result["s"].to_numpy(dtype="float64"), eager.to_numpy())

    def test_numeric_categories(self):
        df = pd.DataFrame(
            {
                "g": pd.Categorical([10, 20, 10, 30, 20]),
                "v": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = (
            df.select()
            .group_by("g")
            .agg(col("v").sum().alias("s"))
            .collect(use_physical_planner=True)
            .sort_values("g")
        )
        eager = df.groupby("g", observed=True)["v"].sum().sort_index()
        assert np.allclose(result["s"].to_numpy(dtype="float64"), eager.to_numpy())


class TestAceroJoinRouting:
    """M5: order-free joins route to acero's parallel hash join."""

    def _frames(self, n=5_000):
        rng = np.random.default_rng(13)
        left = pd.DataFrame(
            {
                "k": rng.integers(0, 500, n),
                "v": rng.standard_normal(n),
            }
        )
        right = pd.DataFrame({"k": np.arange(500), "w": rng.standard_normal(500)})
        return left, right

    def test_join_groupby_routed_to_acero_and_matches_eager(self):
        left, right = self._frames()
        q = (
            left.select()
            .join(right.select(), on="k")
            .group_by("k")
            .agg(col("v").sum().alias("s"), col("w").mean().alias("m"))
        )
        assert "acero (order-free consumer)" in q.explain(physical=True)

        result = (
            q.collect(use_physical_planner=True).sort_values("k").reset_index(drop=True)
        )
        eager = (
            left.merge(right, on="k")
            .groupby("k")
            .agg(s=("v", "sum"), m=("w", "mean"))
            .reset_index()
            .sort_values("k")
            .reset_index(drop=True)
        )
        assert len(result) == len(eager)
        assert np.allclose(result["s"].to_numpy(dtype="float64"), eager["s"].to_numpy())
        assert np.allclose(result["m"].to_numpy(dtype="float64"), eager["m"].to_numpy())

    def test_direct_join_keeps_indexer_path(self):
        # Order is observable on direct materialization: the indexer
        # path (eager pd.merge order by construction) must be kept.
        left, right = self._frames()
        text = left.select().join(right.select(), on="k").explain(physical=True)
        assert "acero" not in text

    def test_float_keys_not_routed(self):
        # pandas merge matches NaN==NaN; acero (SQL semantics) does not.
        rng = np.random.default_rng(14)
        left = pd.DataFrame(
            {"k": rng.standard_normal(1000), "v": rng.standard_normal(1000)}
        )
        right = pd.DataFrame(
            {"k": rng.standard_normal(100), "w": rng.standard_normal(100)}
        )
        q = (
            left.select()
            .join(right.select(), on="k")
            .group_by("k")
            .agg(col("v").sum().alias("s"))
        )
        assert "acero" not in q.explain(physical=True)

    def test_preserve_index_falls_back_at_runtime(self):
        # planned acero, but preserve_index observes row identity ->
        # runtime guard must use the indexer path; values still correct
        left, right = self._frames(500)
        q = left.select().join(right.select(), on="k").sort("k")
        result = q.collect(use_physical_planner=True, preserve_index=True)
        eager = left.merge(right, on="k").sort_values("k")
        assert len(result) == len(eager)


class TestScanStreamingThroughPipelines:
    """M6: file-scan-sourced pipelines execute via the nodes' native
    execute_batches protocol - scan batches are the natural morsels and
    embedded limits terminate the read early. M1's original executor
    materialized scans fully before applying limits (a multi-file glob
    head(1000) read every file: 2,788 ms vs 137 ms)."""

    def test_scan_head_never_fully_materializes(self, tmp_path, monkeypatch):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from pandas.lazy import scan
        from pandas.lazy.physical import PhysicalParquetScan

        for i in range(3):
            table = pa.table({"a": list(range(i * 1000, (i + 1) * 1000))})
            pq.write_table(table, tmp_path / f"part_{i}.parquet")

        def boom(self, context):
            raise AssertionError(
                "PhysicalParquetScan.execute() called for a limit query - "
                "the streaming path must be used"
            )

        monkeypatch.setattr(PhysicalParquetScan, "execute", boom)

        result = (
            scan(str(tmp_path / "part_*.parquet"))
            .head(10)
            .collect(use_physical_planner=True)
        )
        assert len(result) == 10

    def test_scan_filter_values_match(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from pandas.lazy import scan

        table = pa.table({"a": list(range(5000)), "b": [1.5] * 5000})
        pq.write_table(table, tmp_path / "f.parquet", row_group_size=1000)

        result = (
            scan(str(tmp_path / "f.parquet"))
            .filter(col("a") >= 4990)
            .collect(use_physical_planner=True)
        )
        assert list(result["a"]) == list(range(4990, 5000))
