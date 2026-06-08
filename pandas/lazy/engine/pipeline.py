"""
Pipeline graph: the M1 execution model from docs/ENGINE_DESIGN.md.

Every physical plan compiles to an explicit graph of Pipelines. A
Pipeline is a source, an ordered chain of streaming operators, and a
Sink. Pipeline breakers (sort, aggregate, join, distinct, ...) become
sinks; a sink's finalized output is the source of downstream pipelines.

M1 is deliberately a pure refactor: execution still flows through every
node's existing ``execute()`` — the compiler only re-routes plumbing.
Each operator/breaker is applied by swapping its plan input(s) for a
``PrecomputedInput`` adapter holding already-materialized arrays
(``dataclasses.replace``), so operator logic, context mutations, spill
behavior, and semantics are byte-identical to the recursive executor.
The whole input flows as a single morsel; M3 introduces real morsel
partitioning and worker self-dispatch behind the same abstractions.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import dataclasses
from dataclasses import (
    dataclass,
    field,
)
from typing import TYPE_CHECKING

from pandas.lazy.physical import (
    ExecutionContext,
    PhysicalConcat,
    PhysicalConvert,
    PhysicalCSVScan,
    PhysicalFilter,
    PhysicalFusedPipeline,
    PhysicalHashJoin,
    PhysicalLimit,
    PhysicalMaterialize,
    PhysicalParquetScan,
    PhysicalPlan,
    PhysicalProject,
    PhysicalScan,
    PhysicalSortMergeJoin,
)

if TYPE_CHECKING:
    from pandas.lazy.backends.types import ArrayDict
    from pandas.lazy.types import Schema

# Nodes that are pipeline sources (no plan inputs).
SOURCE_TYPES = (PhysicalScan, PhysicalParquetScan, PhysicalCSVScan)

# Nodes that stream morsel-by-morsel and therefore chain inside one
# pipeline. Everything else terminates a pipeline as a sink.
# PhysicalMaterialize is included: with a PrecomputedInput input its
# execute() is a pass-through plus bookkeeping, identical to today.
STREAMING_TYPES = (
    PhysicalFilter,
    PhysicalProject,
    PhysicalFusedPipeline,
    PhysicalLimit,
    PhysicalConvert,
    PhysicalMaterialize,
)


@dataclass
class Morsel:
    """A unit of data flowing through a pipeline.

    M1 uses a single morsel per pipeline (the whole input); ``seq``
    exists for the order-preserving merges of M3+. Index metadata rides
    along when the producing pipeline ran in an isolated sub-context
    (multi-input sinks), mirroring how join/concat capture each side's
    metadata from their per-side cloned contexts.
    """

    arrays: ArrayDict
    seq: int = 0
    index_names: list | None = None
    index_is_multi: bool = False
    user_set_index: bool = False


@dataclass
class PrecomputedInput(PhysicalPlan):
    """Adapter: a plan node whose execution result is already known.

    This is the universal plumbing trick of M1 — any existing node runs
    inside the pipeline engine by replacing its plan input(s) with one
    of these, reusing the node's own execute() unchanged. When the
    producing pipeline ran in an isolated sub-context, the captured
    index metadata is replayed into whatever context this adapter is
    executed under — multi-input nodes execute their inputs inside
    per-side context clones and then merge metadata by their own rules
    (join: left wins; concat: first input wins), so replaying here lets
    that existing merge logic run unchanged.
    """

    arrays: ArrayDict
    schema: Schema | None = None
    index_names: list | None = None
    index_is_multi: bool = False
    user_set_index: bool = False

    def execute(self, context: ExecutionContext) -> ArrayDict:
        if self.index_names is not None:
            context.index_names = self.index_names
            context.index_is_multi = self.index_is_multi
        if self.user_set_index:
            context.user_set_index = True
        return self.arrays

    def children(self) -> list[PhysicalPlan]:
        return []

    @property
    def output_schema(self) -> Schema:
        return self.schema  # type: ignore[return-value]


def with_inputs(node: PhysicalPlan, inputs: list[PhysicalPlan]) -> PhysicalPlan:
    """Return a copy of ``node`` with its plan input(s) replaced."""
    if isinstance(node, (PhysicalHashJoin, PhysicalSortMergeJoin)):
        return dataclasses.replace(node, left=inputs[0], right=inputs[1])
    if isinstance(node, PhysicalConcat):
        return dataclasses.replace(node, inputs=tuple(inputs))
    return dataclasses.replace(node, input=inputs[0])


class Sink(ABC):
    """A pipeline terminator: consumes morsels, produces a result."""

    @abstractmethod
    def consume(self, morsel: Morsel, slot: int, context: ExecutionContext) -> None:
        """Accept a morsel from the pipeline feeding input ``slot``."""

    @abstractmethod
    def finalize(self, context: ExecutionContext) -> ArrayDict:
        """Produce the sink's output after all inputs are consumed."""


class CollectSink(Sink):
    """Terminal sink: the query result."""

    def __init__(self) -> None:
        self._arrays: ArrayDict | None = None

    def consume(self, morsel: Morsel, slot: int, context: ExecutionContext) -> None:
        self._arrays = morsel.arrays

    def finalize(self, context: ExecutionContext) -> ArrayDict:
        assert self._arrays is not None, "CollectSink finalized before consume"
        return self._arrays


class NodeSink(Sink):
    """A pipeline breaker, wrapping the node's existing execution logic.

    M1 form: collect each input pipeline's (single) morsel into its
    slot, then run the node with PrecomputedInput inputs. M4/M5 replace
    specific NodeSinks with truly parallel accumulate/merge sinks
    behind this same interface.
    """

    def __init__(self, node: PhysicalPlan, n_slots: int) -> None:
        self.node = node
        self.n_slots = n_slots
        self._slots: list[Morsel | None] = [None] * n_slots
        self._result: ArrayDict | None = None

    def consume(self, morsel: Morsel, slot: int, context: ExecutionContext) -> None:
        self._slots[slot] = morsel

    def finalize(self, context: ExecutionContext) -> ArrayDict:
        if self._result is None:
            children = self.node.children()
            inputs: list[PhysicalPlan] = [
                PrecomputedInput(
                    arrays=m.arrays,  # type: ignore[union-attr]
                    schema=child.output_schema,
                    index_names=m.index_names,  # type: ignore[union-attr]
                    index_is_multi=m.index_is_multi,  # type: ignore[union-attr]
                    user_set_index=m.user_set_index,  # type: ignore[union-attr]
                )
                for m, child in zip(self._slots, children, strict=True)
            ]
            bound = with_inputs(self.node, inputs)
            self._result = bound.execute(context)
        return self._result


@dataclass
class Pipeline:
    """source → [streaming operators] → sink (one input slot of it)."""

    pid: int
    # Exactly one of the two sources is set:
    source_node: PhysicalPlan | None  # a scan
    source_sink: Sink | None  # upstream breaker's output
    operators: list[PhysicalPlan]
    sink: Sink
    sink_slot: int
    # Set by the decision layer (engine/decisions.py)
    decisions: object | None = None

    def describe(self) -> str:
        if self.source_node is not None:
            src = type(self.source_node).__name__
        else:
            src = f"sink:{type(self.source_sink).__name__}"
            if isinstance(self.source_sink, NodeSink):
                src = f"sink:{type(self.source_sink.node).__name__}"
        ops = " -> ".join(type(op).__name__ for op in self.operators) or "(passthrough)"
        dst = type(self.sink).__name__
        if isinstance(self.sink, NodeSink):
            dst = f"{type(self.sink.node).__name__}Sink"
        text = f"P{self.pid}: {src} -> [{ops}] -> {dst}[slot {self.sink_slot}]"
        if self.decisions is not None:
            text += f"\n      {self.decisions.describe()}"  # type: ignore[attr-defined]
        return text


@dataclass
class PipelineGraph:
    """All pipelines of a query, in execution (topological) order."""

    pipelines: list[Pipeline] = field(default_factory=list)
    collect_sink: CollectSink | None = None

    def describe(self) -> str:
        lines = ["Pipeline graph:"]
        lines += [f"  {p.describe()}" for p in self.pipelines]
        return "\n".join(lines)


@dataclass
class _OpenPipeline:
    """Compiler state: a pipeline whose sink is not yet assigned."""

    source_node: PhysicalPlan | None
    source_sink: Sink | None
    operators: list[PhysicalPlan]


class PipelineCompiler:
    """Compile a physical plan tree into a PipelineGraph."""

    def __init__(self) -> None:
        self._graph = PipelineGraph()
        self._next_pid = 1

    def compile(self, plan: PhysicalPlan) -> PipelineGraph:
        open_pipeline = self._compile(plan)
        sink = CollectSink()
        self._close(open_pipeline, sink, slot=0)
        self._graph.collect_sink = sink
        return self._graph

    def _compile(self, node: PhysicalPlan) -> _OpenPipeline:
        if isinstance(node, SOURCE_TYPES):
            return _OpenPipeline(source_node=node, source_sink=None, operators=[])

        if isinstance(node, STREAMING_TYPES):
            open_pipeline = self._compile(node.children()[0])
            open_pipeline.operators.append(node)
            return open_pipeline

        # Breaker (or any node without special handling): terminate each
        # input's pipeline into a NodeSink wrapping this node.
        children = node.children()
        if not children:
            # Leaf node we don't recognize as a source (defensive):
            # treat as a source.
            return _OpenPipeline(source_node=node, source_sink=None, operators=[])
        sink = NodeSink(node, n_slots=len(children))
        for slot, child in enumerate(children):
            self._close(self._compile(child), sink, slot)
        return _OpenPipeline(source_node=None, source_sink=sink, operators=[])

    def _close(self, open_pipeline: _OpenPipeline, sink: Sink, slot: int) -> None:
        self._graph.pipelines.append(
            Pipeline(
                pid=self._next_pid,
                source_node=open_pipeline.source_node,
                source_sink=open_pipeline.source_sink,
                operators=open_pipeline.operators,
                sink=sink,
                sink_slot=slot,
            )
        )
        self._next_pid += 1


class PipelineExecutor:
    """Run a PipelineGraph. M1: single morsel, in graph order.

    Children close before parents during compilation, so graph order is
    topological — every pipeline's source sink is finalized before the
    pipeline runs.
    """

    def execute(self, graph: PipelineGraph, context: ExecutionContext) -> ArrayDict:
        for pipeline in graph.pipelines:
            # Pipelines feeding a multi-input sink run in an isolated
            # sub-context, mirroring how join/concat executed each side
            # in a clone before; their metadata is captured on the
            # morsel and replayed by PrecomputedInput inside the node's own
            # per-side clones, so the node's existing metadata-merge
            # logic (join: left wins; concat: first wins) is unchanged.
            isolated = isinstance(pipeline.sink, NodeSink) and pipeline.sink.n_slots > 1
            ctx = context.clone_for_subplan() if isolated else context

            # M6 / streaming restoration: file-scan-sourced pipelines run
            # through the nodes' native execute_batches protocol — the
            # scan's batches are the natural morsels, and embedded limits
            # terminate the read early (a head(1000) over a multi-file
            # glob must not read every file). Materializing the scan via
            # execute() before applying the chain — M1's original shape —
            # read 4 files completely for 1000 rows (2,788 ms vs 137 ms).
            if pipeline.source_node is not None and isinstance(
                pipeline.source_node, (PhysicalParquetScan, PhysicalCSVScan)
            ):
                node: PhysicalPlan = pipeline.source_node
                for op in pipeline.operators:
                    node = with_inputs(op, [node])
                batches = list(node.execute_batches(ctx))
                from pandas.lazy.engine.parallel import concat_morsel_results

                arrays = concat_morsel_results(batches) if batches else {}
            else:
                if pipeline.source_node is not None:
                    arrays = pipeline.source_node.execute(ctx)
                else:
                    assert pipeline.source_sink is not None
                    arrays = pipeline.source_sink.finalize(ctx)

                # M3: stateless pipelines over in-memory sources run their
                # operator chain morsel-parallel (engine/parallel.py); the
                # sequential single-morsel path remains the universal
                # default.
                from pandas.lazy.engine.parallel import (
                    MIN_PARALLEL_ROWS,
                    pipeline_is_morsel_parallel,
                    run_morsel_parallel,
                )

                n_rows = len(next(iter(arrays.values()))) if arrays else 0
                if n_rows >= MIN_PARALLEL_ROWS and pipeline_is_morsel_parallel(
                    pipeline
                ):
                    arrays = run_morsel_parallel(pipeline, arrays, ctx, n_rows)
                else:
                    for op in pipeline.operators:
                        child = op.children()[0]
                        bound = with_inputs(
                            op,
                            [
                                PrecomputedInput(
                                    arrays=arrays, schema=child.output_schema
                                )
                            ],
                        )
                        arrays = bound.execute(ctx)

            morsel = Morsel(arrays)
            if isolated:
                morsel.index_names = ctx.index_names
                morsel.index_is_multi = ctx.index_is_multi
                morsel.user_set_index = ctx.user_set_index
            pipeline.sink.consume(morsel, pipeline.sink_slot, ctx)

        assert graph.collect_sink is not None
        return graph.collect_sink.finalize(context)


def execute_as_pipelines(plan: PhysicalPlan, context: ExecutionContext) -> ArrayDict:
    """Compile a physical plan to a pipeline graph and execute it."""
    from pandas.lazy.engine.decisions import annotate_decisions

    graph = annotate_decisions(
        PipelineCompiler().compile(plan), order_relaxed=context.order_relaxed
    )
    return PipelineExecutor().execute(graph, context)


def render_pipelines(plan: PhysicalPlan) -> str:
    """Render the pipeline graph for explain(physical=True)."""
    from pandas.lazy.engine.decisions import annotate_decisions

    return annotate_decisions(PipelineCompiler().compile(plan)).describe()
