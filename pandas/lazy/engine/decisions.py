"""
The physical decision layer (M2, docs/ENGINE_DESIGN.md).

One pass over the compiled PipelineGraph that owns backend, conversion,
and algorithm choices — replacing decisions scattered across the logical
optimizer, runtime thresholds, and per-operator checks (P3: one decision
layer). M2 lands this incrementally: each decision migrates here with
its runtime logic kept as fallback, gated on the suite and benchmark
baselines staying green.

Decisions migrated so far:
- **Groupby backend** (the June routing-bug site): planned from the
  input schema's per-column storage backends using the same
  relevant-columns rule the runtime fix established — Arrow if any
  group key or aggregation value column is Arrow-backed.
- **Filter backend** (schema-decidable case): when every data column is
  Arrow-backed by schema, the Arrow filter path is planned; mixed
  backends remain a runtime threshold decision, where the actual row
  count is strictly better information ("runtime adapts within planned
  bounds").
- **Join build side**: surfaced from the node's existing plan-time row
  estimates (`left/right_rows_estimate`); the runtime re-check against
  actual materialized sizes stays authoritative.

The pass also annotates every pipeline with its per-column backends and
explicit conversion points, surfaced in ``explain(physical=True)`` so a
plan's data movement is visible before it runs (P2: conversions are
plan operations, never side effects).
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import TYPE_CHECKING

from pandas.lazy.engine.pipeline import (
    NodeSink,
    Pipeline,
    PipelineGraph,
)
from pandas.lazy.physical import (
    PhysicalConvert,
    PhysicalDistinct,
    PhysicalFilter,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalSort,
    PhysicalTopK,
)

# Sinks that do not observe their input's row order: a join feeding one
# of these may run through acero's order-destroying parallel hash join.
_ORDER_FREE_SINKS = (
    PhysicalHashAggregate,
    PhysicalSort,
    PhysicalTopK,
    PhysicalDistinct,
)

if TYPE_CHECKING:
    from pandas.lazy.types import Schema


@dataclass
class PipelineDecisions:
    """Planned execution properties of one pipeline."""

    # column name -> "arrow" | "numpy", from the pipeline's output schema
    column_backends: dict[str, str] = field(default_factory=dict)
    # human-readable conversion points (explicit Convert operators)
    conversions: list[str] = field(default_factory=list)
    # sink algorithm/backend chosen at plan time, if any
    sink_decision: str | None = None

    def describe(self) -> str:
        backends = sorted(set(self.column_backends.values()))
        parts = [f"backends={'/'.join(backends) if backends else '?'}"]
        if self.conversions:
            parts.append(f"convert: {', '.join(self.conversions)}")
        if self.sink_decision:
            parts.append(f"sink: {self.sink_decision}")
        return "; ".join(parts)


def _pipeline_output_schema(pipeline: Pipeline) -> Schema | None:
    """Schema of what this pipeline pushes into its sink."""
    if pipeline.operators:
        return pipeline.operators[-1].output_schema
    if pipeline.source_node is not None:
        return pipeline.source_node.output_schema
    if isinstance(pipeline.source_sink, NodeSink):
        return pipeline.source_sink.node.output_schema
    return None


def _plan_groupby_backend(node: PhysicalHashAggregate, input_schema: Schema) -> str:
    """The relevant-columns rule, applied at plan time from the schema.

    Mirrors the runtime rule in PhysicalHashAggregate.execute: decide
    from group keys + aggregation value columns only; Arrow wins if any
    relevant column is Arrow-backed.
    """
    from pandas.lazy.expr import extract_output_name
    from pandas.lazy.ir import (
        Alias,
        Call,
        FieldRef,
    )

    relevant: set[str] = set()
    for expr in node.group_by:
        relevant.add(extract_output_name(expr))
    for expr in node.agg_exprs:
        ir = expr._ir
        if isinstance(ir, Alias):
            ir = ir.arg
        if isinstance(ir, Call):
            for arg in ir.args:
                if isinstance(arg, FieldRef):
                    relevant.add(arg.name)

    backends = {
        input_schema[name].storage_backend for name in relevant if name in input_schema
    }
    if "arrow" in backends:
        return "arrow"
    if backends:
        return "numpy"
    return "numpy"


class DecisionLayer:
    """Annotate a PipelineGraph with planned execution decisions."""

    def annotate(self, graph: PipelineGraph) -> None:
        self._annotate_pipelines(graph)
        self._plan_acero_joins(graph)

    def _annotate_pipelines(self, graph: PipelineGraph) -> None:
        for pipeline in graph.pipelines:
            decisions = PipelineDecisions()

            schema = _pipeline_output_schema(pipeline)
            if schema is not None:
                decisions.column_backends = {
                    name: schema[name].storage_backend for name in schema.names
                }

            for op in pipeline.operators:
                if isinstance(op, PhysicalConvert):
                    target = getattr(op, "target_backend", "?")
                    decisions.conversions.append(f"-> {target}")
                elif isinstance(op, PhysicalFilter):
                    self._plan_filter(op, decisions)

            if isinstance(pipeline.sink, NodeSink):
                self._plan_sink(pipeline.sink, pipeline, decisions)

            pipeline.decisions = decisions

    def _plan_filter(self, op: PhysicalFilter, decisions: PipelineDecisions) -> None:
        """Plan the filter backend when decidable from schema alone."""
        input_schema = op.children()[0].output_schema
        if input_schema is None or not input_schema.names:
            return
        backends = {input_schema[n].storage_backend for n in input_schema.names}
        if backends == {"arrow"}:
            op.planned_backend = "arrow"
            decisions.conversions.append("filter[arrow]")
        # mixed/numpy: runtime threshold decides (actual row count wins)

    def _plan_sink(
        self, sink: NodeSink, pipeline: Pipeline, decisions: PipelineDecisions
    ) -> None:
        node = sink.node

        if isinstance(node, PhysicalHashAggregate):
            input_schema = _pipeline_output_schema(pipeline)
            if input_schema is not None:
                planned = _plan_groupby_backend(node, input_schema)
                node.planned_backend = planned
                decisions.sink_decision = f"groupby[{planned}]"

        elif isinstance(node, PhysicalHashJoin):
            # Surface the build-side expectation from the node's existing
            # plan-time row estimates; the runtime re-check against actual
            # materialized sizes stays authoritative.
            left_est = node.left_rows_estimate
            right_est = node.right_rows_estimate
            if left_est is not None and right_est is not None:
                build = "right" if right_est <= left_est else "left"
                decisions.sink_decision = (
                    f"join[build={build}, est={left_est}x{right_est}]"
                )
            else:
                decisions.sink_decision = "join[build=runtime]"

        elif isinstance(node, (PhysicalSort, PhysicalTopK)):
            input_schema = _pipeline_output_schema(pipeline)
            if input_schema is not None and input_schema.names:
                backends = sorted(
                    {input_schema[n].storage_backend for n in input_schema.names}
                )
                kind = "topk" if isinstance(node, PhysicalTopK) else "sort"
                decisions.sink_decision = f"{kind}[{'/'.join(backends)}]"

    def _plan_acero_joins(self, graph: PipelineGraph) -> None:
        """Route order-free joins to acero's parallel hash join.

        Eligible when (a) the join's output feeds an order-insensitive
        sink — groupby/sort/topk/distinct do not observe input row
        order, so acero's nondeterministic output order is unobservable;
        (b) how is inner/left; (c) every key column is an integer or
        string with no nulls in either input's schema (pandas merge
        matches NaN==NaN and None==None; acero follows SQL and does
        not — excluding nullable/float keys keeps semantics identical).
        Measured at 10M x 1M ints: acero 160 ms vs indexer path 684 ms
        vs Polars ~700 ms. Joins whose order is observable (direct
        materialization) keep the eager-order-matching indexer path.
        """
        consumer_of: dict[int, object] = {}
        feeder_schemas: dict[int, dict[int, object]] = {}
        for p in graph.pipelines:
            if p.source_sink is not None:
                consumer_of[id(p.source_sink)] = p.sink
            if isinstance(p.sink, NodeSink):
                feeder_schemas.setdefault(id(p.sink), {})[p.sink_slot] = (
                    _pipeline_output_schema(p)
                )

        for p in graph.pipelines:
            sink = p.sink
            if not (
                isinstance(sink, NodeSink) and isinstance(sink.node, PhysicalHashJoin)
            ):
                continue
            node = sink.node
            if node.planned_acero:  # already planned via the other feeder
                continue
            if node.how not in ("inner", "left"):
                continue
            consumer = consumer_of.get(id(sink))
            if not (
                isinstance(consumer, NodeSink)
                and isinstance(consumer.node, _ORDER_FREE_SINKS)
            ):
                continue

            schemas = feeder_schemas.get(id(sink), {})
            left_schema, right_schema = schemas.get(0), schemas.get(1)
            if left_schema is None or right_schema is None:
                continue
            if node.on is not None:
                left_keys = right_keys = list(node.on)
            else:
                left_keys = list(node.left_on or ())
                right_keys = list(node.right_on or ())

            def keys_safe(schema, names) -> bool:
                for name in names:
                    if name not in schema:
                        return False
                    dt = schema[name]
                    if dt.nullable:
                        return False
                    if dt.category == "string":
                        continue
                    if dt.category == "numeric" and (
                        dt.numpy_dtype is not None and dt.numpy_dtype.kind in ("i", "u")
                    ):
                        continue
                    return False
                return True

            if keys_safe(left_schema, left_keys) and keys_safe(
                right_schema, right_keys
            ):
                node.planned_acero = True
                if p.decisions is not None:
                    p.decisions.sink_decision = (
                        p.decisions.sink_decision or "join"
                    ) + " -> acero (order-free consumer)"


def annotate_decisions(graph: PipelineGraph) -> PipelineGraph:
    """Run the decision layer over a compiled graph (idempotent)."""
    DecisionLayer().annotate(graph)
    return graph
