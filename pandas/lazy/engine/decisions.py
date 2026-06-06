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
    PhysicalHashAggregate,
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

            if isinstance(pipeline.sink, NodeSink) and isinstance(
                pipeline.sink.node, PhysicalHashAggregate
            ):
                input_schema = _pipeline_output_schema(pipeline)
                if input_schema is not None:
                    planned = _plan_groupby_backend(pipeline.sink.node, input_schema)
                    pipeline.sink.node.planned_backend = planned
                    decisions.sink_decision = f"groupby[{planned}]"

            pipeline.decisions = decisions


def annotate_decisions(graph: PipelineGraph) -> PipelineGraph:
    """Run the decision layer over a compiled graph (idempotent)."""
    DecisionLayer().annotate(graph)
    return graph
