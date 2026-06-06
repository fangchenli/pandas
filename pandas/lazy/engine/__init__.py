"""
Pipeline execution engine (docs/ENGINE_DESIGN.md).

M1: every physical plan compiles to an explicit PipelineGraph —
streaming operator chains between breaker Sinks — executed single-morsel
with behavior identical to the recursive executor.
"""

from pandas.lazy.engine.pipeline import (
    CollectSink,
    Morsel,
    NodeSink,
    Pipeline,
    PipelineCompiler,
    PipelineExecutor,
    PipelineGraph,
    Sink,
    execute_as_pipelines,
    render_pipelines,
)

__all__ = [
    "CollectSink",
    "Morsel",
    "NodeSink",
    "Pipeline",
    "PipelineCompiler",
    "PipelineExecutor",
    "PipelineGraph",
    "Sink",
    "execute_as_pipelines",
    "render_pipelines",
]
