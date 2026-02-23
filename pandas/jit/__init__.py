"""
pandas.jit — JIT compilation for pandas operations.

Public API:
    compilable    Decorator for JIT-compiling pandas functions.
    Tracer        Context manager for tracing pandas operations.
    Backend            Abstract backend base class.
    DataFusionBackend  Execute IR via Apache DataFusion (preferred when available).
    AceroBackend       Execute IR via PyArrow's Acero engine.
    PandasBackend      Pure-pandas IR interpreter backend (fallback).
    infer_schema  Infer IR Schema from a DataFrame.

Power-user modules:
    pandas.jit.ir        IR types (DType, Schema, IRNode, Expr, ...)
    pandas.jit.compiler  SubstraitCompiler, execution plan types
"""

from pandas.jit.compiler import (
    AceroBackend,
    Backend,
    CompiledStage,
    ConnectedPlan,
    DataFusionBackend,
    GraphBreakStage,
    PandasBackend,
    StageSchema,
    infer_schema,
)
from pandas.jit.jit import (
    CompiledFunction,
    DeferredScalar,
    Tracer,
    compilable,
)

__all__ = [
    "AceroBackend",
    "Backend",
    "CompiledFunction",
    "CompiledStage",
    "ConnectedPlan",
    "DataFusionBackend",
    "DeferredScalar",
    "GraphBreakStage",
    "PandasBackend",
    "StageSchema",
    "Tracer",
    "compilable",
    "infer_schema",
]


# Make `pd.jit(fn)` work by making this module callable.
# pd.jit resolves to the pandas.jit subpackage, so we override
# the module's class to allow calling it as a decorator directly.
import sys
import types


class _CallableModule(types.ModuleType):
    def __call__(self, fn=None, *, backend=None):
        return compilable(fn, backend=backend)


_CLASS = "__class__"
setattr(sys.modules[__name__], _CLASS, _CallableModule)
