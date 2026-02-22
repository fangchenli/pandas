"""
pandas.compile — JIT compilation for pandas operations.

Public API:
    compile       Decorator for JIT-compiling pandas functions.
    Tracer        Context manager for tracing pandas operations.
    Backend            Abstract backend base class.
    DataFusionBackend  Execute IR via Apache DataFusion (preferred when available).
    AceroBackend       Execute IR via PyArrow's Acero engine.
    PandasBackend      Pure-pandas IR interpreter backend (fallback).
    infer_schema  Infer IR Schema from a DataFrame.

Power-user modules:
    pandas.compile.ir        IR types (DType, Schema, IRNode, Expr, ...)
    pandas.compile.compiler  SubstraitCompiler, execution plan types
"""

from pandas.compile.compiler import (
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
from pandas.compile.jit import (
    CompiledFunction,
    DeferredScalar,
    Tracer,
    compile,
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
    "compile",
    "infer_schema",
]


# Make `pd.compile(fn)` work by making this module callable.
# pd.compile resolves to the pandas.compile subpackage, so we override
# the module's class to allow calling it as a decorator directly.
import sys
import types


class _CallableModule(types.ModuleType):
    def __call__(self, fn=None, *, backend=None):
        return compile(fn, backend=backend)


_CLASS = "__class__"
setattr(sys.modules[__name__], _CLASS, _CallableModule)
