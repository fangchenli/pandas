"""
Morsel-parallel pipeline execution (M3, docs/ENGINE_DESIGN.md).

Stateless pipelines — chains of row-wise filters and projections over
in-memory sources — partition their input into ~128K-row morsels and run
the whole operator chain per morsel on a shared worker pool. Workers
self-dispatch by claiming morsel indices from a per-pipeline cursor (the
passive-dispatcher pattern: no coordinator thread hands out work), and
results merge in sequence order, preserving the engine's stable row
order contract.

Safety: a pipeline is morsel-parallelizable only when every operator is
provably row-wise. Anything order- or whole-input-dependent — limits,
aggregate expressions (``Call.is_aggregate``), window/positional
functions (shift, lag, cum_*, rank, row_number) — falls back to the
single-morsel path, which remains the universal default.

The GIL question was answered empirically before this module was built
(`../benchmarks/spike_morsel_scaling.py`): NumPy/Arrow kernels release
the GIL completely enough that thread scaling matches free-threaded
Python exactly; the ceilings are memory bandwidth and core asymmetry.
"""

from __future__ import annotations

import itertools
import threading
from typing import TYPE_CHECKING

import numpy as np

from pandas.lazy.physical import (
    ExecutionContext,
    PhysicalConvert,
    PhysicalFilter,
    PhysicalFusedPipeline,
    PhysicalProject,
    PhysicalScan,
)

if TYPE_CHECKING:
    from pandas.lazy.backends.types import ArrayDict
    from pandas.lazy.engine.pipeline import Pipeline

# Tunables live in the cost model (pandas/lazy/cost.py, M2); imported
# under the same names so tests can monkeypatch this module's globals.
from pandas.lazy.cost import (
    MAX_WORKERS,
)

# IR functions that are order-dependent: applying them per morsel or
# feeding them rows in a different order changes results. Shared with
# the decision layer's order-freeness propagation for acero routing.
ORDER_SENSITIVE_FUNCTIONS = frozenset(
    {
        "shift",
        "lag",
        "lead",
        "rank",
        "row_number",
        "cum_sum",
        "cum_max",
        "cum_min",
        "cum_prod",
        "cum_count",
    }
)


def _expr_is_morsel_safe(ir) -> bool:
    """True if an expression IR is row-wise (safe to apply per morsel)."""
    from pandas.lazy.ir import (
        Alias,
        Call,
    )

    if isinstance(ir, Alias):
        return _expr_is_morsel_safe(ir.arg)
    if isinstance(ir, Call):
        if ir.is_aggregate or ir.function in ORDER_SENSITIVE_FUNCTIONS:
            return False
        return all(_expr_is_morsel_safe(arg) for arg in ir.args)
    # FieldRef, Literal, and other leaf nodes are row-wise
    children = getattr(ir, "args", ())
    return all(_expr_is_morsel_safe(c) for c in children)


def _op_is_morsel_safe(op) -> bool:
    if isinstance(op, PhysicalFilter):
        return _expr_is_morsel_safe(op.predicate._ir)
    if isinstance(op, PhysicalProject):
        return all(_expr_is_morsel_safe(e._ir) for e in op.exprs)
    if isinstance(op, PhysicalConvert):
        return True
    if isinstance(op, PhysicalFusedPipeline):
        for fop in op.operations:
            if fop.op_type == "limit":
                return False
            if fop.op_type == "filter" and not _expr_is_morsel_safe(fop.predicate._ir):
                return False
            if fop.op_type == "project" and not all(
                _expr_is_morsel_safe(e._ir) for e in fop.exprs
            ):
                return False
        return True
    # Limit, Materialize, and anything unrecognized: single-morsel path
    return False


# Kernel classes that are compute-bound per byte and therefore gain
# from thread parallelism. Measured (June 2026, quiet Apple Silicon,
# 2M-8M rows): string chains win 1.32-1.43x at EVERY scale, while
# numeric elementwise chains lose ~0.5x at EVERY scale - a single
# thread already saturates memory bandwidth for arithmetic, and the
# same saturation explains the bare filter+select residual (0.87x)
# that M3 part 2's plumbing investigation could not. The gate is
# therefore kernel-CLASS, not op count: parallelize only chains that
# contain at least one compute-bound kernel. Extend this tuple as new
# kernel classes are measured (regex, date parsing are candidates).
_COMPUTE_BOUND_PREFIXES = ("str_",)


def _expr_has_compute_bound(ir) -> bool:
    from pandas.lazy.ir import (
        Alias,
        Call,
    )

    if isinstance(ir, Alias):
        return _expr_has_compute_bound(ir.arg)
    if isinstance(ir, Call):
        if ir.function.startswith(_COMPUTE_BOUND_PREFIXES):
            return True
        return any(_expr_has_compute_bound(a) for a in ir.args)
    return False


def _chain_has_compute_bound(pipeline: Pipeline) -> bool:
    for op in pipeline.operators:
        if isinstance(op, PhysicalFilter):
            if _expr_has_compute_bound(op.predicate._ir):
                return True
        elif isinstance(op, PhysicalProject):
            if any(_expr_has_compute_bound(e._ir) for e in op.exprs):
                return True
        elif isinstance(op, PhysicalFusedPipeline):
            for fop in op.operations:
                if fop.op_type == "filter" and _expr_has_compute_bound(
                    fop.predicate._ir
                ):
                    return True
                if fop.op_type == "project" and any(
                    _expr_has_compute_bound(e._ir) for e in fop.exprs
                ):
                    return True
    return False


def pipeline_is_morsel_parallel(pipeline: Pipeline) -> bool:
    """Can and should this pipeline's operator chain run morsel-parallel?

    Requires: an in-memory source (PhysicalScan or an upstream sink's
    materialized output — file scans become natural morsel sources in
    M6), every operator provably row-wise, and at least one
    compute-bound kernel in the chain (bandwidth-bound chains are
    saturated by a single thread; see _COMPUTE_BOUND_PREFIXES).
    """
    if not pipeline.operators:
        return False
    if pipeline.source_node is not None and not isinstance(
        pipeline.source_node, PhysicalScan
    ):
        return False
    if not all(_op_is_morsel_safe(op) for op in pipeline.operators):
        return False
    return _chain_has_compute_bound(pipeline)


def compile_chain(pipeline: Pipeline):
    """Compile the operator chain into lean per-morsel appliers, once.

    Each applier is the node's existing arrays-level core method
    (``_filter_batch`` / ``_project_batch`` / ``_execute_fused`` — the
    same code ``execute()`` delegates to), bound once per pipeline.
    This removes the per-morsel ``dataclasses.replace`` + ``execute()``
    wrapper that M3's gate measured as the dominant serial fraction,
    with zero semantic divergence: it is literally the same logic.
    """
    appliers = []
    for op in pipeline.operators:
        if isinstance(op, PhysicalFusedPipeline):
            appliers.append(op._execute_fused)
        elif isinstance(op, PhysicalProject):
            appliers.append(op._project_batch)
        elif isinstance(op, PhysicalFilter):
            appliers.append(op._filter_batch)
        elif isinstance(op, PhysicalConvert):

            def convert(arrays, ctx, _op=op):
                from pandas.lazy.backends.convert import ensure_backend

                return {
                    name: ensure_backend(arr, _op.target_backend)
                    for name, arr in arrays.items()
                }

            appliers.append(convert)
        else:  # pragma: no cover - classifier admits only the above
            raise TypeError(f"non-morsel operator: {type(op).__name__}")
    return appliers


def _slice_arrays(arrays: ArrayDict, start: int, end: int) -> ArrayDict:
    return {name: arr[start:end] for name, arr in arrays.items()}


def _concat_column(parts: list):
    """Concatenate one column's morsel results, backend-preserving.

    Backends can differ across morsels for the same column (runtime
    thresholds may route a small tail morsel differently), so normalize:
    if any part is Arrow, all parts merge as Arrow chunks.
    """
    import pyarrow as pa

    if any(isinstance(p, (pa.Array, pa.ChunkedArray)) for p in parts):
        chunks = []
        for p in parts:
            if isinstance(p, pa.ChunkedArray):
                chunks.extend(p.chunks)
            elif isinstance(p, pa.Array):
                chunks.append(p)
            else:
                chunks.append(pa.array(p, from_pandas=True))
        return pa.chunked_array(chunks)
    return np.concatenate(parts)


def concat_morsel_results(results: list[ArrayDict]) -> ArrayDict:
    """Merge per-morsel outputs in sequence order (stable row order)."""
    non_empty = [r for r in results if r and len(next(iter(r.values()))) > 0]
    if not non_empty:
        return results[0] if results else {}
    if len(non_empty) == 1:
        return non_empty[0]
    names = list(non_empty[0].keys())
    return {name: _concat_column([r[name] for r in non_empty]) for name in names}


def run_morsel_parallel(
    pipeline: Pipeline,
    arrays: ArrayDict,
    context: ExecutionContext,
    n_rows: int,
) -> ArrayDict:
    """Run the pipeline's operator chain morsel-parallel over ``arrays``.

    Workers self-dispatch: each claims the next morsel index from a
    shared cursor (one small lock per claim — measured at ~1µs against
    multi-ms kernels in the scaling spike) and runs the entire chain on
    its morsel in a cloned context. Results land in a slot array and
    merge in sequence order.
    """
    from pandas.lazy.cost import get_morsel_size

    morsel_size = get_morsel_size()
    n_morsels = (n_rows + morsel_size - 1) // morsel_size
    results: list[ArrayDict | None] = [None] * n_morsels
    claim = itertools.count()
    claim_lock = threading.Lock()
    errors: list[BaseException] = []

    # Compile the chain once; workers reuse the bound appliers.
    appliers = compile_chain(pipeline)

    def worker() -> None:
        ctx = context.clone_for_subplan()
        while not errors:
            with claim_lock:
                i = next(claim)
            if i >= n_morsels:
                return
            start = i * morsel_size
            end = min(start + morsel_size, n_rows)
            try:
                out = _slice_arrays(arrays, start, end)
                for apply_op in appliers:
                    out = apply_op(out, ctx)
                results[i] = out
            except BaseException as exc:  # propagate to caller
                errors.append(exc)
                return

    import os

    n_workers = min(MAX_WORKERS, os.cpu_count() or 1, n_morsels)
    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise errors[0]
    return concat_morsel_results(results)  # type: ignore[arg-type]
