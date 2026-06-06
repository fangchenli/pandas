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

from pandas.lazy.engine.pipeline import (
    Pipeline,
    PrecomputedInput,
    with_inputs,
)
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

# Tunables (canonical range per Leis et al. / DuckDB; see ENGINE_DESIGN.md)
MORSEL_SIZE = 131_072
MIN_PARALLEL_ROWS = 2 * MORSEL_SIZE
MAX_WORKERS = 8

# IR functions that are order- or whole-input-dependent: applying them
# per morsel would change results.
_NON_MORSEL_FUNCTIONS = frozenset(
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
        if ir.is_aggregate or ir.function in _NON_MORSEL_FUNCTIONS:
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


def _expr_compute_score(ir) -> int:
    """Rough kernel-work weight of an expression (Call nodes)."""
    from pandas.lazy.ir import (
        Alias,
        Call,
    )

    if isinstance(ir, Alias):
        return _expr_compute_score(ir.arg)
    if isinstance(ir, Call):
        own = 3 if ir.function.startswith("str_") else 1
        return own + sum(_expr_compute_score(a) for a in ir.args)
    return 0


def _chain_compute_score(pipeline: Pipeline) -> int:
    score = 0
    for op in pipeline.operators:
        if isinstance(op, PhysicalFilter):
            score += _expr_compute_score(op.predicate._ir)
        elif isinstance(op, PhysicalProject):
            score += sum(_expr_compute_score(e._ir) for e in op.exprs)
        elif isinstance(op, PhysicalFusedPipeline):
            for fop in op.operations:
                if fop.op_type == "filter":
                    score += _expr_compute_score(fop.predicate._ir)
                elif fop.op_type == "project":
                    score += sum(_expr_compute_score(e._ir) for e in fop.exprs)
    return score


# Minimum kernel-work weight for morsel parallelism to pay. Measured
# (June 2026, 10M rows, Apple Silicon): a bare filter+select (score 1)
# runs 0.89x parallel - per-morsel operator plumbing plus the output
# merge eat the bandwidth-bound gain - while a string+arithmetic chain
# (score ~7) runs 1.35x. Reducing the per-morsel plumbing (a compiled
# chain applier instead of per-morsel node rebinding) is the tracked
# next step; until then, parallelism applies only where it wins.
MIN_COMPUTE_SCORE = 3


def pipeline_is_morsel_parallel(pipeline: Pipeline) -> bool:
    """Can and should this pipeline's operator chain run morsel-parallel?

    Requires: an in-memory source (PhysicalScan or an upstream sink's
    materialized output — file scans become natural morsel sources in
    M6), every operator provably row-wise, and enough kernel work per
    row for parallelism to beat its own overhead (MIN_COMPUTE_SCORE).
    """
    if not pipeline.operators:
        return False
    if pipeline.source_node is not None and not isinstance(
        pipeline.source_node, PhysicalScan
    ):
        return False
    if not all(_op_is_morsel_safe(op) for op in pipeline.operators):
        return False
    return _chain_compute_score(pipeline) >= MIN_COMPUTE_SCORE


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
    n_morsels = (n_rows + MORSEL_SIZE - 1) // MORSEL_SIZE
    results: list[ArrayDict | None] = [None] * n_morsels
    claim = itertools.count()
    claim_lock = threading.Lock()
    errors: list[BaseException] = []

    def apply_chain(morsel_arrays: ArrayDict, ctx: ExecutionContext) -> ArrayDict:
        out = morsel_arrays
        for op in pipeline.operators:
            child = op.children()[0]
            bound = with_inputs(
                op, [PrecomputedInput(arrays=out, schema=child.output_schema)]
            )
            out = bound.execute(ctx)
        return out

    def worker() -> None:
        ctx = context.clone_for_subplan()
        while not errors:
            with claim_lock:
                i = next(claim)
            if i >= n_morsels:
                return
            start = i * MORSEL_SIZE
            end = min(start + MORSEL_SIZE, n_rows)
            try:
                results[i] = apply_chain(_slice_arrays(arrays, start, end), ctx)
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
