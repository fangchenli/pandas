# Physical Engine Design: Morsel-Driven Pipelines

**Status**: approved direction (June 2026). This is the comprehensive
design for the physical engine — the answer to "what would it take to
actually compete with Polars" — replacing the grown-not-designed current
execution layer described in [PLANNING.md](PLANNING.md). Migration is
**evolutionary**: every milestone keeps the full test suite, equivalence
suite, and benchmark gates green.

## Why a designed engine, not more fixes

The June fix cycle moved every benchmark category by 2–14x (joins
0.06x→0.31x of Polars, aggregation 0.07x→0.53x, strings 0.27x→1.1x,
filter+select 0.03x→0.39x). Every one of those fixes removed the same
class of defect: **data movement or routing decisions made nowhere in
particular** — a conversion at output assembly, an `np.asarray` at join
input, a backend chosen from the first input column, a mask applied
before a projection. These defects existed *because* the engine has no
designed execution model:

- backend decisions are split across the logical optimizer
  (`EngineSelection`), runtime thresholds, and per-operator ad-hoc checks
- conversions happen as side effects, not as planned operations
- fusion is a post-pass optimization instead of the execution model
- parallelism is five unrelated thread pools (expressions, join sides,
  sort chunks, gathers, concat inputs)
- operators own their memory behavior individually; spill integration is
  bolted onto each breaker

Reactive fixing took the engine from embarrassing to competitive. It
cannot take it to parity: the remaining gap is not in any one operator
but in the *structure* — per-operator materialization, unplanned
conversions, and parallelism that exists only inside individual kernels.

## Design principles (each backed by a measured result)

- **P1 — Thick kernels, thin orchestration.** Python orchestrates
  morsel-sized units; every tight loop lives in C++ (Arrow) or Cython
  (pandas internals). *Evidence*: the join rewrite reusing pandas' own
  `get_join_indexers` beat our pure-NumPy join 6x while inheriting eager
  semantics for free.
- **P2 — Conversions are plan operations, never side effects.** A column
  changes backend only where the plan says so, costed and visible in
  `explain()`. *Evidence*: all six June fixes were unplanned conversions.
- **P3 — One decision layer.** Physical planning owns every backend,
  algorithm, and parallelism choice; runtime adapts only within planned
  bounds (the join skew→sort-merge fallback is the model). *Evidence*:
  the groupby routing bug lived in the gap between three decision sites.
- **P4 — The pipeline is the unit of execution and parallelism.** Not
  the operator. Today's `FusedPipeline` becomes the execution model
  itself, not an optimization that sometimes applies.
- **P5 — Semantics by construction.** Where pandas defines the
  semantics (joins, groupby key handling, stable sorts, NaN-as-missing),
  reuse pandas' machinery or mirror it exactly; the equivalence suite is
  the contract.
- **P6 — Measured evolution.** Each milestone lands with the 1,530-test
  suite green, equivalence holding, and a stated benchmark target
  checked against the baseline workflow.

## Architecture

### Plan compilation: logical plan → pipeline graph

The physical planner stops producing an operator tree that executes
itself and starts producing a **pipeline graph**:

```
Logical (optimized)              Pipeline graph
-------------------              -----------------------------------
Sort                             P2: source=P1.sink → [ ] → SortSink
 └─ Filter                       P1: source=Scan  → [Filter, Project]
     └─ Project                       → feeds SortSink
         └─ Scan
```

- A **Pipeline** is: a *source* (morsel iterator), an ordered list of
  **streaming operators** (stateless batch transforms: filter masks,
  projections/expressions, limits), and a **sink**.
- **Breakers** (sort, aggregate, distinct, join build) become **sinks**:
  thread-safe accumulators with `consume(morsel)` and `finalize()`.
  A sink's `finalize()` output becomes the source of the next pipeline.
- The graph is explicit and explainable: `explain(physical=True)` shows
  pipelines, their operator chains, sink types, and planned conversions.

This generalizes what already exists: `FusedPipeline` is the prototype of
the streaming-operator chain; the streaming aggregation's partial-state
accumulation and the external sorter's run-merge are prototypes of
parallel sinks. The design promotes these from special cases to the
only execution path.

### The data unit: morsels

A **Morsel** is a contiguous row-slice of an ArrayDict plus metadata:
`(arrays, offset, length, seq)` where `seq` is a sequence number for
order-preserving merges. Default morsel size ~128K rows (tuned so that
per-morsel Python overhead is <5% of kernel time; see GIL analysis).

**Column representation is declared per pipeline at plan time** (P2/P3):
each column carries its backend (`arrow` | `numpy`) as a physical plan
property. The planner inserts explicit `Convert` operators only where a
kernel requires the other backend, costed against the alternative of
running a less-preferred kernel. The June lesson is enforced
structurally: a column that no operator touches is *provably* never
converted, because conversion only exists as an operator.

### Execution: the scheduler

One process-wide worker pool (`min(cores, 8+)` threads — kernels release
the GIL, so threads suffice; see GIL analysis):

1. The scheduler walks the pipeline graph in dependency order.
2. For each pipeline, the source partitions input into morsels; workers
   pull morsels, run the *entire* operator chain on their morsel
   (filter→project→…), and push results into the sink.
3. Sinks accumulate thread-locally where possible (partial aggregates,
   sort runs, hash partitions) and merge in `finalize()`.
4. Ordered outputs (limits, stable requirements) merge by `seq`;
   unordered sinks (aggregation) skip the ordering cost.
5. Early termination: a satisfied limit cancels remaining morsels for
   its pipeline (generalizes today's streaming `head()` behavior).

This replaces the five ad-hoc thread pools with one scheduler, and makes
parallelism a property of *every* pipeline rather than of individual
operators that happen to implement it.

### Breakers as parallel algorithms

Each sink is specified with its parallel form — all three have working
prototypes in the current codebase:

| Sink | Parallel form | Existing prototype |
|---|---|---|
| Aggregate | per-worker partial aggregation (Arrow `group_by` per morsel or pandas Cython) → merge partials | streaming aggregation's partial states |
| Sort | per-worker stable sorted runs → k-way merge | `_parallel_argsort` (chunked sort + pairwise merge); external sorter's run merging |
| Join build | partition-by-hash into per-worker hash tables; probe pipelines stream morsels against them | grace hash join partitioning; `get_join_indexers` per partition |
| Distinct | per-worker seen-sets on hashed keys → merge | — |

Memory and spill attach at the sink level (P3): each sink gets a budget
from the spill manager; exceeding it spills partials/runs/partitions
using the existing Arrow IPC infrastructure. Runtime adaptation
(skew→sort-merge) stays, scoped inside the join sink.

### The decision layer

A single **physical optimization pass** replaces the current scatter
(logical `EngineSelection` + `compute.lazy.*` thresholds + per-operator
checks):

- **Inputs**: per-column storage backends (from `Schema`), row estimates
  (`estimate_row_count`, extended with a default-selectivity model for
  filters — the ROADMAP cardinality item is a prerequisite here), kernel
  preference tables (from `backends/router.py`).
- **Outputs**, stamped on the pipeline graph: per-column backend per
  pipeline, explicit Convert operators, sink algorithm choices,
  parallelism degree per pipeline (`max(1, est_rows / morsel_size)`
  capped at pool size), morsel size.
- The existing threshold catalog becomes the **cost model's constants**
  (and the calibration script tunes the cost model rather than scattered
  if-statements). `pd.set_option("compute.lazy.*")` keys keep working.

### What we deliberately do not build

- **Codegen/JIT** — out of reach for a Python-orchestrated engine and
  unnecessary while Arrow/Cython kernels are within 2x of hand-tuned
  code for our workloads.
- **Our own SIMD kernels** — commoditized; Arrow's are fine (P1).
- **Async I/O scheduling** — Parquet row-group reads are already
  parallel inside pyarrow; revisit only if scan profiles demand it.

## GIL analysis (the honest structural question)

The design's viability rests on one claim: **Python orchestration per
morsel·operator is small relative to kernel time, and kernels release
the GIL.**

Budget: orchestration ≈ 30–100µs per morsel·operator (dict handling,
dispatch). At 128K-row morsels, kernels cost ~0.5–5ms per
morsel·operator. Overhead ≈ 2–10%, parallelizable portion ≈ 90%+.
Amdahl gives ~6–7x effective scaling on 8 cores — enough to close the
remaining 2–4x gaps. **Milestone M3 exists specifically to validate this
empirically before deeper investment**; if measured scaling falls
materially short, morsel size increases or the design stops at
pipeline-level (not morsel-level) parallelism, documented honestly.

Upside hedge: pandas already builds free-threaded (`py313-freethreading`
CI env, `-Xfreethreading_compatible` Cython flag). On free-threaded
Python the same architecture scales without the GIL discount — this
design is exactly the shape that benefits, and nothing in it bets
against the GIL existing.

## Migration: evolutionary milestones

Every milestone: full suite green, equivalence suite green, benchmark
gates pass (`--update-baseline` after verified intentional improvements),
explain() output stays truthful.

| # | Milestone | Contents | Target (vs Polars, mixed data) |
|---|---|---|---|
| M1 | Pipeline compiler | Formalize Pipeline/Sink abstractions; compile every plan to a pipeline graph; single-morsel execution (whole input = one morsel) makes this a pure refactor with identical behavior. FusedPipeline, streaming agg, external sort absorb into it | no perf change; structure only |
| M2 | Decision layer | Per-column backend planning + explicit Convert ops; thresholds → cost model; delete per-operator backend checks | no regressions; explain shows conversions |
| M3 | Morsel parallelism for stateless pipelines | Scheduler + worker pool; filter/project/limit pipelines run morsel-parallel with ordered merge; **GIL validation gate** | filter_project 0.22x → ≥0.5x |
| M4 | Parallel sinks: aggregate + sort | partial-agg merge; sorted-run k-way merge | aggregation 0.53x → ≥0.8x; sort 0.27x → ≥0.5x |
| M5 | Parallel join | partitioned build, morsel-parallel probe | join 0.31x → ≥0.6x |
| M6 | Scan-native morsels | Parquet row groups and CSV chunks as natural morsels; scan pushdown feeds the scheduler directly | parquet_scan 0.30x → ≥0.6x; glob scans fixed |

M1–M2 are structural (1–2 sessions each at this codebase's pace);
M3 is the go/no-go gate; M4–M6 are independent once M3 holds.

## Relationship to existing documents

- [PLANNING.md](PLANNING.md) describes the **current** planner; it gains
  a banner pointing here and is updated as milestones land.
- [ARCHITECTURE.md](ARCHITECTURE.md) sections on parallelism, streaming,
  and spilling describe mechanisms this design absorbs into the
  scheduler/sink model.
- [ROADMAP.md](ROADMAP.md) items 3–6 (planning fast paths, sort, join,
  cardinality estimation) are subsumed by milestones M2–M5.
- [PROPOSAL.md](PROPOSAL.md) position (b): this document is the concrete
  engineering answer to "what would competing actually take."
