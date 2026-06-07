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
order-preserving merges. Default morsel size ~128K rows — in the
canonical range (Leis et al. recommend ~100K; DuckDB uses ~100K rows
and a 122,880-row ROW_GROUP_SIZE; the literature shows performance is
flat above ~10K, so the exact value is not load-bearing) — but
**exposed as a tunable** (`compute.lazy.morsel_size`), not hardcoded,
since the optimum is hardware-dependent and our per-morsel Python
overhead is higher than a native engine's (see GIL analysis).

**Column representation is declared per pipeline at plan time** (P2/P3):
each column carries its backend (`arrow` | `numpy`) as a physical plan
property. The planner inserts explicit `Convert` operators only where a
kernel requires the other backend, costed against the alternative of
running a less-preferred kernel. The June lesson is enforced
structurally: a column that no operator touches is *provably* never
converted, because conversion only exists as an operator.

### Execution: the scheduler

One process-wide worker pool (`min(cores, 8+)` threads — kernels release
the GIL, so threads suffice; see GIL analysis).

**The scheduler is a passive data structure, not a dispatcher thread.**
This follows HyPer's explicit rejection of a dedicated dispatcher
("would need a core to run on or might preempt query evaluation threads
and could become a source of contention. Therefore, the dispatcher is
implemented as a lock-free data structure only; the dispatcher's code is
executed by the work-requesting query evaluation thread itself"). The
research flagged a central Python dispatcher as this design's single
biggest risk — per-morsel dispatch through one coordinator is exactly
the contention pattern the canonical engine avoided, and Python
magnifies it. Concretely:

1. Plan state is shared: per-pipeline morsel cursors (an atomic claim
   index over the input range) plus the dependency graph.
2. **Workers self-dispatch**: each worker claims the next morsel range
   directly from the pipeline's cursor (a single atomic/locked
   increment), runs the *entire* operator chain on it
   (filter→project→…), and pushes the result into the sink. No thread
   hands work to another thread.
3. Sinks accumulate thread-locally where possible (partial aggregates,
   sort runs, hash partitions) and merge in `finalize()`.
4. Ordered outputs (limits, stable requirements) merge by `seq`;
   unordered sinks (aggregation) skip the ordering cost.
5. Early termination: a satisfied limit flips the pipeline's cursor to
   exhausted; workers simply find no more morsels (generalizes today's
   streaming `head()` behavior).
6. **Backpressure**: in-flight morsels per pipeline are bounded by a
   semaphore so a fast source cannot queue unbounded sink input —
   Polars' new engine pairs morsel-driven scheduling with exactly such
   a flow-control layer rather than a textbook scheduler.

This replaces the five ad-hoc thread pools with one worker pool over a
passive claim structure, and makes parallelism a property of *every*
pipeline rather than of individual operators that happen to implement
it. **M3 must measure the per-claim cost explicitly** (target: claim +
result handoff < 10% of per-morsel kernel time at the default morsel
size).

### Breakers as parallel algorithms

Each sink is specified with its parallel form — all three have working
prototypes in the current codebase:

| Sink | Parallel form | Existing prototype |
|---|---|---|
| Aggregate | per-worker partial aggregation (Arrow `group_by` per morsel or pandas Cython), **radix over-partitioned** → parallel per-partition merge | streaming aggregation's partial states |
| Sort | per-worker stable sorted runs → **k-way merge with computed non-overlapping intersections** | `_parallel_argsort` (chunked sort; its pairwise merge is upgraded — see below); external sorter's run merging |
| Join build | partition-by-hash into per-worker hash tables; probe pipelines stream morsels against them | grace hash join partitioning; `get_join_indexers` per partition |
| Distinct | per-worker seen-sets on hashed keys → merge | — |

Research-driven specifications for the two hardest sinks:

- **Aggregate — over-partition** (DuckDB's production guidance): each
  worker keeps one internally radix-partitioned table with *more
  partitions than threads* (e.g. 32), so the phase-2 merge parallelizes
  per partition with zero synchronization (identical hashes land in one
  partition) and partitions spill independently under memory pressure.
  Documented weakness of thread-local partials (Xue & Marcus 2025):
  inverse scaling past ~32 threads at high cardinality, with
  O(threads × distinct keys) memory. Our pool is ≤16 threads so the
  scaling cliff is distant, but the memory term is real — the sink
  monitors partial-table cardinality and at high cardinality switches
  to merging more eagerly per partition (and, as a possible later
  refinement, a shared atomic-update table, which the 2025 results show
  matches or beats partitioning for high-cardinality/low-skew loads).
- **Sort — k-way Merge Path, not cascading two-way merges** (DuckDB's
  2025 sort redesign): generate thread-local sorted runs in the sink,
  then compute where the k runs intersect for evenly-sized output
  chunks (searchsorted gives the intersections — our existing
  `_merge_sorted_runs` generalizes) so threads merge disjoint output
  ranges with no synchronization. DuckDB measured ~6.5x at 8 threads
  for k-way vs ~3.5x for the cascading pairwise merge that
  `_parallel_argsort` currently uses — upgrading the merge shape is
  worth roughly as much as parallelizing in the first place.

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

- **Codegen/JIT** — we do not build our own; but see "Future backends"
  below: an XLA-backed kernel backend (JAX) would supply fused codegen
  without us writing a compiler.
- **Our own SIMD kernels** — commoditized; Arrow's are fine (P1).
- **Async I/O scheduling** — Parquet row-group reads are already
  parallel inside pyarrow; revisit only if scan profiles demand it.

### Future backends: the JAX door (kept open, not committed)

The architecture admits an XLA-backed numeric kernel backend with no
structural changes, and several design choices make it unusually cheap:

- The kernel registry is backend-pluggable (`register_kernel(op,
  "jax")`); the router already arbitrates backends per operation.
- **Per-column backends** (M2) contain JAX's hardest limitation — no
  string dtype — by routing numeric columns to JAX while strings stay
  Arrow, a decision the layer already makes per column.
- **Fixed-size morsels neutralize XLA's static-shape requirement**: one
  compilation serves every morsel of a pipeline (pad the tail morsel).
- **Pipelines are the lowering unit**: a streaming-operator chain of
  element-wise ops can compile into a single `jax.jit` kernel — fused
  codegen in one memory pass, which is exactly the lever against the
  measured memory-bandwidth ceiling (~3x) that per-op NumPy/Arrow
  kernels cannot beat.
- **Conversions-as-plan-ops** (P2) price the dlpack/device transfer at
  the boundary, so JAX is chosen only when a pipeline is compute-dense
  enough to pay it.

Scope if pursued: element-wise expression pipelines, reductions,
segment-based aggregations, windows — a kernel backend, never the
engine. Constraints to design around: no nullable types (NaN for
floats, explicit masks for nullable ints), float32 defaults and
accumulation-order differences (equivalence suite needs tolerance
policies for JAX-routed paths), strictly optional dependency. GPU
morsel offload is the speculative extension once the CPU backend
proves itself.

## GIL analysis — MEASURED (June 2026 spike)

The design's viability rested on one claim: **Python orchestration per
morsel·operator is small relative to kernel time, and kernels release
the GIL.** This was validated empirically *before* M3 with a spike
(`../benchmarks/spike_morsel_scaling.py`) that simulates the execution
model faithfully — workers self-dispatching morsel claims from a shared
counter, full filter→project→gather pipelines per 128K morsel — run on
both stock CPython 3.11 (GIL) and free-threaded 3.13t, on Apple Silicon
(4 performance + 4 efficiency cores):

| Workload | GIL 3.11 | Free-threaded 3.13t |
|---|---|---|
| Bandwidth-bound pipeline (filter→project→gather, 16M rows) | 2.85x @ 4t | 3.06x @ 4t |
| Compute-dense (argsort per morsel) | **5.47x @ 8t** | **5.55x @ 8t** |
| Orchestration overhead per morsel | 1µs | 1µs |

Three conclusions, each stronger than the design's original hedge:

1. **The GIL is not a limiting factor.** GIL-threaded scaling matches
   free-threaded scaling *exactly* in both regimes — NumPy/Arrow kernels
   release the GIL completely enough that the 1µs/morsel of serialized
   Python is invisible. The original 30–100µs orchestration budget was
   30–100x too pessimistic.
2. **The real ceilings are physics**: memory bandwidth caps streaming
   pipelines near ~3x on this machine, and core asymmetry caps compute
   near ~5.5x on 4P+4E — the same walls native engines hit. M3/M4
   targets are set against these measured ceilings, not against
   idealized linear scaling.
3. **Free-threaded Python is a compatibility hedge, not a dependency.**
   The architecture needs nothing from it today (and pyarrow's
   free-threaded support is still pending — the `py313-freethreading`
   env carries NumPy only). When the ecosystem lands, the same code
   benefits on Python-heavy edge paths with zero redesign; nothing in
   the design bets against the GIL existing.

M3 remains the integration gate — the spike validates the model with
synthetic pipelines; M3 validates it with the real engine, including the
per-claim overhead measurement (<10% of per-morsel kernel time).

## Research validation (June 2026)

A deep research pass (26 sources, 25 adversarially-verified claims; full
trail in the session record) checked every load-bearing choice against
the literature and the engines we benchmark:

| Design choice | Verdict | Source of truth |
|---|---|---|
| ~128K morsels | **Confirmed** — canonical range (Leis et al. ~100K, DuckDB ~100K/122,880); perf flat above ~10K; made tunable | Leis et al. 2014; DuckDB external aggregation |
| Pipelines-between-breakers, workers run whole chains per morsel | **Confirmed verbatim** against the founding paper | Leis et al. 2014 |
| Central scheduler | **Challenged — amended.** HyPer explicitly rejected a dispatcher thread (contention); dispatcher is a passive lock-free structure, workers self-dispatch. Design updated accordingly; M3 measures per-claim cost | Leis et al. 2014 §dispatcher |
| Thread-local partial-agg sinks | **Confirmed as mainstream** (HyPer/DuckDB/DataFusion all partition); amended with radix over-partitioning and a documented high-cardinality weakness + fallback | DuckDB aggregate-hashtable posts; Xue & Marcus 2025 |
| Sorted-run + k-way merge sort sink | **Confirmed**, sharpened: k-way Merge Path beats cascading pairwise ~6.5x vs ~3.5x at 8 threads | DuckDB 2025 sort redesign; Green et al. Merge Path |
| Morsel-driven for a *dataframe* engine | **Confirmed by the competitor**: Polars' new streaming engine is explicitly morsel-driven (citing the same paper), hybridized with async state machines and flow control — hence the backpressure semaphore added to the scheduler section | Polars engineering posts |

The research found no documented failure of the GIL-threads-over-native-
kernels model at our scale, but also no positive proof — M3 remains the
empirical gate, now with an explicit per-dispatch overhead measurement.

## Migration: evolutionary milestones

Every milestone: full suite green, equivalence suite green, benchmark
gates pass (`--update-baseline` after verified intentional improvements),
explain() output stays truthful.

| # | Milestone | Contents | Target (vs Polars, mixed data) |
|---|---|---|---|
| M1 ✅ | Pipeline compiler (**landed June 2026**) | `pandas/lazy/engine/`: Morsel/Pipeline/Sink/PipelineCompiler/PipelineExecutor; every `collect(use_physical_planner=True)` compiles to and executes through an explicit pipeline graph; nodes run their own `execute()` via the `_Precomputed` input adapter, so behavior is byte-identical; multi-input metadata isolation mirrors join/concat's per-side context clones; `explain(physical=True)` renders the graph | gates: 1,540 tests green, all 4 benchmark baselines pass unchanged |
| M2 ✅ | Decision layer (**complete June 2026**) | `engine/decisions.py` annotates the compiled graph: per-column backends per pipeline, explicit conversion points, sink decisions — all visible in `explain(physical=True)`. First migrated decision: groupby backend planned from the input schema (the June routing-bug site), runtime rule kept as fallback; A/B-verified identical. Parts 2-3: filter backend + join build-side migrated; acero join routing with order-freeness propagation; **thresholds absorbed into the cost model** (`pandas/lazy/cost.py` owns every engine decision constant with measured provenance; `compute.lazy.morsel_size` registered; option-backed catalog unchanged in optimize/config.py) | gates: 1,544 tests green, baselines unchanged; explain shows decisions |
| M3 ⏳ | Morsel parallelism (**part 1 landed June 2026; gate result mixed — honest read below**) | `engine/parallel.py`: in-memory sources partition into 128K morsels; workers self-dispatch via claim cursor; whole chains per morsel in cloned contexts; seq-ordered backend-normalizing merge; safety classifier (limits, aggregates, window/cumulative excluded). **Measured at 10M rows**: compute-dense chains (string + arithmetic) 1.35x; bare filter+select **0.89x** — per-morsel Python operator plumbing (expression re-evaluation, node rebinding) plus the output merge consume the bandwidth-bound gain that the synthetic spike (2.85x) didn't pay. Response per the gate contract: a compute-score gate applies parallelism only where it measurably wins (string/multi-op chains), simple chains stay on the unchanged sequential path. **Part 2 (compiled chain applier) landed — hypothesis disproven**: binding the nodes' arrays-level cores once per pipeline (instead of per-morsel rebinding) changed nothing (1.36x vs 1.35x; 0.87x vs 0.89x), and decomposition shows morselization (~0 ms), result concat (~0 ms), and rebinding were all innocent. **Part 3 (quiet-machine measurement) resolved both open questions**: numeric elementwise chains lose at EVERY scale (~0.5x, 2M-8M — a single thread saturates memory bandwidth, which also fully explains the bare-filter 0.87x residual), while string chains win at every scale (1.32-1.43x — compute-bound per byte). The earlier 'compute-dense 1.35x' was carried entirely by its string op, and the op-count score gate let a 1M numeric chain regress +99% before the kernel-class gate replaced it: parallelism now requires at least one compute-bound kernel (str_*, extensible as new classes are measured). The benchmark gate caught the regression — on its own baselines, as designed | suite 1,552 green; filter+select unchanged (41 ms); compute-dense 228→175 ms |
| M4 ⏳ | Parallel sinks (**sort part landed June 2026**) | k-way segment merge replaces the cascading pairwise merge in `_parallel_argsort`: pivot splits with stability-preserving tie handling (all pivot-equal values left of the boundary; run-order gathering makes local stable argsort honor global (run, position) tie order), parallel per-segment merge. 10M argsort 1449→399 ms (3.63x vs cascade era 1.8x); end-to-end sort @10M 1631→856 ms — inside Polars' own range; gates recorded -33% to -57% on every sort shape (multikey 10M 5208→2230 ms). Stability pinned on dup-heavy/NaN/all-equal/sorted/reversed inputs. **Aggregate part resolved by measurement, not machinery**: acero's hash aggregation is already internally multi-threaded — a Python partial-agg sink would fight it, not help it, so the canonical thread-local-partials design (written for engines that own their hash tables) deliberately does NOT apply here. The measured lever was **key representation**: acero aggregates dictionary keys at 13 ms vs 67 ms for raw large_string @10M (faster than Polars' 18 ms) — but Categorical columns were falling through to per-query object materialization (313 ms!). Fixed end-to-end: extract_array wraps Categorical zero-copy as pa.DictionaryArray, the schema maps CategoricalDtype to arrow storage so the decision layer routes groupby[arrow], output converts back to category dtype. **10M category-key groupby: 313 → 21 ms (15x)** — parity with Polars grouping string keys (18 ms); Polars on its own Categorical dtype runs ~9 ms, so the honest ratio against like-for-like is ~0.43x. Remaining known gap: acero's raw-string hashing (67 vs 18 ms) is a kernel-quality gap; cached dictionary encoding per source is the noted future lever | sort target ≥0.5x: **hit**; category-key aggregation: 15x recovered, ~0.43x vs Polars-categorical |
| M5 ✅ | Parallel join (**landed June 2026 — measurement rewrote the design**) | The spec'd partitioned-build/parallel-probe does NOT work in this engine: pandas' Cython hash join holds the GIL (threaded partition-pairs measured 461→535 ms going 2→8 threads vs 430 serial — first milestone that would genuinely need free-threading). The measured lever: **acero's internally-parallel hash join runs the same 10M×1M join in 160 ms vs 684 ms** — but with nondeterministic output order, while our join contract matches eager pd.merge order. Resolution: the decision layer routes to acero **only when the join feeds an order-insensitive sink** (groupby/sort/topk/distinct) with acero-safe keys (int/string, non-nullable — pandas matches NaN==NaN/None==None, acero doesn't) and no index observation; order-observable joins keep the indexer path. join→groupby @10M×1M: 1319→925 ms (1.43x), 0.42x→**0.60x of Polars** | join ≥0.6x: **hit** on order-free pipelines; direct-materialize joins keep eager-order semantics at 0.31x — a documented semantics price, not a gap |
| M6 ✅ | Scan-native morsels (**landed June 2026, as a regression fix**) | File-scan-sourced pipelines execute through the nodes' native `execute_batches` protocol — scan batches ARE the morsels, embedded limits terminate reads early. This restored streaming that M1's executor had silently broken (it materialized scans via `execute()` before applying limits: glob `head(1000)` read all 4 files — 2,788 ms vs 137 ms pre-M1; invisible to the small-file suite, caught by re-measuring the report queries). Now pinned by a monkeypatch regression test asserting `execute()` is never called for limit queries. glob head: 203 ms; filter.head: 58 ms. **Known remaining gap**: full-scan `select(3).filter` at ~104 ms vs Polars 7.7 — decode-speed/laziness difference, not over-reading; tracked, not hidden | streaming restored + pinned; full-scan decode gap documented |

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
