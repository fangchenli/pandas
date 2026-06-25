# Lazy-pandas performance campaign — summary & index

Capstone for the #1 (join/aggregate-heavy query) campaign. Reads as the
narrative + index over the detailed docs. Bottom line up front, then the arc,
then what shipped / parked / pending, then the durable lessons.

## Bottom line

- **One shipped win:** a parallel partitioned hash-aggregate kernel that **beats
  Polars ~1.9x** on high-cardinality grouped aggregation (isolated 52 vs 102ms;
  bit-exact; all 22 TPC-H validate; 1710 lazy tests pass). The first lever the
  whole campaign that *beats* Polars on a hot path.
- **Everything else was rejected or parked by measurement**, not by hand-waving.
  The residual ~0.45x TPC-H gap is the **execution substrate / whole-pipeline**
  (joins, materialization between operators, the Arrow↔NumPy boundary), not any
  individual kernel — individual kernels are at/near parity once measured in the
  right mode.
- **One candidate remains, de-risked but unproven at scale:** predicate transfer
  (semijoin reduction). Real algorithm, thin margin at SF-3; an EC2 scale-up
  probe is written and ready to make the go/no-go call.

## The arc (each step measured, controlled on/off, validated)

1. **Decomposition + methodology fix** (`QGAP_DECOMP.md`). Localized q20/q21.
   *Critical correction:* must measure with `collect(use_physical_planner=True)`
   — the default `.collect()` is eager pandas (~4x slower) and produced a false
   "count-distinct kernel is 4.7x slow" finding. Under the real mode, n_unique
   and grouped-sum kernels are ~parity. The committed scorecard was always
   measured correctly; only a throwaway harness was wrong.
2. **Three fusion mechanisms rejected** for filter→groupby (`QGAP_DECOMP.md`):
   Acero native (no help, 216 vs 202ms), streaming partial-aggregate+merge
   (*regresses* high-cardinality 2x), custom fused mask→hashagg (substrate-class
   for ~45ms on one shape).
3. **Parallel groupby kernel SHIPPED** (`PARALLEL_GROUPBY_SCOPE.md`,
   `_libs/lazy_groupby.pyx`). Key insight: Arrow's `group_by` is *single-
   threaded* but its serial algorithm is 2.4x better than Polars' serial; Polars
   wins purely by parallelism. So partition by key-hash (nogil counting-sort
   scatter) → run Arrow group_by per bucket on a thread pool → concat. Two
   regressions caught+fixed mid-build (wide-table take; low-card overhead).
   Reach: 6/22 queries; q17 −20% even loaded; win scales with free cores.
4. **Joins don't yield** (`QGAP_DECOMP.md`, `PERF_CEILING.md`). The
   partition-parallel trick does NOT transfer: `pd.merge` is GIL-bound (4×
   concurrent = 2.75x/4) and Arrow's join is already internally threaded (no
   headroom) with a large-output round-trip. **Free-threading probed and
   rejected** on the 3.14t build: `pd.merge` still only 2.09x/4 (sub-GIL
   contention) and FT pandas is 2.6x slower single-threaded today. Asymmetry:
   ops on nogil primitives (Arrow, our kernels) parallelize; pandas' own
   machinery doesn't, GIL or not.
5. **Ceiling analysis** (`PERF_CEILING.md`): parity needs an execution-*model*
   change (whole-plan pushdown / native engine / arrow-native), not more
   kernels — three measured taxes (materialization at breakers, Arrow↔NumPy
   boundary, GIL) that Polars/DuckDB don't pay.
6. **New-engine go/no-go** (`ENGINE_DIFFERENTIATION.md`, `ENGINE_GONOGO_MEMO.md`)
   — backed by a 104-agent deep-research spike, 25 claims adversarially
   verified. **NO-GO** on a from-scratch MLIR data-centric compiler: the
   compiled-vs-vectorized split is *adverse* (vectorization wins the hash-join
   regime that's our gap); LingoDB hasn't beaten a parallel vectorized engine;
   compile latency is a real tax; and the best asymptotic ideas
   (Yannakakis+/predicate transfer) already ship as engine-agnostic plan
   rewrites. Gandiva is expression-only — parity, not a differentiator.
7. **Predicate transfer probed** (`PREDICATE_TRANSFER_PROBE.md`,
   `bench_predicate_transfer.py`): the algorithm avoids real join work (q3
   reduced join 23 vs 196ms), but the reduction cost gates it and at SF-3 the
   margin is thin (q9 ~1.15x, q3 ~break-even). Win should grow with scale →
   EC2 scale-up probe written, pending a run.

## Shipped / parked / pending

**Shipped (origin/lazy-pandas):**
- Parallel partitioned hash-aggregate kernel (`_libs/lazy_groupby.pyx` +
  `PhysicalHashAggregate._grouped_arrow_table`), gated + toggle + tests.
- (Earlier this project) filter→scalar-agg fusion, narrow-inner-join kernel,
  optimizer dispatch cache.

**Parked (measured dead ends — do not repeat):**
- Acero as a general substrate / arrow-across-join (round-trip).
- Engine-side streaming fusion for high-cardinality groupby (regresses).
- Parallel partitioned join (pd.merge GIL-bound; arrow join already threaded).
- Free-threading for joins (sub-GIL contention; FT single-thread overhead).
- From-scratch MLIR / data-centric-compiler engine (NO-GO, evidence-backed).
- Gandiva as a differentiator (expression-only).

**Join→agg fusion — RESOLVED, NOT A WIN (June 2026, full log in
`JOIN_GAP_INVESTIGATION_LOG.md`).** The isolated probe reached Polars parity,
but against the *real* engine it does not: the engine already late-materializes
chains (`PhysicalJoinChain`/`_CompositeJoin`) and routes single joins to acero,
so the buffer-resident fusion has no headroom — the chain extension was
net-negative (q18 regressed 4x on its high-card string group) and reverted.
Direct per-operator profiling (`JOIN_KERNEL_PROFILE.md`) then showed the real
gap: build+probe are **at parity** (218 vs 219ms), and Polars wins by **fusing
the gather into the probe** (cache-local) and **pipelining the join cascade +
streaming the group** — an execution-model gap, not a kernel. Bounded kernel
levers (partitioned join, parallel build, faster gather) all ruled out by
measurement (`JOIN_LEVERS_SCOPE.md`). `PhysicalFusedJoinAgg` landed default-off
as a documented foundation. Net: the join gap is the fused-pipelined engine
(Path A/B), out of scope for the probe.

**Pending (live candidates):**
- Predicate transfer at scale — run `bench_predicate_transfer.py --sf 30/100`
  on EC2. GO (build engine-integrated Bloom-filter PT + optimizer pass) iff
  ≥~1.5x at scale; else close it.

## Durable lessons

- **Always measure in `use_physical_planner=True`** (the scorecard mode); the
  eager `.collect()` default misleads by ~4x.
- **Measurement-first, controlled on/off, probe before building** caught every
  wrong turn this campaign (Acero, count-distinct, 3 fusion mechanisms, parallel
  join, free-threading, the MLIR engine, naive predicate transfer). The probes
  cost little; the avoided builds would have cost months.
- **The asymmetry that explains the whole gap:** operations on nogil-friendly,
  Arrow-native, single-threaded-with-headroom primitives parallelize and can
  beat Polars (groupby); operations that go through pandas' own machinery
  (`pd.merge`, block management, NumPy round-trips) do not — and closing those
  is an execution-model change, not a kernel.
- **Individual kernels are not the gap; the model is.** Confirmed repeatedly:
  under the right mode our kernels are at/beating parity; the residual is
  whole-pipeline (joins + materialization + boundary).

## Doc index
- `QGAP_DECOMP.md` — per-operator decomposition; methodology fix; fusion
  rejections; join-chain localization.
- `PARALLEL_GROUPBY_SCOPE.md` — the shipped kernel: scope, results, reach, load-
  dependence.
- `PERF_CEILING.md` — the three taxes; high-effort paths (A pushdown / B native
  / C free-threading[rejected] / D arrow-native / E kernels).
- `ENGINE_DIFFERENTIATION.md` — what would make a new engine *much* better.
- `ENGINE_GONOGO_MEMO.md` — NO-GO on the MLIR engine (research-spike-backed).
- `PREDICATE_TRANSFER_PROBE.md` — PT probe; thin at SF-3; scale-up needed.
- `bench_predicate_transfer.py` — the EC2 go/no-go probe (ready to run).
