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
  The residual ~0.45x TPC-H gap was long attributed to the execution model
  (pipelining/fusion). **June 2026 correction (`ENGINE_GAP_REFRAMING.md`):** that
  was measured vs Polars's *materializing* engine, which already beats us ~3x;
  its *streaming* engine adds only 10–35%. The real 3x is (1) string/wide-key
  group-by falling back to single-threaded Acero (the parallel kernel never fires
  on string keys) and (2) per-join data movement — both catchable in the current
  materializing model, **not** a new engine.
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
- **JAX/XLA + Pallas custom kernels** (NO-GO, `JAX_CUSTOM_KERNEL_PROBE.md`):
  element-wise loses to NumExpr; factorize/hash is outside XLA; the
  segment-reduction win is beaten ~2x by a single-pass Cython scatter.
- **GPU acceleration** (NO-GO as an engine lever, `GPU_DEVICE_RESIDENT_PROBE.md`):
  measured on Blackwell PCIe + GH200 unified memory; loses the suite, wins only
  heavy aggregate/join queries; even GH200's 17x link is stranded by the
  host-frame software memory model. The from-scratch heterogeneous engine needed
  to capture it is already NO-GO.

**Engine-gap REFRAMED — the gap is NOT the streaming/execution model
(June 2026, `ENGINE_GAP_REFRAMING.md`).** An architecture study of Polars +
DataFusion plus a three-way measurement *corrected* the earlier "execution-model
gap" conclusion. The earlier numbers were all vs Polars's bare `.collect()` =
its **materializing in-memory** engine. New finding: that materializing engine
(our exact model) **already beats us ~3x**, and Polars's *streaming* engine
(gather-into-probe fusion + cascade pipelining) adds only **10–35%** on top — on
q8 it regresses. DataFusion seals it: it does indices-then-`take` (no gather
fusion) and still wins, so **fusion is not the differentiator**. Per-operator
decomposition (`benchmarks/decomp_inmem_gap.py`) localizes the real 3x to:
(1) **string/wide-key group-by falling back to single-threaded Acero**
(`_partition_key_arrays` returns None for non-integer keys → the parallel kernel
never fires on real TPC-H string group keys; q10 117ms vs 35ms, q9 62ms); and
(2) **per-join data movement** (key re-conversion + separate gather + Arrow↔NumPy
round-trips, ~2.5x the cascade despite keys-only parity). **Both are catchable
in the current materializing model — a new streaming engine is the wrong lever**
(recovers at most the 10–35%). Live levers: parallelize string-key group-by;
cut per-join data movement.

*Earlier (now-superseded) framing, kept for the record:* the buffer-resident
join→agg fusion reached isolated parity but was net-negative on the real suite
(q18 4x); `JOIN_KERNEL_PROFILE.md` attributed the gap to gather-into-probe
fusion + cascade pipelining. `PhysicalFusedJoinAgg` remains default-off.

**Pending (live candidates):**
- Predicate transfer at scale — run `bench_predicate_transfer.py --sf 30/100`
  on EC2. GO (build engine-integrated Bloom-filter PT + optimizer pass) iff
  ≥~1.5x at scale; else close it.

## Accelerator line — closed (June 2026)

After the engine NO-GO, the accelerator question was attacked from four angles —
**JAX-CPU fusion, JAX/Pallas custom kernels, PCIe discrete GPU, GH200
unified-memory GPU** (`JAX_CUSTOM_KERNEL_PROBE.md`, `GPU_DEVICE_RESIDENT_PROBE.md`).
All four converged on the same answer, which *is* the finding: **the ceiling is
an execution-and-memory-model problem, not a kernel or hardware problem.** Each
accelerator was bottlenecked at the boundaries (data movement, host-resident
per-op orchestration), never raw compute:

- **JAX-CPU**: the compute-fusion lever is already pulled by NumExpr; the hard
  part (factorize/hash) is outside the dense-array model; the marginal win is
  cheaper in Cython.
- **GPU (PCIe Blackwell)**: wins heavy aggregate/join queries 2–5x (incl. q18,
  confirming our q18 wall is a *CPU*-substrate wall, not fundamental) but loses
  the suite — per-call H2D transfer (~53% of cost) + light queries sink it.
- **GH200 (unified memory)**: the transfer tax is *physically erased* at the
  cuDF-primitive level (q18 inner group 268→46 ms; pinned H2D 17x PCIe) — yet
  the off-the-shelf engine suite **doesn't move** (0.48x). The substrate
  improved 17x and the engine couldn't see it: **the C2C advantage is stranded
  by the software memory model.** (Caveat: tested via global RMM; explicit
  `GPUEngine(memory_resource=...)` untested.)

**Net:** more hardware won't help, and a new engine to exploit it is NO-GO. The
accelerator line is closed. The durable asset is the quantified ceiling and the
unified-memory stranding gap — the live forward thread is the upstream Arrow
track, not further perf work.

## Joins: a Rust kernel BEATS Polars (June 2026) — the Cython ceiling, lifted

The campaign's standing "joins are architecture-bound" conclusion was **partly a
Cython artifact**, corrected by building the kernel (`JOIN_KERNEL_REBUILD_PROBE.md`,
`benchmarks/rust_join_prototype/`):
- Profiling pinned the join cost: index-gen (single-thread build + 2-pass probe)
  + a separate gather, both Python-orchestrated. The Cython fused attempts hit
  the no-OpenMP wall (no real threads; scalar-vs-SIMD gather; row-major transpose
  tax) and reached only pd.merge parity / lost to Polars.
- **A Rust (PyO3 + rayon) fused parallel join+gather, column-major on numpy/Arrow
  buffers (zero-copy, GIL released), beats Polars at every payload width**
  (P=1..21: 125/179/339/727 ms vs Polars 133/192/395/762; vs pd.merge
  293/319/536/927). Correct vs pd.merge. First thing in the campaign to beat
  Polars on a join.
- **Lesson:** the limiter was the **Cython/no-OpenMP toolchain**, not the engine
  model — "Polars is Rust" is the point. A Rust accelerator brings Polars-class
  joins into pandas natively. Open cost: Rust in pandas's build (a real
  project-level decision); generalization (string/null/multi-key) is
  straightforward in arrow-rs; it's bandwidth-bound so it matches/edges Polars.

## DIRECTION CHANGE — Arrow-native Rust engine beats Polars (June 2026)

The campaign's "joins/TPC-H are architecture-bound, we're stuck vs Polars"
conclusion was **wrong** — a failure of approach (`RUST_ENGINE_DIRECTION.md`).
Every attempt hosted a fast kernel inside the Python/Cython engine and paid the
per-operator boundary. Moving execution **into Rust** dissolves it:
- pandas↔Rust Arrow boundary = **0.01 ms** (zero-copy); the tax was per-operator
  Arrow↔NumPy *inside* the Python engine, not the in/out crossing.
- **TPC-H q1 end-to-end in Rust = 30.7 ms vs Polars 243.9 ms (7.95x), correct vs
  DuckDB** (our Cython engine: 0.47x). `benchmarks/rust_engine_prototype/`.

New direction: build the lazy engine's execution on **arrow-rs** (Arrow in once,
Rust operators with rayon, Arrow out once) as the **baseline**, replacing the
~0.43x Cython floor. "Polars is Rust" was the answer; the job is to put execution
where the speed is, not bolt kernels onto a slow host.

## Durable lessons

- **Always measure in `use_physical_planner=True`** (the scorecard mode); the
  eager `.collect()` default misleads by ~4x.
- **The bottleneck follows the execution model, not the silicon.** Triangulated
  from CPU fusion, custom kernels, discrete GPU, and coherent-memory GPU — every
  accelerator stalled at data movement / host-resident orchestration, never
  compute. Even ideal hardware (GH200) couldn't be exploited by a host-frame
  engine. Confirms `PERF_CEILING.md`'s three taxes from a new direction.
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
- `ENGINE_GAP_REFRAMING.md` — **(June 2026, read first for the join/agg gap)**
  architecture study of Polars + DataFusion + three-way measurement; corrects
  the "execution-model gap" conclusion (materializing engine already 3x;
  streaming only +10–35%; real gap = string-groupby fallback + per-join data
  movement).
- `QGAP_DECOMP.md` — per-operator decomposition; methodology fix; fusion
  rejections; join-chain localization.
- `STRING_HASH_AGGREGATE_KERNEL.md` — parallel string-key factorize group-by
  (reaches Polars parity on the operator; integrated default-on; q10 −17%).
- `Q18_DECOMP.md` — q18 blocker = the ultra-high-card int group-by (45%);
  substrate-bound, factorize-int probed equivalent (no bounded lever).
- `PARALLEL_GROUPBY_SCOPE.md` — the shipped kernel: scope, results, reach, load-
  dependence.
- `PERF_CEILING.md` — the three taxes; high-effort paths (A pushdown / B native
  / C free-threading[rejected] / D arrow-native / E kernels).
- `ENGINE_DIFFERENTIATION.md` — what would make a new engine *much* better.
- `ENGINE_GONOGO_MEMO.md` — NO-GO on the MLIR engine (research-spike-backed).
- `PREDICATE_TRANSFER_PROBE.md` — PT probe; thin at SF-3; scale-up needed.
- `bench_predicate_transfer.py` — the EC2 go/no-go probe (ready to run).
- `JAX_CUSTOM_KERNEL_PROBE.md` — NO-GO on JAX custom kernels/Pallas for any hot
  path on CPU (factorize is outside XLA; segment-reduction win beaten ~2x by a
  single-pass Cython scatter); extends the ROADMAP JAX decline.
- `GPU_DEVICE_RESIDENT_PROBE.md` — measured GPU runs (Blackwell PCIe + GH200
  unified memory, real TPC-H): GPU loses the suite (0.48–0.81x) but cracks heavy
  aggregate/join queries incl. q18 at 2–3x. Transfer = 53% of per-call cost on
  PCIe; on GH200 managed memory erases it at the primitive level (q18 268→46 ms,
  pinned H2D 17x PCIe) yet the off-the-shelf engine suite is unchanged — the C2C
  advantage is *stranded by the software memory model*. Lever = execution+memory
  model, a different engine, not a kernel and not just the hardware.
