# The performance ceiling: what closing the gap to Polars actually takes

Written June 2026 after the grouped-aggregate kernel shipped and the parallel-
join probe failed. This is the honest, measurement-grounded answer to "is there
a way to improve performance even with a lot of effort?" The bounded plumbing
levers are exhausted; what remains is architectural. Parity is *achievable* —
Polars is not doing anything magic (it runs near the memory-bandwidth floor,
~80MB/10GBps ≈ 8ms) — but reaching it means changing the execution **model**.

## Root cause: we PLAN like a modern engine but EXECUTE as orchestrated calls

lazy-pandas has a mature logical/physical planner (M1–M6: morsels, decisions,
cost model, join routing, cardinality). But *execution* is Python orchestrating
calls into pandas / Arrow / NumPy kernels. That imposes three taxes Polars and
DuckDB don't pay, each measured this session:

1. **Materialization at every pipeline breaker.** The physical plans show
   `[BREAKER] Materialize` before each join build and each aggregate (q3, q20).
   Each kernel needs its whole input materialized, so intermediates hit DRAM
   between operators. Polars/DuckDB stream morsels through fused operators and
   never materialize the full intermediate (q20: our filter→group materialize
   costs ~45ms that Polars doesn't pay).

2. **The Arrow↔NumPy boundary.** group_by and join are fast in Arrow; pandas
   semantics need NumPy. We convert back and forth. Measured: the Arrow→NumPy
   round-trip on a join's *large* output is what made acero 2–7x slower
   end-to-end — the reason joins can't just use Arrow.

3. **GIL-bound orchestration.** Our parallelism is a ThreadPoolExecutor calling
   kernels, so only *nogil* kernels parallelize (Arrow ops, lazy_join, our new
   partition kernel). The parallel-join probe proved the limit: 4 concurrent
   `pd.merge` = 2.75x/4 (GIL-bound). Polars/DuckDB thread internally in native
   code with no GIL.

The grouped-aggregate win slipped through exactly because it dodged all three:
Arrow `group_by` is single-threaded (so partitioning + a nogil scatter + a
thread pool added the missing parallelism with clean GIL release) and emits
*small* output (no round-trip tax). No other common operator has that profile —
which is why it was the one bounded win and why joins don't yield.

## The high-effort paths (ranked by reward, with the bottleneck each removes)

### Path A — Whole-plan pushdown to a native engine (DuckDB / DataFusion)
Translate our LogicalPlan to the native engine (or Substrait), execute the
*entire* query there, materialize pandas only at the final collect.
- **Removes:** all three taxes at once (one native fused parallel runtime).
- **Reward:** Polars/DuckDB-class — true parity, possibly faster than Polars
  (DuckDB often is).
- **Effort:** high but bounded — plan translation + preserving eager-pandas
  semantics (row order, index, dtypes, NULL/NaN). NOT a new engine.
- **Why earlier probes failed and this wouldn't:** the acero/arrow-across-join
  probes pushed *per operator*, paying the boundary tax at every node. Pushing
  the *whole plan* pays it once, at the end. That's the difference.
- **Catch:** it's "DuckDB with a pandas API," not our own engine. The standing
  view is skepticism toward delegating execution to a competitor engine. But of
  all paths this has by far the best effort-to-parity ratio, and DataFusion
  (Apache, Arrow-native, embeddable) softens the "competitor" objection.

### Path B — Build a native fused execution engine
Write the execution layer in C++/Rust/Cython: one columnar format end to end
(Arrow), operators that stream morsels (no full materialize between breakers),
whole-pipeline parallelism in native threads.
- **Removes:** all three taxes.
- **Reward:** parity, and it's genuinely *our* engine ("lazy pandas competes").
- **Effort:** the real one — multi-person-year. This is rewriting the part of
  Polars/DuckDB that took those teams years.
- **When it's worth it:** only if "an independent native pandas engine" is a
  strategic goal in itself, not just a benchmark number.

### Path C — Free-threaded Python (no-GIL) — TESTED, does NOT unlock joins
Hypothesis was: the GIL caps the parallel join, so a free-threaded build lets
`pd.merge` per partition parallelize. **Probed on the existing 3.14t
free-threaded build (2026-06-17, GIL confirmed off) — the hypothesis fails:**
- 4× concurrent `pd.merge` = **2.09x/4** (~1.9x effective) — better than the
  GIL'd 2.75x/4, but `pd.merge` STILL doesn't parallelize cleanly. It has
  contention *beyond* the GIL (allocator, block/index refcounts, pandas'
  non-thread-tuned paths).
- Free-threaded pandas is currently **2.6x slower single-threaded** (1x merge
  1099ms FT vs 423ms GIL'd 3.11) — the no-GIL refcount/allocator overhead isn't
  optimized away yet in pandas/numpy. So even perfect parallelism couldn't beat
  the GIL'd single-threaded baseline today.
- `DataFrame.iloc`/`concat` are pathologically slow under FT (the partition-
  parallel merge hit 107s) — FT immaturity.
- **BUT** the asymmetry is the real lesson: our shipped **groupby kernel still
  wins 1.9x under FT** (473 vs 914ms) because it's built on Arrow + the nogil
  partition kernel, which parallelize with or without the GIL. **Operations
  built on nogil-friendly primitives scale; `pd.merge` doesn't, GIL or not.**
- **Verdict:** free-threading is NOT the join unlock. To parallelize joins you
  must replace `pd.merge` as the worker with a nogil/native join (lazy_join
  already self-threads; or an Arrow-native join kept Arrow end-to-end) — which
  routes back to Paths A/B/D. Revisit FT later as pandas/numpy reduce its
  single-thread overhead, but it is not a standalone win now.

### Path D — Arrow-native end to end (single conversion at collect)
Keep every operator on Arrow; convert to pandas once, at the final collect.
- **Removes:** tax #2 (and enables more acero routing).
- **Reward:** removes the round-trips; partial. Overlaps Path A/B.
- **Effort:** high — every operator and the eager-order contract must be
  Arrow-faithful. The eager-pandas row-order/index contract is the recurring
  blocker (`collect(order="relaxed")` widens it but not fully).

### Path E — Incremental native kernels (the plumbing-harvest continued)
More single-operator kernels like the groupby one: a parallel count-distinct
(nogil), the string_view upstream-Arrow contribution, etc.
- **Reward:** small, per-operator; does NOT close the whole-pipeline gap (proven
  this session — individual kernels are already ~parity; the gap is the model).
- **Effort:** low–medium each. Good for specific hot ops, not for parity.

## Recommendation

If the goal is **a real shot at parity with bounded (not infinite) effort:**
**Path A (whole-plan pushdown to DataFusion/DuckDB)** is the highest
reward-per-effort — it removes all three taxes by borrowing a finished native
runtime, and the per-operator boundary tax that sank earlier probes disappears
when you push the whole plan at once.

If the goal is **an owned engine that competes on its own terms:** Path B, eyes
open about the multi-person-year cost.

~~If the goal is the best near-term experiment: Path C (free-threading).~~
**Path C was probed and rejected** — free-threading does not unlock joins
(`pd.merge` has sub-GIL contention; FT pandas is 2.6x slower single-threaded
today). So the realistic ranking is now **A (whole-plan pushdown) for parity at
bounded effort**, **B (native engine) for an owned engine**, with C parked
until pandas/numpy shrink their free-threading single-thread overhead.

What none of these are is "a few more kernels." The session proved the residual
gap is the execution model, so the improvement has to be at that layer. The
free-threading probe sharpened *why*: the parts of the pipeline built on
nogil-friendly primitives (Arrow ops, our Cython kernels) already parallelize —
it's the parts that go through pandas' own machinery (`pd.merge`, block
management, the NumPy round-trip) that don't, and removing those is exactly what
a native/arrow-native execution layer (A/B/D) does.
