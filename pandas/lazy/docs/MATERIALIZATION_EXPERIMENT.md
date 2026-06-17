# Where the Polars gap comes from: a controlled substrate experiment (June 2026)

Driven by the question "can we compile kernels / improve Arrow to kill the
materialization-between-kernels cost?" This experiment decomposes the gap on
the **worst non-limit benchmark categories** (`filter_project` family) by
running the *same logical shape* through every execution substrate with
thread count controlled, so each contributing factor is isolated rather than
inferred.

Harness: [`benchmarks/exp_kernel_fusion.py`](../benchmarks/exp_kernel_fusion.py)
(subprocess-per-config so `POLARS_MAX_THREADS` / Arrow CPU count are set
before import; best of 7 after 2 warmups). Raw numbers:
[`benchmarks/exp_kernel_fusion_results.json`](../benchmarks/exp_kernel_fusion_results.json).

## The matrix (10M rows, ms, best of 7, NCPU=8, Apple M-series)

| op | raw_pc | acero_1 | acero_8 | fused_1 | fused_8 | lp (engine) | polars_1 | polars_8 |
|---|---|---|---|---|---|---|---|---|
| `proj` (a+b)            |  9.1 |  5.4 |  6.2 |  –   |  –   | 18.2 |  9.3 |  9.5 |
| `filt_sum` (sum a, a>0) | 31.8 | 80.6 | 18.3 | 35.4 | **7.1** | 43.2 | 8.3 | 8.3 |
| `filt6` (count, 3 preds)|  5.6 | 70.0 | 16.3 |  5.9 | **4.3** | 29.5 | 16.2 | 12.7 |

Columns: `raw_pc` = whole-array `pyarrow.compute` chain; `acero_{1,8}` =
Acero declaration, same kernels, morsel-streamed; `fused_{1,8}` = our Cython
`fused_filter_aggs` kernel (Python-threaded over row slices); `lp` = our
actual `LazyDataFrame.collect(use_physical_planner=True)`; `polars_{1,8}` =
Polars lazy with that thread cap.

## Finding 1 — Acero (morsel streaming) is NOT the free win. DISPROVEN.

The starting hypothesis was that routing our expression chains through Acero
would recover most of the gap for free (a standalone `filter→multiply→sum`
microbench showed Acero 44ms vs whole-array pc 136ms, 3x). **The controlled
matrix kills that for the shapes that matter:**

- Acero helps **map-only projection** (`proj` 10M: 5.4ms, beating even
  Polars' 9.3ms) — there it streams cache-resident morsels with no
  compaction.
- Acero is **catastrophic for filter→reduce**: `filt6` 70ms (1 thread) /
  16ms (8) vs raw_pc's 5.6ms and the fused kernel's 4.3ms. Acero's
  `FilterNode` **physically compacts** the selected rows (a take/gather)
  per morsel before the downstream aggregate — pure waste when the
  aggregate only needs a mask.

The microbench misled because its chain was map-heavy; the real worst
categories are filter→reduce, where compaction is the dominant cost.

## Finding 2 — Polars' power, measured, in three parts

1. **Single-pass fused filter+aggregate at the memory-bandwidth floor.**
   Polars does `filter→sum` in **8.3ms single-threaded — and 8 threads give
   the same 8.3ms.** 80MB of float64 ÷ ~10 GB/s ≈ 8ms: it reads the column
   *once*, mask-and-accumulating in place, never materializing the filtered
   intermediate. It is already at the hardware floor on one core.
2. **No compaction before reduction.** This is the structural source. Every
   path that physically filters first (`raw_pc`, Acero `FilterNode`, our own
   generic `PhysicalFusedPipeline`) pays a take/gather; Polars and our fused
   kernel skip it (mask + accumulate).
3. **Low fixed overhead + per-core tightness.** `proj` 1M: Polars 0.4ms vs
   our engine 2.1ms — framework cost, not kernel cost.

**The substrate can do all of this.** Our hand-written `fused_filter_aggs`
hits the same floor (filt_sum 7.1ms ≈ Polars 8.3ms) and *beats* Polars on
the 3-predicate count (4.3 vs 12.7). This is consistent with the q6 result
(first outright TPC-H win). The substrate (NumPy/Arrow buffers + a tight C
loop) is not the binding constraint for these shapes.

## Finding 3 — our delivered gap is engine plumbing, not substrate

Tracing the `lp` column against our own kernel (10M):

| query as written | physical node selected | time | why |
|---|---|---|---|
| `filter().select(col.sum())`  | `PhysicalFusedPipeline` (compacts) | 42–49ms | scalar-agg `select` never routes to the fused kernel |
| `filter().sum()` (wide frame) | `PhysicalFusedFilterAgg` (kernel)  | 52.9ms  | kernel runs, but processes **all 3 columns** |
| `filter().select("a").sum()`  | `PhysicalFusedFilterAgg` (kernel)  | **10.3ms** | kernel + column-narrow → **beats Polars' 8.3ms one-core, threaded correctly** |

Two corrections to earlier hypotheses fell out of this:
- **Threading is NOT broken on the fused path** (hypothesis disproven): the
  narrow case is 10.3ms ≈ `fused_8` 7.1ms, clearly multi-threaded. The
  52.9ms wide case was column-width, not threading.
- **Plan/optimize is NOT the overhead** (measured): plan+build is 0.3ms; the
  cost is entirely in the execution path chosen.

So the worst `filter_project` categories are gated by two **local** engine
issues, both with a kernel we already own that matches/beats Polars:
1. **Routing**: `filter → select(scalar-agg)` lowers to the compacting
   `PhysicalFusedPipeline` instead of `PhysicalFusedFilterAgg` (the path the
   frame-level `.sum()` already takes). This is the "select(agg-exprs)
   envelope" increment noted in the fused-kernel track.
2. **Column width**: a scalar aggregate processes unreferenced columns;
   pruning into the fused-agg spec took 52.9ms → 10.3ms.

## Finding 4 — implications for "compile kernels in Arrow"

This reorders the priorities honestly:

- **Arrow expression codegen (Gandiva-revival / new JIT) is not the
  ceiling-breaker here.** The fused mask+accumulate kernel already
  matches/beats Polars by hand. Upstream codegen would buy *generality*
  (not hand-writing each filter+agg shape), a coverage/maintenance win — not
  a perf-ceiling win. Worth proposing eventually; not where the gap is. Note
  also that Gandiva is moribund and not even built into shipped pyarrow
  (`import pyarrow.gandiva` → `ModuleNotFoundError` on 23.0.1).
- **The recoverable wins at 1–100M scale are in our engine plumbing**
  (routing + column pruning + avoiding compaction), independent of any
  substrate ceiling. This **qualifies the "substrate-bound at 0.45x"
  conclusion**: that holds at TPC-H SF-3+ where data is large and fixed /
  routing costs amortize, but at small-medium scale we leave ~5x on the
  table in our own execution path.

## Action items (ranked, all local, no upstream dependency)

1. **Route `filter → select(scalar-agg)` to `PhysicalFusedFilterAgg`** —
   same path `.sum()`/`.mean()`/`.count()` frame-level aggs already take.
   **DONE** (`physical.py` `_fuse_filter_aggregates` Shape B). The pass was
   gated on `PhysicalHashAggregate`; an ungrouped `select(col.agg())` lowers
   to a reducing project at the tail of the fused pipeline instead, so it was
   missed and fell to the compacting generic pipeline. The terminal
   aggregate-project is now peeled off and fed through the same translator
   with a no-group shim.
2. **Prune columns into the fused-agg spec** — subsumed by (1): the kernel
   only touches columns named in its predicates/aggs, so a routed
   `select(sum)` lands at the narrow-column time (10.3ms) automatically.
3. (Generic pipeline) when a fused pipeline's terminal op is a reduction,
   avoid compacting upstream — carry the mask. Not needed once (1) routes the
   scalar-agg shapes off that pipeline; left as a note for any future
   non-agg-terminal reduction shape.

**Measured outcome** (controlled on/off, 10M, fix off → on):
`filter().select(sum)` 43.6 → 10.2ms (0.21x → 0.82x);
`filter().select(count)` 32.2 → 4.0ms (0.29x → **2.36x, beats Polars**);
`filter().select(sum(v1*v2))` 51.9 → 12.2ms (0.32x → **1.34x, beats
Polars**). Now tracked as three `filter_project` rows in `bench_vs_polars.py`
and guarded by `tests/lazy/test_fused_agg.py::TestScalarAggSelectFusion`. At
1M the ~1.5ms framework fixed overhead caps the ratio (finding #3, separate).

---

# Part II — the join gap (0.30x), same disease (June 2026)

Follow-on decomposition of the largest single category loss, the inner join
(`inner_join(10M x 1M)` 0.30x — and the dominant cost in TPC-H). Harness:
[`benchmarks/exp_join_decomp.py`](../benchmarks/exp_join_decomp.py); raw
numbers `benchmarks/exp_join_decomp_results.json`. Bench shape is *exploding*:
left=10M, right=1M, 100k keys → ~100M output rows.

## The matrix (inner join, 10M exploding, ms, best of 5, NCPU=8)

| pd_merge | cyth_1 | cyth_8 | acero_1 | acero_8 | lp (engine) | polars_1 | polars_8 |
|---|---|---|---|---|---|---|---|
| 1162 | 1157 | **598** | 1000 | **229** | 1303 | 1299 | 760 |

## Findings

1. **Our engine bails to single-threaded `pd.merge` on every large join.**
   `_try_cython_join` had a blanket `min(sides) > 500_000 → return None`
   build bail, so all large joins (the costly ones) used `pd.merge` (1162ms,
   single-threaded) — that *is* the 0.30x.
2. **Acero's hash join is fastest in isolation (229ms, 3.3x over Polars) but
   the win dies at the materialization boundary.** Returning to NumPy
   (`to_numpy()` on the ~100M-row output) costs ~435ms: acero+round-trip =
   1044ms ≈ pd.merge 1128ms (1.08x). With arrow output kept, 608ms (1.85x).
   This *confirms* the earlier H2O conclusion — acero only wins if the
   pipeline stays Arrow across the join. Same lesson as Part I: the cost is
   the boundary, not the kernel.
3. **Our own threaded CSR kernel stays in NumPy and beats both** pd.merge
   (1.9x) and Polars (1.27x) on the exploding shape — it was just switched
   off by the cap.
4. **But it loses on wide payloads.** Payload-width sweep (10M exploding):
   3 gathered cols 2.33x, 9 cols 1.23x, 21 cols ~break-even — the per-column
   gather of the exploded output overtakes pd.merge's consolidated block
   take. A tight interleaved A/B confirmed TPC-H q7/q21 (lineitem joins,
   moderate payload, high hit) regress ~2-6% if routed onto the kernel.

## Fix landed (conservative, regression-free)

Replace the blanket 500k size bail with payload-width gating in
`_try_cython_join`: relax the bail **only** for very narrow joins
(`n_gather <= 4`), where the kernel wins by a large unambiguous margin;
everything wider keeps the *exact* original gate (500k bail + 0.5
selectivity). Measured: narrow large inner join (the scorecard's
`inner_join` shape, n_gather=3) `pd.merge 1251ms → kernel 1017ms` (1.23x,
min-time), exact vs `pd.merge`; TPC-H q7/q21 preserved within noise (interleaved
A/B); all 22 validate; guarded by
`tests/lazy/test_cython_join.py::TestPayloadAwareGate`.

The bigger lever — large joins that aren't narrow — needs the engine to stay
Arrow across the join (so acero's 1.85x lands without the round-trip), which
is an architectural change, not a cap tweak. Recorded as the next join
target.
