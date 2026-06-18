# #1 campaign — per-operator decomposition of the worst TPC-H queries

Measurement session (June 2026), SF-1, machine loaded. Harness:
`benchmarks/exp_qgap_decomp.py` + isolated microbenchmarks. Localises where
q20/q21 lose before any build — the discipline that kept us from building
Acero / arrow-across-join, and (this session) a count-distinct kernel we
didn't need.

## METHODOLOGY CORRECTION (read first)

An earlier pass measured the lazy side with the **default `.collect()`**, which
is **eager pandas** (`use_physical_planner=False`). The TPC-H scorecard runs
`.collect(use_physical_planner=True)`. The two modes differ wildly, and the
eager numbers sent the investigation down a wrong path (a "count-distinct
kernel is 4.7x slow" finding that is an artifact). Example, q21 `n_unique`
groupby, 6M rows:

| collect mode | ms |
|---|---|
| default `.collect()` (eager pandas) | ~383 |
| `.collect(use_physical_planner=True)` (scorecard mode) | **~98** |
| Polars | ~87 |

**Under the correct mode the n_unique kernel is at ~1.13x — essentially
parity.** All numbers below are `use_physical_planner=True`.

## Headline: the residual gap is CROSS-OPERATOR FUSION, not any single kernel

Individual kernels are at/near Polars parity once measured in the right mode.
The gap is **materialization between operators** + whole-pipeline parallelism —
exactly the seed doc's candidate #1, and consistent with the standing
"substrate-bound ~0.45x" positioning. There is **no single hot kernel to swap
or build**; the count-distinct kernel idea is withdrawn (already parity).

### q20 (scorecard 0.16x) — filter→groupby is not fused

Decomp: the gap is concentrated in the `qty` stage
(`filter(date range) + group_by(l_partkey,l_suppkey).agg(0.5*sum)`); every
stage after it is cheap (small joins, +5/+6/+11ms). Splitting `qty`:

| piece | lp ms | pl ms |
|---|---|---|
| filter only (proj 3 cols, 6M→909K) | ~21 | — |
| grouped sum, **same data, no filter** | ~194 | ~178 |
| **qty = filter + grouped sum** | ~57 | **~31** |

The grouped-sum kernel itself is parity (194 vs 178). Polars does filter+gb in
**one streaming pass** (~31ms ≈ our filter-alone). We **materialize the
filtered 909K-row intermediate (~21ms) then group it (~36ms)** — the physical
plan shows a `Materialize (reason=aggregate)` breaker between the fused-filter
scan pipeline and the hash-aggregate sink. **Lever: stream filtered morsels
directly into the group-by, eliminating that materialize.**

### q21 (scorecard 0.27x) — whole-pipeline diffuse

Full 472ms vs 215ms (~0.46x this run). Reliable per-stage signals:
n_unique `nsupp` 97 vs 76 (1.27x), `late_nsupp` +28 vs +24 (parity). The
cumulative-prefix join numbers blow up (438→789→1208ms) but are an **artifact**
of the method — collecting a join prefix materializes a huge intermediate the
full query never builds (note the large negative deltas once the downstream
filter/limit re-enter: full query 472ms << the 1208ms prefix). So q21's ~2.2x
is **spread thin across many join+groupby stages, each ~1.3x**, compounding —
Polars fuses the whole pipeline and runs stages in parallel. No single stage
dominates; no kernel to build.

## Corrected campaign conclusion

1. **Count-distinct kernel: not needed.** n_unique is ~parity under the real
   execution mode. The "4.7x" was an eager-`.collect()` artifact. Withdrawn.
2. **Joins: not the gap** (seed hypothesis still disproven; routing is fine).
3. **The real lever is cross-operator fusion**, principally **filter→groupby
   streaming fusion** (q20's clean, concentrated case): remove the
   `Materialize (reason=aggregate)` breaker and feed filtered morsels straight
   into the hash-aggregate sink. This is a bounded engine change (the engine
   already has streaming aggregation infra — `can_preaggregate`,
   `_execute_streaming_aggregation`), targeted at the breaker, not a rewrite.
4. q21's residual is the broad whole-pipeline-fusion/parallelism gap — the
   parked architectural item (Polars/DuckDB-style fused engine). Not a
   single-change win.

## Recommended next step

Investigate **filter→groupby streaming fusion** for q20's shape: why the
`Materialize (reason=aggregate)` breaker sits between the fused-filter scan
pipeline and the group-by sink, and whether filtered morsels can pre-aggregate
into the sink (the `can_preaggregate` path) instead of materializing. Measure
controlled on/off, validate full lazy suite + all-22 TPC-H. Expected: q20
`qty` ~57→~35ms; helps any `filter→group_by(sum/count/min/max)` query.

## Fusion investigation (filter→groupby) — measured SF-3

Probed the q20 `qty` shape (lineitem→filter(date)→group_by(2-key).sum) at SF-3
(2.7M filtered rows → 1.6M groups, ~60% unique = very high cardinality).

**Is there a native Arrow way?** Yes — Acero (`source→filter→hash_aggregate`
streams batches, never fully materialising the filtered intermediate). But
**measured it does NOT help**: numpy→Acero→numpy = 216ms vs our current
materialize-then-group = 202ms (Acero slightly *slower*). Acero's per-node
overhead + FilterNode per-batch compaction eat the streaming benefit —
consistent with our prior Acero/arrow-across-join findings. **Acero is not the
fusion path.**

**Where the 92ms gap (202 ours vs 110 Polars) actually is** — isolated the
grouped-sum on the pre-filtered set:

| path | ms |
|---|---|
| our lazy grouped-sum | 139.8 |
| raw Arrow `group_by` | 132.5 |
| **Polars grouped-sum** | **92.9** |
| eager pandas | 244.0 |

So the gap splits ~50/50:
- **~47ms = group kernel substrate.** We already match Arrow's `group_by`
  (140≈132) — our routing is correct — but **Arrow's hash-aggregate is itself
  ~1.5x slower than Polars'** on high cardinality. Closing this needs a
  Polars-class parallel hash-aggregate kernel (a large build competing with
  Arrow C++). Not a plumbing win.
- **~45ms = filter→group materialization.** We compact the 2.7M filtered rows
  (`Materialize (reason=aggregate)`) then group; Polars fuses, so its filter
  adds only ~17ms on top of its group. **Root cause:** the in-memory scan has
  `supports_streaming=False`, so the streaming pre-aggregate path
  (`_can_stream_aggregation` / `_execute_streaming_aggregation`, which exists
  for file/Concat sources) is skipped and the engine materializes. Recoverable
  only by **engine-side morsel pre-aggregation for the in-memory source** (emit
  filtered morsels that partial-aggregate into the group sink + merge) — Acero
  does not deliver it.

**Net:** even perfect fusion lands q20 `qty` at ~140–157ms (the Arrow group
floor + ~17ms fused filter) vs Polars 110ms — i.e. ~0.16x → ~0.22–0.25x. The
other half stays gated on the Arrow-vs-Polars group-kernel substrate gap. So
q20 is **~half bounded-engine-fusion, ~half substrate** — neither a clean
plumbing win; consistent with "substrate-bound ~0.45x."

### Engine-side streaming fusion — DISPROVEN for this shape (do not build)

The engine already has a streaming aggregation path
(`_execute_streaming_aggregation`) that **partial-aggregates each batch then
merges** — but it's gated on `supports_streaming` (false for the in-memory
source), which is why q20 materializes. Tested whether forcing that path would
recover the ~45ms. It does the **opposite**: on q20's high-cardinality shape
(1.6M groups / 2.7M rows ≈ few rows per group), partial-aggregate + merge =
**245ms vs a single group_by's 134ms — ~2x SLOWER** (bs=64K and bs=500K both).
Each batch's partial barely shrinks the data, so the merge is a second
near-full group_by. **Building engine-side streaming fusion on this path would
regress q20, not improve it.** It only helps *low*-cardinality groupbys.

The only mechanism that captures the fusion half on high cardinality is a
**custom fused mask→hash-aggregate kernel** (consume the filter mask + raw
columns, build the group table in one pass, skipping masked rows — no
compaction). But that competes head-on with Arrow's `group_by` (which we
already match at ~132ms and which is itself 1.5x slower than Polars), so it is
a substrate-class build for a ~45ms / one-query-shape payoff.

**Conclusion: q20 is effectively substrate-bound.** Three fusion mechanisms
tried and rejected by measurement — Acero (no help), streaming partial-merge
(regresses), and the only viable one is a substrate-class kernel. No bounded
engine change captures it. This is the same wall as count-distinct: under the
correct execution mode, the residual TPC-H gap is the Arrow-vs-Polars kernel
substrate, not engine plumbing.

## Next-tier decomposition (q3, q10) — the lever is the JOIN chain

After the grouped-aggregate kernel shipped, decomposed the next gated queries
(SF-3, physical planner). Prefix-collect is misleading for joins (carrying all
columns through the join: q3 raw join-chain collect = 1597ms vs 227ms when
projected to the groupby's columns vs 180ms full) — so measure with projection
or by plan inspection, not raw prefixes.

- **q3 (0.38x):** dominated by the 3-table join chain (`PhysicalJoinChain`,
  M5 order-free composition). Groupby keys are packable but the post-join input
  is small. Full 180ms vs Polars ~110ms — the gap is the join.
- **q10 (0.30x):** 4-table join + a **string-key** groupby (c_name, c_phone,
  n_name, c_address, c_comment) → non-packable, so it does NOT hit the parallel
  groupby kernel (falls back to single Arrow group_by). Gap = join chain +
  string groupby.

Join routing today: wide-payload joins use **single-threaded `pd.merge`**;
`lazy_join`'s CSR hash self-threads only narrow inner joins (n_gather<=4, the
shipped fix); acero is 2-7x slower (Arrow round-trip). So wide joins are
single-threaded — the same shape of gap the groupby kernel just closed.

**Candidate next levers (measure before building, as always):**
1. **Parallel partitioned join** — apply the just-proven partition-parallel
   trick to wide `pd.merge` joins (partition both sides by key hash so matching
   keys co-locate, join per partition in parallel, concat). Gates q3/q5/q9/q10.
   Bigger; overlaps prior parked join work, but the partition kernel is a fresh
   angle that worked for groupby.
2. **String-key support in the groupby kernel** — factorize string keys to pack
   them, letting q10's groupby fire the parallel path. Bounded extension.
3. **Fuse the filter mask into the partition scatter** — attack q20's remaining
   ~45ms materialize half; needs a planner change (pass mask to the aggregate
   instead of materializing the filter first).

## Constraints

- Measurement-first; **measure in the mode the scorecard uses**
  (`use_physical_planner=True`); controlled on/off; never ship unexplained
  regressions; validate full lazy suite + all-22 TPC-H vs DuckDB.
