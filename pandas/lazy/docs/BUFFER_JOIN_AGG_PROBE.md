# Probe: can a buffer-resident Cython join→agg reach the join half of the gap?

Measurement session, June 2026. Harness:
`benchmarks/probe_buffer_join_agg.py` (standalone, asserts correctness,
SF-1 + SF-3, `use_physical_planner=True` for the live-engine number). Answers
the open question from `PERF_CEILING.md` / `QGAP_DECOMP.md`: **joins are the
biggest TPC-H tax — can the Arrow/NumPy/Cython "harvest" reach them, or does
the Arrow↔NumPy round-trip on the join's large output cap it (the thing that
sank Acero)?**

## Shape

Representative of q3/q5/q9/q10: `lineitem ⋈ orders` on orderkey (every
lineitem matches one order → join output = full lineitem cardinality, the
"large output" regime), then `group_by(o_orderdate).sum(l_extendedprice)`
(2,406 groups). Four paths produce the **same** asserted-equal result.

## Result — PARITY with Polars, and the round-trip is NOT the wall

| path | SF-1 ms | x vs PL | SF-3 ms | x vs PL |
|---|---|---|---|---|
| **Polars** (native lazy join→group) | 86 | 1.00 | 331 | 1.00 |
| **BUFFER_NARROW** (lazy_join idx → `pc.take` 2 cols → arrow group_by) | 97 | **0.88** | 330 | **1.00** |
| MERGE_NARROW (pd.merge {key,date,price} → group) | 152 | 0.57 | 494 | 0.67 |
| MERGE_WIDE (pd.merge full payload → group) | 488 | 0.18 | 1376 | 0.24 |
| **LIVE ENGINE today** (`collect(use_physical_planner=True)`) | **200** | **0.43** | **678** | **0.49** |

BUFFER_NARROW sub-stage decomposition (SF-3, 18M-row output):

| stage | ms | share |
|---|---|---|
| probe (join indices, lazy_join) | 236 | **72%** |
| gather (2-col `pc.take`, on buffers) | 59 | 18% |
| group (arrow group_by) | 33 | 10% |

## Three findings

1. **The buffer-resident harvest REACHES joins.** A narrow buffer-resident
   join→agg using only kernels we already own (`lazy_join` + `pc.take` +
   arrow `group_by`) hits **parity with Polars** (1.00x at SF-3, 0.88x at
   SF-1) on an 18M-row join output. The join half of the gap is capturable.

2. **The "round-trip caps joins" wall is FALSE — when you project first.**
   The gather of the surviving columns is only **18% of the path and scales
   sub-linearly** (59ms on 18M rows). What sank Acero was gathering the
   *wide* output; in a join→**agg** you never need the wide rows — only
   {group key, agg value} survive the group. Project to those *before* the
   gather and the round-trip is cheap. (Carrying the wide payload through the
   join — MERGE_WIDE — is the catastrophe: 0.18–0.24x. Pushdown alone is
   2.8–3.2x.)

3. **The live engine is 2x slower than its own kernels (678 vs 330ms at
   SF-3).** The engine already has `lazy_join` and arrow `group_by` and (for
   this narrow shape) routes to them — but it **materializes the full join
   intermediate between the join and the group operators** instead of
   streaming the join's output straight into the group. That ~100ms (SF-1) /
   ~350ms (SF-3) is pure orchestration tax, removable with no new kernel.

## The lever this surfaces — join→agg fusion (bounded engine plumbing)

The immediate win is **not a new kernel**. It is fusing the join and the
group so the join's gather-indices feed directly into the group, gathering
only the group-key + agg columns, skipping the intermediate materialization:

- 678 → ~330ms at SF-3 on this shape, i.e. **0.49x → 1.00x**.
- Safest possible fusion to build first: the group is **order-insensitive**,
  so this fusion has **none of the eager row-order contract problem** that
  blocks general join fusion (the group destroys order anyway).
- The kernels exist; this is wiring (project-to-surviving-columns + feed join
  indices into the aggregate sink), exactly the "fuse multiple kernels so the
  intermediate never materializes" pattern that `lazy_fused_agg` already
  proved for filter→agg.

The *second* lever (parity → dominance): the join probe is 72% of the buffer
path. It is already nogil + thread-parallel; optimizing it further is what
would push join→agg **past** Polars, the way `lazy_groupby` did for groupby.

## Caveats (honest scope of the GO)

- Validated on the **2-table inner join → single agg** boundary. Real
  q3/q5/q9/q10 are multi-table join **chains** with extra filters; the fusion
  applies per join→agg boundary, but each intermediate join in a chain still
  materializes — generalizing to chains is the build, not yet measured.
- **Inner** join only; left/semi/anti need the same treatment (and those
  are not order-insensitive unless followed by a group).
- Group here is low-cardinality (2,406). High-card group keys (q3 groups by
  orderkey, ~1.5M) would route the group stage to `lazy_groupby` (the shipped
  parallel kernel) — expected to hold, not yet measured on this path.

## Generalization probe (`benchmarks/probe_buffer_join_agg_gen.py`)

Tested the two open caveats above — a multi-table chain and a high-cardinality
group — at SF-1 and SF-3. The answer **splits the GO by group cardinality**.

SF-3 results (median of 5, asserted equal):

| scenario | Polars | Live engine | Buffer path | buffer vs PL | buffer vs live |
|---|---|---|---|---|---|
| **CHAIN** customer⋈orders⋈lineitem → group_by(orderdate).sum (3 tables, 2,406 groups) | 387ms | 1113ms (0.35x) | **436ms** | **0.89x** | **2.55x** |
| **HIGHCARD** lineitem⋈orders → group_by(l_orderkey).sum (4.5M groups) | 575ms | 696ms (0.83x) | 932ms | 0.62x | 0.75x |

(SF-1 agrees: CHAIN buffer 0.89x / 2.18x-over-live; HIGHCARD buffer 0.52x,
*slower* than live.)

**CHAIN — the win HOLDS, and generalizes across join chains.** Buffer-resident
join→agg through a 2-join chain reaches **0.89x parity** and is **2.55x faster
than the live engine**, which materializes *both* intermediate joins. Each
intermediate join gathers only its surviving columns (next join key + final
group/agg cols), so the chain stays narrow end to end. This is the common
analytical shape and the dominant join-heavy TPC-H queries with a low-card
final group live here: **q5 (n_name, 25), q7, q8 (year), q9 (nation,year),
q12 (shipmode)** — several of the worst (q5 0.39x, q8 0.34x, q9 0.39x).

**HIGHCARD — the win does NOT hold; the wall moves to the group.** When the
post-join group is high-cardinality, neither the buffer path (0.62x) nor the
engine's tuned parallel `lazy_groupby` (0.83x) reaches Polars. The bottleneck
is no longer the join or the materialization — it is the **high-cardinality
group substrate** (Arrow-partitioned group_by vs Polars' fused parallel
group), which join→agg fusion does **not** address. The buffer path is
actually *slower* than the live engine here (extra full passes to gather the
wide high-card key + re-partition), so for this regime the engine's existing
parallel kernel is already the better path and fusion adds nothing. The
high-card-group join queries — **q3 (per-order), q10 (per-customer)** — stay
gated on the group substrate, a separate (already-characterized) wall.

## Engine integration (v1, 2026-06-25) — foundation built, reach is the wall

Built the fusion as a physical node `PhysicalFusedJoinAgg` + planner rewrite
(`_fuse_join_aggregates`), toggle `_FUSE_JOIN_AGG`. Key facts learned:

- **Execution is morsel-driven**, so an execute()-level hook on the aggregate
  can't see the join: at a breaker the join runs in its own pipeline and feeds
  the aggregate as a `PrecomputedInput`. The fusion therefore must be a
  **physical-plan rewrite** that replaces `HashAggregate(Materialize(HashJoin))`
  with one fused node *before* pipeline compilation (mirrors the existing
  `PhysicalFusedFilterAgg`). The node gathers only the join-output columns off
  the Cython join indices, then re-runs the original aggregate over the narrow
  input (reusing backend choice, the cardinality gate / parallel kernel, and
  formatting). Order-insensitive, so no row-order contract.

- **Correct + regression-free, but reach is tiny.** All 22 TPC-H validate;
  controlled A/B (SF-1) total 0.96–0.98x with the only firing query **q4**
  (~0.81–0.93x). The win on the simple synthetic shape is ~15% (200→170ms),
  far below the probe's 97ms — the probe pre-converted Arrow inputs outside the
  timed loop and paid no planning/pipeline overhead; the engine's existing
  acero join-routing + projection-pruning already capture most of the simple
  case.

- **Why only q4: the aggregate is almost never directly over a bare join.**
  Measured aggregate-input shapes across all 22:
  - `Materialize > HashJoin` (direct): q4, q13 — v1 fires.
  - `… > Project/FusedPipeline > HashJoin`: q12, q14, q16, q19, q22 (a filter/
    project sits between the join and the group).
  - `… > JoinChain`: q3, q5, q7, q8, q9, q10, q17, q18, q21 — the multi-join
    chains where the probe's big win (CHAIN 0.89x / 2.55x-over-engine) lives.

  So realizing the win needs two extensions, both larger than v1: **(1) peel
  order-insensitive filter/project between the join and the group** (adds the
  five single-join queries), and **(2) fuse `JoinChain`** (the chain queries —
  the actual lever, and where `_CompositeJoin` late-materialization already
  exists to build on).

**Status:** v1 node + rewrite landed but **DEFAULT-OFF** (`_FUSE_JOIN_AGG=False`)
as the validated foundation; the reach extensions (1)+(2) are the next step.

### Chain-fusion attempt (2026-06-25) — measured NET-NEGATIVE, reverted

Extended the fused node to a generic `producer` (join *or* `PhysicalJoinChain`),
adding a `composite_gather_arrow` entry point on the chain (compose, then gather
only the agg-referenced columns as Arrow off the `_CompositeJoin` indices).
Correct (all 22 validate) but the A/B (SF-1) was a **net loss (1.32x)**:

| query | shape | OFF→ON |
|---|---|---|
| q4 | join-direct | 70→71ms (flat) |
| **q18** | **chain-direct, high-card string group** | **202→829ms (4.09x SLOWER)** |
| q21 | chain-direct | 486→444ms (0.91x, marginal) |

Two decisive facts:

1. **The chain-direct cases that fire are the wrong ones.** q18/q21 group by
   high-cardinality string keys — exactly the HIGHCARD regime the generalization
   probe flagged as a loss. q18 regressed **4x**: gather-then-regroup off the
   composite is far worse than the engine's tuned `_CompositeJoin.gather()` +
   group for a wide high-card string output. The static gate (`int join keys`)
   does **not** see the group cardinality/dtype, so it fires anyway.

2. **The intended low-card chain targets don't fire at all.** q5/q7/q8/q9/q10
   put a `Project`/`FusedPipeline` *between* the chain and the aggregate, so the
   aggregate's input is not directly a `JoinChain` — the rewrite never matches
   them. Only q18/q21 (direct `Materialize > JoinChain`) match, and both are
   high-card.

3. **The engine already late-materializes chains.** `PhysicalJoinChain` +
   `_CompositeJoin` already avoid the intermediate-payload gathers (the bulk of
   the chain win the probe attributed to "the engine materializes both joins" —
   that was vs a naive baseline, not the real chain path). The only remaining
   lever (Arrow-gather + skip the breaker) doesn't beat the tuned path and
   regresses high-card.

So a worthwhile chain fusion would need **both** a runtime cardinality gate
(skip high-card → avoid the q18 4x) **and** tail handling (peel the order-
insensitive `Project`/`FusedPipeline` to reach q5/q7/q8/q9), and even then the
per-query win is uncertain because the chain already late-materializes. Reverted
to the default-off join foundation. **Net conclusion: the engine's existing
JoinChain late-materialization already captures the chain win; the extra fusion
layer is not a win on the realized TPC-H shapes.**

## Verdict

**GO on join→agg fusion, scoped to the low-cardinality-final-group regime**
(join chains terminating in a small group — q5/q7/q8/q9/q12). The probe disproves the
round-trip wall for the agg-terminated shape and shows parity is reachable
with kernels we own; the first deliverable is the bounded engine fusion
(no new kernel) that recovers the 2x orchestration tax, then optional
join-probe kernel tuning to exceed Polars. Measurement-first, controlled
on/off, validate full lazy suite + all-22 TPC-H before shipping.
