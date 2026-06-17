# Scope: parallel partitioned hash-aggregate kernel (the substrate lever)

> **STATUS: BUILT + shipped (2026-06-17, commit 97602ceec6).** Results matched
> the scope: isolated kernel **52ms vs Polars 102ms (1.9x)** on q20's shape,
> bit-exact vs the single Arrow group_by. Controlled per-query A/B (toggle
> on/off): **q20 -16%, q17 -5%**, q1/q13/q18 within noise — no regressions.
> All 22 TPC-H validate; 1710 lazy tests pass. `pandas/_libs/lazy_groupby.pyx`
> (`partition_by_key`) + `PhysicalHashAggregate._grouped_arrow_table`. Two
> regressions were caught and fixed during the build: wide post-join inputs
> (narrow to key+value cols before the per-bucket take) and low-cardinality
> overhead (sample-first distinct-ratio gate, before building the full key).
> Note: full-query geo-mean impact is modest — the grouped aggregate is one
> stage among joins/filters in most TPC-H queries — but it is a clean win on
> grouped-aggregate-heavy work and the first lever measured to *beat* Polars.



Feasibility-probed June 2026 (SF-3, q20's `qty` shape: 2.7M filtered rows →
1.6M groups, very high cardinality). This is the **first concrete lever with a
measured chance to beat Polars** on a core grouped-aggregate hot path.

## Why this exists (the decisive measurement)

Arrow's `group_by` is **single-threaded** — it does NOT scale with
`cpu_count` (124ms @1 core → 133ms @8). But Arrow's *serial* hash-aggregate is
**2.4x better than Polars' serial** (Arrow 125ms vs Polars 1-thread 295ms).
Polars wins on this shape (99ms) **purely by parallelism** (295/~3). So:

> Take Arrow's superior serial algorithm and add the parallelism it lacks.

## Measured feasibility (the ceiling is real)

| approach (SF-3, 2.7M→1.6M groups) | ms |
|---|---|
| Arrow `group_by` (single, current) | 132 |
| Polars (8 threads) | 99 |
| **partition + parallel Arrow group_by per bucket** (free partition) | 25–35 |
| **realistic: parallel `take`+group_by per bucket, 8 threads** (free part.) | **43** |
| partition histogram only (cheap) | 16 |
| partition stable permutation via Python argsort (the slow part) | 190 |

Each per-bucket Arrow `group_by` **releases the GIL**, so a ThreadPoolExecutor
parallelizes them. Partition by hash of the (combined) key so **each group
lands wholly in one bucket** → results just concat, no cross-bucket merge.

The only expensive piece is building the partition permutation; Python's
argsort (190ms) kills the naive version. A **nogil counting-sort scatter**
replaces it (~one memory-bound pass over 2.7M rows ≈ 15–25ms).

**Projected kernel:** ~20ms partition + ~43ms parallel group + ~5ms concat ≈
**~65ms vs Polars 99ms (~1.5x), vs Arrow single 132ms (2x).**

## Kernel design

1. **Combine keys** (multi-key): pack into one int64 (`k0*span1 + k1 …`) with
   an overflow guard → fall back to single-arrow path if it doesn't fit. Cheap,
   vectorized.
2. **Partition** (new nogil Cython, `lazy_groupby_partition` or reuse
   `lazy_radix` histogram/scatter blocks): histogram `hash % T`, prefix-sum,
   scatter row indices into a permutation + per-bucket boundary offsets.
   Unstable is fine (group_by is order-insensitive). ~20ms.
3. **Parallel aggregate**: ThreadPoolExecutor(T); per bucket
   `table.take(bucket_idx).group_by(keys).aggregate(specs)`. **Reuses Arrow's
   group_by** as the worker → mean/sum/min/max/count/var all handled natively;
   no aggregation logic to reimplement.
4. **Concat** the T result tables (disjoint groups). Convert to the engine's
   backend (numpy/arrow) as today.

T = 8–16 (≥ core count; T=16 was faster in the free-partition probe). Tune.

## Integration & gating

- New route inside `_execute_grouped_aggregation` (or a sibling physical sink):
  when the groupby is **high-cardinality** + **mergeable/Arrow-supported aggs**
  (NOT nunique/median) + numeric/packable keys, use the parallel partitioned
  path; else keep the single Arrow `group_by` (low-cardinality doesn't benefit
  and would eat partition overhead).
- Gate on the **cardinality estimate** (engine already has one) — only route
  when estimated groups/rows is high enough that single-thread Arrow is the
  bottleneck. Controlled on/off via a module toggle for A/B.
- **Output order**: partitioned+concat yields groups in bucket-then-hash order,
  different from the current single-group_by order. Must match the groupby
  output contract (or only route when the sink is order-insensitive). Full lazy
  suite + all-22 TPC-H validation gates this.

## Value

Helps **every high-cardinality grouped sum/min/max/count/mean**: q20 directly,
plus q1/q5/q9/q10/q13 and similar. First lever measured to *beat* Polars rather
than approach it. Bounded: one Cython partition kernel + one routing decision;
the aggregation itself is delegated to Arrow.

## Risks / open questions

- Partition kernel must hit ~20ms (memory-bound; estimate says yes, but build
  and measure — controlled on/off, never ship unexplained regressions).
- The per-bucket `take` cost is already inside the measured 43ms, so it's
  accounted for; but verify at other scales (SF-1 small, SF-10 large).
- Thread-pool reuse: create one shared pool, not per-call (pool spin-up cost).
- Low-cardinality / few-rows: must fall back; partition overhead would dominate.
- Free-threading / nested parallelism: this sits under the morsel engine's own
  parallelism — ensure no oversubscription (gate so only one level threads).
