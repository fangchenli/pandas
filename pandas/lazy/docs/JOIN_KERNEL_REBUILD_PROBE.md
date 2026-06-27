# Joins DO yield to kernels — measured rebuild targets (June 2026)

Reopens the "joins don't yield / matching Polars on joins is architectural"
conclusion (`PERF_CEILING.md`, `MATERIALIZATION_EXPERIMENT.md` III). Driven by:
*there is no reason we can't write better parallel/SIMD kernels; find the
bottleneck, beat it.* Done — the bottleneck is precisely located and every
component is measured-beatable with hand-written parallel kernels.

## Decomposition of a representative TPC-H join
Shape: 10M probe (fact) ⋈ 2.5M unique build (dim), ~1:1 hit (the q3/q5/q9/q10
fact→dim pattern). Repo env, pyarrow path, 8 cores. `benchmarks/` scratch
profiles (`join_gather_profile.py`, `index_decomp.py`, `parallel_gather_test.py`,
`bucket_join_isolated.py`).

| stage | current | what it is |
|---|---|---|
| index-gen TOTAL | **346 ms** | monolithic CSR kernel |
| — build (single-thread) | 79 ms | hash table on 2.5M keys, **not parallel** |
| — probe_count (threaded) | 93 ms | a full pass just to **size** the output |
| — probe_fill + alloc | ~161 ms | a **second** full probe pass |
| gather P=9 (per-col `take`) | 903 ms | our wide-payload path → why we bail to pd.merge |

vs Polars whole join (build+probe+gather): **P=1 132 ms, P=9 392 ms**. Our
index-gen *alone* (346 ms) is 2.4x Polars's entire join — index-gen is the
dominant cost, not the kernels' algorithm.

## Both components beaten by parallel kernels (measured)

**1. Gather — row-chunk parallel `take` (ThreadPoolExecutor, np.take with `out=`,
zero Cython):**

| | P=1 | P=3 | P=9 | P=21 |
|---|---|---|---|---|
| per-col take (current) | 93 | 296 | 903 | 2124 |
| **row-chunk parallel** | **32** | **102** | **323** | **850** |
| pd.merge (block take) | 287 | 327 | 531 | 968 |
| polars (whole join) | 132 | 188 | 392 | 771 |

Row-chunk parallel gather **beats pd.merge at every width and Polars's whole-join
time through P=9**. Directly removes the reason the engine bails to pd.merge on
wide payloads.

**2. Index-gen — partitioned parallel hash join:** with partitioning pre-done,
parallel per-bucket index-gen is **128 ms (B=16) vs 346 ms monolithic = 2.7x**,
matching Polars's whole-join time. The monolithic single-thread build + 2-pass
probe is the loss; partitioning parallelizes the build and keeps hash tables
cache-resident. (Naive prototype with `argsort` partitioning was slower —
partitioning MUST be O(N) radix/counting, which we already have in
`lazy_radix.pyx`; the isolated bucket-join number above excludes the partition
cost, which radix adds back at ~tens of ms, parallelizable.)

## The build plan (all pieces measured-feasible)
A **fused, radix-partitioned, parallel hash-join kernel**:
1. Radix-partition both sides by key-hash into B≈16 buckets (`lazy_radix`, O(N),
   parallel) — fixes the single-thread build and the 2-pass probe.
2. Per bucket in parallel: build small (cache-resident) table, probe, and
   **emit gathered payload directly** (fused probe+gather, one pass) — kills the
   separate gather pass that even our fast parallel gather still pays.
3. This addresses all measured costs at once; target ≤ Polars at every width.

## The one design fork that gates it: row order
Our current kernel guarantees **exact `pd.merge` inner row order** (probe left in
row order). Partitioning emits in **bucket order**, breaking that contract.
Options:
- (A) Restore pd.merge order with a stable reorder of the output by left-row
  index (a radix sort of the ~10M output, parallelizable, ~tens of ms) — keeps
  the exact eager-pandas contract, costs a pass.
- (B) Define the lazy engine's join as **order-not-guaranteed** (like Polars/SQL)
  and skip the reorder — fastest, but a semantic change to document + may shift
  results of order-sensitive tests/queries.

This is the decision before building; everything else is measured-ready.

## Consequence
Reverses "joins are architectural." A large part of the join gap is **kernel
work we can do** (parallel build, single-pass probe, fused gather), not a
whole-query-fusion rewrite. The remaining genuinely-architectural piece (Polars
pipelining the probe stream across a join cascade) is separate and smaller than
the per-join kernel wins available here.
