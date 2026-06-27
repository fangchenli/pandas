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

## BUILD + VALIDATE result (June 2026): components win, composition doesn't yet
Built `partition_by_bucket` (O(N) nogil counting-sort radix partition, in
`_libs/lazy_join.pyx`) + a partitioned parallel join driver (fast unordered core
+ opt-in `pd.merge`-order reorder). **Correctness verified** vs `inner_join_indexers_i8`:
ordered output == pd.merge exactly; unordered == pd.merge multiset.

Measured (Python-orchestrated driver, the honest end-to-end):

| | index-gen | full join + gather (P=9) |
|---|---|---|
| monolithic (current) | 298 ms | — |
| **partitioned UNORDERED B=16** | **211 ms** | 541 ms |
| partitioned ORDERED B=16 | 434 ms | — |
| pd.merge | — | 507 ms |
| polars | ~132 (P=1) | **385 ms** |

Partitioned-unordered index-gen beats monolithic (211 vs 298), but **composed
end-to-end it's only ~pd.merge parity and still behind Polars** (541 vs 385).
The isolated 128 ms per-bucket number did **not** survive composition. Taxes
that ate it: (1) the partition counting-sort is **single-threaded** (~83 ms ×2
sides); (2) **Python orchestration** over 16 bucket tasks + the 16-way
`concatenate`; (3) the gather is still a **separate pass** (not fused into the
probe); (4) order preservation via `argsort` costs **+222 ms** (must become an
O(N) radix reorder).

This is the project's recurring **isolation trap** (cf. the scan-in-place
groupby regression): nogil kernels win standalone, but Python orchestration
*between* them is the tax (PERF_CEILING tax #3). The lesson is precise and
actionable: the win requires a **single fused nogil kernel** — partition +
per-bucket build + probe + **emit gathered payload in the same pass** — driven by
one thread pool over buckets, with an O(N) radix reorder for the ordered mode.
That removes all four taxes at once. Polars's 385 ms is exactly one fused pass;
our 211 ms index-gen + 323 ms gather = 534 ms is two passes plus orchestration.

## FUSED KERNEL BUILT — beats Polars on wide payloads (June 2026)
Built the fused kernel: `join_gather_bucket_rm` (`_libs/lazy_join.pyx`) +
`partitioned_join_gather` driver (`backends/numpy/join.py`). Per partition it
builds a cache-resident hash table on the unique right (dim) keys, probes the
left, and **`memcpy`s the matched right row's payload directly** into a
**row-major** output block — fused, no index array. Two layout attempts:
- **Column-major scalar gather** (first try): LOSES — a scalar per-column copy
  can't match SIMD `np.take` (P=21 1763 ms). Dropped.
- **Row-major `memcpy`-per-match** (one contiguous cache-line copy/row): WINS.
  Row-major gather measured ~2.2x faster than column-major at high P; the
  dim-side transpose to row-major is cheap (small build side, done once).

Validated (`benchmarks/bench_partitioned_join.py`, 10M⋈2.5M unique build,
best-of-5; correctness vs pd.merge exact for ordered, multiset for unordered):

| P | fused unordered | fused ordered | pd.merge | polars |
|---|---|---|---|---|
| 1 | 171 | 277 | 285 | **136** |
| 3 | 220 | 401 | 326 | **193** |
| 9 | **308** | 563 | 531 | 397 |
| 21 | **538** | 963 | 961 | 774 |

**The unordered fast core beats Polars at P≥9 (308 vs 397; 538 vs 774) and
beats pd.merge at every width** — the first kernel in the campaign to beat
Polars on a join. At small P Polars still wins (its index-gen is leaner; our
partition overhead dominates when there's little payload to amortize it).
**Ordered mode loses** — the O(N) position-map reorder copies the whole output
block (a second pass); acceptable as the opt-in correctness fallback, but it
needs a gather-into-sorted-slot kernel to be competitive.

## Integration check — the win does NOT survive the column-major engine (June 2026)
Before wiring, measured the honest round-trip the engine requires: its columns
are **column-major** (`ArrayDict`) and it returns column-major, but the fast
fused gather needs **row-major** in and produces row-major out. Adding the
column→row-major transpose (input) + row-major→column split (output):

| P | block-only (microbench) | + round-trip (engine-real) | pd.merge | polars |
|---|---|---|---|---|
| 3 | 230 | 282 | 328 | 194 |
| 9 | 365 | **573** | 527 | 402 |

The round-trip adds ~208 ms at P=9 — the 365 ms win becomes **573 ms, losing to
Polars (402) and ~tying pd.merge**. Two compounding problems:
1. **Layout round-trip tax.** Row-major gather is fast in isolation, but a
   column-major engine pays a transpose in + split out that ~equals the gather
   saved. This is `PERF_CEILING.md` tax #2 (the layout/boundary) at the kernel
   level — the same isolation trap, now nailed with a *built, measured* kernel.
2. **Mis-targeted side.** The kernel memcpy-gathers the **build (dim)** payload;
   TPC-H's wide float64 payload (measures) is on the **probe (fact)** side, and
   dim payloads are mostly strings/ints → the float64-right-payload + unique-
   build gate would rarely fire on the real suite anyway.

**Decision: do NOT wire it (as-is).** It would be a regression-or-no-op,
violating the campaign rule against shipping unexplained regressions. Wiring +
all-22 was not run because the integration cost is *measured* to negate the win
before execution.

## Consequence
The kernel **beats Polars in isolation** (block in/out, build-side payload) —
proving the *kernel* is not the bottleneck. But the win is **stranded by the
engine's column-major data contract**: every place we speed a join, the
layout/orchestration boundary eats it. This is the definitive, kernel-level
confirmation of the standing conclusion — matching Polars on joins requires the
engine to be **row-major/block-native end-to-end** (carry blocks through joins,
no per-operator column round-trips), i.e. the Polars/DuckDB architecture, which
is the already-NO-GO from-scratch engine (`ENGINE_GONOGO_MEMO.md`). The kernel
is a validated artifact and a clean proof; it is correctly **not shipped**.

Salvage paths (all require more than a kernel): (a) make the engine carry
row-major blocks across the join so input/output transposes vanish; (b) a
probe-side fused gather (the fact transpose + output split costs are larger, not
smaller); (c) keep it for a future block-native execution mode only.
