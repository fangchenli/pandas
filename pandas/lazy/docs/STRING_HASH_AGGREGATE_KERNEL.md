# Parallel string-hash aggregate kernel — ceiling validated (June 2026)

Closes the one gap left open by `ENGINE_GAP_REFRAMING.md`: high-cardinality
**string-key** group-by, where the engine falls back to single-threaded Acero
(~57ms on q10's group) and Polars parallel-hashes it (~30ms). Arrow's
`group_by` requests threads but does not split a single in-memory Table into
morsels, so it stays serial; dict-encoding the keys doesn't help at high
cardinality (encoding near-unique strings costs as much as the group). The only
way to parity is a genuine **parallel hash-aggregate over the raw Arrow string
buffers** — no encode, no per-bucket gather.

## Result — Polars parity (q10 group input, SF-3, 8 threads)

344,528 rows → 114,218 groups, 7 keys (5 strings incl. long `c_comment`), sum:

| path | time |
|------|------|
| arrow single `group_by` (current fallback) | ~57 ms |
| kernel, **serial** hashing | ~54 ms |
| kernel, **parallel** hashing | **~30 ms** |
| Polars | ~30 ms |

Stage breakdown (serial-hash kernel): **hash 37.6ms**, partition 1.8ms,
bucket-aggregate 2.7ms (parallel), gather-keys-at-reps 12.0ms. **The entire win
is parallelizing the string hashing**; the bucket hash-aggregate itself is
~free. Reproduce: `benchmarks/probe_strhash_kernel.py`.

## Design (does not copy Polars/DataFusion; reuses our nogil+ThreadPool pattern)

Three nogil kernels in `_libs/lazy_groupby.pyx`, driven from Python (same
GIL-released-kernel + `ThreadPoolExecutor` pattern as the existing
`partition_by_key` path):

1. `hash_string_col(offsets, data, acc, first, lo, hi)` /
   `hash_int_col(vals, acc, first, lo, hi)` — fold each key column's per-row
   hash (FNV-1a over the string bytes; multiplicative mix for ints) into a
   combined `acc[lo:hi]`. The `[lo,hi)` range lets the driver hash disjoint row
   ranges on separate threads (this is the parallelism that buys the win).
2. `partition_by_key(acc, n_buckets)` — existing counting-sort partition so each
   group lands wholly in one bucket.
3. `bucket_factorize(perm, lo, hi, h1, h2, codes)` — per-bucket open-addressing
   factorize keyed by the **128-bit** hash `(h1, h2)` (run per bucket on the
   ThreadPool). Writes a dense local code per row and returns each group's
   representative row; the driver offsets local codes by the bucket base and
   concatenates reps. The actual key columns are then gathered at the rep rows
   once at the end (one narrow take, not per bucket — avoids the string-gather
   tax that sinks the partition+Arrow-group_by route).

The aggregation itself is **not** in the kernel: the driver builds a
`[code] + value cols` table and runs the existing `arrow_group_by` on the cheap
dense int codes (covers every agg func, null semantics, dict decode), then
replaces the code column with the real keys gathered at the rep rows. This keeps
the new kernel surface tiny (factorize only) and reuses battle-tested
aggregation.

## Status: PRODUCTIONIZED + INTEGRATED (default-on)

Engine integration (`physical.py`): `_grouped_arrow_table` routes string-keyed
groups to `_string_hash_grouped_table`, which classifies keys
(int/temporal/bool/float-bits + string), cardinality-gates on a cheap strided
sample (so low-card string groups like q1 don't pay), factorizes in parallel,
aggregates on the codes, and attaches keys at the rep rows. Module toggle
`_STRING_HASH_GROUPBY` (default True) for A/B.

Exactness: groups by a **128-bit** key hash (two independent FNV-1a/multiplicative
hashes computed in one byte pass). Collision ~n²/2^128 — below the engine's
existing float-sum nondeterminism. Float keys are bit-viewed to match Arrow's
group_by (which treats -0.0/+0.0 as distinct); float keys containing NaN fall
back (Arrow/pandas NaN-group semantics differ). Decimal keys fall back (a future
enhancement — q18's `o_totalprice` is decimal, so q18 doesn't yet fire).

Validation: all-22 TPC-H validate vs DuckDB at SF-3 with the path on; 1718 lazy
tests pass (incl. `TestStringHashGroupBy`). Warm A/B (min of 15): q10 −17%
(362→300 ms), no regressions (q1 −5%, others neutral; the earlier apparent q1
regression was cold-start + machine noise).

## Honest end-to-end caveat

The kernel reaches parity on the **group operator**, but the group is only ~1/5
of the queries that have high-card string groups; joins + gather dominate. So
the realized whole-query win is q10 ≈ −17% (0.28x→~0.37x), and it currently
helps mainly q10 (q18 blocked on decimal-key support). It IS a clean substrate
win (parity on a hot operator, upstreamable as the AG4 string-hash gap), but its
suite-wide impact is modest — consistent with the reframing finding that the 3x
is distributed across many operators.

## Remaining TODO (not blocking)
- Decimal128 key support (would let q18 fire).
- A fused per-bucket sum path (the prototype `bucket_hash_sum` was ~16ms faster
  than factorize+arrow-agg on the sum-only shape) if the in-engine overhead
  (arrow-agg-on-codes + attach) proves worth trimming.
