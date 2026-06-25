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
3. `bucket_hash_sum(perm, lo, hi, keyhash, values)` — per-bucket open-addressing
   hash-aggregate (run per bucket on the ThreadPool), returns each group's
   representative row index + sum. The driver gathers the actual key columns at
   the representative rows once at the end (one narrow take, not per bucket —
   avoids the string-gather tax that sinks the partition+Arrow-group_by route).

## Status: VALIDATED PROTOTYPE — not yet production / not yet integrated

What works: correct (bit-identical group sums vs Arrow), reaches Polars parity
standalone. Kept in `lazy_groupby.pyx` as the validated foundation; **not wired
into the engine routing**.

Productionization TODO before integration:
- **Exact key verification.** `bucket_hash_sum` currently groups by the 64-bit
  `keyhash` directly (no key-equality check). Collision probability at these
  scales is ~1e-9/query — fine for this probe, NOT acceptable for an engine that
  must pass exact TPC-H validation reliably. Fix: 128-bit hash key (one extra
  multiply/byte, collision ~1e-19) or true key comparison against the rep row.
- **Multi-aggregate** support (count/mean/min/max + multiple aggs), nulls,
  decimal/temporal keys (q10's `c_acctbal` is decimal — probed as float64 bits).
- **Engine routing**: extend `_grouped_arrow_table` / `_partition_key_arrays` to
  route high-card string-key groups here behind a gate, matching output schema
  (incl. `preserve_index`); validate all-22 TPC-H + the 1710 lazy tests.

## Honest end-to-end caveat (decides whether to productionize)

The kernel reaches parity on the **group operator**, but the group is only ~1/5
of the queries that have high-card string groups. q10 is ~365ms (vs Polars-mem
~105ms); the group is ~30–60ms of that, joins + gather dominate. So a fully
integrated kernel moves q10 from ~0.28x to ~0.31x, and helps mainly q10/q18.
It IS a clean substrate win (parity on a hot operator, upstreamable as the AG4
string-hash gap), but its whole-query impact is modest — consistent with the
reframing finding that the 3x is distributed across many operators.
