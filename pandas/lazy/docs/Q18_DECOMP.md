# q18 decomposition — the blocker is the ultra-high-card int group-by (June 2026)

Per-operator decomposition of TPC-H q18 (lazy ~678 ms vs Polars-mem ~281 ms,
0.42x at SF-3). Investigated after the string-hash kernel landed, to check
whether q18 had a catchable lever (it had been wrongly tagged "blocked on
decimal-key support" — see `STRING_HASH_AGGREGATE_KERNEL.md`; q18's string group
is only ~1,253 rows and never fires the string path).

## Measured breakdown (in-context wall-clock, SF-3)

| stage | our time | share |
|-------|----------|-------|
| **inner `l_orderkey` group-by** (18M rows → 4.5M groups, for `HAVING sum>300`) | **~306 ms** | **45%** |
| 3 joins (customer ⋈ orders ⋈ big ⋈ lineitem) | ~150 ms | ~22% |
| `lineitem` scanned twice (column re-extraction) + other scans | ~150 ms | ~22% |
| outer 5-key group (only ~1,253 rows) | ~1 ms | ~0% |
| sort + limit 100 | small | |

q18's shape (`benchmarks/bench_tpch.py:lp_q18`): a `big` subquery groups **all**
lineitem by `l_orderkey`, sums quantity, keeps `sum>300`; the main query joins
customer/orders/`big`/lineitem and groups the survivors by 5 keys.

**The dominant blocker is the inner ultra-high-cardinality int group-by** —
aggregating all 18M lineitem rows into 4.5M per-order sums to evaluate the
`HAVING`. This is fundamental work (you cannot know which orders exceed 300
without summing every one); Polars does the same in ~210 ms.

## No clean bounded lever (probed, measured)

- **Factorize+bincount — equivalent, no win.** The int analog of the new
  string-hash factorize kernel (`bench` in `/tmp`, parallel `hash_int_col` →
  `partition_by_key` → `bucket_factorize` → `np.bincount`) measured **531 ms vs
  the current parallel path's 537 ms** on identical isolated data — no
  improvement. We are already at our substrate's limit for ultra-high-card int
  groups. (Both ~530 ms isolated; Polars ~231 ms.)
- The ~100 ms in-context residual vs Polars is the **partitioned threaded
  group-by** in Polars's *in-memory* engine (what bare `.collect()` uses —
  `crates/polars-core/src/frame/group_by/hashing.rs`,
  `group_by_threaded_slice`/`_iter`): partitions by `hash_to_partition`, each
  thread scans all rows but inserts only its partition's keys into a thread-local
  hashmap (source comment: *"the hash-table currently is local to the thread so
  in hot cache"*). Our `partition_by_key` → per-bucket `take` (**physically
  gathers 18M rows**) → Arrow group_by → concat a 4.5M-row intermediate pays a
  scatter+gather+concat that Polars avoids.

  **PROFILED, not inferred (macOS `sample` on the unstripped polars 1.37.1
  runtime, tight loop over the synthesized 18M→4.5M group at SF-3 scale; self-
  time by symbol):**

  | Polars frame | self samples | what it is |
  |---|---|---|
  | `group_by_threaded_slice` | **32,922** | per-thread hashmap building per-group **index** vectors (GroupsIdx/UnitVec) |
  | `agg_sum` (Float64) | **~25,000** | the sum, read *through* the group indices |
  | `__psynch_cvwait` | 17,260 | idle / thread-sync (not work) |
  | `hashbrown`/foldhash | 2,499 | key hashing |
  | jemalloc (`_rjem_*`) | ~5,000 | allocation |
  | `gather` (take `PrimitiveArray`) | **1,024** | **negligible (~1.5% of real work)** |

  This is direct evidence for the mechanism: Polars builds per-group **index**
  lists in parallel thread-local hashmaps and sums through them; it does **not**
  physically gather the data (gather is 1,024 samples vs ~58,000 for build+sum).
  No `HotGrouper`/`PreAgg`/`SpillFrame` frames appear — confirming the cache-aware
  **hot/cold** grouper (4096-entry eviction) is the *streaming* engine
  (`polars-stream`) and is NOT used by `.collect()`; do not attribute it here.
  *Caveat:* profiled a numpy-synthesized reproduction of the inner group (same
  cardinality), not the literal q18 pipeline, and the sample counts are relative.
- **Inner→outer redundancy is NOT costly.** The per-order sum is computed twice
  (inner `big` for the HAVING, outer for output), but the outer group is only
  ~1,253 rows, so reusing it would save ~nothing.
- **Double `lineitem` scan** (inner group + final join) is column re-extraction
  from an in-memory frame, not I/O — secondary to the group. Optimizer-level
  scan-sharing is a bounded but small lever (~tens of ms).

### Measurement caveat (recorded so it isn't re-tripped)
Isolated `_grouped_arrow_table` on a fresh Arrow table read the inner group at
~537 ms; **in-context it is ~306 ms** (the scan feeds NumPy-backed columns, a
more favorable layout). Trust the in-context wall-clock, not the isolated call —
the isolation over-stated the cost ~1.7x and inverted an on/off A/B.

## Scan-in-place kernel — replicated Polars, but a net regression integrated

Built a Polars-style scan-in-place fused aggregate to replicate
`group_by_threaded_*` (`lazy_groupby.scan_partition_agg` + `compute_hash_part`;
`physical._scan_in_place_grouped_table`): hash the key, then each thread scans
all rows SEQUENTIALLY and accumulates only its hash-partition into a thread-local
open-addressing table — no permutation, no data gather. Keyed by 128-bit hash
(any key type), sum/count/mean over clean (null/NaN-free) value columns; reps
recover the actual keys.

**In ISOLATION it works and matches Polars:** on q18's inner group (18M→4.5M)
the lean prototype hit ~252 ms vs our partition+take path ~558 ms (2.2x) and
Polars ~220 ms — confirming scan-in-place is *the* mechanism.

**Integrated it REGRESSES and was set default-off** (`_SCAN_IN_PLACE_GROUPBY`):
clean warm A/B (min of 13) showed q18 **+22%**, q1 +38%, q17 **+92%**, q20 +18%.
Why the isolated win evaporated: (1) in-context the current arrow path is already
~306 ms (numpy-backed input), not the 558 ms the isolation implied — so the
baseline it must beat is ~2x better than the probe suggested; (2) the
production kernel is heavier than the lean probe (128-bit key + multi-agg +
rep-tracking state) and the Python-orchestrated multi-step path (full
hash → value extract → scan → assemble → key gather) adds overhead that exceeds
the benefit. Estimate-based table sizing (avoiding 8x over-zeroing) did not
rescue it. All-22 validate and 1725 lazy tests pass with it forced on; it's kept
as a correct, validated foundation, not shipped on.

## Conclusion

q18 is **substrate-bound on CPU**, consistent with `ENGINE_GAP_REFRAMING.md`: the
gap is distributed, and the high-cardinality group-by is ~1.5x Polars because of
the parallel hash-aggregate substrate. **Measured corollary (June 2026,
`GPU_DEVICE_RESIDENT_PROBE.md`): the inner group-by is a *CPU*-substrate wall,
not fundamental** — on GPU it runs 2x faster than 128-core Polars (SF-30, growing
with scale) and 7.8x a single-thread CPU device-resident; but ~half the per-call
GPU cost is H2D transfer, so capturing it needs the device-resident execution
model (a different engine), not a kernel in the morsel loop. We can *replicate* Polars's scan-in-place
algorithm and match it in isolation, but **not capture it through our
Python-orchestrated engine** — the in-context arrow path is already close, and
the orchestration overhead negates the kernel win. Confirms (now with a built,
measured attempt) that closing this needs the execution model, not a kernel; the
double-lineitem-scan optimizer lever remains the only small bounded option.
