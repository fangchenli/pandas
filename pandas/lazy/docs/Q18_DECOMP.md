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
  `group_by_threaded_slice`/`_iter`). Verified against source: it partitions by
  `hash_to_partition` and **each thread scans all rows but inserts only its
  partition's keys into a thread-local (cache-hot) hashmap — no gather/scatter
  pass** (source comment: *"the hash-table currently is local to the thread so
  in hot cache"*). Our `partition_by_key` → per-bucket `take` (gathers 18M rows)
  → Arrow group_by → concat a 4.5M-row intermediate pays a scatter+concat that
  Polars's scan-in-place avoids — that extra work is the measured mechanism of
  the gap.
  *Provenance:* the ~100 ms split is inference from our own measured take+concat
  overhead plus reading Polars source; we have **not** profiled Polars's
  internals on this query. (Note: the cache-aware **hot/cold** grouper with
  4096-entry eviction is Polars's *streaming* engine, `polars-stream`, which
  `.collect()` does NOT use — do not attribute it to this measurement.)
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

## Conclusion

q18 is **substrate-bound**, consistent with `ENGINE_GAP_REFRAMING.md`: the gap
is distributed, and the high-cardinality group-by is ~1.5x Polars because of the
parallel hash-aggregate substrate, not a missing kernel. The only levers are
non-bounded: (a) a scan-in-place partitioned threaded group-by (per-thread
cache-local hashmap, no scatter pass) matching Polars's in-memory
`group_by_threaded_*` — which would require replacing the partition+take+arrow
path, since Arrow's group_by gives us no scan-in-place primitive; or (b)
optimizer scan-sharing for the double lineitem scan (small). Neither is a clean
kernel swap; no bounded win was found.
