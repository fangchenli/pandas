# Engine-gap reframing — the Polars gap is NOT the streaming/execution model (June 2026)

This doc **corrects** the standing conclusion in `JOIN_GAP_INVESTIGATION_LOG.md`
and `PERF_CAMPAIGN_SUMMARY.md` ("the join gap is the execution model:
gather-into-probe fusion + cascade pipelining"). Direct architecture study of
Polars and Apache DataFusion + a three-way measurement show that conclusion was
measured against the wrong target and mis-attributes the gap.

Bottom line up front: **Polars's own *materializing* (in-memory) engine — the
same execution model as ours — already beats us ~3x. Its streaming engine adds
only 10–35% on top of that. So a streaming/pipelined engine is NOT where the 3x
is. The 3x is (1) string/wide-key group-by falling back to single-threaded
Acero, and (2) per-join data movement.

**FOLLOW-UP (probed, June 2026): neither is a clean *bounded* lever.** Lever (1)
was over-weighted from cProfile inflation — real group cost ~62ms vs Polars
~30ms (not 117 vs 35), and every way to close it dead-ends on the string columns
being expensive to move/hash (naive parallel regresses; the integer-superkey
skip needs metadata we don't track). Lever (2) was already scoped to
bounded/architectural last campaign (`JOIN_LEVERS_SCOPE.md`: gather is
bandwidth-bound, fusion conflicts with chain late-materialization, cascade
pipelining is the architectural engine). **Net: the 3x vs the materializing
engine is the *aggregate* of several ~30–50ms parallel-substrate advantages
(parallel string hashing, fused/cache-local gather, no Python/Arrow↔NumPy
per-operator boundary), none individually a clean catchable win.** This refines
— and is consistent with — the standing "substrate-bound" conclusion; it just
corrects the *mechanism* (distributed substrate taxes within the materializing
model, NOT a missing streaming engine).**

## What Polars and DataFusion actually are (architecture study)

- Both are **interpreted-vectorized** engines, **not compilers**. DataFusion's
  Cranelift JIT (`datafusion-jit`) was **removed in 2023** for being *slower*
  than calling precompiled arrow kernels; Polars never had codegen. The
  from-scratch MLIR compiler the campaign already NO-GO'd was never the bar.
- **Polars has two engines.** The **in-memory** engine (default `.collect()`,
  `crates/polars-mem-engine`) is recursive pull, **fully materializing every
  intermediate — identical model to ours**. Only the opt-in
  `engine="streaming"` (`crates/polars-stream`) is morsel-driven push, fuses the
  payload gather into the join probe (`equi_join.rs` `gather_extend`), and
  pipelines join cascades.
- **DataFusion** is async pull-Volcano over Arrow `RecordBatch`
  (`SendableRecordBatchStream`), parallel via the **exchange** operator
  (`RepartitionExec` + tokio), explicitly **not** morsel-driven. Decisively: it
  **does NOT fuse gather into the probe** — `build_batch_from_indices` does
  `arrow::compute::take` per output column, *the same indices-then-gather we do*
  — yet it still beats engines like ours. Its only execution-model lever over us
  is **pipelining** the probe stream so only build sides materialize.

The DataFusion fact is the key one: an engine that materializes via `take` just
like us, with no gather fusion, is still fast. So **gather-into-probe fusion is
not the differentiator** — refuting the prior `JOIN_KERNEL_PROFILE.md` claim.

## The measurement (the reframing)

Our whole campaign measured "≈0.45x vs Polars" against bare `.collect()` =
the **materializing in-memory engine**. Three-way at SF-3
(`benchmarks/three_way_engine_compare.py`; >1.0x = Polars faster):

| q  | lazy (ms) | pl in-mem | pl streaming | in-mem/lazy | streaming/lazy |
|----|-----------|-----------|--------------|-------------|----------------|
| 3  | 195       | 76        | 67           | 0.39x       | 0.35x          |
| 5  | 338       | 99        | 94           | 0.29x       | 0.28x          |
| 7  | 483       | 367       | 240          | 0.76x       | 0.50x          |
| 8  | 96        | 31        | 42           | 0.33x       | 0.44x (slower!)|
| 9  | 550       | 209       | 168          | 0.38x       | 0.31x          |
| 10 | 374       | 105       | 92           | 0.28x       | 0.25x          |
| 18 | 717       | 301       | 239          | 0.42x       | 0.33x          |

- **The materializing engine (our exact model) already beats us ~3x.**
- **Streaming adds only ~10–35%** over in-memory — and on q8 it *regresses*.
- Therefore fusion + pipelining (the whole streaming-engine value) accounts for
  at most the 10–35% delta, **not** the 3x.

## Where the 3x actually is (per-operator decomposition)

From `benchmarks/decomp_inmem_gap.py` (cProfile of our engine vs Polars
in-memory per-node profile):

1. **String/wide-key group-by → single-threaded Acero.** `_partition_key_arrays`
   returns `None` for any non-integer/temporal key, so the parallel partitioned
   hash-aggregate kernel never fires on real TPC-H (string) group keys. **UPDATE
   (probed, `benchmarks/probe_strkey_groupby.py`) — NOT a clean catchable
   lever; the cProfile 117ms was inflated.** Clean wall-clock on q10's group
   input (344k rows, 114k groups): our full 7-key group_by = **~62ms**, Polars =
   **~30ms** (gap ~32ms, not 82ms). Every attempt to close it dead-ends on the
   strings being expensive to move/hash:
   - **Naive parallel REGRESSES (93ms > 62ms).** Partitioning by key requires
     `take`-ing the wide string columns (incl. long `c_comment`) into each
     bucket; that gather costs more than the parallelism saves — the same
     "gather tax" that killed partitioned join.
   - **The only fast path is to not hash the strings at all:** group by the
     integer key (`c_custkey`, ~10ms vs 62ms) — but that's correct only if it's
     a *superkey*, which we don't track, and there is no cheap exact check (the
     min/max self-check via string aggregation costs ~140ms; the
     group-then-attach path is ~57ms because the attach `take` of 6 string cols
     is itself ~47ms).
   - Polars's ~30ms comes from genuinely parallel string hashing. Matching it
     needs a from-scratch parallel nogil string-hash aggregate kernel (the AG4
     string-hash substrate gap + a new aggregate), for a ~32ms/query win on the
     two queries with high-card string groups (q10, q18). Substrate-kernel
     class, not a bounded plumbing lever. **PARKED.**

2. **Per-join data-movement tax.** Keys-only build+probe is at parity (218 vs
   219ms), but the *realized* cascade is ~2.5x slower (q5: 338 vs 99ms with a
   trivial group — it's all joins+gather+plumbing). The overhead is i64 key
   re-conversion (`_join_key_arrays_i64`), the separate random-access gather pass
   (`_take_all_columns` / `_compose_step`), Arrow↔NumPy round-trips, and
   column-dict rebuilds — compounded across 5 joins.

3. **Corroboration:** our best ratio (q3, 0.58x) has **no string group-by and
   fewer joins**; the worst (q10, 0.28x) leans on the string group + gather.

## Verdict on "build a new engine" (ignoring engineering cost)

We *could* build a DataFusion-style streaming engine (single-core operators +
`RecordBatch` streams + exchange parallelism over our nogil kernels; no JIT
needed). **But the measurement says it is the wrong lever:** it recovers at most
the 10–35% that Polars's streaming engine gains over its own materializing one.

**And the catchable-within-the-materializing-model levers turned out NOT to be
clean bounded wins either** (probed June 2026):

- **String/wide-key group-by — PARKED.** Real gap ~32ms/query on two queries;
  every bounded approach regresses or needs superkey metadata; a true fix is a
  from-scratch parallel string-hash aggregate kernel (substrate class).
- **Per-join data movement — bounded/architectural** (`JOIN_LEVERS_SCOPE.md`):
  gather is memory-bandwidth-bound, fuse-into-probe conflicts with chain
  late-materialization (helps only single-join queries), cascade pipelining IS
  the architectural engine.

So the honest end state: the 3x vs the *materializing* Polars engine is the
aggregate of several ~30–50ms parallel-substrate advantages (parallel string
hashing, fused/cache-local gather, no per-operator Python/Arrow↔NumPy boundary),
none individually a clean catchable win. Closing it means either (a) a fleet of
substrate kernels (parallel string-hash aggregate, fused probe-gather — each a
real Cython/Arrow project for a per-query tens-of-ms), or (b) the architectural
fused engine — which only buys the additional 10–35% over the materializing
model anyway. Under `PROBE_CHARTER.md`, this precise, corrected quantification
is the deliverable.

A streaming engine remains the wrong first move: it buys only the 10–35%
pipelining delta, while the larger 3x lives in the parallel-substrate kernels —
so a substrate-kernel investment dominates an execution-model rewrite on
reward/effort.

## Artifacts
- `benchmarks/three_way_engine_compare.py` — lazy vs Polars in-mem vs streaming.
- `benchmarks/decomp_inmem_gap.py` — per-operator decomposition vs in-mem.
- `benchmarks/probe_strkey_groupby.py` — string-key group-by lever (measured
  non-win: naive parallel regresses; integer-superkey skip needs metadata).
- Architecture study (Polars `crates/polars-{stream,mem-engine,ops}`;
  DataFusion `datafusion/physical-plan/src/{joins,aggregates,repartition}`).
