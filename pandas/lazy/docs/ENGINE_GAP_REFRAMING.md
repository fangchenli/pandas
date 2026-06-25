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
Acero, and (2) per-join data movement — both catchable inside the current
materializing model, no new engine required.**

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

1. **String/wide-key group-by → single-threaded Acero (the biggest lever).**
   `_partition_key_arrays` returns `None` for any non-integer/temporal key, so
   the parallel partitioned hash-aggregate kernel (the one that "beats Polars
   1.9x" on integer keys) **never fires on real TPC-H group keys**, which are
   strings. q10 (7-key string group: `c_name, c_phone, n_name, c_address,
   c_comment`): **Acero `_group_by` 117ms vs Polars 35ms** (~82ms gap on the
   group alone). q9 (n_name): 62ms. Polars parallelizes string group-by via key
   row-encoding + hash partitioning. **This is a kernel-routing gap, not an
   execution-model gap.**

2. **Per-join data-movement tax.** Keys-only build+probe is at parity (218 vs
   219ms), but the *realized* cascade is ~2.5x slower (q5: 338 vs 99ms with a
   trivial group — it's all joins+gather+plumbing). The overhead is i64 key
   re-conversion (`_join_key_arrays_i64`), the separate random-access gather pass
   (`_take_all_columns` / `_compose_step`), Arrow↔NumPy round-trips, and
   column-dict rebuilds — compounded across 5 joins.

3. **Corroboration:** our best ratio (q3, 0.58x) has **no string group-by and
   fewer joins**; the worst (q10, 0.28x) is dominated by the 117ms string group.

## Verdict on "build a new engine" (ignoring engineering cost)

We *could* build a DataFusion-style streaming engine (single-core operators +
`RecordBatch` streams + exchange parallelism over our nogil kernels; no JIT
needed). **But the measurement says it is the wrong lever:** it recovers at most
the 10–35% that Polars's streaming engine gains over its own materializing one.
The 3x is catchable inside the current materializing model:

- **Parallelize string/wide-key group-by** — row-encode/factorize keys to int
  codes, then route through the existing partitioned kernel. ~80ms/query on the
  group-heavy queries (q9, q10). Highest-value concrete win.
- **Cut per-join data movement** — fuse gather into the probe (the *one* fusion
  that helps), eliminate Arrow↔NumPy round-trips and redundant key conversions
  per join. Targets the cascade queries (q5, q9).

A streaming engine should be revisited only if, after both levers land, a
residual gap remains that is *demonstrably* the 10–35% pipelining delta.

## Artifacts
- `benchmarks/three_way_engine_compare.py` — lazy vs Polars in-mem vs streaming.
- `benchmarks/decomp_inmem_gap.py` — per-operator decomposition vs in-mem.
- Architecture study (Polars `crates/polars-{stream,mem-engine,ops}`;
  DataFusion `datafusion/physical-plan/src/{joins,aggregates,repartition}`).
