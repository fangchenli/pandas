# Join-gap investigation — complete log of what we tried (June 2026)

Chronological record of the join/aggregate-gap investigation, every approach
and its measured outcome, so nothing is re-tried blind. Detail docs are linked.
Bottom line up front: **the join gap is the execution model (Polars fuses the
gather into the probe and pipelines the join cascade); our individual join
kernels are at parity; every bounded kernel lever was measured to dead-end.**

## The arc

1. **Buffer-resident join→agg probe** (`BUFFER_JOIN_AGG_PROBE.md`,
   `benchmarks/probe_buffer_join_agg.py`). Hypothesis: gather only the surviving
   {group key, agg value} columns off the join indices and group directly,
   skipping full-join materialization. **Result: reaches Polars parity** in
   isolation (1.00x at SF-3, 18M-row output); the Arrow↔NumPy round-trip is NOT
   the wall when you project before the gather (gather = 18% of the path).
   *Caveat that later proved decisive: the baseline was a naive hand-coded path,
   not the real engine.*

2. **Generalization probe** (`probe_buffer_join_agg_gen.py`). CHAIN (3-table →
   low-card group) held at 0.89x / 2.55x-over-naive-engine; HIGHCARD (join →
   high-card group) did NOT reach parity — wall moves to the group substrate.

3. **Built `PhysicalFusedJoinAgg` + `_fuse_join_aggregates` rewrite** (commit
   `37c46b43d3`). Engine integration learning: execution is morsel-driven, the
   join feeds the agg as a `PrecomputedInput` at the breaker, so fusion must be
   a plan rewrite (not an execute() hook), mirroring `PhysicalFusedFilterAgg`.
   **Result: correct, regression-free, but fires only on q4** — real queries put
   a filter/project or a JoinChain between the join and the group, so the
   direct-adjacency match has ~no reach. Default-off.

4. **Chain extension** (generic producer = join | JoinChain;
   `composite_gather_arrow`). **Result: NET-NEGATIVE (1.32x), reverted** (commit
   `e4db4faa02`). q18 regressed **4.09x** (high-card string group — the HIGHCARD
   loss regime); the intended low-card targets q5/q7/q8/q9/q10 don't fire
   (Project/FusedPipeline tail). **Key discovery: the engine ALREADY
   late-materializes chains** (`PhysicalJoinChain` + `_CompositeJoin`), so the
   probe's 2.55x "engine materializes both joins" gap was vs a naive baseline,
   not the real chain path.

5. **User pushback → direct per-operator profiling** (`JOIN_KERNEL_PROFILE.md`,
   commit `2161a47e4b`). Polars `LazyFrame.profile()` vs cProfile on q5. **Both
   engines spend ~all time in joins; the group is 0.4ms in Polars and ~free in
   ours** — so the whole join→agg group-fusion direction targeted the wrong
   operator.

6. **Join kernel decomposition** (single big join, lineitem 18M ⋈ orders 4.5M):
   - keys-only build+probe = **parity** (ours 218ms ≈ Polars 219ms).
   - `build_join_table_i8` is single-threaded (115ms) but that's a **red
     herring** — keys-only still matches Polars.

7. **Radix-partitioned parallel join over our nogil kernel** (partition both
   sides by key-hash via `partition_by_key`, per-bucket build+probe on threads).
   **Result: REGRESSES ~2x (409ms vs 218), DEAD LEVER** — build isn't the
   bottleneck, so partitioning only adds partition + gather-by-permutation
   overhead. (Distinct from the earlier-rejected parallel-join probe, which
   partitioned `pd.merge`/Arrow's already-threaded join.)

8. **Gather analysis.** Our standalone gather is **memory-bandwidth-bound**
   (~64ms for 4 cols × 18M via parallel `np.take`); within-column chunking
   (87ms) and cols×rows parallel (78ms) both *regress* — more threads don't
   help. Polars's full join (build+probe+**gather**, ~228ms) costs barely more
   than our build+probe *alone* (218ms): it **fuses the gather into the probe**
   (each row gathered while hot in cache), while we do a separate random-access
   take pass.

9. **Scoped both remaining levers** (`JOIN_LEVERS_SCOPE.md`, commit
   `a632a74010`): (1) fuse-gather-into-probe — ~50–65ms/join, single-join only,
   conflicts with chain late-materialization; (2a) trim late-mat compose
   overhead — ~50ms of the ~176ms q5 gap, regression-risky; (2b) true cascade
   pipelining — the real chain win (~120ms) but architectural (Path A/B).

## What is ruled out by measurement (do not re-try)

- Join→agg **group fusion** (buffer-resident or chain) as a TPC-H win — the
  group is ~free; the engine already late-materializes; net-negative on the real
  suite (q18 4x regression).
- **Radix-partitioned parallel join** over our kernel — regresses; build is not
  the bottleneck.
- **Parallelizing the build** alone — keys-only is already at parity.
- **Faster standalone gather** (more threads / chunking) — memory-bandwidth-
  bound; chunking regresses.

## What remains (and why it's not bounded)

- **Fuse gather into the probe** (kernel rewrite emitting payload, not indices):
  real ~50ms/single-join, but conflicts with the chain's late-materialization
  and only helps the few single-join queries → small.
- **Pipeline the join cascade + stream the group** (Path A/B): the real chain
  win (~120ms/query), but it requires rebuilding execution from breaker-per-
  operator to fused morsel streaming — the multi-person architectural item the
  campaign already NO-GO'd as a from-scratch engine.

## Conclusion

The join gap is fully characterized at operator grain: **execution model
(gather-into-probe fusion + cascade pipelining), not a swappable kernel.** Our
kernels are at parity; Polars wins by never materializing intermediate join
outputs as separate arrays. This is the finest-grained confirmation of the
standing "~0.45x substrate/execution-model-bound" finding. Under the
`PROBE_CHARTER.md` framing (lazy-pandas is a substrate probe, not a product),
this precise quantified gap is the deliverable; making it faster is the
architectural engine, out of scope for a probe.

## Artifacts (this investigation)

- `BUFFER_JOIN_AGG_PROBE.md`, `benchmarks/probe_buffer_join_agg.py`,
  `benchmarks/probe_buffer_join_agg_gen.py`
- `JOIN_KERNEL_PROFILE.md`, `JOIN_LEVERS_SCOPE.md`
- `PhysicalFusedJoinAgg` + `_fuse_join_aggregates` (in `physical.py`,
  **default-off** `_FUSE_JOIN_AGG=False`) — the validated foundation, kept for
  the record; not a win on the realized TPC-H shapes.
