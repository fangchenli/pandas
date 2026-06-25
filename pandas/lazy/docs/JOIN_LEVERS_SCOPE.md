# Scoping the two join levers (after the per-operator profile)

Follows `JOIN_KERNEL_PROFILE.md`. The profile pinned the join gap to the
execution model: build+probe at parity, Polars wins by (1) fusing the gather
into the probe and (2) pipelining/overlapping the join cascade + streaming the
group. This doc scopes both as concrete builds with their measured headroom.

## Lever 1 — fuse the gather into the probe

**Goal:** emit payload columns *during* the probe (cache-hot rows, one pass)
instead of producing indices then doing a separate random-access take.

**Measured headroom (single big join, 18M⋈4.5M, SF-3):**
- build+probe (keys): 218ms == Polars 219ms (parity).
- our standalone gather (4 cols, parallel `np.take` across columns): ~64ms,
  and it is **memory-bandwidth-bound** — within-column chunking (87ms) and
  cols×rows parallel (78ms) both *regress*; more threads don't help.
- Polars full join (build+probe+**gather**): ~228ms ≈ our build+probe alone.
- So the fusable gap is **~50–65ms per payload join**, entirely cache-locality:
  Polars touches each matched row once (probe+gather together) while we touch
  build rows twice (probe records the index, gather re-reads it cold).

**Design:** a new Cython join that, per match, writes the probe-side payload
(sequential, cheap) and gathers the build-side payload (random, but hot from
the probe) directly into output buffers — single nogil pass, thread-parallel
over probe chunks. Returns materialized columns, not indices.

**Catch (decisive):** this **conflicts with the chain's late-materialization.**
`PhysicalJoinChain` deliberately does NOT gather intermediate payloads — it
composes indices and gathers once at the end (that is the existing chain win).
A gather-fusing join re-introduces an intermediate gather at every chain step,
which is exactly what late-mat avoids. So lever 1 only helps **single-join**
queries (q4/q12/q14/q19/q22), and even there the gather is already narrowed by
projection pushdown (the group needs few columns), shrinking the ~50–65ms gain.

**Verdict:** real but small, and a genuine kernel rewrite. Net ~50ms on a
handful of single-join queries; does not touch the chains (the bigger gap).

## Lever 2 — pipeline/overlap the join cascade + stream the group

**Goal:** match Polars's whole-cascade execution — morsels flow through join 1 →
join 2 → … → group concurrently, instead of our breaker-by-breaker materialize.

**Measured headroom (q5, SF-3): lazy 297ms vs Polars 121ms (the 2.5x).**
Our breakdown: 5 join probes ≈181ms (sequential, each internally parallel),
compose bookkeeping (`column` key-gather 22ms + `extend` base re-index 29ms =
51ms, late-mat-specific), final gather 60ms. Polars overlaps ~195ms of join
work into 121ms wall and the group is streamed (0.4ms).

**Two sub-levers:**
- **(2a) Reduce the late-mat compose overhead** (51ms, ours-specific). The
  `extend` step re-indexes every base's index array at each chain step
  (O(output × n_bases)); `column` re-gathers the running key for the next
  probe. Bounded, in `_CompositeJoin` — but only ~50ms of the ~176ms gap.
- **(2b) True morsel pipelining of the cascade.** This is the architectural
  item: our engine is breaker-per-operator (each join fully materializes before
  the next), so overlapping joins means rebuilding execution around streaming
  morsels through fused operators — Path A/B in `PERF_CEILING.md`
  (multi-person-quarter+). Even per-join parallelism is already saturating
  cores during each probe, so the win is overlap + fused gather, not more
  threads.

**Verdict:** (2a) is a bounded but small (~50ms) cleanup; (2b) is the real
chain win and is architectural, not a kernel.

## Bottom line

Both levers are now scoped with numbers, and **neither is a clean bounded win:**

| lever | reach | gain | effort | blocker |
|---|---|---|---|---|
| 1 gather-into-probe | single-join queries only | ~50ms/join (narrowed) | kernel rewrite | conflicts with chain late-mat |
| 2a compose-overhead trim | chains | ~50ms (of 176ms gap) | medium | only part of the gap |
| 2b cascade pipelining | chains | the rest (~120ms) | multi-person | architectural (Path A/B) |

This is the finest-grained confirmation of the standing conclusion: the join
gap is the execution model (fusion + pipelining), our kernels are at parity, and
the bounded kernel levers (partitioned join, faster gather, parallel build) are
exhausted or marginal. Closing it for real is the fused-pipelined engine
(Path A/B), not another kernel.
