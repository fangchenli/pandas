# Join kernel profile — where the Polars gap actually is (q5, SF-3)

Measurement session 2026-06-25. Direct side-by-side profile of q5 (6-table
join chain → group by n_name (25) → sort) using **Polars' built-in per-node
profiler** vs **cProfile** of lazy-pandas. This corrects the earlier
"individual kernels are at/near parity" claim: **for joins it is false.**

## Per-node profile (q5, SF-3): lazy 297ms vs Polars 121ms (0.41x)

**Polars** (`LazyFrame.profile()`), time concentrated in joins, group is free:

| node | ms |
|---|---|
| join(o_orderkey) | 91 |
| join(l_suppkey) | 85 |
| join(c_custkey) | 19 |
| filter(c_nationkey==s_nationkey) | 19 |
| **streaming_group_by** | **0.4** |
| sort(revenue) | 0.03 |

(joins run in parallel/pipelined, so they don't sum into the 121ms wall.)

**lazy-pandas** (cProfile cumulative) — also entirely in the joins:

| frame | ms | note |
|---|---|---|
| JoinChain.execute | 300 | the whole query |
| `inner_join_indexers_i8` | **181** | **61% of the query** — our hash join |
| compose `column` (key gather per step) | 22 | late-materialization bookkeeping |
| compose `extend` (re-index bases) | 29 | " |
| final `gather` (payload take) | 60 | |

The group/sort/filter are tiny for us too. **The join chain is the gap, not the
group** — the join→agg group fusion explored earlier this session was aimed at
an operator that is already ~free in both engines.

## Isolating one big join (lineitem 18M ⋈ orders 4.5M)

| path | ms |
|---|---|
| OURS — index probe only (`inner_join_indexers_i8`, no gather) | 220 |
| OURS — index probe + gather (4 cols) | 319 |
| **POLARS — full join incl. gather** | **233** |

Our **index probe alone** is as slow as Polars' **entire join**. Decomposed:

| component | ms | threaded? |
|---|---|---|
| **`build_join_table_i8`** (CSR hash build, 4.5M keys) | **115** | **single-threaded** |
| probe count pass (18M, one chunk) | 121 | parallel in production |
| Polars full join (keys only) | 182 | fully multithreaded |

## What's actually slow — measured layer by layer (corrects two false starts)

Drilling in overturned two intermediate hypotheses; the real answer is the
execution model, now proven at the operator level.

**Keys-only single join: PARITY.** Current `inner_join_indexers_i8` = 218ms,
Polars keys-only join = 219ms. So the build+probe is *not* the gap — even though
`build_join_table_i8` is single-threaded (115ms), the full keys-only join
matches Polars. (My "single-threaded build is the #1 lever" was wrong.)

**Partitioned parallel join: REGRESSES — dead lever.** Radix-partition both
sides (`partition_by_key`) + per-bucket build+probe on a thread pool = **409ms**
(vs 218 current / 219 Polars), correct but ~2x slower. Because build isn't the
bottleneck, partitioning only adds partition + gather-by-permutation overhead.
Do not pursue.

**The gap is the GATHER, and Polars fuses it into the probe.** Single join with
4 payload columns:

| | ms |
|---|---|
| our build+probe (keys only) | 218 |
| + our gather (4 cols × 18M, parallel `np.take`) | +108 → **326** |
| arrow `pc.take` per col (alt gather) | 86 |
| **Polars full join (build+probe+gather, 4 col)** | **243** |

Polars's *entire* join (243) costs barely more than our build+probe *alone*
(218): it materializes payload **during** the probe — one multithreaded pass,
each row gathered while it is hot in cache. We produce indices, then do a
**separate** 108ms random-access gather pass that cold-misses on 18M scattered
rows. That two-pass-vs-fused difference, not kernel speed, is the per-join gap.

**The chain gap is pipelining + overlap.** q5 (297 vs 121ms): our 5 join probes
(181ms) run **sequentially** breaker-by-breaker, plus late-materialization
compose overhead (`column` 22ms + `extend` 29ms) and the final gather (60ms).
Polars **pipelines morsels across the join cascade and streams the group**, so
its ~195ms of join work overlaps into 121ms wall and the group is ~free.

## Conclusion — execution model, not a swappable kernel

Our individual join kernels (build+probe) are at parity with Polars. Polars
wins by **fusing the gather into the probe** (cache-local, single pass) and
**pipelining/overlapping** the join cascade + streaming the group — it never
materializes intermediate join outputs as separate arrays the way our
breaker-per-operator model does. This is the execution-model tax (cf.
`PERF_CEILING.md`), now measured per-operator rather than asserted.

The only kernel-level lever left is **fusing gather into the probe** (a join
kernel that emits payload directly instead of indices) — a substantial kernel
rewrite that captures the ~80ms single-join gather gap but still not the
cross-join pipelining. The pipelining/overlap is the architectural item
(Path A/B). Partitioned parallel join and single-threaded-build are both ruled
out by measurement.
