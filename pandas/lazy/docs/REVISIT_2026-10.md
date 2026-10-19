# Revisit (October 2026): is there a path to Polars parity?

Second look at the "no path to parity" conclusion that closed the engine work
in July 2026. Measured on the same box as the earlier campaigns (Apple M1,
8 cores, 16 GB), SF-1, every result validated against DuckDB. Drivers:
`benchmarks/bench_polars2_bar.py` (§3, `--sf`, `--skip-lz`) and
`benchmarks/rust_engine_prototype/bench_all22.py` (§4); both build on
`bench_tpch.py` and `translate_datafusion.py`.

## 1. The premise, corrected

We did not stop in July because no path existed. Two paths had been measured
against the Polars bar of the time (the 1.x **in-memory** engine):

| Path | Status in July 2026 | Evidence |
|---|---|---|
| Owned Python/Cython engine | 0.45–0.48x geomean, plumbing levers exhausted | `PERF_CAMPAIGN_SUMMARY.md`, `ROADMAP.md` |
| Owned Rust prototype (`lazy_engine_rs`) | 22/22 correct; beats Polars on aggregate shapes (q1 1.7x, q6 2.6x), parity on q3 (0.95x hand-tuned), **0.15x general** because multi-join chains fall to take-every-column materialization | `RUST_ENGINE_DIRECTION.md` |
| LogicalPlan lowered into DataFusion | 22/22 correct; **geomean 1.00x (parity)** execute-only at SF-3 | `EXECUTION_RESEARCH_SURVEY.md` §5 |

We stopped for positioning reasons, not technical ones: the API we built is a
new Polars-style expression API (not pandas), so parity would not justify a
product (`PROBE_CHARTER.md`); delegating execution to another engine was judged
"not a winning strategy" (`ADOPTION.md`); and productizing the owned Rust
engine is a multi-person-year commitment plus a Rust toolchain in the pandas
build.

## 2. What changed since July

**Polars 2.0 shipped (final release 2026-10-06; RC 2026-09-02) and made the
streaming engine the default for `collect()`.** The in-memory engine that every
prior number was measured against is now opt-in
(`pl.Config.set_engine_affinity("in-memory")`). Row order is no longer
guaranteed for join / group_by / unpivot unless `maintain_order=True` is passed,
so Polars also stopped paying the ordering cost our engine pays by contract.
Polars' own PDS-H numbers (May 2025, 96 vCPU): streaming 2.5x faster than
in-memory at SF-10, 6.4x at SF-100. Still in flight for 2.x: out-of-core,
cost-based planner, join reordering.

**DataFusion:** core 54.1.0 (2026-07-21), datafusion-python 54.0.0. The
subplan-materialization epic (#22676, the cause of our q2/q11/q15/q21 recompute
tax) is active: `MATERIALIZED` CTE syntax merged, 1/11 sub-issues done, three
PRs ready for review as of 2026-10-16. The single-distinct DataFrame-API fix we
proposed (AG10, PR #23403) was closed stale on 2026-09-15 — our
`_rewrite_n_unique` workaround remains necessary.

**Arrow:** our Substrait consumer fix (PR #50635, AG18) is still open,
awaiting a code-owner review (last activity 2026-09-08).

## 3. Re-measurement against the Polars 2.0 bar (SF-1, best-of-3, ms)

LZ = owned Python/Cython engine; DF = DataFusion lowering (sources registered
once per query, execute-only); PLs = Polars 2.0 streaming (new default);
PLm = Polars 2.0 in-memory (the old bar). Ratios > 1 mean the row engine is
slower than Polars streaming.

| q | LZ | DF | PLs | PLm | PLs/LZ | PLs/DF | PLm/PLs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 267.3 | 93.4 | 52.8 | 85.9 | 0.20 | 0.57 | 1.63 |
| 2 | 75.7 | 186.9 | 47.0 | 42.0 | 0.62 | 0.25 | 0.89 |
| 3 | 85.4 | 30.3 | 28.0 | 36.0 | 0.33 | 0.92 | 1.29 |
| 4 | 124.1 | 65.0 | 15.0 | 35.0 | 0.12 | 0.23 | 2.34 |
| 5 | 97.5 | 57.0 | 27.1 | 50.0 | 0.28 | 0.48 | 1.84 |
| 6 | 13.1 | 18.4 | 5.0 | 10.3 | 0.38 | 0.27 | 2.06 |
| 7 | 229.3 | 191.4 | 60.1 | 98.4 | 0.26 | 0.31 | 1.64 |
| 8 | 110.6 | 52.1 | 20.7 | 15.6 | 0.19 | 0.40 | 0.75 |
| 9 | 166.6 | 93.8 | 71.5 | 95.0 | 0.43 | 0.76 | 1.33 |
| 10 | 146.8 | 63.1 | 67.3 | 54.7 | 0.46 | 1.07 | 0.81 |
| 11 | 23.3 | 81.4 | 13.3 | 13.6 | 0.57 | 0.16 | 1.02 |
| 12 | 155.6 | 87.4 | 22.5 | 72.7 | 0.14 | 0.26 | 3.23 |
| 13 | 137.6 | 41.5 | 41.5 | 102.6 | 0.30 | 1.00 | 2.47 |
| 14 | 37.2 | 22.6 | 5.6 | 8.3 | 0.15 | 0.25 | 1.49 |
| 15 | 54.0 | 14.2 | 10.4 | 15.0 | 0.19 | 0.73 | 1.44 |
| 16 | 56.9 | 31.6 | 22.6 | 13.8 | 0.40 | 0.71 | 0.61 |
| 17 | 108.3 | 159.4 | 44.1 | 77.8 | 0.41 | 0.28 | 1.76 |
| 18 | 199.7 | 219.9 | 69.1 | 123.9 | 0.35 | 0.31 | 1.79 |
| 19 | 164.3 | 34.9 | 32.1 | 75.7 | 0.20 | 0.92 | 2.36 |
| 20 | 178.8 | 83.0 | 70.2 | 96.5 | 0.39 | 0.85 | 1.37 |
| 21 | 878.9 | 1529.8 | 343.3 | 301.5 | 0.39 | 0.22 | 0.88 |
| 22 | 43.4 | 33.0 | 23.7 | 25.9 | 0.55 | 0.72 | 1.09 |
| **geomean** | | | | | **0.30** | **0.45** | **1.42** |
| **total** | 3354 | 3190 | 1093 | 1450 | | | |

All 22 lazy-engine and DataFusion results validate against DuckDB. (The
harness flagged Polars q3; that is a column-name artifact — Polars emits
`o_orderkey`, DuckDB `l_orderkey`, and the validator aligns by sorted column
name. Values are identical on both Polars engines.)

Readings:

- **The bar moved 1.42x.** Polars 2.0 streaming beats its own in-memory engine
  by 1.42x geomean at SF-1 on 8 cores (up to 3.2x on q12, 2.5x on q13); it
  loses only on a few tiny dimension-heavy queries (q8, q16, q2, q21). The gap
  grows with scale (Polars' own SF-10 figure is 2.5x), so SF-3+ is worse for us
  than this table.
- **Owned Python/Cython engine: 0.30x** (was 0.45–0.48x). Nothing in this
  engine changed; the target did. Every lever here was already measured as
  exhausted.
- **DataFusion lowering: 0.45x** (was ~1.0x vs in-memory). It is still at
  parity on the join+aggregate shapes (q3 0.92, q10 1.07, q13 1.00, q19 0.92,
  q20 0.85) and loses in two known places: (a) the shared-subplan /
  count-distinct queries q2/q11/q21 (0.16–0.25x) where our eager
  materialization workaround for DataFusion's missing subplan CSE dominates;
  (b) the small scan→filter→aggregate queries q4/q6/q12/q14 (0.23–0.27x),
  5–20 ms queries where DataFusion's per-batch exchange overhead shows at SF-1.
  Both are DataFusion-side items with upstream work in flight, not our
  frontend.
- **Owned Rust prototype:** see §4.

### 3a. SF-3 re-baseline: DataFusion lowering vs Polars 2.0 (the July comparison, re-run)

Same method as July's 1.00x number (execute-only, SF-3, datafusion-python
51.0.0), Polars 2.0.0. Lazy engine skipped (its SF-3 number, 0.44–0.48x vs
in-memory, is already on record in `ROADMAP.md`).

| q | DF | PLs | PLm | PLs/DF | PLm/PLs |
|---|---:|---:|---:|---:|---:|
| 1 | 272.7 | 219.3 | 1106.9 | 0.80 | 5.05 |
| 2 | 541.7 | 94.8 | 164.4 | 0.17 | 1.73 |
| 3 | 145.6 | 109.8 | 244.6 | 0.75 | 2.23 |
| 4 | 188.3 | 31.7 | 126.9 | 0.17 | 4.00 |
| 5 | 184.1 | 82.9 | 129.9 | 0.45 | 1.57 |
| 6 | 55.4 | 14.4 | 33.0 | 0.26 | 2.29 |
| 7 | 788.6 | 369.5 | 512.9 | 0.47 | 1.39 |
| 8 | 39.2 | 59.7 | 40.1 | 1.52 | 0.67 |
| 9 | 232.6 | 161.1 | 401.1 | 0.69 | 2.49 |
| 10 | 180.7 | 140.5 | 128.3 | 0.78 | 0.91 |
| 11 | 205.1 | 21.4 | 33.2 | 0.10 | 1.55 |
| 12 | 206.0 | 52.5 | 188.5 | 0.25 | 3.59 |
| 13 | 139.3 | 176.5 | 335.0 | 1.27 | 1.90 |
| 14 | 39.2 | 16.5 | 32.1 | 0.42 | 1.95 |
| 15 | 28.4 | 15.3 | 28.4 | 0.54 | 1.86 |
| 16 | 123.2 | 36.6 | 29.6 | 0.30 | 0.81 |
| 17 | 379.0 | 223.0 | 444.5 | 0.59 | 1.99 |
| 18 | 582.1 | 253.3 | 368.0 | 0.44 | 1.45 |
| 19 | 178.5 | 61.6 | 219.0 | 0.35 | 3.56 |
| 20 | 190.5 | 230.1 | 248.4 | 1.21 | 1.08 |
| 21 | 5827.2 | 1337.5 | 1334.3 | 0.23 | 1.00 |
| 22 | 117.3 | 173.1 | 76.1 | 1.48 | 0.44 |
| **geomean** | | | | **0.47** | **1.68** |
| **total** | 10645 | 3881 | 6225 | | |

- **At SF-3 the bar moved 1.68x** (streaming vs in-memory), more than the
  1.42x at SF-1, consistent with Polars' own scale trend. q1 alone is 5x: the
  in-memory engine's materializing group-by over 18M rows is exactly the
  cache-miss regime the streaming engine was built to avoid.
- **DataFusion vs the old bar is unchanged** (DF total 10.6 s vs in-memory
  6.2 s is dominated by q21's 5.8 s; without q21 the two are within 10%, the
  July parity). **Vs the new bar it is 0.47x geomean** — identical to SF-1's
  0.45x, so this is not a per-query-overhead artifact. DF still wins q8, q13,
  q20, q22 and is within 0.75–0.80x on q1/q3/q10.
- The losses are the same two buckets as SF-1: shared-subplan / count-distinct
  (q2 0.17, q11 0.10, q21 0.23) and small scan→filter→aggregate (q4 0.17,
  q6 0.26, q12 0.25).

### 3b. datafusion-python 51 → 54 (SF-1, execute-only)

Upgraded `pandas-dev` to datafusion 54.0.0, re-ran the lowering, then
restored 51.0.0. **22/22 still validate against DuckDB on 54** (no lowering
breakage across three releases). Geomean vs Polars streaming 0.48x (0.45x on
51) — within run-to-run noise; no query moved by more than noise in either
direction (first 54 run was discarded: Polars' own times doubled on several
queries, the box was contended). The release upgrade is not a lever; the
in-flight upstream work (subplan materialization #22676) is.

## 4. Owned Rust prototype vs Polars 2.0 streaming (SF-1)

Rebuilt `lazy_engine_rs` from the checkpoint source (maturin, release), plan
compiled and Arrow tables built once per query outside the timed region,
timed = Rust execute + `to_pandas`. **22/22 validate against DuckDB.**

| q | RS (ms) | PLs (ms) | PLs/RS |
|---|---:|---:|---:|
| 1 | 116.1 | 42.5 | 0.37 |
| 2 | 509.0 | 37.9 | 0.07 |
| 3 | 338.6 | 21.2 | 0.06 |
| 4 | 387.1 | 12.1 | 0.03 |
| 5 | 198.4 | 26.8 | 0.13 |
| 6 | 11.8 | 4.7 | 0.40 |
| 7 | 362.8 | 55.2 | 0.15 |
| 8 | 58.6 | 17.5 | 0.30 |
| 9 | 161.5 | 58.2 | 0.36 |
| 10 | 228.8 | 29.1 | 0.13 |
| 11 | 187.8 | 9.8 | 0.05 |
| 12 | 144.1 | 22.6 | 0.16 |
| 13 | 172.0 | 36.2 | 0.21 |
| 14 | 26.2 | 5.2 | 0.20 |
| 15 | 52.4 | 8.9 | 0.17 |
| 16 | 32.6 | 19.3 | 0.59 |
| 17 | 412.2 | 57.6 | 0.14 |
| 18 | 1349.7 | 81.1 | 0.06 |
| 19 | 207.9 | 28.0 | 0.13 |
| 20 | 531.2 | 81.0 | 0.15 |
| 21 | 3345.5 | 462.8 | 0.14 |
| 22 | 31.3 | 12.4 | 0.39 |
| **geomean** | | | **0.16** |

Same picture as the July checkpoint (0.15x vs in-memory at SF-3), now against
the harder bar: correct everywhere, slow on every multi-join chain for the one
known reason (take-every-column materialization between joins; only one
join+aggregate pair is fused). The aggregate-bound shapes that beat Polars
in-memory at SF-3 (q1, q6) no longer beat Polars streaming at SF-1
(0.37x, 0.40x). The prototype remains what the checkpoint called it: proof
that the architecture is not the barrier, not an engine.

## 5. Assessment

**Is there a path to parity with Polars?** Three different questions hide in
that sentence, and they have different answers.

1. *Can the engine we own (Python/Cython) reach parity?* **No**, and this is
   now more certain, not less. It is at 0.30x of a bar that just moved 1.4x and
   will keep moving (Polars 2.x roadmap: cost-based planner, join reordering,
   out-of-core). The execution-model diagnosis in `PERF_CEILING.md` is
   unchanged; nothing in this layer closes a 3x gap.

2. *Can an owned native engine reach parity?* **Architecturally yes, as a
   sustained program, not a project.** The Rust prototype already beats Polars
   on aggregate-bound shapes and reached join parity on q3, and its general-path
   deficit is a known, scoped gap (late materialization across join chains).
   But "parity" is parity with a 2.0 release that is itself a new engine from a
   funded team. Catching a moving target requires matching their rate of
   investment, plus a Rust toolchain in pandas' build and a maintainer
   community that does not exist today. The July conclusion ("the engineering
   is not the blocker, the commitment is") stands.

3. *Can a lazy pandas reach parity by lowering to a native backend?* **Yes,
   within ~2x today and converging**, with Apache DataFusion as the obvious
   backend: Arrow-native, Apache-governed (not a competing product), already
   22/22 correct from our frontend, and its remaining losses map onto work
   already in its tracker (subplan materialization; distinct rewrite). This was
   parity in July against the then-default Polars engine. It is the only path
   whose cost is bounded by *our* frontend work rather than by Polars' roadmap.

**The question that still matters more than speed.** `PROBE_CHARTER.md`'s
finding is unchanged: what we built is a new Polars-style API. At any speed,
a new API that is at best as fast as Polars has no reason to exist. A lazy
mode that executes the *real pandas API* with pandas semantics is the only
product form with a reason to exist, and for that product the honest success
metric is "how much faster than eager pandas, with no code changes", not
"parity with Polars". By that metric even the current Cython engine is already
a large win (Polars' own PDS-H run has eager pandas 94x behind Polars streaming
at SF-10; we are ~3x behind). The frontend fork (`RUST_ENGINE_DIRECTION.md`
§Frontend fork, option 6b) is where that project would start, and it is
FireDucks/Bodo-sized.

## 6. Recommendation

- **Do not reopen "owned engine to Polars parity."** The evidence base that
  closed it is stronger after Polars 2.0, not weaker. Keep the Rust prototype
  as the probe baseline it already is.
- **If the lazy branch is to have a product future, change the goal, not the
  engine:** pandas-faithful frontend over a pluggable native backend, with
  DataFusion first and speed measured against eager pandas. Decide that
  explicitly before spending more engine effort.
- **Cheap, probe-consistent follow-ups** (each a measurement, none a grind):
  1. Re-baseline the DataFusion lowering at SF-3 against Polars 2.0 streaming
     (the July 1.00x was SF-3; SF-1 over-weights per-query overhead).
  2. Upgrade `datafusion` 51 → 54 in `pandas-dev` and re-measure; re-validate
     the lowering (22/22) on the newer release.
  3. When #22676 (subplan materialization) lands, drop the eager
     materialize-shared-subplan workaround in `translate_datafusion.py` and
     re-measure q2/q11/q15/q21.
  4. Polars 2.0's dropped order guarantee is a benchmark-fairness item: our
     `collect(order="relaxed")` is the matching mode and should be the one
     compared against default `collect()` in Polars 2.x.
