# Query-execution systems survey — advanced techniques for the arrow-rs engine (2026-07-08)

A cited literature survey of advanced query-execution techniques, filtered
through **our** engine and the **probe mission** (instrument to find/quantify
substrate gaps; realistic ceiling on join-heavy shapes = parity with Polars, not
domination). Companion to `RUST_ENGINE_DIRECTION.md` (engine shape + honest
perf). Commissioned as a background research agent; every claim carries a primary
citation.

## The one-line diagnosis the literature confirms

**Our 0.15x geomean is a data-movement problem, not a code-quality problem.** We
materialize whole columns operator-at-a-time and `take` *every* column through
join chains. Kersten et al. (VLDB 2018) — the definitive compiled-vs-vectorized
head-to-head — shows compilation's advantage is **smallest** exactly on
memory-bound workloads. That licenses our instinct: **do not build a JIT.** The
techniques that matter for us are the unglamorous ones (vectorized morsel
streaming, late materialization, collision-free keys, selectivity-gated Bloom
pre-filtering), and they line up with the checkpoint plan already on file.

## 1. Execution-model taxonomy

| Model | Seminal source | Verdict for us |
|---|---|---|
| **Vectorized / pull** (Volcano with batches) | MonetDB/X100 — Boncz, Zukowski, Nes, CIDR 2005 | **The core gap.** Arrow arrays *are* the vectors, but we run operator-at-a-time (materialize the whole column) instead of vector-at-a-time (materialize a 64K morsel, pass it on, reuse the buffer). Lowest-risk highest-leverage structural change. Adopt. |
| **Data-centric compilation / push** (produce–consume) | Neumann, VLDB 2011 (HyPer) | Theoretical ceiling, **wrong tool for a probe.** JIT is multi-quarter and wins mostly on compute-bound expression chains, not our join-materialization gap. |
| **Compiled vs vectorized head-to-head** | Kersten et al., VLDB 2018 | Our justification memo: **choose vectorization.** Our gap is memory-bound — precisely where compilation's edge is smallest. |
| **Morsel-driven parallelism** | Leis, Boncz, Kemper, Neumann, SIGMOD 2014 | We already do the *aggregate* half correctly (per-thread partial tables merged via rayon → parity). Lesson: make morsel-driven the *general* substrate, not two special cases. |
| **Push-based streaming in practice** | DuckDB (push), DataFusion (pull), Polars new-streaming | Push-vs-pull is **second-order** — DataFusion's paper shows pull matches DuckDB's push scheduler. Don't agonize; the win is *streaming morsels at all* + *not materializing every column*. |

## 2. Beyond simple fusion — advanced pipelining / codegen

- **Relaxed Operator Fusion** (Menon, Mowry, Pavlo, VLDB 2017): insert a
  vector-sized buffer in front of random-access operators (hash probes) to enable
  software prefetch/SIMD. *Verdict:* steal the **idea** (prefetch buffer at the
  probe) as a scoped experiment *after* late-materialization; skip the compiled
  staging machinery.
- **Adaptive execution of compiled queries** (Kohn, Leis, Neumann, ICDE 2018) &
  **Permutable compiled queries** (Menon et al., VLDB 2020): solve compile
  latency we don't have. *Verdict:* out of scope as built; steal **only** the
  runtime selectivity-based predicate/probe reordering heuristic (~30 lines).
- **Cranelift / LLVM query codegen:** relevant only if we ever go compiled;
  Cranelift's fast compile would suit ad-hoc TPC-H. *Verdict:* over-hyped for our
  bottleneck — codegen makes *compute* faster, not *gather* cheaper.

**Section verdict:** nothing here is worth *building* for a memory-bound
boundary-once interpreter. Two cheap *ideas* worth stealing without their
machinery: prefetch buffer at hash probes, and selectivity-based reordering.

## 3. Join research — our biggest gap

- **Radix-partitioned vs non-partitioned hash join** — Balkesen et al. (VLDB
  2013, "Sort vs. Hash Revisited"); **Bandle, Giceva, Neumann, SIGMOD 2021**
  ("To Partition, or Not to Partition, That is the Join Question in a Real
  System" — *this is the correct attribution; not "Schuh SIGMOD'21"*); Schuh,
  Chen, Dittrich (SIGMOD 2016, "Thirteen Relational Equi-Joins"). *Verdict:*
  **second-order for us and often not worth it** in a real system. Not our
  bottleneck — the join is slow because it gathers every column, not because
  probes miss cache. The radix micro-benchmark literature measures *key-only*
  joins on cache-resident data and systematically understates the payload/gather
  cost that is our entire problem. Measure *after* late materialization.
- **Sideways information passing — LIP & Predicate Transfer** — Zhu et al. (VLDB
  2017, "Looking Ahead Makes Query Plans Robust" / LIP); **Yang, Zhao, Yu,
  Koutris, CIDR 2024 ("Predicate Transfer")**; follow-ups Robust Predicate
  Transfer (SIGMOD 2025, arXiv 2502.15181) and Parachute (arXiv 2506.13670,
  2025). *Verdict:* **highest-leverage join-specific technique for our gap** — it
  shrinks inputs *before* the gather, and its wins are largest exactly on TPC-H
  multi-join shapes (Q3/Q5/Q7/Q8/Q9/Q10 — our worst). We already have an internal
  probe (`PREDICATE_TRANSFER_PROBE.md`). Limit the literature flags: the win
  depends on join selectivity (large when joins filter heavily, ~zero/negative on
  non-reducing joins) → **gate it with a runtime selectivity check so it
  self-disables.** The probe's job is to quantify SF-3→SF-100 scaling and the
  reducing-vs-non-reducing crossover.
- **Worst-case optimal joins** — Ngo, Porat, Ré, Rudra (PODS 2012, theory);
  Freitag et al. (VLDB 2020, practical in Umbra). *Verdict:* **wrong shape for
  TPC-H — do not build.** WCOJ wins on *cyclic*/graph queries where binary joins
  explode intermediates; TPC-H is acyclic PK-FK and even Umbra falls back to
  binary joins on it. Frequently cited as a cure-all; a distraction for us.
- **Row encoding for collision-free multi-key keys** — arrow-rs
  `arrow_row::RowConverter` (Apache Arrow Rust blog, 2022); used by DataFusion to
  >3x multi-column sort. *Verdict:* **do this now — it's a correctness fix, not
  just perf.** We fold multi-key joins to one i64 hash, which is
  collision-*unsafe*. `arrow_row::RowConverter` is **already in our dep tree**,
  gives a canonical collision-free composite key, and is the exact primitive
  DuckDB/DataFusion/Polars use. A few dozen lines; removes a hazard.

## 4. Production engines on our substrate

- **Apache DataFusion** (Lamb et al., SIGMOD 2024) — **our exact substrate
  (arrow-rs).** Pull-based streaming: each `ExecutionPlan` returns a
  `SendableRecordBatchStream` of `RecordBatch` morsels; parallelism via
  `RepartitionExec`; mature `HashJoinExec` (with join filters + dynamic
  filter/bloom pushdown), two-phase `AggregateExec`, projection/predicate
  **pushdown optimizer rules**, and `RowConverter` multi-key keys. **Almost
  everything in our chain-fusion plan already exists here, on our substrate.**
- **DuckDB** — vectorized push-based morsel engine; thread-local hash tables at
  build sinks merged at pipeline end (confirms our aggregate design is
  state-of-the-art); **late-materialization discipline** (carry row-ids/selection
  through the probe, gather only projected columns, last) = the direct fix for
  our all-column `take`.
- **Velox** (Pedreira et al., VLDB 2022) — composable vectorized C++ execution
  library, no optimizer (mirrors our boundary-once philosophy). Transferable
  ideas: lazy/dictionary vectors (don't materialize a gathered column until
  needed) and runtime adaptive filter reordering. Design reference, not code.
- **Polars new-streaming** — morsel-driven hybrid push/pull engine on Rust async
  state machines, spillable sinks. Proves the target: a Rust, arrow-ish,
  morsel-driven engine at the tier we want **with no JIT**. Matching its
  *architecture* is how we reach parity.

## 5. Decision — the DataFusion fork: LOWER into DataFusion

The strongest strategic finding: since our mission is *measuring substrate gaps,
not owning execution code*, and DataFusion is on our **exact substrate**, the
honest fastest route to a fair Polars-parity baseline is to stop re-deriving
joins that already exist on arrow-rs.

**Decision (2026-07-08): if we take the DataFusion route, LOWER our
`LogicalPlan` INTO DataFusion — do not lift DataFusion operators into our
hand-rolled interpreter.** Concretely:

- lazy-pandas stays the **frontend**: our IR, our optimizer, our pandas semantics.
- Emit a **DataFusion `LogicalPlan`** (or drive the `DataFrame`/`SessionContext`
  API) from our optimized plan — a *lowering* pass analogous to today's
  `translate.py` (which lowers to our JSON plan), but targeting DataFusion.
- DataFusion's physical planner + streaming operators + pushdown rules **execute**;
  Arrow tables still cross the Python boundary once.
- Our hand-rolled `engine.rs` becomes a **reference/probe artifact** — the
  controlled "naive baseline" we measure DataFusion (and our own fusion ideas)
  against, not the thing we grind to parity.

Why lower-into rather than lift-from: lifting operators means owning and
maintaining a fork of DataFusion's join/agg/pushdown code inside our interpreter
— all the cost, little of the leverage. Lowering keeps DataFusion as an
upstream dependency we track, isolates our novel effort to (a) the frontend
lowering and (b) the **probe** (Predicate-Transfer scaling, substrate-gap
instrumentation) — which is the actual deliverable. Hand-rolling is justified
only where we specifically want to probe a gap DataFusion doesn't expose.

## Ranked actions

| # | Technique | Impact on join gap | Cost | Verdict |
|---|---|---|---|---|
| 1 | Row-format composite keys (`arrow_row`) | Med (+ **correctness fix**) | Very low | **Do first** — it's a bug fix |
| 2 | Projection pushdown + **late materialization** into joins | **Very high** | Low–med | Do now (checkpoint #1) |
| 3 | Morsel streaming (not operator-at-a-time) | **Very high** | Med | Do now (checkpoint #2) |
| 4 | Predicate Transfer / LIP (selectivity-gated) | **High** on reducing joins | Med | Scoped probe (we have one) |
| 5 | ROF-lite prefetch buffer at probe | Low–med | Low | Scoped probe (after #2) |
| 6 | Runtime selectivity reordering | Low–med | Low | Steal the idea |
| 7 | Radix-partitioned probe | Low–med (2nd-order) | Med–high | Measure then decide |
| 8 | Worst-case optimal joins | ~0 on TPC-H | High | Out of scope (wrong shape) |
| 9 | Full JIT (LLVM/Cranelift) | Low (memory-bound) | Very high | Out of scope for a probe |
| 10 | Adaptive/permutable compiled machinery | ~0 (no compile latency) | High | Out of scope (keep only #6) |
| 11 | **Lower `LogicalPlan` into DataFusion** | Delivers #2,#3, join stack | Low (integration) | **Strong buy-vs-build; preferred route** |

## 6. Frontend fork — how does the plan get *built*? (lazy-plan now vs eager-capture later)

The DataFusion decision (§5) settles the **backend** (execution). It leaves open
the **frontend**: how a chain of pandas ops becomes the plan we lower. Two levels,
very different in cost. Recording both so the choice is on the record.

### 6a. Lower the existing lazy `LogicalPlan` (what we have) — DO FIRST
The lazy-pandas `LogicalPlan` already *is* "a chain of (deferred) pandas ops
compiled into a plan." Our TPC-H queries are exactly that: deferred ops →
`LogicalPlan` → optimizer → engine. So compiling-a-chain-into-a-plan is already
done; the only new work is a **lowering pass** `LogicalPlan → DataFusion
LogicalPlan`/`DataFrame` API (analogous to today's `translate.py`, which lowers
to our JSON plan). Small, reuses the whole frontend, immediately yields a fair
Polars-parity baseline to instrument. **This is the probe-appropriate slice.**

### 6b. Capture *eager* pandas (`import ... as pd`) — a separate, much larger pivot
The ambitious version: don't require the lazy API — **intercept ordinary eager
`df.merge(...).groupby(...).agg(...)`, trace into a plan behind the scenes, and
execute on DataFusion.** High value (works on *existing* pandas code, not a new
API) but this is a product effort, not a probe deliverable. The hard parts:

1. **Eager observation forces materialization.** pandas is eager — users
   `print(df)`, `if len(df) > 0`, index intermediates, hand them to numpy/sklearn.
   Every observation must trigger a flush → lazy eval with automatic
   materialization + guards (a tracing-JIT shape). This is the core engineering.
2. **Partial coverage → the fallback boundary leaks the win.** A chain accelerates
   only if *every* op maps to DataFusion. Real pandas code is full of ops with no
   clean relational form (`apply` with arbitrary Python, `stack`/`unstack`,
   MultiIndex). One fallback op materializes and pays the boundary; deciding where
   to cut is most of the work.
3. **The index.** pandas has a first-class row index; DataFusion (relational)
   doesn't. Round-tripping index semantics is a real modeling problem.
4. **Bit-for-bit semantics.** DataFusion's null/type/ordering/overflow edge cases
   differ from pandas. Exact-compat across the API is what has sunk most
   "pandas on X" efforts.

**Prior art — validated AND occupied:** FireDucks (NEC) does exactly transparent
pandas-capture → IR → compiled backend (`import fireducks.pandas as pd`); Ibis
does the *explicit* dataframe API → many backends incl. DataFusion; Modin does
pandas API → distributed. So the idea is proven and has well-funded incumbents —
a product here needs a differentiator (the obvious one: true pandas-API compat
that Ibis doesn't attempt, i.e. exactly FireDucks' hard problem).

### MEASURED (2026-07-08) — 6a built, and it lands at Polars parity
Built `benchmarks/translate_datafusion.py` (the 6a lowering pass) and ran the
full suite. **All 22 TPC-H queries lower into DataFusion and validate bit-correct
vs DuckDB** (SF-0.5 and SF-1). Benchmarked execute-only vs Polars at SF-3
(boundary paid once via `from_pandas`/registration, like Polars; best-of-5):

**geomean PL/DF = 0.95x — Polars parity — vs the hand-rolled engine's 0.15x.**
Same lazy-pandas frontend; the only change is the backend. DataFusion *beats*
Polars on 9 queries (q19 3.69x, q13 2.95x, q12 1.86x, q1 1.62x, q20 1.62x, q22
1.51x, q17 1.25x, q7 1.11x, q9 1.04x) and is near-parity on most others. Sole
catastrophic outlier: **q21 0.14x** (5.7s vs 0.8s) — a *frontend plan-shape*
problem (our q21 lowers to a giant inner self-join on `l_orderkey` where Polars
uses a semi/anti formulation), NOT a DataFusion execution weakness; it drags the
naive totals ratio to 0.46 while the outlier-robust geomean stays 0.95.

This is the probe's headline result: **the join-heavy gap vs Polars was never
architectural.** A mature arrow-rs engine on our exact substrate, driven by our
own optimized plan, reaches Polars parity — confirming the "buy vs build"
finding and retiring the hand-rolled chain-fusion grind. What the lowering
needed (all mechanical): strip pandas schema metadata on registration (it poisons
DataFusion's `join_selection` optimizer); `substring` (3-arg) not `substr`;
`F.count(x).distinct().build()` for n_unique; cross join via `join_on` with no
predicates; and pandas-faithful join suffixing (rename right keys to temps,
suffix overlapping non-key columns, drop/rename per `on` vs `left_on/right_on`)
because DataFusion neither auto-suffixes nor reliably coalesces `on=` keys when an
input is itself a join result. Note: this route makes the `arrow_row` multi-key
correctness fix **moot** — DataFusion does multi-key joins natively/correctly.

Open follow-up: q21's plan shape (make our optimizer emit a semi/anti-join for
the correlated-exists pattern instead of a materialized self-join).

### Decision on the record (2026-07-08)
- **Now (probe):** 6a — lower the existing `LogicalPlan` into DataFusion. Cheap,
  reuses the frontend, gives the baseline to measure.
- **Deferred (product pivot):** 6b — eager-pandas capture. A FireDucks-sized
  commitment; pursue only if the mission changes from "quantify substrate gaps"
  to "ship a faster pandas." Not on the probe path.

## Sources
- MonetDB/X100 — Boncz, Zukowski, Nes, CIDR 2005 — https://www.cidrdb.org/cidr2005/papers/P19.pdf
- Neumann, Compiling Efficient Query Plans, VLDB 2011 — http://www.vldb.org/pvldb/vol4/p539-neumann.pdf
- Kersten et al., Compiled vs Vectorized, VLDB 2018 — https://www.vldb.org/pvldb/vol11/p2209-kersten.pdf
- Leis et al., Morsel-Driven Parallelism, SIGMOD 2014 — https://db.in.tum.de/~leis/papers/morsels.pdf
- Menon et al., Relaxed Operator Fusion, VLDB 2017 — https://db.cs.cmu.edu/papers/2017/p1-menon.pdf
- Kohn, Leis, Neumann, Adaptive Execution of Compiled Queries, ICDE 2018 — https://15721.courses.cs.cmu.edu/spring2019/papers/19-compilation/kohn-icde2018.pdf
- Balkesen et al., Sort vs Hash Revisited, VLDB 2013 — http://www.vldb.org/pvldb/vol7/p85-balkesen.pdf
- Bandle, Giceva, Neumann, To Partition or Not, SIGMOD 2021 — https://db.in.tum.de/~bandle/papers/bandle-partitionVsNonPartition.pdf
- Zhu et al., Looking Ahead / LIP, VLDB 2017 — https://www.vldb.org/pvldb/vol10/p889-zhu.pdf
- Yang et al., Predicate Transfer, CIDR 2024 — https://www.cidrdb.org/cidr2024/papers/p22-yang.pdf
- Robust Predicate Transfer, SIGMOD 2025 (arXiv) — https://arxiv.org/pdf/2502.15181
- Freitag et al., Worst-Case Optimal Joins, VLDB 2020 — https://www.vldb.org/pvldb/vol13/p1891-freitag.pdf
- Lamb et al., Apache Arrow DataFusion, SIGMOD 2024 — https://dl.acm.org/doi/10.1145/3626246.3653368
- Pedreira et al., Velox, VLDB 2022 — https://dl.acm.org/doi/10.14778/3554821.3554829
- Apache Arrow Rust multi-column sort / row format, 2022 — https://arrow.apache.org/blog/2022/11/07/multi-column-sorts-in-arrow-rust-part-1/
- DuckDB push-based execution — https://zz-jason.github.io/posts/duckdb-execution-model/
- Polars streaming engine — https://pola.rs/posts/polars-in-aggregate-dec25/

**Verification note:** all papers matched to a primary source via search. Two
cited from knowledge without pulling the exact PDF this session (both confidently
real): Ngo/Porat/Ré/Rudra WCOJ (PODS 2012, foundational) and Menon et al.
Permutable Compiled Queries (VLDB 2020, DOI 10.14778/3425879.3425882). The two
2025 follow-ups are preprints, not yet final-venue.
