# Go/No-Go memo: a new engine "much better" than Polars/DuckDB?

Decision memo from the research spike (deep-research harness: 104 agents, 22
sources, 25 claims adversarially verified, 20 confirmed / 5 refuted, June 2026).
Question: build a from-scratch analytical engine (the Path-B / MLIR
data-centric-compiler thesis from ENGINE_DIFFERENTIATION.md) that is
*materially* better than Polars/DuckDB on complex multi-join queries?

## Verdict: NO-GO on the MLIR-compiler / from-scratch-engine thesis

The exciting thesis — "an MLIR query compiler specialized for complex joins,
with Yannakakis/WCOJ join algorithms, beats Polars/DuckDB by a wide margin" —
is **not supported by the evidence**. Four verified pillars, each fatal on its
own:

1. **The architectural lever points at the WRONG regime.** The canonical
   apples-to-apples study (Kersten et al., VLDB 2018, same algorithms/data
   structures/parallelism) and TUM's InkFuse (2024) both find: *compilation
   wins calculation-heavy queries (Q1 +74%); vectorization wins the
   hash-join-heavy queries (Q3, Q4, Q9, Q13)* — i.e. precisely the complex-join
   regime that is our bottleneck. Compiling buys us the regime we're already
   fine in and concedes the one we care about. [Kersten VLDB'18; InkFuse 2024]

2. **The marquee MLIR engine has not beaten a parallel vectorized engine.**
   LingoDB beats DuckDB 3.5x only single-threaded / no-index / SF1, is itself
   **1.3x slower than HyPer**, and is *still single-threaded* — parallelism and
   GPU are explicitly unshipped future work. There is no demonstration of an
   MLIR compiler beating multi-threaded DuckDB/Polars, the engines we must beat.
   [Jungmair et al., PVLDB vol15 p2389, 2022]

3. **JIT compile latency is a real, expensive, already-solved-by-others
   problem.** Even at TPC-H SF1, query compile time often *exceeded* execution
   time; HyPer/Umbra needed adaptive compilation (custom IR + fast VM, escalate
   to LLVM only when justified) to fix it. A fresh whole-query MLIR compiler
   inherits this in a worse form and must redo engineering the competition has
   already done. [Neumann, PVLDB vol14 p3207; InkFuse]

4. **The moat is already collapsing.** The genuinely promising asymptotic
   ideas for acyclic analytical joins — Yannakakis-style semijoin reduction and
   predicate transfer — are being absorbed into existing engines as **plan-level
   rewrites that need NO new engine**: Robust Predicate Transfer (~1.5x in
   DuckDB), Yannakakis+ (avg 2.41x, validated in 4 *unmodified* engines incl.
   DuckDB/Postgres/Spark, beating native plans on 160/162 queries). The best
   ideas ride on top of vectorized engines; they don't require building one.
   [RPT, SIGMOD'25 arXiv:2502.15181; Yannakakis+, SIGMOD'25 arXiv:2504.03279]

Worst-case-optimal joins (Leapfrog Triejoin) are real but **niche**: the proven
advantage is on cyclic / skewed / graph-pattern queries, demonstrated in a
Datalog system — *not* a measured TPC-H win, since q7/q9/q21 are largely acyclic
where WCOJ offers little. [Veldhuizen, ICDT 2014]. GPU/heterogeneous as a
differentiator is **unsupported** by the evidence gathered — MLIR→GPU is
architectural intent with no shipped numbers; no verified GPU-vs-DuckDB
complex-join comparison was found.

(Verification also *refuted* the opposite over-claims — "compilation
conceptually dominates vectorization" and "the industry has decisively moved to
vectorization" both failed 0-3 — so the honest reading is genuinely "no uniform
winner," which is itself fatal to a bet that needs a *wide* margin.)

## The reframe that DOES fit our constraints

The spike's own narrow recommendation was "a semijoin-reduction optimizer pass
that lowers to DuckDB/Polars" — but that brushes against the stated exclusion
("delegating to DuckDB is a different project"). The sharper, constraint-
respecting reading:

**The "much better on complex joins" lever is ALGORITHMIC, not architectural —
and it lives in a layer we already own: the optimizer.**

Predicate transfer / Yannakakis-style semijoin reduction is a **LogicalPlan
rewrite**: before any join runs, pass Bloom filters between tables so each
relation is reduced to just the rows that survive the full join graph. That
attacks the *intermediate-result explosion* we measured directly in the
prefix-collect blowups (q3 raw join chain 1597ms) — and it helps **our own
execution backend** (smaller inputs into our pd.merge / lazy_join), with **no
new engine and no delegation**. The same rewrite is backend-agnostic, so it's
robustly useful and not throwaway. Documented margin: 1.5-2.4x on join-heavy
queries. Effort: weeks-to-months in the optimizer we already have, not a
multi-year compiler.

This is the honest resolution: there is **no** path that is simultaneously
(much better than DuckDB) AND (a from-scratch owned engine) AND (evidence-
backed) — the moat isn't there. But there IS an owned, evidence-backed lever
for our actual bottleneck (joins), and it's an optimizer pass, not an engine.

## Recommendation

- **NO-GO** on the multi-year MLIR / from-scratch compiler. Wrong regime, no
  demonstrated moat, compile-latency tax, and the key ideas are commoditizing
  into pluggable rewrites.
- **Candidate GO (small, owned, on our actual gap):** a predicate-transfer /
  semijoin-reduction optimization pass in the lazy-pandas optimizer, feeding our
  existing execution. Validate with a probe first (does it shrink q3/q9/q21 join
  inputs and beat our current plan?) — same measurement-first discipline.
- **Park:** WCOJ (revisit only for cyclic/graph workloads), GPU (no evidence of
  a single-IR CPU+GPU win), MLIR compiler.
- **Time-sensitivity:** RPT/Yannakakis+/COM are all 2024-25; the trend is *away*
  from custom engines toward pluggable rewrites — which further argues against a
  multi-year engine bet.

## Open questions (if revisited)
- LingoDB TODAY vs *multi-threaded* DuckDB/Polars at SF10-100 with indexes — the
  2022 single-threaded/SF1 numbers can't answer the real question.
- Are q7/q9/q21 purely acyclic (cheap Yannakakis+/RPT wins) or do they hide
  cyclic/skew patterns where WCOJ's edge would actually appear?
- Does our-optimizer + predicate-transfer + our-execution capture most of the
  hypothetical compiler's win — making the engine bet redundant?
