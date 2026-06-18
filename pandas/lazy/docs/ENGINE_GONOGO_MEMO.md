# Go/No-Go memo: a new engine "much better" than Polars/DuckDB?

Decision memo from the research spike (deep-research harness: 104 agents, 22
sources, 25 claims adversarially verified, 20 confirmed / 5 refuted, June 2026).
Question: build a from-scratch analytical engine (the Path-B / MLIR
data-centric-compiler thesis from ENGINE_DIFFERENTIATION.md) that is
*materially* better than Polars/DuckDB on complex multi-join queries?

## Verdict: NO-GO on the *speed* thesis (the flexibility thesis is a separate, open question — see pillar 2)

The exciting thesis — "an MLIR query compiler specialized for complex joins,
with Yannakakis/WCOJ join algorithms, beats Polars/DuckDB *by a wide margin on
speed*" — is **not supported by the evidence**. (A *different* thesis —
extensibility / heterogeneous-hardware reach — is where MLIR engines actually
aim; not evaluated here, see pillar 2.) Four verified pillars against the speed
thesis, each fatal on its own:

1. **The architectural lever points at the WRONG regime.** The canonical
   apples-to-apples study (Kersten et al., VLDB 2018, same algorithms/data
   structures/parallelism) and TUM's InkFuse (2024) both find: *compilation
   wins calculation-heavy queries (Q1 +74%); vectorization wins the
   hash-join-heavy queries (Q3, Q4, Q9, Q13)* — i.e. precisely the complex-join
   regime that is our bottleneck. Compiling buys us the regime we're already
   fine in and concedes the one we care about. [Kersten VLDB'18; InkFuse 2024]

2. **The marquee MLIR engine has not *demonstrated* it is much faster than a
   parallel vectorized engine — and its own pitch isn't speed.** (CORRECTED
   2026-06 after pushback that the original 2022 citation was stale.) The 2022
   result — 3.5x over DuckDB, but only single-threaded / no-index / SF1, and
   1.3x *slower* than HyPer — is outdated: LingoDB **added thread-scaling
   parallelism** in the sub-operators work (Jungmair & Giceva, VLDB 2023), so
   "still single-threaded" is wrong. However, the updated picture does not
   rescue the *speed* thesis: (a) no published head-to-head shows
   multi-threaded LingoDB beating multi-threaded DuckDB/Polars on TPC-H; (b) GPU
   remains *"ongoing work"* in LingoDB's own VLDB 2025 paper, not shipped with
   measured wins; (c) LingoDB's 2025 positioning is **extensibility / flexibility
   / future-proofing across heterogeneous hardware** *"without sacrificing
   performance"* — i.e. competitive, NOT dominant-on-speed. So the leading MLIR
   engine doesn't even claim to be much faster than DuckDB; its differentiator
   is flexibility + one-IR-to-CPU/GPU/PIM. [Jungmair et al., PVLDB 15 p2389,
   2022; Jungmair & Giceva, *Declarative Sub-Operators*, PVLDB 16 p3461, 2023;
   Jungmair & Giceva, *Towards Future-Proof Data Processing Systems*, PVLDB 18
   p3988, 2025]

   **Implication (sharpens, not reverses, the verdict):** a bet justified by
   "much faster than DuckDB on complex joins" stays NO-GO — even SOTA MLIR
   engines don't demonstrate it. But a bet justified by "extensible,
   future-proof, heterogeneous-hardware (CPU+GPU) from one IR" is where the MLIR
   frontier actually is — a *different value proposition than beating Polars on
   speed*, and the one that matches the "generic / reusable elsewhere" instinct.
   Open question we could not settle from public sources: current multi-threaded
   LingoDB vs DuckDB at SF-10+ (would need to run LingoDB ourselves).

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

- **NO-GO if the goal is "much faster than DuckDB/Polars on complex joins"** via
  a multi-year MLIR / from-scratch compiler. Wrong regime, no demonstrated moat
  (even SOTA MLIR engines don't claim speed-dominance), compile-latency tax, and
  the key speed ideas are commoditizing into pluggable rewrites.
- **Different question, not settled here:** if the goal were instead an
  **extensible, future-proof, heterogeneous-hardware (CPU+GPU+PIM) engine from
  one IR** — LingoDB's actual thesis — the MLIR frontier genuinely points there.
  That's a flexibility/reach bet, not a speed bet, and would need its own
  go/no-go against a *different* success metric (and ideally running current
  multi-threaded LingoDB ourselves, since public sources don't answer it).
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
