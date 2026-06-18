# What would make a new engine MUCH better than Polars/DuckDB?

The Path-B question (docs/PERF_CEILING.md), sharpened: building "another good
vectorized columnar engine" is pointless — Polars and DuckDB are excellent and
near the hardware limits. A new engine is only worth it if it's *materially*
better, not on par. This doc separates the non-differentiators from the genuine
step-change angles, grounded in where our measured gap actually is (joins /
complex multi-operator queries) and in the published research.

## Gandiva: useful, but NOT the differentiator

Gandiva is Arrow's **LLVM expression compiler**. It JIT-compiles a tree of
*scalar expressions* (arithmetic, comparisons, conditionals, some functions)
to native code that runs over Arrow batches. What it is and isn't:

- **Is:** fast, fused *expression* evaluation. `a*(1-b)*(1+c)` compiles to one
  tight loop with no intermediate arrays — a generic version of the hand-fused
  expression kernels we already wrote for q1/q6.
- **Is NOT:** a query compiler. It does **not** compile joins, aggregations, or
  fuse *across* operators. It's projection/filter-scoped, and it's not wired
  into Acero's join/aggregate nodes.

So Gandiva would bring our *expression* evaluation to Polars parity (Polars
already SIMD-vectorizes and partially fuses expressions) — a coverage win, not
superiority. **It cannot make us better than Polars**, because expression eval
isn't the gap; joins and whole-pipeline execution are. If we build a compiler,
Gandiva is at most inspiration for the expression layer — we'd want full
*query* compilation, of which expression codegen is one small piece.

## What does NOT make us "much better" (the trap)

- A faster vectorized push-based engine in Rust/C++ → that's literally Polars /
  DuckDB. Better engineering of the same model yields ~parity, not a leap.
- More SIMD / better memory layout → they already do this; near bandwidth floor.
- Free-threading parallelism → probed, doesn't even unlock our joins
  (PERF_CEILING.md Path C).

## The genuine step-change angles

### 1. Whole-query data-centric COMPILATION (the model advantage)
Polars and DuckDB are *vectorized-interpreted*: they push batches through
pre-compiled operators, materializing an intermediate vector between each
operator. The alternative is **data-centric compilation** (HyPer, Neumann 2011;
Umbra, Neumann/Freitag 2020): generate machine code per query so a tuple flows
filter→join-probe→aggregate **entirely in registers**, touching memory only for
hash tables — no per-operator intermediate materialization at all.

- **Why it can beat them:** on complex multi-operator queries (TPC-H q7/q9/q21
  — exactly our worst), compilation eliminates the per-batch materialization and
  dispatch the vectorized model still pays. This is the produce/consume model.
- **Honest caveat:** Kersten et al. (VLDB 2018, "Everything You Always Wanted
  to Know About Compiled and Vectorized Queries…") found compiled ≈ vectorized
  overall, each winning in different regimes (compiled: compute-heavy, complex;
  vectorized: scan/bandwidth-bound). DuckDB *chose* vectorized deliberately
  (compile latency, debuggability). So compilation is "much better" **only for
  the complex-query regime** — which happens to be our gap, but it is not a
  blanket win.

### 2. Better JOIN ALGORITHMS (asymptotic, hits our exact gap)
Polars and DuckDB execute joins as **binary hash-join trees** — and the killer
on many-table queries is *intermediate result explosion* (a join chain
materializes huge mid-results even when the final output is small; we saw this
in our prefix-collect experiments). Two research lines beat this:

- **Yannakakis-style semijoin reduction** (Yannakakis 1981; recent systems work
  incl. DuckDB-team "robust join" / SIPS, 2023-24): for acyclic queries (most
  of TPC-H), pre-reduce each relation via semijoins so no intermediate exceeds
  the final output — runtime linear in (input + output). Asymptotically avoids
  the explosion binary plans suffer.
- **Worst-case-optimal joins** (Ngo et al.; LeapFrog Triejoin, Veldhuizen 2014):
  for cyclic/complex joins, provably better than any binary plan.

This is a *genuine* "different and better," not faster engineering — and it
directly attacks the join chain that our decomposition pinned as the dominant
residual cost (q3/q9/q21). DuckDB has only partial SIPS; a join engine built
around Yannakakis + WCOJ from the start could be materially better on join-heavy
analytics.

### 3. Heterogeneous hardware (GPU / accelerators)
Polars/DuckDB are CPU. A columnar engine that compiles to **GPU** (cf. cuDF,
HeavyDB, Theseus) is 10-100x on large scan/join/aggregate workloads that stream
through device memory. A genuine step-change for the right (large, GPU-resident)
workloads; data-transfer-bound and pointless for small data. Only "much better"
if the engine can *target* heterogeneous hardware from one query representation
— which points directly at the generic-substrate angle below.

## The "generic, reusable elsewhere" angle → MLIR

Your second instinct is the strongest unifier. Build the engine as a
**data-centric query compiler on MLIR** (LLVM's multi-level IR). This is exactly
what LingoDB does (Jungmair/Giceva, TUM, "A Practical Approach to Query
Compilation with MLIR", VLDB 2024): relational + finer-grained "sub-operator"
dialects lowered through MLIR to native code, designed to be extensible and to
target heterogeneous hardware.

Why MLIR makes it both *much better* AND *generic*:
- It is the substrate for **whole-query compilation (#1)** — the IR *is* the
  fused pipeline; you get the produce/consume model for free.
- One representation lowers to **CPU and GPU (#3)** — heterogeneous targets from
  a single compiler, which no current dataframe engine offers cleanly.
- It is **reusable beyond dataframes**: MLIR is a general compiler framework
  (ML, HPC, tensor algebra all live there). A relational/sub-operator dialect +
  lowerings is infrastructure other projects (and the broader LLVM ecosystem)
  can build on — the engine becomes a reusable *compiler*, not a one-off
  runtime. The "sub-operator IR" abstraction is the generic, reusable unit.
- Advanced join algorithms (#2) can be expressed as IR transforms/rewrites
  rather than hand-coded operators.

So the sweet spot that satisfies "much better" *and* "generic/reusable" is:
**a data-centric query compiler on MLIR, with semijoin/WCOJ join planning and a
GPU lowering path** — not a hand-built vectorized runtime, and not Gandiva.

## Honest risk assessment

- This is **frontier, research-grade** work. Umbra is Neumann's life's work;
  LingoDB is a multi-year TUM research effort. "Much better than DuckDB" is
  genuinely hard — DuckDB is world-class.
- Where a focused effort *can* win: the **complex-join regime** (compilation +
  Yannakakis/WCOJ), because that's (a) where the vectorized model is weakest and
  (b) exactly our measured gap. A narrow, deep bet on "the fastest complex-join
  analytical engine via a compiling MLIR backend" is more defensible than
  "beat DuckDB at everything."
- The pandas connection: lazy-pandas already has the front end (LogicalPlan,
  optimizer, eager-semantics contract). The new engine is a *backend* — lower
  the existing LogicalPlan to the MLIR dialect. That reuses everything built and
  keeps the eager-pandas API, so this is additive, not a from-scratch product.

## Recommended next step (research spike, not a build)
Before committing: a literature + feasibility spike on (1) MLIR query-compiler
prior art (LingoDB sub-operators, the MLIR `linalg`/`sparse_tensor` story) and
(2) Yannakakis/SIPS join processing — to decide whether the "compiling MLIR
engine specialized for complex joins" thesis holds and where it would beat
DuckDB by a margin worth the multi-year effort. Output: a go/no-go design memo.
