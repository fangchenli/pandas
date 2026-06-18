# lazy-pandas as a substrate probe — project charter

This supersedes the "lazy pandas as a product" framing. A blunt finding settled
it (see `ENGINE_GONOGO_MEMO.md` and the discussion that followed): what was
built is a **new, Polars-style expression API** (`df.select().filter(col(...))
.group_by(...).agg(...).collect()`), *not* the eager pandas method-chain API and
*not* pandas-exact in surface. As a new lazy API that is slower than Polars and
younger, it has **no product justification** — anyone who must learn a new API
and rewrite would pick the faster, mature one. The only product form that would
be justified — *lazy evaluation of the real pandas API, pandas-exact* — is a
different, multi-year semantics project (the hard part Dask/Modin still haven't
solved) that we did not build.

So this document reframes the mission to what the project actually does well.

## Mission

**Use the lazy engine as an instrument to discover, quantify, and upstream the
gaps in the analytics substrate (Arrow, NumPy, pandas, CPython).** The engine is
the probe, not the product. Its job is to *find things* and turn them into
ecosystem contributions — which benefit pandas, Polars, and DuckDB alike, since
they share the substrate.

Under this lens, this session's headline conclusion — "the residual gap is
substrate-bound, ~0.45x, not closable in the plan/kernel layer" — is not a
disappointment. **The precise, workload-grounded map of where and why the
substrate falls short IS the deliverable.**

## Why a lazy engine is a good probe

- **Whole-pipeline stress.** It exercises the substrate across realistic
  end-to-end shapes (TPC-H: scan → filter → multi-join → groupby → sort →
  fusion → parallelism → out-of-core), surfacing gaps that micro-benchmarks
  miss (e.g. cross-operator materialization, the Arrow↔NumPy boundary tax).
- **It forces honest comparison.** Validated bit-exact vs eager pandas and
  against DuckDB/Polars, so a "gap" is a measured delta with a cause, not a hunch.
- **It spans backends.** Routing across numpy/arrow/acero exposes where each
  substrate wins and loses on the *same* workload.

## Discipline (what makes it a probe, not a journal)

1. **Measurement-first, controlled on/off.** Every gap is an A/B with a "why."
2. **Adversarial verification.** Probe before building; refute claims (the
   deep-research spike, the count-distinct/fusion/join reversals this session).
3. **Workload-grounded + scale-aware.** Measure in the mode that ships
   (`use_physical_planner=True`); note where a gap only appears at scale.
4. **THE OBLIGATION — output is upstream.** A gap's value is realized only when
   it becomes a *filed issue, a reproducible benchmark, a PR, or a tool*.
   Internal docs alone don't count. Prioritize by **ecosystem leverage** (a fix
   in Arrow helps everyone; a pandas-only tweak helps fewer).

## Gap inventory (the probe's output to date)

Status: **Q** = quantified with controlled measurement; **U?** = upstreamable;
**lev** = ecosystem leverage. Evidence in the linked docs.

| # | Gap | Status | U? | Lev | Evidence |
|---|---|---|---|---|---|
| A | **Arrow `group_by` does not parallelize** (single-threaded; 124ms@1core→133@8). Partition-parallel wrapping beats it ~2.5x and beats Polars ~1.9x on high-cardinality. | Q | **yes (strongest)** | high | `PARALLEL_GROUPBY_SCOPE.md` |
| B | **Acero non-competitive for in-process filtered pipelines** — per-node overhead, FilterNode compaction, no zero-copy NumPy handoff (large-output round-trip). | Q | yes (document limitation) | med | `QGAP_DECOMP.md`, `MATERIALIZATION_EXPERIMENT.md` |
| C | **Arrow `count_distinct` slower than pandas' Cython** on high-card. | Q | yes | med | `QGAP_DECOMP.md` |
| D | **No predicate-transfer / Bloom-semijoin primitive** in Arrow/Acero (RPT shows it as 2 operators in DuckDB). | partial | tool gap | med | `PREDICATE_TRANSFER_PROBE.md` |
| E | **`pd.merge` is GIL-bound** (sub-GIL contention, measured even free-threaded). | Q | pandas FT roadmap | med | `PERF_CEILING.md` |
| F | **Free-threaded pandas ~2.6x slower single-threaded** today. | Q | CPython/pandas FT input | med | `PERF_CEILING.md` |
| G | **No core fused-expression evaluation** (numexpr/Gandiva exist, not in core). | scoping | tool gap | low-med | `ENGINE_DIFFERENTIATION.md` |
| H | **`np.argsort` single-threaded** (motivated `lazy_radix`). | Q | tool/algo | low | `lazy_radix.pyx` |
| I | **Arrow `string_view` kernel coverage** (at-scale/string gains). | planned | **PR planned** | high | `STRING_VIEW_CONTRIBUTION_PLAN.md` |
| J | **Arrow crash reports** (process-aborting hash-agg row table). | filed | **filed** | high | `UPSTREAM_ISSUES.md` |
| K | **Materialization-between-operators + Arrow↔NumPy boundary tax** — the structural substrate cost. | Q | architectural | high | `PERF_CEILING.md` |

## Prioritization

1. **Already-moving, high-leverage:** string_view PR (I), crash reports (J) —
   keep them going.
2. **Highest new candidate:** parallel hash-aggregate (A) — we have a design, a
   working prototype that beats Polars, and bit-exact validation. Best
   evidence-to-impact ratio. → first upstream artifact (plan:
   `UPSTREAM_PARALLEL_GROUPBY_PLAN.md`), verification-gated, not yet filed.
3. **Document-as-limitation:** Acero (B), count_distinct (C) — issues with
   benchmarks, lower build cost.
4. **Feed the FT roadmap:** pd.merge GIL contention (E), FT single-thread
   overhead (F) — data points for the CPython/pandas free-threading effort.
5. **Tool proposals (scope before build):** Bloom-semijoin (D), fused
   expressions (G).

## What this is NOT

- Not a product; nobody must adopt the engine.
- Not a Polars/DuckDB speed competitor (proven unwinnable, and beside the point).
- Not a place to ship a slower-Polars new API as if it were pandas.

The asset is **a validated map of substrate gaps + a pipeline of upstream
contributions** — see `PERF_CAMPAIGN_SUMMARY.md` for the full evidence trail.
