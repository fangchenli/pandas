# Lazy Execution for pandas — Design Proposal

| | |
|---|---|
| **Status** | Experimental prototype, seeking maintainer feedback |
| **Branch** | [`fangchenli/pandas@lazy-pandas`](https://github.com/fangchenli/pandas/tree/lazy-pandas) |
| **Author** | Fangchen Li |
| **Discussion** | (not yet opened — this document is the starting point) |

## Abstract

This branch prototypes an opt-in lazy execution mode for pandas: queries are
built as logical plans, optimized (predicate pushdown, projection pruning,
filter fusion, TopK, …), and executed through either pandas itself or an
array-native physical engine with Arrow/NumPy kernels, streaming, and
spill-to-disk. It is a pure addition — no existing pandas behavior changes —
with a single integration point (`DataFrame.select()`) and everything else
isolated under a new `pandas.lazy` module.

The prototype is complete enough to evaluate the design: ~30k lines of
implementation, ~1,500 tests (including an eager↔physical equivalence suite),
and an honest benchmark story (some 2–10x wins over eager pandas; still well
behind Polars in several categories — see [Status](#status-and-performance)).

**This document is a request for direction, not a merge request.** See
[Positioning and open questions](#positioning-and-open-questions) — in
particular the positioning question, which the author considers the real
one.

## Motivation

A growing share of the "pandas is slow / pandas can't handle my data" exodus
to Polars and DuckDB is not about per-operation speed — pandas' Cython and
Arrow paths are competitive — but about what eager evaluation structurally
cannot do:

1. **No query optimization.** `df[df.a > 0][["a"]]` materializes the full
   filtered frame before dropping columns. Users hand-optimize operation
   order; a planner should do it.
2. **No multi-step fusion.** Each operation allocates an intermediate result.
   Chained filters, filter→compute→filter pipelines, and `sort().head(n)`
   all do redundant work.
3. **No deferred I/O.** `read_parquet` reads everything before the first
   filter runs. Predicate/projection pushdown into file scans requires
   knowing the query before reading.
4. **No out-of-core story.** Larger-than-memory workloads leave the library
   entirely (Dask, DuckDB, Polars streaming).

pandas already has the substrate these engines are built on — Arrow-backed
columns, nullable dtypes, Copy-on-Write, a mature type system. What is
missing is the layer between the API and execution: a logical plan. This
prototype builds that layer to answer, with working code and measured
numbers, whether it belongs in pandas.

### Why not "just use Polars"?

Users with existing pandas codebases, the pandas API muscle memory, and the
pandas ecosystem (sklearn, statsmodels, matplotlib, …) pay a real migration
cost. An opt-in lazy mode lets them keep pandas interoperability and adopt
deferred execution only where it pays — one pipeline at a time, with
`collect()` returning an ordinary `DataFrame`.

A caveat this document should not gloss over: the prototype keeps pandas
*values and interoperability*, not full pandas semantics — several defaults
deliberately follow Polars/Arrow conventions instead (see
[Semantics](#semantics-what-is-kept-and-what-deviates)). How far "pandas
semantics" should be a hard requirement is part of the positioning question.

## Goals and non-goals

**Goals**

- Opt-in, zero impact on eager pandas (additive API only)
- Value-compatible results with eager pandas, enforced by an equivalence
  test suite (known divergences are tracked as strict xfails — see
  [Semantics](#semantics-what-is-kept-and-what-deviates))
- Real optimizer wins: pushdown, pruning, fusion, TopK, early termination
- Out-of-core capability: streaming batches + disk spill
- Lazy file scans (Parquet, CSV) with predicate/projection pushdown

**Non-goals (for this prototype)**

- Distributed execution
- Replacing eager pandas or changing any default
- API stability — naming and shape are exactly what this review is for

## Proposed API

Two entry points produce a `LazyDataFrame`:

```python
import pandas as pd
from pandas.lazy import col, lit, when, scan

ldf = df.select()                  # from an in-memory DataFrame
ldf = scan("events/*.parquet")     # from files, reads nothing yet
```

Queries chain Polars-style verbs over an expression API:

```python
result = (
    scan("events/*.parquet")
    .filter((col("status") == "ok") & (col("value") > 100))
    .with_columns((col("value") * 0.01).alias("value_pct"))
    .group_by("region")
    .agg(col("value_pct").sum().alias("total"), col("value").count().alias("n"))
    .sort("total", descending=True)
    .head(10)
    .collect()                     # returns an ordinary pd.DataFrame
)
```

Public surface: `LazyDataFrame`, `col`, `lit`, `when`, `coalesce`, `scan`,
`concat`; verbs `select / filter / with_columns / group_by / agg / join /
sort / head / tail / limit / distinct / set_index / reset_index /
with_row_index`; plus `explain()` and `collect()`.

`collect()` exposes the execution knobs:

```python
ldf.collect()                                  # eager pandas evaluation
ldf.collect(use_physical_planner=True)         # array-native engine
ldf.collect(streaming=True, batch_size=65536)  # iterator of batches
ldf.collect(spill_config=SpillConfig(...))     # out-of-core
ldf.collect(preserve_index=True)               # keep source index labels
ldf.collect(optimize=False)                    # debugging reference path
```

`explain()` shows the plan before paying for it:

```python
print(ldf.explain())                # optimized logical plan (text|tree|json)
print(ldf.explain(physical=True))   # physical plan, marks [BREAKER] nodes
```

## Design overview

```
Expr API → Logical Plan → Optimizer (11 passes) → Physical Planner → Execution
                                                        │
                                  ┌─────────────────────┤
                                  ▼                     ▼
                          pandas evaluator      array-native engine
                          (default, fallback)   (ArrayDict, Arrow/NumPy
                                                 kernels, streaming, spill)
```

Two execution paths share one plan: a pandas-based evaluator (always
available, used as the fallback when a kernel is missing) and a physical
engine that operates on raw arrays with per-operation backend routing
(string/null ops → Arrow; numeric ops follow data; thresholds decide
crossovers). The two engines are held to identical results by an
equivalence suite; the one known divergence (NaN in Arrow aggregation) is
pinned as a strict xfail.

The deeper documents, in reading order for a design review:

1. [ARCHITECTURE.md](ARCHITECTURE.md) — plan/IR/type system, array
   execution, joins (grace hash + sort-merge fallback), streaming, spilling
2. [PLANNING.md](PLANNING.md) — logical plan construction and the physical
   planner: node mapping, materialization boundaries, operator fusion,
   plan-time vs run-time decisions
3. [OPTIMIZER.md](OPTIMIZER.md) — the 11 passes, ordering rationale, and the
   safety rules (lineage tracking, dependency DAGs, required-columns)
4. [KERNELS.md](KERNELS.md) — which op runs on which backend, with
   performance notes
5. [THRESHOLDS.md](THRESHOLDS.md) — the cost-model knobs
   (`pd.set_option("compute.lazy.*")`), calibration, adaptive tuning
6. [ROADMAP.md](ROADMAP.md) — known gaps and what we'd build next

A runnable tour lives in [`examples.py`](examples.py) (stdlib + pandas +
pyarrow only):

```bash
python pandas/lazy/docs/examples.py
```

## Compatibility and dependencies

- **No behavior changes** to eager pandas. The only touch outside
  `pandas/lazy/` is the added `DataFrame.select()` method, pandas option
  registration (`compute.lazy.*`), and build config.
- **pyarrow is not required to import** `pandas.lazy` — all pyarrow imports
  are deferred. The NumPy backend covers arithmetic, comparisons, groupby,
  joins, rolling/fill (Bottleneck-accelerated when present). Arrow is
  required for Parquet scans and is strongly preferred for string/null ops.
- Nullable dtypes with `pd.NA` semantics throughout; physical-engine output
  prefers Arrow-backed dtypes (near zero-copy). Making output dtypes fully
  consistent across operators is a known gap (see
  [Semantics](#semantics-what-is-kept-and-what-deviates)).

## Status and performance

What works today (each backed by tests):

| Area | State |
|---|---|
| Logical plan + optimizer | 11 passes, iterated to fixpoint, plan caching, `explain()` in 3 formats |
| Physical engine | Filter/Project/Aggregate/Sort/TopK/Distinct/Join/Limit/Concat, operator fusion, parallel expressions & join sides |
| Joins | Hash join w/ build-side selection, semi/anti, grace hash (spill), sort-merge fallback on pathological skew |
| Streaming | Batch iterator, early termination for `head()`, pipeline-breaker marking |
| Out-of-core | `SpillConfig` — Arrow IPC spill files, external sort, grace hash join |
| File scans | Parquet (predicate incl. row-group stats + projection pushdown), CSV; glob + fsspec URLs |
| Index contract | Default RangeIndex / `preserve_index=True`, identical between engines |
| Tests | 1,489 in `pandas/tests/lazy`: 1,488 passing + 1 strict xfail pinning a known engine divergence; incl. eager↔physical equivalence suite (12 query shapes × index modes) |

Honest numbers (Apple Silicon; details in
[`../benchmarks/`](../benchmarks/README.md)):

- **vs eager pandas**: chained filters 1.5–2×, filter+`head(N)` 2–10×
  (10×+ on multi-file globs), Arrow string pipelines 2–10×, plus
  streaming/spill enabling workloads eager pandas cannot run. Single simple
  operations are 2–5× *slower* (planning overhead) — lazy is for pipelines.
- **vs Polars**: behind almost everywhere except `str.lower`-style string
  kernels (≈2× faster). Aggregations ~0.2×, joins ~0.15×, sort ~0.2×,
  `head()` is Polars' best case and our worst. [ROADMAP.md](ROADMAP.md)
  itemizes the causes (no fast-path for trivial plans, no cardinality
  estimation, single-threaded sort).

The prototype's claim is not "faster than Polars" — it is that the
architecture is sound, results are value-compatible with eager pandas, and
the optimizer wins are real, with a clear path on the remaining gaps.

## Semantics: what is kept and what deviates

"Keeps pandas semantics" would overstate the current prototype. The honest
breakdown, verified empirically:

**Kept**: values (one known exception below), NA presence, column order and
naming (including pandas' `_x`/`_y` join suffixes), `collect()` returning an
ordinary `DataFrame` that the whole ecosystem accepts.

**Deliberately Polars/Arrow-style instead of pandas-style:**

| Behavior | Eager pandas | Lazy default |
|---|---|---|
| Index after ops | label-preserving | positional `RangeIndex` (`preserve_index=True` opts back in) |
| GroupBy result | keys as sorted index | keys as columns, first-appearance order |
| Nulls from `lag`/window ops | `NaN` + float64 upcast | `pd.NA` + nullable dtype (no upcast) |
| Output dtypes (physical engine) | NumPy dtypes | may be Arrow-backed |

**Known gaps / divergences (bugs, not choices):**

- Arrow-backed groupby aggregation treats float `NaN` as a value, not
  missing: `sum([1.0, NaN])` is `1.0` in eager pandas and the eager lazy
  path but `NaN` through the physical engine. The engines currently
  disagree; the equivalence suite must grow NaN-payload cases.
- Physical-engine output dtypes are not stable across operators and data
  sizes (NumPy-backed for some paths, Arrow-backed for others).
- Duplicate column labels are unsupported and fail with an unhelpful error.
- The optimizer can change observable results for position-dependent
  expressions (`with_row_index` before a pushed-down filter).
- No UDFs (`apply`/`map`) — only the implemented expression vocabulary.

Whether the deliberate deviations are features (modern defaults) or bugs
(breaking the "no migration" promise of position (a)) is part of the
positioning question — if "pandas semantics" is the differentiator, the
defaults above arguably have it backwards.

## Code size and maintenance

| Component | Lines |
|---|---|
| `pandas/lazy/` implementation | ~30k |
| `pandas/tests/lazy/` | ~19k |
| `pandas/lazy/benchmarks/` (not shipped) | ~13k |

All isolated under `pandas/lazy/`; benchmarks and these docs are excluded
from wheels. Note that the layers (IR, logical plan, optimizer, physical
engine) were designed as one system and do not split into independently
useful PRs — how something this size could land is itself one of the open
questions below.

## How to try it

```bash
git clone https://github.com/fangchenli/pandas.git --branch lazy-pandas
cd pandas
# standard pandas dev setup:
mamba env create --file environment.yml && mamba activate pandas-dev
python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true

python pandas/lazy/docs/examples.py        # runnable tour
python -m pytest pandas/tests/lazy -q      # ~1,500 tests, ~2s
```

## Positioning and open questions

### 1. What is lazy pandas *for*? — the real question

An honest self-assessment first: judged as a Polars competitor, the current
prototype is "Polars, but worse" — a similar API over a slower engine (see
[Status](#status-and-performance)). That framing would not justify the
maintenance cost, so the positioning question has to be answered before any
technical one. The candidate positions, with very different bars:

- **(a) A native optimization layer for pandas users.** The comparison
  target is *eager pandas*, not Polars: existing pandas users get pushdown,
  fusion, streaming, and out-of-core for their pipelines while keeping
  pandas values, objects, and the ecosystem, with no migration. The
  prototype clears the performance bar in several categories, but note
  that this position makes *pandas semantics* the differentiator — and the
  current defaults deviate from them (see
  [Semantics](#semantics-what-is-kept-and-what-deviates)); committing to
  (a) likely means flipping those defaults back toward pandas.
- **(b) A genuine competitor to Polars/DuckDB for data pipelines.** This
  requires a real breakthrough, not parity-chasing — something engines that
  require a rewrite structurally cannot offer. A research pass over prior
  art ([COMPETITIVE_RESEARCH.md](COMPETITIVE_RESEARCH.md)) identifies the
  candidate: **transparent lazy capture of existing eager pandas code**
  (validated piecewise by cudf.pandas, LaFP, Dias, Modin) **combined with
  pluggable best-of-breed execution backends via IR** (validated by
  Ibis→Substrait→DuckDB; engine kernels are commoditizing per the
  composable-data-systems thesis). Incumbent API + owned semantics +
  commoditized execution is a position Polars/DuckDB structurally cannot
  take. Absent committing to that composite, (b) is not worth pursuing.
- **(c) Incubation outside the main tree** (a `pandas-lazy` package or
  long-lived experimental branch) until (a)-vs-(b) is settled by usage.

The author's current read: (a) is defensible today; (b) should be treated
as a research question to answer deliberately — not an assumption; the
in-tree vs (c) choice is process, not design. Maintainer views on this
framing are the most valuable feedback this document can get.

### 2. Entry point and API shape

`df.select()` as the lazy entry point has already drawn significant
pushback (it reads as relational "select columns" while actually switching
evaluation modes and return type). Alternatives to weigh:

- `df.lazy()` — Polars precedent, unambiguous about what it does
- top-level constructors only — `pd.lazy.from_dataframe(df)`,
  `pd.lazy.scan(...)`
- `scan()` as the *only* entry point — arguably honest, since lazy
  execution pays off on I/O-rooted pipelines

The expression API itself (`col`/`lit`/`when` + chained verbs) has been
less controversial; whether pandas would prefer `query`/`eval`-style string
expressions remains open.

### 3. Namespace and process — positions, not questions

- **Namespace**: `pandas.lazy`, not `pandas.api.lazy`.
- **Process**: if this moves forward in any form, it will be a PDEP. This
  document is the pre-PDEP draft, circulated with the working prototype
  because concrete code makes a better discussion anchor than an abstract
  proposal.

### 4. How could something this size land?

Merging piece-by-piece is not realistic: the IR, logical plan, optimizer,
and physical engine were designed together, and the early pieces are not
independently useful (a plan/IR layer with no engine helps nobody).
Realistic options:

- one large, explicitly **experimental** merge — module marked unstable,
  API exempt from the deprecation policy until graduation
- incubate out-of-tree (`pandas-lazy`) with pandas reserving the namespace,
  absorbing it only after the positioning question is settled
- keep it a fork/branch as a design probe, harvesting individual ideas
  (e.g. kernel dispatch, threshold calibration) into pandas piecemeal

Feedback is welcome directly to the author; a discussion thread will be
linked from the header once opened.
