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
implementation, 1,488 tests (including an eager↔physical equivalence suite),
and an honest benchmark story (some 2–10x wins over eager pandas; still well
behind Polars in several categories — see [Status](#status-and-performance)).

**This document is a request for direction, not a merge request.** See
[Open questions](#open-questions-for-maintainers).

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
cost. An opt-in lazy mode lets them keep pandas semantics and
interoperability and adopt deferred execution only where it pays — one
pipeline at a time, with `collect()` returning an ordinary `DataFrame`.

## Goals and non-goals

**Goals**

- Opt-in, zero impact on eager pandas (additive API only)
- Same results as eager pandas (nullable/NA semantics included), enforced by
  an equivalence test suite
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
crossovers). Results are identical by construction and by test.

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
  uses Arrow-backed dtypes (near zero-copy) by design.

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
| Tests | 1,488 in `pandas/tests/lazy`, incl. eager↔physical equivalence suite (12 query shapes × index modes) |

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
architecture is sound, the semantics are pandas', and the optimizer wins are
real, with a clear path on the remaining gaps.

## Code size and maintenance

| Component | Lines |
|---|---|
| `pandas/lazy/` implementation | ~30k |
| `pandas/tests/lazy/` | ~19k |
| `pandas/lazy/benchmarks/` (not shipped) | ~13k |

All isolated under `pandas/lazy/`; benchmarks and these docs are excluded
from wheels. If the direction is approved, the integration plan would be
reviewed piece-by-piece (plan/IR → optimizer → physical engine → scans),
not as one PR.

## How to try it

```bash
git clone https://github.com/fangchenli/pandas.git --branch lazy-pandas
cd pandas
# standard pandas dev setup:
mamba env create --file environment.yml && conda activate pandas-dev
python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true

python pandas/lazy/docs/examples.py        # runnable tour
python -m pytest pandas/tests/lazy -q      # 1,488 tests, ~2s
```

## Open questions for maintainers

1. **Is there appetite for lazy execution in pandas at all**, or is the
   ecosystem position "that's Polars/DuckDB territory"? This is the
   threshold question; everything below assumes a qualified yes.
2. **API shape** — is a Polars-style expression API (`col`, `lit`,
   chained verbs) acceptable, or should this look more like
   `DataFrame.query`/`eval` string expressions? Is `df.select()` the right
   entry point, or e.g. `df.lazy()`?
3. **Namespace** — `pandas.lazy` vs `pandas.api.lazy` vs a separate
   distribution (`pandas-lazy`) that pandas could later absorb?
4. **Process** — should this become a PDEP? The author is happy to convert
   this document; circulating the working prototype first seemed more
   useful than an abstract proposal.
5. **Scope of a first merge** — if the direction is right, what is the
   smallest piece worth reviewing on its own (e.g. plan/IR + optimizer +
   eager evaluator, deferring the physical engine)?

Feedback welcome by issue/discussion on the fork, or directly to the author.
