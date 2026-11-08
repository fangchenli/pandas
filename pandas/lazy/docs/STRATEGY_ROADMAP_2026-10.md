# Strategy roadmap (October 2026): Arrow-native pandas, eager first, lazy last

Follows from `REVISIT_2026-10.md`. The conclusion there: pandas cannot and
should not enter the engine race; what it can do is remove the taxes its users
actually pay, and reach Polars-class speed only by owning the API and renting
the engine. This document turns that into an ordered plan where **every phase
pays on its own even if the next never ships**, which is the property a
maintainer-capacity effort needs.

Why this order (our own measurements): eager pandas is 20–90x behind Polars;
Polars' streaming engine beats its own eager-style materializing engine by only
1.4x (SF-1) to 1.7x (SF-3). Most of the gap is substrate and per-operation
execution, not cross-operation fusion. The eager steps also sidestep the
semantics problem that sank every lazy drop-in.

## Phase 0 — close the Arrow fallbacks (now, continuous)

Already running: the fallback registry (`NUMPY_FALLBACK_REGISTRY.md`, 24
AF-series items, public tracker pandas-dev/pandas#66626), re-synced against
pyarrow 26.

- [ ] Keep landing AF items; prioritize by **trigger frequency on real
      workloads**, not by ease.
- [ ] Add an opt-in `PerformanceWarning` (or counter) when an Arrow-backed
      operation leaves Arrow for NumPy/object/Python loop. Gives users a
      signal and gives the registry real-world frequencies.
- [ ] Gate metric: fraction of operations in a workload corpus (pandas' own
      benchmarks + a notebook corpus) that stay in Arrow end to end.

## Phase 1 — Arrow-resident frames: the residuals (no new manager)

**Correction (2026-10-21):** the storage layer for Arrow columns is already
Arrow-native. An Arrow-backed column is a 1D, non-consolidating
`ExtensionBlock` whose values *are* the `ArrowExtensionArray`; the BlockManager
is placement bookkeeping around it, and a `Table`-shaped manager would change
nothing for column-wise work (the `ArrayManager` experiment, tried and removed
in 2.x, is the precedent for exactly that non-result). CoW (pandas 3.0) already
made Arrow immutability the model. So Phase 1 is not a storage rewrite; it is
the short list of places where NumPy-shaped machinery still wraps Arrow
columns:

- [ ] **Index engines.** An Arrow-backed `Index` falls to
      `libindex.ExtensionEngine`, the generic EA lookup path
      (`pandas/core/indexes/base.py` `_engine`; the typed libindex engines are
      noted in-tree as "much more performant"). Lookup and alignment are on the
      hot path of every binary op, `loc`, and index-based merge. Probe the gap
      vs the typed engines; fix = a typed engine over Arrow buffers (or a
      hash index built once from the Arrow array).
- [ ] **Frame-level algorithms that are NumPy-shaped around Arrow columns** —
      `merge` (factorize keys → libjoin indexers → per-block `EA.take`),
      `groupby` (EA `_groupby_op` → pyarrow `group_by` for some aggs, else
      masked/NumPy), `sort_values`, `concat`, `reindex`. These are Phase 2's
      target list; Phase 1 only *measures* where each leaves Arrow and what it
      costs (the registry already does this per EA op; extend it to frame ops).
- [ ] **EA-internal costs**, already tracked as AF items: chunk handling and
      `combine_chunks` copies (AF-22), 2 GiB offset overflow (AF-23), scalar
      `__getitem__` and `np.asarray` paths in generic code.
- [ ] **Zero-copy frame ↔ `pyarrow.Table`.** Already near-free for all-Arrow
      frames (`pa.table` over `_pa_array` per column); verify the PyCapsule
      import path and that index columns round-trip. This is the 0.01 ms engine
      boundary later phases rely on.
- [ ] The 2D edges stay explicit: `df.values`, transpose, `apply(axis=1)`,
      `iterrows` pay a conversion by nature.
- [ ] Gates: index lookup/alignment on Arrow-backed indexes within ~1.5x of the
      typed engines; registry extended with frame-op entries; no small-frame
      regression in asv.

## Phase 2 — per-operation native execution (eager, semantics intact)

The cudf.pandas model on CPU: heavy operations on Arrow-native frames dispatch
to a native kernel or an embedded engine **per call**, falling back otherwise.
Eager, so pandas semantics hold by construction; the test suite is the oracle.
Why it failed before: the Arrow↔NumPy boundary on every call. After Phase 1
that boundary is gone, which is the whole reason for the ordering.

Target operations (by measured payoff): `merge` (`pd.merge` is GIL-bound,
measured 2.75x/4 threads), `groupby.agg` (our partitioned hash-aggregate beats
Polars 1.9x on high-cardinality keys), `sort_values`, `value_counts`,
`drop_duplicates`, string predicates, `concat`.

- [ ] **First concrete bridge from the lazy branch:** port
      `pandas/_libs/lazy_groupby.pyx` (parallel partitioned hash-aggregate,
      nogil) into core groupby for Arrow/int keys. Already measured; smallest
      step with a Polars-beating payoff.
- [ ] Engine choice per op, decided by measurement, not policy:
      in-tree Cython nogil kernels (portable, no dependency; proven for
      groupby/join indexers) · pyarrow compute/Acero (already a dependency;
      single-threaded `group_by`, per-node overhead measured) · DataFusion
      (fast and parallel; a Rust wheel dependency → **optional engine only**,
      via the existing `engine=` precedent: numexpr, numba, pyarrow-csv).
- [ ] Dispatch rule: all inputs Arrow-backed → native path; else current path.
      Document result-order and null semantics per op (groupby `sort=True`
      default, merge left-order preservation, sum-of-all-null = 0, NA vs NaN).
- [ ] Gates per op: ≥2x over current pandas on Arrow frames at 1–10M rows; the
      op's full test file passes on the native path; no regression below
      ~10k rows.
- [ ] Phase gate: an eager TPC-H-style pandas script on Arrow frames ≥5x the
      same script on NumPy frames, and within ~3x of Polars in-memory.

## Phase 2b — lazy data movement in eager pandas (the CoW pattern)

Added 2026-11-07. CoW deferred a *copy* until a write was observed, and it was
safe because (i) the deferred action has no observable effect except memory and
time, (ii) every observation point is interceptable in one place, (iii) the
result is bit-identical. The same three conditions hold for **data movement**
(gather, scan, column load) and fail for **computation** (deferring it moves
where exceptions raise and, in pandas, sometimes which dtype results). So this
phase defers data movement only; deferred computation stays in Phase 3.
Semantics are unchanged by construction, which is why these land as ordinary
PERF changes rather than a mode.

- [ ] **Late materialization (deferred gather).** `df[mask]`, `sort_values`,
      `iloc`/`head`, `reindex`, and the assembly step of `merge` all compute an
      indexer and immediately apply it to every column. Instead a frame carries
      a *pending indexer* applied per column on first access; indexer-of-
      indexer composes into one integer gather. Observation hooks = the CoW
      write hook + column access + `values`/`to_numpy`. This is the manager-
      level feature (pending indexer on blocks; `EA.take` on access for Arrow
      columns) and is the same lever that removes the take-every-column join
      cost measured in both engines. Idioms it fixes: `df[mask][["a", "b"]]`
      (copies 50 columns to keep 2), `df.sort_values("x").head(10)` (argsort
      one column then move all columns then slice), `merge(...)[["a", "b"]]`.
      - [ ] Probe first: argsort-only vs argsort + full gather on a wide
            frame (the top-k idiom), and `merge` indexer-only vs indexer +
            gather, at 1–10M rows × 20–50 columns. Expected: gather dominates.
      - [ ] Memory retention rule: `df = df[mask]` at low selectivity keeps the
            parent alive until a column is touched → materialize eagerly below
            a selectivity threshold; reuse CoW's reference tracking.
      - [ ] Gates: bit-identical results across the affected test files; no
            regression on narrow frames (1–3 columns, where the deferral buys
            nothing and must cost ~nothing).
- [ ] **Deferred scan with pushdown.** `df = read_parquet(p); df =
      df[df.year == 2024][["a", "b"]]` is the commonest out-of-core-adjacent
      pattern. Defer the scan until first observation; if the first operations
      are filters/projections, push them into the pyarrow dataset reader
      (already supports both) and materialize the result; otherwise
      materialize at once. Result identical to eager. Opt-in flag first
      (`read_parquet(..., lazy=True)` or an option), default later under the
      CoW-style warning-then-flip process.
      - [ ] Scope the pushable predicate set (comparisons, `isin`, `&`/`|`,
            `isna`) and column-projection via `__getitem__`/`loc`.
      - [ ] Gate: a filter-at-read workload reads only the surviving row
            groups/columns (verify with pyarrow's scan statistics).
- [ ] **Lazy column loading.** Load Parquet/IPC columns on first access
      instead of at read. Projection pushdown at the eager level with no API
      change; subsumed by the deferred scan if that ships, so do it only if the
      scan item stalls.
- [ ] **Larger-than-memory, what this buys and what it doesn't.** Sources
      can exceed RAM (deferred scan + pushdown; Arrow IPC/Feather memory
      mapping for read-mostly single-pass work). Intermediates of `groupby`/
      `merge`/`sort` whose *inputs* exceed RAM cannot: eager semantics require
      every intermediate to exist, and chunked eager execution is Modin's
      design, which lost to both pandas and the lazy engines. True out-of-core
      pipelines are Phase 3 (validated there at SF-100/300 beyond RAM).

## Phase 3 — deferred mode (lazy), opt-in, last

Once Phases 1–2 exist, laziness is a fusion optimization over operations that
already run natively, not a new semantics project. Its distinct new capability
is **out-of-core** (lazy scans, data > RAM — the lazy branch validated 22/22 at
SF-100/300 beyond RAM).

- [ ] **Go/no-go probe — can start now, in parallel, on NumPy frames:** wire
      the eager-call tracer (`compile-ir` branch, `pandas/jit/`) in front of
      the lazy `LogicalPlan` + the 22/22-correct DataFusion lowering
      (`benchmarks/translate_datafusion.py`) and produce three numbers:
      1. pass rate of `pandas/tests/frame/methods` (then groupby, merge) in
         deferred mode with eager pandas as the oracle;
      2. fraction of operations that lower vs fall back;
      3. graph-break frequency on a corpus of real notebooks (observation
         points: `len`, `shape`, `head`, `print`, data-dependent column names,
         UDFs).
- [ ] If go: `pd.options.mode.deferred` (context manager), materialize at
      observation points, fallback to eager, order preserved via a row-id
      column and restored at collect (relaxed when unobservable), lower to
      DataFusion first; the Rust engine stays a probe baseline.
- [ ] Known semantic edges from the lowering work, to encode as tests: sort
      null placement, null-key joins, sum of all-null, integer division, shared
      subplans (DataFusion #22676), float-reduction bit nondeterminism under
      parallelism (documented-behavior equality, not bit equality).
- [ ] Gates: ≥1.5x over Phase-2 eager on chained workloads (the measured
      fusion dividend); zero test-suite regressions in deferred mode.

## What happens to the lazy-pandas branch

Keep as the probe it is. Reuse map:

| keep | retire |
|---|---|
| `LogicalPlan` IR, optimizer passes, index-as-columns schema | the Polars-style surface API (`select`, `col`) as a user API |
| `translate_datafusion.py` (22/22) + `validate_datafusion.py` | the Python/Cython executor as an engine (0.30x) |
| `differential_probe.py`, TPC-H/H2O harnesses, SF-100/300 validation | — |
| `_libs/lazy_groupby.pyx`, `lazy_join.pyx`, `lazy_radix.pyx` → Phase 2 | — |
| `rust_engine_prototype/` as reference baseline | — |
| upstream hand-offs (`docs/upstream/`) | — |

## Sequencing and capacity

- Phase 0 is continuous and already staffed (the Arrow PR track).
- Phase 1 is now a measurement pass plus the index-engine fix: weeks, not
  quarters; it mostly feeds Phase 2's target list.
- Phase 2 lands op by op over 2–4 quarters; the groupby kernel port is first.
- Phase 3's probe can run now (no dependency on Phases 1–2); the mode itself
  waits for Phase 1.
- Everything public (issues, PDEPs, PRs) goes through the usual per-action
  go-ahead.
