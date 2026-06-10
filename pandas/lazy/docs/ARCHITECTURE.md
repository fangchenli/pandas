# Lazy Pandas Architecture

This document describes the lazy pandas execution architecture as implemented.
For plan construction and the physical planner in depth, see
[PLANNING.md](PLANNING.md); for the optimizer, [OPTIMIZER.md](OPTIMIZER.md);
for per-operation backend choices, [KERNELS.md](KERNELS.md).

## Execution Pipeline

```
User Query (Expr API)
    │
    ▼
Logical Plan (plan.py)            built incrementally by LazyDataFrame methods
    │
    ▼
Optimizer (optimize/)             12 rule-based passes, cached per LazyDataFrame
    │
    ▼
Physical Planner (physical.py)    logical → physical operators, operator fusion
    │                             (only when use_physical_planner=True)
    ▼
Execution                         array-based (ArrayDict) or pandas fallback
    │
    ▼
pd.DataFrame                      constructed once, at collect()
```

Two execution paths exist:

1. **Pandas evaluator** (`eval.py`) — operates on DataFrames/Series. Default
   path and the fallback when a kernel is unavailable.
2. **Physical planner** (`physical.py`) — operates on raw arrays via backend
   kernels. Enabled with `collect(use_physical_planner=True)`. Supports
   streaming, spilling, and parallel execution.

## Expression IR (`ir.py`)

User expressions built via the `Expr` API (`expr.py`) lower to a small IR:

| Node | Meaning |
|------|---------|
| `FieldRef` | Column reference |
| `Literal` | Constant value |
| `Alias` | Named expression |
| `Call` | Function call (carries `is_aggregate` flag) |
| `Cast` | Type conversion |

`Call.function` names (e.g. `add`, `str_lower`, `dt_year`, `groupby_sum`) are
the keys into the backend kernel registry.

## Logical Plan (`plan.py`)

| Node | Meaning |
|------|---------|
| `DataFrameSource` | Leaf wrapping an in-memory DataFrame |
| `ParquetSource` / CSV source | Leaf for lazy file scans (carries pushed-down predicate + projection) |
| `Concat` | Multi-file / multi-input concatenation |
| `Project` | Column selection/computation |
| `Filter` | Row filtering by predicate |
| `Aggregate` | Grouping and aggregation |
| `Sort`, `Limit`, `TopK` | Ordering and row limiting (`TopK` produced by the optimizer from Sort+Limit) |
| `Distinct` | Deduplication |
| `Join` | Two-input join |
| `Convert` | Explicit backend conversion boundary (inserted by the optimizer) |

## Type System (`types.py`)

`Schema` maps column names to `LazyDtype`, which tracks **both**
`numpy_dtype` and `arrow_type` plus a category ("numeric", "string",
"datetime", "boolean", "object"). This dual tracking is what enables
backend-aware planning: storage backend (where data lives) is distinguished
from execution backend (which kernel runs), and data can be converted when
profitable.

Null values use nullable dtypes with `pd.NA` semantics.

## Physical Execution

### ArrayDict Convention

Physical operators pass `dict[str, ArrayLike]` (NumPy `ndarray`, `pa.Array`,
or `pa.ChunkedArray`) between nodes instead of DataFrames:

```
PhysicalScan → {"__index__": ..., "col_a": ..., "col_b": ...}
    → PhysicalFilter → PhysicalProject → ... → arrays_to_dataframe()
```

Design decisions:

| Decision | Choice |
|----------|--------|
| Backend granularity | Per-node, with per-expression overrides for high-value ops (string/null ops force Arrow) |
| Mixed formats | Allowed — some columns Arrow, some NumPy in the same ArrayDict |
| Index handling | Stored as `__index__` column (`__index_0__`, `__index_1__`, ... for MultiIndex); names kept in `ExecutionContext` |
| DataFrame construction | Once, at the end of `collect()` |
| Output dtypes | A schema-driven contract (`convert.arrays_to_dataframe`) returns the dtypes the eager path would, regardless of which kernels ran: numeric/bool → NumPy, strings → default `str` (zero-copy Arrow-buffer reinterpret), genuine `pd.ArrowDtype` and Categorical preserved. Pipeline-breaker outputs assemble `copy=False` (no consolidation copy); projections keep the copy that isolates passthrough views |

### ChunkedArray Strategy

pandas Arrow-backed columns are `pa.ChunkedArray`. Most `pyarrow.compute`
functions accept chunked input directly, so kernels keep chunks as-is and only
call `combine_chunks()` when an operation requires contiguous memory.
Misaligned chunk boundaries between two inputs are handled by PyArrow
internally.

### Kernel Registry and Routing

`backends/__init__.py` keeps a registry keyed by `(function_name, backend)`,
populated via `@register_kernel("add", "numpy")`-style decorators in
`backends/numpy/*` and `backends/arrow/*`. The router
(`backends/router.py`) decides the backend per operation:

- **Arrow-preferred ops** (string ops, null handling) always route to Arrow.
- **Neutral ops** (arithmetic, comparison, aggregation) follow the input
  format, with threshold-based switching for large data (see
  [THRESHOLDS.md](THRESHOLDS.md)).
- **Bottleneck** accelerates rolling-window and ffill/bfill kernels when
  installed and `compute.use_bottleneck` is enabled
  (`backends/_bottleneck.py`).

When no kernel exists: strict mode raises `NotImplementedError`; normal mode
falls back to the pandas evaluator with a warning.

### Expression Evaluation

`backends/array_eval.py:ArrayEvaluator` evaluates IR trees against an
ArrayDict, dispatching each `Call` to the chosen backend kernel and converting
operands when needed. It integrates:

- **NumExpr fusion** (`backends/numexpr_fusion.py`) — arithmetic/comparison
  subtrees over large NumPy arrays (≥100K elements) are compiled into a single
  multi-threaded NumExpr evaluation, avoiding intermediate allocations
  (2–10x for complex expressions).
- **Memory pooling** (`backends/memory_pool.py`) — `PoolingStrategy.SCRATCH`
  (default) rotates pre-allocated buffers through expression chains (~3x for
  chained arithmetic); `ArrowPoolBackend` selects the Arrow allocator
  (mimalloc default).

## Joins

`PhysicalHashJoin` chooses a strategy after materializing both sides:

- **`pd.merge` (primary, in-memory equi-joins)** — the default path
  (`_execute_pandas_merge`). `pd.merge` *is* the eager-pandas join semantics
  this engine promises to match, so it is row-order- and null-correct by
  construction, and it is the fastest path measured (it beats both the custom
  indexer hash join and Arrow/acero — see `benchmarks/H2O_BENCHMARK.md`). Used
  whenever the index is not observed downstream.
- **acero hash join** — for joins feeding an order-free sink (or under
  `collect(order="relaxed")`) with acero-safe keys; an internally-parallel
  fallback, now rarely faster than `pd.merge` for in-memory data.
- **Indexer hash join** — the build/probe path (build on the smaller side;
  left/right fix the build side; semi/anti via pandas Cython
  `hashtable.ismember`). Kept for index-observing joins (preserves the left
  index).
- **Grace hash join (size-triggered, out-of-core)** — used *only* when
  spilling is enabled **and** the materialized inputs exceed the operator
  budget; small joins with spill enabled still take the fast `pd.merge` path.
  Both sides are hash-partitioned and spilled, partition pairs joined in
  memory (each pair itself via `pd.merge`), with a **sort-merge fallback**
  on pathological skew (`physical.py:_execute_sort_merge_fallback`).
- **Parallel side execution** — left and right subplans execute concurrently.

### Join Column Disambiguation

Overlapping non-key columns are suffixed in the join output schema (pandas
convention: `a` → `a_x`/`a_y`); join keys appear once. Expressions after a
join must use the suffixed names — ambiguous references raise with a message
naming both candidates. The optimizer tracks provenance (output name → source
side and original name) to push predicates and prune columns through joins
correctly. Nested joins compound suffixes; self-joins suffix every non-key
column.

## Streaming Execution

`collect(streaming=True, batch_size=...)` yields results batch-by-batch:

- **Pipeline operators** (Filter, Project, Limit) process one batch at a time.
- **Pipeline breakers** (Sort, Aggregate, Join build side) materialize their
  input before producing output. `explain(physical=True)` marks these
  boundaries with `[BREAKER]`.
- **Early termination** — `head(N)`/`limit(N)` stops scanning once N rows are
  produced (2–2.5x on large scans; up to ~10x on multi-file globs).
- Default batch size 65,536 rows is L3-cache friendly.

Streaming aggregation accumulates per-batch partial states using Arrow
kernels and merges at the end.

## Spilling (`backends/spill.py`)

Out-of-core execution is opt-in via `collect(spill_config=SpillConfig(...))`:

- `SpillManager` + `MemoryTracker` spill the largest tracked buffers when the
  memory threshold is exceeded; per-operator budgets are configurable.
- Spill files use Arrow IPC format, reloaded zero-copy via memory mapping.
- `ExternalSorter` (sorted-run merge) and `GraceHashJoiner` build on the same
  infrastructure.
- Files are session-scoped and cleaned up on manager close.

## Parallelism

All thread-pool parallelism relies on NumPy and Arrow kernels releasing the
GIL; pools are sized `min(8, cpu_count)` and gated by row thresholds so
small data never pays pool overhead.

- **Parallel expression evaluation** — projections with ≥8 expressions
  (configurable) evaluate in a `ThreadPoolExecutor`, one `ArrayEvaluator` per
  thread.
- **Parallel join sides** — independent subplans of a join run concurrently.
- **Sort** — large numeric single-key sorts use a Cython **thread-parallel
  LSD radix argsort** (`pandas/_libs/lazy_radix.pyx`; keys built vectorized in
  NumPy, ~130 ms at 10M float64, faster than Polars' `arg_sort`). Multi-key
  sorts use a **radix lexsort** (stable per-key radix composed
  least-significant-first); string keys are factorized into order-preserving
  codes first, so they ride the same path. Non-numeric/unsupported keys fall
  back to Arrow's table `sort_indices`, then `np.lexsort`. The NumPy k-way
  merge (`_parallel_argsort`) remains the fallback when the radix kernel is
  unavailable.
  **Lazy sorts are stable by contract** (ties keep their original relative
  order): it is the only tie order the eager evaluator, NumPy engine, and
  Arrow engine (whose sorts are inherently stable) can all agree on, and
  the equivalence suite asserts it.
- **Parallel gather** (`physical.py:_take_all_columns`) — applying sort
  indices to result columns fans out per column (~2.5x for 4 columns at
  10M rows); used by single- and multi-key sort paths.
- **Parallel concat inputs** — `PhysicalConcat.execute()` runs independent
  input subtrees (typically per-file scans) concurrently, preserving input
  order in the output.
- Arrow's internal threading is enabled where available: `Table.group_by`
  (except order-dependent first/last), `Table.join`, Parquet reads.

Measured effect at 10M rows × 5 columns: single-key sort 1.37x *faster*
than eager pandas (previously ~2x slower); multi-key sort 1.73x faster than
eager.

## Lazy File Scanning (`scan.py`)

| Format | Status | Pushdown |
|--------|--------|----------|
| Parquet | Implemented | Predicate (incl. row-group statistics skipping) + projection |
| CSV | Implemented | Projection; streaming-friendly |
| JSON | Not yet implemented | — |

Paths may be local files, glob patterns (`data/*.parquet`), or fsspec URLs
(`s3://`, `gs://`, `https://`). Multi-file scans plan as `Concat` of
per-file sources, so pushdown applies to every file.

## Index Preservation

By default results get a fresh `RangeIndex` (materialized as int64).
`collect(preserve_index=True)` carries the original index through the plan as
`__index__` column(s), including through joins and groupby
(`with_row_index`/`set_index`/`reset_index` are also available on
`LazyDataFrame`). Group keys become the index for grouped results, matching
eager pandas.

## Invariants

Every optimization and execution path targets **value equivalence** with
eager pandas under nullable-dtype semantics:

- same values and column order (dtypes may differ: nullable/Arrow-backed
  output is documented behavior)
- same NA presence and NaN-as-missing semantics: aggregation kernels mask
  float NaN to null (Arrow) or drop it (NumPy), and rows with missing
  group keys are dropped, matching pandas `skipna`/`dropna` defaults
- `collect(optimize=False)` and `collect(use_physical_planner=False)` are
  always available as reference paths; equivalence is enforced by
  `tests/lazy/test_optimizer_equivalence.py`

Note that *index, groupby shape, and null-representation defaults
deliberately differ* from eager pandas (Polars-style positional results);
see PROPOSAL.md for the full kept-vs-deviates breakdown.
