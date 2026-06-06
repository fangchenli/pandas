# Lazy Pandas Roadmap

Open work, known gaps, and design questions. Implemented work is documented
in [ARCHITECTURE.md](ARCHITECTURE.md), [PLANNING.md](PLANNING.md), and
[OPTIMIZER.md](OPTIMIZER.md); dated performance reports live in
`../benchmarks/`.

## Competitive Standing (vs Polars, June 2026 — physical engine)

From `../benchmarks/LAZY_VS_POLARS_BENCHMARK.md` (1M–10M rows, mixed-dtype
data, Apple Silicon). Speedup > 1.0 = lazy pandas faster. The January
report had measured the eager path by mistake; all numbers below are the
physical engine after the June conversion/routing fix cycle:

| Category | Avg | Best | Movement this cycle |
|----------|-----|------|---------------------|
| string | **1.10x** | 2.43x (`str.lower`) | 0.27x → win (pass-through output fix) |
| aggregation | 0.53x | **2.11x** (multi-agg) | 0.07x → partial win (groupby routing fix) |
| join | 0.31x | 0.39x | 0.06x → competitive (Cython indexers + native gathers) |
| parquet_scan | 0.30x | 0.74x | ~flat |
| sort | 0.27x | 0.34x | 0.17x → (parallel argsort, Arrow multi-key) |
| filter_project | 0.22x | 0.39x | 0.03x worst case → 0.39x (fused-mask pruning) |
| limit | 0.03x | — | ~0x → ~200 µs absolute (trivial-slice fast path) |

Common thread of the cycle: kernels were never the bottleneck — data
movement between them was. Every fix removed conversions or routing
mistakes on columns no operation touched.

## High-Impact Opportunities

1. **Stop converting pass-through columns.** *Output side fixed*: the
   final DataFrame assembly was round-tripping untouched Arrow columns
   through object arrays (379 ms → 51 ms for `with_columns` at 10M with
   one string column; physical `str.lower` went from a 0.39x loss to a
   2.4x win over Polars; up to -70% on physical pipeline baselines).
   Output now wraps Arrow columns zero-copy and leaves NumPy columns
   alone — locked by `test_passthrough_arrow_column_not_copied`.
   *Join side fixed too*: the NumPy join np.asarray'd every column
   (Arrow strings → object) and used a pure-NumPy indexer ~6x slower
   than pandas' Cython hash join. The kernel now computes indexers via
   `pandas.core.reshape.merge.get_join_indexers` (the same machinery as
   eager `pd.merge` — row order and NaN-key semantics match by
   construction) and gathers payload columns in their native backend.
   10M×1M inner join: ~13,000 ms → **846 ms** (eager merge: 636 ms;
   Polars: ~712 ms). *Groupby fixed too*: a routing bug chose the
   backend from the *first* input column, sending Arrow-string-keyed
   groupbys down the NumPy path (object factorize). The choice now
   considers only group keys + aggregation columns. 10M wide-frame
   groupby-sum: 290 ms → **58 ms**; multi-agg 975 ms → **67 ms**
   (Polars: 116 ms — first aggregation win). Output dtype consistency
   improved as a side effect: string columns always come out as
   pandas' default str dtype on every output path.

2. ~~**`head()`/limit fast path.**~~ **Done**: `collect()` short-circuits
   `Limit` over pure column selections of an in-memory source, slicing
   directly (CoW makes the views safe). 9,000–14,000 µs → **~200 µs** at
   any size. The fix also removed an O(n) `isna` scan per column from
   `Schema.from_dataframe` — plan construction is now O(1) per column,
   which benefits *every* lazy query, not just head. Remaining gap to
   Polars' ~10 µs is plan-construction overhead (see item 3).

3. **Planning-overhead fast paths.** Single filter/select queries pay
   60–80% of their lazy overhead in the optimizer. Heuristics worth
   implementing (from `bench_planning_phases.py`):

   | Condition | Skip | Savings |
   |-----------|------|---------|
   | Single filter/select | entire optimizer | 60–80% of overhead |
   | No repeated expressions | CSE pass | 10–20% |
   | Flat query (no nesting) | PredicatePushdown | 5–10% |
   | No limits | LimitPushdown, SortLimitToTopK | ~5% |
   | < 10K rows | lazy path entirely (use eager) | all of it |

4. **Sort performance.** 0.2x of Polars; needs parallel sort and/or Arrow
   sort improvements; multi-key sort is the worst case.

5. **Join performance.** 0.15x of Polars despite build/probe + parallel
   sides; profile hash table build (a swisstable-style hash table is one
   candidate) and output materialization.

6. **Cardinality estimation.** Row estimates exist for sources and
   size-preserving nodes but stop at filters (`estimate_row_count()`
   returns None — no selectivity model). Simple selectivity estimates
   (histogram/sample/default-10%) would make engine selection and
   join-side choice cost-based instead of threshold-based.

## Known Semantic Issues (bugs, not design choices)

- ~~**NaN vs null in aggregation**~~ **Fixed**: Arrow kernels mask NaN→null
  before aggregating (`mask_nan_to_null`), NumPy kernels drop NaN values and
  negative factorize codes, and rows with missing group keys are dropped
  (pandas `dropna=True`). Locked by `TestNaNAggregationSemantics`.
- **Output dtype instability**: the physical engine returns Arrow-backed
  dtypes on some paths (large filters) and NumPy dtypes on others
  (small data, groupby outputs). Pick one contract and enforce it.
- **Duplicate column labels** crash with `AttributeError` instead of a
  clear "unsupported" error at plan construction.
- **`shift` is unimplemented in the eager evaluator** while `lag` works in
  both engines — audit Expr-API coverage parity between engines.

## Smaller Items

- **JSON scanning** — `scan.py` accepts `format="json"` but raises
  `NotImplementedError`.
- **Partition-aware execution** — parallel processing of pre-partitioned
  (e.g. hive-partitioned Parquet) data.
- **Adaptive thresholds maturation** — the EMA-based tuner
  (`optimize/adaptive.py`) is experimental and off by default; needs
  evaluation under mixed workloads before enabling.
- **Nullable dtype preservation** — physical execution can widen nullable
  ints through NumPy kernels; track original dtypes in the execution context
  and restore on output.
- **RangeIndex preservation** — `preserve_index=True` materializes a
  RangeIndex as int64 values; could carry the range representation through
  the plan instead.

## Where Lazy Already Wins (keep protected by benchmarks)

| Scenario | Typical speedup vs eager | Mechanism |
|----------|--------------------------|-----------|
| Sequential filters | 1.5–2x | filter fusion |
| filter + `head(N)` | 2–10x (10x+ on multi-file) | streaming early termination |
| Arrow string pipelines | 2–10x | Arrow kernels |
| Multi-step pipelines | 1.2–1.5x | reduced materialization |
| Larger-than-memory | n/a (enables) | streaming + spill |

Regressions here should be caught by `benchmarks/bench_optimizer_quality.py`
(plan shape) and the benchmark suite (timings).
