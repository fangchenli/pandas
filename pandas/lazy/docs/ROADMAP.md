# Lazy Pandas Roadmap

Open work, known gaps, and design questions. Implemented work is documented
in [ARCHITECTURE.md](ARCHITECTURE.md), [PLANNING.md](PLANNING.md),
[OPTIMIZER.md](OPTIMIZER.md), and — for the execution engine —
[ENGINE_DESIGN.md](ENGINE_DESIGN.md), whose six milestones are all landed
or measurement-gated as of June 2026. Dated performance reports live in
`../benchmarks/`.

## Competitive Standing (vs Polars, June 2026 — engine era)

From `../benchmarks/LAZY_VS_POLARS_BENCHMARK.md` (1M–10M rows, mixed-dtype
data, Apple Silicon; physical engine). Speedup > 1.0 = lazy pandas faster.
Two development phases produced these numbers: the conversion fix cycle
(June 6), then the designed engine (M1–M6, June 6–7):

| Category | Standing | Driver |
|----------|----------|--------|
| string | **2.07x avg — wins** (`str.lower` 4.38x, `contains` 1.71x) | pass-through fix + compute-bound morsel parallelism |
| aggregation | **1.17x avg — wins** (multi-agg 3.47x, groupby-sum 1.02x @10M) | groupby routing + dictionary-encoding cache (warm from 2nd query) |
| parquet scan | 0.86x avg — glob `head()` **wins 1.55x** (6.7 ms vs 10.3) | limit pushdown into scans + direct ParquetFile path + vectorized index |
| join→groupby composite | 0.58x | acero routing for order-free joins |
| sort | 0.42x | k-way segment merge (3.6x kernel) |
| category-key groupby | 0.43x vs Polars-categorical | zero-copy dictionary flow (was 313 ms, now 21) |
| full-scan select+filter | ~0.37x (21 ms) | vectorized index column (was 104 ms) |
| filter_project | 0.21x | bandwidth-bound; single thread saturates (measured) |
| join (order-preserving) | 0.42x default; **0.89x with `order="relaxed"`** | eager `pd.merge` row order by default; `collect(order="relaxed")` routes to acero (measured 10M×1M inner: 298→142 ms, 2.1x; left: 232→71 ms, 3.2x) |
| limit (in-memory) | ~300 µs absolute | plan-construction floor |

Themes, each measured: kernels were never the bottleneck (the fix cycle);
parallelism belongs in internally-parallel C++ kernels routed by plan-time
decisions, with Python threads only for compute-bound kernels (the engine);
and representation — dictionary keys, vectorized index columns — is worth
more than threading on bandwidth-bound paths.

## Open Opportunities (ranked)

1. **JAX/XLA kernel backend** (the breakthrough candidate — see
   ENGINE_DESIGN.md "Future backends"). Fused codegen is the only
   identified lever against the measured memory-bandwidth ceiling that
   caps bandwidth-bound chains at ~1x for *any* thread-based approach;
   GPU morsels are the speculative extension. The prerequisites (pipeline
   objects, per-column backend planning, conversion costing) all exist.
2. **Acero raw-string hash gap.** acero groups raw `large_string` keys at
   67 ms/10M vs Polars' 18; the dictionary cache solves repeated queries
   but first-query and one-shot workloads still pay. Upstream Arrow work
   or a pre-hashing trick are the options.
3. **Planning-overhead fast paths.** Single-op queries pay 60–80% of lazy
   overhead in the optimizer (`bench_planning_phases.py`); skip passes by
   plan shape. The ~300 µs limit floor is this item.
4. **Cardinality estimation.** Row estimates stop at filters (no
   selectivity model); estimates feed the decision layer's join build-side
   and parallelism-degree choices, so better estimates compound.
5. **Free-threaded partition joins.** pandas' Cython hash join holds the
   GIL (measured: threaded partition-pairs 461→535 ms at 2→8 threads vs
   430 serial). On free-threaded Python the M5-spec'd partitioned join
   becomes buildable; the engine architecture needs no changes to exploit
   it.

## Known Semantic Issues (bugs, not design choices)

- **Output dtype instability**: ~~settled~~ — a schema-driven output
  contract (`convert.arrays_to_dataframe`, June 2026) makes the physical
  engine return the dtypes eager would, regardless of which kernels ran:
  numeric/bool → NumPy (int widens to float64 on nulls, like eager),
  strings → default `str`, genuine `pd.ArrowDtype` preserved, Categorical
  preserved. No more `double[pyarrow]` leaking from acero groupby/join.
  Enforced on every user-facing path; gated by the `schema` arg so internal
  join/spill round-trips are untouched. Residual: masked nullable dtypes
  (see below).
- **Duplicate column labels** crash with `AttributeError` instead of a
  clear "unsupported" error at plan construction.
- **`shift` is unimplemented in the eager evaluator** while `lag` works in
  both engines — audit Expr-API coverage parity between engines.

## Smaller Items

- **JSON scanning** — `scan.py` accepts `format="json"` but raises
  `NotImplementedError`.
- **CSV limit pushdown** — Parquet scans got `ParquetSource.limit` +
  the direct small-limit path; CSV scans have neither.
- **Partition-aware execution** — parallel processing of pre-partitioned
  (e.g. hive-partitioned Parquet) data.
- **Adaptive thresholds maturation** — the EMA-based tuner
  (`optimize/adaptive.py`) is experimental and off by default; the cost
  model (`pandas/lazy/cost.py`) is now the natural calibration target.
- **Nullable dtype preservation** — the output dtype contract preserves
  genuine `pd.ArrowDtype` columns but *not* pandas masked nullable dtypes
  (`Int64`/`Float64`): once a join/aggregate marks the schema nullable, a
  masked-nullable source is indistinguishable from its NumPy counterpart,
  so it comes out NumPy-backed. Fixing this needs the schema to track
  "originated as a masked extension dtype" through joins/aggregates.
- **RangeIndex preservation** — `preserve_index=True` materializes a
  RangeIndex as int64 values; could carry the range representation through
  the plan instead.
- **Compute-bound kernel classes** — morsel parallelism currently
  recognizes `str_*` only; regex and date parsing are unmeasured
  candidates (`cost.py` / `engine/parallel.py`).

## Where Lazy Already Wins (keep protected by benchmarks)

| Scenario | Typical speedup vs eager | Mechanism |
|----------|--------------------------|-----------|
| Sequential filters | 1.5–2x | filter fusion |
| filter + `head(N)` | 2–10x (10x+ on multi-file) | streaming early termination + scan limit pushdown |
| Arrow string pipelines | 2–10x | Arrow kernels + morsel parallelism |
| Repeated string-key groupbys | ~3x from 3rd query | dictionary-encoding cache |
| Multi-step pipelines | 1.2–1.5x | reduced materialization |
| Larger-than-memory | n/a (enables) | streaming + spill |

Regressions here should be caught by `benchmarks/bench_optimizer_quality.py`
(plan shape) and the benchmark suite (timings).
