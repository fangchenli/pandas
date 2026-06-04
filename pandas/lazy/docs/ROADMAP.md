# Lazy Pandas Roadmap

Open work, known gaps, and design questions. Implemented work is documented
in [ARCHITECTURE.md](ARCHITECTURE.md) / [OPTIMIZER.md](OPTIMIZER.md); dated
performance reports live in `../benchmarks/`.

## Competitive Standing (vs Polars, Jan 2026)

From `../benchmarks/LAZY_VS_POLARS_BENCHMARK.md` (1M–10M rows, Apple
Silicon). Speedup > 1.0 = lazy pandas faster:

| Category | Avg speedup | Notes |
|----------|-------------|-------|
| string | 0.86x (up to 2.1x) | `str.lower` beats Polars; `contains`/`len` behind |
| parquet_scan | 0.28x | scan+filter+head competitive (0.68x); glob scans behind |
| sort | 0.22x | |
| aggregation | 0.19x | multi-agg best case 0.43x |
| join | 0.15x | |
| filter_project | 0.14x | |
| limit | ~0x | `head(N)` is Polars' best case (µs) vs our ms — see below |

## High-Impact Opportunities

1. **`head()`/limit fast path.** Polars answers `head(10)` on in-memory data
   in ~10µs; we spend milliseconds planning and copying. Needs a trivial-plan
   short-circuit that slices without entering the physical pipeline.

2. **Planning-overhead fast paths.** Single filter/select queries pay
   60–80% of their lazy overhead in the optimizer. Heuristics worth
   implementing (from `bench_planning_phases.py`):

   | Condition | Skip | Savings |
   |-----------|------|---------|
   | Single filter/select | entire optimizer | 60–80% of overhead |
   | No repeated expressions | CSE pass | 10–20% |
   | Flat query (no nesting) | PredicatePushdown | 5–10% |
   | No limits | LimitPushdown, SortLimitToTopK | ~5% |
   | < 10K rows | lazy path entirely (use eager) | all of it |

3. **Sort performance.** 0.2x of Polars; needs parallel sort and/or Arrow
   sort improvements; multi-key sort is the worst case.

4. **Join performance.** 0.15x of Polars despite build/probe + parallel
   sides; profile hash table build (candidate for the swisstable work) and
   output materialization.

5. **Cardinality estimation.** Plan nodes carry no row estimates; filters
   assume nothing about selectivity. Row-count propagation + simple
   selectivity estimates (histogram/sample/default-10%) would make engine
   selection and join-side choice cost-based instead of threshold-based.

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
