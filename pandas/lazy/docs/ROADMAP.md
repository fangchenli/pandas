# Lazy Pandas Roadmap

What we plan to do next, what just landed, and the known gaps. Implemented
architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md),
[PLANNING.md](PLANNING.md), [OPTIMIZER.md](OPTIMIZER.md), and — for the
execution engine — [ENGINE_DESIGN.md](ENGINE_DESIGN.md) (six milestones, all
landed). Dated performance reports live in `../benchmarks/`.

## Competitive Standing (vs Polars, June 2026 — engine era)

Two benchmarks: the **H2O db-benchmark** (`../benchmarks/H2O_BENCHMARK.md`) is
the authoritative cross-engine standard (group-by + join, same harness as
Polars/DuckDB) — there lazy pandas now **supports all 10 group-by and all 5
join queries and beats Polars on 6 group-by queries**; int-key joins are
roughly at parity (run-to-run noisy) with string-key/left joins the genuine
remaining losses. The table below is from the custom `LAZY_VS_POLARS_BENCHMARK.md`
microbenchmark (1M–10M rows, mixed-dtype, Apple Silicon). Speedup > 1.0 = lazy
pandas faster.

| Category | Standing | Driver |
|----------|----------|--------|
| string | **2.07x avg — wins** (`str.lower` 4.38x, `contains` 1.71x) | pass-through fix + compute-bound morsel parallelism |
| aggregation | **wins** (H2O group-by: 6/10 beat Polars, string-key sums up to ~7x, `corr` ~3x) | `groupby_prefers_arrow` routing (numeric + arrow-string → acero) + post-aggregation projection |
| parquet scan | 0.86x avg — glob `head()` **wins 1.55x** (6.7 ms vs 10.3) | limit pushdown into scans + direct ParquetFile path + vectorized index |
| join | **`pd.merge` path** — int-key joins ~parity (noisy); string-key (H2O q4) and left (q3) joins lag Polars' parallel hash join — see H2O_BENCHMARK.md | `pd.merge` is the in-memory equi-join path (eager-correct, fastest); acero/Grace fallbacks |
| sort | numeric ~0.9x (argsort beats Polars); string-key multi-sort 0.46→0.88x; full mixed sort gather-bound | radix argsort + radix lexsort (numeric & factorized string keys) + breaker copy-free output; string payload gather is the wall |
| category-key groupby | 0.43x vs Polars-categorical | zero-copy dictionary flow (was 313 ms, now 21) |
| full-scan select+filter | ~0.37x (21 ms) | vectorized index column (was 104 ms) |
| filter_project | 0.21x | gather-bound (string `take`) + pandas' mutation-safety copy, not bandwidth |
| limit (in-memory) | ~300 µs absolute | plan-construction floor |

Themes, each measured: kernels were rarely the bottleneck; parallelism
belongs in internally-parallel C++ kernels routed by plan-time decisions,
with Python threads only over GIL-releasing kernels; representation
(dictionary keys, vectorized index, radix keys) beats threading on
bandwidth-bound paths; and the residual losers (sort gather, filter_project)
are bounded by the Arrow `large_string` layout and pandas' mutation
semantics, not by missing optimizations — see "Blocked on upstream" below.

## Planned Work (ranked)

The forward list — open items only. Landed work is recorded further down.

1. **Planning-overhead fast paths.** Single-op queries pay 60–80% of lazy
   overhead in the optimizer (`bench_planning_phases.py`); skip passes by
   plan shape. The ~300 µs limit floor is this item. (Low leverage:
   sub-millisecond, invisible on real data.)

This is the only remaining open performance item, and it is low-leverage. The
engine's real-operation gaps are now either upstream-blocked (Arrow
string-view `take`) or structural (pandas mutation semantics); see the
competitive-standing table.

### Considered and set aside (with measurements)

- **Free-threaded partition joins** — *mechanism validated, niche too narrow*
  (tested June 2026 on a CPython 3.14t build; `spike_freethreaded_join.py`).
  The hypothesis holds: with the GIL off, the partitioned hash join scales
  1→8 threads (1332→487 ms, **2.67x** over a single `pd.merge`), where on 3.11
  the same approach was *slower* than serial. numpy 2.4, pyarrow 24, and our
  pandas all keep the GIL disabled on import — the engine is free-threading
  ready. But **acero's internally-parallel C++ join is still faster** (337 vs
  487 ms at 8M), and acero needs no free-threaded interpreter. So the
  partitioned join only earns its keep for joins acero *cannot* do — nullable
  or float keys (acero uses SQL null semantics, not pandas NaN==NaN) and index
  preservation — under relaxed output order. Deferred until that niche matters
  or free-threading is the common deployment. (Most lazy kernels already
  release the GIL via numpy/arrow/nogil Cython, so they need no change.)

- **H2O q10 hash pre-partition** — *does not help* (measured June 2026). For a
  near-unique 6-key group-by (~10M groups/10M rows), our multi-key arrow path
  is 1731 ms — already 6x faster than a raw single `pa.Table.group_by` on six
  keys (11 s) and faster than every partitioned-acero variant (×16 parallel
  1923 ms; full split+agg+concat 4077 ms). The 0.66x residual vs Polars is its
  radix-partitioned contiguous-group-state aggregate engine — structural, not
  reachable by Python-level partitioning.
- **H2O q4 string-key join via factorize** — *slower, not faster* (measured
  June 2026). Factorizing both sides to a shared int code space (1726 ms) and
  categorical-encoding (1228 ms) both lose to plain `pd.merge` on the string
  keys (824 ms): they re-hash the 10M strings that `pd.merge` already
  factorizes internally once. The real fix is Arrow string-view storage
  (length-independent hashing + zero-copy gather) — large and structural.
- **Acero raw-string hash gap** — *the gap does not exist* (re-measured June
  2026). acero groups raw `large_string` keys at **80 ms/10M — faster than
  Polars' 118 ms** on the same raw strings. The ROADMAP's "Polars 18 ms" was
  Polars' *categorical* path (7.5 ms here), not raw-string. Pre-encoding the
  key costs more than it saves (dictionary_encode alone is 160 ms). The
  residual **categorical** groupby gap (0.45x, `engine_pipeline`) decomposes
  as the acero kernel on dictionary keys (~11 ms — already at Polars' pace)
  plus ~4.5 ms of pandas-NaN-semantics detection on the value column (a
  `skipna` correctness requirement, near-irreducible — `mask_nan_to_null`
  already short-circuits the masking pass when no NaN is present). No clean
  in-process win.
- **JAX/XLA kernel backend** — *declined June 2026 on measured grounds*, was
  the former "breakthrough candidate." Two findings sink it:
  1. **The CPU fusion lever is already pulled by NumExpr.** The engine
     already routes fuseable arithmetic chains (≥100K elems) through
     `numexpr_fusion.py`; measured 1.4x (`a+b`, bandwidth-bound) to 5.6x
     (`sqrt(exp(a)+sin(b)*cos(c))`, compute-bound) over naive NumPy. JAX-CPU
     (XLA) would compete with NumExpr, not naive NumPy — both fuse and
     multithread, so the marginal gain doesn't justify a heavy dependency.
  2. **GPU can't beat a bandwidth ceiling by adding a bandwidth-bound
     transfer.** A 10M-row chain moves ~240–480 MB; a discrete-GPU
     host→device→host round-trip (~15–30 ms over PCIe) already exceeds what
     NumExpr does the *whole* op in (9–16 ms). GPU only pays off when data
     is device-*resident* across many ops (the cuDF model), not converted
     per-chain. The narrow exception — transcendental-heavy numeric chains
     (ML preprocessing) — argues for a device-resident sub-pipeline, a
     different execution model, not a kernel backend. Revisit only for a
     workload that is both compute-bound *and* chains many such ops.

### API gaps vs Polars LazyFrame

The basic manipulation verbs are landed (`rename`, `drop`, `drop_nulls`,
`fill_null`, `cast`, `pipe`, frame aggregations, `unpivot`/`melt`); group-by
also has `agg` with arithmetic-over-aggregates + `corr`, and `head(n)` /
grouped top-k (`GroupByHead`). Remaining, roughly by value:

- **Time-series** — `join_asof` (as-of/nearest join), `group_by_dynamic`,
  `rolling` (time-windowed aggregation). The biggest area; a project on its
  own.
- **Row ops** — frame-level `top_k`/`bottom_k`, `reverse`, `slice`, `gather`,
  `gather_every` (mostly sort/limit reuse; grouped top-k already exists via
  `sort → group_by().head(k)`).
- **Reshape** — `pivot` (output schema is *data-dependent*, so it needs
  materialization — awkward in a lazy plan), `explode`/`unnest` (nested
  dtypes).
- **Niche** — `sql`, `merge_sorted`, `interpolate`, `update`, `set_sorted`,
  `map_batches`.

### Smaller items

- **Nullable dtype preservation** — the output contract preserves genuine
  `pd.ArrowDtype` but *not* pandas masked nullable (`Int64`/`Float64`): once
  a join/aggregate marks the schema nullable, a masked source is
  indistinguishable from its NumPy counterpart and comes out NumPy-backed.
  Needs the schema to track "originated as a masked extension dtype".
- **CSV limit pushdown** — Parquet scans got `ParquetSource.limit` + the
  direct small-limit path; CSV scans have neither.
- **JSON scanning** — `scan.py` accepts `format="json"` but raises
  `NotImplementedError`.
- **Partition-aware execution** — parallel processing of pre-partitioned
  (e.g. hive-partitioned Parquet) data.
- **Adaptive thresholds maturation** — the EMA tuner (`optimize/adaptive.py`)
  is experimental and off by default; the cost model (`cost.py`) is the
  natural calibration target.
- **RangeIndex preservation** — `preserve_index=True` materializes a
  RangeIndex as int64 values; could carry the range representation through.
- **Compute-bound kernel classes** — morsel parallelism recognizes `str_*`
  only; regex and date parsing are unmeasured candidates (`cost.py` /
  `engine/parallel.py`).

### Known bugs (not design choices)

- **Duplicate column labels** crash with `AttributeError` instead of a clear
  "unsupported" error at plan construction.
- **`shift` is unimplemented in the eager evaluator** while `lag` works in
  both engines — audit Expr-API coverage parity between engines.

## Blocked on upstream

- **Faster string gather (needs Arrow string-view `take`).** Arrow
  `large_string` `take` is ~271–375 ms/10M column — the bottleneck for both
  sort gather and `filter`; Polars does it in ~92 ms (4x). Investigated June
  2026 and found to be a **memory-bandwidth wall on the `large_string`
  layout**, not a missing parallelization:
  - A custom parallel gather (validated in numba) scales only 3.3x (1→8
    threads, 568→173 ms), tapering after 4. `_take_all_columns` already
    overlaps string columns across cores, so a per-column parallel kernel
    oversubscribes and doesn't cleanly win the multi-string-column case.
  - No free pyarrow threading: chunked input (313 ms) and `Table.take`
    (350 ms) are not faster than a single `take` (284 ms).
  - Dictionary `take` is 2.8x (101 ms) only when already-encoded; one-shot
    encode (107 ms) erases it. Categorical/cached columns get this free via
    the keycache; plain `large_string` does not.
  - **Polars wins by moving less data** via string-view ("German strings":
    16-byte inline views, no offset indirection or variable byte copy).
    That's the only way past the bandwidth bound, and **pyarrow 23 and 24
    both lack** `string_view` support in `array_take`, `hash_aggregate`, and
    `hash_join` (re-tested on 24.0.0 — identical failures). The
    dictionary-encoding "bridge" was scoped and then **measured-and-disproven**
    (`docs/STRING_STORAGE_SCOPING.md`): end-to-end it regresses 13x on q3
    because the one-shot encode (~180 ms) + output decode (~1900 ms) dwarf the
    gather saving, and string payload is only ~40% of the merge cost anyway.
    The only winning case (already-encoded input, Categorical output) is
    already covered by the Categorical→DictionaryArray path. So the remaining
    string gaps are genuinely structural; revisit only when pyarrow C++ ships
    `string_view` compute kernels.

## Recently Landed (June 2026 cycle)

Newest first. Detail in the git log and the docs above.

- **H2O db-benchmark + numeric-key group-by routing** — added a faithful port
  of the H2O.ai db-benchmark (`benchmarks/bench_h2o.py`, group-by + join vs
  Polars; see `docs/BENCHMARK_SUITES.md`). It immediately exposed that
  integer-keyed aggregation ran the slow pandas NumPy path (q4 0.06x vs
  Polars). Fixed by routing numeric-keyed aggregation to acero
  (`groupby_prefers_arrow`): **q4 226→19 ms (0.06x→0.68x), q5 0.76x→1.16x**.
  A benchmark-methodology fix (warm-up before timing, vs the original 2-run
  convention) then showed the apparent string-key group-by gap was a pure
  measurement artifact — string keys are `str`-dtype/arrow-backed and already
  beat Polars at steady state (q1 1.2x, q2 6.6x, q3 2.1x). Group-by is now
  competitive-to-winning across the board.
- **pd.merge join path** — the H2O joins (0.15–0.58x) ran a custom indexer
  hash join or Arrow/acero, both slower than just calling `pd.merge` — which
  *is* the eager semantics the join promises to match (row-order- and
  null-correct by construction) and avoids acero's Arrow↔pandas round-trip on
  payload columns. Routing in-memory equi-joins to `pd.merge` moved joins to
  **0.25–1.40x**: q2 0.32→1.40x (ahead of Polars), q5 10M×10M 0.20→0.88x
  (5236→733 ms). Remaining join gaps: string-key (q4 0.25x) and left (q3).
- **Core frame verbs** — `rename`, `drop`, `drop_nulls`, `fill_null`, `cast`,
  `pipe`, frame aggregations (`sum`/`mean`/`min`/`max`/`std`/`var`/`median`/
  `count` → one-row frame), and `unpivot`/`melt`, closing the most visible
  Polars LazyFrame API gaps. Also fixed a global-aggregation backend bug
  (NumPy column routed to an Arrow kernel in a mixed frame).
- **Cardinality column statistics** — sampled NDV (random sample, exact for
  low cardinality, extrapolated for high) + exact min/max, computed lazily
  and cached on `DataFrameSource`, propagated through pass-through ops.
  Equality sizes by `1/NDV`, ranges by min/max interpolation, grouped-agg
  row count by the key's NDV (was a sqrt heuristic). Costs nothing for
  join-free queries (only computed when a join/agg estimate is requested).
  `ParquetSource` gets the same from row-group footer metadata (min/max +
  null count) with zero data read — range and is_null selectivity for scans.
  Range selectivity uses an **equi-depth histogram** (sampled quantiles) so it
  tracks skew — `col < median` of a lognormal estimates ~0.5, where flat
  min/max interpolation collapses to ~0.001.
- **String-key multi-column sort** — a string sort key is factorized
  (`sort=True`) into order-preserving float codes (nulls as NaN), so the
  radix lexsort handles it with per-key descending and nulls-last for free;
  `sort(group[str], value1)` 10M went 3150→1643 ms (0.46→~0.88x of Polars),
  ~3.7x faster than the Arrow string-key argsort. Non-numeric/non-string
  keys (datetime) still fall back to Arrow.
- **Predicate-aware cardinality (System R selectivity)** — `Filter` /
  pushed-down Parquet predicates size by operator (equality 0.1, range 0.33,
  AND/OR/NOT compose) instead of a flat 0.3; flips join build-side selection
  to the actually-smaller filtered side. Statistics-driven refinement (NDV,
  histograms) is the remaining Planned Work item.
- **Radix multi-key (lexsort) sort** — `radix_lexsort` composes per-key
  parallel-radix sorts (least-significant first); all-numeric multi-key
  sorts now run ~6x faster than Arrow's table `sort_indices`
  (`sort(group, value1)` 10M: 2332→765 ms), with per-key descending/NaN
  handled natively. String-keyed multi-key sorts still use Arrow.
- **Thread-parallel radix argsort** — `_radix_sort_parallel` over the nogil
  phase kernels in `pandas/_libs/lazy_radix.pyx`; ~130 ms @10M float64,
  **1.6x faster than Polars**. 11-bit digits keep each chunk's histogram
  cache-resident; serial 16-bit kernel below `RADIX_PARALLEL_MIN_ROWS` (1M).
- **Cython LSD radix argsort** — replaced the k-way merge for large numeric
  single-key sorts; keys built vectorized in NumPy (sign/float transforms,
  -0.0 normalized), stable, matched Polars serially (~215 ms).
- **String-gather investigation** — characterized the bandwidth wall above;
  no clean in-process win, recorded the upstream dependency.
- **Breaker copy-free output** — sort/groupby/join/distinct/topk assemble
  with `copy=False` (every column is a fresh take/aggregate); projections
  keep the safety copy. `TestCollectDoesNotAliasSource` pins both.
- **Output dtype contract** — `convert.arrays_to_dataframe` returns the
  dtypes eager would (numeric/bool → NumPy, strings → `str`, ArrowDtype and
  Categorical preserved); no more `double[pyarrow]` leaking from acero.
  Schema-gated so internal join/spill round-trips are untouched.
- **`collect(order="relaxed")`** — widens acero join routing to terminal
  joins when the user opts out of eager row order (10M×1M inner 298→142 ms,
  2.1x; left 232→71 ms, 3.2x), with intermediate order-dependent ops still
  blocking it.

## Where Lazy Already Wins (keep protected by benchmarks)

| Scenario | Typical speedup vs eager | Mechanism |
|----------|--------------------------|-----------|
| Sequential filters | 1.5–2x | filter fusion |
| filter + `head(N)` | 2–10x (10x+ on multi-file) | streaming early termination + scan limit pushdown |
| Arrow string pipelines | 2–10x | Arrow kernels + morsel parallelism |
| Numeric single-key sort | argsort 1.6x over Polars | thread-parallel radix kernel |
| Repeated string-key groupbys | ~3x from 3rd query | dictionary-encoding cache |
| Multi-step pipelines | 1.2–1.5x | reduced materialization |
| Larger-than-memory | n/a (enables) | streaming + spill |

Regressions here should be caught by `benchmarks/bench_optimizer_quality.py`
(plan shape) and the benchmark suite (timings).
