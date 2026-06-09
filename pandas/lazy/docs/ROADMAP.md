# Lazy Pandas Roadmap

What we plan to do next, what just landed, and the known gaps. Implemented
architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md),
[PLANNING.md](PLANNING.md), [OPTIMIZER.md](OPTIMIZER.md), and — for the
execution engine — [ENGINE_DESIGN.md](ENGINE_DESIGN.md) (six milestones, all
landed). Dated performance reports live in `../benchmarks/`.

## Competitive Standing (vs Polars, June 2026 — engine era)

From `../benchmarks/LAZY_VS_POLARS_BENCHMARK.md` (1M–10M rows, mixed-dtype
data, Apple Silicon; physical engine). Speedup > 1.0 = lazy pandas faster.

| Category | Standing | Driver |
|----------|----------|--------|
| string | **2.07x avg — wins** (`str.lower` 4.38x, `contains` 1.71x) | pass-through fix + compute-bound morsel parallelism |
| aggregation | **1.17x avg — wins** (multi-agg 3.47x, groupby-sum 1.02x @10M) | groupby routing + dictionary-encoding cache (warm from 2nd query) |
| parquet scan | 0.86x avg — glob `head()` **wins 1.55x** (6.7 ms vs 10.3) | limit pushdown into scans + direct ParquetFile path + vectorized index |
| join (order-preserving) | 0.42x default; **0.89x with `order="relaxed"`** | eager `pd.merge` row order by default; `collect(order="relaxed")` routes to acero |
| join→groupby composite | 0.58x | acero routing for order-free joins |
| sort | numeric single + multi-key argsort **beats Polars**; full mixed sort gather-bound | thread-parallel radix argsort + radix lexsort + breaker copy-free output; string gather is the wall |
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

1. **String-key multi-column sort.** The radix lexsort (landed) covers
   all-numeric multi-key sorts; a multi-key sort that includes a string
   column still falls back to Arrow's table `sort_indices`. A dictionary-
   encoded key (int32 codes feeding the radix lexsort) would extend the win
   there, bounded by the same `large_string` gather wall (Blocked, below).
2. **Acero raw-string hash gap.** acero groups raw `large_string` keys at
   67 ms/10M vs Polars' 18; the dictionary cache solves repeated queries
   but first-query and one-shot workloads still pay. Upstream Arrow work or
   a pre-hashing trick are the options.
3. **Cardinality estimation.** Row estimates stop at filters (no selectivity
   model); estimates feed the decision layer's join build-side and
   parallelism-degree choices, so better estimates compound.
4. **Planning-overhead fast paths.** Single-op queries pay 60–80% of lazy
   overhead in the optimizer (`bench_planning_phases.py`); skip passes by
   plan shape. The ~300 µs limit floor is this item. (Low leverage:
   sub-millisecond, invisible on real data.)
5. **Free-threaded partition joins.** pandas' Cython hash join holds the GIL
   (measured: threaded partition-pairs 461→535 ms at 2→8 threads vs 430
   serial). On free-threaded Python the M5-spec'd partitioned join becomes
   buildable; the engine architecture needs no changes to exploit it.

### Considered and set aside (with measurements)

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
    That's the only way past the bandwidth bound, and pyarrow 23 has no
    `array_take` kernel for `string_view`. Revisit when pyarrow ships it.

## Recently Landed (June 2026 cycle)

Newest first. Detail in the git log and the docs above.

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
