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

**TPC-H is the other side of the story — and the honest one for *pipelines*.**
The TPC-H/PDS-H harness (`../benchmarks/bench_tpch.py`, every query validated
exact against DuckDB's `PRAGMA tpch(n)`) stress-tests full analytical pipelines
(multi-table joins + filters + group-by + sort). **All 22 queries are
implemented and validate exact against DuckDB; Polars wins every one,
0.06–0.67x at SF-1.** The single-op H2O wins do *not* extend to multi-op
pipelines: our engine pays a pandas↔Arrow conversion at each operator boundary,
while Polars stays in one native columnar representation end to end, so the
conversions compound across a deep pipeline. (Measured fairly — each engine on
its native input, query only; an earlier revision timed Polars' `from_pandas`
per run and so inflated our ratios into apparent wins. Corrected.) So:
**competitive-to-winning on single operations (H2O); behind on full analytical
pipelines (TPC-H).** Building out the 22 queries also surfaced **three real
engine bugs, now fixed**: a datetime-filter comparison (~150x on our own prior
path), a CSE expression-conflation correctness bug, and `is_null`/`is_not_null`
missing float NaN on the Arrow backend (which silently broke the left-join
anti-join idiom).

**Where the pipeline gap actually is (measured on Q5, not assumed):** profiling
puts ~89% of Q5 in the joins and ~52% in `pd.merge`'s key factorization — *not*
in pandas↔Arrow conversions (which don't appear) and *not* in a slow per-join
kernel (a single TPC-H join is `pd.merge` 68ms ≈ acero 67ms ≈ Polars 53ms, only
~1.3x). The 9x chain gap is Polars' **cost-based join execution**: it reorders
the join tree and pipelines without materializing intermediates. This is not a
cheap win, and **naive join reordering backfires** — hand-reordering Q5 to apply
the `region='ASIA'` restriction first made it *25x slower* (5.5s vs 0.22s),
because the early joins then land on low-cardinality `nationkey` and explode the
supplier×customer intermediate. Closing the gap needs a real cost-based join
reorderer (cardinality-driven) plus pipelined joins — mature-optimizer work.

**Join-reorder prototype (built, opt-in via `compute.lazy.join_reorder`, OFF by
default — `optimize/join_reorder.py`).** A greedy left-deep reorderer with a
System-R cardinality model (NDV from the existing sampler). It confirms the
lever: on a controlled star join written in a bad order it cuts time ~1.8x and
the *optimal* order makes our engine **beat Polars** (49 ms vs 122 ms — and
Polars does not cost-reorder multi-way joins at all, so a working reorderer
would let us leapfrog it). But it is **off by default because the cardinality
estimator is the wall**, exactly as Leis et al. (VLDB 2015) predict: sampled
NDV underestimates (`l_partkey` 200k→56k), which hides that joining a *filtered*
dimension early is restrictive — so on hand-ordered TPC-H Q9 the model is
confidently *backwards* (rates the bad order 3x cheaper) and a confidence margin
can't save a backwards model. The honest blockers, in order: (1) reliable
distinct-count estimation (HyperLogLog-class), (2) bushy enumeration (GOO) so a
restrictive filtered dimension can stay a small separate subtree, (3) pipelined
joins. The enumeration is the easy part; cardinality is the real work.

| Category | Standing | Driver |
|----------|----------|--------|
| string | **2.14x avg — wins** (`str.lower` 5.30x, `contains` up to 1.55x) | pass-through fix + compute-bound morsel parallelism |
| aggregation | **wins** (H2O group-by: 6/10 beat Polars, string-key sums up to ~7x, `corr` ~3x) | `groupby_prefers_arrow` routing (numeric + arrow-string → acero) + post-aggregation projection |
| parquet scan | 0.88x avg — glob `head()` **wins 1.5x** | limit pushdown into scans + direct ParquetFile path + vectorized index |
| join | **`pd.merge` path** (eager-correct, fastest — 5x over acero on this shape). H2O 1:1-ish joins ~parity (noisy); the microbenchmark's *many-to-many* ~100M-row-result join is ~0.3x (result materialization, where Polars parallelizes); string-key/left joins lag — see H2O_BENCHMARK.md | `pd.merge` in-memory equi-join; acero/Grace fallbacks |
| sort | string-key multi-sort **wins 1.09x at 10M**; numeric ~parity (0.91–0.97x at 10M); smaller sorts and full mixed sort gather-bound (0.4–0.8x) | radix argsort + radix lexsort (numeric & factorized string keys) + breaker copy-free output; string payload gather is the wall |
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

## The Plan: Competing with Polars

Goal stated June 2026: actually compete. Everything below is grounded in the
measured record (H2O, TPC-H, the join investigation), not aspiration.

### What "compete" means — two scenarios, one metric each

- **S1 — native vs native** (engine quality): each engine on its own format,
  query only. TPC-H SF-1 geometric mean today: **0.22x** (range 0.06–0.67;
  5/22 queries ≥0.4x, 11/22 below 0.2x). H2O single-op: we win 6/10 group-by.
- **S2 — pandas-resident** (the data already lives in pandas DataFrames):
  Polars must pay `from_pandas` once; we never do. **Measured (P0, June
  2026): the moat is thin for heavy analytics.** Converting all 8 SF-1
  tables costs only ~400 ms one-time, while our suite deficit is ~3.2 s per
  pass — so converting to Polars pays for itself within 1–10 runs of most
  queries (for the heaviest, q21, from the very first run). The moat
  genuinely protects only light or few-run usage (q6/q11/q14 break-evens of
  ~20–30 runs). See `../benchmarks/TPCH_BENCHMARK.md`.

The P0 measurement reframed the bet: **S2 does not rescue the standing — S1
engine work is the only real path.** The realistic goal is still not full
native parity across all 22 queries against a decade of Rust/SIMD/Rayon; it
is: (1) close S1 enough (**≥0.5x geo-mean** mid-term) that the per-query
deficit shrinks toward the conversion cost and the S2 break-evens swing our
way; (2) keep and extend the outright wins (group-by analytics, string ops,
large sorts); (3) publish both scenarios honestly — the S2 break-even table
is the user-facing decision tool either way.

### Where the S1 gap lives (measured, with disproven paths marked)

1. **Join-chain execution dominates** — q5 profile: ~89% in joins, ~52% in
   `pd.merge`'s single-threaded `_factorize_keys`. A *single* join is ~1.3x
   off Polars (68 vs 53 ms); chains are 3–7x off because per-join factorize +
   full intermediate materialization compound, while Polars runs a parallel
   hash join and keeps intermediates flowing.
2. ~~Column pruning through joins~~ — **disproven** (manually pre-projecting
   q3's inputs changes nothing; the optimizer already prunes).
3. ~~Join order on hand-written queries~~ — order is the lever only for
   *naively written* queries (star join: optimal order beats Polars 49 vs
   122 ms); the reorderer exists but is blocked on NDV estimation quality.
4. **String gather wall** (`large_string` bandwidth; `string_view` blocked
   upstream in pyarrow 23/24) — caps filter_project (0.21x) and string-key
   joins.
5. **Planning overhead floor** (~300 µs; 60–80% of single-op lazy overhead in
   the optimizer) — hurts small interactive queries only.

### Roadmap (phased, each with a measured gate)

- **P0 — Scorecard. DONE (June 2026).** `bench_tpch --report` now emits the
  dual-scenario table (`../benchmarks/TPCH_BENCHMARK.md`): S1 never times
  `from_pandas` (the fixed fairness rule stays); S2 times the conversion
  exactly once and reports per-query **break-evens** (never charged per
  query). *Result: S1 geo-mean 0.23x; conversion ~400 ms vs ~3.2 s/pass
  suite deficit — the S2 moat is thin for heavy analytics, so P1/P2 carry
  the plan.*
- **P1 — Parallel join kernel (in progress).** The single biggest S1 lever.
  **Candidate measurements done (June 2026), on q3's real filtered shapes**
  (c 30k ⋈ o 727k ⋈ li 3.2M; Polars reference on identical inputs: **6.8 ms**
  vs our `pd.merge` chain 84 ms — 12x):
  - ~~(a) acero~~ — 69 ms (1.2x): parallel C++ but the Arrow round-trip and
    its probe cost leave it nowhere near Polars. Insufficient.
  - ~~(b) partitioned parallel `pd.merge`~~ — 86 ms: **no win** (GIL +
    concat overhead eat the parallelism).
  - ~~(c′) NumPy direct-address (LUT) join~~ — 3.1x standalone on the
    dominant join (72→23 ms, dense unique int build keys), but **in-engine it
    is suite-neutral (1.01x total across all 22 queries)** and unstable
    per-query even after five eligibility gates (build-side screen, density
    vs probe, sampled selectivity pre-probe, hashtable uniqueness pre-check,
    probe/build ratio cap). Root cause: the multi-pass NumPy probe has a
    ~8.5 ms/M-row floor, vs `pd.merge`'s shape-dependent 5–25 ms/M — the
    crossover is too narrow to gate reliably. Built, measured, **reverted**.
  - **(c) Cython single-pass hash join — LANDED (June 2026,
    `pandas/_libs/lazy_join.pyx`).** CSR-grouped hash table built on the
    small side; `nogil` count/fill probe passes driven thread-parallel
    (no-OpenMP pattern, like `lazy_radix`); output is `pd.merge`'s exact
    inner order by construction (probe-in-row-order; one stable integer
    argsort restores it when the build side is the left). At the indexer
    level it is **7–11x over `pd.merge`** on the dominant shapes (q3's big
    join: 6.5 ms vs 72.9 ms). In-engine, gated to *selective* joins
    (sampled hit-rate ≤0.5, build side ≤500k — on high-hit wide
    intermediates the per-column gather loses to pd.merge's consolidated
    block take, measured q7 1.85x before the gates): **q3 0.60x→, q5
    0.72x→, q10 0.87x→, q18 0.63x→ of their former times, no real
    regressions** (borderline cases are ≤1.03x at 4 reps). All 22 queries
    still validate; 1677 tests pass.
  - **P1 remaining:** the bottleneck has *moved* — on selective joins the
    kernel is no longer the cost; on high-hit joins the payload gather of
    wide intermediates is (pd.merge's block-take advantage). That is P2
    territory (pipelined joins / block-aware gather). *Gate progress:
    q3/q5/q10 at 1.15–1.7x of the ≥2x target; S1 geo-mean re-measure due.*
- **P2 — Late-materialization join chains. LANDED (June 2026,
  `PhysicalJoinChain`).** A planner post-pass collapses left-deep trees of
  eligible inner single-int-key joins into one breaker fed the BASE
  relations; it composes the Cython kernel's indexers step by step (only
  each probe key is gathered) and gathers every payload column exactly once.
  Two hard-won lessons: (1) the executor rebinds breaker children to
  pre-materialized inputs, so the chain had to be a planner-level node — a
  direct-recursion version could never fire; (2) **order-freeness is the
  win condition on full-hit chains** — the decision layer (acero's
  order-unobservability walk) marks chains feeding group-by/sort sinks
  `order_free`, skipping cascade-order restoration (q18 flipped from a
  1.83x loss to a 0.70x win on that flag alone). Controlled on/off, all 22
  validated: q5 0.48x, q2 0.49x, q11 0.69x, q18 0.70x, q7 0.72x, q3 0.75x —
  8 wins, zero regressions. **S1 geo-mean: 0.22x (pre-P1) → 0.27x.**
  *Gate status: q7 1.39x / q18 1.43x of the ≥1.5x target (q21's cost is not
  its joins); geo-mean 0.27 vs ≥0.35 — the join side has largely delivered;
  the laggards are now non-join costs: q19 (big-OR filter evaluation,
  561 ms), q21 (n_unique-heavy group-bys, 1.4 s), q22/q15/q16 (strings &
  small-query floors). The next profile-driven target list lives there.*

- **P2.5 — DONE (June 2026): q19 + q21 attacked, geo-mean 0.27 → 0.32x.**
  - **q19 550 → 138 ms (0.13 → 0.55x)**: the probe disproved the conversion
    hypothesis (arrays were already Arrow; `pc.is_in` itself costs ~58 ms on
    6M `large_string` rows) — the real fix was **predicate derivation** in
    PredicatePushdown's cannot-push join branch: side-only conjuncts of a
    mixed AND push fully (and drop from the upper filter); a mixed
    OR-of-conjunctions pushes the implied side-only OR in addition.
  - **q21 1400 → 780 ms (0.13 → 0.25x)**: acero `count_distinct` disproven
    (448 ms — slower than the pandas fallback). Landed a packed-dedup NumPy
    `n_unique` kernel (pack (key,value) into int64, `np.unique`, run-length
    count: 183 ms) and excluded `n_unique` from arrow groupby preference so
    it actually runs (the rerouting also exposed and fixed a multi-key
    `n_unique` spelling crash).

- **P3-next — ranked from the fresh scorecard (S1 geo-mean 0.32x, suite
  3.0 s vs Polars 1.0 s; biggest remaining laggards by absolute time):**
  1. **q21 remainder (746 ms, 0.25x — the largest single chunk).** The
     group-bys are fixed; what remains is its big⋈big join chain (late
     3.8M ⋈ orders-F 730k ⋈ two 1.5M-row aggregate sides). Suspect the
     Cython-join build cap (≤500k) and the chain's eligibility exclude
     these; profile, then consider raising the cap for the parallel kernel
     / letting order-free chains take big builds.
  2. **String-heavy q22 (222 ms, 0.15x) and q13 (211 ms, 0.48x).** q13's
     left join is chain-ineligible and its `str.contains` regex runs over
     1.5M comments single-threaded; q22 is `str.slice` + cross + anti.
     Leads: morsel-parallel string kernels, left-join chain support.
  3. **Small-query floors: q15/q16/q20/q8/q14 (20–60 ms absolute).**
     Dominated by per-operator overhead and planning (the long-standing
     "planning-overhead fast paths" item) — worth one profiling pass to
     see if a shared fix moves all five.
  *Gate to close P3-next: S1 geo-mean ≥ 0.4x.*

- **Scale check (June 2026): SF-3 is now the scorecard standard**
  (`../benchmarks/TPCH_BENCHMARK_SF3.md`; SF-1 kept for fast inner-loop
  comparisons). Rationale: at SF-1 half the queries run in 20–150 ms where
  run-to-run noise is a meaningful fraction; the machine has 16 GB, so SF-5+
  risks the swap distortion seen before (SF-3 ≈ 7 GB resident across all
  three engines). Findings at SF-3 (geo-mean **0.34x**, slightly better
  than SF-1's 0.32 — the engine scales):
  - **q7 reaches 0.95x — near parity with Polars** on a 5-table 18M-row
    join pipeline.
  - **q22 is the scale anomaly: 0.15x → 0.05x** (LP 222 → 1006 ms, ~4.5x
    for 3x data — superlinear; Polars 47 ms). Suspects: `distinct` over the
    4.5M-row `o_custkey`, the left-join anti-join path, or `str.slice` —
    profile first. This jumps to the top of the target list with q21
    (2.6 s at SF-3).
  - Small-query floors matter less at scale, as expected (q15/q16 ratios
    improved); q20 scales linearly on both engines (ratio stable at 0.18).
- **q21 decomposition (SF-3 per-operator profile, June 2026):** the two
  n_unique aggregates 531+312 ms (now ~2x faster via stable-radix-sort
  dedup in the packed kernel), the big-big JoinChain 499 ms, and ~1.2 s of
  distributed cost (sub-60 ms operators; the `late` filter subtree likely
  executes twice — no common-subplan reuse across pipelines). q21
  2759 → 2087 ms. Remaining leads: common-subplan caching, the chain's
  build caps at scale.
- **P3 — Cardinality, then reorder default-on.** Exact NDV for small relations
  (dimensions are cheap to count exactly), HyperLogLog-class sketches for fact
  tables; re-test the q9 backwards-model case; then enable `JoinReorder` by
  default behind its confidence gate; bushy (GOO) after. *Gate: naive-order
  queries ≥2x with zero regressions on hand-ordered ones.*
- **P4 — String representation (parallel track, long lead).** The only path
  past the bandwidth wall is `string_view` ("German strings"): contribute
  take/hash kernels to Arrow upstream, or vendor a Cython string-view gather.
  *Gate: filter_project 0.21x → ≥0.5x; string-key join parity.*
- **P5 — Positioning.** Take the dual-scenario scorecard to the PDEP /
  maintainer discussion (`pandas.lazy` namespace, `select()` entry point —
  the contested questions need the numbers this plan produces).

Floor work continues alongside: masked-`Float64` storage flag in the type
model, the arrow-group-by masked-`boolean` crash, `rank` dtype, CI.

## Planned Work (ranked)

The forward list — open items only; the phases above sequence the big ones.
Landed work is recorded further down.

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

- **Nullable dtype preservation** — *partially fixed.* The output contract now
  restores masked **`Int*`/`UInt*`/`boolean`** (NumPy int/bool can't be null, so
  a `nullable` int/bool dtype can only be a masked input; aggregate inference
  was also fixed to propagate input nullability so `sum(numpy int)` stays
  `int64` while `sum(Int64)` stays `Int64`). **`Float64` remains a gap**: NumPy
  `float64` holds NaN and is also flagged `nullable`, so LazyDtype cannot tell
  masked `Float64` from NumPy `float64` — fully fixing it needs the type model
  to carry a real masked-storage flag (set in `from_pandas_dtype`, propagated
  through `infer_expr_dtype`). Genuine `pd.ArrowDtype` is preserved as before.
- **Masked `boolean` in the arrow group-by** crashes (`ArrowInvalid: Could not
  convert <NA> ... to boolean`) when a masked boolean column with `pd.NA` is
  present in a grouped frame — the acero path can't ingest it. Pre-existing;
  needs a null-safe conversion in `_execute_arrow_table_groupby`.
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

- ~~**Duplicate column labels** crash with `AttributeError`~~ — fixed: now a
  clear `NotImplementedError` at plan construction.
- ~~**`shift` unimplemented in the eager evaluator**~~ — fixed. The parity
  audit it prompted also fixed `clip` and `diff` (physical-only → both engines).
- **Minor parity gaps remaining** (low priority): `rank` returns int on the
  physical path vs float (pandas) on the eager path — the dtype is fixed by the
  type model but overridden somewhere in the physical dispatch (multiple rank
  code paths); and `abs`/`round` have no `Expr` method yet (API addition, not a
  parity bug).

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
