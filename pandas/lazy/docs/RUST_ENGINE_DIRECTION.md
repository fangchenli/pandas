# The direction: an Arrow-native Rust execution engine (June 2026)

This supersedes the campaign's recurring "joins/TPC-H are architecture-bound /
we're stuck vs Polars" conclusion. **That conclusion was wrong** — it was a
failure of approach, not a real wall.

## The error
Every join/perf attempt bolted a fast kernel onto the **Python/Cython
column-major engine** and paid the boundary (Python gather, Arrow↔NumPy per
operator, GIL orchestration), then blamed "the architecture." The
contradiction that exposed it: a **Rust kernel already beat Polars** on the join
gather (`JOIN_KERNEL_REBUILD_PROBE.md`: 125/179/339/727 ms vs Polars
133/192/395/762), yet the integrated all-22 *regressed* — because the gather
went back through the Python engine. The fix was never "give up"; it was **move
the execution into Rust** so the boundary is paid once, not per operator. The
wall was self-imposed.

## The proof (SF-3, 8 cores; `benchmarks/rust_engine_prototype/`)
- **The pandas↔Rust Arrow boundary is free.** RecordBatch round-trip
  pandas→Rust→pandas = **0.01 ms** (zero-copy Arrow C-data interface); numpy↔arrow
  = 0.001 ms for numeric. The "boundary tax" was the *per-operator* Arrow↔NumPy
  churn *inside the Python engine* — an Arrow-native Rust engine has none of it.
- **TPC-H q1 executed end-to-end in Rust** (filter + group-by + multi-agg, on
  Arrow, rayon, one boundary crossing): **30.7 ms vs Polars 243.9 ms = 7.95x**,
  **correct vs DuckDB**. Our Cython engine on the same query: 0.47x of Polars.

- **TPC-H q3 executed end-to-end in Rust** (3 filters + 2 joins + high-card
  group-by + top-k — the hard, join-heavy shape where Polars is strong): **80.9
  ms vs Polars 77.0 ms = 0.95x (parity), correct vs DuckDB.** Our Cython engine:
  0.43x (200 ms) — so the Rust engine is **2.3x ours** and at Polars parity.
  (A fast i64 hasher took it 0.87x→0.95x; swapping the HashMap semi-joins for the
  proven partitioned join kernel would push past Polars.)

So across the spectrum: favorable shape (q1) **8x** Polars; hard join-heavy
shape (q3) **parity** — always ~2–8x our Cython engine. The point stands:
**Arrow-native Rust execution, boundary-once, matches/beats Polars** — "be Polars
under the pandas API," which we have the kernels for.

## General plan→Rust executor — built, correct, FUSED, beats Polars
Built a general executor (`benchmarks/rust_engine_prototype/src/engine.rs`): a
serialized plan (JSON: scan/filter/project/aggregate/sort/limit + an expression
interpreter) executes over Arrow tables — queries route through it, not a
hand-written `run_qN`. **Correct vs DuckDB on q1 and q6.**

Two stages, measured (SF-3):
1. *Naive materializing* (operator-at-a-time over full batches): q6 0.78x, q1
   0.25x — materializes between operators (copy at filter, arithmetic
   intermediates at project, re-scan at aggregate).
2. **Fused / morsel-pipelined aggregate** (`Aggregate ← (Filter|Project)* ←
   Scan` → cache-resident 64K sub-morsels, row-wise chain fused per morsel,
   thread-local partial aggregates merged, rayon-parallel): **q6 ~2.6x, q1
   ~1.7x — beats Polars**, through the *general* JSON-plan path.

The key was **cache-resident morsels**: a first cut used N/nthreads (~2.25M-row)
morsels whose intermediates spilled to DRAM (q1 only 0.69x); fixing to 64K
sub-morsels made the fusion real (q1 → 1.7x). This is the Polars/DuckDB
vectorized-morsel model, now ours.

3. **Join operator + join-aggregate fusion** (`Aggregate ← [Project ←] Join`):
   build the hash once on the build side, then morsel-probe the probe side and,
   per cache-resident morsel, gather the joined rows + apply the outer project +
   partial-aggregate — one fused pass (the run_q3 shape, generalized). TPC-H q3
   (2 joins + high-card group-by + top-k) through the general plan path:
   **115.9 ms vs Polars 86.2 = 0.74x**, correct vs DuckDB — up from 0.37x
   (naive join) and **2.7x our Cython engine** (0.43x).

So the full TPC-H operator set (scan/filter/project/join/aggregate/sort/limit)
now runs through the general plan path: **aggregate-heavy queries beat Polars
(q1 1.7x, q6 2.6x); the join-heavy q3 is 0.74x** (2.7x our engine). The q3
residual vs Polars is optimization headroom the hand-written run_q3 (0.95x)
shows is reachable — semijoin instead of a materialized build side, and fewer
per-morsel gathers.

## LogicalPlan → Rust translator — real queries auto-route (June 2026)
Built `benchmarks/rust_engine_prototype/translate.py`: walks an optimized
`LogicalPlan` → the engine's JSON plan + Arrow tables (df→Arrow once, the
boundary), executes via `lazy_engine_rs.execute`, restores output dtypes from
the plan schema. Maps the IR (`FieldRef`/`Literal`/`Alias`/`Call`/`Cast`,
canonical names add/subtract/multiply/divide, less/greater/equal/…, and_/or_,
sum/mean/min/max/count), hoists non-trivial agg-input expressions into a project,
and handles inner single-key joins. Unsupported → `NotSupported` (caller falls
back). Added numeric coercion in the engine (int-literal vs float-column → f64).

**Real lazy queries now auto-route into the Rust engine** (all-22 at SF-3,
execute-only timing, validated vs DuckDB):
- **q1 = 1.77x Polars, correct — fully automatic** (plan → Rust, no hand-code).
- q6 0.59x (translated plan doesn't hit the fused pattern), q5 0.13x (naive
  multi-join), q17 exec-error.
- **Coverage 4/22 translate.** The other 18 use ops/exprs not yet built: TopK
  (q2/3/10/18/21), Distinct (q4/20), case_when (q8/12/14), dt_year (q7/9),
  cross/left join (q11/15/13), isin (q19), n_unique (q16), is_null (q22).

So the *integration is proven* — the architecture routes real plans and q1 beats
Polars automatically. Reaching all-22 is a **coverage + per-shape fusion grind**:
add the missing operators/exprs, make fusion trigger on more translated shapes
(q6), fuse multi-joins (q5), fix q17. Mechanical, not architectural.

**Coverage grind COMPLETE (June 2026): 4 → 22/22 translate, 22/22 validated
vs DuckDB.** The entire TPC-H suite now compiles from a lazy-pandas
`LogicalPlan` and executes through the general Arrow-native Rust engine,
bit-correct against DuckDB. What it took (all mechanical — operators/exprs +
bug fixes, zero architecture change):

- **Exprs:** `dt_year` (cast→timestamp→date_part), `is_null`, `invert`, `isin`,
  `case_when` (zip), `str_startswith/endswith` (literal), `str_contains` (real
  regex via the `regex` crate — it was literal substring, so `special.*requests`
  on q13 matched 0), `str_slice` (substring).
- **Operators:** `TopK`→sort+limit, `Distinct` (dedup keep-first), `n_unique`
  (null-aware), **left outer join** (NULL right-index → null right cols),
  **cross join** (cartesian; right is a 1-row scalar threshold in TPC-H),
  **multi-key join** (composite i64 hash of the key columns).
- **Bug fixes:** Project/apply_ops broadcast bare-literal columns to `num_rows`
  (the `__g=liti` global-aggregate const-key was length-1 → panic on q14/q17/q22);
  join_exec keeps the right key column unless its name collides with a left
  column (q3 groups by `l_orderkey`, which was being dropped); null-aware
  count/mean/sum/min/max/n_unique (left-join unmatched rows must not count, fixed
  the missing `c_count=0` bucket on q13); `fused_join_aggregate` bails to naive
  on `how!=inner` / multi-key (it was silently doing an inner single-key join);
  `exec_compiled` restores datetimes via the real `Schema` API; the validate
  harness compares datetimes resolution-agnostically (us vs ns are the same
  instant — Polars emits us, an arrow-rs engine emits ns).

So the engine is no longer a q1/q3/q6 proof-of-concept: it runs **all 22**
end-to-end and correct. Per-shape fusion (making every translated query *fast*,
not just correct) is the remaining performance work — aggregate-heavy queries
already beat Polars; some multi-join shapes still hit naive paths.

## The direction
Build the lazy engine's execution in **Rust on arrow-rs**: `LogicalPlan` → Arrow
in once → execute operators in Rust (rayon, no GIL, hand-tuned hot kernels; port
Polars' join where useful) → Arrow out once. This becomes the **baseline** the
project measures from and builds on, replacing the ~0.43x Cython engine as the
performance floor. Status: hand-written queries (q1 8x, q3 parity) prove the
*ceiling*; the general executor proves the *translator*; **operator fusion** is
the bridge between them.

Path (incremental, each validated vs DuckDB + benched vs Polars):
1. Operator coverage: joins (the proven fused kernel), high-card group-by, sort,
   project/filter, limit/top-k — the TPC-H set.
2. `LogicalPlan` → Rust execution (a thin translator; the `pl_q*` functions show
   our plans map cleanly).
3. Wire as the engine's execution backend (Arrow-native end-to-end), fall back
   to the Cython engine only for pandas-specific semantics it can't express.
4. Build integration decision: Rust via maturin as an optional accelerator vs a
   hard dependency (this is the real cost; the engineering is not the blocker).

## Durable lesson
"Polars is Rust" was the answer all along. A coding agent can write the kernel in
Rust/C/asm or port Polars' — so "stuck" was never the right conclusion. The job
was to **stop hosting fast kernels in a slow Python engine** and put the
execution where the speed is. Boundary-once Arrow-native Rust is that place.

---

## Checkpoint (2026-07-08) — engine shape, honest perf, next lever

Paused here at a clean checkpoint. Branch `lazy-pandas`, all work committed
(coverage grind `0eb8fcb359`..`cba294db4b`, merged up to `upstream/main`).
Nothing in flight.

### The shape of the engine (as built)
`benchmarks/rust_engine_prototype/` — NOT in the pandas build; maturin module
`lazy_engine_rs` (~1021 lines `engine.rs` + ~12KB `lib.rs` + 281-line
`translate.py`). It is a **GIL-released, rayon-parallel, tree-walking Arrow
interpreter** with two fused fast-paths:

- **Boundary (`lib.rs`):** one real entry `execute(plan_json: str, tables:
  dict[str, RecordBatch]) -> RecordBatch`. Python builds Arrow tables ONCE
  (`translate.py`), hands JSON plan + tables across the Arrow C-data interface
  (zero-copy), Rust deserializes (serde) and runs the whole thing under
  `py.allow_threads` (GIL released). Legacy hand-written `run_q1`/`run_q3`
  probes still exported but unused by the general path.
- **Plan IR** (serde `tag="op"`): `Scan | Filter | Project | Aggregate | Sort |
  Limit | Join | Distinct`. **Expr IR** (`tag="t"`): `Col | Liti | Litf |
  LitStr | Bin | Unary | Isin | Case | Str | Slice`, tree-walked by `eval()`
  over arrow-rs compute kernels.
- **Execution = `execute(plan)` recursive tree-walker, operator-at-a-time.**
  Most ops MATERIALIZE full output. `Aggregate` is the only smart node — it
  tries two fusions before falling back:
  1. `fused_join_aggregate` (`Aggregate ← [Project ←] Join`): build hash once,
     morsel-probe, per-64K-morsel gather+project+partial-agg in one pass (the q3
     shape). **Bails to naive on `how != inner` or multi-key.**
  2. `fused_aggregate` (`Aggregate ← (Filter|Project)* ← Scan`): `peel()`
     collapses the chain to `RowOp`s; rayon over threads, each in cache-resident
     `const MORSEL = 65_536` sub-morsels, row-wise chain fused into a thread-local
     hashbrown partial-agg table, merged at end. Intermediates never hit DRAM.
     **This is the path that beats Polars.**
  3. else → naive null-aware `aggregate()` (full re-scan).
- **Joins (`join_exec`):** keys folded to i64 (single downcast / `composite_key`
  FNV-fold for multi-key); `join_indices` (inner, rayon-partitioned) /
  `join_indices_left` (null right-index) / cross (cartesian); then arrow `take`
  to gather **ALL columns**. This is the slow path.

### Honest performance (general translator path, SF-3, clean/uncontended)
**Geomean ≈ 0.15x Polars.** Two regimes, and they map exactly onto which
fast-path a query hits:
- **Near-parity / wins:** aggregate-bound shapes that hit `fused_aggregate` —
  q1 0.92x, q6 0.67x, q13 0.59x. (Hand-tuned isolated q1 hit 1.77–8x; the
  general translated plan doesn't always trigger the best fused pattern.)
- **Slow:** every multi-join chain — q15 0.01x, q11 0.03x, q2 0.04x, q4 0.04x,
  q10 0.06x, q3 0.07x, q20 0.08x, q18 0.13x. Single, specific cause:
  `fused_join_aggregate` fuses only ONE join+agg, so ≥2-join chains fall to
  `join_exec`'s take-EVERY-column full materialization between operators.

So: **correctness is complete and architectural; the perf residual is one
scoped, known engineering gap — not a wall.** The thesis ("boundary-once
Arrow-native Rust matches/beats Polars") is *proven*, but its join half rests on
a single query (q3 at 0.74–0.95x).

### Systems-literature survey (2026-07-08) — see `EXECUTION_RESEARCH_SURVEY.md`
A cited survey of advanced execution techniques confirmed the diagnosis: **our
0.15x is a data-movement problem, not a code-quality one** (Kersten VLDB'18 —
compilation's edge is smallest on memory-bound workloads, so **no JIT**). Net new
actionable items it surfaced: (a) **row-format composite keys**
(`arrow_row::RowConverter`, already in our dep tree) — a **correctness fix** for
our collision-unsafe i64-folded multi-key join, do first; (b) "late
materialization" is the precise name for the join fix (carry row-ids/selection
through the probe, gather only projected cols, last). Out-of-scope rabbit holes it
ruled out: WCOJ (wrong shape — TPC-H is acyclic PK-FK), full JIT, radix probe
(2nd-order). Predicate Transfer (CIDR'24) is the highest-leverage join-specific
lever, selectivity-gated; we already have `PREDICATE_TRANSFER_PROBE.md`.

**DataFusion decision (2026-07-08): if we take the DataFusion route, LOWER our
`LogicalPlan` INTO DataFusion** (lazy-pandas = frontend/optimizer; DataFusion =
execution backend on our exact arrow-rs substrate), NOT lift its operators into
our interpreter. Our `engine.rs` becomes the reference/probe baseline, not a
thing we grind to parity. Rationale in `EXECUTION_RESEARCH_SURVEY.md` §5. This
may replace the hand-rolled chain-fusion grind entirely — decide before the next
work session.

**RESULT (2026-07-08): 6a built and it hits Polars parity.**
`benchmarks/translate_datafusion.py` lowers our optimized `LogicalPlan` into a
DataFusion DataFrame. All 22 queries validate bit-correct vs DuckDB; execute-only
SF-3 vs Polars = **geomean 1.00x / totals 0.94x (parity), up from the hand-rolled
engine's 0.15x** — same frontend, DataFusion backend. Beats Polars on 9 queries.
q21 started at 0.14x; root cause was DataFusion's `single_distinct_to_groupby`
optimizer rule firing for SQL but NOT the DataFrame API (identical query: API
3518ms / one distinct:true aggregate, vs SQL 895ms / nested double-aggregate).
Our `_rewrite_n_unique` replicates the SQL rewrite → q21 0.52x, suite geomean
0.95→1.00. Two DataFusion findings surfaced and are handed off:
`upstream/AG10-datafusion-singledistinct-dataframe.md` (the distinct rule), and
the small dimension-join queries (q11/q15/q2 ~0.4–0.6x) = DataFusion has no
common-*subplan* elimination (recomputes reused subtrees; controlled q15 cache
experiment 37→0.8ms confirms; Polars shares them). The join-heavy gap was never
architectural. This retires the chain-fusion grind; `engine.rs` is now the
reference/probe baseline. Detail + method in `EXECUTION_RESEARCH_SURVEY.md` §5.

**Frontend fork (survey §6):** DataFusion settles the *backend*; the *frontend*
(how the plan gets built) has two levels. **6a (do first, probe):** lower our
existing lazy `LogicalPlan` into DataFusion — a small lowering pass, since the
plan already IS "a chain of deferred pandas ops"; reuses the whole frontend.
**6b (deferred, product pivot):** capture *eager* `import ... as pd` chains and
trace them into a plan transparently — high value (works on existing pandas code)
but a FireDucks-sized effort (eager-observation materialization/guards, partial-
coverage fallback boundary, index semantics, bit-for-bit compat) and already has
incumbents (FireDucks/Ibis/Modin). Not on the probe path; pursue only if the
mission flips to "ship a faster pandas."

### Recommended next lever (NOT started — gated, not an open grind)
The realistic ceiling for join-heavy shapes via this engine is **parity, not
domination** (hand-tuned q3 topped at 0.95x). So the goal of more work is to
upgrade the finding from "parity on 1 join query" to "parity across the
join-heavy suite" — stronger, upstreamable evidence — NOT to ship a product.

1. **Projection pushdown through joins (highest ROI, do first).** `join_exec`
   gathers all columns; most TPC-H join queries need 3–5 downstream. Narrow the
   `take` to surviving columns — small, isolated change, clean controlled A/B,
   large payoff on the take-everything cost.
2. **Chain fusion (only if #1 doesn't close it).** Generalize
   `fused_join_aggregate` to survive multi-join chains — stream morsels across
   the join cascade instead of materializing between joins.
3. **Go/no-go gate:** measure on the 3 worst (q15/q11/q2). If they reach
   ~0.6–0.9x → bank the strengthened suite-wide finding. **If they wall below
   parity at a specific operator, THAT wall is the finding** — record it in
   `ARROW_GAPS.md` and stop. Either outcome is a probe deliverable.

Alternatives considered and rejected: *bank now* (defensible — thesis proven —
but one cheap step buys a far more credible plank); *open-ended "make all 22
fast"* (that's product engineering, ceiling is only parity, charter says no).

### Side note — postpython (openteams-ai/postpython), assessed 2026-07-07
Evaluated for reuse: **nothing to pull in.** It's an AOT compiler for a *typed
Python subset → C99 shared libraries* (ufuncs / scipy.special reimpl), draft
spec, no license. Orthogonal to query execution: no join/groupby/aggregate/
morsel engine anywhere; its `LazyFrame`/`DataFrame` "profile" is unimplemented
spec prose. Its stable C-ABI array view is NumPy-ufunc-shaped, strictly inferior
to the Arrow C-data interface we already use. Revisit only if it ever ships a
real LazyFrame execution backend.
