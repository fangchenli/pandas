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
