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

q1 is a favorable shape (4 groups → a fused single-pass filter+accumulate beats
Polars' general machinery). High-cardinality group-by and joins will land closer
to parity (match, not 8x). The point stands: **Arrow-native Rust execution,
boundary-once, matches/beats Polars** — exactly "be Polars under the pandas API,"
which we have the kernels for.

## The direction
Build the lazy engine's execution in **Rust on arrow-rs**: `LogicalPlan` → Arrow
in once → execute operators in Rust (rayon, no GIL, hand-tuned hot kernels; port
Polars' join where useful) → Arrow out once. This becomes the **baseline** the
project measures from and builds on, replacing the ~0.43x Cython engine as the
performance floor.

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
