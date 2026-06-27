# Rust/arrow-rs execution engine — prototype (the direction, NOT in the build)

Proves the conclusion that the lazy-pandas join/TPC-H gap vs Polars was the
**Python/Cython host**, not anything fundamental: execute the plan in Rust on
Arrow, cross the pandas boundary **once**. See
`../../docs/RUST_ENGINE_DIRECTION.md`.

Measured (SF-3, 8 cores):
- Arrow pandas↔Rust boundary round-trip: **0.01 ms** (zero-copy C-data interface).
- **TPC-H q1 end-to-end in Rust: 30.7 ms vs Polars 243.9 ms = 7.95x**, correct
  vs DuckDB. (Our Cython engine: 0.47x of Polars.)

q1 is a favorable shape (4 groups → a fused single-pass filter+accumulate beats
Polars' general machinery); high-cardinality group-by and joins will be closer
to parity (match, not 8x). The point is the architecture: Arrow-native Rust
execution, boundary-once, matches/beats Polars.

## Build + run (throwaway; needs Rust toolchain + maturin)
```
cd pandas/lazy/benchmarks/rust_engine_prototype
CONDA_PREFIX=$ENV maturin develop --release
cd .. && python rust_engine_prototype/bench_q1.py
```

Scope so far: Arrow FFI round-trip + `run_q1` (filter+group-by+multi-agg). Next:
operator coverage (joins via the proven fused kernel, high-card group-by, sort),
LogicalPlan→Rust execution, and wiring as the engine baseline.
