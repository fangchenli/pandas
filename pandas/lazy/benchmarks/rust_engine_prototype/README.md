# Rust/arrow-rs execution engine — prototype (the direction, NOT in the build)

Proves the conclusion that the lazy-pandas join/TPC-H gap vs Polars was the
**Python/Cython host**, not anything fundamental: execute the plan in Rust on
Arrow, cross the pandas boundary **once**. See
`../../docs/RUST_ENGINE_DIRECTION.md`.

Measured (SF-3, 8 cores):
- Arrow pandas↔Rust boundary round-trip: **0.01 ms** (zero-copy C-data interface).
- **q1** (filter + low-card group-by + multi-agg): **30.7 ms vs Polars 243.9 ms
  = 7.95x**, correct vs DuckDB. (Cython engine: 0.47x.)
- **q3** (3 filters + 2 joins + high-card group-by + top-k): **80.9 ms vs Polars
  77.0 ms = 0.95x (parity)**, correct vs DuckDB. (Cython engine: 0.43x — Rust is
  2.3x ours.)

Favorable shape → 8x Polars; hard join-heavy shape → parity; always ~2–8x our
Cython engine. The architecture (Arrow-native Rust, boundary-once) matches/beats
Polars. Pushing q3 past Polars = swap the HashMap semi-joins for the proven
partitioned join kernel.

## Build + run (throwaway; needs Rust toolchain + maturin)
```
cd pandas/lazy/benchmarks/rust_engine_prototype
CONDA_PREFIX=$ENV maturin develop --release
cd .. && python rust_engine_prototype/bench_q1.py
```

Scope so far: Arrow FFI round-trip + `run_q1` (filter+group-by+multi-agg). Next:
operator coverage (joins via the proven fused kernel, high-card group-by, sort),
LogicalPlan→Rust execution, and wiring as the engine baseline.
