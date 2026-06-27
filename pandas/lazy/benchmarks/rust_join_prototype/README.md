# Rust fused join+gather prototype (validation, NOT in the pandas build)

Validates that a Rust (PyO3 + rayon) fused parallel inner-join + column-major
gather **beats Polars across all payload widths** — the result Cython could not
reach (no real threads, no fused gather). See
`../../docs/JOIN_KERNEL_REBUILD_PROBE.md` for the full write-up.

Measured (10M ⋈ 2.5M unique build, 8 cores, vs Polars whole-join):

| P | rust (unordered) | pd.merge | polars |
|---|---|---|---|
| 1 | 125 | 293 | 133 |
| 3 | 179 | 319 | 192 |
| 9 | 339 | 536 | 395 |
| 21 | 727 | 927 | 762 |

Column-major (engine-native, operates on numpy/Arrow numeric buffers zero-copy,
no transpose), GIL released, unique-build inner join, float64 payload.

## Build + run (throwaway; requires the Rust toolchain + maturin)
```
cd pandas/lazy/benchmarks/rust_join_prototype
CONDA_PREFIX=$ENV maturin develop --release
python bench.py
```

This is a **prototype to prove the path**, deliberately not wired into the
pandas build (adding Rust to pandas's build system is a separate project
decision). Limitations: numeric (i64 key, f64 payload) only; single-threaded
hashbrown build; build-side gather (probe-side is the same column-major pattern).
