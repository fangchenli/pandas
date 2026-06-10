# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1243 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.41x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 7531 ms vs convert+Polars 4793 ms (1243 + 3550); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 443.5 | 263.5 | 0.59x | ≤6 |
| q2 | OK | 162.5 | 64.8 | 0.40x | ≤12 |
| q3 | OK | 159.0 | 62.8 | 0.40x | ≤12 |
| q4 | OK | 185.8 | 84.6 | 0.46x | ≤12 |
| q5 | OK | 225.8 | 85.4 | 0.38x | ≤8 |
| q6 | OK | 68.7 | 25.9 | 0.38x | ≤29 |
| q7 | OK | 403.0 | 339.5 | 0.84x | ≤19 |
| q8 | OK | 85.8 | 31.5 | 0.37x | ≤22 |
| q9 | OK | 465.9 | 194.2 | 0.42x | ≤4 |
| q10 | OK | 339.0 | 101.1 | 0.30x | ≤5 |
| q11 | OK | 45.6 | 19.6 | 0.43x | ≤47 |
| q12 | OK | 334.2 | 228.8 | 0.68x | ≤11 |
| q13 | OK | 649.1 | 306.8 | 0.47x | ≤3 |
| q14 | OK | 81.3 | 16.7 | 0.21x | ≤19 |
| q15 | OK | 112.9 | 20.2 | 0.18x | ≤13 |
| q16 | OK | 113.8 | 24.2 | 0.21x | ≤13 |
| q17 | OK | 487.6 | 283.0 | 0.58x | ≤6 |
| q18 | OK | 725.2 | 284.1 | 0.39x | ≤2 |
| q19 | OK | 356.0 | 225.9 | 0.63x | ≤9 |
| q20 | OK | 454.2 | 128.1 | 0.28x | ≤3 |
| q21 | OK | 1563.3 | 708.4 | 0.45x | ≤1 |
| q22 | OK | 68.7 | 51.1 | 0.74x | ≤70 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
