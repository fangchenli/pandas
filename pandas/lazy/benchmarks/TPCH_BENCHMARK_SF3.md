# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1739 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.42x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 9433 ms vs convert+Polars 5453 ms (1739 + 3714); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 559.2 | 238.7 | 0.43x | ≤5 |
| q2 | OK | 150.1 | 63.1 | 0.42x | ≤20 |
| q3 | OK | 179.4 | 68.9 | 0.38x | ≤15 |
| q4 | OK | 185.4 | 86.1 | 0.46x | ≤17 |
| q5 | OK | 226.3 | 87.5 | 0.39x | ≤12 |
| q6 | OK | 21.0 | 24.9 | 1.18x | always |
| q7 | OK | 392.8 | 342.4 | 0.87x | ≤34 |
| q8 | OK | 86.1 | 29.5 | 0.34x | ≤30 |
| q9 | OK | 477.7 | 184.5 | 0.39x | ≤5 |
| q10 | OK | 343.6 | 103.0 | 0.30x | ≤7 |
| q11 | OK | 47.3 | 20.0 | 0.42x | ≤63 |
| q12 | OK | 337.4 | 232.0 | 0.69x | ≤16 |
| q13 | OK | 657.4 | 314.0 | 0.48x | ≤5 |
| q14 | OK | 82.0 | 17.5 | 0.21x | ≤26 |
| q15 | OK | 69.0 | 24.0 | 0.35x | ≤38 |
| q16 | OK | 117.1 | 24.5 | 0.21x | ≤18 |
| q17 | OK | 511.4 | 301.6 | 0.59x | ≤8 |
| q18 | OK | 768.3 | 326.8 | 0.43x | ≤3 |
| q19 | OK | 377.5 | 262.9 | 0.70x | ≤15 |
| q20 | OK | 979.0 | 154.9 | 0.16x | ≤2 |
| q21 | OK | 2786.1 | 756.5 | 0.27x | never |
| q22 | OK | 79.4 | 50.5 | 0.64x | ≤60 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
