# TPC-H (PDS-H): lazy pandas vs Polars

SF-1.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **361 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.23x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 4634 ms vs convert+Polars 1377 ms (361 + 1016); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 149.9 | 82.8 | 0.55x | ≤5 |
| q2 | OK | 114.2 | 20.2 | 0.18x | ≤3 |
| q3 | OK | 131.4 | 18.5 | 0.14x | ≤3 |
| q4 | OK | 96.1 | 25.8 | 0.27x | ≤5 |
| q5 | OK | 216.2 | 29.8 | 0.14x | ≤1 |
| q6 | OK | 21.1 | 9.0 | 0.42x | ≤29 |
| q7 | OK | 314.8 | 92.1 | 0.29x | ≤1 |
| q8 | OK | 61.4 | 10.0 | 0.16x | ≤7 |
| q9 | OK | 140.4 | 53.3 | 0.38x | ≤4 |
| q10 | OK | 145.0 | 30.1 | 0.21x | ≤3 |
| q11 | OK | 28.1 | 6.6 | 0.23x | ≤16 |
| q12 | OK | 112.8 | 74.4 | 0.66x | ≤9 |
| q13 | OK | 202.7 | 85.1 | 0.42x | ≤3 |
| q14 | OK | 25.6 | 6.1 | 0.24x | ≤18 |
| q15 | OK | 56.0 | 7.3 | 0.13x | ≤7 |
| q16 | OK | 53.3 | 8.9 | 0.17x | ≤8 |
| q17 | OK | 144.6 | 70.3 | 0.49x | ≤4 |
| q18 | OK | 318.8 | 71.8 | 0.23x | ≤1 |
| q19 | OK | 546.7 | 75.2 | 0.14x | never |
| q20 | OK | 206.3 | 35.4 | 0.17x | ≤2 |
| q21 | OK | 1335.8 | 188.6 | 0.14x | never |
| q22 | OK | 212.9 | 14.3 | 0.07x | ≤1 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
