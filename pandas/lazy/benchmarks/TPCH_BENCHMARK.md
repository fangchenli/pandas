# TPC-H (PDS-H): lazy pandas vs Polars

SF-1.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **400 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.27x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 4213 ms vs convert+Polars 1427 ms (400 + 1028); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 155.3 | 83.9 | 0.54x | ≤5 |
| q2 | OK | 58.6 | 19.9 | 0.34x | ≤10 |
| q3 | OK | 60.8 | 18.4 | 0.30x | ≤9 |
| q4 | OK | 101.6 | 25.6 | 0.25x | ≤5 |
| q5 | OK | 70.1 | 31.3 | 0.45x | ≤10 |
| q6 | OK | 22.3 | 8.6 | 0.39x | ≤29 |
| q7 | OK | 227.7 | 93.2 | 0.41x | ≤2 |
| q8 | OK | 50.7 | 10.7 | 0.21x | ≤9 |
| q9 | OK | 154.2 | 53.4 | 0.35x | ≤3 |
| q10 | OK | 110.8 | 30.7 | 0.28x | ≤4 |
| q11 | OK | 17.9 | 7.1 | 0.40x | ≤37 |
| q12 | OK | 118.1 | 77.0 | 0.65x | ≤9 |
| q13 | OK | 206.7 | 85.8 | 0.42x | ≤3 |
| q14 | OK | 26.2 | 6.1 | 0.23x | ≤19 |
| q15 | OK | 57.0 | 8.1 | 0.14x | ≤8 |
| q16 | OK | 54.7 | 9.0 | 0.16x | ≤8 |
| q17 | OK | 158.3 | 70.9 | 0.45x | ≤4 |
| q18 | OK | 175.7 | 73.7 | 0.42x | ≤3 |
| q19 | OK | 560.8 | 75.7 | 0.13x | never |
| q20 | OK | 202.2 | 36.0 | 0.18x | ≤2 |
| q21 | OK | 1405.8 | 189.1 | 0.13x | never |
| q22 | OK | 217.5 | 13.2 | 0.06x | ≤1 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
