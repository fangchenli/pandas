# TPC-H (PDS-H): lazy pandas vs Polars

SF-1.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **386 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.26x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 4585 ms vs convert+Polars 1518 ms (386 + 1133); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 162.3 | 89.4 | 0.55x | ≤5 |
| q2 | OK | 115.0 | 20.8 | 0.18x | ≤4 |
| q3 | OK | 80.0 | 30.4 | 0.38x | ≤7 |
| q4 | OK | 104.8 | 29.9 | 0.29x | ≤5 |
| q5 | OK | 157.8 | 32.3 | 0.20x | ≤3 |
| q6 | OK | 21.9 | 8.6 | 0.39x | ≤29 |
| q7 | OK | 323.0 | 93.2 | 0.29x | ≤1 |
| q8 | OK | 58.3 | 10.1 | 0.17x | ≤8 |
| q9 | OK | 149.7 | 53.3 | 0.36x | ≤4 |
| q10 | OK | 127.6 | 30.0 | 0.24x | ≤3 |
| q11 | OK | 29.1 | 6.9 | 0.24x | ≤17 |
| q12 | OK | 117.3 | 75.2 | 0.64x | ≤9 |
| q13 | OK | 205.3 | 85.0 | 0.41x | ≤3 |
| q14 | OK | 27.4 | 6.6 | 0.24x | ≤18 |
| q15 | OK | 55.2 | 7.2 | 0.13x | ≤8 |
| q16 | OK | 54.5 | 9.0 | 0.16x | ≤8 |
| q17 | OK | 144.5 | 70.6 | 0.49x | ≤5 |
| q18 | OK | 209.2 | 89.7 | 0.43x | ≤3 |
| q19 | OK | 557.2 | 75.7 | 0.14x | never |
| q20 | OK | 212.4 | 84.5 | 0.40x | ≤3 |
| q21 | OK | 1448.0 | 209.6 | 0.14x | never |
| q22 | OK | 224.0 | 14.5 | 0.06x | ≤1 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
