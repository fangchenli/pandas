# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1401 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.41x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 7917 ms vs convert+Polars 5093 ms (1401 + 3692); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 525.3 | 237.9 | 0.45x | ≤4 |
| q2 | OK | 163.7 | 63.6 | 0.39x | ≤13 |
| q3 | OK | 159.2 | 67.2 | 0.42x | ≤15 |
| q4 | OK | 187.7 | 111.7 | 0.60x | ≤18 |
| q5 | OK | 223.8 | 83.7 | 0.37x | ≤10 |
| q6 | OK | 68.0 | 25.2 | 0.37x | ≤32 |
| q7 | OK | 383.2 | 335.3 | 0.87x | ≤29 |
| q8 | OK | 89.2 | 29.1 | 0.33x | ≤23 |
| q9 | OK | 570.2 | 195.2 | 0.34x | ≤3 |
| q10 | OK | 338.4 | 104.8 | 0.31x | ≤5 |
| q11 | OK | 45.3 | 19.6 | 0.43x | ≤54 |
| q12 | OK | 330.7 | 228.5 | 0.69x | ≤13 |
| q13 | OK | 649.6 | 300.5 | 0.46x | ≤4 |
| q14 | OK | 80.7 | 19.3 | 0.24x | ≤22 |
| q15 | OK | 115.4 | 20.9 | 0.18x | ≤14 |
| q16 | OK | 116.9 | 24.1 | 0.21x | ≤15 |
| q17 | OK | 484.7 | 275.8 | 0.57x | ≤6 |
| q18 | OK | 718.7 | 350.4 | 0.49x | ≤3 |
| q19 | OK | 382.8 | 240.6 | 0.63x | ≤9 |
| q20 | OK | 487.1 | 141.1 | 0.29x | ≤4 |
| q21 | OK | 1729.2 | 765.5 | 0.44x | ≤1 |
| q22 | OK | 67.2 | 51.8 | 0.77x | ≤90 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
