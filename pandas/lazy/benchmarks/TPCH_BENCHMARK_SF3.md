# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1233 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.36x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 9366 ms vs convert+Polars 4624 ms (1233 + 3391); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 440.8 | 228.3 | 0.52x | ≤5 |
| q2 | OK | 157.4 | 57.6 | 0.37x | ≤12 |
| q3 | OK | 155.7 | 64.4 | 0.41x | ≤13 |
| q4 | OK | 559.1 | 81.5 | 0.15x | ≤2 |
| q5 | OK | 236.6 | 84.3 | 0.36x | ≤8 |
| q6 | OK | 67.0 | 24.8 | 0.37x | ≤29 |
| q7 | OK | 383.7 | 328.5 | 0.86x | ≤22 |
| q8 | OK | 85.0 | 29.4 | 0.35x | ≤22 |
| q9 | OK | 548.0 | 181.2 | 0.33x | ≤3 |
| q10 | OK | 327.8 | 98.9 | 0.30x | ≤5 |
| q11 | OK | 45.8 | 19.4 | 0.42x | ≤46 |
| q12 | OK | 330.9 | 225.1 | 0.68x | ≤11 |
| q13 | OK | 638.8 | 302.5 | 0.47x | ≤3 |
| q14 | OK | 80.0 | 16.7 | 0.21x | ≤19 |
| q15 | OK | 113.7 | 20.0 | 0.18x | ≤13 |
| q16 | OK | 108.6 | 23.5 | 0.22x | ≤14 |
| q17 | OK | 474.9 | 271.6 | 0.57x | ≤6 |
| q18 | OK | 697.6 | 263.6 | 0.38x | ≤2 |
| q19 | OK | 358.2 | 224.7 | 0.63x | ≤9 |
| q20 | OK | 732.2 | 127.6 | 0.17x | ≤2 |
| q21 | OK | 2758.5 | 670.2 | 0.24x | never |
| q22 | OK | 66.1 | 47.2 | 0.71x | ≤65 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
