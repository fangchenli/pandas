# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1712 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.39x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 9211 ms vs convert+Polars 5257 ms (1712 + 3545); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 566.7 | 262.9 | 0.46x | ≤5 |
| q2 | OK | 184.1 | 75.5 | 0.41x | ≤15 |
| q3 | OK | 171.7 | 86.1 | 0.50x | ≤19 |
| q4 | OK | 183.8 | 93.7 | 0.51x | ≤19 |
| q5 | OK | 240.7 | 86.3 | 0.36x | ≤11 |
| q6 | OK | 67.5 | 24.9 | 0.37x | ≤40 |
| q7 | OK | 379.9 | 335.7 | 0.88x | ≤38 |
| q8 | OK | 85.9 | 30.0 | 0.35x | ≤30 |
| q9 | OK | 637.0 | 193.5 | 0.30x | ≤3 |
| q10 | OK | 326.3 | 101.0 | 0.31x | ≤7 |
| q11 | OK | 45.0 | 19.9 | 0.44x | ≤68 |
| q12 | OK | 328.8 | 230.2 | 0.70x | ≤17 |
| q13 | OK | 640.3 | 298.7 | 0.47x | ≤5 |
| q14 | OK | 80.2 | 16.6 | 0.21x | ≤26 |
| q15 | OK | 113.6 | 20.0 | 0.18x | ≤18 |
| q16 | OK | 106.4 | 24.3 | 0.23x | ≤20 |
| q17 | OK | 477.8 | 271.2 | 0.57x | ≤8 |
| q18 | OK | 691.4 | 273.4 | 0.40x | ≤4 |
| q19 | OK | 354.4 | 230.4 | 0.65x | ≤13 |
| q20 | OK | 731.8 | 131.0 | 0.18x | ≤2 |
| q21 | OK | 2730.9 | 687.5 | 0.25x | never |
| q22 | OK | 66.7 | 52.1 | 0.78x | ≤116 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
