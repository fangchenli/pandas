# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1256 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.41x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 7618 ms vs convert+Polars 4815 ms (1256 + 3560); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 432.1 | 247.3 | 0.57x | ≤6 |
| q2 | OK | 162.0 | 65.3 | 0.40x | ≤12 |
| q3 | OK | 162.5 | 64.4 | 0.40x | ≤12 |
| q4 | OK | 187.7 | 87.4 | 0.47x | ≤12 |
| q5 | OK | 232.9 | 87.2 | 0.37x | ≤8 |
| q6 | OK | 69.8 | 25.1 | 0.36x | ≤28 |
| q7 | OK | 390.6 | 341.0 | 0.87x | ≤25 |
| q8 | OK | 86.2 | 29.8 | 0.35x | ≤22 |
| q9 | OK | 470.8 | 184.6 | 0.39x | ≤4 |
| q10 | OK | 343.9 | 102.4 | 0.30x | ≤5 |
| q11 | OK | 45.9 | 19.5 | 0.43x | ≤47 |
| q12 | OK | 335.8 | 228.1 | 0.68x | ≤11 |
| q13 | OK | 646.8 | 313.8 | 0.49x | ≤3 |
| q14 | OK | 80.3 | 16.6 | 0.21x | ≤19 |
| q15 | OK | 114.2 | 20.5 | 0.18x | ≤13 |
| q16 | OK | 116.5 | 25.8 | 0.22x | ≤13 |
| q17 | OK | 489.4 | 282.4 | 0.58x | ≤6 |
| q18 | OK | 710.7 | 286.9 | 0.40x | ≤2 |
| q19 | OK | 360.7 | 228.7 | 0.63x | ≤9 |
| q20 | OK | 452.3 | 134.9 | 0.30x | ≤3 |
| q21 | OK | 1657.9 | 714.9 | 0.43x | ≤1 |
| q22 | OK | 68.4 | 52.8 | 0.77x | ≤80 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
