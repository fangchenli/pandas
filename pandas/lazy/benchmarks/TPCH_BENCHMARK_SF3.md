# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1240 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.40x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 7977 ms vs convert+Polars 4788 ms (1240 + 3548); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 422.8 | 252.1 | 0.60x | ≤7 |
| q2 | OK | 158.4 | 65.1 | 0.41x | ≤13 |
| q3 | OK | 158.9 | 66.7 | 0.42x | ≤13 |
| q4 | OK | 186.8 | 85.4 | 0.46x | ≤12 |
| q5 | OK | 240.0 | 84.8 | 0.35x | ≤7 |
| q6 | OK | 67.7 | 25.6 | 0.38x | ≤29 |
| q7 | OK | 391.1 | 336.9 | 0.86x | ≤22 |
| q8 | OK | 85.6 | 30.7 | 0.36x | ≤22 |
| q9 | OK | 566.6 | 184.4 | 0.33x | ≤3 |
| q10 | OK | 341.2 | 102.8 | 0.30x | ≤5 |
| q11 | OK | 46.0 | 19.8 | 0.43x | ≤47 |
| q12 | OK | 333.3 | 228.3 | 0.68x | ≤11 |
| q13 | OK | 649.0 | 304.3 | 0.47x | ≤3 |
| q14 | OK | 82.5 | 17.1 | 0.21x | ≤18 |
| q15 | OK | 116.8 | 20.1 | 0.17x | ≤12 |
| q16 | OK | 108.5 | 24.0 | 0.22x | ≤14 |
| q17 | OK | 488.5 | 289.7 | 0.59x | ≤6 |
| q18 | OK | 702.1 | 282.2 | 0.40x | ≤2 |
| q19 | OK | 359.7 | 228.5 | 0.64x | ≤9 |
| q20 | OK | 739.7 | 131.2 | 0.18x | ≤2 |
| q21 | OK | 1663.1 | 718.1 | 0.43x | ≤1 |
| q22 | OK | 68.9 | 50.5 | 0.73x | ≤67 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
