# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1220 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.43x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 7492 ms vs convert+Polars 4817 ms (1220 + 3597); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 466.8 | 259.6 | 0.56x | ≤5 |
| q2 | OK | 162.6 | 66.0 | 0.41x | ≤12 |
| q3 | OK | 167.9 | 71.1 | 0.42x | ≤12 |
| q4 | OK | 188.9 | 114.0 | 0.60x | ≤16 |
| q5 | OK | 236.3 | 96.8 | 0.41x | ≤8 |
| q6 | OK | 68.1 | 25.8 | 0.38x | ≤28 |
| q7 | OK | 388.1 | 350.7 | 0.90x | ≤32 |
| q8 | OK | 85.7 | 44.2 | 0.52x | ≤29 |
| q9 | OK | 471.3 | 184.1 | 0.39x | ≤4 |
| q10 | OK | 338.7 | 107.3 | 0.32x | ≤5 |
| q11 | OK | 49.7 | 21.0 | 0.42x | ≤42 |
| q12 | OK | 332.6 | 226.3 | 0.68x | ≤11 |
| q13 | OK | 656.3 | 305.5 | 0.47x | ≤3 |
| q14 | OK | 88.1 | 21.1 | 0.24x | ≤18 |
| q15 | OK | 116.2 | 20.8 | 0.18x | ≤12 |
| q16 | OK | 116.5 | 27.0 | 0.23x | ≤13 |
| q17 | OK | 480.5 | 278.1 | 0.58x | ≤6 |
| q18 | OK | 699.4 | 271.2 | 0.39x | ≤2 |
| q19 | OK | 354.6 | 226.7 | 0.64x | ≤9 |
| q20 | OK | 442.8 | 128.0 | 0.29x | ≤3 |
| q21 | OK | 1514.4 | 702.1 | 0.46x | ≤1 |
| q22 | OK | 66.2 | 49.3 | 0.75x | ≤72 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
