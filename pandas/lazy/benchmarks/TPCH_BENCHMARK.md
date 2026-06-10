# TPC-H (PDS-H): lazy pandas vs Polars

SF-1.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **363 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.32x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 3029 ms vs convert+Polars 1426 ms (363 + 1062); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 152.5 | 86.6 | 0.57x | ≤5 |
| q2 | OK | 59.5 | 20.7 | 0.35x | ≤9 |
| q3 | OK | 64.2 | 19.2 | 0.30x | ≤8 |
| q4 | OK | 99.9 | 26.8 | 0.27x | ≤4 |
| q5 | OK | 74.7 | 31.7 | 0.42x | ≤8 |
| q6 | OK | 21.7 | 8.7 | 0.40x | ≤27 |
| q7 | OK | 120.8 | 91.2 | 0.75x | ≤12 |
| q8 | OK | 51.2 | 10.1 | 0.20x | ≤8 |
| q9 | OK | 149.1 | 52.7 | 0.35x | ≤3 |
| q10 | OK | 114.2 | 30.1 | 0.26x | ≤4 |
| q11 | OK | 18.6 | 6.8 | 0.37x | ≤30 |
| q12 | OK | 117.2 | 76.1 | 0.65x | ≤8 |
| q13 | OK | 210.7 | 101.7 | 0.48x | ≤3 |
| q14 | OK | 28.3 | 6.6 | 0.23x | ≤16 |
| q15 | OK | 59.5 | 7.5 | 0.13x | ≤6 |
| q16 | OK | 49.5 | 8.9 | 0.18x | ≤8 |
| q17 | OK | 150.0 | 71.0 | 0.47x | ≤4 |
| q18 | OK | 174.8 | 74.4 | 0.43x | ≤3 |
| q19 | OK | 139.1 | 76.9 | 0.55x | ≤5 |
| q20 | OK | 205.5 | 35.1 | 0.17x | ≤2 |
| q21 | OK | 746.3 | 185.5 | 0.25x | never |
| q22 | OK | 221.5 | 34.0 | 0.15x | ≤1 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
