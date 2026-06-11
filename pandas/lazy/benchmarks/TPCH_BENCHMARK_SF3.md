# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1829 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.45x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 8295 ms vs convert+Polars 5630 ms (1829 + 3801); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 472.8 | 288.9 | 0.61x | ≤9 |
| q2 | OK | 151.0 | 64.8 | 0.43x | ≤21 |
| q3 | OK | 164.6 | 69.5 | 0.42x | ≤19 |
| q4 | OK | 187.1 | 90.5 | 0.48x | ≤18 |
| q5 | OK | 229.9 | 89.2 | 0.39x | ≤12 |
| q6 | OK | 21.6 | 25.4 | 1.18x | always |
| q7 | OK | 408.7 | 365.9 | 0.90x | ≤42 |
| q8 | OK | 93.7 | 31.1 | 0.33x | ≤29 |
| q9 | OK | 483.2 | 190.4 | 0.39x | ≤6 |
| q10 | OK | 344.5 | 102.8 | 0.30x | ≤7 |
| q11 | OK | 43.1 | 21.5 | 0.50x | ≤84 |
| q12 | OK | 343.3 | 229.4 | 0.67x | ≤16 |
| q13 | OK | 692.7 | 331.8 | 0.48x | ≤5 |
| q14 | OK | 85.1 | 18.2 | 0.21x | ≤27 |
| q15 | OK | 64.6 | 22.1 | 0.34x | ≤43 |
| q16 | OK | 116.5 | 31.5 | 0.27x | ≤21 |
| q17 | OK | 561.4 | 311.3 | 0.55x | ≤7 |
| q18 | OK | 819.2 | 332.2 | 0.41x | ≤3 |
| q19 | OK | 367.6 | 230.9 | 0.63x | ≤13 |
| q20 | OK | 724.9 | 161.9 | 0.22x | ≤3 |
| q21 | OK | 1840.9 | 739.8 | 0.40x | ≤1 |
| q22 | OK | 78.4 | 51.8 | 0.66x | ≤68 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
