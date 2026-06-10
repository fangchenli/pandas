# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1242 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.42x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 8109 ms vs convert+Polars 4935 ms (1242 + 3694); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 489.9 | 247.0 | 0.50x | ≤5 |
| q2 | OK | 169.4 | 64.5 | 0.38x | ≤11 |
| q3 | OK | 163.1 | 86.6 | 0.53x | ≤16 |
| q4 | OK | 191.6 | 90.1 | 0.47x | ≤12 |
| q5 | OK | 238.1 | 87.0 | 0.37x | ≤8 |
| q6 | OK | 68.4 | 25.5 | 0.37x | ≤28 |
| q7 | OK | 401.7 | 345.2 | 0.86x | ≤21 |
| q8 | OK | 88.3 | 31.6 | 0.36x | ≤21 |
| q9 | OK | 475.9 | 192.1 | 0.40x | ≤4 |
| q10 | OK | 344.2 | 106.6 | 0.31x | ≤5 |
| q11 | OK | 47.4 | 20.1 | 0.42x | ≤45 |
| q12 | OK | 335.5 | 231.7 | 0.69x | ≤11 |
| q13 | OK | 654.1 | 316.9 | 0.48x | ≤3 |
| q14 | OK | 82.4 | 17.6 | 0.21x | ≤19 |
| q15 | OK | 122.1 | 22.1 | 0.18x | ≤12 |
| q16 | OK | 113.8 | 24.8 | 0.22x | ≤13 |
| q17 | OK | 507.5 | 310.7 | 0.61x | ≤6 |
| q18 | OK | 743.1 | 289.8 | 0.39x | ≤2 |
| q19 | OK | 366.1 | 230.3 | 0.63x | ≤9 |
| q20 | OK | 479.9 | 146.8 | 0.31x | ≤3 |
| q21 | OK | 1959.5 | 753.8 | 0.38x | ≤1 |
| q22 | OK | 66.7 | 52.9 | 0.79x | ≤89 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
