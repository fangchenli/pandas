# TPC-H (PDS-H): lazy pandas vs Polars

SF-3.0, every query validated exact against DuckDB `PRAGMA tpch(n)`. pandas 3.1.0.dev0+1220.g55b305e82a, polars 1.37.1, macOS-26.5.1-arm64-arm-64bit.

Two scenarios, two questions:

- **S1 — native vs native** (engine quality): each engine on its own
  format, query only. `from_pandas` is **never** timed here.
- **S2 — pandas-resident** (the data already lives in pandas): converting
  all 8 tables to Polars costs **1674 ms once**; the break-even
  column is the number of runs of that query below which staying in
  lazy pandas is faster end-to-end. Conversion is a one-time cost —
  it is never charged per query.

**S1 geometric mean: 0.34x** (PL/LP; >1 means lazy pandas faster).
**S2 whole-suite**: one pass of all queries — lazy pandas 10015 ms vs convert+Polars 5267 ms (1674 + 3593); converting to Polars wins from the very first pass.

| query | valid | LP (ms) | PL (ms) | S1 PL/LP | S2 break-even (runs) |
|---|---|---|---|---|---|
| q1 | OK | 506.8 | 233.8 | 0.46x | ≤6 |
| q2 | OK | 157.3 | 63.0 | 0.40x | ≤17 |
| q3 | OK | 156.6 | 80.8 | 0.52x | ≤22 |
| q4 | OK | 280.6 | 100.9 | 0.36x | ≤9 |
| q5 | OK | 239.6 | 81.8 | 0.34x | ≤10 |
| q6 | OK | 67.2 | 24.8 | 0.37x | ≤39 |
| q7 | OK | 409.5 | 390.8 | 0.95x | ≤89 |
| q8 | OK | 97.6 | 30.8 | 0.32x | ≤25 |
| q9 | OK | 555.9 | 218.6 | 0.39x | ≤4 |
| q10 | OK | 347.9 | 106.5 | 0.31x | ≤6 |
| q11 | OK | 45.9 | 20.2 | 0.44x | ≤65 |
| q12 | OK | 330.9 | 230.6 | 0.70x | ≤16 |
| q13 | OK | 650.1 | 318.0 | 0.49x | ≤5 |
| q14 | OK | 80.4 | 17.1 | 0.21x | ≤26 |
| q15 | OK | 116.8 | 21.5 | 0.18x | ≤17 |
| q16 | OK | 108.5 | 24.3 | 0.22x | ≤19 |
| q17 | OK | 478.4 | 270.9 | 0.57x | ≤8 |
| q18 | OK | 700.3 | 269.5 | 0.38x | ≤3 |
| q19 | OK | 359.4 | 227.2 | 0.63x | ≤12 |
| q20 | OK | 724.2 | 127.8 | 0.18x | ≤2 |
| q21 | OK | 2594.8 | 687.3 | 0.26x | never |
| q22 | OK | 1006.2 | 46.6 | 0.05x | ≤1 |

Regenerate: `python pandas/lazy/benchmarks/bench_tpch.py --sf 1 --report pandas/lazy/benchmarks/TPCH_BENCHMARK.md`
