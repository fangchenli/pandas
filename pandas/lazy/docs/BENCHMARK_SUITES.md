# Standard Benchmark Suites (Polars / DuckDB) — and what lazy-pandas adopts

Research into the benchmark suites Polars and DuckDB use to evaluate and market
their engines, so lazy-pandas can compare on *recognized, comparable* ground
rather than only our custom `bench_vs_polars.py` microbenchmarks.

## Summary

| Benchmark | Used by | Tests | #Queries | Laptop scale | DataFrame fit |
|---|---|---|---|---|---|
| **H2O.ai db-benchmark** | Both (DuckDB Labs maintains; Polars is an entrant) | group-by + join | 10 + 5 | 1e7 (~0.5 GB), 1e8 (~5 GB) | **Native** |
| **TPC-H / PDS-H** | Both (Polars markets it) | full analytical SQL | 22 | SF-1→SF-10 | SQL→DataFrame translation |
| **ClickBench** | Both (Polars added Jan 2026) | wide-table OLAP scan | 43 | ~14 GB | SQL-only — poor fit |
| **TPC-DS** | DuckDB (mostly internal) | complex SQL | 99 | any SF | Poor fit |

## H2O.ai db-benchmark ("Database-like ops benchmark") — our primary target

The de-facto cross-engine DataFrame benchmark. Originally h2oai (dormant since
2021), revived and maintained by **DuckDB Labs**:
<https://github.com/duckdblabs/db-benchmark>. Polars, pandas, Dask, DuckDB,
DataFusion, and Arrow are all entrants in the *same harness*, so results are
directly comparable. Tests exactly the two primitives a DataFrame engine lives
or dies on — **group-by and join** — with no SQL parser needed.

- **Group-by:** 10 queries (5 basic + 5 advanced). Vary low/high cardinality,
  integer vs string keys, small/large value columns.
- **Join:** 5 queries — small/medium/large right-hand tables (RHS sized
  LHS/1e6, LHS/1e3, LHS), integer and varchar keys, including a left join.
- **Scales:** 1e7 ≈ 0.5 GB, 1e8 ≈ 5 GB, 1e9 ≈ 50 GB (out-of-core tier).
- **Methodology:** each query run twice; both cold and hot reported.
- DuckDB Labs reference hardware: AWS c6id.metal, 128 cores, local NVMe.
- Data-gen: `_data/groupby-datagen.R <nrow> <k> <na> <sort>`.
- Blog: <https://duckdb.org/2023/04/14/h2oai>,
  <https://duckdb.org/2023/11/03/db-benchmark-update>.

### Group-by dataset schema (G1_<nrow>_<k>_<na>_<sort>)

`id1,id2,id3` string keys ("id001"…); `id4,id5,id6` integer keys; `v1,v2`
small integer values; `v3` float value. Key cardinality is controlled by `k`.

### The 10 group-by queries (Polars reference)

1. `sum(v1)` by `id1`
2. `sum(v1)` by `id1, id2`
3. `sum(v1), mean(v3)` by `id3`
4. `mean(v1), mean(v2), mean(v3)` by `id4`
5. `sum(v1), sum(v2), sum(v3)` by `id6`
6. `median(v3), std(v3)` by `id4, id5` *(advanced)*
7. `max(v1) - min(v2)` as range by `id3` *(advanced — agg arithmetic)*
8. largest-two `v3` by `id6` *(advanced — grouped top-k + explode)*
9. `corr(v1, v2)^2` by `id2, id4` *(advanced — corr aggregation)*
10. `sum(v3), count` by `id1..id6` *(high-cardinality group-by)*

### The 5 join queries (Polars reference)

LHS table `x` (≈ nrow rows) joined to three right tables:
1. inner join `small` (nrow/1e6 rows) on `id1`
2. inner join `medium` (nrow/1e3 rows) on `id2`
3. left join `medium` on `id2`
4. inner join `medium` on `id5`
5. inner join `big` (nrow rows) on `id3`

## TPC-H / PDS-H — strong second target

22 decision-support SQL queries; Polars brands its adaptation **PDS-H** and runs
it against pandas/Dask/Spark/DuckDB on every benchmarks post
(<https://pola.rs/posts/benchmarks/>). Polars' repo
<https://github.com/pola-rs/polars-benchmark> has hand-written
Polars-expression translations of all 22 — a ready reference for porting to a
lazy DataFrame API. DuckDB ships a first-party `tpch` extension
(`CALL dbgen(sf=…)`, `PRAGMA tpch(1..22)`). SF-1→SF-10 are laptop-tractable;
SF-100 is data-center scale. **Gotcha:** correlated subqueries (Q2/Q17/Q20/Q21)
are the translation risk; validate outputs against DuckDB's `PRAGMA tpch(n)`.

## Deprioritized

- **ClickBench** — SQL-only against a single wide table; the 43 queries assume a
  SQL engine and reward columnar-storage/indexing tricks over DataFrame-op
  efficiency. Low fit. <https://github.com/ClickHouse/ClickBench>.
- **TPC-DS** — 99 complex SQL queries, heavy SQL→DataFrame translation, not a
  benchmark Polars markets. Out of scope.

## Plan for lazy-pandas

1. **H2O db-benchmark first** — group-by + join at 1e7 (and 1e8 if RAM allows),
   replicating the cold/hot convention. Directly comparable to the published
   board; exercises exactly what our recent work (cardinality, order-relaxed
   joins, dictionary-key group-by) targets. See `benchmarks/bench_h2o.py`.
2. **PDS-H / TPC-H second** — SF-1→SF-10, porting Polars' 22 reference queries;
   budget for SQL→lazy translation and output validation against DuckDB.
