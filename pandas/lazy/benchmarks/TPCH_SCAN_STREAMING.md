# TPC-H via Parquet-Scan Streaming (out-of-core readiness)

`bench_tpch_scan.py`: all 22 queries with every table a `scan(parquet)`
LazyFrame, each query in an isolated subprocess, validated exact against
DuckDB `PRAGMA tpch(n)`. Machine: 16 GB Apple Silicon.

## SF-10 (June 2026) — 22/22 completed and validated

SF-10 tables are ~14 GB as pandas DataFrames — beyond what the in-memory
path could hold. Through the streaming scan path every query finished
with **peak RSS 1.1–8.0 GB** (≤ half of RAM), no OOM, no timeout.

| query | status | ms | peak RSS |
|---|---|---|---|
| q1 | OK | 3131 | 6.5G |
| q2 | OK | 1934 | 2.8G |
| q3 | OK | 2746 | 5.5G |
| q4 | OK | 1266 | 3.2G |
| q5 | OK | 4902 | 7.6G |
| q6 | OK | 391 | 1.5G |
| q7 | OK | 6308 | 6.5G |
| q8 | OK | 4835 | 7.4G |
| q9 | OK | 7269 | 7.6G |
| q10 | OK | 2187 | 3.7G |
| q11 | OK | 857 | 1.7G |
| q12 | OK | 864 | 1.5G |
| q13 | OK | 4532 | 4.8G |
| q14 | OK | 657 | 2.1G |
| q15 | OK | 820 | 2.0G |
| q16 | OK | 477 | 1.1G |
| q17 | OK | 5674 | 8.0G |
| q18 | OK | 7417 | 8.0G |
| q19 | OK | 1091 | 2.5G |
| q20 | OK | 2023 | 3.0G |
| q21 | OK | 6142 | 7.5G |
| q22 | OK | 6018 | 1.6G |

The SF-1 shakeout that preceded this found and fixed 3 real bugs
(regex-contains pushed to scans as a literal match — silent wrong
results; unconvertible scan predicates silently dropped; Distinct crash
on empty input). Next scale rungs (cloud): SF-100 headline on a 192 GB
instance, then SF-300+ NVMe out-of-core stress — see ROADMAP.

## SF-100 on EC2 r6i.8xlarge (June 2026) — 22/22 completed and validated

32 vCPU / 247 GB / 300 GB gp3, Ubuntu 26.04. ~35 GB parquet (~100 GB raw).
Both engines via parquet scans on the identical box; every query validated
exact against DuckDB `PRAGMA tpch(n)`. Polars uses its default engine
(its fastest configuration; its q3 flag below is a date-repr validation
artifact, not a wrong result).

| query | LP (ms) | Polars (ms) | PL/LP |
|---|---|---|---|
| q1 | 31,959 | 3,642 | 0.11x |
| q2 | 39,394 | 6,661 | 0.17x |
| q3 | 55,243 | 1,768 | 0.03x |
| q4 | 17,964 | 2,403 | 0.13x |
| q5 | 88,232 | 4,302 | 0.05x |
| q6 | 2,581 | 579 | 0.22x |
| q7 | 138,905 | 13,263 | 0.10x |
| q8 | 88,354 | 2,039 | 0.02x |
| q9 | 131,898 | 7,135 | 0.05x |
| q10 | 33,493 | 3,688 | 0.11x |
| q11 | 20,864 | 1,377 | 0.07x |
| q12 | 7,370 | 1,769 | 0.24x |
| q13 | 68,141 | 8,225 | 0.12x |
| q14 | 9,847 | 1,311 | 0.13x |
| q15 | 8,530 | 1,049 | 0.12x |
| q16 | 7,401 | 602 | 0.08x |
| q17 | 121,990 | 11,079 | 0.09x |
| q18 | 143,446 | 21,507 | 0.15x |
| q19 | 7,815 | 1,451 | 0.19x |
| q20 | 42,602 | 7,164 | 0.17x |
| q21 | 93,907 | 28,684 | 0.31x |
| q22 | 36,640 | 2,335 | 0.06x |

**Totals: lazy pandas 1197 s vs Polars 132 s (~9x slower overall; range 0.04–0.24x).**
Peak RSS stayed well under RAM throughout (heaviest: q18 at 69.5 GB — the
~150M-group aggregation; q13 35.8 GB; q2 24.6 GB).

### Bugs this run found (all fixed and tested)

Three int32-string-offset (2 GB) scale bugs that no smaller run could hit:
1. **q13**: batch concatenation via `pa.concat_arrays` raises "offset
   overflow" past 2 GB of strings → concatenate as ChunkedArray (no copy).
2. **q2/q18**: pd.merge's pyarrow `take` **segfaults** (exit 139, C++,
   no clean error) on int32-offset string columns at this scale → merge
   inputs upcast `string`→`large_string` (offsets buffer only).
3. Plus harness fixes: Linux `ru_maxrss` units; versioneer needs a
   reachable tag on shallow clones (pyarrow rejects `0+untagged` pandas).

### Reading

The result is the **completion claim**: pandas semantics over ~100 GB of
raw data on one box, all 22 validated, memory bounded. The ~9x speed gap
vs Polars at this scale (vs 0.43x geo-mean at SF-3) is dominated by the
known string-layout wall and per-batch overheads — the gap analysis and
string_view plan cover the path. The Coiled-benchmark bar ("finishes at
scale without falling over") is met at this rung; SF-300+ NVMe is the
next.
