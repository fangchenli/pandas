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
