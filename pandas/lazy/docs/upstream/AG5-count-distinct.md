# Hand-off: AG5 — Acero `count_distinct` slower than pandas' Cython

**For:** a fresh agent. Read `README.md` first. **Priority 5.** Needs an
isolated benchmark before it's file-able (claim is real but not yet
standalone-verified like AG4).

> **RESOLVED (2026-07-09) — the vs-pandas premise is SHELVED; a real vs-Polars
> gap took its place.** Built + ran the isolated benchmark (`/tmp/ag5_countdistinct.py`,
> also see the standalone shape below) on **latest pyarrow 24.0.0** / pandas 3.0.3 /
> polars 1.42.1, 10M rows, threads controlled:
> - **vs pandas: NO GAP — SHELVE.** Acero grouped `count_distinct` is *faster*
>   than `pandas.groupby().nunique()` at 1 core (acero/pandas = 0.6–0.8×) and
>   ~parity multi-core. The original "Acero slower than pandas Cython" claim does
>   NOT hold on latest — it was the eager-vs-physical measurement artifact the
>   positioning memory already flagged. Do not file the vs-pandas framing.
> - **vs Polars: REAL, reproducible gap (new AG5').** Acero grouped
>   `count_distinct` is **~3.8–5.3× slower than Polars at ≥10k groups** (and 3.9×
>   at 1M groups). Mechanism (measured): Acero's `count_distinct` is
>   **thread-insensitive** — ~flat 500–600 ms across 1/2/4/8 cores at 10k groups —
>   while Polars parallelizes. (Earlier 11.9× figure was a noisy run; corrected.)
> - **Dup-search (apache/arrow):** no issue reports this. Closest open item is
>   **#38372 "[Discuss][C++] Replace MemoTable with a SwissTable implementation"**
>   — the grouping/distinct hash-table perf discussion, a natural home to attach
>   our data point rather than file anew. (#29633 closed = the original kernel
>   impl.) Not a duplicate.
> - **Note the cross-ecosystem pattern:** this is the Acero-C++ sibling of AG10
>   (DataFusion count-distinct) — count-distinct is a weak spot across the Arrow
>   stack; Polars and DataFusion-SQL (distinct-then-count rewrite) are both fast.
>
> **CONSOLIDATED with AG3 (2026-07-09):** the AG3 probe found the same root —
> Arrow grouped `sum` also saturates parallel scaling at high cardinality (1M int
> ≈1.2×, string ≈1.3×) while low-card scales 4.3×. AG3's broad "no parallelism"
> claim was refuted; its residual is this same hash-table (MemoTable) high-card
> ceiling. Treat AG5' + AG3-residual as ONE finding for #38372: "Arrow grouped
> hash-aggregate parallel scaling saturates at high cardinality vs Polars."
>
> **Next step (needs go-ahead):** either comment our data point on #38372, or
> file a scoped `[C++][Acero] grouped count_distinct does not parallelize / ~4–5×
> behind Polars` enhancement. NOT the old vs-pandas issue. Re-verify on latest at
> file time.

## Goal
Verify, then (if it holds and is non-duplicate) report that Acero's grouped
`count_distinct` / `hash_count_distinct` is slower than pandas' Cython
count-distinct on high-cardinality data — or, if it doesn't hold standalone,
record that and shelve.

## Context
The lazy-pandas engine routes `n_unique`/`count_distinct` to a pandas-Cython
path rather than Acero because Acero's `count_distinct` measured slower (see
`../QGAP_DECOMP.md`). This was observed in-engine, **not** isolated in a
standalone benchmark — so the first job is to isolate it.

## What to do
1. **Build a standalone benchmark** (model it on
   `../../benchmarks/bench_arrow_string_groupby.py`): N rows, a group key + a
   high-cardinality value column; compare grouped distinct-count via
   - Acero `Table.group_by(key).aggregate([(val, "count_distinct")])`,
   - pandas `df.groupby(key)[val].nunique()`,
   - polars `group_by(key).agg(pl.col(val).n_unique())`,
   across cardinalities and `set_cpu_count`; assert correctness. Run on latest
   pyarrow (playbook §1).
2. If Acero is materially slower (and the gap isn't just the AG3 single-`Table`
   parallelism confound — control threads): run the **duplicate search** and
   draft an enhancement issue. If not reproducible standalone, **shelve** and
   note it.

## Gates
- [ ] Standalone benchmark built; gap confirmed on latest pyarrow with threads
      controlled (separate the kernel-quality question from AG3 parallelism).
- [ ] Duplicate search recorded.
- [ ] Human approval.

## Definition of done
Result (verified+filed, or shelved) recorded in `../ARROW_GAPS.md` AG5 +
`README.md`. Keep the pandas-Cython routing regardless.
