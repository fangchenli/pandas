# Hand-off: AG5 — Acero `count_distinct` slower than pandas' Cython

**For:** a fresh agent. Read `README.md` first. **Priority 5.** Needs an
isolated benchmark before it's file-able (claim is real but not yet
standalone-verified like AG4).

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
