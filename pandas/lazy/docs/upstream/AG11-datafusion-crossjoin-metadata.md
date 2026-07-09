# Hand-off: AG11 — DataFusion cross join fails on tables carrying schema metadata

**For:** a fresh agent. Read `README.md` first. **Target project: `apache/datafusion`.**
**Priority: 3** — small, isolated, reproducible on latest; a scoped follow-up to
an already-fixed bug.

## The finding (one line)
A **cross join** (`DataFrame.join_on(other, how="inner")` with no predicates, i.e.
a `NestedLoopJoin`) over tables whose Arrow schema carries **metadata** (e.g. the
`pandas` blob that `pyarrow.RecordBatch.from_pandas` attaches) fails with an
`Internal` schema-mismatch error. Stripping the metadata makes it pass. Keyed
(hash) joins on the same metadata-carrying tables work.

## Context — this is a scoped follow-up, NOT a fresh bug
Issue **#15754** ("Join on pandas dataframe from python API fails due to schema
metadata") was **fixed by PR #16221** ("fix: metadata of join schema"), closed
2025-06-02. That fix covers **keyed/hash joins**. The **cross / nested-loop join
path was not covered** and still fails on datafusion **51.0.0**. So this is
either an incomplete fix or a parallel path missed by #16221 — frame it as such
(reference #15754/#16221) rather than as a novel discovery.

## Evidence (datafusion 51.0.0)
| join | metadata present | result |
|---|---|---|
| keyed inner (`join(on=["k"])`), same tables | yes | OK |
| keyed inner, two **different** tables (different `pandas` blobs) | yes | OK |
| **cross (`join_on`, no predicates)** | **yes** | **FAIL** — `Internal("Physical input schema should be the same as the one …")` |
| cross (`join_on`, no predicates) | no (stripped) | OK |

Where we hit it: lowering TPC-H q11/q15/q22 (each has a cross join against a
1-row scalar threshold) into DataFusion via `benchmarks/translate_datafusion.py`.
Our workaround is `_df_to_arrow` → `b.replace_schema_metadata(None)` on
registration; that is why our lowering strips metadata unconditionally.

## Standalone repro (pure datafusion + pyarrow + pandas)
```python
import datafusion as d, pyarrow as pa, pandas as pd
from datafusion import SessionContext
print("datafusion", d.__version__)

def batch(df, keep_meta):
    b = pa.RecordBatch.from_pandas(df, preserve_index=False)   # attaches `pandas` metadata
    return b if keep_meta else b.replace_schema_metadata(None)

for keep_meta in (True, False):
    ctx = SessionContext()
    ctx.register_record_batches("l", [[batch(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), keep_meta)]])
    ctx.register_record_batches("r", [[batch(pd.DataFrame({"thr": [9.0]}),          keep_meta)]])
    try:
        n = ctx.table("l").join_on(ctx.table("r"), how="inner").count()   # cross join
        print(f"metadata={keep_meta}: OK ({n} rows)")
    except Exception as e:
        print(f"metadata={keep_meta}: FAIL -> {str(e)[:100]}")
```
Expect: `metadata=True: FAIL`, `metadata=False: OK` on 51.0.0.

## Hypothesis (state as hypothesis, not fact)
The cross/nested-loop physical planner (or its `join_selection` / schema-check
step) compares an input's physical schema against a logical schema that had its
metadata normalized differently — the same class of bug #16221 fixed for hash
joins, on the code path #16221 didn't touch. For maintainers to confirm.

## Gates (ALL before filing — README playbook)
- [ ] **Reproduce on the latest `datafusion` release** (env here is 51.0.0; the
      cross-join path may already be fixed on a newer release).
- [ ] **Duplicate search** on `apache/datafusion`: cross-reference **#15754** /
      **#16221** and search `cross join schema metadata`, `nested loop join
      metadata`, `Physical input schema should be the same`. Record searches. If
      #15754 is the right home, prefer **commenting there / asking to reopen with
      the cross-join repro** over a brand-new issue.
- [ ] Standalone repro above runs and shows FAIL-with-metadata / OK-without.
- [ ] **Human approval** before filing (outward-facing; guardrail).

## Definition of done
Filed (with #, or a comment on #15754) or shelved (fixed-on-latest), recorded here
+ in `README.md`. Keep the `replace_schema_metadata(None)` workaround in
`translate_datafusion.py` regardless — it is independently harmless and also
sidesteps the `join_selection` metadata-comparison entirely.
