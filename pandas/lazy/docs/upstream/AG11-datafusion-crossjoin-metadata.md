# Hand-off: AG11 — DataFusion `CrossJoinExec` mismatches schema metadata

**For:** a fresh agent. Read `README.md` first. **Target project: `apache/datafusion`.**
**Priority: 2** — root cause pinned to one line; repro verified on latest; not a
duplicate. Ready to file pending human go-ahead.

> **STATUS (2026-07-09): VERIFIED ON LATEST + ROOT-CAUSED + DUP-SEARCHED — file-ready.**
> Confirmed two independent ways: (1) **executed** the repro on datafusion
> **54.0.0** (the latest PyPI release, 2026-06-29) — pure cross join over
> metadata-carrying tables fails; (2) **source-read** the offending code, byte-
> identical at tags 51.0.0, 54.0.0, and `main`. **Not fixed.** Not a duplicate
> (see §Dup-search). Remaining gate: human approval to file.

## The finding (one line)
A **pure cross join** (`DataFrame.join_on(other, how="inner")` with no predicates
→ `CrossJoinExec`) over tables whose Arrow schema carries **metadata** (e.g. the
`pandas` blob from `pyarrow.RecordBatch.from_pandas`) fails with an `Internal`
error; `replace_schema_metadata(None)` works around it. Hash/keyed joins — and
even a *filtered* nested-loop join — on the same tables succeed.

## Evidence (executed on datafusion 54.0.0, latest)
| join shape | metadata present | result |
|---|---|---|
| **pure cross (`join_on`, no predicate)** | **yes** | **FAIL** — `Internal error: Physical input schema should be the same as the one converted from logical input schema…` |
| pure cross | no (stripped) | OK |
| nested-loop with non-equi filter (`a < thr`) | yes | OK |
| equijoin / hash (`join_on lk==rk`, `join on=[k]`) | yes | OK |
| inner / left / full keyed join | yes | OK |

So it is specifically the **pure cross-product `CrossJoinExec`** path — NOT
NestedLoopJoin generally (filtered NLJ is fine) and NOT hash joins (#16221 fixed
those). Present on 51.0.0 → 54.0.0 → `main`. Where we hit it: lowering TPC-H
q11/q15/q22 (each cross-joins a 1-row scalar threshold) via
`benchmarks/translate_datafusion.py`; our workaround is
`_df_to_arrow → replace_schema_metadata(None)` on registration.

## Root cause (confirmed in source — `apache/datafusion` `main`)
For an **inner** join whose two inputs carry *different* values under the same
metadata key (`pandas`):
- **Logical** schema (`datafusion/expr/src/logical_plan/builder.rs`,
  `build_join_schema`, post-#16221) is **left-biased** → keeps LEFT's value.
- **Physical `HashJoinExec` / `NestedLoopJoinExec`** use the physical
  `build_join_schema` (`datafusion/physical-plan/src/joins/utils.rs` ~L320-328,
  also patched by #16221) → **left-biased** → matches logical → no error.
- **Physical `CrossJoinExec`** (`datafusion/physical-plan/src/joins/cross_join.rs`,
  `CrossJoinExec::new`) does NOT use that helper; it merges with
  `let mut metadata = left.metadata().clone(); metadata.extend(right…)` →
  `HashMap::extend` is **right-biased** → keeps RIGHT's value.

Logical=LEFT vs physical CrossJoin=RIGHT ⇒ the physical-plan verification
`schema_satisfied_by` (`datafusion/core/src/schema_equivalence.rs:24-25`, whose
first check is `original.metadata() == candidate.metadata()`) fails, raising the
error at `datafusion/core/src/physical_planner.rs:1072`. #16221 fixed the shared
`build_join_schema` helper (hash + nested-loop) but **never touched
`cross_join.rs`** (its diff was only `builder.rs` + `joins/utils.rs` + tests), so
`CrossJoinExec` alone still diverges.

## Standalone repro (pure datafusion + pyarrow + pandas; run OUTSIDE the repo dir
so the source `pandas/` doesn't shadow the installed one)
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
        n = ctx.table("l").join_on(ctx.table("r"), how="inner").count()   # pure cross join
        print(f"metadata={keep_meta}: OK ({n} rows)")
    except Exception as e:
        print(f"metadata={keep_meta}: FAIL -> {str(e)[:100]}")
```
Observed on 54.0.0: `metadata=True: FAIL`, `metadata=False: OK (2 rows)`.

## Dup-search (recorded — apache/datafusion, issues AND PRs)
Searched `cross join metadata`, `nested loop join metadata`, `Physical input
schema should be the same`, `CrossJoinExec metadata`, `join schema metadata`.
**No existing issue covers the cross-join case.** Related, all distinct:
- **#15754** (CLOSED, fixed by **#16221** merged 2025-06-02) — a *different* error
  (`join_selection` physical-optimizer, LEFT/hash join), not this
  `physical_planner` assertion. #16221 touched only `builder.rs` +
  `joins/utils.rs` — cross join untouched. **Do NOT reopen** (different operator +
  code path).
- **#19049** (CLOSED, fixed by **#21127** merged 2026-03-26,
  `intersect_metadata_for_union`) — the *same assertion* for the **UNION**
  operator. This is the precedent: maintainers scope metadata fixes per-operator.
- #18337 / #19069 (union UX dups); #12734 / #12729 / #12974 (2024, cross-join
  metadata *dropped*, not mismatched — different bug); #4432 (2022, context).

## Filing recommendation: ONE new scoped issue (do NOT reopen #15754)
- **Title:** `CrossJoinExec mismatches schema metadata: "Physical input schema
  should be the same as the one converted from logical input schema" on cross
  join over metadata-carrying tables`.
- **Body:** the repro above + `replace_schema_metadata(None)` workaround + "hash/
  keyed joins on the same tables pass". Then the root cause: `CrossJoinExec::new`
  merges metadata right-biased (`left.extend(right)`) while logical + physical
  `build_join_schema` are left-biased for inner joins after #16221, so
  `CrossJoinExec` alone diverges and trips `schema_satisfied_by`. Reference
  #15754/#16221 ("hash/nested-loop fixed there, `CrossJoinExec` missed") and
  #21127 (analogous union fix). Question + evidence + offer-to-help tone.
- **Suggested fix to offer:** align `CrossJoinExec::new`'s metadata merge with the
  join-type-aware selection in `build_join_schema` (inner ⇒ left-wins), or route
  cross join through `build_join_schema`; add a `test_files/metadata.slt` case
  mirroring #21127.
- **Labels:** `bug`, `datafusion` / `physical-plan` (as the project uses).

## Gates
- [x] **Reproduces on latest release** — executed on datafusion 54.0.0 (FAIL with
      metadata / OK without); source byte-identical at 54.0.0 + `main`.
- [x] **Duplicate search recorded** — not a dup; #15754 is a different path,
      #19049/#21127 is the per-operator precedent.
- [x] **Standalone repro** — above, pure datafusion.
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Definition of done
Filed (with #) referencing #15754/#16221 + #21127, or shelved if fixed on a newer
release, recorded here + in `README.md`. Keep the `replace_schema_metadata(None)`
workaround in `translate_datafusion.py` regardless.
