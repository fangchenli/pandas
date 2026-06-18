# Hand-off: AG2 — `hash_aggregate` aborts (SIGABRT) on >2 GB string keys

**For:** a fresh agent. Read `README.md` first. **Priority 2.** **Type: bug**
(uncatchable process abort; must be a `Status`/`ArrowInvalid`).

## Goal
File an apache/arrow bug: grouping a table by string key column(s) whose bytes
exceed 2 GB makes Acero's `hash_aggregate` row table throw a C++
`std::length_error` that escapes a `noexcept` boundary → `std::terminate` →
**SIGABRT (exit 134), uncatchable from Python**, instead of returning a clean
error from the row-table size check.

## Context & full detail
Found at TPC-H q10, SF-300 — grouping a ~20M-row joined intermediate by 7
customer columns incl. `c_comment`/`c_address`. Full detail — symptom, expected,
repro sketch, our mitigation (byte-gate at `_execute_arrow_table_groupby`,
commits `e4825bbb74`/`6f4a9f4ae3`) — is in **`UPSTREAM_ISSUES.md` → Issue 2**.

## What to do
1. **Build a standalone repro** (does not exist yet): `pa.table` with string key
   column(s) totaling > 2 GB; `Table.group_by(keys).aggregate([...])` (routes
   through `pyarrow.acero._group_by`) → expect SIGABRT. ~2.5 GB of string keys;
   16 GB machine; pure pyarrow + numpy.
2. **Run gates** (README §2): reproduce on **latest** pyarrow; duplicate search
   (acero / hash_aggregate / row table / `std::length_error` / 2GB string keys;
   check `string_view` umbrella **GH-44336**).
3. **File as a bug**, `[C++][Acero]`, linking GH-44336, noting the workload.
   Draft from `UPSTREAM_ISSUES.md` Issue 2.

## Gates
- [ ] Standalone repro built + SIGABRT confirmed on latest pyarrow.
- [x] Duplicate search done (2026-06-18) → **non-duplicate** (no existing report
      of the acero hash_aggregate abort on >2 GB string keys). Re-skim before
      filing in case something lands.
- [ ] Human approval.

## Definition of done
Filed → number recorded in `../ARROW_GAPS.md` AG2 + `UPSTREAM_ISSUES.md` Status
+ `README.md`. Byte-gate mitigation stays regardless (a clean error would still
fail the query; the gate completes it).
