# Upstream Arrow Issues To File (found by the TPC-H scale runs, June 2026)

Two process-killing crash classes hit at SF-100/SF-300 on pyarrow 24.0.0
(Arrow C++ 24). Both are **upstream defects** (a library must raise, never
kill the host process), distinct from the underlying *by-design* limitation
(plain `string` uses int32 offsets → 2 GB per array; `large_string` /
`string_view` exist for this). Our engine now routes around both either
way; filing them upstream documents production-shaped pressure on the
small-string types and supports the `string_view` kernel work
([STRING_VIEW_CONTRIBUTION_PLAN.md](STRING_VIEW_CONTRIBUTION_PLAN.md)).

## Issue 1 — `take` SEGFAULTS on >2 GB int32-offset string data

- **Seen at**: TPC-H q2/q18, SF-100, inside `pd.merge`'s reindex
  (`pyarrow.compute.take` on an ArrowExtensionArray) — exit 139, no Python
  error, found only via `faulthandler`.
- **Expected**: `ArrowInvalid` ("offset overflow ... consider casting to
  `large_string`"), as the concat kernels already do.
- **Repro sketch**: one `string` ChunkedArray whose chunks total > 2^31
  bytes; `pc.take` with int64 indices whose gathered output also exceeds
  2 GB → segfault.
- **Our mitigation** (commit `596fee591c`): merge inputs upcast
  `string → large_string` / `binary → large_binary` before `pd.merge`.

## Issue 2 — acero hash_aggregate ABORTS (uncatchable) on >2 GB string keys

- **Seen at**: TPC-H q10, SF-300 — grouping a ~20M-row joined intermediate
  by 7 customer columns incl. `c_comment`/`c_address`. C++
  `std::length_error` from `vector::_M_default_append` escapes a
  `noexcept` boundary → `std::terminate` → SIGABRT (exit 134). Cannot be
  caught from Python.
- **Expected**: a `Status`/`ArrowInvalid` from the row-table size check.
- **Repro sketch**: `pa.table` with string key column(s) totaling > 2 GB,
  `Table.group_by(keys).aggregate([...])` (routes through
  `pyarrow.acero._group_by`).
- **Our mitigation** (commits `e4825bbb74`, `6f4a9f4ae3`): byte-gate at
  the `_execute_arrow_table_groupby` chokepoint (≥1.5 GB of string key
  bytes → numpy multi-key path), whose own int64 cardinality-product
  overflow was fixed with Horner packing + periodic re-densification.

## Already-clean behavior, for contrast (no issue needed)

`pa.concat_arrays` / `combine_chunks` past 2 GB of `string` raises a clean
`ArrowInvalid` ("offset overflow while concatenating arrays...") — the
behavior the two cases above should match. (Hit at q13 SF-100; our fix:
concatenate as ChunkedArray, commit `596fee591c`.)

## Status

- [ ] Synthesize standalone repro scripts (~2.5 GB strings each; runnable
      on a 16 GB machine)
- [ ] File both against apache/arrow, linking GH-44336 (view-type kernel
      umbrella) and noting the workload context
- [ ] Note issue numbers here once filed

Keep-the-gate note: even after upstream fixes, our mitigations stay — a
clean `ArrowInvalid` would still fail the query; the gates complete it.
