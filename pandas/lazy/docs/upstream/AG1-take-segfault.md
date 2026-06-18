# Hand-off: AG1 — `take` segfaults on >2 GB int32-offset string data

**For:** a fresh agent. Read `README.md` first. **Priority 2.** **Type: bug**
(a library must raise, never kill the host process).

## Goal
File an apache/arrow bug: `pyarrow.compute.take` on a `string` (int32-offset)
ChunkedArray **segfaults** (exit 139, no Python error) when the gathered output
exceeds 2 GB, instead of raising `ArrowInvalid` ("offset overflow … cast to
`large_string`") as the concat kernels already do.

## Context & full detail
Found in TPC-H q2/q18 at SF-100 inside `pd.merge`'s reindex. The underlying 2 GB
limit (int32 offsets) is **by design**; the bug is the **crash instead of a
catchable error**. Complete detail — symptom, expected behavior, repro sketch,
our mitigation (commit `596fee591c`) — is in **`UPSTREAM_ISSUES.md` → Issue 1**.

## What to do
1. **Build a standalone repro** (does not exist yet): one `string` ChunkedArray
   whose chunks total > 2³¹ bytes; `pc.take` with int64 indices whose gathered
   output also exceeds 2 GB → expect segfault. Use `faulthandler` to capture it.
   ~2.5 GB of strings; must run on a 16 GB machine. Pure pyarrow + numpy.
2. **Run gates** (README §2): reproduce on **latest** pyarrow; duplicate search
   (`take` / offset overflow / segfault / `string` 2GB — and check the
   `string_view` umbrella **GH-44336** and linked issues); confirm
   `pa.concat_arrays` past 2 GB already raises cleanly (the contrast case).
3. **File as a bug**, `[C++]`/`[Python]`, linking GH-44336 and noting the
   `pd.merge`/reindex workload context. Draft the body from `UPSTREAM_ISSUES.md`
   Issue 1 (symptom → expected → repro → context).

## Gates
- [ ] Standalone repro built + segfault confirmed on latest pyarrow.
- [ ] Duplicate search recorded.
- [ ] Human approval.

## Definition of done
Filed (or attached to an existing umbrella issue) → number recorded in
`../ARROW_GAPS.md` AG1 + `UPSTREAM_ISSUES.md` Status + `README.md`. Our upcast
mitigation stays regardless.
