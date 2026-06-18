# Hand-off: AG1 — `take` segfaults on >2 GB int32-offset string data

**For:** a fresh agent. Read `README.md` first. **Priority 2.**

> **DUP-SEARCH RESULT (2026-06-18): this is effectively a duplicate of the open
> issue #25822** "[C++] Take kernel can't handle ChunkedArrays that don't fit in
> an Array" — same root cause (Take concatenates first, breaks past 2 GB). **Do
> NOT file a new issue.** Our distinct contribution is that it currently
> **segfaults instead of raising** — add that as a comment + standalone repro on
> #25822. Related open issues: #33049/#41890/#44164 (concat offset overflow),
> #46814 (large-data segfault).

## Goal
**Comment on apache/arrow #25822** with a minimal repro showing
`pyarrow.compute.take` on a `string` (int32-offset) ChunkedArray **segfaults**
(exit 139, no Python error) when the gathered output exceeds 2 GB — i.e. even
before #25822's requested "handle it" fix lands, it should at least raise
`ArrowInvalid` rather than crash the process. (Only open a *new* issue if a
maintainer asks to split the crash-vs-handle aspects.)

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
2. **Confirm on latest pyarrow** (README §1) that it still segfaults (vs raises).
   Duplicate search already done (→ #25822); re-skim #25822's recent comments in
   case the crash aspect is already covered.
3. **Comment on #25822** with the repro + the "segfaults instead of raising"
   point (detail/context in `UPSTREAM_ISSUES.md` Issue 1: symptom → expected →
   repro → the `pd.merge`/reindex workload). Don't open a new issue unless asked.

## Gates
- [ ] Standalone repro built + segfault confirmed on latest pyarrow.
- [x] Duplicate search done → #25822 (re-skim its comments before posting).
- [ ] Human approval to post.

## Definition of done
Comment posted on #25822 (or new issue if a maintainer requests the split) →
recorded in `../ARROW_GAPS.md` AG1 + `UPSTREAM_ISSUES.md` Status + `README.md`.
Our upcast mitigation stays regardless.
