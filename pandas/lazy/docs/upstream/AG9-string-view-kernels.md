# Hand-off: AG9 — Arrow `string_view` kernel coverage (contribution)

**For:** a fresh agent. Read `README.md` first. **Priority 3.** This is a
**code contribution (PR)**, not just an issue — larger and longer-running than
AG1/AG2/AG4.

## Goal
Improve Arrow's `string_view` compute-kernel coverage so pandas' Arrow-backed
string dtype gets the at-scale/string performance wins (avoiding the 2 GB
int32-offset cliff behind AG1/AG2). High ecosystem leverage — benefits all Arrow
consumers.

> **DUP-SEARCH RESULT (2026-06-18): this is ACTIVELY IN-FLIGHT upstream — do NOT
> file or start coding blind.** The umbrella **#44336 "[C++] Binary View Compute
> Kernels" is open**, with open PRs **#50164** (view arrays in selection
> kernels), **#50166** (cast-to-view null buffers), **#48734** (eliminate Array
> boxing in scalar string kernels) and open sub-issues **#43010**
> (STRING_VIEW/BINARY_VIEW in array_take/array_filter), **#46128** (cast-to-
> StringView memory). The contribution is to **join the existing effort**, not
> open new work.

## Context & full plan
`string_view` (German-style strings) is the modern Arrow string layout but its
compute-kernel coverage lags `string`/`large_string`. Detailed scope, effort
estimate (~4–6 person-weeks), and expected impact (est. +0.04–0.07 TPC-H
geo-mean) are in **`STRING_VIEW_CONTRIBUTION_PLAN.md`** (this folder).

## What to do
1. Read `STRING_VIEW_CONTRIBUTION_PLAN.md`, then **read #44336 and the open PRs
   above** to see what's already done / in progress — do NOT duplicate them.
2. **Re-scope against current Arrow** (playbook §1): enumerate which kernels we
   actually need that are *still* unaddressed after the open PRs land.
3. **Engage on #44336** (comment offering to take a specific unclaimed kernel)
   before writing code; align with maintainers.
4. Implement + test + PR per Arrow's C++ process, referencing #44336.

## Gates
- [x] Duplicate/overlap search done → #44336 + PRs #50164/#50166/#48734, issues
      #43010/#46128. Re-check these are still the live set before engaging.
- [ ] A specific unaddressed kernel identified (not covered by the open PRs).
- [ ] Maintainer alignment on #44336.
- [ ] Human approval before any outward comment/PR.

## Definition of done
PR(s) merged or in review with maintainer buy-in → recorded in `../ARROW_GAPS.md`
AG9 + `README.md`. Note: this is the highest-effort item here; treat as a
multi-step project, not a single sitting.
