# Hand-off: AG9 — Arrow `string_view` kernel coverage (contribution)

**For:** a fresh agent. Read `README.md` first. **Priority 3.** This is a
**code contribution (PR)**, not just an issue — larger and longer-running than
AG1/AG2/AG4.

## Goal
Improve Arrow's `string_view` compute-kernel coverage so pandas' Arrow-backed
string dtype gets the at-scale/string performance wins (avoiding the 2 GB
int32-offset cliff behind AG1/AG2). High ecosystem leverage — benefits all Arrow
consumers.

## Context & full plan
`string_view` (German-style strings) is the modern Arrow string layout but its
compute-kernel coverage lags `string`/`large_string`. The detailed scope, effort
estimate (~4–6 person-weeks), expected impact (est. +0.04–0.07 TPC-H geo-mean,
big at-scale/string gains), and approach are in
**`STRING_VIEW_CONTRIBUTION_PLAN.md`** (this folder). The Arrow umbrella is
**GH-44336** (view-type kernel coverage).

## What to do
1. Read `STRING_VIEW_CONTRIBUTION_PLAN.md` fully.
2. **Re-scope against current Arrow** (playbook §1): `string_view` kernel
   coverage moves fast — enumerate which kernels we need are *still* missing on
   the latest release (don't work from the plan's possibly-stale list).
3. **Engage maintainers first** (GH-44336 / dev list) before writing code —
   confirm scope, avoid duplicating in-flight work, agree on which kernels.
4. Implement + test + PR per Arrow's C++ contribution process. Expect iteration.

## Gates
- [ ] Current-Arrow coverage gap re-confirmed (specific missing kernels listed).
- [ ] Maintainer alignment on scope (comment on GH-44336 / dev list).
- [ ] Human approval before any outward comment/PR.

## Definition of done
PR(s) merged or in review with maintainer buy-in → recorded in `../ARROW_GAPS.md`
AG9 + `README.md`. Note: this is the highest-effort item here; treat as a
multi-step project, not a single sitting.
