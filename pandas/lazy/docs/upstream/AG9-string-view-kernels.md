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
>
> **UPDATE (2026-07-08): PR #50166 (GH-49740, "cast to view types leaving null
> variadic buffers") MERGED 2026-07-01.** Umbrella #44336 remains open; #50164
> and #48734 still open; issues #43010 and #46128 still open. Net: the effort is
> progressing but NOT complete — before engaging, re-pull the live set (#50166 is
> done, don't re-scope around it) and re-check which selection/take/filter view
> kernels are still unaddressed.
>
> **🎉 OUR CONTRIBUTION MERGED (2026-07-08): PR #50224 (GH-50223, "[C++][Compute]
> Support string_view/binary_view keys in the hash-aggregate Grouper") — authored
> by fangchenli, merged by pitrou 2026-07-07, xref #44336.** This is the AG9 "join
> the existing effort" strategy bearing fruit: `Grouper` (behind Acero's
> hash-aggregate / `Table.group_by`) previously rejected view keys with
> `NotImplemented: Keys of type string_view`; the PR adds `BinaryViewKeyEncoder`
> so `group_by` on a view-typed key works. This is exactly the view-native path
> that sidesteps the AG1/AG2 >2 GB int32-offset cliff for grouped aggregation.
> **AG9 is now an active in-flight contribution, not just a watch item.** Next
> unaddressed kernels remain (selection/take/filter view kernels — #50164/#43010).
>
> **NEXT KERNEL SCOPED (2026-07-09) → `AG9-next-string-predicate-kernels.md`.**
> take/filter (#50164/#43010) is already claimed + approved + near-merge by
> Periecle — do NOT duplicate. Recommended unclaimed next PR: **view support for
> scalar string predicate kernels** (`match_substring`/`match_like`/`starts_with`/
> `ends_with`/`find_substring`/`*_length` in `scalar_string_ascii.cc`) — the
> missing middle of the filter pipeline (post-#50164 you can take/filter view
> arrays but can't evaluate a string WHERE clause on them), highest TPC-H-LIKE
> payoff, Medium difficulty. That doc has the full state table, gap map, files,
> template (#50164), and checklist.
>
> **✅ THAT PR IS NOW FILED (2026-07-11) AND IN REVIEW: #50479 (GH-50478,
> "[C++][Compute] Support string_view/binary_view in scalar string predicate
> kernels"), authored by fangchenli — OPEN, MERGEABLE, +323 −33 / 6 files, no
> approval yet.** `zanmato1984` (a #50164 reviewer, as the hand-off recommended)
> engaged 2026-07-13; his one ask — a `utf8_view` + `ignore_case` + non-ASCII test
> to lock the `StringViewType`-vs-generic-`BinaryViewType` dispatch distinction —
> was addressed 07-14 in `8082d31`. A Copilot null-slot/OOB hazard (unvalidated
> view null headers) was addressed via `VisitArrayValuesInline` + null-visitor;
> Copilot re-reviewed clean 07-14. **Second AG9 contribution in flight; nothing
> outstanding on our side.** Full state → `AG9-next-string-predicate-kernels.md`.
>
> **⚠ Status hygiene (2026-07-14).** A report that "string_view/binary_view
> support for scalar string predicate kernels was merged" was checked and found
> **false**: #50479 is still open, #50164 (take/filter) is still open, tracker
> #39634 still has take/filter unchecked, and Arrow `main`'s
> `scalar_string_ascii.cc`/`scalar_string_utf8.cc`/`scalar_string_internal.h` have
> **zero** Arrow view-type references (their many `std::string_view` hits are the
> C++ stdlib type — the name collision is the trap). What *did* merge is #49964
> (view **comparison** kernels, Periecle, 2026-06-09) — a different family, already
> recorded in `STRING_VIEW_CONTRIBUTION_PLAN.md`. **Verify view coverage by reading
> `main` source, not by PR-title search.**

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
      (2026-07-08: #50166 MERGED; #50164/#48734/#43010/#46128 + umbrella still open.)
- [x] A specific unaddressed kernel identified (not covered by the open PRs).
      Round 1: hash-agg Grouper view keys → #50224 (merged 2026-07-07).
      Round 2 (2026-07-09): scalar string predicate kernels → #50479 (open).
- [x] Maintainer alignment: #50224 merged by pitrou (xref #44336); #50479 under
      review by zanmato1984 since 2026-07-13.
- [x] Human approval before any outward comment/PR — obtained for both #50224 and
      #50479. **Still required for every future one** (standing guardrail).

## Definition of done
PR(s) merged or in review with maintainer buy-in → recorded in `../ARROW_GAPS.md`
AG9 + `README.md`. Note: this is the highest-effort item here; treat as a
multi-step project, not a single sitting.

**Progress: 1 merged (#50224), 1 in review (#50479).** The umbrella gap is NOT
closed — take/filter (#50164/#43010) is claimed-but-open, and sort-view
(`array_sort_indices`) + row-table hash-join key encoding (`encode_internal.cc`,
Large/hard) remain unclaimed. Next steps in cost order live at the foot of
`AG9-next-string-predicate-kernels.md`.
