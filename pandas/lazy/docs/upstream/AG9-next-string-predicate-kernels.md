# Hand-off: AG9-next — `string_view`/`binary_view` support for scalar string predicate kernels

**For:** a fresh agent. Read `README.md` (playbook) first, then
`AG9-string-view-kernels.md` (the umbrella context). **Target: `apache/arrow`
C++ compute.** This is a **code contribution (PR)** — the concrete next step on
the AG9 view-kernel track after our #50224 (grouper keys) merged 2026-07-07.
Scoped by research on 2026-07-09; **re-verify the live state before coding** (the
landscape moves).

> **QUANTIFIED by the differential probe (2026-07-10, `../DIFFERENTIAL_PROBE.md`
> COVERAGE class).** The kernel type-coverage matrix confirms on pyarrow 24.0.0
> that **all 17 tested scalar string kernels** (`utf8_length`, `utf8_upper/lower`,
> `utf8_reverse`, `utf8_slice_codeunits`, `utf8_capitalize`, `utf8_trim_whitespace`,
> `utf8_is_alpha`, `match_substring`, `match_substring_regex`, `match_like`,
> `starts_with`, `ends_with`, `find_substring`, `count_substring`,
> `replace_substring`, `split_pattern`) work on `string`/`large_string` but are
> `NotImplemented` on `string_view` (and on `dict<string>`). That's the exact
> unclaimed target list below, now measured — and the probe re-runs each release
> to track which flip to `ok` as sub-PRs land. Run it to re-scope before coding.

## BOTTOM LINE
Add `STRING_VIEW`/`BINARY_VIEW` support to the **scalar string predicate /
measurement** kernels in `scalar_string_ascii.cc` (+ `scalar_string_utf8.cc`):
`match_substring`, `match_substring_regex`, `match_like`, `starts_with`,
`ends_with`, `find_substring`, `count_substring`, `binary_length`/`utf8_length`,
`string_is_ascii`, `utf8_is_*`. The obvious targets — take/filter (#50164/#43010)
— are **already claimed, approved, and near-merge** (Periecle), so this is the
honest unclaimed adjacent gap: post-#50164 you can group/cast/take/filter view
arrays but **cannot evaluate a string WHERE clause on them** without casting to
`utf8` first. Highest TPC-H-filter payoff (LIKE predicates), self-contained,
**Medium** difficulty. Confidence: **High** it's real + unclaimed; **Medium** on
difficulty.

## Live state of the umbrella (verified 2026-07-09 — RE-CHECK at execution)
Umbrella **#44336** (open, pointer); real checklist in tracking issue **#39634**
(open): array-equality ✅, cast ✅ (#43302), unique/dictionary ✅, take/filter ❌.

| Item | Type | State | Who |
|---|---|---|---|
| #50224 grouper keys | PR | **MERGED 2026-07-07** | us (fangchenli) |
| #50166 cast null buffers | PR | **MERGED** | — |
| #50164 selection (take/filter) | PR | **OPEN, APPROVED, near-merge** | Periecle — active; "Closes #43010" |
| #43010 take/filter | issue | open (closes via #50164) | felipecrv / Periecle |
| #48734 de-box scalar string kernels | PR | **OPEN, STALLED ~6mo** | HyukjinKwon |
| #46128 cast-to-StringView memory | issue | open, active | andishgar |

**Take/filter is taken (don't duplicate #50164).** #48734 edits the SAME FILE we
target but is a de-boxing perf refactor (orthogonal, dormant) — coordinate/rebase,
no semantic overlap. No new view-*kernel* PR/issue since 2026-07.

## Coverage gap map (after #50224 + #50166 + #50164 landing)
| operator | view support | status |
|---|---|---|
| hash-agg grouper keys (legacy RowEncoder) | ✅ #50224 | done |
| cast → view | ✅ #50166 | done |
| take / filter | ⏳ #50164 | claimed, near-merge |
| **scalar string predicates** (LIKE/substring/prefix/length) | ❌ | **UNCLAIMED — this hand-off** |
| sort (`array_sort_indices`) — view has no physical-type mapping | ❌ | unclaimed (future) |
| row-table hash-join key encoding (`encode_internal.cc`) | ❌ | unclaimed, **Large/hard** (future) |

So the predicate gap is the cheapest, highest-value of the three unclaimed gaps.

## Scope (first PR)
Do the **input-view, fixed-output-type subset only**: predicates/measures that
READ view arrays and emit bool/int (`match_*`, `starts_with`, `ends_with`,
`find_substring`, `count_substring`, `*_length`, `*_is_ascii`, `utf8_is_*`).
**Defer** kernels that *emit* strings (`utf8_upper/lower`, `replace_substring`,
`utf8_trim`) — output-view construction is the hard part, a follow-up PR.

Mechanism: a view array already yields `std::string_view` per element via
`GetView(i)`, so the matcher logic is unchanged — the work is templating the exec
functor over an `ArraySpan` of view layout + adding registration entries.
**Watch:** kernels using the contiguous-`Transform` fast path assume packed
offset buffers and need a per-element path for views.

- **Files:** `cpp/src/arrow/compute/kernels/scalar_string_ascii.cc`,
  `.../scalar_string_utf8.cc`, `.../scalar_string_internal.h` (the
  `GenerateVarBinaryBase` / `BaseBinaryTypes()` dispatch helpers), tests
  `.../scalar_string_test.cc`, docs `docs/source/cpp/compute.rst` (type-support
  tables).
- **Template to follow:** **#50164** (selection kernels) — the current
  reviewer-blessed pattern for extending `VisitType`/registration dispatch to
  view arrays + preserve-variadic-buffer tests. Mirror its structure. Use
  **#50224** (our merged grouper PR) as the template for the
  umbrella-engagement + benchmark + review workflow.
- **TPC-H payoff:** Q2 (`p_type LIKE '%BRASS'`→`ends_with`), Q9 (`p_name LIKE
  '%green%'`→`match_substring`), Q13 (`o_comment NOT LIKE`), Q14/Q16/Q20
  (`LIKE 'X%'`→`starts_with`).

## Gates / checklist (guardrail: NO outward comment/issue/PR without human go-ahead)
1. **Re-verify unclaimed** at execution: re-pull #44336, #39634, #50164 (merged?),
   #48734 (revived?); re-search `string_view` + scalar/substring/utf8 PRs+issues.
2. **Engage first (draft, hold):** prepare a #44336 (and/or #39634) comment stating
   intent, noting #50164 covers take/filter and #48734 is an orthogonal de-boxing
   refactor of the same file (offer to rebase-coordinate). Post only after approval.
3. **Build:** Arrow C++ from latest `main`, compute + tests enabled; clean baseline.
4. **Implement** the predicate subset per the #50164 dispatch/registration pattern;
   rebase over #48734 if it lands.
5. **Test:** extend `scalar_string_test.cc` — view cases: empty, inlined ≤12-byte,
   out-of-line, sliced, null-in-variadic; assert view == utf8 results per kernel.
6. **Benchmark (lazy-pandas rule):** view vs cast-to-utf8 inputs, quantify the
   avoid-the-cast win, on latest `main`; numbers go in the PR description.
7. **Lint/docs:** `pre-commit run cpp`; update `compute.rst` type-support tables.
8. **Only after human go-ahead:** PR `GH-<new-issue>: [C++][Compute] Support
   string_view/binary_view in scalar string predicate kernels`, link
   #44336/#39634, request #50164 reviewers (pitrou, zanmato1984).

## Definition of done
PR merged or in review with maintainer buy-in → recorded in
`AG9-string-view-kernels.md`, `../ARROW_GAPS.md` AG9, and `README.md`. This is the
next single step; sort-view and row-table-view encoding are the subsequent AG9
steps (larger).

## Provenance / caveats
State above pulled live via `gh` + `raw.githubusercontent.com/apache/arrow/main/`
on 2026-07-09 (source: `scalar_string_ascii.cc`, `vector_sort_internal.h`,
`encode_internal.cc`) — current-`main` but not pinned to a SHA. Re-verify against
exact HEAD at execution. "Rides #50224 momentum" = the umbrella's momentum, not
the grouper/row-format subsystem specifically (that's the deferred row-table gap).
