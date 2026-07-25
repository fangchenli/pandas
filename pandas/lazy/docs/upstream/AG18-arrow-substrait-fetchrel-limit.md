# Hand-off: AG18 — Arrow's Substrait consumer silently returns 0 rows for every `LIMIT`

> ## ✅ EXECUTED — FILED as [PR #50635](https://github.com/apache/arrow/pull/50635) (issue GH-50634), **OPEN / draft**
>
> **Status as of 2026-07-25** (verified live via `gh`). Filed by fangchenli; marked
> a **"Critical Fix"** (produces incorrect data). No reviews yet.
>
> | | |
> |---|---|
> | **PR** | [#50635](https://github.com/apache/arrow/pull/50635) `GH-50634: [C++][Engine] Fix Substrait consumer silently dropping modern FetchRel/AggregateRel fields` — opened 2026-07-25 |
> | **State** | OPEN, **draft**, MERGEABLE, +449 −16 across 5 files. No reviews yet. |
> | **Closes** | issue **#50634** |
>
> **The filed fix GENERALIZED the root cause beyond this doc's FetchRel scope.** The
> vendored-proto version lag (v0.44.0) silently drops *several* modern fields, not
> just `count_expr`. PR #50635 fixes the whole family in one bump:
> - **Bump** the vendored substrait proto **v0.44.0 → v0.63.0** (matches DataFusion's
>   substrait-rs; deprecated arms remain for back-compat) — exactly what this
>   hand-off recommended.
> - **FetchRel** (this doc, AG18a): read the `count_mode`/`offset_mode` oneofs, eval
>   `count_expr`/`offset_expr` to a constant, unset count = ALL, unset offset = 0,
>   skip a no-op fetch. Fixes silent `LIMIT 0`.
> - **AggregateRel** (a *new sibling* found while fixing this — same root cause):
>   GROUP BY keys moved from deprecated `Grouping.grouping_expressions` to
>   `Grouping.expression_references` (indexing the AggregateRel-level
>   `grouping_expressions`). Reading only the deprecated field **dropped the group
>   keys → collapsed every row into one group** — a second silent-wrong-result.
> - **JoinRel**: update the renamed `JoinType` enum (`JOIN_TYPE_SEMI/ANTI` →
>   `JOIN_TYPE_LEFT_SEMI/LEFT_ANTI`) surfaced by the bump.
> - Regression tests for the expr arms, unset-count = ALL, and grouping-by-reference.
>
> ⚠ **Do not conflate with AG19** (`No conversion function … starts_with`): that is
> a *function-coverage* gap (`extension_set.cc` registry), a different fix, still
> unfiled. AG18 is the *deprecated-field/stale-proto* class.
>
> ---
> Everything below is the original 2026-07-24 scoping (FetchRel only), kept as
> provenance for how the finding was characterized and gated before filing.

**For:** a fresh agent. Read `README.md` first. **Target project: `apache/arrow`**
(the C++ `arrow::engine`/Substrait **consumer**). **Priority: 2** — the *severity*
is the highest class on the board (a **silent wrong result**, no error), but the
fix is a heavy vendored-proto bump (see "The fix"), which lowers actionability and
likelihood of quick acceptance. File as a **correctness bug**, not an enhancement.

> **✅ `main`-FIRST GATE CLOSED (2026-07-24) — verified against `apache/arrow` HEAD
> `62d2dd8270`, not the release.** The bug is **CONFIRMED LIVE at HEAD** and the
> root cause was **sharpened** from the original "ignores a oneof" framing to the
> real one: a **stale vendored Substrait proto**. Filing-ready **pending human
> go-ahead** (guardrail — the probe characterizes; a human files).
>
> _History: found by the Substrait roundtrip probe (`../SUBSTRAIT_ROUNDTRIP.md`);
> originally recorded inside `AG17-...md` §residual against pyarrow 25.0.0. Promoted
> to this self-contained hand-off 2026-07-24 after the `main`-first gate confirmed
> it live and re-derived the mechanism from source._

## The finding (one line)
Arrow's Substrait **consumer** reads `FetchRel` via the **scalar** `fetch.count()`
accessor from a **vendored Substrait proto pinned to v0.44.0** (≈2 years / ~19
releases stale). Any spec-current producer (DataFusion emits the `count_expr`
expression arm, added in Substrait **v0.63.0**) sends a `FetchRel` whose scalar
`count` field is unset → `fetch.count()` returns its default **0** →
`FetchNodeOptions(offset, 0)` → **`LIMIT 0` → the query silently returns 0 rows,
with no error.**

## Root cause — a vendored-proto version lag (this is the crux)
Arrow pins **substrait v0.44.0** (Arrow's `versions.txt`,
`ARROW_SUBSTRAIT_BUILD_VERSION=v0.44.0`). The `FetchRel` message changed shape
across releases:

| substrait version | `FetchRel` count field | who uses it |
|---|---|---|
| **v0.44.0** | scalar `int64 count = 4;` only (no expression form) | **Arrow's pinned proto** — `fetch.count()` reads field 4 |
| v0.54.0 | scalar `int64 count = 4;` (comment adds "use -1 = ALL") | — |
| v0.60.0 | still scalar only | — |
| **v0.63.0** | `oneof count_mode { int64 count = 4 [deprecated]; Expression count_expr = 6; }` — **the oneof + expression form lands here**; "unset = ALL" | **DataFusion's pinned substrait-rs (0.63.0)** — emits the `count_expr` arm |
| substrait `main` | `reserved 3, 4;` — the scalar `count`/`offset` are **removed**; only `offset_expr`/`count_expr` remain | (unreleased) |

Because `count` and `count_expr` are the two arms of a **`oneof`** from v0.63.0 on,
a producer that sets `count_expr` (field 6) leaves `count` (field 4) **unset** —
that is exactly how oneofs work. Arrow's v0.44.0 proto has no field 6 at all, so on
the wire that field lands in protobuf **unknown-fields** and is silently dropped;
`fetch.count()` then returns the scalar default **0**.

> **Note the earlier framing was imprecise.** The original AG18 write-up said
> "Acero ignores the `oneof count_mode`." That describes v0.63/`main`, but Arrow's
> pinned proto (v0.44.0) has **no oneof to ignore** — it predates the whole
> migration. The accurate root cause is the *version lag*, and the fix follows from
> that (bump the proto), not from "check the oneof" in the current code.

## Mechanism, end to end
1. DataFusion (substrait-rs 0.63.0) lowers a SQL/DataFrame `LIMIT n` to a
   `FetchRel` with `count_mode = CountMode::CountExpr(literal n)`
   (`datafusion/substrait/src/logical_plan/producer/rel/fetch_rel.rs`, sets
   `offset_mode`/`count_mode`, never the scalar arms).
2. The plan is serialized to Substrait protobuf. `count_expr` is field 6.
3. Arrow parses it with the **v0.44.0** proto (no field 6) → field 6 → unknown
   fields, dropped. Scalar `count` (field 4) is absent → default `0`.
4. `relation_internal.cc` FetchRel handler (`case …kFetch:`, HEAD lines ~782-786):
   ```cpp
   int64_t offset = fetch.offset();
   int64_t count  = fetch.count();      // <-- returns 0 (field 4 unset)
   acero::Declaration fetch_dec{
       "fetch", {input.declaration}, acero::FetchNodeOptions(offset, count)};
   ```
   → `FetchNodeOptions(offset, 0)` → Acero fetch node emits **0 rows**.
5. Result: **every `LIMIT` query silently returns an empty result** — no error, no
   warning. (`offset` has the same lag but its default `0` is a harmless no-op, so
   only `count` manifests as a visible wrong answer.)

## HEAD verification (the gate)
- Consumer still scalar at HEAD `62d2dd8270`:
  `cpp/src/arrow/engine/substrait/relation_internal.cc:782-786` — `fetch.count()` /
  `fetch.offset()`, no `count_expr`/`offset_expr`/`count_mode` handling anywhere in
  the file (grep: 0 hits for `count_expr|offset_expr|count_mode|has_count`).
- Proto pin still v0.44.0 at HEAD (`versions.txt`).
- So the defect is **not** a released-artifact mirage — it is live on `main`.

## Severity / why it matters
Highest class: a **silent wrong result** (empty output, no exception) on **every**
`LIMIT` query produced by any current-Substrait producer. In the probe's TPC-H
Substrait fan-out this hit **q3 / q10 / q18 / q21** (all use `LIMIT`/top-N). Unlike
AG19 (a hard `No conversion function` error you can't miss), this one fails *quietly*
— the most dangerous failure mode.

## How it was found + empirical repro + our workaround
The Substrait roundtrip probe (`../SUBSTRAIT_ROUNDTRIP.md`,
`../../benchmarks/substrait_roundtrip.py`) lowered the 22 TPC-H plans through
DataFusion → Substrait → Acero and saw the top-N queries return 0 rows against a
DuckDB oracle. `substrait_fixup.py` works around it by **mirroring the count
literal back into the deprecated scalar `count` arm (field 4)** that Arrow's
v0.44.0 proto reads — which restores correct row counts, confirming the mechanism
(Arrow reads field 4; the producer never set it).

Standalone repro (throwaway venv, latest releases —
`pip install -U pyarrow datafusion duckdb substrait`): build a DataFusion plan with
a `LIMIT`, lower to Substrait, feed to `pyarrow.substrait.run_query`; observe 0
rows vs the expected top-N. (Fill `ScalarFunction.output_type` first per AG17 or it
errors earlier.) Confirmed on pyarrow 23.0.1 and 25.0.0.

## The fix (bigger than a registry add — be honest about scope)
There is **no narrow fix**: Arrow's v0.44.0 proto literally cannot represent
`count_expr`, so reading it requires newer generated code. Two coupled changes:
1. **Bump the vendored Substrait proto** from v0.44.0 to **≥ v0.63.0** (where
   `count_mode`/`offset_mode` land). This is the heavy part: a ~19-release jump
   spans changes across *every* rel/type/expression message and could ripple into
   other consumer paths — likely why it hasn't been done. It is the real ask.
2. **Rewrite the `FetchRel` consumer** to read the oneof: evaluate `count_expr` /
   `offset_expr` (Acero needs a **constant** — accept a literal, error on a
   non-literal expression), treat **unset `count_mode` = ALL** (no limit, *not* 0)
   and **unset `offset_mode` = 0**. Keep reading the deprecated scalar arm as a
   fallback for old producers until `main`'s `reserved 3,4` forces its removal.

Frame the issue around **(1)**: "the vendored Substrait proto is pinned to v0.44.0,
so the consumer silently drops `count_expr` from any current producer and returns
0 rows for every `LIMIT`." The FetchRel rewrite is the concrete consumer-side
consequence.

## AG18(b) `precision_timestamp` — RESOLVED upstream, DO NOT file
The original AG18 bundled a second manifestation: the consumer erroring on
`precision_timestamp`. **This is fixed at HEAD** — `type_internal.cc:136-145`
(consumer `FromProto`) handles `kPrecisionTimestamp`/`kPrecisionTimestampTz` and
`:328,:334` (producer `ToProto`) emit them. It was true only against the pyarrow
25.0.0 *release*; an AG17-class released-artifact mirage the gate caught. Our
`legacy_timestamp` fixup is now floor-debt, not an Arrow gap. **AG18 narrows to the
single silent-`LIMIT 0` bug.**

## Dup-search (recorded — apache/arrow, 2026-07-24)
- `gh search` → **zero** open Arrow issues on `FetchRel` / `count_mode` /
  `substrait fetch limit` / `substrait version bump` / `update substrait 0.44`.
  **Non-dup.**
- The AG9/AG19 umbrella #13285 ("register tricky Substrait functions") is about
  *function coverage*, not `FetchRel`, and was auto-closed as stale 2025-12-05 —
  unrelated to this.
- Refresh at file time: `substrait FetchRel count`, `substrait limit returns
  empty/0 rows`, `bump/update vendored substrait version`, and scan open PRs
  touching `relation_internal.cc` or `versions.txt`.

## What "desired behavior" rests on (be honest)
The Substrait spec is unambiguous: from v0.63.0, count is a `oneof count_mode` and
**unset = ALL**; a compliant consumer must read `count_expr` and must not treat a
producer that set the expression arm as `LIMIT 0`. Arrow returning 0 rows here is a
genuine spec-conformance **correctness** bug, not a preference. The only judgment
call is the *fix cost* (the proto bump) — which is why this is filed as a bug with
a clear repro, leaving the maintainers to decide the bump's scope/timing.

## Recommendation
File on **apache/arrow** as a correctness bug: *"Substrait consumer silently
returns 0 rows for `LIMIT` — vendored substrait proto pinned to v0.44.0 drops
`count_expr` from current producers."* Include the standalone repro, the HEAD
source citation (`relation_internal.cc:782-786`), the version table above, and the
DataFusion-0.63-emits-`count_expr` mechanism. Do **not** bundle `precision_timestamp`
(resolved) or the AG19 function-coverage gap (separate issue). **Nothing filed
without explicit human go-ahead** (guardrail).

## Definition of done
**✅ FILED as PR #50635 (GH-50634), OPEN/draft, 2026-07-25** — the version-lag
diagnosis held and generalized (FetchRel + AggregateRel + JoinRel enum, one proto
bump). Remaining: mark PR ready-for-review, land review, merge. Keep the
`substrait_fixup.py` scalar-count mirror (and the grouping/join fixups) in our
lowering until a fixed Arrow ships in a release we depend on, then retire them.

_History: characterized, `main`-first-gated (HEAD `62d2dd8270`), mechanism
re-derived from source, non-dup, repro on latest releases — was hand-off-ready
2026-07-24, filed 2026-07-25._
