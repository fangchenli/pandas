# Upstream contributions — backlog & hand-off playbook

This folder is the home for upstream contributions surfaced by the lazy-pandas
**substrate probe** (see `../PROBE_CHARTER.md`; gap registry in
`../ARROW_GAPS.md`). Each actionable feature has a **self-contained hand-off
doc** below that a fresh agent can execute end-to-end without prior context.

**Read this playbook first**, then open the hand-off for your feature.

> **Finding new gaps:** two standing discovery instruments —
> 1. `../DIFFERENTIAL_PROBE.md` (`../../benchmarks/differential_probe.py`) — a
>    fixed workload grid across `{pandas, polars, acero, datafusion-sql,
>    datafusion-df}` emitting RESULT/PLAN/PERF divergences. Rediscovered
>    AG10/AG4/AG5′ from scratch.
> 2. `../SUBSTRAIT_ROUNDTRIP.md` (`../../benchmarks/substrait_roundtrip.py`) —
>    lowers the 22 TPC-H plans, roundtrips each through **Substrait** (the
>    portable IR), and scores survival across DataFusion + Acero consumers.
>    Cross-engine divergences on one IR are free findings; it found **AG17**.
>
> Run both on each new release to generate the next backlog rows (and auto-retire
> fixed ones) instead of finding by accident.

## Backlog index

| Feature | Gap | Status | Priority | Hand-off |
|---|---|---|---|---|
| AG4 | Acero string-key hash-agg 3.5–10× slower than dict-encoded | verified, **non-dup**, draft ready | **1** | `AG4-acero-string-key-hashing.md` |
| AG1 | `take` segfaults on >2 GB int32-offset string | **dup → comment on #25822** | 2 | `AG1-take-segfault.md` |
| AG2 | `hash_aggregate` SIGABRT on >2 GB string keys | **non-dup**, repro pending | 2 | `AG2-hashagg-abort.md` |
| AG9 | Arrow `string_view` kernel coverage (contribution) | **active contribution — OUR PR #50224 merged 2026-07-07** (Grouper view keys); #44336 open, #50166 merged | 3 | `AG9-string-view-kernels.md` |
| AG9-next | Arrow view support for **scalar string predicate kernels** (LIKE/substring/prefix) — next PR (take/filter #50164 already claimed) | **scoped 2026-07-09, contribution-ready** | 3 | `AG9-next-string-predicate-kernels.md` |
| AG3 | ~~`Table.group_by` doesn't parallelize~~ **REFUTED on pyarrow 24 (scales 4.3× low-card, chunking irrelevant)**; residual high-card scaling folds into AG5'/#38372 | **resolved/closed 2026-07-09** | — | `AG3-table-groupby-parallelism.md` |
| AG5 | ~~Acero `count_distinct` slower than pandas~~ **SHELVED (Acero faster than pandas on pyarrow 24)**; NEW AG5' = Acero count_distinct ~4–5× behind Polars / doesn't parallelize | **resolved 2026-07-09; AG5' non-dup (rel. #38372)** | 4 | `AG5-count-distinct.md` |
| AG10 | **DataFusion** `single_distinct_to_groupby` fires for SQL but not the DataFrame API (~3.9× slower count-distinct) | **FILED #23401 + PR #23403 open** | **2** | `AG10-datafusion-singledistinct-dataframe.md` |
| AG11 | **DataFusion** `CrossJoinExec` mismatches schema metadata (right-biased `extend` vs left-biased `build_join_schema`; #16221 missed cross join) | **FILED #23434 + PR #23442 open** | **2** | `AG11-datafusion-crossjoin-metadata.md` |
| AG13 | **DataFusion-Python** `register_record_batches` **panics** on an empty table (unchecked `partitions[0][0]` at `context.rs:835`; reachable via `register(name, [tbl.to_batches()])` when `tbl` is empty) | **FILED #1626 + PR #1627 open** (apache/datafusion-python) | **2** | `AG13-datafusion-register-empty-panic.md` |
| AG14 | **DataFusion** inlines shared subplans (recompute; both DataFrame API + SQL CTEs) — novel **correctness** angle: a recomputed non-deterministic parallel float `SUM` under exact `==` gives wrong results (TPC-H q15 → 0 rows vs DuckDB 1; flaky, ~50% @ target_partitions=8) | **SHELVED — not actionable for us now** (flaky repro + generalization already accepted upstream #22676/#8777); **workaround kept**, comment not posted | — | `AG14-datafusion-subplan-cse.md` |
| AG15 | **NumPy** `StringDType(na_object=…)` — no coherent NA policy across the ~43 `np.strings` ufuncs (5 behaviors; **12 ops reject NA in 3 different message styles**); case/`encode` ops leak `descriptor 'upper' … doesn't apply to a 'float'` (legacy `_vec_string`) where siblings propagate. Separate: `partition`/`rpartition` unimplemented for `StringDType` | **found by the NumPy/CPython scout + exhaustively characterized on numpy 2.5.1 + non-dup — hand-off-ready** | 3 | `AG15-numpy-stringdtype-na-ufuncs.md` |
| AG16 | **NumPy** additive nan-reductions (`nansum`/`nanmean`/`nancumsum`) silently return `NaT` on `timedelta64` with a missing value, while the 7 order-based ones (`nanmin`/`max`/`median`/`percentile`/`quantile`/`argmin`/`argmax`) skip it — root cause: `_replace_nan` only masks object/inexact dtypes, passes `timedelta64` through, though `np.isnan(td)` works. Sibling of the fixed #5222 (plain min/max NaT) | **found by the NumPy/CPython scout (datetime64/NaT one-shot) + full-family matrix + root-caused on numpy 2.5.1 + non-dup — hand-off-ready** | **2** | `AG16-numpy-nan-reductions-timedelta-nat.md` |
| AG17 | ~~**DataFusion** omits `ScalarFunction.output_type`~~ | **RESOLVED-UPSTREAM / NOT FILING** — fixed on `main` by PR #20597 (merged 2026-05-27, +regression test); absent only from the 54.0.0 *release* (branch-54 cut before the merge). The probe tested the released pip pkg and wrongly concluded "regression persists on main." Live salvage → **AG20**. Durable value: forced the `main`-first gate (§2.2) | — | `AG17-...md` (post-mortem) |
| AG20 | **DataFusion** producer omits aggregate `Measure.output_type` **and** emits `phase: UNSPECIFIED` (`aggregate_function.rs`) + LIKE/ILIKE `output_type: None` (`scalar_function.rs:296,312`) — the paths #20597 **missed**. Blocks every aggregating query on Acero. **Verified on `main` HEAD (source read)** + causally proven (isolate-inject: both needed → Acero consumes) + novel-open | **the live salvage of AG17 — `main`-first-gated, causally proven on datafusion 54.0.0, hand-off-ready** | **2** | `AG20-datafusion-substrait-aggregate-output-type-phase.md` |
| AG19 | **Arrow** Substrait **consumer** has no mapping for the **standard** string functions `starts_with`/`ends_with`/`substring` → `No conversion function …` (blocks TPC-H `LIKE`/substring; unmapped even when anchored). **Kernels exist** (`pc.starts_with` etc.) — small additive registry fix; AG9/AG11 sibling-miss of closed #13285. Adjacent Acero gap: `count(distinct)` DISTINCT invocation. **Spec-split (review):** `date_part`/`regexp_like` are NOT standard Substrait (canonical `extract`/`regexp_match_substring`) → **DataFusion producer** issue, not Arrow; + DataFusion emits all fns unanchored (`urn_reference` unset) | **found by the Substrait probe + per-function coverage matrix (`substrait_fn_coverage.py`) + spec-scope-checked + attribution-tested + reproduces on pyarrow 23.0.1 AND 25.0.0 (releases) — hand-off-ready; ⚠ `main`-first gate (§2.2) owed vs Arrow C++ HEAD before filing** | 3 | `AG19-arrow-substrait-consumer-function-coverage.md` |
| AG18 | **Arrow/Acero** consumer doesn't follow the Substrait spec (spec is correct — verified vs substrait-io/substrait): (a) **silent-wrong** — `FetchRel.count` is a `oneof count_mode`; spec (PR #748 + proto docs) says check the oneof, unset = ALL. DataFusion sets `count_expr`; Acero ignores the oneof, reads deprecated `count`=0, treats as `LIMIT 0` → **every `LIMIT` query silently returns 0 rows** (TPC-H q3/q10/q18/q21); (b) no `precision_timestamp` support. Plus **function-coverage** gaps (standard `starts_with`/`ends_with`/`substring`) and cross/non-equi JoinRel rejection. All worked around in `substrait_fixup.py` (0→11/22) | **Reproduces on the pyarrow 25.0.0 *release* (not just 23.0.1); spec-checked = Arrow-consumer bug, NOT Substrait. ⚠ `main`-first gate (§2.2) still owed: verified against the released pkg, not Arrow C++ HEAD — check before filing**; documented in AG17 §residual + `../SUBSTRAIT_ROUNDTRIP.md` | 3 | (in `AG17-...md` §residual) |
| AG6 | Acero filter→reduce compaction tax | **SCOPED → don't file** (resolved multithreaded on 24.0.0) | — | `AG6-acero-filter-compaction.md` |
| AG7 | Arrow↔NumPy/pandas boundary tax | **SCOPED → not an Arrow gap** (dtype-model mismatch; clean numeric already zero-copy) | — | `AG7-boundary-tax.md` |
| AG8 | Gandiva expression-JIT unpackaged / not in Acero | **SCOPED → low value** (Acero project already removes the materialization tax; chain-JIT redundant) | — | `../MATERIALIZATION_EXPERIMENT.md` addendum + `../ARROW_GAPS.md` R2 |
| AG12 | Temporal unit downcast has no floor rounding (`*_temporal` missing `duration` kernel; `cast` truncates toward zero) | **DUP → comment on #50395** (gap #1); gap #2 no issue | 3 | `AG12-arrow-temporal-duration-floor.md` |
| E–H | pd.merge GIL; FT overhead; np.argsort | **not hand-off-ready** (characterization / needs scoping) | — | tracked in `../ARROW_GAPS.md` + `../PERF_CEILING.md` |

## Shared playbook (every hand-off assumes these)

### 1. Environment
The repo's `pandas-dev` env is pinned to **old pyarrow (23.0.1)** — do NOT
benchmark/verify upstream claims against it. Spin up a fresh, throwaway venv on
the **current** releases:
```
python -m venv /tmp/upstream-venv && . /tmp/upstream-venv/bin/activate
pip install -U pyarrow polars numpy pandas
python -c "import pyarrow, polars, numpy; print(pyarrow.__version__, polars.__version__)"
```
Run the feature's standalone benchmark here. Record the versions in the issue.

### 2. Standard gates (ALL must pass before filing)
1. **Reproduce on the latest release** (not 23.0.1) — the gap may be fixed.
2. **Confirm the defect exists on the project's `main`/HEAD, not just the released
   package** (added 2026-07-12 after AG17 — a released-artifact mirage). The
   release answers "is it *shipped*"; only `main` answers "is it *fixed*." A
   producer/optimizer/consumer defect can be fixed on `main` yet still present in
   the newest pip/conda binary if the release branch was cut before the fix
   merged. Before any "still broken / regression / incomplete-fix" claim:
   - `git grep`/read the actual source path on `main` for the field or behavior
     (e.g. GitHub raw file), and
   - `git merge-base --is-ancestor <fix-commit> <release-tag>` (or the GitHub
     `compare/<tag>...<sha>` API — `diverged`/`ahead` = fix NOT in the release).

   If a candidate fix is on `main` but not in the release you tested, the finding
   is an **unreleased fix** (zero action — self-resolves next release), NOT a
   regression. AG17 died on exactly this; AG20 is its salvage, gated correctly.
3. **Duplicate search** on the target repo: `gh api -X GET search/issues -f
   q="repo:apache/<repo> <terms>"` across several phrasings **plus** manual
   review of adjacent issues **and the closed PRs that touched the code path**.
   Record what you searched. If a dup exists, attach our benchmark as a data point
   instead of filing anew. Distinguish an *issue* number from the *PR* that closed
   it — they are one fix, not two precedents (an AG17 misread).
4. **Self-contained repro** — a script using only the target library (+ a peer
   like polars for reference), synthetic data, asserts correctness, prints
   versions. Must run on a modest machine.
5. **Human approval** — filing is outward-facing (see guardrails).

### 3. Filing conventions (apache/arrow)
- Title prefix by area: `[C++][Acero]`, `[C++][Compute]`, `[Python]`, etc.
- Labels: `Type: enhancement` / `Type: bug`, `Component: C++` / `Component: Python`.
- Tone: **question + evidence + offer to help**, not a demand. Lead with the
  enhancement framing where it's a perf gap (not a crash).
- Attach/inline the repro; include a results table and the env header.

### 4. Guardrails (non-negotiable)
- **Never open an issue or PR without explicit human go-ahead.** It is
  outward-facing, irreversible, and carries the human's GitHub identity.
- Don't over-claim internal mechanism — state measurements as fact, mechanism as
  hypothesis for maintainers to confirm.
- Keep the engine's local workaround regardless of upstream outcome.

### 5. Definition of done
gates green → draft approved by human → filed → **issue/PR number recorded back
in `../ARROW_GAPS.md`** (and here) → optional: offer a PR if the fix is in scope.

## Maintenance
New upstream item → add a row above + a hand-off doc here, and a registry row in
`../ARROW_GAPS.md`. This folder is the single source of truth for the upstream
track's execution; `../ARROW_GAPS.md` is the source of truth for the findings.
