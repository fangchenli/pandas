# Upstream contributions — backlog & hand-off playbook

This folder is the home for upstream contributions surfaced by the lazy-pandas
**substrate probe** (see `../PROBE_CHARTER.md`; gap registry in
`../ARROW_GAPS.md`). Each actionable feature has a **self-contained hand-off
doc** below that a fresh agent can execute end-to-end without prior context.

**Read this playbook first**, then open the hand-off for your feature.

## Backlog index

| Feature | Gap | Status | Priority | Hand-off |
|---|---|---|---|---|
| AG4 | Acero string-key hash-agg 3.5–10× slower than dict-encoded | verified, draft ready, non-dup | **1** | `AG4-acero-string-key-hashing.md` |
| AG1 | `take` segfaults on >2 GB int32-offset string | repro pending | 2 | `AG1-take-segfault.md` |
| AG2 | `hash_aggregate` SIGABRT on >2 GB string keys | repro pending | 2 | `AG2-hashagg-abort.md` |
| AG9 | Arrow `string_view` kernel coverage (contribution) | scoped (PR) | 3 | `AG9-string-view-kernels.md` |
| AG3 | `Table.group_by` parallelism (likely footgun) | verify-first | 4 | `AG3-table-groupby-parallelism.md` |
| AG5 | Acero `count_distinct` slower than pandas Cython | needs bench | 5 | `AG5-count-distinct.md` |
| AG6/7/8, E–H | Acero per-node overhead; Arrow↔NumPy boundary tax; Gandiva availability; pd.merge GIL; FT overhead; fused-expr; np.argsort | **not hand-off-ready** (characterization / needs scoping) | — | tracked in `../ARROW_GAPS.md` + `../PERF_CEILING.md` |

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
2. **Duplicate search** on `apache/arrow`: `gh api -X GET search/issues -f
   q="repo:apache/arrow <terms>"` across several phrasings **plus** manual
   review of adjacent issues. Record what you searched. If a dup exists, attach
   our benchmark as a data point instead of filing anew.
3. **Self-contained repro** — a script using only the target library (+ a peer
   like polars for reference), synthetic data, asserts correctness, prints
   versions. Must run on a modest machine.
4. **Human approval** — filing is outward-facing (see guardrails).

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
