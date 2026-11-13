# Pandas runtime redesign research

Research date: 2026-11-12. Status: M0–M4 Python prototype implemented and evaluated against deferred-take pandas (2026-11-13). The [M5 decision report](report.md) concludes against a general runtime replacement and for the incremental deferred-take route.

The proposal merits a bounded experiment. Its strongest opportunity is reducing movement of heterogeneous payload columns across structural operations while preserving pandas behavior. The first decision is whether that benefit survives materialization, mutation, and interoperability costs, and exceeds a targeted deferred-take implementation.

## Contents

- [Original proposal](proposal.md): the user-supplied proposal, preserved separately from the review.
- [Design review](review.md): assessment, semantic gaps, recommended changes, and a proposed initial architecture.
- [Annotated references](references.md): primary sources, reading priorities, applicability, and limitations.
- [Experimental plan](experiments.md): correctness cases, controlled baselines, workloads, instrumentation, and decision criteria.
- [M5 decision report](report.md): measured findings, retained memory, hypotheses, what was not measured, and when to reopen.
- [Python prototype design](prototype.md): API, shared ownership model, eager/compact/tree strategies, backend contracts, an early JAX validation milestone, and implementation milestones.
- [Unified logical types](logical_types.md): verified PDEP-13 comparison, type identity, operation/result-type rules, and backend-independent contracts.

Start with the review, then the prototype design. The [runnable prototype](prototype/README.md) implements eager, compact, and tree strategies with NumPy/Arrow adapters. The [remaining decisions](decisions.md) record key semantics, custom-type boundaries, and follow-up experiments. The experimental plan describes the broader evaluation; the initial workload smoke checks do not establish a speedup.

The [PDEP-16 review and missing-value contract](missing_values.md) adds operation-specific null rules, float-policy evidence, and 2D workload controls. It distinguishes the open draft from existing pandas behavior.

## Repository provenance

| Item | Pinned revision |
| --- | --- |
| Research branch | `investigate-pandas-runtime` |
| Local `main` and initial research checkout | `f1cc1a1bc979b83e641facf72ec6526593d61420` |
| Experimental `perf-arrow-deferred-take` inspected | `629c2c8f89b6b3d6f15fa4176b4d765fb6a531f2` |
| Merge base of those revisions | `f1cc1a1bc979b83e641facf72ec6526593d61420` |

The experimental branch contains unrelated changes as well as deferred take. Its complete diff is not an isolated performance treatment. The original checkout's untracked experiments were not imported into this worktree.

Local source paths in these notes refer to the pinned main revision unless another revision is specified. References to experimental files can be read with `git show <revision>:<path>`. Web sources were consulted on the research date; live documentation may change. Search-engine crawl dates are not publication dates. Bibliographic links and annotations are stored here rather than copies of entire external websites.

## Evidence boundaries

The review combines local source inspection, primary documentation, and research publications. It makes architectural recommendations, not measured performance claims. The bibliography covers the proposal's named systems and closely related materialization, encoding, and interoperability work; it is not an exhaustive literature survey.

PDEP-13 has now been verified from the [PR supplied by the user](https://github.com/pandas-dev/pandas/pull/58455). Will Ayd's draft was closed unmerged on 2025-04-21; it is a research reference, not an accepted pandas specification. The [logical type design](logical_types.md) records the pinned source and distinguishes its proposals from our implementation choices.

## Authorship

`proposal.md` is user-supplied text. Everything else is agent-generated, by two harnesses:

- Codex harness, GPT-6 family (exact model variant and reasoning effort not exposed): the research documents (`review.md`, `references.md`, `experiments.md`, `prototype.md`, `logical_types.md`, `missing_values.md`, `decisions.md`) and the first M0–M3 prototype (`98269289d6`).
- Claude Code, Claude Opus 5.5: all later prototype code, tests and benchmarks (from `67a7f1dedd`: review fixes, type-layer cleanups, M4, the Arrow default, joins, the pandas comparison), the pod results, `report.md`, and later edits to `prototype.md`, `logical_types.md`, `experiments.md` and this README.

Each commit's `Co-Authored-By` trailer identifies Claude's commits; Codex commits carry none. Before external contribution, apply the repository's [automated contributions policy](../../doc/source/development/contributing.rst): human review and revision, agent disclosure, and explicit quotation of any verbatim agent text used in community discussion.
