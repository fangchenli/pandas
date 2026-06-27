# Arrow substrate gaps — consolidated registry

Canonical list of Arrow/Acero gaps the probe (`PROBE_CHARTER.md`) surfaced,
gathered from the docs where they were found. This is the **Arrow contribution
backlog**: each gap has a claim, the measured evidence + source doc, an honest
status, the upstream framing, and ecosystem leverage. Crash-class defects keep
their detail in `upstream/UPSTREAM_ISSUES.md`; this registry links them, doesn't dupe.

Status legend: **filed-track** (defect, repro pending) · **planned** (PR
intended) · **needs-verification** (claim not yet airtight) · **quantified /
worked-around** (real, we route around it) · **observed**.

| # | Gap | Status | Leverage | Source |
|---|---|---|---|---|
| AG1 | `take` **segfaults** on >2 GB int32-offset string data (should raise `ArrowInvalid`) | **dup → comment on #25822** | high | upstream/UPSTREAM_ISSUES.md #1 |
| AG2 | acero `hash_aggregate` **aborts (SIGABRT, uncatchable)** on >2 GB string keys | non-dup, repro pending | high | upstream/UPSTREAM_ISSUES.md #2 |
| AG3 | `Table.group_by()` on a **single Table doesn't parallelize** (no scaling 1→8 cores) | **needs-verification** | high (if real) | PARALLEL_GROUPBY_SCOPE.md |
| AG4 | acero **raw-string key hashing 3.5–10x slower than dict-encoding** the same keys (and ~3.8x slower than Polars per-thread) | **VERIFIED (standalone)** | med-high | ENGINE_DESIGN.md M4; `benchmarks/bench_arrow_string_groupby.py` |
| AG5 | acero `count_distinct` **slower than pandas' Cython** on high-cardinality | quantified / worked-around | med | QGAP_DECOMP.md |
| AG6 | Acero **per-node overhead** — filter→reduce catastrophic, per-join-node overhead at filtered scale | quantified / worked-around | med | MATERIALIZATION_EXPERIMENT.md I & III |
| AG7 | Acero join/agg wins **die at the Arrow→NumPy round-trip** (boundary tax) | quantified (structural) | high | MATERIALIZATION_EXPERIMENT.md II; PERF_CEILING.md |
| AG8 | **Gandiva**: optional/often-unshipped pyarrow + **not wired into Acero** (expression codegen unavailable in practice) | observed | low-med | MATERIALIZATION_EXPERIMENT.md F4 (corrected below) |
| AG9 | `string_view` **kernel coverage** (at-scale/string gains) | **in-flight upstream (#44336)** | high | upstream/STRING_VIEW_CONTRIBUTION_PLAN.md |

## Upstream duplicate-search results (apache/arrow, 2026-06-18)

Searched issues **and** PRs across many phrasings (`gh api search/issues`):

- **AG9 — already actively in-flight; do NOT file.** Umbrella **#44336 "[C++]
  Binary View Compute Kernels" (open)** with open PRs **#50164** (view arrays in
  selection kernels), **#50166** (cast-to-view null buffers), **#48734**
  (eliminate Array boxing in scalar string kernels) and open sub-issues
  **#43010** (STRING_VIEW/BINARY_VIEW in array_take/array_filter), **#46128**
  (cast-to-StringView memory). Contribution = **join this effort** (re-scope to
  an unaddressed kernel, engage on #44336), not a new issue.
- **AG1 — effectively a duplicate.** **#25822 (open)** "[C++] Take kernel can't
  handle ChunkedArrays that don't fit in an Array" is the same root limitation
  (Take concatenates first, breaks past 2 GB). Our angle (it **segfaults**
  rather than raising) is best added as a **comment + repro on #25822**, not a
  new issue. Related open: #33049/#41890/#44164 (concat offset overflow),
  #46814 (large-data segfault).
- **AG2 — non-duplicate.** No existing report of the acero `hash_aggregate`
  SIGABRT on >2 GB string keys.
- **AG4 — non-duplicate** (issues + PRs). Only DataFusion #27498 (closed, Rust,
  different angle).
- **AG3 / AG5 / AG8 — no specific existing issue** (none is file-ready anyway).

## Two cross-doc contradictions, resolved

### R1 — Is Arrow's hash aggregate single-threaded? (AG3 vs AG4/M4)

`PARALLEL_GROUPBY_SCOPE.md` measured **`Table.group_by()` not scaling** with
`cpu_count` (124→133 ms, 2-key, ~1.6M groups, single in-memory Table) and a
partition-parallel wrapper beating it ~2.5x. But `ENGINE_DESIGN.md` M4 states
**"acero's hash aggregation is already internally multi-threaded"** (and
measured acero aggregating *dictionary* keys at 13 ms @10M, beating Polars'
18 ms). These are not actually contradictory — they reconcile via **how the
input reaches the kernel**:

- Acero parallelizes across **morsels/batches**. A **single `Table`** handed to
  the `group_by` convenience is effectively one unit → single-threaded (AG3).
- A proper **streaming Acero plan over many batches** engages acero's internal
  threading (M4).

**So AG3 is most likely the single-`Table`-convenience footgun, not a missing
capability** — and the partition-parallel win is essentially "manually split the
Table into morsels so acero's existing parallelism engages." This is precisely
`upstream/UPSTREAM_PARALLEL_GROUPBY_PLAN.md` **gate 1**, and M4 is strong internal
counter-evidence that the broad claim ("Arrow can't parallelize group-by") is
**wrong**. Open question gate 1 must still settle: does multi-batch streaming
acero parallelize a **high-cardinality** (~1.6M-group) aggregate, or does the
shared hash table contend there even when low-cardinality dict-key aggregates
parallelize cleanly? Until tested, AG3 stays **needs-verification** and must be
framed as the footgun, not a capability gap.

**Further weakening (2026-06, from the AG4 benchmark):** `Table.group_by()`
*did* scale with `set_cpu_count` in that benchmark — acero_dict at K=100 went
44.7 → 10.9 ms going 1→8 cores (~4x), and raw-string K=100 96→392... i.e. the
convenience API is **not** unconditionally single-threaded. The original
PARALLEL_GROUPBY_SCOPE no-scaling result (124→133 ms) was a 2-key high-
cardinality *numeric* case; whether that specific shape fails to parallelize
(vs. the string/dict cases that clearly do) is now the open question. Net: AG3
is **weaker than first stated** — likely shape-specific, not a blanket gap.

### R2 — Gandiva is NOT "moribund" (AG8 correction)

`MATERIALIZATION_EXPERIMENT.md` Finding 4 calls Gandiva "moribund." That is
**stale/incorrect** (corrected during the engine-differentiation work): Gandiva
is **actively developed** (commits through 2026). The accurate gap is narrower:
Gandiva is an **expression-only** LLVM compiler, **frequently not shipped in
pyarrow wheels** (`import pyarrow.gandiva` → `ModuleNotFoundError` on the conda
build), and **not wired into Acero** (so it can't fuse joins/aggregates or be
used from a normal pyarrow pipeline). See `ENGINE_DIFFERENTIATION.md`. Treat AG8
as "expression codegen exists but is practically unavailable + non-composable,"
not "abandoned."

**Refresh (2026-06-26) — and AG8 is now LOW value.** `import pyarrow.gandiva`
still `ModuleNotFoundError` on **both 23.0.1 and 24.0.0** wheels (unchanged).
But a measured re-test of the underlying question — *"JIT-compile a pc chain to
remove intermediate materialization?"* (`benchmarks/arrow_fusion_probe.py`,
`MATERIALIZATION_EXPERIMENT.md` addendum) — shows **Acero's `project` node
already removes ~4.4x of the materialization tax on a deep compute chain without
any JIT** (197→45 ms, within ~12% of NumExpr), because morsel-streaming keeps
intermediates cache-resident. So chain-JIT/Gandiva would add only the last ~12%
on compute-bound chains and ~nothing on bandwidth-bound ones. **Do not pursue a
chain-JIT contribution.** The real remaining materialization gap is **filter→
reduce compaction** in Acero's `FilterNode` (gathers selected rows before the
aggregate; `filt6` 70 ms vs raw_pc 5.6 ms) — fix is mask/selection-vector
push-down to the aggregate, not expression JIT. That is the file-able Acero
enhancement worth scoping (currently characterization-stage, not hand-off-ready).

## Detail for the non-obvious gaps

- **AG4 (string-key hashing). VERIFIED** by a standalone benchmark
  (`benchmarks/bench_arrow_string_groupby.py`, 10M rows, pyarrow 23.0.1 /
  polars 1.37.1, controlled cardinality × thread count):
  - **raw `large_string` keys are 3.5–10x slower than dict-encoding the same
    keys** (raw/dict: K=100 8.5x, K=10K 9.5–9.8x, K=1M 3.5–6x) — robust across
    cardinality and thread count. Correctness identical across methods.
  - Per-thread (1 core) acero-raw is **3.8x slower than Polars** at K=100
    (388 vs 102 ms) — reproduces the original ~3.7x claim.
  - **Dict-encoding makes acero beat Polars** at K=100/10K; **honest residual:**
    at K=1M Polars' high-card hash still edges acero_dict (342 vs 446 ms multi-
    thread), so dict-encoding doesn't fully close the gap at extreme cardinality.
  - **Mechanism hypothesis:** acero re-hashes raw key bytes per row rather than
    interning; dict-encoding eliminates the repeated hashing. The clean,
    file-able framing: *acero's hash-aggregate should dictionary-encode (intern)
    string keys internally* — Polars effectively does. Best upstream candidate:
    well-measured, workaround-validated, reproducible, no morsel-nuance confound.
  - **Duplicate search done (apache/arrow, 2026-06-18):** ~12 query phrasings +
    manual review of adjacent Grouper issues — **no existing report.** Closest
    is DataFusion #27498 (Rust, different angle). → **non-duplicate, file-able.**
    Issue draft in `upstream/UPSTREAM_AG4_STRING_HASH.md`. Not yet filed.
  - **Latest-Arrow gate GREEN (2026-06-26):** re-ran on **pyarrow 24.0.0 /
    polars 1.42.0** (fresh venv) — gap unchanged from 23.0.1 (1-core raw/dict
    8.4x/9.5x/4.0x at K=100/10K/1M; 8-core 7.7x/.../5.3x). Not fixed in two
    releases. Draft refreshed with the 24.0.0 numbers. **Only remaining gate:
    human approval to file.**
- **AG6 (Acero per-node overhead).** filter→reduce in Acero is *catastrophic*
  (`filt6` 70 ms/1-thread vs raw `pc` 5.6 ms and our fused kernel 4.3 ms);
  per-join-node overhead made acero end-to-end 2x slower than our pd.merge on
  real q3. Acero helps map-only projection but loses on filtered/reduced
  pipelines — a "when does Acero pay" characterization, not a single bug.
  **SCOPED + CLOSED (2026-06-26, `upstream/AG6-acero-filter-compaction.md`):** the
  filter→reduce compaction tax is **largely gone on pyarrow 24.0.0 at the default
  multithreaded path** — `filt6` acero_8 is now **1.1×** raw_pc (was 2.9×) and
  simple filter→sum acero_8 *beats* the no-gather masked reduce (0.3×). A
  single-threaded residual remains (2–5×, cost scales with selectivity = the
  compaction signature) but single-thread isn't realistic Acero usage. **Verdict:
  LOW value, do NOT file.** Keep the engine's `fused_filter_aggs` kernel anyway.
- **AG7 (boundary tax).** Acero's hash join is fastest in isolation (229 ms,
  3.3x over Polars) but the `to_numpy()` round-trip on a ~100M-row output costs
  ~435 ms → acero+round-trip ≈ pd.merge. Acero only wins if the pipeline **stays
  Arrow across the operator**. Structural Arrow↔NumPy boundary cost; motivates
  zero-copy handoff / arrow-native execution rather than a single kernel fix.
  **SCOPED + CLOSED (2026-06-26, `upstream/AG7-boundary-tax.md`): NOT an Arrow
  gap.** Decomposed on pyarrow 24.0.0: clean numeric Arrow→NumPy is **zero-copy
  (0 ms)**; all the cost is NumPy/pandas **dtype-model mismatch** — strings
  (159–212 ms = CPython `str` object creation, not Arrow), nulls (no NumPy mask),
  BlockManager consolidation (a pandas cost). The zero-copy escape exists
  (`to_pandas(types_mapper=pd.ArrowDtype)` = **0.1 ms**) but only *defers* the tax
  to the next NumPy consumer. **Do NOT file against Arrow** — Arrow's conversions
  are already zero-copy where dtype models match. Engine-side ArrowDtype output is
  the only lever, and whole-pipeline-Arrow is the already-NO-GO rewrite.

## Upstream priority (Arrow-specific)

1. **Moving / high-leverage:** AG9 `string_view` PR; AG1/AG2 crash reports
   (repro scripts pending, per upstream/UPSTREAM_ISSUES.md).
2. **Cleanest new candidate:** **AG4** (string-key hash kernel quality) — well-
   measured, workaround-validated, no morsel confound.
3. **Verify-then-decide:** **AG3** (parallel group-by) — run gate 1 first; likely
   downgrades to the single-`Table` footgun report.
4. **Document-as-characterization:** AG6 (when Acero pays), AG7 (boundary tax) —
   issues/benchmarks, lower urgency.
5. **Note, not file yet:** AG5 (count_distinct), AG8 (Gandiva availability).

## Maintenance

When a new Arrow gap is found, add a row here (not buried in a session doc), and
when one is filed/fixed upstream, record the issue/PR number. This registry is
the single source of truth for the Arrow track; `PROBE_CHARTER.md`'s inventory
links here for the Arrow subset.
