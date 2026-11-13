# M5 decision report

Date: 2026-11-13, revised the same day after a second pod run. Status: evaluation complete for the workloads listed below. This closes M5 of the [prototype plan](prototype.md) for the measured scope.

## Decision

Do not pursue a general runtime replacement. Continue with the incremental pandas route: the deferred-take PR (pandas-dev/pandas#72023), three per-column fixes inside it, and a follow-up that defers takes with missing rows. Keep the prototype as a semantic reference and test oracle. No native-code port is justified by these results.

The [decision criteria](experiments.md#decision-criteria) say a general runtime is justified only if its gains extend beyond the Arrow string/binary deferred take and survive realistic mutation and interoperability. Against #72023 the prototype now leads on four workloads, by 1.1–4.5x, and the lead grows with frame width. Profiling attributes the growth to three per-column costs in #72023 that its own design can remove, not to a property of the prototype's representation. A fifth workload, W04, is a coverage gap in #72023 with a fix already scoped. The prototype ties on two workloads (full, W14) and loses on two (W03, W10). The mutation and interoperability conditions were not tested at scale. Because the gains are explained within the deferred take, the decision does not depend on them.

## Revision note

The first version of this report used the [first pod run](prototype/results/2026-11-13-pod/README.md). Profiling its unexplained results found two prototype bugs, both fixed in `414a04649b`:

- the Arrow ingress copy kept a doubled IPC buffer alive, which inflated the prototype's memory by about 1.7x;
- map composition always took the slow missing-aware path, which slowed every prototype workload and produced the first run's "tree wins W11".

All figures below come from the [second pod run](prototype/results/2026-11-13-pod-2/README.md), with the fixes. The decision is unchanged; its reasons are sharper.

## Evidence

- **Machine:** one 16 vCPU pod on an AMD Ryzen Threadripper 7960X host, chosen for low load after two attempts on a loaded host were discarded.
- **Data:** 4 Int64 columns plus 8, 32 or 64 string columns of 32 characters; 1M rows, with 100k and 4M rows at fewer widths.
- **Timing:** 5–7 timed repeats per arm, in two passes in opposite order. Pass drift for the compared arms is at most 5% at 1M rows with 8 and 32 strings and at most 7% at 4M rows. It reaches 25% for some 100k-row runs, which take milliseconds.
- **Correctness:** every arm returned identical results.

### Time against deferred-take pandas

The comparator is #72023 at `fc0a774af3` (`pandas-deferred`). Ratios above 1x mean the prototype's compact strategy (`proto-compact`) is faster.

| Workload | What it does | 1M, 8 | 1M, 32 | 1M, 64 | 100k, 32 | 4M, 8 |
|---|---|---|---|---|---|---|
| narrow | filter, take, filter; read one Int64 and one string column | 1.09x | 1.28x | 2.20x | 2.04x | 1.23x |
| deep | five filters keeping ~90% each; read two columns | 1.32x | 1.54x | 1.96x | 1.24x | 1.38x |
| W11 | eight filter-and-reverse steps; read 11 scalars from every column | 1.33x | 2.83x | 4.45x | 2.62x | 1.25x |
| W01 | full sort on Int64, first 100 rows, read two columns | 1.32x | 1.44x | 1.57x | 1.12x | 1.49x |
| full | narrow's chain; read every column | 1.00x | 1.00x | 1.02x | 1.08x | 1.03x |
| W14 | take and filter of one frame, labeled addition across the two | 1.01x | 0.97x | 0.96x | 0.92x | 0.96x |
| W10 | append 16 slices one at a time, then a random take of half the rows | 0.76x | 0.90x | 0.98x | 1.10x | 0.87x |
| W03 | inner join with a 500-row dimension, project three columns | 0.55x | 0.62x | 0.72x | 0.81x | 0.80x |
| W04 | reindex to half the labels plus 100k absent ones, shuffled | 4.38x | 16.51x | 34.75x | 13.15x | 4.68x |

### What each finding means for the design

1. **Deferral is the large effect, and pandas already has it.** Against eager execution, deferral wins 2–50x on every string-heavy workload, in both the prototype and #72023; the one exception is W04 in #72023 (finding 4).
2. **The prototype's lead is per-column overhead in #72023, and #72023 can remove it.** #72023's time on narrow, deep and W11 grows with the number of string columns, while the prototype's stays flat (W11: 69, 163 and 280 ms at 8, 32 and 64 strings, against 52, 58 and 63 ms). Profiling the #72023 build on a laptop finds three causes:
   - **Bounds checks per column.** `ArrowExtensionArray.take` runs `min()` and `max()` on the shared indexer for every column at every step: on deep, 40 calls per run at 8 strings and 160 at 32, with reduction time rising from 14 to 32 ms. The frame already checks the indexer once when it shares it.
   - **Sharing lost after a slice.** After `iloc[::-1]`, each of 32 columns composes the next selection on its own, where without the slice one column composes and 31 reuse it. Sharing therefore does nothing on W11: the `pandas-unshared` arm takes 161.8 ms against 163.4 ms.
   - **Eager Int64 gathers.** #72023 defers strings only; its masked Int64 gathers cost about 7–19 ms per run, independent of string width.

   W01's lead (1.1–1.6x) is not attributed; its sort and eager Int64 gathers are candidates.
3. **Sharing one selection across columns matters, and pandas has it.** This is hypothesis H2. In #72023, sharing took narrow at 32 strings from 64.1 to 22.8 ms and deep from 202.8 to 62.4 ms. Finding 2 shows where it still falls short.
4. **W04 is a gap in #72023, not a property of the design.** Reindex passes `-1` for absent labels, and #72023 defers only takes without `-1`, so its deferred arm matches the eager one and grows with width (34.8x behind at 64 strings). A left merge has the same gap: on the #72023 build, measured locally, it reuses the left columns without copying but gathers the right side's strings eagerly. The fix is scoped for a stacked PR once #72023 lands.
5. **The prototype loses on joins and appends.** W03 runs 0.55–0.81x and W10 0.76–0.98x at 1M and 4M rows. With NumPy Int64, W03 ties #72023 at 32 strings (26.3 against 25.4 ms) but not at 8 (40.8 against 23.2 ms). Neither loss is attributed.
6. **Compact dominates tree.** Compact is as fast or faster at 1M and 4M rows on every workload: 1.48–1.70x on W11 and 1.12–1.23x on full, where tree's per-read cache recomposes the chain once per column. Tree is faster only at 100k rows with 32 strings on W01, W03 and W10 (1.2–1.35x, runs of 4–11 ms), which did not reproduce at other sizes. Tree has no measured niche.
7. **NumPy Int64 is the better prototype default for time.** The NumPy-default ablation (`proto-*-numpy`) takes 12–36% less time on W11, and at 32 strings on narrow, deep, W14 and W03. Arrow is faster only on integer gathers (`gather_ints`: 8.7 against 20.1 ms at 8 strings).
8. **The eager control is a fair baseline.** The prototype's eager strategy is within 6% of pandas-eager on narrow, full, deep, W01, W04 and W14. It is faster on W10 (115 against 150 ms at 8 strings) and slower on W03 (69 against 52 ms) and W11 (637 against 402 ms). The deferred comparisons are therefore not inflated by a slimmer API, which is the risk the [baselines section](experiments.md#baselines-and-controls) names.

### Retained memory

This is Arrow memory still alive after the source frame is dropped and the result consumed. It counts the Arrow memory pool only, so NumPy-backed Int64 columns are excluded. `proto-compact-numpy` keeps Int64 in NumPy, as pandas does, so it is the like-for-like column. The figures are deterministic: a local re-measurement matches the pod to the decimal.

| Workload, 32 strings | pandas-eager | pandas-deferred | proto-compact-numpy | proto-compact |
|---|---|---|---|---|
| narrow | 153 MB | 1,187 MB | 1,187 MB | 1,206 MB |
| deep | 715 MB | 1,205 MB | 1,205 MB | 1,227 MB |
| full | 153 MB | 153 MB | 153 MB | 157 MB |
| W04 | 637 MB | 637 MB | 1,203 MB | 1,227 MB |
| W03 | 38 MB | 38 MB | 38 MB | 42 MB |

- **The two deferred designs retain the same memory** on every workload except W04, where #72023 does not defer. The Arrow-default prototype adds its 30.5 MB of Arrow Int64.
- **Deferral keeps the whole source alive** until every deferred column is read or copied. A column that is never read holds its source indefinitely. This is the main cost of both designs, and it is already documented in #72023's `PendingTake`.
- **The missing-row follow-up will extend it to reindex.** W04 in pandas would move from 637 MB to about 1,203 MB.

## Hypotheses

| Hypothesis | Result |
|---|---|
| H1 structural efficiency | Supported against eager execution; captured by #72023's per-column deferral, apart from the per-column costs in finding 2 |
| H2 shared representations | Supported; in #72023 as frame-level selection sharing, which a slice currently breaks |
| H3 native computation | Not tested beyond materialize-then-native; no bottleneck found that NumPy/PyArrow cannot handle |
| H4 mutation | Correctness covered by the test suite (snapshot isolation, copy-on-write, duplicate gathers); cost not measured at scale; retention measured above |
| H5 simplicity | About 2,200 lines of source and 295 tests for Int64, Boolean and UTF-8. The `Segments` and join map costs are where complexity showed up as time, and the two bugs above were in the map and ingress layers. |

## Expressiveness

Expressiveness did not limit the prototype: every measured workload fit its representation. The costs were in time, from dense maps and second gathers through `Segments`.

In compact mode a column is a buffer (`Physical`), a `Constant`, a `Segments` laid end to end, or one `Select` of one of those by a row map. A row map is `Identity`, `Affine` or dense `Positions` with an optional mask for missing rows. Tree mode expresses the same set and differs only in when maps compose.

`Select` over `Segments` picks each output row from any of several same-typed sources, a one-row `Constant` included, or leaves it missing. `test_concat_then_take_picks_rows_across_sources` checks three such picks under every strategy and both backends: `where(cond, a, b)`, a reindex with a non-NA fill, and a sparse write as a patch. By the same construction, without new node types, the representation can express the following; only the three cases above are tested:

- row selection: take, filter, slice, sort, head, sample, deduplication;
- missing rows: reindex, shift, unstack and pivot, both sides of left and outer joins;
- repeats: many-to-many and cross joins, `repeat`;
- appends: concat;
- row-wise choice between sources: `where`, `mask`, `combine_first`, `fillna` with another column, `update`, the coalesced key of an outer join;
- melt and stack, as columns laid end to end and then reordered;
- categoricals, as a `Select` of the categories by the codes.

It cannot express:

1. computed values (arithmetic, casts, string operations), which always produce a buffer, by design;
2. one column over sources of different types or backends, since `Segments` requires them equal;
3. operations across columns or in 2D: transpose, `axis=1` reductions, a homogeneous block for `to_numpy` (W15);
4. nested types, such as `explode` or list and struct columns;
5. deferred or hierarchical axes: labels are flat and eager, with no MultiIndex.

Some row selections it can express cost more than they need to. A dense position array stores a filter keeping 999,000 of 1M rows in 8 MB, where #72023 keeps a 125 KB mask. Repeats and sorted runs are not compressed, and a patch-style write needs an O(n) map per write. The prototype implements none of these: a write copies the whole column.

For pandas, the multi-source pick is a design target for #72023's `PendingTake`, which holds one parent and one selection. The missing-row follow-up covers reindex and left joins. A selection over several parents would add `where` and `combine_first`, and cheap sparse writes, without other new machinery.

## Not measured

These planned items were not run. None of them could rescue the general-runtime case, because the measured gains are already explained within the deferred take.

- W02 and W05–W09 have correctness-checked runners, but only 10,000-row pilot timings, which predated later fixes and were removed. W12, W13 and W16 were not run.
- W15 (2D homogeneous arithmetic) was skipped on purpose. The plan requires it only before a general replacement claim, and this report makes none.
- M3a, the JAX adapter, was not built. Neither were the Arrow string-view treatment, the Polars and DuckDB context runs, peak RSS, or the pandas integration experiment.
- Left and outer joins were checked for deferral on a laptop (finding 4) but not benchmarked on the pod. The follow-up will check TPC-H q13.

## When to reopen

- #72023 or its missing-row follow-up is not accepted upstream, which would leave the incremental route unavailable.
- #72023 gains the three per-column fixes in finding 2 and the prototype still leads by more than about 1.5x on narrow, deep or W11 at 32 or more string columns. The remaining lead would then be a representation effect.
- Memory retention by deferred columns becomes a user-facing problem that needs ownership machinery beyond `copy()`.

## Authorship

This report is agent-generated: Claude Code, model Claude Opus 5.5. It summarizes measurements produced by the prototype's benchmark scripts. Before external use, apply the repository's [automated contributions policy](../../doc/source/development/contributing.rst).
