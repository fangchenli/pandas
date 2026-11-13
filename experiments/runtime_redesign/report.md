# M5 decision report

Date: 2026-11-13. Status: evaluation complete for the workloads listed below. This closes M5 of the [prototype plan](prototype.md) for the measured scope.

## Decision

Do not pursue a general runtime replacement. Continue with the incremental pandas route: the deferred-take PR (pandas-dev/pandas#72023), then a follow-up that defers takes with missing rows. Keep the prototype as a semantic reference and test oracle. No native-code port is justified by these results.

The [decision criteria](experiments.md#decision-criteria) say a general runtime is justified only if its gains extend beyond the Arrow string/binary deferred take and survive realistic mutation and interoperability. Measured against #72023 with frame-level selection sharing, the prototype is ahead by 1.2–1.7x on two workloads. It is at parity or behind on six, and ahead by 4–16x on one, W04. W04's lead is a coverage gap in the PR, not a property of the representation. The mutation and interoperability conditions were not tested at scale. Because the first condition already fails, the decision does not depend on them.

## Evidence

All figures come from the [2026-11-13 pod evaluation](prototype/results/2026-11-13-pod/README.md):

- **Machine:** one 16 vCPU AMD EPYC CPU pod.
- **Data:** 1M rows, 4 Int64 columns plus 8 or 32 string columns of 32 characters.
- **Timing:** 7 timed repeats per arm, in two passes in opposite order.
- **Correctness:** every arm returned identical results.

Raw observations are in `s8.jsonl` and `s32.jsonl` next to that README.

### Time against deferred-take pandas

The comparator is #72023 at `fc0a774af3` (`pandas-deferred`). Ratios above 1x mean the prototype's compact strategy (`proto-compact`) is faster.

| Workload | What it does | 8 strings | 32 strings |
|---|---|---|---|
| narrow | filter, take, filter; read one Int64 and one string column | 1.23x | 1.73x |
| W01 | full sort on Int64, first 100 rows, read two columns | 1.36x | 1.42x |
| full | narrow's chain; read every column | 1.07x | 1.00x |
| deep | five filters keeping ~90% each; read two columns | 0.94x | 1.12x |
| W14 | take and filter of one frame, labeled addition across the two | 0.88x | 0.96x |
| W11 | eight filter-and-reverse steps; read 11 scalars from every column | 0.43x | 0.90x |
| W03 | inner join with a 500-row dimension, project three columns | 0.81x | 0.78x |
| W10 | append 16 slices one at a time, then a random take of half the rows | 0.76x | 0.87x |
| W04 | reindex to half the labels plus 100k absent ones, shuffled | 4.27x | 15.51x |

At 8 strings, pandas-deferred also beats the prototype's tree strategy on W11, 53 ms against 100 ms. The [results README](prototype/results/2026-11-13-pod/README.md) compares tree only with compact on W11, so this table adds the comparison with pandas.

### What each finding means for the design

1. **Deferral is the large effect, and pandas already has it.** Against eager execution, deferral wins 2–30x on every string-heavy workload, in both the prototype and #72023; the one exception is W04 in #72023 (finding 4). Released pandas 3.1.1 and the PR with deferral patched off agree within about 10%.
2. **Sharing one selection across columns matters, and pandas now has that too.** This is hypothesis H2. In #72023, sharing took narrow at 32 strings from 40.1 to 18.8 ms and deep from 149.8 to 52.2 ms. With sharing, the PR's narrow time grows only from 16 ms to 19 ms between 8 and 32 strings.
3. **The prototype's remaining lead is small and not attributed.** On narrow it widens with string width, 1.23x to 1.73x. Two candidate causes, neither measured:
   - pandas gathers its Int64 columns eagerly, while the prototype defers them;
   - pandas carries per-column object overhead.

   `gather_ints` takes half the rows of the four Int64 columns and reads them all: 7–8 ms in every pandas arm and in the Arrow-default prototype. So deferring numeric columns could save at most a few milliseconds per selection.
4. **W04 is a gap in #72023, not a property of the design.** Reindex passes `-1` for absent labels, and #72023 defers only takes without `-1`. Its deferred arm therefore matches the eager one (180 and 662 ms). The fix is scoped for a stacked PR once #72023 lands.
5. **The prototype's representation costs show up on joins and appends.**
   - *Joins:* the prototype builds output-length position maps for both sides.
   - *Appends:* its `Segments` reads gather twice when positions cross segments out of order.

   Each costs 13–24% against the PR's per-column deferral.
6. **Neither compact nor tree dominates.**
   - Tree beats compact on W11 by 1.23x and 1.26x, because it resolves the long chain only at the few positions read.
   - It loses on full reads (0.79x and 0.86x), where its per-read cache recomposes the chain once per column.

   Both are kept as reference strategies. Picking one is not needed for the decision above.
7. **The Arrow and NumPy defaults are roughly even.** The prototype defaults Int64 to Arrow; the NumPy-default ablation (`proto-*-numpy`) is:
   - 10–20% faster where Int64 arithmetic or comparison dominates (W04, W14 at 32 strings, deep at 32 strings);
   - slower on integer gathers (`gather_ints`: 10–13 ms against 7 ms).
8. **The eager control is a fair baseline.** The prototype's eager strategy is within 10% of pandas-eager on narrow, full, deep, W01, W04 and W14. It is faster on W10 (108 against 137 ms at 8 strings) and slower on W03 and W11 (W11: 528 against 349 ms at 8 strings). The deferred comparisons are therefore not inflated by a slimmer API, which is the risk the [baselines section](experiments.md#baselines-and-controls) names.

### Retained memory

This is Arrow memory still alive after the source frame is dropped and the result consumed. It counts the Arrow memory pool only, so pandas' NumPy-backed Int64 columns are excluded.

| Workload, 32 strings | pandas-eager | pandas-deferred | proto-compact |
|---|---|---|---|
| narrow | 153 MB | 1,187 MB | 2,008 MB |
| deep | 715 MB | 1,205 MB | 2,028 MB |
| full | 153 MB | 153 MB | 157 MB |
| W04 | 637 MB | 637 MB | 2,029 MB |
| W03 | 38 MB | 38 MB | 42 MB |

- **Deferral keeps the whole source alive** until every deferred column is read or copied. A column that is never read holds its source indefinitely. This is the main cost of both deferred designs, and it is already documented in #72023's `PendingTake`.
- **The missing-row follow-up will extend it to reindex.** W04 in pandas would move from 637 MB toward the prototype's figure.
- **The prototype's source is larger than pandas' for the same data:** 2,080 MB against 1,221 MB at 32 strings, about 64 MB against 38 MB per string column. The cause was not investigated. It affects the prototype's absolute memory figures, not the decision.

## Hypotheses

| Hypothesis | Result |
|---|---|
| H1 structural efficiency | Supported against eager execution; mostly captured by #72023's per-column deferral |
| H2 shared representations | Supported; now in #72023 as frame-level selection sharing |
| H3 native computation | Not tested beyond materialize-then-native; no bottleneck found that NumPy/PyArrow cannot handle |
| H4 mutation | Correctness covered by the test suite (snapshot isolation, copy-on-write, duplicate gathers); cost not measured at scale; retention measured above |
| H5 simplicity | About 2,200 lines of source and 287 tests for Int64, Boolean and UTF-8. The `Segments` and join map costs are where complexity showed up as time. |

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

These planned items were not run. None of them could rescue the general-runtime case, because that case already fails the first criterion.

- W02 and W05–W09 have correctness-checked runners, but only 10,000-row pilot timings, which predated later fixes and were removed. W12, W13 and W16 were not run.
- W15 (2D homogeneous arithmetic) was skipped on purpose. The plan requires it only before a general replacement claim, and this report makes none.
- M3a, the JAX adapter, was not built. Neither were the Arrow string-view treatment, the Polars and DuckDB context runs, peak RSS, or the pandas integration experiment.
- Left and outer joins were not benchmarked. W03 is an inner join. In #72023, left and outer merges are expected to share W04's gap; the follow-up will check TPC-H q13.

## When to reopen

- #72023 or its missing-row follow-up is not accepted upstream, which would leave the incremental route unavailable.
- A workload shows a frame-wide representation effect that per-column deferral cannot reach. The candidates seen here, sort-prefix and narrow reads, are worth 1.2–1.7x, which is not enough.
- Memory retention by deferred columns becomes a user-facing problem that needs ownership machinery beyond `copy()`.

## Authorship

This report is agent-generated: Claude Code, model Claude Opus 5.5. It summarizes measurements produced by the prototype's benchmark scripts. Before external use, apply the repository's [automated contributions policy](../../doc/source/development/contributing.rst).
