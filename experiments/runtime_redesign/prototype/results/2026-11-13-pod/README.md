# Pod evaluation, 2026-11-13

**Superseded** by [pod evaluation 2](../2026-11-13-pod-2/README.md). This run's prototype figures include two prototype bugs, fixed in `414a04649b`: an Arrow ingress copy that kept a doubled buffer alive, and a slow map composition path. Its pandas figures and its W04 finding stand. Kept as the record behind the first version of the report.

Prototype strategies against released pandas and the deferred-take pandas PR
(pandas-dev/pandas#72023), from `benchmarks/compare_pandas.sh`.

## Setup

- RunPod CPU pod `cpu5c`: 16 vCPU AMD EPYC 4565P, 32 GB, Python 3.13.18.
- numpy 2.4.6 and pyarrow 22.0.0 in every arm.
- `pandas` arm: the pandas 3.1.1 wheel. Every other arm: a release build of the
  #72023 branch at `fc0a774af3`, which includes frame-level selection sharing.
  The shallow clone had no tags, so the build's version string was set to
  `3.2.0.dev0+248.gfc0a774` (pyarrow rejects `0+untagged`); the code is unchanged.
- Prototype at `2a0fac4884` plus the benchmark changes committed with these
  results.
- 1M rows, 4 Int64 columns plus 8 or 32 string columns (`s8.jsonl`,
  `s32.jsonl`). Each arm runs in its own process, 7 timed repeats, in two passes
  with the second in reverse order. Every arm returned identical results.

`summary.md` is the output of
`benchmarks/summarize_compare.py s8.jsonl s32.jsonl`; pass drift is the spread
between the two passes' medians. Memory figures count the Arrow pool only, so
Arrow-backed Int64 columns (the prototype's default) appear there and NumPy ones
do not.

## Findings

Speedup of `proto-compact` over `pandas-deferred` (above 1x: prototype faster):

| Workload | 8 strings | 32 strings |
|---|---|---|
| narrow | 1.23x | 1.73x |
| W01 sort, prefix | 1.36x | 1.42x |
| full | 1.07x | 1.00x |
| deep | 0.94x | 1.12x |
| W14 labeled arithmetic | 0.88x | 0.96x |
| W03 join | 0.81x | 0.78x |
| W10 append, gather | 0.76x | 0.87x |
| W04 reindex with missing labels | 4.3x | 15.5x |

1. With selection sharing, deferred pandas grows little with width (narrow: 16
   ms at 8 strings, 19 ms at 32). The prototype keeps a 1.2-1.7x lead on narrow
   reads and sort-prefix, ties on full, deep and alignment, and loses on join
   and append.
2. W04's lead comes from the PR, not the representation: reindex with missing
   labels does not defer in #72023 (its deferred arm matches eager).
3. Selection sharing in the PR (`pandas-unshared` against `pandas-deferred`):
   narrow at 32 strings 40.1 to 18.8 ms, deep 149.8 to 52.2 ms.
4. Tree against compact: tree wins W11 (long chain, scalar reads) by
   1.23-1.26x and loses `full` (0.79x, 0.86x), where its per-read cache
   recomposes the chain for each column.
5. Arrow-default Int64 against the NumPy ablation is mostly even: NumPy is
   10-20% faster where Int64 arithmetic or comparisons dominate, Arrow is faster
   on integer gathers.
6. Both deferred designs retain the whole source until it is consumed.
