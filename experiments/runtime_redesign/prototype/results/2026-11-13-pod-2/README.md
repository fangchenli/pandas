# Pod evaluation 2, 2026-11-13

A rerun of the [first pod evaluation](../2026-11-13-pod/README.md) after two prototype fixes found by profiling (`414a04649b`), with more widths and sizes. It supersedes the first run's prototype figures.

## What changed since the first run

- **Arrow ingress copy:** the prototype's ownership copy wrote through a growing IPC buffer, and the copied arrays kept its doubled capacity alive (64 MB held for 38 MB of strings). It now allocates exactly.
- **Map composition:** composing two maps without missing rows took the general missing-aware path, about five passes over the map instead of one gather. A slice now composes as a strided view. On a laptop this took W11 (compact) from 343 to 61 ms.
- **Empty descending slice:** a bug in the new strided-view path, found by Hypothesis on the first attempt of this run and fixed in `5aa9cf6423` before the measurements.

## Setup

- RunPod CPU pod `cpu5c` in EU-RO-1: 16 vCPU on an AMD Ryzen Threadripper 7960X host, 32 GB, Python 3.13.18. The first run used an AMD EPYC 4565P host, so absolute times differ between the two runs; compare arms within a run.
- Two earlier attempts in EUR-IS-1 landed on a loaded EPYC 9575F host (load average 27–60, pass drift up to 48%). They were discarded and are not reported.
- numpy 2.4.6 and pyarrow 22.0.0 in every arm. `pandas`: the 3.1.1 wheel. Every other pandas arm: a release build of #72023 at `fc0a774af3`, version string patched to `3.2.0.dev0+248.gfc0a774` because the shallow clone had no tags.
- Prototype at `5aa9cf6423`. All 295 prototype tests passed on the pod. Every arm returned identical results in every file.

| File | Rows | String columns | Arms | Repeats |
|---|---|---|---|---|
| `r1m_s8`, `r1m_s32` | 1M | 8, 32 | all 10 | 7 × 2 passes |
| `r1m_s64` | 1M | 64 | pandas-eager, pandas-deferred, proto-compact, proto-tree | 5 × 2 passes |
| `r100k_s8`, `r100k_s32` | 100k | 8, 32 | the same four | 5 × 2 passes |
| `r4m_s8` | 4M | 8 | the same four | 5 × 2 passes |

Each `.md` file is the output of `benchmarks/summarize_compare.py` on the `.jsonl` file of the same name. Pass drift for pandas-deferred, proto-compact and proto-tree is at most 5% at 1M rows with 8 and 32 strings, and at most 7% at 4M rows. It reaches 23% for a few proto-tree rows at 64 strings and 25% for some 100k-row rows, where runs take 1–20 ms. The eager arms drift the most.

## Findings

Speedup of proto-compact over pandas-deferred (above 1x: prototype faster):

| Workload | 100k, 8 | 100k, 32 | 1M, 8 | 1M, 32 | 1M, 64 | 4M, 8 |
|---|---|---|---|---|---|---|
| narrow | 1.62x | 2.04x | 1.09x | 1.28x | 2.20x | 1.23x |
| deep | 1.42x | 1.24x | 1.32x | 1.54x | 1.96x | 1.38x |
| W11 chain and scalars | 1.53x | 2.62x | 1.33x | 2.83x | 4.45x | 1.25x |
| W01 sort, prefix | 1.26x | 1.12x | 1.32x | 1.44x | 1.57x | 1.49x |
| full | 1.18x | 1.08x | 1.00x | 1.00x | 1.02x | 1.03x |
| W14 labeled arithmetic | 0.99x | 0.92x | 1.01x | 0.97x | 0.96x | 0.96x |
| W10 append, gather | 1.04x | 1.10x | 0.76x | 0.90x | 0.98x | 0.87x |
| W03 join | 0.74x | 0.81x | 0.55x | 0.62x | 0.72x | 0.80x |
| W04 reindex with missing labels | 3.92x | 13.15x | 4.38x | 16.51x | 34.75x | 4.68x |
| gather_ints | 1.60x | 1.56x | 1.22x | 1.21x | 1.21x | 1.18x |

1. The prototype's lead on narrow, deep and W11 grows with the number of string columns because #72023's time does (W11: 69, 163 and 280 ms at 8, 32 and 64 strings, against 52, 58 and 63 ms for compact). Profiling the #72023 build locally attributes this to three per-column costs:
   - `ArrowExtensionArray.take` checks `min()` and `max()` of the shared indexer for every column at every step (deep: 40 calls per run at 8 strings, 160 at 32);
   - selection sharing does not survive a slice: after `iloc[::-1]`, every column composes the next selection on its own (32 compositions against 1 without the slice), and the `pandas-unshared` arm matches `pandas-deferred` on W11 (161.8 and 163.4 ms at 32 strings);
   - Int64 columns are gathered eagerly (about 7–19 ms per run on the laptop, independent of the string width).
2. W04's lead is #72023's missing-row gap and grows linearly with width (34.8x at 64 strings).
3. The prototype loses on W03 (0.55–0.81x) and on W10 at 1M and 4M rows (0.76–0.98x). With NumPy Int64 (`proto-compact-numpy`), W03 at 32 strings ties #72023 (26.3 against 25.4 ms) but at 8 strings does not (40.8 against 23.2 ms). Neither loss is attributed.
4. Compact is as fast as tree or faster everywhere at 1M and 4M rows. Tree loses W11 (compact is 1.48–1.70x faster) and full (1.12–1.23x). The first run's "tree wins W11" came from the composition bug. Tree is faster only at 100k rows with 32 strings on W01, W03 and W10 (1.2–1.35x), where runs take 4–11 ms; this did not reproduce at other sizes.
5. Selection sharing in #72023 (`pandas-unshared` against `pandas-deferred`, 32 strings): narrow 64.1 to 22.8 ms, deep 202.8 to 62.4 ms, W11 no change (finding 1).
6. The NumPy-Int64 ablation takes 12–36% less time than the Arrow default on W11, and at 32 strings on narrow, deep, W14 and W03. Arrow is faster on `gather_ints` (8.7 against 20.1 ms at 8 strings, 8.8 against 12.5 at 32).
7. The eager control: proto-eager is within 6% of pandas-eager on narrow, full, deep, W01, W04 and W14, faster on W10 (115 against 150 ms at 8 strings), and slower on W03 (69 against 52 ms) and W11 (637 against 402 ms).
8. Retained Arrow memory is deterministic and matches a local re-measurement. With NumPy Int64, the prototype retains the same Arrow memory as #72023 on every workload except W04, where #72023 does not defer. The Arrow-default prototype adds its 30.5 MB of Arrow Int64.
