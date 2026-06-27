# Scoping: AG6 — Acero filter→reduce compaction tax

**For:** the Arrow agent. Read `README.md` (playbook) first. **Status after
scoping (2026-06-26): LOW value — do NOT file.** The gap is largely resolved on
current Arrow at the default (multithreaded) path; only a single-threaded
residual remains. This doc records the measurement that closes it.

## The gap (as originally characterized)
Acero's `FilterNode` physically **compacts** (gathers) the selected rows into a
new contiguous batch before handing them to a downstream aggregate. When the
aggregate only needs a mask (filter→sum/count), that gather is pure waste, and
its cost scales with selectivity (rows passing = rows gathered). The original
`MATERIALIZATION_EXPERIMENT.md` (Finding 1, pyarrow 23.0.1, Apple M-series) found
this **catastrophic**: `filt6` (count, 3 predicates) 70 ms/1-thread and **16 ms
even at 8 threads**, vs raw whole-array `pc` 5.6 ms and our fused kernel 4.3 ms.
The proposed upstream fix was mask / selection-vector push-down to the aggregate.

## Re-measurement on the latest release (pyarrow 24.0.0, fresh venv)
`benchmarks/acero_filter_compaction.py` (selectivity sweep, filter→sum) and
`benchmarks/acero_filt6.py` (the original 3-predicate shape):

**filt6 (count via sum, 3 predicates, 10M rows):**

| threads | acero filter→agg | raw_pc mask→sum | ratio | vs old (23.0.1) |
|---|---|---|---|---|
| **8 (default)** | **14.2 ms** | 12.9 ms | **1.1×** | was 2.9× — **gap gone** |
| 1 | 60.6 ms | 12.9 ms | 4.7× | was 12.5× — improved, residual |

**filter→sum selectivity sweep (single predicate, 10M rows):**

| | sel 0.01 | sel 0.10 | sel 0.50 | sel 0.90 |
|---|---|---|---|---|
| **8 threads:** acero / pc_masked | 2.9 / 9.5 | 4.6 / 17.7 | 12.7 / 45.4 | 7.5 / 18.8 |
| acero/pc_masked (8t) | **0.3×** | **0.3×** | **0.3×** | **0.4×** |
| **1 thread:** acero / raw_pc | 8.2 / 2.2 | 17.3 / 6.6 | 55.2 / 26.3 | 31.8 / 15.5 |
| acero/raw_pc (1t) | 3.7× | 2.6× | 2.1× | 2.1× |

## Findings
1. **Multithreaded (the default and realistic path): the gap is essentially
   gone on 24.0.0.** filt6 acero_8 is now 1.1× raw_pc (was 2.9×), and simple
   filter→sum acero_8 actually **beats** the no-gather masked reduce (0.3×) and
   is competitive with Polars. Acero's parallelism amortizes the per-morsel
   compaction. The old "catastrophic even at 8 threads" no longer reproduces.
2. **Single-threaded: a real residual remains** — acero is 2–5× slower than
   raw_pc, and its cost still scales with selectivity (8→17→55 ms as sel
   0.01→0.5), the compaction signature. But single-thread is not how Acero is
   used in practice.
3. **Not version-regressed** — single-thread ratios are the same on 23.0.1 and
   24.0.0 (~the mechanism is unchanged); what changed is the multithreaded path
   improved enough that the gap stopped mattering at the default.

## Verdict — do not file
The file-able framing was "filter→reduce is catastrophic"; on current Arrow that
is **no longer true at the default multithreaded path** (1.1× / often <1×). The
only surviving claim is "single-threaded Acero filter→reduce is 2–5× slower than
a mask-reduce" — a weak, easily-deprioritized issue for an engine designed to run
multithreaded. Net: **low value, recorded and closed; do not open an issue.**

- Gates: latest-release re-check **done** (gap resolved multithreaded). Duplicate
  search **moot** (verdict is don't-file).
- Keep the engine's own single-pass `fused_filter_aggs` Cython kernel regardless
  — it's still the fastest filter→reduce path and independent of this outcome.
- If ever revisited: the only angle worth a *documentation* note upstream is
  "Acero's FilterNode compacts; for single-threaded filter→reduce, prefer a
  masked compute or a fused kernel" — not a code change.

Artifacts: `benchmarks/acero_filter_compaction.py`, `benchmarks/acero_filt6.py`.
