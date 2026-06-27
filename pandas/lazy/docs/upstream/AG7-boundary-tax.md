# Scoping: AG7 — the Arrow↔NumPy/pandas boundary tax

**For:** the Arrow agent. Read `README.md` (playbook) first. **Status after
scoping (2026-06-26): NOT an Arrow gap — do NOT file against apache/arrow.** The
tax is a structural NumPy/pandas **dtype-model mismatch**, not an Arrow
inefficiency. The actionable lever is engine-side (Arrow-backed output), and it
*moves* the tax rather than removing it unless the whole pipeline stays Arrow —
which is the already-NO-GO architectural rewrite.

## The gap (as characterized)
`MATERIALIZATION_EXPERIMENT.md` II/III: Acero's hash join is fastest in isolation
(229 ms, 3.3× Polars) but `to_numpy()` on a ~100M-row output costs ~435 ms, so
acero+round-trip ≈ `pd.merge`. "Acero only wins if the pipeline stays Arrow."
AG7 was tagged **high** severity, structural. Question this scoping answers: **is
the conversion itself a file-able Arrow inefficiency, or inherent to NumPy/pandas?**

## Decomposition by dtype (pyarrow 24.0.0, pandas 3.0.3, 10M rows)
`benchmarks/arrow_boundary_tax.py`:

**Arrow array → NumPy (single column):**

| column | to_numpy | zero-copy? | nature |
|---|---|---|---|
| f64 / i64, no null, 1 chunk | **0.0 ms** | **ZERO-COPY** | no tax |
| f64 / i64, 10% nulls | 22 ms | copy | NumPy has no null mask → must fill |
| string (low/high card) | 159 / 212 ms | copy | NumPy object array = 10M CPython `str` objects |
| f64 chunked ×8 | 19 ms | copy | concatenation to contiguous |

**Arrow table → pandas (4 cols incl 1 string + 1 nullable):**

| path | ms | |
|---|---|---|
| `to_pandas()` default | 25.8 | string materialize + BlockManager consolidate |
| `to_pandas(split_blocks=True)` | 24.8 | ~same |
| numeric-only `to_pandas()` | 19.8 | BlockManager block-consolidation copy (a *pandas* cost) |
| **`to_pandas(types_mapper=pd.ArrowDtype)`** | **0.1** | **zero-copy — Arrow buffers kept** |

## Findings — the tax is inherent, not an Arrow bug
1. **Clean numeric is already zero-copy** (0 ms). For the case Arrow controls
   fully, there is no tax.
2. **The cost lives entirely in dtype-model mismatches NumPy/pandas impose, not
   Arrow:**
   - **strings** → NumPy's only representation is an object array of Python
     `str`; the 159–212 ms is ~16–21 ns/value of *CPython object creation*, not
     Arrow work. Arrow cannot make this cheaper — there is no native NumPy string.
   - **nulls** → NumPy primitives have no null mask, so Arrow must materialize
     (fill/float-cast). Inherent.
   - **BlockManager consolidation** (numeric `to_pandas` 19.8 ms) is a **pandas**
     cost (building consolidated 2-D blocks), not an Arrow conversion.
3. **The zero-copy escape exists and is ~free** (`types_mapper=pd.ArrowDtype`,
   0.1 ms) — but it returns **Arrow-backed** pandas. That doesn't *remove* the
   tax; it *defers* it to whenever a downstream NumPy/object consumer touches the
   column. It only truly wins if the entire pipeline stays Arrow-native end-to-end.

## Verdict — do not file (against Arrow); structural, engine-side only
- **No file-able apache/arrow item.** Arrow's conversions are already zero-copy
  where the dtype models match; everything slow is the NumPy/pandas model
  (object strings, no null mask, block consolidation) — Arrow can't fix that.
  Duplicate search **moot** (nothing to file).
- **Engine-side lever (real, bounded):** return `pd.ArrowDtype`-backed columns
  for string/nullable outputs to make `collect()` zero-copy at the boundary —
  but this is a public-API/return-type change (out of scope per
  `MATERIALIZATION_EXPERIMENT.md` Part IV's "non-pandas return" floor), and it
  only pays if consumers stay Arrow; otherwise it relocates the same cost.
- **Whole-pipeline-Arrow** (the only way to truly avoid it) is the
  already-disproven path: `MATERIALIZATION_EXPERIMENT.md` Part III showed Acero
  end-to-end is *slower* than our pd.merge engine on real q3 (127 vs 65 ms) and
  6× behind Polars — there is no faster Arrow substrate to stay in.

**Net:** AG7 is the same shape as the GH200 finding (`GPU_DEVICE_RESIDENT_PROBE.md`)
— the cost is the *consumer's data model*, not the substrate. Closed as
structural/characterization; no Arrow issue. Artifact:
`benchmarks/arrow_boundary_tax.py`.
