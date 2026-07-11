# Hand-off: AG16 — NumPy `nansum`/`nanmean` don't skip `NaT` on `timedelta64` (silent `NaT`)

**For:** a fresh agent. Read `README.md` first. **Target project: `numpy/numpy`.**
**Priority: 2** — a small, sharp, mechanistically-pinned bug with a clear fix and
a precedent (#5222); silent-wrong-result class. File-worthy after a latest-version
re-check.

> **STATUS (2026-07-11): FOUND BY THE NUMPY/CPYTHON SCOUT (datetime64/NaT one-shot)
> + ROOT-CAUSED + DUP-SEARCHED — hand-off-ready (not filed).** Verified on numpy
> **2.5.1**; mechanism confirmed from the numpy source. Novel (see §Dup-search).
> Nothing filed without go-ahead (guardrail).

## The finding (one line)
Across the **full `nan*` reduction family on `timedelta64`**, the **additive**
reductions (`nansum`, `nanmean`, `nancumsum`) **silently return `NaT`** instead of
skipping it, while all 7 order-based reductions correctly skip — a silent wrong
result on the dtype's missing value.

## Evidence (numpy 2.5.1) — the full family matrix on `timedelta64` + `NaT`
`td = np.array([1, np.timedelta64("NaT"), 3, 2], dtype="timedelta64[D]")`
(skipping `NaT` → values `[1, 3, 2]` days)

| behavior | # | ops | note |
|---|---|---|---|
| **SKIP-NaT** (correct) | 7 | `nanmin`, `nanmax`, `nanmedian`, `nanpercentile`, `nanquantile`, `nanargmin`, `nanargmax` | order/sort-based path handles `NaT` |
| **LEAK-NaT** (silent bug) | 3 | `nansum`, `nanmean`, `nancumsum` | return `NaT`; `nancumsum` → `[1 day, NaT, NaT, NaT]` |
| **ERROR** (defensible) | 4 | `nanprod`, `nanstd`, `nanvar`, `nancumprod` | `multiply`/`square` undefined for timedelta — product/variance of durations is ill-defined |

So the bug is precisely the **additive `_replace_nan`-based subfamily** — and those
are exactly the timedelta reductions that make sense and that pandas needs
(`sum`/`mean`/`cumsum` of durations, skipping `NaT`). The whole point of the
`nan*` family is to skip missing values (`nansum` of a float with `NaN` treats it
as 0); on `timedelta64` these three silently don't, and don't error either.

## What "desired behavior" rests on (be honest — the docs don't settle it)
The `nansum`/`nanmean`/`nanmin` docstrings all say *"NaN"* ("treating NaNs as
zero" / "ignoring NaNs"); **none mentions `NaT`, `timedelta64`/`datetime64`, or a
dtype restriction.** So this is **not** a documented-contract violation, and the
report must not claim it is. The basis that `NaT` *should* be skipped, in
descending strength:
1. **Internal inconsistency** (strongest, needs no external spec): on the *same*
   array, **7 sibling `nan*` reductions skip `NaT`** (matrix above) and 3 don't —
   all sharing the "ignoring NaNs" wording. NumPy contradicts itself.
2. **Maintainer precedent** #5222: they already treated the analogous
   NaT-across-reductions inconsistency as a bug and fixed it toward consistency.
3. **Unambiguous regardless of policy**: `nansum` returns `NaT` **silently** — not
   an error, not a documented rejection. Silent-wrong-output is bad whether the
   intended answer is skip-`NaT` *or* raise.

**Posture:** file as a **question** ("is this intended? sibling ops + #5222 say
no"), not an assertion that the docs mandate skipping.

## Root cause (confirmed in numpy source — `numpy/lib/_nanfunctions_impl.py`)
`nansum`/`nanmean`/`nanvar`/`nanstd` all mask NaNs via `_replace_nan(a, val)`,
which builds the mask **only** for object or inexact dtypes:
```python
if a.dtype == np.object_:
    mask = np.not_equal(a, a, dtype=bool)
elif issubclass(a.dtype.type, np.inexact):   # float / complex only
    mask = np.isnan(a)
# else (timedelta64 'm', datetime64 'M', int): falls through
return a, None      # <- no mask, `a` unchanged: NaT is NOT replaced
```
For `timedelta64` (`kind == 'm'`), `_replace_nan` returns `(a, None)` — it does
nothing — so `NaT` flows straight through the plain `add.reduce` → the result is
`NaT`. **The kicker:** `np.isnan(td)` **already works** on `timedelta64`
(`np.isnan(td) == [False, True, False, False]`, identical to `np.isnat`), so the
mask is available — `_replace_nan` just never reaches for it. `nanmin`/`nanmax`/
`nanmedian` take a different, NaT-skipping path, which is why they're consistent.

Confirmation: `np.nansum(td_without_NaT) == 6 days` (works), `np.nansum(td_with_NaT)
== NaT` (leaks).

## Also breaks `datetime64` (via differences — the pandas mean-date idiom)
The leak follows through anything that produces a `timedelta64` with `NaT`:
```python
dt = np.array(["2020-01-01","NaT","2020-03-01","2020-02-01"], dtype="datetime64[D]")
np.nansum(dt - dt[0])        # NaT   (differences from a ref)
np.nansum(np.diff(dt))       # NaT   (consecutive differences)
dt[0] + np.nanmean(dt - dt[0])   # NaT  <- the pandas "mean date, skip NaT" idiom, broken
```
This last line matters: NumPy **can't** `mean`/`nanmean` a `datetime64` array
directly (`ufunc 'add' cannot use operands with type datetime64`), so the *only*
way to get a nan-skipping mean date is the offset idiom `ref + mean(values-ref)` —
which AG16 silently turns into `NaT`. (Separately, `np.nanmedian(dt)` also errors
on `add` for `datetime64`, though it works fine for `timedelta64` — a smaller
adjacent datetime-median gap.)

## Why it matters (pandas relevance)
pandas' `Series.mean()`/`.sum()` on a `timedelta64` column **skip `NaT`** (the
`skipna=True` default). NumPy's `nanmean`/`nansum`/`nancumsum` — the natural
primitives for exactly that, and the building blocks of the mean-date idiom above
— give the wrong answer (`NaT`) on the same data, so they can't be relied on for
nullable-timedelta/datetime reductions; a consumer must special-case `NaT` itself.
Any code assuming `nan*` means "skip missing" is silently wrong here.

## Standalone repro (pure numpy)
```python
import numpy as np
print("numpy", np.__version__)
td = np.array([1, np.timedelta64("NaT"), 3, 2], dtype="timedelta64[D]")
print("nanmin :", np.nanmin(td))   # 1 days  (skips NaT)
print("nanmax :", np.nanmax(td))   # 3 days  (skips NaT)
print("nansum :", np.nansum(td))   # NaT     (BUG: should be 6 days)
print("nanmean:", np.nanmean(td))  # NaT     (BUG: should be 2 days)
print("isnan works:", np.isnan(td).tolist())  # [False, True, False, False]
```

## Dup-search (recorded — numpy/numpy, thorough)
Searched (issues **and** PRs): `nansum nanmean timedelta NaT not skipped`, `nan
functions timedelta64 NaT isnan`, `nanmean datetime timedelta NaT`, `nansum
nanmean NaT timedelta`, `nanmean nansum datetime timedelta`, `_replace_nan
timedelta datetime`, `nan functions not support datetime timedelta`, `nanmean
timedelta returns NaT`, `nansum NaT not treated as zero`, `nanfunctions datetime
timedelta support`, `reduction skipna datetime timedelta`. **No open or closed
issue reports the additive-nan-reduction `NaT` leak on `timedelta64`/`datetime64`**
— all hits were unrelated (groupby ENH, MIPS `nanmin`, float16→datetime UB).
- **#5222 [closed 2020] "BUG: (nan)?(arg)?(max|min) handling of NaT inconsistent"**
  — the PRECEDENT, but a *different* case: it fixed the **plain** `min`/`max`/
  `argmin`/`argmax` NaT consistency (they now propagate NaT like NaN). It did
  **not** touch the **nan-aware additive** reductions. This is the surviving
  sibling — same "NaT-handling inconsistent across reductions" class, unfixed for
  `nansum`/`nanmean`/`nancumsum` (the AG11 pattern: a fix that missed a sibling).

## Recommendation
**File one focused issue** on numpy/numpy, framed as a **question** (not a
docs-violation claim — see "desired behavior" above): the additive nan-reductions
`nansum`/`nanmean`/`nancumsum` silently return `NaT` on `timedelta64`, where the 7
order-based nan-reductions skip it — is the inconsistency intended? Lead with the
matrix + #5222 precedent. Root cause +
fix: extend `_replace_nan`'s mask branch to cover datetime/timedelta
(`a.dtype.kind in "mM"`) using `np.isnat` (or `np.isnan`, which already works),
then fill `NaT` with the zero-timedelta. Reference #5222 as the precedent for the
min/max half. (The 4 ERROR ops — `nanprod`/`nanstd`/`nanvar`/`nancumprod` — are
out of scope: product/variance of durations is genuinely ill-defined.)

## Gates
- [x] **Reproduces** — numpy 2.5.1, repro above; root cause read from
      `_nanfunctions_impl._replace_nan`.
- [x] **Duplicate search recorded (thorough, 11 phrasings, issues+PRs)** — no
      issue reports it; #5222 is the closed plain-reduction sibling; the nan-aware
      additive case is novel/unfixed. Datetime manifestation (mean-date idiom)
      confirmed too.
- [ ] **Re-verify on the latest numpy release** at file time (2.5.1 may not be newest).
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Definition of done
Filed (with #) referencing #5222, or shelved, recorded here + in `README.md` +
`../ARROW_GAPS.md`. The datetime64/NaT scout's genuine find — the "one shot" that
paid off (see AG15 for the string-NA sibling; the interop scout between them was
low-yield).
