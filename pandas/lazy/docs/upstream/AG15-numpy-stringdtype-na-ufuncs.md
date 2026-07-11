# Hand-off: AG15 — NumPy `StringDType` string ufuncs handle NA (`na_object`) inconsistently

**For:** a fresh agent. Read `README.md` first. **Target project: `numpy/numpy`**
(NOT Arrow/DataFusion). **Priority: 3** — a real, reproducible, pandas-relevant
substrate gap with a sharp concrete bug inside it; likely file-worthy after a
latest-version re-check + dup-search refresh.

> **STATUS (2026-07-11): FOUND BY THE NUMPY/CPYTHON SCOUT + CHARACTERIZED —
> hand-off-ready (not filed).** The differential-probe method (vary one axis,
> watch behavior diverge) pointed at NumPy 2.x `StringDType`. Verified on numpy
> **2.5.1**. Novel (see §Dup-search). Deliverable is a NumPy report; per the
> guardrail, nothing filed without go-ahead.

## The finding (one line)
For `np.dtypes.StringDType(na_object=…)` — whose defining feature is missing-value
support — the `np.strings` ufuncs have **no coherent NA policy**: across the full
~43-op surface they split into **five** behaviors, and **12 ops error on NA in
three *different* message styles** (one of which is a broken internals leak).

## Evidence (numpy 2.5.1) — the full NA-behavior matrix (all ~43 `np.strings` ops)
```python
import numpy as np
a = np.array(["alpha", np.nan, "gamma"], dtype=np.dtypes.StringDType(na_object=np.nan))
```
| behavior | # | ops | NA element result |
|---|---|---|---|
| **PROPAGATE-NA** (sensible/pandas-like) | 10 | `add`, `multiply`, `replace`, `strip`, `lstrip`, `rstrip`, `center`, `ljust`, `rjust`, `zfill` | NA in → NA out |
| **NA→False (predicates)** | 17 | `isalpha`, `isalnum`, `isdigit`, `isdecimal`, `isnumeric`, `isspace`, `islower`, `isupper`, `istitle`, `startswith`, `endswith`, `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal` | returns `False` for NA (no NA propagation) |
| **CLEAN-ERR** `"…not supported for null values that are not strings"` | 6 | `str_len`, `find`, `rfind`, `index`, `rindex`, `count` | `ValueError` |
| **LEAKY-ERR** `"descriptor 'upper' for 'str' objects doesn't apply to a 'float' object"` (a bug) | 6 | `upper`, `lower`, `capitalize`, `swapcase`, `title`, `encode` | `TypeError` (internals leak) |
| **3rd error style** `"Cannot slice null string"` | 1 | `slice` | `TypeError` |

So the *same* missing value yields **propagate / False / three different errors**
depending only on which op you call. **12 ops (CLEAN-ERR + LEAKY-ERR + slice)
outright reject NA, in three inconsistent message styles** — on the dtype whose
headline feature is `na_object`.

## Separate sub-finding — `partition`/`rpartition` unsupported for `StringDType`
`np.strings.partition` / `rpartition` fail on `StringDType` **even with no NA
present** — `UFuncTypeError: ufunc '_partition' did not contain a loop with the
correct signature`. This is a *coverage gap* (the kernel isn't implemented for
the dtype at all), independent of the NA issue — the NumPy sibling of the Arrow
view-kernel gaps (AG9). Worth a one-line mention or its own issue.

## The sharp, file-worthy bug: case/encode ops leak the legacy `_vec_string` path
`upper`/`lower`/`capitalize`/`swapcase`/`title`/`encode` are still implemented via
the legacy element-wise helper (traceback: `numpy/_core/strings.py … `
`return _vec_string(a_arr, a_arr.dtype, 'upper')`), which calls the Python
`str.upper` descriptor on each element — including the `na_object` (`nan`) —
producing the nonsensical `descriptor 'upper' for 'str' objects doesn't apply to
a 'float' object`. The analogous **native** string ufuncs on the same array either
**propagate NA** (`replace`, `strip`) or raise a **clean domain error** (`find`,
`str_len`). So these ops are the odd ones out on two counts: wrong error text
*and* a leaked implementation detail. Clear fix: route them through the native
path (propagate NA like `replace`) or, at minimum, raise the same clean "not
supported for null values" error as `find`.

Note the broader inconsistency this sits inside: NA rejection uses **three
different messages** — `"…not supported for null values that are not strings"`
(find-family), `"descriptor … doesn't apply to a 'float'"` (case/encode, the
leak), and `"Cannot slice null string"` (`slice`). Even the engines that *do*
reject NA don't agree on how.

## Why it matters (pandas relevance)
pandas needs NA in string columns. If pandas were to back a string dtype with
NumPy `StringDType(na_object=…)`, `Series.str.upper()` on a column containing NA
would surface this broken `TypeError`, while `Series.str.replace()` would work —
an inconsistency users would hit immediately. This is the NumPy sibling of the
Arrow view-kernel coverage story (AG9): the modern string dtype's op coverage is
incomplete/inconsistent on its headline feature.

## Standalone repro (pure numpy)
```python
import numpy as np
print("numpy", np.__version__)
a = np.array(["alpha", np.nan, "gamma"], dtype=np.dtypes.StringDType(na_object=np.nan))
print("replace:", np.strings.replace(a, "a", "X"))   # ok: ['XlphX' nan 'gXmmX']  (propagates)
try: np.strings.upper(a)
except Exception as e: print("upper  :", type(e).__name__, "-", str(e)[:60])  # LEAKY TypeError
try: np.strings.find(a, "a")
except Exception as e: print("find   :", type(e).__name__, "-", str(e)[:60])  # clean ValueError
```
Expected on 2.5.1: `replace` propagates NA; `upper` raises the leaky
`descriptor … doesn't apply to a 'float'`; `find` raises the clean "not
supported for null values".

## Dup-search (recorded — numpy/numpy)
Searched `StringDType na_object np.strings null`, `StringDType null upper
descriptor float`, `not supported for null values string ufunc`, `na_object
strings upper lower error`, `StringDType missing value ufunc inconsistent`.
- **#25693 [open] "TSK: Follow-up things for stringdtype"** — a follow-up task
  list, but about *internals* (size arg, promoters, cython API, arena bitfields);
  it does **not** cover NA handling across the string ufuncs. Could host the data
  point, but the concrete case-op bug warrants its own issue.
- **#26198 [closed] "ensure find-like ufuncs convert arguments to common dtypes"**
  and **#25347 (NEP 55 introduction)** — related, not this. No open issue frames
  the NA-consistency gap or the case-op leaky error.

## Recommendation
Two candidate reports (the first is the sharp, small, unambiguous one):
1. **The case/encode leaky-error bug** — `upper`/`lower`/`capitalize`/`swapcase`/
   `title`/`encode` raise a wrong internals-leaking `TypeError` on NA where
   siblings propagate or error cleanly. Lead with this + the full matrix; small
   and actionable, clear fix (native path / clean error).
2. **The broader NA-policy inconsistency** — 5 behaviors, 3 error styles — as
   motivation for a coherent, documented NA policy across `np.strings`
   (propagate vs reject, and if reject then one message). Fits #25693 or its own
   discussion.

Also worth a **one-line mention**: `partition`/`rpartition` are unimplemented for
`StringDType` (coverage gap, NA-independent). Lead outward with #1; it's the most
defensible single bug.

## Gates
- [x] **Reproduces** — numpy 2.5.1, full ~43-op matrix + standalone repro above;
      `partition` coverage gap and `slice`/case-op error styles all confirmed.
- [x] **Duplicate search recorded** — novel; #25693 is internals-only, #26198/#25347 related-not-this.
- [ ] **Re-verify on the latest numpy release** at file time (2.5.1 may not be newest).
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Adjacent: `StringDType` interop (scouted 2026-07-11 — LOW-YIELD, recorded so we don't re-scout)
Probed `StringDType` round-tripping with Arrow / pandas / object for a *second*
finding. Verdict: nothing new to file.
- **`pa.array(StringDType)` fails** (`ArrowNotImplementedError: Unsupported numpy
  type 2056`; forcing `type=large_string` → `ArrowTypeError`) — so NumPy's modern
  string dtype has no path to Arrow. **Already tracked: Arrow #42018 [open]
  "[Python] Conversion to/from numpy 2.0+ new StringDType"** → attach-data if we
  ever care, don't file.
- **NA gets stringified on cast** (`None`→`"None"`, `pd.NA`→`"<NA>"`, and NA→`"nan"`
  casting to `<U`) — but this is **sentinel-mismatch, expected**: `astype(SD)`
  only treats the exact declared `na_object` as NA (`object(None)→SD(na_object=None)`
  correctly yields NA; `<U` has no NA slot so the sentinel is stringified). A
  silent footgun, but defensible by design — not a clean bug.
- **`pd.Series(SD)` → object dtype** (pandas doesn't recognize `StringDType`;
  expected — no pandas support yet).

## Definition of done
Filed (with #) or shelved, recorded here + in `README.md` + `../ARROW_GAPS.md`.
This is a pure-numpy substrate gap surfaced by the NumPy/CPython scout — the
scout's first genuine find (the DuckDB scout before it was low-yield). The
interop follow-up (above) was low-yield; the `np.strings` NA gap remains the
file-worthy item.
