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
support — the `np.strings` ufuncs have **four mutually-inconsistent NA
behaviors**, and the case-transform ops raise a **broken, internals-leaking
error** on NA.

## Evidence (numpy 2.5.1) — the NA-behavior matrix
```python
import numpy as np
a = np.array(["alpha", np.nan, "gamma"], dtype=np.dtypes.StringDType(na_object=np.nan))
```
| behavior | ops | what happens on the NA element |
|---|---|---|
| **PROPAGATE-NA** (the sensible/pandas-like result) | `add`, `multiply`, `replace`, `strip`, `lstrip`, `ljust`, `zfill` | NA in → NA out |
| **CLEAN-ERR** | `str_len`, `find`, `rfind`, `count`, `index` | raises, e.g. `ValueError: 'find' not supported for null values that are not strings` |
| **LEAKY-ERR** (a bug) | `upper`, `lower`, `capitalize`, `swapcase`, `title` | raises `TypeError: descriptor 'upper' for 'str' objects doesn't apply to a 'float' object` |
| **NA-as-value** | `isalpha`, `isdigit`, `startswith`, `endswith`, `equal` | returns a `bool` for the NA slot (no propagation) |

So the *same* missing value yields propagate / clean-error / leaky-error / bool
depending only on which op you call — there is **no coherent NA policy**.

## The sharp, file-worthy bug: case ops leak the legacy `_vec_string` path
`upper`/`lower`/`capitalize`/`swapcase`/`title` are still implemented via the
legacy element-wise helper (traceback: `numpy/_core/strings.py … `
`return _vec_string(a_arr, a_arr.dtype, 'upper')`), which calls Python
`str.upper` on each element — including the `na_object` (`nan`) — producing the
nonsensical `descriptor 'upper' for 'str' objects doesn't apply to a 'float'
object`. The analogous **native** string ufuncs on the same array either
**propagate NA** (`replace`, `strip`) or raise a **clean domain error**
(`find`, `str_len`). So the case ops are the odd ones out on two counts: wrong
error text *and* an implementation detail leaked to the user. Clear fix: port
them to the native path (propagate NA like `replace`) or, at minimum, raise the
same clean "not supported for null values" error as `find`.

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
**File one focused bug** on numpy/numpy for the case-op leaky error
(`upper`/`lower`/`capitalize`/`swapcase`/`title` raise a wrong internals-leaking
`TypeError` on NA where sibling ops propagate or error cleanly), **and** mention
the broader four-way inconsistency as motivation for a coherent NA policy (or add
it to #25693). Lead with the concrete bug + the matrix; it's small and actionable.

## Gates
- [x] **Reproduces** — numpy 2.5.1, matrix + standalone repro above.
- [x] **Duplicate search recorded** — novel; #25693 is internals-only, #26198/#25347 related-not-this.
- [ ] **Re-verify on the latest numpy release** at file time (2.5.1 may not be newest).
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Definition of done
Filed (with #) or shelved, recorded here + in `README.md` + `../ARROW_GAPS.md`.
This is a pure-numpy substrate gap surfaced by the NumPy/CPython scout — the
scout's first genuine find (the DuckDB scout before it was low-yield).
