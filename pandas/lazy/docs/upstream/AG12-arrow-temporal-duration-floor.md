# AG12 — Arrow temporal unit downcast has no floor rounding (duration kernel missing)

**Status:** verified on pyarrow 23.0.1; **duplicate exists for the core gap →
apache/arrow#50395 (open).** Do NOT file a duplicate. Candidate action is a
data-point comment (and optionally the C++ kernel PR). Human go-ahead required
before any outward-facing action (see README guardrails).

**Substrate:** Arrow C++ compute — `cast` + `round_temporal`/`floor_temporal`/
`ceil_temporal` kernels.
**Downstream consumer:** pandas `ArrowExtensionArray._dt_as_unit`
(`pandas/core/arrays/arrow/array.py`).
**Surfaced by:** pandas PR #63573 (PERF: Arrow-native `as_unit`) introduced a
regression → fixed by PR #66218 with a hand-rolled int64 floor.

## Claim

Casting a PyArrow `timestamp`/`duration` array to a **coarser** unit (e.g.
`ns → s`) cannot produce numpy/SQL floor-toward-−∞ semantics. `pc.cast` offers
only two behaviors, both wrong for pandas:

- `safe=True`  → **refuses** any lossy value: `ArrowInvalid: would lose data`.
- `safe=False` → **truncates toward zero** (not toward −∞).

`CastOptions` exposes only boolean `allow_time_truncate` / `allow_time_overflow`
— **no rounding-mode enum** on cast, though Arrow has a `RoundMode`
(floor/ceil/half_to_even) used by the `*_temporal` kernels. The temporal-rounding
kernels floor correctly **but have no `duration` kernel**, so the one clean
Arrow-native floor path is unavailable for durations.

## Evidence (pyarrow 23.0.1, standalone)

```python
import pyarrow as pa, pyarrow.compute as pc

arr = pa.array([-93784567890123], type=pa.duration("ns"))
pc.cast(arr, pa.duration("s"), safe=False).to_pylist()
# -> timedelta(-93784s);  numpy/pandas floor -> -93785s   (truncates toward zero)
pc.cast(arr, pa.duration("s"), safe=True)
# -> ArrowInvalid: Casting from duration[ns] to duration[s] would lose data

# floor_temporal floors correctly across a unit — for TIMESTAMPS:
pc.floor_temporal(pa.array([-1], type=pa.timestamp("ns")), unit="second")
# -> 1969-12-31 23:59:59   (correct; a following safe cast to s is exact)

# ...but NO duration kernel:
pc.floor_temporal(pa.array([-93784567890123], type=pa.duration("ns")), unit="second")
# -> ArrowNotImplementedError: Function 'floor_temporal' has no kernel
#    matching input types (duration[ns])
```

## Gaps, ranked

1. **`floor_temporal`/`ceil_temporal`/`round_temporal` have no `duration` kernel**
   — registered only for `date32/64`, `time32/64`, `timestamp`
   (`WithDates, WithTimes, WithTimestamps` in `scalar_temporal_unary.cc`).
   *Filable, well-scoped, has a concrete downstream consumer — and the issue
   already exists: apache/arrow#50395.* With it, pandas could floor a duration
   then `safe`-cast instead of hand-rolling int64 math.
2. **No rounding-mode control on temporal `cast`** — `safe=False` hardcodes
   toward-zero; no floor/ceil/round even though `RoundMode` exists elsewhere.
   Broader enhancement; #1 is the tactical subset. **No upstream issue found**
   (searched 2026-07-09).
3. **`safe=False` silently wraps on int64 overflow during upcast** (e.g.
   `duration[s]` year-2600 → `ns` returns garbage). *Not filable* — `safe=False`
   is documented as "no checks". pandas correctly uses `safe=True` on the upcast
   path and re-raises `OutOfBounds{Datetime,Timedelta}`.

## How pandas worked around it (PR #66218)

`_dt_as_unit` was made direction-aware:

- **Upcast / same-res** (finer target): `pc.cast` with default `safe=True`, catch
  `pa.ArrowInvalid`, re-raise as `OutOfBoundsDatetime` / `OutOfBoundsTimedelta`.
- **Downcast** (coarser target): explicit int64 floor —
  `quotient = i8 // ratio`; subtract 1 where remainder < 0 — because neither
  `cast` nor `floor_temporal` (no duration kernel) can floor.

If gap #1 lands upstream, the manual duration downcast becomes
`floor_temporal(..., unit) → safe cast` and the int64 dance can be deleted.

## Duplicate search (2026-07-09, apache/arrow)

`gh api -X GET search/issues -f q="repo:apache/arrow <terms>"` across:
`floor_temporal duration`, `round_temporal duration kernel`, `ceil_temporal
duration`, `temporal cast rounding mode`, `cast duration truncate toward zero`.

- **Hit (gap #1):** **#50395** "[C++] Support duration type in floor_temporal/etc"
  — open, `Type: enhancement` / `Component: C++`, 0 comments, no assignee,
  filed 2026-07-06. Body names the exact registration site
  (`WithDates, WithTimes, WithTimestamps`). <https://github.com/apache/arrow/issues/50395>
- **No hit** for gap #2 (rounding-mode on `cast`).

## Action

- **Do not duplicate #50395.** Gates 1–3 already green (repro above, latest-release
  re-check pending only if we escalate).
- Candidate contributions (each needs explicit human go-ahead):
  1. Comment on #50395 with the pandas downstream motivation + repro (0 comments
     today; a concrete consumer strengthens it).
  2. Optionally implement the C++ duration kernel in `scalar_temporal_unary.cc`.
  3. File gap #2 (rounding-mode on `cast`) only if #50395 stalls — larger design
     change, no issue yet.
- **Keep the pandas workaround regardless** of upstream outcome.
