# Hand-off: AG19 — Arrow's Substrait consumer doesn't map common string/date functions (kernels exist)

**For:** a fresh agent. Read `README.md` first. **Target project: `apache/arrow`**
(the C++ `arrow::engine`/Substrait consumer). **Priority: 3** — an AG9-class
coverage gap: the kernels already exist, only the Substrait-function→Arrow-compute
mapping is unregistered, so the fix is small and additive. Precedent #13285 (a
closed "register more Substrait functions" issue that missed these). File-worthy
after a latest-version re-check + dup-search refresh.

> **STATUS (2026-07-11): FOUND BY THE SUBSTRAIT ROUNDTRIP PROBE + pinned to a
> per-function coverage matrix + kernels-exist-confirmed + dup-searched —
> hand-off-ready (not filed).** Reproduces IDENTICALLY on pinned pyarrow 23.0.1
> AND latest **25.0.0** (`substrait_fn_coverage.py`). Nothing filed without
> go-ahead (guardrail).

## The finding (one line)
Arrow's Substrait **consumer** (`pyarrow.substrait.run_query` / `arrow::engine`)
has no function-extension mapping for five common Substrait scalar functions —
`starts_with`, `ends_with`, `regexp_like`, `substring`, `date_part` — so any
Substrait plan using them fails with `No conversion function exists to convert the
Substrait function <X> to an Arrow call expression`, **even though Arrow already
implements every one of those kernels** (`pc.starts_with`, `pc.ends_with`,
`pc.match_substring_regex`, `pc.utf8_slice_codeunits`, `pc.year`).

## How it was found
The Substrait roundtrip probe (`../SUBSTRAIT_ROUNDTRIP.md`) lowered the 22 TPC-H
plans and, after the AG17/AG18 portability fixup, still left 8/22 unconsumable by
Acero with this error. `../../benchmarks/substrait_fn_coverage.py` then reduced it
to one minimal plan per function — a clean coverage matrix rather than a per-query
first-error.

## Evidence — the coverage matrix (identical on pyarrow 23.0.1 AND 25.0.0)
Each function emitted via a minimal DataFusion expression → Substrait → AG17/AG18
fixup → Acero:

| Acero verdict | # | functions |
|---|---|---|
| **OK** | 14 | `equal`, `gt/lt/gte/lte`, `or`, `not`, `is_null`, `add/subtract`, `multiply`, `divide`, `in_list`, `case_when`, `sum`, `avg`, `min/max`, `count` |
| **NO-CONVERSION** (the gap) | 5 | **`starts_with`, `ends_with`, `regexp_like`, `substring`, `date_part`** |
| **other** | 1 | `count(distinct)` — `Unsupported aggregation invocation 'AGGREGATION_INVOCATION_DISTINCT'` (a *separate* consumer gap, see below) |

The gap is coherent: arithmetic, comparison, logical, `in_list`, `case_when` and
the standard aggregates all map fine; it's specifically the **string-predicate /
substring / temporal-extract** family that's unregistered.

## The kernels already exist — so it's a pure mapping gap (cheap fix)
| Substrait fn | existing Arrow compute kernel |
|---|---|
| `starts_with` | `pc.starts_with` |
| `ends_with` | `pc.ends_with` |
| `regexp_like` | `pc.match_substring_regex` (or `pc.match_like`) |
| `substring` | `pc.utf8_slice_codeunits` |
| `date_part(year)` | `pc.year` (temporal component kernels) |

All five confirmed present (`hasattr(pc, …) == True`). So no kernel work is
needed — only registering the Substrait standard-function URIs → these calls in
the consumer's extension registry (the same table #13285 extended). This is the
direct sibling of the Arrow **string-view kernel coverage** contribution (AG9):
the modern surface is incomplete on exactly the string/date ops.

## Adjacent gap (record together, likely its own issue): `count(distinct)`
Acero's Substrait consumer rejects the DISTINCT aggregate invocation:
`Unsupported aggregation invocation 'AGGREGATION_INVOCATION_DISTINCT'`. Arrow has
`count_distinct`, so again the kernel exists and only the Substrait
invocation→kernel path is missing. Separate from the scalar-function coverage but
same "consumer doesn't wire an existing kernel" class.

## Why it matters (probe relevance)
This is the last wall between the Substrait fan-out and full cross-engine coverage:
after AG17 (producer output_type) + AG18 (deprecated-field bridges) are worked
around, **8/22 TPC-H queries still can't run on Acero solely because of these five
unmapped functions** (every TPC-H `LIKE`/`substring`/`extract(year)` predicate).
Registering them would take Acero from 11/22 toward full coverage and turn the
differential probe's Substrait fan-out fully live (cross-engine *result*
divergences instead of consumer-coverage ones).

## Standalone repro (pure datafusion + pyarrow + substrait)
`../../benchmarks/substrait_fn_coverage.py` — run in a throwaway venv on the latest
releases (`pip install -U pyarrow datafusion duckdb substrait`). Minimal inline
version:
```python
import pyarrow as pa, pyarrow.substrait as pas, pyarrow.compute as pc
from datafusion import SessionContext, col, functions as F, lit
import datafusion.substrait as ss
# NB: fill ScalarFunction.output_type first (AG17) or it fails earlier; see
# substrait_fixup.fill_output_types. With that done:
print("kernel exists:", hasattr(pc, "starts_with"))   # True
# a Substrait plan using starts_with -> Acero raises:
#   ArrowNotImplementedError: No conversion function exists to convert the
#   Substrait function starts_with to an Arrow call expression
```
Report the versions (datafusion producer, pyarrow/Acero consumer, `substrait`
pkg). Confirmed on pyarrow 23.0.1 and 25.0.0.

## Dup-search (recorded — apache/arrow)
- **#13285 [closed] "ARROW-15582: [C++] Add support for registering tricky
  functions with the Substrait consumer"** — the PRECEDENT: it registered a batch
  of Substrait functions but did **not** cover `starts_with`/`ends_with`/
  `regexp_like`/`substring`/`date_part` (still NO-CONVERSION on 25.0.0). Same
  incomplete-coverage / sibling-miss pattern as AG9/AG11 — reference it.
- **#39691 [closed] "pyarrow.compute.Expression.to_substrait() is missing
  conversions"** — *producer* side (Arrow → Substrait), the opposite direction;
  not this.
- Refresh at file time: `substrait consumer function extension starts_with`,
  `No conversion function Substrait Arrow call`, `substrait date_part substring
  consumer`, and check open PRs.

## What "desired behavior" rests on (be honest)
The Substrait spec defines these as standard functions
(`functions_string.yaml` / `functions_datetime.yaml`), and Arrow implements the
kernels — so this is a genuine **coverage/enhancement** gap, framed as "please
register the mapping", not a correctness bug. #13285 establishes that maintainers
treat "register more Substrait functions" as in-scope and welcome.

## Recommendation
**File one enhancement** on apache/arrow: the Substrait consumer lacks
function-extension mappings for `starts_with`/`ends_with`/`regexp_like`/
`substring`/`date_part` (kernels already exist — table above), blocking common
`LIKE`/`substring`/`extract` plans (8/22 TPC-H). Reference #13285 as precedent,
offer the mapping (it's a small, additive registry change), and mention the
`count(distinct)` invocation gap as a likely-related follow-up. On-charter (Arrow
is a named substrate) and the direct sibling of AG9.

## Gates
- [x] **Reproduces** — coverage matrix on pyarrow 23.0.1 AND 25.0.0
      (`substrait_fn_coverage.py`); 5 NO-CONVERSION functions pinned.
- [x] **Kernels-exist confirmed** — all five `pc.*` present, so it's a mapping
      gap, not missing kernels (fix is small/additive).
- [x] **Reproduces on latest** — pyarrow 25.0.0 identical to pinned 23.0.1.
- [x] **Duplicate search recorded** — #13285 closed precedent (incomplete); #39691
      is producer-side; novel as an open consumer-coverage report.
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Definition of done
Filed (with #) referencing #13285, or shelved, recorded here + in `README.md` +
`../SUBSTRAIT_ROUNDTRIP.md`. The Substrait fan-out's last coverage wall — closing
it (upstream) takes Acero from 11/22 toward full TPC-H and makes the differential
probe's one-plan-many-engines fan-out fully live.
