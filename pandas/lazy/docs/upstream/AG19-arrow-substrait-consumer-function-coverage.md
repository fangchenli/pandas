# Hand-off: AG19 — Arrow's Substrait consumer doesn't map standard string functions (kernels exist)

> **Scope note (revised after a spec check):** of the 5 functions that first
> looked like one Arrow gap, only **3 are standard Substrait** (`starts_with`,
> `ends_with`, `substring`) and belong to Arrow; the other 2 (`date_part`,
> `regexp_like`) are **non-canonical names DataFusion emits** and go to the
> producer. See "Scope check first" below. This doc is now narrowly the
> Arrow-consumer piece.

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

## Scope check first — "are these even Substrait's job?" (they split 3 ways)
Before blaming the consumer, the honest question (raised in review): are these
five functions *standard Substrait* at all? Checked against the spec's function
extensions (`substrait-io/substrait` `extensions/functions_string.yaml` +
`functions_datetime.yaml`). They split into **three distinct findings**, and only
the first is really an Arrow-consumer coverage gap:

| function | in the Substrait spec? | who's at fault | class |
|---|---|---|---|
| `starts_with` | **yes** (`functions_string.yaml`) | **Arrow consumer** — unmapped even when correctly anchored (test below) | AG19 core |
| `ends_with` | **yes** (`functions_string.yaml`) | **Arrow consumer** | AG19 core |
| `substring` | **yes** (`functions_string.yaml`) | **Arrow consumer** | AG19 core |
| `regexp_like` | **NO** — canonical is `regexp_match_substring` | **DataFusion producer** emits a non-canonical name | producer (→ AG17 theme) |
| `date_part` | **NO** — canonical is `extract` (component arg) | **DataFusion producer** emits a non-canonical name | producer (→ AG17 theme) |

Plus a **third, orthogonal producer gap** found while checking: DataFusion emits
*every* function extension **unanchored** — `extension_urn_reference = 4294967295`
(the uint32 "unset" sentinel) and no URN/URI declared in the plan. So even the
standard-named functions arrive with no pointer to the YAML that defines them.

So the reviewer's instinct was right for `date_part`/`regexp_like`: those aren't
standard Substrait, so a compliant consumer *can't* be expected to map them — the
fix is on DataFusion (emit `extract` / `regexp_match_substring`), not Arrow.

## The finding (one line, corrected)
Arrow's Substrait **consumer** has no mapping for three **standard** Substrait
string functions — `starts_with`, `ends_with`, `substring` — so plans using them
fail with `No conversion function exists to convert the Substrait function <X> to
an Arrow call expression`, **even though Arrow implements the kernels**
(`pc.starts_with`, `pc.ends_with`, `pc.utf8_slice_codeunits`) **and even when the
function is correctly anchored to `functions_string.yaml`** (attribution test
below). The other two originally-lumped-in functions (`regexp_like`, `date_part`)
are DataFusion-producer non-canonicality, tracked separately.

## Attribution test (separates consumer gap from producer anchoring)
`starts_with` fails on Acero as-emitted (unanchored). Injecting the standard
`functions_string.yaml` URN into the plan and pointing the function at it — so it
is *correctly anchored* — **still** yields `No conversion function … starts_with`.
So Acero genuinely lacks the mapping; it isn't merely DataFusion's missing anchor.
(Caveat: URN-vs-URI proto naming skews across datafusion 54 / pyarrow 25 / the
`substrait` pkg; corroborate by enumerating Arrow's `default_extension_id_registry`
at file time.)

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
| **NO-CONVERSION** | 5 | standard-but-unmapped: **`starts_with`, `ends_with`, `substring`** (AG19 core) · non-canonical producer names: **`regexp_like`, `date_part`** |
| **other** | 1 | `count(distinct)` — `Unsupported aggregation invocation 'AGGREGATION_INVOCATION_DISTINCT'` (a *separate* consumer gap; DISTINCT invocation IS in Substrait scope, so a real Acero gap) |

The gap is coherent: arithmetic, comparison, logical, `in_list`, `case_when` and
the standard aggregates all map fine; it's specifically the **string-predicate /
substring** family (standard) plus the two non-canonical temporal/regex names.

## The kernels already exist — so the consumer-side fix is a pure mapping (cheap)
For the three **standard** functions, Arrow already has the kernel — only the
Substrait-function→Arrow-compute registration is missing:

| standard Substrait fn | existing Arrow compute kernel |
|---|---|
| `starts_with` | `pc.starts_with` |
| `ends_with` | `pc.ends_with` |
| `substring` | `pc.utf8_slice_codeunits` |

Confirmed present (`hasattr(pc, …) == True`). No kernel work — only registering
the mapping in the consumer's extension registry (the table #13285 extended). The
direct sibling of the Arrow **string-view kernel coverage** contribution (AG9): the
modern surface is incomplete on exactly the string ops. (The canonical
`regexp_match_substring` → `pc.match_substring_regex` and `extract` →
`pc.year`/temporal kernels also exist — so *if* DataFusion emitted the canonical
names, they'd be the same easy consumer-registration. But as-emitted they're the
producer issue below.)

## The producer side (`date_part`, `regexp_like`, + unanchored) — DataFusion, not Arrow
- `date_part` and `regexp_like` are **not** Substrait standard functions; the
  canonical spec names are `extract` (with a component argument) and
  `regexp_match_substring`. DataFusion emits the non-canonical names, so **no**
  compliant consumer maps them — this is a DataFusion Substrait-producer
  canonicality gap (fits the AG17 "producer emits non-portable Substrait" theme),
  not an Arrow coverage gap. Fix is on DataFusion (emit the canonical functions).
- **All** function extensions are emitted **unanchored**
  (`extension_urn_reference = 4294967295`, no URN declared) — a third producer gap:
  even the standard-named functions carry no pointer to their defining extension.
- These belong in the AG17 hand-off's "producer portability" scope; recorded here
  so the split is explicit and AG19 stays narrowly the Arrow-consumer piece.

## Adjacent consumer gap (record together, likely its own issue): `count(distinct)`
Acero's Substrait consumer rejects the DISTINCT aggregate invocation:
`Unsupported aggregation invocation 'AGGREGATION_INVOCATION_DISTINCT'`. This IS in
Substrait scope (`AggregateFunction.invocation`), and Arrow has `count_distinct`,
so again the kernel exists and only the invocation→kernel path is missing — a real
Acero consumer gap, same class as the string functions above.

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
  of Substrait functions but did **not** cover the standard `starts_with`/
  `ends_with`/`substring` (still NO-CONVERSION on 25.0.0). Same incomplete-coverage
  / sibling-miss pattern as AG9/AG11 — reference it.
- **#39691 [closed] "pyarrow.compute.Expression.to_substrait() is missing
  conversions"** — *producer* side (Arrow → Substrait), the opposite direction;
  not this.
- Refresh at file time: `substrait consumer function extension starts_with`,
  `No conversion function Substrait Arrow call`, `substrait date_part substring
  consumer`, and check open PRs.

## What "desired behavior" rests on (be honest)
`starts_with`/`ends_with`/`substring` **are** in `functions_string.yaml`, Arrow
implements the kernels, and the mapping is still missing even when the function is
correctly anchored — so this is a genuine **coverage/enhancement** gap, framed as
"please register the mapping", not a correctness bug. #13285 shows maintainers
treat "register more Substrait functions" as in-scope. Crucially, `date_part`/
`regexp_like` are **not** in the spec, so this hand-off does **not** ask Arrow to
map them — that would be asking Arrow to support a non-standard name; those go to
DataFusion.

## Recommendation (two targets — keep them separate)
1. **apache/arrow (this hand-off, AG19):** register consumer function-extension
   mappings for the three **standard** functions `starts_with`/`ends_with`/
   `substring` (kernels exist — table above), + the `count(distinct)` DISTINCT
   invocation. Small additive registry change; reference #13285. This alone lifts
   the TPC-H `LIKE`/`substring` predicates.
2. **apache/datafusion (folds into AG17's producer theme):** emit **canonical**
   Substrait — `extract` instead of `date_part`, `regexp_match_substring` instead
   of `regexp_like` — and **anchor** function extensions to their defining URN
   (currently `extension_urn_reference` is the unset sentinel). Without this, those
   ops are non-portable to *any* compliant consumer, not just Acero.

## Gates
- [x] **Reproduces** — coverage matrix on pyarrow 23.0.1 AND 25.0.0
      (`substrait_fn_coverage.py`).
- [x] **Scope-checked against the Substrait spec** — `starts_with`/`ends_with`/
      `substring` are standard (`functions_string.yaml`); `date_part`/`regexp_like`
      are NOT (canonical: `extract` / `regexp_match_substring`) → split to the
      producer. Prevents mis-filing a non-standard name as an Arrow gap.
- [x] **Attribution test** — `starts_with` still NO-CONVERSION on Acero even when
      anchored to `functions_string.yaml` → genuine consumer gap (caveat: URN/URI
      proto skew; corroborate via Arrow's `default_extension_id_registry`).
- [x] **Kernels-exist confirmed** — `pc.starts_with`/`ends_with`/
      `utf8_slice_codeunits` present → mapping gap, not missing kernels.
- [x] **Reproduces on latest** — pyarrow 25.0.0 identical to pinned 23.0.1.
- [x] **Duplicate search recorded** — #13285 closed precedent (incomplete); #39691
      is producer-side; novel as an open consumer-coverage report.
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Definition of done
Filed (with #) referencing #13285, or shelved, recorded here + in `README.md` +
`../SUBSTRAIT_ROUNDTRIP.md`. The Substrait fan-out's last coverage wall — closing
it (upstream) takes Acero from 11/22 toward full TPC-H and makes the differential
probe's one-plan-many-engines fan-out fully live.
