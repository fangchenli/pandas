# Hand-off: AG17 — ~~DataFusion omits `ScalarFunction.output_type`~~ **RESOLVED UPSTREAM (not filing)**

**Target project:** `apache/datafusion`. **STATUS: NOT FILEABLE — already fixed
on `main`.** The live salvage moved to **`AG20-datafusion-substrait-aggregate-output-type-phase.md`**.

> **STATUS (2026-07-12): RESOLVED-UPSTREAM / DO NOT FILE.** The scalar-function
> `output_type` omission was fixed by **PR #20597 "fix: Set Substrait output types
> for expressions"** (merged to `main` 2026-05-27, with a regression test
> `binary_expr_output_type`). It is **not a regression and not an incomplete fix**
> — it's a merged fix that simply hadn't shipped in the **54.0.0** release this
> probe tested (branch-54 was cut before #20597 merged; it wasn't backported).

## Post-mortem — why this doc was wrong (the methodology lesson)
This finding failed a check it never ran, and it's the important takeaway:

1. **Tested the release, concluded about `main`.** Verified on the pip package
   `datafusion 54.0.0` and marked "reproduces on latest" `[x]`. But 54.0.0
   (tagged 2026-06-03) is `diverged` from #20597's merge commit
   (`77240f9f`, 2026-05-27) — `git merge-base` shows the fix is **not** an
   ancestor of the tag. The released binary lacks the fix; `main` has it. "Still
   omits on current / regression persists" is true of the *artifact* pip-installed
   and **false of the codebase**. Confirmed by reading `main`:
   `scalar_function.rs` now sets `output_type: Some(...)` (L129/160/339/367) with
   the regression test at L463.
2. **Misread the precedent as "fixed twice."** #15831 is the *issue*; #20597 is
   the *PR* that closed it — one recent fix, same day. The "regression with two
   closed-fix precedents (AG11/AG16 class)" framing rested on that misread and
   collapses.
3. **The causal proof was solid but aimed at an already-solved target** — the
   inject-`output_type`→Acero-consumes repro proves the value of a fix that
   already exists.

**The gate this should have had (now standard — see `README.md` §2):** *any*
"still broken / regression / incomplete-fix" claim must be verified against the
repo's `main`/HEAD source, not the released package. Released version answers "is
it shipped"; only `main` answers "is it fixed." Check with `git grep` on `main` +
`git merge-base --is-ancestor <fix-commit> <release-tag>`. AG20 applies this gate.

## What survives → AG20
The **aggregate** path #20597 didn't touch is the live finding: on current `main`,
aggregate Measures still emit `output_type: None` + `phase: Unspecified`
(`aggregate_function.rs`), and the LIKE/ILIKE builder still emits
`output_type: None` (`scalar_function.rs:296,312`). Since every real query
aggregates, *that* is what blocks Acero on `main`. Causally proven + verified on
`main` in **`AG20-datafusion-substrait-aggregate-output-type-phase.md`**.

---
*Historical record of the (resolved) scalar finding follows.*

## How it was found (the experiment)
The question: *of the 22 TPC-H plans that lower cleanly to the DataFusion
**DataFrame API** today (`translate_datafusion.run`, 22/22 vs DuckDB), how many
survive a roundtrip through **Substrait** — the portable IR that DataFusion,
Acero, DuckDB… all consume?* Instrument:
`../../benchmarks/substrait_roundtrip.py` (a sibling of `differential_probe.py`).

Two headline numbers, and they are the finding:
- **22/22 survive the DataFusion→Substrait(bytes)→DataFusion roundtrip** and
  still match DuckDB. So Substrait is a viable *portable lowering for the
  DataFusion route* — DataFusion happily re-ingests its own Substrait.
- **0/22 are consumable by Acero** (the same Substrait bytes, a *second*
  compliant engine). DataFusion's Substrait only roundtrips **to itself**.

That 22-vs-0 split is the tell: DataFusion's producer emits Substrait that its
*own* consumer tolerates (it re-derives missing pieces) but that a second engine
rejects — i.e. **non-portable Substrait**, which defeats the entire point of the
format. Bisecting the Acero failures on trivial plans isolated the dominant
cause.

## The finding (one line)
DataFusion's Substrait producer emits every `Expression.ScalarFunction`
(comparisons **and** arithmetic) with **no `output_type` field set**. Substrait
marks `output_type` **required**; Acero's consumer needs it and fails with a
blank-typed error, so any DataFusion-produced plan containing a scalar function
(≈ every real query — every filter/arithmetic) is unconsumable downstream.

## Evidence (datafusion 51.0.0)
Decoding the emitted protobuf — `scalar_function.HasField("output_type")` is
**False** across the board, via *both* front-ends:

| plan (front-end) | scalar fns | `output_type` set |
|---|---|---|
| DataFrame `filter(col("k") > 1)` (compare) | 1 | `[False]` |
| DataFrame `select(col("v") + col("k"))` (arith) | 1 | `[False]` |
| DataFrame `filter((col("v")*2) > col("k"))` (nested) | 2 | `[False, False]` |
| **SQL** `WHERE k > 1` | 1 | `[False]` |
| **SQL** `SELECT v + k` | 1 | `[False]` |

It is **not** DataFrame-API-specific (unlike AG10) — SQL omits it too, so it's in
the shared expression-serialization path.

Acero's resulting error (pyarrow 23.0.1) is the uninformative
`conversion to arrow::DataType from Substrait type ` (blank — Acero's
`FromProto(Type)` hits its `default:` on the unset `kind` oneof).

## Causal proof (this is *the* blocker, not a symptom)
On the minimal `filter(col("k") > 1)` plan, a bare scan and a scan+project
**consume into Acero fine** (i64/fp64/string leaf types convert). Adding the
filter breaks it. Injecting the single missing field into the emitted protobuf —
`scalar_function.output_type.bool.nullability = NULLABILITY_NULLABLE` — makes
Acero **consume the identical plan and return the correct 2 rows**. So the
omitted `output_type` is the sole filter-path blocker. (Repro below.)

## Local workaround built + measured (strengthens the finding)
`../../benchmarks/substrait_fixup.py` is the engine's keep-the-workaround fix: it
post-processes the emitted protobuf to re-derive what the producer omitted —
propagate types bottom-up and fill every missing `ScalarFunction.output_type`,
stamp aggregate phase `INITIAL_TO_RESULT`, plus the two portability downgrades
below. Applying it (`substrait_roundtrip.py --acero --fix`) takes **Acero
consumption from 0/22 → 11/22 fully correct vs DuckDB**
(q1/3/4/5/6/10/12/17/18/19/21), with the `conversion to arrow::DataType` error
gone entirely. That a ~200-line consumer-blind protobuf patch — filling exactly
the producer's omissions — flips 0→11 is direct evidence the omissions are the
root blockers, not symptoms.

The remaining 11 stratify into **downstream Arrow/Acero-consumer** gaps (separate
from this DataFusion-producer finding, and all on pinned **pyarrow 23.0.1** — the
playbook says re-verify these on the latest pyarrow before treating as findings):
- **AG18 (Arrow) — the consumer reads *deprecated* Substrait fields.**
  → **Now promoted to its own self-contained hand-off:
  [`AG18-arrow-substrait-fetchrel-limit.md`](AG18-arrow-substrait-fetchrel-limit.md)**
  (root cause re-derived to a vendored-proto version lag; `precision_timestamp`
  dropped as resolved-upstream). The notes below are kept as provenance.
  Two manifestations, both worked around in the fixup, both because Acero's
  Substrait consumer lags the spec's deprecated→new field moves:
  - *`precision_timestamp`* (the type/literal that *deprecated* legacy
    `timestamp`) → `substrait literal did not have any literal type set` (an
    error). Fixup: opt-in `legacy_timestamp` downgrade → `timestamp` (ns→µs).
    **⚠ RESOLVED at Arrow HEAD (gate 2026-07-24 — see "GATE CLOSED" below):
    `type_internal.cc` now handles `precision_timestamp` both ways; this arm is
    floor-debt, NOT a live Arrow gap. Drop from any AG18 filing.**
  - ***`FetchRel.count_expr` (silent-wrong, worst kind) — an Arrow-consumer bug,
    NOT a Substrait gap.*** `count` lives in a `oneof count_mode { count,
    count_expr }`; the spec (proto docs + PR #748) says consumers must check the
    oneof and that an unset `count` means **ALL**. DataFusion sets the `count_expr`
    arm; Acero ignores the oneof, reads the deprecated `count` as its default 0,
    and treats that as `LIMIT 0` → **silently returns 0 rows**, no error (TPC-H
    q3/q10/q18/q21). So Arrow is wrong *twice* (ignores the oneof; treats unset as
    0 not ALL). Fixup mirrors the literal into the deprecated arm. Highest-severity
    item here — a silent wrong result across every LIMIT query — and a clean Arrow
    bug (the spec anticipated exactly this ambiguity and handled it).
- **Arrow function coverage (AG9-class):** `No conversion function exists to
  convert the Substrait function <ends_with|date_part|starts_with|regexp_like|…>
  to an Arrow call expression` (8 queries) — Acero registers only a subset of
  Substrait functions. Genuinely unimplemented in Acero → not workaround-able by
  us; an upstream-Arrow coverage item.
- **Acero cross/non-equi join:** `A join rel's expression must be a simple
  equality between keys` (3 queries) — Acero's Substrait JoinRel rejects anything
  but key-equality, so cross joins are unconsumable. A genuine Acero limitation.

## What "desired behavior" rests on (be honest, but this one is strong)
Unlike the NumPy finds (AG15/AG16, where docs were silent), this has a
**documented-contract basis plus precedent plus cross-engine demonstration** —
three independent legs:
1. **Spec**: the Substrait `Expression.ScalarFunction.output_type` field is
   documented **Required** ("The output type of the scalar function"). Omitting a
   required field is a producer defect regardless of any consumer's leniency.
   *(Re-confirm against the current substrait.io spec text at file time.)*
2. **Maintainer precedent (the AG11 pattern)**: DataFusion already fixed this
   exact field twice —
   - **#15831 [closed] "Ensure Substrait producer for `BinaryExpr` includes
     `output_type`"**
   - **#20597 [closed] "fix: Set Substrait output types for expressions"**

   yet 51.0.0 still omits it on comparison **and** arithmetic scalar functions,
   via both front-ends → a **regression or incomplete fix**, not a design choice.
3. **Cross-engine, policy-independent**: DataFusion's Substrait roundtrips only to
   itself (22/22) and to no second engine (0/22 Acero). Whatever the spec says,
   Substrait whose only valid consumer is its own producer is not portable — the
   demonstrable defect the experiment measured.

**Posture:** file as a **regression/bug** (strongest of the three legs is the
precedent — reference #15831/#20597), with the cross-engine repro as proof and
the spec line as the contract.

## Second, adjacent finding (same class — record together, likely its own issue)
DataFusion's producer also emits aggregate rels with the **aggregation phase
unset** → defaults to `AGGREGATION_PHASE_UNSPECIFIED`; Acero rejects with
`Unsupported aggregation phase 'AGGREGATION_PHASE_UNSPECIFIED'` (it needs a
concrete phase, e.g. `INITIAL_TO_RESULT`). Same "producer omits a field a
compliant consumer requires" class; separable from AG17's scalar-function issue.
Not yet causally isolated the way `output_type` was — a follow-up.

## Standalone repro (pure datafusion + pyarrow, no pandas)
```python
import pyarrow as pa, pyarrow.substrait as pas
from datafusion import SessionContext, col
import datafusion.substrait as ss
from substrait.proto import Plan, Type

ctx = SessionContext()
a = pa.RecordBatch.from_pydict({"k": [1, 2, 3, 1], "v": [1.0, 2.0, 3.0, 4.0]})
ctx.register_record_batches("A", [[a]])
raw = ss.Producer.to_substrait_plan(
    ctx.table("A").filter(col("k") > 1).logical_plan(), ctx).encode()

p = Plan(); p.ParseFromString(raw)
sf = p.relations[0].root.input.filter.condition.scalar_function
print("output_type set?", sf.HasField("output_type"))           # False  <-- the bug

def tp(names, schema): return pa.table(
    {"k": [1, 2, 3, 1], "v": [1.0, 2.0, 3.0, 4.0]}).select(schema.names)

try:                                                             # as-emitted: fails
    pas.run_query(pa.py_buffer(raw), table_provider=tp).read_all()
except Exception as e:
    print("Acero as-emitted:", str(e)[:55])                     # conversion to arrow::DataType from ...

sf.output_type.bool.nullability = Type.Nullability.NULLABILITY_NULLABLE  # inject it
out = pas.run_query(pa.py_buffer(p.SerializeToString()), table_provider=tp).read_all()
print("Acero after inject: rows =", out.num_rows)               # 2  <-- fixed
```
Versions in the report: datafusion producer, pyarrow/Acero consumer, `substrait`
python pkg. Re-run on the latest datafusion release before filing.

## Dup-search (recorded — apache/datafusion + apache/arrow)
- **#15831 [closed]** and **#20597 [closed]** — the two prior output_type fixes
  (see leg 2); the current state is a regression/incomplete fix relative to them.
  **The report should reference both** and show 51.0.0 still omitting it.
- apache/arrow #40614 (closed, "Running TPCH queries in Acero through Substrait")
  and #33566 are Acero-consumer-side context, not this producer defect.
- Search more phrasings at file time: `substrait producer scalar function
  output_type`, `substrait output_type missing regression`, `to_substrait_plan
  output type`, and check open PRs.

## Recommendation — DO NOT FILE (resolved)
The scalar `output_type` omission is **fixed on `main`** by #20597; filing it
would report an already-merged fix. Take instead the live sibling — **AG20**
(aggregate `output_type`/`phase` + LIKE `output_type`), which #20597 missed and
which is verified unfixed on `main`.

## Gates (corrected)
- [x] **Reproduced on the 51.0.0 / 54.0.0 *releases*** — but that only proved the
      *release artifact* omits it, not the codebase (the error below).
- [x] **~~Reproduces on the latest~~ → CORRECTED: fixed on `main`.** #20597 (merged
      2026-05-27) sets `output_type` in `scalar_function.rs` on `main`; the 54.0.0
      *tag* (2026-06-03) is `diverged` from that commit (`merge-base` not an
      ancestor). "Regression persists on current" was a claim about the pip
      package, not `main`. **Verified by reading `main` source.**
- [x] **Causally proven** — inject → Acero consumes. Solid work, but aimed at an
      already-solved target.
- [~] **Duplicate search** — MISREAD: #15831 is the *issue*, #20597 the *PR* that
      closed it (one fix, not two precedents). The "regression / two closed fixes"
      framing is retracted.
- [x] **NEW GATE (now standard, `README.md` §2): defect confirmed on `main`/HEAD,
      not just the release** — this doc FAILED it; AG20 passes it.

## Note on the AG18/AG19 (Arrow-consumer) findings
Those were verified on **released** pyarrow 25.0.0, not Arrow C++ `main`. By the
same lesson, before filing AG18/AG19, run the `main`-first gate against
`apache/arrow` HEAD (the consumer registration / FetchRel-oneof handling may have
landed since the release). Their repros stand on the released artifact; their
"still unfixed" status needs the source check.

> ### ✅ GATE CLOSED (2026-07-24) — read `apache/arrow` HEAD `62d2dd8270`
> Ran the `main`-first gate by reading the C++ consumer source at HEAD (not the
> released pkg). Results — **two still live, one already fixed:**
>
> | claim | HEAD verdict | evidence (main @ `62d2dd8270`) |
> |---|---|---|
> | **AG18(a)** `FetchRel` silent-`LIMIT 0` | **✅ CONFIRMED LIVE — root cause sharpened** | `relation_internal.cc:782-786` reads `int64_t count = fetch.count();` — a **scalar** accessor. **The real root cause is a substrait-version lag, not "ignores a oneof":** Arrow pins **substrait v0.44.0** (Arrow's `versions.txt`, `ARROW_SUBSTRAIT_BUILD_VERSION=v0.44.0`), whose `FetchRel` has only `int64 count = 4` (no expression form exists yet). DataFusion (main) uses **substrait-rs 0.63.0** and its producer emits `count_mode = CountMode::CountExpr(..)` (`datafusion/substrait/.../producer/rel/fetch_rel.rs`). Arrow's v0.44.0 proto has no `count_expr` field → it lands in protobuf **unknown-fields** and is dropped; `fetch.count()` returns the scalar default **0** → `FetchNodeOptions(offset, 0)` → **LIMIT 0 → silent 0 rows**. Fix scope is therefore *not* a small registry add: **bump the vendored substrait proto (≥~0.61, where `count_mode`/`offset_mode` land) + teach the FetchRel consumer to read `count_expr`/`offset_expr`, eval the literal, unset count = ALL.** Bigger/riskier (a proto bump ripples across all rels) but fixes a silent-wrong-result bug. |
> | **AG18(b)** no `precision_timestamp` support | **❌ RESOLVED at HEAD — DROP from filing** | `type_internal.cc:136-145` (consumer `FromProto`) handles `kPrecisionTimestamp`/`kPrecisionTimestampTz`; `:328,:334` (producer `ToProto`) emit them. Fixed upstream since our pyarrow-25 read. **This is an AG17-class released-artifact mirage — must NOT appear in any AG18 filing.** Our `legacy_timestamp` fixup is now floor-debt, not an Arrow gap. |
> | **AG19** string-fn coverage (`starts_with`/`ends_with`/`substring`) | **✅ CONFIRMED LIVE** | `extension_set.cc` `DefaultExtensionIdRegistry` is the complete consumer map; it registers exactly **one** `kSubstraitStringFunctionsUri` name — `"concat"` (line 1148). No `starts_with`/`ends_with`/`substring`/`like`/`match_substring`. Kernels (`pc.starts_with`, …) exist → additive registry fix. Unchanged. |
>
> **Dup check (2026-07-24):** non-dup — `gh search` finds **zero** Arrow issues on
> substrait string-function coverage or on `FetchRel`/`count_mode`. The referenced
> umbrella **#13285** ("register tricky Substrait functions with the consumer") was
> **auto-closed as stale 2025-12-05** (365-day stalebot), *not* fixed — so the one
> issue that could have covered AG19 lapsed without landing anything.
>
> **Net:** AG18 narrows to the single silent-`LIMIT 0` bug (its highest-severity
> manifestation anyway); AG19 stands as scoped. Both now pass the `main`-first
> gate and are filing-ready **pending human go-ahead** (guardrail).

## Definition of done
**Done: RESOLVED-UPSTREAM, not filing.** Recorded here + in `README.md`; live work
handed to `AG20-datafusion-substrait-aggregate-output-type-phase.md`. The durable
value of AG17 is the methodology gate it forced (test `main`, not the release).
