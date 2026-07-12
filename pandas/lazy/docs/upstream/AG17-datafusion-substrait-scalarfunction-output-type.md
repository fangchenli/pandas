# Hand-off: AG17 — DataFusion's Substrait producer omits `ScalarFunction.output_type` (breaks portability)

**For:** a fresh agent. Read `README.md` first. **Target project:
`apache/datafusion`** (the `datafusion-substrait` crate; surfaced via
`datafusion-python`). **Priority: 2** — a broad, clean, causally-proven producer
defect with **two closed-fix precedents** (regression/incomplete-fix class, like
AG11/AG16); it makes DataFusion-produced Substrait unconsumable by any other
compliant engine. File-worthy after a latest-version re-check.

> **STATUS (2026-07-11): FOUND BY THE SUBSTRAIT ROUNDTRIP-SURVIVAL PROBE +
> ROOT-CAUSED + CAUSALLY PROVEN + DUP-SEARCHED — hand-off-ready (not filed).**
> Verified on **datafusion 51.0.0** producer, consumed by **pyarrow/Acero
> 23.0.1** and DataFusion's own consumer. Nothing filed without go-ahead
> (guardrail).

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
stamp aggregate phase `INITIAL_TO_RESULT`. Applying it (`substrait_roundtrip.py
--acero --fix`) takes **Acero consumption from 0/22 → 7/22 fully correct vs
DuckDB** (q1/4/5/6/12/17/19), with the `conversion to arrow::DataType` error gone
entirely. That a ~150-line consumer-blind protobuf patch — filling exactly the
producer's omissions — flips 0→7 is direct evidence the omission is the root
blocker, not a symptom.

The remaining 15 stratify into **downstream Arrow/Acero-consumer** gaps (separate
from this DataFusion-producer finding, and all on pinned **pyarrow 23.0.1** — the
playbook says re-verify these on the latest pyarrow before treating as findings):
- **AG18 (Arrow, candidate):** Acero's Substrait consumer doesn't support
  `precision_timestamp` (the type/literal that *deprecated* legacy `timestamp`) —
  `substrait literal did not have any literal type set`. The fixup's opt-in
  `legacy_timestamp` downgrade (→ `timestamp`, ns→µs) works around it; that's what
  lifts 5 of the 7.
- **Arrow function coverage (AG9-class):** `No conversion function exists to
  convert the Substrait function <ends_with|date_part|starts_with|…> to an Arrow
  call expression` (8 queries) — Acero registers only a subset of Substrait
  functions.
- **Acero cross/non-equi join:** `A join rel's expression must be a simple
  equality between keys` (3 queries) — Acero's Substrait JoinRel rejects anything
  but key-equality, so cross joins are unconsumable.
- **0-row correctness divergence:** 4 queries consume + execute but return 0 rows
  vs DuckDB — a separate execution-semantics divergence to investigate.

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

## Recommendation
**File one focused issue** on apache/datafusion, framed as a **regression** of
#15831/#20597: `to_substrait_plan` emits `ScalarFunction` with no `output_type`
(comparison + arithmetic, DataFrame + SQL) on 51.0.0, making the output
unconsumable by Acero; reference the two closed fixes + the causal repro. Fix:
set `output_type` when serializing `ScalarFunction` in the producer (the type is
already known from the logical plan's schema). Mention the aggregation-phase
omission as a likely-related second issue. This is **on-charter** (DataFusion is a
named substrate) and directly serves the differential probe: a portable-Substrait
producer would let one lowering fan across DataFusion + Acero + DuckDB, turning
every cross-engine result divergence into a free finding.

## Gates
- [x] **Reproduces** — datafusion 51.0.0 producer; `HasField("output_type")` False
      on all scalar functions (both front-ends); Acero 23.0.1 consumer fails;
      protobuf decoded.
- [x] **Causally proven** — inject `output_type` → Acero consumes the identical
      plan → correct rows. The field is the sole filter-path blocker.
- [x] **Duplicate search recorded** — two closed precedents (#15831, #20597);
      current state is a regression/incomplete fix; novel as an open report.
- [ ] **Re-verify on the latest datafusion release** at file time (51.0.0 may not
      be newest) — the fix may already be in-flight.
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Definition of done
Filed (with #) referencing #15831/#20597, or shelved, recorded here + in
`README.md`. The Substrait roundtrip-survival probe's first genuine find — and
the answer to "is Substrait a portable lowering for us?": **yes for the
DataFusion route (22/22), no across engines yet — and this is why.**
