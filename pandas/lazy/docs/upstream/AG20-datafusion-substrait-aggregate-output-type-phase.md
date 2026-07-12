# Hand-off: AG20 — DataFusion Substrait producer omits aggregate `output_type` + `phase` (the live salvage of AG17)

**For:** a fresh agent. Read `README.md` first. **Target project:
`apache/datafusion`** (`datafusion-substrait` producer). **Priority: 2** — the
genuine, **verified-on-main** producer defect that AG17 should have been: the
aggregate path #20597 didn't touch. Blocks every aggregating query on a strict
Substrait consumer. Causally proven. Novel-open.

> **STATUS (2026-07-12): FOUND (salvaged from AG17) + VERIFIED ON MAIN HEAD (not
> just a release) + CAUSALLY PROVEN + DUP-SEARCHED — hand-off-ready (not filed).**
> This is the finding that survives after AG17's scalar-output_type story turned
> out to be an already-merged, not-yet-released fix (#20597). See AG17 for the
> post-mortem + the methodology gate that produced this doc.

## The finding (one line)
DataFusion's Substrait producer emits every aggregate `Measure` with
`output_type: None` **and** `phase: AGGREGATION_PHASE_UNSPECIFIED`
(`aggregate_function.rs`), and the LIKE/ILIKE scalar builder with
`output_type: None` (`scalar_function.rs`) — the paths PR #20597 ("Set Substrait
output types for expressions") fixed for general scalars but **missed**. A strict
consumer (Acero) then can't type the aggregate or accept the phase, so **every
query that aggregates** (≈ all of TPC-H) is unconsumable.

## Verified on main HEAD — the gate AG17 failed (read this first)
Confirmed by reading the **current `main` source** (not a released package — the
mistake that sank AG17's primary claim):

- `datafusion/substrait/src/logical_plan/producer/expr/aggregate_function.rs`
  (the `Measure { measure: AggregateFunction { … } }` builder):
  ```rust
  output_type: None,
  invocation: match distinct { true => Distinct, false => All } as i32,  // set!
  phase: AggregationPhase::Unspecified as i32,                            // not set
  ```
  So `invocation` is set correctly but `output_type` and `phase` are not.
- `datafusion/substrait/src/logical_plan/producer/expr/scalar_function.rs` (~L296,
  L312) — the hand-built `substrait_like` `ScalarFunction` (and its negated
  `not(...)` wrapper) still set `output_type: None`. #20597 fixed the *general*
  scalar path (`to_substrait_type(...)` → `Some(output_type)` at L129/160/339/367,
  with regression test `binary_expr_output_type` at L463) but not this manual
  builder.

Both are **on main today**, unfixed → this is a real live defect, not an
unreleased fix. (Contrast AG17's scalar-output_type primary, which *is* fixed on
main and merely absent from the 54.0.0 release.)

## Causal proof (isolate-and-inject — the proof AG17 admitted it lacked here)
Pure `GROUP BY k → sum(v)` (no scalar functions, so nothing else can confound),
producer datafusion 54.0.0, consumer Acero pyarrow 25.0.0. Injecting each field
into the emitted protobuf and re-running Acero:

| plan | Acero result |
|---|---|
| raw (both omitted) | `Unsupported aggregation phase 'AGGREGATION_PHASE_UNSPECIFIED'` |
| + `phase = INITIAL_TO_RESULT` only | `conversion to arrow::DataType from Substrait type` (now hits the missing `output_type`) |
| + `output_type` only | `Unsupported aggregation phase …` (still) |
| + **both** | **OK, 2 rows** ✓ |

So each omission is independently a blocker and the two together are sufficient —
a clean causal isolation. (The emitted Measure confirms `output_type set = False`,
`phase = 0`.) This omission is **identical on the 54.0.0 release and on main**
(verified in source), so a release-based causal proof is valid here — unlike the
scalar case, where release and main diverge.

## Why it matters
It's the *actual* reason DataFusion-produced Substrait can't fan out to a second
engine: after the scalar fix (#20597) ships, comparison/arithmetic will carry
their `output_type`, but **every aggregate still won't**, and since every
non-trivial analytical query aggregates, the portability wall stays up. The
engine's local workaround (`../../benchmarks/substrait_fixup.py`) already fills
both (`_agg_out` sets `output_type`, phase → `INITIAL_TO_RESULT`), which is what
lifts the aggregate cases 0→OK in the Substrait fan-out — that fixup is the
inline evidence of what the producer should emit.

## Standalone repro (pure datafusion + pyarrow + substrait)
```python
import pyarrow as pa, pyarrow.substrait as pas
from datafusion import SessionContext, col, functions as F
import datafusion.substrait as ss
from substrait.proto import Plan

ctx = SessionContext()
ctx.register_record_batches("A", [[pa.RecordBatch.from_pydict({"k":[1,1,2],"v":[1.,2.,3.]})]])
df = ctx.table("A").aggregate([col("k")], [F.sum(col("v")).alias("m")])
raw = ss.Producer.to_substrait_plan(df.logical_plan(), ctx).encode()
p = Plan(); p.ParseFromString(raw)
m = p.relations[0].root.input.aggregate.measures[0].measure
print("output_type set:", m.HasField("output_type"), "| phase:", m.phase)  # False | 0

def tp(n, s): return pa.Table.from_batches(ctx.table(n[-1]).collect()).select(s.names)
try:
    pas.run_query(pa.py_buffer(raw), table_provider=tp).read_all()          # as-emitted: raises
except Exception as e: print("as-emitted:", str(e)[:55])
m.phase = 3; m.output_type.fp64.nullability = 1                            # inject both
print("after inject:", pas.run_query(pa.py_buffer(p.SerializeToString()),
      table_provider=tp).read_all().num_rows, "rows")                       # OK
```
Report versions. Re-verify `aggregate_function.rs` still omits both on main at
file time (it may get fixed as a follow-up to #20597).

## Dup-search (recorded — apache/datafusion, 2026-07-12)
No open issue tracks the aggregate `output_type`/`phase` omission or the LIKE
`output_type` residual. Hits were unrelated: #12541 (nested expressions), #17910
(multiple grouping sets), #15344 [closed] (empty aggregation functions — a
different bug). #20597 [closed] / #15831 [closed] are the scalar fix that missed
these paths — reference as the sibling fix. Refresh phrasings at file time:
`substrait aggregate measure output_type`, `substrait aggregation phase
unspecified`, `substrait like output_type`, and check open PRs (a #20597
follow-up may be in flight).

## Recommendation
**File one issue** (or offer a PR — the fix mirrors #20597 exactly) on
apache/datafusion: the Substrait producer omits `output_type` + `phase` on
aggregate Measures (`aggregate_function.rs`) and `output_type` on the LIKE builder
(`scalar_function.rs`), which #20597 missed; strict consumers (Acero) can't
consume aggregating plans. Fix: set the Measure's `output_type` from the aggregate
return type and `phase = INITIAL_TO_RESULT` (single-stage); set the LIKE
`output_type` to boolean. Reference #20597 as the precedent + sibling. On-charter
(DataFusion is a named substrate); the direct continuation of the scalar fix.

## Gates
- [x] **Reproduces** — datafusion 54.0.0 producer; emitted Measure
      `output_type=None`, `phase=0`; Acero can't consume; protobuf decoded.
- [x] **Confirmed the defect exists on `main`/HEAD, not just the released
      package** — read `aggregate_function.rs` + `scalar_function.rs:296,312` on
      `main`; both still emit `output_type: None` (and phase `Unspecified`). THIS
      is the gate AG17 skipped.
- [x] **Causally proven** — isolate-and-inject: both fields independently block,
      both together → Acero consumes (table above).
- [x] **Duplicate search recorded** — novel-open; #20597/#15831 are the scalar
      fix that missed these paths.
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Definition of done
Filed (with #) referencing #20597, or shelved, recorded here + in `README.md`.
The finding that survives AG17's post-mortem — same class (producer omits
`output_type`), correctly re-pointed at the paths that are *actually* unfixed on
main, and gated on a main-first source check so it can't be a released-artifact
mirage.
