# Substrait roundtrip-survival probe — is Substrait a portable lowering for us?

**Instrument:** `../benchmarks/substrait_roundtrip.py`. **Companion to** the
DataFrame-API lowering (`translate_datafusion.py`, `SUBSTRAIT`-free, 22/22 vs
DuckDB) and the `DIFFERENTIAL_PROBE.md` grid.

## The question
`translate_datafusion.py` lowers the lazy-pandas optimized `LogicalPlan` into the
**DataFusion DataFrame API** — a *direct, single-engine* route. DataFusion (and
Acero, DuckDB, Velox…) also consume **Substrait**, a portable relational IR.
Should the lowering target Substrait instead?

The decision rule: **a portable IR pays off exactly when you target >1 engine.**
- For **validating/benchmarking one engine** (what `translate_datafusion` does):
  **no** — Substrait inserts a lossy protobuf hop and its consumers lag the native
  API, so you trade coverage for nothing.
- For the **differential probe** (the mission): **yes, additively** — one
  Substrait plan fanned across N consumers turns every "engine X can't consume
  node Y" and every cross-engine result divergence into a free finding, without N
  hand-written adapters. On-charter because DataFusion + Acero are named
  substrates. (Caveat: Substrait itself is *not* a charter substrate — use it as
  plumbing to multiply findings about the substrates we care about, don't become a
  Substrait-conformance project. Clean consumers today are DataFusion + Acero;
  DuckDB's extension is stale, Polars' is producer-mostly.)

## The experiment
For each of the 22 TPC-H plans: lower once to a DataFusion DataFrame (reusing
`translate_datafusion` internals), take its `logical_plan()`, then

```
Producer.to_substrait_plan → Plan.encode()→bytes → Serde.deserialize_bytes
   ├─ datafusion Consumer.from_substrait_plan → create_dataframe_from_logical_plan → exec
   └─ pyarrow.substrait.run_query(bytes, table_provider) → exec        (2nd engine)
```
compared to DuckDB `PRAGMA tpch(n)`. Each stage is classified (PRODUCE-FAIL /
DF-CONSUME-FAIL / EXEC-FAIL / RESULT-DIVERGE / SURVIVES) so we see *where* a plan
dies.

## Result (datafusion 51.0.0, pyarrow/Acero 23.0.1, SF-0.1)
| roundtrip | survive | meaning |
|---|---|---|
| DataFusion → Substrait(bytes) → **DataFusion** | **22/22** | Substrait IS a viable portable lowering for the DataFusion route — it re-ingests its own Substrait, correct vs DuckDB, in 0.8–5.5 KB/plan |
| DataFusion → Substrait(bytes) → **Acero** | **0/22** | DataFusion's Substrait roundtrips only to *itself* — non-portable |

The 22-vs-0 split is itself the finding. Bisected on trivial plans to a
producer `output_type` omission. **Correction (2026-07-12):** the *scalar*
`output_type` omission (originally AG17) turned out to be **fixed on datafusion
`main`** by PR #20597 — absent only from the 54.0.0 *release* we tested, so AG17
is retracted (RESOLVED-UPSTREAM; a released-artifact mirage). The **live** producer
defect is the aggregate path #20597 missed → **AG20**: `Measure.output_type: None`
+ `phase: UNSPECIFIED` on `main`, causally proven (inject both → Acero consumes),
blocking every aggregating query. (See AG17 doc's post-mortem for the `main`-first
methodology gate this produced.)

## The fix — a local portability pass (`substrait_fixup.py`, `--fix`)
Keep-the-workaround principle: post-process the emitted protobuf to re-derive what
the producer omitted (propagate types bottom-up, fill every missing `output_type`,
stamp aggregate phase; plus two deprecated-field bridges). With `--fix`, **Acero
climbs 0/22 → 11/22 fully correct vs DuckDB** — proving the producer omissions are
the root blockers. The residual 11 peel into *downstream* Arrow/Acero-consumer
layers (all on pinned pyarrow 23.0.1 — re-verify on latest):

| what the fixup does | lifts | class |
|---|---|---|
| fill `ScalarFunction`/`AggregateFunction` `output_type` + agg phase | 7 | DataFusion producer: scalar=fixed on `main` (~~AG17~~ #20597); aggregate=**AG20** |
| downgrade `precision_timestamp` → legacy `timestamp` | +? | AG18 (Arrow consumer reads deprecated field) |
| mirror `FetchRel.count_expr` → deprecated `count` (**silent 0-row LIMIT fix**) | +4 | AG18 (silent-wrong; Acero read `count==0`) |

| residual Acero blocker (not ours to fix) | # | class |
|---|---|---|
| `No conversion function … <ends_with/date_part/starts_with/…>` | 8 | Arrow Substrait fn coverage → **AG19** (5 fns pinned via `substrait_fn_coverage.py`; kernels exist) |
| `join rel's expression must be a simple equality` (cross/non-equi) | 3 | Acero JoinRel limitation |

So one ~200-line consumer-blind patch turns DataFusion's Substrait portable enough
to run **11** TPC-H queries correctly on a *second* engine, and cleanly stratifies
what's left as genuine Arrow-consumer coverage findings rather than our-producer
ones. The standout find en route: a **silent** 0-row result on every `LIMIT` query
because Acero reads Substrait's deprecated `FetchRel.count` (0) while DataFusion
writes only the newer `count_expr`.

## Is any of this a *Substrait* (spec) contribution? — No.
Checked every finding against the spec (substrait-io/substrait). The spec is
correct on all of them; every gap is an **engine** failing to implement it:
- `FetchRel.count` is a `oneof count_mode` and the spec (PR #748 + proto docs)
  says consumers must check the oneof / unset = ALL → the silent 0-rows is an
  **Arrow-consumer** bug, not a spec ambiguity (#748 even pre-flagged the hazard).
- `output_type` is spec-**required** → **DataFusion-producer** bug: scalar case
  fixed on `main` (#20597, retracted AG17); aggregate case still open → **AG20**.
- `starts_with`/`ends_with`/`substring`, `precision_timestamp`, and the
  `DISTINCT` invocation are all standard spec surface → **Arrow-consumer** gaps.
- `date_part`/`regexp_like` aren't standard names → **DataFusion-producer** issue.

So the fixes go to **DataFusion** (AG20 aggregate output_type/phase, canonical names) and **Arrow** (AG18/AG19),
never to Substrait. The only conceivable Substrait-project contribution is
*conformance test vectors* (their consumer-testing effort would catch exactly
these engine bugs) — test infra, not a spec fix. Net: a quiet endorsement of the
Substrait spec's design.

## Standing conclusion
Keep the **DataFrame-API route** as the faithful single-engine oracle
(`translate_datafusion.py`). The **Substrait fan-out layer is now partially live**:
with the `substrait_fixup.py` workaround (`--fix`), one plan runs correctly on
both DataFusion (22/22) and Acero (7/22), and every remaining cross-engine
divergence is a characterized finding (AG18 + the Arrow function-coverage /
cross-join / 0-row layers above). Re-run `substrait_roundtrip.py --acero --fix` on
each datafusion/pyarrow release: the fixup's scalar-`output_type` step auto-retires
once #20597 ships and its aggregate step once **AG20** lands, and the Acero OK
count tracks the Arrow consumer's Substrait coverage as it improves. That rising number is the fan-out
going live — at which point the same instrument starts emitting cross-engine
*result* divergences (the free findings) instead of consumer-coverage ones.
