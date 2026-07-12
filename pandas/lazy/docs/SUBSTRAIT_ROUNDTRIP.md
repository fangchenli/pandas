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

The 22-vs-0 split is itself the finding. Bisected on trivial plans to a concrete,
causally-proven producer defect → **AG17**: `to_substrait_plan` omits the
required `ScalarFunction.output_type` (a regression of closed fixes #15831/#20597;
inject the field → Acero consumes). Second, adjacent: aggregate rels emitted with
phase `UNSPECIFIED`, which Acero rejects.

## Standing conclusion
Keep the **DataFrame-API route** as the faithful single-engine oracle
(`translate_datafusion.py`). Treat **Substrait as an additive fan-out layer** for
the differential probe once AG17 (and the phase omission) are fixed upstream —
until then, DataFusion-produced Substrait can't reach a second engine, so the
fan-out yields findings *about that gap* rather than cross-engine result
divergences. Re-run `substrait_roundtrip.py --acero` on each datafusion/pyarrow
release; when Acero consumption climbs off 0/22, the fan-out becomes live.
