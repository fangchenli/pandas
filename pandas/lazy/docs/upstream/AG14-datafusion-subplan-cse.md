# Hand-off: AG14 — DataFusion recomputes shared subplans (no common-subplan CSE)

**For:** a fresh agent. Read `README.md` first. **Target project: `apache/datafusion`.**
**Priority: 3** — already tracked upstream, so this is an **attach-a-data-point**
task, NOT a new filing. Our novel contribution is the **correctness** angle (below).

> **STATUS (2026-07-10): TRACKED UPSTREAM — attach data, do NOT file new.** The gap
> is a known open epic; our job is to add a data point that strengthens the case.
> Two open issues cover it:
> - **#22676 [EPIC] "Subplan Materalization"** (sic) — the umbrella.
> - **#8777 "Avoid recompute CTEs (common table expressions) / share input plans"**.
>
> **What neither issue captures — our angle:** the missing CSE is not just a
> *performance* cost (recomputing a reused subtree); when the reused subtree
> contains a **non-deterministic floating-point aggregate** (a parallel `SUM`)
> and its result feeds an **exact equality**, recomputation with different
> rounding produces **WRONG RESULTS**, not just slow ones. We hit this as TPC-H
> **q15 returning 0 rows**. That reframing — CSE absence as a *correctness* issue
> for float aggregates under exact comparison — is the data point to add.

## The finding (one line)
A subplan referenced more than once in a DataFusion **DataFrame-API** plan is
**recomputed**, not shared/materialized — so `rev` used in two branches becomes
two independent `Aggregate` nodes in the optimized plan.

## Evidence 1 — the recompute (standalone, deterministic; datafusion 54.0.0)
```python
import pyarrow as pa
from datafusion import SessionContext, col, functions as F
ctx = SessionContext()
b = pa.RecordBatch.from_arrays([pa.array([1,1,2,2]), pa.array([1.0,2.0,3.0,4.0])], ["k","v"])
ctx.register_record_batches("t", [[b.replace_schema_metadata(None)]])
rev = ctx.table("t").aggregate([col("k")], [F.sum(col("v")).alias("rev")])   # shared subplan
mx  = rev.aggregate([], [F.max(col("rev")).alias("mx")])                      # rev reused
joined = rev.join_on(mx, how="inner").filter(col("rev") == col("mx"))        # cross join + filter
print(joined.optimized_logical_plan().display_indent())
```
Optimized plan — **`rev`'s `Aggregate` appears twice** (recomputed, not shared):
```
Inner Join: rev = mx
  Aggregate: groupBy=[[t.k]], aggr=[[sum(t.v) AS rev]]          <-- rev, copy 1
    TableScan: t projection=[k, v]
  Aggregate: groupBy=[[]], aggr=[[max(rev) AS mx]]
    Projection: rev
      Aggregate: groupBy=[[t.k]], aggr=[[sum(t.v) AS rev]]      <-- rev, copy 2
        TableScan: t projection=[k, v]
```

## Evidence 2 — the correctness impact (TPC-H q15, the novel angle)
q15 ("Top Supplier") builds a per-supplier revenue view `rev =
Σ l_extendedprice·(1−l_discount)` (a **float** `SUM`), then filters
`total_revenue == max(total_revenue)`. `rev` is referenced twice (the value, and
its max). With no CSE, DataFusion recomputes `rev` in both branches; a parallel
float `SUM` is **not associative**, so the two recomputations can round to
slightly different `total_revenue` values → the exact `==` never matches → **0
rows** instead of the correct 1. DuckDB and a materializing engine return 1 row.

This is the sharp point for the epic: **subplan CSE is a correctness requirement,
not only an optimization, whenever a reused float aggregate feeds an exact
predicate.** (A recompute that is bit-identical would be merely slow; a parallel
float aggregate is not guaranteed bit-identical.)

## Our workaround (keep regardless — `benchmarks/translate_datafusion.py`)
Lower each node once (memoize by identity); a node referenced ≥2× is materialized
to Arrow once and **each reference gets a fresh uniquely-named registered table**
over the cached batches. Fixes q15 (identical values → exact equality holds) and
also q2 (a **self-join** of a shared subplan — reusing one DataFrame handle trips
`"Projections require unique expression names"`, the #14147 duplicate-qualified-
field limit; distinct table qualifiers avoid it). Took the lowering from 21/22 →
**22/22** vs DuckDB (SF-0.1).

## Dup-search (recorded — apache/datafusion)
Searched `common subexpression subplan elimination`, `DataFrame reuse recompute
subplan`, `common subplan elimination dataframe`, `shared subplan
nondeterministic float aggregate`. Covered by **#22676** (epic) + **#8777** (CTE
recompute); **no issue frames it as a correctness bug for float aggregates under
exact comparison** — that is the gap our data point fills. Not a new-filing case.

## Recommendation
**Comment on #22676 (and/or #8777)** with: the standalone recompute plan above,
the q15 0-rows correctness manifestation + mechanism (non-associative parallel
float `SUM` recomputed twice ≠ under exact `==`), and note the materialize-once
workaround. Frame as "another motivation for subplan materialization: it is a
correctness requirement, not just perf, for this pattern." Offer the repro.

## Gates
- [x] **Reproduces on latest** — recompute plan on datafusion 54.0.0; q15 0-rows
      on our lowering (pre-fix).
- [x] **Duplicate search recorded** — tracked by #22676 + #8777; correctness
      angle not present in either.
- [x] **Standalone repro** — above, pure datafusion + pyarrow.
- [ ] **Human approval before commenting** (outward-facing; guardrail).

## Definition of done
Data point posted on #22676/#8777 (with #), or shelved, recorded here + in
`README.md` + `../ARROW_GAPS.md`. Keep the materialize-shared-subplan workaround
in `translate_datafusion.py` regardless.
