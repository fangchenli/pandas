# Hand-off: AG14 — DataFusion inlines shared subplans (recompute) → a *correctness* gap

**For:** a fresh agent. Read `README.md` first. **Target project: `apache/datafusion`.**
**Priority: 3** — the gap is a known open epic; the deliverable is a **comment /
data-point on #22676**, NOT a PR (the fix is epic-scale). Its one novel angle is
**correctness**; everything else is already understood upstream.

> **STATUS (2026-07-10): REVISED after review (verified against datafusion-python
> 54.0.0 built from local `main`).** Recomputation confirmed; the *correctness*
> failure is now reproduced (flaky, characterized below). Scope of the pitch
> narrowed to the one thing upstream hasn't said. Recommendation: post the
> correctness data point on #22676 with the flakiness caveat, and keep our
> workaround. Gate: human approval to comment.

## The finding (one line)
DataFusion's **logical optimizer inlines a shared subplan at every reference
instead of materializing it once** — so a subplan used twice becomes two
independent `Aggregate` nodes. This is **not** DataFrame-API-specific: an
explicit SQL `WITH rev AS (…)` **CTE** inlines identically (DataFusion has no
auto-materialization; see #8777).

## The novel angle — this is a CORRECTNESS issue, not just perf
Both upstream threads (#22676, #8777) frame subplan reuse as **performance /
efficiency**. Neither frames it as **correctness**. But recompute breaks
**intra-query self-consistency**: `x == max(x)` over the *same* subplan must hold
for the max row, yet if `x` is a **non-deterministic parallel float `SUM`**
recomputed twice, the two copies can combine partials in different orders and
round differently — so the equality never matches. **TPC-H q15** (a spec query)
returns **0 rows** where DuckDB returns **1**. That reframe — "materialization is
required for correctness under exact predicates on non-deterministic aggregates,
not merely an optimization" — is the only thing this hand-off adds upstream.

## Evidence 1 — recomputation (deterministic; datafusion 54.0.0)
Both front-ends produce **3 `Aggregate` nodes** for a twice-used `rev` (`sum`
appears once per reference):

```python
import pyarrow as pa
from datafusion import SessionContext
ctx = SessionContext()
b = pa.RecordBatch.from_arrays([pa.array([1,1,2,2]), pa.array([1.0,2.0,3.0,4.0])], ["k","v"])
ctx.register_record_batches("t", [[b.replace_schema_metadata(None)]])

# SQL CTE — inlined, NOT shared (DataFrame API behaves identically):
sql = ("WITH rev AS (SELECT k, sum(v) AS rev FROM t GROUP BY k) "
       "SELECT r.rev FROM rev r CROSS JOIN (SELECT max(rev) mx FROM rev) m WHERE r.rev = m.mx")
print(ctx.sql(sql).optimized_logical_plan().display_indent())   # -> two `Aggregate: groupBy=[[t.k]]`
```
This proves **recompute only** — on tiny data it returns the correct 1 row; it is
*not* the correctness bug.

## Evidence 2 — the correctness failure (FLAKY, characterized)
The 0-rows failure needs **scale + parallelism**: the two recomputed float `SUM`s
must combine partials in different orders. It is a **race on partial-combine
order**, so it is **not bit-deterministic** — it triggers on a *fraction* of runs
at `target_partitions ≥ 2`, and not at all at `target_partitions = 1`.

```python
import numpy as np, pyarrow as pa
from datafusion import SessionContext, SessionConfig, col, functions as F
rng = np.random.default_rng(0)
N, G = 2_000_000, 500
k = rng.integers(0, G, N).astype("int64")
v = (rng.random(N) * 100).round(2)
big = rng.random(N) < 0.001                    # a few huge values => SUM strongly
v[big] = rng.random(big.sum()) * 1e9           # combine-order dependent
tbl = pa.table({"k": k, "v": v})

def matched(tp):
    ctx = SessionContext(SessionConfig().with_target_partitions(tp))
    ctx.register_record_batches("t", [tbl.to_batches()])
    rev = ctx.table("t").aggregate([col("k")], [F.sum(col("v")).alias("rev")])
    mx  = rev.aggregate([], [F.max(col("rev")).alias("mx")])
    return sum(b.num_rows for b in rev.join_on(mx, how="inner").filter(col("rev") == col("mx")).collect())

fails = sum(matched(8) == 0 for _ in range(20))
print(f"{fails}/20 runs returned 0 rows (correctness failure)")   # observed ~11/20
```
Measured on datafusion 54.0.0: **~50% of runs return 0 rows** at
`target_partitions=8` (and 16) with this magnitude-spread data; **0 failures at
`target_partitions=1`**. With ordinary TPC-H decimals the rate is lower but real —
q15 at SF-0.1 through our lowering failed reliably enough to surface in a single
validation run. **The failure is real and reproducible but flaky; there is no
bit-deterministic trigger** (that is the nature of the race). State this plainly
when posting.

## Upstream state (both OPEN — verified 2026-07-10)
- **#22676 [EPIC] "Subplan Materialization"** (nathanb9, May 2026) — the live
  umbrella; POC PRs **#22551 / #22675**. Tpt's first comment already proposes a
  `Materialized`/`Cached` node for **duplicated subplans (beyond CTEs)** and
  nathanb9 agreed — so "generalize CTE-reuse to any duplicated subplan" is
  **already accepted upstream; do NOT pitch it, it adds nothing.**
- **#8777 "Avoid recompute CTEs / share input plans"** — the deep execution-design
  thread (streaming reuse, an open **join-deadlock** blocker, spill fallback).

## Recommendation — comment on #22676, do NOT file / PR
The fix is epic-scale with an unresolved execution-design blocker (#8777); it is
out of scope for a contribution. Post a short **data point on #22676** leading
with the correctness reframe (the only novel content), honest about the flakiness.

### Draft comment (for #22676)
> A data point in favour of materialization as a **correctness** requirement, not
> only an optimization: when a duplicated subplan contains a non-deterministic
> parallel float `SUM` feeding an exact predicate, recompute can return **wrong
> results**. TPC-H **q15** (`total_revenue = max(total_revenue)` over the same
> revenue view) returns **0 rows** vs DuckDB's 1, because the two recomputed
> `SUM`s combine partials in different orders. Minimal repro (datafusion 54):
> [the Evidence-2 snippet] — ~50% of runs return 0 rows at `target_partitions=8`,
> never at `=1`. Materializing the shared subplan once makes the equality
> self-consistent. (Flaky by nature — it's a partial-combine race — so not
> bit-deterministic.) Happy to expand if useful.

## Our workaround (keep regardless — `benchmarks/translate_datafusion.py`)
Lower each shared node once, materialize to Arrow, and give **each reference a
fresh registered table** (identical values → q15's equality holds; distinct
qualifiers → q2's self-join dodges the #14147 duplicate-field limit). Took the
lowering 21/22 → 22/22 vs DuckDB (SF-0.1 and SF-1).

## Gates
- [x] **Reproduces on latest** — recompute plan (DataFrame + SQL CTE) and the
      flaky 0-rows failure both on datafusion 54.0.0.
- [x] **Correctness failure reproduced + characterized** — flaky (~50% @ tp=8),
      race on partial-combine order, none at tp=1. Not bit-deterministic.
- [x] **Duplicate search** — #22676 (epic, POC #22551/#22675) + #8777; the
      "duplicated-subplan generalization" is already accepted, the **correctness**
      angle is not present in either.
- [ ] **Human approval before commenting** (outward-facing; guardrail).

## Definition of done
Correctness data point posted on #22676 (with the flakiness caveat) — or shelved
if the reviewer judges a flaky repro too weak to post. Keep the materialize-
shared-subplan workaround in `translate_datafusion.py` regardless. Record the
outcome here + in `README.md` + `../ARROW_GAPS.md`.
