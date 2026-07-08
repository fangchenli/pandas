# Hand-off: AG10 — DataFusion `SingleDistinctToGroupBy` doesn't fire for the DataFrame API

**For:** a fresh agent. Read `README.md` first. **Target project: `apache/datafusion`**
(NOT `apache/arrow` like AG1–AG9). **Priority: 2** — sharp, reproducible, single
root cause, likely a small fix.

## The finding (one line)
A grouped `count(DISTINCT col)` built via DataFusion's **Python DataFrame API**
is **~3.9× slower** than the **identical query written in SQL**, because the
`single_distinct_to_groupby` logical-optimizer rule rewrites the SQL plan
(distinct-then-count) but does **not** fire for the DataFrame-API-constructed
plan.

## Evidence (datafusion 51.0.0, TPC-H SF-3 lineitem, 18M rows, 4.5M orderkeys)
Same query — `SELECT l_orderkey, count(DISTINCT l_suppkey) GROUP BY l_orderkey`:

| construction path | time | groups |
|---|---|---|
| DataFrame API (`.aggregate([col("l_orderkey")], [F.count(col("l_suppkey")).distinct().build()])`) | **3518 ms** | 4.5M |
| SQL (`ctx.sql(...)`) | **895 ms** | 4.5M |

Optimized logical plans (the smoking gun):
- **SQL** → a *nested double aggregate*: inner `GROUP BY (l_orderkey, l_suppkey)`
  with no aggregate, then outer `GROUP BY l_orderkey` with a plain
  `count(alias1)`. This is `single_distinct_to_groupby` applied.
- **DataFrame API** → a *single* `Aggregate` with `aggr_expr = count(l_suppkey)
  distinct: true` — **unrewritten**. The rule never fired.

Real-world impact: TPC-H q21 (two such aggregates) was **5.7 s → 1.5 s** once we
rewrote it manually in our lowering (`_rewrite_n_unique` in
`benchmarks/translate_datafusion.py`); that manual rewrite is exactly what the
SQL optimizer does for free.

## Hypothesis (for maintainers to confirm — state as hypothesis, not fact)
The `single_distinct_to_groupby` rule's matcher does not recognize the aggregate
expression shape produced by the DataFrame API's aggregate builder
(`Expr::AggregateFunction { distinct: true, .. }` via the UDF `count`), whereas
the SQL planner produces a shape the rule matches. I.e. the two front-ends emit
logically-equivalent but structurally-different `LogicalPlan`s, and the rule only
matches one. Whether the divergence is in the builder or the rule's pattern is
for maintainers to determine.

## Standalone repro (pure datafusion, no pandas/lazy deps)
```python
import time
import pyarrow as pa
import datafusion as d
from datafusion import SessionContext, col, functions as F

n, g = 18_000_000, 4_500_000          # rows, distinct group keys
import numpy as np                     # only to synthesize data
key = (np.arange(n) % g).astype("int64")
val = (np.arange(n) % 7).astype("int64")
batch = pa.RecordBatch.from_arrays([pa.array(key), pa.array(val)], ["k", "v"])

ctx = SessionContext()
ctx.register_record_batches("t", [[batch]])
t = ctx.table("t")

def tm(fn, warm=1, runs=3):
    for _ in range(warm): fn()
    b = float("inf")
    for _ in range(runs):
        s = time.perf_counter(); fn(); b = min(b, (time.perf_counter()-s)*1000)
    return b

api = lambda: t.aggregate([col("k")], [F.count(col("v")).distinct().build().alias("n")]).collect()
sql = lambda: ctx.sql("SELECT k, count(DISTINCT v) n FROM t GROUP BY k").collect()
assert sum(b.num_rows for b in api()) == sum(b.num_rows for b in sql())   # same result
print("DataFrame API:", round(tm(api)), "ms")
print("SQL          :", round(tm(sql)), "ms")
print("API optimized plan:\n", t.aggregate([col("k")],
      [F.count(col("v")).distinct().build().alias("n")]).optimized_logical_plan())
print("SQL optimized plan:\n",
      ctx.sql("SELECT k, count(DISTINCT v) n FROM t GROUP BY k").optimized_logical_plan())
print("datafusion", d.__version__)
```
Expect: API materially slower; API plan shows a single `distinct: true`
aggregate, SQL plan shows the nested double aggregate.

## Gates (ALL before filing — see README playbook)
- [ ] **Reproduce on latest `datafusion` release** (pin the version; the env here
      is 51.0.0). The rule/builder may already be aligned on a newer release.
- [ ] **Duplicate search** on `apache/datafusion`:
      `gh api -X GET search/issues -f q="repo:apache/datafusion single_distinct_to_groupby dataframe"`
      and phrasings: "count distinct dataframe api slow", "single distinct to
      group by not applied", "optimizer rule dataframe vs sql". Record searches.
- [ ] Standalone repro above runs on a modest machine and shows the gap + the
      two plans. Confirm both front-ends produce the SAME result.
- [ ] **Human approval** before filing (outward-facing; guardrail).

## Filing (apache/datafusion, if gates pass)
- Title: `[Optimizer] single_distinct_to_groupby not applied to DataFrame API count(DISTINCT)`.
- Framing: enhancement/bug — "DataFrame API and SQL produce logically-equivalent
  plans but only SQL gets `single_distinct_to_groupby`; here's a repro + the two
  optimized plans + timings." Offer to help.
- Labels: `enhancement`, `optimizer` (or as the project uses).

## Definition of done
Result (filed with #, or shelved as fixed-on-latest / duplicate) recorded here +
in `README.md` backlog. Keep `_rewrite_n_unique` in our lowering regardless — it
is the local workaround and is independently correct.
