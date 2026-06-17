# #1 campaign — per-operator decomposition of the worst TPC-H queries

Measurement session (June 2026), SF-1, machine loaded. Harness:
`benchmarks/exp_qgap_decomp.py` (cumulative-prefix timing per stage) +
isolated kernel microbenchmarks. This **localises** where q20/q21 lose before
any build — the discipline that kept us from building Acero / arrow-across-join.

> **Correction recorded in-line:** an early pass mis-attributed q20 to a numpy
> groupby kernel that could be routed to Arrow ("Finding A"). That was an
> apples-to-oranges error (full lazy pipeline incl. filter vs Arrow groupby on
> pre-filtered rows). **The groupby already runs on Arrow** (`groupby[arrow]`
> in the physical plan). Finding A is withdrawn. The corrected findings stand
> below. Probing before building paid off again.

## Headline: the gap is the GROUP-BY AGGREGATE stage, not joins

The seed doc (`FUSION_GAP_DESIGN_SEED.md`) hypothesised joins / cross-breaker
fusion. **The measurement disproves that.** Both worst queries put their
dominant cost in a grouped-aggregate stage; the join chains route fine (all
reach `PhysicalJoinChainSink`, order-free composition) and are cheap
(forest⋈partsupp = 9ms).

## q20 (0.16x) — diffuse plumbing, no single kernel

Dominant stage is `qty`
(`filter + group_by(l_partkey,l_suppkey).agg(0.5*sum(l_quantity))`).
Isolated (SF-1, lineitem 6M → 909K filtered, 543K output groups — very
high-cardinality):

| component | lazy ms | reference ms |
|---|---|---|
| filter + scan only (6M→909K, collect) | ~46 | polars whole query 35 |
| groupby only (pre-filtered, collect) | ~57 | raw Arrow gb + build DF 31 / polars gb 25 |
| **qty full (filter+gb+collect)** | **~105** | **polars 35 (3x)** |

The groupby **already routes to Arrow**. Its ~26ms overhead over raw
`pa.table(...).group_by()` is scan + sink/morsel machinery (output DataFrame
build is only ~2ms). The filter+scan of 6M rows is ~46ms — alone it exceeds
Polars' entire query. So q20's 3x is **diffuse**: ~half filter/scan, ~half
groupby-sink overhead. Harvestable in pieces (~1.3x), but there is no clean
4x and no wrong-kernel to swap.

## q21 (0.27x) — grouped count-distinct: a real substrate kernel gap

Dominant stages are two `n_unique` group-bys: `nsupp` on full lineitem
**411ms lp vs 77ms pl (5.3x)** and `late_nsupp` incremental ~300ms vs 22ms.
Join chain routes fine. Kernel isolation (6M rows, single-key count-distinct):

| path | ms |
|---|---|
| pandas Cython `nunique` (what we use today, the floor) | ~366 |
| lazy full (scan+gb+collect) | ~402 |
| Arrow `hash_count_distinct` | ~448 (slower) |
| **Polars `n_unique`** | **~77** |

We **already use the best pandas/Arrow path**; lazy ≈ the Cython floor. Polars
is 4.7x faster with a parallel count-distinct kernel that neither pandas nor
Arrow exposes. This is the single clean, high-value lever in the whole #1 gap.

### Feasibility probe (de-risking, done)

Count-distinct is embarrassingly parallel if you hash-partition by group key
(a group lands wholly in one partition). But a **naive Python-threaded**
prototype floors at ~247ms (T=8), because:
- the partition step (boolean-mask gather, T full passes) costs 109–135ms, and
- the GIL + Python orchestration kill thread scaling (T=4→8 barely helps).

So parity needs a **nogil Cython kernel**: single-pass radix partition by key
hash + per-partition factorise/dedup across threads, in the mold of
`lazy_radix.pyx` / `lazy_join.pyx`. Idealised budget (T=8): ~30ms partition +
~46ms parallel dedup ≈ ~76ms ≈ Polars. Real, buildable, but a multi-day kernel
project with parity **likely 2–3x, possibly full** — not a plumbing harvest.

## Campaign conclusion / recommendation

- **Joins / cross-breaker fusion: not the gap** (seed hypothesis disproven by
  measurement). Don't build there.
- **q20-class (grouped sum):** diffuse plumbing (filter/scan + sink overhead),
  groupby already Arrow. Small harvestable wins only.
- **q21-class (grouped count-distinct):** the one clean, high-value target — a
  **parallel count-distinct Cython kernel**. Recurs across the suite (q21 ×2),
  proven 4.7x headroom, infra precedent exists (lazy_radix/lazy_join). This is
  the recommended next build, scoped as its own kernel project.

## Constraints carried into any build

- Measurement-first; controlled on/off; never ship unexplained regressions;
  validate full lazy suite + all-22 TPC-H vs DuckDB after every change.
- Group-by output order: Arrow `group_by` returns hash order, already
  reconciled in `_execute_arrow_table_groupby`; any new kernel must match the
  established groupby output contract (tests will gate this).
- nunique stays off the Arrow path (Arrow count_distinct is slower — verified
  pyarrow 23); the new kernel replaces the pandas-Cython hybrid path.
