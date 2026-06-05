# Logical and Physical Planning

How a lazy query becomes an executable plan: construction of the logical
plan, and the physical planner that turns the optimized logical plan into
executable operators. The optimization passes between the two layers are
covered in [OPTIMIZER.md](OPTIMIZER.md); execution conventions (ArrayDict,
kernels, joins, spilling) in [ARCHITECTURE.md](ARCHITECTURE.md).

```
LazyDataFrame verbs ──build──▶ Logical Plan ──optimize──▶ Optimized Plan
                                                              │
                                              PhysicalPlanner.plan()
                                                              │
                                          ┌───────────────────┴─────┐
                                          │ 1. _plan_recursive      │  node mapping
                                          │ 2. materialize breakers │  boundaries
                                          │ 3. _apply_fusion        │  pipelines
                                          └───────────────────┬─────┘
                                                              ▼
                                                       Physical Plan
                                                              │
                                       execute() / execute_batches(context)
```

## Logical Planning

### Construction

Every `LazyDataFrame` verb is non-executing and immutable: it wraps the
current plan in one new node and returns a new `LazyDataFrame`.

```python
def filter(self, predicate: Expr) -> LazyDataFrame:
    return LazyDataFrame(Filter(self._plan, predicate), self._schema)
```

Schema-changing verbs (`select`, `with_columns`, `group_by().agg`, `join`,
`set_index`, …) call `new_plan.resolve_schema()` **at construction time**, so
schema errors surface where the user wrote the bug, not at `collect()`:
unknown columns, ambiguous post-join references (with a message naming both
suffixed candidates), incompatible `concat` inputs, missing aliases.

### The `LogicalPlan` contract

Every node implements three things (`plan.py`):

| Method | Purpose |
|---|---|
| `resolve_schema()` | Output `Schema`; computed once and cached per node (`_cached_schema`) since physical planning re-reads schemas repeatedly |
| `children()` | Tree traversal for the optimizer and planners |
| `estimate_row_count()` | Best-effort cardinality: sources report lengths, Parquet uses file metadata, `Limit` caps, `Concat` sums; `Filter` returns `None` (unknown selectivity) and `None` poisons upward |

Row estimates exist purely to inform physical decisions (join build-side
selection); they are never required for correctness.

### Optimization and caching

`LazyDataFrame._get_optimized_plan()` memoizes the optimizer's output per
object, so `explain()` followed by `collect()` plans once. Plans are shared
structurally (nodes are immutable), so reusing a `LazyDataFrame` as the base
for several queries duplicates nothing.

## Physical Planning

`PhysicalPlanner.plan(logical_plan)` (in `physical.py`) runs three steps.

### 1. Node mapping (`_plan_recursive`)

A per-node-type translation; mostly 1:1, with execution strategy baked into
the physical node:

| Logical | Physical | Strategy decided at plan time |
|---|---|---|
| `DataFrameSource` | `PhysicalScan` | array extraction, index → `__index__` cols |
| `ParquetSource` | `PhysicalParquetScan` | pushed predicate → pyarrow filter expression, row-group statistics skipping, column subset |
| `CSVSource` | `PhysicalCSVScan` | streaming-friendly batch reads |
| `Project` | `PhysicalProject` | parallel expression evaluation above threshold |
| `Filter` | `PhysicalFilter` | — (backend chosen at runtime, see below) |
| `Aggregate` | `PhysicalHashAggregate` | input wrapped in `PhysicalMaterialize` |
| `Sort` | `PhysicalSort` | input materialized; quicksort default |
| `TopK` (from optimizer) | `PhysicalTopK` | partial-sort selection |
| `Limit` | `PhysicalLimit` | streaming early termination (`head`); `tail` non-streaming |
| `Distinct` | `PhysicalDistinct` | input materialized |
| `Join` | `PhysicalHashJoin` | both sides materialized; left/right row estimates attached for build-side choice |
| `Convert` (from optimizer) | `PhysicalConvert` | backend conversion point |
| `Concat` | `PhysicalConcat` | sequential batch streaming across inputs |
| `SetIndex`/`ResetIndex` | `PhysicalSetIndex`/`PhysicalResetIndex` | sets `user_set_index` semantics |

### 2. Explicit materialization boundaries

Inputs to pipeline breakers (sort, aggregate, distinct, both join sides) are
wrapped in `PhysicalMaterialize(input, reason)`. This is deliberate
bookkeeping, not an extra copy — it makes every spot where streaming must
stop *visible in the plan*, which buys:

- `explain(physical=True)` marks `[BREAKER]` nodes honestly
- one place to hang spill management (materialize is where memory pressure
  concentrates)
- correct fusion boundaries (fusion never crosses a materialize)
- a natural location for backend conversion

### 3. Operator fusion (`_apply_fusion`)

A bottom-up post-pass over the physical tree. `_try_fuse` walks chains of
adjacent `PhysicalFilter` / `PhysicalProject` / `PhysicalLimit` (head only —
`tail` is excluded since it needs the end of the stream) and collapses them
into one `PhysicalFusedPipeline` holding an ordered `FusedOperation` list.
An existing fused pipeline encountered mid-chain is absorbed rather than
nested. Single-operation chains are left alone (no benefit).

A fused pipeline processes each batch through all its operations before
touching the next batch — filter masks gate expression evaluation, and an
embedded limit counts rows across batches and stops the upstream scan:

```
Scan → Filter(a>0) → Project(a, b*2) → Limit(100)
   becomes
Scan → FusedPipeline([Filter(a>0), Project(a, b*2), Limit(100)])
```

This is what makes `scan(...).filter(...).head(n)` terminate early without
any operator knowing about its neighbors.

## The `PhysicalPlan` execution contract

Every physical node implements:

| Member | Contract |
|---|---|
| `execute(context) → ArrayDict` | full materialization of this subtree |
| `execute_batches(context) → Iterator[ArrayDict]` | streaming; default implementation yields `execute()` as one batch |
| `supports_streaming` | `True` only if the node processes batches incrementally |
| `is_pipeline_breaker` | `True` if all input must be consumed before any output |
| `output_schema` | schema, mirrors the logical node |

The streaming driver (`frame.py:_collect_streaming`) simply iterates
`physical_plan.execute_batches(context)` and converts each `ArrayDict` batch
to a DataFrame. Breakers don't break the *driver* — a `PhysicalSort` in a
streaming plan consumes its input inside `execute()` and then yields its
sorted result as batches; the boundary is internal and marked in `explain`.

## Plan-time vs run-time decisions

A deliberate split: the planner fixes plan *shape*; data-dependent choices
stay at runtime where actual sizes are known.

**Fixed at plan time** — operator selection (Sort+Limit→TopK happened in the
optimizer), materialization boundaries, fusion grouping, scan pushdowns,
join row estimates.

**Decided at run time** (via `ExecutionContext` + the threshold system, see
[THRESHOLDS.md](THRESHOLDS.md)):

- Filter backend: Arrow vs NumPy by row count (`filter_arrow_threshold`)
- GroupBy backend: Arrow vs pandas-Cython by rows/cardinality thresholds
- Hash join build side: smaller side by estimate, re-checked against
  actual materialized sizes
- Grace hash join: only when a `spill_manager` is present; after
  partitioning, **skew statistics are inspected and the join falls back to
  `PhysicalSortMergeJoin`** when partitions are pathological (high skew
  ratio or a partition exceeding the memory budget)
- Expression parallelism: thread pool only above `parallel_expr_threshold`
- NumExpr fusion of arithmetic subtrees by array size

`ExecutionContext` carries this state: preferred backend, strict mode,
threshold config, index metadata (`index_names`, `user_set_index`,
`preserve_index`), batch size, spill manager, worker count, and a CSE cache.

## Known limitations (honest notes)

- `_choose_backend_for_exprs` currently just returns the user preference —
  per-expression plan-time backend analysis is a TODO; today backend choice
  is effectively runtime-threshold-driven. (The *logical* `EngineSelection`
  pass does insert `Convert` nodes, so the two layers overlap; unifying them
  is on the roadmap.)
- Hash join materializes **both** sides; streaming the probe side is a known
  future optimization.
- Row estimates stop at filters (no selectivity model) — see
  [ROADMAP.md](ROADMAP.md) on cardinality estimation.
