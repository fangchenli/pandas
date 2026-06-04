# Lazy Pandas Query Optimizer

The optimizer (`pandas/lazy/optimize/`) transforms a logical plan into an
equivalent, more efficient one before execution. It is rule-based: a sequence
of passes, each traversing the plan tree, run iteratively (up to 3 rounds)
until a fixpoint, so one pass can create opportunities for another.

The optimized plan is memoized per `LazyDataFrame`
(`frame.py:_get_optimized_plan`), so repeated `collect()`/`explain()` calls on
the same object pay planning cost once. `collect(optimize=False)` bypasses
optimization for debugging.

## Passes and Default Order

Defined in `optimize/base.py`; implementations in `optimize/passes.py` and
`optimize/engine.py`.

| # | Pass | Effect |
|---|------|--------|
| 1 | `ConstantFolding` | Evaluate constant subexpressions at plan time (`2 + 3` → `5`) |
| 2 | `ExpressionSimplification` | Algebraic identities: `x*1→x`, `x+0→x`, De Morgan's laws, self-cancellation (`x-x→0`, `x==x→True`), idempotence (`x&x→x`) |
| 3 | `DeadCodeElimination` | Remove no-op nodes: `Filter(True)`, identity projections |
| 4 | `FilterFusion` | `Filter(p1, Filter(p2, X))` → `Filter(p1 AND p2, X)` |
| 5 | `PredicatePushdown` | Move filters toward sources (through Project/Join/Concat/file sources) |
| 6 | `AggregatePushdown` | Push aggregates through pass-through projects |
| 7 | `ProjectionPruning` | Drop columns not needed downstream; push column selection into file scans |
| 8 | `LimitPushdown` | Push `Limit` through `Project`; merge consecutive limits |
| 9 | `SortLimitToTopK` | Sort+Limit → TopK: O(n log k) instead of O(n log n) |
| 10 | `EngineSelection` | Analyze backend requirements; insert explicit `Convert` nodes |
| 11 | `ConversionElimination` | Remove redundant `Convert` nodes |

`CommonSubexpressionElimination` is available but not in the default
pipeline (enable explicitly).

### Why This Order

```
FilterFusion         before pushdown: a single fused filter is easier to push
PredicatePushdown    before pruning: filters may reference columns that
                     pruning would otherwise remove
ProjectionPruning    after pushdown: filter columns are now in final position
LimitPushdown        after pruning: fewer columns flow through limits
EngineSelection      last on the logical plan: all operation movement is done,
                     so backend decisions are final
ConversionElimination cleanup of the converts EngineSelection inserted
```

Anti-patterns: pruning before pushdown (drops filter columns), engine
selection before structural passes (operations move after backend decisions).

## Safety Rules

Every pass must be semantics-preserving: same values, column order, dtypes,
and NA propagation. The main correctness machinery:

### Predicate Pushdown: Lineage

A filter above a `Project` may reference pass-through, renamed, or computed
columns. `build_project_lineage()` maps each output name to its source
expression; pushdown then:

- **pass-through column** → push as-is
- **renamed column** → push, rewritten to the original name
- **computed column** → either keep the filter in place (conservative
  default), or rewrite the predicate by inlining the defining expression —
  bounded by a complexity limit so large expressions are not duplicated
  (`rewrite_predicate_through_project`)
- a conjunction mixing pushable and non-pushable parts is split on `AND` and
  pushed partially

Filters never push below an `Aggregate` whose output they reference.

### Predicate Pushdown Through Joins

Using join provenance (output name → side + original name,
see ARCHITECTURE.md "Join Column Disambiguation"):

- predicate references only left columns → push to left input
- only right columns → push to right input
- only join-key columns → push to **both** sides
- mixes non-key columns from both sides → not pushed

### Projection Pruning: Required-Columns Analysis

Pruning is driven top-down by what each node *requires* of its input, not
just the final output: filter predicates, join keys, group-by keys,
aggregation inputs, sort keys, and distinct subsets all contribute.
`Distinct` without a subset requires every column (full-row uniqueness), so
nothing is pruned below it. For joins, downstream (possibly suffixed) names
are mapped back through provenance to per-side required sets. An empty
requirement set keeps at least one column so row count is preserved.

### Expression Reordering: Dependency DAG

Expressions inside a `Project` can reference earlier outputs and can
overwrite the same name. Reordering is only legal within a topological level
of the dependency DAG, where:

- an edge exists from each expression to the latest earlier producer of any
  column it references, and
- successive producers of the **same** output name are chained, preserving
  overwrite order (last-write-wins must stay observable).

### Engine Selection

`optimize/engine.py` models per-operation backend requirements
(`BackendRequirements`: supported / preferred / required backend +
conversion-cost estimate), combines them bottom-up over the expression tree,
picks a dominant backend per plan node, and materializes the decision as
explicit `Convert` nodes in the plan. Benefits of explicit conversion nodes:

- `explain()` shows exactly where format conversions happen
- `ConversionElimination` can cancel `Convert(Convert(x, A), A)` and converts
  into a backend the input already has
- the evaluator treats hints as advisory and may still adapt to actual data

Final row-count-dependent choices (e.g. Arrow vs NumPy filter at runtime) use
the threshold system — see [THRESHOLDS.md](THRESHOLDS.md), including the
experimental EMA-based adaptive tuner (`optimize/adaptive.py`).

## explain()

```python
ldf.explain()                      # logical plan, optimized (default)
ldf.explain(optimized=False)       # plan as written
ldf.explain(format="tree")         # text | tree | json
ldf.explain(physical=True)         # physical plan, marks [BREAKER] nodes
```

JSON output is stable enough for tooling; the physical form shows pipeline
materialization boundaries and chosen backends.

## Testing

- Per-pass unit tests: `tests/lazy/test_optimize.py` (~165 cases)
- **Equivalence tests**: `tests/lazy/test_optimizer_equivalence.py` runs
  query corpora with optimization on/off and physical planner on/off and
  asserts identical results — including NaN/NA semantics and column order.
- Plan-quality regression: `benchmarks/bench_optimizer_quality.py` asserts
  expected structural effects (filters fused, TopK conversions, nodes
  reduced) so refactors can't silently disable a pass.
