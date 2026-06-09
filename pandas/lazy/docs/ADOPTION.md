# Adopting the Competitive Directions

How the two directions from
[COMPETITIVE_RESEARCH.md](COMPETITIVE_RESEARCH.md) map onto the existing
prototype. Short version: **everything funnels through the layer we already
own** — the logical plan. Direction 2 adds engines below it; Direction 1
adds a capture layer above it. Neither requires rearchitecting.

```
                    Direction 1 (capture)
   eager pandas calls ──proxy──▶ LogicalPlan ◀──── existing Expr API
                                      │
                            optimizer (existing, 11 passes)
                                      │
                          backend selector (new, per subtree)
                          ┌───────────┼───────────────┐
                          ▼           ▼               ▼
                    eager eval   Arrow/NumPy      DuckDB / …
                    (existing)   engine           Direction 2 (new)
                                 (existing →
                                  semantics fallback)
```

## Direction 2: pluggable execution backends

> **Status: contested.** The author's objection: moving execution to
> engines that are themselves competitors (DuckDB) is not obviously a
> winning strategy — pandas would own the API while ceding the part that
> wins benchmarks, and maintainers may reasonably balk at a hard or soft
> dependency on a competing project. The counter-argument is the
> commoditization thesis (engines are becoming interchangeable
> infrastructure, like compilers — nobody considers LLVM a "competitor").
> This section documents *how* delegation would work if pursued; whether
> to pursue it is an open positioning question, and nothing in Direction 1
> depends on it — capture works identically over our built-in engine.

Technically lower risk, and immediately attacks our worst gaps (join
0.15x, sort 0.2x, aggregation 0.2x vs Polars — all DuckDB strengths);
the prototype is ~90% shaped for it already.

> **Note (June 2026):** those gap figures were the *motivating* pre-engine-work
> numbers. The own-engine cycle has since closed most of them — sort and
> aggregation now beat Polars on several H2O queries and int-key joins are
> ~parity (see `benchmarks/H2O_BENCHMARK.md`). The remaining real gaps
> (string-key/left joins, exact median) are structural. This does not change
> the strategic argument for capture/delegation, but the specific "worst gap"
> numbers are stale.

### Why the prototype is ready

- The **optimized logical plan** is engine-neutral; `PhysicalPlanner`
  already maps it per-node (PLANNING.md) — a delegated subtree is just one
  more mapping target.
- The **`__index__`-column convention** transfers directly: index levels are
  ordinary columns inside any SQL engine, exactly as they are in our
  ArrayDict. `preserve_index=True` survives delegation for free.
- Our **positional-by-default contract** (RangeIndex) is precisely the
  semantics SQL engines provide — the deliberate Polars-style deviations we
  documented turn out to be the delegation-friendly choice.
- **Arrow is the interchange**: `duckdb.from_arrow()` in, Arrow out,
  zero-copy both ways (verified working with duckdb 1.4.3 in the dev env).
- The **equivalence suite** parametrizes over engines already; a third
  engine is one more parameter value.

### Implementation steps

1. **`ExecutionBackend` interface** (`backends/engine.py`):
   `supports(plan_node) -> bool` and
   `execute(plan, sources: dict[str, pa.Table]) -> pa.Table`. Register like
   kernels.
2. **DuckDB translator** (`backends/duckdb_backend.py`): logical plan →
   DuckDB *relational API* (not SQL strings):
   `from_arrow → .filter/.project/.aggregate/.join/.order/.limit`. Our IR
   `Call` nodes render to DuckDB expressions; unsupported functions mark the
   node non-delegable. **Not Substrait yet** — the DuckDB substrait
   extension currently lags DuckDB releases (verified: not installable for
   1.4.3). Keep the translator behind the interface so a Substrait producer
   can slot in later and unlock DataFusion/Acero/Velox.
3. **Subtree delegation in the physical planner**: in
   `PhysicalPlanner._plan_recursive`, greedily find maximal delegable
   subtrees and wrap them in a new `PhysicalDelegated(backend, logical_subtree)`
   leaf. Everything above it continues through our operators. This reuses
   the materialize-boundary machinery; `explain(physical=True)` shows
   `[DELEGATED:duckdb]` the same way it shows `[BREAKER]` — fallback
   visibility built in from day one (the cudf.pandas lesson).
4. **Output schema enforcement**: cast engine output to the plan's resolved
   `Schema` (`pa.Table.cast`). Concrete known case: DuckDB `sum(BIGINT)`
   returns `DECIMAL/HUGEINT` — without the cast, dtypes drift. This also
   resolves our "output dtype instability" known issue in one place.
5. **Selection policy**: reuse the threshold system —
   `compute.lazy.engine` option (`"auto" | "builtin" | "duckdb"`), with
   auto delegating only above a row threshold (engine handoff has fixed
   cost) and only for subtree shapes where DuckDB measured faster
   (join/sort/agg) — the calibration script extends naturally.
6. **Tests**: extend `TestEagerVsPhysicalIndexContract` with
   `engine="duckdb"`; add NaN-payload cases (the NaN→NULL mapping must be
   decided once and enforced for both our Arrow kernels and DuckDB — this
   is the same fix as the known NaN bug).

**Prerequisites from the known-issues list**: ~~fix NaN-vs-null in
aggregation~~ (done — kernels now enforce pandas skipna/dropna semantics)
and pin the output-dtype contract *before* adding a third engine —
otherwise we multiply the divergence surface.

**Spike to prove it** (small): translate
`Filter→Aggregate(group_by)→Sort→Limit` chains only, benchmark
`bench_join.py`/`bench_aggregations.py` shapes against the 0.15–0.2x
gaps. If delegated subtrees don't land within ~2x of Polars, the
commoditization premise fails early and cheaply.

## Direction 1: transparent capture of eager pandas

Higher risk and higher ceiling — and unlike Direction 2, uncontested as a
strategy: it strengthens pandas with no dependence on anyone else's engine.
It runs over whatever execution layer exists (the built-in engine today;
delegated backends later only if Direction 2 is ever adopted). The main
sequencing concern stands regardless: capture is only compelling if the
execution underneath makes captured pipelines visibly faster than plain
eager pandas, so the optimizer-win categories (fusion, pushdown, early
termination, streaming) should headline the first demo.

### Design (each piece has prior art)

1. **Explicit scope first, not import interception**
   (`pandas/lazy/capture.py`):

   ```python
   with pd.lazy.capture() as cap:
       out = df[df.a > 0].groupby("g")["v"].sum()
   # `out` materialized at scope exit / on observation
   print(cap.report())   # what was captured, what fell back, why
   ```

   cudf.pandas-style global interception (`ModuleAccelerator`) is Phase 3,
   only if the scoped form proves out. Scoped capture is honest about its
   boundary and testable.
2. **Proxy objects** wrapping `DataFrame`/`Series`: each supported method
   appends to a `LogicalPlan` instead of executing. The mapping table is
   our existing node set — Modin's result (240 pandas ops → compact
   algebra at >85% coverage) says this table is finite and tractable:
   `__getitem__[mask]`→Filter, `[cols]`→Project, `assign`→with_columns,
   `groupby().agg`→Aggregate, `merge`→Join, `sort_values`→Sort,
   `head`→Limit, …
3. **Materialization triggers** — the eager-semantics leaks, enumerated and
   handled, not discovered in production: `__repr__` (collect via
   TopK/limit — our streaming early termination makes repr ~free; LaFP's
   "Lazy Print"), `len()` (row-count fast path), `.values`/`.to_numpy()`/
   iteration/boolean coercion (full collect), `.iloc` scalar (positional
   slice).
4. **Fallback contract**: unsupported method → materialize inputs, run real
   pandas, rewrap result as a new `DataFrameSource`. CPU-to-CPU, so cheap —
   but **always recorded**: `cap.report()` lists every fallback with the
   triggering method. Silent fallback is the one unsolved problem at NVIDIA
   scale; visibility is our differentiator, and the explain infrastructure
   is built for it.
5. **Pandas semantics mode**: capture mode flips the documented deviations
   back to pandas defaults — `preserve_index` semantics on, groupby
   keys-as-index, NaN behavior matching eager. The semantics audit
   (PROPOSAL.md) is the spec; the `__index__` machinery already implements
   the hard part. *This is why owning the semantics layer matters — it is
   exactly what Ibis-on-pandas and the distributed clones lacked.*
6. **Correctness gate**: Dias-style precondition checks — ops are captured
   only when verified semantics-equivalent; the equivalence harness runs
   captured-vs-pure-eager over test corpora, same pattern as the existing
   eager-vs-physical suite.

### Phasing

| Phase | Scope | Exit criterion |
|---|---|---|
| 2a (after D2 spike) | Capture context manager, ~8 whitelisted ops, loud fallback, report() | captured == eager on the equivalence corpus; a real notebook runs with ≥1 pipeline accelerated |
| 2b | Widen op coverage (Modin algebra map), repr/len fast paths | fallback rate < 20% on sampled real-world notebooks |
| 3 | Import-interception mode (whole-script) | only if 2a/2b show wins users actually keep enabled |

## Sequence summary

1. ~~Fix the NaN-vs-null bug~~ (done) + pin the output dtype contract
2. **D1 Phase 2a**: scoped capture over the existing plan layer — the
   uncontested direction goes first
3. Decide the Direction 2 question (delegate vs build vs neither) with
   maintainers as part of the positioning discussion; if delegation is
   accepted, the spike (DuckDB relational backend for agg/join/sort
   subtrees, benchmarked against the known gaps) is a cheap
   kill-or-commit signal
4. Revisit Substrait when the DuckDB extension catches up (unlocks
   DataFusion/Velox as backends with zero new translation code)
