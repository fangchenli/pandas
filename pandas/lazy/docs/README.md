# Lazy Pandas Documentation

Lazy pandas adds deferred, optimized query execution to pandas. Instead of
executing each operation eagerly, it builds a logical query plan that is
optimized and executed only when results are requested.

## Quick Start

```python
import pandas as pd
from pandas.lazy import col, scan

# From an existing DataFrame
df = pd.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
result = (
    df.select()                      # enter lazy mode
    .filter(col("a") > 1)
    .select((col("a") * 2).alias("a2"), col("s").str.upper().alias("S"))
    .collect()                       # execute, returns DataFrame
)

# From files (Parquet/CSV, with predicate + projection pushdown)
ldf = scan("data/*.parquet").filter(col("value") > 100).select("id", "value")
result = ldf.collect(use_physical_planner=True)

# Inspect what will run
print(ldf.explain())                 # logical plan (text | tree | json)
print(ldf.explain(physical=True))    # physical plan with pipeline boundaries
```

## Document Map

| Document | Contents |
|----------|----------|
| [PROPOSAL.md](PROPOSAL.md) | **Start here for a design review** — motivation, API tour, status, open questions for maintainers |
| [examples.py](examples.py) | Runnable end-to-end tour (`python pandas/lazy/docs/examples.py`) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design: expression IR, logical/physical plans, array-based execution, backends, joins, streaming, spilling |
| [PLANNING.md](PLANNING.md) | Logical plan construction and the physical planner: node mapping, materialization boundaries, operator fusion, plan-time vs run-time decisions |
| [ENGINE_DESIGN.md](ENGINE_DESIGN.md) | **The engine** (all milestones landed/gated, June 2026): morsel-driven pipeline architecture — principles, decision layer, parallel sinks, measured gates and the disproven-hypothesis trail |
| [OPTIMIZER.md](OPTIMIZER.md) | Query optimizer: passes, ordering rationale, safety rules, engine selection |
| [KERNELS.md](KERNELS.md) | Kernel reference: which operations run on which backend (Arrow/NumPy/Bottleneck) and how they perform |
| [THRESHOLDS.md](THRESHOLDS.md) | Cost-based decision thresholds: configuration, calibration, adaptive tuning |
| [ROADMAP.md](ROADMAP.md) | Known gaps, open questions, and future work |
| [COMPETITIVE_RESEARCH.md](COMPETITIVE_RESEARCH.md) | Researched answer to "can this compete with Polars/DuckDB?" — prior-art post-mortems, ranked directions |
| [ADOPTION.md](ADOPTION.md) | How the competitive directions map onto the prototype: pluggable engine backends, then transparent capture — phased plan |
| [../benchmarks/README.md](../benchmarks/README.md) | Benchmark suite: how to run, methodology, key findings |

## Module Layout

```
pandas/lazy/
├── __init__.py       # Public API: col, lit, scan, LazyDataFrame
├── expr.py           # Expr API (user-facing expression builder)
├── ir.py             # Expression IR nodes (FieldRef, Literal, Call, ...)
├── plan.py           # Logical plan nodes (Project, Filter, Join, ...)
├── types.py          # LazyDtype / Schema (dual NumPy+Arrow dtype tracking)
├── frame.py          # LazyDataFrame: collect(), explain(), plan caching
├── eval.py           # Pandas-based evaluator (fallback path)
├── scan.py           # Lazy file scanning (Parquet, CSV)
├── physical.py       # Physical planner and operators
├── cost.py           # Engine cost model: decision constants + provenance
├── engine/           # Pipeline engine: graph compiler, executor,
│                     # decision layer, morsel parallelism (ENGINE_DESIGN.md)
├── optimize/         # Query optimizer (passes, engine selection, config)
├── backends/         # Kernel registry, router, Arrow/NumPy kernels,
│                     # memory pools, NumExpr fusion, spilling
├── benchmarks/       # Standalone benchmark suite
└── docs/             # This documentation
```
