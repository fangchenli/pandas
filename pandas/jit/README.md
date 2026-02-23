# pandas.jit — JIT Compilation for pandas

`pandas.jit` traces pandas operations at runtime, builds a relational
intermediate representation (IR), and compiles the plan to
[Substrait](https://substrait.io/) for execution on an optimized backend
(DataFusion, PyArrow Acero, or a pure-pandas interpreter). Operations that
cannot be represented in the IR trigger *graph breaks* — the traced computation
is materialized and tracing resumes automatically.

## Quick start

```python
import pandas as pd

@pd.jit.compilable
def process(df):
    df["revenue"] = df["price"] * df["quantity"]
    big = df[df["revenue"] > 500]
    return big.sort_values("revenue", ascending=False).head(10)

result = process(sales_df)
```

`@pd.jit.compilable` can also be called with a specific backend:

```python
from pandas.jit import PandasBackend

@pd.jit.compilable(backend=PandasBackend())
def process(df):
    return df[df["price"] > 100]
```

## Architecture

```
User function
     │
     ▼
TracedDataFrame / TracedSeries   (proxy objects)
     │  intercept pandas API calls
     ▼
IR graph  (ReadTable → Filter → AddColumn → Sort → Limit → ...)
     │
     ├──► SubstraitCompiler  → Substrait protobuf  → DataFusionBackend (datafusion)
     │
     ├──► SubstraitCompiler  → Substrait protobuf  → AceroBackend (pyarrow)
     │
     └──► PandasBackend      → direct pandas execution (fallback)
```

### Modules

| Module | Purpose |
|--------|---------|
| `ir.py` | IR types: `DType`, `Schema`, `IRNode` subclasses, `Expr` subclasses, `explain_ir`/`explain_expr` |
| `compiler.py` | `SubstraitCompiler`, backends, `ExecutionPlan`, `ConnectedPlan`, `SchemaGuard` |
| `jit.py` | `@compilable` decorator, `Tracer`, `DeferredScalar`, all `Traced*` proxy classes |

### Backends

| Backend | How it works | When used |
|---------|-------------|-----------|
| `DataFusionBackend` | Compiles IR → Substrait protobuf → `datafusion.SessionContext` | Default when `datafusion` is installed |
| `AceroBackend` | Compiles IR → Substrait protobuf → `pyarrow.substrait.run_query()` | Fallback when only pyarrow is installed |
| `PandasBackend` | Interprets the IR tree using pandas operations directly | Fallback, or when explicitly requested |

`default_backend()` selects the best available: DataFusion > Acero > Pandas.
All Substrait-based backends fall back to PandasBackend on errors.

## Supported operations — fully traced

These operations are captured in the IR and compiled to Substrait / executed on
any backend without materialization.

### DataFrame operations

| pandas API | IR node | Notes |
|-----------|---------|-------|
| `df[col]` | `ColRef` | Column access returns `TracedSeries` |
| `df[[col1, col2]]` | `Project` | Column subset |
| `df[bool_series]` | `Filter` | Boolean indexing with `TracedSeries` |
| `df["new"] = expr` | `AddColumn` | Add/replace column |
| `df.assign(col=expr)` | `AddColumn` | Functional style; supports callables that return `TracedSeries` |
| `df.sort_values(by)` | `Sort` | Single or multi-column, ascending/descending |
| `df.head(n)` | `Limit` | Take first N rows |
| `df.iloc[:n]` | `Limit` | Equivalent to head |
| `df.iloc[start:stop]` | `Limit(offset=start)` | Offset + count without materialization |
| `df.nlargest(n, col)` | `Sort` + `Limit` | Descending sort then limit |
| `df.nsmallest(n, col)` | `Sort` + `Limit` | Ascending sort then limit |
| `df.query("price > 100")` | `Filter` | Parses query string via AST; supports comparisons, arithmetic, `and`/`or`, `~` |
| `df.drop(columns=[...])` | `Project` | Keep remaining columns |
| `df.rename(columns={...})` | `RenameColumns` | Column renaming |
| `df.dropna(subset=[...])` | `Filter(NOT IS_NULL)` | Chained null checks per column |
| `df.fillna(value)` | `AddColumn(COALESCE)` | Per-column or scalar |
| `df.astype(dtype)` | `AddColumn(CastExpr)` | For recognized pandas/numpy dtypes |
| `df.merge(right, on=...)` | `Join` | inner/left/right/outer; single or composite keys |
| `df.where(cond, other)` | `AddColumn(IfThenExpr)` | Conditional replacement per column |
| `df.mask(cond, other)` | `AddColumn(IfThenExpr)` | Inverse of where |
| `df.drop_duplicates()` | `Distinct` | When `keep="first"` and subset covers all columns |
| `df.cumsum()` | `Window` | Expanding window with sum |
| `df.cummax()` | `Window` | Expanding window with max |
| `df.cummin()` | `Window` | Expanding window with min |
| `df.cumprod()` | `Window` | Expanding window with product |
| `df.rolling(w).mean()` | `Window` | See window functions below |
| `df.expanding().sum()` | `Window` | Unbounded window |
| `pd.concat([df1, df2])` | `Union` | UNION ALL; N-ary; mixed traced + raw DataFrames |
| `df.groupby(by).sum()` | `Aggregate` | See aggregations below |
| `df.shift(n)` | `Window(lag/lead)` | Positive n → lag, negative → lead; all columns shifted |
| `df.reset_index(drop=True)` | no-op | Relational algebra has no row index |
| `df.copy()` | no-op | Returns same proxy |
| `df.pipe(func)` | pass-through | Calls `func(self)`, stays traced if func uses traced API |

### Series operations

| Method | IR expression | Returns |
|--------|---------------|---------|
| `series > x` (and `>=`, `<`, `<=`, `==`, `!=`) | `BinOp` | `TracedSeries[BOOL]` |
| `series + x` (and `-`, `*`, `/`) | `BinOp` | `TracedSeries` |
| `series & other`, `series \| other` | `BinOp("and"/"or")` | `TracedSeries[BOOL]` |
| `~series` | `UnaryOp("not")` | `TracedSeries[BOOL]` |
| `-series` | `UnaryOp("negate")` | `TracedSeries` |
| `abs(series)` / `series.abs()` | `UnaryOp("abs")` | `TracedSeries` |
| `series.isin(values)` | `SingularOrList` | `TracedSeries[BOOL]` |
| `series.isna()` | `UnaryOp("is_null")` | `TracedSeries[BOOL]` |
| `series.notna()` | `NOT(IS_NULL)` | `TracedSeries[BOOL]` |
| `series.fillna(val)` | `BinOp("coalesce")` | `TracedSeries` |
| `series.between(lo, hi)` | `AND(GTE, LTE)` | `TracedSeries[BOOL]` |

### Aggregation functions

Available on `groupby()`, `groupby()[col]`, and `groupby()[[cols]]`:

| Method | Substrait function | Notes |
|--------|-------------------|-------|
| `sum()` | `sum:i64` | Numeric columns only |
| `mean()` | `avg:i64` | Returns FLOAT64 |
| `min()` | `min:i64` | Preserves source dtype |
| `max()` | `max:i64` | Preserves source dtype |
| `count()` | `count:any` | Returns INT64; includes non-numeric columns |
| `std()` | `std_dev:fp64` | Returns FLOAT64 |
| `var()` | `variance:fp64` | Returns FLOAT64 |
| `size()` | `count:any` | Returns single column named `"size"` |
| `agg({col: func})` | varies | Dict mapping columns to function names |
| `first()` | — | **Graph break** (no Substrait equivalent) |
| `last()` | — | **Graph break** (no Substrait equivalent) |
| `rank()` | — | **Graph break** on `groupby()[col]`; returns pandas Series |

### DeferredScalar — series-level aggregations

Series-level aggregations (`series.sum()`, `series.mean()`, etc.) return a
`DeferredScalar` — a lazy proxy that stays symbolic as long as it's used in
arithmetic with traced objects. This avoids graph breaks for common patterns
like normalization:

```python
@pd.jit.compilable
def normalize(df):
    df["pct"] = df["price"] / df["price"].sum()   # no graph break!
    return df
```

Under the hood, `DeferredScalar` wraps a `ScalarSubquery` expression that maps
to Substrait's `Expression.Subquery.Scalar`. The subquery is inlined into the
plan and executed by the backend in a single pass.

If Python needs the actual value (e.g., `if total > 0:`, `print(total)`), the
scalar materializes on demand.

| Method | Returns | Stays lazy in arithmetic |
|--------|---------|------------------------|
| `series.sum()` | `DeferredScalar` | Yes |
| `series.mean()` | `DeferredScalar` | Yes |
| `series.min()` | `DeferredScalar` | Yes |
| `series.max()` | `DeferredScalar` | Yes |
| `series.count()` | `DeferredScalar` | Yes |
| `series.std()` | `DeferredScalar` | Yes |
| `series.var()` | `DeferredScalar` | Yes |

### Window functions

Available on `df.rolling(window)` and `df.expanding()` for numeric columns:

| Method | Window spec | Notes |
|--------|------------|-------|
| `rolling(w).mean()` | `rows(w-1, 0)` | Fixed-width trailing window |
| `rolling(w).sum()` | `rows(w-1, 0)` | |
| `rolling(w).std()` | `rows(w-1, 0)` | |
| `rolling(w).var()` | `rows(w-1, 0)` | |
| `rolling(w).min()` | `rows(w-1, 0)` | |
| `rolling(w).max()` | `rows(w-1, 0)` | |
| `rolling(w).count()` | `rows(w-1, 0)` | |
| `expanding().mean()` | `rows(unbounded, 0)` | Unbounded lower bound |
| `expanding().sum()` | `rows(unbounded, 0)` | |
| `expanding().std()` | `rows(unbounded, 0)` | |
| `expanding().var()` | `rows(unbounded, 0)` | |
| `expanding().min()` | `rows(unbounded, 0)` | |
| `expanding().max()` | `rows(unbounded, 0)` | |

Both support `partition_by` and `order_by` via the `Window` IR node.

### Positional window functions

| Method | Window function | Notes |
|--------|----------------|-------|
| `df.shift(n)` (n > 0) | `lag(col, n)` | NaN-fills first n rows |
| `df.shift(n)` (n < 0) | `lead(col, abs(n))` | NaN-fills last n rows |
| `df.shift(0)` | `lag(col, 0)` | Identity |

### Rank window functions

The IR and compiler support `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()` as
parameterless window functions with `ORDER BY` and optional `PARTITION BY`.
At the tracer level, `series.rank()` and `groupby().rank()` currently
graph-break because window function results can't compose as scalar
expressions in `assign()` / `filter()`.

| Substrait function | pandas equivalent | Notes |
|-------------------|-------------------|-------|
| `rank:` | `rank(method="min")` | Ties get lowest rank, gaps after ties |
| `dense_rank:` | `rank(method="dense")` | Ties get lowest rank, no gaps |
| `row_number:` | `rank(method="first")` | No ties, order of appearance |

### Cumulative functions

DataFrame-level cumulative functions are traced as expanding windows:

| Method | Window function | Notes |
|--------|----------------|-------|
| `df.cumsum()` | `sum` over `rows(unbounded, 0)` | Numeric columns only |
| `df.cummax()` | `max` over `rows(unbounded, 0)` | |
| `df.cummin()` | `min` over `rows(unbounded, 0)` | |
| `df.cumprod()` | `product` over `rows(unbounded, 0)` | |

Series-level cumulative (`series.cumsum()`) graph-breaks because Window IR
is a relational operator that can't be composed as an expression in `assign()`.

### Datetime accessor (`.dt`)

Traced via `FunctionCall("extract", ...)`. Supported on all backends.

| Property | Substrait component |
|----------|-------------------|
| `.dt.year` | `YEAR` |
| `.dt.month` | `MONTH` |
| `.dt.day` | `DAY` |
| `.dt.hour` | `HOUR` |
| `.dt.minute` | `MINUTE` |
| `.dt.second` | `SECOND` |
| `.dt.quarter` | `QUARTER` |
| `.dt.dayofweek` / `.dt.day_of_week` | `MONDAY_DAY_OF_WEEK` |
| `.dt.dayofyear` / `.dt.day_of_year` | `DAY_OF_YEAR` |

### String accessor (`.str`)

Traced via `FunctionCall("str_*", ...)`.

| Method | Substrait function |
|--------|-------------------|
| `.str.upper()` | `upper:str` |
| `.str.lower()` | `lower:str` |
| `.str.strip()` | `trim:str` |
| `.str.lstrip()` | `ltrim:str` |
| `.str.rstrip()` | `rtrim:str` |
| `.str.len()` | `char_length:str` |
| `.str.contains(pat)` | `contains:str_str` |
| `.str.startswith(pat)` | `starts_with:str_str` |
| `.str.endswith(pat)` | `ends_with:str_str` |
| `.str.replace(pat, repl)` | `replace:str_str_str` |
| `.str.slice(start, stop)` | `substring:str_i32_i32` |

## Graph-break operations

These operations materialize the current IR, run the operation in plain pandas,
then resume tracing with a fresh `ReadTable` from the result. They work
transparently — the user doesn't need to do anything special.

### DataFrame graph breaks

| Operation | What happens |
|-----------|-------------|
| `len(df)`, `df.shape`, `df.empty` | Materialize, return scalar |
| `df.values`, `df.to_numpy()` | Materialize, return array |
| `df.iloc[int]`, `df.iloc[[0,1,2]]` | Materialize, return slice (fancy/scalar indexing) |
| `df.loc[...]` | Materialize, return slice |
| `df.iterrows()`, `df.itertuples()` | Materialize, iterate |
| `df.apply(func)` | Materialize, apply, re-wrap result |
| `df.pipe(func)` | Tries traced first; materializes on failure |
| `df.to_csv()`, `df.to_parquet()` | Materialize, write |
| `df.describe()`, `df.info()`, `df.value_counts()` | Materialize, return |
| `df.drop_duplicates(subset=[...])` | Materialize when partial subset or `keep != "first"` |
| `df.diff(n)` | Materialize, compute differences, re-wrap |
| `df.rank(...)` | Materialize, rank all columns, re-wrap |
| `df.reset_index(drop=False)` | Materialize when index is meaningful (non-default RangeIndex) |
| `df.set_index(keys)` | Materialize, set index, re-wrap |
| `df.pivot_table(...)` | Materialize, pivot, re-wrap |
| `df.melt(...)` | Materialize, melt, re-wrap |
| `df.stack()`, `df.unstack()` | Materialize, reshape |

### Series graph breaks

| Operation | What happens |
|-----------|-------------|
| `series.apply(func)` | Materialize, apply |
| `series.map(func)` | Materialize, map |
| `series.unique()` | Materialize, return array |
| `series.nunique()` | Materialize, return int |
| `series.value_counts()` | Materialize, return Series |
| `series.cumsum()` etc. | Materialize (Window IR is relational) |
| `series.shift(n)` | Materialize, shift values |
| `series.diff(n)` | Materialize, compute differences |
| `series.rank(...)` | Materialize, rank values |
| `series.reset_index(drop=False)` | Materialize, returns DataFrame |
| `bool(series)`, `series.item()` | Materialize scalar value |
| `len(series)` | Materialize, return int |

After a graph break, tracing continues automatically:

```python
@pd.jit.compilable
def f(df):
    deduped = df.drop_duplicates(subset=["id"])    # graph break
    return deduped[deduped["price"] > 100]          # tracing resumes with filter
```

## What cannot be implemented

Some operations fundamentally cannot be traced into relational algebra and will
always require graph breaks. These are architectural limitations, not missing
features.

### Arbitrary Python UDFs

`apply()`, `map()`, and `transform()` accept arbitrary Python callables that
operate row-by-row or element-by-element. These cannot be expressed in
Substrait or any relational algebra IR. They will always materialize.

### Row iteration

`iterrows()`, `itertuples()`, and `for row in df` are inherently row-iteration
APIs with no relational equivalent.

### Data-dependent schemas

`pivot_table()`, `pivot()`, `unstack()`, and `crosstab()` produce output
columns determined by the actual data values (e.g., unique categories become
column names). Substrait plans require a fixed schema at compile time, so
these operations cannot be expressed without materializing first.

`melt()` and `stack()` (the inverses) have similar issues with Substrait's
lack of an UNPIVOT operator.

### Index-based operations

The IR uses flat column names with no concept of row index. Operations
that depend on row labels — `.loc`, label-based alignment, index joins,
and MultiIndex — will always require materialization. `set_index()` and
`reset_index(drop=False)` are supported but graph-break transparently.

### Operations requiring row count

`tail(n)` and `iloc[start:]` (open-ended slices) need to know the total
number of rows to compute the offset. Since the IR doesn't track cardinality,
these materialize. (Contrast with `head(n)` and `iloc[:n]` which work without
knowing row count.)

### Chained comparisons in query strings

The query parser supports single comparisons and `and`/`or` conjunctions, but
rejects chained comparisons like `1 < x < 10`. Use `df.query("x > 1 and x < 10")`
instead.

### Operations without Substrait equivalents

| Operation | Why it can't be compiled |
|-----------|------------------------|
| `groupby().first()` / `last()` | No first/last aggregation function in Substrait |
| `resample()` | Temporal bucketing has no Substrait equivalent |

### Missing accessor methods

Many `.dt` and `.str` methods are not yet traced. They fall through to pandas
via graph break but produce correct results.

**Not-yet-traced `.dt` methods:** `date`, `time`, `nanosecond`, `microsecond`,
`is_month_start/end`, `is_quarter_start/end`, `is_year_start/end`,
`is_leap_year`, `tz`, `freq`, `normalize()`, `strftime()`, `tz_localize()`,
`tz_convert()`, `floor()`, `ceil()`, `round()`, `total_seconds()`.

**Not-yet-traced `.str` methods:** `split`, `rsplit`, `extract`, `findall`,
`match`, `pad`, `center`, `zfill`, `wrap`, `get`, `join`, `cat`, `repeat`,
`normalize`, `encode`, `decode`, `translate`, `capitalize`, `title`,
`swapcase`, `isnumeric`, `isalpha`, `isdigit`, `isspace`, `islower`,
`isupper`, `istitle`.

### Plan caching limitation

When a cached plan contains graph breaks (eager segments), it is always
re-traced because the eager functions may depend on runtime values. Only
fully-compiled plans (zero graph breaks) benefit from caching.

## Introspection

### `explain()` — view the execution plan

```python
@pd.jit.compilable
def f(df):
    return df[df["price"] > 100].sort_values("price").head(5)

print(f.explain(sales_df))
# ExecutionPlan for f():
#   Backend: datafusion
#   Segments: 1
#
#   [0] COMPILED -> __mat_1
#       Limit(5)
#         Sort(price ASC)
#           Filter(($price > 100))
#             ReadTable(df) [id:INT64, region:STRING, ...]
#
#   Output: __mat_1
```

### Substrait export

```python
# As protobuf objects
plans = f.to_substrait(sales_df)
plan_bytes = plans[0].SerializeToString()

# As JSON
json_str = f.to_substrait_json(sales_df)
```

### ConnectedPlan — rich DAG export

When a function has graph breaks, `to_substrait()` returns disconnected plans.
`to_connected_plan()` provides a richer export with full metadata: schemas,
table connections, and graph-break reasons, forming a linked DAG.

```python
@pd.jit.compilable
def f(df):
    df["revenue"] = df["price"] * df["quantity"]
    total = len(df)                                    # graph break
    return df[df["revenue"] > 500].head(10)

cp = f.to_connected_plan(sales_df)

# Iterate stages in order
for stage in cp.stages:
    if isinstance(stage, CompiledStage):
        print(f"Compiled stage {stage.index}: {stage.description}")
        print(f"  Inputs: {stage.input_tables}")
        print(f"  Output: {stage.output_table}")
        print(f"  Schema: {stage.output_schema.columns}")
    elif isinstance(stage, GraphBreakStage):
        print(f"Graph break {stage.index}: {stage.reason}")

# Backward-compatible: list of Substrait Plan protos
plans = cp.plans

# JSON-serializable dict with full metadata
metadata = cp.to_dict()
```

Key classes:

| Class | Purpose |
|-------|---------|
| `ConnectedPlan` | Top-level container. Properties: `stages`, `compiled_stages`, `graph_breaks`, `plans`, `final_output` |
| `CompiledStage` | A Substrait plan with `plan`, `plan_bytes`, `input_tables`, `output_table`, `output_schema` |
| `GraphBreakStage` | A materialization point with `reason`, `input_tables`, `output_tables` |
| `StageSchema` | Schema metadata: `table_name`, `columns: dict[str, str]` |

## Context manager API

For more control, use `Tracer` directly:

```python
from pandas.jit import Tracer, PandasBackend

with Tracer(backend=PandasBackend()) as t:
    df = t.input(sales_df, "sales")
    df["revenue"] = df["price"] * df["quantity"]
    filtered = df[df["revenue"] > 500]
    t.output(filtered)

result = t.result()
print(t.explain())

# Substrait export
plans = t.to_substrait()
json_str = t.to_substrait_json()

# Connected plan export
cp = t.to_connected_plan()
```

## Caching

`CompiledFunction` caches execution plans keyed by a `SchemaGuard` — a tuple
of (input DataFrame schemas, non-DataFrame positional args, non-DataFrame
keyword args). If the same function is called again with DataFrames of the
same schema and the same scalar arguments, the cached plan is reused without
re-tracing.

Plans that contain eager segments (graph breaks) are always re-traced since
the eager functions may depend on runtime values.

## IR types reference

### Nodes (`ir.IRNode` subclasses) — 12 total

| Node | Fields |
|------|--------|
| `ReadTable` | `name: str`, `schema: Schema` |
| `Filter` | `input: IRNode`, `predicate: Expr` |
| `Project` | `input: IRNode`, `columns: list[str]` |
| `AddColumn` | `input: IRNode`, `name: str`, `expr: Expr`, `dtype: DType` |
| `Sort` | `input: IRNode`, `keys: list[tuple[str, bool]]` |
| `Limit` | `input: IRNode`, `n: int`, `offset: int = 0` |
| `Aggregate` | `input: IRNode`, `group_keys: list[str]`, `agg_specs: list[tuple[str, str, str]]` |
| `Join` | `left: IRNode`, `right: IRNode`, `left_on: list[str]`, `right_on: list[str]`, `how: str` |
| `Union` | `inputs: list[IRNode]` |
| `Distinct` | `input: IRNode`, `columns: list[str]` |
| `Window` | `input: IRNode`, `window_funcs: list[tuple]`, `window_spec: WindowSpec`, `partition_by`, `order_by` |
| `RenameColumns` | `input: IRNode`, `mapping: dict[str, str]` |

### Expressions (`ir.Expr` subclasses) — 9 total

| Expr | Fields |
|------|--------|
| `ColRef` | `name: str` |
| `Literal` | `value: Any`, `dtype: DType` |
| `BinOp` | `op: str`, `left: Expr`, `right: Expr` |
| `UnaryOp` | `op: str`, `operand: Expr` |
| `IfThenExpr` | `condition: Expr`, `then_expr: Expr`, `else_expr: Expr` |
| `CastExpr` | `input: Expr`, `target_dtype: DType` |
| `SingularOrList` | `value: Expr`, `options: list[Expr]` |
| `FunctionCall` | `func_name: str`, `args: list[Expr]`, `options: dict`, `return_dtype: DType` |
| `ScalarSubquery` | `agg_node: IRNode`, `dtype: DType` |

### DType enum — 19 values

`INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`,
`FLOAT32`, `FLOAT64`, `STRING`, `BINARY`, `BOOL`, `DATE`, `TIME`,
`TIMESTAMP`, `TIMESTAMP_TZ`, `TIMEDELTA`, `DECIMAL`

### Substrait function registry — 43 entries

| Category | Functions |
|----------|----------|
| Comparison | `gt`, `gte`, `lt`, `lte`, `eq`, `ne`, `is_null`, `coalesce` |
| Arithmetic | `add`, `sub`, `mul`, `div`, `abs`, `negate` |
| Boolean | `and`, `or`, `not` |
| Aggregate | `sum`, `avg`/`mean`, `min`, `max`, `count`, `std`, `var`, `product` |
| Positional window | `lag`, `lead` |
| Rank window | `rank`, `dense_rank`, `row_number` |
| Datetime | `extract` |
| String | `str_upper`, `str_lower`, `str_strip`, `str_lstrip`, `str_rstrip`, `str_len`, `str_contains`, `str_startswith`, `str_endswith`, `str_replace`, `str_slice` |

## Public API

All public names are importable from `pandas.jit`:

```python
from pandas.jit import (
    AceroBackend,
    Backend,
    CompiledFunction,
    CompiledStage,
    ConnectedPlan,
    DataFusionBackend,
    DeferredScalar,
    GraphBreakStage,
    PandasBackend,
    StageSchema,
    Tracer,
    compilable,
    infer_schema,
)
```

Power-user modules (not re-exported but importable):
- `pandas.jit.ir` — `DType`, `Schema`, `IRNode`, `Expr` subclasses
- `pandas.jit.compiler` — `SubstraitCompiler`, `ExecutionPlan`
