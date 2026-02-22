# pandas.compile — JIT Compilation for pandas

`pandas.compile` traces pandas operations at runtime, builds a relational
intermediate representation (IR), and executes the plan on an optimized backend
(PyArrow Acero via Substrait, or a pure-pandas interpreter). Operations that
cannot be represented in the IR trigger *graph breaks* — the traced computation
is materialized and tracing resumes automatically.

## Quick start

```python
import pandas as pd

@pd.compile
def process(df):
    df["revenue"] = df["price"] * df["quantity"]
    big = df[df["revenue"] > 500]
    return big.sort_values("revenue", ascending=False).head(10)

result = process(sales_df)
```

`@pd.compile` can also be called with a specific backend:

```python
from pandas.compile import PandasBackend

@pd.compile(backend=PandasBackend())
def process(df):
    return df[df["price"] > 100]
```

> **Note:** `compile` is a Python built-in, so always use `pd.compile` to
> avoid shadowing it. The `pandas.compile` subpackage is made callable via
> a `_CallableModule` wrapper so `pd.compile(fn)` works as a decorator.

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
     ├──► SubstraitCompiler  → Substrait protobuf  → AceroBackend (pyarrow)
     │
     └──► PandasBackend      → direct pandas execution (fallback)
```

### Modules

| Module | Purpose |
|--------|---------|
| `ir.py` | IR types: `DType`, `Schema`, `IRNode` subclasses, `Expr` subclasses, `explain_ir`/`explain_expr` |
| `compiler.py` | `SubstraitCompiler`, `PandasBackend`, `AceroBackend`, `ExecutionPlan`, `SchemaGuard` |
| `jit.py` | `@pd.compile` decorator, `Tracer` context manager, all `Traced*` proxy classes |

### Backends

| Backend | How it works | When used |
|---------|-------------|-----------|
| `AceroBackend` | Compiles IR → Substrait protobuf → `pyarrow.substrait.run_query()` | Default when pyarrow is installed |
| `PandasBackend` | Interprets the IR tree using pandas operations directly | Fallback, or when explicitly requested |

## Supported operations

### Fully traced (IR nodes)

These operations are captured in the IR and compiled to Substrait / executed on
either backend without materialization.

| pandas API | IR node | Notes |
|-----------|---------|-------|
| `df[col]` | `ColRef` | Column access returns `TracedSeries` |
| `df[[col1, col2]]` | `Project` | Column subset |
| `df[bool_series]` | `Filter` | Boolean indexing |
| `df["new"] = expr` | `AddColumn` | Add/replace column |
| `df.assign(col=expr)` | `AddColumn` | Same, functional style |
| `df.sort_values(by)` | `Sort` | Single or multi-column |
| `df.head(n)` | `Limit` | Take first N rows |
| `df.nlargest(n, col)` | `Sort` + `Limit` | |
| `df.nsmallest(n, col)` | `Sort` + `Limit` | |
| `df.query("price > 100")` | `Filter` | Parses query string to expression |
| `df.drop(columns=[...])` | `Project` | |
| `df.rename(columns={...})` | `RenameColumns` | |
| `df.dropna(subset=[...])` | `Filter(NOT IS_NULL)` | |
| `df.fillna(value)` | `AddColumn(COALESCE)` | Per-column or scalar |
| `df.merge(right, on=...)` | `Join` | inner, left, right, outer |
| `df.groupby(by).sum()` | `Aggregate` | See aggregations below |

### Aggregation functions

These are available on `groupby()`, `groupby()[col]`, and `groupby()[[cols]]`:

| Method | Substrait function | Backend support |
|--------|-------------------|-----------------|
| `sum()` | `sum:i64` | Acero + Pandas |
| `mean()` | `avg:i64` | Acero + Pandas |
| `min()` | `min:i64` | Acero + Pandas |
| `max()` | `max:i64` | Acero + Pandas |
| `count()` | `count:any` | Acero + Pandas |
| `std()` | `std_dev:fp64` | Acero + Pandas |
| `var()` | `variance:fp64` | Acero + Pandas |
| `size()` | `count:any` | Acero + Pandas |
| `first()` | — | Graph break |
| `last()` | — | Graph break |
| `agg({col: func})` | varies | Acero + Pandas |

Series-level aggregations (`series.sum()`, `series.mean()`, etc.) trigger a
graph break — the IR is materialized and the scalar value is returned.

### Arithmetic and comparison operators

All of these produce `BinOp` / `UnaryOp` expression nodes:

| Operator | IR op | | Operator | IR op |
|----------|-------|-|----------|-------|
| `>` | `gt` | | `+` | `add` |
| `>=` | `gte` | | `-` | `sub` |
| `<` | `lt` | | `*` | `mul` |
| `<=` | `lte` | | `/` | `div` |
| `==` | `eq` | | `&` | `and` |
| `!=` | `ne` | | `\|` | `or` |
| `-x` | `negate` | | `~x` | `not` |
| `abs(x)` | `abs` | | | |

### Series methods (traced)

| Method | IR expression | Returns |
|--------|---------------|---------|
| `series.isin(values)` | `OR(eq, eq, ...)` | `TracedSeries[BOOL]` |
| `series.isna()` | `is_null(col)` | `TracedSeries[BOOL]` |
| `series.notna()` | `NOT(is_null(col))` | `TracedSeries[BOOL]` |
| `series.fillna(val)` | `coalesce(col, val)` | `TracedSeries` |
| `series.abs()` | `abs(col)` | `TracedSeries` |
| `series.between(lo, hi)` | `AND(gte, lte)` | `TracedSeries[BOOL]` |

### Datetime accessor (`.dt`)

Traced via `FunctionCall("extract", ...)`. Supported on both backends.

```python
@pd.compile
def f(df):
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    return df
```

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

Traced via `FunctionCall("str_*", ...)`. **PandasBackend only** — Acero does
not support these Substrait string functions.

```python
@pd.compile(backend=PandasBackend())
def f(df):
    df["upper_name"] = df["name"].str.upper()
    return df[df["name"].str.contains("Ali")]
```

| Method | IR function |
|--------|------------|
| `.str.upper()` | `str_upper` |
| `.str.lower()` | `str_lower` |
| `.str.strip()` | `str_strip` |
| `.str.lstrip()` | `str_lstrip` |
| `.str.rstrip()` | `str_rstrip` |
| `.str.len()` | `str_len` |
| `.str.contains(pat)` | `str_contains` |
| `.str.startswith(pat)` | `str_startswith` |
| `.str.endswith(pat)` | `str_endswith` |
| `.str.replace(pat, repl)` | `str_replace` |
| `.str.slice(start, stop)` | `str_slice` |

### Graph-break operations

These operations materialize the current IR, run the operation in plain pandas,
then resume tracing with a fresh `ReadTable` from the result. They work
transparently — the user doesn't need to do anything special.

| Operation | What happens |
|-----------|-------------|
| `len(df)`, `df.shape`, `df.empty` | Materialize, return scalar |
| `df.values`, `df.to_numpy()` | Materialize, return array |
| `df.iloc[...]`, `df.loc[...]` | Materialize, return slice |
| `df.iterrows()`, `df.itertuples()` | Materialize, iterate |
| `df.apply(func)`, `df.pipe(func)` | Materialize, apply, re-register |
| `df.to_csv()`, `df.to_parquet()` | Materialize, write |
| `df.describe()`, `df.info()` | Materialize, return |
| `df.drop_duplicates()` | Materialize, deduplicate, re-register |
| `df.rolling(w).mean()` | Materialize, apply rolling, re-register |
| `df.expanding().sum()` | Materialize, apply expanding, re-register |
| `df.pivot_table(...)` | Materialize, pivot, re-register |
| `df.melt(...)` | Materialize, melt, re-register |
| `df.stack()`, `df.unstack()` | Materialize, reshape, re-register |
| `df.astype(dtype)` | Materialize, cast, re-register |
| `df.where(cond)`, `df.mask(cond)` | Materialize, apply, re-register |
| `series.apply(func)` | Materialize, apply |
| `series.map(func)` | Materialize, map |
| `series.unique()`, `series.nunique()` | Materialize, return |
| `series.value_counts()` | Materialize, return |
| `bool(series)`, `series.item()` | Materialize scalar |
| `groupby().first()`, `groupby().last()` | Materialize, apply, re-register |

After a graph break, tracing continues automatically:

```python
@pd.compile
def f(df):
    rolled = df[["id", "price"]].sort_values("id").rolling(2).mean()  # graph break
    return rolled[rolled["price"] > 100]  # tracing resumes with filter
```

## Introspection

### `explain()` — view the execution plan

```python
@pd.compile
def f(df):
    return df[df["price"] > 100].sort_values("price").head(5)

print(f.explain(sales_df))
# ExecutionPlan for f():
#   Backend: acero
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

## Context manager API

For more control, use `Tracer` directly:

```python
from pandas.compile import Tracer, PandasBackend

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

### Nodes (`ir.IRNode` subclasses)

| Node | Fields |
|------|--------|
| `ReadTable` | `name: str`, `schema: Schema` |
| `Filter` | `input: IRNode`, `predicate: Expr` |
| `Project` | `input: IRNode`, `columns: list[str]` |
| `AddColumn` | `input: IRNode`, `name: str`, `expr: Expr`, `dtype: DType` |
| `Sort` | `input: IRNode`, `keys: list[tuple[str, bool]]` |
| `Limit` | `input: IRNode`, `n: int` |
| `Aggregate` | `input: IRNode`, `group_keys: list[str]`, `agg_specs: list[tuple[str, str, str]]` |
| `Join` | `left: IRNode`, `right: IRNode`, `left_on: str`, `right_on: str`, `how: str` |
| `RenameColumns` | `input: IRNode`, `mapping: dict[str, str]` |

### Expressions (`ir.Expr` subclasses)

| Expr | Fields |
|------|--------|
| `ColRef` | `name: str` |
| `Literal` | `value: Any`, `dtype: DType` |
| `BinOp` | `op: str`, `left: Expr`, `right: Expr` |
| `UnaryOp` | `op: str`, `operand: Expr` |
| `FunctionCall` | `func_name: str`, `args: list[Expr]`, `options: dict[str, str]`, `return_dtype: DType` |

### DType enum

`INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`,
`FLOAT32`, `FLOAT64`, `STRING`, `BINARY`, `BOOL`, `DATE`, `TIME`,
`TIMESTAMP`, `TIMESTAMP_TZ`, `TIMEDELTA`, `DECIMAL`
