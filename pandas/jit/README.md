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
| `df.abs()` | `AddColumn(UnaryOp("abs"))` | Numeric columns only; non-numeric pass through |
| `df.clip(lower, upper)` | `AddColumn(IfThenExpr)` | Scalar bounds; dict/Series bounds graph-break |
| `df.round(decimals)` | `AddColumn(FunctionCall("round"))` | Numeric columns only |
| `df.rank(method=...)` | `Window(rank/dense_rank/row_number)` | Per-column; `min`/`dense`/`first` traced; others graph-break |
| `df.eval("col = expr")` | `AddColumn` via `_parse_query_string` | Assignment form only; bare expressions graph-break |
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
| `df.diff(n)` | `Window(lag) + AddColumn(sub)` | Per-column lag then subtract; numeric only |
| `df.reset_index(drop=True)` | no-op | Relational algebra has no row index |
| `df.copy()` | no-op | Returns same proxy |
| `df.pipe(func)` | pass-through | Calls `func(self)`, stays traced if func uses traced API |
| `df.filter(items=...)` | `Project` | Column selection by name, `like=`, or `regex=` |
| `df.reindex(columns=...)` | `Project` | Column reindex only; row reindex graph-breaks |
| `df.get(key, default)` | `ColRef` or default | Returns `TracedSeries` if key exists, else default |
| `df + scalar` / `df - scalar` / `df * scalar` / `df / scalar` | `AddColumn(BinOp)` | Element-wise arithmetic on numeric columns; DataFrame operand graph-breaks |
| `-df` | `AddColumn(BinOp("mul", -1))` | Negate numeric columns |
| `list(df)` / `for col in df` | schema iteration | Iterates column names without materialization |
| `df.items()` | schema iteration | Yields `(col_name, TracedSeries)` pairs without materialization |
| `df.keys()` | alias for `df.columns` | No materialization |

### Series operations

| Method | IR expression | Returns |
|--------|---------------|---------|
| `series > x` (and `>=`, `<`, `<=`, `==`, `!=`) | `BinOp` | `TracedSeries[BOOL]` |
| `series + x` (and `-`, `*`, `/`, `//`, `%`, `**`) | `BinOp` | `TracedSeries`; reverse ops supported |
| `series & other`, `series \| other` | `BinOp("and"/"or")` | `TracedSeries[BOOL]` |
| `~series` | `UnaryOp("not")` | `TracedSeries[BOOL]` |
| `-series` | `UnaryOp("negate")` | `TracedSeries` |
| `abs(series)` / `series.abs()` | `UnaryOp("abs")` | `TracedSeries` |
| `series.isin(values)` | `SingularOrList` | `TracedSeries[BOOL]` |
| `series.isna()` | `UnaryOp("is_null")` | `TracedSeries[BOOL]` |
| `series.notna()` | `NOT(IS_NULL)` | `TracedSeries[BOOL]` |
| `series.fillna(val)` | `BinOp("coalesce")` | `TracedSeries` |
| `series.between(lo, hi)` | `AND(GTE, LTE)` | `TracedSeries[BOOL]` |
| `series.clip(lower, upper)` | `IfThenExpr` | Scalar bounds only; Series bounds graph-break |
| `series.where(cond, other)` | `IfThenExpr` | Scalar `other`; Series cond stays traced |
| `series.mask(cond, other)` | `IfThenExpr(NOT)` | Inverse of where |
| `series.astype(dtype)` | `CastExpr` | Recognized pandas/numpy dtypes; unknown graph-break |
| `series.round(decimals)` | `FunctionCall("round")` | `TracedSeries` |
| `series.sqrt()` | `FunctionCall("sqrt")` | `TracedSeries[FLOAT64]` |
| `series.log()` | `FunctionCall("log")` | `TracedSeries[FLOAT64]`; natural log |
| `series.log10()` | `FunctionCall("log10")` | `TracedSeries[FLOAT64]` |
| `series.exp()` | `FunctionCall("exp")` | `TracedSeries[FLOAT64]` |
| `series.ceil()` | `FunctionCall("ceil")` | `TracedSeries` |
| `series.floor()` | `FunctionCall("floor")` | `TracedSeries` |
| `series.sign()` | `FunctionCall("sign")` | `TracedSeries` |
| `np.sqrt(series)` | `__array_ufunc__` → `FunctionCall` | Intercepts numpy ufuncs |
| `series.rank(method=...)` | `Window(rank/dense_rank/row_number)` | `TracedSeries[FLOAT64]`; methods `min`/`dense`/`first` |
| `series.cumsum()` | `Window(sum, expanding)` | `TracedSeries`; composable via cross-IR |
| `series.cummax()` | `Window(max, expanding)` | `TracedSeries` |
| `series.cummin()` | `Window(min, expanding)` | `TracedSeries` |
| `series.cumprod()` | `Window(product, expanding)` | `TracedSeries` |
| `series.diff(n)` | `Window(lag) + BinOp(sub)` | `TracedSeries`; col minus lagged col |
| `series.pct_change(n)` | `Window(lag) + BinOp(sub,div)` | `TracedSeries[FLOAT64]`; `(x - lag(x,n)) / lag(x,n)` |
| `series.shift(n)` | `Window(lag/lead)` | `TracedSeries`; positive n → lag, negative → lead |
| `series.copy()` | no-op | Returns new `TracedSeries` with same IR |
| `series.replace(dict)` | `IfThenExpr` chain | Dict and scalar-to-scalar traced; complex patterns graph-break |
| `series.map(dict)` | `IfThenExpr` chain + `AddColumn` | Dict arg traced (unmapped → NaN); callable arg graph-breaks |

### Series properties

| Property | Returns | Notes |
|----------|---------|-------|
| `series.name` | `str` | Column name (no materialization) |
| `series.dtype` | `pandas.dtype` | Mapped from IR dtype |
| `series.ndim` | `1` | Always 1 (no materialization) |
| `series.hasnans` | `bool` | Graph break — materializes |
| `series.is_unique` | `bool` | Graph break — materializes |
| `series.is_monotonic_increasing` | `bool` | Graph break — materializes |
| `series.is_monotonic_decreasing` | `bool` | Graph break — materializes |
| `series.empty` | `bool` | Graph break — materializes |

### DataFrame properties

| Property | Returns | Notes |
|----------|---------|-------|
| `df.columns` | `pd.Index` | From IR schema (no materialization) |
| `df.dtypes` | `pd.Series` | From IR schema (no materialization) |
| `df.ndim` | `2` | Always 2 (no materialization) |
| `df.shape` | `tuple` | Graph break — materializes |
| `df.index` | `pd.Index` | Graph break — materializes |
| `df.empty` | `bool` | Graph break — materializes |

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
| `prod()` / `product()` | `multiply:i64` | Numeric columns only |
| `size()` | `count:any` | Returns single column named `"size"` |
| `nunique()` | `count_distinct:any` | Returns count of unique values per column |
| `agg({col: func})` | varies | Dict mapping columns to function names |
| `first()` | `Window(row_number) + Filter` | Traced: partitioned row_number = 1 |
| `last()` | — | **Graph break** |
| `head(n)` | `Window(row_number) + Filter` | Traced: partitioned row_number ≤ n |
| `nth(n)` | `Window(row_number) + Filter` | Traced for n ≥ 0; negative n graph-breaks |
| `rank(method=...)` | `Window(rank/dense_rank/row_number)` | Traced for `min`/`dense`/`first`; others graph-break |
| `cumsum()` | `Window(sum, expanding)` | Partitioned by group keys |
| `cummax()` | `Window(max, expanding)` | Partitioned by group keys |
| `cummin()` | `Window(min, expanding)` | Partitioned by group keys |
| `cumprod()` | `Window(product, expanding)` | Partitioned by group keys |
| `cumcount()` | `Window(row_number) - 1` | Traced; 0-based; supports `ascending=False` |
| `shift(n)` | `Window(lag/lead)` | Traced with partition_by group keys |
| `diff(n)` | `Window(lag) + BinOp(sub)` | Traced; numeric columns only |
| `pct_change(n)` | `Window(lag) + BinOp(sub,div)` | Traced; numeric columns only |
| `transform(func)` | `Window` | Traced for `sum`/`mean`/`min`/`max`/`count`/`std`/`var`; others graph-break |

All aggregation and window methods above also work on `groupby()[[col1, col2]]` (multi-column
selection via `TracedGroupByMulti`), including `shift()`, `diff()`, `cumsum()`/`cummax()`/`cummin()`/`cumprod()`, `rank()`, `first()`, `head()`, and `nth()`.

### GroupBySeries operations (`groupby()[col]`)

Available on `TracedGroupBySeries` — single-column groupby access:

| Method | How it works | Notes |
|--------|-------------|-------|
| `sum()`, `mean()`, `count()`, `min()`, `max()`, `std()`, `var()`, `nunique()`, `prod()` | `Aggregate` | Traced; returns `TracedDataFrame` |
| `cumsum()`, `cummax()`, `cummin()`, `cumprod()` | `Window(expanding)` | Traced; returns `TracedSeries` |
| `transform(func)` | `Window` | Traced for string funcs; others graph-break |
| `rank(method=...)` | `Window(rank/dense_rank/row_number)` | Traced for `min`/`dense`/`first` |
| `first()` | `Window(row_number) + Filter` | Traced; returns `TracedSeries` |
| `last()` | — | Graph break |
| `shift(n)` | `Window(lag/lead)` | Traced; returns `TracedSeries` |
| `diff(n)` | `Window(lag) + BinOp(sub)` | Traced; returns `TracedSeries` |
| `head(n)` | `Window(row_number) + Filter` | Traced |
| `nth(n)` | `Window(row_number) + Filter` | Traced for n ≥ 0 |
| `apply(func)` | — | Graph break |
| `ffill()`, `bfill()` | — | Graph break |
| `median()`, `quantile()` | — | Graph break |
| `describe()`, `value_counts()` | — | Graph break |

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

### Rolling/Expanding graph-break methods

These methods materialize the IR and delegate to pandas. Available on both
`rolling()` and `expanding()`:

| Method | Notes |
|--------|-------|
| `median()` | Statistical median |
| `quantile(q)` | Quantile computation |
| `skew()` | Skewness |
| `kurt()` | Kurtosis |
| `sem()` | Standard error of the mean |
| `rank()` | Rolling/expanding rank |
| `apply(func)` | Custom rolling/expanding function |
| `corr(other)` | Correlation |
| `cov(other)` | Covariance |
| `agg(func)` / `aggregate(func)` | Dispatches to traced method for known string funcs (`mean`, `sum`, etc.); otherwise graph-breaks |

### Positional window functions

| Method | Window function | Notes |
|--------|----------------|-------|
| `df.shift(n)` (n > 0) | `lag(col, n)` | NaN-fills first n rows |
| `df.shift(n)` (n < 0) | `lead(col, abs(n))` | NaN-fills last n rows |
| `df.shift(0)` | `lag(col, 0)` | Identity |

### Rank window functions

The IR and compiler support `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()` as
parameterless window functions with `ORDER BY` and optional `PARTITION BY`.
`series.rank()` and `groupby().rank()` stay traced for supported methods
(`min`, `dense`, `first`) via cross-IR composition — the Window node is
transplanted into the target IR when used in `assign()`, `__setitem__()`,
or `__getitem__()`. Unsupported methods (`average`, `pct=True`) graph-break.

| Substrait function | pandas equivalent | Notes |
|-------------------|-------------------|-------|
| `rank:` | `rank(method="min")` | Ties get lowest rank, gaps after ties |
| `dense_rank:` | `rank(method="dense")` | Ties get lowest rank, no gaps |
| `row_number:` | `rank(method="first")` | No ties, order of appearance |

### Cross-IR composition

When a `TracedSeries` is backed by a different IR tree (e.g., a `Window` node
from `rank()` or `shift()`), it can't be directly composed with the parent
DataFrame's IR via simple expression splicing. The tracer detects this case
and transplants the Window function into a new `Window` node attached to the
target IR:

- **`assign()` / `__setitem__()`**: The Window's output column is renamed to
  the target key. The new Window uses the DataFrame's current IR as input.
- **`__getitem__()` (boolean mask)**: Temporary column names are generated to
  avoid overwriting existing columns. The mask expression is rewritten to
  reference the temp names, then `Filter` + `Project` drops the temps.
- **Fallback**: If the Window's input isn't an ancestor of the target IR,
  both sides are materialized.

This enables patterns like `df.assign(r=df["a"].rank(method="min"))` and
`df[df["a"].rank(method="min") <= 3]` to stay fully traced.

### Cumulative functions

DataFrame-level cumulative functions are traced as expanding windows:

| Method | Window function | Notes |
|--------|----------------|-------|
| `df.cumsum()` | `sum` over `rows(unbounded, 0)` | Numeric columns only |
| `df.cummax()` | `max` over `rows(unbounded, 0)` | |
| `df.cummin()` | `min` over `rows(unbounded, 0)` | |
| `df.cumprod()` | `product` over `rows(unbounded, 0)` | |

Series-level cumulative (`series.cumsum()`, etc.) is also traced using expanding
windows and supports cross-IR composition — see the Series operations table above.

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
| `.dt.microsecond` | `MICROSECOND` |
| `.dt.nanosecond` | via extraction |
| `.dt.week` / `.dt.weekofyear` | `ISO_WEEK` |
| `.dt.date` | via extraction |

**Boolean properties** (traced): `.dt.is_month_start`, `.dt.is_month_end`,
`.dt.is_quarter_start`, `.dt.is_quarter_end`, `.dt.is_year_start`, `.dt.is_year_end`

**Graph-break methods**: `.dt.strftime()`, `.dt.tz_localize()`, `.dt.tz_convert()`,
`.dt.normalize()`, `.dt.ceil()`, `.dt.floor()`, `.dt.round()`, `.dt.month_name()`,
`.dt.day_name()`

### Timedelta accessor (`.td`)

| Property | Notes |
|----------|-------|
| `.dt.days` | Traced via extraction |
| `.dt.seconds` | Traced via extraction |
| `.dt.microseconds` | Traced via extraction |
| `.dt.total_seconds()` | Traced via extraction |

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
| `.str.capitalize()` | `capitalize:str` |
| `.str.title()` | `title:str` |
| `.str.swapcase()` | `swapcase:str` |
| `.str.isdigit()` | `is_digit:str` |
| `.str.isalpha()` | `is_alpha:str` |
| `.str.isnumeric()` | `is_numeric:str` |
| `.str.isspace()` | `is_space:str` |
| `.str.islower()` | `is_lower:str` |
| `.str.isupper()` | `is_upper:str` |
| `.str.count(sub)` | `count_substring:str_str` |
| `.str.find(sub)` | `strpos:str_str` |
| `.str.cat(sep)` | `concat:str_str` |
| `.str.pad(width, side)` | `str_pad:str_i32` |
| `.str.center(width)` | `str_center:str_i32` |
| `.str.zfill(width)` | `str_zfill:str_i32` |
| `.str.repeat(n)` | `str_repeat:str_i32` |

**Graph-break `.str` methods**: `.str.split()`, `.str.rsplit()`, `.str.get()`,
`.str.extract()`, `.str.match()`, `.str.fullmatch()`, `.str.wrap()`,
`.str.ljust()`, `.str.rjust()`

### IR optimization

The module includes a **projection pruning** pass (`optimize_projections()` in `ir.py`)
that is applied automatically before execution. It walks the IR tree top-down and
inserts `Project` nodes to eliminate unused columns early, reducing memory and
compute for wide-table pipelines. The pass is enabled by default and can be
disabled with `optimize=False` on `PandasBackend.execute()` or `SubstraitCompiler.compile()`.

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
| `df.rank(method="average")` | Materialize; `min`/`dense`/`first` are now traced |
| `df.reset_index(drop=False)` | Materialize when index is meaningful (non-default RangeIndex) |
| `df.set_index(keys)` | Materialize, set index, re-wrap |
| `df.pivot_table(...)` | Materialize, pivot, re-wrap |
| `df.melt(...)` | Materialize, melt, re-wrap |
| `df.stack()`, `df.unstack()` | Materialize, reshape |
| `df.corr()`, `df.cov()` | Materialize, return correlation/covariance matrix |
| `df.nunique()` | Materialize, return Series of unique counts |
| `df.prod()` / `df.product()` | Materialize, return Series of products |
| `df.isin(values)` | Materialize, return boolean DataFrame |
| `df.idxmin()`, `df.idxmax()` | Materialize, return Series of index labels |
| `df.join(other)` | Materialize both sides, join, re-wrap |
| `df.combine_first(other)` | Materialize, combine, re-wrap |
| `df.update(other)` | Materialize, update in-place, re-wrap |
| `df.equals(other)` | Materialize both sides, compare |
| `df.compare(other)` | Materialize both sides, compare |
| `df.explode(column)` | Materialize, explode, re-wrap |
| `df.take(indices)` | Materialize, return rows |
| `df.pop(item)` | Materialize, remove column, return Series |
| `df.xs(key)` | Materialize, return cross-section |
| `df.memory_usage()` | Materialize, return memory info |
| `df.insert(loc, col, val)` | Materialize, insert column, re-wrap |
| `df.T` / `df.transpose()` | Materialize, return transposed DataFrame |
| `df.sample(...)` | Materialize, return random sample |
| `df.agg(func)` / `df.aggregate(func)` | Materialize, aggregate, return |
| `df.map(func)` | Materialize, apply element-wise, re-wrap |
| `df.any()` / `df.all()` | Materialize, return Series |
| `df + other_df` | Materialize both sides, operate, re-wrap |

### GroupBy graph breaks

| Operation | What happens |
|-----------|-------------|
| `groupby().median()` | Materialize, aggregate, re-wrap |
| `groupby().quantile(q)` | Materialize, aggregate, re-wrap |
| `groupby().sem()` | Materialize, aggregate, re-wrap |
| `groupby().skew()` | Materialize, aggregate, re-wrap |
| `groupby().kurt()` | Materialize, aggregate, re-wrap |
| `groupby().idxmin()`, `idxmax()` | Materialize, return |
| `groupby().filter(func)` | Materialize, filter groups, re-wrap |
| `groupby().apply(func)` | Materialize, apply, re-wrap |
| `groupby().describe()` | Materialize, return |
| `groupby().ffill()`, `bfill()` | Materialize, fill, re-wrap |
| `groupby().value_counts()` | Materialize, return |
| `groupby().ngroups` | Materialize, return int |
| `groupby().groups` | Materialize, return dict |
| `groupby().get_group(name)` | Materialize, extract group, re-wrap |

### Series graph breaks

| Operation | What happens |
|-----------|-------------|
| `series.apply(func)` | Materialize, apply |
| `series.map(func)` | Materialize for callable arg; dict arg is traced |
| `series.unique()` | Materialize, return array |
| `series.nunique()` | Materialize, return int |
| `series.value_counts()` | Materialize, return Series |
| `series.rank(method="average")` | Materialize; `min`/`dense`/`first` stay traced |
| `series.reset_index(drop=False)` | Materialize, returns DataFrame |
| `bool(series)`, `series.item()` | Materialize scalar value |
| `len(series)` | Materialize, return int |
| `series.median()` | Materialize, return scalar |
| `series.quantile(q)` | Materialize, return scalar |
| `series.prod()` / `product()` | Materialize, return scalar |
| `series.sem()`, `skew()`, `kurt()` | Materialize, return scalar |
| `series.corr(other)`, `cov(other)` | Materialize both, return scalar |
| `series.idxmin()`, `idxmax()` | Materialize, return index label |
| `series.duplicated(keep)` | Materialize, return boolean TracedSeries |
| `series.factorize()` | Materialize, return (codes, uniques) tuple |
| `series.explode()` | Materialize, return TracedSeries |
| `series.searchsorted(value)` | Materialize, return insertion index |
| `series.tolist()` / `to_list()` | Materialize, return list |
| `series.__array__()` | Materialize, return numpy array |
| `series.ffill()`, `bfill()` | Materialize, fill, re-wrap |
| `series.describe()`, `mode()` | Materialize, return |
| `series.nlargest(n)`, `nsmallest(n)` | Materialize, return |
| `series.replace(...)` | Materialize for complex patterns; dict/scalar traced |
| `series.to_numpy()` | Materialize, return numpy array |
| `series.to_dict()` | Materialize, return dict |
| `series.to_csv()` / `to_json()` | Materialize, write |
| `series.items()` | Materialize, iterate |
| `for val in series` | Materialize, iterate |
| `val in series` | Materialize, check membership |
| `series.values` | Materialize, return numpy array |
| `series.shape` | Materialize, return tuple |
| `series.index` | Materialize, return Index |

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
| `resample()` | Temporal bucketing has no Substrait equivalent |

Note: `groupby().first()` is now traced via `Window(row_number) + Filter(eq, 1)`.
`groupby().last()` still graph-breaks as there's no efficient way to get the last
row without knowing group sizes.

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

### Substrait function registry — 66 entries

| Category | Functions |
|----------|----------|
| Comparison | `gt`, `gte`, `lt`, `lte`, `eq`, `ne`, `is_null`, `coalesce` |
| Arithmetic | `add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`, `abs`, `negate`, `round`, `sqrt`, `log`, `log10`, `exp`, `ceil`, `floor`, `sign` |
| Boolean | `and`, `or`, `not` |
| Aggregate | `sum`, `avg`/`mean`, `min`, `max`, `count`, `std`, `var`, `product` |
| Positional window | `lag`, `lead` |
| Rank window | `rank`, `dense_rank`, `row_number` |
| Datetime | `extract` |
| String | `str_upper`, `str_lower`, `str_strip`, `str_lstrip`, `str_rstrip`, `str_len`, `str_contains`, `str_startswith`, `str_endswith`, `str_replace`, `str_slice`, `str_capitalize`, `str_title`, `str_swapcase`, `str_isdigit`, `str_isalpha`, `str_isnumeric`, `str_isspace`, `str_islower`, `str_isupper`, `str_count`, `str_find`, `str_cat` |

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
