# Query Optimization Plan for Lazy Pandas

## Overview

Query optimization transforms a logical plan into an equivalent but more efficient logical plan before execution. This document outlines the optimization passes we'll implement for lazy pandas.

## Current Architecture

```
User Query -> Expr API -> Logical Plan (plan.py) -> Eager Evaluation (frame.py)
```

**Plan Nodes** (from `plan.py`):
- `DataFrameSource` - Leaf node wrapping a DataFrame
- `Project` - Column selection/computation
- `Filter` - Row filtering by predicate
- `Aggregate` - Grouping and aggregation
- `Sort` - Ordering
- `Limit` - Row limiting
- `Distinct` - Deduplication
- `Join` - Two-input join

**IR Nodes** (from `ir.py`):
- `FieldRef` - Column reference
- `Literal` - Constant value
- `Alias` - Named expression
- `Call` - Function call (with `is_aggregate` flag)
- `Cast` - Type conversion

**Type System** (from `types.py`):
- `LazyDtype` - Tracks both `numpy_dtype` and `arrow_type`
- `Schema` - Column names and their LazyDtype

## Array Backend Considerations

### The Problem

pandas columns can be backed by different array types:
1. **NumPy arrays** - Traditional backend, `np.ndarray`
2. **Arrow arrays** - Via `pd.ArrowDtype`, stored as PyArrow ChunkedArray

When operations mix backends, **format conversion** occurs:
- Arrow → NumPy: Can be expensive (memory copy, type coercion)
- NumPy → Arrow: Also expensive, plus potential precision loss

**Goal**: Minimize format conversions by grouping operations by compatible backend.

### Backend Detection

We already track backend info in `LazyDtype`:
```python
@dataclass
class LazyDtype:
    category: str           # "numeric", "string", "datetime", "boolean", "object"
    numpy_dtype: np.dtype   # Present for NumPy-backed columns
    arrow_type: pa.DataType # Present for Arrow-backed columns
```

Detection at schema construction time:
```python
# In Schema.from_dataframe()
if isinstance(dtype, pd.ArrowDtype):
    # Arrow-backed column
    fields[col] = LazyDtype.from_arrow_type(dtype.pyarrow_dtype)
else:
    # NumPy-backed column
    fields[col] = LazyDtype.from_pandas_dtype(dtype)
```

### Backend-Aware Optimization Strategy

#### 1. Track Backend Through Plan

Each plan node's schema already contains `LazyDtype` with backend info. We can determine:
- Which columns are Arrow-backed vs NumPy-backed
- Which operations prefer which backend

#### 2. Operation Backend Preferences

| Operation | NumPy Native | Arrow Native | Notes |
|-----------|--------------|--------------|-------|
| Arithmetic (+, -, *, /) | ✓ | ✓ | Both efficient |
| Comparison (<, >, ==) | ✓ | ✓ | Both efficient |
| String ops (str.*) | ✗ | ✓ | Arrow much faster |
| Filter (boolean mask) | ✓ | ✓ | Both efficient |
| Aggregations (sum, mean) | ✓ | ✓ | Arrow can be faster |
| Group-by | ✓ | ✓ | Arrow can be faster |
| Join | ✓ | ✓ | Depends on key types |
| Sort | ✓ | ✓ | Both efficient |

#### 3. Backend Boundary Optimization

**Principle**: Group operations that share the same backend preference.

```python
# Before optimization (causes 2 conversions):
df.select()
  .with_columns(col("s").str.lower())   # Arrow-preferred
  .with_columns(col("n") + 1)           # NumPy fine
  .with_columns(col("s").str.upper())   # Arrow-preferred
  .collect()

# After optimization (0 conversions if Arrow-backed):
df.select()
  .with_columns(
      col("s").str.lower(),
      col("s").str.upper(),             # Grouped string ops
  )
  .with_columns(col("n") + 1)
  .collect()
```

#### 4. Conversion Boundary Insertion

When we must cross backends, explicit conversion is better than implicit:
- Mark conversion points in the plan
- Evaluator can batch conversions
- Future: Choose optimal conversion point

### Implementation: Backend-Aware Passes

#### Critical Safety Constraint: Expression Dependencies

**Problem**: Reordering expressions is NOT always safe. Expressions can have dependencies:

```python
# UNSAFE to reorder - b depends on a
df.select()
  .with_columns((col("x") + 1).alias("a"))
  .with_columns((col("a") + 1).alias("b"))  # References "a" created above!

# This is NOT equivalent to:
df.select()
  .with_columns((col("a") + 1).alias("b"))  # WRONG: "a" doesn't exist yet!
  .with_columns((col("x") + 1).alias("a"))
```

**Additional concern - Column overwrites**: If a column name is produced multiple times (overwrites), ordering is observable:

```python
# Order matters when same column is assigned twice
df.select()
  .with_columns((col("x") + 1).alias("a"))   # a = x + 1
  .with_columns((col("x") + 2).alias("a"))   # a = x + 2 (overwrites!)
# Final value of "a" depends on order!
```

**Rules for safe reordering**:
1. Never reorder across dependency levels
2. Never reorder expressions that produce the same column name
3. Only reorder within the same topological level where names are unique

**Solution**: Build a dependency DAG and only reorder within topological levels.

#### Expression Dependency Analysis

```python
def get_referenced_columns(expr: IRNode) -> set[str]:
    """Extract all column names referenced by an expression."""
    if isinstance(expr, FieldRef):
        return {expr.name}
    elif isinstance(expr, Call):
        refs = set()
        for arg in expr.args:
            refs |= get_referenced_columns(arg)
        return refs
    elif isinstance(expr, Alias):
        return get_referenced_columns(expr.arg)
    # ... handle other node types
    return set()

def get_produced_column(expr: Expr) -> str:
    """Get the output column name of an expression."""
    return extract_output_name(expr)

def build_expression_dag(exprs: list[Expr]) -> tuple[dict[int, set[int]], list[str]]:
    """
    Build dependency graph using expression indices (not names, to handle overwrites).

    Returns:
        - dag: expr_index -> set of expr_indices it depends on
        - names: list of output names (for grouping by backend later)

    Uses indices because the same column name can appear multiple times
    (overwrites), and we need to preserve their relative order.
    """
    names = [get_produced_column(e) for e in exprs]

    # Map from column name to the LATEST index that produces it
    # (for dependency tracking - refs always see the most recent assignment)
    latest_producer: dict[str, int] = {}
    for i, name in enumerate(names):
        latest_producer[name] = i

    dag: dict[int, set[int]] = {}
    for i, expr in enumerate(exprs):
        refs = get_referenced_columns(expr._ir)
        deps = set()
        for ref in refs:
            if ref in latest_producer:
                dep_idx = latest_producer[ref]
                if dep_idx < i:  # Only depend on earlier expressions
                    deps.add(dep_idx)
        dag[i] = deps

        # Update latest_producer AFTER computing deps
        # (so expr doesn't depend on itself)
        latest_producer[names[i]] = i

    # Add implicit ordering for overwrites: if same name produced twice,
    # later one depends on earlier one
    name_occurrences: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        name_occurrences.setdefault(name, []).append(i)

    for name, indices in name_occurrences.items():
        if len(indices) > 1:
            # Chain them: each depends on the previous
            for j in range(1, len(indices)):
                dag[indices[j]].add(indices[j - 1])

    return dag, names

def topological_levels(dag: dict[int, set[int]]) -> list[set[int]]:
    """
    Partition expression indices into levels where nodes in same level
    have no deps on each other.

    Level 0: expressions with no dependencies
    Level 1: expressions depending only on level 0
    etc.
    """
    remaining = {k: set(v) for k, v in dag.items()}
    levels = []
    while remaining:
        # Find nodes with no remaining dependencies
        level = {n for n, deps in remaining.items() if not deps}
        if not level:
            raise ValueError("Circular dependency detected in expressions")
        levels.append(level)
        # Remove this level's nodes from remaining deps
        remaining = {
            n: deps - level
            for n, deps in remaining.items()
            if n not in level
        }
    return levels
```

#### Safe Backend Grouping

```python
class BackendGrouping(OptimizationPass):
    """
    Reorder expressions within topological levels to minimize backend switches.

    SAFETY RULES:
    1. Never reorder across dependency levels (respects data dependencies)
    2. Never reorder expressions that overwrite the same column
    3. Only reorder independent expressions within the same topological level

    The DAG construction handles both rules by:
    - Adding edges for column references
    - Adding edges between same-name assignments (preserves overwrite order)
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        if not isinstance(plan, Project):
            return self._recurse(plan)

        exprs = list(plan.exprs)
        if len(exprs) <= 1:
            return plan  # Nothing to reorder

        dag, names = build_expression_dag(exprs)
        levels = topological_levels(dag)

        # Within each level, group by backend preference
        input_schema = plan.input.resolve_schema()
        reordered = []
        for level in levels:
            level_exprs = [exprs[i] for i in sorted(level)]  # Get exprs by index
            # Sort by backend: group arrow-preferred together, etc.
            level_exprs.sort(key=lambda e: self._backend_sort_key(e, input_schema))
            reordered.extend(level_exprs)

        # Only create new plan if order changed
        if reordered == exprs:
            return plan

        return Project(self._recurse(plan.input), tuple(reordered))

    def _backend_sort_key(self, expr: Expr, schema: Schema) -> int:
        """Return sort key: 0=arrow-preferred, 1=any, 2=numpy-preferred."""
        pref = get_backend_preference(expr._ir, schema)
        return {"arrow": 0, "any": 1, "numpy": 2}.get(pref, 1)

    def _recurse(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively optimize child plans."""
        # Implementation handles each plan type...
        pass
```

#### Expression Backend Analysis (Richer Model)

The simple "arrow/numpy/any" model is insufficient. Backend choice depends on:

1. **Available kernel for op + dtype**: Not all ops have Arrow compute kernels
2. **Missing-value semantics**: Arrow has native nulls, NumPy uses NaN/sentinel
3. **Physical storage**: Where the data currently lives (Arrow ChunkedArray vs ndarray)
4. **Conversion costs**: Moving data between backends is expensive

**Key distinction**: Storage backend ≠ Execution backend
- Data can be Arrow-backed but executed via NumPy (after conversion)
- We want to minimize conversions, not blindly follow storage

```python
@dataclass(frozen=True)
class BackendRequirements:
    """
    Rich backend analysis for an expression.

    Attributes
    ----------
    supported_backends : frozenset[str]
        Backends that CAN execute this operation: {"arrow", "numpy"} or subset.
    preferred_backend : str | None
        Backend that executes most efficiently, if there's a clear winner.
        None means no strong preference (both equally good).
    requires_backend : str | None
        Backend that MUST be used (no alternative). E.g., some string ops
        only have Arrow implementations in our codebase.
    conversion_cost : float
        Estimated cost if conversion is needed (0.0 to 1.0 scale).
        Higher = more expensive. Used to break ties.
    """
    supported_backends: frozenset[str]
    preferred_backend: str | None = None
    requires_backend: str | None = None
    conversion_cost: float = 0.0

    @classmethod
    def any_backend(cls) -> BackendRequirements:
        """Operation works equally well on any backend."""
        return cls(frozenset({"arrow", "numpy"}), None, None, 0.0)

    @classmethod
    def prefer_arrow(cls, required: bool = False) -> BackendRequirements:
        """Operation prefers or requires Arrow."""
        if required:
            return cls(frozenset({"arrow"}), "arrow", "arrow", 0.0)
        return cls(frozenset({"arrow", "numpy"}), "arrow", None, 0.3)

    @classmethod
    def prefer_numpy(cls, required: bool = False) -> BackendRequirements:
        """Operation prefers or requires NumPy."""
        if required:
            return cls(frozenset({"numpy"}), "numpy", "numpy", 0.0)
        return cls(frozenset({"arrow", "numpy"}), "numpy", None, 0.3)


# Operation -> Backend requirements mapping
# This is the "kernel registry" - which backends support which ops
BACKEND_KERNELS: dict[str, BackendRequirements] = {
    # Arithmetic - both backends efficient
    "add": BackendRequirements.any_backend(),
    "subtract": BackendRequirements.any_backend(),
    "multiply": BackendRequirements.any_backend(),
    "divide": BackendRequirements.any_backend(),
    "negate": BackendRequirements.any_backend(),
    "abs": BackendRequirements.any_backend(),

    # Comparison - both backends efficient
    "equal": BackendRequirements.any_backend(),
    "not_equal": BackendRequirements.any_backend(),
    "less": BackendRequirements.any_backend(),
    "greater": BackendRequirements.any_backend(),
    "less_equal": BackendRequirements.any_backend(),
    "greater_equal": BackendRequirements.any_backend(),

    # Logical - both backends
    "and_": BackendRequirements.any_backend(),
    "or_": BackendRequirements.any_backend(),
    "invert": BackendRequirements.any_backend(),

    # String operations - Arrow strongly preferred (or required)
    # In MVP, we might only implement Arrow path
    "str_lower": BackendRequirements.prefer_arrow(required=True),
    "str_upper": BackendRequirements.prefer_arrow(required=True),
    "str_len": BackendRequirements.prefer_arrow(required=True),
    "str_contains": BackendRequirements.prefer_arrow(required=True),
    "str_startswith": BackendRequirements.prefer_arrow(required=True),
    "str_endswith": BackendRequirements.prefer_arrow(required=True),
    "str_strip": BackendRequirements.prefer_arrow(required=True),
    "str_replace": BackendRequirements.prefer_arrow(required=True),
    "str_slice": BackendRequirements.prefer_arrow(required=True),

    # Datetime - both have support, Arrow slightly preferred
    "dt_year": BackendRequirements.prefer_arrow(),
    "dt_month": BackendRequirements.prefer_arrow(),
    "dt_day": BackendRequirements.prefer_arrow(),
    "dt_hour": BackendRequirements.prefer_arrow(),
    "dt_minute": BackendRequirements.prefer_arrow(),
    "dt_second": BackendRequirements.prefer_arrow(),

    # Aggregations - both efficient
    "sum": BackendRequirements.any_backend(),
    "mean": BackendRequirements.any_backend(),
    "min": BackendRequirements.any_backend(),
    "max": BackendRequirements.any_backend(),
    "count": BackendRequirements.any_backend(),

    # Null handling - Arrow has native support
    "is_null": BackendRequirements.prefer_arrow(),
    "is_not_null": BackendRequirements.prefer_arrow(),
    "fill_null": BackendRequirements.any_backend(),
    "coalesce": BackendRequirements.any_backend(),
}


def analyze_backend_requirements(
    expr: IRNode,
    schema: Schema
) -> BackendRequirements:
    """
    Analyze backend requirements for an expression tree.

    This considers:
    1. The operation's kernel availability
    2. Input column storage backends
    3. Nested expression requirements

    Returns combined requirements that satisfy the whole expression.
    """
    if isinstance(expr, FieldRef):
        # Column reference - report storage backend, but no execution requirement
        dtype = schema[expr.name]
        storage = "arrow" if dtype.arrow_type is not None else "numpy"
        # Storage is a hint, not a requirement - we CAN convert
        return BackendRequirements(
            supported_backends=frozenset({"arrow", "numpy"}),
            preferred_backend=storage,  # Prefer to stay in current storage
            requires_backend=None,
            conversion_cost=0.0,  # No conversion if we use storage backend
        )

    if isinstance(expr, Literal):
        # Literals can be used in any backend
        return BackendRequirements.any_backend()

    if isinstance(expr, Call):
        # Look up kernel requirements
        op_reqs = BACKEND_KERNELS.get(expr.function, BackendRequirements.any_backend())

        # Analyze children
        child_reqs = [
            analyze_backend_requirements(arg, schema)
            for arg in expr.args
        ]

        # Combine requirements
        return _combine_requirements(op_reqs, child_reqs)

    if isinstance(expr, Alias):
        return analyze_backend_requirements(expr.arg, schema)

    if isinstance(expr, Cast):
        return analyze_backend_requirements(expr.arg, schema)

    return BackendRequirements.any_backend()


def _combine_requirements(
    op_reqs: BackendRequirements,
    child_reqs: list[BackendRequirements]
) -> BackendRequirements:
    """
    Combine operation requirements with child requirements.

    Rules:
    - If op requires a backend, that wins
    - Otherwise, intersect supported backends
    - Preferred backend: op's preference, or majority of children
    - Conversion cost: sum of costs if mismatched
    """
    if op_reqs.requires_backend:
        # Operation requires specific backend
        required = op_reqs.requires_backend
        # Calculate conversion cost for children not in required backend
        cost = sum(
            0.5 for c in child_reqs
            if c.preferred_backend and c.preferred_backend != required
        )
        return BackendRequirements(
            supported_backends=frozenset({required}),
            preferred_backend=required,
            requires_backend=required,
            conversion_cost=cost,
        )

    # Intersect supported backends
    supported = op_reqs.supported_backends
    for c in child_reqs:
        if c.requires_backend:
            supported = supported & frozenset({c.requires_backend})

    if not supported:
        # No common backend - this shouldn't happen with good kernel registry
        supported = frozenset({"numpy"})  # Fallback

    # Determine preferred backend
    # Priority: op preference > majority of children > "numpy" default
    if op_reqs.preferred_backend and op_reqs.preferred_backend in supported:
        preferred = op_reqs.preferred_backend
    else:
        # Count child preferences
        arrow_count = sum(1 for c in child_reqs if c.preferred_backend == "arrow")
        numpy_count = sum(1 for c in child_reqs if c.preferred_backend == "numpy")
        if arrow_count > numpy_count and "arrow" in supported:
            preferred = "arrow"
        elif numpy_count > 0 and "numpy" in supported:
            preferred = "numpy"
        else:
            preferred = next(iter(supported))  # Pick any

    # Compute conversion cost
    cost = sum(c.conversion_cost for c in child_reqs)
    # Add cost for children that need conversion to preferred
    cost += sum(
        0.3 for c in child_reqs
        if c.preferred_backend and c.preferred_backend != preferred
    )

    return BackendRequirements(
        supported_backends=supported,
        preferred_backend=preferred,
        requires_backend=None,
        conversion_cost=cost,
    )
```

#### Updated Backend Grouping with Rich Requirements

```python
class BackendGrouping(OptimizationPass):
    """
    Reorder expressions within topological levels to minimize backend switches
    and conversion costs.
    """

    def _backend_sort_key(self, expr: Expr, schema: Schema) -> tuple[int, float]:
        """
        Return sort key: (backend_priority, conversion_cost).

        Sorts by:
        1. Required backend first (group required-arrow, then required-numpy)
        2. Then by preferred backend
        3. Then by conversion cost (lower first)
        """
        reqs = analyze_backend_requirements(expr._ir, schema)

        if reqs.requires_backend == "arrow":
            priority = 0
        elif reqs.requires_backend == "numpy":
            priority = 3
        elif reqs.preferred_backend == "arrow":
            priority = 1
        elif reqs.preferred_backend == "numpy":
            priority = 2
        else:
            priority = 2  # Default to numpy group

        return (priority, reqs.conversion_cost)
```

#### Schema Extension for Backend Tracking

```python
@dataclass
class LazyDtype:
    # Existing fields...

    @property
    def storage_backend(self) -> str:
        """Return the physical storage backend: 'numpy' or 'arrow'."""
        if self.arrow_type is not None:
            return "arrow"
        return "numpy"
```

### Engine Hints and Explicit Conversions

**Problem**: Pure logical optimization cannot reason about conversion minimization if backend decisions are deferred entirely to evaluation time. We need a middle ground.

**Solution**: Introduce engine hints and explicit Convert nodes - still "logical" but conversions are now explicit and optimizable.

#### Engine Hint IR Node

```python
@dataclass
class EngineHint:
    """
    Hint for preferred execution engine on a subexpression.

    This is advisory - the evaluator may override based on actual data.
    But it allows the optimizer to reason about and minimize conversions.
    """
    arg: IRNode
    engine: str  # "arrow" | "numpy" | "auto"

    def __repr__(self) -> str:
        return f"EngineHint({self.engine}: {self.arg!r})"
```

#### Explicit Convert Nodes

```python
@dataclass
class Convert(LogicalPlan):
    """
    Explicit conversion between backends.

    Inserted by the "Engine Selection Pass" to make conversions visible
    in the plan. This enables:
    - explain() to show conversion points
    - Optimizer to minimize/eliminate conversions
    - Cost estimation for backend choices
    """
    input: LogicalPlan
    target_backend: str  # "arrow" | "numpy"

    def resolve_schema(self) -> Schema:
        # Schema unchanged, only backend representation changes
        return self.input.resolve_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def __repr__(self) -> str:
        return f"Convert(to={self.target_backend!r})"
```

#### Engine Selection Pass

This pass analyzes backend requirements and inserts explicit Convert nodes:

```python
class EngineSelectionPass(OptimizationPass):
    """
    Analyze expression backend requirements and insert explicit conversions.

    This runs AFTER other optimizations (predicate pushdown, etc.) because:
    1. Those passes may move operations that affect backend decisions
    2. We want the final logical structure before deciding on backends

    Algorithm:
    1. For each plan node, analyze backend requirements of all expressions
    2. Determine the "dominant" backend for each node
    3. Insert Convert nodes at boundaries where backend changes
    4. Try to push conversions up/down to minimize total conversions
    """

    def optimize_node(self, node: LogicalPlan) -> LogicalPlan:
        # First, recursively optimize children
        node = self._optimize_children(node)

        if isinstance(node, Project):
            return self._select_project_engines(node)
        elif isinstance(node, Filter):
            return self._select_filter_engine(node)
        # ... other node types

        return node

    def _select_project_engines(self, project: Project) -> LogicalPlan:
        """
        Determine backend for Project and insert conversions if needed.
        """
        schema = project.input.resolve_schema()

        # Analyze all expressions
        expr_reqs = [
            analyze_backend_requirements(e._ir, schema)
            for e in project.exprs
        ]

        # Find required backends (any expression that MUST use specific backend)
        required_arrow = any(r.requires_backend == "arrow" for r in expr_reqs)
        required_numpy = any(r.requires_backend == "numpy" for r in expr_reqs)

        if required_arrow and required_numpy:
            # Conflict - need to split expressions (advanced case)
            # For MVP, just use arrow for required-arrow, numpy for rest
            return self._split_project_by_backend(project, expr_reqs)

        # Determine dominant backend
        if required_arrow:
            target = "arrow"
        elif required_numpy:
            target = "numpy"
        else:
            # Use preferred backend from majority
            arrow_pref = sum(1 for r in expr_reqs if r.preferred_backend == "arrow")
            numpy_pref = sum(1 for r in expr_reqs if r.preferred_backend == "numpy")
            target = "arrow" if arrow_pref > numpy_pref else "numpy"

        # Check if input needs conversion
        input_backend = self._get_plan_backend(project.input)
        if input_backend != target and input_backend != "auto":
            # Insert conversion
            converted_input = Convert(project.input, target)
            return Project(converted_input, project.exprs)

        return project

    def _get_plan_backend(self, plan: LogicalPlan) -> str:
        """
        Determine the output backend of a plan node.

        Returns "auto" if mixed or unknown.
        """
        if isinstance(plan, Convert):
            return plan.target_backend
        elif isinstance(plan, DataFrameSource):
            # Check actual DataFrame column backends
            schema = plan.resolve_schema()
            backends = {schema[n].storage_backend for n in schema.names}
            if len(backends) == 1:
                return backends.pop()
            return "auto"  # Mixed
        else:
            # Propagate from child
            children = plan.children()
            if children:
                return self._get_plan_backend(children[0])
            return "auto"
```

#### Conversion Elimination Pass

After engine selection, we can optimize away unnecessary conversions:

```python
class ConversionEliminationPass(OptimizationPass):
    """
    Eliminate redundant or unnecessary Convert nodes.

    Patterns eliminated:
    1. Convert(Convert(x, "arrow"), "arrow") -> Convert(x, "arrow")
    2. Convert(x, backend) where x already produces that backend
    3. Adjacent converts that cancel: Convert(Convert(x, "arrow"), "numpy")
       where the arrow region is empty (nothing uses arrow-only ops)
    """

    def optimize_node(self, node: LogicalPlan) -> LogicalPlan:
        if not isinstance(node, Convert):
            return node

        # Pattern 1: Redundant double convert to same backend
        if isinstance(node.input, Convert):
            if node.input.target_backend == node.target_backend:
                # Convert(Convert(x, A), A) -> Convert(x, A)
                return Convert(node.input.input, node.target_backend)

        # Pattern 2: Input already in target backend
        input_backend = self._get_plan_backend(node.input)
        if input_backend == node.target_backend:
            return node.input  # Eliminate unnecessary convert

        return node
```

#### Updated explain() Output

With explicit Convert nodes, `explain()` can show backend decisions:

```
LazyDataFrame Plan:
├─ Project(['id', 'name_upper', 'score'])
│  └─ Convert(to='arrow')                    # <-- Now visible!
│     └─ Filter(col("active") == True)
│        └─ DataFrameSource(columns=['id', 'name', 'score', 'active'], rows=1000)

Expressions:
  - id: col("id")
  - name_upper: str_upper(col("name"))       # Requires Arrow
  - score: col("score")
```

#### Schema Backend Tracking

To support this, extend Schema/LazyDtype:

```python
@dataclass
class LazyDtype:
    # Existing fields...

    @property
    def storage_backend(self) -> str:
        """Return the physical storage backend: 'numpy' or 'arrow'."""
        if self.arrow_type is not None:
            return "arrow"
        return "numpy"


# In Schema
def dominant_backend(self) -> str:
    """Return the dominant storage backend for this schema."""
    backends = [self.fields[n].storage_backend for n in self.names]
    arrow_count = sum(1 for b in backends if b == "arrow")
    return "arrow" if arrow_count > len(backends) // 2 else "numpy"
```

### Execution-Time Backend Handling

The evaluator uses the engine hints and Convert nodes as guidance:

```python
class Evaluator:
    def evaluate_plan(self, plan: LogicalPlan) -> pd.DataFrame:
        if isinstance(plan, Convert):
            # Explicit conversion requested by optimizer
            result = self.evaluate_plan(plan.input)
            return self._convert_dataframe(result, plan.target_backend)
        # ... rest of evaluation

    def _convert_dataframe(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Convert DataFrame columns to target backend."""
        if target == "arrow":
            return df.astype({
                col: pd.ArrowDtype(pa.from_numpy_dtype(df[col].dtype))
                for col in df.columns
                if not isinstance(df[col].dtype, pd.ArrowDtype)
            })
        else:  # numpy
            # Convert Arrow columns to numpy
            result = df.copy()
            for col in df.columns:
                if isinstance(df[col].dtype, pd.ArrowDtype):
                    result[col] = df[col].to_numpy()
            return result
```

### Why This Middle Ground?

**Benefits over pure eval-time decisions:**
1. **Explainability**: `explain()` shows where conversions happen
2. **Optimization**: Can minimize conversions in the optimizer
3. **Predictability**: Same query produces consistent backend behavior
4. **Cost estimation**: Can estimate conversion costs for query planning

**Benefits over full physical planning:**
1. **Simplicity**: Still just logical nodes with hints
2. **Flexibility**: Evaluator can still adapt to actual data
3. **Incremental**: Can add full physical planning later if needed

### Future: Full Physical Planning

If needed later, the architecture supports introducing **physical planning**:

```
Logical Plan -> Optimizer -> Optimized Logical Plan -> Physical Planner -> Physical Plan
```

Physical plan nodes would be fully backend-specific:
- `ArrowProject` vs `NumPyProject`
- `ArrowFilter` vs `NumPyFilter`
- Explicit data movement between operators

The current engine hints approach provides 80% of the benefit while keeping the door open for full physical planning.

## Proposed Architecture

```
User Query -> Expr API -> Logical Plan -> **Optimizer** -> Optimized Plan -> Evaluation
```

New module: `pandas/lazy/optimize.py`

## Optimization Passes

We'll implement a **rule-based optimizer** with multiple transformation passes. Each pass traverses the plan tree and applies transformations.

### Phase 1: Core Optimizations (High Impact)

#### 1.1 Predicate Pushdown

**Goal**: Move Filter nodes closer to data sources to reduce data volume early.

**Critical Issue: Name Resolution and Lineage**

Predicates may reference:
1. **Pass-through columns** - columns that exist in input and are passed through unchanged
2. **Computed columns** - columns created by expressions in a Project (aliases)
3. **Renamed columns** - columns where input name differs from output name

**Example problem**:
```python
# Project creates alias "a" from expression "x + 1"
Project: [col("x"), (col("x") + 1).alias("a")]
Filter: col("a") > 0

# WRONG: Cannot just push Filter below Project - "a" doesn't exist in input!
# OPTION 1: Don't push (conservative)
# OPTION 2: Rewrite predicate to (col("x") + 1) > 0 (requires lineage tracking)
```

**Lineage Tracking**:
```python
def build_project_lineage(project: Project) -> dict[str, IRNode | None]:
    """
    Build mapping from output column names to their source expressions.

    Returns:
        dict mapping output_name -> source expression (or None if pass-through)

    For a Project with exprs like:
        [col("x"), (col("y") + 1).alias("a"), col("z").alias("renamed")]

    Returns:
        {"x": FieldRef("x"),           # pass-through
         "a": Call("add", ...),         # computed
         "renamed": FieldRef("z")}      # renamed
    """
    lineage = {}
    for expr in project.exprs:
        output_name = extract_output_name(expr)
        ir = expr._ir

        # Unwrap Alias to get the actual expression
        if isinstance(ir, Alias):
            source_ir = ir.arg
        else:
            source_ir = ir

        lineage[output_name] = source_ir
    return lineage

def is_simple_column_ref(ir: IRNode) -> bool:
    """Check if IR is just a column reference (possibly aliased)."""
    if isinstance(ir, FieldRef):
        return True
    if isinstance(ir, Alias) and isinstance(ir.arg, FieldRef):
        return True
    return False

def get_source_column(ir: IRNode) -> str | None:
    """Get the source column name if IR is a simple column reference."""
    if isinstance(ir, FieldRef):
        return ir.name
    if isinstance(ir, Alias) and isinstance(ir.arg, FieldRef):
        return ir.arg.name
    return None
```

**Predicate Pushdown Strategies**:

**Strategy 1: Conservative (MVP)** - Only push if ALL referenced columns are pass-through
```python
def can_push_filter_through_project(filter_plan: Filter, project: Project) -> bool:
    """Check if filter can be pushed below project (conservative approach)."""
    lineage = build_project_lineage(project)
    input_schema = project.input.resolve_schema()

    pred_columns = get_referenced_columns(filter_plan.predicate._ir)

    for col_name in pred_columns:
        if col_name not in lineage:
            return False  # Column doesn't exist in project output

        source_ir = lineage[col_name]
        source_col = get_source_column(source_ir)

        if source_col is None:
            # It's a computed expression, not a simple column ref
            return False

        if source_col not in input_schema:
            return False  # Source column doesn't exist in input

    return True
```

**Strategy 2: Rewriting (Advanced)** - Rewrite predicate using lineage
```python
def rewrite_predicate_for_pushdown(
    predicate: IRNode,
    lineage: dict[str, IRNode],
    max_expr_size: int = 10
) -> IRNode | None:
    """
    Rewrite predicate in terms of input columns using lineage.

    Returns None if rewriting would create expressions larger than max_expr_size.

    Example:
        predicate: col("a") > 0
        lineage: {"a": Call("add", (FieldRef("x"), Literal(1)))}
        result: Call("add", (FieldRef("x"), Literal(1))) > 0
    """
    def rewrite(node: IRNode) -> IRNode | None:
        if isinstance(node, FieldRef):
            if node.name in lineage:
                replacement = lineage[node.name]
                if count_nodes(replacement) > max_expr_size:
                    return None  # Too complex, don't duplicate
                return replacement
            return node  # Not in lineage, assume it's from input

        elif isinstance(node, Call):
            new_args = []
            for arg in node.args:
                rewritten = rewrite(arg)
                if rewritten is None:
                    return None
                new_args.append(rewritten)
            return Call(node.function, tuple(new_args), node.kwargs, node.is_aggregate)

        # Handle other node types...
        return node

    return rewrite(predicate)

def count_nodes(ir: IRNode) -> int:
    """Count total nodes in IR tree (for complexity estimation)."""
    if isinstance(ir, (FieldRef, Literal)):
        return 1
    elif isinstance(ir, Call):
        return 1 + sum(count_nodes(arg) for arg in ir.args)
    elif isinstance(ir, Alias):
        return 1 + count_nodes(ir.arg)
    return 1
```

**Rules** (updated):
- `Filter(Project(...))` → Push only if predicate references pass-through columns (MVP)
- `Filter(Project(...))` → Optionally rewrite predicate using lineage (advanced, controlled by flag)
- `Filter(Join(...))` → Push filter to left/right child if columns are from one side only
- `Filter(Filter(...))` → Combine predicates with AND

**Example (MVP - conservative)**:
```python
# CAN push: filter on pass-through column
Project: [col("x"), col("y"), (col("x") + 1).alias("a")]
Filter: col("x") > 0
# x is pass-through, so Filter can move below Project

# CANNOT push: filter on computed column
Project: [col("x"), (col("x") + 1).alias("a")]
Filter: col("a") > 0
# a is computed, Filter stays above Project (unless we rewrite)
```

**Implementation**:
1. Build lineage mapping for the Project
2. For each column in predicate, check if it's a pass-through
3. Only push if ALL columns are pass-throughs (MVP)
4. (Advanced) Optionally rewrite predicate using lineage, with complexity limit

#### 1.2 Projection Pruning

**Goal**: Only read/compute columns that are actually used downstream.

**Critical Issue: Required Columns Analysis**

Pruning is NOT just about output columns. Must account for columns needed by:
- **Filters** - predicate references
- **Join keys** - left_on, right_on, on columns
- **GroupBy keys** - grouping columns
- **Aggregation inputs** - columns being aggregated
- **Sort keys** - sorting expressions
- **Distinct subset** - columns used for uniqueness
- **Window partitions** - partition_by columns

**Required Columns Analysis**:

```python
def compute_required_columns(plan: LogicalPlan, needed_downstream: set[str]) -> set[str]:
    """
    Compute columns required from this plan node's input(s).

    Parameters:
        plan: The plan node to analyze
        needed_downstream: Columns needed by nodes above this one

    Returns:
        Set of column names this node requires from its input(s)
    """
    if isinstance(plan, DataFrameSource):
        # Source node: return what's needed (will be used for column selection)
        return needed_downstream

    elif isinstance(plan, Project):
        # Project needs: columns referenced by expressions that produce needed output
        required = set()
        for expr in plan.exprs:
            output_name = extract_output_name(expr)
            if output_name in needed_downstream:
                # This expression is needed, so we need its input columns
                required |= get_referenced_columns(expr._ir)
        return required

    elif isinstance(plan, Filter):
        # Filter needs: downstream columns + predicate columns
        pred_cols = get_referenced_columns(plan.predicate._ir)
        return needed_downstream | pred_cols

    elif isinstance(plan, Aggregate):
        # Aggregate needs: groupby columns + aggregation input columns
        required = set()
        for expr in plan.group_by:
            required |= get_referenced_columns(expr._ir)
        for expr in plan.agg_exprs:
            required |= get_referenced_columns(expr._ir)
        return required

    elif isinstance(plan, Sort):
        # Sort needs: downstream columns + sort key columns
        sort_cols = set()
        for expr in plan.by:
            sort_cols |= get_referenced_columns(expr._ir)
        return needed_downstream | sort_cols

    elif isinstance(plan, Limit):
        # Limit passes through requirements unchanged
        return needed_downstream

    elif isinstance(plan, Distinct):
        # Distinct needs: downstream + subset columns (or all if no subset)
        if plan.subset:
            return needed_downstream | set(plan.subset)
        else:
            # No subset means all columns used for uniqueness
            return needed_downstream  # But we can't prune - need all

    elif isinstance(plan, Join):
        # Join is complex - see below
        raise NotImplementedError("Join requires split analysis")

    return needed_downstream
```

**Join Column Requirements (Complex Case)**:

```python
def compute_join_required_columns(
    join: Join,
    needed_downstream: set[str]
) -> tuple[set[str], set[str]]:
    """
    Compute columns required from left and right inputs of a join.

    Returns:
        (left_required, right_required)
    """
    left_schema = join.left.resolve_schema()
    right_schema = join.right.resolve_schema()

    left_required = set()
    right_required = set()

    # 1. Join keys are always required
    if join.on:
        for col in join.on:
            left_required.add(col)
            right_required.add(col)
    elif join.left_on and join.right_on:
        left_required |= set(join.left_on)
        right_required |= set(join.right_on)

    # 2. Downstream columns - map back to source side
    for col in needed_downstream:
        # Handle suffixed columns (from overlapping names)
        if col.endswith(join.suffix[0]):
            base = col[:-len(join.suffix[0])]
            if base in left_schema:
                left_required.add(base)
                continue
        if col.endswith(join.suffix[1]):
            base = col[:-len(join.suffix[1])]
            if base in right_schema:
                right_required.add(base)
                continue

        # Non-suffixed column - could be from either side
        if col in left_schema:
            left_required.add(col)
        if col in right_schema and (join.on is None or col not in join.on):
            right_required.add(col)

    return left_required, right_required
```

**Top-Down Pruning Pass**:

```python
class ProjectionPruning(OptimizationPass):
    """
    Remove unnecessary columns from projections.

    Algorithm:
    1. Start at root with output columns as "needed"
    2. Walk down the tree, computing required columns at each level
    3. Modify Project nodes to only include required expressions
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        # Start with all output columns as needed
        output_cols = set(plan.resolve_schema().names)
        return self._prune(plan, output_cols)

    def _prune(self, plan: LogicalPlan, needed: set[str]) -> LogicalPlan:
        if isinstance(plan, DataFrameSource):
            # Can't prune source in logical plan (would need physical plan)
            return plan

        elif isinstance(plan, Project):
            # Filter expressions to only those producing needed columns
            new_exprs = []
            child_needed = set()
            for expr in plan.exprs:
                output_name = extract_output_name(expr)
                if output_name in needed:
                    new_exprs.append(expr)
                    child_needed |= get_referenced_columns(expr._ir)

            if not new_exprs:
                # Edge case: nothing needed? Keep at least one column
                new_exprs = [plan.exprs[0]]
                child_needed = get_referenced_columns(plan.exprs[0]._ir)

            new_input = self._prune(plan.input, child_needed)
            return Project(new_input, tuple(new_exprs))

        elif isinstance(plan, Filter):
            pred_cols = get_referenced_columns(plan.predicate._ir)
            child_needed = needed | pred_cols
            new_input = self._prune(plan.input, child_needed)
            return Filter(new_input, plan.predicate)

        elif isinstance(plan, Aggregate):
            # Aggregates define their own column requirements
            child_needed = set()
            for expr in plan.group_by:
                child_needed |= get_referenced_columns(expr._ir)
            for expr in plan.agg_exprs:
                child_needed |= get_referenced_columns(expr._ir)
            new_input = self._prune(plan.input, child_needed)
            return Aggregate(new_input, plan.group_by, plan.agg_exprs)

        elif isinstance(plan, Sort):
            sort_cols = set()
            for expr in plan.by:
                sort_cols |= get_referenced_columns(expr._ir)
            child_needed = needed | sort_cols
            new_input = self._prune(plan.input, child_needed)
            return Sort(new_input, plan.by, plan.descending)

        elif isinstance(plan, Limit):
            new_input = self._prune(plan.input, needed)
            return Limit(new_input, plan.n, plan.offset)

        elif isinstance(plan, Distinct):
            if plan.subset:
                child_needed = needed | set(plan.subset)
            else:
                # Without subset, all columns matter for uniqueness
                child_needed = set(plan.input.resolve_schema().names)
            new_input = self._prune(plan.input, child_needed)
            return Distinct(new_input, plan.subset)

        elif isinstance(plan, Join):
            left_needed, right_needed = compute_join_required_columns(plan, needed)
            new_left = self._prune(plan.left, left_needed)
            new_right = self._prune(plan.right, right_needed)
            return Join(
                new_left, new_right,
                plan.on, plan.left_on, plan.right_on,
                plan.how, plan.suffix
            )

        return plan
```

**Example**:
```python
# User code
df.select("a", "b", "c").filter(col("a") > 0).select("a")

# Required columns analysis (top-down):
# - Final select needs: {"a"}
# - Filter needs: {"a"} (downstream) ∪ {"a"} (predicate) = {"a"}
# - First select needs: {"a"} from source

# Before optimization: Project(a,b,c) -> Filter -> Project(a)
# After: Project(a) -> Filter -> Project(a)
# (First project pruned to just "a")
```

**Implementation**:
1. Implement `compute_required_columns()` for each plan node type
2. Implement `compute_join_required_columns()` for Join
3. Walk plan top-down, pruning Project nodes
4. Handle edge cases (empty projects, distinct without subset)

#### 1.3 Filter Combination (Predicate Fusion)

**Goal**: Merge consecutive filters into one.

**Rule**: `Filter(pred1, Filter(pred2, X))` → `Filter(pred1 AND pred2, X)`

**Benefit**: Single filter pass instead of multiple.

### Phase 2: Join Optimizations (Medium Impact)

#### 2.1 Filter Pushdown Through Joins

**Rules**:
- If filter predicate only references left table columns → push to left child
- If filter predicate only references right table columns → push to right child
- Split predicates on AND to push each part independently

#### 2.2 Join Reordering (Future)

Not in initial implementation. Would require:
- Cost model for estimating result sizes
- Statistics collection

### Phase 3: Limit Optimizations (Quick Wins)

#### 3.1 Limit Pushdown

**Rules**:
- `Limit(Project(...))` → `Project(Limit(...))` - apply limit first
- `Limit(n, Limit(m, X))` → `Limit(min(n, m), X)` - combine limits
- `Limit(Filter(...))` - limit can't push through filter (filter might eliminate rows)

#### 3.2 Distinct + Limit

For `distinct().head(n)` patterns, we can short-circuit early when distinct count reaches n.

### Phase 4: Expression Simplification (Low Impact but Useful)

#### 4.1 Constant Folding

**Rule**: Evaluate constant expressions at plan time.

**Examples**:
- `Literal(2) + Literal(3)` → `Literal(5)`
- `col("a") * Literal(1)` → `col("a")`
- `col("a") + Literal(0)` → `col("a")`

#### 4.2 Dead Expression Elimination

Remove expressions from Project that are never used downstream.

## Implementation Plan

### File Structure

```
pandas/lazy/
├── optimize.py          # NEW: Optimizer module
│   ├── Optimizer class
│   ├── OptimizationPass (base class)
│   ├── PredicatePushdown
│   ├── ProjectionPruning
│   ├── FilterFusion
│   ├── LimitPushdown
│   └── ConstantFolding
└── frame.py             # Modified: integrate optimizer in collect()
```

### Key Classes

```python
# optimize.py

class OptimizationPass(ABC):
    """Base class for optimization passes."""

    @abstractmethod
    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Transform a plan, returning optimized version."""
        pass

class Optimizer:
    """Coordinates optimization passes."""

    def __init__(self, passes: list[OptimizationPass] | None = None):
        self.passes = passes or self._default_passes()

    def _default_passes(self) -> list[OptimizationPass]:
        """
        Default optimization pass ordering.

        IMPORTANT: Order matters! Rationale for each position:

        1. FilterFusion - Combine filters first so pushdown sees single filters
        2. PredicatePushdown - Push filters down while plan is still wide
           (before projection pruning removes columns that filters might need)
        3. ProjectionPruning - After pushdown, so we don't prune filter-needed cols
        4. LimitPushdown - After pruning; could also optimize sort->limit to top-k
        5. EngineSelection - Insert Convert nodes, decide engine regions
           (runs last on logical plan because earlier passes may move operations)
        6. ConversionElimination - Clean up redundant conversions

        NOT included by default (optional/future):
        - ConstantFolding - Only if stable; can enable explicitly
        - CSE (Common Subexpression Elimination) - Advanced optimization

        Note: BackendGrouping is folded into EngineSelection. Expression
        reordering within Project nodes happens as part of engine region
        analysis, respecting topological dependencies.
        """
        return [
            FilterFusion(),           # 1. Combine consecutive filters
            PredicatePushdown(),      # 2. Push filters toward sources
            ProjectionPruning(),      # 3. Remove unused columns
            LimitPushdown(),          # 4. Push limits toward sources
            EngineSelection(),        # 5. Insert backend conversions
            ConversionElimination(),  # 6. Remove redundant conversions
        ]

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply all optimization passes."""
        for pass_ in self.passes:
            plan = pass_.optimize(plan)
        return plan
```

### Pass Ordering Rationale

The ordering of passes is critical for correctness and optimization quality:

```
┌─────────────────┐
│  Original Plan  │
└────────┬────────┘
         ▼
┌─────────────────┐   Combine: Filter(Filter(x, p1), p2) → Filter(x, p1 AND p2)
│  FilterFusion   │   WHY FIRST: Single filter is easier to push down
└────────┬────────┘
         ▼
┌─────────────────┐   Move filters closer to data sources
│PredicatePushdown│   WHY BEFORE PRUNING: Filters may reference columns
└────────┬────────┘   that would otherwise be pruned
         ▼
┌─────────────────┐   Remove columns not needed downstream
│ProjectionPruning│   WHY AFTER PUSHDOWN: Filter columns now in correct
└────────┬────────┘   position, won't be incorrectly pruned
         ▼
┌─────────────────┐   Push LIMIT toward source (or combine with SORT for top-k)
│  LimitPushdown  │   WHY AFTER PRUNING: Fewer columns to sort/limit
└────────┬────────┘
         ▼
┌─────────────────┐   Analyze backend requirements, insert Convert nodes
│ EngineSelection │   WHY LAST ON LOGIC: All operation movements done,
└────────┬────────┘   can now make final backend decisions
         ▼
┌─────────────────┐   Remove Convert(Convert(x, A), A) patterns
│ConversionElim.  │   WHY AFTER ENGINE: Clean up any redundant converts
└────────┬────────┘
         ▼
┌─────────────────┐
│  Optimized Plan │
└─────────────────┘
```

**Anti-patterns to avoid:**
- ProjectionPruning BEFORE PredicatePushdown: May prune columns needed by filters
- EngineSelection BEFORE logical optimizations: Operations may move after engine decision
- LimitPushdown BEFORE ProjectionPruning: More columns to process through limit

### Utility Functions Needed

```python
def get_referenced_columns(expr: Expr | IRNode) -> set[str]:
    """Extract all column names referenced in an expression."""
    pass

def substitute_columns(node: IRNode, mapping: dict[str, str]) -> IRNode:
    """Rename columns in an expression tree."""
    pass

def transform_plan(plan: LogicalPlan, visitor: Callable) -> LogicalPlan:
    """Apply transformation function to each node bottom-up."""
    pass
```

### Integration with collect()

```python
# In frame.py

def collect(self, *, optimize: bool = True, ...) -> DataFrame:
    plan = self._plan
    if optimize:
        from pandas.lazy.optimize import Optimizer
        plan = Optimizer().optimize(plan)
    return self._execute(plan)
```

### Integration with explain()

```python
def explain(self, *, optimized: bool = True, ...) -> str:
    if optimized:
        from pandas.lazy.optimize import Optimizer
        plan = Optimizer().optimize(self._plan)
    else:
        plan = self._plan
    return self._format_plan(plan)
```

## Testing Strategy

### Unit Tests for Each Pass

```python
class TestPredicatePushdown:
    def test_filter_through_project(self):
        # Filter after project pushes down
        pass

    def test_filter_preserves_when_column_computed(self):
        # Filter on computed column stays in place
        pass

    def test_filter_through_join_left(self):
        # Filter on left columns pushes to left child
        pass

class TestProjectionPruning:
    def test_unused_column_removed(self):
        pass

    def test_filter_column_preserved(self):
        # Column used in filter but not output is preserved until filter
        pass

class TestFilterFusion:
    def test_consecutive_filters_merged(self):
        pass
```

### Integration Tests

```python
def test_optimization_produces_same_result():
    """Optimized and unoptimized plans produce identical results."""
    df = pd.DataFrame(...)
    query = df.select().filter(...).select(...)

    result_unopt = query.collect(optimize=False)
    result_opt = query.collect(optimize=True)

    tm.assert_frame_equal(result_unopt, result_opt)
```

### Performance Tests (Optional)

Benchmark optimized vs unoptimized on large DataFrames to verify optimization benefit.

## Implementation Order

### Step 1: Framework & Utilities
- Create `optimize.py` with base classes (`Optimizer`, `OptimizationPass`)
- Add `get_referenced_columns()` utility function
- Add `optimize` parameter to `collect()`
- Update `explain()` to show both plans
- Tests for utility functions

### Step 2: Filter Fusion
- Combine consecutive `Filter` nodes: `Filter(p1, Filter(p2, X))` → `Filter(p1 AND p2, X)`
- Create `and_` IR node combiner
- Tests

### Step 3: Predicate Pushdown (MVP - Conservative)
- Add `build_project_lineage()` utility
- Add `is_simple_column_ref()` and `get_source_column()` utilities
- Implement `can_push_filter_through_project()` (conservative: pass-through only)
- Push filter through `Project` only when ALL predicate columns are pass-throughs
- Push filter through `Join` (to left/right child based on column ownership)
- Tests for each pushdown case
- Tests for cases where pushdown is NOT safe (computed columns)

### Step 3b: Predicate Pushdown (Advanced - Optional)
- Add `rewrite_predicate_for_pushdown()` with complexity limit
- Add `count_nodes()` for expression size estimation
- Enable rewriting for small expressions (configurable threshold)
- Tests for predicate rewriting

### Step 4: Projection Pruning
- Column usage analysis (walk plan to find used columns)
- Prune unnecessary columns from `Project` nodes
- Tests

### Step 5: Limit Pushdown
- Push `Limit` through `Project`
- Combine consecutive `Limit` nodes
- Tests

### Step 6: Backend-Aware Optimization (Phase 2)
- Add `backend` property to `LazyDtype`
- Implement `get_backend_preference()` for expressions
- Add `BackendGrouping` pass to merge/reorder projections
- Tests with Arrow-backed DataFrames

### Step 7: Evaluator Backend Awareness
- Detect column backends at evaluation time
- Batch format conversions
- Use native Arrow compute when available
- Tests for mixed-backend scenarios

## Considerations

### Safety

Every optimization must be **semantics-preserving**:
- Same output for same input
- Same column order
- Same data types

### Debugging

When `explain(optimized=False)` users can see original plan.
When `explain(optimized=True)` users can see what optimizer did.

### Opt-out

`collect(optimize=False)` for debugging or when optimization causes issues.

## Critical Design Decision: Join Column Disambiguation

### The Problem

When both sides of a Join have a column named "a", `FieldRef("a")` becomes ambiguous after the join:

```python
left = df1.select()   # has column "a"
right = df2.select()  # also has column "a"
joined = left.join(right, on="id")  # Output has "a_x" and "a_y"

# User writes:
joined.filter(col("a") > 0)  # Which "a"? Ambiguous!
```

This affects:
- **Predicate pushdown**: Need to know which side owns a column to push filter correctly
- **Projection pruning**: Need to map suffixed output names back to input names
- **Expression evaluation**: `col("a")` after join is invalid

### Solution: Suffix-Based Disambiguation (Option B)

We adopt pandas' approach: overlapping non-join columns get suffixes in the output schema.

**Key Principles:**

1. **Join output schema uses suffixes**: Column "a" from left becomes "a_x", from right becomes "a_y"
2. **Users must use suffixed names after join**: `col("a_x")` not `col("a")`
3. **Join columns (on=) are NOT suffixed**: They appear once in output
4. **Optimizer tracks column provenance**: Maps suffixed names back to source columns

### Implementation Details

#### 1. Schema Already Handles This

The existing `Join.resolve_schema()` already adds suffixes for overlapping columns:

```python
# In Join.resolve_schema() - already implemented
if name in right_schema and name not in join_cols:
    columns[name + self.suffix[0]] = dtype  # "a" -> "a_x"
```

#### 2. Column Provenance Tracking for Optimization

```python
@dataclass
class JoinColumnMapping:
    """
    Tracks where columns in a Join output come from.

    Used by optimizer to:
    - Push predicates to correct side
    - Prune columns from correct input
    """
    left_columns: dict[str, str]   # output_name -> left_input_name
    right_columns: dict[str, str]  # output_name -> right_input_name
    join_columns: set[str]         # columns used for join (appear once)

def build_join_column_mapping(join: Join) -> JoinColumnMapping:
    """Build mapping from join output columns to their source."""
    left_schema = join.left.resolve_schema()
    right_schema = join.right.resolve_schema()

    left_cols = {}
    right_cols = {}
    join_cols = set()

    # Determine join columns
    if join.on:
        join_cols = set(join.on)

    # Map left columns
    for name in left_schema.names:
        if name in right_schema and name not in join_cols:
            # Overlapping -> gets suffix
            output_name = name + join.suffix[0]
        else:
            output_name = name
        left_cols[output_name] = name

    # Map right columns
    for name in right_schema.names:
        if join.on and name in join_cols:
            continue  # Skip, already from left
        if name in left_schema and name not in join_cols:
            output_name = name + join.suffix[1]
        else:
            output_name = name
        right_cols[output_name] = name

    return JoinColumnMapping(left_cols, right_cols, join_cols)
```

#### 3. Predicate Pushdown Through Join

```python
def can_push_predicate_through_join(
    predicate_cols: set[str],
    join_mapping: JoinColumnMapping
) -> tuple[bool, bool, set[str], set[str]]:
    """
    Determine if/how predicate can be pushed through join.

    Returns:
        (can_push_left, can_push_right, left_cols, right_cols)

    Rules:
    - Predicate on only left columns -> push to left
    - Predicate on only right columns -> push to right
    - Predicate on join columns -> push to BOTH sides
    - Predicate mixing left/right non-join columns -> cannot push
    """
    left_refs = set()
    right_refs = set()

    for col in predicate_cols:
        if col in join_mapping.join_columns:
            # Join column exists in both - can push to either
            left_refs.add(col)
            right_refs.add(col)
        elif col in join_mapping.left_columns:
            left_refs.add(join_mapping.left_columns[col])
        elif col in join_mapping.right_columns:
            right_refs.add(join_mapping.right_columns[col])
        else:
            # Column doesn't exist in join output
            return (False, False, set(), set())

    # Can only push if ALL columns are from one side (or join columns)
    only_left = len(right_refs - join_mapping.join_columns) == 0
    only_right = len(left_refs - join_mapping.join_columns) == 0

    return (only_left, only_right, left_refs, right_refs)
```

#### 4. Validation at Query Build Time

```python
def validate_column_ref(col_name: str, schema: Schema, plan: LogicalPlan) -> None:
    """
    Validate that a column reference is unambiguous.

    Raises ValueError with helpful message if ambiguous.
    """
    if col_name not in schema:
        # Check if this might be an ambiguous reference after join
        if isinstance(plan, Join):
            left_schema = plan.left.resolve_schema()
            right_schema = plan.right.resolve_schema()
            if col_name in left_schema and col_name in right_schema:
                raise ValueError(
                    f"Column '{col_name}' is ambiguous after join. "
                    f"Use '{col_name}{plan.suffix[0]}' for left or "
                    f"'{col_name}{plan.suffix[1]}' for right."
                )
        raise KeyError(f"Column '{col_name}' not found in schema. "
                      f"Available: {schema.names}")
```

### Edge Cases

1. **Nested joins**: Each join level adds its own suffixes
   - `a` -> `a_x` after first join
   - `a_x` -> `a_x_x` if joined again with table having `a_x`

2. **Join on suffixed column**: User can join on `a_x` if it exists
   - Treated as normal column name

3. **Self-join**: Same table joined with itself
   - All columns get suffixed (except join key)
   - `df.join(df, on="id")` produces `col_x`, `col_y` for all non-id columns

### Testing

```python
class TestJoinColumnDisambiguation:
    def test_overlapping_columns_get_suffix(self):
        left = pd.DataFrame({"id": [1], "a": [10]})
        right = pd.DataFrame({"id": [1], "a": [20]})
        result = left.select().join(right.select(), on="id").collect()
        assert "a_x" in result.columns
        assert "a_y" in result.columns
        assert "a" not in result.columns

    def test_filter_on_suffixed_column(self):
        left = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
        right = pd.DataFrame({"id": [1, 2], "a": [100, 200]})
        result = (
            left.select()
            .join(right.select(), on="id")
            .filter(col("a_x") > 15)
            .collect()
        )
        assert len(result) == 1
        assert result["a_x"].iloc[0] == 20

    def test_ambiguous_ref_raises_error(self):
        left = pd.DataFrame({"id": [1], "a": [10]})
        right = pd.DataFrame({"id": [1], "a": [20]})
        joined = left.select().join(right.select(), on="id")
        with pytest.raises(ValueError, match="ambiguous"):
            joined.filter(col("a") > 0).collect()

    def test_predicate_pushdown_through_join(self):
        # Predicate on left-only column can be pushed
        # Predicate on right-only column can be pushed
        # Predicate on join column can be pushed to both
        # Predicate mixing sides cannot be pushed
        pass
```

## Edge Cases to Handle

### General Edge Cases
1. **Self-referential expressions**: `col("a") + col("a")` - column "a" referenced twice
2. **Aliased columns**: Filter on aliased name vs original name
3. **Window functions**: Partitions may reference columns not in output
4. **Aggregations**: Filter on aggregate result can't push below aggregation
5. **Joins with overlapping names**: Now handled via suffix disambiguation (see above)

### Predicate Pushdown Edge Cases
10. **Filter on computed column**: `filter(col("a") > 0)` where "a" is `(col("x") + 1).alias("a")` - cannot push without rewriting
11. **Filter on renamed column**: `filter(col("renamed") > 0)` where "renamed" is `col("original").alias("renamed")` - CAN push but must use "original" in pushed predicate
12. **Mixed predicate**: `filter((col("x") > 0) & (col("a") > 0))` where "x" is pass-through but "a" is computed - can split and push only the "x" part (advanced)
13. **Stacked Projects**: Multiple Projects in sequence, each with different lineage
14. **Filter through Aggregate**: Cannot push filter on aggregate result below the aggregation

### Testing for Predicate Pushdown
```python
class TestPredicatePushdown:
    def test_push_through_passthrough(self):
        """Filter on pass-through column can be pushed."""
        # Project: [col("x"), col("y")]
        # Filter: col("x") > 0
        # Result: Project over Filter over Source
        pass

    def test_no_push_computed_column(self):
        """Filter on computed column stays above Project."""
        # Project: [col("x"), (col("x") + 1).alias("a")]
        # Filter: col("a") > 0
        # Result: Filter stays above Project (no pushdown)
        pass

    def test_push_renamed_column(self):
        """Filter on renamed column pushes with original name."""
        # Project: [col("x").alias("renamed")]
        # Filter: col("renamed") > 0
        # Result: Filter(col("x") > 0) pushed below Project
        pass

    def test_no_push_through_aggregate(self):
        """Filter on aggregate result cannot push below aggregation."""
        # Aggregate: group_by("g").agg(col("x").sum().alias("total"))
        # Filter: col("total") > 100
        # Result: Filter stays above Aggregate
        pass
```

### Backend Grouping Edge Cases
6. **Expression dependencies**: `(col("x") + 1).alias("a")` then `(col("a") + 1).alias("b")` - cannot reorder
7. **Column overwrites**: Multiple assignments to same column name must preserve order
8. **Circular dependencies**: Should be detected and raise error (shouldn't happen in valid plans)
9. **Mixed dependencies and overwrites**: `a = x + 1`, `a = a + 1` (overwrite that also references itself)

### Testing for Dependency Analysis
```python
class TestExpressionDependencies:
    def test_no_dependencies(self):
        """Independent expressions can be freely reordered."""
        exprs = [
            (col("x") + 1).alias("a"),
            (col("y") + 1).alias("b"),
        ]
        dag, names = build_expression_dag(exprs)
        assert dag == {0: set(), 1: set()}  # No deps

    def test_simple_dependency(self):
        """b depends on a - cannot reorder."""
        exprs = [
            (col("x") + 1).alias("a"),
            (col("a") + 1).alias("b"),  # refs "a"
        ]
        dag, names = build_expression_dag(exprs)
        assert dag[0] == set()
        assert dag[1] == {0}  # b depends on a

    def test_overwrite_same_column(self):
        """Multiple assignments to same column must preserve order."""
        exprs = [
            (col("x") + 1).alias("a"),
            (col("x") + 2).alias("a"),  # overwrites!
        ]
        dag, names = build_expression_dag(exprs)
        assert dag[1] == {0}  # second "a" depends on first

    def test_self_referencing_overwrite(self):
        """a = a + 1 depends on previous value of a."""
        exprs = [
            (col("x") + 1).alias("a"),
            (col("a") + 1).alias("a"),  # refs and overwrites
        ]
        dag, names = build_expression_dag(exprs)
        assert dag[1] == {0}

    def test_topological_levels(self):
        """Verify level partitioning."""
        # a and b independent, c depends on a
        exprs = [
            (col("x") + 1).alias("a"),
            (col("y") + 1).alias("b"),
            (col("a") + col("b")).alias("c"),
        ]
        dag, _ = build_expression_dag(exprs)
        levels = topological_levels(dag)
        assert levels[0] == {0, 1}  # a and b in level 0
        assert levels[1] == {2}     # c in level 1
```

## Success Criteria

1. All existing tests pass
2. `explain()` shows both raw and optimized plans
3. Optimization can be toggled via `collect(optimize=...)`
4. At least 3 optimization passes working:
   - Filter fusion
   - Predicate pushdown through Project
   - Projection pruning
5. Comprehensive tests for each optimization rule
6. Backend-aware grouping reduces conversions for mixed-type DataFrames

## Architecture Decision: Logical-Only vs Logical+Physical

### Option A: Logical Optimization Only (Recommended for Initial Implementation)

```
Logical Plan -> Logical Optimizer -> Optimized Logical Plan -> Evaluator
```

**Pros:**
- Simpler implementation
- Evaluator already handles backend differences
- Good enough for most use cases

**Cons:**
- Backend decisions made at evaluation time, not plan time
- Harder to guarantee zero-conversion execution paths

### Option B: Logical + Physical Planning (Future Enhancement)

```
Logical Plan -> Logical Optimizer -> Optimized Logical Plan
                                           ↓
                                    Physical Planner
                                           ↓
                                    Physical Plan -> Executor
```

Physical plan nodes would be backend-specific:
- `NumPyProject`, `ArrowProject`
- `NumPyFilter`, `ArrowFilter`
- `ConvertToArrow`, `ConvertToNumPy` (explicit conversion nodes)

**Pros:**
- Complete control over backend choices
- Can guarantee zero-conversion paths
- Better for advanced optimization (e.g., push projection into Arrow read)

**Cons:**
- More complex
- Requires physical node implementations
- May be premature optimization

### Recommendation

Start with **Option A** (logical-only), but design the optimizer framework to allow adding physical planning later:

1. Keep `OptimizationPass` generic enough to work on any plan type
2. The `Optimizer` class can later be extended with `PhysicalPlanner`
3. Backend analysis utilities (`get_backend_preference()`) will be reusable

The `BackendGrouping` pass in logical optimization gives us 80% of the benefit with 20% of the complexity. Physical planning can be added when we have concrete performance data showing it's needed.
