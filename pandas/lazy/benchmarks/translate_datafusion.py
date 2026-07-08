"""Lower a lazy-pandas LogicalPlan -> a DataFusion DataFrame, and execute it.

This is the "6a" frontend fork from EXECUTION_RESEARCH_SURVEY.md: lazy-pandas
stays the frontend + optimizer; DataFusion (on our exact arrow-rs substrate)
becomes the execution backend. Mirrors translate.py (which lowers to our hand-
rolled Rust engine's JSON plan), but targets DataFusion's Python DataFrame API.

Source dfs are registered as Arrow record batches ONCE (the boundary, like
Polars' from_pandas). Unsupported nodes/exprs raise NotSupported.
"""

from __future__ import annotations

import datafusion as D
from datafusion import (
    SessionContext,
    col,
    functions as F,
    lit,
)
import numpy as np
import pyarrow as pa

import pandas as pd
from pandas.lazy.ir import (
    Alias,
    Call,
    Cast,
    FieldRef,
    Literal,
)
from pandas.lazy.plan import (
    Aggregate,
    DataFrameSource,
    Distinct,
    Filter,
    Join,
    Limit,
    Project,
    Sort,
    TopK,
)

# alias a builtin (a custom Exception subclass would trip the errors-location hook)
NotSupported = NotImplementedError

_AGG = {"sum", "mean", "min", "max", "count", "n_unique"}


def _df_to_arrow(df: pd.DataFrame) -> pa.RecordBatch:
    """pandas -> Arrow, preserving dtypes (datetimes stay timestamps, not int64
    as in the Rust-engine path — DataFusion handles Arrow temporal types).

    Schema metadata is stripped: pandas attaches a per-table ``pandas`` blob that
    otherwise poisons DataFusion's ``join_selection`` optimizer (it compares
    intermediate schemas *including* metadata, so two tables that differ only in
    that blob trip a spurious schema-mismatch)."""
    b = pa.RecordBatch.from_pandas(df, preserve_index=False)
    return b.replace_schema_metadata(None)


def _lit(v):
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        # a timestamp[ns] literal so it compares against Arrow timestamp columns
        return lit(pa.scalar(pd.Timestamp(v).to_pydatetime(), type=pa.timestamp("ns")))
    if isinstance(v, bool):
        return lit(v)
    if isinstance(v, (int, np.integer)):
        return lit(int(v))
    if isinstance(v, (float, np.floating)):
        return lit(float(v))
    if isinstance(v, str):
        return lit(v)
    raise NotSupported(f"literal {v!r}")


def _expr(node) -> D.Expr:
    if isinstance(node, FieldRef):
        return col(node.name)
    if isinstance(node, Literal):
        return _lit(node.value)
    if isinstance(node, Cast):
        return _expr(node.arg)  # DataFusion coerces; ignore cast
    if isinstance(node, Alias):
        return _expr(node.arg).alias(node.name)
    if isinstance(node, Call):
        return _call(node)
    raise NotSupported(type(node).__name__)


def _call(node: Call) -> D.Expr:
    fn = node.function
    a = node.args

    if len(a) == 2:
        left, right = _expr(a[0]), _expr(a[1])
        binop = {
            "add": lambda: left + right,
            "subtract": lambda: left - right,
            "multiply": lambda: left * right,
            "divide": lambda: left / right,
            "less": lambda: left < right,
            "less_equal": lambda: left <= right,
            "greater": lambda: left > right,
            "greater_equal": lambda: left >= right,
            "equal": lambda: left == right,
            "not_equal": lambda: left != right,
            "and_": lambda: left & right,
            "or_": lambda: left | right,
        }.get(fn)
        if binop is not None:
            return binop()
        if fn in ("str_startswith", "str_endswith", "str_contains"):
            pat = a[1].value if isinstance(a[1], Literal) else None
            if not isinstance(pat, str):
                raise NotSupported(f"{fn} non-literal pattern")
            sarg = _expr(a[0])
            if fn == "str_startswith":
                return F.starts_with(sarg, lit(pat))
            if fn == "str_endswith":
                return F.ends_with(sarg, lit(pat))
            return F.regexp_like(sarg, lit(pat))  # contains == regex, matches engine

    if len(a) == 1:
        arg = _expr(a[0])
        if fn == "dt_year":
            return F.date_part(lit("year"), arg).cast(pa.int64())
        if fn == "is_null":
            return arg.is_null()
        if fn == "invert":
            return ~arg
        if fn == "str_slice":
            start = int(node.kwargs.get("start") or 0)
            stop = node.kwargs.get("stop")
            # DataFusion substr is 1-indexed; length = stop - start
            if stop is not None:
                return F.substring(arg, lit(start + 1), lit(int(stop) - start))
            return F.substr(arg, lit(start + 1))

    if fn == "isin":
        vals = list(node.kwargs.get("values", ()))
        return F.in_list(_expr(a[0]), [_lit(v) for v in vals], negated=False)

    if fn == "case_when":
        cases = node.kwargs["cases"]
        otherwise = node.kwargs["otherwise"]
        builder = None
        for cond, val in cases:
            if builder is None:
                builder = F.when(_expr(cond), _expr(val))
            else:
                builder = builder.when(_expr(cond), _expr(val))
        return builder.otherwise(_expr(otherwise))

    raise NotSupported(f"call {fn}/{len(a)}")


def _agg_expr(node) -> D.Expr:
    name = node.name if isinstance(node, Alias) else None
    call = node.arg if isinstance(node, Alias) else node
    if not isinstance(call, Call) or call.function not in _AGG:
        raise NotSupported(f"agg {getattr(call, 'function', '?')}")
    inner = _expr(call.args[0])
    func = call.function
    e = {
        "sum": lambda: F.sum(inner),
        "mean": lambda: F.avg(inner),
        "min": lambda: F.min(inner),
        "max": lambda: F.max(inner),
        "count": lambda: F.count(inner),
        "n_unique": lambda: F.count(inner).distinct().build(),
    }[func]()
    return e.alias(name) if name else e


def _name_of(node) -> str:
    if isinstance(node, Alias):
        return node.name
    if isinstance(node, FieldRef):
        return node.name
    raise NotSupported("unnamed expr")


def _lower(plan, ctx: SessionContext, ctr: list) -> D.DataFrame:
    if isinstance(plan, DataFrameSource):
        nm = f"t{ctr[0]}"
        ctr[0] += 1
        ctx.register_record_batches(nm, [[_df_to_arrow(plan.df)]])
        return ctx.table(nm)

    if isinstance(plan, Filter):
        return _lower(plan.input, ctx, ctr).filter(_expr(_ir(plan.predicate)))

    if isinstance(plan, Project):
        exprs = []
        for e in plan.exprs:
            node = _ir(e)
            exprs.append(_expr(node).alias(_name_of(node)))
        return _lower(plan.input, ctx, ctr).select(*exprs)

    if isinstance(plan, Aggregate):
        df = _lower(plan.input, ctx, ctr)
        groups = [col(_name_of(_ir(g))) for g in plan.group_by]
        aggs = [_agg_expr(_ir(e)) for e in plan.agg_exprs]
        return df.aggregate(groups, aggs)

    if isinstance(plan, Sort):
        df = _lower(plan.input, ctx, ctr)
        keys = [
            col(_name_of(_ir(b))).sort(ascending=not bool(d))
            for b, d in zip(plan.by, plan.descending, strict=False)
        ]
        return df.sort(*keys)

    if isinstance(plan, Limit):
        return _lower(plan.input, ctx, ctr).limit(int(plan.n), int(plan.offset or 0))

    if isinstance(plan, TopK):
        df = _lower(plan.input, ctx, ctr)
        keys = [
            col(_name_of(_ir(b))).sort(ascending=not bool(d))
            for b, d in zip(plan.by, plan.descending, strict=False)
        ]
        return df.sort(*keys).limit(int(plan.k))

    if isinstance(plan, Distinct):
        return _lower(plan.input, ctx, ctr).distinct()

    if isinstance(plan, Join):
        left = _lower(plan.left, ctx, ctr)
        right = _lower(plan.right, ctx, ctr)
        how = plan.how
        if how == "cross":
            # DataFusion has no cross method; join_on with no predicates == cross.
            return left.join_on(right, how="inner")
        return _join(left, right, plan)

    raise NotSupported(type(plan).__name__)


def _join(left, right, plan) -> D.DataFrame:
    """Pandas-faithful equi-join. DataFusion doesn't auto-suffix overlapping
    columns and its ``on=`` coalesce is unreliable when an input is itself a
    join result, so we do it explicitly: rename the right keys to temps (so we
    can drop/rename them deterministically), suffix overlapping non-key columns
    the way ``pd.merge`` does, then join on the temp keys."""
    how = plan.how
    lnames = left.schema().names
    rnames = right.schema().names
    if plan.on is not None:
        lkeys = rkeys = list(plan.on)
        coalesce = True  # `on=`: one shared key column in the output
    elif plan.left_on and plan.right_on:
        lkeys, rkeys = list(plan.left_on), list(plan.right_on)
        coalesce = False  # left_on/right_on: both key columns kept
    else:
        raise NotSupported("join keys")

    sfx0, sfx1 = plan.suffix
    overlap = {c for c in lnames if c not in lkeys} & {
        c for c in rnames if c not in rkeys
    }

    # right keys -> temp names (deterministic drop/rename after the join)
    tmp_rkeys = [f"__rk{i}" for i in range(len(rkeys))]
    for rk, tk in zip(rkeys, tmp_rkeys, strict=True):
        right = right.with_column_renamed(rk, tk)
    # suffix overlapping non-key columns (pd.merge semantics)
    for c in overlap:
        left = left.with_column_renamed(c, c + sfx0)
        right = right.with_column_renamed(c, c + sfx1)

    df = left.join(right, left_on=lkeys, right_on=tmp_rkeys, how=how)
    if coalesce:
        df = df.drop(*tmp_rkeys)  # keep only the left copy of the shared key
    else:
        for rk, tk in zip(rkeys, tmp_rkeys, strict=True):  # restore right-key names
            df = df.with_column_renamed(tk, rk)
    return df


def _ir(e):
    return e._node if hasattr(e, "_node") else e


def run(ldf) -> pd.DataFrame:
    plan = ldf._get_optimized_plan()
    ctx = SessionContext()
    df = _lower(plan, ctx, [0])
    return df.to_pandas()
