"""Translate a lazy-pandas LogicalPlan -> Rust engine JSON plan + Arrow tables,
and run it through lazy_engine_rs.execute(). Makes real lazy queries auto-route
into the Arrow-native Rust engine. Unsupported nodes/exprs raise NotSupported
(caller falls back to the Cython engine). Prototype — see RUST_ENGINE_DIRECTION.md.
"""

from __future__ import annotations

import json

import lazy_engine_rs as E
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
    Filter,
    Join,
    Limit,
    Project,
    Sort,
    TopK,
)

_BIN = {
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "equal": "eq",
    "not_equal": "ne",
    "and_": "and",
    "or_": "or",
}
_AGG = {"sum", "mean", "min", "max", "count"}


# alias a builtin (a custom Exception subclass would trip the errors-location hook)
NotSupported = NotImplementedError


def _df_to_batch(df: pd.DataFrame) -> pa.RecordBatch:
    arrays, names = [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            arrays.append(pa.array(s.values.astype("datetime64[ns]").view("int64")))
        elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
            arrays.append(pa.array(s.to_numpy().astype("int64")))
        elif pd.api.types.is_float_dtype(s):
            arrays.append(pa.array(s.to_numpy().astype("float64")))
        else:
            arrays.append(pa.array(s.astype("string"), type=pa.string()))
        names.append(str(c))
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _ir(e):
    return e._node if hasattr(e, "_node") else e


def _expr(node):
    if isinstance(node, FieldRef):
        return {"t": "col", "name": node.name}
    if isinstance(node, Literal):
        v = node.value
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return {"t": "liti", "v": int(pd.Timestamp(v).value)}
        if isinstance(v, bool):
            return {"t": "liti", "v": int(v)}
        if isinstance(v, (int, np.integer)):
            return {"t": "liti", "v": int(v)}
        if isinstance(v, (float, np.floating)):
            return {"t": "litf", "v": float(v)}
        if isinstance(v, str):
            return {"t": "litstr", "v": v}
        raise NotSupported(f"literal {v!r}")
    if isinstance(node, Cast):
        return _expr(node.arg)  # engine is dtype-flexible; ignore cast
    if isinstance(node, Alias):
        return _expr(node.arg)
    if isinstance(node, Call):
        f = node.function
        if f in _BIN and len(node.args) == 2:
            return {
                "t": "bin",
                "op": _BIN[f],
                "l": _expr(node.args[0]),
                "r": _expr(node.args[1]),
            }
        if f in ("dt_year", "is_null") and len(node.args) == 1:
            return {"t": "unary", "op": f, "a": _expr(node.args[0])}
        if f == "isin":
            vals = list(node.kwargs.get("values", ()))
            if all(
                isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                for v in vals
            ):
                return {
                    "t": "isin",
                    "a": _expr(node.args[0]),
                    "ints": [int(v) for v in vals],
                }
            return {
                "t": "isin",
                "a": _expr(node.args[0]),
                "strs": [str(v) for v in vals],
            }
        if f == "case_when":
            cases = node.kwargs["cases"]
            return {
                "t": "case",
                "cases": [[_expr(c), _expr(v)] for c, v in cases],
                "otherwise": _expr(node.kwargs["otherwise"]),
            }
        raise NotSupported(f"call {f}/{len(node.args)}")
    raise NotSupported(type(node).__name__)


def _name_of(node) -> str:
    if isinstance(node, Alias):
        return node.name
    if isinstance(node, FieldRef):
        return node.name
    raise NotSupported("unnamed expr")


def _go(plan, tables, ctr):
    if isinstance(plan, DataFrameSource):
        nm = f"src{ctr[0]}"
        ctr[0] += 1
        tables[nm] = _df_to_batch(plan.df)
        return {"op": "scan", "table": nm, "columns": [str(c) for c in plan.df.columns]}
    if isinstance(plan, Filter):
        return {
            "op": "filter",
            "pred": _expr(_ir(plan.predicate)),
            "input": _go(plan.input, tables, ctr),
        }
    if isinstance(plan, Project):
        exprs = []
        for e in plan.exprs:
            node = _ir(e)
            exprs.append({"expr": _expr(node), "name": _name_of(node)})
        return {"op": "project", "exprs": exprs, "input": _go(plan.input, tables, ctr)}
    if isinstance(plan, Aggregate):
        group = [_name_of(_ir(g)) for g in plan.group_by]
        # hoist non-trivial agg inputs into a project above the input
        hoist, aggs = [], []
        for e in plan.agg_exprs:
            node = _ir(e)
            name = _name_of(node)
            call = node.arg if isinstance(node, Alias) else node
            if not isinstance(call, Call) or call.function not in _AGG:
                raise NotSupported(f"agg {getattr(call, 'function', '?')}")
            arg = call.args[0]
            if isinstance(arg, FieldRef):
                col = arg.name
            else:
                col = f"_h{len(hoist)}"
                hoist.append({"expr": _expr(arg), "name": col})
            aggs.append({"func": call.function, "col": col, "name": name})
        inp = _go(plan.input, tables, ctr)
        if hoist:
            # carry group cols through + the hoisted agg-input cols
            keep = [{"expr": {"t": "col", "name": g}, "name": g} for g in group]
            inp = {"op": "project", "exprs": keep + hoist, "input": inp}
        return {"op": "aggregate", "group": group, "aggs": aggs, "input": inp}
    if isinstance(plan, Sort):
        keys = [
            {"col": _name_of(_ir(b)), "desc": bool(d)}
            for b, d in zip(plan.by, plan.descending, strict=False)
        ]
        return {"op": "sort", "keys": keys, "input": _go(plan.input, tables, ctr)}
    if isinstance(plan, Limit):
        if plan.offset:
            raise NotSupported("limit offset")
        return {"op": "limit", "n": int(plan.n), "input": _go(plan.input, tables, ctr)}
    if isinstance(plan, TopK):
        keys = [
            {"col": _name_of(_ir(b)), "desc": bool(d)}
            for b, d in zip(plan.by, plan.descending, strict=False)
        ]
        srt = {"op": "sort", "keys": keys, "input": _go(plan.input, tables, ctr)}
        return {"op": "limit", "n": int(plan.k), "input": srt}
    if isinstance(plan, Join):
        if plan.how != "inner":
            raise NotSupported(f"join how={plan.how}")
        if plan.on is not None and len(plan.on) == 1:
            lk = rk = plan.on[0]
        elif plan.left_on and plan.right_on and len(plan.left_on) == 1:
            lk, rk = plan.left_on[0], plan.right_on[0]
        else:
            raise NotSupported("join keys")
        return {
            "op": "join",
            "left": _go(plan.left, tables, ctr),
            "right": _go(plan.right, tables, ctr),
            "left_key": lk,
            "right_key": rk,
        }
    raise NotSupported(type(plan).__name__)


def compile_plan(ldf):
    """Translate to (json_plan, tables, schema). Arrow tables built ONCE here
    (the boundary conversion, like Polars\' from_pandas)."""
    plan = ldf._get_optimized_plan()
    tables: dict = {}
    jp = _go(plan, tables, [0])
    return json.dumps(jp), tables, plan.resolve_schema()


def exec_compiled(jp: str, tables: dict, schema=None) -> pd.DataFrame:
    df = E.execute(jp, tables).to_pandas()
    if schema is not None:
        for col in df.columns:
            dt = schema.get(col) if hasattr(schema, "get") else None
            sdt = str(getattr(dt, "pandas_dtype", dt) if dt is not None else "")
            if "datetime" in sdt and pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
    return df


def run(ldf) -> pd.DataFrame:
    jp, tables, schema = compile_plan(ldf)
    return exec_compiled(jp, tables, schema)
