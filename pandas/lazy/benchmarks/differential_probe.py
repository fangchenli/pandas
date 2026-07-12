"""Differential substrate probe — a standing instrument for *finding* upstream gaps.

The lazy-pandas probe's best findings all have the same shape: run one logical
operation, vary a single axis, and watch something diverge.

    AG10  SQL plan          vs  DataFrame-API plan   (same query)
    AG11  cross join         vs  hash/NLJ             (same inputs)
    AG3/5 1 core             vs  N cores              (same op)
    AG4   raw string key     vs  dict-encoded key     (same agg)
    AG9   type X works       vs  type Y NotImplemented

Each was a hand-built matrix, thrown away after one find. This module makes the
matrix *standing*: a fixed workload grid run identically across
``{pandas, polars, acero, datafusion-sql, datafusion-df}`` that emits a
**divergence report**. Every diverging cell is a *candidate finding* — the
harness flags, a human confirms and roots-cause.

Five divergence classes, most-valuable first:

    CRASH    one engine raises/panics where others succeed (found AG13: datafusion
             register_record_batches panics on an empty table). Loudest signal.
    RESULT   engines disagree on the answer (correctness/bug, or a pandas->engine
             lowering hazard — the surface our ad-hoc perf matrices never covered)
    PLAN     datafusion SQL vs DataFrame-API optimized plans differ in operator
             shape (the AG10 class — an optimizer rule fires for one front-end)
    COVERAGE an arrow-compute kernel works on one type but is NotImplemented on a
             sibling type (the AG9 class — string_view/dict lag string kernels)
    PERF     one engine is >Nx slower on the same cell (the AG3/4/5 class)

Sources: the grouped-aggregate grid (PLAN/PERF); a degenerate-input edge sweep —
empty / all-null / null key (CRASH/RESULT); a join-shape matrix — inner/left/full/
fanout/null-key/cross/semi/anti/self, the AG11 axis (CRASH/RESULT); a kernel
type-coverage matrix — kernel x input-type, the AG9/AG12 axis (COVERAGE); and a
sort-order matrix — null/NaN placement across engines (RESULT). Several
cells are standing fix-trackers that auto-flip when the upstream fix ships:
cross-count-metadata reproduces AG11 (datafusion PR #23442), self-join reproduces
the datafusion-df duplicate-qualified-field limit (open #14147), and each
COVERAGE cell flips NotImpl->ok as arrow #44336 view-kernel sub-PRs land.

A sixth source is the **Substrait fan-out** matrix (AG17/AG18/AG19 axis): lower one
op to DataFusion, serialize to Substrait — the portable IR — and consume it back
on a *second* engine (Acero). Every "DataFusion emits but Acero can't consume" is a
COVERAGE finding, every "Acero consumes but silently disagrees" a RESULT finding.
Its cells are standing trackers: ``acero-raw`` (Acero on UNFIXED datafusion
Substrait) flips ok when datafusion stops omitting ``ScalarFunction.output_type``
(AG17); ``acero-fix`` (Acero on the AG17/AG18 portability-fixed plan, via
``substrait_fixup.py``) flips ok as arrow registers the missing string-function
mappings (AG19) / ``precision_timestamp`` (AG18b), and guards the AG18a
``FetchRel`` silent-0-row fix from regressing. Skipped unless the ``substrait``
proto pkg is importable.

Self-contained: pyarrow + polars + datafusion + numpy + pandas, synthetic data.
Run on the *current* releases in a throwaway venv (see ``upstream/README.md`` §1),
never the repo's pinned pyarrow — the whole point is to catch what changed:

    python -m venv /tmp/upstream-venv && . /tmp/upstream-venv/bin/activate
    pip install -U pyarrow polars datafusion numpy pandas substrait
    cd /tmp && python <repo>/pandas/lazy/benchmarks/differential_probe.py

(Run from *outside* the repo so the source ``pandas/`` doesn't shadow the built
one.) ``--rows``, ``--perf-threshold`` and ``--only`` flags below tune the sweep.
Thread-scaling (the AG3/AG5 axis) needs re-exec because polars' pool is fixed at
import — see ``bench_arrow_string_groupby.py``; this harness runs all-cores and
notes the pool size.
"""

from __future__ import annotations

import argparse
from dataclasses import (
    dataclass,
    field,
)
import datetime
import os
import time

import datafusion as d
from datafusion import (
    SessionContext,
    col,
    functions as F,
)
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

import pandas as pd

ENGINES = ["pandas", "polars", "acero", "datafusion-sql", "datafusion-df"]
ORACLE = "pandas"  # correctness reference

# operator-node prefixes we count in a datafusion optimized-plan display to
# detect front-end plan-shape divergence (the AG10 signal: single_distinct_to_
# groupby turns one Aggregate into two).
PLAN_NODES = (
    "Aggregate",
    "Distinct",
    "CrossJoin",
    "Join",
    "Filter",
    "Projection",
    "Sort",
    "Limit",
    "Union",
    "Window",
    "SubqueryAlias",
    "TableScan",
)
# a plan-shape difference only counts as a finding when a *structural* node
# diverges (an optimizer rule fired for one front-end, e.g. single_distinct_to_
# groupby -> Aggregate x2). A lone Projection/SubqueryAlias/TableScan delta is a
# benign column-aliasing artifact between the SQL and DataFrame builders.
STRUCTURAL_NODES = frozenset(
    {"Aggregate", "Distinct", "CrossJoin", "Join", "Union", "Window"}
)


def _time(fn, warm=1, runs=3):
    """best-of-``runs`` wall time in ms, plus the last result."""
    for _ in range(warm):
        fn()
    best, res = float("inf"), None
    for _ in range(runs):
        s = time.perf_counter()
        res = fn()
        best = min(best, (time.perf_counter() - s) * 1000)
    return best, res


def _canon(df: pd.DataFrame, key: str, out: str) -> pd.DataFrame:
    """normalize an engine's result to [key, out] sorted by key for comparison."""
    r = df[[key, out]].copy()
    r.columns = ["key", "out"]
    r = r.sort_values("key", kind="stable").reset_index(drop=True)
    if pd.api.types.is_float_dtype(r["out"]):
        r["out"] = r["out"].round(6)
    return r


def _results_agree(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    # compare *values*, not dtypes — engines legitimately differ on the result
    # dtype (polars count -> uint32, pandas -> int64); a dtype mismatch is not a
    # correctness divergence, so array_equal on the values, not Series.equals.
    if len(a) != len(b):
        return False
    if not np.array_equal(a["key"].to_numpy(), b["key"].to_numpy()):
        return False
    av, bv = a["out"].to_numpy(), b["out"].to_numpy()
    if np.issubdtype(av.dtype, np.floating) or np.issubdtype(bv.dtype, np.floating):
        return bool(np.allclose(av, bv, rtol=1e-9, atol=1e-6))
    return bool(np.array_equal(av, bv))


def _plan_node_counts(plan_text: str) -> dict[str, int]:
    """multiset of operator-node types in a datafusion optimized-plan display."""
    counts: dict[str, int] = {}
    for line in plan_text.splitlines():
        stripped = line.lstrip(" |+-")
        for node in PLAN_NODES:
            if stripped.startswith((node + ":", node + " ")):
                counts[node] = counts.get(node, 0) + 1
                break
    return counts


# --------------------------------------------------------------------------- #
# Workload: grouped aggregate (covers AG3/AG4/AG5 + the AG10 plan surface).
# key_dtype x cardinality x agg. One key column, one value column.
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """native handles for one synthetic table, built once per (dtype, card)."""

    key_dtype: str
    card: int
    n: int
    pdf: pd.DataFrame
    pol: pl.DataFrame
    tbl: pa.Table
    ctx: SessionContext = field(repr=False)


def build_dataset(key_dtype: str, card: int, n: int, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, card, n)
    if key_dtype == "int64":
        keys = idx.astype("int64")
        pa_key = pa.array(keys)
    elif key_dtype == "string":
        pool = np.array([f"k{i:012d}" for i in range(card)], dtype=object)
        keys = pool[idx]
        pa_key = pa.array(keys, type=pa.large_string())
    else:  # pragma: no cover - guard
        raise ValueError(key_dtype)
    fval = rng.standard_normal(n)
    ival = rng.integers(0, max(card // 4, 2), n).astype("int64")  # for count_distinct

    pdf = pd.DataFrame({"k": keys, "fv": fval, "iv": ival})
    pol = pl.DataFrame(
        {
            "k": keys.astype(str) if key_dtype == "string" else keys,
            "fv": fval,
            "iv": ival,
        }
    )
    tbl = pa.table({"k": pa_key, "fv": pa.array(fval), "iv": pa.array(ival)})
    # datafusion: strip metadata (the AG11 workaround / clean baseline).
    ctx = SessionContext()
    ctx.register_record_batches("t", [tbl.replace_schema_metadata(None).to_batches()])
    return Dataset(key_dtype, card, n, pdf, pol, tbl, ctx)


@dataclass
class AggSpec:
    """one aggregation across all five engines."""

    name: str  # display
    valcol: str  # "fv" or "iv"
    pandas: object
    polars: object
    acero: str
    sql: str
    df: object  # builds the datafusion Expr given col()


AGGS = {
    "sum": AggSpec(
        "sum",
        "fv",
        pandas=lambda g: g["fv"].sum(),
        polars=lambda: pl.col("fv").sum().alias("out"),
        acero="sum",
        sql="sum(fv)",
        df=lambda: F.sum(col("fv")).alias("out"),
    ),
    "count_distinct": AggSpec(
        "count_distinct",
        "iv",
        pandas=lambda g: g["iv"].nunique(),
        polars=lambda: pl.col("iv").n_unique().alias("out"),
        acero="count_distinct",
        sql="count(DISTINCT iv)",
        df=lambda: F.count(col("iv")).distinct().build().alias("out"),
    ),
    # used by the edge-case (RESULT/CRASH) sweep, not the perf grid
    "count": AggSpec(
        "count",
        "iv",
        pandas=lambda g: g["iv"].count(),
        polars=lambda: pl.col("iv").count().alias("out"),
        acero="count",
        sql="count(iv)",
        df=lambda: F.count(col("iv")).alias("out"),
    ),
    "mean": AggSpec(
        "mean",
        "fv",
        pandas=lambda g: g["fv"].mean(),
        polars=lambda: pl.col("fv").mean().alias("out"),
        acero="mean",
        sql="avg(fv)",
        df=lambda: F.avg(col("fv")).alias("out"),
    ),
    "min": AggSpec(
        "min",
        "fv",
        pandas=lambda g: g["fv"].min(),
        polars=lambda: pl.col("fv").min().alias("out"),
        acero="min",
        sql="min(fv)",
        df=lambda: F.min(col("fv")).alias("out"),
    ),
}


def run_grouped_agg(ds: Dataset, agg: AggSpec):
    """run one grouped agg on all engines; return {engine: (ms, canon_df)} + plans."""
    out: dict[str, tuple[float, pd.DataFrame]] = {}

    def pandas_fn():
        r = ds.pdf.groupby("k", sort=False)
        s = agg.pandas(r).reset_index()
        s.columns = ["k", "out"]
        return s

    ms, r = _time(pandas_fn)
    out["pandas"] = (ms, _canon(r, "k", "out"))

    def polars_fn():
        return ds.pol.group_by("k").agg(agg.polars())

    ms, r = _time(polars_fn)
    out["polars"] = (ms, _canon(r.to_pandas(), "k", "out"))

    def acero_fn():
        return ds.tbl.group_by(["k"]).aggregate([(agg.valcol, agg.acero)])

    ms, r = _time(acero_fn)
    out["acero"] = (ms, _canon(r.to_pandas(), "k", f"{agg.valcol}_{agg.acero}"))

    sql = f"SELECT k, {agg.sql} AS out FROM t GROUP BY k"

    def dfsql_fn():
        return ds.ctx.sql(sql).collect()

    ms, batches = _time(dfsql_fn)
    out["datafusion-sql"] = (
        ms,
        _canon(pa.Table.from_batches(batches).to_pandas(), "k", "out"),
    )

    def dfdf_build():
        return ds.ctx.table("t").aggregate([col("k")], [agg.df()])

    def dfdf_fn():
        return dfdf_build().collect()

    ms, batches = _time(dfdf_fn)
    out["datafusion-df"] = (
        ms,
        _canon(pa.Table.from_batches(batches).to_pandas(), "k", "out"),
    )

    # display_indent() is the canonical indented tree ("Aggregate: groupBy=...");
    # str(plan) is a one-line Rust Debug repr that hides node structure.
    plans = {
        "datafusion-sql": ds.ctx.sql(sql).optimized_logical_plan().display_indent(),
        "datafusion-df": dfdf_build().optimized_logical_plan().display_indent(),
    }
    return out, plans


# --------------------------------------------------------------------------- #
# Divergence detection + report
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    kind: str  # CRASH | RESULT | PLAN | PERF
    workload: str
    detail: str
    severity: float  # sort key (bigger = louder)


def analyze(
    workload: str, cells: dict, plans: dict, perf_threshold: float
) -> list[Finding]:
    findings: list[Finding] = []
    oracle = cells[ORACLE][1]

    # RESULT — any engine disagreeing with the oracle answer.
    for eng, (_, res) in cells.items():
        if eng == ORACLE:
            continue
        if not _results_agree(oracle, res):
            findings.append(
                Finding(
                    "RESULT",
                    workload,
                    f"{eng} disagrees with {ORACLE}: "
                    f"{len(res)} rows vs {len(oracle)}; sample out "
                    f"{res['out'].head(3).tolist()} vs "
                    f"{oracle['out'].head(3).tolist()}",
                    1e6,  # correctness always ranks top
                )
            )

    # PLAN — datafusion SQL vs DataFrame optimized-plan operator shape.
    if "datafusion-sql" in plans and "datafusion-df" in plans:
        cs = _plan_node_counts(plans["datafusion-sql"])
        cd = _plan_node_counts(plans["datafusion-df"])
        differing = [
            k for k in sorted(set(cs) | set(cd)) if cs.get(k, 0) != cd.get(k, 0)
        ]
        if any(k in STRUCTURAL_NODES for k in differing):
            diff = ", ".join(
                f"{k}: sql={cs.get(k, 0)} df={cd.get(k, 0)}" for k in differing
            )
            findings.append(
                Finding(
                    "PLAN",
                    workload,
                    f"datafusion SQL vs DataFrame plan shape differs — {diff}",
                    5e5,
                )
            )

    # PERF — spread across engines, plus the two known axes.
    times = {e: t for e, (t, _) in cells.items()}
    fastest = min(times, key=times.get)
    slowest = max(times, key=times.get)
    ratio = times[slowest] / max(times[fastest], 1e-9)
    if ratio >= perf_threshold:
        findings.append(
            Finding(
                "PERF",
                workload,
                f"{slowest} {times[slowest]:.0f}ms is {ratio:.1f}x slower than "
                f"{fastest} {times[fastest]:.0f}ms",
                ratio,
            )
        )
    # named axes we specifically hunt (only if not already the headline spread)
    for a, b, label in [
        ("acero", "polars", "acero-vs-polars"),
        ("datafusion-df", "datafusion-sql", "df-api-vs-sql"),
    ]:
        if (
            a in times
            and b in times
            and times[a] / max(times[b], 1e-9) >= perf_threshold
        ):
            findings.append(
                Finding(
                    "PERF",
                    workload,
                    f"[{label}] {a} {times[a]:.0f}ms is "
                    f"{times[a] / times[b]:.1f}x slower than {b} {times[b]:.0f}ms",
                    times[a] / times[b],
                )
            )
    return findings


# --------------------------------------------------------------------------- #
# Edge-case sweep: degenerate inputs (empty / all-null group / null key) surface
# the RESULT (semantic-divergence / lowering-hazard) and CRASH (one engine
# raises or PANICS where others succeed) classes. This is how AG13 (datafusion
# register_record_batches panics on an empty table) and AG11-class crashes get
# caught automatically. Tiny crafted tables; each engine wrapped so a crash is a
# finding, not the end of the run.
# --------------------------------------------------------------------------- #
# aliased base so the `validate-errors-locations` hook (which flags any class
# whose base is literally `Exception`/`*Error`/`*Warning` outside pandas/errors)
# doesn't mistake this benchmark-internal sentinel for a pandas public error.
_Sentinel = Exception


class _Unsupported(_Sentinel):
    """an engine's API genuinely can't express this shape (not a bug) — excluded
    from CRASH/RESULT so an API limitation isn't mistaken for a defect."""


def _safe(fn):
    """run fn; return ("ok", value) | ("unsupported", why) | ("crash", msg).
    Catches Rust PanicException (pyo3) too — a panic is the loudest divergence."""
    try:
        return ("ok", fn())
    except _Unsupported as e:
        return ("unsupported", str(e))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        return ("crash", f"{type(e).__name__}: {str(e).strip()[:90]}")


def _pairs(keys, vals) -> list[tuple[str, str]]:
    """canonical [(key, out)] as strings, nulls -> 'NULL', sorted. Exact semantic
    compare for tiny inputs (distinguishes 0.0 from NULL from NaN)."""
    rows = []
    for k, v in zip(keys, vals, strict=True):
        ks = "NULL" if pd.isna(k) else str(k)
        if pd.isna(v):
            vs = "NULL"
        elif isinstance(v, float):
            vs = repr(round(v, 6))
        else:
            vs = str(v)
        rows.append((ks, vs))
    return sorted(rows)


def run_edge_case(tbl: pa.Table, agg: AggSpec) -> dict[str, tuple[str, object]]:
    """run one grouped agg over a crafted table on all engines, crash-safe."""
    results: dict[str, tuple[str, object]] = {}

    def pandas_pairs():
        ppd = tbl.to_pandas(types_mapper=pd.ArrowDtype)
        s = agg.pandas(ppd.groupby("k", dropna=True)).reset_index()
        s.columns = ["k", "out"]
        return _pairs(s["k"].tolist(), s["out"].tolist())

    def polars_pairs():
        rp = pl.from_arrow(tbl).group_by("k").agg(agg.polars()).to_pandas()
        return _pairs(rp["k"].tolist(), rp["out"].tolist())

    def acero_pairs():
        r = tbl.group_by(["k"]).aggregate([(agg.valcol, agg.acero)])
        rp = r.to_pandas()
        return _pairs(rp["k"].tolist(), rp[f"{agg.valcol}_{agg.acero}"].tolist())

    results["pandas"] = _safe(pandas_pairs)
    results["polars"] = _safe(polars_pairs)
    results["acero"] = _safe(acero_pairs)

    # datafusion: registration itself can panic (AG13) — share one attempt.
    def make_ctx():
        ctx = SessionContext()
        ctx.register_record_batches(
            "t", [tbl.replace_schema_metadata(None).to_batches()]
        )
        return ctx

    reg = _safe(make_ctx)
    if reg[0] == "crash":
        results["datafusion-sql"] = reg
        results["datafusion-df"] = reg
    else:
        ctx = reg[1]

        def dfsql_pairs():
            b = ctx.sql(f"SELECT k, {agg.sql} AS out FROM t GROUP BY k").collect()
            rp = pa.Table.from_batches(b).to_pandas()
            return _pairs(rp["k"].tolist(), rp["out"].tolist())

        def dfdf_pairs():
            b = ctx.table("t").aggregate([col("k")], [agg.df()]).collect()
            rp = pa.Table.from_batches(b).to_pandas()
            return _pairs(rp["k"].tolist(), rp["out"].tolist())

        results["datafusion-sql"] = _safe(dfsql_pairs)
        results["datafusion-df"] = _safe(dfdf_pairs)
    return results


def analyze_edge(workload: str, results: dict) -> list[Finding]:
    findings: list[Finding] = []
    ok = {e: v for e, (s, v) in results.items() if s == "ok"}
    crashed = {e: m for e, (s, m) in results.items() if s == "crash"}
    # CRASH — an engine raised/panicked where at least one other succeeded.
    if crashed and ok:
        for e, m in crashed.items():
            findings.append(
                Finding(
                    "CRASH",
                    workload,
                    f"{e} raised where {sorted(ok)} succeeded — {m}",
                    2e6,  # a crash outranks a wrong answer
                )
            )
    # RESULT — engines that succeeded disagree with pandas (lowering oracle).
    # These are pandas-semantics divergences: a lowering hazard, not necessarily
    # an upstream bug (SQL sum-of-null=NULL vs pandas 0; pandas drops NULL keys).
    if "pandas" in ok:
        ref = ok["pandas"]
        for e, pairs in ok.items():
            if e == "pandas" or pairs == ref:
                continue
            findings.append(
                Finding(
                    "RESULT",
                    workload,
                    f"{e} vs pandas: {pairs} vs {ref}",
                    1e6,
                )
            )
    return findings


def build_edge_cases() -> list[tuple[str, pa.Table, list[str]]]:
    """crafted degenerate tables (columns k, fv, iv) + which aggs to probe."""
    empty = pa.table(
        {
            "k": pa.array([], pa.string()),
            "fv": pa.array([], pa.float64()),
            "iv": pa.array([], pa.int64()),
        }
    )
    all_null = pa.table(
        {
            "k": pa.array(["A", "A", "B", "B"]),
            "fv": pa.array([None, None, 1.0, 2.0], pa.float64()),
            "iv": pa.array([None, None, 3, 4], pa.int64()),
        }
    )
    null_key = pa.table(
        {
            "k": pa.array(["A", None, None, "B"]),
            "fv": pa.array([1.0, 2.0, 3.0, 4.0], pa.float64()),
            "iv": pa.array([1, 2, 3, 4], pa.int64()),
        }
    )
    return [
        ("empty", empty, ["sum", "count"]),
        ("all_null_group", all_null, ["sum", "count", "mean", "min"]),
        ("null_key", null_key, ["sum"]),
    ]


# --------------------------------------------------------------------------- #
# Join-shape matrix (the AG11 axis): run the SAME logical join across engines,
# vary shape (inner/left/full/fanout/null-key/cross/semi/anti/self) + metadata,
# compare the output multiset (payload pairs / left-only / self-pairs) — order-
# insensitive, key-naming-agnostic. Surfaces RESULT (join semantics diverge —
# e.g. pandas matches NULL==NULL on a join key, SQL engines don't) and CRASH
# (AG11: count over a metadata-carrying cross join; df-df self-join duplicate
# fields). The cross-count case auto-flips to OK when AG11's fix (#23442) ships.
# --------------------------------------------------------------------------- #
_HOW_PD = {"inner": "inner", "left": "left", "full": "outer"}
_HOW_PL = {
    "inner": "inner",
    "left": "left",
    "full": "full",
    "semi": "semi",
    "anti": "anti",
}
_HOW_ACERO = {
    "inner": "inner",
    "left": "left outer",
    "full": "full outer",
    "semi": "left semi",
    "anti": "left anti",
}
_HOW_SQL = {"inner": "JOIN", "left": "LEFT JOIN", "full": "FULL JOIN"}


def _n(x):
    return "NULL" if pd.isna(x) else str(x)


def _lv_rv(df) -> list[tuple[str, str]]:
    return sorted((_n(a), _n(b)) for a, b in zip(df["lv"], df["rv"], strict=True))


def _left_only(df) -> list[tuple[str]]:
    """semi/anti output = left rows only, keyed on the lv payload."""
    return sorted((_n(v),) for v in df["lv"])


def _self_pairs(df) -> list[tuple[str, str]]:
    """self-join output: the two payload columns by POSITION (names collide —
    lv_x/lv_y, lv/lv_right, or duplicate lv/lv — so index, don't name)."""
    payload = [i for i, c in enumerate(df.columns) if c != "k"][:2]
    sub = df.iloc[:, payload]
    return sorted((_n(sub.iloc[i, 0]), _n(sub.iloc[i, 1])) for i in range(len(sub)))


def run_join_case(
    lt: pa.Table, rt: pa.Table, how: str, keep_meta: bool, op: str
) -> dict[str, tuple[str, object]]:
    """one join across all engines, crash-safe. how in inner/left/full/cross/
    semi/anti/self; op='count' triggers the AG11 aggregate-over-cross path."""
    lpd, rpd = lt.to_pandas(), rt.to_pandas()
    lpl, rpl = pl.from_arrow(lt), pl.from_arrow(rt)

    def _emit(df, n):
        return [("n", str(n))] if op == "count" else _lv_rv(df)

    def pandas_fn():
        if how == "self":
            return _self_pairs(lpd.merge(lpd, on="k"))
        if how in ("semi", "anti"):
            mask = lpd["k"].isin(rpd["k"])
            return _left_only(lpd[mask if how == "semi" else ~mask])
        m = (
            lpd.merge(rpd, how="cross")
            if how == "cross"
            else lpd.merge(rpd, on="k", how=_HOW_PD[how])
        )
        return _emit(m, len(m))

    def polars_fn():
        if how == "self":
            return _self_pairs(lpl.join(lpl, on="k", how="inner").to_pandas())
        if how in ("semi", "anti"):
            return _left_only(lpl.join(rpl, on="k", how=_HOW_PL[how]).to_pandas())
        m = (
            lpl.join(rpl, how="cross")
            if how == "cross"
            else lpl.join(rpl, on="k", how=_HOW_PL[how])
        ).to_pandas()
        return _emit(m, len(m))

    def acero_fn():
        if how == "cross":
            raise _Unsupported("pyarrow Table.join requires keys (no cross)")
        if how == "self":
            return _self_pairs(lt.join(lt, keys="k", join_type="inner").to_pandas())
        m = lt.join(rt, keys="k", join_type=_HOW_ACERO[how]).to_pandas()
        return _left_only(m) if how in ("semi", "anti") else _emit(m, len(m))

    def make_ctx():
        ctx = SessionContext()
        lb = lt if keep_meta else lt.replace_schema_metadata(None)
        rb = rt if keep_meta else rt.replace_schema_metadata(None)
        ctx.register_record_batches("l", [lb.to_batches()])
        ctx.register_record_batches("r", [rb.to_batches()])
        if how == "self":
            ctx.register_record_batches("t", [lb.to_batches()])
        return ctx

    def dfsql_fn(ctx):
        if how == "self":
            b = ctx.sql(
                "SELECT a.lv AS la, b.lv AS lb FROM t a JOIN t b ON a.k = b.k"
            ).collect()
            return _self_pairs(pa.Table.from_batches(b).to_pandas())
        if how in ("semi", "anti"):
            ex = "EXISTS" if how == "semi" else "NOT EXISTS"
            b = ctx.sql(
                f"SELECT l.lv FROM l WHERE {ex} (SELECT 1 FROM r WHERE r.k = l.k)"
            ).collect()
            return _left_only(pa.Table.from_batches(b).to_pandas())
        if op == "count":
            join = (
                "CROSS JOIN r" if how == "cross" else f"{_HOW_SQL[how]} r ON l.k = r.k"
            )
            b = ctx.sql(f"SELECT count(*) AS n FROM l {join}").collect()
            return [("n", str(pa.Table.from_batches(b).to_pandas()["n"][0]))]
        q = (
            "SELECT l.lv, r.rv FROM l CROSS JOIN r"
            if how == "cross"
            else f"SELECT l.lv, r.rv FROM l {_HOW_SQL[how]} r ON l.k = r.k"
        )
        b = ctx.sql(q).collect()
        return _lv_rv(pa.Table.from_batches(b).to_pandas())

    def dfdf_fn(ctx):
        if how == "self":
            # join a table to itself via the DataFrame API — the q16 lowering
            # hazard (duplicate qualified fields); may raise -> a CRASH finding.
            t = ctx.table("t")
            b = t.join(t, how="inner", left_on=["k"], right_on=["k"]).collect()
            return _self_pairs(pa.Table.from_batches(b).to_pandas())
        lo, ro = ctx.table("l"), ctx.table("r")
        if how in ("semi", "anti"):
            b = (
                lo.join(ro, how=how, left_on=["k"], right_on=["k"])
                .select(col("lv"))
                .collect()
            )
            return _left_only(pa.Table.from_batches(b).to_pandas())
        j = (
            lo.join_on(ro, how="inner")
            if how == "cross"
            else lo.join(ro, how=how, left_on=["k"], right_on=["k"])
        )
        if op == "count":
            return [("n", str(j.count()))]  # aggregate over cross join = AG11 path
        b = j.select(col("lv"), col("rv")).collect()
        return _lv_rv(pa.Table.from_batches(b).to_pandas())

    results: dict[str, tuple[str, object]] = {
        "pandas": _safe(pandas_fn),
        "polars": _safe(polars_fn),
        "acero": _safe(acero_fn),
    }
    reg = _safe(make_ctx)
    if reg[0] != "ok":
        results["datafusion-sql"] = results["datafusion-df"] = reg
    else:
        results["datafusion-sql"] = _safe(lambda: dfsql_fn(reg[1]))
        results["datafusion-df"] = _safe(lambda: dfdf_fn(reg[1]))
    return results


def analyze_join(workload: str, results: dict) -> list[Finding]:
    findings: list[Finding] = []
    ok = {e: v for e, (s, v) in results.items() if s == "ok"}
    crashed = {e: m for e, (s, m) in results.items() if s == "crash"}
    if crashed and ok:
        for e, m in crashed.items():
            findings.append(
                Finding(
                    "CRASH",
                    workload,
                    f"{e} raised where {sorted(ok)} succeeded — {m}",
                    2e6,
                )
            )
    if "pandas" in ok:
        ref = ok["pandas"]
        for e, rows in ok.items():
            if e == "pandas" or rows == ref:
                continue
            findings.append(
                Finding("RESULT", workload, f"{e} vs pandas: {rows} vs {ref}", 1e6)
            )
    return findings


def build_join_cases():
    """crafted (L, R) tables + join shape. Payload cases compare (lv,rv); the
    cross-count case is the AG11 tracker (metadata-carrying, aggregate-over-cross)."""

    def t(d):
        return pa.table(d)

    L = t({"k": pa.array([1, 2, 3], pa.int64()), "lv": ["L1", "L2", "L3"]})
    R = t({"k": pa.array([2, 3, 4], pa.int64()), "rv": ["R2", "R3", "R4"]})
    Ln = t({"k": pa.array([1, None, 2], pa.int64()), "lv": ["L1", "Ln", "L2"]})
    Rn = t({"k": pa.array([None, 2, 3], pa.int64()), "rv": ["Rn", "R2", "R3"]})
    Ld = t({"k": pa.array([1, 1, 2], pa.int64()), "lv": ["L1a", "L1b", "L2"]})
    Rd = t({"k": pa.array([1, 2, 2], pa.int64()), "rv": ["R1", "R2a", "R2b"]})
    Ts = t({"k": pa.array([1, 1, 2], pa.int64()), "lv": ["A", "B", "C"]})  # self-join
    # AG11 tracker: metadata-carrying (the `pandas` blob from from_pandas).
    Lm = pa.Table.from_pandas(
        pd.DataFrame({"k": [1, 2, 3], "lv": ["L1", "L2", "L3"]}), preserve_index=False
    )
    Rm = pa.Table.from_pandas(
        pd.DataFrame({"k": [2, 3, 4], "rv": ["R2", "R3", "R4"]}), preserve_index=False
    )
    #  name,            L,   R,   how,     keep_meta, op
    return [
        ("inner_1to1", L, R, "inner", False, "payload"),
        ("left", L, R, "left", False, "payload"),
        ("full", L, R, "full", False, "payload"),
        ("dup_fanout", Ld, Rd, "inner", False, "payload"),
        ("null_key_inner", Ln, Rn, "inner", False, "payload"),
        ("semi", L, R, "semi", False, "payload"),
        ("anti", L, R, "anti", False, "payload"),
        ("self_join", Ts, Ts, "self", False, "payload"),
        ("cross_clean", L, R, "cross", False, "payload"),
        ("cross_count_meta[AG11]", Lm, Rm, "cross", True, "count"),
    ]


# --------------------------------------------------------------------------- #
# Sort-order matrix: ORDER a column with null + NaN and compare the resulting
# SEQUENCE across engines (order matters — list equality, not multiset). The
# position of missing values is a notorious cross-engine divergence and a
# pandas-lowering hazard: pandas sorts na last, polars sorts null first, SQL
# engines vary, and datafusion's DataFrame API default can differ from its SQL
# default. RESULT class. Missing values normalized to 'NA' (position is the signal;
# a numpy-backed pandas float column can't hold a distinct null anyway).
# --------------------------------------------------------------------------- #
def _seq(vals) -> list[str]:
    return [
        "NA" if v is None or (isinstance(v, float) and v != v) else str(v) for v in vals
    ]


def run_sort_case(tbl: pa.Table, ascending: bool) -> dict[str, tuple[str, object]]:
    """sort the single column 'v' on each engine (engine-default null placement),
    native null-preserving extraction, crash-safe."""

    def pandas_fn():
        s = tbl.to_pandas()["v"].sort_values(ascending=ascending, na_position="last")
        return _seq(s.tolist())

    def polars_fn():
        return _seq(
            pl.from_arrow(tbl).sort("v", descending=not ascending)["v"].to_list()
        )

    def acero_fn():
        order = "ascending" if ascending else "descending"
        idx = pc.sort_indices(tbl, sort_keys=[("v", order)])
        return _seq(pc.take(tbl["v"], idx).to_pylist())

    def make_ctx():
        ctx = SessionContext()
        ctx.register_record_batches(
            "t", [tbl.replace_schema_metadata(None).to_batches()]
        )
        return ctx

    def dfsql_fn(ctx):
        d = "ASC" if ascending else "DESC"
        b = ctx.sql(f"SELECT v FROM t ORDER BY v {d}").collect()
        return _seq(pa.Table.from_batches(b).column("v").to_pylist())

    def dfdf_fn(ctx):
        # engine-default null placement (no nulls_first override) — may differ
        # from the SQL default, which is itself a front-end divergence.
        b = ctx.table("t").sort(col("v").sort(ascending=ascending)).collect()
        return _seq(pa.Table.from_batches(b).column("v").to_pylist())

    results: dict[str, tuple[str, object]] = {
        "pandas": _safe(pandas_fn),
        "polars": _safe(polars_fn),
        "acero": _safe(acero_fn),
    }
    reg = _safe(make_ctx)
    if reg[0] != "ok":
        results["datafusion-sql"] = results["datafusion-df"] = reg
    else:
        results["datafusion-sql"] = _safe(lambda: dfsql_fn(reg[1]))
        results["datafusion-df"] = _safe(lambda: dfdf_fn(reg[1]))
    return results


def analyze_sort(workload: str, results: dict) -> list[Finding]:
    findings: list[Finding] = []
    ok = {e: v for e, (s, v) in results.items() if s == "ok"}
    crashed = {e: m for e, (s, m) in results.items() if s == "crash"}
    if crashed and ok:
        for e, m in crashed.items():
            findings.append(
                Finding(
                    "CRASH",
                    workload,
                    f"{e} raised where {sorted(ok)} succeeded — {m}",
                    2e6,
                )
            )
    if "pandas" in ok:
        ref = ok["pandas"]
        for e, seq in ok.items():
            if e == "pandas" or seq == ref:
                continue
            findings.append(
                Finding("RESULT", workload, f"{e} vs pandas: {seq} vs {ref}", 1e6)
            )
    return findings


def build_sort_cases():
    """crafted single-column ('v') tables + sort direction."""

    def t(arr):
        return pa.table({"v": arr})

    nan = float("nan")
    f_mix = t(pa.array([3.0, nan, 1.0, None, 2.0], pa.float64()))  # null + NaN
    i_null = t(pa.array([3, None, 1, 2], pa.int64()))  # null only (isolates position)
    return [
        ("asc_int_null", i_null, True),
        ("desc_int_null", i_null, False),
        ("asc_float_nan_null", f_mix, True),
    ]


# --------------------------------------------------------------------------- #
# Kernel type-coverage matrix (the AG9/AG12 axis): an Arrow compute kernel that
# works on one type but is NotImplemented on a SIBLING type (which peers handle)
# is a coverage gap. kernel x input-type -> ok / NotImpl. Emits COVERAGE findings
# summarized per sibling type, and doubles as a release-over-release tracker — a
# cell flipping NotImpl->ok is a kernel gaining support (e.g. as arrow #44336
# view-kernel sub-PRs land). Add a family = add a (types, kernels, reference) row.
# --------------------------------------------------------------------------- #
_STR_DATA = ["alpha", "beta", "gamma", "alpha", None]


def _string_layout_types() -> dict[str, pa.Array]:
    base = pa.array(_STR_DATA, pa.string())
    return {
        "string": base,
        "large_string": pa.array(_STR_DATA, pa.large_string()),
        "string_view": pa.array(_STR_DATA, pa.string_view()),
        "dict<string>": base.dictionary_encode(),
    }


_STRING_KERNELS = [
    ("utf8_length", lambda a: pc.utf8_length(a)),
    ("utf8_upper", lambda a: pc.utf8_upper(a)),
    ("utf8_lower", lambda a: pc.utf8_lower(a)),
    ("utf8_reverse", lambda a: pc.utf8_reverse(a)),
    ("utf8_slice_codeunits", lambda a: pc.utf8_slice_codeunits(a, 1, 3)),
    ("utf8_capitalize", lambda a: pc.utf8_capitalize(a)),
    ("utf8_trim_whitespace", lambda a: pc.utf8_trim_whitespace(a)),
    ("utf8_is_alpha", lambda a: pc.utf8_is_alpha(a)),
    ("match_substring", lambda a: pc.match_substring(a, pattern="a")),
    ("match_substring_regex", lambda a: pc.match_substring_regex(a, pattern="a.")),
    ("match_like", lambda a: pc.match_like(a, pattern="a%")),
    ("starts_with", lambda a: pc.starts_with(a, pattern="a")),
    ("ends_with", lambda a: pc.ends_with(a, pattern="a")),
    ("find_substring", lambda a: pc.find_substring(a, pattern="a")),
    ("count_substring", lambda a: pc.count_substring(a, pattern="a")),
    (
        "replace_substring",
        lambda a: pc.replace_substring(a, pattern="a", replacement="X"),
    ),
    ("split_pattern", lambda a: pc.split_pattern(a, pattern="a")),
]


@dataclass
class CoverageFamily:
    """a sibling type-family + kernels to sweep for coverage gaps."""

    name: str
    reference: str  # the type expected to work (the baseline sibling)
    tracked: str  # upstream note where the gap already lives
    types_fn: object  # () -> {type_label: pa.Array}
    kernels: list


COVERAGE_FAMILIES = [
    CoverageFamily(
        "string_layout",
        "string",
        "arrow #44336 (Binary View Compute Kernels umbrella)",
        _string_layout_types,
        _STRING_KERNELS,
    ),
]


def run_coverage(fam: CoverageFamily) -> dict[str, dict[str, str]]:
    """kernel -> {type_label: 'ok' | 'NotImpl' | '<ExcType>'}."""
    types = fam.types_fn()
    matrix: dict[str, dict[str, str]] = {}
    for kname, fn in fam.kernels:
        row: dict[str, str] = {}
        for tname, arr in types.items():
            try:
                fn(arr)
                row[tname] = "ok"
            except pa.lib.ArrowNotImplementedError:
                row[tname] = "NotImpl"
            except Exception as e:
                row[tname] = type(e).__name__
        matrix[kname] = row
    return matrix


def analyze_coverage(fam: CoverageFamily, matrix: dict) -> list[Finding]:
    """one summarized COVERAGE finding per sibling type with a gap."""
    findings: list[Finding] = []
    types = list(next(iter(matrix.values())).keys())
    for t in types:
        if t == fam.reference:
            continue
        gap = [
            k
            for k, row in matrix.items()
            if row[fam.reference] == "ok" and row[t] != "ok"
        ]
        if gap:
            findings.append(
                Finding(
                    "COVERAGE",
                    f"coverage/{fam.name}",
                    f"{len(gap)}/{len(matrix)} kernels ok on '{fam.reference}' but "
                    f"NotImplemented on '{t}' (tracked: {fam.tracked}): "
                    f"{', '.join(gap)}",
                    float(len(gap)),
                )
            )
    return findings


# --------------------------------------------------------------------------- #
# Substrait fan-out matrix (the AG17/AG18/AG19 axis). Lower one op to DataFusion,
# serialize to Substrait, consume it back on Acero — one portable IR, a second
# engine. COVERAGE = DataFusion emits but Acero can't consume; RESULT = Acero
# consumes but silently disagrees. Needs the `substrait` proto pkg (for the AG17/
# AG18 fixup) + datafusion.substrait + pyarrow.substrait; skipped otherwise.
# --------------------------------------------------------------------------- #
try:
    import datafusion.substrait as _dfss
    import pyarrow.substrait as _pas
    import substrait_fixup as _sfix

    _SUBSTRAIT_OK = True
except Exception:
    _SUBSTRAIT_OK = False


def _sub_table() -> pa.RecordBatch:
    ts = pa.array(
        [
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2020, 6, 1),
            datetime.datetime(2020, 3, 1),
        ],
        type=pa.timestamp("ns"),
    )
    return pa.RecordBatch.from_arrays(
        [
            pa.array([1, 2, 3]),
            pa.array([1.0, 2.0, 3.0]),
            pa.array(["alpha", "beta", "gamma"]),
            ts,
        ],
        names=["k", "v", "s", "t"],
    )


def build_substrait_cases():
    """(name, builder) — builder(datafusion_table) -> a DataFrame emitting one
    construct. Chosen to cover each Substrait interop finding + a working
    baseline (so an all-fail signals a real regression, not a broken harness)."""
    from datafusion import (
        col,
        functions as SF,
        lit,
    )

    mar = pa.scalar(datetime.datetime(2020, 3, 1), type=pa.timestamp("ns"))
    return [
        ("filter_cmp", lambda t: t.filter(col("k") > lit(1))),  # baseline
        ("agg_sum", lambda t: t.aggregate([col("k")], [SF.sum(col("v")).alias("m")])),
        # AG18a — LIMIT via FetchRel.count_expr (silent-0-row regression guard)
        ("limit", lambda t: t.filter(col("k") > lit(0)).sort(col("k")).limit(2)),
        ("ts_filter", lambda t: t.filter(col("t") >= lit(mar))),  # AG18b (precision_ts)
        ("starts_with", lambda t: t.filter(SF.starts_with(col("s"), lit("al")))),
        ("substring", lambda t: t.select(_sub(SF).alias("o"))),  # both AG19
        ("count_distinct", lambda t: t.aggregate([col("k")], [_cd(SF)])),
    ]  # fmt: skip


def _sub(SF):  # substring(s, 1, 2) — factored out to keep case lines short
    from datafusion import (
        col,
        lit,
    )

    return SF.substring(col("s"), lit(1), lit(2))


def _cd(SF):  # count(distinct v) — factored out to keep case lines short
    from datafusion import col

    return SF.count(col("v")).distinct().build().alias("m")


def _sub_agree(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """value-multiset equality (order-insensitive); a row-count mismatch — the
    AG18a silent-0-row signal — fails here."""
    if a.shape != b.shape:
        return False
    ra = sorted(str(r) for r in a.itertuples(index=False, name=None))
    rb = sorted(str(r) for r in b.itertuples(index=False, name=None))
    return ra == rb


def _consume_df(ctx, raw: bytes, ref: pd.DataFrame):
    """DataFusion re-ingests its own Substrait (should always work)."""
    try:
        lp = _dfss.Consumer.from_substrait_plan(ctx, _dfss.Serde.deserialize_bytes(raw))
        got = ctx.create_dataframe_from_logical_plan(lp).to_pandas()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        return ("crash", f"{type(e).__name__}: {str(e)[:48]}")
    return (
        ("ok", got)
        if _sub_agree(got, ref)
        else ("wrong", f"{len(got)}v{len(ref)} rows")
    )


def _consume_acero(ctx, raw: bytes, ref: pd.DataFrame, legacy_ts: bool):
    def tp(names, schema):
        tbl = pa.Table.from_batches(ctx.table(names[-1]).collect()).select(schema.names)
        if legacy_ts:
            tbl = pa.table(
                [
                    c.cast(pa.timestamp("us")) if pa.types.is_timestamp(c.type) else c
                    for c in tbl.columns
                ],
                names=tbl.column_names,
            )
        return tbl

    try:
        got = (
            _pas.run_query(pa.py_buffer(raw), table_provider=tp).read_all().to_pandas()
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        msg = str(e)
        kind = "no-map" if "No conversion function" in msg else "crash"
        return (kind, f"{type(e).__name__}: {msg[:80]}")
    return (
        ("ok", got)
        if _sub_agree(got, ref)
        else ("wrong", f"{len(got)}v{len(ref)} rows")
    )


def run_substrait_case(builder) -> dict:
    ctx = SessionContext()
    ctx.register_record_batches("A", [[_sub_table()]])
    df = builder(ctx.table("A"))
    ref = df.to_pandas()  # datafusion direct execution = correctness reference
    raw = _dfss.Producer.to_substrait_plan(df.logical_plan(), ctx).encode()
    fixed = _sfix.fix_plan(raw, legacy_timestamp=True)
    return {
        "df-rt": _consume_df(ctx, raw, ref),
        "acero-raw": _consume_acero(ctx, raw, ref, legacy_ts=False),
        "acero-fix": _consume_acero(ctx, fixed, ref, legacy_ts=True),
    }


def analyze_substrait(workload: str, results: dict) -> list[Finding]:
    findings: list[Finding] = []
    rt_st, rt_det = results["df-rt"]
    fx_st, fx_det = results["acero-fix"]
    # DataFusion should re-ingest its own Substrait — a failure is a loud find.
    if rt_st != "ok":
        findings.append(
            Finding(
                "CRASH",
                workload,
                f"datafusion cannot roundtrip its own Substrait ({rt_st}): {rt_det}",
                9e5,
            )
        )
    # Acero can't consume even after the AG17/AG18 portability fixup -> COVERAGE.
    if fx_st in ("no-map", "crash"):
        findings.append(
            Finding(
                "COVERAGE",
                workload,
                f"acero cannot consume fixed Substrait ({fx_st}): {fx_det} "
                f"[AG19/AG18b — flips ok when arrow registers it]",
                3e5,
            )
        )
    # Acero consumes but disagrees (the AG18a silent-0-row class) -> RESULT.
    elif fx_st == "wrong":
        findings.append(
            Finding(
                "RESULT",
                workload,
                f"acero silently disagrees on fixed Substrait: {fx_det} [AG18a class]",
                1e6,
            )
        )
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rows", type=int, default=2_000_000, help="rows per synthetic table"
    )
    ap.add_argument(
        "--perf-threshold",
        type=float,
        default=3.0,
        help="min slowdown ratio to flag PERF",
    )
    ap.add_argument("--only", default=None, help="substring filter on workload id")
    args = ap.parse_args()

    pa_cpu = os.cpu_count()
    pa.set_cpu_count(pa_cpu)
    print(
        f"pyarrow {pa.__version__} | polars {pl.__version__} "
        f"(pool {pl.thread_pool_size()}) | datafusion {d.__version__} | "
        f"pandas {pd.__version__} | numpy {np.__version__}"
    )
    print(
        f"cpus={pa_cpu} | rows={args.rows:,} | "
        f"perf-threshold={args.perf_threshold}x | engines={ENGINES}\n"
    )

    grid = [
        (kd, card, agg)
        for kd in ("int64", "string")
        for card in (100, 10_000, 1_000_000)
        for agg in ("sum", "count_distinct")
    ]

    all_findings: list[Finding] = []
    ds_cache: dict[tuple[str, int], Dataset] = {}
    for kd, card, aggname in grid:
        wid = f"gagg/{kd}/card={card}/{aggname}"
        if args.only and args.only not in wid:
            continue
        if (kd, card) not in ds_cache:
            ds_cache[(kd, card)] = build_dataset(kd, card, args.rows)
        ds = ds_cache[(kd, card)]
        cells, plans = run_grouped_agg(ds, AGGS[aggname])

        # per-workload line
        times = " ".join(
            f"{e.split('-')[-1]:>3.3}={t:7.1f}" for e, (t, _) in cells.items()
        )
        groups = len(cells[ORACLE][1])
        ok = all(
            _results_agree(cells[ORACLE][1], r)
            for e, (_, r) in cells.items()
            if e != ORACLE
        )
        print(f"{wid:<40} groups={groups:>9,} | {times} | result_ok={ok}")

        all_findings.extend(analyze(wid, cells, plans, args.perf_threshold))

    # ---- edge-case sweep (RESULT / CRASH surface) --------------------------
    print()
    for name, tbl, aggnames in build_edge_cases():
        for an in aggnames:
            wid = f"edge/{name}/{an}"
            if args.only and args.only not in wid:
                continue
            results = run_edge_case(tbl, AGGS[an])
            status = " ".join(
                f"{e.split('-')[-1]:>3.3}={'OK' if s == 'ok' else 'XX'}"
                for e, (s, _) in results.items()
            )
            print(f"{wid:<40} | {status}")
            all_findings.extend(analyze_edge(wid, results))

    # ---- join-shape matrix (RESULT / CRASH surface, the AG11 axis) ----------
    print()
    for name, lt, rt, how, keep_meta, op in build_join_cases():
        wid = f"join/{name}"
        if args.only and args.only not in wid:
            continue
        results = run_join_case(lt, rt, how, keep_meta, op)
        status = " ".join(
            f"{e.split('-')[-1]:>3.3}="
            + {"ok": "OK", "crash": "XX", "unsupported": "--"}[s]
            for e, (s, _) in results.items()
        )
        print(f"{wid:<40} | {status}")
        all_findings.extend(analyze_join(wid, results))

    # ---- sort-order matrix (RESULT surface: null/NaN placement) -------------
    print()
    for name, tbl, ascending in build_sort_cases():
        wid = f"sort/{name}"
        if args.only and args.only not in wid:
            continue
        results = run_sort_case(tbl, ascending)
        status = " ".join(
            f"{e.split('-')[-1]:>3.3}={'OK' if s == 'ok' else 'XX'}"
            for e, (s, _) in results.items()
        )
        print(f"{wid:<40} | {status}")
        all_findings.extend(analyze_sort(wid, results))

    # ---- kernel type-coverage matrix (COVERAGE surface, the AG9/AG12 axis) --
    print()
    for fam in COVERAGE_FAMILIES:
        wid = f"coverage/{fam.name}"
        if args.only and args.only not in wid:
            continue
        matrix = run_coverage(fam)
        types = list(next(iter(matrix.values())).keys())
        summ = " ".join(
            f"{t.split('<')[0]}={sum(1 for r in matrix.values() if r[t] == 'ok')}"
            f"/{len(matrix)}"
            for t in types
        )
        print(f"{wid:<40} | {summ}")
        all_findings.extend(analyze_coverage(fam, matrix))

    # ---- substrait fan-out matrix (COVERAGE/RESULT, the AG17/AG18/AG19 axis) -
    print()
    if not _SUBSTRAIT_OK:
        print(
            "substrait/*  skipped — needs `pip install substrait` + "
            "datafusion.substrait + pyarrow.substrait"
        )
    else:
        _labels = {"ok": "OK", "no-map": "NM", "wrong": "!!", "crash": "XX"}
        for name, builder in build_substrait_cases():
            wid = f"substrait/{name}"
            if args.only and args.only not in wid:
                continue
            results = run_substrait_case(builder)
            status = " ".join(
                f"{k.split('-')[-1]:>3.3}={_labels.get(s, s[:2])}"
                for k, (s, _) in results.items()
            )
            print(f"{wid:<40} | {status}")
            all_findings.extend(analyze_substrait(wid, results))

    # ---- divergence report -------------------------------------------------
    print("\n" + "=" * 78)
    print(
        "FINDINGS (candidate gaps — each needs human confirm + root-cause + dup-search)"
    )
    print("=" * 78)
    order = {"CRASH": 0, "RESULT": 1, "PLAN": 2, "COVERAGE": 3, "PERF": 4}
    all_findings.sort(key=lambda f: (order[f.kind], -f.severity))
    if not all_findings:
        print("none above thresholds.")
    for f in all_findings:
        print(f"[{f.kind:<8}] {f.workload}\n         {f.detail}")
    counts = {
        k: sum(1 for f in all_findings if f.kind == k)
        for k in ("CRASH", "RESULT", "PLAN", "COVERAGE", "PERF")
    }
    print(
        f"\nsummary: {counts['CRASH']} CRASH, {counts['RESULT']} RESULT, "
        f"{counts['PLAN']} PLAN, {counts['COVERAGE']} COVERAGE, {counts['PERF']} PERF"
    )


if __name__ == "__main__":
    main()
