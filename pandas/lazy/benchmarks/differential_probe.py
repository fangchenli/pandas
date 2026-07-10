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

Three divergence classes, most-valuable first:

    RESULT  engines disagree on the answer (correctness/bug — highest value; the
            surface our ad-hoc perf matrices never covered)
    PLAN    datafusion SQL vs DataFrame-API optimized plans differ in operator
            shape (the AG10 class — an optimizer rule fires for one front-end)
    PERF    one engine is >Nx slower on the same cell (the AG3/4/5 class)

Self-contained: pyarrow + polars + datafusion + numpy + pandas, synthetic data.
Run on the *current* releases in a throwaway venv (see ``upstream/README.md`` §1),
never the repo's pinned pyarrow — the whole point is to catch what changed:

    python -m venv /tmp/upstream-venv && . /tmp/upstream-venv/bin/activate
    pip install -U pyarrow polars datafusion numpy pandas
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
    kind: str  # RESULT | PLAN | PERF
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

    # ---- divergence report -------------------------------------------------
    print("\n" + "=" * 78)
    print(
        "FINDINGS (candidate gaps — each needs human confirm + root-cause + dup-search)"
    )
    print("=" * 78)
    order = {"RESULT": 0, "PLAN": 1, "PERF": 2}
    all_findings.sort(key=lambda f: (order[f.kind], -f.severity))
    if not all_findings:
        print("none above thresholds.")
    for f in all_findings:
        print(f"[{f.kind:<6}] {f.workload}\n         {f.detail}")
    counts = {
        k: sum(1 for f in all_findings if f.kind == k)
        for k in ("RESULT", "PLAN", "PERF")
    }
    print(
        f"\nsummary: {counts['RESULT']} RESULT, "
        f"{counts['PLAN']} PLAN, {counts['PERF']} PERF"
    )


if __name__ == "__main__":
    main()
