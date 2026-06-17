"""Per-operator decomposition of the worst TPC-H queries (#1 campaign).

Localises *where* q20 (0.16x) and q21 (0.27x) lose vs Polars by timing each
cumulative prefix of the query for both engines, then differencing to get a
per-stage cost. Also dumps the chosen physical plan so we can see which joins
reached acero (M5) vs fell to single-threaded pd.merge, and where groupby /
sort / distinct land.

Lazy collects use ``use_physical_planner=True`` — the mode the TPC-H
benchmark/scorecard actually runs (the default ``.collect()`` is eager pandas
and gives misleading numbers; an earlier pass mismeasured this).

Read-only measurement; SF-1 by default (disk-safe — no large spill). Run:

    python pandas/lazy/benchmarks/exp_qgap_decomp.py --sf 1 --query 21

Writes a markdown decomposition to docs/QGAP_DECOMP.md and prints a summary.
"""

from __future__ import annotations

import argparse
import time

import duckdb
import polars as pl

import pandas as pd
from pandas.lazy import col

TABLES = [
    "lineitem",
    "orders",
    "customer",
    "supplier",
    "part",
    "partsupp",
    "nation",
    "region",
]


def make_tables(sf: float):
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    import decimal

    pdt = {}
    plt = {}
    for name in TABLES:
        df = con.execute(f"SELECT * FROM {name}").df()
        for c in df.columns:
            if (
                df[c].dtype == object
                and len(df)
                and isinstance(df[c].iloc[0], decimal.Decimal)
            ):
                df[c] = df[c].astype("float64")
        pdt[name] = df
        plt[name] = pl.from_pandas(df)
    con.close()
    return pdt, plt


def _lp(df):
    return df.select()


def _time(build, warm=1, runs=3):
    """Time a thunk that builds + collects a query prefix. Returns best ms."""
    for _ in range(warm):
        build()
    best = float("inf")
    for _ in range(runs):
        s = time.perf_counter()
        build()
        best = min(best, (time.perf_counter() - s) * 1000)
    return best


# ---------------------------------------------------------------------------
# Q20 — cumulative prefixes (lazy + polars), each ends in .collect()
# ---------------------------------------------------------------------------
def q20_lp_stages(t):
    lo, hi = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")

    def forest():
        return (
            _lp(t["part"])
            .filter(col("p_name").str.startswith("forest"))
            .select(col("p_partkey"))
        )

    def qty():
        return (
            _lp(t["lineitem"])
            .filter((col("l_shipdate") >= lo) & (col("l_shipdate") < hi))
            .group_by("l_partkey", "l_suppkey")
            .agg((0.5 * col("l_quantity").sum()).alias("half_qty"))
        )

    def target_join1():
        return _lp(t["partsupp"]).join(
            forest(), left_on="ps_partkey", right_on="p_partkey"
        )

    def target_join2():
        return target_join1().join(
            qty(),
            left_on=["ps_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )

    def target():
        return (
            target_join2()
            .filter(col("ps_availqty") > col("half_qty"))
            .select(col("ps_suppkey"))
            .distinct()
        )

    def final():
        return (
            _lp(t["supplier"])
            .join(
                _lp(t["nation"]).filter(col("n_name") == "CANADA"),
                left_on="s_nationkey",
                right_on="n_nationkey",
            )
            .join(target(), left_on="s_suppkey", right_on="ps_suppkey")
            .select(col("s_name"), col("s_address"))
            .sort("s_name")
        )

    c = {"use_physical_planner": True}
    return [
        ("forest (filter part)", lambda: forest().collect(**c)),
        ("qty (filter+gb lineitem)", lambda: qty().collect(**c)),
        ("+ partsupp⋈forest", lambda: target_join1().collect(**c)),
        ("+ ⋈qty", lambda: target_join2().collect(**c)),
        ("+ filter+distinct→target", lambda: target().collect(**c)),
        ("+ supplier⋈nation⋈target+sort", lambda: final().collect(**c)),
    ], final


def q20_pl_stages(t):
    lo, hi = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")

    def forest():
        return (
            t["part"]
            .lazy()
            .filter(pl.col("p_name").str.starts_with("forest"))
            .select("p_partkey")
        )

    def qty():
        return (
            t["lineitem"]
            .lazy()
            .filter((pl.col("l_shipdate") >= lo) & (pl.col("l_shipdate") < hi))
            .group_by("l_partkey", "l_suppkey")
            .agg((0.5 * pl.col("l_quantity").sum()).alias("half_qty"))
        )

    def target_join1():
        return (
            t["partsupp"]
            .lazy()
            .join(forest(), left_on="ps_partkey", right_on="p_partkey", how="semi")
        )

    def target_join2():
        return target_join1().join(
            qty(),
            left_on=["ps_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )

    def target():
        return (
            target_join2()
            .filter(pl.col("ps_availqty") > pl.col("half_qty"))
            .select("ps_suppkey")
            .unique()
        )

    def final():
        return (
            t["supplier"]
            .lazy()
            .join(
                t["nation"].lazy().filter(pl.col("n_name") == "CANADA"),
                left_on="s_nationkey",
                right_on="n_nationkey",
            )
            .join(target(), left_on="s_suppkey", right_on="ps_suppkey", how="semi")
            .select("s_name", "s_address")
            .sort("s_name")
        )

    return [
        ("forest (filter part)", lambda: forest().collect()),
        ("qty (filter+gb lineitem)", lambda: qty().collect()),
        ("+ partsupp⋈forest", lambda: target_join1().collect()),
        ("+ ⋈qty", lambda: target_join2().collect()),
        ("+ filter+distinct→target", lambda: target().collect()),
        ("+ supplier⋈nation⋈target+sort", lambda: final().collect()),
    ]


# ---------------------------------------------------------------------------
# Q21 — cumulative prefixes
# ---------------------------------------------------------------------------
def q21_lp_stages(t):
    def li():
        return _lp(t["lineitem"])

    def nsupp():
        return (
            li().group_by("l_orderkey").agg(col("l_suppkey").n_unique().alias("nsupp"))
        )

    def late():
        return li().filter(col("l_receiptdate") > col("l_commitdate"))

    def late_nsupp():
        return (
            late()
            .group_by("l_orderkey")
            .agg(col("l_suppkey").n_unique().alias("late_nsupp"))
        )

    def o():
        return _lp(t["orders"]).filter(col("o_orderstatus") == "F")

    def nation():
        return _lp(t["nation"]).filter(col("n_name") == "SAUDI ARABIA")

    def j1():
        return late().join(o(), left_on="l_orderkey", right_on="o_orderkey")

    def j2():
        return j1().join(nsupp(), on="l_orderkey")

    def j3():
        return j2().join(late_nsupp(), on="l_orderkey")

    def filt():
        return j3().filter((col("nsupp") > 1) & (col("late_nsupp") == 1))

    def j4():
        return filt().join(
            _lp(t["supplier"]), left_on="l_suppkey", right_on="s_suppkey"
        )

    def j5():
        return j4().join(nation(), left_on="s_nationkey", right_on="n_nationkey")

    def final():
        return (
            j5()
            .group_by("s_name")
            .agg(col("l_orderkey").count().alias("numwait"))
            .sort("numwait", "s_name", descending=[True, False])
            .limit(100)
        )

    c = {"use_physical_planner": True}
    return [
        ("nsupp (gb n_unique)", lambda: nsupp().collect(**c)),
        ("late (filter lineitem)", lambda: late().collect(**c)),
        ("late_nsupp (gb n_unique)", lambda: late_nsupp().collect(**c)),
        ("late⋈orders", lambda: j1().collect(**c)),
        ("+ ⋈nsupp", lambda: j2().collect(**c)),
        ("+ ⋈late_nsupp", lambda: j3().collect(**c)),
        ("+ filter nsupp/late_nsupp", lambda: filt().collect(**c)),
        ("+ ⋈supplier", lambda: j4().collect(**c)),
        ("+ ⋈nation", lambda: j5().collect(**c)),
        ("+ gb count+sort+limit", lambda: final().collect(**c)),
    ], final


def q21_pl_stages(t):
    def li():
        return t["lineitem"].lazy()

    def nsupp():
        return (
            li()
            .group_by("l_orderkey")
            .agg(pl.col("l_suppkey").n_unique().alias("nsupp"))
        )

    def late():
        return li().filter(pl.col("l_receiptdate") > pl.col("l_commitdate"))

    def late_nsupp():
        return (
            late()
            .group_by("l_orderkey")
            .agg(pl.col("l_suppkey").n_unique().alias("late_nsupp"))
        )

    def o():
        return t["orders"].lazy().filter(pl.col("o_orderstatus") == "F")

    def nation():
        return t["nation"].lazy().filter(pl.col("n_name") == "SAUDI ARABIA")

    def j1():
        return late().join(o(), left_on="l_orderkey", right_on="o_orderkey")

    def j2():
        return j1().join(nsupp(), on="l_orderkey")

    def j3():
        return j2().join(late_nsupp(), on="l_orderkey")

    def filt():
        return j3().filter((pl.col("nsupp") > 1) & (pl.col("late_nsupp") == 1))

    def j4():
        return filt().join(
            t["supplier"].lazy(), left_on="l_suppkey", right_on="s_suppkey"
        )

    def j5():
        return j4().join(nation(), left_on="s_nationkey", right_on="n_nationkey")

    def final():
        return (
            j5()
            .group_by("s_name")
            .agg(pl.len().alias("numwait"))
            .sort(["numwait", "s_name"], descending=[True, False])
            .limit(100)
        )

    return [
        ("nsupp (gb n_unique)", lambda: nsupp().collect()),
        ("late (filter lineitem)", lambda: late().collect()),
        ("late_nsupp (gb n_unique)", lambda: late_nsupp().collect()),
        ("late⋈orders", lambda: j1().collect()),
        ("+ ⋈nsupp", lambda: j2().collect()),
        ("+ ⋈late_nsupp", lambda: j3().collect()),
        ("+ filter nsupp/late_nsupp", lambda: filt().collect()),
        ("+ ⋈supplier", lambda: j4().collect()),
        ("+ ⋈nation", lambda: j5().collect()),
        ("+ gb count+sort+limit", lambda: final().collect()),
    ]


REG = {
    20: (q20_lp_stages, q20_pl_stages),
    21: (q21_lp_stages, q21_pl_stages),
}


def run(qn, pdt, plt):
    lp_stages_fn, pl_stages_fn = REG[qn]
    lp_stages, final_builder = lp_stages_fn(pdt)
    pl_stages = pl_stages_fn(plt)

    bar = "=" * 72
    print(f"\n{bar}\nQ{qn} per-stage cumulative decomposition (SF data loaded)\n{bar}")
    print(f"{'stage':<38}{'lp(ms)':>9}{'pl(ms)':>9}{'lp Δ':>9}{'pl Δ':>9}")
    rows = []
    prev_lp = prev_pl = 0.0
    for (name, lp_b), (_, pl_b) in zip(lp_stages, pl_stages, strict=False):
        lp_ms = _time(lp_b)
        pl_ms = _time(pl_b)
        # Δ only meaningful for the linear chain; the first independent
        # sub-results (forest/qty/nsupp/late/late_nsupp) are standalone.
        d_lp = lp_ms - prev_lp
        d_pl = pl_ms - prev_pl
        print(f"{name:<38}{lp_ms:>9.1f}{pl_ms:>9.1f}{d_lp:>9.1f}{d_pl:>9.1f}")
        rows.append((name, lp_ms, pl_ms))
        prev_lp, prev_pl = lp_ms, pl_ms

    full = final_builder()
    print(f"\n--- Q{qn} physical plan (chosen ops) ---")
    plan = full.explain(physical=True)
    print(plan)
    return rows, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("--query", type=int, default=None, help="20 or 21; default both")
    args = ap.parse_args()

    print(f"Generating TPC-H SF-{args.sf} ...")
    pdt, plt = make_tables(args.sf)
    print("rows:", {k: len(v) for k, v in pdt.items()})

    qs = [args.query] if args.query else [21, 20]
    for qn in qs:
        run(qn, pdt, plt)


if __name__ == "__main__":
    main()
