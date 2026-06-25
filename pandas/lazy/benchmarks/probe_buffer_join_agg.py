#!/usr/bin/env python3
"""Probe: can a buffer-resident nogil Cython join->agg reach the join half of
the Polars gap, or does the payload gather / Arrow<->NumPy round-trip cap it?

Decision question (see docs/QGAP_DECOMP.md, PERF_CEILING.md): joins are the
biggest TPC-H tax. ``_try_cython_join`` already beats ``pd.merge`` at the
*indexer* level but is gated OFF wide payloads, because "the per-column gather
of an exploded wide payload loses to pd.merge's consolidated block take." In a
join->agg pipeline you never need the wide payload -- only {group key, agg
value} survive the group. So: if we PROJECT to those columns *before* the
gather and stay on Arrow/NumPy buffers, does a Cython join + buffer-resident
group beat merge-then-group and approach Polars?

Representative shape (q3/q5/q9/q10): lineitem |><| orders on orderkey, then
group by o_orderdate, sum(l_extendedprice). Every lineitem matches one order,
so the join output is the full lineitem cardinality (the "large output" that
sank Acero); the group is low-cardinality (~2400 dates).

Paths measured (all produce the SAME grouped result, asserted equal):
  POLARS          native lazy join->group->agg, collect           (target)
  MERGE_WIDE      pd.merge ALL columns, then groupby (no pushdown) (worst)
  MERGE_NARROW    pd.merge only {key,date,price}, then groupby     (pushdown)
  BUFFER_NARROW   lazy_join indices -> pc.take {date,price} only
                  -> arrow group_by                                (hypothesis)

BUFFER_NARROW is decomposed into probe / gather / group sub-stages so we can
see whether the gather (the round-trip wall) dominates when narrow.

Read-only. SF-1 default (lineitem ~6M). Run:
    python pandas/lazy/benchmarks/probe_buffer_join_agg.py --sf 1
"""

from __future__ import annotations

import argparse
import time

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

import pandas as pd
from pandas.lazy.backends.numpy.join import inner_join_indexers_i8


def _t(fn, n=5):
    """Median of n timed runs (ms), plus the result of the last run."""
    out = None
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), out


def make_data(sf: float):
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    li = con.execute(
        "SELECT l_orderkey, l_extendedprice, l_quantity, l_discount, "
        "l_tax, l_partkey, l_suppkey, l_shipmode, l_comment FROM lineitem"
    ).df()
    orders = con.execute(
        "SELECT o_orderkey, o_orderdate, o_custkey, o_orderpriority, "
        "o_clerk, o_comment FROM orders"
    ).df()
    con.close()
    # orderdate -> int64 days (a packable group key). Resolution-robust:
    # cast to datetime64[D] then view as int (pandas 3.x to_datetime is not
    # guaranteed ns, so dividing a raw int64 by a fixed ns/day factor is wrong).
    od = pd.to_datetime(orders["o_orderdate"]).to_numpy().astype("datetime64[D]")
    orders["o_orderdate"] = od.astype("int64")
    for c in ("l_extendedprice", "l_quantity", "l_discount", "l_tax"):
        li[c] = li[c].astype("float64")
    return li, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    print(f"pyarrow {pa.__version__}  polars {pl.__version__}  sf={args.sf}")
    li, orders = make_data(args.sf)
    print(
        f"lineitem={len(li):,}  orders={len(orders):,}  "
        f"groups={orders['o_orderdate'].nunique():,}"
    )

    lk = li["l_orderkey"].to_numpy()
    ok = orders["o_orderkey"].to_numpy()
    price = li["l_extendedprice"].to_numpy()
    odate = orders["o_orderdate"].to_numpy()

    # ---- reference result (pandas merge-narrow + groupby) ----
    ref = (
        pd.DataFrame({"l_orderkey": lk, "l_extendedprice": price})
        .merge(
            pd.DataFrame({"o_orderkey": ok, "o_orderdate": odate}),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .groupby("o_orderdate")["l_extendedprice"]
        .sum()
        .sort_index()
    )

    def check(s, name):
        got = s.sort_index()
        assert np.allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-6), (
            f"{name} mismatch"
        )
        assert np.array_equal(got.index.to_numpy(), ref.index.to_numpy()), name

    results = {}

    # ---- POLARS ----
    li_pl = pl.DataFrame({"l_orderkey": lk, "l_extendedprice": price})
    ord_pl = pl.DataFrame({"o_orderkey": ok, "o_orderdate": odate})

    def polars_run():
        return (
            li_pl.lazy()
            .join(ord_pl.lazy(), left_on="l_orderkey", right_on="o_orderkey")
            .group_by("o_orderdate")
            .agg(pl.col("l_extendedprice").sum())
            .collect()
        )

    results["POLARS"], pout = _t(polars_run, args.n)
    check(
        pd.Series(
            pout["l_extendedprice"].to_numpy(),
            index=pd.Index(pout["o_orderdate"].to_numpy(), name="o_orderdate"),
        ),
        "POLARS",
    )

    # ---- MERGE_WIDE: carry the full payload through the join, then group ----
    li_full = li.copy()
    ord_full = orders.copy()

    def merge_wide():
        m = li_full.merge(ord_full, left_on="l_orderkey", right_on="o_orderkey")
        return m.groupby("o_orderdate")["l_extendedprice"].sum()

    results["MERGE_WIDE"], mw = _t(merge_wide, args.n)
    check(mw, "MERGE_WIDE")

    # ---- MERGE_NARROW: projection pushed down to {key,date,price} ----
    li_n = pd.DataFrame({"l_orderkey": lk, "l_extendedprice": price})
    ord_n = pd.DataFrame({"o_orderkey": ok, "o_orderdate": odate})

    def merge_narrow():
        m = li_n.merge(ord_n, left_on="l_orderkey", right_on="o_orderkey")
        return m.groupby("o_orderdate")["l_extendedprice"].sum()

    results["MERGE_NARROW"], mn = _t(merge_narrow, args.n)
    check(mn, "MERGE_NARROW")

    # ---- BUFFER_NARROW: cython join indices -> narrow Arrow take -> group ----
    price_pa = pa.array(price)
    odate_pa = pa.array(odate)

    def buffer_narrow():
        li_idx, oi_idx = inner_join_indexers_i8(lk, ok, max_hit_fraction=None)
        g_price = pc.take(price_pa, pa.array(li_idx))
        g_date = pc.take(odate_pa, pa.array(oi_idx))
        tbl = pa.table({"o_orderdate": g_date, "l_extendedprice": g_price})
        grouped = tbl.group_by("o_orderdate").aggregate([("l_extendedprice", "sum")])
        return pd.Series(
            grouped["l_extendedprice_sum"].to_numpy(),
            index=pd.Index(grouped["o_orderdate"].to_numpy(), name="o_orderdate"),
        )

    results["BUFFER_NARROW"], bn = _t(buffer_narrow, args.n)
    check(bn, "BUFFER_NARROW")

    # ---- decompose BUFFER_NARROW into probe / gather / group ----
    def sub_probe():
        return inner_join_indexers_i8(lk, ok, max_hit_fraction=None)

    t_probe, (li_idx, oi_idx) = _t(sub_probe, args.n)
    li_idx_pa, oi_idx_pa = pa.array(li_idx), pa.array(oi_idx)

    def sub_gather():
        return pc.take(price_pa, li_idx_pa), pc.take(odate_pa, oi_idx_pa)

    t_gather, (g_price, g_date) = _t(sub_gather, args.n)

    def sub_group():
        tbl = pa.table({"o_orderdate": g_date, "l_extendedprice": g_price})
        return tbl.group_by("o_orderdate").aggregate([("l_extendedprice", "sum")])

    t_group, _ = _t(sub_group, args.n)

    # ---- report ----
    print(f"\n{'path':<16}{'ms':>9}   x vs Polars")
    pl_ms = results["POLARS"]
    for name in ["POLARS", "MERGE_WIDE", "MERGE_NARROW", "BUFFER_NARROW"]:
        ms = results[name]
        x = pl_ms / ms if ms else float("nan")
        print(f"{name:<16}{ms:>9.1f}   {x:>5.2f}x")

    print(f"\nBUFFER_NARROW sub-stage decomposition (n_out={len(li_idx):,}):")
    print(f"  probe (join indices)   {t_probe:>8.1f} ms")
    print(f"  gather (2-col pc.take)  {t_gather:>8.1f} ms")
    print(f"  group (arrow group_by) {t_group:>8.1f} ms")
    print(f"  sum of stages          {t_probe + t_gather + t_group:>8.1f} ms")

    print("\nleverage:")
    print(
        f"  wide->narrow payload pushdown saves "
        f"{results['MERGE_WIDE'] - results['MERGE_NARROW']:.1f} ms "
        f"({results['MERGE_WIDE'] / results['MERGE_NARROW']:.2f}x)"
    )
    print(
        f"  buffer vs merge_narrow: "
        f"{results['MERGE_NARROW'] / results['BUFFER_NARROW']:.2f}x"
    )
    print(
        f"  buffer vs Polars:       "
        f"{results['POLARS'] / results['BUFFER_NARROW']:.2f}x  "
        f"(>=1.0 => buffer path reaches Polars)"
    )


if __name__ == "__main__":
    main()
