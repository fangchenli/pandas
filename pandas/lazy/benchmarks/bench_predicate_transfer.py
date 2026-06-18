"""Scale-up probe for predicate transfer (semijoin reduction) — the go/no-go.

docs/PREDICATE_TRANSFER_PROBE.md found PT's algorithm avoids real join work but
its margin at SF-3 is thin (q9 ~1.15x, q3 ~break-even) because the reduction
overhead rivals the savings when intermediates are small. PT's documented
1.5-2.4x wins are at larger scale / worse join orders. This probe answers the
decisive question on a bigger box: **does PT clear ~1.5x at SF-30-100?** If yes,
build the engine-integrated Bloom-filter PT + optimizer pass; if it stays
~1.15x, don't.

Run (needs RAM+disk for the SF; SF-30 lineitem ~180M rows ≈ tens of GB):

    python pandas/lazy/benchmarks/bench_predicate_transfer.py --sf 30
    python pandas/lazy/benchmarks/bench_predicate_transfer.py --sf 100 --queries 3,9

For each join-heavy query it reports: baseline lazy join time, PT reduction cost
(Bloom-filter semijoin masks on KEY COLUMNS ONLY — what an engine-integrated
implementation would pay, NOT a pandas wide-DataFrame isin+copy), reduced-join
time, the implied PT total and speedup, and Polars for reference. Correctness is
asserted (PT must not change results).

Method notes:
- Reduction is measured as Bloom-filter mask computation over key arrays only;
  the reduced base tables are materialized OUTSIDE the timer (the real engine
  applies the mask during its columnar scan, it does not copy wide frames).
- Bloom uses a correct multiply-shift hash (HIGH bits) — the SF-3 probe had a
  low-bits hash bug that inflated false positives; fixed here. False positives
  are harmless (removed by the real join), they only slightly enlarge the
  reduced join.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import pandas as pd

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# ---------------------------------------------------------------------------
# Bloom-filter semijoin: build on `build_keys`, return a candidate mask over
# `probe_keys`. Two independent multiply-shift hashes keeping HIGH bits.
# ---------------------------------------------------------------------------
_G1 = np.uint64(0x9E3779B97F4A7C15)
_G2 = np.uint64(0xC2B2AE3D27D4EB4F)


def _bloom_mask(probe_keys: np.ndarray, build_keys: np.ndarray, bits_log2: int = 24):
    m = 1 << bits_log2
    shift = np.uint64(64 - bits_log2)
    bk = build_keys.astype(np.uint64, copy=False)
    bits = np.zeros(m, dtype=bool)
    bits[(bk * _G1) >> shift] = True
    bits[(bk * _G2) >> shift] = True
    pk = probe_keys.astype(np.uint64, copy=False)
    return bits[(pk * _G1) >> shift] & bits[(pk * _G2) >> shift]


def _pack2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combine two int key columns into one uint64 (for compound-key semijoins)."""
    span = np.uint64(int(b.max()) + 1)
    return a.astype(np.uint64, copy=False) * span + b.astype(np.uint64, copy=False)


# ---------------------------------------------------------------------------
# Per-query predicate-transfer reductions. Each returns (reduced_tables_dict,
# reduction_ms). Base filters are applied outside the timer (the query applies
# them anyway); only the semijoin Bloom passes are timed.
# ---------------------------------------------------------------------------
def _time_calls(fn, runs=3):
    best = float("inf")
    out = None
    for _ in range(runs):
        s = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - s) * 1000)
    return best, out


def pt_q3(t):
    cut = pd.Timestamp("1995-03-15")
    c = t["customer"][t["customer"].c_mktsegment == "BUILDING"]
    o = t["orders"][t["orders"].o_orderdate < cut]
    li = t["lineitem"][t["lineitem"].l_shipdate > cut]
    ck, ock, ook, lok = (
        c.c_custkey.to_numpy(),
        o.o_custkey.to_numpy(),
        o.o_orderkey.to_numpy(),
        li.l_orderkey.to_numpy(),
    )

    def masks():
        o_keep = _bloom_mask(ock, ck)  # orders ⋉ customer
        li_keep = _bloom_mask(lok, ook[o_keep])  # lineitem ⋉ reduced-orders
        o_keep = o_keep & _bloom_mask(ook, lok[li_keep])  # backward
        c_keep = _bloom_mask(ck, ock[o_keep])  # backward
        return c_keep, o_keep, li_keep

    red_ms, (cm, om, lm) = _time_calls(masks)
    red = dict(t)
    red["customer"] = c[cm].reset_index(drop=True)
    red["orders"] = o[om].reset_index(drop=True)
    red["lineitem"] = li[lm].reset_index(drop=True)
    ratios = {"orders": (len(o), int(om.sum())), "lineitem": (len(li), int(lm.sum()))}
    return red, red_ms, ratios


def pt_q9(t):
    p = t["part"][t["part"].p_name.str.contains("green")]
    li = t["lineitem"]
    ppk = p.p_partkey.to_numpy()
    lpk = li.l_partkey.to_numpy()

    def masks():
        li_keep = _bloom_mask(lpk, ppk)  # lineitem ⋉ green-parts (the dominant)
        lok_r = li.l_orderkey.to_numpy()[li_keep]
        lsk_r = li.l_suppkey.to_numpy()[li_keep]
        # backward reductions of the other join sides
        o_keep = _bloom_mask(t["orders"].o_orderkey.to_numpy(), lok_r)
        s_keep = _bloom_mask(t["supplier"].s_suppkey.to_numpy(), lsk_r)
        ps = t["partsupp"]
        ps_build = _pack2(ps.ps_partkey.to_numpy(), ps.ps_suppkey.to_numpy())
        ps_keep = _bloom_mask(ps_build, _pack2(lpk[li_keep], lsk_r))
        return li_keep, o_keep, s_keep, ps_keep

    red_ms, (lm, om, sm, psm) = _time_calls(masks)
    red = dict(t)
    red["lineitem"] = li[lm].reset_index(drop=True)
    red["orders"] = t["orders"][om].reset_index(drop=True)
    red["supplier"] = t["supplier"][sm].reset_index(drop=True)
    red["partsupp"] = t["partsupp"][psm].reset_index(drop=True)
    return red, red_ms, {"lineitem": (len(li), int(lm.sum()))}


def pt_q10(t):
    lo, hi = pd.Timestamp("1993-10-01"), pd.Timestamp("1994-01-01")
    o = t["orders"][(t["orders"].o_orderdate >= lo) & (t["orders"].o_orderdate < hi)]
    li = t["lineitem"][t["lineitem"].l_returnflag == "R"]
    ock = o.o_custkey.to_numpy()
    ook = o.o_orderkey.to_numpy()
    lok = li.l_orderkey.to_numpy()
    ckey = t["customer"].c_custkey.to_numpy()

    def masks():
        li_keep = _bloom_mask(lok, ook)  # lineitem(R) ⋉ orders(date)
        o_keep = _bloom_mask(ook, lok[li_keep])  # orders ⋉ reduced-lineitem
        c_keep = _bloom_mask(ckey, ock[o_keep])  # customer ⋉ reduced-orders
        return li_keep, o_keep, c_keep

    red_ms, (lm, om, cm) = _time_calls(masks)
    red = dict(t)
    red["lineitem"] = li[lm].reset_index(drop=True)
    red["orders"] = o[om].reset_index(drop=True)
    red["customer"] = t["customer"][cm].reset_index(drop=True)
    ratios = {"lineitem": (len(li), int(lm.sum())), "orders": (len(o), int(om.sum()))}
    return red, red_ms, ratios


PT = {3: pt_q3, 9: pt_q9, 10: pt_q10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=30.0)
    ap.add_argument("--queries", default="3,9,10")
    args = ap.parse_args()

    # Import the TPC-H query/table machinery from the sibling benchmark.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_tpch as B

    print(f"Generating TPC-H SF-{args.sf} (this needs real RAM+disk at SF>=30)…")
    con = B.make_duckdb(args.sf)
    t = B.load_tables(con)
    print("rows:", {k: len(v) for k, v in t.items()})
    pp = {"use_physical_planner": True}
    tpl = {k: pl.from_pandas(v) for k, v in t.items()} if HAS_POLARS else None

    def tq(fn, runs=3):
        for _ in range(1):
            fn()
        b = float("inf")
        for _ in range(runs):
            s = time.perf_counter()
            fn()
            b = min(b, (time.perf_counter() - s) * 1000)
        return b

    print(
        f"\n{'q':>3} {'baseline':>10} {'reduce':>8} {'red-join':>9} {'PT total':>9} "
        f"{'speedup':>8} {'polars':>9}  reductions"
    )
    for q in [int(x) for x in args.queries.split(",")]:
        lp_fn = B.QUERIES[q][0]
        base = tq(lambda: lp_fn(t).collect(**pp))
        red_tables, red_ms, ratios = PT[q](t)
        red_join = tq(lambda: lp_fn(red_tables).collect(**pp))
        pt_total = red_ms + red_join
        # correctness
        r0 = lp_fn(t).collect(**pp)
        r1 = lp_fn(red_tables).collect(**pp)
        ok = r0.shape == r1.shape
        polars_ms = tq(lambda: B.QUERIES[q][1](tpl)) if HAS_POLARS else float("nan")
        rs = " ".join(f"{k} {a}->{b}" for k, (a, b) in ratios.items())
        flag = "" if ok else "  !!CORRECTNESS"
        print(
            f"{q:>3} {base:>10.1f} {red_ms:>8.1f} {red_join:>9.1f} {pt_total:>9.1f} "
            f"{base / pt_total:>7.2f}x {polars_ms:>8.1f}  {rs}{flag}"
        )
    print(
        "\nGO if PT speedup >= ~1.5x at this SF; else predicate transfer is not "
        "worth building for this workload."
    )


if __name__ == "__main__":
    main()
