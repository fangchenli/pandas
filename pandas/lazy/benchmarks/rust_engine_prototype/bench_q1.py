"""TPC-H q1 executed end-to-end in the Rust/arrow-rs engine vs Polars vs our
Cython engine. Proves the Arrow-native, boundary-once architecture (see
../../docs/RUST_ENGINE_DIRECTION.md). Run from pandas/lazy/benchmarks/.
"""

from __future__ import annotations

import time

import bench_tpch as B
import lazy_engine_rs as E
import numpy as np
import pyarrow as pa

import pandas as pd


def _best(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def main(sf: float = 3.0):
    import polars as pl

    con = B.make_duckdb(sf)
    tables = B.load_tables(con)
    li = tables["lineitem"]
    sd_ns = li["l_shipdate"].values.astype("datetime64[ns]").view("int64")
    cutoff = int((pd.Timestamp("1998-12-01") - pd.Timedelta(days=90)).value)
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(sd_ns),
            pa.array(li["l_returnflag"], type=pa.string()),
            pa.array(li["l_linestatus"], type=pa.string()),
            pa.array(li["l_quantity"].to_numpy()),
            pa.array(li["l_extendedprice"].to_numpy()),
            pa.array(li["l_discount"].to_numpy()),
            pa.array(li["l_tax"].to_numpy()),
        ],
        names=["sd", "rf", "ls", "q", "p", "d", "t"],
    )
    df = (
        E.run_q1(batch, cutoff)
        .to_pandas()
        .sort_values(["l_returnflag", "l_linestatus"])
        .reset_index(drop=True)
    )
    ref = con.execute("PRAGMA tpch(1)").df()
    cols = ["sum_qty", "sum_base_price", "sum_disc_price", "sum_charge"]
    ok = (
        all(np.allclose(df[c], ref[c], rtol=1e-6) for c in cols)
        and (df["count_order"].to_numpy() == ref["count_order"].to_numpy()).all()
    )
    rust = _best(lambda: E.run_q1(batch, cutoff))
    lp = _best(lambda: B.lp_q1(tables).collect(use_physical_planner=True))
    pl_tables = {n: pl.from_pandas(d) for n, d in tables.items()}
    pol = _best(lambda: B.pl_q1(pl_tables))
    print(f"q1 @ SF-{sf}  correct={ok}")
    print(f"  RUST {rust:.1f} ms | polars {pol:.1f} ms | our-lazy {lp:.1f} ms")
    print(f"  rust/polars = {pol / rust:.2f}x   our-lazy/polars = {pol / lp:.2f}x")


if __name__ == "__main__":
    main()
