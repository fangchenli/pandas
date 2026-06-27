"""Benchmark the fused partitioned parallel join+gather kernel vs pd.merge and
Polars on the representative TPC-H fact->dim shape (unique build), sweeping
payload width. See pandas/lazy/docs/JOIN_KERNEL_REBUILD_PROBE.md.

    python bench_partitioned_join.py
"""

from __future__ import annotations

import time

import numpy as np

import pandas as pd
from pandas.lazy.backends.numpy.join import partitioned_join_gather


def _best(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def main():
    try:
        import polars as pl

        have_pl = True
    except ImportError:
        have_pl = False

    n, m = 10_000_000, 2_500_000
    rng = np.random.default_rng(0)
    lk = rng.integers(0, m, n).astype(np.int64)
    rk = np.arange(m, dtype=np.int64)
    print(f"fact={n:,} dim={m:,} (unique build) | fused kernel vs pd.merge vs polars")
    print(f"{'P':>3} {'fused_un':>9} {'fused_ord':>10} {'pd.merge':>9} {'polars':>9}")
    for p in (1, 3, 9, 21):
        cols = [rng.standard_normal(m) for _ in range(p)]
        rm = np.ascontiguousarray(np.array(cols).T)  # (m, P) row-major dim block
        fused_un = _best(
            lambda rm=rm: partitioned_join_gather(lk, rk, rm, preserve_order=False)
        )
        fused_ord = _best(
            lambda rm=rm: partitioned_join_gather(lk, rk, rm, preserve_order=True)
        )
        ldf = pd.DataFrame({"key": lk})
        rdf = pd.DataFrame({"key": rk, **{f"p{j}": cols[j] for j in range(p)}})
        pm = _best(lambda: pd.merge(ldf, rdf, on="key", how="inner"))
        if have_pl:
            lpl = pl.DataFrame({"key": lk})
            rpl = pl.DataFrame({"key": rk, **{f"p{j}": cols[j] for j in range(p)}})
            pol = _best(lambda: lpl.join(rpl, on="key", how="inner"))
        else:
            pol = float("nan")
        print(f"{p:>3} {fused_un:9.1f} {fused_ord:10.1f} {pm:9.1f} {pol:9.1f}")


if __name__ == "__main__":
    main()
