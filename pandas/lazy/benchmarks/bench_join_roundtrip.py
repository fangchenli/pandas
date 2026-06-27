"""Does the fused-kernel win survive the column<->row-major round-trip the
engine needs? Engine has column-major arrays and wants column-major out.
Fused kernel needs row-major in (transpose) and produces row-major out (split)."""

import time

import numpy as np
import polars as pl

import pandas as pd
from pandas.lazy.backends.numpy.join import partitioned_join_gather

N, M = 10_000_000, 2_500_000
rng = np.random.default_rng(0)
lk = rng.integers(0, M, N).astype(np.int64)
rk = np.arange(M, dtype=np.int64)


def best(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


print(f"{'P':>3} {'block-only':>11} {'+roundtrip':>11} {'pd.merge':>9} {'polars':>9}")
for P in (3, 9):
    cols = [
        rng.standard_normal(M) for _ in range(P)
    ]  # dim payload, column-major (engine form)

    def fused_block():  # what the microbench measured (block in/out)
        rm = np.ascontiguousarray(np.array(cols).T)
        return partitioned_join_gather(lk, rk, rm, preserve_order=False)

    def fused_roundtrip():  # honest: row-major in, split output back to columns
        rm = np.ascontiguousarray(np.array(cols).T)
        out, lrow = partitioned_join_gather(lk, rk, rm, preserve_order=False)
        out_cols = [
            np.ascontiguousarray(out[:, j]) for j in range(P)
        ]  # back to columns
        return out_cols

    bo = best(fused_block)
    rt = best(fused_roundtrip)
    ldf = pd.DataFrame({"key": lk})
    rdf = pd.DataFrame({"key": rk, **{f"p{j}": cols[j] for j in range(P)}})
    pm = best(lambda: pd.merge(ldf, rdf, on="key", how="inner"))
    lpl = pl.DataFrame({"key": lk})
    rpl = pl.DataFrame({"key": rk, **{f"p{j}": cols[j] for j in range(P)}})
    pol = best(lambda: lpl.join(rpl, on="key", how="inner"))
    print(f"{P:>3} {bo:11.1f} {rt:11.1f} {pm:9.1f} {pol:9.1f}")
