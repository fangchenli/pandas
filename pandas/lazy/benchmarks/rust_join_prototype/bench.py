import time

from lazyjoin_rs import fused_join_gather
import numpy as np
import polars as pl

import pandas as pd


def best(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


# correctness
rng = np.random.default_rng(7)
nL, nR, P = 5000, 1200, 3
lk = rng.integers(0, nR, nL).astype(np.int64)
rk = np.arange(nR, dtype=np.int64)
cols = np.ascontiguousarray(rng.standard_normal((P, nR)))
out, lrow = fused_join_gather(lk, rk, cols, True)
ldf = pd.DataFrame({"key": lk, "_lrow": np.arange(nL)})
rdf = pd.DataFrame({"key": rk, **{f"p{j}": cols[j] for j in range(P)}})
m = pd.merge(ldf, rdf, on="key", how="inner")
ok = np.array_equal(lrow, m["_lrow"].to_numpy()) and all(
    np.allclose(out[j], m[f"p{j}"].to_numpy()) for j in range(P)
)
print("correctness:", ok)

N, M = 10_000_000, 2_500_000
lk = rng.integers(0, M, N).astype(np.int64)
rk = np.arange(M, dtype=np.int64)
print(f"\n{'P':>3} {'rust_unord':>10} {'rust_ord':>9} {'pdmerge':>9} {'polars':>9}")
for P in (1, 3, 9, 21):
    cols = np.ascontiguousarray(rng.standard_normal((P, M)))
    ru = best(lambda: fused_join_gather(lk, rk, cols, False))
    ro = best(lambda: fused_join_gather(lk, rk, cols, True))
    ldf = pd.DataFrame({"key": lk})
    rdf = pd.DataFrame({"key": rk, **{f"p{j}": cols[j] for j in range(P)}})
    pm = best(lambda: pd.merge(ldf, rdf, on="key", how="inner"))
    lpl = pl.DataFrame({"key": lk})
    rpl = pl.DataFrame({"key": rk, **{f"p{j}": cols[j] for j in range(P)}})
    pol = best(lambda: lpl.join(rpl, on="key", how="inner"))
    print(f"{P:>3} {ru:10.1f} {ro:9.1f} {pm:9.1f} {pol:9.1f}")
