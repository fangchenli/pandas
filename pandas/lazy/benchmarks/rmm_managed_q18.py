import time

import numpy as np
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(managed_memory=True, pool_allocator=True, initial_pool_size="8GB")
import cupy as cp

cp.cuda.set_allocator(rmm_cupy_allocator)
import cudf

NROWS = 180_000_000
norders = NROWS // 4
rng = np.random.default_rng(0)
ok = rng.integers(0, norders, size=NROWS).astype(np.int64)
q = rng.integers(1, 51, size=NROWS).astype(np.float64)


def sync():
    cp.cuda.runtime.deviceSynchronize()


def best(fn, n=5):
    fn()
    sync()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def percall():
    g = cudf.DataFrame({"k": ok, "q": q}).groupby("k", sort=False).agg({"q": "sum"})
    return g[g["q"] > 300]


m = best(percall)
out = percall()
print(
    f"RMM-managed per-call (df-from-numpy + group + HAVING): "
    f"{m:.1f} ms; survivors={len(out):,}"
)
