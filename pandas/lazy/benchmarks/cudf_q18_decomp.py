"""Decompose the q18 inner high-card group-by on GPU: transfer vs compute.

The Q18_DECOMP.md wall: group ALL lineitem by l_orderkey (180M rows @ SF-30 ->
~45M groups), sum l_quantity, keep sum>300. CPU-substrate-bound at ~306ms
(SF-3) / scales up. This isolates GPU H2D transfer vs the actual group-by
compute, and the device-resident ceiling (run the agg repeatedly without
re-transfer).
"""

import time

import cudf
import cupy as cp
import numpy as np

SF = 30
NROWS = 6_000_000 * SF  # lineitem rows at SF
# l_orderkey: ~ NROWS/4 distinct orders, repeated; l_quantity: 1..50
norders = NROWS // 4
rng = np.random.default_rng(0)
orderkey = rng.integers(0, norders, size=NROWS).astype(np.int64)
quantity = rng.integers(1, 51, size=NROWS).astype(np.float64)
print(f"SF-{SF}: {NROWS:,} rows, ~{norders:,} distinct orders")
nbytes = orderkey.nbytes + quantity.nbytes
print(f"working set: {nbytes / 1e9:.2f} GB (2 cols)")


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


# --- H2D transfer cost (host numpy -> device) ---
def h2d():
    d_k = cp.asarray(orderkey)
    d_q = cp.asarray(quantity)
    return d_k, d_q


h2d_ms = best(h2d)
print(
    f"\nH2D transfer (2 cols, {nbytes / 1e9:.2f} GB): {h2d_ms:.1f} ms "
    f"=> {nbytes / 1e9 / (h2d_ms / 1000):.0f} GB/s"
)

# --- device-resident group-by compute (data already on GPU) ---
d_k, d_q = h2d()
gdf = cudf.DataFrame({"k": d_k, "q": d_q})


def gpu_group_resident():
    g = gdf.groupby("k", sort=False).agg({"q": "sum"})
    return g[g["q"] > 300]


comp_ms = best(gpu_group_resident)
out = gpu_group_resident()
print(
    f"GPU group-by+HAVING (device-resident): {comp_ms:.1f} ms; survivors={len(out):,}"
)


# --- per-call (transfer + compute, the realistic per-query cost) ---
def gpu_group_percall():
    dk = cp.asarray(orderkey)
    dq = cp.asarray(quantity)
    g = cudf.DataFrame({"k": dk, "q": dq}).groupby("k", sort=False).agg({"q": "sum"})
    return g[g["q"] > 300]


percall_ms = best(gpu_group_percall)
print(f"GPU group-by+HAVING (per-call incl H2D): {percall_ms:.1f} ms")


# --- CPU reference (numpy bincount-based, host-resident) ---
def cpu_group():
    s = np.bincount(orderkey, weights=quantity, minlength=norders)
    return np.count_nonzero(s > 300)


cpu_ms = best(cpu_group, n=3)
print(f"\nCPU bincount group+HAVING (1 thread-ish): {cpu_ms:.1f} ms")
print(f"\n=> device-resident compute speedup vs CPU bincount: {cpu_ms / comp_ms:.1f}x")
print(f"=> per-call (with transfer) speedup: {cpu_ms / percall_ms:.1f}x")
print(f"=> transfer is {h2d_ms / percall_ms * 100:.0f}% of per-call GPU time")
