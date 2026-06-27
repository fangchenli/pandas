import time

import cupy as cp
import numpy as np

N = 360_000_000  # 2.88 GB float64
host = np.random.default_rng(0).random(N)
nb = host.nbytes


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


# A. pageable cp.asarray
def pageable():
    return cp.asarray(host)


a = best(pageable)
print(f"A pageable cp.asarray : {a:7.1f} ms  {nb / 1e9 / (a / 1000):6.0f} GB/s")
# B. pinned host -> device
pin = cp.cuda.alloc_pinned_memory(nb)
hp = np.frombuffer(pin, dtype=np.float64, count=N)
hp[:] = host
dst = cp.empty(N, dtype=np.float64)


def pinned():
    dst.set(hp)


b = best(pinned)
print(f"B pinned set         : {b:7.1f} ms  {nb / 1e9 / (b / 1000):6.0f} GB/s")
# C. managed (cudaMallocManaged) - prefetch to GPU then sum (C2C migration)
mm = cp.cuda.malloc_managed(nb)
dm = cp.ndarray(
    N,
    dtype=np.float64,
    memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(mm.ptr, nb, mm), 0),
)
dm[:] = cp.asarray(host)  # init


def managed_sum():
    cp.cuda.runtime.memPrefetchAsync(mm.ptr, nb, 0, 0)  # prefetch to GPU0
    return float(dm.sum())


try:
    c = best(managed_sum)
    print(f"C managed prefetch+sum: {c:7.1f} ms  {nb / 1e9 / (c / 1000):6.0f} GB/s")
except Exception as e:
    print("C managed:", type(e).__name__, str(e)[:60])
