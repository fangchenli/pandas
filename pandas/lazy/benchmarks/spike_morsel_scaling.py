"""M3 spike: morsel-driven scaling under GIL vs free-threaded Python.

Simulates the ENGINE_DESIGN.md execution model faithfully:
- workers self-dispatch by claiming morsel indices from a shared counter
  (the passive lock-free dispatcher pattern)
- each worker runs a whole pipeline (filter mask -> project compute ->
  gather) on its morsel, with realistic Python orchestration per step
  (dict handling, dispatch indirection)
- kernels are NumPy (release the GIL for large arrays)

Reports throughput scaling at 1/2/4/8 threads plus the per-morsel
orchestration overhead, on whichever interpreter runs it.
"""

import itertools
import sys
import threading
import time

import numpy as np

N_ROWS = 16_000_000
MORSEL = 131_072  # 128K, the design default


def make_data(n):
    rng = np.random.default_rng(42)
    return {
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.integers(0, 1000, n).astype(np.int64),
    }


def run_pipeline_on_morsel(data, start, end):
    """Filter(a>0) -> Project(d=a+b, c) -> gather. Mirrors fused exec."""
    # orchestration: dict slicing per morsel (the Python-level cost)
    morsel = {k: v[start:end] for k, v in data.items()}
    # kernel 1: filter mask
    mask = morsel["a"] > 0.0
    indices = np.flatnonzero(mask)
    # kernel 2: project compute
    d = morsel["a"] + morsel["b"]
    # kernel 3: gather (backend-preserving take)
    out = {
        "d": np.take(d, indices),
        "c": np.take(morsel["c"], indices),
    }
    return len(indices), out


def run_threads(data, n_threads):
    n_morsels = (N_ROWS + MORSEL - 1) // MORSEL
    claim = itertools.count()  # atomic under GIL; thread-safe enough for spike
    results = [None] * n_morsels
    lock = threading.Lock()  # only to make claim safe on free-threaded

    def worker():
        while True:
            with lock:
                i = next(claim)
            if i >= n_morsels:
                return
            start = i * MORSEL
            end = min(start + MORSEL, N_ROWS)
            results[i] = run_pipeline_on_morsel(data, start, end)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    total = sum(r[0] for r in results if r)
    return elapsed, total


def measure_orchestration_overhead(data):
    """Time the Python-only portion (dict slicing, no kernels) per morsel."""
    n_morsels = N_ROWS // MORSEL
    t0 = time.perf_counter()
    for i in range(n_morsels):
        start = i * MORSEL
        _ = {k: v[start : start + MORSEL] for k, v in data.items()}
    per_morsel = (time.perf_counter() - t0) / n_morsels
    return per_morsel


def main():
    gil = "GIL" if getattr(sys, "_is_gil_enabled", lambda: True)() else "FREE-THREADED"
    print(f"Python {sys.version.split()[0]} [{gil}]  numpy {np.__version__}")
    print(f"rows={N_ROWS:,}  morsel={MORSEL:,}  morsels={N_ROWS // MORSEL}")

    data = make_data(N_ROWS)
    overhead = measure_orchestration_overhead(data)
    print(f"orchestration (dict slice) per morsel: {overhead * 1e6:.0f} us")

    # warmup
    run_threads(data, 1)
    base = None
    print(f"{'threads':>8} {'time(ms)':>10} {'speedup':>8}")
    for nt in (1, 2, 4, 8):
        best = min(run_threads(data, nt)[0] for _ in range(3))
        if base is None:
            base = best
        print(f"{nt:>8} {best * 1e3:>10.0f} {base / best:>7.2f}x")


if __name__ == "__main__":
    main()


# Compute-dense variant: per-morsel argsort (CPU-bound, low bandwidth).
# Run main() above for the bandwidth-bound pipeline; run compute_dense()
# for this one. June 2026 results (Apple Silicon 4P+4E):
#   bandwidth-bound: GIL 2.85x @4t / free-threaded 3.06x @4t (identical)
#   compute-dense:   GIL 5.47x @8t / free-threaded 5.55x @8t (identical)
# Conclusion: NumPy kernels release the GIL completely enough that
# GIL-threaded morsel execution matches free-threaded Python; the
# ceilings are memory bandwidth and core asymmetry, not the GIL.
def compute_dense():
    import itertools
    import threading

    n_rows, morsel = 16_000_000, MORSEL
    data = np.random.default_rng(42).standard_normal(n_rows)
    n_morsels = n_rows // morsel

    def run(n_threads):
        claim = itertools.count()
        lock = threading.Lock()

        def worker():
            while True:
                with lock:
                    i = next(claim)
                if i >= n_morsels:
                    return
                s = i * morsel
                np.argsort(data[s : s + morsel], kind="stable")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return time.perf_counter() - t0

    run(1)
    base = None
    for nt in (1, 2, 4, 8):
        best = min(run(nt) for _ in range(3))
        if base is None:
            base = best
        print(f"{nt:>3} threads: {best * 1e3:7.0f} ms  {base / best:5.2f}x")


if __name__ == "__main__" and "--compute" in sys.argv:
    compute_dense()
