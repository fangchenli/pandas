"""TPC-H device-resident GPU ceiling probe: Polars-CPU vs Polars-GPU (cudf).

Same query plans (bench_tpch.pl_q*), same data, same box. GPU uses
raise_on_fail=True so silent CPU fallbacks are detected, not hidden. Reports
per-query CPU ms, GPU ms, speedup, and GPU status (GPU / FALLBACK).
"""

import math
import sys
import time
import types

# stub pandas.lazy so bench_tpch imports without our built editable pandas
_m = types.ModuleType("pandas.lazy")
_m.col = _m.lit = _m.when = lambda *a, **k: None
sys.modules["pandas.lazy"] = _m

sys.path.insert(0, "/root")
import bench_tpch as B
import polars as pl

SF = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
QUERIES = (
    [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else sorted(B.QUERIES)
)
REPEAT = 5

t0 = time.perf_counter()
con = B.make_duckdb(SF)
tables = B.load_tables(con)
gen_s = time.perf_counter() - t0
tconv = time.perf_counter()
pl_tables = {n: pl.from_pandas(df) for n, df in tables.items()}
conv_ms = (time.perf_counter() - tconv) * 1000
print(
    f"[SF-{SF}: generated in {gen_s:.1f}s; lineitem={len(tables['lineitem']):,} rows; "
    f"pandas->polars conv (one-time) {conv_ms:.0f} ms]"
)
print(f"{'query':>6} {'CPU ms':>9} {'GPU ms':>9} {'speedup':>8}  status")
print("-" * 50)

_orig_collect = pl.LazyFrame.collect
GPU = pl.GPUEngine(raise_on_fail=True)


def gpu_collect(self, *a, **k):
    k["engine"] = GPU
    return _orig_collect(self, *a, **k)


def best(fn, n=REPEAT):
    fn()  # warm
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


rows = []
for q in QUERIES:
    if q not in B.QUERIES:
        continue
    _, pl_fn = B.QUERIES[q]
    try:
        cpu = best(lambda: pl_fn(pl_tables))
    except Exception as e:
        print(f"q{q:>5} {'CPU-ERR':>9}  {type(e).__name__}: {str(e)[:40]}")
        continue
    pl.LazyFrame.collect = gpu_collect
    status = "GPU"
    try:
        gpu = best(lambda: pl_fn(pl_tables))
    except pl.exceptions.ComputeError as e:
        gpu = float("nan")
        status = "FALLBACK"
        msg = str(e).replace("\n", " ")[:50]
    except Exception as e:
        gpu = float("nan")
        status = f"ERR:{type(e).__name__}"
        msg = str(e).replace("\n", " ")[:50]
    finally:
        pl.LazyFrame.collect = _orig_collect
    spd = (cpu / gpu) if gpu == gpu else float("nan")
    rows.append((q, cpu, gpu, spd, status))
    extra = "" if status == "GPU" else f"  ({msg})"
    print(f"q{q:>5} {cpu:9.1f} {gpu:9.1f} {spd:8.2f}  {status}{extra}")

print("-" * 50)
gpu_ok = [r for r in rows if r[4] == "GPU"]
if gpu_ok:
    geo = math.exp(sum(math.log(r[3]) for r in gpu_ok) / len(gpu_ok))
    tcpu = sum(r[1] for r in gpu_ok)
    tgpu = sum(r[2] for r in gpu_ok)
    print(
        f"GPU-native queries: {len(gpu_ok)}/{len(rows)} | "
        f"geo-mean speedup {geo:.2f}x | "
        f"suite CPU {tcpu:.0f} ms vs GPU {tgpu:.0f} ms ({tcpu / tgpu:.2f}x)"
    )
