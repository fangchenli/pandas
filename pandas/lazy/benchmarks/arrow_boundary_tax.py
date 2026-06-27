"""Decompose the Arrow<->NumPy/pandas boundary tax by dtype, and test whether
it's a file-able Arrow inefficiency or an inherent dtype-model mismatch.

For each dtype: is Arrow->NumPy zero-copy? what does the unavoidable case cost?
does ArrowDtype-backed pandas (types_mapper) avoid it entirely?
"""

import time

import numpy as np
import pyarrow as pa

import pandas as pd

N = 10_000_000
rng = np.random.default_rng(0)


def best(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def zc(arr):
    """does to_numpy(zero_copy_only=True) succeed?"""
    try:
        arr.to_numpy(zero_copy_only=True)
        return "ZERO-COPY"
    except Exception as e:
        return f"COPY ({type(e).__name__})"


print(f"pyarrow {pa.__version__} | pandas {pd.__version__} | N={N:,}\n")

# --- single-column Arrow -> NumPy by dtype ---
f64 = pa.array(rng.standard_normal(N))  # no nulls, 1 chunk
i64 = pa.array(rng.integers(0, 1 << 60, N))
mask = rng.random(N) < 0.1
f64n = pa.array(rng.standard_normal(N), mask=mask)  # 10% nulls
i64n = pa.array(rng.integers(0, 1000, N), mask=mask)
slo = pa.array(rng.integers(0, 100, N).astype(str))  # low-card strings
shi = pa.array([f"k{i}" for i in rng.integers(0, 9_000_000, N)])  # high-card
# chunked float (8 chunks)
f64c = pa.chunked_array(np.array_split(rng.standard_normal(N), 8))

print("Arrow array -> NumPy (single column):")
for name, arr in [
    ("f64 (no null,1chunk)", f64),
    ("i64 (no null)", i64),
    ("f64 (10% null)", f64n),
    ("i64 (10% null)", i64n),
    ("string low-card", slo),
    ("string high-card", shi),
    ("f64 chunked x8", f64c),
]:
    t = best(lambda a=arr: a.to_numpy(zero_copy_only=False))
    print(f"  {name:24} to_numpy {t:8.1f} ms   [{zc(arr)}]")

# --- full table -> pandas: default vs ArrowDtype-backed (zero-copy escape) ---
tbl = pa.table({"a": f64, "b": i64, "c": f64n, "s": shi})
print("\nArrow table -> pandas (4 cols incl 1 string, 1 nullable):")
print(f"  to_pandas() default            : {best(lambda: tbl.to_pandas()):8.1f} ms")
print(
    f"  to_pandas(self_destruct,split) : "
    f"{best(lambda: tbl.to_pandas(self_destruct=True, split_blocks=True)):8.1f} ms"
)
print(
    f"  to_pandas(types_mapper=ArrowDt): "
    f"{best(lambda: tbl.to_pandas(types_mapper=pd.ArrowDtype)):8.1f} ms"
    f"  <- Arrow-backed, ~zero-copy"
)

# --- reverse: NumPy -> Arrow ---
na = rng.standard_normal(N)
sa = np.array([f"k{i}" for i in range(100)])[rng.integers(0, 100, N)]
print("\nNumPy -> Arrow:")
print(f"  pa.array(float64)              : {best(lambda: pa.array(na)):8.1f} ms")
print(f"  pa.array(object strings)       : {best(lambda: pa.array(sa)):8.1f} ms")
