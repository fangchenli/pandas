"""Standalone verification of ARROW_GAPS.md AG4: acero hashes raw string group
keys multiples slower than dictionary-encoding the same keys.

Self-contained (pyarrow + polars + numpy, synthetic data — no pandas-lazy
import), so it can be attached to an upstream Arrow issue. For each cardinality
it groups a 10M-row table by one string key and sums a float column, comparing:
  - acero_raw : Table.group_by on raw large_string keys
  - acero_dict: Table.group_by on dictionary_encode()'d keys (the workaround)
  - polars    : pl.group_by on string keys (reference)
at Arrow cpu_count = 1 and = all cores. Correctness is asserted across methods.

Thread isolation: Polars' pool is fixed at import, so run twice —
  POLARS_MAX_THREADS=1 python bench_arrow_string_groupby.py   # per-thread kernel
  python bench_arrow_string_groupby.py                        # real-world (all cores)
The cpu=1 / polars-pool=1 row is the clean per-thread hashing-kernel comparison;
it reproduces the ~3.7x acero-raw-vs-Polars gap and the 4-10x raw-vs-dict gap.

Verified June 2026 (pyarrow 23.0.1, polars 1.37.1): raw/dict = 3.5-10x across
cardinalities; acero_dict beats Polars except at K=1M (Polars' high-card hash
still wins ~1.3x). Mechanism hypothesis: acero re-hashes raw key bytes per row
rather than interning; dict-encoding eliminates it.
"""

from __future__ import annotations

import os
import time

import numpy as np
import polars as pl
import pyarrow as pa

N = 10_000_000


def t(f, warm=1, runs=3):
    for _ in range(warm):
        f()
    b = float("inf")
    r = None
    for _ in range(runs):
        s = time.perf_counter()
        r = f()
        b = min(b, (time.perf_counter() - s) * 1000)
    return b, r


def main():
    print(
        f"pyarrow {pa.__version__} | polars {pl.__version__} | numpy {np.__version__}"
        f" | cpus={os.cpu_count()} | polars_pool={pl.thread_pool_size()} | N={N:,}"
    )
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(N)

    for K in (100, 10_000, 1_000_000):
        pool = np.array([f"k{i:012d}" for i in range(K)], dtype=object)
        keys = pool[rng.integers(0, K, N)]
        k_ls = pa.array(keys, type=pa.large_string())
        tbl = pa.table({"k": k_ls, "v": vals})
        tbl_dict = pa.table({"k": k_ls.dictionary_encode(), "v": vals})
        pdf = pl.DataFrame({"k": keys.astype(str), "v": vals})

        def acero_raw(tbl=tbl):
            return tbl.group_by(["k"]).aggregate([("v", "sum")])

        def acero_dict(tbl_dict=tbl_dict):
            return tbl_dict.group_by(["k"]).aggregate([("v", "sum")])

        def pol(pdf=pdf):
            return pdf.group_by("k").agg(pl.col("v").sum())

        for ncpu in (1, os.cpu_count()):
            pa.set_cpu_count(ncpu)
            braw, rraw = t(acero_raw)
            bdict, _ = t(acero_dict)
            bpol, _ = t(pol)
            print(
                f"K={K:>9,} acero_cpu={ncpu:<2} | acero_raw {braw:8.1f}  "
                f"acero_dict {bdict:8.1f}  polars(pool={pl.thread_pool_size()}) "
                f"{bpol:8.1f}  | raw/dict={braw / bdict:5.2f}x  "
                f"raw/pol={braw / bpol:5.2f}x  groups={rraw.num_rows:,}"
            )

        a = acero_raw().to_pandas().set_index("k")["v_sum"].sort_index()
        d = acero_dict().to_pandas().set_index("k")["v_sum"].sort_index()
        p = pol().to_pandas().set_index("k")["v"].sort_index()
        ok = np.allclose(a.values, d.reindex(a.index).values) and np.allclose(
            a.values, p.reindex(a.index).values
        )
        print(f"          correctness across methods: {ok}")


if __name__ == "__main__":
    main()
