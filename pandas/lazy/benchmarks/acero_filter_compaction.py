"""Isolate Acero's filter->reduce compaction tax.

Query: SELECT sum(v) WHERE pred, varying selectivity. If FilterNode physically
compacts (gathers) selected rows before the aggregate, its cost scales with the
number of rows passing the filter -- whereas a mask-based reduce (no gather) is
roughly flat. That divergence is the smoking gun.

Paths:
  acero      : source -> filter -> aggregate(sum)        [compacts per morsel]
  raw_pc     : pc.sum(pc.filter(v, mask))                [whole-array, gathers once]
  pc_masked  : pc.sum(pc.if_else(mask, v, 0.0))          [NO gather -- the floor]
  numpy_mask : np.where(mask, v, 0).sum()                [reference floor]
  polars     : pl.filter(pred).select(sum)               [peer]
"""

import time

import numpy as np
import polars as pl
import pyarrow as pa
from pyarrow import acero
import pyarrow.compute as pc

N = 10_000_000
rng = np.random.default_rng(0)
v = rng.standard_normal(N)
key = rng.random(N)  # threshold on this gives exact selectivity
va = pa.array(v)


def best(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def make_acero(thr):
    tbl = pa.table({"v": va, "k": pa.array(key)})
    decl = acero.Declaration.from_sequence(
        [
            acero.Declaration("table_source", acero.TableSourceNodeOptions(tbl)),
            acero.Declaration(
                "filter", acero.FilterNodeOptions(pc.less(pc.field("k"), thr))
            ),
            acero.Declaration(
                "aggregate", acero.AggregateNodeOptions([("v", "sum", None, "s")])
            ),
        ]
    )
    return lambda: decl.to_table()


pl_df = pl.DataFrame({"v": v, "k": key})

print(
    f"pyarrow {pa.__version__} | polars {pl.__version__} | "
    f"N={N:,} | cpus={pa.cpu_count()}"
)
print(
    f"{'sel':>6} {'acero':>9} {'raw_pc':>9} {'pc_mask':>9} "
    f"{'np_mask':>9} {'polars':>9}  {'acero/mask':>10}"
)
for sel in (0.01, 0.1, 0.5, 0.9):
    thr = sel
    mask = pc.less(pa.array(key), thr)
    nmask = key < thr
    acero_t = best(make_acero(thr))
    raw_pc = best(lambda: pc.sum(pc.filter(va, mask)))
    pc_masked = best(lambda: pc.sum(pc.if_else(mask, va, 0.0)))
    np_mask = best(lambda: np.where(nmask, v, 0.0).sum())
    pol = best(
        lambda: (
            pl_df.lazy().filter(pl.col("k") < thr).select(pl.col("v").sum()).collect()
        )
    )
    print(
        f"{sel:>6.2f} {acero_t:9.1f} {raw_pc:9.1f} {pc_masked:9.1f} "
        f"{np_mask:9.1f} {pol:9.1f}  {acero_t / pc_masked:>9.1f}x"
    )
