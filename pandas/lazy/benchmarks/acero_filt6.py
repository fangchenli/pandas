import sys
import time

import numpy as np
import pyarrow as pa
from pyarrow import acero
import pyarrow.compute as pc

NCPU = int(sys.argv[1]) if len(sys.argv) > 1 else 8
pa.set_cpu_count(NCPU)
N = 10_000_000
rng = np.random.default_rng(0)
a = pa.array(rng.standard_normal(N))
b = pa.array(rng.standard_normal(N))
c = pa.array(rng.standard_normal(N))
tbl = pa.table({"a": a, "b": b, "c": c})


def best(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


# filt6: count rows where a>0 & b>0 & c>0  (~12.5% pass)
def acero_filt6():
    pred = (pc.field("a") > 0.0) & (pc.field("b") > 0.0) & (pc.field("c") > 0.0)
    decl = acero.Declaration.from_sequence(
        [
            acero.Declaration("table_source", acero.TableSourceNodeOptions(tbl)),
            acero.Declaration("filter", acero.FilterNodeOptions(pred)),
            acero.Declaration(
                "aggregate", acero.AggregateNodeOptions([("a", "sum", None, "s")])
            ),
        ]
    )
    return decl.to_table()


def raw_pc_filt6():
    mask = pc.and_(pc.and_(pc.greater(a, 0.0), pc.greater(b, 0.0)), pc.greater(c, 0.0))
    return pc.sum(pc.cast(mask, pa.int64()))


print(f"pyarrow {pa.__version__} | cpu={NCPU} | filt6: count(a>0 & b>0 & c>0)")
av = best(acero_filt6)
rv = best(raw_pc_filt6)
print(f"  acero filter->count : {av:7.1f} ms")
print(f"  raw_pc mask->sum    : {rv:7.1f} ms   (acero/raw = {av / rv:.1f}x)")
