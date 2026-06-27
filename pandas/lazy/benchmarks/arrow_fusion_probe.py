"""Does any AVAILABLE Arrow path remove intermediate materialization in a
compute-function chain? Measure a deep chain vs the fused alternatives.

chain: sqrt(exp(a) + sin(b)*cos(c))  -- 6 kernels, ~5 intermediates.
"""

import time

import numpy as np
import pyarrow as pa
from pyarrow import acero
import pyarrow.compute as pc

try:
    import numexpr as ne

    HAVE_NE = True
except Exception:
    HAVE_NE = False

N = 10_000_000
rng = np.random.default_rng(0)
a = pa.array(rng.standard_normal(N))
b = pa.array(rng.standard_normal(N))
c = pa.array(rng.standard_normal(N))
na, nb, nc = a.to_numpy(), b.to_numpy(), c.to_numpy()
tbl = pa.table({"a": a, "b": b, "c": c})


def best(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def raw_pc():
    # whole-array compute: each call materializes a full 80MB intermediate
    return pc.sqrt(pc.add(pc.exp(a), pc.multiply(pc.sin(b), pc.cos(c))))


def acero_project():
    # Acero project: streams morsels; per-batch kernels (peak mem reduced)
    expr = pc.sqrt(
        pc.add(
            pc.exp(pc.field("a")),
            pc.multiply(pc.sin(pc.field("b")), pc.cos(pc.field("c"))),
        )
    )
    decl = acero.Declaration.from_sequence(
        [
            acero.Declaration("table_source", acero.TableSourceNodeOptions(tbl)),
            acero.Declaration("project", acero.ProjectNodeOptions([expr], ["r"])),
        ]
    )
    return decl.to_table()


def numpy_chain():
    return np.sqrt(np.exp(na) + np.sin(nb) * np.cos(nc))


def numexpr_chain():
    return ne.evaluate("sqrt(exp(na)+sin(nb)*cos(nc))")


print(f"pyarrow {pa.__version__} | N={N:,} | chain sqrt(exp(a)+sin(b)*cos(c))")
print(f"  raw pc (materializes each intermediate) : {best(raw_pc):7.1f} ms")
print(f"  acero project (morsel-streamed)         : {best(acero_project):7.1f} ms")
print(f"  numpy (materializes, ref)               : {best(numpy_chain):7.1f} ms")
if HAVE_NE:
    print(f"  numexpr (fused, no intermediates)       : {best(numexpr_chain):7.1f} ms")


# bandwidth-bound chain too (a+b*c): fusion buys less
def raw_pc2():
    return pc.add(a, pc.multiply(b, c))


def ne2():
    return ne.evaluate("a+b*c", {"a": na, "b": nb, "c": nc})


print(f"\n  simple a+b*c  raw pc                     : {best(raw_pc2):7.1f} ms")
if HAVE_NE:
    print(f"  simple a+b*c  numexpr (fused)           : {best(ne2):7.1f} ms")
