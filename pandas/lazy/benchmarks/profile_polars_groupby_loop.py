#!/usr/bin/env python3
"""Tight loop over Polars's high-cardinality int group-by, for native profiling.

Used to profile WHERE Polars's in-memory group-by spends time (q18's inner
l_orderkey group: 18M rows -> 4.5M groups), to verify the mechanism claimed in
docs/Q18_DECOMP.md. Polars's runtime .so (_polars_runtime) ships WITH Rust
symbols (not stripped), so macOS `sample` resolves real frames.

Workflow (macOS):
    python profile_polars_groupby_loop.py > /tmp/pid.txt 2>&1 &
    PID=$!
    sample $PID 12 -file /tmp/pl_sample.txt
    wait $PID
then parse /tmp/pl_sample.txt for self-time per symbol.

Finding (polars 1.37.1, SF-3 scale): self-time is dominated by
`group_by_threaded_slice` (building per-group INDEX vectors in parallel
thread-local hashmaps) + `agg_sum`; the `take`/gather frame is ~1.5% — i.e.
Polars groups in place via index lists and does NOT physically gather the data,
unlike our partition_by_key -> per-bucket take -> arrow group_by -> concat path.
"""

import os
import time

import numpy as np
import polars as pl

rng = np.random.default_rng(0)
N_ORDERS = 4_500_000
TARGET = 18_000_000

reps = rng.integers(1, 8, N_ORDERS)
ok = np.repeat(np.arange(1, N_ORDERS + 1, dtype=np.int64), reps)
if len(ok) >= TARGET:
    ok = ok[:TARGET]
else:
    ok = np.concatenate([ok, rng.integers(1, N_ORDERS, TARGET - len(ok))])
rng.shuffle(ok)
qty = rng.random(len(ok)) * 50.0
df = pl.DataFrame({"l_orderkey": ok, "q": qty})

df.group_by("l_orderkey").agg(pl.col("q").sum())  # warm
print("PID", os.getpid(), "rows", len(ok), flush=True)

t_end = time.time() + 16
nit = 0
while time.time() < t_end:
    df.group_by("l_orderkey").agg(pl.col("q").sum())
    nit += 1
print("iters", nit, flush=True)
