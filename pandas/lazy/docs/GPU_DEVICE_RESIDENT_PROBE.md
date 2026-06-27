# GPU device-resident ceiling — measured on real TPC-H (June 2026)

First *measured* GPU probe (the ROADMAP GPU decline was reasoned, not measured).
Runs the full TPC-H suite + drills the q18 wall on a real GPU, comparing
**Polars-CPU vs Polars-GPU (cudf-polars backend)** on identical query plans and
identical data, same box. Answers: does GPU crack our substrate-bound queries,
and is the per-call transfer ceiling real?

**Bottom line:** GPU does **not** win the suite against a strong multicore CPU
(per-query host→device transfer + light/transfer-bound queries + one pathology
sink it), **but it genuinely cracks the heavy aggregate/join queries — including
q18, our CPU-substrate wall — at 2–3x, and that advantage *grows* with scale.**
The win is gated by the **execution model**: ~half the GPU cost on heavy ops is
H2D transfer, so the payoff needs **device-resident** data across the plan, not
per-call offload from host-resident frames. This *confirms and quantifies* the
ROADMAP decline rather than overturning it.

**GH200 update (measured, June 2026):** on Grace-Hopper unified memory the
per-call transfer tax *is* eliminated at the cuDF-primitive level (q18 inner
group 268→46 ms with managed memory; pinned H2D 17x PCIe) — but the off-the-shelf
polars-GPU **suite is unchanged** (0.48x) because the engine doesn't route
ingestion through the C2C path. The hardware advantage is real but **stranded by
the software memory model**. See the measured GH200 section below.

## Hardware (note: not the GH200 we intended)
RunPod provisioned **x86_64, 128 cores, 1.5 TiB RAM + NVIDIA RTX PRO 6000
Blackwell (sm_120), 96 GB VRAM, PCIe** (CUDA 12.8 toolkit / driver 580 / cudf
26.6.0, polars 1.39.3). This is the **PCIe** case (the transfer ceiling
applies), *not* unified memory — see the GH200 extrapolation at the end. The
128-core CPU is a strong baseline: Polars-CPU q18 @ SF-3 = 125 ms here vs ~281
ms on the dev Mac, so GPU is fighting ~2x our local CPU.

Method: `gpu_tpch_driver.py` (`pandas/lazy/benchmarks/`) reuses `bench_tpch.pl_q*` (the Polars
queries), times each `.collect()` CPU vs GPU, **best-of-5 warm**. GPU uses
`pl.GPUEngine(raise_on_fail=True)` so silent CPU fallbacks are detected — **all
22 ran GPU-native at both scales, zero fallbacks.** `pandas->polars` conversion
is one-time and excluded; H2D transfer per collect **is** included (the honest
per-query cost from host-resident data).

## Suite results (speedup = CPU ms / GPU ms; >1 = GPU faster)

| query | SF-3 | SF-30 | | query | SF-3 | SF-30 |
|---|---|---|---|---|---|---|
| q1  | 0.72 | 0.82 | | q12 | **1.69** | **1.75** |
| q2  | 0.74 | **1.60** | | q13 | **2.69** | 0.51 |
| q3  | 0.40 | 0.44 | | q14 | 0.30 | 0.33 |
| q4  | 0.60 | 0.48 | | q15 | 0.22 | 0.27 |
| q5  | 0.34 | 0.39 | | q16 | 0.25 | 0.29 |
| q6  | 0.35 | 0.40 | | q17 | 0.82 | **2.06** |
| q7  | 0.93 | 0.72 | | **q18** | **1.31** | **2.02** |
| q8  | 0.17 | 0.25 | | q19 | 0.73 | **0.13** |
| q9  | **1.12** | **1.35** | | q20 | 0.82 | **1.20** |
| q10 | 0.50 | 0.59 | | q21 | **1.24** | 0.78 |
| q11 | 0.28 | 0.74 | | q22 | 0.72 | **3.02** |

Suite: **SF-3 = 0.81x, SF-30 = 0.56x** (geo-mean 0.61 / 0.67). GPU loses the
suite at both scales, and the SF-30 suite is *worse* — dragged by q19's
pathology (0.13x, 14.5 s GPU vs 1.95 s CPU; complex OR-predicate query) and the
many light transfer-bound queries (q8/q14/q15/q16 all <0.35x).

### The signal under the suite number
Sort by what changed SF-3 → SF-30: the **heavy aggregate/join queries move
decisively toward the GPU and several flip from loss to win** — q17
0.82→**2.06**, q22 0.72→**3.02**, q2 0.74→**1.60**, q20 0.82→**1.20**, q18
1.31→**2.02**, q9 1.12→1.35, q12 1.69→1.75. The transfer-bound light queries
stay losing (compute too small to amortize H2D). So GPU value is **per-query and
scale-dependent**: it pays when compute-per-byte is high and the data is large.
(q13 and q21 regressed at scale — GPU hash-join/aggregate cost is not monotone;
worth a look but not central.)

## q18 wall decomposition — transfer vs compute (direct cudf)
The `Q18_DECOMP.md` wall = group all lineitem by `l_orderkey` (180M→45M groups @
SF-30), sum quantity, keep `sum>300`. Isolated with `cudf_q18_decomp.py`:

| stage | ms | vs CPU |
|---|---|---|
| H2D transfer (2.88 GB, 2 cols, pageable) | 131 | — (22 GB/s) |
| GPU group-by + HAVING (**device-resident**) | 109 | **7.8x** |
| GPU group-by + HAVING (**per-call, incl H2D**) | 246 | **3.5x** |
| CPU `np.bincount` group + HAVING (single-thread) | 849 | 1x |

**Transfer = 53% of the per-call GPU time.** Eliminating it (device-resident
across the plan) would ~2x the GPU advantage. Caveats: the bincount baseline is
single-threaded (the fair end-to-end CPU number is Polars-128c, where full-q18
GPU is the 2.02x above); transfer used **pageable** host memory — pinned +
async would raise the 22 GB/s and shrink the 53%.

## Conclusions

1. **Our q18 wall is a CPU-substrate wall, not fundamental.** The high-card int
   group-by we found substrate-bound on CPU (`Q18_DECOMP.md`) runs 2x faster
   than 128-core Polars on GPU at SF-30, growing with scale, and 7.8x a
   single-thread CPU device-resident. GPU hash-aggregation is the textbook win
   and it shows.
2. **But GPU is not a blanket win** against a strong multicore CPU: the suite is
   0.56–0.81x. Light/transfer-bound queries and pathologies (q19) lose, and
   per-call H2D from host-resident frames is ~half the cost of even the heavy
   wins.
3. **The lever is the execution model, not a kernel** — exactly the ROADMAP
   "device-resident sub-pipeline" exception, now measured: GPU pays off only for
   a *device-resident* sub-plan of heavy aggregate/join ops on large data
   (cuDF's model), never per-chain offload. This is consistent with — and a
   measured instance of — the JAX GPU decline (`JAX_CUSTOM_KERNEL_PROBE.md`,
   `ROADMAP.md`).
4. **For lazy-pandas specifically:** a GPU backend would have to (a) keep a
   sub-plan device-resident to beat transfer, (b) route only heavy/large
   aggregate-join queries to it (a cost-model decision), and (c) fall back
   honestly elsewhere — i.e. cuDF's architecture, a different engine, not a
   kernel plugged into the morsel loop. The from-scratch heterogeneous engine is
   already NO-GO (`ENGINE_GONOGO_MEMO.md`); the actionable finding is the
   **quantified ceiling** (2–3x on heavy queries, device-residency worth ~2x of
   that, transfer ~half the per-call cost), not a build.

## GH200 / unified memory — MEASURED (June 2026): real, but stranded by the stack
Reran on a **Lambda GH200 480GB** (Grace 64-core Neoverse-V2 aarch64 + Hopper
H200 96 GB HBM3e, NVLink-C2C, `Addressing Mode: ATS`; same cudf 26.6.0 / polars
1.39.3). The PCIe probe predicted the transfer tax would ~vanish. It does at the
**primitive** level — but the off-the-shelf engine does **not** capture it.

**Memory-path bandwidth (2.88 GB, `mem_paths.py`) — the C2C link is real:**

| H2D path | time | bandwidth | vs PCIe (22 GB/s) |
|---|---|---|---|
| pageable `cp.asarray` (the *default* cuDF ingestion) | 212 ms | **14 GB/s** | **0.6x — worse!** |
| pinned host → device | 7.6 ms | **381 GB/s** | **17x** |
| managed, prefetched + resident read | 1.9 ms | **1489 GB/s** | — (HBM-bound) |

**q18 inner group, per-call (`cudf_q18_decomp.py` / `rmm_managed_q18.py`):**

| | ms |
|---|---|
| device-resident compute (Hopper HBM3e) | **45** (vs Blackwell 109 = 2.4x) |
| per-call, **default pageable** | 268 (transfer = 82%) |
| per-call, **RMM managed memory** | **46** — *equals* device-resident |

So at the cuDF-primitive level the transfer tax is **eliminated** by managed
memory (268→46 ms): the GPU reaches host data over C2C on-demand instead of a
slow pageable bulk copy. The PCIe extrapolation is **confirmed**.

**But the full polars-GPU suite is UNCHANGED by managed memory** (SF-30, default
vs `rmm.reinitialize(managed_memory=True)` A/B on the same box):

| | suite | geo-mean | q18 GPU | q1 GPU (pure transfer-heavy) |
|---|---|---|---|---|
| default (pageable) | 0.48x | 0.46x | 1268 ms | 1331 ms |
| managed (C2C) | 0.49x | 0.46x | 1268 ms | 1325 ms |

Identical to the millisecond. The transfer-heavy q1 (re-ingests all lineitem
each collect) didn't move — **cudf-polars does not route its host-frame
ingestion through the managed/C2C path**, so the 17x link sits idle. Off-the-
shelf polars-GPU on GH200 still *loses* the suite (0.48x — even worse than PCIe
Blackwell's 0.56x, because the Grace CPU baseline is strong and the default
pageable copy on Grace is slower than PCIe). Heavy queries still win
(q13 5.16x, q12 1.96x, q22 1.25x) on compute, not transfer.

**The finding (substrate-class):** GH200's unified memory *hardware* delivers a
17x ingestion speedup that fully erases the per-call transfer tax — but it is
**stranded by the software memory model**. Capturing it needs an engine that
allocates host buffers as pinned/managed end-to-end (a memory-model-aware
data-ingestion path), not just running existing tooling on better hardware. This
**reinforces** the central conclusion: the lever is the *execution + memory
model*, not a kernel and not merely the hardware. Even on ideal silicon, the gap
is in the stack. (A genuine instrument finding: the substrate improved 17x and
the engine couldn't see it.)

## Reproduce
On a CUDA GPU box: `pip install "polars[gpu]" cudf-cu12 pyarrow pandas duckdb`,
copy `bench_tpch.py` + `gpu_tpch_driver.py`, run
`python gpu_tpch_driver.py <SF> [q,q,...]`. q18 drill: `python
cudf_q18_decomp.py`. GH200 memory paths: `mem_paths.py`; managed A/B:
`rmm_managed_q18.py` and `run_managed.py <SF>` (wraps the driver with an RMM
managed pool). Scripts in `pandas/lazy/benchmarks/`.

## Reproduce
On a CUDA GPU box: `pip install "polars[gpu]" cudf-cu12 pyarrow pandas duckdb`,
copy `bench_tpch.py` + `gpu_tpch_driver.py`, run
`python gpu_tpch_driver.py <SF> [q,q,...]`. q18 drill: `python
cudf_q18_decomp.py`. Scripts in the session scratchpad.
