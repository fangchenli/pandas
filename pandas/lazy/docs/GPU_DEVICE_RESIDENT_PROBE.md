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

## Hardware (note: not the GH200 we intended)
RunPod provisioned **x86_64, 128 cores, 1.5 TiB RAM + NVIDIA RTX PRO 6000
Blackwell (sm_120), 96 GB VRAM, PCIe** (CUDA 12.8 toolkit / driver 580 / cudf
26.6.0, polars 1.39.3). This is the **PCIe** case (the transfer ceiling
applies), *not* unified memory — see the GH200 extrapolation at the end. The
128-core CPU is a strong baseline: Polars-CPU q18 @ SF-3 = 125 ms here vs ~281
ms on the dev Mac, so GPU is fighting ~2x our local CPU.

Method: `gpu_tpch_driver.py` (scratchpad) reuses `bench_tpch.pl_q*` (the Polars
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

### GH200 / unified-memory extrapolation (untested)
This was PCIe. On Grace-Hopper (NVLink-C2C coherent unified memory, ~900 GB/s,
~40x this PCIe path's measured 22 GB/s), the transfer half of the per-call cost
~vanishes, so the per-call and device-resident columns converge: the heavy
queries' 2–3x would hold *without* needing an explicit device-resident plan, and
several currently-losing mid-weight queries (q1, q7, q10, q11) would likely
cross to GPU. The light queries stay CPU-favored (compute too small regardless).
A GH200 rerun of `gpu_tpch_driver.py` would confirm; the harness is ready.

## Reproduce
On a CUDA GPU box: `pip install "polars[gpu]" cudf-cu12 pyarrow pandas duckdb`,
copy `bench_tpch.py` + `gpu_tpch_driver.py`, run
`python gpu_tpch_driver.py <SF> [q,q,...]`. q18 drill: `python
cudf_q18_decomp.py`. Scripts in the session scratchpad.
