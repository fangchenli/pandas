# JAX custom kernels for a hot path — measured NO-GO (June 2026)

Follow-up to the JAX/XLA decline in `ROADMAP.md` (§"declined June 2026").
That decline covered JAX as an **element-wise fusion backend** (loses to
NumExpr) and the **GPU transfer tax**. This probe covers the angle it did
**not**: can you *roll custom kernels* in JAX (Pallas / `jax.jit` codegen), and
does any of our actual hot paths — group-by, join, sort — benefit?

Answer: **no, on CPU.** XLA/Pallas categorically cannot express the dominant
hot-path cost (the factorize / hash-table build), and the one slice it *can*
express (segmented reduction) is at best break-even and is beaten ~2x by a
single-pass Cython scatter kernel. GPU (JAX's real payoff) is unavailable on
this Apple-Silicon box (`CpuDevice` only) and was already declined on the
transfer ceiling.

Measured with `jax 0.10.2` (CPU), `jax_enable_x64`, best-of-N warm (JIT compile
excluded). Scripts in scratchpad: `jax_hotpath_probe.py`,
`jax_multiagg_probe.py`, `jax_vs_acero.py`, `segsum_passes.py`.

## Can you roll custom kernels in JAX? Yes — three mechanisms

1. **`jax.jit`** — XLA auto-fuses dense array ops (element-wise chains,
   reductions, `segment_sum`/scatter) into one compiled kernel. Static shapes
   required (our fixed-size morsels neutralize this, as `ENGINE_DESIGN.md`
   notes).
2. **Pallas** — explicit block-level kernel DSL (Triton-like). Imports fine, but
   its real backends are **GPU (Triton/Mosaic) and TPU (Mosaic)**; CPU lowering
   is interpret-only/experimental. Aimed at dense tensor compute (attention,
   matmul), **not** hash tables or data-dependent shapes.
3. **`ffi` / `custom_call`** — wrap external C/CUDA. This is just FFI — no
   different from the Cython we already write, and buys nothing on CPU.

The blocker is not expressiveness of the DSL; it's that **XLA and Pallas are
built for static-shape dense linear algebra**, and our hot paths are
hash-table / gather-scatter / data-dependent-shape operations.

## Hot-path-by-hot-path (measured)

### A. Element-wise transcendental chain — NumExpr already wins
`sqrt(exp(a)+sin(b)*cos(c))`, N=10M:

| | ms |
|---|---|
| numpy (naive) | 215 |
| **numexpr** | **41** |
| jax.jit (data device-resident) | 56 |
| jax.jit (+ np↔jax conversion) | 67 |

JAX-CPU is **1.4–1.6x slower than NumExpr**, which the engine already routes
fuseable chains through (`backends/numexpr_fusion.py`). Lever already pulled —
confirms the prior decline directly.

### B. Group-by — the dominant cost is outside XLA entirely
The group-by hot path (q18 inner group, 18M rows → 4.5M groups) is
**factorize (hash-table build) → segmented reduction**. Our Polars profiling
(`Q18_DECOMP.md`) put the hash/group-build at ~57% and the sum at ~43%.

- **Factorize / hash build** — *not XLA- or Pallas-expressible* (dynamic output
  shape, hash table). JAX cannot touch the larger half of the cost. Full stop.
- **Segmented reduction given dense codes** — the *only* XLA-friendly slice.
  Single sum over 18M→4.5M:

  | | ms |
  |---|---|
  | np.bincount(weights) | 118 |
  | jax.jit segment_sum (device-resident) | 109 |
  | jax.jit segment_sum (+ conversion) | **138** |

  ~8% faster *only* if data is already device-resident; **17% slower** once you
  pay the numpy↔jax conversion a numpy-backed morsel must pay each call.

### C. JAX's strongest case — multi-agg fusion — is beaten by one Cython pass
Group-by computing sum+sumsq+count (≈ mean+var+count), 18M→4.5M, agg-only:

| | ms | passes |
|---|---|---|
| 1× bincount (floor: one scatter) | 119 | 1 |
| 3× bincount (numpy, no fusion) | 365 | 3 |
| **jax.jit fused (device-resident)** | **289** | ~2.4 |
| jax.jit fused (+ conversion) | 316 | ~2.4 |

XLA fuses the three reductions to ~2.4 scatter-equivalents (reads `v` once,
computes `v*v` inline) — a ~1.15–1.25x win **vs naive multi-pass numpy**. But
the bottleneck is the **scatter** (random writes into 4.5M groups), and XLA
still emits ~one scatter per output. A **single-pass Cython scatter kernel**
(one loop updating sum/sumsq/count accumulators per row — exactly the technique
already in `lazy_fused_agg.pyx`, which beats Polars on the *ungrouped* q6
filter+multi-agg: 10.8 ms vs 26 ms) would do **one** scatter ≈ 130–150 ms —
**~2x faster than JAX**, with no heavy GPU-oriented dependency and no conversion
tax. The fusion win JAX offers is real but **more cheaply captured in Cython**.

### Aside: acero is the wrong kernel for high-card int group-by
`pa.table.group_by(codes).aggregate([sum,mean,count])` (factorize **included**)
measured **1683 ms** — ~3x the factorize+bincount path (`Q18_DECOMP.md`: ~531
ms). Not a JAX comparison (acero does the factorize JAX can't), but a standing
reason our int group-by avoids acero for high cardinality.

## Conclusion

JAX *can* compile and roll custom kernels, but **no lazy-pandas hot path
benefits on CPU**:
1. Element-wise fusion — NumExpr already wins (JAX-CPU 1.4x slower).
2. The group-by's dominant half (factorize/hash) is categorically outside
   XLA/Pallas.
3. The segmented-reduction tail: single-agg is break-even-to-slower with the
   conversion tax; multi-agg's ~1.2x fusion win is beaten ~2x by a single-pass
   Cython scatter kernel we already know how to write.
4. Pallas targets GPU/TPU; CPU is experimental and still can't express hash
   tables. GPU offload needs device-*resident* data across many ops (the cuDF
   execution model), not a per-call kernel backend — already declined on the
   transfer ceiling.

This **extends and confirms** the ROADMAP JAX decline with the custom-kernel /
segmented-reduction angle. Revisit only under the same exception already on
record: a workload that is compute-bound *and* chains many ops on
device-resident data. The actionable takeaway is not JAX — it's that **if
grouped multi-agg ever becomes a bottleneck, a single-pass fused-scatter Cython
kernel (extend `lazy_fused_agg` to the grouped case) is the cheaper lever.**
