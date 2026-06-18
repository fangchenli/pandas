# Predicate-transfer probe — the algorithm works, the SF-3 margin is thin

Probe (June 2026, SF-3) of the go/no-go memo's recommended lever: a
predicate-transfer / semijoin-reduction optimizer pass that reduces join inputs
before the join chain runs. Measurement-first, before building.

## What we measured

**The algorithm is highly effective at avoiding join work:**
- q3 (3 selective filters): exact semijoin reduction shrinks orders 2.18M→34K
  and lineitem 9.7M→90K (exactly the surviving rows). The **reduced join is
  23ms vs 196ms baseline — 8x on the join itself**.
- q9 (1 selective filter, part="green" 5.4%): reduces lineitem 18M→980K; reduced
  join 365ms vs 510ms baseline.

**But the reduction COST is the gating factor, and it's large relative to the
SF-3 baseline:**
- Mechanism matters enormously (lineitem 18M→980K semijoin):
  `isin(set)` 239ms · `pandas isin(ndarray)` 235ms · **`np.isin` 73ms** ·
  `searchsorted` 1739ms · **Bloom 1M-bit 93ms** (1.6% false positives, removed
  by the real join).
- Naive full PT on q3 (pandas isin + wide-DataFrame boolean-copy): **1767ms** —
  9x slower than baseline. The wide-DataFrame copy, not the isin, dominates.
- Engine-realistic full PT on q3 (Bloom filters, **key columns only, no copy**):
  **165ms** of reduction — vs a savings budget of only ~172ms (196 baseline −
  24 reduced join). **≈ break-even.**

## Verdict: real algorithm, scale-dependent margin — thin at SF-3

| query | mechanism | result at SF-3 |
|---|---|---|
| q9 | 1 cheap exact reduction (np.isin 73ms) | **~1.15x** (438 vs 510ms) |
| q3 | multi-pass Bloom reduction (165ms) | **~break-even** (≈190 vs 196ms) |

PT wins when **one cheap reduction eliminates a lot of downstream join work**
(q9). It only breaks even when the **reduction needs multiple passes whose cost
rivals the savings** (q3) — because at SF-3 the baseline is already small
(~200ms) and our join orders aren't catastrophically bad, so there isn't a big
explosion to avoid.

This is more sobering than the memo's "1.5–2.4x documented" — those wins are
**averaged / at larger scale / on worse join orders**. The mechanism lesson
also repeats the groupby-kernel story: right algorithm, and the win needs an
engine-integrated implementation (Bloom-filter bitmask applied during the
columnar scan on key columns), never pandas `isin` + wide-DataFrame copy.

## Why the margin should GROW with scale (the open question)

The reduction cost scales ~linearly with input rows, but the **work it avoids
grows super-linearly** when intermediates explode (bigger fact tables, worse
join blow-up). RPT/Yannakakis+ report their 1.5–2.4x at larger scales and over
*random* join orders (robustness). Our scorecard is SF-3 only because the laptop
OOMs at SF-10; we validated correctness to SF-300 on EC2, and large data is the
actual target. So PT is plausibly a real win at SF-30–300 even though it's
thin/break-even at SF-3.

## Recommendation: do NOT build it on the SF-3 evidence alone

- The SF-3 margin (q9 1.15x, q3 break-even) does **not** justify building a
  Bloom-filter PT subsystem + optimizer pass, and it **risks regressions** on
  queries where reduction overhead exceeds savings (exactly q3).
- **Decisive next measurement before any build:** run the q3/q9/q21 PT probe at
  **SF-30–100** (needs EC2 / more RAM+disk than this laptop). If PT shows ≥1.5x
  at scale, build the engine-integrated Bloom-filter version (key-column
  bitmask during scan + a forward/backward optimizer pass, RPT-style). If it
  stays ~1.15x, it isn't worth it.
- If built, gate it on a cost/cardinality estimate (only fire when the predicted
  reduction outweighs the multi-pass cost) — same discipline as the parallel
  groupby cardinality gate.

Net: the probe **de-risked the algorithm** (it genuinely avoids join work) but
**did not confirm a margin worth building at the scale we can currently
measure**. The honest next step is a scale-up measurement, not a build.
