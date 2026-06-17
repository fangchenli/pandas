# Design seed: closing the join/aggregate-heavy query gap (#1)

Seed for a deliberate design session (written June 2026, after the
plumbing-harvest session). Read this first; it corrects the framing and
fixes the starting facts so the design does not begin from a wrong premise.

## The honest standing (regenerated this session)

- **SF-3 TPC-H S1 geo-mean ~0.42x** (run-to-run 0.42–0.45x; machine was
  loaded). q6 is the lone outright win (1.18x). The gap is **concentrated in
  the join/aggregate-heavy queries**: q20 0.16x, q21 0.27x, q10 0.30x,
  q8 0.34x, q15 0.35x, q3 0.38x, q5/q9 0.39x. The small-query categories are
  at/above parity after this session.
- This session's wins (filter→scalar-agg fusion 1.4–2.4x, narrow-join kernel,
  optimizer dispatch −10% fixed overhead) are **real and recorded but do NOT
  move the TPC-H geo-mean** — they target shapes that aren't the hot path.
  The geo-mean is gated by the join/agg queries above. That is what #1 is.

## CRITICAL framing correction

#1 is **NOT** a from-scratch "build fusion." The engine is already
morsel-driven and mature — see ENGINE_DESIGN.md milestones M1–M6 (all
landed):
- M1 pipeline compiler (Morsel/Pipeline/Sink/PipelineGraph in `engine/`).
- M2 decision layer (`engine/decisions.py` + cost model `cost.py`).
- M3 morsel parallelism — **measured law**: numeric elementwise chains LOSE
  to parallelism (single thread saturates memory bandwidth ~0.5x); only
  compute-bound kernels (string/multi-op) win. Parallelism is gated on a
  kernel-class score.
- M4 parallel radix sort (`lazy_radix.pyx`), category-key groupby 15x.
- M5 **parallel join already routes to acero** when the join feeds an
  order-insensitive sink (groupby/sort/topk/distinct) with acero-safe keys
  and no index observation; `collect(order="relaxed")` widens this to
  terminal joins. join→groupby 10M×1M 1319→925ms (0.42→0.60x).
- M6 scan-native morsels (streaming, early limit termination).
- Plus: cardinality estimation, free-threading validated.

So the engine is not a naive operator-at-a-time materializer. #1 is
**locate and close the RESIDUAL** join/agg-query gap in this mature engine.

## What we already disproved this session (do NOT repeat)

- **Acero as a general join substrate**: its win dies at the Arrow→NumPy
  round-trip (1044ms ≈ pd.merge on the 10M exploding join). Only wins if
  results stay Arrow — which M5 already exploits for order-free sinks.
- **"Arrow-across-join" end-to-end on a real query**: probed on q3
  (`probe_arrow_across_join.py`) — acero end-to-end is 2x SLOWER than our
  pd.merge engine (127 vs 65ms SF-1) because acero's per-join-node overhead
  dominates at realistic filtered scale. Polars' q3 win is whole-query fused
  parallel execution, not a borrowable join kernel.

## The design question (sharpened)

Given M1–M6, **where does q21 (0.27x) / q20 (0.16x) / q3 (0.38x) actually
lose** vs Polars? Candidate mechanisms, to be measured per-operator before
any build:
1. **Materialization between operators** — does join→groupby→sort
   materialize full intermediates the way Polars fuses them? (We have
   morsels, but do multi-operator chains across a *pipeline breaker* like
   join/aggregate re-materialize?)
2. **Join routing** — are these joins actually reaching acero (M5), or
   falling to single-threaded pd.merge because of the eager-order contract
   or non-acero-safe keys? q21/q20 have semi/anti joins and nested
   structure — check what the decision layer picks.
3. **Per-operator parallelism gaps** — which operators in q21 run
   single-threaded? (groupby? the nested subqueries?)
4. **Optimizer plan quality** — is q20/q21's plan shape (join order, pushdown)
   as tight as DuckDB/Polars? Cardinality-driven join reorder exists; is it
   firing well on these?

## Proposed first step (the design session opens with measurement)

Per-operator decomposition of **q21 and q20** (the two worst) in the CURRENT
engine: instrument or `explain(physical=True)` + time each stage, identify
the single dominant stage, and compare that stage in isolation to Polars'
equivalent. Only after the gap is localized do we choose the mechanism
(cross-breaker fusion vs routing fix vs a parallel operator vs a plan fix).
This mirrors the discipline that made the plumbing wins land and kept us from
building the wrong thing twice (Acero, arrow-across-join).

## Standing constraints (from memory + this project)

- Measurement-first; controlled on/off; never ship unexplained regressions;
  every change validated by the full lazy suite AND all-22 TPC-H vs DuckDB.
- Output must stay eager-pandas-equivalent (row order, dtypes) unless
  `order="relaxed"`. The eager-order contract is the main thing blocking
  wider acero routing — a recurring tension worth revisiting in the design.
- PDEP deferred; findings are the asset. This is engineering, not advocacy.
