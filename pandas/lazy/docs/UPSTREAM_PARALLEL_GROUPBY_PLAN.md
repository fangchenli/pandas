# Upstream plan: Arrow parallel hash-aggregate gap (PLAN — do NOT file yet)

The first probe-output candidate (`PROBE_CHARTER.md` gap A). This is a **plan**:
it defines the claim, the evidence, the **verification gates that must pass
before we open anything upstream**, the reproducible benchmark, and the framing.
Nothing gets filed until the gates below are green — the claim may be too broad
as stated, and filing a sloppy or duplicate report would burn credibility.

## The claim (as currently measured, SF-3, pyarrow ~23/24)

Arrow's `Table.group_by(...).aggregate(...)` does **not scale with cores** on a
high-cardinality grouped aggregate: 124ms @ cpu_count=1 → 133ms @ 8 (i.e. no
speedup; `use_threads=False` was marginally *faster*). Its *serial* algorithm is
strong (2.4x faster than Polars' single-thread, ~132 vs ~295ms), but Polars wins
(~95-110ms) purely by parallelizing. Wrapping Arrow's group_by in a hash
partition (each group lands wholly in one bucket) + a thread pool + concat hits
~52ms — **~2.5x over single Arrow and ~1.9x over Polars**, bit-exact. So the gap
is "no parallel hash-aggregate," and there is measured headroom.

## ⚠️ Verification gates — ALL must pass before filing

These are the ways the claim could be **wrong, too broad, or already known**.
Resolve every one first.

1. **Acero morsel-parallelism nuance (HIGHEST RISK).** Acero parallelizes across
   *morsels/batches*, and a single in-memory `Table` may be handed to the
   hash-aggregate as effectively one unit → single-threaded. The real question
   is whether a **proper Acero streaming plan** (`Declaration` with a
   multi-batch source / a Table sliced into many RecordBatches) and
   `use_threads=True` **does** parallelize the hash-aggregate. **Test this
   before claiming "Arrow can't parallelize group-by."** Likely outcomes:
   - If multi-batch Acero parallelizes → the gap narrows to *"`Table.group_by`
     convenience / single-batch input isn't split into morsels"* — still real
     and worth reporting, but a **different, weaker claim** (usability/perf
     footgun, not a missing capability).
   - If even multi-batch Acero hash-aggregate stays single-threaded → the strong
     claim holds. Either way, the framing depends on this result.
2. **Latest Arrow.** Re-run on the **current** Arrow C++ / pyarrow release (we
   measured on 23/24). Confirm the no-scaling behavior still reproduces; a recent
   release may already parallelize.
3. **Duplicate search.** Search Arrow GitHub issues + the `dev@arrow.apache.org`
   archives + JIRA-migrated issues for existing reports/discussions on parallel
   hash aggregation / Acero aggregate threading. Do **not** duplicate; if it
   exists, attach our benchmark as a data point instead.
4. **Partition-approach caveats hold up.** Confirm our partition-parallel result
   (a) is bit-exact across dtypes and agg funcs (sum/mean/min/max/count), (b)
   reproduces standalone (below), and (c) that we're honestly reporting the
   partition cost (the nogil scatter) and the output-order change, not hiding
   them.

## Reproducible benchmark (standalone, no pandas-lazy dependency)

A self-contained script Arrow devs can run, demonstrating (i) `group_by`
no-scaling vs `cpu_count`, (ii) the proper-Acero-streaming result from gate 1,
and (iii) the partition-parallel speedup. Requirements:
- Pure pyarrow + numpy (no pandas-lazy import); generate or load TPC-H-like
  high-cardinality keys (e.g. 2.7M rows → ~1.6M groups; include a synthetic
  generator so it runs without TPC-H).
- Report wall-time at `pa.set_cpu_count(1,2,4,8)` for: `Table.group_by`,
  Acero streaming hash-aggregate (`use_threads` on/off, multi-batch), and the
  partition+per-bucket+concat approach.
- Assert bit-exactness of the partitioned result vs `Table.group_by`.
- Print machine/Arrow-version header for reproducibility.
- Derive it from `benchmarks/bench_predicate_transfer.py`'s structure +
  `_libs/lazy_groupby.partition_by_key` (or inline a pure-numpy partition so the
  script has no pandas build dependency).

## Framing of the upstream artifact (decide AFTER gates)

- **If strong claim holds:** a GitHub Discussion / issue on `apache/arrow`
  titled around *"Hash aggregate does not use multiple threads; partition-
  parallel wrapping is ~2.5x"* — present the benchmark, ask whether parallel
  hash-aggregate is in scope, and **offer to contribute** (we have a working
  partition approach + a nogil partition kernel design). Keep it a question +
  evidence, not a demand.
- **If it narrows to the single-batch/Table.group_by footgun:** a smaller issue
  /doc note: *"`Table.group_by` on a single Table runs single-threaded; feed
  multiple batches / use Acero with use_threads for parallelism"* + benchmark.
  Possibly a docs PR.
- **Venue:** start on the Arrow dev list / GitHub Discussions (architecture
  question), not a PR — let maintainers steer scope before any code.

## Objections to preempt

- *"Just use Acero with threads / more batches."* → gate 1 answers this with data.
- *"Partitioning changes group order."* → acknowledge; note groups are disjoint
  across buckets so it's a concat, and order can be restored if required.
- *"The partition cost eats the win."* → report it explicitly (nogil scatter
  ~8ms/2.7M); show net speedup including it.
- *"Single-threaded by design (composability with external parallelism)."* →
  fair; frame as "expose an opt-in parallel mode," not "change the default."

## Decision rule

File **only if** gates 1–4 are green AND the result is non-duplicate AND
reproducible standalone. Otherwise: downgrade to the narrower footgun report, or
shelve with the benchmark kept as evidence. Either outcome is a valid probe
result — the point is an honest, verified contribution, not a filed issue count.
