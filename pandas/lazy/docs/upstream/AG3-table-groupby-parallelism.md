# Hand-off: AG3 — `Table.group_by` parallelism (RESOLVED — broad claim REFUTED; residual folds into AG5'/#38372)

**For:** a fresh agent. Read `README.md` first. **Priority: closed.**

> **RESOLVED (2026-07-09) — the broad "Table.group_by doesn't parallelize" claim
> is REFUTED on latest pyarrow 24.0.0; do NOT file it.** Ran the decisive test
> (`/tmp/ag3_groupby_parallel.py`: single-Table vs many-chunk, int + string keys,
> low + high cardinality, `set_cpu_count(1,2,4,8)`, 20M rows):
> | shape | 1→8 core | verdict |
> |---|---|---|
> | int key, 1k groups | **82→19 ms = 4.3×**, beats Polars (126ms) | parallelizes great |
> | int key, 1M groups | 610→517 ms = 1.2× (8 worse than 4) | saturates high-card |
> | string key, 100k groups | 326→253 ms = 1.3× | saturates high-card |
>
> Two hypotheses REFUTED: (1) "Table.group_by is single-threaded" — false, it
> scales **4.3×** at low cardinality; (2) "a single Table isn't morsel-split so
> stays serial" (the old reconciliation) — false, **single-chunk and 8-chunk
> scale identically**; chunking is irrelevant. The old 124→133ms non-scaling
> (`PARALLEL_GROUPBY_SCOPE.md`) does not reproduce on pyarrow 24 — it was likely a
> since-improved release and/or a specific 2-key high-card shape.
>
> **Residual real finding:** grouped-aggregate parallel scaling **saturates at
> high cardinality** (1M int ≈1.2×, string ≈1.3×; loses to Polars 379ms at 1M
> int). This is the **same root as AG5'** (count_distinct is thread-insensitive)
> — Arrow's grouping hash-table. **CONSOLIDATE: AG3-residual + AG5' → one finding
> = "Arrow grouped hash-aggregate parallel scaling saturates at high cardinality
> vs Polars," home = open discussion #38372 (MemoTable→SwissTable).** Attach our
> data there rather than filing AG3 or AG5' separately. AG3 as a standalone item
> is CLOSED/shelved.

---
_Original hand-off (superseded by the resolution above) follows._

**Priority 4.** ⚠️ This one is
**not confirmed** and may well dissolve on investigation — do the verification
before assuming there's anything to file.

## Goal
Determine whether there is a real, file-able Arrow gap in `Table.group_by`
parallelism, or whether the observed slowness is just a single-`Table` usage
footgun (most likely). Only file if a genuine gap survives.

## Why it's shaky (read before doing anything)
- `PARALLEL_GROUPBY_SCOPE.md` measured `Table.group_by` **not scaling** with
  `cpu_count` (124→133 ms) on a **2-key high-cardinality numeric** aggregate.
- BUT `ENGINE_DESIGN.md` M4 says acero's hash aggregation **is** internally
  multi-threaded, and the AG4 benchmark observed `Table.group_by` **scaling ~4×**
  with cores for a dict-key low-cardinality case.
- Reconciliation (see `../ARROW_GAPS.md` R1): Acero parallelizes across
  **morsels/batches**; a single in-memory `Table` may not be split → looks
  single-threaded. So the broad "Arrow can't parallelize group-by" claim is
  most likely **wrong**.

## The one decisive test (gate 1, from `UPSTREAM_PARALLEL_GROUPBY_PLAN.md`)
Build a standalone benchmark comparing, on the **2-key high-cardinality numeric**
shape (the one that didn't scale):
- `Table.group_by` on a single Table, vs
- a **proper Acero streaming plan** over the same data sliced into **many
  RecordBatches** with `use_threads=True`,
across `set_cpu_count(1,2,4,8)`. (See `UPSTREAM_PARALLEL_GROUPBY_PLAN.md` for the
full gate list + objections.)

## Decision rule
- If multi-batch Acero **parallelizes** the high-card aggregate → there is **no
  capability gap**; at most a **docs/usability note** ("feed multiple batches /
  use Acero for parallel group-by; a single Table runs one thread"). File only
  if maintainers would value the note; otherwise **shelve** with the benchmark.
- If even multi-batch Acero stays single-threaded on high cardinality → a real
  gap; then do the standard gates and draft from the plan doc.

## Definition of done
Gate-1 result recorded in `../ARROW_GAPS.md` AG3 + `README.md`, with the
decision (file small note / file gap / shelve). Do **not** file the broad
"Arrow can't parallelize group-by" claim — the evidence already contradicts it.
