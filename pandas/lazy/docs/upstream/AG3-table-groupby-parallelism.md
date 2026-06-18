# Hand-off: AG3 — `Table.group_by` parallelism (VERIFY FIRST — likely a footgun, may not be file-able)

**For:** a fresh agent. Read `README.md` first. **Priority 4.** ⚠️ This one is
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
