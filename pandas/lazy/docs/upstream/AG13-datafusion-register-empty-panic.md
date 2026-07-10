# Hand-off: AG13 — DataFusion-Python `register_record_batches` panics on an empty table

**For:** a fresh agent. Read `README.md` first. **Target project:
`apache/datafusion-python`** (the pyo3 binding, NOT `apache/datafusion` core or
`apache/arrow`). **Priority: 2** — crash-class, root cause pinned to one line,
repro is three lines, verified on latest, non-duplicate. File-ready pending human
go-ahead.

> **STATUS (2026-07-09): FOUND BY THE DIFFERENTIAL PROBE + ROOT-CAUSED + DUP-SEARCHED
> — file-ready.** This is the differential probe's (`../DIFFERENTIAL_PROBE.md`)
> first *new* find: widening to the RESULT/robustness surface immediately surfaced
> it via the degenerate-input (empty table) sweep. Verified two ways: (1)
> **executed** on datafusion 54.0.0 / pyarrow 24.0.0 (panics); (2) **source-read**
> the offending line, **byte-identical at tag 54.0.0 and `main`** — not fixed.
> Distinct from the fixed #575 (see §Dup-search). Remaining gate: human approval.

## The finding (one line)
`SessionContext.register_record_batches(name, partitions)` **panics** (a Rust
`panic!` unwind surfaced to Python as `pyo3_runtime.PanicException`) instead of
raising a graceful `PyDataFusionError`, whenever a partition contains **zero
record batches** — the natural result of `register_record_batches(name,
[tbl.to_batches()])` when `tbl` is an **empty (0-row) table**, because
`pyarrow.Table.to_batches()` returns `[]` for an empty table.

## Evidence (executed on datafusion 54.0.0 / pyarrow 24.0.0)
| construction | partitions arg | result |
|---|---|---|
| `register_record_batches("t", [empty.to_batches()])` | `[[]]` (0 batches) | **PANIC** — `PanicException: index out of bounds: the len is 0 but the index is 0` |
| `register_record_batches("t", [[zero_row_batch]])` | `[[<0-row batch>]]` | OK (0 agg rows) |
| `register_record_batches("t", [[b], []])` (mixed) | one real + one empty partition | OK |
| `ctx.from_arrow(empty_table)` | — | OK (this path was fixed by #613) |

So the trigger is specifically a partition with **zero batches**, not a zero-*row*
batch. A `panic!` is never a valid response to valid, well-typed input — the
method signature is `PyDataFusionResult<()>`, i.e. it is contracted to return
errors, not unwind. Where we hit it: `benchmarks/translate_datafusion.py` and
`benchmarks/differential_probe.py` both register via
`register_record_batches(name, [tbl.to_batches()])`; any empty input frame
crashes the process.

## Root cause (confirmed in source — `apache/datafusion-python`)
`crates/core/src/context.rs`, `register_record_batches` (byte-identical at
`54.0.0` and `main`):
```rust
pub fn register_record_batches(
    &self,
    name: &str,
    partitions: PyArrowType<Vec<Vec<RecordBatch>>>,
) -> PyDataFusionResult<()> {
    let schema = partitions.0[0][0].schema();          // <-- unchecked [0][0]
    let table = MemTable::try_new(schema, partitions.0)?;
    self.ctx.register_table(name, Arc::new(table))?;
    Ok(())
}
```
`partitions.0[0][0]` indexes the first partition's first batch to derive the
schema with **no length check**. If `partitions` is empty, or its first partition
is empty (`[[]]`), the `[0]`/`[0]` index panics. The function already returns a
`PyDataFusionResult`, so the fix is to detect the no-batch-available case and
return a `PyDataFusionError` (or accept a schema / infer an empty `MemTable`)
rather than indexing blindly.

## Standalone repro (pure datafusion + pyarrow; run OUTSIDE the repo dir)
```python
import datafusion as d, pyarrow as pa
from datafusion import SessionContext
print("datafusion", d.__version__, "pyarrow", pa.__version__)

empty = pa.table({"k": pa.array([], pa.string()), "v": pa.array([], pa.float64())})
print("to_batches():", empty.to_batches())          # []  -> partitions arg becomes [[]]
ctx = SessionContext()
ctx.register_record_batches("t", [empty.to_batches()])   # PanicException
```
Observed on 54.0.0: `PanicException: index out of bounds: the len is 0 but the
index is 0` at `crates/core/src/context.rs:835`.

**Workarounds (keep in our lowering regardless):** guard empty frames before
registering; or pass an explicit zero-row batch
(`register_record_batches(name, [[pa.RecordBatch.from_arrays([...], schema=sch)]])`);
or use `ctx.from_arrow(tbl)` (already handles empty).

## Dup-search (recorded — apache/datafusion-python, issues AND PRs)
Searched `register_record_batches panic`, `empty table panic index out of
bounds`, `register_record_batches empty`, `panic empty pyarrow table`.
- **#575** (CLOSED/completed 2024-04-13, fixed by **#613**) — "Panic when reading
  empty `pyarrow.Table`": a *different* method (`from_arrow_table`) panicking at
  `src/context.rs:294:37` on a 0-**row**-but-has-columns table. #613 fixed that
  path (our `from_arrow` test now passes, and an explicit zero-row batch is fine).
  It did **not** touch `register_record_batches`, which still indexes `[0][0]`
  and panics on a zero-**batch** partition. **This is the surviving sibling** —
  same "empty pyarrow input panics" class, distinct unfixed method + trigger
  (0 batches vs 0 rows). Precedent for scoping a per-method fix: exactly the
  AG11 pattern (#16221 fixed hash/NLJ join schema, missed `cross_join.rs`).
- No open issue covers `register_record_batches`.

## Filing recommendation: ONE new scoped issue (do NOT reopen #575)
- **Title:** `register_record_batches panics ("index out of bounds") on an empty
  partition / empty table instead of returning an error`.
- **Body:** the 3-line repro + the `[0][0]` root cause + "reachable via the common
  `register_record_batches(name, [tbl.to_batches()])` idiom when `tbl` is empty,
  because `to_batches()` returns `[]`". Reference #575/#613 ("the `from_arrow`
  path was fixed there; `register_record_batches` has the same class of bug on a
  zero-batch partition"). Question + evidence + offer-to-help tone.
- **Suggested fix to offer:** guard the empty case —
  `if partitions.0.iter().all(|p| p.is_empty()) { return Err(PyDataFusionError::…) }`
  (graceful error), or derive an empty `MemTable` from a caller-supplied/first
  available schema. Add a `test_register_empty` mirroring #613's test.
- **Labels:** `bug`, `python` / `core` (as the project uses).

## Gates
- [x] **Reproduces on latest release** — executed on datafusion 54.0.0 (panic);
      source byte-identical at `54.0.0` + `main`.
- [x] **Duplicate search recorded** — not a dup; #575/#613 is a different method
      (`from_arrow_table`) + trigger (0 rows), already fixed; this method unfixed.
- [x] **Standalone repro** — above, three lines, pure datafusion + pyarrow.
- [ ] **Human approval before filing** (outward-facing; guardrail).

## Definition of done
Filed (with #) referencing #575/#613, or shelved if fixed on a newer release,
recorded here + in `README.md` + `../ARROW_GAPS.md`. Keep the empty-frame guard in
`translate_datafusion.py` / `differential_probe.py` regardless.
