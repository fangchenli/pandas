# Hand-off: AG4 — Acero string-key hash-aggregate is 3.5–10× slower than dict-encoded

**For:** a fresh agent. Read `README.md` (playbook) first. **Priority 1** — the
most ready upstream item. **Status:** verified + non-duplicate; one gate left
(latest-Arrow re-check) + human approval, then file.

## Goal
File an apache/arrow enhancement issue reporting that Acero's hash aggregate on
raw `string`/`large_string` group keys is 3.5–10× slower than on the
dictionary-encoded form of the same keys, with a reproducible benchmark; offer
to help. (Optionally, scope an internal-dict-encoding fix — discuss first.)

## Context (1 paragraph)
The lazy-pandas probe found that routing groupby through Acero is fast only when
string keys are dictionary-encoded first; raw string keys are multiples slower.
The engine already dict-encodes as a workaround. This is the cleanest upstream
candidate: large, robust, reproducible, no morsel-parallelism confound.

## Evidence (already gathered, pyarrow 23.0.1, 10M rows, 1 string key + float sum)
| distinct keys | raw `large_string` | dict-encoded | ratio | vs Polars (1 core) |
|---|---|---|---|---|
| 100 (1 core) | 388 ms | 46 ms | 8.5× | acero 3.8× slower than polars |
| 10,000 (1 core) | 518 ms | 53 ms | 9.7× | — |
| 1,000,000 (1 core) | 2404 ms | 595 ms | 4.0× | — |
| 100 / 1M (8 cores) | 96 / 2527 ms | 12 / 446 ms | 7.9× / 6.2× | — |
Results identical across methods. Honest residual: at K=1M, Polars' high-card
hash still edges acero_dict (342 vs 446 ms multi-thread) — dict-encoding doesn't
*fully* close the gap at extreme cardinality. Mechanism hypothesis: the grouper
re-hashes raw key bytes per row instead of interning.

## Reproducible artifact (exists)
`../../benchmarks/bench_arrow_string_groupby.py` — standalone (pyarrow + polars +
numpy, synthetic data, cardinality × thread matrix, asserts correctness).
Run per `README.md` §1:
```
POLARS_MAX_THREADS=1 python bench_arrow_string_groupby.py   # per-thread kernel
python bench_arrow_string_groupby.py                        # real-world
```

## Gates (status)
- [x] Verified (above). [x] Duplicate search done — non-duplicate (closest is
  DataFusion #27498, Rust, different). See `UPSTREAM_AG4_STRING_HASH.md` for the
  searched phrasings.
- [ ] **DO THIS:** re-run the benchmark on the **latest** pyarrow (playbook §1),
  confirm the gap persists, and capture the fresh numbers + version for the
  issue. If it's been fixed, record that in `../ARROW_GAPS.md` AG4 and stop.
- [ ] Human approval to file.

## Deliverable
The ready issue draft (title, labels, body with the table + repro + hypothesis)
is in **`UPSTREAM_AG4_STRING_HASH.md`** → "Draft issue". Update its numbers with
the latest-Arrow run, then file with `gh issue create --repo apache/arrow`
(only after approval). Title:
`[C++][Acero] Hash aggregation on string group keys is 3.5–10× slower than on dictionary-encoded keys`.

## Definition of done
Latest-Arrow gate green → draft refreshed + approved → filed → issue number
recorded in `../ARROW_GAPS.md` AG4 and `README.md` index. Optional next: ask
maintainers whether internal string-key dict-encoding is in scope; if yes, scope
a C++ PR (separate, larger task).
