# Upstream AG4: Acero string-key hash-aggregate slowness — issue draft

Ready-to-file draft for ARROW_GAPS.md gap **AG4**, verified by
`../../benchmarks/bench_arrow_string_groupby.py`. **Not yet filed** — needs the
final version gate (below) and your go-ahead (filing is outward-facing).

## Gate status

- [x] **Verified** (standalone benchmark, 10M rows, cardinality × thread matrix,
      correctness asserted).
- [x] **Duplicate search (apache/arrow)** — ~12 query phrasings + manual review
      of adjacent Grouper issues (#32431 memory-prealloc, #45821/#45822 API,
      #41036 benchmark, #41233 num_groups bug). **No existing issue reports
      this gap.** Closest is DataFusion #27498 (Rust; grouping *on* dict data —
      different engine, different angle). → **non-duplicate.**
- [ ] **Latest-Arrow re-check** — measured on pyarrow 23.0.1 / Arrow C++ 23.
      Confirm the gap still reproduces on the current release before filing
      (no closed issue suggests a fix landed, but verify). This is the one
      remaining gate.
- [ ] **Your approval to file.**

## Draft issue

**Title:** `[C++][Acero] Hash aggregation on string group keys is 3.5–10× slower than on dictionary-encoded keys`

**Labels (suggest):** Type: enhancement / Component: C++

**Body:**

> ### Describe the enhancement requested
>
> Grouping a table by a **`string`/`large_string`** key and aggregating is
> **3.5–10× slower** than grouping by the **dictionary-encoded** version of the
> exact same keys. Since dictionary-encoding is purely a key representation
> change (identical groups, identical results), this points to the hash
> aggregate re-hashing raw key bytes per row rather than interning string keys
> internally.
>
> Measured with pyarrow 23.0.1 on 10M rows, one string key + a float64 `sum`,
> sweeping cardinality and `pa.set_cpu_count`:
>
> | distinct keys | `group_by` raw `large_string` | `group_by` dict-encoded | ratio |
> |---|---|---|---|
> | 100 (1 core) | 388 ms | 46 ms | **8.5×** |
> | 10,000 (1 core) | 518 ms | 53 ms | **9.7×** |
> | 1,000,000 (1 core) | 2404 ms | 595 ms | **4.0×** |
> | 100 (8 cores) | 96 ms | 12 ms | **7.9×** |
> | 1,000,000 (8 cores) | 2527 ms | 446 ms | **6.2×** |
>
> Results are identical across both paths. For reference, Polars grouping the
> same raw string keys is ~3.8× faster than Acero per-thread at low cardinality
> (102 ms vs 388 ms, single-threaded), i.e. the raw-string path is also behind a
> peer engine — so this isn't just "dict-encoding is faster," the raw-string
> hashing itself appears to leave a large factor on the table.
>
> ### Repro
>
> Self-contained script (pyarrow + numpy, plus polars for the reference column);
> generates synthetic data, asserts correctness across methods: [attach
> `bench_arrow_string_groupby.py`].
>
> ### Expected
>
> Grouping by string keys should be competitive with grouping by their
> dictionary-encoded form — ideally the aggregate would dictionary-encode
> (intern) string keys internally so callers don't have to. At minimum,
> documenting "dictionary-encode string group keys for large speedups" would
> help.
>
> ### Component / environment
>
> C++ Acero / `arrow::compute::Grouper` (row-table key encoding). pyarrow
> 23.0.1; [confirm on latest before filing].
>
> ### Hypothesis (for maintainers to confirm)
>
> The grouper's row-table key encoder hashes the full variable-length key bytes
> per row for string types; interning/dictionary-encoding string keys once would
> avoid the repeated hashing (the measured dict-encoded path already does this
> externally). Not confirmed from source — happy to provide more benchmarks.

## Notes for us

- Framing is a question + evidence + offer, not a demand (per UPSTREAM plan
  conventions). Lead with the *enhancement* angle (dict-encode internally),
  not "bug."
- Keep our dict-encode workaround regardless of upstream outcome (it's already
  in the engine's groupby routing).
- After filing: record the issue number in ARROW_GAPS.md AG4.
