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
- [x] **Latest-Arrow re-check (June 2026)** — re-ran on **pyarrow 24.0.0** /
      polars 1.42.0 in a fresh venv. **Gap persists, essentially unchanged**
      across two releases (8.35x/9.54x/4.05x at 1 core vs the 23.0.1
      8.5x/9.7x/4.0x). Not fixed. Fresh numbers in the table below.
- [ ] **Your approval to file.** ← only remaining gate.

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
> Measured with **pyarrow 24.0.0** on 10M rows, one string key + a float64
> `sum`, sweeping cardinality and `pa.set_cpu_count` (gap is unchanged from
> 23.0.1):
>
> | distinct keys | `group_by` raw `large_string` | `group_by` dict-encoded | ratio |
> |---|---|---|---|
> | 100 (1 core) | 369 ms | 44 ms | **8.4×** |
> | 10,000 (1 core) | 503 ms | 53 ms | **9.5×** |
> | 1,000,000 (1 core) | 2378 ms | 588 ms | **4.0×** |
> | 100 (8 cores) | 83 ms | 11 ms | **7.7×** |
> | 1,000,000 (8 cores) | 2128 ms | 398 ms | **5.3×** |
>
> Results are identical across both paths. For reference, Polars grouping the
> same raw string keys is ~3.7× faster than Acero per-thread at low cardinality
> (100 ms vs 369 ms, single-threaded), i.e. the raw-string path is also behind a
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
> 24.0.0 (gap also present on 23.0.1).
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
