# E — Parallelize `libjoin` without Rust (free-threading path)

**Target:** pandas core, `pandas/_libs/join.pyx` — GH issue
[#51364](https://github.com/pandas-dev/pandas/issues/51364) ("PERF: parallelize
libjoin calls", open; labels *Multithreading / Performance / Reshaping*).

**One-line goal:** parallelize the numeric merge-join indexers by releasing the
GIL and chunking, and evaluate whether **free-threaded CPython (PEP 703)** makes
the chunk-in-Python / nogil-kernel-per-chunk pattern actually scale — i.e. get
Polars-class join parallelism *in-tree*, no Rust, no OpenMP.

> **Status: hand-off-ready, characterization done, nothing filed.** This is an
> *implementation + measurement* task (a would-be PR to pandas), not a
> bug-find. The parallelism premise is already proven by our own lazy-engine
> kernels; the open question this hand-off answers is *how much free-threading
> removes the composition tax that sank the naive approach* (and the issue
> author's own prototype).

---

## 0. Why this is tractable without Rust (read first)

Our lazy-engine join probe (`../JOIN_KERNEL_REBUILD_PROBE.md`) already built
GIL-releasing parallel join kernels **in Cython** and measured them end to end.
Its headline "a Rust kernel beats Polars" is narrower than it reads: the
bottleneck it named is the **Cython/no-OpenMP toolchain** (no in-kernel threads,
no fused SIMD gather), not the language. Two facts drive this hand-off:

1. **We already parallelize joins in pure Cython today.** `pandas/_libs/lazy_join.pyx`
   has `with nogil` chunk kernels (`probe_count_chunk`, `probe_fill_chunk`,
   `build_join_table_i8`, `partition_by_bucket`, `join_gather_bucket_rm`) driven
   by a `ThreadPoolExecutor` (`pandas/lazy/backends/numpy/join.py:777`
   `inner_join_indexers_i8`, `:882` `partitioned_join_gather`). `nogil` releases
   the GIL, so those sections genuinely run concurrently **on a stock GIL build**.
2. **The thing that erased the win is the "isolation trap"** — nogil chunks win
   standalone, but the Python orchestration + the final concat give it back. This
   is *exactly* the issue author's own result: their chunk+concat is **3.9 s vs
   2.89 s monolithic** because "the concat is 1.05 s." Our probe named the fix:
   avoid the concat (write disjoint output regions) and stop paying the Python
   boundary per chunk.

Free-threading attacks precisely that composition tax: on a GIL build the driver
serializes, threads thrash re-acquiring the GIL on exit, and the assembly can't
run in parallel from Python. On a free-threaded build all three run
concurrently — the same "thread pool over nogil kernels" model Polars gets from
rayon, but native to CPython.

---

## 1. The target, precisely

`inner_join_indexer(ndarray[numeric_object_t] left, ndarray[numeric_object_t] right)`
(`pandas/_libs/join.pyx:465`) is a **sorted merge-join** indexer: it assumes both
inputs are sorted and walks them in lockstep, emitting `(left_indexer,
right_indexer)`. Siblings: `left_join_indexer` (:354), `outer_join_indexer`
(:565), and the `_unique` fast paths. All are fused over `numeric_object_t`
(numeric **and** object).

The issue's proposal, restated:
- For **non-object** dtypes the inner loop touches no Python objects, so it can
  run under `nogil`.
- Split the sorted `left` at its midpoint; `searchsorted` the split key in
  `right` to get the matching `right` sub-range; join the two halves
  independently; concatenate.
- Author's numbers (10⁸ ⋈ 10⁸ sorted int64): monolithic **2.89 s**; two chunks
  **1.47 s each**; concat **1.05 s** → naive total **3.9 s** (a regression).

So the algorithm parallelizes; the *packaging* is what loses. That is the whole
problem to solve.

> This is the **sorted-merge** indexer (used by `pd.merge` on already-sorted
> keys, `MultiIndex` joins, and `merge_asof`-adjacent paths). It is a **different
> algorithm** from our lazy hash join (`inner_join_indexers_i8`), but it shares
> the nogil-chunk + thread-pool + avoid-concat pattern and every lesson below.

---

## 2. Plan (no Rust; free-threading-first, GIL-build-compatible)

Do the steps in order; each is independently measurable and independently
shippable. Steps 1–3 help on **both** interpreters; step 5 is the free-threading
payoff.

**Step 1 — Release the GIL for the non-object path (the issue's ask).**
Split `inner_join_indexer` so the numeric specializations run the merge loop in a
`with nogil` block (object dtype keeps the current GIL-holding path). This is the
enabling change; on its own it does not speed up a single call, but it lets a
caller run several concurrently. Small, self-contained, upstreamable alone.

**Step 2 — Parallelize the merge via searchsorted-split (the issue's sketch,
done right).** A driver (Cython or Python) that: picks `k` split points in
`left`, `searchsorted`s them in `right`, runs the `k` nogil sub-joins on a thread
pool. Correctness: the sub-ranges must be **disjoint and gap-free** — a run of
equal keys straddling a split must go to exactly one chunk (split on a *key
boundary*, not a raw index; use `left`-value split points and
`side='left'`/`'right'` consistently). Add a regression test for duplicate keys
across a boundary.

**Step 3 — Kill the concat (the 1.05 s).** Do **not** concatenate per-chunk
outputs. Two passes with disjoint writes:
  - a cheap `nogil` **count** pass per chunk → per-chunk output sizes → exclusive
    prefix-sum → each chunk's output offset;
  - allocate the two output indexers **once** at the total size;
  - a `nogil` **fill** pass per chunk writing into its own `[offset:offset+size)`
    slice — no locks, no merge.
  This mirrors our `probe_count_chunk`/`probe_fill_chunk` split in
  `lazy_join.pyx` and removes the author's dominant cost outright. (Two passes is
  cheaper than one pass + a 1.05 s concat; our probe measured the same tradeoff.)

**Step 4 — (if the gather is in scope) parallelize the payload take.** Not part
of the indexer itself, but the real wide-payload sink downstream. `np.take(...,
out=...)` over row-chunks in a thread pool already wins with zero Cython
(`../JOIN_KERNEL_REBUILD_PROBE.md`, "Gather" section) — note it for the merge
caller, don't fold it into `inner_join_indexer`.

**Step 5 — Measure GIL vs free-threaded.** Same code, two interpreters. The
hypothesis to confirm/refute: on the GIL build steps 1–3 give a *partial* win
(nogil compute overlaps, but driver + allocation serialize and threads contend
on GIL re-entry); on the free-threaded build the driver + disjoint-write assembly
also run concurrently, so scaling approaches the ideal `monolithic / k` the
author's per-chunk timings imply (~1.5 s at k=2, less at higher k until
bandwidth-bound). **This measurement is the deliverable** — it tells pandas
whether free-threading is the answer to #51364.

---

## 3. Environment

Build/verify against **current pandas `main`** (not this repo's branch, and not
an old release), on two interpreters:

```bash
# GIL build (baseline)
python -m venv /tmp/join-gil && . /tmp/join-gil/bin/activate
pip install -U cython numpy meson-python ninja
pip install -e . --no-build-isolation          # from a fresh pandas main checkout

# Free-threaded build (the experiment)
uv python install 3.13t                          # or 3.14t; a --disable-gil CPython
uv venv --python 3.13t /tmp/join-ft && . /tmp/join-ft/bin/activate
pip install -U 'cython>=3.1' numpy meson-python ninja
pip install -e . --no-build-isolation
python -c "import sys; print('gil_enabled=', sys._is_gil_enabled())"
```

- **Cython free-threading:** the `_libs` module(s) touched must be marked
  free-threading-compatible (`# cython: freethreading_compatible = True`) and the
  nogil sections must not touch Python objects. Expect that pandas `_libs` may
  not yet build/import cleanly free-threaded — **surfacing that is itself a
  finding** (feeds the README "E–H" row / a pandas free-threading tracking
  issue). Do not force it; record where it breaks.
- Record `pandas`, `numpy`, `cython`, interpreter version + `gil_enabled` in
  every result.

---

## 4. Reference implementation (steal the pattern, not the code)

Our kernels are lazy-engine-specific (int64 keys, **hash** join, unique-build
assumption) and are **not** a drop-in for the general sorted-merge
`inner_join_indexer` — but they are a working, measured template for every step:

| Concept | Where |
|---|---|
| nogil count / fill split (step 3) | `pandas/_libs/lazy_join.pyx` `probe_count_chunk` (:170), `probe_fill_chunk` (:184) |
| thread-pool driver over nogil chunks | `pandas/lazy/backends/numpy/join.py` `inner_join_indexers_i8` (:777) |
| disjoint-write partitioned gather | `join.py` `partitioned_join_gather` (:882); kernel `join_gather_bucket_rm` (`lazy_join.pyx:313`) |
| full decomposition + measured numbers + the isolation trap | `../JOIN_KERNEL_REBUILD_PROBE.md` |
| the parallel groupby precedent (same nogil+threadpool model, shipped) | `../PARALLEL_GROUPBY_SCOPE.md` |

---

## 5. Gates (all green before proposing anything upstream)

1. **Reproduce the baseline on current pandas `main`** — confirm `inner_join_indexer`
   is still single-threaded / GIL-held for numeric dtypes there (read
   `join.pyx` on `main`; it may have changed since the 2023 issue).
2. **Correctness**: the parallel indexer is **bit-identical** to the serial
   `inner_join_indexer` output (same `(left_indexer, right_indexer)`), including
   duplicate keys straddling chunk boundaries, empty side, no-match, and all
   numeric dtypes. Use the serial function as the oracle.
3. **Self-contained benchmark**: one script, synthetic sorted int64 (and a
   couple of dtypes), best-of-N warm, prints the version/gil header; compares
   serial vs parallel on GIL and free-threaded builds; sweeps `k`
   (2/4/8/cores). Modest-machine runnable. Add an `asv` benchmark under
   `asv_bench/benchmarks/join.py` for the eventual PR.
4. **Duplicate/prior-art search** on pandas: the issue itself (#51364), plus any
   PRs that touched `join.pyx` threading, plus the pandas free-threading tracking
   issues (`gh api -X GET search/issues -f q="repo:pandas-dev/pandas libjoin
   nogil parallel"` and `... free-threading join`). Record what you searched.

---

## 6. Guardrails (non-negotiable — same as every hand-off)

- **Never open an issue/PR or post a comment on #51364 without explicit human
  go-ahead.** It is outward-facing and carries the human's GitHub identity.
  Draft in chat; the human posts. (Project rule: no GitHub comments from the
  agent.)
- The deliverable is **measured evidence + a reviewable branch**, not a filed PR.
- State measurements as fact; state mechanism (why free-threading helps) as a
  hypothesis for maintainers to confirm.
- Nothing here changes the lazy engine's own join path — it already routes
  in-memory equi-joins to `pd.merge` and has its own kernels. This is an
  upstream-pandas contribution track; keep the two separate.

---

## 7. Definition of done

Gates green → a branch on a pandas `main` checkout implementing steps 1–3 (step
4/5 as scoped) → a results table contrasting **serial vs parallel × GIL vs
free-threaded × k** with correctness asserted → a short write-up of whether
free-threading dissolves the isolation trap (the answer to #51364's viability
without Rust) → **draft PR/comment text prepared for the human to review and
post.** Record the outcome back in this file, the README "E–H" row, and
`../ARROW_GAPS.md` (rename that registry's scope note if needed — this is a
pandas item, not an Arrow one).

---

## 8. Open questions / known risks

- **Memory bandwidth is the hard floor.** Free-threading gives cheap threads,
  not more bandwidth. Our probe measured wide/string gather scaling only ~3.3×,
  tapering after 4 threads. The **numeric indexer (this issue) benefits**; don't
  promise wide-payload wins.
- **Free-threaded pandas maturity.** `_libs` may not build/import free-threaded
  yet; Cython 3.1 free-threading support is new. If it doesn't build, the
  fallback is the **OpenMP-in-Cython** route (`prange`) — the *other* no-Rust
  option (thread inside the kernel instead of via Python). Note the tradeoff:
  free-threading keeps parallelism in Python where a scheduler can compose it;
  OpenMP keeps it in C and needs a build-system change (pandas builds without
  OpenMP today — the exact gap the probe named).
- **Refcount tax.** Free-threaded CPython adds atomic refcounting overhead to
  pure-Python hot loops. Keep the driver/assembly out of per-element Python (all
  per-row work stays in the nogil kernel).
- **Object dtype stays serial** by design (the loop touches Python objects) —
  scope the change to the numeric specializations only.
- **Callers.** `inner_join_indexer` feeds `pd.merge`/`MultiIndex` join paths;
  a change must preserve their exact contract (row order within equal-key runs,
  indexer values). The serial oracle in gate 2 covers this.
