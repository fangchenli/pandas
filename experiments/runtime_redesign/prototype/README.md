# Python runtime prototype

This standalone research package implements the M0–M3 numerical/string vertical slice from [the design](../prototype.md). It does not modify pandas itself. NumPy and Arrow share logical type rules and three execution strategies: immediate selection (`eager`), map composition on construction (`compact`), and map resolution on consumption (`tree`).

## Run it

From this directory, use an isolated Python environment with NumPy, PyArrow, pandas, pytest, and Hypothesis installed. Install the package with `python -m pip install -e '.[test]'`, or set `PYTHONPATH=src` for the commands below. The parent pandas checkout need not be compiled. Run from this directory to avoid importing the parent source tree as the pandas oracle.

```sh
python -m pytest -q
PYTHONPATH=src python benchmarks/run.py --rows 10000 --width 4 --repeat 3 --output results/pilot.json
```

```python
from pdruntime import Runtime

rt = Runtime(strategy="tree", metrics=True)
df = rt.frame(
    {
        "price": rt.array([10, 20, 30], dtype="int64", backend="numpy"),
        "name": rt.array(["a", "b", "c"], dtype="utf8", backend="arrow"),
    }
)
selected = df.take([1, 1, 2])
selected.set_at(row=0, column="price", value=99)
assert selected.column("price").to_list() == [99, 20, 30]
assert df.column("price").to_list() == [10, 20, 30]
```

## Implemented surface

- Nullable Int64, Boolean, and Arrow UTF-8; canonical type identity, scalar checks, explicit result binding, wrapping integer addition/sum.
- Owned ingress, negative positional indexing, slices, missing-aware internal maps, filtering, constants through `Runtime.full`, composition, and subset/scalar reads.
- Flat integer/string axes, shared maps across columns, Series alignment on equal or unique compatible axes, projection, and detached pandas export (Arrow strings are wrapped as `ArrowStringArray`, not boxed). UTF-8 is stored as Arrow `large_string`, matching pandas' `StringDtype('pyarrow')`. `Series.array` returns the unlabeled values without copying, for positional operations such as `Frame.filter`.
- Arrow is the default backend: `Runtime(backend="arrow")` is where arrays start when a constructor names no backend, and where numeric operations on mixed NumPy/Arrow operands run, converting the other operand once. `Runtime(backend="numpy")` is the NumPy-first ablation; UTF-8 always starts on Arrow, its only adapter.
- Addition, equality, integer comparison/sum, mixed CPU NumPy/Arrow numeric dispatch, scalar/vector assignment, and column replacement. Scalar operands broadcast natively instead of materializing a constant column. NumPy and Arrow scalars are accepted wherever a Python scalar is, against either backend; each adapter unboxes its own scalar types. `Frame.replace_column` aligns a Series on labels, as pandas column assignment does; an Array is positional.
- Cell-based snapshot isolation, synchronized materialization with retry after failure, explicit compaction, native-buffer reports, and optional event counters.
- Detached NumPy `(values, validity)` and Arrow exports, explicit CPU backend conversion, correctness-checked W02 and W05–W09 runners.
- M4 structural operations, each resolving its keys and labels before returning:
  - `Frame.sort_values(name)`: one Int64 key, stable, ascending, missing keys last. One shared permutation map for every column.
  - `Frame.reindex(labels)`: unique source labels, missing fill only. Target labels may repeat; one shared missing-aware map.
  - `Frame.join(right, on=, how=, suffixes=)`: inner or left equijoin on one non-null Int64 key, in left key order (then right key order) under a fresh RangeIndex, as `pandas.merge` orders them. Right keys may repeat (many-to-one and many-to-many). Keys are matched by a pandas key-only adapter (`keys.py`), one public `Index.join` call returning row indexers; a left join whose rows each match once keeps an identity left map. A many-to-many join grows two output-length position maps; each source payload is retained once. Left and right columns keep separate maps into their own sources.
  - `Runtime.concat(frames)`: identical ordered columns, dtypes and backends. Eager concatenates payloads; compact and tree build a `Segments` descriptor. Compact flattens nested segments at construction, as it composes maps; tree keeps the nesting. A read through segments flattens nested segments without recursion and reads each touched child once. Columns of one operation share the offsets array and the routing of a map across children. When the positions cross segments out of order, or the map introduces missing rows, the read gathers a second time to restore output order; slices over segments are gathered rather than sliced. Both are deliberate baselines for W10 to measure.
- `benchmarks/compare_pandas.py` runs W01 (sort, prefix), W03 (join, narrow projection), W04 (reindex with absent labels, arithmetic), W10 (append 16 slices one at a time, then gather) and W14 (labeled arithmetic on unequal axes) against pandas; every arm must return the same result.

Public entry points are the handles and factory functions exported from `pdruntime`, plus the methods listed above. Internal cells, descriptors, backend objects, and their buffers are implementation details, even where Python permits attribute access. The prototype does not defend against deliberate mutation of its internals. Concurrent reads are supported; simultaneous writes to one handle are not.

## Deliberate limitations and costs

JAX, GPU execution, general custom dtype registration, native key kernels, general merge/reindex/concat options (right and outer joins, multiple keys, non-null fill, mixed-backend concat), Float64, Boolean AND/OR, and serialization remain later milestones. The package layout combines the initial adapters in `backends.py` and strategy logic in `execution.py`; split modules when another adapter warrants it.

Missing input is `NA` or `None`; `pd.NA` is rejected because the core type layer does not import pandas, and there is no pandas import adapter yet. Ingress validates Python iterables element by element; native input arrays and selection/computation use bulk kernels. The first NumPy ingress path shares Arrow's typed input validation before copying into NumPy-owned storage. Arrow ingress, compaction, and detached Arrow export use an IPC copy boundary to guarantee independent ownership. This is deliberately conservative; a pilot measurement found it runs at roughly memcpy speed and is 3–7x faster than an identity `take` copy. Gather results (selection, Arrow replacement) are already freshly allocated and are not copied again. It is measured identically across strategies; it is not evidence about an optimized ownership path.

Copy events report logical buffer-byte estimates, not exact memory traffic through IPC or native allocation counts. Memory reports cover known underlying buffer owners and map arrays; they exclude allocator overhead, Python metadata, and some logical-size details. Frame reports list axis bytes separately. Peak RSS and device memory measurements are future work.

The supplied pilot runner compares representation strategies with identical adapters and validates full values outside timed regions. Timing includes the named workload's actual consumers and a detached export of the consumed columns. Diagnostic event collection runs separately. It does not compare against pandas; `benchmarks/compare_pandas.py` does (see the validation record below). The installed pandas version and import location are recorded as the semantic oracle.

## Validation record

The current suite has 295 pytest cases.

The [second pod evaluation](results/2026-11-13-pod-2/README.md) compares every strategy, and a NumPy-default ablation, with pandas 3.1.1 and the deferred-take PR (pandas-dev/pandas#72023) at 100k, 1M and 4M rows and 8 to 64 string columns; every arm returned identical results. The prototype leads on narrow reads, deep filters, long chains and sort-prefix, by more as the frame widens (up to 4.5x at 64 strings); profiling attributes that to per-column costs inside the PR. It ties on full consumption and alignment and loses on join and append. Its large reindex lead reflects a path the PR does not yet defer. The [first pod evaluation](results/2026-11-13-pod/README.md) is kept for the record; its prototype figures include two bugs since fixed. Tested dependency versions are in [requirements-tested.txt](requirements-tested.txt). The earlier 10,000-row pilot records, which predated several fixes, were removed.

## Follow-up experiments

The [M5 decision report](../report.md) closes the evaluation for the measured scope and keeps this package as a reference implementation. The JAX adapter (M3a), W15, and the [remaining decisions](../decisions.md) experiments (key semantics, custom types, bounded reads, axis costs, pandas integration) are not scheduled; the report lists when to reopen them.
