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
- Flat integer/string axes, shared maps across columns, Series alignment on equal or unique compatible axes, projection, and detached pandas export (Arrow strings are wrapped as `ArrowStringArray`, not boxed).
- Addition, equality, integer comparison/sum, mixed CPU NumPy/Arrow numeric dispatch, scalar/vector assignment, and column replacement. Scalar operands broadcast natively instead of materializing a constant column. NumPy and Arrow scalars are accepted wherever a Python scalar is, against either backend; each adapter unboxes its own scalar types. `Frame.replace_column` aligns a Series on labels, as pandas column assignment does; an Array is positional.
- Cell-based snapshot isolation, synchronized materialization with retry after failure, explicit compaction, native-buffer reports, and optional event counters.
- Detached NumPy `(values, validity)` and Arrow exports, explicit CPU backend conversion, correctness-checked W02 and W05–W09 runners.

Public entry points are the handles and factory functions exported from `pdruntime`, plus the methods listed above. Internal cells, descriptors, backend objects, and their buffers are implementation details, even where Python permits attribute access. The prototype does not defend against deliberate mutation of its internals. Concurrent reads are supported; simultaneous writes to one handle are not.

## Deliberate limitations and costs

JAX, GPU execution, general custom dtype registration, key kernels, joins, sorting, public reindexing, concatenation/segments, Float64, Boolean AND/OR, and serialization remain later milestones. The package layout combines the initial adapters in `backends.py` and strategy logic in `execution.py`; split modules when another adapter warrants it.

Missing input is `NA` or `None`; `pd.NA` is rejected because the core type layer does not import pandas, and there is no pandas import adapter yet. Ingress validates Python iterables element by element; native input arrays and selection/computation use bulk kernels. The first NumPy ingress path shares Arrow's typed input validation before copying into NumPy-owned storage. Arrow ingress, compaction, and detached Arrow export use an IPC copy boundary to guarantee independent ownership. This is deliberately conservative; a pilot measurement found it runs at roughly memcpy speed and is 3–7x faster than an identity `take` copy. Gather results (selection, Arrow replacement) are already freshly allocated and are not copied again. It is measured identically across strategies; it is not evidence about an optimized ownership path.

Copy events report logical buffer-byte estimates, not exact memory traffic through IPC or native allocation counts. Memory reports cover known underlying buffer owners and map arrays; they exclude allocator overhead, Python metadata, and some logical-size details. Frame reports list axis bytes separately. Peak RSS and device memory measurements are future work.

The supplied pilot runner compares representation strategies with identical adapters and validates full values outside timed regions. Timing includes the named workload's actual consumers and a detached export of the consumed columns. Diagnostic event collection runs separately. It does not compare against pinned pandas main, the deferred-take branch, or external engines, and does not justify a general speedup claim. The installed pandas version and import location are recorded as the semantic oracle.

## Validation record

The current suite has 132 pytest cases. The initial run passed 117 pytest cases, including generated cases, all repository pre-commit checks, and editable installation. At 10,000 rows and four columns, 108 timing runs and 36 separate diagnostic runs passed value checks. The timing run was performed after the test process finished.

See [environment and source fingerprint](results/environment.json), [diagnostic summary](results/smoke_summary.json), [raw pilot observations](results/pilot_observations.jsonl), and [tested dependency versions](requirements-tested.txt). These records use the installed pandas 3.1.1 wheel as the semantic oracle, not a build of pinned pandas main. They do not establish an architectural performance advantage. They predate the scalar-broadcast, filter, column-alignment, and export fixes, so their source fingerprint no longer matches `src/`.

## Follow-up experiments

Validate the third-backend boundary with the narrow JAX adapter (M3a), then add structural operations (M4). W15 covers 2D regressions before any general replacement claim; the [remaining decisions](../decisions.md) specify key semantics, custom types, bounded reads, axis costs, and a pandas integration experiment.
