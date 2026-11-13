# Experimental plan for the pandas runtime redesign

Status: broader evaluation plan, dated 2026-11-12. The measured subset and its conclusion are in the [M5 decision report](report.md); the items it lists as not measured remain open.

The first objective is to determine whether structural sharing delivers an advantage after paying for ownership, map composition, real consumption, and mutation. Implement the smallest runtime that can answer that question before broadening API coverage. The supporting evidence and semantic issues are in the [review](review.md) and [references](references.md).

The [Python prototype design](prototype.md) defines the concrete API and milestones. It compares eager, compact, and tree modes early using shared semantic and backend code. The first implementation supports nullable integer, boolean, and string data; float workloads below remain scheduled extensions pending an explicit float semantic contract.

## Phase zero establishes the contract

Write executable specifications for owned input snapshots, valid position maps, missing positions, dtype resolution, and mutation isolation. Start with one-dimensional arrays and ordered columns. Use explicit logical types for int64, float64, nullable integer/boolean, and strings with a chosen missing policy. Test additional pandas string policies separately. Initially reject unsupported types with a documented error rather than silently coercing them.

Define the eager boundary: shape, labels, validity, maps, operation errors, and values are fixed at return. Storage allocation can be deferred. Document delayed resource errors. Import/export ownership and object-dtype limitations must be explicit.

Use pandas as a differential oracle for the supported API contract. Use a direct eager implementation with the same narrow API to isolate representation overhead. Mathematical properties supplement the oracle; they do not replace it.

## Baselines and controls

| Baseline | Purpose | Required control |
| --- | --- | --- |
| Pinned main `f1cc1a1bc979b83e641facf72ec6526593d61420` | Current pandas behavior and performance | Explicit dtypes, Copy-on-Write behavior, and dependency versions |
| Deferred-take branch `629c2c8f89b6b3d6f15fa4176b4d765fb6a531f2` | Incremental optimization comparator | Build a matched branch control without the deferred feature, or isolate its changes; the branch also contains unrelated changes |
| Eager minimal runtime | Cost of the same narrow API with physical selection | Same labels, nulls, ownership, native kernels, and validation |
| Compact representation runtime | Main experimental treatment | Bounded maps and deterministic policies |
| Small representation tree | Test the complexity/performance tradeoff | Same operations and ownership; instrument depth and normalization |
| Direct NumPy and PyArrow | Kernel and materialization lower-level controls | Include conversion and ingress separately from resident-data runs |
| Arrow string-view storage | Alternative way to reduce payload copying | Same values; include initial conversion and retained buffers |
| Polars eager; optionally DuckDB and Polars lazy | External workload context | Match ordering, nulls, keys, projection, threads, and output consumption; report semantic mismatches |

A runtime with fewer semantic checks can beat pandas without demonstrating a better representation. The eager minimal-runtime control is therefore mandatory. External query-engine comparisons do not replace the incremental pandas control.

Pin Python, NumPy, PyArrow, pandas commits, compiler/build mode, hardware, and thread settings. Report string physical layouts explicitly. Keep an immutable dataset/seed manifest and benchmark commands alongside results. Do not compare one implementation's first run against another's warmed cache.

## Correctness cases

| Area | Required cases |
| --- | --- |
| Map laws | Identity, total-map composition, missing-aware composition, slices with negative steps, and chains of gathers |
| Validation | Empty arrays, empty maps, negative positions, out-of-bounds indices, invalid fill indices, zero-length constants, and malformed boolean masks |
| Missing data | Source nulls plus absent positions; all-null and no-null inputs; non-null fill values; NaN distinct from validity where the type requires it |
| Constants | A missing-introducing take of a non-null constant; a null constant; invalid indices must still raise |
| Concatenation | Empty typed children, multiple sources, mismatched physical chunk boundaries, resolved promotions, and gathers crossing segments |
| Snapshot ownership | Caller mutates an input array or indexer after selection; caller retains an alias before flags change; owners survive garbage collection |
| Copy-on-Write | Mutate parent, child, and sibling independently; duplicate gather `[1, 1, 2]`; overlapping slices; whole-column replacement |
| Assignment semantics | Duplicate destinations, broadcast assignment, incompatible values, and failure without corrupting cached representations |
| Axes | Duplicate row/column labels, different axis orders, nonidentical labels in arithmetic, empty axes, and independent maps after joins |
| Computation | Two columns with different maps, mixed backends, integer overflow policy, reduction null options, signed zero, and float tolerance policy |
| Boundary behavior | Scalar access, display, iteration, NumPy and Arrow conversion, serialization, and writable-export isolation |
| Cache behavior | Repeated materialization, overlapping consumers, concurrent first reads, exception/retry, and release of original owners |

Generate randomized structural-operation sequences and compare both logical results and mutation traces with the eager reference. Include small exhaustive index domains, which often expose validity composition bugs. Keep representation invariants independently testable: no invalid dereference, bounded depth, map ownership, and correct length/dtype after every operation.

Treat object arrays as a compatibility lane. Check reference semantics without claiming recursive deep-copy behavior. Unsupported advanced labels such as MultiIndex should be explicitly outside the first prototype, with representative cases retained in the later compatibility backlog.

## Workload matrix

| ID | Workload and terminal observation | Main hypothesis | Failure it exposes |
| --- | --- | --- | --- |
| W01 | Sort a wide mixed frame, take a short prefix, then export selected columns | H1, H2 | Copying avoided only until export; expensive full ordering |
| W02 | Several filters/takes, then sum one numeric column | H1, H3 | Map composition dominates narrow or cheap values |
| W03 | Join with large string payload, project a few columns, then consume | H1, H2 | Matching/map costs dominate; narrow projection already solves most copying |
| W04 | Reindex with missing labels, then arithmetic and export | H1, H3 | Validity and dtype handling force repeated conversion |
| W05 | Random gather of int32/int64 data, then full scan of every column | H3 | Indirection, map size, and cache misses exceed saved work |
| W06 | Read the same selection 1, 2, 5, and 20 times | H3 | Repeated gathers or decoding instead of reuse |
| W07 | Retain a tiny selection, drop the parent, then compact | H4 | Large source or map allocations remain live |
| W08 | Duplicate gather, one-cell write, then read parent and child | H4 | Incorrect scatter or excessive mutation amplification |
| W09 | Repeated sparse writes and whole-column replacement | H4 | Materializing every write or unnecessarily reading replaced columns |
| W10 | Repeated concatenation of small chunks, take across chunks, then export | H5 | Segment explosion and quadratic metadata maintenance |
| W11 | Long selection chain with intermediate objects retained and released | H1, H5 | Descriptor depth, composition allocations, and owner lifetime |
| W12 | Export repeatedly to NumPy, Arrow, and serialization | H3, H4 | Hidden fallback copies and unsafe shared writable memory |
| W13 | Small interactive frames with scalar reads and display | H5 | Python dispatch and object overhead dominate |
| W14 | Labeled binary arithmetic with unequal axes and independent column maps | H2, H3 | Axis work or forced frame-wide materialization dominates |
| W15 | Homogeneous frame arithmetic, row-wise sum, consumed transpose, contiguous 2D NumPy export | H3, H5 | Per-column dispatch and packing lose existing 2D throughput |
| W16 | Restricted nullable Int64 groupby-sum after repeated selection | H3, H5 | Key equality, null groups, or row multiplicity diverge across backends |

For W01–W04, distinguish structural construction from consumption, but report their sum as the primary end-to-end result. A workload that consumes only a subset is legitimate if both implementations expose the same observable output. Also run full-consumption controls. Do not require a deferred implementation to expand genuinely unused columns, or allow it to skip columns the reference actually returns to a terminal consumer.

Start with a few representative sizes, then sweep the dimensions that change the answer:

- Rows: small interactive inputs, medium cache-resident inputs, and larger memory-bandwidth inputs that safely fit the machine.
- Width: 1, 8, and 64 columns; homogeneous and heterogeneous cases.
- Selection: 0%, 1%, 10%, 50%, and 100%; monotone, random, repeated, and reversed positions.
- Values: fixed-width integers/floats, null density, short strings, and long strings with skewed lengths.
- Consumption: one column versus all columns; one read versus reuse; immediate versus delayed export.
- Structure: one versus many chunks, few versus many distinct shared maps, shallow versus deep chains, and nontrivial labels.

These are proposed experiment parameters, not promised scalability limits. Avoid a huge Cartesian product: screen for crossovers, then expand around them.

W15 follows the [PDEP-16 discussion](missing_values.md). Compare the column-based runtime with a direct 2D values-plus-validity reference and pinned pandas under matching semantics. Include all-valid/null-heavy data, contiguous/gathered rows, and a column write with live sibling views. Charge packing, validity transformation, ownership copies, and export. Use Int64 first where supported; Float64 follows its semantic milestone. These controls precede any general replacement claim but need not block M2/M3.

For a later float extension, record the oracle's `future.distinguish_nan_and_na` setting and test both candidate policies separately across construction, assignment, computed NaNs, reductions, and export. The existence of an option in pinned pandas is source evidence, not a substitute for behavior tests.

The [remaining decisions](decisions.md) specify W16 and follow-up bounded-read, axis-cost, and integration experiments. These are scheduled extensions, not prerequisites for the first selection prototype.

## Instrumentation

Measure wall time in separate regions: ingress, key/predicate computation, mapping and label construction, normalization, first consumption, repeated consumption, mutation, and export. Validate values outside the timed region while ensuring the timed consumer actually performs the work. A checksum alone is insufficient for semantic validation of order and missing values.

Report allocation and memory measurements at the same boundaries:

- Logical payload size, map/validity/descriptor bytes, and axis storage.
- Unique backing-buffer capacity reachable from live results, deduplicated by allocation ownership rather than summing view lengths.
- Bytes explicitly copied/materialized and why; hidden backend allocations as separately measured totals when exact attribution is unavailable.
- Peak RSS in fresh subprocesses, steady-state retention after parents are dropped, and retention after compaction.
- Native allocator counters where available, alongside Python allocation tracking. Python-only tracers miss native allocations; allocator caching can keep RSS high after logical owners are released.
- Kernel/decode/fallback counts, cache hit rate, map-composition work, and maximum node/segment count.

Benchmark resident-data and import-to-export workflows separately. Report both single-thread and explicitly configured native threading when relevant. Use repeated independent runs, randomize baseline order, and report distributions/confidence intervals with raw observations. Record GC and allocator settings. Avoid interpreting retained RSS alone as proof of a live reference leak.

## Ablations

Change one feature at a time: shared versus per-column maps; eager versus deferred map composition; affine versus dense maps; boolean mask versus positions; cache on versus off; whole-column detachment versus any later finer-grained update; compact descriptors versus bounded trees; traditional strings versus string views.

The first computation path should be materialize-then-native. Add direct indexed execution only for a small operation/type pair if a compatible backend facility exists or if an explicitly separate custom-kernel experiment is justified. Record implementation effort and semantic restrictions with its performance result.

The initial materialization policy uses only present state: input encoding, size, ownership, and the next operation's requirements. A future-aware optimal choice may be reported as an offline upper bound, never as a deployable runtime result.

## Third-backend validation

Run the prototype's M3a JAX experiment after the NumPy/Arrow vertical slice and before broadening structural operations. Its purpose is to test the adapter boundary with immutable arrays and explicit devices. Keep it separate from the main representation-performance conclusion.

Use nullable Int64/Boolean import, composed gather, addition, integer sum, and assignment with duplicate read mappings. Compare eager, compact, and tree using identical JAX kernels, then check values, validity, and dtype against the NumPy/Arrow contracts. Cover empty/all-missing outputs, source nulls combined with absent positions, alias isolation, disabled 64-bit mode, and rejected implicit backend/device transfers. JAX remains an optional test dependency; record when this suite is skipped.

Record JAX/jaxlib versions, 64-bit configuration, device, map/mask placement, and compilation state. Run CPU first; accelerator measurements are optional. Synchronize timed native work. Report cold compilation and warmed execution separately, plus transfers and strict-eager wait costs; include ingress and export in full-workflow totals. Count explicit map transfers as well as payload transfers. Native device memory requires device-aware metrics; host RSS alone is insufficient. See [JAX sources R28–R30](references.md).

The boundary passes when the adapter preserves semantics without backend-specific branches throughout handles, type rules, and representation traversal. A small shared numerical helper layer is justified by actual reuse; a public plugin API, whole-Frame JIT, and autodiff remain outside this experiment.

## Decision criteria

Set numerical thresholds before running the full suite, after learning the measurement noise from a pilot. The following gates define what the evidence must establish:

| Hypothesis | Evidence needed | Reason to narrow or reject |
| --- | --- | --- |
| H1 structural efficiency | Lower end-to-end cost after required consumption across multiple structural workloads | Wins disappear after terminal consumption or matched eager control |
| H2 shared representations | Map memory and composition scale with distinct coordinate maps, rather than column count | Cross-column bookkeeping cancels the saving |
| H3 native computation | Useful throughput with direct execution or bounded fallback overhead | Most operations repeatedly decode/materialize or need extensive new kernels |
| H4 mutation | Correct isolation, measurable retention, and acceptable write amplification | Small updates or external ownership impose prohibitive copies |
| H5 simplicity | Bounded depth and segments, few normalization rules, understandable dispatch, and maintained correctness | Complexity merely relocates into maps, owners, and type adapters |

Track code size, normalization-rule count, dispatch combinations, fallback frequency, and test burden for H5. These are proxies for complexity, not proof of maintainability. Document negative results and crossover points; do not summarize only the best speedup.

An incremental pandas improvement remains a successful research outcome if it captures most benefits at lower complexity. A general runtime is justified only if its gains extend beyond the existing Arrow string/binary deferred-take cases and survive realistic mutation and interoperability.

## Deliverables after this review

1. A small supported-semantics matrix and ownership/representation specification.
2. An eager reference, compact descriptors, and a small tree built against the same Python API and backend adapters, compared from the selection milestone onward.
3. Reproducible benchmark scripts, pinned environment metadata, raw results, and correctness checks.
4. A decision report separating measured findings from projections and documenting retained-memory tradeoffs.

Future code, benchmark output, and reports should stay under this research directory until there is evidence to propose integration. No runtime code or benchmark results were created as part of this initial review.
