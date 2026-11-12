# Review of the pandas runtime redesign

Date: 2026-11-12. Assessment: proceed with a constrained research prototype, with an explicit semantic contract and controlled baselines. There is not yet evidence that a new runtime is simpler or faster than an incremental change to pandas.

## What is compelling

Separating logical selection from payload movement is a useful organizing principle. Wide heterogeneous tables, long strings, repeated selection, and joins followed by narrow projection are plausible beneficiaries. The proposal correctly treats duplicate gathers as read mappings rather than write destinations, recognizes retained memory, and proposes native computation with deterministic dispatch.

The potential contribution is an engineering synthesis: durable representations for interactive, labeled objects; predictable ownership and mutation; and interoperable computation. Composition of index functions itself is established. The important result would be a measured benefit under those combined constraints, including unfavorable workloads.

## Changes to the framing

**Representation optimization and query optimization are orthogonal.** Velox uses dictionary wrappers for filters and joins and combines wrapper mappings for access. DuckDB has several physical vector encodings behind a common access interface. These techniques coexist with query execution. Present the table in section 2 as a statement of this project's scope, not a classification of mutually exclusive system architectures. [Velox dictionary encoding](https://facebookincubator.github.io/velox/develop/dictionary-encoding.html), [DuckDB execution format](https://duckdb.org/docs/2.0/internals/vector).

**Pandas already separates some logical behavior from storage.** PDEP-7 makes derived objects behave as copies while allowing sharing internally. ExtensionArray supports storage other than NumPy, not just Arrow. PDEP-14 separates string missing-value behavior from its storage implementation. The problem to demonstrate is duplication or friction across today's interfaces, rather than the absence of any separation. See local [PDEP-7](../../web/pandas/pdeps/0007-copy-on-write.md), [PDEP-14](../../web/pandas/pdeps/0014-string-dtype.md), and [ExtensionArray](../../pandas/core/arrays/base.py).

**The clean-sheet advantage is a hypothesis.** An independent implementation makes experiments easier; it also assumes responsibility for dtype resolution, axes, scalar boxing, extension types, and ecosystem contracts. Measure the incremental approach alongside it. Performance comparisons must use equivalent semantics and dtypes.

**Clarify two reference claims.** The inspected main revision does not contain the experimental deferred-take module. The branch provides code and tests, but this review has not reproduced its performance evidence. PDEP-13 is now verified from the [PR supplied by the user](https://github.com/pandas-dev/pandas/pull/58455): it is Will Ayd's draft, closed unmerged, rather than an accepted specification in main. The [logical type comparison](logical_types.md) records its source and the implications for this prototype.

## Specify what eager means

Define a successful operation as fixing the result's shape, axes, dtype, validity, and logical values relative to an owned snapshot. Sorting, predicate evaluation, label lookup, join matching, validation, and result-type decisions finish during that operation. Only access to an already determined sequence is represented indirectly.

In `sort_values(...).iloc[:100]`, the full ordering still has to be determined under the proposed contract. Deferral saves payload rearrangement; it does not automatically convert sorting into top-k selection. Likewise, a join still pays for matching and producing its row maps. Arbitrary many-to-many join results can require enormous maps even when payloads remain shared.

Specify the boundary for exceptions. Invalid indices, invalid casts, unsupported operations, and user callbacks should fail at the initiating operation. Allocation failure may move to first materialization; the experimental deferred-take documentation already states this behavior. Resource-error timing is an observable exception to strict equivalence with physical eagerness. UDF side effects and external I/O cannot be delayed under this contract.

Do not represent general Python functions as mappings. In the prototype, a map is validated integer data or a closed structural descriptor. This keeps arbitrary computation outside the representation language.

## Complete the algebra before implementation

The proposed total-function composition identity is correct when shapes match and all maps are valid. Extend the model to typed arrays with an explicit validity domain. Let `Option(T)` contain valid values of type `T` and a missing value `⊥`; NaN is not automatically identical to `⊥`.

For a missing-aware map `f: I_m → I_n ∪ {⊥}`, define:

```text
take(A, f)[j] = missing                         if f[j] = ⊥
               A[f[j]]                         otherwise

(f ⋆ g)[j] = ⊥                                if g[j] = ⊥
             f[g[j]]                           otherwise

valid(take(A, f), j) = false                   if f[j] = ⊥
                      valid(A, f[j])           otherwise
```

Then `take(take(A, f), g) = take(A, f ⋆ g)`. The implementation must test validity before reading an index or source element. A missing slot may contain an unusable physical index.

The constant identity needs a condition. For `A = Constant(7, 3)` and `f = [0, ⊥]`, the result is `[7, missing]`, not `Constant(7, 2)`. Preserve a validity overlay or restrict that rewrite to total maps. Awkward's indexed option representation and Velox's separate wrapper validity illustrate relevant designs. [Awkward layouts](https://awkward-array.org/doc/main/user-guide/how-to-create-constructors.html), [Velox null handling](https://facebookincubator.github.io/velox/develop/dictionary-encoding.html).

There are further preconditions:

- Normalize negative positional indices before internal composition. Pandas' `take(..., allow_fill=True)` assigns a different meaning to `-1`; the internal representation must not confuse these conventions.
- A non-null `fill_value` is not a missing sentinel. It needs an explicit filled-selection rule or an eager fallback. An existing null in the source must not be replaced merely because an absent position uses a fill value.
- Validate before simplifying. Rewriting an invalid take of a constant into a constant would otherwise suppress an error. An empty result should preserve dtype and release unnecessary owners.
- `Concat` requires a resolved logical dtype, conversion rules, lengths, and defined order. Representation associativity applies to already compatible operands; it does not establish that arbitrary pandas concatenations with intermediate dtype inference can be reassociated.
- Define equivalence using typed, validity-aware observations. Ordinary elementwise `==` is insufficient for NaN, nullable comparisons, signed zero, categorical metadata, timezone units, and object identity. Mutation additionally requires an ownership invariant over execution traces.

The algebra should include the semantics of each constructor and a short list of guarded normalization rules. A proof of read equivalence does not prove mutation safety or preservation of exception timing.

## Ownership is the first implementation problem

Immutable descriptors are necessary but insufficient. A NumPy array can have writable aliases, and its owner may permit its writeable flag to be restored. Arrow objects can also reference externally supplied buffers. Snapshot guarantees require controlling the underlying owner, not inferring immutability solely from the wrapper type. [NumPy writeability rules](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html).

For the prototype, copy externally writable inputs and indexers at ingress unless an explicit ownership-transfer or trusted immutable-buffer contract applies. Count these copies in end-to-end measurements. An internal read-only view protects against accidental writes through that view; it does not revoke existing aliases. The experimental branch already takes special care with immutable indexer ownership; preserve that lesson.

Maintain owners independently of logical objects and maps. An owner must outlive every descriptor, cached result, and exported buffer using it. Copy-on-Write must protect both directions: changing the original cannot change a previously selected result, and changing the selection cannot change its original or siblings.

Initially, materialize the entire affected logical column into private storage, apply the update, and replace its descriptor. This handles duplicate maps and overlapping views simply. It may amplify small writes, which must be measured. Whole-column replacement can usually install a new owner without reading the old column. If an operation fails, keep the original descriptor intact; document any differences from pandas' behavior on partially failing assignments.

Python object arrays need a separate limitation: copying a pointer array does not deep-copy the referenced Python objects. Do not promise recursive object immutability. Use a compatibility fallback for objects rather than applying algebraic computation rewrites to arbitrary Python behavior.

## Native computation is the largest performance uncertainty

An arbitrary gather is not a strided NumPy view. NumPy's indexing documentation distinguishes basic views from advanced-indexing copies. Arrow defines dictionary and view layouts, but a supported storage layout does not imply every compute kernel executes over it without decoding. Its compute documentation explicitly describes dictionary expansion for comparisons. [NumPy copies and views](https://numpy.org/doc/stable/user/basics.copies.html), [Arrow compute](https://arrow.apache.org/docs/cpp/compute.html).

Start with materialize-then-native as the universally available path. Track a capability table keyed by operation, logical type and options, physical encoding, backend version, and ownership requirements. Distinguish direct execution, execution with hidden decode/allocation, and unsupported execution. A wrapper accepting an input is not evidence of zero-copy computation.

The plan to compare indexed kernels is in tension with the non-goal of implementing kernels. Resolve this by making custom indexed kernels an optional, narrowly scoped experiment. Report their implementation cost separately. Do not quietly make a growing set of custom arithmetic and reduction kernels essential to the architecture.

Avoid moving arithmetic onto unselected parent rows just because a pointwise equation appears valid. Such a rewrite can evaluate an overflowing or otherwise invalid value that was never selected, invoke Python operators extra times, and compute unnecessary results. A repeated gather also changes multiplicities for reductions. Floating-point reductions over chunks may change accumulation order. Set an explicit comparison policy rather than assuming bitwise interchangeability.

## Keep labels and types in the cost model

The array algebra alone does not model a DataFrame. Row order, column order, duplicate labels, alignment, and axis metadata must be specified. `s1 + s2` aligns labels before it is a positional binary kernel. A join may use independent left and right maps; a subsequently assigned column may already be in output order. One mandatory DataFrame-wide map cannot represent all three uniformly.

Use per-column representations that reference shared immutable map objects where their coordinate systems agree. The DataFrame owns ordered axes and column slots. This permits a shared map for a sort, distinct maps after a join, and an identity map for a replacement column, without requiring a new block-group manager.

Logical types need more than a name and bit width. Define null and NaN behavior, overflow, promotion, scalar conversion, equality, ordering, and reduction options. Timezone, timestamp unit, decimal precision, and categorical ordering can affect logical meaning. Backend identity should not define the logical type, but a backend may lack an exact implementation. Prefer an explicit error or declared fallback over silent semantic drift. Begin with a small compatibility matrix rather than redesigning all pandas dtypes at once.

Include labels in allocation and retention accounting. If axes are materialized on every structural operation, they can dominate a narrow frame even while payload columns remain deferred. Petersohn's work is relevant because it models ordered axes and data/metadata relationships, not merely column buffers. [Dataframe systems thesis](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-193.html).

## Recommended initial representation

Treat the syntax tree as a specification and optional experimental control. Prefer a bounded execution form:

```text
Frame(ordered_axes, ordered_column_slots)
Column(logical_type, representation)

representation:
    physical(owner, offset, length, stride_or_chunk_layout)
    selected(base, immutable_map, optional_validity)
    constant(typed_value, length)
    segmented(compatible_children, cumulative_lengths)  # later phase

map:
    identity | affine(start, step, length) | owned_positions
```

This is a recommendation, not a settled design. Keep at most one composed arbitrary gather above a physical leaf initially. Compose affine maps without an index allocation. Flatten gather chains once per shared map, not once per column. Arbitrary map composition generally costs `O(k)` for a result of length `k`; it is not a free metadata operation. Boolean masks are worth a measured alternative: they require scanning to count selections, but can be smaller than position vectors and can match native filter kernels.

Handle concatenation with bounded segmentation, explicit lengths, and a materialization escape path. A take over many segments can require per-row segment lookup; eagerly distributing every selection across a tree can explode metadata. Do not equate arbitrary permutations with affine layouts. Futhark ultimately simplified its layout representation substantially, accepting some additional copying; that supports investigating a bound, not importing its compiler representation unchanged. [Futhark representation history](https://futhark-lang.org/blog/2024-03-06-array-representation.html).

Cache materialization outside the immutable semantic descriptor. Define synchronized first access, owner release, cache lifetime, and whether aliases share cached results. The experimental branch's overlapping-slice and concurrent-read tests are useful requirements. Immutability by itself does not make a cache thread-safe.

## Memory and interoperability are decision boundaries

For `m` selected rows, map width `s`, `q` distinct maps, and fixed-width payload widths `w_j`, a rough incremental comparison is:

```text
map bytes                 ≈ q * m * s
eager selected payload    ≈ m * sum(w_j)
```

This is a size estimate, not a latency model. An eight-byte position map is twice the payload size of a selected int32 column, before validity and object overhead; one such map shared by many wide columns is much more attractive. Variable-length data, axes, temporary maps, and existing buffers need separate accounting. If selection keeps an otherwise dead large parent alive, total live memory can exceed the eager alternative despite fewer bytes copied.

Report unique backing allocations, logical bytes, metadata, incremental allocations, peak resident memory, and retained memory after parents are dropped. Sliced index maps can themselves retain a large owner. A materialized string view may still retain payload buffers; payload compaction and logical materialization are distinct operations. Arrow's string-view layout and Polars' adoption make this a required comparator. [Arrow format](https://arrow.apache.org/docs/format/Columnar.html), [Polars string representation](https://pola.rs/posts/polars-string-type/).

Provide explicit materialize and compact/detach operations in the research API. Initially use deterministic policies: normalize on construction; materialize when the next operation requires it; detach on mutation; release all owners for empty results; bound segmentation. Explore retention and reuse heuristics only after measuring these defaults. The `argmin` in section 11 is a useful ideal, but future workload `W` is unknown in an interactive session. Do not use future accesses to choose a policy in the main benchmark.

Define export boundaries early: NumPy conversion, Arrow C/PyCapsule export, serialization, and Python iteration. A generic NumPy consumer cannot interpret a private gather descriptor. Arrow can carry certain encodings, but the receiving consumer and requested schema may require conversion. Exported buffers must remain alive, and writable exports must not bypass Copy-on-Write. [Arrow C interface](https://arrow.apache.org/docs/format/CDataInterface.html), [Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html).

## What should happen next

Add a semantic phase before the proposal's Phase 1 and start measurement with the first implementation. The smallest useful experiment combines owned snapshots, selection composition, one shared map across mixed columns, an eager computation fallback, duplicate-map mutation, and export. Sorting and joins can initially supply precomputed maps for representation-only measurements, but full-operation benchmarks must include their production costs.

Advance only if the prototype beats an eager implementation with the same narrow API and shows value beyond the experimental deferred take. If improvements are concentrated in Arrow string payload movement, an incremental improvement or string-view integration may capture most of the benefit. If the advantages survive nullable numeric types, repeated consumption, mixed maps, mutation, and native export, the case for a general runtime becomes stronger. The [experimental plan](experiments.md) makes those outcomes testable.
