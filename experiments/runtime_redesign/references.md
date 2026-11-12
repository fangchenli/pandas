# Annotated references for the pandas runtime redesign

Consulted: 2026-11-12. Primary sources only. “Essential” identifies the first reading pass; “supporting” identifies material for a particular design question. An annotation describes relevance, not evidence that the proposed runtime will be faster. Live documentation versions are those served during this review, not the versions of a tested environment.

## Representation and materialization

### R01 Velox dictionary encoding

**Essential.** Velox maintainers. [Dictionary Encoding](https://facebookincubator.github.io/velox/develop/dictionary-encoding.html). Documentation; relevant sections read: Introduction, Nulls, DecodedVector.

The closest execution precedent: index wrappers can select or duplicate rows, and decoding combines nested mappings and validity. Useful for the read representation and shared-map design. Velox's kernel ecosystem is part of its solution; its capabilities cannot be assumed for NumPy or PyArrow.

### R02 DuckDB vector encodings

**Essential.** DuckDB maintainers. [Execution Format](https://duckdb.org/docs/2.0/internals/vector). Documentation served as 2.0; vector formats and Unified Vector Format read.

Flat, constant, dictionary, and sequence vectors share a generic access interface. This directly challenges a strict query-centric/data-centric dichotomy. Study how the interface avoids a separate kernel for every encoding combination; do not assume its bounded execution batches provide long-lived pandas ownership semantics.

### R03 Futhark representation history

**Essential.** Troels Henriksen. [Evolution of Futhark's Array Representation](https://futhark-lang.org/blog/2024-03-06-array-representation.html), 2024-03-06. Author's technical article; read through its concluding discussion.

Documents the move from flexible structural mappings through composed LMADs to a simpler representation. This is evidence that representational flexibility can hinder analysis. LMADs address affine layout transformations, not arbitrary pandas row permutations or multiple-source joins.

### R04 Futhark memory representation paper

**Supporting; full paper pending.** Philip Munksgaard, Cosmin Oancea, Troels Henriksen. *Compiling a Functional Array Language with Non-Semantic Memory Information*, IFL 2022. [Project publication record](https://futhark-lang.org/publications.html), [author PDF](https://futhark-lang.org/publications/ifl22.pdf).

The project index verifies authors, title, venue, and year. Full-text retrieval timed out; only indexed excerpts and the publication record were inspected. Read before deriving formal representation invariants from Futhark. R03 supports the current review's conclusions independently.

### R05 Awkward indexed layouts

**Essential.** Awkward Array maintainers. [Direct constructors](https://awkward-array.org/doc/main/user-guide/how-to-create-constructors.html), especially IndexedArray, IndexedOptionArray, and masked arrays. Relevant sections read.

IndexedArray changes element access without changing its content type; IndexedOptionArray adds missing positions. Record layouts also show how an index can avoid immediate propagation into every field. This is unusually close prior art for heterogeneous structural selection, although its mutation and axis semantics differ from pandas.

### R06 R ALTREP

**Essential.** Luke Tierney, Gabe Becker, Tomas Kalibera. [ALTREP: Alternative Representations for R Objects](https://svn.r-project.org/R/branches/ALTREP/ALTREP.html). Historical design document; design, guidelines, and compact-vector sections read.

Separates element/region access from requesting a physical data pointer and addresses duplication and serialization. Particularly useful for defining materialization boundaries behind an ordinary interface. The historical branch text should not be treated as a current exhaustive API reference.

### R07 Repa delayed arrays

**Supporting; partial review.** Ben Lippmeier and Gabriele Keller. *Efficient Parallel Stencil Convolution in Haskell*, Haskell 2011. [Author-hosted PDF](https://repa.ouroborus.net/papers/2011-stencil/stencil-haskell2011.pdf). Indexed abstract and representation excerpts reviewed; direct full-text open timed out.

Repa distinguishes manifest arrays from index-to-value functions and uses delayed representations for fusion. This is useful conceptual background, but its deferred computations are broader than this proposal's structural mappings. Do not use its compiler-dependent performance as evidence for a Python runtime.

### R08 MLIR MemRef

**Supporting.** LLVM project. [MemRef dialect](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop). The `memref.subview` section was read.

Defines offsets, sizes, strides, and composition relative to an input view. Useful for affine descriptor invariants and distinguishing relative coordinates from physical storage offsets. It does not provide the complete missing-aware gather algebra, labeled axes, or Copy-on-Write policy needed here.

### R09 Late materialization in column stores

**Essential.** Daniel J. Abadi, Daniel S. Myers, David J. DeWitt, Samuel R. Madden. *Materialization Strategies in a Column-Oriented DBMS*, ICDE 2007. [Author PDF](https://www.cs.umd.edu/~abadi/papers/abadiicde2007.pdf); [MIT extended technical report](https://dspace.mit.edu/entities/publication/cbb87087-bee0-4e96-9d91-09a4c05e2b4e), issued 2006-11-27. Abstract and introductory strategy discussion read.

Adds a major missing lineage: retaining positions and postponing tuple reconstruction. Its use of “materialization” concerns combining columns into tuples, so it is related rather than identical to pandas gather deferral. Read the evaluation before designing locality and consumption experiments.

### R10 MonetDB X100

**Supporting.** Peter Boncz, Marcin Zukowski, Niels Nes. *MonetDB/X100: Hyper-Pipelining Query Execution*, CIDR 2005. [CWI publication record](https://ir.cwi.nl/pub/16497), [paper copy](https://15721.courses.cs.cmu.edu/spring2019/papers/15-execution/boncz-cidr2005.pdf). Institutional abstract and bibliographic record reviewed.

Relevant to the cost of full intermediates versus per-element interpretation, and to considering region access without introducing a query planner. CWI's generated citation inconsistently names VLDB; the paper identifies CIDR 2005. Its reported database speedups are not transferred to this proposal.

## DataFrame semantics and pandas contracts

### R11 DataFrame algebra

**Essential.** Devin Petersohn, Stephen Macke, Doris Xin, William Ma, Doris Lee, Xiangxi Mo, Joseph E. Gonzalez, Joseph M. Hellerstein, Anthony D. Joseph, Aditya G. Parameswaran. *Towards Scalable Dataframe Systems*, PVLDB 13(11), 2033–2046, 2020. DOI: [10.14778/3407790.3407807](https://doi.org/10.14778/3407790.3407807). [Publisher PDF](https://www.vldb.org/pvldb/vol13/p2033-petersohn.pdf), [author preprint](https://arxiv.org/abs/2001.00888). Abstract and introduction reviewed.

Establishes ordering, flexible schemas, row/column symmetry, and data/metadata fluidity as central concerns. Use it to specify the logical layer missing from a pure array algebra. It is a broader system agenda, not a complete specification of every pandas edge case.

### R12 Petersohn thesis

**Supporting.** Devin Petersohn. *Dataframe Systems: Theory, Architecture, and Implementation*. UC Berkeley dissertation, UCB/EECS-2021-193, 2021-08-13. [University record and PDF link](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-193.html). Bibliographic record and abstract reviewed.

The longer treatment connects a data model and algebra to Modin. Useful follow-up for axis semantics, decomposition, and the burden of preserving pandas behavior. The dissertation has not been read in full for this review.

### R13 Pandas Copy-on-Write

**Essential.** Joris Van den Bossche. [PDEP-7](https://pandas.pydata.org/pdeps/0007-copy-on-write.html), created July 2021; marked Implemented. [Pinned local text](../../web/pandas/pdeps/0007-copy-on-write.md) and [implementation notes](../../doc/source/development/copy_on_write.rst) reviewed.

This is the semantic starting point for isolation of derived pandas objects. The local implementation notes explain block-level weak-reference tracking. A new owner graph must preserve the behavior even when sharing spans indirect mappings.

### R14 Pandas string semantics

**Essential.** Joris Van den Bossche. [PDEP-14](https://pandas.pydata.org/pdeps/0014-string-dtype.html), created 2024-05-03; marked Accepted. [Pinned local text](../../web/pandas/pdeps/0014-string-dtype.md) reviewed.

Provides a concrete example of separating storage choice from missing-value behavior, including the distinction between `str` and nullable string semantics. Use it to avoid reducing logical typing to “same bits, same type.”

### R15 Pandas mutation typing

**Supporting.** Marco Gorelli. [PDEP-6: Ban upcasting in setitem-like operations](https://pandas.pydata.org/pdeps/0006-ban-upcasting.html), created 2022-12-23; marked Implemented. Abstract and scope reviewed; [local text](../../web/pandas/pdeps/0006-ban-upcasting.md).

Mutation needs an explicit type-validity policy, not just storage detachment. It also distinguishes assignment into an existing column from replacing a column. Use this when building the mutation compatibility matrix.

### R16 ExtensionArray interface

**Essential.** pandas maintainers. [Extending pandas](https://pandas.pydata.org/pandas-docs/stable/extending.html), plus local [ExtensionArray](../../pandas/core/arrays/base.py) and [ArrowExtensionArray](../../pandas/core/arrays/arrow/array.py). Extension overview and local `take` contracts reviewed.

Documents an existing non-NumPy extension boundary and the requirements inherited by indexing/alignment. `allow_fill`, negative indexing, fill-value boxing, and conversion must be preserved or explicitly scoped out. External array protocols alone do not replace pandas extension-type semantics.

### R17 PDEP-13 logical type system

**Essential; verified draft.** Will Ayd. *PDEP-13: The pandas Logical Type System*, created 2024-04-27, revision 3. [PR 58455](https://github.com/pandas-dev/pandas/pull/58455), [pinned source](https://github.com/WillAyd/pandas/blob/af9da9771ce739c95c7c4c5df922d9f2545ed7c8/web/pandas/pdeps/0013-logical-type-system.md). Full source and selected discussion comments reviewed. Closed unmerged on 2025-04-21; its header retains the draft status “Under discussion.”

Relevant to backend-independent dtype construction and result typing. See the [logical type comparison](logical_types.md) for provenance, limitations, and our design choices. This resolves the earlier missing-source note; absence from local main was not evidence that the proposal did not exist.

### R18 Local deferred take experiment

**Essential; local implementation evidence.** Branch `perf-arrow-deferred-take` at `629c2c8f89b6b3d6f15fa4176b4d765fb6a531f2`.

Inspected paths:

- `pandas/core/arrays/arrow/deferred.py`: `PendingTake`, map composition, materialization, scalar access, immutable-indexer checks, parent release, and overlapping-slice handling.
- `pandas/tests/arrays/arrow/test_deferred_take.py`: test inventory for shared maps, missing positions, mutation, pickling, and concurrent reads.
- `doc/source/user_guide/pyarrow.rst`: supported types and documented gather boundaries.
- `asv_bench/benchmarks/indexing.py`: benchmark source inspection; not a benchmark result.

The branch defers supported Arrow string/binary selections, shares indexers, and composes further selection. Missing-introducing selections and other dtypes are documented as eager. This is the main incremental baseline; performance must be reproduced independently. Inspect without checking out the branch:

```sh
git show 629c2c8f89b6b3d6f15fa4176b4d765fb6a531f2:pandas/core/arrays/arrow/deferred.py
```

## Storage formats and computation boundaries

### R19 Arrow layouts

**Essential.** Apache Arrow project. [Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html), documentation served as v26.0.1. Layout overview, dictionary/validity descriptions, and view-layout descriptions inspected.

Provides concrete encodings for validity, dictionary selection, string views, list views, and run-end encoding. Logical selection, decoding, and payload compaction should be distinguished. Having a format in the specification does not establish compute support in a chosen PyArrow version.

### R20 Arrow compute contracts

**Essential.** Apache Arrow project. [C++ Compute Functions](https://arrow.apache.org/docs/cpp/compute.html) and [Python compute index](https://arrow.apache.org/docs/python/compute.html), served as v26.0.1. Relevant operation/type tables and dictionary comparison note inspected.

Use these to build a versioned capability table. In particular, documented dictionary expansion during comparisons defeats a blanket assumption that encoding-aware storage means direct encoded computation. Probe actual kernels and count allocations in the prototype.

### R21 Polars string views

**Essential.** Ritchie Vink. [Why we have rewritten the string data type](https://pola.rs/posts/polars-string-type/), 2024-01-30. Technical article; motivation and layout sections read.

Explains the payload-copy cost of traditional variable-width strings and the view representation adopted by Polars. Include this as a storage-layout comparator so improvements are not attributed exclusively to a new runtime. The article's benchmarks were not reproduced here.

### R22 NumPy views and copies

**Essential.** NumPy developers. [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html), served as v2.5. Relevant view and indexing sections read.

Basic slicing and arbitrary advanced indexing have different physical consequences. Use this to separate affine descriptor cases from gather cases and to establish the straightforward materialization baseline.

### R23 NumPy ownership and writeability

**Essential.** NumPy developers. [ndarray.setflags](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html), served as v2.5. Writeability conditions read.

Read-only status is conditional on ownership and can sometimes be changed. The runtime must own or explicitly borrow buffers and maps; setting a flag is not sufficient to guarantee a snapshot in the presence of external aliases.

### R24 Arrow C data interface

**Supporting.** Apache Arrow project. [The Arrow C data interface](https://arrow.apache.org/docs/format/CDataInterface.html). Ownership/release callback sections inspected.

Provides the interoperability ownership contract. A representation cache and its owners must survive exported arrays until release. This interface describes Arrow-compatible data, not arbitrary private representation nodes.

### R25 Arrow PyCapsule interface

**Supporting.** Apache Arrow project. [The Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html). Array/stream export and requested-schema sections read.

Defines Python-facing array and stream export protocols and schema requests. Plan export adapters early; schema negotiation and consumer capabilities can require materialization even when internal sharing is possible.

### R26 Python Array API

**Supporting.** Consortium for Python Data API Standards. [Python array API standard](https://data-apis.org/array-api/latest/), [purpose and scope](https://data-apis.org/array-api/latest/purpose_and_scope.html), served as 2025.12. Overview and scope inspected.

A numerical interoperability reference, not a specification of pandas axes, heterogeneous tables, null semantics, or shared indirect storage. Its role should stay limited to operations whose semantics and inputs match.

### R27 Polars eager and lazy APIs

**Supporting.** Polars maintainers. [Lazy API](https://docs.pola.rs/user-guide/concepts/lazy-api/). Eager/lazy comparison and pushdown example read.

Polars supports both modes. Use eager operation-by-operation runs for the closest workflow comparison, and separately report optimized lazy runs where appropriate. The latter can eliminate computation that this proposal intentionally performs eagerly.

## Further reading and source limitations

The essential sources cover every named prior-art family in the proposal and add database late materialization, DuckDB encodings, and string-view layouts. Full proof review of Futhark/Repa and the complete Petersohn dissertation remains follow-up work. No numerical result in those sources is treated as a measurement of this runtime.

Before implementing representation-aware kernels, read the full Futhark papers linked from R04 and the scalar/vector function documentation linked from R01. Before supporting additional logical dtypes, extend the concrete compatibility matrix against the pinned pandas source; R17 informs the direction but does not specify every rule. Retain source versions with any later benchmark report.
