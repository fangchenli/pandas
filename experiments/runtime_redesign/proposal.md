# A Data-Centric Redesign of Pandas

**Status:** Research proposal / Initial design
**Date:** November 2026

## 1. Motivation

Pandas faces increasing competition from modern DataFrame systems, particularly Polars and DuckDB. These systems have significant architectural advantages in query planning, columnar execution, streaming, and analytical workloads.

However, building another query engine is unlikely to be the right direction for pandas. Competing directly with specialized analytical engines would require substantial engineering resources while maintaining pandas' considerably broader API and compatibility requirements.

Pandas has a different set of strengths: labeled data, heterogeneous types, advanced indexing, alignment, mutation, and deep integration with Python's scientific computing ecosystem.

Rather than reproducing the architecture of modern analytical query engines, we propose fundamentally reconsidering how pandas represents, transforms, accesses, and modifies data.

The existing pandas architecture evolved around NumPy. Its `BlockManager` combines physical storage management, indexing, memory sharing, and other responsibilities. Alternative storage implementations, such as Arrow, were subsequently integrated through `ExtensionArray`, introducing additional layers and implementation complexity.

Incrementally evolving this architecture toward a fundamentally different design may be more complicated than developing a new runtime.

We therefore propose a clean-sheet redesign, using existing pandas as a semantic reference and performance baseline.

## 2. Problem Definition

**Can we design a composable, backend-independent data representation system that minimizes unnecessary materialization, copying, and data movement while preserving eager, mutable, labeled DataFrame semantics?**

The central observation is that logical computation and physical materialization do not necessarily need to occur simultaneously.

Consider:

```python
df2 = df.sort_values("price")
df3 = df2.iloc[:100]
```

The sorting operation must determine the ordering of rows. However, the remaining columns do not necessarily need to be physically rearranged.

Instead, `df2` can reference the original data through a permutation mapping. The subsequent positional selection can compose with that mapping.

The resulting DataFrame has well-defined logical contents even though its underlying values have not been copied.

This is fundamentally different from lazy query execution.

### Query-centric versus data-centric systems

| Property | Query-centric systems | Proposed data-centric system |
|---|---|---|
| Primary abstraction | Computation pipeline | Data representation |
| Optimization target | Query execution | Data storage, access, and movement |
| Laziness | Deferred computation | Deferred physical materialization |
| Intermediate representation | Logical/physical query plan | Composable data representations |
| Indexing | An operation in the pipeline | Fundamental part of the data model |
| Mutation | Generally outside the lazy model | First-class, with Copy-on-Write |
| Execution | Planned and scheduled | Eager, using backend-native kernels |

Our objective is not to optimize arbitrary sequences of user operations through a query planner.

Instead, we aim to make individual DataFrame operations produce efficient representations of their results, allowing subsequent operations to reuse and simplify those representations.

## 3. Theoretical Foundation: A Data Representation Algebra

An array can be modeled as a function from a finite index space to a value domain:

\[
A:I_n\rightarrow T
\]

where:

\[
I_n=\{0,1,\ldots,n-1\}
\]

A structural transformation can be represented by an index mapping:

\[
f:I_m\rightarrow I_n
\]

The transformed array is:

\[
B=A\circ f
\]

This defines a new logical array without requiring the physical values to be copied.

### Composition

Suppose a second transformation uses an index mapping:

\[
g:I_k\rightarrow I_m
\]

Then:

\[
C=(A\circ f)\circ g=A\circ(f\circ g)
\]

Therefore:

\[
\operatorname{Take}(\operatorname{Take}(A,f),g)
=
\operatorname{Take}(A,f\circ g)
\]

This follows directly from the associativity of function composition.

Other useful identities include:

\[
\operatorname{Take}(A,\operatorname{id})=A
\]

\[
\operatorname{Take}(\operatorname{Constant}(v,n),f)
=
\operatorname{Constant}(v,|f|)
\]

These identities provide a mathematical basis for simplifying physical data representations.

### Representation algebra

We propose starting with a small representation language:

\[
\begin{aligned}
R ::= {}&
\operatorname{Physical}(B)\\
&\mid \operatorname{Take}(R,f)\\
&\mid \operatorname{Concat}(R_1,R_2)\\
&\mid \operatorname{Constant}(v,n)
\end{aligned}
\]

Here:

- `Physical` references backend-managed storage.
- `Take` describes logical indexing through an index mapping.
- `Concat` represents a logical sequence of existing representations.
- `Constant` represents repeated values without allocating a full value buffer.

Slices and permutations are specialized forms of `Take`.

Missing positions can be modeled using partial mappings or mappings into an extended index space containing a missing-position sentinel.

The representation algebra describes the current logical values of data, not a deferred program that computes them.

### Representation equivalence

Two representations are equivalent when they produce identical logical values under the same type semantics:

\[
R_1\equiv R_2
\iff
\llbracket R_1\rrbracket
=
\llbracket R_2\rrbracket
\]

This allows us to define correctness-preserving transformations independently of the physical storage backend.

However, equivalent representations need not have equivalent performance.

An important part of the research is determining which representation should be retained and when physical materialization is preferable.

## 4. Proposed Architecture

The runtime is organized into four conceptual components.

### 4.1 Logical DataFrame and Type System

The logical layer defines:

- DataFrame and Series semantics.
- Row and column labels.
- Indexing and alignment.
- Logical data types.
- Missing-value behavior.
- Type promotion and conversions.
- Mutation and Copy-on-Write semantics.

Logical types are independent of their physical storage implementations.

For example, a nullable 64-bit integer has a defined logical meaning regardless of whether its values are stored using NumPy or Arrow.

PDEP-13, *The pandas Logical Type System*, provides an important starting point for this design.

We distinguish logical type properties from physical storage properties. Backend identity, buffer layout, and physical null encoding belong to the storage representation rather than the logical type.

### 4.2 Data Representation Layer

The representation layer describes how logical values map to underlying physical storage.

It supports structural transformations such as:

- Slicing.
- Deferred gathers.
- Permutations.
- Reindexing with missing positions.
- Concatenation.
- Constant/broadcast representations.

Representations may share physical storage and index mappings.

For example, multiple DataFrame columns may share a row permutation generated by sorting.

After a join, different groups of columns may use different mappings into their respective source arrays.

The representation layer is responsible for composition, simplification, sharing, and materialization decisions.

It must not become a general-purpose query optimizer.

### 4.3 Physical Storage

The initial implementation supports two first-class storage backend families:

**NumPy**
- Dense and strided arrays.
- Native numerical operations.
- Python object storage where required.
- Writable buffers where supported.

**Apache Arrow**
- Columnar memory layouts.
- Validity bitmaps.
- Chunked arrays.
- String and nested types.
- Native Arrow computational kernels.

Other array libraries participate primarily through interoperability protocols.

The objective is not unrestricted backend extensibility. Supporting NumPy and Arrow deeply is more valuable than supporting many backends superficially.

### 4.4 Computation

Computation remains eager and uses backend-native kernels.

The runtime determines:

1. The logical operation and its required semantics.
2. The physical representation of each input.
3. Which backend-native kernels support those inputs.
4. Whether materialization or conversion is necessary.

For example:

```python
selected = df.iloc[indices]
result = selected["a"] + 1
```

The addition may:

- Execute directly through the index mapping if a compatible kernel exists.
- Materialize the selected values and invoke a native kernel.
- Convert to a compatible physical representation if required and permitted.

Kernel dispatch is based on operation, logical dtype, backend, and representation capabilities.

The initial runtime should use simple deterministic dispatch rules rather than cost-based query planning.

The Python Array API may be useful for selected numerical operations, but it is not the foundation of the representation model.

## 5. Mutation and Copy-on-Write

Physical storage can be shared among multiple logical arrays.

The runtime must ensure that modifying one logical array does not unexpectedly modify another.

Consider:

```python
a = [10, 20, 30]
b = take(a, [1, 1, 2])

b[0] = 99
```

Both `b[0]` and `b[1]` initially reference `a[1]`.

However, under Copy-on-Write semantics:

- `a` must remain unchanged.
- `b[1]` must remain unchanged.
- Only `b[0]` receives the new value.

Consequently, an index mapping is a read representation, not automatically a valid write mapping.

Mutation may require detaching shared storage, materializing part of the representation, or constructing a new representation containing updated values.

We propose immutable representation descriptors with controlled mutation of the logical objects referencing them.

The initial implementation may materialize affected columns on mutation. More granular update mechanisms can be investigated later.

## 6. Design Principles

1. **Data-centric rather than query-centric.** Optimize data representations rather than computational pipelines.
2. **Eager semantics, deferred materialization.** Logical operations complete immediately, but physical data movement may be postponed.
3. **Composable structural transformations.** Indexing and related operations should be expressible through a small algebra.
4. **Backend-independent logical types.** User-visible semantics should not depend unnecessarily on physical storage.
5. **Backend-native computation.** Reuse NumPy and Arrow kernels rather than reimplementing them.
6. **Copy-on-Write by construction.** Memory sharing and mutation must be fundamental design considerations.
7. **Minimal representation complexity.** More expressive representations are not automatically better. Prefer a compact set of efficiently analyzable primitives.
8. **Measured materialization decisions.** Deferred indexing can save copies but introduce indirection, poor locality, and memory retention.
9. **End-to-end performance.** Optimizations must improve complete workloads, not only isolated microbenchmarks.

## 7. Scope and Non-Goals

### In scope

- Array and DataFrame representation.
- Indexing and alignment.
- Shared physical storage.
- Copy-on-Write and mutation.
- Backend-independent logical dtypes.
- NumPy and Arrow interoperability.
- Computation over indirect representations.
- Representation composition and simplification.
- Memory management and materialization policies.

### Out of scope

- A Polars-style lazy DataFrame API.
- General-purpose query planning and optimization.
- Distributed execution.
- A new analytical database engine.
- Universal support for arbitrary computational backends.
- Reimplementing NumPy or Arrow kernels.
- Full pandas API compatibility in the first prototype.

Streaming and out-of-memory execution are not primary architectural objectives. They may become possible for certain operations, but should not drive the system toward a query engine.

## 8. Relevant Prior Art

The proposal builds upon several established ideas.

| System / Research | Relevant ideas |
|---|---|
| Futhark | Index functions, layout algebra, representation simplification |
| Velox | Encoded physical vectors, dictionary indexing, kernels over indirect representations |
| Awkward Array | Composable array layouts, indexed arrays, missing-position representations |
| R ALTREP | Alternative physical array representations behind an eager interface |
| Repa | Delayed arrays, representation-based fusion |
| MLIR MemRef | Structured memory views, offsets, strides, subviews |
| Petersohn / Modin | Formal DataFrame semantics and operation decomposition |
| PDEP-13 | Separation of logical types from physical storage |
| Existing pandas deferred `take` | Empirical evidence for deferring structural data movement |

The mathematical foundations of indexing transformations and representation composition are well established.

We do not claim novelty for deferred indexing, representation DAGs, or algebraic simplification independently.

The potential contribution is their integration into a general-purpose runtime supporting heterogeneous, labeled, mutable DataFrames with backend-native computation.

## 9. Research Hypotheses

### Primary hypothesis

A small, composable algebra of data representations can substantially reduce unnecessary copying, allocation, and data movement in common DataFrame workloads without sacrificing eager semantics or native computational performance.

### Secondary hypotheses

**H1 — Structural transformation efficiency**

Repeated indexing, sorting, reindexing, and related operations can be represented and composed more efficiently than repeated materialization.

**H2 — Shared representation efficiency**

DataFrame-level sharing of row mappings across heterogeneous columns can reduce memory consumption and indexing overhead.

**H3 — Native computation compatibility**

Backend-native kernels can operate efficiently on selected indirect representations, or materialize them with sufficiently low overhead.

**H4 — Mutation efficiency**

Copy-on-Write can be implemented coherently over shared physical storage and indirect representations without excessive memory amplification.

**H5 — Representation simplicity**

A constrained representation algebra can capture most useful optimizations while remaining simpler and more efficient than an unrestricted representation DAG.

## 10. Initial Experimental Plan

The project should begin as an independent experimental runtime, using pandas for semantic validation and benchmarking.

### Phase 1: Array representation

Implement:

- NumPy and Arrow physical storage.
- Identity, slice, and gather representations.
- Index mapping composition.
- Explicit materialization.
- Basic positional access.

Compare a compact descriptor-based design against a small composable representation tree.

### Phase 2: DataFrame representation

Implement:

- Heterogeneous columns.
- Row and column axes.
- Shared row mappings.
- Positional indexing.
- Column replacement.
- Basic Copy-on-Write.

### Phase 3: Computation

Implement:

- Arithmetic.
- Comparisons.
- Reductions.
- Native backend dispatch.
- Materialization fallbacks.

Compare indexed-kernel execution with materialization followed by native computation.

### Phase 4: Structural transformations

Implement:

- Sorting.
- Reindexing.
- Concatenation.
- Selected join operations.

Evaluate whether the representation model remains efficient as structural transformations become more complex.

### Phase 5: Evaluation

Measure:

- End-to-end latency.
- Peak memory consumption.
- Retained physical memory.
- Bytes copied or materialized.
- Number of allocations.
- Index mapping overhead.
- Random-access performance.
- Native-kernel throughput.
- Mutation overhead.

Existing pandas, including its deferred-`take` implementation, is an essential baseline.

The goal is to establish measurable architectural advantages beyond individual operation-level optimizations.

## 11. Open Research Questions

### Representation structure

Should physical representations form a general composable DAG, or should they be normalized into compact descriptors?

A hybrid approach may offer both flexibility and low runtime overhead.

### Representation optimization

Given equivalent representations, how should the runtime choose the most efficient one?

A possible formulation is:

\[
R^*=\arg\min_{R'\equiv R}\operatorname{Cost}(R',W)
\]

where \(W\) represents an expected access workload.

The cost model may consider:

- Materialization cost.
- Index indirection.
- Cache locality.
- Access frequency.
- Reuse.
- Memory retention.
- Backend kernel capabilities.

Unlike query optimization, this problem concerns the physical representation of existing logical data.

### Missing values

How should missing positions introduced by indexing interact with backend-specific validity encodings and logical null semantics?

### Mutation

How should updates operate on representations with shared storage, duplicate index mappings, or multiple physical sources?

### Logical typing

How should we define exact type equality, logical type compatibility, nullability, type promotion, and backend-specific exceptions?

### Shared mappings

Should row mappings live at the array, column-group, or DataFrame level?

How can we support multiple mappings within a heterogeneous DataFrame without recreating the complexity of `BlockManager`?

## 12. Expected Contribution

The immediate goal is to determine whether a representation-algebra-based runtime can improve the performance, memory efficiency, and architectural simplicity of pandas.

Potential contributions include:

1. A formal model of composable data representations for labeled, heterogeneous arrays.
2. A compact runtime implementation supporting multiple physical storage backends.
3. An efficient representation composition and materialization strategy.
4. A coherent Copy-on-Write mutation model over indirect representations.
5. Experimental evidence comparing this architecture with conventional eager DataFrame implementations.

The longer-term objective is to establish a new architectural foundation for pandas rather than introduce another competing analytical query engine.

## 13. Summary

We propose a clean-sheet, data-centric redesign of pandas based on a composable algebra of physical data representations.

The fundamental abstraction is a labeled collection of logical arrays whose values may be accessed through structural transformations over shared physical storage.

Indexing transformations can compose without immediately materializing their results. Computation remains eager and uses backend-native kernels. NumPy and Arrow serve as the initial physical backends, while a shared logical type system defines user-visible semantics.

The core research challenge is not whether deferred indexing is possible—it already is—but whether a coherent representation algebra can deliver significant end-to-end improvements without introducing prohibitive runtime complexity.

**Our central research question is:**

*Can a formally defined, composable data representation system provide a simpler, faster, and more memory-efficient foundation for general-purpose DataFrames than today's materialization-oriented architecture?*

This project aims to answer that question through a combination of formal modeling, runtime design, and experimental evaluation.
