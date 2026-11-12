# Remaining architecture decisions and experiments

Date: 2026-11-12. These decisions complement the representation, backend, and missing-value designs. Implement M0–M3 first, then use the following experiments to test boundaries before broadening compatibility.

## Key semantics belong to logical operations

Elementwise equality can return NA. Key matching must instead decide whether rows belong to the same equivalence class. A proposed internal binding is `bind_keys(logical_types, purpose, options) -> KeySignature`, separate from arithmetic/comparison binding. It supplies equality, hash compatibility, ordering, and missing-key policy; physical adapters implement that signature or decline it. Equal keys must have equal hashes within an execution. Hash values need not be portable across backends or processes unless a future interchange contract requires it.

For the first join, keep the existing non-null Int64 restriction. For a later restricted groupby-sum, use one nullable Int64 key and one Int64 value column. Make key ordering and dropping/retaining null keys explicit. If missing keys are retained, give them one equivalence class independently of elementwise `NA == NA`. Specify value-column `skipna` and `min_count` independently of key handling. Do not use backend defaults as semantic rules. pandas exposes missing-key behavior through `dropna`; its [GroupBy contract](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) is a comparison reference.

Float NaN keys, signed zero, categoricals, string normalization/collation, and multikey grouping require separate bindings and tests before entering scope. The initial key signature is a design boundary, not an implemented hashing library.

## Custom types and compute backends are separate extensions

A JAX adapter provides numerical storage and kernels. A domain-specific logical type may instead need scalar validation, semantic parameters, operation signatures, and serialization metadata. Reserve those responsibilities for a logical-type registry, with adapter capabilities attached separately. Type identity must include immutable semantic parameters and a stable type identifier, not the Python class or storage backend alone.

The first registry remains closed to Int64, Boolean, and UTF-8. Unknown types and unsupported operations raise; there is no implicit object-array conversion. Do not publish a registration API until a small custom-type experiment tests round-trip metadata, selection, missing values, and unsupported computation. Existing pandas [extension types and conformance tests](https://pandas.pydata.org/docs/development/extending.html) are an integration reference, not a requirement to make every type NumPy-backed.

## Experiments after the vertical slice

| Experiment | Required observation | Boundary it tests |
| --- | --- | --- |
| W16 restricted groupby-sum | Correct key classes, repeated-row multiplicity, missing keys, value nulls, explicit result ordering | Key semantics versus elementwise comparison |
| Bounded reduction reads | Compare full materialization with selected batches across chunk boundaries; report temporary memory and native call count | Whether scalar/full-array access leaves a useful middle ground |
| Axis-cost sweep | Run the same selection chain with RangeIndex, explicit integer labels, and string labels; include construction and retention | Whether eager axes erase payload savings |
| pandas integration | One narrow adapter and one real downstream consumer; count conversions and verify dtype/null/mutation behavior | Whether isolated runtime gains survive interoperability |

Begin bounded reads with integer sum. For floating reductions, batching may change accumulation order; require an explicit tolerance/determinism contract before comparison. This experiment processes existing logical data and does not introduce a query plan. Preserve the materialize-then-native baseline.

Use measured axis costs before introducing an indirect index representation. RangeIndex and explicit labels need not have identical physical strategies. Keep positional coordinates separate from label lookup, and preserve duplicates in every supported structural operation.

For integration, start with detached exports already present in the prototype. A later Arrow PyCapsule or pandas-facing adapter must preserve ownership and report when it materializes or changes encoding. [Arrow PyCapsule](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) defines an interchange mechanism, not full pandas behavior. A result limited to Arrow payload selection may justify an incremental pandas change instead of a replacement runtime.

Do not expand the first implementation to complete all these experiments at once. The next planned implementation step after M0–M3 is the narrow JAX validation, followed by structural workloads and these targeted extensions. Stop expanding scope if the matched eager baseline captures the benefit with less complexity.
