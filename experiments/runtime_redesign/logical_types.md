# Unified logical types for the prototype

Date: 2026-11-12. Status: design refinement following review of PDEP-13; the initial three-type rules are implemented in the prototype. This document complements the [prototype design](prototype.md).

The prototype should have a unified logical type system from its first implementation, with only three initially implemented types. “Small set of types” must not mean “backend-specific type inference with common names.” Constructors, operations, representation changes, mutation, and exports should all use the same logical type objects and rules.

PDEP-16 is a complementary reference for missing-value behavior. The [missing-value review](missing_values.md) records its pinned source, incomplete/open status, and implications for operation signatures and 2D evaluation. Our physical null encodings remain backend-specific.

## PDEP-13 source and status

Will Ayd's [PDEP-13 pull request](https://github.com/pandas-dev/pandas/pull/58455) was opened on 2024-04-27 and closed unmerged on 2025-04-21. The author said the work was inactive when closing it; closure should not be described as a formal rejection of its principles. The source header still says “Under discussion,” which is distinct from the PR's current state. [Closure comment](https://github.com/pandas-dev/pandas/pull/58455#issuecomment-2817496880).

The inspected text is revision 3, at commit `af9da9771ce739c95c7c4c5df922d9f2545ed7c8` in `WillAyd/pandas`, file `web/pandas/pdeps/0013-logical-type-system.md`, blob `b9fcd533cf1e7355d7d15ae7e0559ea50b85f045`. The full source was read and checked against the PR diff. Selected discussion comments were reviewed; the entire 301-entry discussion was not reviewed. [Pinned proposal](https://github.com/WillAyd/pandas/blob/af9da9771ce739c95c7c4c5df922d9f2545ed7c8/web/pandas/pdeps/0013-logical-type-system.md).

The draft proposes consistent dtype construction and logical result types while allowing storage to change. It reuses existing pandas dtype classes where practical, proposes additional types, exposes storage/physical-type/missing-value metadata, and discusses an NA default with a legacy option. It does not supply a complete promotion or casting calculus. [Proposal text](https://github.com/WillAyd/pandas/blob/af9da9771ce739c95c7c4c5df922d9f2545ed7c8/web/pandas/pdeps/0013-logical-type-system.md).

## The useful lesson from the discussion

The key distinction is between a common spelling for a dtype and a common behavioral contract. Maintainers discussed matching important behavior across storage implementations, documenting exceptions, and making dtype construction consistent. They also distinguished storage from computation and observed that a logical type need not have an implementation in every backend. These are discussion positions, not adopted specifications. [Brock's summary](https://github.com/pandas-dev/pandas/pull/58455#issuecomment-2284757244), [Joris's response](https://github.com/pandas-dev/pandas/pull/58455#issuecomment-2284859591), [Joris's summary](https://github.com/pandas-dev/pandas/pull/58455#issuecomment-2284908678).

For this experiment, choose a stricter, smaller contract: every supported operation has defined logical behavior, and any backend path must satisfy it or decline the operation. Backend differences cannot silently determine a result dtype or missing-value policy. A documented numerical tolerance may eventually be part of the float contract; an unspecified backend-dependent result is not.

## Where the prototype follows and differs

| Concern | Prototype decision |
| --- | --- |
| User-facing dtype | A canonical logical type object, independent of storage |
| Result dtype | Determined by logical operation rules before physical dispatch |
| Storage choice | Separate construction/execution policy; explicit overrides remain useful for experiments |
| Storage metadata | Inspectable on the physical representation, excluded from logical equality and hashing |
| Missing behavior | One NA policy initially; legacy modes are out of scope |
| Supported families | Int64, Boolean, and UTF-8 initially; broader families require actual semantic and backend support |
| Multiple backends | Optional per logical type; NumPy strings are not required for the first milestone |
| pandas dtype classes | Used by oracle and import/export adapters; no subclassing requirement for this standalone runtime |
| Promotion and casting | Explicit interfaces with a deliberately small supported rule set |

Separating storage metadata from the dtype object is our architectural choice. The draft exposes that metadata through dtype construction; it does not define the exact internal identity model we specify below. Likewise, the prototype's independent type classes are not an implementation of the draft's pandas migration plan.

## Type identity and physical metadata

Use one immutable, hashable class per type family, with semantic parameters as named, keyword-only, validated fields. A `TypeKind` enum supplies the stable family identifier, and a derived structural key gives the class-independent identity used for serialization:

```text
SignedInteger(bits=64)  key: (signed_integer, ((bits, 64),), NA)
Boolean()               key: (boolean, (), NA)
Utf8()                  key: (utf8, (), NA)
```

Each class shares a `missing_policy` field from the `LogicalType` base, which is abstract. A family rejects parameters it cannot support at construction: `SignedInteger(bits=32)` raises until 32-bit integers have operations and backend support. This follows the pyarrow and pandas precedent of typed families with named parameters, such as a timestamp's unit and timezone, rather than an untyped kind string with positional parameters.

All initial types admit missing values. A column currently containing no missing values still has the same dtype. There is no schema-level non-nullability guarantee in this milestone, so selection/reindexing can introduce missingness without changing its dtype.

A family's fields contain only semantic parameters and are recursively immutable. Future examples include timestamp unit/timezone, decimal precision/scale, or a list's element type. These are examples for the identity model, not accepted initial constructor arguments. Do not create apparently supported types whose operations all fail later.

`missing_policy` names behavior, not a pointer to `np.nan` or a physical null encoding. Initially its sole supported value is `NA`; its rules are the prototype's declared nullable semantics. If a legacy policy is later implemented, its distinction must survive type equality, operation binding, and serialization.

Define these separate notions:

- **Logical equality:** same kind, semantic parameters, and missing policy. `dtype == other_dtype` uses this relation.
- **Operation compatibility:** an explicit rule exists for the operation and operand types. Equality does not imply support for every operation.
- **Physical compatibility:** an adapter can interpret a particular storage encoding as that logical type without altering its semantics.
- **Conversion permission:** a declared conversion rule exists for the requested casting mode; it is not inferred from whichever backend accepts the data.

The physical representation carries backend, device, encoding, chunking, offsets/strides, validity encoding, and buffer ownership. These fields never participate in logical type hashing. Integer storage in NumPy and Arrow can therefore have equal logical dtypes while requiring different physical dispatch.

Keep overflow and reduction defaults in the operation contract rather than inventing a separate dtype for each algorithm. If such options become user-configurable, include them in the operation binding/cache key.

## Constructors and aliases

Provide small factory functions and canonical aliases:

```python
int64()  # SignedInteger(bits=64)
boolean()
utf8()

a = rt.array([1, 2, None], dtype=int64(), backend="numpy")
b = rt.array([1, 2, None], dtype=int64(), backend="arrow")
assert a.dtype == b.dtype
```

The existing prototype spellings `"int64"`, `"boolean"`, and `"utf8"` normalize to these same objects. In this research API `"int64"` is nullable; it is not a promise to reproduce the meaning of the same constructor string in every existing library. Generic NumPy/PyArrow dtype-object acceptance and inference are deferred until explicit normalization rules exist.

Inferred Python scalar types must not accidentally leak into dispatch. Bind a scalar to its operand's logical domain with range/type checks before constructing a kernel argument. For example, adding Python `1` to Int64 validates it as int64; `True` is not accepted as an integer, and a Python integer outside the int64 range raises. Missing scalar inputs bind to the expected nullable domain. Backend-native scalars are unboxed by the adapter that owns their type before validation: NumPy scalars via `item()`, and Arrow scalars via `as_py()`, with a null Arrow scalar becoming `NA`. The type layer validates only canonical Python scalars and imports no backend, so a new adapter supplies its own unboxing without changing `types.py`. Unboxing never bypasses validation: `np.float64(5)` or `pa.scalar(True)` is still rejected for Int64, and a nested scalar is rejected rather than treated as a sequence.

Keep the explicit `backend=` constructor argument for controlled experiments. It selects physical storage and does not modify `dtype`. The initial CPU NumPy/Arrow dispatch rule may change storage later. Experimental JAX/device transfers require an explicit conversion. The result still carries the type established by the logical rules.

## Operation binding before backend dispatch

Add a semantic binding stage:

```text
normalize dtype aliases and scalar arguments
    → resolve axes and positional alignment
    → bind operation against logical operand types and explicit options
    → obtain result logical type and semantic contract
    → choose physical conversion and kernel path
    → execute eagerly
    → verify and attach the declared result type
```

Use a small `TypeRules` registry rather than placing backend dispatch in dtype subclasses:

```text
normalize_dtype(spec) -> LogicalType
validate_scalar(value, expected_type) -> TypedScalar
bind(op, operand_types, options) -> OperationSignature
common_type(types, context) -> LogicalType
resolve_cast(source_type, target_type, mode) -> CastRule
```

`OperationSignature` contains operand logical types, result logical type, scalar/array result shape category, and normalized semantic options. It does not select a backend. Lengths and label alignment are checked by the logical array/frame layer. A physical adapter receives this signature and cannot replace it with a backend-inferred dtype.

Initial bindings:

| Operation | Input types | Result type | Contract |
| --- | --- | --- | --- |
| Take/slice/filter | Any supported type | Same type | Preserve order, duplicates, and validity |
| Reindex with missing fill | Any supported type | Same type | Missing positions become NA |
| Add | Int64, Int64 | Int64 | Wrapping arithmetic; missing propagation |
| Equal | Two of the same supported type | Boolean | Missing propagation |
| Less than | Int64, Int64 | Boolean | Missing propagation |
| Sum | Int64 | Int64 scalar | Explicit skipna/min_count and wrapping accumulator |
| Concatenate | Identical logical types | Same type | Preserve order; no implicit promotion |
| Assign into a column | Target type and validated values of that type | Target type | Reject incompatible input before mutation |

The first `common_type` implementation accepts only identical logical types. The first cast rule accepts identity conversion; other conversions explicitly raise an unsupported-rule error. This is intentionally limited, but the interface prevents backend fallbacks from becoming an accidental promotion system. Physical conversion between NumPy and Arrow under the same logical type is not a logical cast.

Column replacement is distinct from assignment into an existing column: a replacement Array may carry a different supported logical type if its length matches. The frame schema then changes for that slot. The prototype should not accidentally apply the no-upcasting assignment rule to whole-column replacement.

## Result types must survive representation changes

Every `ValueCell` holds a `LogicalType`. `Select`, `Constant`, `Segments`, and `Physical` must agree with it. Materialization replaces the cell's physical state but preserves logical type identity. Backend conversion does the same. `Segments` resolves its logical result type before accepting children; a future conversion cannot be postponed if it changes error timing or values.

For example, two integer columns with identical labels but different storage backends have this path:

```text
bind(add, (Int64, Int64), options) → Int64 result
physical selection              → materialize each selected input
mixed-backend policy            → convert the NumPy input to the default (Arrow) storage
native addition                 → values and validity
new ValueCell                   → dtype Int64, physical Arrow storage
```

The physical dispatcher may choose a different implementation in a later experiment without changing the binding. A string equality result is Boolean regardless of the string storage encoding. Broader result-type examples can be added as operations enter scope; they are not prerequisites for the first prototype.

## What metadata needs to be observable

Expose `array.dtype` as the logical type and add a diagnostic `array.storage_info()` for current backend/encoding/representation metadata. Diagnostics must not materialize the array simply to answer which representation it holds. A segmented value may report several physical sources rather than inventing one authoritative backend.

Keep construction preference and observed storage separate. A request to start an integer array on NumPy is not a permanent property of its dtype. This allows backend experiments while retaining predictable logical equality.

Serialize a versioned logical type descriptor alongside values and axes. Physical encoding information may describe how the payload is stored, but loading it must reconstruct the declared logical semantics or fail. The serialization version protects future changes to missing-value or scalar policies.

## Acceptance tests and implementation changes

Add `type_rules.py` next to `types.py`. `types.py` owns canonical logical types, aliases, and scalar validation; `type_rules.py` owns operation signatures, common-type resolution, and cast rules. No backend import is required to compare or hash logical types.

Before representation benchmarks, require:

1. The same logical type compares and hashes equally across backend, slicing, gather, materialization, compaction, and export/import round trips.
2. Missing insertion preserves the dtype of an initially complete nullable column.
3. Every supported operation has the same logical result dtype and null behavior in eager, compact, and tree modes.
4. Unsupported type combinations fail before payload mutation or an implicit backend conversion.
5. Mixed-backend addition returns the declared Int64 result even when its physical output differs from an input.
6. Scalar range checks, integer wrapping, equality/ordering, and sum options agree across adapter implementations.
7. Whole-column replacement can change a schema slot; incompatible in-column assignment cannot.
8. Type metadata survives serialization independently of backend encoding.
9. Experimental JAX operations preserve Int64/Boolean semantics and reject unavailable 64-bit support rather than narrowing; device and namespace do not enter type identity.
10. Arrow validity remains native, while numerical adapters combine masks and structural presence without changing logical null behavior.
11. Missing scalar reads expose the same `NA` singleton across adapters, scalar truth testing raises, and pandas conversion maps the singleton to/from `pd.NA`. Future Boolean operators must pass Kleene truth tables rather than reuse the addition validity rule.

These tests are a small semantic system, not a requirement to implement every type in PDEP-13. Add one type family or promotion rule at a time, with its oracle and backend capability tests. The first implementation remains Python; this layer is small metadata and rule resolution, not a reason to introduce a Rust core.
