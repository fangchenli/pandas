# Missing-value semantics and PDEP-16

Date: 2026-11-12. Status: research/design refinement; the initial scalar and operation rules are covered by prototype tests. Float and 2D experiments remain pending. This document complements the [logical type design](logical_types.md) and [prototype](prototype.md).

PDEP-13 informs type identity and construction. PDEP-16 supplies a complementary missing-value direction. Our runtime needs both, while keeping semantic rules independent of storage, device, and representation strategy.

## Source and status

[PR 58988](https://github.com/pandas-dev/pandas/pull/58988), opened by Joris Van den Bossche, is titled *PDEP-16: Consistent missing value handling*. It was open and unmerged when inspected. The revision-1 document names Patrick Hoefler and Joris Van den Bossche as authors, dates its creation to March 2024, and says “Under discussion.” Its abstract, scalar details, semantic details, compatibility section, and timeline still contain placeholders. It is an incomplete proposal, not an accepted pandas specification.

The full 107-line source was read and matched to the PR diff: repository `phofl/pandas`, commit `d7dcbdb90d4d2e1658c34cff16dc99990aa6cdb0`, path `web/pandas/pdeps/0016-consistent-missing-values.md`, blob `a74f7c42eef4f939cf79a2477f9875332789a6fa`. [Pinned source](https://github.com/phofl/pandas/blob/d7dcbdb90d4d2e1658c34cff16dc99990aa6cdb0/web/pandas/pdeps/0016-consistent-missing-values.md).

All substantive bodies in the connector's 28 returned discussion/review entries were read. The linked historical threads in issues 32265 and 58243 were not reviewed in full. Discussion positions below are attributed to their authors; they do not establish consensus or acceptance.

## What the draft proposes

The draft proposes nullable types throughout pandas, a single public `pd.NA` indicator, and consistent nullable behavior. It leaves masks, sentinels, bitmap layouts, and Arrow use to implementations. Dtype-constructor design is delegated to PDEP-13. It also proposes preserving 2D block capability through 2D ExtensionArrays. For compatibility, its interim policy treats NaN and NaT as missing on input and through operations, leaving a future NaN/NA distinction to separate discussion. [Proposal](https://github.com/phofl/pandas/blob/d7dcbdb90d4d2e1658c34cff16dc99990aa6cdb0/web/pandas/pdeps/0016-consistent-missing-values.md).

## Consequences for our semantic contract

Keep one prototype `NA` singleton for observable missing scalar results across all supported types. It is a runtime-owned scalar; pandas adapters translate it to and from `pd.NA`. A single missing indicator does not require identical physical encoding, and it does not by itself decide whether a valid floating NaN is missing.

The first three logical types remain nullable Int64, Boolean, and UTF-8. Missing insertion preserves type identity. NumPy/JAX adapters can use values plus validity; Arrow keeps native validity. In selection, structural presence and source validity combine before a scalar is exposed. No separate sentinel should leak from a particular backend.

Specify rules per operation rather than treating null propagation as one universal mask-AND rule:

| Operation | Prototype contract or scheduled requirement |
| --- | --- |
| Scalar access | Return the runtime's `NA` for missing positions and source nulls |
| Int64 addition and comparisons | Propagate missing operands; comparison result is nullable Boolean |
| Boolean filtering | Missing predicates do not select a row |
| Integer sum | Explicit `skipna` and `min_count`, including empty/all-missing inputs |
| Scalar truth testing | `bool(NA)` raises `TypeError` |
| Boolean AND/OR, when added | Kleene logic: `False & NA` is false and `True | NA` is true; unknown outcomes remain NA |
| Further arithmetic, when added | Bind exceptions explicitly; do not infer every operator's behavior from addition |

These are our concrete requirements, informed by existing pandas behavior rather than completed detail in the PDEP. The pinned pandas [missing-data guide](../../doc/source/user_guide/missing_data.rst) documents scalar truth ambiguity, propagation exceptions, and reduction behavior; the [Boolean guide](../../doc/source/user_guide/boolean.rst) supplies filtering and Kleene truth tables. Boolean operators remain outside the first computation milestone, but their eventual semantics constrain the dispatcher now.

## Float policy is a separate decision

The discussion exposes a real tradeoff. Marco Gorelli raises the cost of checking for NaNs beyond maintaining validity and argues for distinguishing them. Will Ayd describes the compatibility rationale for interpreting historical NaN assignment as missing. These are useful arguments, not performance measurements. [Cost/semantic concern](https://github.com/pandas-dev/pandas/pull/58988#discussion_r1638029519), [compatibility rationale](https://github.com/pandas-dev/pandas/pull/58988#discussion_r1638298747).

The draft predates relevant behavior already documented in our pinned main revision `f1cc1a1bc979b83e641facf72ec6526593d61420`. Its [3.0 release notes](../../doc/source/whatsnew/v3.0.0.rst) describe `future.distinguish_nan_and_na`: enabled, NaN is a floating value distinct from NA and is rejected by integer dtypes. The local [configuration](../../pandas/core/config_init.py) defaults it to false, and [FloatingArray documentation](../../pandas/core/arrays/floating.py) confirms the distinction. This source check supports those specific claims; it is not a full runtime conformance test or validation of every later PR comment.

Float64 remains deferred. Before enabling it, choose a coherent policy for construction, assignment, generated NaNs, comparisons, missing predicates, reductions, backend conversion, and export. A separate validity channel permits distinguishing NaN from NA; that is an architectural capability, not an adopted float policy. Compare the compatibility policy with a distinct-NaN policy in an isolated extension experiment, recording the pandas oracle option explicitly. Avoid changing interpretation when a column changes backend. A plain floating NumPy export that collapses NA into NaN must be explicitly requested; a values-plus-validity export can preserve the distinction.

## Preserve the ability to evaluate 2D storage

Joris explains that adopting nullable arrays should not implicitly remove pandas' 2D blocks. Brock Mendel describes a broader 2D EA interface and later highlights arithmetic, row-wise operations, and conversion to NumPy as motivating workloads. Their comments differ on how widely 2D support should apply. [Joris's clarification](https://github.com/pandas-dev/pandas/pull/58988#discussion_r1636961138), [Brock's interface view](https://github.com/pandas-dev/pandas/pull/58988#discussion_r1637371150), [workload rationale](https://github.com/pandas-dev/pandas/pull/58988#discussion_r2193035821).

Our first implementation still uses one logical value cell per column. That is a deliberate experimental restriction, not evidence that physical grouping is unnecessary. Multiple physical column views can retain a common owner, so the ownership model should not assume one allocation per column. Grouped execution and its coordinate/mutation rules need a separate design if measurements justify it.

Add W15 controls for homogeneous numerical frames: whole-frame arithmetic, row-wise sum, transpose followed by consumption, and export to a contiguous 2D NumPy array. Compare column-wise execution with a direct 2D values-plus-validity reference and pinned pandas where semantics match. Include all-valid and null-heavy cases, contiguous and gathered rows, and writes to one column while sibling views remain live. Charge grouping, validity transformation, materialization, and export costs. Begin with Int64 where supported; repeat with Float64 only after its contract is fixed.

These controls are required before claiming the architecture is a general replacement for pandas. They need not block the initial gather experiments. Do not mandate 2D Arrow storage, add a BlockManager replacement, or promise a win before running them.

## Design and evaluation changes

Keep null semantics in `TypeRules`/operation signatures, native validity in adapters, and structural presence in row maps. Add scalar-NA and cross-backend behavior cases to M0/M3; retain Kleene operations and Float64 as explicit extensions. Extend the evaluation with W15 and record the source/status distinction between PDEP-16, discussion comments, and pinned pandas behavior. The original user proposal remains unchanged.
