# Arrow `string_view` Kernel Contribution Plan

A self-contained work plan for contributing `Utf8View`/`BinaryView`
("German strings") compute-kernel support to Apache Arrow C++ — written to
hand to an engineer/agent with no prior context. Researched June 2026
against `apache/arrow` main (`fea7897`), pyarrow 23.0.1/24.0.0, and a local
kernel probe.

## Why (the motivating measurement)

The lazy-pandas engine is memory-bandwidth-walled on Arrow `large_string`:
`take` on a 10M-row string column is ~271–375 ms vs Polars' ~92 ms (4x) —
Polars wins by *moving less data* via 16-byte inline views. The
`string_view` type exists in Arrow, and pandas/parquet round-trip it, but
the compute kernels we need all throw `ArrowNotImplementedError` on
pyarrow 23 AND 24 (locally verified): `array_take`, `array_filter`,
`equal`*, `is_in`, `match_substring`, `sort_indices`, group-by keys
("Keys of type string_view"), acero join keys ("not supported in join key
field"). (*comparisons merged upstream 2026-06-09, ship in pyarrow 25.)

This is the single structural item capping string-heavy queries
(filter_project 0.21x, H2O join q4 0.13x, TPC-H q13/q16) — everything else
in the engine gap is ordinary downstream engineering.

## Upstream state (June 2026)

- Umbrella issues: [GH-39634](https://github.com/apache/arrow/issues/39634)
  (original basic-functionality tracker),
  [GH-44336](https://github.com/apache/arrow/issues/44336) (kernel-parity
  umbrella, stale-bot pinged Nov 2025).
- `take`/`filter`: [GH-43010](https://github.com/apache/arrow/issues/43010)
  open, assigned felipecrv, **no PR**; felipecrv invited "a simple
  implementation" (July 2024 comment) after his selection-kernel refactor
  ([PR #41700](https://github.com/apache/arrow/pull/41700)).
- Comparisons: [PR #49964](https://github.com/apache/arrow/pull/49964)
  (contributor: Periecle) merged 2026-06-09 → pyarrow 25 (~July 2026).
  Periecle's stated sequence: comparisons → set-lookup → take/filter.
- **Group-by keys (Grouper) and acero hash-join keys: NO TRACKING ISSUES
  EXIST.** Nobody upstream is planning them.
- `CompactArray` for view buffers:
  [PR #46229](https://github.com/apache/arrow/pull/46229), stalled since
  Jul 2025 (relevant to take-output buffer retention, not a blocker).
- What already works on views: `cast`, `unique`, `value_counts`,
  `dictionary_encode`, slicing, concatenation, IPC, Parquet.

## The three kernels, technically

### K1 — `array_take` / `array_filter` (small-medium, ~1–2 person-weeks)

Files: `cpp/src/arrow/compute/kernels/vector_selection_take_internal.cc`,
`vector_selection_filter_internal.cc`, `vector_selection_internal.cc`.

The dispatch table (~line 740 of the take file) has entries for
`BinaryLike`/`LargeBinaryLike`/`FixedSizeBinaryLike`/list-view but none for
`BINARY_VIEW`/`STRING_VIEW`. The semantics are favorable: taking a view
array is a **fixed-width 16-byte gather** of the view headers (the existing
`FixedWidthTakeExec` machinery handles FSB16) plus attaching the input's
variadic data buffers to the output `ArrayData` **unchanged** — no string
bytes are copied; this IS the German-strings win.

Care points:
- The kernel framework's output preallocation doesn't know about variadic
  buffers — the exec must attach them manually
  (`out->buffers` beyond index 1 / `ArrayData::buffers` semantics for view
  types: validity, views, then N data buffers).
- ChunkedArray take: chunk concatenation already supports views
  (`Concatenate`), buffers accumulate — correct but unreferenced data is
  retained; this is a *quality* question (see CompactArray), explicitly NOT
  a stated precondition. Note it in the PR.
- Filter is the same story with the filter-to-indices or bitmap path.

### K2 — Grouper (group-by keys) (medium, ~2–3 person-weeks)

Files: `cpp/src/arrow/compute/row/grouper.cc`,
`cpp/src/arrow/compute/row/row_encoder_internal.{h,cc}`.

Two grouper impls:
- `GrouperImpl` (generic): per-column `KeyEncoder`s — there are
  `FixedWidthKeyEncoder`, `VarLengthKeyEncoder<BinaryType/LargeBinaryType>`,
  Dictionary/Boolean/Null encoders. **Add a view-aware KeyEncoder**:
  encode = read the 16-byte header, copy the string bytes into the row
  buffer (same wire format as VarLength); decode = rebuild a `string_view`
  (or plain `string` — get reviewer buy-in on output type; plain string is
  acceptable v1 since group keys in output are small).
- `GrouperFastImpl` (Swiss-table): `CanUse()` (~line 553) fails to exclude
  views, so they enter the fast path and hit `NotImplemented` (~line 599).
  Its `KeyColumnMetadata`/`KeyColumnArray`
  (`compute/light_array_internal.cc`, `ColumnMetadataFromDataType` ~line
  125) hard-assume ≤3 buffers — views are unrepresentable without redesign.
  **v1: make `CanUse()` reject views** so they fall back to `GrouperImpl`.

Note: `hash_min_max`/`hash_sum` etc. on view *payload* columns are separate
kernels (`hash_aggregate*.cc`) — out of scope; `sum(int) GROUP BY
string_view_key` needs only the Grouper fix.

### K3 — acero hash join keys (medium functional / large fast)

Files: `cpp/src/arrow/acero/hash_join_node.cc`, `hash_join.cc`,
`swiss_join.cc`, `compute/row/encode_internal.cc`.

Three layers:
- Gate: `HashJoinSchema::IsTypeSupported` (~line 68) returns
  `is_fixed_width || is_binary_like || is_large_binary_like` — views are
  rejected as key AND payload fields. Relax for views.
- Routing: `use_swiss_join = !HasDictionaries() && !HasLargeBinary()`
  (~line 768) — extend to also avoid Swiss for views, routing to the basic
  `HashJoinImpl` (`hash_join.cc`), which uses the **same
  `RowEncoder`/`KeyEncoder` machinery as `GrouperImpl`** — so the K2
  KeyEncoder makes view-key joins functional nearly for free
  (~1–2 weeks incremental: gate + routing + payload materialization +
  tests).
- SwissJoin fast path: `KeyColumnArray`'s 3-buffer assumption blocks views
  in `compute/row/*` (encode/compare/hash + AVX2 specializations +
  `ResizableArrayData` output). **4–8+ person-weeks, separate project; do
  not attempt in v1.**

## Related ammunition: two upstream crash reports

The SF-100/SF-300 runs hit two process-killing Arrow defects on int32-
offset string data (`take` segfault; acero hash_aggregate abort) — see
[UPSTREAM_ISSUES.md](UPSTREAM_ISSUES.md). File those alongside step 1:
they document production pressure on the small-string types from the same
workloads motivating the view kernels.

## Sequencing (the plan to execute)

1. **File the missing issues first** (1 day): one for Grouper view keys,
   one for acero join view keys, both linking GH-44336; comment on
   GH-43010 + GH-44336 stating intent and coordinating with Periecle
   (avoid duplicating their set-lookup/take plans — if they've started
   take/filter, take K2 first).
2. **K1 take/filter PR** (~1–2 wk): dispatch entries + view exec via the
   fixed-width-16 gather + buffer attach; tests incl. sliced arrays,
   chunked, nulls, empty, multiple data buffers; benchmark in the PR
   description (gather µs/element vs large_string).
3. **K2 KeyEncoder + Grouper PR** (~2–3 wk): view KeyEncoder
   (encode/decode + row-width accounting), `GrouperFastImpl::CanUse` gate,
   group_by tests (string_view keys × {sum,count,min,max,mean} values,
   nulls, empties, high cardinality).
4. **K3 functional-join PR** (~1–2 wk): relax `IsTypeSupported`, extend
   the swiss-routing predicate, payload handling, join tests (inner/left/
   semi/anti, view keys both sides, view payloads, nulls).
5. **Track pyarrow releases**: comparisons land in 25; ours target 26.

## Interim bridge (already usable downstream)

`dictionary_encode` works on view arrays today, and **dictionary keys are
supported by both the Grouper fast path and the hash join**
(`hash_join_dict.cc`). For low/mid-cardinality string keys:
`string_view → dictionary(int32)` once, then gather/group/join on int32
codes — preserving most of the bandwidth win. Degrades toward a full cast
at high cardinality. The lazy engine's keycache/dictionary flow can adopt
this before any upstream work ships.

## Acceptance criteria for the whole effort

- pyarrow (nightly) passes: `pc.take/filter`, `Table.group_by` with a
  `string_view` key, `Table.join` on `string_view` keys — all currently
  throwing calls in our probe script
  (see `pandas/lazy/docs/ROADMAP.md` G5 entry for the probe).
- Measured on our 10M-row gather benchmark: view take within ~2x of the
  16-byte theoretical floor (i.e. ≥3x faster than large_string take).
- Engine integration follow-up (downstream, separate): route string
  columns to `string_view` storage at scan/conversion boundaries and
  re-measure filter_project (0.21x) and H2O join q4 (0.13x).
