# Scoping: string storage migration (string-view vs dictionary)

The remaining H2O performance gaps that survive measurement are all
**string-bound**: the string-key join (q4, 0.25x), the left join with string
payload (q3, 0.34x), and string-payload gathers generally. The cause is
Arrow `large_string`: hashing it is length-dependent and `take` copies the
underlying bytes. Polars/DuckDB avoid this with **German-style string-view**
storage (inline prefix → length-independent hashing; gather copies 16-byte
views, not bytes). This doc scopes adopting that class of fix.

## Feasibility gate (measured June 2026, pyarrow 23.0.1 **and 24.0.0**)

**The literal string-view migration is upstream-blocked.** pyarrow exposes the
`pa.string_view` *type*, but the compute kernels we route through do **not**
accept it — confirmed identical failures on both pyarrow 23.0.1 and 24.0.0
(upgraded and re-tested):

| kernel | `string_view` (pa 23 **and** 24) |
|---|---|
| `pc.take` (gather) | ✗ `array_take has no kernel for (string_view, int64)` |
| `Table.group_by` | ✗ `Keys of type string_view` unsupported |
| `Table.join` | ✗ `string_view is not supported in join key field` |

So we cannot get the string-view benefits through pyarrow's C++ compute layer
today. (arrow-rs *does* have these kernels — that's what Polars/DataFusion
use — but our engine is pyarrow/C++-based. The C++ compute kernels lag the
Rust ones here.) Re-test on each pyarrow upgrade with the probe above; nothing
to do until they land.

**Dictionary encoding delivers the same wins and is supported now:**

| operation (2M rows, 10k distinct) | `large_string` | `dictionary` |
|---|---|---|
| `take` (gather) | 37 ms | **1.3 ms (28x)** |
| `group_by` | works | ✅ works (fast, ~11 ms/10M from prior work) |
| `join` | — | ✅ works |
| one-time `dictionary_encode` | — | 28 ms / column |

Gathering a dictionary column takes the **integer codes**, not the string
bytes — the same structural win string-view's view-copy `take` would give.
Hashing dictionary codes is also length-independent.

## Track A Phase 1 — MEASURED AND DISPROVEN (June 2026)

Before building Phase 1, the end-to-end approach was measured on q3 (left join,
10M, string payload `id4`/`id5`). **It regresses badly — do not implement:**

| q3 left join, 10M | time |
|---|---|
| str payload merge (current) | **182 ms** |
| dict payload, full e2e (encode → merge → decode) | 2344 ms (**13x slower**) |
| dict merge, no final decode | 440 ms (still slower) |
| encode `id5` (str→dict) alone | 179 ms |
| left join numeric-only (no string payload) | 98 ms |
| Polars | ~60 ms |

Why the 28x micro-benchmark did not translate:
- The 28x (`take` on codes) only holds for an **already-encoded** column
  gathered repeatedly. A single join→output query pays the **encode**
  (~180 ms to hash 10M strings into a dictionary) *and* the **decode**
  (~1900 ms to materialize 10M `str` values from codes at output, required to
  match eager). Together they dwarf any gather saving.
- String payload is only **~71 ms** of q3's 169 ms merge (169 with strings vs
  98 numeric-only) — not the dominant cost. And even eliminating it entirely
  (98 ms) would not reach Polars' 60 ms: **half the gap is the join algorithm**
  (Polars' parallel hash join), not the gather.
- This confirms the earlier ROADMAP note: "dictionary `take` is 2.8x only when
  *already-encoded*; one-shot encode erases it."

**Consequence:** dictionary encoding adds nothing over the existing Categorical
path for plain-`str` single-pass queries — the only case it wins (already
zero-copy-encoded input, output stays Categorical, no decode) is exactly what
the Categorical handling already covers. Track A is **not pursued**. The
remaining string gaps are genuinely structural — Track B (string-view),
upstream-blocked.

## Decision (superseded by the measurement above)

Two tracks, sequenced:

- **Track A — dictionary-encode string columns through the engine (do now).**
  Achievable with today's pyarrow; delivers the gather + hash/join wins for
  **low/medium-cardinality** strings (the common case for keys and many
  payloads).
- **Track B — true string-view (upstream-blocked; track only).** Handles
  *all* cardinalities with no dictionary overhead, and is the long-term right
  answer. Blocked until pyarrow C++ adds `string_view` to `array_take`,
  `hash_aggregate`, and `hash_join`. Track the Arrow issues; revisit when it
  ships.

## Track A — design

### What exists today (the foundation is already there)

- `extract_array` (`backends/convert.py:144`) already maps **Categorical →
  Arrow `DictionaryArray`** zero-copy on the codes, and the schema routes
  dictionary-keyed group-by to acero (the M4 "category-key groupby 313→21 ms"
  win). The dtype contract already decodes `dictionary → Categorical` at
  output.
- `arrow_take` (`backends/arrow/core.py:609`) is just `pc.take` — it already
  gathers a dictionary array cheaply; no kernel change needed.
- The key-encoding cache (`backends/keycache.py`) already dictionary-encodes
  *join/group keys*.

So the machinery exists for **Categorical** and for **keys**. The migration is
to extend it to **plain string columns** (`str` dtype / `large_string`),
including **payload** columns that get gathered.

### The four changes

1. **Cardinality-gated dictionary encoding of string columns.** Blanket
   encoding is wrong — a near-unique string column gets a huge dictionary,
   slow encode, and no benefit. Gate on an NDV estimate (reuse
   `optimize/cardinality.py`): dictionary-encode a string column when
   `ndv << rows` (e.g. ndv/rows < ~0.5). Apply at the source/first breaker so
   the encoding is paid once and reused.
2. **Carry codes through the pipeline.** Once encoded, joins, group-bys, and
   `take`s operate on the dictionary array (cheap). This mostly *already*
   works once the array is a `DictionaryArray`; the main work is making the
   planner/decision-layer keep it encoded rather than decoding mid-pipeline.
3. **Decode to `str` (not Categorical) at output.** The dtype contract
   currently decodes any dictionary → `Categorical`. An *internally*
   dict-encoded string column must decode back to `str` to match eager output.
   Add a schema flag (`originated_as_string`) so the contract knows which
   dictionaries to decode to `str` vs `Categorical`.
4. **Dictionary unification at joins.** Two sides may have independent
   dictionaries; acero's join handled separate dictionaries in testing, but
   validate unification correctness (and that left-join null rows decode
   correctly).

### Phasing (each phase measured before the next)

1. **Payload-gather only** (lowest risk, clear win): at a join, dictionary-
   encode low-cardinality **string payload** columns before the gather, take
   codes, decode at output. Targets q3/q4's gather cost directly. Validate q3
   moves toward parity.
2. **String join keys**: dictionary-encode string join keys (cardinality-
   gated) and route the join through acero-on-dictionary; A/B vs `pd.merge`
   on the raw string (which is the current best — so this must *beat* 824 ms
   to ship). Measurement-gated: prior turns showed factorize-to-codes was
   *slower*, so this needs to prove out.
3. **String group-by keys**: already fast via the key cache; fold any
   remaining raw-string group-by into the dictionary path.
4. **Source-level encoding**: dictionary-encode qualifying string columns at
   scan/ingestion so the benefit compounds across a whole query.

### Risks & caveats

- **Cardinality heuristic.** The whole approach is a *loss* on high-cardinality
  strings (big dictionary, encode cost, no gather win). The NDV gate must be
  reliable; mis-gating regresses those columns. Mitigation: conservative
  threshold + fall back to `large_string`.
- **Dtype round-trip correctness.** The `str` vs `Categorical` output
  distinction (change #3) is the subtle correctness risk — easy to
  accidentally return `Categorical` where eager returns `str`. The output
  dtype contract tests are the guard.
- **Encode cost amortization.** 28 ms/column one-time. Worth it only when the
  column is gathered/joined/grouped at least once (usually true for keys and
  join payload; not for pass-through-then-output columns).
- **acero dictionary unification** across join sides — validate, especially
  left-join null handling and the decoded values.
- **Not a universal fix.** High-cardinality string payloads still hit the
  `large_string` wall until Track B. Be honest in the benchmark notes about
  which workloads benefit.

### Effort

Medium-to-large: ~2–4 focused sessions. The kernels and Categorical/key
dictionary plumbing already exist; the new work is (a) the cardinality-gated
encode decision, (b) the schema flag + `str`-vs-`Categorical` output decode,
(c) keeping columns encoded through the planner, and (d) per-phase
measurement. Medium risk, concentrated in the dtype round-trip.

### Expected payoff

- **q4 string join (0.25x)** and **q3 left join (0.34x)**: the string-payload
  gather drops ~28x for low/medium-cardinality columns; should move both
  materially toward parity (exact gain is measurement-gated).
- Any string-heavy gather/join/group on low/medium-cardinality data.
- **Not** high-cardinality string columns — those wait for Track B.

## Track B — true string-view (upstream tracking)

The complete fix. When pyarrow C++ adds `string_view` support to `array_take`,
the hash-aggregate, and the hash-join kernels, string-view handles **all**
cardinalities with **no** dictionary overhead and view-copy gather. Until then
it is not actionable from a pyarrow-based engine.

- **Action:** track the Arrow C++ kernel issues; re-run the feasibility gate
  (the `pc.take` / `group_by` / `join` on `pa.string_view` probe above) on
  each pyarrow upgrade. When they pass, scope the migration of the block
  manager's string representation to `string_view`.

## Recommendation (updated after measurement)

**Do not pursue Track A.** Phase 1 was measured on q3 and regresses 13x
(encode + decode overhead dwarfs the gather saving; the gather isn't even the
dominant cost). The only case dictionary encoding wins — already-encoded input
with Categorical output and no decode — is already handled by the existing
Categorical → `DictionaryArray` path. There is no dictionary "bridge" for
plain-`str` single-pass join/output queries.

The remaining string gaps (q3 0.34x, q4 0.25x) are therefore **genuinely
structural and upstream-blocked**: the fix is Arrow string-view, and pyarrow's
C++ `take`/`group_by`/`join` kernels do not support it (confirmed on 23 and
24). The queries are **correct, just slower**; this is a documented,
bounded gap, not a missing capability.

**Action:** none until pyarrow ships `string_view` compute kernels. Re-run the
feasibility probe (top of this doc) on each pyarrow upgrade; when `take` /
`group_by` / `join` accept `string_view`, scope migrating the block manager's
string representation to it (Track B) — that fix needs no per-column encode or
decode and handles all cardinalities.
