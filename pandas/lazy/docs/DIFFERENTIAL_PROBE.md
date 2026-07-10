# Differential substrate probe — the standing finding instrument

**What:** `pandas/lazy/benchmarks/differential_probe.py` — a fixed workload grid
run identically across `{pandas, polars, acero, datafusion-sql, datafusion-df}`
that emits a **divergence report**. Every diverging cell is a *candidate finding*
the probe surfaces automatically; a human confirms, root-causes, dup-searches,
and (per the guardrail) files. This is the substrate probe's *discovery* engine —
it replaces the hand-built, thrown-away diff matrices that produced AG10/AG11/AG4.

**Why it exists:** our best findings all have one shape — run a logical op, vary
one axis, watch something diverge:

| finding | the differential |
|---|---|
| AG10 | SQL plan **vs** DataFrame-API plan (same query) |
| AG11 | cross join **vs** hash/NLJ (same inputs) |
| AG3/AG5′ | 1 core **vs** N cores (same op) |
| AG4 | raw string key **vs** dict-encoded key (same agg) |
| AG9/AG1/AG12 | type X works **vs** type Y NotImplemented |

Each was a manual matrix built once and discarded. This makes the matrix
*standing*, so the next find is generated, not accidental.

## Divergence classes (most-valuable first)

- **CRASH** — one engine **raises or panics** (incl. a Rust `PanicException` from
  pyo3) on an input the others handle. The loudest divergence; a crash on valid
  input is almost always an upstream bug. Surfaced by the degenerate-input
  (empty / all-null / null-key) edge sweep, each engine wrapped so a crash is a
  *finding*, not the end of the run. This is how **AG13** (datafusion-python
  `register_record_batches` panics on an empty table) was found, and how an
  AG11-class crash gets caught automatically.
- **RESULT** — engines disagree on the *answer*. Highest value (a bug, not an
  enhancement). This is the surface our ad-hoc perf matrices never covered.
  Compares *values*, not dtypes (polars count → `uint32`, pandas → `int64` is
  **not** a divergence).
- **PLAN** — datafusion SQL vs DataFrame-API **optimized plans** differ in
  *structural* operator shape (Aggregate/Join/Distinct/Union/Window). This is the
  AG10 class: an optimizer rule fires for one front-end only. Gated to structural
  nodes — a lone Projection/alias delta is benign builder noise and is **not**
  flagged.
- **COVERAGE** — an Arrow compute kernel works on one type but is
  `NotImplemented` on a **sibling** type that peers handle (the AG9/AG12 class).
  `kernel × input-type → ok / NotImpl`, findings summarized per sibling type.
- **PERF** — one engine is >Nx slower on the same cell (default 3×). The AG3/AG4/
  AG5′ class. Reports the overall spread plus two named axes we specifically hunt:
  `acero-vs-polars` and `df-api-vs-sql`.

## Run it (on the *current* releases, not the repo's pinned pyarrow)

```bash
python -m venv /tmp/upstream-venv && . /tmp/upstream-venv/bin/activate
pip install -U pyarrow polars datafusion numpy pandas
cd /tmp && python <repo>/pandas/lazy/benchmarks/differential_probe.py
#   --rows N            rows per synthetic table (default 2,000,000)
#   --perf-threshold X  min slowdown ratio to flag PERF (default 3.0)
#   --only SUBSTR       filter to workloads whose id contains SUBSTR
```

Run from **outside** the repo so the source `pandas/` doesn't shadow the built
one. The whole point is the *current* substrate — never benchmark upstream claims
against the repo's pinned pyarrow 23.

## First-run validation (2026-07-09 — pyarrow 24.0.0 / polars 1.42.1 /
## datafusion 54.0.0 / pandas 3.0.3, 2M rows)

The instrument **rediscovered three already-known findings from scratch,
unprompted** — the proof it would have caught them:

- **AG10** → 6 PLAN findings, `Aggregate: sql=2 df=1` on every `count_distinct`
  (SQL gets the `single_distinct_to_groupby` nested double-aggregate; the
  DataFrame API keeps a lone `distinct: true`). Cross-confirmed on the perf axis
  (df 86ms vs sql 44ms at int64/1M).
- **AG4** → acero raw string-key `sum` 8–12× slower than the fastest engine.
- **AG5′** → acero `count_distinct` 7–8× slower than polars across cardinalities.

It also flagged pandas as the laggard in several low-cardinality cells (expected;
pandas is the oracle, not a target).

### First *new* find — the RESULT/CRASH surface paid off immediately (2026-07-09)
Widening to the degenerate-input edge sweep (`empty` / `all_null_group` /
`null_key` × sum/count/mean/min) on the same releases produced, on its first run:

- **AG13 (CRASH, new upstream find)** — `datafusion-sql` and `datafusion-df` both
  **panic** (`PanicException: index out of bounds`) on an **empty table**, where
  pandas/polars/acero return an empty result. Root-caused to one line in
  datafusion-python (`register_record_batches` indexes `partitions[0][0]`
  unchecked); non-dup (sibling of the fixed #575). Hand-off:
  `upstream/AG13-datafusion-register-empty-panic.md`.
- **Two RESULT lowering-hazards** (semantics, not upstream bugs): `sum` of an
  **all-null group** → datafusion + acero return NULL, pandas + polars return
  `0.0`; **NULL in the group key** → datafusion/acero/polars keep the null group,
  pandas drops it (`dropna=True`). These are `translate_datafusion.py`
  correctness hazards — a pandas→engine lowering silently changes the answer —
  so they're worth tracking even though they're documented SQL-vs-pandas choices,
  not bugs to file. `count`/`mean`/`min` on the all-null group agreed everywhere
  (correctly *not* flagged).

### Two harness weaknesses it caught in *itself* on run 1 (now fixed)
The instrument's first output was audited before trusting it — and it exposed two
of its own detector bugs, exactly what a finding tool must not ship with:
1. **RESULT false positives** — dtype-strict comparison flagged `uint32` vs
   `int64` counts as "disagreement" though the values were identical. Fixed:
   compare values via `np.array_equal`/`allclose`, ignore dtype.
2. **PLAN false negative** — parsed the one-line Rust `Debug` repr and scored
   every plan 0, so it would have **missed AG10**. Fixed: use the canonical
   `optimized_logical_plan().display_indent()` tree.

The lesson is baked into the doc because it generalizes: **audit the instrument's
findings against known ground truth before believing a new one.**

## Coverage today, and the axes to widen (all pure finding work)

Implemented: (1) **grouped aggregate** (`sum`, `count_distinct`) × key dtype
(`int64`, `string`) × cardinality (100 / 10k / 1M) — the AG3/AG4/AG5′/AG10
surface; (2) a **degenerate-input edge sweep** (empty / all-null / null-key) —
the RESULT/CRASH surface that found AG13; (3) a **join-shape matrix**
(inner/left/full/fanout/null-key/cross/semi/anti/self) — the AG11 axis; (4) a
**kernel type-coverage matrix** — the AG9/AG12 axis.

The coverage matrix (added 2026-07-10) quantifies the AG9 gap on its first run:
of 17 scalar string kernels (`utf8_length`, `match_substring`, `starts_with`,
`match_like`, `find_substring`, `replace_substring`, …), **all 17 work on
`string`/`large_string` but are `NotImplemented` on both `string_view` and
`dict<string>`** on pyarrow 24.0.0. This is exactly the AG9 area (tracked under
umbrella #44336) — the matrix turns "view kernels lag" into the precise 17-kernel
list AG9-next needs to scope its PR, and each cell auto-flips `NotImpl→ok` as a
view-kernel sub-PR lands (so re-running measures #44336's progress release over
release). Add a family = add a `(types, kernels, reference)` row to
`COVERAGE_FAMILIES`; next families to add: temporal-unit casts (the AG12 floor
axis) and decimal arithmetic.

The join matrix (added 2026-07-10) already earns its keep on the first run:
- **CRASH** — its `cross_count_meta[AG11]` case (a `count` aggregate over a
  metadata-carrying cross join) **reproduces AG11**: both datafusion front-ends
  raise `Internal error: Physical input schema should be the same…` where
  pandas/polars succeed. It's encoded as a standing tracker — when AG11's fix
  (datafusion PR #23442) ships, this cell auto-flips to OK, retiring the finding
  with zero manual re-verification. (Pinning the trigger took care: a plain
  cross-join `collect` masks it; only an *aggregate over* the cross join trips
  the schema check. acero can't express a keyless cross join, correctly reported
  `unsupported` — not a crash.)
- **RESULT (lowering hazard)** — `null_key_inner`: **pandas matches `NULL==NULL`
  on a join key** (yielding an extra matched row), while polars/acero/datafusion
  all use SQL semantics (null never matches). Lowering a pandas merge on a
  null-containing key to any of these engines silently drops the null-null
  matches — a `translate_datafusion.py` correctness hazard. inner/left/full/
  fanout/clean-cross/semi/anti all agreed (correctly not flagged).
- **CRASH (self-join, tracked upstream)** — `self_join`: **datafusion-df raises
  `Schema contains duplicate qualified field name t.k`** joining a table to
  itself, where pandas/polars/acero/datafusion-**sql** all succeed. This is the
  exact q16/q21 lowering issue (why `translate_datafusion.py._join` renames the
  right keys to temporaries). Already tracked upstream — **open #14147** (+
  regression #14112) — so *attach data, don't file*; the cell is a standing
  fix-tracker that flips to OK when #14147 lands.

Both `cross_count_meta[AG11]` and `self_join` double as **cross-version
regression/fix trackers**: re-run on a new datafusion release and a cell flipping
CRASH→OK is the fix landing, OK→CRASH a regression.

Extend by adding entries to `AGGS` / `build_dataset` / `build_edge_cases` /
`build_join_cases`:

- **RESULT surface** — the highest-value, least-covered. Add ops where engines
  plausibly disagree: null handling in agg, empty groups, NaN vs null, integer
  overflow in `sum`, string collation in `min/max`, timezone/temporal casts
  (AG12 lives here). A result-diff find is a *bug*, the strongest hand-off.
- **PLAN surface** — widen the DataFusion SQL-vs-DataFrame sweep. AG10 came from
  7 optimizer rules with only 1 diverging; DataFusion has dozens. Add join,
  filter-pushdown, limit-pushdown, distinct, window workloads and diff the plans.
- **Join-shape matrix** — the AG11 axis (cross vs hash vs NLJ, metadata on/off).
  A natural second operator family.
- **Kernel type-coverage matrix** — the AG9/AG12 axis: op × input-type →
  {ok, NotImplemented, slow}. Mechanically enumerable; each `NotImplemented` cell
  is a candidate.
- **Cross-version diff** — re-run against each new pyarrow/datafusion release and
  diff the report: auto-retire hand-offs that got fixed, auto-surface
  regressions. Directly kills the "re-verify on latest at file time" tax. A cron
  could own this.

## Guardrail
The probe **finds and characterizes**; it never files. Filing/implementation is a
separate track (a different agent) and still requires explicit human go-ahead per
`upstream/README.md` §4. This instrument's output is the *input* to that track.
