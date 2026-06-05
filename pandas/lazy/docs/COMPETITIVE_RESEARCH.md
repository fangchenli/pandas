# Can Lazy Pandas Compete with Polars/DuckDB? — Research Findings

A deep-research pass (29 sources, 137 extracted claims, 25 adversarially
verified) on the positioning question in
[PROPOSAL.md](PROPOSAL.md#positioning-and-open-questions): what direction, if
any, would justify competing with Polars/DuckDB rather than settling for
"optimization layer for pandas users". Conducted 2026-06-04.

**Headline: the "compete" position is justifiable, but not by building a
faster engine.** The defensible play is *pandas as the universal frontend* —
capture existing eager pandas code into our logical plan, optimize it, and
execute on commoditized best-of-breed engines. That combination is
structurally unavailable to Polars and DuckDB (they require rewrites and are
not the incumbent API), and every component of it has now been validated
independently by someone else.

## What actually makes Polars/DuckDB fast — and what it implies

- Polars' speed decomposes into: a lazy rule-based optimizer (5–10x by
  itself), L1-cache-sized vectors (1024–2048 items) with SIMD, a Rust
  work-stealing thread pool across all cores, Arrow columnar memory, and
  plan-wide schema tracking
  ([endjin](https://endjin.com/blog/2026/01/under-the-hood-what-makes-polars-so-scalable-and-fast)).
  DuckDB: columnar-vectorized execution, lazy-only, automatic spill
  ([CMU 15-721](https://15721.courses.cs.cmu.edu/spring2024/notes/20-duckdb.pdf)).
- We already have the optimizer part. The rest — parallel SIMD kernels,
  morsel-driven scheduling — is a from-scratch engine effort.
- The key strategic fact: embeddable columnar engines (DuckDB, Velox,
  DataFusion) are **commoditizing exactly that layer**. The composable-data-
  systems thesis predicts roughly equivalent vectorized execution everywhere
  within years
  ([CIDR 2021](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper08.pdf),
  [Velox, VLDB'23](https://www.vldb.org/pvldb/vol16/p2679-pedreira.pdf)).
  Hand-building kernels is therefore not a durable advantage — for us *or*
  for Polars.

## Prior attempts at transparent pandas acceleration — post-mortems

| Attempt | Approach | Outcome / lesson |
|---|---|---|
| **cudf.pandas** (NVIDIA) | Import interception → proxy objects, GPU-first with CPU fallback | Production success: 10–50x, zero code change. Solved the chained-call fallback cliff via minimal rewind-and-replay. Unsolved: **silent fallback masks performance cliffs** — only a profiler reveals it; maintainers themselves asked for a disable-fallback mode ([docs](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/how-it-works/), [cudf#15724](https://github.com/rapidsai/cudf/issues/15724)) |
| **Modin** | Compact dataframe algebra (~240 pandas ops → small core) + parallel execution | >85% API coverage with one import change; 19–30x on big multicore groupbys ([Petersohn, VLDB'20](https://vldb.org/pvldb/vol13/p2033-petersohn.pdf)). Proves broad API capture is tractable |
| **Dias** (SIGMOD'24) | JIT rewriting of notebook cells with dynamic precondition checks | Up to 57x per cell, 3.6x whole notebooks; beats Modin by up to 1909x per cell on EDA — **lightweight rewriting beats heavyweight parallel engines for exploratory code** ([arXiv 2303.16146](https://arxiv.org/abs/2303.16146)) |
| **LaFP** (2025) | Drop-in lazy pandas wrapper (2-line change) + JIT static analysis, pluggable backends | **Direct validation of lazy capture**: 3x on pandas / 10x Modin / 19x Dask backends; rescued 7 of 8 out-of-memory programs via pushdown; beat Dask's own optimizer; includes a "Lazy Print" optimization for the repr-materialization problem ([arXiv 2501.08207](https://arxiv.org/abs/2501.08207)) |
| **Ibis pandas backend** | Lazy expression API executed *on* eager pandas | **Deprecated** — "fundamental model mismatch": most pandas-specific code of any backend, NaN/NULL semantics headaches, often slower than calling pandas directly ([farewell-pandas](https://ibis-project.org/posts/farewell-pandas/)) |
| **pandas-API-on-distributed** (Koalas/Spark, Dask) | API parity over lazy/distributed engines | Ordering/iloc semantics cliffs (~15x for positional access), optimization choices that invert across engines, eager-on-lazy duplicated work — API parity without owning semantics is a trap ([post-mortem](https://towardsdatascience.com/why-pandas-like-interfaces-are-sub-optimal-for-distributed-computing-322dacbce43/)) |

The pattern: capture-and-accelerate **works** (cudf.pandas, LaFP, Dias) when
the accelerator owns a real plan representation and handles fallback
honestly; it **fails** when a lazy API is bolted *onto* eager pandas
execution (Ibis) or when API parity is chased without owning semantics
(distributed pandas clones).

## Delegating execution: validated and increasingly standard

- Ibis runs on 16–22 engines, picked DuckDB as default because it beat its
  pandas/Dask backends on both performance and features, and keeps pandas
  only for I/O interchange ([why-duckdb](https://ibis-project.org/posts/why-duckdb/)).
- A working pipeline already exists for "frontend builds plan → standard IR
  → engine executes": Ibis → Substrait → `duckdb.from_substrait()`
  ([demo](https://ibis-project.org/posts/ibis_substrait_to_duckdb/)).
  Substrait is explicitly positioned as the portable alternative to SQL
  dialects, though engine support is still maturing.
- Arrow zero-copy interop between pandas/Polars/DuckDB makes mixed-engine
  architectures nearly free.
- dask-expr retrofitted exactly our optimizer design (expression plans,
  filter/projection pushdown) onto Dask — convergent evolution toward
  plan-based dataframes.

## Where the moat actually is (and isn't)

**Is**: pandas is the API every Python developer reads; sklearn,
statsmodels, and the viz ecosystem integrate with pandas directly while
Polars/DuckDB users convert at the ML boundary; ecosystem tools that claimed
Polars support were often silently converting to pandas underneath.

**Isn't**: the index (users routinely `reset_index` to escape it; Polars
dropped it deliberately), the type system (three ways to spell int64, more
for strings), or the API's looseness — when ecosystem libraries chose a
neutral dataframe API, **Narwhals chose a strict Polars subset, not the
pandas API**, and 29+ libraries (Altair, Plotly, Bokeh, marimo) adopted it
([narwhals](https://github.com/narwhals-dev/narwhals),
[McKinney–Gorelli transcript](https://wesmckinney.com/transcripts/2025-12-16-test-set-marco-gorelli)).
The moat is the *installed base and ecosystem edges*, not the API's design.

## Ranked directions

| # | Direction | Structural advantage | Feasibility | Evidence |
|---|---|---|---|---|
| 1 | **Transparent lazy capture of eager pandas code** — proxy/intercept eager calls, build our logical plan behind the scenes, optimize, execute, fall back honestly | Unique to pandas: zero-rewrite acceleration of the world's installed pandas code. Polars/DuckDB structurally cannot offer this | Hard but de-risked: every component proven separately (cudf.pandas proxying + rewind-replay; LaFP lazy capture + Lazy Print; Modin's compact algebra; Dias's precondition checks). Our plan/optimizer is the missing middle those projects lacked | Strong |
| 2 | **Pluggable execution backends via IR** — keep our logical plan + semantics layer; add DuckDB (and later Velox/DataFusion) as physical engines beside our Arrow/NumPy engine, which becomes the semantics-preserving fallback | Rides engine commoditization instead of fighting it; instantly closes the 5–7x kernel gap where plans map cleanly | Medium: Substrait path demonstrated end-to-end by Ibis→DuckDB; the hard part is the semantics mapping (ordering, index, NaN/NULL) — which is precisely the layer we own and have already audited | Strong |
| 3 | **Own-engine parity** (parallel kernels, morsel-driven scheduling, SIMD) | None — arrives years late to a commoditizing layer | From-scratch engine effort inside pandas | Against: commoditization thesis; Polars' wins are exactly this layer |
| 4 | **pandas-native expression standard** (`pd.col`) | Defensive: keeps the ecosystem's neutral-API choice from defaulting to Polars-shaped (Narwhals) | Small, already started upstream | Enabler for #1/#2, not a standalone play |

Directions 1 and 2 **compose**: capture turns eager code into plans; pluggable
backends make those plans fast; our current engine guarantees pandas
semantics where external engines can't express them (index ops, exotic
dtypes). That composite — *incumbent API + owned semantics + commoditized
execution* — is the breakthrough candidate position (b) requires.

## Caveats

- Speedups quoted from papers/vendors are best-case; LaFP (Jan 2025) is
  fresh research, not production-proven.
- Polars/DuckDB internals claims draw partly on secondary sources.
- The silent-fallback UX problem is unsolved even at NVIDIA scale — any
  capture design must treat fallback *visibility* as a first-class
  requirement (our `explain()` infrastructure is a head start).
- Eager-semantics capture has sharp edges documented by every prior attempt:
  ordering/`iloc`, NaN-vs-NULL, repr forcing materialization. Our semantics
  audit (PROPOSAL.md) already maps this terrain.
