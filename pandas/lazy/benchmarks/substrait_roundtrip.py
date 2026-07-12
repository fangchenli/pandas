"""Substrait roundtrip-survival probe over the 22 TPC-H plans.

The question this answers (see docs/upstream + the Substrait-vs-DataFrame-API
analysis): of the TPC-H plans that lower cleanly to the DataFusion **DataFrame
API** today (`translate_datafusion.run` — 22/22 vs DuckDB), how many *survive a
roundtrip through Substrait*? Substrait is the portable IR that DataFusion,
Acero, DuckDB, ... all consume; if a plan roundtrips and still matches DuckDB,
Substrait is a viable portable lowering (→ fan one plan across N engines for the
differential probe). If it drops, the failure list IS a findings batch:
DataFusion / Acero Substrait-*consumer* coverage gaps (the AG9 coverage-matrix
class), on-charter because both engines are named substrates.

For each query we lower once to a DataFusion DataFrame (reusing the existing
`translate_datafusion` internals — NOT re-implementing the lowering), then:

  DataFrame.logical_plan()
      │  Producer.to_substrait_plan(lp, ctx)         ── PRODUCE
      ▼
  Substrait Plan ──► Plan.encode() ─► bytes ─► Serde.deserialize_bytes  ── SERDE
      │
      ├─► datafusion Consumer.from_substrait_plan(ctx, plan)             ── DF-CONSUME
      │       └─► ctx.create_dataframe_from_logical_plan(...).to_pandas() ─► exec
      │
      └─► pyarrow.substrait.run_query(bytes, table_provider=...)     ── ACERO-CONSUME
              └─► reader.read_all().to_pandas()                      ─► exec

Each stage is classified so we can see *where* a plan dies:
  SURVIVES        roundtrips through DataFusion-Substrait AND matches DuckDB
  RESULT-DIVERGE  roundtrips + executes, but result != DuckDB (silent-wrong)
  PRODUCE-FAIL    DataFusion can't emit Substrait for this plan
  DF-CONSUME-FAIL DataFusion can't re-ingest its own Substrait
  EXEC-FAIL       consumed plan won't execute
  BASELINE-FAIL   the direct DataFrame-API lowering itself failed (excluded)

The Acero consumer is scored separately (a second, independent Substrait engine
— cross-engine divergences are free findings).

Run (repo pandas-dev has datafusion + pyarrow-substrait + duckdb):
    PATH=<pandas-dev>/bin:$PATH python substrait_roundtrip.py --sf 0.1
    python substrait_roundtrip.py --sf 0.1 --queries 1,3,15 --acero
"""

from __future__ import annotations

import argparse

import bench_tpch as B
from datafusion import SessionContext
import datafusion.substrait as ss
import translate_datafusion as T


def lower_to_df(ldf, ctx: SessionContext):
    """Reuse translate_datafusion's internals to get the DataFusion DataFrame
    (before .to_pandas()), so we can reach .logical_plan()."""
    plan = ldf._get_optimized_plan()
    return T._lower(plan, ctx, [0], {}, T._count_refs(plan), {})


def _acero_run(raw: bytes, tables: dict):
    """Feed the same Substrait protobuf to pyarrow/Acero. Acero resolves named
    tables via a table_provider callback; we serve the source frames as Arrow."""
    import pyarrow as pa
    import pyarrow.substrait as pas

    # arrow-table view of each registered source, keyed by the name the lowering
    # used (t0, t1, ...). We don't know the exact name map, so provide by schema:
    # Acero passes the requested names + expected schema to the callback.
    providers = {}
    for name, df in tables.items():
        providers[name] = pa.Table.from_pandas(df, preserve_index=False)

    def table_provider(names, schema):
        # names: list[str] identifying the requested table. Match by column set.
        want = set(schema.names)
        for tbl in providers.values():
            if want.issubset(set(tbl.schema.names)):
                return tbl.select(schema.names)
        raise KeyError(f"no provider for {names} / {schema.names}")

    reader = pas.run_query(pa.py_buffer(raw), table_provider=table_provider)
    return reader.read_all().to_pandas()


def probe_query(n: int, tables: dict, con, do_acero: bool) -> dict:
    r = {"q": n, "status": "", "detail": "", "acero": ""}
    lp_fn, _ = B.QUERIES[n]

    # --- baseline: does the direct DataFrame-API lowering work + match DuckDB? ---
    try:
        ldf = lp_fn(tables)
        ctx = SessionContext()
        df = lower_to_df(ldf, ctx)
        lp = df.logical_plan()
        direct = df.to_pandas()
    except Exception as e:
        r["status"] = "BASELINE-FAIL"
        r["detail"] = f"{type(e).__name__}: {str(e)[:55]}"
        return r

    # --- PRODUCE ---
    try:
        sub = ss.Producer.to_substrait_plan(lp, ctx)
        raw = sub.encode()
    except Exception as e:
        r["status"] = "PRODUCE-FAIL"
        r["detail"] = f"{type(e).__name__}: {str(e)[:55]}"
        return r

    # --- SERDE + DF-CONSUME + EXEC ---
    try:
        sub2 = ss.Serde.deserialize_bytes(raw)
        consumed = ss.Consumer.from_substrait_plan(ctx, sub2)
    except Exception as e:
        r["status"] = "DF-CONSUME-FAIL"
        r["detail"] = f"{type(e).__name__}: {str(e)[:55]} [{len(raw)}B]"
        return r
    try:
        rt = ctx.create_dataframe_from_logical_plan(consumed).to_pandas()
    except Exception as e:
        r["status"] = "EXEC-FAIL"
        r["detail"] = f"{type(e).__name__}: {str(e)[:55]}"
        return r

    # --- correctness vs DuckDB ---
    try:
        ref = con.execute(f"PRAGMA tpch({n})").df()
        ok, msg = B.validate(rt, ref)
    except Exception as e:
        r["status"] = "VAL-ERR"
        r["detail"] = f"{type(e).__name__}: {str(e)[:55]}"
        return r
    if ok:
        r["status"] = "SURVIVES"
        r["detail"] = f"{len(raw)}B"
    else:
        # roundtripped + executed, but wrong: distinguish from a baseline that was
        # already wrong (shouldn't happen — baseline is 22/22).
        base_ok, _ = B.validate(direct, ref)
        r["status"] = "RESULT-DIVERGE" if base_ok else "BASELINE-WRONG"
        r["detail"] = str(msg)[:60]

    # --- second engine: Acero (independent Substrait consumer) ---
    if do_acero:
        try:
            aout = _acero_run(raw, tables)
            aok, amsg = B.validate(aout, ref)
            r["acero"] = "OK" if aok else f"DIVERGE:{str(amsg)[:30]}"
        except Exception as e:
            r["acero"] = f"{type(e).__name__}:{str(e)[:34]}"
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sf", type=float, default=0.1)
    ap.add_argument("--queries", default=None, help="comma-separated ids")
    ap.add_argument(
        "--acero", action="store_true", help="also score the pyarrow/Acero consumer"
    )
    args = ap.parse_args()

    which = (
        [int(q) for q in args.queries.split(",")] if args.queries else sorted(B.QUERIES)
    )
    con = B.make_duckdb(args.sf)
    tables = B.load_tables(con)
    print(
        f"Substrait roundtrip survival | SF-{args.sf} | "
        f"lineitem={len(tables['lineitem']):,} | datafusion + "
        f"{'acero + ' if args.acero else ''}duckdb\n"
    )
    hdr = f"{'q':>4} {'status':<16} {'detail':<50}"
    if args.acero:
        hdr += " acero"
    print(hdr)
    print("-" * (len(hdr) + 6))

    tally: dict[str, int] = {}
    for n in which:
        if n not in B.QUERIES:
            continue
        r = probe_query(n, tables, con, args.acero)
        tally[r["status"]] = tally.get(r["status"], 0) + 1
        line = f"q{n:<3} {r['status']:<16} {r['detail']:<50}"
        if args.acero:
            line += f" {r['acero']}"
        print(line)

    total = sum(tally.values())
    surv = tally.get("SURVIVES", 0)
    print(f"\n{surv}/{total} SURVIVE the DataFusion-Substrait roundtrip")
    print("breakdown: " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
