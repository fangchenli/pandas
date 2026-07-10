"""Validate the DataFusion lowering (``translate_datafusion.run``) across all 22
TPC-H queries against DuckDB.

Complements ``bench_tpch.py`` (which validates the *lazy engine's* ``.collect()``
vs DuckDB) — this drives each query's lazy plan through the DataFusion lowering
instead, and checks the result against DuckDB's ``PRAGMA tpch(n)``. Reuses
``bench_tpch``'s data generation (``make_duckdb`` / ``load_tables``), query
definitions (``QUERIES``), and comparison (``validate``).

Reports one line per query: PASS / WRONG (correctness) / UNSUPPORTED (a node/expr
the lowering doesn't handle) / ERROR (a DataFusion or build failure). Known-good:
22/22 PASS at SF-0.1 and SF-1 (2026-07-10, after the shared-subplan materialize
fix — see translate_datafusion.py and docs/upstream/AG14).

Run in an env with pandas.lazy built + datafusion + duckdb (the repo's pandas-dev
has all three); prepend the env's bin to PATH so the editable rebuild finds its
tools:

    PATH=<pandas-dev>/bin:$PATH python validate_datafusion.py --sf 0.1
    # subset:  python validate_datafusion.py --sf 1 --queries 2,15
"""

from __future__ import annotations

import argparse

import bench_tpch as B
import translate_datafusion as T


def validate_query(n: int, tables: dict, con) -> tuple[str, str]:
    """-> (status, detail) for one query n."""
    lp_fn, _ = B.QUERIES[n]
    try:
        ldf = lp_fn(tables)
    except Exception as e:
        return ("BUILD-ERR", f"{type(e).__name__}: {str(e)[:60]}")
    try:
        got = T.run(ldf)
    except T.NotSupported as e:
        return ("UNSUPPORTED", str(e)[:60])
    except Exception as e:
        return ("ERROR", f"{type(e).__name__}: {str(e)[:60]}")
    try:
        ref = con.execute(f"PRAGMA tpch({n})").df()
        ok, msg = B.validate(got, ref)
        return ("PASS", "") if ok else ("WRONG", str(msg)[:70])
    except Exception as e:
        return ("VAL-ERR", f"{type(e).__name__}: {str(e)[:60]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sf", type=float, default=0.1, help="TPC-H scale factor")
    ap.add_argument(
        "--queries", default=None, help="comma-separated query ids (default: all)"
    )
    args = ap.parse_args()

    which = (
        [int(q) for q in args.queries.split(",")] if args.queries else sorted(B.QUERIES)
    )
    con = B.make_duckdb(args.sf)
    tables = B.load_tables(con)
    print(
        f"DataFusion lowering vs DuckDB | SF-{args.sf} | "
        f"lineitem={len(tables['lineitem']):,} rows\n"
    )
    print(f"{'q':>4} {'status':<12} detail")
    print("-" * 70)
    npass = 0
    for n in which:
        if n not in B.QUERIES:
            print(f"q{n:<3} {'not-impl':<12}")
            continue
        status, detail = validate_query(n, tables, con)
        npass += status == "PASS"
        print(f"q{n:<3} {status:<12} {detail}")
    print(f"\n{npass}/{len(which)} PASS")
    return 0 if npass == len(which) else 1


if __name__ == "__main__":
    raise SystemExit(main())
