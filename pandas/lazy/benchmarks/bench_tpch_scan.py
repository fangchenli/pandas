#!/usr/bin/env python3
"""
TPC-H through the PARQUET-SCAN STREAMING path (out-of-core readiness).

The in-memory `bench_tpch.py` holds all tables as pandas DataFrames — a
hard RAM wall at scale (SF-10 ≈ 14 GB of frames on a 16 GB machine). This
harness runs the SAME 22 queries with every table a `scan(parquet)`
LazyFrame, so execution exercises the streaming scan path, pushdowns, and
the engine's materialization discipline. Completion + correctness (every
query validated against DuckDB) is the primary result; time and peak RSS
are recorded per query.

Each query runs in its own SUBPROCESS (timeout + crash isolation): an OOM
or wedge marks that query and the run continues — exactly what an
unattended at-scale shakeout needs.

Data: DuckDB ``dbgen`` (chunked at SF>=3 to bound generator memory) into a
disk-backed .duckdb file, exported to parquet with dtype-parity casts
(DECIMAL→DOUBLE, DATE→TIMESTAMP) so query literals (`pd.Timestamp`)
compare identically to the in-memory harness. The same .duckdb file serves
``PRAGMA tpch(n)`` reference results.

Usage::

    python bench_tpch_scan.py --sf 1            # full run (gen + 22 queries)
    python bench_tpch_scan.py --sf 10
    python bench_tpch_scan.py --sf 10 --one 9   # internal: single query
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import bench_tpch as bt
import duckdb

DATA_ROOT = os.environ.get("TPCH_SCAN_DIR", "/tmp/tpch_scan")
TIMEOUT_S = int(os.environ.get("TPCH_SCAN_TIMEOUT_S", "900"))


def gen_parquet(sf: float) -> str:
    out = f"{DATA_ROOT}/sf{sf:g}"
    marker = f"{out}/.complete"
    if os.path.exists(marker):
        return out
    os.makedirs(out, exist_ok=True)
    db = f"{out}/ref.duckdb"
    if os.path.exists(db):
        Path(db).unlink()
    con = duckdb.connect(db)
    con.execute("INSTALL tpch; LOAD tpch")
    t0 = time.perf_counter()
    children = max(1, int(sf))
    if children > 1:
        for step in range(children):
            con.execute(f"CALL dbgen(sf={sf}, children={children}, step={step})")
    else:
        con.execute(f"CALL dbgen(sf={sf})")
    print(f"[dbgen sf={sf}: {time.perf_counter() - t0:.0f}s]", flush=True)
    for name in bt.TABLES:
        cols = con.execute(f"DESCRIBE {name}").fetchall()
        sel = []
        for cname, ctype, *_ in cols:
            if ctype.startswith("DECIMAL"):
                sel.append(f'CAST("{cname}" AS DOUBLE) AS "{cname}"')
            elif ctype == "DATE":
                sel.append(f'CAST("{cname}" AS TIMESTAMP) AS "{cname}"')
            else:
                sel.append(f'"{cname}"')
        con.execute(
            f"COPY (SELECT {', '.join(sel)} FROM {name}) TO "
            f"'{out}/{name}.parquet' (FORMAT PARQUET)"
        )
    con.close()
    open(marker, "w").write("ok")
    print(f"[parquet export done: {time.perf_counter() - t0:.0f}s]", flush=True)
    return out


class _ScanTable:
    """Shim so the bench_tpch queries work on scan-backed tables.

    Queries reach tables two ways: ``_lp(t[name])`` (patched to identity
    here) and direct ``t[name].select(...)``. This shim delegates to the
    underlying scan LazyFrame, with no-arg ``select()`` meaning "as is".
    """

    def __init__(self, lf):
        self._lf = lf

    def select(self, *exprs):
        return self._lf.select(*exprs) if exprs else self._lf

    def __getattr__(self, name):
        return getattr(self._lf, name)


def scan_tables(data_dir: str) -> dict:
    from pandas.lazy import scan

    return {n: _ScanTable(scan(f"{data_dir}/{n}.parquet")) for n in bt.TABLES}


def run_one(q: int, sf: float, data_dir: str, engine: str = "lp") -> None:
    """Child process: run + validate one query, print one JSON line."""
    if engine == "pl":
        import polars as pl

        tables = {n: pl.scan_parquet(f"{data_dir}/{n}.parquet") for n in bt.TABLES}
        _, pl_fn = bt.QUERIES[q]
        t0 = time.perf_counter()
        res = pl_fn(tables)  # pl_* queries collect internally (default engine)
        ms = (time.perf_counter() - t0) * 1000
        if hasattr(res, "to_pandas"):
            res = res.to_pandas()
    else:
        bt._lp = lambda x: x if isinstance(x, _ScanTable) else x.select()
        t = scan_tables(data_dir)
        lp_fn, _ = bt.QUERIES[q]
        t0 = time.perf_counter()
        res = lp_fn(t).collect(use_physical_planner=True)
        ms = (time.perf_counter() - t0) * 1000
    con = duckdb.connect(f"{data_dir}/ref.duckdb", read_only=True)
    con.execute("LOAD tpch")
    ref = con.execute(f"PRAGMA tpch({q})").df()
    ok, msg = bt.validate(res, ref)
    # ru_maxrss is BYTES on macOS but KILOBYTES on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9
    print(
        "RESULT "
        + json.dumps({"q": q, "ok": ok, "msg": msg, "ms": ms, "rss_gb": rss_gb})
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("--one", type=int, default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--engine", default="lp", choices=["lp", "pl"])
    args = ap.parse_args()

    data_dir = gen_parquet(args.sf)
    if args.one is not None:
        run_one(args.one, args.sf, data_dir, args.engine)
        return

    which = (
        [int(x) for x in args.queries.split(",")]
        if args.queries
        else sorted(bt.QUERIES)
    )
    print(f"TPC-H SCAN-STREAMING run [{args.engine}], SF-{args.sf:g} ({data_dir})")
    print(f"{'query':>6} {'status':>22} {'ms':>9} {'peakRSS':>8}")
    rows = []
    for q in which:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--sf",
            str(args.sf),
            "--one",
            str(q),
            "--engine",
            args.engine,
        ]
        try:
            p = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            line = next(
                (ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")),
                None,
            )
            if line:
                r = json.loads(line[7:])
                status = "OK" if r["ok"] else f"FAIL:{r['msg'][:14]}"
                print(
                    f"{'q' + str(q):>6} {status:>22} {r['ms']:9.0f} {r['rss_gb']:7.1f}G"
                )
                rows.append(r)
            else:
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:]
                print(f"{'q' + str(q):>6} {'CRASH':>22}  {tail}")
                rows.append({"q": q, "ok": False, "msg": "crash"})
        except subprocess.TimeoutExpired:
            print(f"{'q' + str(q):>6} {'TIMEOUT':>22}")
            rows.append({"q": q, "ok": False, "msg": "timeout"})
    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"-- {n_ok}/{len(rows)} queries completed and validated --")


if __name__ == "__main__":
    main()
