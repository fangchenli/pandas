"""Per-function Substrait->Acero coverage matrix.

The Substrait roundtrip probe (`substrait_roundtrip.py --acero --fix`) left 8/22
TPC-H queries failing with ``No conversion function exists to convert the
Substrait function <X> to an Arrow call expression`` — Acero's Substrait consumer
maps only a subset of Substrait functions to Arrow compute calls. This probe pins
*exactly* which functions, one minimal plan per function, so the gap is a clean
coverage matrix (the AG9-class deliverable) rather than a per-query first-error.

Each function is emitted via a minimal DataFusion expression, lowered to Substrait,
run through the AG17/AG18 portability fixup, and fed to Acero. Classified:
  OK              Acero mapped the function and executed
  NO-CONVERSION   `No conversion function … to an Arrow call expression` (the gap)
  OTHER           any other error (printed)

Self-contained (datafusion + pyarrow + substrait + substrait_fixup) — no pandas,
so it runs in a throwaway venv on the *latest* releases (the authoritative check;
the repo pins pyarrow 23.0.1):

    python -m venv /tmp/sv && . /tmp/sv/bin/activate
    pip install -U pyarrow datafusion duckdb substrait
    python substrait_fn_coverage.py
"""

from __future__ import annotations

import datetime
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from datafusion import (
    SessionContext,
    col,
    functions as F,
    lit,
)
import datafusion.substrait as ss
import pyarrow as pa
import pyarrow.substrait as pas
import substrait_fixup


def _ctx():
    ctx = SessionContext()
    ts = pa.array(
        [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 6, 1)],
        type=pa.timestamp("ns"),
    )
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2]), pa.array([1.0, 2.0]), pa.array(["alpha", "beta"]), ts],
        names=["k", "v", "s", "t"],
    )
    ctx.register_record_batches("A", [[batch]])
    return ctx


# (label, builder) — builder(ctx.table("A")) -> a DataFusion DataFrame emitting the
# target Substrait function.
def _filter(pred):
    return lambda tb: tb.filter(pred(tb))


def _project(expr):
    return lambda tb: tb.select(expr(tb).alias("out"))


def _agg(measure):
    return lambda tb: tb.aggregate([col("k")], [measure(tb).alias("m")])


CASES = [
    ("equal", _filter(lambda t: col("k") == lit(1))),
    ("gt/lt/gte/lte", _filter(lambda t: (col("k") > lit(1)) & (col("k") <= lit(2)))),
    ("or", _filter(lambda t: (col("k") < lit(1)) | (col("k") > lit(1)))),
    ("not", _filter(lambda t: ~(col("k") == lit(1)))),
    ("is_null", _filter(lambda t: col("k").is_null())),
    ("add/subtract", _project(lambda t: col("v") + col("k") - lit(1))),
    ("multiply", _project(lambda t: col("v") * lit(2.0))),
    ("divide", _project(lambda t: col("v") / lit(2.0))),
    (
        "in_list",
        _filter(lambda t: F.in_list(col("k"), [lit(1), lit(2)], negated=False)),
    ),
    (
        "case_when",
        _project(lambda t: F.when(col("k") > lit(1), lit(1)).otherwise(lit(0))),
    ),
    ("starts_with", _filter(lambda t: F.starts_with(col("s"), lit("al")))),
    ("ends_with", _filter(lambda t: F.ends_with(col("s"), lit("ha")))),
    ("regexp_like", _filter(lambda t: F.regexp_like(col("s"), lit("lph")))),
    ("substring", _project(lambda t: F.substring(col("s"), lit(1), lit(2)))),
    ("date_part(year)", _project(lambda t: F.date_part(lit("year"), col("t")))),
    ("sum", _agg(lambda t: F.sum(col("v")))),
    ("avg", _agg(lambda t: F.avg(col("v")))),
    ("min/max", _agg(lambda t: F.min(col("v")))),
    ("count", _agg(lambda t: F.count(col("v")))),
    ("count(distinct)", _agg(lambda t: F.count(col("v")).distinct().build())),
]


def _classify(label, builder, ctx) -> str:
    try:
        df = builder(ctx.table("A"))
        raw = substrait_fixup.fix_plan(
            ss.Producer.to_substrait_plan(df.logical_plan(), ctx).encode(),
            legacy_timestamp=True,
        )
    except Exception as e:  # producing/fixing failed (not the point here)
        return f"BUILD-ERR: {type(e).__name__}: {str(e)[:40]}"

    def tp(names, schema):
        tbl = pa.Table.from_batches(ctx.table(names[-1]).collect()).select(schema.names)
        cols = [
            c.cast(pa.timestamp("us")) if pa.types.is_timestamp(c.type) else c
            for c in tbl.columns
        ]
        return pa.table(cols, names=tbl.column_names)

    try:
        pas.run_query(pa.py_buffer(raw), table_provider=tp).read_all()
        return "OK"
    except Exception as e:
        msg = str(e)
        if "No conversion function" in msg:
            return "NO-CONVERSION"
        return f"OTHER: {type(e).__name__}: {msg[:45]}"


def main() -> int:
    import datafusion

    print(
        f"Substrait->Acero function coverage | datafusion {datafusion.__version__} "
        f"| pyarrow {pa.__version__}\n"
    )
    print(f"{'function':<18} {'Acero':<14}")
    print("-" * 34)
    tally: dict[str, int] = {}
    for label, builder in CASES:
        ctx = _ctx()
        verdict = _classify(label, builder, ctx)
        key = verdict.split(":")[0]
        tally[key] = tally.get(key, 0) + 1
        print(f"{label:<18} {verdict}")
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print(
        "\nNO-CONVERSION = Arrow's Substrait consumer has no function-extension "
        "mapping\nfor that Substrait function (the AG9-class coverage gap)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
