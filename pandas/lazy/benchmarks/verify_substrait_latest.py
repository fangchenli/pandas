"""Reproduce-on-latest gate for AG17 / AG18 (Substrait interop findings).

Runs the minimal repros against the *current* datafusion + pyarrow releases (not
the repo's pinned pyarrow 23.0.1) — the playbook requires this before treating a
substrate gap as real. Pure datafusion + pyarrow + substrait; no pandas, so it
runs in a throwaway venv:

    python -m venv /tmp/sv && . /tmp/sv/bin/activate
    pip install -U pyarrow datafusion duckdb substrait
    python verify_substrait_latest.py

Isolates AG18a (FetchRel `count_expr`) and AG18b (`precision_timestamp`) with ONLY
the AG17 `output_type` fill applied (no fetch-mirror, no ts downgrade), so we see
latest Acero's NATIVE handling of each. Confirmed SURVIVES on datafusion 54.0.0 +
pyarrow 25.0.0 (2026-07-11)."""

import datetime
import sys

sys.path.insert(
    0, "/Users/fangchenli/Workspace/pandas-fangchenli/pandas/lazy/benchmarks"
)
import datafusion
from datafusion import (
    SessionContext,
    col,
    lit,
)
import datafusion.substrait as ss
import pyarrow as pa
import pyarrow.substrait as pas
import substrait_fixup as sfx

print(f"datafusion {datafusion.__version__} | pyarrow {pa.__version__}\n")

fill_types_only = sfx.fill_output_types  # AG17 fix only; isolates AG18 behavior


# AG18a: filter + sort + LIMIT 2 (count_expr; deprecated count=0)
ctx = SessionContext()
ctx.register_record_batches(
    "A",
    [
        [
            pa.RecordBatch.from_pydict(
                {"k": [1, 2, 3, 4, 5], "v": [1.0, 2.0, 3.0, 4.0, 5.0]}
            )
        ]
    ],
)
raw = fill_types_only(
    ss.Producer.to_substrait_plan(
        ctx.table("A").filter(col("k") > 2).sort(col("k")).limit(2).logical_plan(), ctx
    ).encode()
)


def tp(n, s):
    return pa.Table.from_batches(ctx.table(n[-1]).collect()).select(s.names)


print("AG18a  LIMIT 2 via count_expr, latest Acero (output_type filled, count kept):")
try:
    out = pas.run_query(pa.py_buffer(raw), table_provider=tp).read_all()
    v = (
        "FIXED on latest (reads count_expr)"
        if out.num_rows == 2
        else f"SURVIVES: silent {out.num_rows} rows (expected 2)"
    )
    print(f"   -> {out.num_rows} rows :: {v}")
except Exception as e:
    print(f"   -> {type(e).__name__}: {str(e)[:70]}")

# AG18b: precision_timestamp filter
ctx2 = SessionContext()
ts = pa.array(
    [
        datetime.datetime(2020, 1, 1),
        datetime.datetime(2020, 2, 1),
        datetime.datetime(2020, 3, 1),
    ],
    type=pa.timestamp("ns"),
)
ctx2.register_record_batches(
    "B", [[pa.RecordBatch.from_arrays([pa.array([1, 2, 3]), ts], names=["k", "t"])]]
)
rawb = fill_types_only(
    ss.Producer.to_substrait_plan(
        ctx2.table("B")
        .filter(
            col("t")
            >= lit(pa.scalar(datetime.datetime(2020, 2, 1), type=pa.timestamp("ns")))
        )
        .logical_plan(),
        ctx2,
    ).encode()
)


def tp2(n, s):
    return pa.Table.from_batches(ctx2.table(n[-1]).collect()).select(s.names)


print("\nAG18b  precision_timestamp filter, latest Acero (output_type filled):")
try:
    out = pas.run_query(pa.py_buffer(rawb), table_provider=tp2).read_all()
    print(f"   -> Acero OK, {out.num_rows} rows :: FIXED on latest (precision_ts ok)")
except Exception as e:
    print(f"   -> {type(e).__name__}: {str(e)[:70]} :: SURVIVES")
