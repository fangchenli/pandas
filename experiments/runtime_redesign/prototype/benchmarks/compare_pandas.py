"""Compare prototype strategies with eager and deferred-take pandas.

Run one arm per process; ``compare_pandas.sh`` drives every arm and writes
JSON lines. Construction is untimed. Each timed run covers a workload's whole
structural chain (predicates, filters, takes) and its consumers. Consumers
compute on the selected values (an Int64 sum, a string equality count), so
every arm must produce those values and none pays an export copy.

Arms:
    pandas            pandas as installed (eager unless it defers natively)
    pandas-eager      the deferred-take build with ``defer`` patched to gather
    pandas-deferred   the deferred-take build as is
    proto-<strategy>  the prototype with the eager, compact, or tree strategy
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa

VOCABULARY = 10_000
STRING_WIDTH = 32
LITERAL_INDEX = 7


def make_data(rows: int, ints: int, strings: int, seed: int):
    rng = np.random.default_rng(seed)
    int_columns = {
        f"i{k}": rng.integers(0, 1000, rows, dtype=np.int64) for k in range(ints)
    }
    letters = rng.integers(97, 123, size=(VOCABULARY, STRING_WIDTH), dtype=np.uint8)
    vocabulary = pa.array([bytes(row).decode() for row in letters])
    string_columns = {
        f"s{k}": vocabulary.take(pa.array(rng.integers(0, VOCABULARY, rows)))
        for k in range(strings)
    }
    return int_columns, string_columns, vocabulary[LITERAL_INDEX].as_py()


def random_half(size: int, seed: int) -> np.ndarray:
    """Half of ``range(size)`` in random order: an unsorted gather."""
    return np.random.default_rng(seed).permutation(size)[: size // 2]


# Workloads take an ``ops`` adapter and return their consumed results.


def narrow(ops, frame, plan):
    f = ops.filter(frame, "i0", 500)
    f = ops.take(f, plan["take_after_i0_lt_500"])
    f = ops.filter(f, "i1", 500)
    return [ops.sum(f, "i2"), ops.eq_count(f, "s0", plan["literal"])]


def full(ops, frame, plan):
    f = ops.filter(frame, "i0", 500)
    f = ops.take(f, plan["take_after_i0_lt_500"])
    f = ops.filter(f, "i1", 500)
    return [
        ops.sum(f, name)
        if name.startswith("i")
        else ops.eq_count(f, name, plan["literal"])
        for name in ops.columns(f)
    ]


# Five filters keeping about 90% each (the last 8/9 of what remains).
DEEP_FILTERS = (("i0", 900), ("i1", 900), ("i2", 900), ("i3", 900), ("i0", 800))


def deep(ops, frame, plan):
    f = frame
    for column, threshold in DEEP_FILTERS:
        f = ops.filter(f, column, threshold)
    return [ops.sum(f, "i2"), ops.eq_count(f, "s0", plan["literal"])]


def gather_ints(ops, frame, plan):
    f = ops.take(ops.project(frame, plan["int_names"]), plan["take_all"])
    return [ops.sum(f, name) for name in plan["int_names"]]


WORKLOADS: dict[str, Callable] = {
    "narrow": narrow,
    "full": full,
    "deep": deep,
    "gather_ints": gather_ints,
}


class PandasOps:
    def __init__(self, deferral: str):
        import pandas as pd

        self.pd = pd
        self.deferral = deferral
        if deferral == "off":
            import pandas.core.arrays.arrow.array as arrow_array

            def eager_defer(parent, positions, *, copy=True):
                # Gather at once with public pyarrow kernels, as eager pandas
                # does: a mask filters, positions take.
                if positions.dtype == np.bool_:
                    return parent.filter(pa.array(positions))
                return parent.take(pa.array(positions))

            arrow_array.defer = eager_defer

    def build(self, int_columns, string_columns):
        pd = self.pd
        data = {name: pd.array(v, dtype="Int64") for name, v in int_columns.items()}
        for name, values in string_columns.items():
            chunked = pa.chunked_array([values.cast(pa.large_string())])
            data[name] = pd.arrays.ArrowStringArray(chunked)
        frame = pd.DataFrame(data)
        assert all(str(frame[n].dtype) == "string" for n in string_columns)
        self.check_deferral(frame, next(iter(string_columns)))
        return frame

    def check_deferral(self, frame, string_name):
        """Fail loudly if the arm does not defer the way it claims to."""
        array = frame.take([0, 1])[string_name].array
        pending = getattr(array, "_pending_take", None) is not None
        expected = self.deferral == "on"
        if pending != expected:
            raise RuntimeError(f"deferral={self.deferral} but pending={pending}")

    def filter(self, frame, column, threshold):
        return frame[frame[column] < threshold]

    def take(self, frame, positions):
        return frame.take(positions)

    def project(self, frame, names):
        return frame[list(names)]

    def columns(self, frame):
        return list(frame.columns)

    def sum(self, frame, column):
        return int(frame[column].sum())

    def eq_count(self, frame, column, literal):
        return int((frame[column] == literal).sum())


class PrototypeOps:
    def __init__(self, strategy: str):
        from pdruntime import Runtime
        from pdruntime.api import Array

        self.runtime = Runtime(strategy=strategy)
        self.Array = Array

    def build(self, int_columns, string_columns):
        rt = self.runtime
        columns = {
            name: rt.array(v, dtype="int64", backend="numpy")
            for name, v in int_columns.items()
        }
        for name, values in string_columns.items():
            columns[name] = rt.array(values, dtype="utf8", backend="arrow")
        return rt.frame(columns)

    def filter(self, frame, column, threshold):
        predicate = frame.column(column).lt(threshold)
        # Frame.filter rejects a Series (filtering is positional) and has no
        # public Series-to-Array conversion yet; rewrap the cell, no copy.
        return frame.filter(self.Array(self.runtime, predicate.cell))

    def take(self, frame, positions):
        return frame.take(positions)

    def project(self, frame, names):
        return frame.project(names)

    def columns(self, frame):
        return list(frame.columns)

    def sum(self, frame, column):
        return frame.column(column).sum()

    def eq_count(self, frame, column, literal):
        values, valid = frame.column(column).eq(literal).to_numpy()
        return int(np.count_nonzero(values if valid is None else values & valid))


def make_ops(arm: str):
    if arm == "pandas":
        return PandasOps("native")
    if arm == "pandas-eager":
        return PandasOps("off")
    if arm == "pandas-deferred":
        return PandasOps("on")
    if arm.startswith("proto-"):
        return PrototypeOps(arm.removeprefix("proto-"))
    raise ValueError(arm)


def environment(arm: str) -> dict[str, object]:
    import pandas as pd

    return {
        "arm": arm,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "pandas": pd.__version__,
        "pandas_location": pd.__file__,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--ints", type=int, default=4)
    parser.add_argument("--strings", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ints < 4 or args.strings < 1:
        parser.error("workloads need at least 4 Int64 and 1 string column")

    int_columns, string_columns, literal = make_data(
        args.rows, args.ints, args.strings, args.seed
    )
    kept = int(np.count_nonzero(int_columns["i0"] < 500))
    plan = {
        "literal": literal,
        "int_names": tuple(int_columns),
        "take_after_i0_lt_500": random_half(kept, args.seed + 1),
        "take_all": random_half(args.rows, args.seed + 2),
    }
    ops = make_ops(args.arm)
    started = time.perf_counter()
    frame = ops.build(int_columns, string_columns)
    build_seconds = time.perf_counter() - started
    env = environment(args.arm)

    with args.output.open("a") as out:
        for name in args.workloads:
            workload = WORKLOADS[name]
            result = workload(ops, frame, plan)  # warm-up and the checked result
            seconds = []
            for _ in range(args.repeat):
                gc.collect()
                started = time.perf_counter()
                repeated = workload(ops, frame, plan)
                seconds.append(time.perf_counter() - started)
                assert repeated == result
            record = {
                **env,
                "workload": name,
                "rows": args.rows,
                "ints": args.ints,
                "strings": args.strings,
                "build_seconds": build_seconds,
                "seconds": seconds,
                "median_seconds": statistics.median(seconds),
                "result": result,
            }
            out.write(json.dumps(record) + "\n")
            print(
                f"{args.arm:16s} {name:12s} strings={args.strings:<3d} "
                f"median {record['median_seconds'] * 1e3:8.1f} ms",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
