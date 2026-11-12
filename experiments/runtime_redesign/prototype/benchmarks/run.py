"""Run M3 correctness-checked pilot workloads; no pandas speedup claim."""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from pdruntime import Runtime

CASES = ("W02", "W05", "W06", "W07", "W08", "W09")


def run_once(strategy, backend, case, rows, width, seed, *, metrics=False):
    rng = np.random.default_rng(seed)
    inputs = {f"c{i}": np.arange(rows, dtype=np.int64) + i for i in range(width)}
    first = rng.integers(0, rows, size=max(1, rows // 2), dtype=np.int64)
    second = rng.integers(0, len(first), size=max(1, rows // 20), dtype=np.int64)
    runtime = Runtime(strategy=strategy, metrics=metrics)
    start = time.perf_counter()
    frame = runtime.frame(
        {
            name: runtime.array(values, dtype="int64", backend=backend)
            for name, values in inputs.items()
        }
    )
    imported = time.perf_counter()
    memory = None
    original = None
    if case == "W02":
        result = frame.take(first).take(second)
        expected = {name: values[first][second] for name, values in inputs.items()}
        consumed = result.column("c0").sum()
    elif case == "W05":
        result = frame.take(first)
        expected = {name: values[first] for name, values in inputs.items()}
        consumed = [result.column(name).sum() for name in result.columns]
    elif case == "W06":
        result = frame.take(first)
        expected = {name: values[first] for name, values in inputs.items()}
        consumed = [result.column("c0").sum() for _ in range(20)]
    elif case == "W07":
        result = frame.slice(slice(0, min(10, rows)))
        expected = {name: values[: min(10, rows)] for name, values in inputs.items()}
        del frame
        gc.collect()
        before = result.memory_report()
        result = result.compact()
        gc.collect()
        memory = {"before_compact": before, "after_compact": result.memory_report()}
        consumed = result.column("c0").sum()
    elif case == "W08":
        result = frame.take([1, 1, 2])
        expected = {name: values[[1, 1, 2]].copy() for name, values in inputs.items()}
        expected["c0"][0] = -99
        result.set_at(row=0, column="c0", value=-99)
        consumed = result.column("c0").sum()
        original = frame
    elif case == "W09":
        result = frame.take(first)
        expected = {name: values[first].copy() for name, values in inputs.items()}
        for value in range(5):
            result.set_at(row=0, column="c0", value=-value)
        replacement = np.full(len(result), 42, dtype=np.int64)
        result.replace_column(
            "c0", runtime.array(replacement, dtype="int64", backend=backend)
        )
        expected["c0"] = replacement
        consumed = result.column("c0").sum()
        original = frame
    else:
        raise ValueError(case)
    completed = time.perf_counter()
    # Export only the columns the workload consumes. Full diagnostics below are
    # outside the timed region, so W02 remains a legitimate narrow-consumer case.
    names = result.columns if case == "W05" else ("c0",)
    exported = {name: result.column(name).to_numpy() for name in names}
    finished = time.perf_counter()
    events = list(runtime.metrics.events)
    for name, (values, valid) in exported.items():
        np.testing.assert_array_equal(values, expected[name])
        assert valid is None
    for name in result.columns:
        values, valid = result.column(name).to_numpy()
        np.testing.assert_array_equal(values, expected[name])
        assert valid is None
    if original is not None:
        for name, values in inputs.items():
            np.testing.assert_array_equal(original.column(name).to_numpy()[0], values)
    totals = [int(np.sum(expected[name], dtype=np.int64)) for name in names]
    expected_consumed = (
        totals if case == "W05" else [totals[0]] * 20 if case == "W06" else totals[0]
    )
    assert consumed == expected_consumed
    return {
        "strategy": strategy,
        "backend": backend,
        "workload": case,
        "ingress_seconds": imported - start,
        "resident_seconds": completed - imported,
        "export_seconds": finished - completed,
        "total_seconds": finished - start,
        "memory": memory,
        "events": events if metrics else None,
    }


def manifest():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "pandas": pd.__version__,
        "pandas_location": pd.__file__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "base_commit": commit.stdout.strip() or None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "arrow_cpu_count": pa.cpu_count(),
        "limitations": (
            "pilot only; no pinned-main/deferred-take comparison or peak RSS; "
            "IPC ownership copies included; GC included in W07"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 3 or args.width < 1 or args.repeat < 1:
        parser.error("rows >= 3, width >= 1, repeat >= 1 required")
    tasks = [
        (strategy, backend, case)
        for strategy in ("eager", "compact", "tree")
        for backend in ("numpy", "arrow")
        for case in CASES
    ]
    rng = np.random.default_rng(args.seed)
    results = []
    for repetition in range(args.repeat):
        for position in rng.permutation(len(tasks)):
            strategy, backend, case = tasks[position]
            result = run_once(strategy, backend, case, args.rows, args.width, args.seed)
            result["repetition"] = repetition
            results.append(result)
    diagnostics = [
        run_once(s, b, c, args.rows, args.width, args.seed, metrics=True)
        for s, b, c in tasks
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "manifest": manifest(),
                "parameters": vars(args) | {"output": str(args.output)},
                "results": results,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Validated {len(results)} timing runs and {len(diagnostics)} diagnostic runs; "
        f"saved {args.output}"
    )


if __name__ == "__main__":
    main()
