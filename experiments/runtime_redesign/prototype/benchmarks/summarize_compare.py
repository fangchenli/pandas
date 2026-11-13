"""Summarize compare_pandas.py JSON lines as Markdown tables.

For each workload and width, report the median over both passes of every
arm, the drift between the two passes, and speedups against released pandas
(the eager reference) and deferred-take pandas; then the Arrow memory each
chain result retains. Results must agree across arms, or the script fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = (
    "pandas",
    "pandas-eager",
    "pandas-deferred",
    "proto-eager",
    "proto-compact",
    "proto-tree",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    runs: dict[tuple, list[dict]] = defaultdict(list)
    for path in args.paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record["workload"], record["strings"], record["rows"])
            runs[key].append(record)

    timing = [
        "| workload | strings | arm | median ms | pass drift | "
        "vs pandas | vs pandas-deferred |",
        "|---|---|---|---|---|---|---|",
    ]
    memory = [
        "| workload | strings | arm | source MB | retained MB | "
        "retained after consume MB |",
        "|---|---|---|---|---|---|",
    ]
    for (workload, strings, _rows), records in sorted(runs.items()):
        results = {json.dumps(r["result"]) for r in records}
        if len(results) != 1:
            raise SystemExit(f"{workload}/{strings}: arms disagree: {results}")
        by_arm = defaultdict(list)
        for record in records:
            by_arm[record["arm"]].append(record)
        medians = {
            arm: statistics.median(s for r in rs for s in r["seconds"])
            for arm, rs in by_arm.items()
        }
        for arm in (a for a in ARMS if a in by_arm):
            passes = [r["median_seconds"] for r in by_arm[arm]]
            drift = max(passes) / min(passes) - 1
            timing.append(
                f"| {workload} | {strings} | {arm} | {medians[arm] * 1e3:.1f} | "
                f"{drift:.0%} | {speedup(medians, 'pandas', arm)} | "
                f"{speedup(medians, 'pandas-deferred', arm)} |"
            )
            # The memory run is deterministic; report the first pass.
            arrow = by_arm[arm][0].get("arrow_bytes")
            if arrow is not None:
                memory.append(
                    f"| {workload} | {strings} | {arm} | {mb(arrow['source'])} | "
                    f"{mb(arrow['retained_before_consume'])} | "
                    f"{mb(arrow['retained_after_consume'])} |"
                )
    print("\n".join(timing))
    if len(memory) > 2:
        print("\nArrow memory pool only; NumPy-backed Int64 columns are excluded.\n")
        print("\n".join(memory))


def speedup(medians: dict[str, float], reference: str, arm: str) -> str:
    if reference not in medians:
        return "n/a"
    return f"{medians[reference] / medians[arm]:.2f}x"


def mb(value: int) -> str:
    return f"{value / 2**20:.1f}"


if __name__ == "__main__":
    main()
