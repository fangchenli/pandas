"""Summarize compare_pandas.py JSON lines as a Markdown table.

For each workload and width, report the median over both passes of every
arm, the drift between the two passes, and speedups against eager and
deferred-take pandas. Results must agree across arms, or the script fails.
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

    print(
        "| workload | strings | arm | median ms | pass drift | "
        "vs pandas-eager | vs pandas-deferred |"
    )
    print("|---|---|---|---|---|---|---|")
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
        for arm in ARMS:
            if arm not in by_arm:
                continue
            passes = [r["median_seconds"] for r in by_arm[arm]]
            drift = max(passes) / min(passes) - 1 if len(passes) > 1 else 0.0
            eager = medians["pandas-eager"] / medians[arm]
            deferred = medians["pandas-deferred"] / medians[arm]
            print(
                f"| {workload} | {strings} | {arm} | {medians[arm] * 1e3:.1f} | "
                f"{drift:.0%} | {eager:.2f}x | {deferred:.2f}x |"
            )


if __name__ == "__main__":
    main()
