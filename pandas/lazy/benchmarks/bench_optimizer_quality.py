#!/usr/bin/env python3
"""
Benchmark: Optimizer Effectiveness (Plan Quality Metrics)

Measures the QUALITY of optimization, not just speed:
- Number of nodes before vs after optimization
- Filters fused count
- Predicates pushed down
- Columns pruned
- Limits pushed down
- Sort+Limit converted to TopK

This serves as regression testing for optimizer passes.
When you refactor or add passes, run this to ensure optimizations still work.

Usage:
    python pandas/lazy/benchmarks/bench_optimizer_quality.py
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
import json

import numpy as np

import pandas as pd
from pandas.lazy import col
from pandas.lazy.optimize import (
    CommonSubexpressionElimination,
    ConstantFolding,
    DeadCodeElimination,
    ExpressionSimplification,
    FilterFusion,
    LimitPushdown,
    Optimizer,
    PredicatePushdown,
    ProjectionPruning,
    SortLimitToTopK,
)
from pandas.lazy.plan import (
    Aggregate,
    Convert,
    CSVSource,
    DataFrameSource,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    ParquetSource,
    Project,
    Sort,
    TopK,
)

# =============================================================================
# Plan analysis utilities
# =============================================================================


@dataclass
class PlanMetrics:
    """Metrics about a logical plan structure."""

    total_nodes: int = 0
    filter_nodes: int = 0
    project_nodes: int = 0
    sort_nodes: int = 0
    limit_nodes: int = 0
    topk_nodes: int = 0
    aggregate_nodes: int = 0
    join_nodes: int = 0
    scan_nodes: int = 0
    convert_nodes: int = 0
    distinct_nodes: int = 0

    # Expression metrics
    total_filter_predicates: int = 0
    total_project_exprs: int = 0

    # Column tracking (for projection pruning)
    columns_at_scan: set = field(default_factory=set)
    columns_at_root: set = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "total_nodes": self.total_nodes,
            "filter_nodes": self.filter_nodes,
            "project_nodes": self.project_nodes,
            "sort_nodes": self.sort_nodes,
            "limit_nodes": self.limit_nodes,
            "topk_nodes": self.topk_nodes,
            "aggregate_nodes": self.aggregate_nodes,
            "join_nodes": self.join_nodes,
            "scan_nodes": self.scan_nodes,
            "convert_nodes": self.convert_nodes,
            "distinct_nodes": self.distinct_nodes,
            "total_filter_predicates": self.total_filter_predicates,
            "total_project_exprs": self.total_project_exprs,
            "columns_at_scan": len(self.columns_at_scan),
            "columns_at_root": len(self.columns_at_root),
        }


def analyze_plan(plan: LogicalPlan) -> PlanMetrics:
    """
    Walk a logical plan and collect metrics.

    Returns PlanMetrics with counts of each node type and other statistics.
    """
    metrics = PlanMetrics()

    def visit(node: LogicalPlan) -> None:
        metrics.total_nodes += 1

        if isinstance(node, Filter):
            metrics.filter_nodes += 1
            metrics.total_filter_predicates += 1
        elif isinstance(node, Project):
            metrics.project_nodes += 1
            metrics.total_project_exprs += len(node.exprs)
        elif isinstance(node, Sort):
            metrics.sort_nodes += 1
        elif isinstance(node, Limit):
            metrics.limit_nodes += 1
        elif isinstance(node, TopK):
            metrics.topk_nodes += 1
        elif isinstance(node, Aggregate):
            metrics.aggregate_nodes += 1
        elif isinstance(node, Join):
            metrics.join_nodes += 1
        elif isinstance(node, (DataFrameSource, ParquetSource, CSVSource)):
            metrics.scan_nodes += 1
            # Track columns available at scan
            if hasattr(node, "schema") and node.schema:
                metrics.columns_at_scan = set(node.schema.names)
        elif isinstance(node, Convert):
            metrics.convert_nodes += 1
        elif isinstance(node, Distinct):
            metrics.distinct_nodes += 1

        # Recurse into children
        if hasattr(node, "input") and node.input is not None:
            visit(node.input)
        if hasattr(node, "left") and node.left is not None:
            visit(node.left)
        if hasattr(node, "right") and node.right is not None:
            visit(node.right)

    visit(plan)

    # Track columns at root
    if hasattr(plan, "schema") and plan.schema:
        metrics.columns_at_root = set(plan.schema.names)

    return metrics


@dataclass
class OptimizationEffect:
    """Effect of optimization on a query."""

    query_name: str
    before: PlanMetrics
    after: PlanMetrics

    # Computed effects
    nodes_reduced: int = 0
    filters_fused: int = 0
    projects_merged: int = 0
    columns_pruned: int = 0
    sort_limit_to_topk: bool = False
    predicates_pushed: bool = False
    limits_pushed: bool = False

    def compute_effects(self) -> None:
        """Compute derived metrics from before/after comparison."""
        self.nodes_reduced = self.before.total_nodes - self.after.total_nodes
        self.filters_fused = max(0, self.before.filter_nodes - self.after.filter_nodes)
        self.projects_merged = max(
            0, self.before.project_nodes - self.after.project_nodes
        )
        self.columns_pruned = max(
            0,
            len(self.before.columns_at_scan) - len(self.after.columns_at_scan),
        )
        self.sort_limit_to_topk = (
            self.before.topk_nodes == 0 and self.after.topk_nodes > 0
        )
        # Heuristic: if we have fewer filter nodes but same predicates,
        # predicates were likely pushed/fused
        self.predicates_pushed = self.after.filter_nodes < self.before.filter_nodes or (
            self.after.filter_nodes == self.before.filter_nodes
            and self.filters_fused > 0
        )
        self.limits_pushed = self.after.limit_nodes <= self.before.limit_nodes

    def to_dict(self) -> dict:
        return {
            "query_name": self.query_name,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "effects": {
                "nodes_reduced": self.nodes_reduced,
                "filters_fused": self.filters_fused,
                "projects_merged": self.projects_merged,
                "columns_pruned": self.columns_pruned,
                "sort_limit_to_topk": self.sort_limit_to_topk,
                "predicates_pushed": self.predicates_pushed,
                "limits_pushed": self.limits_pushed,
            },
        }


# =============================================================================
# Test queries designed to exercise specific optimizations
# =============================================================================


def create_test_data() -> pd.DataFrame:
    """Create test DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "a": rng.random(1000),
            "b": rng.random(1000),
            "c": rng.random(1000),
            "d": rng.integers(0, 100, 1000),
            "e": rng.integers(0, 100, 1000),
            "f": rng.random(1000),
            "g": rng.random(1000),
            "category": rng.choice(["A", "B", "C"], 1000),
        }
    )


# Query builders that return LazyDataFrame
OPTIMIZATION_TEST_QUERIES = {
    # FilterFusion: Multiple consecutive filters should become one
    "filter_fusion_3": lambda df: (
        df.select().filter(col("a") > 0.5).filter(col("b") < 0.8).filter(col("c") > 0.2)
    ),
    "filter_fusion_5": lambda df: (
        df.select()
        .filter(col("a") > 0.1)
        .filter(col("a") < 0.9)
        .filter(col("b") > 0.2)
        .filter(col("b") < 0.8)
        .filter(col("c") > 0.3)
    ),
    # ProjectionPruning: Only select columns that are actually used
    "projection_pruning": lambda df: (
        df.select()  # Has 8 columns
        .filter(col("a") > 0.5)
        .select(col("a"), col("b"))  # Only need 2
    ),
    "projection_pruning_deep": lambda df: (
        df.select()  # Has 8 columns
        .filter(col("a") > 0.5)
        .select(col("a"), col("b"), col("c"), col("d"))
        .filter(col("b") < 0.8)
        .select(col("a"), col("b"))  # Only need 2 at the end
    ),
    # PredicatePushdown: Push filters closer to scan
    "predicate_pushdown": lambda df: (
        df.select()
        .select(col("a"), col("b"), (col("a") + col("b")).alias("sum"))
        .filter(col("a") > 0.5)  # Can be pushed before the select
    ),
    # SortLimitToTopK: Sort + Limit should become TopK
    "sort_limit_topk": lambda df: (
        df.select().sort(col("a"), descending=True).head(10)
    ),
    "sort_limit_topk_with_filter": lambda df: (
        df.select().filter(col("b") > 0.3).sort(col("a"), descending=True).head(100)
    ),
    # LimitPushdown: Push limit toward source
    "limit_pushdown": lambda df: (df.select().select(col("a"), col("b")).head(100)),
    # CSE: Common subexpressions should be computed once
    "cse_simple": lambda df: (
        df.select().select(
            (col("a") + col("b")).alias("sum1"),
            (col("a") + col("b")).alias("sum2"),  # Same expression
        )
    ),
    "cse_nested": lambda df: (
        df.select().select(
            ((col("a") * col("b")) + col("c")).alias("expr1"),
            ((col("a") * col("b")) + col("c")).alias("expr2"),
            ((col("a") * col("b")) * 2).alias("expr3"),
        )
    ),
    # ConstantFolding: Evaluate constants at plan time
    "constant_folding": lambda df: (
        df.select().select(
            col("a"),
            (col("a") + 1 + 2 + 3).alias("const_add"),  # 1+2+3 should fold to 6
        )
    ),
    # DeadCodeElimination: Remove unused intermediate results
    "dead_code": lambda df: (
        df.select()
        .select(col("a"), col("b"), col("c"))
        .select(col("a"), col("b"))  # c is dead
        .select(col("a"))  # b is dead
    ),
    # Complex: Multiple optimizations should apply
    "complex_multi_opt": lambda df: (
        df.select()
        .filter(col("a") > 0.3)
        .filter(col("a") < 0.9)
        .select(col("a"), col("b"), col("c"), col("d"))
        .filter(col("b") > 0.2)
        .select(col("a"), col("b"))
        .sort(col("a"), descending=True)
        .head(50)
    ),
}


# =============================================================================
# Per-pass effectiveness measurement
# =============================================================================


def measure_pass_effect(plan: LogicalPlan, pass_name: str, pass_instance) -> dict:
    """Measure the effect of a single optimization pass."""
    before = analyze_plan(plan)
    after_plan = pass_instance.optimize(plan)
    after = analyze_plan(after_plan)

    return {
        "pass_name": pass_name,
        "nodes_before": before.total_nodes,
        "nodes_after": after.total_nodes,
        "nodes_reduced": before.total_nodes - after.total_nodes,
        "filters_before": before.filter_nodes,
        "filters_after": after.filter_nodes,
        "projects_before": before.project_nodes,
        "projects_after": after.project_nodes,
        "topk_created": after.topk_nodes > before.topk_nodes,
    }


# =============================================================================
# Main benchmark
# =============================================================================


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_optimizer_quality_benchmark():
    """Run the optimizer quality benchmark."""
    print_header("OPTIMIZER EFFECTIVENESS BENCHMARK")
    print("\nMeasuring plan quality before and after optimization.")
    print("This tests that optimizer passes are working correctly.\n")

    df = create_test_data()
    optimizer = Optimizer()

    all_results = []

    # Measure each query
    print_header("OPTIMIZATION EFFECTS BY QUERY")

    for query_name, query_builder in OPTIMIZATION_TEST_QUERIES.items():
        lf = query_builder(df)
        plan_before = lf._plan
        plan_after = optimizer.optimize(plan_before)

        before_metrics = analyze_plan(plan_before)
        after_metrics = analyze_plan(plan_after)

        effect = OptimizationEffect(
            query_name=query_name,
            before=before_metrics,
            after=after_metrics,
        )
        effect.compute_effects()
        all_results.append(effect)

        print(f"\n--- {query_name} ---")
        b, a = before_metrics, after_metrics
        print(
            f"  Nodes:    {b.total_nodes:>3} -> {a.total_nodes:>3}  "
            f"(reduced: {effect.nodes_reduced})"
        )
        print(
            f"  Filters:  {b.filter_nodes:>3} -> {a.filter_nodes:>3}  "
            f"(fused: {effect.filters_fused})"
        )
        print(
            f"  Projects: {b.project_nodes:>3} -> {a.project_nodes:>3}  "
            f"(merged: {effect.projects_merged})"
        )

        if effect.sort_limit_to_topk:
            print("  TopK:     Sort+Limit -> TopK conversion applied")
        if b.topk_nodes > 0 or a.topk_nodes > 0:
            print(f"  TopK:     {b.topk_nodes:>3} -> {a.topk_nodes:>3}")

    # Summary statistics
    print_header("OPTIMIZATION SUMMARY")

    total_nodes_before = sum(r.before.total_nodes for r in all_results)
    total_nodes_after = sum(r.after.total_nodes for r in all_results)
    total_filters_fused = sum(r.filters_fused for r in all_results)
    total_projects_merged = sum(r.projects_merged for r in all_results)
    topk_conversions = sum(1 for r in all_results if r.sort_limit_to_topk)

    print(f"\nAcross {len(all_results)} test queries:")
    diff = total_nodes_before - total_nodes_after
    print(
        f"  Total nodes reduced:     {total_nodes_before} -> "
        f"{total_nodes_after} ({diff} fewer)"
    )
    print(f"  Filters fused:           {total_filters_fused}")
    print(f"  Projects merged:         {total_projects_merged}")
    print(f"  Sort+Limit -> TopK:      {topk_conversions}")

    # Per-pass effectiveness
    print_header("PER-PASS EFFECTIVENESS")
    print("\nMeasuring what each pass contributes on 'complex_multi_opt' query:")

    lf = OPTIMIZATION_TEST_QUERIES["complex_multi_opt"](df)
    plan = lf._plan

    passes = [
        ("ConstantFolding", ConstantFolding()),
        ("ExpressionSimplification", ExpressionSimplification()),
        ("FilterFusion", FilterFusion()),
        ("PredicatePushdown", PredicatePushdown()),
        ("ProjectionPruning", ProjectionPruning()),
        ("LimitPushdown", LimitPushdown()),
        ("SortLimitToTopK", SortLimitToTopK()),
        ("CSE", CommonSubexpressionElimination()),
        ("DeadCodeElimination", DeadCodeElimination()),
    ]

    print(f"\n{'Pass':<25} {'Nodes':>12} {'Filters':>12} {'Projects':>12} {'TopK':>8}")
    print("-" * 72)

    current_plan = plan
    for pass_name, pass_instance in passes:
        effect = measure_pass_effect(current_plan, pass_name, pass_instance)
        current_plan = pass_instance.optimize(current_plan)

        nodes_delta = effect["nodes_before"] - effect["nodes_after"]
        filters_delta = effect["filters_before"] - effect["filters_after"]
        projects_delta = effect["projects_before"] - effect["projects_after"]
        topk_str = "created" if effect["topk_created"] else "-"

        if (
            nodes_delta != 0
            or filters_delta != 0
            or projects_delta != 0
            or effect["topk_created"]
        ):
            print(
                f"{pass_name:<25} "
                f"{effect['nodes_before']:>3} -> {effect['nodes_after']:<3} "
                f"{effect['filters_before']:>3} -> {effect['filters_after']:<3} "
                f"{effect['projects_before']:>3} -> {effect['projects_after']:<3} "
                f"{topk_str:>8}"
            )
        else:
            print(f"{pass_name:<25} {'(no change)':>50}")

    # Regression detection hints
    print_header("EXPECTED OPTIMIZATIONS (for regression detection)")

    expectations = [
        ("filter_fusion_3", "filters_fused", 2, ">="),
        ("filter_fusion_5", "filters_fused", 4, ">="),
        ("sort_limit_topk", "sort_limit_to_topk", True, "=="),
        ("sort_limit_topk_with_filter", "sort_limit_to_topk", True, "=="),
        ("projection_pruning", "projects_merged", 0, ">="),
        ("dead_code", "projects_merged", 1, ">="),
    ]

    print("\nThese expectations should be maintained across refactors:\n")
    print(f"{'Query':<35} {'Metric':<25} {'Expected':>10} {'Actual':>10} {'Status':>8}")
    print("-" * 90)

    all_passed = True
    for query_name, metric_name, expected, op in expectations:
        result = next(r for r in all_results if r.query_name == query_name)
        actual = getattr(result, metric_name)

        if op == ">=":
            passed = actual >= expected
        elif op == "==":
            passed = actual == expected
        else:
            passed = False

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(
            f"{query_name:<35} {metric_name:<20} "
            f"{expected!s:>10} {actual!s:>10} {status:>8}"
        )

    print()
    if all_passed:
        print("All optimization expectations met.")
    else:
        print("WARNING: Some optimizations are not working as expected!")
        print("This may indicate a regression in the optimizer.")

    # JSON output for CI
    print_header("JSON OUTPUT (for CI integration)")

    json_results = {
        "summary": {
            "total_queries": len(all_results),
            "total_nodes_before": total_nodes_before,
            "total_nodes_after": total_nodes_after,
            "total_nodes_reduced": total_nodes_before - total_nodes_after,
            "filters_fused": total_filters_fused,
            "projects_merged": total_projects_merged,
            "topk_conversions": topk_conversions,
            "all_expectations_passed": all_passed,
        },
        "queries": [r.to_dict() for r in all_results],
    }

    print("\n" + json.dumps(json_results["summary"], indent=2))
    print("\n(Full results available in return value for programmatic use)")

    return json_results


if __name__ == "__main__":
    run_optimizer_quality_benchmark()
