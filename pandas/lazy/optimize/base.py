"""
Base classes for query optimization.

This module provides:
- OptimizationPass: Abstract base class for optimization passes
- PlanVisitor: Base class with visitor pattern for plan traversal
- Optimizer: Coordinates multiple optimization passes
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas.lazy.plan import LogicalPlan


class OptimizationPass(ABC):
    """Base class for optimization passes."""

    @abstractmethod
    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """
        Transform a plan, returning optimized version.

        Parameters
        ----------
        plan : LogicalPlan
            The plan to optimize.

        Returns
        -------
        LogicalPlan
            The optimized plan (may be same object if no changes).
        """


class PlanVisitor(OptimizationPass):
    """
    Base class for optimization passes using the visitor pattern.

    Subclasses can override specific visit_* methods to customize behavior
    for particular node types, while the default implementation handles
    recursive traversal.

    This eliminates the boilerplate of manually recursing through all node
    types in each optimization pass.

    Example
    -------
    class MyPass(PlanVisitor):
        def visit_filter(self, plan: Filter) -> LogicalPlan:
            # Custom filter handling
            new_input = self.visit(plan.input)
            # ... custom logic ...
            return Filter(new_input, plan.predicate)
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply the visitor to transform the plan."""
        return self.visit(plan)

    def visit(self, plan: LogicalPlan) -> LogicalPlan:
        """
        Dispatch to the appropriate visit_* method based on node type.

        Parameters
        ----------
        plan : LogicalPlan
            The plan node to visit.

        Returns
        -------
        LogicalPlan
            The transformed plan node.
        """
        # Import here to avoid circular imports
        from pandas.lazy.plan import (
            Aggregate,
            DataFrameSource,
            Distinct,
            Filter,
            Join,
            Limit,
            Project,
            Sort,
            TopK,
        )

        # Dispatch based on node type
        if isinstance(plan, DataFrameSource):
            return self.visit_source(plan)
        elif isinstance(plan, Project):
            return self.visit_project(plan)
        elif isinstance(plan, Filter):
            return self.visit_filter(plan)
        elif isinstance(plan, Aggregate):
            return self.visit_aggregate(plan)
        elif isinstance(plan, Sort):
            return self.visit_sort(plan)
        elif isinstance(plan, Limit):
            return self.visit_limit(plan)
        elif isinstance(plan, Distinct):
            return self.visit_distinct(plan)
        elif isinstance(plan, Join):
            return self.visit_join(plan)
        elif isinstance(plan, TopK):
            return self.visit_topk(plan)
        else:
            # Handle Convert and any other node types
            return self._visit_default(plan)

    def _visit_default(self, plan: LogicalPlan) -> LogicalPlan:
        """
        Default handler for unknown node types.

        Tries to handle nodes with standard children() interface.
        """
        from pandas.lazy.plan import Convert

        if isinstance(plan, Convert):
            return self.visit_convert(plan)
        return plan

    # ==========================================================================
    # Default visit methods - subclasses override these
    # ==========================================================================

    def visit_source(self, plan) -> LogicalPlan:
        """Visit a DataFrameSource node. Default: return unchanged."""
        return plan

    def visit_project(self, plan) -> LogicalPlan:
        """Visit a Project node. Default: recurse into input."""
        from pandas.lazy.plan import Project

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return Project(new_input, plan.exprs)
        return plan

    def visit_filter(self, plan) -> LogicalPlan:
        """Visit a Filter node. Default: recurse into input."""
        from pandas.lazy.plan import Filter

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return Filter(new_input, plan.predicate)
        return plan

    def visit_aggregate(self, plan) -> LogicalPlan:
        """Visit an Aggregate node. Default: recurse into input."""
        from pandas.lazy.plan import Aggregate

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return Aggregate(new_input, plan.group_by, plan.agg_exprs)
        return plan

    def visit_sort(self, plan) -> LogicalPlan:
        """Visit a Sort node. Default: recurse into input."""
        from pandas.lazy.plan import Sort

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return Sort(new_input, plan.by, plan.descending)
        return plan

    def visit_limit(self, plan) -> LogicalPlan:
        """Visit a Limit node. Default: recurse into input."""
        from pandas.lazy.plan import Limit

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return Limit(new_input, plan.n, plan.offset)
        return plan

    def visit_distinct(self, plan) -> LogicalPlan:
        """Visit a Distinct node. Default: recurse into input."""
        from pandas.lazy.plan import Distinct

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return Distinct(new_input, plan.subset)
        return plan

    def visit_join(self, plan) -> LogicalPlan:
        """Visit a Join node. Default: recurse into both inputs."""
        from pandas.lazy.plan import Join

        new_left = self.visit(plan.left)
        new_right = self.visit(plan.right)
        if new_left is not plan.left or new_right is not plan.right:
            return Join(
                new_left,
                new_right,
                plan.on,
                plan.left_on,
                plan.right_on,
                plan.how,
                plan.suffix,
            )
        return plan

    def visit_topk(self, plan) -> LogicalPlan:
        """Visit a TopK node. Default: recurse into input."""
        from pandas.lazy.plan import TopK

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return TopK(new_input, plan.k, plan.by, plan.descending)
        return plan

    def visit_convert(self, plan) -> LogicalPlan:
        """Visit a Convert node. Default: recurse into input."""
        from pandas.lazy.plan import Convert

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return Convert(new_input, plan.target_backend)
        return plan


class Optimizer:
    """
    Coordinates optimization passes.

    Parameters
    ----------
    passes : list[OptimizationPass] or None
        Custom list of passes. If None, uses default passes.
    """

    def __init__(self, passes: list[OptimizationPass] | None = None) -> None:
        self.passes = passes if passes is not None else self._default_passes()

    def _default_passes(self) -> list[OptimizationPass]:
        """
        Default optimization pass ordering.

        IMPORTANT: Order matters! Rationale for each position:

        1. ConstantFolding - Evaluate constants first to simplify expressions
        2. DeadCodeElimination - Remove no-op nodes early (Filter(True), etc.)
        3. FilterFusion - Combine filters first so pushdown sees single filters
        4. PredicatePushdown - Push filters down while plan is still wide
           (before projection pruning removes columns that filters might need)
        5. AggregatePushdown - Push aggregates through pass-through projects
        6. ProjectionPruning - After pushdown, so we don't prune needed columns
        7. LimitPushdown - After pruning; push limits toward sources
        8. SortLimitToTopK - After limit pushdown, combine Sort+Limit into TopK

        Note: CommonSubexpressionElimination is available but not in default
        passes because it requires evaluator support for intermediate columns.
        """
        # Import here to avoid circular imports
        from pandas.lazy.optimize.passes import (
            AggregatePushdown,
            ConstantFolding,
            DeadCodeElimination,
            FilterFusion,
            LimitPushdown,
            PredicatePushdown,
            ProjectionPruning,
            SortLimitToTopK,
        )

        return [
            ConstantFolding(),
            DeadCodeElimination(),
            FilterFusion(),
            PredicatePushdown(),
            AggregatePushdown(),
            ProjectionPruning(),
            LimitPushdown(),
            SortLimitToTopK(),
        ]

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """
        Apply all optimization passes.

        Parameters
        ----------
        plan : LogicalPlan
            The plan to optimize.

        Returns
        -------
        LogicalPlan
            The optimized plan.
        """
        for pass_ in self.passes:
            plan = pass_.optimize(plan)
        return plan
