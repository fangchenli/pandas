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
        self._visit_memo: dict[int, LogicalPlan] = {}
        return self.visit(plan)

    def visit(self, plan: LogicalPlan) -> LogicalPlan:
        """
        Dispatch to the appropriate visit_* method based on node type.

        Memoized per ``optimize()`` run by input-node identity: LazyFrame
        reuse makes plans DAGs (one subtree object, several parents), and an
        unmemoized rebuild visits each parent's copy separately — returning
        two distinct rebuilt objects and silently destroying the sharing that
        common-subplan caching (PhysicalCachedSubplan) depends on. With the
        memo, an identical input node maps to the identical output object, so
        the DAG survives every pass.

        Parameters
        ----------
        plan : LogicalPlan
            The plan node to visit.

        Returns
        -------
        LogicalPlan
            The transformed plan node.
        """
        memo = getattr(self, "_visit_memo", None)
        if memo is not None:
            cached = memo.get(id(plan))
            if cached is not None:
                return cached
        result = self._dispatch(plan)
        if memo is not None:
            memo[id(plan)] = result
        return result

    def _dispatch(self, plan: LogicalPlan) -> LogicalPlan:
        # Import here to avoid circular imports
        from pandas.lazy.plan import (
            Aggregate,
            DataFrameSource,
            Distinct,
            Filter,
            GroupByHead,
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
        elif isinstance(plan, GroupByHead):
            return self.visit_group_by_head(plan)
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

    def visit_group_by_head(self, plan) -> LogicalPlan:
        """Visit a GroupByHead node. Default: recurse into input."""
        from pandas.lazy.plan import GroupByHead

        new_input = self.visit(plan.input)
        if new_input is not plan.input:
            return GroupByHead(new_input, plan.group_by, plan.n)
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

        Phase 1 - Normalize expressions early:
        1. ConstantFolding - Evaluate constants first to simplify expressions
        2. ExpressionSimplification - Algebraic simplifications (x*1->x, etc.)

        Phase 2 - Make filters simple and close to source:
        3. FilterFusion - Combine filters so pushdown sees single filters
        4. PredicatePushdown - Push filters down while plan is still wide
           (before projection pruning removes columns that filters might need)
        5. AggregatePushdown - Push aggregates through pass-through projects

        Phase 3 - Prune columns late (after we know what's needed):
        6. ProjectionPruning - After pushdown, so we don't prune needed columns
        7. LimitPushdown - After pruning; push limits toward sources
        8. SortLimitToTopK - After limit pushdown, combine Sort+Limit into TopK

        Phase 4 - Structural cleanups at the end:
        9. CommonSubexpressionElimination - Late because folding/simplification
           increases exact matches. Uses two-stage projection to hide temps.
        10. DeadCodeElimination - After structural rewrites to remove no-ops
            introduced by earlier passes (e.g., identity projections from CSE)

        Phase 5 - Backend selection:
        11. EngineSelection - Analyze backend requirements, insert Convert nodes
        12. ConversionElimination - Remove redundant/unnecessary conversions
        """
        # Import here to avoid circular imports
        from pandas.lazy.optimize.engine import (
            ConversionElimination,
            EngineSelection,
        )
        from pandas.lazy.optimize.join_reorder import JoinReorder
        from pandas.lazy.optimize.passes import (
            AggregatePushdown,
            CommonSubexpressionElimination,
            ConstantFolding,
            DeadCodeElimination,
            ExpressionSimplification,
            FilterFusion,
            LimitPushdown,
            PredicatePushdown,
            ProjectionPruning,
            SortLimitToTopK,
        )

        # Cost-based join reordering is experimental and OFF by default: it
        # helps badly-ordered multi-join queries, but the sampled-NDV
        # cardinality model is not reliable enough to avoid regressing
        # well-ordered many-table joins (see join_reorder.py). Opt in via
        # ``compute.lazy.join_reorder``.
        join_reorder_enabled = False
        try:
            from pandas import get_option

            join_reorder_enabled = bool(get_option("compute.lazy.join_reorder"))
        except Exception:
            pass

        passes = [
            # Phase 1: Normalize expressions
            ConstantFolding(),
            ExpressionSimplification(),
            # Phase 2: Push filters toward sources
            FilterFusion(),
            PredicatePushdown(),
            AggregatePushdown(),
        ]
        if join_reorder_enabled:
            # After pushdown (relation sizes reflect filters) and before pruning
            # (join tree still intact). Bails to the original order on ambiguity.
            passes.append(JoinReorder())
        passes += [
            # Phase 3: Prune and limit
            ProjectionPruning(),
            LimitPushdown(),
            SortLimitToTopK(),
            # Phase 4: Structural cleanups
            CommonSubexpressionElimination(),
            DeadCodeElimination(),
            # Phase 5: Backend selection
            EngineSelection(),
            ConversionElimination(),
        ]
        return passes

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
