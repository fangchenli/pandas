"""
Base classes for query optimization.

This module provides:
- OptimizationPass: Abstract base class for optimization passes
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
        2. FilterFusion - Combine filters first so pushdown sees single filters
        3. PredicatePushdown - Push filters down while plan is still wide
           (before projection pruning removes columns that filters might need)
        4. ProjectionPruning - After pushdown, so we don't prune filter-needed cols
        5. LimitPushdown - After pruning; push limits toward sources
        6. SortLimitToTopK - After limit pushdown, combine Sort+Limit into TopK
        """
        # Import here to avoid circular imports
        from pandas.lazy.optimize.passes import (
            ConstantFolding,
            FilterFusion,
            LimitPushdown,
            PredicatePushdown,
            ProjectionPruning,
            SortLimitToTopK,
        )

        return [
            ConstantFolding(),
            FilterFusion(),
            PredicatePushdown(),
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
