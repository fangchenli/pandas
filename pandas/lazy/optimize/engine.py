"""
Engine selection and backend-related optimization passes.

This module contains:
- BackendRequirements: Tracks backend requirements for operations
- EngineSelection: Insert Convert nodes for backend decisions
- ConversionElimination: Remove redundant conversions
"""

from __future__ import annotations

from dataclasses import dataclass

from pandas.lazy.ir import (
    Alias,
    Call,
    Cast,
    FieldRef,
    IRNode,
    Literal,
)
from pandas.lazy.optimize.base import PlanVisitor
from pandas.lazy.plan import (
    Convert,
    DataFrameSource,
    Filter,
    LogicalPlan,
    Project,
)

# =============================================================================
# Backend Requirements Analysis
# =============================================================================


@dataclass(frozen=True)
class BackendRequirements:
    """
    Rich backend analysis for an expression.

    Attributes
    ----------
    supported_backends : frozenset[str]
        Backends that CAN execute this operation: {"arrow", "numpy"} or subset.
    preferred_backend : str | None
        Backend that executes most efficiently, if there's a clear winner.
        None means no strong preference (both equally good).
    requires_backend : str | None
        Backend that MUST be used (no alternative). E.g., some string ops
        only have Arrow implementations in our codebase.
    conversion_cost : float
        Estimated cost if conversion is needed (0.0 to 1.0 scale).
        Higher = more expensive. Used to break ties.
    """

    supported_backends: frozenset[str]
    preferred_backend: str | None = None
    requires_backend: str | None = None
    conversion_cost: float = 0.0

    @classmethod
    def any_backend(cls) -> BackendRequirements:
        """Operation works equally well on any backend."""
        return cls(frozenset({"arrow", "numpy"}), None, None, 0.0)

    @classmethod
    def prefer_arrow(cls, required: bool = False) -> BackendRequirements:
        """Operation prefers or requires Arrow."""
        if required:
            return cls(frozenset({"arrow"}), "arrow", "arrow", 0.0)
        return cls(frozenset({"arrow", "numpy"}), "arrow", None, 0.3)

    @classmethod
    def prefer_numpy(cls, required: bool = False) -> BackendRequirements:
        """Operation prefers or requires NumPy."""
        if required:
            return cls(frozenset({"numpy"}), "numpy", "numpy", 0.0)
        return cls(frozenset({"arrow", "numpy"}), "numpy", None, 0.3)


# Operation -> Backend requirements mapping
# This is the "kernel registry" - which backends support which ops
BACKEND_KERNELS: dict[str, BackendRequirements] = {
    # Arithmetic - both backends efficient
    "add": BackendRequirements.any_backend(),
    "subtract": BackendRequirements.any_backend(),
    "multiply": BackendRequirements.any_backend(),
    "divide": BackendRequirements.any_backend(),
    "negate": BackendRequirements.any_backend(),
    "abs": BackendRequirements.any_backend(),
    "floor_divide": BackendRequirements.any_backend(),
    "modulo": BackendRequirements.any_backend(),
    "power": BackendRequirements.any_backend(),
    # Comparison - both backends efficient
    "equal": BackendRequirements.any_backend(),
    "not_equal": BackendRequirements.any_backend(),
    "less": BackendRequirements.any_backend(),
    "greater": BackendRequirements.any_backend(),
    "less_equal": BackendRequirements.any_backend(),
    "greater_equal": BackendRequirements.any_backend(),
    # Logical - both backends
    "and_": BackendRequirements.any_backend(),
    "or_": BackendRequirements.any_backend(),
    "invert": BackendRequirements.any_backend(),
    # String operations - Arrow strongly preferred (or required)
    "str_lower": BackendRequirements.prefer_arrow(required=True),
    "str_upper": BackendRequirements.prefer_arrow(required=True),
    "str_len": BackendRequirements.prefer_arrow(required=True),
    "str_contains": BackendRequirements.prefer_arrow(required=True),
    "str_startswith": BackendRequirements.prefer_arrow(required=True),
    "str_endswith": BackendRequirements.prefer_arrow(required=True),
    "str_strip": BackendRequirements.prefer_arrow(required=True),
    "str_lstrip": BackendRequirements.prefer_arrow(required=True),
    "str_rstrip": BackendRequirements.prefer_arrow(required=True),
    "str_replace": BackendRequirements.prefer_arrow(required=True),
    "str_slice": BackendRequirements.prefer_arrow(required=True),
    # Datetime - both have support, Arrow slightly preferred
    "dt_year": BackendRequirements.prefer_arrow(),
    "dt_month": BackendRequirements.prefer_arrow(),
    "dt_day": BackendRequirements.prefer_arrow(),
    "dt_hour": BackendRequirements.prefer_arrow(),
    "dt_minute": BackendRequirements.prefer_arrow(),
    "dt_second": BackendRequirements.prefer_arrow(),
    "dt_weekday": BackendRequirements.prefer_arrow(),
    "dt_dayofyear": BackendRequirements.prefer_arrow(),
    "dt_quarter": BackendRequirements.prefer_arrow(),
    "dt_date": BackendRequirements.prefer_arrow(),
    "dt_is_month_start": BackendRequirements.prefer_arrow(),
    "dt_is_month_end": BackendRequirements.prefer_arrow(),
    "dt_is_year_start": BackendRequirements.prefer_arrow(),
    "dt_is_year_end": BackendRequirements.prefer_arrow(),
    # Aggregations - both efficient
    "sum": BackendRequirements.any_backend(),
    "mean": BackendRequirements.any_backend(),
    "min": BackendRequirements.any_backend(),
    "max": BackendRequirements.any_backend(),
    "count": BackendRequirements.any_backend(),
    "std": BackendRequirements.any_backend(),
    "var": BackendRequirements.any_backend(),
    # Null handling - Arrow has native support
    "is_null": BackendRequirements.prefer_arrow(),
    "is_not_null": BackendRequirements.prefer_arrow(),
    "fill_null": BackendRequirements.any_backend(),
    "coalesce": BackendRequirements.any_backend(),
    # Window functions - both backends
    "window": BackendRequirements.any_backend(),
    "rank": BackendRequirements.any_backend(),
    "dense_rank": BackendRequirements.any_backend(),
    "row_number": BackendRequirements.any_backend(),
    "lag": BackendRequirements.any_backend(),
    "lead": BackendRequirements.any_backend(),
    "cum_sum": BackendRequirements.any_backend(),
    "cum_min": BackendRequirements.any_backend(),
    "cum_max": BackendRequirements.any_backend(),
    "cum_mean": BackendRequirements.any_backend(),
    "cum_prod": BackendRequirements.any_backend(),
    # Conditional
    "case_when": BackendRequirements.any_backend(),
}


def analyze_backend_requirements(
    expr: IRNode,
    schema,
) -> BackendRequirements:
    """
    Analyze backend requirements for an expression tree.

    This considers:
    1. The operation's kernel availability
    2. Input column storage backends
    3. Nested expression requirements

    Returns combined requirements that satisfy the whole expression.
    """
    if isinstance(expr, FieldRef):
        # Column reference - report storage backend, but no execution requirement
        dtype = schema[expr.name]
        storage = dtype.storage_backend
        # Storage is a hint, not a requirement - we CAN convert
        return BackendRequirements(
            supported_backends=frozenset({"arrow", "numpy"}),
            preferred_backend=storage,
            requires_backend=None,
            conversion_cost=0.0,
        )

    if isinstance(expr, Literal):
        # Literals can be used in any backend
        return BackendRequirements.any_backend()

    if isinstance(expr, Call):
        # Look up kernel requirements
        op_reqs = BACKEND_KERNELS.get(expr.function, BackendRequirements.any_backend())

        # Analyze children
        child_reqs = [analyze_backend_requirements(arg, schema) for arg in expr.args]

        # Combine requirements
        return _combine_requirements(op_reqs, child_reqs)

    if isinstance(expr, Alias):
        return analyze_backend_requirements(expr.arg, schema)

    if isinstance(expr, Cast):
        return analyze_backend_requirements(expr.arg, schema)

    return BackendRequirements.any_backend()


def _combine_requirements(
    op_reqs: BackendRequirements,
    child_reqs: list[BackendRequirements],
) -> BackendRequirements:
    """
    Combine operation requirements with child requirements.

    Rules:
    - If op requires a backend, that wins
    - Otherwise, intersect supported backends
    - Preferred backend: op's preference, or majority of children
    - Conversion cost: sum of costs if mismatched
    """
    if op_reqs.requires_backend:
        # Operation requires specific backend
        required = op_reqs.requires_backend
        # Calculate conversion cost for children not in required backend
        cost = sum(
            0.5
            for c in child_reqs
            if c.preferred_backend and c.preferred_backend != required
        )
        return BackendRequirements(
            supported_backends=frozenset({required}),
            preferred_backend=required,
            requires_backend=required,
            conversion_cost=cost,
        )

    # Intersect supported backends
    supported = op_reqs.supported_backends
    for c in child_reqs:
        if c.requires_backend:
            supported = supported & frozenset({c.requires_backend})

    if not supported:
        # No common backend - this shouldn't happen with good kernel registry
        supported = frozenset({"numpy"})  # Fallback

    # Determine preferred backend
    # Priority: op preference > majority of children > "numpy" default
    if op_reqs.preferred_backend and op_reqs.preferred_backend in supported:
        preferred = op_reqs.preferred_backend
    else:
        # Count child preferences
        arrow_count = sum(1 for c in child_reqs if c.preferred_backend == "arrow")
        numpy_count = sum(1 for c in child_reqs if c.preferred_backend == "numpy")
        if arrow_count > numpy_count and "arrow" in supported:
            preferred = "arrow"
        elif numpy_count > 0 and "numpy" in supported:
            preferred = "numpy"
        elif supported:
            preferred = next(iter(supported))
        else:
            preferred = "numpy"

    # Compute conversion cost
    cost = sum(c.conversion_cost for c in child_reqs)
    # Add cost for children that need conversion to preferred
    cost += sum(
        0.3
        for c in child_reqs
        if c.preferred_backend and c.preferred_backend != preferred
    )

    return BackendRequirements(
        supported_backends=supported,
        preferred_backend=preferred,
        requires_backend=None,
        conversion_cost=cost,
    )


# =============================================================================
# EngineSelection Pass
# =============================================================================


class EngineSelection(PlanVisitor):
    """
    Analyze expression backend requirements and insert explicit Convert nodes.

    This runs AFTER other optimizations (predicate pushdown, etc.) because:
    1. Those passes may move operations that affect backend decisions
    2. We want the final logical structure before deciding on backends

    Algorithm:
    1. For each plan node, analyze backend requirements of all expressions
    2. Determine the "dominant" backend for each node
    3. Insert Convert nodes at boundaries where backend changes
    """

    def visit_project(self, plan: Project) -> LogicalPlan:
        new_input = self.visit(plan.input)
        new_plan = (
            Project(new_input, plan.exprs) if new_input is not plan.input else plan
        )
        return self._select_project_engine(new_plan)

    def visit_filter(self, plan: Filter) -> LogicalPlan:
        new_input = self.visit(plan.input)
        new_plan = (
            Filter(new_input, plan.predicate) if new_input is not plan.input else plan
        )
        return self._select_filter_engine(new_plan)

    def _select_project_engine(self, project: Project) -> LogicalPlan:
        """Determine backend for Project and insert conversion if needed."""

        schema = project.input.resolve_schema()

        # Analyze all expressions
        expr_reqs = [analyze_backend_requirements(e._ir, schema) for e in project.exprs]

        # Find required backends
        required_arrow = any(r.requires_backend == "arrow" for r in expr_reqs)
        required_numpy = any(r.requires_backend == "numpy" for r in expr_reqs)

        if required_arrow and required_numpy:
            # Conflict - for MVP, prefer arrow (string ops more common)
            target = "arrow"
        elif required_arrow:
            target = "arrow"
        elif required_numpy:
            target = "numpy"
        else:
            # Use preferred backend from majority
            arrow_pref = sum(1 for r in expr_reqs if r.preferred_backend == "arrow")
            numpy_pref = sum(1 for r in expr_reqs if r.preferred_backend == "numpy")
            target = "arrow" if arrow_pref > numpy_pref else "numpy"

        # Check if input needs conversion
        input_backend = self._get_plan_backend(project.input)
        if input_backend not in (target, "auto"):
            converted_input = Convert(project.input, target)
            return Project(converted_input, project.exprs)

        return project

    def _select_filter_engine(self, filter_plan: Filter) -> LogicalPlan:
        """Determine backend for Filter and insert conversion if needed."""

        schema = filter_plan.input.resolve_schema()

        # Analyze predicate
        pred_reqs = analyze_backend_requirements(filter_plan.predicate._ir, schema)

        if pred_reqs.requires_backend:
            target = pred_reqs.requires_backend
        elif pred_reqs.preferred_backend:
            target = pred_reqs.preferred_backend
        else:
            target = self._get_plan_backend(filter_plan.input)
            if target == "auto":
                target = "numpy"

        # Check if input needs conversion
        input_backend = self._get_plan_backend(filter_plan.input)
        if input_backend not in (target, "auto"):
            converted_input = Convert(filter_plan.input, target)
            return Filter(converted_input, filter_plan.predicate)

        return filter_plan

    def _get_plan_backend(self, plan: LogicalPlan) -> str:
        """
        Determine the output backend of a plan node.

        Returns "auto" if mixed or unknown.
        """

        if isinstance(plan, Convert):
            return plan.target_backend
        elif isinstance(plan, DataFrameSource):
            schema = plan.resolve_schema()
            backends = {schema[n].storage_backend for n in schema.names}
            if len(backends) == 1:
                return backends.pop()
            return "auto"
        else:
            children = plan.children()
            if children:
                return self._get_plan_backend(children[0])
            return "auto"


# =============================================================================
# ConversionElimination Pass
# =============================================================================


class ConversionElimination(PlanVisitor):
    """
    Eliminate redundant or unnecessary Convert nodes.

    Patterns eliminated:
    1. Convert(Convert(x, "arrow"), "arrow") -> Convert(x, "arrow")
    2. Convert(x, backend) where x already produces that backend
    3. Back-to-back converts: Convert(Convert(x, A), B) -> Convert(x, B)
    """

    def visit_convert(self, plan: Convert) -> LogicalPlan:
        new_input = self.visit(plan.input)

        # Pattern 1 & 3: Nested converts
        if isinstance(new_input, Convert):
            if new_input.target_backend == plan.target_backend:
                # Convert(Convert(x, A), A) -> Convert(x, A)
                return Convert(new_input.input, plan.target_backend)
            else:
                # Convert(Convert(x, A), B) -> Convert(x, B)
                return Convert(new_input.input, plan.target_backend)

        # Pattern 2: Unnecessary convert
        input_backend = self._get_plan_backend(new_input)
        if input_backend == plan.target_backend:
            return new_input

        if new_input is not plan.input:
            return Convert(new_input, plan.target_backend)
        return plan

    def _get_plan_backend(self, plan: LogicalPlan) -> str:
        """Determine the output backend of a plan node."""
        if isinstance(plan, Convert):
            return plan.target_backend
        elif isinstance(plan, DataFrameSource):
            schema = plan.resolve_schema()
            backends = {schema[n].storage_backend for n in schema.names}
            if len(backends) == 1:
                return backends.pop()
            return "auto"
        else:
            children = plan.children()
            if children:
                return self._get_plan_backend(children[0])
            return "auto"
