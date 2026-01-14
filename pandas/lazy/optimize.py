"""
Query optimization for lazy pandas.

This module implements a rule-based optimizer with multiple transformation passes.
Each pass traverses the logical plan tree and applies transformations.

Pass Ordering (critical for correctness):
1. ConstantFolding - Evaluate constant expressions first
2. FilterFusion - Combine consecutive filters
3. PredicatePushdown - Push filters toward data sources
4. ProjectionPruning - Remove unused columns
5. LimitPushdown - Push limits toward sources
6. SortLimitToTopK - Combine Sort+Limit into TopK
7. EngineSelection - Insert Convert nodes for backend decisions
8. ConversionElimination - Remove redundant conversions
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pandas.lazy.ir import (
    Alias,
    Call,
    Cast,
    FieldRef,
    IRNode,
    Literal,
)
from pandas.lazy.plan import (
    Aggregate,
    DataFrameSource,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Sort,
    TopK,
)

if TYPE_CHECKING:
    from pandas.lazy.expr import Expr


# =============================================================================
# Utility Functions
# =============================================================================


def get_referenced_columns(node: IRNode | Expr) -> set[str]:
    """
    Extract all column names referenced in an expression.

    Parameters
    ----------
    node : IRNode or Expr
        The expression to analyze.

    Returns
    -------
    set[str]
        Set of column names referenced in the expression.

    Examples
    --------
    >>> get_referenced_columns(FieldRef("a"))
    {'a'}
    >>> get_referenced_columns(Call("add", (FieldRef("a"), FieldRef("b"))))
    {'a', 'b'}
    """
    # Handle Expr wrapper
    if hasattr(node, "_ir"):
        node = node._ir

    columns: set[str] = set()
    _collect_columns(node, columns)
    return columns


def _collect_columns(node: IRNode, columns: set[str]) -> None:
    """Recursively collect column references from an IR node."""
    if isinstance(node, FieldRef):
        columns.add(node.name)
    elif isinstance(node, Alias):
        _collect_columns(node.arg, columns)
    elif isinstance(node, Call):
        for arg in node.args:
            _collect_columns(arg, columns)
        # Handle kwargs that might contain IR nodes (e.g., case_when)
        for value in node.kwargs.values():
            if isinstance(value, IRNode):
                _collect_columns(value, columns)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, tuple):
                        # case_when cases: ((condition, value), ...)
                        for sub_item in item:
                            if isinstance(sub_item, IRNode):
                                _collect_columns(sub_item, columns)
                    elif isinstance(item, IRNode):
                        _collect_columns(item, columns)
    elif isinstance(node, Cast):
        _collect_columns(node.arg, columns)
    elif isinstance(node, Literal):
        pass  # Literals don't reference columns


def substitute_columns(node: IRNode, mapping: dict[str, str]) -> IRNode:
    """
    Rename column references in an expression tree.

    Parameters
    ----------
    node : IRNode
        The expression to transform.
    mapping : dict[str, str]
        Mapping from old column names to new column names.

    Returns
    -------
    IRNode
        New expression with column names substituted.
    """
    if isinstance(node, FieldRef):
        new_name = mapping.get(node.name, node.name)
        return FieldRef(new_name) if new_name != node.name else node

    elif isinstance(node, Alias):
        new_arg = substitute_columns(node.arg, mapping)
        return Alias(new_arg, node.name) if new_arg is not node.arg else node

    elif isinstance(node, Call):
        new_args = tuple(substitute_columns(arg, mapping) for arg in node.args)
        # Handle kwargs
        new_kwargs = {}
        kwargs_changed = False
        for key, value in node.kwargs.items():
            if isinstance(value, IRNode):
                new_value = substitute_columns(value, mapping)
                if new_value is not value:
                    kwargs_changed = True
                new_kwargs[key] = new_value
            else:
                new_kwargs[key] = value

        if new_args != node.args or kwargs_changed:
            return Call(node.function, new_args, new_kwargs, node.is_aggregate)
        return node

    elif isinstance(node, Cast):
        new_arg = substitute_columns(node.arg, mapping)
        return Cast(new_arg, node.target_dtype) if new_arg is not node.arg else node

    return node


def extract_output_name(expr: Expr | IRNode) -> str:
    """
    Extract the output column name from an expression.

    Parameters
    ----------
    expr : Expr or IRNode
        The expression to analyze.

    Returns
    -------
    str
        The output column name.

    Raises
    ------
    ValueError
        If no output name can be determined.
    """
    # Handle Expr wrapper
    if hasattr(expr, "_ir"):
        node = expr._ir
    else:
        node = expr

    if isinstance(node, Alias):
        return node.name
    elif isinstance(node, FieldRef):
        return node.name
    else:
        raise ValueError(f"Cannot determine output name for {node!r}")


# =============================================================================
# Lineage Tracking (for predicate pushdown)
# =============================================================================


def build_project_lineage(project: Project) -> dict[str, IRNode]:
    """
    Build mapping from output column names to their source expressions.

    This is used for predicate pushdown to determine if a column is:
    - A pass-through column (FieldRef with same name)
    - A renamed column (FieldRef with different name via Alias)
    - A computed column (any other expression)

    Parameters
    ----------
    project : Project
        The Project node to analyze.

    Returns
    -------
    dict[str, IRNode]
        Mapping from output column name to source expression.
        The source expression has Alias unwrapped.
    """
    lineage: dict[str, IRNode] = {}
    for expr in project.exprs:
        ir = expr._ir
        output_name = extract_output_name(ir)

        # Unwrap Alias to get the actual expression
        if isinstance(ir, Alias):
            source_ir = ir.arg
        else:
            source_ir = ir

        lineage[output_name] = source_ir
    return lineage


def get_source_column_name(lineage: dict[str, IRNode], col_name: str) -> str | None:
    """
    Get the source column name for a column that is pass-through or renamed.

    Parameters
    ----------
    lineage : dict[str, IRNode]
        Lineage mapping from build_project_lineage.
    col_name : str
        Output column name to trace.

    Returns
    -------
    str or None
        The source column name if the column is pass-through or a simple rename,
        None if the column is computed or doesn't exist.
    """
    if col_name not in lineage:
        return None

    source = lineage[col_name]
    if isinstance(source, FieldRef):
        return source.name
    return None


def is_pass_through_column(lineage: dict[str, IRNode], col_name: str) -> bool:
    """
    Check if a column is a pass-through (unchanged from input).

    A column is pass-through if:
    - It's in the lineage
    - Its source is a FieldRef with the same name

    Parameters
    ----------
    lineage : dict[str, IRNode]
        Lineage mapping from build_project_lineage.
    col_name : str
        Column name to check.

    Returns
    -------
    bool
        True if the column is pass-through.
    """
    if col_name not in lineage:
        return False
    source = lineage[col_name]
    return isinstance(source, FieldRef) and source.name == col_name


def can_push_predicate_through_project(
    predicate_cols: set[str],
    lineage: dict[str, IRNode],
) -> tuple[bool, dict[str, str]]:
    """
    Determine if a predicate can be pushed through a Project.

    A predicate can be pushed if ALL referenced columns are either:
    - Pass-through columns (same name in input)
    - Simple renames (FieldRef with different name)

    Parameters
    ----------
    predicate_cols : set[str]
        Column names referenced in the predicate.
    lineage : dict[str, IRNode]
        Lineage mapping from build_project_lineage.

    Returns
    -------
    tuple[bool, dict[str, str]]
        (can_push, column_mapping)
        - can_push: True if predicate can be pushed
        - column_mapping: Mapping from predicate column names to input column names
    """
    column_mapping: dict[str, str] = {}

    for col in predicate_cols:
        source_name = get_source_column_name(lineage, col)
        if source_name is None:
            # Column is computed, cannot push
            return False, {}
        column_mapping[col] = source_name

    return True, column_mapping


# =============================================================================
# Join Column Mapping (for predicate pushdown through joins)
# =============================================================================


@dataclass
class JoinColumnMapping:
    """
    Tracks where columns in a Join output come from.

    Used by optimizer to:
    - Push predicates to correct side
    - Prune columns from correct input
    """

    left_columns: dict[str, str]  # output_name -> left_input_name
    right_columns: dict[str, str]  # output_name -> right_input_name
    join_columns: set[str]  # columns used for join (appear once)


def build_join_column_mapping(join: Join) -> JoinColumnMapping:
    """Build mapping from join output columns to their source."""
    left_schema = join.left.resolve_schema()
    right_schema = join.right.resolve_schema()

    left_cols: dict[str, str] = {}
    right_cols: dict[str, str] = {}
    join_cols: set[str] = set()

    # Determine join columns
    if join.on:
        join_cols = set(join.on)

    # Map left columns
    for name in left_schema.names:
        if name in right_schema.names and name not in join_cols:
            # Overlapping -> gets suffix
            output_name = name + join.suffix[0]
        else:
            output_name = name
        left_cols[output_name] = name

    # Map right columns
    for name in right_schema.names:
        if join.on and name in join_cols:
            continue  # Skip, already from left
        if name in left_schema.names and name not in join_cols:
            output_name = name + join.suffix[1]
        else:
            output_name = name
        right_cols[output_name] = name

    return JoinColumnMapping(left_cols, right_cols, join_cols)


def can_push_predicate_through_join(
    predicate_cols: set[str],
    join_mapping: JoinColumnMapping,
) -> tuple[bool, bool, set[str], set[str]]:
    """
    Determine if/how predicate can be pushed through join.

    Parameters
    ----------
    predicate_cols : set[str]
        Column names referenced in the predicate.
    join_mapping : JoinColumnMapping
        Column mapping from build_join_column_mapping.

    Returns
    -------
    tuple[bool, bool, set[str], set[str]]
        (can_push_left, can_push_right, left_cols, right_cols)

    Rules:
    - Predicate on only left columns -> push to left
    - Predicate on only right columns -> push to right
    - Predicate on join columns -> push to BOTH sides
    - Predicate mixing left/right non-join columns -> cannot push
    """
    left_refs: set[str] = set()
    right_refs: set[str] = set()

    for col in predicate_cols:
        if col in join_mapping.join_columns:
            # Join column exists in both - can push to either
            left_refs.add(col)
            right_refs.add(col)
        elif col in join_mapping.left_columns:
            left_refs.add(join_mapping.left_columns[col])
        elif col in join_mapping.right_columns:
            right_refs.add(join_mapping.right_columns[col])
        else:
            # Column doesn't exist in join output
            return (False, False, set(), set())

    # Can only push if ALL columns are from one side (or join columns)
    only_left = len(right_refs - join_mapping.join_columns) == 0
    only_right = len(left_refs - join_mapping.join_columns) == 0

    return (only_left, only_right, left_refs, right_refs)


# =============================================================================
# Base Classes
# =============================================================================


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
        return [
            ConstantFolding(),
            FilterFusion(),
            PredicatePushdown(),
            ProjectionPruning(),
            LimitPushdown(),
            SortLimitToTopK(),
        ]

    def optimize(self, plan: LogicalPlan, max_iterations: int = 3) -> LogicalPlan:
        """
        Apply all optimization passes iteratively until fixpoint.

        The optimizer runs passes multiple times because one pass may create
        new optimization opportunities for another. For example:
        - PredicatePushdown may expose new opportunities for ProjectionPruning
        - FilterFusion after pushdown may enable SortLimitToTopK

        Parameters
        ----------
        plan : LogicalPlan
            The plan to optimize.
        max_iterations : int, default 3
            Maximum number of optimization iterations. Each iteration runs
            all passes once. Stops early if no changes are made.

        Returns
        -------
        LogicalPlan
            The optimized plan.
        """
        for _ in range(max_iterations):
            prev_plan = plan
            for pass_ in self.passes:
                plan = pass_.optimize(plan)

            # Check if plan changed (simple equality check on repr)
            # This is a heuristic - full structural equality would be more robust
            if repr(plan) == repr(prev_plan):
                break

        return plan


# =============================================================================
# ConstantFolding Pass
# =============================================================================


class ConstantFolding(OptimizationPass):
    """
    Evaluate constant expressions at optimization time.

    Transforms:
        lit(1) + lit(2) -> lit(3)
        lit("hello") + lit(" world") -> lit("hello world")
        lit(True) & lit(False) -> lit(False)

    This reduces runtime computation and enables further optimizations.
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform_plan(plan)

    def _transform_plan(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan, folding constants in expressions."""
        if isinstance(plan, DataFrameSource):
            return plan

        elif isinstance(plan, Project):
            new_input = self._transform_plan(plan.input)
            new_exprs = tuple(self._fold_expr(e) for e in plan.exprs)
            if new_input is not plan.input or new_exprs != plan.exprs:
                return Project(new_input, new_exprs)
            return plan

        elif isinstance(plan, Filter):
            new_input = self._transform_plan(plan.input)
            new_pred = self._fold_expr(plan.predicate)
            if new_input is not plan.input or new_pred is not plan.predicate:
                return Filter(new_input, new_pred)
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform_plan(plan.input)
            new_group_by = tuple(self._fold_expr(e) for e in plan.group_by)
            new_agg_exprs = tuple(self._fold_expr(e) for e in plan.agg_exprs)
            if (
                new_input is not plan.input
                or new_group_by != plan.group_by
                or new_agg_exprs != plan.agg_exprs
            ):
                return Aggregate(new_input, new_group_by, new_agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform_plan(plan.input)
            new_by = tuple(self._fold_expr(e) for e in plan.by)
            if new_input is not plan.input or new_by != plan.by:
                return Sort(new_input, new_by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._transform_plan(plan.input)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform_plan(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform_plan(plan.left)
            new_right = self._transform_plan(plan.right)
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

        elif isinstance(plan, TopK):
            new_input = self._transform_plan(plan.input)
            new_by = tuple(self._fold_expr(e) for e in plan.by)
            if new_input is not plan.input or new_by != plan.by:
                return TopK(new_input, plan.k, new_by, plan.descending)
            return plan

        return plan

    def _fold_expr(self, expr: Expr) -> Expr:
        """Fold constants in an expression."""
        from pandas.lazy.expr import Expr

        new_ir = self._fold_ir(expr._ir)
        if new_ir is not expr._ir:
            return Expr(new_ir)
        return expr

    def _fold_ir(self, node: IRNode) -> IRNode:
        """Recursively fold constants in an IR node."""
        if isinstance(node, Literal):
            return node

        elif isinstance(node, FieldRef):
            return node

        elif isinstance(node, Alias):
            new_arg = self._fold_ir(node.arg)
            if new_arg is not node.arg:
                return Alias(new_arg, node.name)
            return node

        elif isinstance(node, Cast):
            new_arg = self._fold_ir(node.arg)
            # If casting a literal, try to fold
            if isinstance(new_arg, Literal):
                try:
                    folded = self._cast_literal(new_arg.value, node.target_dtype)
                    if folded is not None:
                        return Literal(folded)
                except (ValueError, TypeError):
                    pass
            if new_arg is not node.arg:
                return Cast(new_arg, node.target_dtype)
            return node

        elif isinstance(node, Call):
            # First, fold children
            new_args = tuple(self._fold_ir(arg) for arg in node.args)

            # Check if all args are literals
            if all(isinstance(arg, Literal) for arg in new_args):
                folded = self._fold_call(node.function, new_args)
                if folded is not None:
                    return folded

            if new_args != node.args:
                return Call(node.function, new_args, node.kwargs, node.is_aggregate)
            return node

        return node

    def _cast_literal(self, value, target_dtype):
        """Cast a literal value to a target dtype."""
        import numpy as np

        if hasattr(target_dtype, "numpy_dtype"):
            target_dtype = target_dtype.numpy_dtype

        if target_dtype == np.dtype("int64"):
            return int(value)
        elif target_dtype == np.dtype("float64"):
            return float(value)
        elif target_dtype == np.dtype("bool"):
            return bool(value)

        return None

    def _fold_call(self, function: str, args: tuple[IRNode, ...]) -> IRNode | None:
        """Try to fold a function call with literal arguments."""
        # Extract values from literals
        values = [arg.value for arg in args if isinstance(arg, Literal)]
        if len(values) != len(args):
            return None

        try:
            result = self._evaluate_function(function, values)
            if result is not None:
                return Literal(result)
        except (ValueError, TypeError, ZeroDivisionError, ArithmeticError):
            pass

        return None

    def _evaluate_function(self, function: str, values: list):
        """Evaluate a function with literal values."""
        if len(values) == 0:
            return None

        if len(values) == 1:
            v = values[0]
            # Unary operations
            if function == "negate":
                return -v
            elif function == "abs":
                return abs(v)
            elif function == "invert":
                return not v

        elif len(values) == 2:
            a, b = values
            # Binary arithmetic
            if function == "add":
                return a + b
            elif function == "subtract":
                return a - b
            elif function == "multiply":
                return a * b
            elif function == "divide":
                if b == 0:
                    return None  # Don't fold division by zero
                return a / b
            elif function == "floor_divide":
                if b == 0:
                    return None
                return a // b
            elif function == "modulo":
                if b == 0:
                    return None
                return a % b
            elif function == "power":
                return a**b
            # Comparison
            elif function == "equal":
                return a == b
            elif function == "not_equal":
                return a != b
            elif function == "less":
                return a < b
            elif function == "greater":
                return a > b
            elif function == "less_equal":
                return a <= b
            elif function == "greater_equal":
                return a >= b
            # Logical
            elif function == "and_":
                return a and b
            elif function == "or_":
                return a or b

        return None


# =============================================================================
# FilterFusion Pass
# =============================================================================


class FilterFusion(OptimizationPass):
    """
    Combine consecutive Filter nodes into a single filter.

    Transforms:
        Filter(Filter(x, p1), p2) -> Filter(x, p1 AND p2)

    This simplifies the plan and makes subsequent optimizations easier.
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan."""
        if isinstance(plan, Filter):
            # First, transform the input
            new_input = self._transform(plan.input)

            # Check if input is also a Filter
            if isinstance(new_input, Filter):
                # Combine predicates with AND
                from pandas.lazy.expr import Expr

                combined_predicate = Expr(
                    Call(
                        "and_",
                        (new_input.predicate._ir, plan.predicate._ir),
                    )
                )
                return Filter(new_input.input, combined_predicate)

            # Input changed but isn't a Filter
            if new_input is not plan.input:
                return Filter(new_input, plan.predicate)
            return plan

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Project(new_input, plan.exprs)
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        elif isinstance(plan, DataFrameSource):
            return plan

        return plan


# =============================================================================
# PredicatePushdown Pass
# =============================================================================


class PredicatePushdown(OptimizationPass):
    """
    Push Filter nodes closer to data sources.

    This reduces the amount of data processed by subsequent operations.

    Conservative MVP implementation:
    - Only pushes through Project if ALL predicate columns are pass-through
    - Does not push through Aggregate (filter on aggregate result stays)
    - Pushes through Sort, Limit, Distinct (schema-preserving)
    - Pushes through Join to appropriate side when possible
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan, pushing filters down."""
        if isinstance(plan, Filter):
            # First transform the input
            new_input = self._transform(plan.input)

            # Try to push the filter down
            return self._push_filter(new_input, plan.predicate)

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Project(new_input, plan.exprs)
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        elif isinstance(plan, DataFrameSource):
            return plan

        return plan

    def _push_filter(self, input_plan: LogicalPlan, predicate: Expr) -> LogicalPlan:
        """
        Try to push a filter down through the input plan.

        Returns either:
        - A new plan with filter pushed down
        - Filter(input_plan, predicate) if cannot push
        """
        from pandas.lazy.expr import Expr

        pred_cols = get_referenced_columns(predicate)

        if isinstance(input_plan, Project):
            lineage = build_project_lineage(input_plan)
            can_push, col_mapping = can_push_predicate_through_project(
                pred_cols, lineage
            )

            if can_push:
                # Rewrite predicate with input column names
                new_pred_ir = substitute_columns(predicate._ir, col_mapping)
                new_pred = Expr(new_pred_ir)

                # Recursively try to push further
                new_input = self._push_filter(input_plan.input, new_pred)
                return Project(new_input, input_plan.exprs)

            # Cannot push through Project
            return Filter(input_plan, predicate)

        elif isinstance(input_plan, Filter):
            # Combine filters and try to push the combined filter
            combined_ir = Call("and_", (input_plan.predicate._ir, predicate._ir))
            combined_pred = Expr(combined_ir)
            return self._push_filter(input_plan.input, combined_pred)

        elif isinstance(input_plan, Sort):
            # Filter can pass through Sort
            new_input = self._push_filter(input_plan.input, predicate)
            return Sort(new_input, input_plan.by, input_plan.descending)

        elif isinstance(input_plan, Limit):
            # Cannot push filter below Limit (would change semantics)
            return Filter(input_plan, predicate)

        elif isinstance(input_plan, Distinct):
            # Filter can pass through Distinct
            new_input = self._push_filter(input_plan.input, predicate)
            return Distinct(new_input, input_plan.subset)

        elif isinstance(input_plan, Aggregate):
            # Cannot push filter below Aggregate (filter on aggregate result)
            return Filter(input_plan, predicate)

        elif isinstance(input_plan, Join):
            # Try to push to appropriate side
            join_mapping = build_join_column_mapping(input_plan)
            can_left, can_right, left_cols, right_cols = (
                can_push_predicate_through_join(pred_cols, join_mapping)
            )

            if can_left and not can_right:
                # Push to left side only
                # Build column mapping for predicate rewrite
                col_mapping = {
                    out: inp
                    for out, inp in join_mapping.left_columns.items()
                    if out in pred_cols or inp in pred_cols
                }
                new_pred_ir = substitute_columns(predicate._ir, col_mapping)
                new_pred = Expr(new_pred_ir)

                new_left = self._push_filter(input_plan.left, new_pred)
                return Join(
                    new_left,
                    input_plan.right,
                    input_plan.on,
                    input_plan.left_on,
                    input_plan.right_on,
                    input_plan.how,
                    input_plan.suffix,
                )

            elif can_right and not can_left:
                # Push to right side only
                col_mapping = {
                    out: inp
                    for out, inp in join_mapping.right_columns.items()
                    if out in pred_cols or inp in pred_cols
                }
                new_pred_ir = substitute_columns(predicate._ir, col_mapping)
                new_pred = Expr(new_pred_ir)

                new_right = self._push_filter(input_plan.right, new_pred)
                return Join(
                    input_plan.left,
                    new_right,
                    input_plan.on,
                    input_plan.left_on,
                    input_plan.right_on,
                    input_plan.how,
                    input_plan.suffix,
                )

            # Cannot push through Join
            return Filter(input_plan, predicate)

        elif isinstance(input_plan, DataFrameSource):
            # Cannot push below source
            return Filter(input_plan, predicate)

        # Unknown node type, don't push
        return Filter(input_plan, predicate)


# =============================================================================
# ProjectionPruning Pass
# =============================================================================


class ProjectionPruning(OptimizationPass):
    """
    Remove unnecessary columns from projections.

    Algorithm:
    1. Start at root with output columns as "needed"
    2. Walk down the tree, computing required columns at each level
    3. Modify Project nodes to only include required expressions
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        # Start with all output columns as needed
        output_cols = set(plan.resolve_schema().names)
        return self._prune(plan, output_cols)

    def _prune(self, plan: LogicalPlan, needed: set[str]) -> LogicalPlan:
        """Recursively prune the plan."""
        if isinstance(plan, DataFrameSource):
            # Can't prune source in logical plan
            return plan

        elif isinstance(plan, Project):
            # Filter expressions to only those producing needed columns
            new_exprs = []
            child_needed: set[str] = set()

            for expr in plan.exprs:
                output_name = extract_output_name(expr)
                if output_name in needed:
                    new_exprs.append(expr)
                    child_needed |= get_referenced_columns(expr)

            if not new_exprs:
                # Edge case: nothing needed? Keep at least one column
                new_exprs = [plan.exprs[0]]
                child_needed = get_referenced_columns(plan.exprs[0])

            new_input = self._prune(plan.input, child_needed)

            # Only create new Project if something changed
            if len(new_exprs) != len(plan.exprs) or new_input is not plan.input:
                return Project(new_input, tuple(new_exprs))
            return plan

        elif isinstance(plan, Filter):
            pred_cols = get_referenced_columns(plan.predicate)
            child_needed = needed | pred_cols
            new_input = self._prune(plan.input, child_needed)
            if new_input is not plan.input:
                return Filter(new_input, plan.predicate)
            return plan

        elif isinstance(plan, Aggregate):
            # Aggregates define their own column requirements
            child_needed: set[str] = set()
            for expr in plan.group_by:
                child_needed |= get_referenced_columns(expr)
            for expr in plan.agg_exprs:
                child_needed |= get_referenced_columns(expr)

            new_input = self._prune(plan.input, child_needed)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            sort_cols: set[str] = set()
            for expr in plan.by:
                sort_cols |= get_referenced_columns(expr)
            child_needed = needed | sort_cols

            new_input = self._prune(plan.input, child_needed)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._prune(plan.input, needed)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            if plan.subset:
                child_needed = needed | set(plan.subset)
            else:
                # Without subset, all columns matter for uniqueness
                child_needed = set(plan.input.resolve_schema().names)

            new_input = self._prune(plan.input, child_needed)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            left_needed, right_needed = self._compute_join_required(plan, needed)

            new_left = self._prune(plan.left, left_needed)
            new_right = self._prune(plan.right, right_needed)

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

        return plan

    def _compute_join_required(
        self, join: Join, needed_downstream: set[str]
    ) -> tuple[set[str], set[str]]:
        """Compute columns required from each side of a join."""
        left_schema = join.left.resolve_schema()
        right_schema = join.right.resolve_schema()

        left_required: set[str] = set()
        right_required: set[str] = set()

        # Join keys are always required
        if join.on:
            for col in join.on:
                left_required.add(col)
                right_required.add(col)
        elif join.left_on and join.right_on:
            left_required |= set(join.left_on)
            right_required |= set(join.right_on)

        # Map downstream columns back to source sides
        for col in needed_downstream:
            # Handle suffixed columns
            if col.endswith(join.suffix[0]):
                base = col[: -len(join.suffix[0])]
                if base in left_schema.names:
                    left_required.add(base)
                    continue
            if col.endswith(join.suffix[1]):
                base = col[: -len(join.suffix[1])]
                if base in right_schema.names:
                    right_required.add(base)
                    continue

            # Non-suffixed column
            if col in left_schema.names:
                left_required.add(col)
            if col in right_schema.names and (join.on is None or col not in join.on):
                right_required.add(col)

        return left_required, right_required


# =============================================================================
# LimitPushdown Pass
# =============================================================================


class LimitPushdown(OptimizationPass):
    """
    Push Limit nodes closer to data sources where safe.

    Can push through:
    - Project (schema change doesn't affect row count)

    Cannot push through:
    - Filter (would change which rows are selected)
    - Aggregate (would change aggregation results)
    - Sort (need all rows to sort before limiting)
    - Join (would change join results)
    - Distinct (need all rows to find unique ones)

    Special case: Sort followed by Limit could become top-k optimization
    (not implemented in this MVP).
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan."""
        if isinstance(plan, Limit):
            new_input = self._transform(plan.input)
            return self._push_limit(new_input, plan.n, plan.offset)

        elif isinstance(plan, Filter):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Filter(new_input, plan.predicate)
            return plan

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Project(new_input, plan.exprs)
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        elif isinstance(plan, DataFrameSource):
            return plan

        return plan

    def _push_limit(self, input_plan: LogicalPlan, n: int, offset: int) -> LogicalPlan:
        """Try to push a limit down through the input plan."""
        if isinstance(input_plan, Project):
            # Limit can pass through Project
            new_input = self._push_limit(input_plan.input, n, offset)
            return Project(new_input, input_plan.exprs)

        elif isinstance(input_plan, Limit):
            # Combine limits: take the more restrictive one
            # If outer has offset, it's complex - just use outer
            if offset > 0:
                return Limit(input_plan.input, n, offset)
            # No outer offset: combine
            combined_n = min(n, input_plan.n)
            combined_offset = input_plan.offset
            return Limit(input_plan.input, combined_n, combined_offset)

        # Cannot push through other nodes
        return Limit(input_plan, n, offset)


# =============================================================================
# SortLimitToTopK Pass
# =============================================================================


class SortLimitToTopK(OptimizationPass):
    """
    Combine Sort followed by Limit into a TopK operation.

    Transforms:
        Limit(Sort(x, by, desc), k) -> TopK(x, k, by, desc)

    TopK can be evaluated more efficiently than full sort + limit:
    - Uses heap-based selection: O(n log k) vs O(n log n)
    - Lower memory usage: only keep k elements in memory
    - Better cache behavior for large datasets

    This pass runs AFTER LimitPushdown so that limits have been
    pushed as close to sorts as possible.
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan, combining Sort+Limit into TopK."""
        if isinstance(plan, Limit):
            new_input = self._transform(plan.input)

            # Check if input is a Sort (the pattern we're looking for)
            if isinstance(new_input, Sort) and plan.offset == 0:
                # Transform Sort + Limit -> TopK
                return TopK(
                    new_input.input,
                    plan.n,
                    new_input.by,
                    new_input.descending,
                )

            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Filter):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Filter(new_input, plan.predicate)
            return plan

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Project(new_input, plan.exprs)
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        elif isinstance(plan, TopK):
            # Already a TopK, just recurse
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return TopK(new_input, plan.k, plan.by, plan.descending)
            return plan

        elif isinstance(plan, DataFrameSource):
            return plan

        return plan


# =============================================================================
# Common Subexpression Elimination (CSE) Pass
# =============================================================================


class CommonSubexpressionElimination(OptimizationPass):
    """
    Identify and eliminate common subexpressions.

    When the same expression appears multiple times in a Project node,
    we can compute it once and reuse the result. This is particularly
    useful for expensive computations like string operations.

    Transforms:
        Project([col("a").str.upper().alias("x"),
                 col("a").str.upper().alias("y")])
        ->
        Project([col("a").str.upper().alias("__cse_0"),
                 col("__cse_0").alias("x"),
                 col("__cse_0").alias("y")])

    Note: This is a simplified CSE that works within a single Project node.
    A more sophisticated version could track expressions across nodes.
    """

    def __init__(self) -> None:
        self._cse_counter = 0

    def _next_cse_name(self) -> str:
        """Generate a unique CSE temp column name."""
        name = f"__cse_{self._cse_counter}"
        self._cse_counter += 1
        return name

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        self._cse_counter = 0  # Reset counter for each optimization run
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan, eliminating common subexpressions."""
        if isinstance(plan, DataFrameSource):
            return plan

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)
            new_exprs = self._eliminate_cse_in_project(plan.exprs)
            if new_input is not plan.input or new_exprs != plan.exprs:
                return Project(new_input, new_exprs)
            return plan

        elif isinstance(plan, Filter):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Filter(new_input, plan.predicate)
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        elif isinstance(plan, TopK):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return TopK(new_input, plan.k, plan.by, plan.descending)
            return plan

        return plan

    def _eliminate_cse_in_project(self, exprs: tuple[Expr, ...]) -> tuple[Expr, ...]:
        """
        Find and eliminate common subexpressions within a Project.

        Returns the potentially modified tuple of expressions.
        """
        from pandas.lazy.expr import Expr

        # Build a map of IR node fingerprints to their occurrences
        # We use a string representation as a simple fingerprint
        ir_to_exprs: dict[str, list[tuple[int, Expr]]] = {}

        for i, expr in enumerate(exprs):
            # Get the core IR (unwrap alias)
            ir = expr._ir
            if isinstance(ir, Alias):
                core_ir = ir.arg
            else:
                core_ir = ir

            # Only consider non-trivial expressions (not simple FieldRefs or Literals)
            if isinstance(core_ir, (FieldRef, Literal)):
                continue

            fingerprint = self._fingerprint_ir(core_ir)
            if fingerprint not in ir_to_exprs:
                ir_to_exprs[fingerprint] = []
            ir_to_exprs[fingerprint].append((i, expr))

        # Find duplicates (fingerprints with multiple occurrences)
        duplicates = {
            fp: indices for fp, indices in ir_to_exprs.items() if len(indices) > 1
        }

        if not duplicates:
            return exprs

        # Build the new expression list
        new_exprs = list(exprs)
        cse_exprs: list[Expr] = []  # CSE definitions to prepend

        for fingerprint, occurrences in duplicates.items():
            # Create a CSE temp column
            cse_name = self._next_cse_name()

            # First occurrence: compute and store
            first_idx, first_expr = occurrences[0]
            first_ir = first_expr._ir
            if isinstance(first_ir, Alias):
                core_ir = first_ir.arg
            else:
                core_ir = first_ir

            # Create the CSE definition: expr.alias(__cse_N)
            cse_def = Expr(Alias(core_ir, cse_name))
            cse_exprs.append(cse_def)

            # Replace all occurrences with references to CSE column
            for idx, expr in occurrences:
                ir = expr._ir
                if isinstance(ir, Alias):
                    # Keep the original alias name, but reference CSE column
                    new_exprs[idx] = Expr(Alias(FieldRef(cse_name), ir.name))
                else:
                    # This shouldn't happen for our case, but handle it
                    new_exprs[idx] = Expr(FieldRef(cse_name))

        # Return CSE definitions followed by modified expressions
        # But we need to remove the CSE temp columns from final output
        # Actually, for this simple implementation, we'll keep them
        # A more sophisticated version would add a cleanup projection
        return tuple(cse_exprs) + tuple(new_exprs)

    def _fingerprint_ir(self, ir: IRNode) -> str:
        """
        Create a fingerprint string for an IR node.

        This is used to identify identical expressions.
        """
        if isinstance(ir, FieldRef):
            return f"FieldRef({ir.name})"
        elif isinstance(ir, Literal):
            return f"Literal({ir.value!r})"
        elif isinstance(ir, Alias):
            return f"Alias({self._fingerprint_ir(ir.arg)},{ir.name})"
        elif isinstance(ir, Call):
            args_fp = ",".join(self._fingerprint_ir(a) for a in ir.args)
            return f"Call({ir.function},{args_fp})"
        elif isinstance(ir, Cast):
            return f"Cast({self._fingerprint_ir(ir.arg)},{ir.target_dtype})"
        else:
            return repr(ir)


# =============================================================================
# Dead Code Elimination Pass
# =============================================================================


class DeadCodeElimination(OptimizationPass):
    """
    Remove unreachable or useless code from the plan.

    This pass identifies and removes:
    1. Filter nodes with constant True predicate (no filtering needed)
    2. Project nodes that don't change anything (identity projections)
    3. Limit nodes with n >= input row count (when statically known)

    Note: This is complementary to ProjectionPruning, which removes
    unused columns. DeadCodeElimination removes entire nodes.
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan, eliminating dead code."""
        if isinstance(plan, DataFrameSource):
            return plan

        elif isinstance(plan, Filter):
            new_input = self._transform(plan.input)

            # Check if predicate is constant True
            pred_ir = plan.predicate._ir
            if isinstance(pred_ir, Literal) and pred_ir.value is True:
                # Filter with True predicate is dead code
                return new_input

            # Check if predicate is constant False
            # Note: This removes all rows - could be intentional, but unusual
            # We leave this case alone for now

            if new_input is not plan.input:
                return Filter(new_input, plan.predicate)
            return plan

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)

            # Check if this is an identity projection
            if self._is_identity_projection(plan):
                return new_input

            if new_input is not plan.input:
                return Project(new_input, plan.exprs)
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)

            # Check if sorting by zero columns (no-op)
            if len(plan.by) == 0:
                return new_input

            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        elif isinstance(plan, TopK):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return TopK(new_input, plan.k, plan.by, plan.descending)
            return plan

        return plan

    def _is_identity_projection(self, project: Project) -> bool:
        """
        Check if a Project is an identity transformation.

        An identity projection:
        - Has the same columns as input (in same order)
        - Each column is a simple pass-through (FieldRef with same name)
        """
        input_schema = project.input.resolve_schema()
        input_names = input_schema.names

        # Must have same number of columns
        if len(project.exprs) != len(input_names):
            return False

        # Each expression must be a simple pass-through
        for expr, expected_name in zip(project.exprs, input_names, strict=False):
            ir = expr._ir
            # Handle both FieldRef and Alias(FieldRef, same_name)
            if isinstance(ir, FieldRef):
                if ir.name != expected_name:
                    return False
            elif isinstance(ir, Alias):
                if ir.name != expected_name:
                    return False
                if not isinstance(ir.arg, FieldRef):
                    return False
                if ir.arg.name != expected_name:
                    return False
            else:
                return False

        return True


# =============================================================================
# Aggregate Pushdown Pass
# =============================================================================


class AggregatePushdown(OptimizationPass):
    """
    Push aggregations closer to data sources where safe.

    This optimization:
    1. Pushes Aggregate through Project when the projection doesn't affect
       the columns needed for aggregation
    2. Combines Aggregate with Filter below it (pre-aggregation filtering)

    Benefits:
    - Reduces data volume before expensive grouping operations
    - Enables better column pruning before aggregation

    Transforms:
        Aggregate(Project(source, exprs), group_by, aggs)
        -> Project(Aggregate(source, group_by', aggs'), exprs')
        (when group_by and agg columns are pass-through in Project)

        Aggregate(Filter(source, pred), group_by, aggs)
        -> stays as is (filter before aggregate is already optimal)
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan, pushing aggregates down."""
        if isinstance(plan, DataFrameSource):
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)

            # Try to push aggregate through Project
            if isinstance(new_input, Project):
                pushed = self._push_aggregate_through_project(
                    new_input, plan.group_by, plan.agg_exprs
                )
                if pushed is not None:
                    return pushed

            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Project(new_input, plan.exprs)
            return plan

        elif isinstance(plan, Filter):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Filter(new_input, plan.predicate)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        elif isinstance(plan, TopK):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return TopK(new_input, plan.k, plan.by, plan.descending)
            return plan

        return plan

    def _push_aggregate_through_project(
        self,
        project: Project,
        group_by: tuple[Expr, ...],
        agg_exprs: tuple[Expr, ...],
    ) -> LogicalPlan | None:
        """
        Try to push an Aggregate through a Project.

        This is valid when:
        1. All group_by columns are pass-through or simple renames in the Project
        2. All columns referenced by agg_exprs are pass-through or simple renames

        Returns the pushed plan or None if not possible.
        """
        from pandas.lazy.expr import (
            Expr,
            extract_output_name,
        )

        lineage = build_project_lineage(project)

        # Collect all columns needed by group_by and agg_exprs
        needed_cols: set[str] = set()
        col_mapping: dict[str, str] = {}  # output_name -> input_name

        # Check group_by columns
        for expr in group_by:
            cols = get_referenced_columns(expr)
            for col_name in cols:
                source_name = get_source_column_name(lineage, col_name)
                if source_name is None:
                    # Column is computed, cannot push
                    return None
                col_mapping[col_name] = source_name
                needed_cols.add(source_name)

        # Check agg_exprs columns
        for expr in agg_exprs:
            cols = get_referenced_columns(expr)
            for col_name in cols:
                source_name = get_source_column_name(lineage, col_name)
                if source_name is None:
                    # Column is computed, cannot push
                    return None
                col_mapping[col_name] = source_name
                needed_cols.add(source_name)

        # All columns are pass-through or simple renames - we can push!
        # Rewrite group_by expressions with source column names
        new_group_by = tuple(
            Expr(substitute_columns(e._ir, col_mapping)) for e in group_by
        )

        # Rewrite agg_exprs with source column names
        new_agg_exprs = tuple(
            Expr(substitute_columns(e._ir, col_mapping)) for e in agg_exprs
        )

        # Create the pushed Aggregate
        pushed_agg = Aggregate(project.input, new_group_by, new_agg_exprs)

        # We need to preserve the original output column names
        # Build a Project on top to rename back if needed
        output_exprs: list[Expr] = []

        # Add group_by columns (may need renaming)
        for orig_expr in group_by:
            orig_name = extract_output_name(orig_expr)
            source_name = col_mapping.get(orig_name, orig_name)
            if orig_name != source_name:
                # Need to rename back
                output_exprs.append(Expr(Alias(FieldRef(source_name), orig_name)))
            else:
                output_exprs.append(Expr(FieldRef(orig_name)))

        # Add agg columns (keep original names from aliases)
        for orig_expr in agg_exprs:
            orig_name = extract_output_name(orig_expr)
            # Agg expressions keep their alias names
            output_exprs.append(Expr(FieldRef(orig_name)))

        # Check if we need the renaming Project
        needs_rename = any(
            col_mapping.get(extract_output_name(e), extract_output_name(e))
            != extract_output_name(e)
            for e in group_by
        )

        if needs_rename:
            return Project(pushed_agg, tuple(output_exprs))
        else:
            return pushed_agg


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


class EngineSelection(OptimizationPass):
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

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan, inserting Convert nodes."""
        from pandas.lazy.plan import Convert

        if isinstance(plan, DataFrameSource):
            return plan

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)
            return self._select_project_engine(
                Project(new_input, plan.exprs) if new_input is not plan.input else plan
            )

        elif isinstance(plan, Filter):
            new_input = self._transform(plan.input)
            return self._select_filter_engine(
                Filter(new_input, plan.predicate)
                if new_input is not plan.input
                else plan
            )

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        elif isinstance(plan, Convert):
            # Already has conversion
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Convert(new_input, plan.target_backend)
            return plan

        return plan

    def _select_project_engine(self, project: Project) -> LogicalPlan:
        """Determine backend for Project and insert conversion if needed."""
        from pandas.lazy.plan import Convert

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
        from pandas.lazy.plan import Convert

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
        from pandas.lazy.plan import Convert

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


class ConversionElimination(OptimizationPass):
    """
    Eliminate redundant or unnecessary Convert nodes.

    Patterns eliminated:
    1. Convert(Convert(x, "arrow"), "arrow") -> Convert(x, "arrow")
    2. Convert(x, backend) where x already produces that backend
    3. Back-to-back converts: Convert(Convert(x, A), B) -> Convert(x, B)
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        return self._transform(plan)

    def _transform(self, plan: LogicalPlan) -> LogicalPlan:
        """Recursively transform the plan, eliminating redundant converts."""
        from pandas.lazy.plan import Convert

        if isinstance(plan, DataFrameSource):
            return plan

        elif isinstance(plan, Convert):
            new_input = self._transform(plan.input)

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

        elif isinstance(plan, Project):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Project(new_input, plan.exprs)
            return plan

        elif isinstance(plan, Filter):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Filter(new_input, plan.predicate)
            return plan

        elif isinstance(plan, Aggregate):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Aggregate(new_input, plan.group_by, plan.agg_exprs)
            return plan

        elif isinstance(plan, Sort):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Sort(new_input, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Limit):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Limit(new_input, plan.n, plan.offset)
            return plan

        elif isinstance(plan, Distinct):
            new_input = self._transform(plan.input)
            if new_input is not plan.input:
                return Distinct(new_input, plan.subset)
            return plan

        elif isinstance(plan, Join):
            new_left = self._transform(plan.left)
            new_right = self._transform(plan.right)
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

        return plan

    def _get_plan_backend(self, plan: LogicalPlan) -> str:
        """Determine the output backend of a plan node."""
        from pandas.lazy.plan import Convert

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
