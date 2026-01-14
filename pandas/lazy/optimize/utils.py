"""
Utility functions for query optimization.

This module provides helper functions for:
- Column reference analysis
- Expression substitution
- Output name extraction
- Lineage tracking for predicate pushdown
- Join column mapping analysis
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from pandas.lazy.expr import Expr
    from pandas.lazy.plan import (
        Join,
        Project,
    )


# =============================================================================
# Column Reference Analysis
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
