"""
Optimization passes for lazy pandas.

This module contains all the optimization pass implementations:
- ConstantFolding: Evaluate constant expressions at compile time
- FilterFusion: Combine consecutive filters
- PredicatePushdown: Push filters toward data sources
- ProjectionPruning: Remove unused columns
- LimitPushdown: Push limits toward sources
- SortLimitToTopK: Combine Sort+Limit into TopK
- CommonSubexpressionElimination: Eliminate duplicate computations
- DeadCodeElimination: Remove identity projections
- AggregatePushdown: Push aggregations through projections

All passes use PlanVisitor base class to reduce boilerplate.
"""

from __future__ import annotations

import numpy as np

from pandas.lazy.expr import Expr
from pandas.lazy.ir import (
    Alias,
    Call,
    Cast,
    FieldRef,
    IRNode,
    Literal,
)
from pandas.lazy.optimize.base import PlanVisitor
from pandas.lazy.optimize.utils import (
    build_join_column_mapping,
    build_project_lineage,
    can_push_predicate_through_join,
    can_push_predicate_through_project,
    extract_output_name,
    get_referenced_columns,
    get_source_column_name,
    rewrite_predicate_through_project,
    substitute_columns,
)
from pandas.lazy.plan import (
    Aggregate,
    Concat,
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
from pandas.lazy.types import infer_expr_dtype

# =============================================================================
# ConstantFolding Pass
# =============================================================================


class ConstantFolding(PlanVisitor):
    """
    Evaluate constant expressions at optimization time.

    Transforms:
        lit(1) + lit(2) -> lit(3)
        lit("hello") + lit(" world") -> lit("hello world")
        lit(True) & lit(False) -> lit(False)

    This reduces runtime computation and enables further optimizations.
    """

    def visit_project(self, plan: Project) -> LogicalPlan:
        new_input = self.visit(plan.input)
        new_exprs = tuple(self._fold_expr(e) for e in plan.exprs)
        if new_input is not plan.input or new_exprs != plan.exprs:
            return Project(new_input, new_exprs)
        return plan

    def visit_filter(self, plan: Filter) -> LogicalPlan:
        new_input = self.visit(plan.input)
        new_pred = self._fold_expr(plan.predicate)
        if new_input is not plan.input or new_pred is not plan.predicate:
            return Filter(new_input, new_pred)
        return plan

    def visit_aggregate(self, plan: Aggregate) -> LogicalPlan:
        new_input = self.visit(plan.input)
        new_group_by = tuple(self._fold_expr(e) for e in plan.group_by)
        new_agg_exprs = tuple(self._fold_expr(e) for e in plan.agg_exprs)
        if (
            new_input is not plan.input
            or new_group_by != plan.group_by
            or new_agg_exprs != plan.agg_exprs
        ):
            return Aggregate(new_input, new_group_by, new_agg_exprs)
        return plan

    def visit_sort(self, plan: Sort) -> LogicalPlan:
        new_input = self.visit(plan.input)
        new_by = tuple(self._fold_expr(e) for e in plan.by)
        if new_input is not plan.input or new_by != plan.by:
            return Sort(new_input, new_by, plan.descending)
        return plan

    def visit_topk(self, plan: TopK) -> LogicalPlan:
        new_input = self.visit(plan.input)
        new_by = tuple(self._fold_expr(e) for e in plan.by)
        if new_input is not plan.input or new_by != plan.by:
            return TopK(new_input, plan.k, new_by, plan.descending)
        return plan

    def _fold_expr(self, expr: Expr) -> Expr:
        """Fold constants in an expression."""

        new_ir = self._fold_ir(expr._ir)
        if new_ir is not expr._ir:
            return Expr(new_ir)
        return expr

    def _fold_ir(self, node: IRNode) -> IRNode:
        """Recursively fold constants in an IR node."""
        if isinstance(node, (Literal, FieldRef)):
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

            # Check if all args are literals AND kwargs doesn't contain non-Literal
            # IR nodes (kwargs can contain plain Python values or IR nodes)
            kwargs_are_constant = self._kwargs_are_constant(node.kwargs)
            if (
                all(isinstance(arg, Literal) for arg in new_args)
                and kwargs_are_constant
            ):
                folded = self._fold_call(node.function, new_args, node.kwargs)
                if folded is not None:
                    return folded

            if new_args != node.args:
                return Call(node.function, new_args, node.kwargs, node.is_aggregate)
            return node

        return node

    def _cast_literal(self, value, target_dtype):
        """Cast a literal value to a target dtype."""

        if hasattr(target_dtype, "numpy_dtype"):
            target_dtype = target_dtype.numpy_dtype

        if target_dtype == np.dtype("int64"):
            return int(value)
        elif target_dtype == np.dtype("float64"):
            return float(value)
        elif target_dtype == np.dtype("bool"):
            return bool(value)

        return None

    def _kwargs_are_constant(self, kwargs: dict) -> bool:
        """
        Check if all kwargs values are constant (no non-Literal IR nodes).

        kwargs can contain:
        - Plain Python values (int, str, bool, etc.) - always constant
        - IR nodes - only constant if they are Literal nodes
        - Tuples/lists of the above

        Returns True if kwargs can be considered constant for folding purposes.
        """
        for value in kwargs.values():
            if isinstance(value, IRNode):
                if not isinstance(value, Literal):
                    return False
            elif isinstance(value, (tuple, list)):
                for item in value:
                    if isinstance(item, IRNode) and not isinstance(item, Literal):
                        return False
        return True

    def _fold_call(
        self, function: str, args: tuple[IRNode, ...], kwargs: dict | None = None
    ) -> IRNode | None:
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


class FilterFusion(PlanVisitor):
    """
    Combine consecutive Filter nodes into a single filter.

    Transforms:
        Filter(Filter(x, p1), p2) -> Filter(x, p1 AND p2)

    This simplifies the plan and makes subsequent optimizations easier.
    """

    def visit_filter(self, plan: Filter) -> LogicalPlan:
        # First, transform the input
        new_input = self.visit(plan.input)

        # Check if input is also a Filter
        if isinstance(new_input, Filter):
            # Combine predicates with AND

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


# =============================================================================
# PredicatePushdown Pass
# =============================================================================


class PredicatePushdown(PlanVisitor):
    """
    Push Filter nodes closer to data sources.

    This reduces the amount of data processed by subsequent operations.

    Conservative MVP implementation:
    - Only pushes through Project if ALL predicate columns are pass-through
    - Does not push through Aggregate (filter on aggregate result stays)
    - Pushes through Sort, Limit, Distinct (schema-preserving)
    - Pushes through Join to appropriate side when possible
    """

    def visit_filter(self, plan: Filter) -> LogicalPlan:
        # First transform the input
        new_input = self.visit(plan.input)

        # Try to push the filter down
        return self._push_filter(new_input, plan.predicate)

    @staticmethod
    def _flatten_ir(ir, function: str) -> list:
        """Flatten a binary and_/or_ tree into its operand list."""
        if isinstance(ir, Call) and ir.function == function:
            out = []
            for arg in ir.args:
                out.extend(PredicatePushdown._flatten_ir(arg, function))
            return out
        return [ir]

    def _derive_or_predicate(self, predicate: Expr, side_columns) -> Expr | None:
        """Side-only predicate implied by an OR of conjunctions, or None.

        For ``(a1 & b1) | (a2 & b2) | ...`` where the ``a`` parts reference only
        one join side, ``a1 | a2 | ...`` is implied (any surviving row
        satisfies some disjunct, hence that disjunct's side-only part).
        Returns None unless the predicate is a 2+-way OR and EVERY disjunct
        contributes at least one side-only conjunct (otherwise the derived
        predicate would not be implied). Column names are mapped back to the
        side's input names via ``side_columns`` (output -> input).
        """
        import functools

        disjuncts = self._flatten_ir(predicate._ir, "or_")
        if len(disjuncts) < 2:
            return None
        available = set(side_columns)
        derived_disjuncts = []
        for d in disjuncts:
            side_only = [
                c
                for c in self._flatten_ir(d, "and_")
                if get_referenced_columns(Expr(c)) <= available
            ]
            if not side_only:
                return None
            derived_disjuncts.append(
                functools.reduce(lambda x, y: Call("and_", (x, y)), side_only)
            )
        derived = functools.reduce(lambda x, y: Call("or_", (x, y)), derived_disjuncts)
        mapping = {out: inp for out, inp in side_columns.items() if out != inp}
        if mapping:
            derived = substitute_columns(derived, mapping)
        return Expr(derived)

    def _push_filter(self, input_plan: LogicalPlan, predicate: Expr) -> LogicalPlan:
        """
        Try to push a filter down through the input plan.

        Returns either:
        - A new plan with filter pushed down
        - Filter(input_plan, predicate) if cannot push
        """

        pred_cols = get_referenced_columns(predicate)

        if isinstance(input_plan, Project):
            lineage = build_project_lineage(input_plan)
            can_push, col_mapping = can_push_predicate_through_project(
                pred_cols, lineage
            )

            if can_push:
                # Simple case: all predicate columns are pass-through
                # Rewrite predicate with input column names
                new_pred_ir = substitute_columns(predicate._ir, col_mapping)
                new_pred = Expr(new_pred_ir)

                # Recursively try to push further
                new_input = self._push_filter(input_plan.input, new_pred)
                return Project(new_input, input_plan.exprs)

            # Advanced case: try expression rewriting for computed columns
            # This enables pushing filters like `filter(a_plus_b > 10)` through
            # `with_columns(a_plus_b = col("a") + col("b"))`
            rewritten_ir = rewrite_predicate_through_project(
                predicate._ir, lineage, max_complexity=20
            )
            if rewritten_ir is not None:
                new_pred = Expr(rewritten_ir)
                # Recursively try to push the rewritten predicate further
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
            # Try to push to appropriate side (join-type aware)
            join_mapping = build_join_column_mapping(input_plan)
            can_left, can_right, left_cols, right_cols = (
                can_push_predicate_through_join(
                    pred_cols, join_mapping, join_how=input_plan.how
                )
            )

            if can_left and not can_right:
                # Push to left side only
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

            # Cannot push the predicate itself. For an OR of conjunctions a
            # weaker side-only predicate is still implied: each disjunct must
            # hold for a surviving row, so OR(over disjuncts, AND(of that
            # disjunct's side-only conjuncts)) can be pushed below the join
            # while the original filter stays above (classic predicate
            # derivation — TPC-H q19's brand/container/size OR shrinks the
            # part side ~50x before the join). Inner joins only: derivation
            # changes which rows match, which outer joins observe via
            # null-extension.
            if input_plan.how == "inner":
                import functools

                conjuncts = self._flatten_ir(predicate._ir, "and_")
                left_avail = set(join_mapping.left_columns)
                right_avail = set(join_mapping.right_columns)
                new_left = input_plan.left
                new_right = input_plan.right
                remainder = []
                changed = False

                def _mapped(ir, side_columns):
                    mapping = {
                        out: inp for out, inp in side_columns.items() if out != inp
                    }
                    return Expr(substitute_columns(ir, mapping) if mapping else ir)

                for c in conjuncts:
                    cols = get_referenced_columns(Expr(c))
                    if cols <= left_avail:
                        # Side-only conjunct: push fully; for inner joins the
                        # surviving rows all satisfy it, so it can be dropped
                        # from the upper filter.
                        new_left = self._push_filter(
                            new_left, _mapped(c, join_mapping.left_columns)
                        )
                        changed = True
                        continue
                    if cols <= right_avail:
                        new_right = self._push_filter(
                            new_right, _mapped(c, join_mapping.right_columns)
                        )
                        changed = True
                        continue
                    # Mixed-side conjunct: stays above, but an OR of
                    # conjunctions still implies weaker side-only predicates
                    # that can be pushed in ADDITION (kept above too).
                    d_left = self._derive_or_predicate(
                        Expr(c), join_mapping.left_columns
                    )
                    if d_left is not None:
                        new_left = self._push_filter(new_left, d_left)
                        changed = True
                    d_right = self._derive_or_predicate(
                        Expr(c), join_mapping.right_columns
                    )
                    if d_right is not None:
                        new_right = self._push_filter(new_right, d_right)
                        changed = True
                    remainder.append(c)

                if changed:
                    new_join = Join(
                        new_left,
                        new_right,
                        input_plan.on,
                        input_plan.left_on,
                        input_plan.right_on,
                        input_plan.how,
                        input_plan.suffix,
                    )
                    if not remainder:
                        return new_join
                    rem_ir = functools.reduce(
                        lambda x, y: Call("and_", (x, y)), remainder
                    )
                    return Filter(new_join, Expr(rem_ir))

            # Cannot push through Join
            return Filter(input_plan, predicate)

        elif isinstance(input_plan, TopK):
            # Cannot push filter below TopK (would change which rows are selected)
            return Filter(input_plan, predicate)

        elif isinstance(input_plan, ParquetSource):
            # Push predicate into ParquetSource for pushdown to Parquet reader
            # Combine with any existing predicate
            if input_plan.predicate is not None:
                combined_ir = Call("and_", (input_plan.predicate._ir, predicate._ir))
                combined_pred = Expr(combined_ir)
            else:
                combined_pred = predicate

            return ParquetSource(
                path=input_plan.path,
                columns=input_plan.columns,
                predicate=combined_pred,
            )

        elif isinstance(input_plan, CSVSource):
            # Push predicate into CSVSource for filtering after read
            # Combine with any existing predicate
            if input_plan.predicate is not None:
                combined_ir = Call("and_", (input_plan.predicate._ir, predicate._ir))
                combined_pred = Expr(combined_ir)
            else:
                combined_pred = predicate

            return CSVSource(
                path=input_plan.path,
                columns=input_plan.columns,
                predicate=combined_pred,
                sep=input_plan.sep,
                header=input_plan.header,
                skip_rows=input_plan.skip_rows,
                n_rows=input_plan.n_rows,
            )

        elif isinstance(input_plan, Concat):
            # Push filter through Concat to all inputs
            # This enables predicate pushdown to each source independently
            new_inputs = tuple(
                self._push_filter(inp, predicate) for inp in input_plan.inputs
            )
            return Concat(new_inputs)

        # Cannot push below source or unknown node type
        return Filter(input_plan, predicate)


# =============================================================================
# ProjectionPruning Pass
# =============================================================================


class ProjectionPruning(PlanVisitor):
    """
    Remove unnecessary columns from projections.

    Algorithm:
    1. Start at root with output columns as "needed"
    2. Walk down the tree, computing required columns at each level
    3. Modify Project nodes to only include required expressions
    """

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        self._visit_memo = {}
        # Shared-subtree handling: pruning is context-dependent (required
        # columns flow top-down), so a naive recursion rebuilds a shared
        # subtree once per parent — destroying the DAG identity that
        # subplan caching (PhysicalCachedSubplan) depends on. Measured
        # consequence: q15's shared `rev` computed twice through different
        # paths with different float summation orders, so its
        # total_revenue == max equality returned EMPTY at SF-300. Shared
        # non-source nodes are pruned ONCE against their full output
        # schema (the safe union of every parent's needs) and memoized by
        # identity. Sources stay per-parent (a shared scan must still
        # narrow to each consumer's columns).
        self._shared_memo: dict[int, LogicalPlan] = {}
        self._parent_counts: dict[int, int] = {}
        seen: set[int] = set()

        def walk(n) -> None:
            nid = id(n)
            self._parent_counts[nid] = self._parent_counts.get(nid, 0) + 1
            if nid in seen:
                return
            seen.add(nid)
            for c in n.children():
                walk(c)

        walk(plan)
        # Start with all output columns as needed
        output_cols = set(plan.resolve_schema().names)
        return self._prune(plan, output_cols)

    def _prune(self, plan: LogicalPlan, needed: set[str]) -> LogicalPlan:
        nid = id(plan)
        if self._parent_counts.get(nid, 0) >= 2 and not isinstance(
            plan, (DataFrameSource, ParquetSource, CSVSource)
        ):
            cached = self._shared_memo.get(nid)
            if cached is None:
                cached = self._prune_impl(plan, set(plan.resolve_schema().names))
                self._shared_memo[nid] = cached
            return cached
        return self._prune_impl(plan, needed)

    def _prune_impl(self, plan: LogicalPlan, needed: set[str]) -> LogicalPlan:
        """Recursively prune the plan."""

        if isinstance(plan, DataFrameSource):
            # Can't prune source in logical plan
            return plan

        elif isinstance(plan, ParquetSource):
            # Push column selection into ParquetSource
            # Only select columns that are actually needed
            full_schema = plan.resolve_schema()
            available = set(full_schema.names)

            # Include columns needed by the predicate
            predicate_cols = (
                get_referenced_columns(plan.predicate)
                if plan.predicate is not None
                else set()
            )
            all_needed = needed | predicate_cols

            columns_to_read = tuple(sorted(all_needed & available))

            if columns_to_read and columns_to_read != plan.columns:
                return ParquetSource(
                    path=plan.path,
                    columns=columns_to_read if columns_to_read else None,
                    predicate=plan.predicate,
                )
            return plan

        elif isinstance(plan, CSVSource):
            # Push column selection into CSVSource
            # Only select columns that are actually needed
            full_schema = plan.resolve_schema()
            available = set(full_schema.names)

            # Include columns needed by the predicate
            predicate_cols = (
                get_referenced_columns(plan.predicate)
                if plan.predicate is not None
                else set()
            )
            all_needed = needed | predicate_cols

            columns_to_read = tuple(sorted(all_needed & available))

            if columns_to_read and columns_to_read != plan.columns:
                return CSVSource(
                    path=plan.path,
                    columns=columns_to_read if columns_to_read else None,
                    predicate=plan.predicate,
                    sep=plan.sep,
                    header=plan.header,
                    skip_rows=plan.skip_rows,
                    n_rows=plan.n_rows,
                )
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

        elif isinstance(plan, TopK):
            sort_cols: set[str] = set()
            for expr in plan.by:
                sort_cols |= get_referenced_columns(expr)
            child_needed = needed | sort_cols

            new_input = self._prune(plan.input, child_needed)
            if new_input is not plan.input:
                return TopK(new_input, plan.k, plan.by, plan.descending)
            return plan

        elif isinstance(plan, Concat):
            # Push projection requirements to all inputs
            new_inputs = tuple(self._prune(inp, needed) for inp in plan.inputs)
            if any(
                new != old for new, old in zip(new_inputs, plan.inputs, strict=True)
            ):
                return Concat(new_inputs)
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
            # IMPORTANT: Suffixes are only generated when a column exists on BOTH
            # sides. So if we need value_x, we must keep 'value' on both sides
            # to ensure the suffix is still generated after pruning.
            if col.endswith(join.suffix[0]):
                base = col[: -len(join.suffix[0])]
                if base in left_schema.names and base in right_schema.names:
                    # Need base on both sides to trigger suffix generation
                    left_required.add(base)
                    right_required.add(base)
                    continue
            if col.endswith(join.suffix[1]):
                base = col[: -len(join.suffix[1])]
                if base in right_schema.names and base in left_schema.names:
                    # Need base on both sides to trigger suffix generation
                    left_required.add(base)
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


class LimitPushdown(PlanVisitor):
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
    """

    def visit_limit(self, plan: Limit) -> LogicalPlan:
        new_input = self.visit(plan.input)
        return self._push_limit(new_input, plan.n, plan.offset)

    def _push_limit(self, input_plan: LogicalPlan, n: int, offset: int) -> LogicalPlan:
        """Try to push a limit down through the input plan."""
        if isinstance(input_plan, Project):
            # Limit can pass through Project
            new_input = self._push_limit(input_plan.input, n, offset)
            return Project(new_input, input_plan.exprs)

        elif isinstance(input_plan, Limit):
            # Combine limits carefully - only safe when both are simple head() ops
            # with no offset (offset=0 means head, offset=-1 means tail, offset>0
            # means skip)
            #
            # Safe to combine: head(50).head(10) -> head(10)
            # NOT safe to combine:
            #   - head(50).tail(10) - need rows 40-49, not rows 0-9
            #   - tail(50).head(10) - need first 10 of last 50
            #   - head(50).skip(5).head(10) - need rows 5-14
            if offset == 0 and input_plan.offset == 0:
                # Both are simple head() with no offset - can combine
                combined_n = min(n, input_plan.n)
                return Limit(input_plan.input, combined_n, 0)
            # Otherwise, keep both limits - cannot safely combine
            return Limit(input_plan, n, offset)

        elif isinstance(input_plan, ParquetSource) and offset == 0:
            # Push the limit into the scan: enables early read
            # termination and the direct small-limit ParquetFile path
            # (head(1000) over a multi-file glob reads ~6 ms of one file
            # instead of paying the Dataset scanner's ~50 ms readahead
            # startup). The Limit node is kept - the scan's limit is an
            # upper bound on rows read; the Limit enforces exactness.
            import dataclasses

            scan_limit = n if input_plan.limit is None else min(n, input_plan.limit)
            new_source = dataclasses.replace(input_plan, limit=scan_limit)
            return Limit(new_source, n, offset)

        # Cannot push through other nodes
        return Limit(input_plan, n, offset)


# =============================================================================
# SortLimitToTopK Pass
# =============================================================================


class SortLimitToTopK(PlanVisitor):
    """
    Combine Sort followed by Limit into a TopK operation.

    Transforms:
        Limit(Sort(x, by, desc), k) -> TopK(x, k, by, desc)

    TopK can be evaluated more efficiently than full sort + limit:
    - Uses heap-based selection: O(n log k) vs O(n log n)
    - Lower memory usage: only keep k elements in memory
    - Better cache behavior for large datasets
    """

    def visit_limit(self, plan: Limit) -> LogicalPlan:
        new_input = self.visit(plan.input)

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


# =============================================================================
# Common Subexpression Elimination (CSE) Pass
# =============================================================================

# Volatile functions that should never be CSE'd because they may return
# different values each time they're called, even with the same inputs.
VOLATILE_FUNCTIONS = frozenset(
    {
        "row_index",  # Returns position, changes per row
        "row_number",  # Window function, context-dependent
        "random",  # Non-deterministic
        "now",  # Time-dependent
        "today",  # Date-dependent
        "uuid",  # Unique per call
    }
)


def _is_volatile(ir: IRNode) -> bool:
    """
    Check if an IR node contains any volatile (non-deterministic) functions.

    Volatile functions should never be CSE'd because they may return different
    values each time they're called.
    """
    if isinstance(ir, Call):
        if ir.function in VOLATILE_FUNCTIONS:
            return True
        # Check arguments recursively
        for arg in ir.args:
            if _is_volatile(arg):
                return True
    elif isinstance(ir, Alias):
        return _is_volatile(ir.arg)
    elif isinstance(ir, Cast):
        return _is_volatile(ir.arg)
    return False


class CommonSubexpressionElimination(PlanVisitor):
    """
    Identify and eliminate common subexpressions.

    When the same expression appears multiple times in a Project node,
    we can compute it once and reuse the result.

    Implementation uses a two-stage projection pattern:
        Stage 1: Project(input, cse_defs + passthrough_cols_needed_for_final)
        Stage 2: Project(stage1, original_exprs_rewritten_to_use_cse_refs)

    This ensures that internal __cse_* columns never leak to user-visible output.

    Volatile functions (row_index, random, now, etc.) are excluded from CSE
    because they may return different values each time they're called.

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
        self._visit_memo = {}
        self._cse_counter = 0  # Reset counter for each optimization run
        return self.visit(plan)

    def visit_project(self, plan: Project) -> LogicalPlan:
        new_input = self.visit(plan.input)
        result = self._eliminate_cse_in_project(new_input, plan.exprs)
        if result is not None:
            return result
        if new_input is not plan.input:
            return Project(new_input, plan.exprs)
        return plan

    def _eliminate_cse_in_project(
        self, input_plan: LogicalPlan, exprs: tuple[Expr, ...]
    ) -> LogicalPlan | None:
        """
        Find and eliminate common subexpressions using two-stage projection.

        Returns a new plan if CSE was applied, or None if no CSE was found.

        Two-stage approach:
        1. Stage 1: Compute CSE values + passthrough columns needed by final exprs
        2. Stage 2: Project the final user-visible columns using CSE refs

        This ensures __cse_* columns are internal and never leak to output.
        """

        # Build a map of IR node fingerprints to their occurrences
        ir_to_exprs: dict[str, list[tuple[int, Expr, IRNode]]] = {}

        for i, expr in enumerate(exprs):
            ir = expr._ir
            if isinstance(ir, Alias):
                core_ir = ir.arg
            else:
                core_ir = ir

            # Only consider non-trivial expressions
            if isinstance(core_ir, (FieldRef, Literal)):
                continue

            # Skip volatile functions - they must not be CSE'd
            if _is_volatile(core_ir):
                continue

            fingerprint = self._fingerprint_ir(core_ir)
            if fingerprint not in ir_to_exprs:
                ir_to_exprs[fingerprint] = []
            ir_to_exprs[fingerprint].append((i, expr, core_ir))

        # Find duplicates (expressions that appear more than once)
        duplicates = {
            fp: occurrences
            for fp, occurrences in ir_to_exprs.items()
            if len(occurrences) > 1
        }

        if not duplicates:
            return None

        # Build CSE definitions and track which columns from input we need
        cse_defs: list[Expr] = []  # __cse_N = expr definitions
        cse_mapping: dict[str, str] = {}  # fingerprint -> __cse_N name
        input_cols_needed: set[str] = set()

        for fingerprint, occurrences in duplicates.items():
            cse_name = self._next_cse_name()
            cse_mapping[fingerprint] = cse_name

            # Use the core IR from the first occurrence
            _, _, core_ir = occurrences[0]
            cse_def = Expr(Alias(core_ir, cse_name))
            cse_defs.append(cse_def)

            # Track input columns needed by this CSE expression
            input_cols_needed |= get_referenced_columns(cse_def)

        # Build the final (Stage 2) expressions by rewriting CSE occurrences
        final_exprs: list[Expr] = []
        for i, expr in enumerate(exprs):
            ir = expr._ir
            if isinstance(ir, Alias):
                core_ir = ir.arg
                output_name = ir.name
            else:
                core_ir = ir
                output_name = None

            # Check if this expression matches a CSE'd subexpression
            if not isinstance(core_ir, (FieldRef, Literal)) and not _is_volatile(
                core_ir
            ):
                fingerprint = self._fingerprint_ir(core_ir)
                if fingerprint in cse_mapping:
                    cse_name = cse_mapping[fingerprint]
                    if output_name is not None:
                        # Preserve the original alias: __cse_N -> original_name
                        final_exprs.append(Expr(Alias(FieldRef(cse_name), output_name)))
                    else:
                        # No alias, just reference the CSE column
                        final_exprs.append(Expr(FieldRef(cse_name)))
                    continue

            # Not a CSE'd expression - keep original and track its input deps
            final_exprs.append(expr)
            input_cols_needed |= get_referenced_columns(expr)

        # Stage 1: Compute CSE values + passthrough columns needed by final
        # We need to pass through input columns that final_exprs reference
        stage1_exprs: list[Expr] = list(cse_defs)

        # Add passthrough columns for non-CSE expressions in final
        input_schema = input_plan.resolve_schema()
        stage1_exprs.extend(
            Expr(FieldRef(col_name))
            for col_name in sorted(input_cols_needed)
            if col_name in input_schema
        )

        stage1 = Project(input_plan, tuple(stage1_exprs))

        # Stage 2: Project final user-visible columns
        stage2 = Project(stage1, tuple(final_exprs))

        return stage2

    def _fingerprint_ir(self, ir: IRNode) -> str:
        """Create a fingerprint string for an IR node."""
        if isinstance(ir, FieldRef):
            return f"FieldRef({ir.name})"
        elif isinstance(ir, Literal):
            return f"Literal({ir.value!r})"
        elif isinstance(ir, Alias):
            return f"Alias({self._fingerprint_ir(ir.arg)},{ir.name})"
        elif isinstance(ir, Call):
            args_fp = ",".join(self._fingerprint_ir(a) for a in ir.args)
            # kwargs must be part of the fingerprint: case_when carries its
            # branches (cases/otherwise) entirely in kwargs with empty args, so
            # ignoring kwargs makes two different case_when expressions collide
            # and CSE wrongly merges them. is_aggregate likewise distinguishes
            # e.g. sum(x) from x.
            kwargs_fp = ",".join(
                f"{k}={self._fingerprint_value(ir.kwargs[k])}"
                for k in sorted(ir.kwargs)
            )
            return (
                f"Call({ir.function},[{args_fp}],{{{kwargs_fp}}},agg={ir.is_aggregate})"
            )
        elif isinstance(ir, Cast):
            return f"Cast({self._fingerprint_ir(ir.arg)},{ir.target_dtype})"
        else:
            return repr(ir)

    def _fingerprint_value(self, value) -> str:
        """Fingerprint a kwargs value, which may be an IR node, a (nested)
        tuple/list of values, or a plain Python value."""
        if isinstance(value, IRNode):
            return self._fingerprint_ir(value)
        if isinstance(value, (tuple, list)):
            return "(" + ",".join(self._fingerprint_value(v) for v in value) + ")"
        return repr(value)


# =============================================================================
# Dead Code Elimination Pass
# =============================================================================


class DeadCodeElimination(PlanVisitor):
    """
    Remove unreachable or useless code from the plan.

    This pass identifies and removes:
    1. Filter nodes with constant True predicate (no filtering needed)
    2. Project nodes that don't change anything (identity projections)
    3. Sort nodes with zero columns (no-op)
    """

    def visit_filter(self, plan: Filter) -> LogicalPlan:
        new_input = self.visit(plan.input)

        # Check if predicate is constant True
        pred_ir = plan.predicate._ir
        if isinstance(pred_ir, Literal) and pred_ir.value is True:
            # Filter with True predicate is dead code
            return new_input

        if new_input is not plan.input:
            return Filter(new_input, plan.predicate)
        return plan

    def visit_project(self, plan: Project) -> LogicalPlan:
        new_input = self.visit(plan.input)

        # Check if this is an identity projection
        if self._is_identity_projection(plan):
            return new_input

        if new_input is not plan.input:
            return Project(new_input, plan.exprs)
        return plan

    def visit_sort(self, plan: Sort) -> LogicalPlan:
        new_input = self.visit(plan.input)

        # Check if sorting by zero columns (no-op)
        if len(plan.by) == 0:
            return new_input

        if new_input is not plan.input:
            return Sort(new_input, plan.by, plan.descending)
        return plan

    def _is_identity_projection(self, project: Project) -> bool:
        """Check if a Project is an identity transformation."""
        input_schema = project.input.resolve_schema()
        input_names = input_schema.names

        if len(project.exprs) != len(input_names):
            return False

        for expr, expected_name in zip(project.exprs, input_names, strict=False):
            ir = expr._ir
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


class AggregatePushdown(PlanVisitor):
    """
    Push aggregations closer to data sources where safe.

    Transforms:
        Aggregate(Project(source, exprs), group_by, aggs)
        -> Project(Aggregate(source, group_by', aggs'), exprs')
        (when group_by and agg columns are pass-through in Project)
    """

    def visit_aggregate(self, plan: Aggregate) -> LogicalPlan:
        new_input = self.visit(plan.input)

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

    def _push_aggregate_through_project(
        self,
        project: Project,
        group_by: tuple[Expr, ...],
        agg_exprs: tuple[Expr, ...],
    ) -> LogicalPlan | None:
        """Try to push an Aggregate through a Project."""

        lineage = build_project_lineage(project)

        # Collect all columns needed by group_by and agg_exprs
        needed_cols: set[str] = set()
        col_mapping: dict[str, str] = {}

        # Check group_by columns
        for expr in group_by:
            cols = get_referenced_columns(expr)
            for col_name in cols:
                source_name = get_source_column_name(lineage, col_name)
                if source_name is None:
                    return None
                col_mapping[col_name] = source_name
                needed_cols.add(source_name)

        # Check agg_exprs columns
        for expr in agg_exprs:
            cols = get_referenced_columns(expr)
            for col_name in cols:
                source_name = get_source_column_name(lineage, col_name)
                if source_name is None:
                    return None
                col_mapping[col_name] = source_name
                needed_cols.add(source_name)

        # All columns are pass-through or simple renames - we can push!
        new_group_by = tuple(
            Expr(substitute_columns(e._ir, col_mapping)) for e in group_by
        )
        new_agg_exprs = tuple(
            Expr(substitute_columns(e._ir, col_mapping)) for e in agg_exprs
        )

        pushed_agg = Aggregate(project.input, new_group_by, new_agg_exprs)

        # Build output Project for renaming
        output_exprs: list[Expr] = []

        for orig_expr in group_by:
            orig_name = extract_output_name(orig_expr)
            source_name = col_mapping.get(orig_name, orig_name)
            if orig_name != source_name:
                output_exprs.append(Expr(Alias(FieldRef(source_name), orig_name)))
            else:
                output_exprs.append(Expr(FieldRef(orig_name)))

        for orig_expr in agg_exprs:
            orig_name = extract_output_name(orig_expr)
            output_exprs.append(Expr(FieldRef(orig_name)))

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
# ExpressionSimplification Pass
# =============================================================================


class ExpressionSimplification(PlanVisitor):
    """
    Simplify algebraic expressions to reduce runtime computation.

    Identity-with-Literal Transformations (always safe):
        x * 1 -> x           x * 0 -> 0
        x / 1 -> x           x + 0 -> x
        x - 0 -> x           x ** 1 -> x
        x ** 0 -> 1

    Logical Transformations (always safe with boolean literals):
        x & True -> x        x & False -> False
        x | False -> x       x | True -> True
        !!x -> x             --x -> x (double negation)

    De Morgan's Laws:
        !(a & b) -> !a | !b
        !(a | b) -> !a & !b

    Self-referential Transformations (only for non-nullable integer types):
        x - x -> 0           (safe for non-nullable integers, no NaN)
        x == x -> True       (safe for non-nullable integers)
        x != x -> False      (safe for non-nullable integers)
        x < x -> False       (safe for non-nullable integers)
        x > x -> False       (safe for non-nullable integers)
        x <= x -> True       (safe for non-nullable integers)
        x >= x -> True       (safe for non-nullable integers)
        x & x -> x           (safe for non-nullable booleans)
        x | x -> x           (safe for non-nullable booleans)

    Note: Self-referential patterns are NOT safe for:
        - Floating-point types (NaN behavior)
        - Nullable types (pd.NA behavior)
        - x / x (even integers: 0/0 is undefined)

    This pass complements ConstantFolding, which only handles
    expressions where ALL operands are constants.
    """

    def __init__(self) -> None:
        self._current_schema = None

    def visit_project(self, plan: Project) -> LogicalPlan:
        new_input = self.visit(plan.input)
        # Set schema context for expression simplification
        self._current_schema = new_input.resolve_schema()
        new_exprs = tuple(self._simplify_expr(e) for e in plan.exprs)
        if new_input is not plan.input or new_exprs != plan.exprs:
            return Project(new_input, new_exprs)
        return plan

    def visit_filter(self, plan: Filter) -> LogicalPlan:
        new_input = self.visit(plan.input)
        # Set schema context for expression simplification
        self._current_schema = new_input.resolve_schema()
        new_pred = self._simplify_expr(plan.predicate)
        if new_input is not plan.input or new_pred is not plan.predicate:
            return Filter(new_input, new_pred)
        return plan

    def visit_aggregate(self, plan: Aggregate) -> LogicalPlan:
        new_input = self.visit(plan.input)
        # Set schema context for expression simplification
        self._current_schema = new_input.resolve_schema()
        new_group_by = tuple(self._simplify_expr(e) for e in plan.group_by)
        new_agg_exprs = tuple(self._simplify_expr(e) for e in plan.agg_exprs)
        if (
            new_input is not plan.input
            or new_group_by != plan.group_by
            or new_agg_exprs != plan.agg_exprs
        ):
            return Aggregate(new_input, new_group_by, new_agg_exprs)
        return plan

    def visit_sort(self, plan: Sort) -> LogicalPlan:
        new_input = self.visit(plan.input)
        # Set schema context for expression simplification
        self._current_schema = new_input.resolve_schema()
        new_by = tuple(self._simplify_expr(e) for e in plan.by)
        if new_input is not plan.input or new_by != plan.by:
            return Sort(new_input, new_by, plan.descending)
        return plan

    def visit_topk(self, plan: TopK) -> LogicalPlan:
        new_input = self.visit(plan.input)
        # Set schema context for expression simplification
        self._current_schema = new_input.resolve_schema()
        new_by = tuple(self._simplify_expr(e) for e in plan.by)
        if new_input is not plan.input or new_by != plan.by:
            return TopK(new_input, plan.k, new_by, plan.descending)
        return plan

    def _simplify_expr(self, expr: Expr) -> Expr:
        """Simplify an expression."""

        new_ir = self._simplify_ir(expr._ir)
        if new_ir is not expr._ir:
            return Expr(new_ir)
        return expr

    def _simplify_ir(self, node: IRNode) -> IRNode:
        """Recursively simplify an IR node."""
        if isinstance(node, (Literal, FieldRef)):
            return node

        elif isinstance(node, Alias):
            new_arg = self._simplify_ir(node.arg)
            if new_arg is not node.arg:
                return Alias(new_arg, node.name)
            return node

        elif isinstance(node, Cast):
            new_arg = self._simplify_ir(node.arg)
            if new_arg is not node.arg:
                return Cast(new_arg, node.target_dtype)
            return node

        elif isinstance(node, Call):
            # First, simplify children
            new_args = tuple(self._simplify_ir(arg) for arg in node.args)

            # Try algebraic simplification
            simplified = self._simplify_call(node.function, new_args, node.kwargs)
            if simplified is not None:
                return simplified

            if new_args != node.args:
                return Call(node.function, new_args, node.kwargs, node.is_aggregate)
            return node

        return node

    def _simplify_call(
        self, function: str, args: tuple[IRNode, ...], kwargs: dict
    ) -> IRNode | None:
        """Try to simplify a function call algebraically."""
        if len(args) != 2:
            # Also handle unary simplifications
            if len(args) == 1:
                return self._simplify_unary(function, args[0])
            return None

        left, right = args

        # Check for self-referential patterns (x op x)
        # Only safe for non-nullable, non-floating-point types
        if self._ir_equal(left, right):
            result = self._simplify_self_op(function, left)
            if result is not None:
                return result

        # Check for identity patterns with literals
        left_is_lit = isinstance(left, Literal)
        right_is_lit = isinstance(right, Literal)

        if not (left_is_lit or right_is_lit):
            return None

        # Multiplication simplifications
        if function == "multiply":
            if right_is_lit:
                if right.value == 1:
                    return left  # x * 1 -> x
                if right.value == 0:
                    return Literal(0)  # x * 0 -> 0
            if left_is_lit:
                if left.value == 1:
                    return right  # 1 * x -> x
                if left.value == 0:
                    return Literal(0)  # 0 * x -> 0

        # Division simplifications
        elif function == "divide":
            if right_is_lit and right.value == 1:
                return left  # x / 1 -> x

        # Addition simplifications
        elif function == "add":
            if right_is_lit and right.value == 0:
                return left  # x + 0 -> x
            if left_is_lit and left.value == 0:
                return right  # 0 + x -> x

        # Subtraction simplifications
        elif function == "subtract":
            if right_is_lit and right.value == 0:
                return left  # x - 0 -> x

        # Power simplifications
        elif function == "power":
            if right_is_lit:
                if right.value == 1:
                    return left  # x ** 1 -> x
                if right.value == 0:
                    return Literal(1)  # x ** 0 -> 1

        # Logical AND simplifications
        elif function == "and_":
            if right_is_lit:
                if right.value is True:
                    return left  # x & True -> x
                if right.value is False:
                    return Literal(False)  # x & False -> False
            if left_is_lit:
                if left.value is True:
                    return right  # True & x -> x
                if left.value is False:
                    return Literal(False)  # False & x -> False

        # Logical OR simplifications
        elif function == "or_":
            if right_is_lit:
                if right.value is False:
                    return left  # x | False -> x
                if right.value is True:
                    return Literal(True)  # x | True -> True
            if left_is_lit:
                if left.value is False:
                    return right  # False | x -> x
                if left.value is True:
                    return Literal(True)  # True | x -> True

        return None

    def _simplify_unary(self, function: str, arg: IRNode) -> IRNode | None:
        """Simplify unary operations."""
        # Double negation elimination: !!x -> x
        if function == "invert":
            if (
                isinstance(arg, Call)
                and arg.function == "invert"
                and len(arg.args) == 1
            ):
                return arg.args[0]

            # De Morgan's Laws: !(a & b) -> !a | !b, !(a | b) -> !a & !b
            if isinstance(arg, Call) and len(arg.args) == 2:
                if arg.function == "and_":
                    # !(a & b) -> !a | !b
                    left_inverted = Call("invert", (arg.args[0],))
                    right_inverted = Call("invert", (arg.args[1],))
                    return Call("or_", (left_inverted, right_inverted))
                elif arg.function == "or_":
                    # !(a | b) -> !a & !b
                    left_inverted = Call("invert", (arg.args[0],))
                    right_inverted = Call("invert", (arg.args[1],))
                    return Call("and_", (left_inverted, right_inverted))

        # Double negate elimination: --x -> x
        elif function == "negate":
            if (
                isinstance(arg, Call)
                and arg.function == "negate"
                and len(arg.args) == 1
            ):
                return arg.args[0]

        return None

    def _simplify_self_op(self, function: str, operand: IRNode) -> IRNode | None:
        """
        Simplify operations where both operands are the same (x op x).

        SAFETY: Only applies to non-nullable, non-floating-point types.
        This is critical because:
            - NaN - NaN = NaN (not 0)
            - NaN / NaN = NaN (not 1)
            - NaN == NaN = False (not True)
            - pd.NA has similar 3-valued logic issues

        Patterns (when safe):
            x - x -> 0           (NOT x / x, since 0/0 is undefined)
            x == x -> True
            x != x -> False
            x < x -> False
            x > x -> False
            x <= x -> True
            x >= x -> True
            x & x -> x           (boolean idempotence)
            x | x -> x           (boolean idempotence)
        """
        # Check if the operand type is safe for self-referential optimization
        if not self._is_safe_for_self_op(operand, function):
            return None

        # Arithmetic self-cancellation (NOT division - 0/0 is undefined)
        if function == "subtract":
            return Literal(0)  # x - x -> 0

        # Comparison tautologies/contradictions
        elif function == "equal":
            return Literal(True)  # x == x -> True

        elif function == "not_equal":
            return Literal(False)  # x != x -> False

        elif function == "less":
            return Literal(False)  # x < x -> False

        elif function == "greater":
            return Literal(False)  # x > x -> False

        elif function == "less_equal":
            return Literal(True)  # x <= x -> True

        elif function == "greater_equal":
            return Literal(True)  # x >= x -> True

        # Logical idempotence (for non-nullable booleans)
        elif function == "and_":
            return operand  # x & x -> x

        elif function == "or_":
            return operand  # x | x -> x

        return None

    def _is_safe_for_self_op(self, node: IRNode, function: str) -> bool:
        """
        Check if a node's dtype is safe for self-referential optimization.

        Safe types are:
        - Non-nullable integer types (no NaN, no pd.NA)
        - Non-nullable boolean types (for logical operations)

        Unsafe types are:
        - Floating-point types (can have NaN)
        - Nullable types (can have pd.NA)
        - String types (comparison semantics may vary)
        - Object types (unknown behavior)
        """

        # Need schema context to determine dtype
        if self._current_schema is None:
            return False

        try:
            dtype = infer_expr_dtype(node, self._current_schema)
        except (KeyError, TypeError):
            # Cannot determine dtype - not safe
            return False

        # Reject nullable types
        if dtype.nullable:
            return False

        # For logical operations, require boolean type
        if function in {"and_", "or_"}:
            return dtype.is_boolean()

        # For arithmetic/comparison operations, require non-float numeric
        if dtype.is_numeric() and dtype.numpy_dtype is not None:
            # Only allow integer types (no NaN possible)
            return np.issubdtype(dtype.numpy_dtype, np.integer)

        return False

    def _ir_equal(self, a: IRNode, b: IRNode) -> bool:
        """
        Check if two IR nodes are structurally equal.

        This is a conservative equality check that returns True only when
        the nodes are guaranteed to produce the same result.
        """
        if type(a) is not type(b):
            return False

        if isinstance(a, Literal):
            return a.value == b.value

        elif isinstance(a, FieldRef):
            return a.name == b.name

        elif isinstance(a, Alias):
            return a.name == b.name and self._ir_equal(a.arg, b.arg)

        elif isinstance(a, Cast):
            return a.target_dtype == b.target_dtype and self._ir_equal(a.arg, b.arg)

        elif isinstance(a, Call):
            if a.function != b.function or len(a.args) != len(b.args):
                return False
            return all(
                self._ir_equal(aa, bb) for aa, bb in zip(a.args, b.args, strict=True)
            )

        return False
