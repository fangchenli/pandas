"""Physical plan — planner operators (split from physical.py)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np

from pandas.lazy.backends.convert import arrays_to_dataframe
from pandas.lazy.expr import (
    Expr,
    extract_output_name,
)
from pandas.lazy.ir import (
    Alias,
    Call,
    FieldRef,
    Literal as IRLiteral,
)
from pandas.lazy.physical.base import (
    ExecutionContext,
    PhysicalMaterialize,
    PhysicalPlan,
)
from pandas.lazy.physical.fused import (
    FUSE_JOIN_AGG,
    FusedAggSpec,
    FusedOperation,
    PhysicalFusedFilterAgg,
    PhysicalFusedJoinAgg,
    PhysicalFusedPipeline,
)
from pandas.lazy.physical.groupby import PhysicalHashAggregate
from pandas.lazy.physical.join import (
    PhysicalHashJoin,
    PhysicalJoinChain,
)
from pandas.lazy.physical.project_filter import (
    PhysicalFilter,
    PhysicalProject,
)
from pandas.lazy.physical.reshape import (
    PhysicalCachedSubplan,
    PhysicalConcat,
    PhysicalConvert,
    PhysicalResetIndex,
    PhysicalSetIndex,
)
from pandas.lazy.physical.scans import (
    PhysicalCSVScan,
    PhysicalParquetScan,
    PhysicalScan,
)
from pandas.lazy.physical.sort_limit import (
    PhysicalDistinct,
    PhysicalGroupByHead,
    PhysicalLimit,
    PhysicalSort,
    PhysicalTopK,
)
from pandas.lazy.plan import (
    Aggregate,
    Concat,
    Convert,
    CSVSource,
    DataFrameSource,
    Distinct,
    Filter,
    GroupByHead,
    Join,
    Limit,
    LogicalPlan,
    ParquetSource,
    Project,
    ResetIndex,
    SetIndex,
    Sort,
    TopK,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from pandas.lazy.backends.spill import SpillConfig

# Execute-once caching of shared subplans (see PhysicalPlanner.plan). The
# q15 regression that kept this off is diagnosed and fixed: the childless
# wrapper hid its inner subtree from the fusion/chain post-passes.
_SUBPLAN_CACHE_ENABLED = True


def _unwrap_materialize(plan):
    """Look through the PhysicalMaterialize bookkeeping wrapper."""
    if isinstance(plan, PhysicalMaterialize):
        return plan.input
    return plan


# =============================================================================
# Physical Planner
# =============================================================================


class PhysicalPlanner:
    """
    Converts optimized logical plans to physical plans.

    The planner makes execution decisions based on:
    - Data characteristics
    - Operation requirements
    - User preferences

    Pipeline Boundaries
    -------------------
    The planner inserts explicit PhysicalMaterialize nodes before pipeline
    breaker operators (sort, aggregate, distinct, join build side). This
    makes materialization points visible in the physical plan for:

    - Clear debugging and explain output
    - Centralized spill management
    - Correct fusion boundaries
    - Backend conversion points
    """

    def __init__(
        self,
        preferred_backend: Literal["auto", "arrow", "numpy"] = "auto",
    ) -> None:
        self.preferred_backend = preferred_backend

    def _materialize_for_breaker(
        self,
        logical_input: LogicalPlan,
        reason: str,
    ) -> PhysicalPlan:
        """
        Plan an input and wrap in Materialize node for a pipeline breaker.

        This ensures inputs to pipeline breakers (sort, aggregate, distinct,
        join) are explicitly materialized, making the boundary visible in
        the plan.

        Parameters
        ----------
        logical_input : LogicalPlan
            The logical input to plan and materialize.
        reason : str
            Why materialization is needed (for debugging/explain).

        Returns
        -------
        PhysicalPlan
            The input wrapped in PhysicalMaterialize.
        """
        physical_input = self._plan_recursive(logical_input)
        return PhysicalMaterialize(input=physical_input, reason=reason)

    def plan(
        self, logical_plan: LogicalPlan, *, enable_fusion: bool = True
    ) -> PhysicalPlan:
        """
        Convert a logical plan to a physical plan.

        Parameters
        ----------
        logical_plan : LogicalPlan
            The optimized logical plan.
        enable_fusion : bool, default True
            If True, apply operator fusion as a post-processing step.
            Fusion combines chains of Filter/Project/Limit into single
            fused operators for better performance.

        Returns
        -------
        PhysicalPlan
            The physical execution plan.
        """
        # Common-subplan detection: LazyFrame reuse makes the logical plan a
        # DAG (preserved through optimization by PlanVisitor's visit memo);
        # shared non-source subtrees can be planned once and wrapped in an
        # execute-once PhysicalCachedSubplan. DEFAULT-OFF pending diagnosis:
        # the wrapper wins modestly where expected (q21 0.96x) but makes
        # q15's *surrounding* graph ~3.5x slower through a mechanism not yet
        # understood (probed and ruled out: the aggregate itself, the order
        # contract, morsel-parallelism loss — inner runs through the full
        # pipeline engine). See ROADMAP "common-subplan caching".
        self._shared_ids = (
            self._find_shared_subplans(logical_plan)
            if _SUBPLAN_CACHE_ENABLED
            else set()
        )
        self._shared_wrappers: dict[int, PhysicalCachedSubplan] = {}

        physical_plan = self._plan_recursive(logical_plan)

        if enable_fusion:
            physical_plan = self._apply_fusion(physical_plan)

        physical_plan = self._collapse_join_chains(physical_plan)
        physical_plan = self._fuse_filter_aggregates(physical_plan)
        physical_plan = self._fuse_join_aggregates(physical_plan)

        # PhysicalCachedSubplan is childless (the pipeline compiler must
        # treat it as a source), so the tree-walking post-passes above never
        # see its inner subtree — without this, a shared subtree loses
        # operator fusion and chain collapse (measured: q15's shared filter
        # ran unfused at 663 ms vs 44 ms fused). Apply them to each wrapper's
        # inner explicitly; nested shared wrappers are in the dict too.
        for wrapper in getattr(self, "_shared_wrappers", {}).values():
            inner = wrapper.inner
            if enable_fusion:
                inner = self._apply_fusion(inner)
            inner = self._collapse_join_chains(inner)
            inner = self._fuse_filter_aggregates(inner)
            wrapper.inner = self._fuse_join_aggregates(inner)

        return physical_plan

    @staticmethod
    def _find_shared_subplans(plan: LogicalPlan) -> set[int]:
        """ids of non-source logical nodes referenced by 2+ parents."""
        from pandas.lazy.plan import (
            CSVSource,
            DataFrameSource,
            ParquetSource,
        )

        counts: dict[int, int] = {}

        def walk(node) -> None:
            nid = id(node)
            counts[nid] = counts.get(nid, 0) + 1
            if counts[nid] == 1:  # recurse each subtree once (DAG-safe)
                for child in node.children():
                    walk(child)

        walk(plan)
        shared: set[int] = set()
        seen: set[int] = set()

        def collect(node) -> None:
            nid = id(node)
            if nid in seen:
                return
            seen.add(nid)
            if counts.get(nid, 0) >= 2 and not isinstance(
                node,
                (DataFrameSource, ParquetSource, CSVSource),
            ):
                # Aggregates were once excluded here (q15 anomaly hunt — the
                # real cause was the childless wrapper hiding its inner from
                # the fusion post-passes, since fixed). The exclusion later
                # turned HARMFUL: in scan mode q15's shared root optimizes to
                # the Aggregate itself, sharing silently failed, `rev` was
                # computed twice with ULP-different float sums, and the
                # total_revenue == mx equality returned EMPTY at SF-300.
                shared.add(nid)
            for child in node.children():
                collect(child)

        collect(plan)
        return shared

    def _fuse_filter_aggregates(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Collapse ungrouped scan->filter/project->aggregate subtrees onto
        the fused single-pass Cython kernel (PhysicalFusedFilterAgg).

        Conservative: every predicate conjunct must be a numeric/datetime
        range over a source column and every aggregate must resolve
        (through the fused projections) to sum/sum-of-product/count/min/
        max/mean of source columns; anything else leaves the plan
        untouched. The original subtree rides along as runtime fallback.
        """
        children = plan.children()
        if children:
            new_children = [self._fuse_filter_aggregates(c) for c in children]
            if any(n is not o for n, o in zip(new_children, children, strict=True)):
                if isinstance(plan, PhysicalHashJoin):
                    plan = dataclasses.replace(
                        plan, left=new_children[0], right=new_children[1]
                    )
                elif isinstance(plan, PhysicalConcat):
                    plan = dataclasses.replace(plan, inputs=tuple(new_children))
                elif isinstance(plan, PhysicalJoinChain):
                    plan = dataclasses.replace(plan, bases=tuple(new_children))
                else:
                    plan = dataclasses.replace(plan, input=new_children[0])

        # Shape A: scan -> fused(filter/project) -> HashAggregate.
        if isinstance(plan, PhysicalHashAggregate):
            node = plan.input
            if isinstance(node, PhysicalMaterialize):
                node = node.input
            if not isinstance(node, PhysicalFusedPipeline):
                return plan
            if not isinstance(node.input, PhysicalScan):
                return plan
            spec = self._translate_fused_agg(node, plan)
            if spec is None:
                return plan
            return PhysicalFusedFilterAgg(
                scan=node.input,
                spec=spec,
                fallback=plan,
                schema=plan.output_schema,
            )

        # Shape B: scan -> fused(filter/project, ..., scalar-agg project).
        # An ungrouped `select(col.sum())` lowers to a reducing project at
        # the tail of the fused pipeline rather than a HashAggregate node, so
        # Shape A misses it and it falls to the row-compacting generic
        # pipeline (measured 5x slower than the kernel — see
        # docs/MATERIALIZATION_EXPERIMENT.md). Peel the terminal aggregate
        # project off, reuse the same translator with a no-group shim.
        if isinstance(plan, PhysicalFusedPipeline) and isinstance(
            plan.input, PhysicalScan
        ):
            ops = plan.operations
            if not ops or ops[-1].op_type != "project":
                return plan
            term_exprs = list(ops[-1].exprs or ())

            def _is_agg_expr(e):
                ir = e._ir.arg if isinstance(e._ir, Alias) else e._ir
                return isinstance(ir, Call) and ir.is_aggregate

            if not term_exprs or not all(_is_agg_expr(e) for e in term_exprs):
                return plan
            prefix = dataclasses.replace(plan, operations=ops[:-1])
            shim = SimpleNamespace(group_by=[], agg_exprs=term_exprs)
            spec = self._translate_fused_agg(prefix, shim)
            if spec is None:
                return plan
            return PhysicalFusedFilterAgg(
                scan=plan.input,
                spec=spec,
                fallback=plan,
                schema=plan.output_schema,
            )

        return plan

    @staticmethod
    def _join_keys_integer(join: PhysicalHashJoin) -> bool:
        """Whether the join's key columns are int (packable by the kernel).

        Static gate so we only build a fused node for joins whose keys the
        Cython indexer can actually pack — string/datetime-keyed joins would
        always fall back at runtime (re-running the sides), a regression.
        """
        if join.on is not None:
            lkeys, rkeys = list(join.on), list(join.on)
        elif join.left_on is not None and join.right_on is not None:
            lkeys, rkeys = list(join.left_on), list(join.right_on)
        else:
            return False
        if len(lkeys) not in (1, 2) or len(lkeys) != len(rkeys):
            return False
        try:
            ls, rs = join.left.output_schema, join.right.output_schema
            for sch, keys in ((ls, lkeys), (rs, rkeys)):
                for k in keys:
                    dt = getattr(sch[k], "numpy_dtype", None)
                    if dt is None or dt.kind not in "iu" or dt == np.dtype("uint64"):
                        return False
        except (KeyError, AttributeError, TypeError):
            return False
        return True

    def _fuse_join_aggregates(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Collapse ``HashAggregate(Materialize?(inner HashJoin))`` onto a
        single PhysicalFusedJoinAgg that gathers only the group/agg columns off
        the join indices, skipping the full-join materialization.

        Static gate: inner join with int keys feeding a non-empty group. The
        aggregate may be anything (re-run unchanged over the narrow input);
        runtime ``_build_narrow`` falls back if a referenced column cannot be
        resolved off one side. See PhysicalFusedJoinAgg /
        docs/BUFFER_JOIN_AGG_PROBE.md.
        """
        if not FUSE_JOIN_AGG:
            return plan

        children = plan.children()
        if children:
            new_children = [self._fuse_join_aggregates(c) for c in children]
            if any(n is not o for n, o in zip(new_children, children, strict=True)):
                if isinstance(plan, (PhysicalHashJoin, PhysicalFusedJoinAgg)):
                    plan = dataclasses.replace(
                        plan, left=new_children[0], right=new_children[1]
                    )
                elif isinstance(plan, PhysicalConcat):
                    plan = dataclasses.replace(plan, inputs=tuple(new_children))
                elif isinstance(plan, PhysicalJoinChain):
                    plan = dataclasses.replace(plan, bases=tuple(new_children))
                else:
                    plan = dataclasses.replace(plan, input=new_children[0])

        if not isinstance(plan, PhysicalHashAggregate) or not plan.group_by:
            return plan
        node = plan.input
        if isinstance(node, PhysicalMaterialize):
            node = node.input
        if not isinstance(node, PhysicalHashJoin) or node.how != "inner":
            return plan
        if not self._join_keys_integer(node):
            return plan

        return PhysicalFusedJoinAgg(
            left=node.left,
            right=node.right,
            join=node,
            agg=plan,
            fallback=plan,
            schema=plan.output_schema,
        )

    @staticmethod
    def _translate_fused_agg(fused, agg):
        import numpy as _np

        INT_MIN, INT_MAX = -(2**63) + 1, 2**63 - 1

        # Predicate slot (int64 vs float64) follows the COLUMN dtype, not
        # the literal: `l_quantity < 24` is an int literal over a float64
        # column — int-range semantics (hi = 23) would silently change the
        # result, and the runtime dtype check would bail to the fallback.
        col_kinds: dict = {}
        try:
            schema = fused.input.output_schema
            for name in schema.names:
                dt = getattr(schema[name], "numpy_dtype", None)
                if dt is not None:
                    col_kinds[name] = dt.kind
        except Exception:
            pass

        def lit_value(ir):
            if not isinstance(ir, IRLiteral):
                return None
            v = ir.value
            if hasattr(v, "value") and hasattr(v, "tz"):  # Timestamp
                return ("i", int(v.value)) if v.tz is None else None
            if isinstance(v, _np.datetime64):
                return ("i", int(_np.datetime64(v, "ns").view("int64")))
            if isinstance(v, (bool,)):
                return None
            if isinstance(v, (int, _np.integer)):
                return ("n", int(v))
            if isinstance(v, (float, _np.floating)):
                return ("f", float(v))
            return None

        env: dict = {}

        def resolve(ir):
            if isinstance(ir, Alias):
                return resolve(ir.arg)
            if isinstance(ir, FieldRef):
                return env.get(ir.name, ir)
            if isinstance(ir, Call):
                return Call(ir.function, tuple(resolve(a) for a in ir.args))
            return ir

        spec = FusedAggSpec()
        i64r: dict = {}
        f64r: dict = {}

        def add_range(col_ir, op, val):
            if not isinstance(col_ir, FieldRef):
                return False
            kind, v = val
            name = col_ir.name
            ck = col_kinds.get(name)
            if ck == "f" and kind in ("n", "f"):
                kind = "f"
            elif ck in ("i", "m", "M") and kind == "n":
                kind = "i"
            elif ck is None:
                return False
            if kind == "i" or (kind == "n" and op in ("ge", "gt", "le", "lt", "eq")):
                lo, hi = i64r.get(name, (INT_MIN, INT_MAX))
                iv = int(v)
                if op == "ge":
                    lo = max(lo, iv)
                elif op == "gt":
                    lo = max(lo, iv + 1)
                elif op == "le":
                    hi = min(hi, iv)
                elif op == "lt":
                    hi = min(hi, iv - 1)
                else:
                    lo, hi = max(lo, iv), min(hi, iv)
                i64r[name] = (lo, hi)
                return True
            if kind in ("f", "n"):
                lo, hi = f64r.get(name, (-_np.inf, _np.inf))
                fv = float(v)
                if op == "ge":
                    lo = max(lo, fv)
                elif op == "gt":
                    lo = max(lo, _np.nextafter(fv, _np.inf))
                elif op == "le":
                    hi = min(hi, fv)
                elif op == "lt":
                    hi = min(hi, _np.nextafter(fv, -_np.inf))
                else:
                    lo, hi = max(lo, fv), min(hi, fv)
                f64r[name] = (lo, hi)
                return True
            return False

        CMP = {
            "greater_equal": "ge",
            "greater": "gt",
            "less_equal": "le",
            "less": "lt",
            "equal": "eq",
        }
        FLIP = {"ge": "le", "gt": "lt", "le": "ge", "lt": "gt", "eq": "eq"}

        def conjuncts(ir, out):
            if isinstance(ir, Call) and ir.function == "and_":
                for a in ir.args:
                    conjuncts(a, out)
            else:
                out.append(ir)

        for op in fused.operations:
            if op.op_type == "filter":
                pred = getattr(op, "predicate", None) or getattr(op, "expr", None)
                if pred is None:
                    return None
                parts: list = []
                conjuncts(resolve(pred._ir), parts)
                for c in parts:
                    if not (isinstance(c, Call) and len(c.args) == 2):
                        return None
                    a, b = c.args
                    cmp = CMP.get(c.function)
                    if cmp is None:
                        return None
                    va, vb = lit_value(a), lit_value(b)
                    if vb is not None and not isinstance(a, IRLiteral):
                        if not add_range(a, cmp, vb):
                            return None
                    elif va is not None and not isinstance(b, IRLiteral):
                        if not add_range(b, FLIP[cmp], va):
                            return None
                    else:
                        return None
            elif op.op_type == "project":
                new_env = {}
                for e in op.exprs or ():
                    ir = e._ir
                    name = extract_output_name(e)
                    if isinstance(ir, Alias):
                        ir = ir.arg
                    new_env[name] = resolve(ir)
                env = new_env
            else:
                return None  # limit etc: not fusable

        for g in agg.group_by:
            g_ir = resolve(g._ir if not isinstance(g._ir, Alias) else g._ir.arg)
            if isinstance(g_ir, Alias):
                g_ir = g_ir.arg
            if not isinstance(g_ir, FieldRef):
                return None
            spec.group_cols.append(g_ir.name)

        AGGK = {"sum": 0, "count": 2, "min": 3, "max": 4}
        for e in agg.agg_exprs:
            out_name = extract_output_name(e)
            ir = e._ir
            if isinstance(ir, Alias):
                ir = ir.arg
            if not (isinstance(ir, Call) and ir.is_aggregate and ir.args):
                return None
            fn = ir.function
            arg = resolve(ir.args[0])

            def one_minus(x):
                # subtract(1, F) or subtract(F-from-lit ...): match 1 - F
                if (
                    isinstance(x, Call)
                    and x.function == "subtract"
                    and len(x.args) == 2
                    and isinstance(x.args[0], IRLiteral)
                    and x.args[0].value == 1
                    and isinstance(x.args[1], FieldRef)
                ):
                    return x.args[1].name
                return None

            def one_plus(x):
                if isinstance(x, Call) and x.function == "add" and len(x.args) == 2:
                    a0, a1 = x.args
                    if (
                        isinstance(a0, IRLiteral)
                        and a0.value == 1
                        and isinstance(a1, FieldRef)
                    ):
                        return a1.name
                    if (
                        isinstance(a1, IRLiteral)
                        and a1.value == 1
                        and isinstance(a0, FieldRef)
                    ):
                        return a0.name
                return None

            def classify(a):
                """-> (kind, col_a, col_b, col_c) or None."""
                if isinstance(a, FieldRef):
                    return (0, a.name, None, None)
                if not (isinstance(a, Call) and a.function == "multiply"):
                    return None
                if len(a.args) != 2:
                    return None
                x, y = a.args
                if isinstance(x, FieldRef) and isinstance(y, FieldRef):
                    return (1, x.name, y.name, None)
                if isinstance(x, FieldRef) and one_minus(y) is not None:
                    return (5, x.name, one_minus(y), None)
                if isinstance(y, FieldRef) and one_minus(x) is not None:
                    return (5, y.name, one_minus(x), None)
                # (F * (1-F)) * (1+F) in either nesting order
                inner, outer = (x, y) if isinstance(x, Call) else (y, x)
                if (
                    isinstance(inner, Call)
                    and inner.function == "multiply"
                    and len(inner.args) == 2
                ):
                    sub = classify(inner)
                    if sub is not None and sub[0] == 5:
                        cc = one_plus(outer)
                        if cc is not None:
                            return (6, sub[1], sub[2], cc)
                return None

            cls = classify(arg)
            if fn == "count":
                spec.aggs.append((out_name, 2, None, None, None))
            elif fn == "sum":
                if cls is None:
                    return None
                k, ca, cb, cc = cls
                kind = 1 if k == 1 else (k if k in (5, 6) else 0)
                spec.aggs.append((out_name, kind, ca, cb, cc))
            elif fn in ("min", "max"):
                if cls is None or cls[0] != 0:
                    return None
                spec.aggs.append((out_name, AGGK[fn], cls[1], None, None))
            elif fn == "mean":
                if cls is None or cls[0] != 0:
                    return None
                s_slot = len(spec.aggs)
                spec.aggs.append((f"__fused_sum_{out_name}", 0, cls[1], None, None))
                c_slot = len(spec.aggs)
                spec.aggs.append((f"__fused_cnt_{out_name}", 2, None, None, None))
                spec.mean_outs[out_name] = (s_slot, c_slot)
            else:
                return None
        for name, (lo, hi) in i64r.items():
            spec.i64_preds.append((name, lo, hi))
        for name, (lo, hi) in f64r.items():
            spec.f64_preds.append((name, lo, hi))
        if not spec.aggs:
            return None
        if not spec.group_cols and any(a[1] in (5, 6) for a in spec.aggs):
            return None  # product-minus forms only in the grouped kernel
        if spec.group_cols:
            # GROUPED FUSION IS OFF — measured losses across the board
            # (controlled on/off at SF-3): q1 1.05x (11-slot per-row
            # scatter can't auto-vectorize; acero's SIMD grouped agg wins),
            # q15 1.20x and q20 1.26x (full-column fetch copies + group-code
            # derivation cost more than the already-fused morsel-parallel
            # baseline on selective filters). The grouped kernel and
            # translation stay as tested infrastructure; revisiting needs
            # zero-copy column access and/or SIMD scatter. Ungrouped
            # (q6-class) fusion is a clean 0.31x win and stays on.
            return None
        return spec

    def _collapse_join_chains(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Collapse left-deep trees of eligible inner joins into one
        PhysicalJoinChain breaker (late materialization — see that class).

        Statically eligible step: inner, single-key equi. Runtime gates
        (key dtypes, name overlap, spill/index contexts) live in the chain
        node, which degrades per step to the original join semantics.
        """

        def is_step(node) -> bool:
            if not (isinstance(node, PhysicalHashJoin) and node.how == "inner"):
                return False
            if node.on is not None:
                pairs = [(k, k) for k in node.on]
            elif (
                node.left_on is not None
                and node.right_on is not None
                and len(node.left_on) == len(node.right_on)
            ):
                pairs = list(zip(node.left_on, node.right_on, strict=True))
            else:
                return False
            if len(pairs) == 1:
                return True
            if len(pairs) != 2:
                return False
            # Two-key steps collapse only when both keys are integer (they
            # pack into one int64 at runtime). Non-int composite keys keep
            # the nested joins so the decision layer can still route them
            # to acero (chains have no acero fallback inside).
            try:
                ls = node.left.output_schema
                rs = node.right.output_schema
                for ln, rn in pairs:
                    for schema, name in ((ls, ln), (rs, rn)):
                        dt = schema[name]
                        np_dt = getattr(dt, "numpy_dtype", None)
                        if np_dt is None or np_dt.kind not in "iu":
                            return False
            except Exception:
                return False
            return True

        def rewrite(node: PhysicalPlan) -> PhysicalPlan:
            inner = _unwrap_materialize(node)
            if is_step(inner) and is_step(_unwrap_materialize(inner.left)):
                # Collect the maximal left-deep chain bottom-up.
                steps: list[PhysicalHashJoin] = []
                cur = inner
                while is_step(cur):
                    steps.append(cur)
                    nxt = _unwrap_materialize(cur.left)
                    if not is_step(nxt):
                        break
                    cur = nxt
                steps.reverse()  # bottom-up
                bases = [rewrite(steps[0].left)] + [rewrite(s.right) for s in steps]
                chain = PhysicalJoinChain(
                    bases=tuple(bases),
                    steps=tuple(steps),
                    schema=inner.output_schema,
                )
                if isinstance(node, PhysicalMaterialize):
                    return dataclasses.replace(node, input=chain)
                return chain
            children = node.children()
            if not children:
                return node
            new_children = [rewrite(c) for c in children]
            if all(n is o for n, o in zip(new_children, children, strict=True)):
                return node
            if isinstance(node, PhysicalHashJoin):
                return dataclasses.replace(
                    node, left=new_children[0], right=new_children[1]
                )
            if isinstance(node, PhysicalConcat):
                return dataclasses.replace(node, inputs=tuple(new_children))
            return dataclasses.replace(node, input=new_children[0])

        return rewrite(plan)

    def _plan_recursive(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        """Recursively convert logical plan to physical plan."""

        nid = id(logical_plan)
        shared = getattr(self, "_shared_ids", None)
        if shared is not None and nid in shared:
            wrapper = self._shared_wrappers.get(nid)
            if wrapper is None:
                shared.discard(nid)  # avoid recursing into this hook
                inner = self._plan_recursive(logical_plan)
                shared.add(nid)
                wrapper = PhysicalCachedSubplan(
                    inner=inner, key=nid, schema=logical_plan.resolve_schema()
                )
                self._shared_wrappers[nid] = wrapper
            return wrapper

        if isinstance(logical_plan, DataFrameSource):
            return self._plan_scan(logical_plan)

        elif isinstance(logical_plan, ParquetSource):
            return self._plan_parquet_scan(logical_plan)

        elif isinstance(logical_plan, CSVSource):
            return self._plan_csv_scan(logical_plan)

        elif isinstance(logical_plan, Project):
            return self._plan_project(logical_plan)

        elif isinstance(logical_plan, Filter):
            return self._plan_filter(logical_plan)

        elif isinstance(logical_plan, Aggregate):
            return self._plan_aggregate(logical_plan)

        elif isinstance(logical_plan, Sort):
            return self._plan_sort(logical_plan)

        elif isinstance(logical_plan, TopK):
            return self._plan_topk(logical_plan)

        elif isinstance(logical_plan, Limit):
            return self._plan_limit(logical_plan)

        elif isinstance(logical_plan, Distinct):
            return self._plan_distinct(logical_plan)

        elif isinstance(logical_plan, GroupByHead):
            return self._plan_group_by_head(logical_plan)

        elif isinstance(logical_plan, Join):
            return self._plan_join(logical_plan)

        elif isinstance(logical_plan, Convert):
            return self._plan_convert(logical_plan)

        elif isinstance(logical_plan, SetIndex):
            return self._plan_set_index(logical_plan)

        elif isinstance(logical_plan, ResetIndex):
            return self._plan_reset_index(logical_plan)

        elif isinstance(logical_plan, Concat):
            return self._plan_concat(logical_plan)

        else:
            raise NotImplementedError(
                f"Physical planning not implemented for: {type(logical_plan)}"
            )

    def _plan_scan(self, node) -> PhysicalScan:
        """Plan a DataFrameSource."""
        return PhysicalScan(
            df=node.df,
            schema=node.resolve_schema(),
        )

    def _plan_parquet_scan(self, node) -> PhysicalParquetScan:
        """Plan a ParquetSource."""
        return PhysicalParquetScan(
            path=node.path,
            schema=node.resolve_schema(),
            columns=node.columns,
            predicate=node.predicate,
            limit=node.limit,
        )

    def _plan_csv_scan(self, node) -> PhysicalPlan:
        """Plan a CSVSource.

        Unlike Parquet, CSV doesn't support native predicate pushdown.
        If there's a predicate, we wrap the scan with a PhysicalFilter.
        """
        scan = PhysicalCSVScan(
            path=node.path,
            schema=node.resolve_schema(),
            columns=node.columns,
            predicate=None,  # Predicate applied by filter, not scan
            sep=node.sep,
            header=node.header,
            skip_rows=node.skip_rows,
            n_rows=node.n_rows,
        )

        # If there's a predicate, add a filter step after scanning
        if node.predicate is not None:
            return PhysicalFilter(
                input=scan,
                predicate=node.predicate,
                schema=node.resolve_schema(),
                backend=self._choose_backend_for_exprs((node.predicate,)),
            )

        return scan

    def _plan_project(self, node) -> PhysicalProject:
        """Plan a Project."""
        return PhysicalProject(
            input=self._plan_recursive(node.input),
            exprs=node.exprs,
            schema=node.resolve_schema(),
            backend=self._choose_backend_for_exprs(node.exprs),
        )

    def _plan_filter(self, node) -> PhysicalFilter:
        """Plan a Filter."""
        return PhysicalFilter(
            input=self._plan_recursive(node.input),
            predicate=node.predicate,
            schema=node.resolve_schema(),
            backend=self._choose_backend_for_exprs((node.predicate,)),
        )

    def _plan_aggregate(self, node) -> PhysicalHashAggregate:
        """Plan an Aggregate."""
        # Aggregate is a pipeline breaker - needs all rows per group
        # Wrap input in Materialize to make boundary explicit
        return PhysicalHashAggregate(
            input=self._materialize_for_breaker(node.input, "aggregate"),
            group_by=node.group_by,
            agg_exprs=node.agg_exprs,
            schema=node.resolve_schema(),
        )

    def _plan_sort(self, node) -> PhysicalSort:
        """Plan a Sort."""
        # Sort is a pipeline breaker - needs all data for global ordering
        # Wrap input in Materialize to make boundary explicit
        return PhysicalSort(
            input=self._materialize_for_breaker(node.input, "sort"),
            by=node.by,
            descending=node.descending,
            schema=node.resolve_schema(),
            algorithm="quicksort",
        )

    def _plan_topk(self, node) -> PhysicalTopK:
        """Plan a TopK."""
        return PhysicalTopK(
            input=self._plan_recursive(node.input),
            k=node.k,
            by=node.by,
            descending=node.descending,
            schema=node.resolve_schema(),
        )

    def _plan_limit(self, node) -> PhysicalLimit:
        """Plan a Limit."""
        return PhysicalLimit(
            input=self._plan_recursive(node.input),
            n=node.n,
            offset=node.offset,
            schema=node.resolve_schema(),
        )

    def _plan_distinct(self, node) -> PhysicalDistinct:
        """Plan a Distinct."""
        # Distinct is a pipeline breaker - needs all values to deduplicate
        # Wrap input in Materialize to make boundary explicit
        return PhysicalDistinct(
            input=self._materialize_for_breaker(node.input, "distinct"),
            subset=node.subset,
            schema=node.resolve_schema(),
        )

    def _plan_group_by_head(self, node) -> PhysicalGroupByHead:
        """Plan a GroupByHead (group-wise head; a pipeline breaker)."""
        group_keys = tuple(extract_output_name(e) for e in node.group_by)
        return PhysicalGroupByHead(
            input=self._materialize_for_breaker(node.input, "group_by_head"),
            group_keys=group_keys,
            n=node.n,
            schema=node.resolve_schema(),
        )

    def _plan_join(self, node) -> PhysicalHashJoin:
        """Plan a Join."""
        # Get row count estimates for build/probe optimization
        left_rows = node.left.estimate_row_count()
        right_rows = node.right.estimate_row_count()

        # Hash join: build side must be fully materialized to build hash table
        # Currently we materialize both sides; future optimization could stream
        # the probe side through the join.
        #
        # Note: PhysicalHashJoin internally chooses which side is build vs probe
        # based on row estimates. Both sides need materialization for now.
        return PhysicalHashJoin(
            left=self._materialize_for_breaker(node.left, "hash_join_build"),
            right=self._materialize_for_breaker(node.right, "hash_join_build"),
            on=node.on,
            left_on=node.left_on,
            right_on=node.right_on,
            how=node.how,
            suffix=node.suffix,
            schema=node.resolve_schema(),
            left_rows_estimate=left_rows,
            right_rows_estimate=right_rows,
        )

    def _plan_convert(self, node) -> PhysicalConvert:
        """Plan a Convert (backend conversion)."""
        return PhysicalConvert(
            input=self._plan_recursive(node.input),
            target_backend=node.target_backend,
            schema=node.resolve_schema(),
        )

    def _plan_set_index(self, node) -> PhysicalSetIndex:
        """Plan a SetIndex."""
        return PhysicalSetIndex(
            input=self._plan_recursive(node.input),
            keys=node.keys,
            drop=node.drop,
            schema=node.resolve_schema(),
        )

    def _plan_reset_index(self, node) -> PhysicalResetIndex:
        """Plan a ResetIndex."""
        return PhysicalResetIndex(
            input=self._plan_recursive(node.input),
            drop=node.drop,
            schema=node.resolve_schema(),
        )

    def _plan_concat(self, node) -> PhysicalConcat:
        """Plan a Concat."""
        return PhysicalConcat(
            inputs=tuple(self._plan_recursive(inp) for inp in node.inputs),
            schema=node.resolve_schema(),
        )

    def _apply_fusion(self, plan: PhysicalPlan) -> PhysicalPlan:
        """
        Apply operator fusion to optimize the physical plan.

        Detects chains of fuseable operators (Filter, Project, Limit) and
        combines them into single PhysicalFusedPipeline operators.

        Fusion Rules
        ------------
        1. Filter → Project: Fuse to evaluate predicate before expressions
        2. Project → Project: Combine into single projection
        3. Filter → Filter: Combine predicates (AND)
        4. Filter → Limit: Short-circuit when limit is reached
        5. Project → Limit: Fuse to stop early
        6. Any combination of above

        Fusion Boundaries
        -----------------
        Fusion stops at:
        - Pipeline breakers (Sort, Aggregate, Join, Distinct)
        - Materialize nodes
        - Scan nodes (fusion starts fresh after)

        Parameters
        ----------
        plan : PhysicalPlan
            The physical plan to optimize.

        Returns
        -------
        PhysicalPlan
            Optimized plan with fused operators.
        """
        # First, recursively apply fusion to children
        plan = self._apply_fusion_to_children(plan)

        # Then check if this node can be fused with its input
        return self._try_fuse(plan)

    def _apply_fusion_to_children(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Recursively apply fusion to all children of a plan node."""
        from dataclasses import replace

        children = plan.children()
        if not children:
            return plan

        # Recursively optimize children
        new_children = [self._apply_fusion(child) for child in children]

        # If no changes, return original
        if all(new is old for new, old in zip(new_children, children, strict=True)):
            return plan

        # Create new node with optimized children
        if hasattr(plan, "input") and len(new_children) == 1:
            return replace(plan, input=new_children[0])
        elif hasattr(plan, "left") and hasattr(plan, "right"):
            if len(new_children) == 2:
                return replace(plan, left=new_children[0], right=new_children[1])
        elif hasattr(plan, "inputs"):
            return replace(plan, inputs=tuple(new_children))
        else:
            # Unknown structure, return as-is
            return plan

    def _try_fuse(self, plan: PhysicalPlan) -> PhysicalPlan:
        """
        Try to fuse this plan node with its input(s) into a FusedPipeline.

        Only Filter, Project, and Limit can be fused.
        """
        # Check if this is a fuseable operator
        if not isinstance(plan, (PhysicalFilter, PhysicalProject, PhysicalLimit)):
            return plan

        # Don't fuse tail operations (offset=-1)
        if isinstance(plan, PhysicalLimit) and plan.offset == -1:
            return plan

        # Collect the chain of fuseable operations
        operations: list[FusedOperation] = []
        current = plan
        base_input = None

        while True:
            if isinstance(current, PhysicalFilter):
                operations.append(
                    FusedOperation(op_type="filter", predicate=current.predicate)
                )
                current = current.input

            elif isinstance(current, PhysicalProject):
                operations.append(
                    FusedOperation(op_type="project", exprs=current.exprs)
                )
                current = current.input

            elif isinstance(current, PhysicalLimit) and current.offset != -1:
                # Only fuse head() operations, not tail()
                operations.append(FusedOperation(op_type="limit", limit_n=current.n))
                current = current.input

            elif isinstance(current, PhysicalFusedPipeline):
                # Already fused - absorb its operations. They are stored
                # in execution (bottom-up) order, but this collection loop
                # builds a top-down list that is reversed at the end, so
                # they must be added reversed to survive that reversal.
                # (Getting this wrong made later projections drop columns
                # that earlier fused filters still referenced.)
                operations.extend(reversed(current.operations))
                current = current.input

            else:
                # Not fuseable - this is the base input
                base_input = current
                break

        # Reverse to get operations in execution order (bottom-up to top-down)
        operations.reverse()

        # A single FILTER always fuses: PhysicalFusedPipeline is what carries
        # morsel parallelism and prune-before-mask, so a bare PhysicalFilter
        # over a large input is ~15-20x slower than the same filter fused
        # (q15's shared filter: 691 vs 44 ms; q20's filter-on-join-output:
        # 929 ms bare). Single project/limit chains keep the no-benefit
        # exemption — they have no mask/prune work to parallelize.
        if len(operations) == 1 and operations[0].op_type == "filter":
            return PhysicalFusedPipeline(
                input=base_input,
                operations=tuple(operations),
                schema=plan.output_schema,
            )
        # If only one operation and no fusion benefit, return original
        if len(operations) == 1:
            return plan

        # Check if fusion is beneficial
        # At minimum we need: Filter+Project, Filter+Limit, or Project+Limit
        has_filter = any(op.op_type == "filter" for op in operations)
        has_project = any(op.op_type == "project" for op in operations)
        has_limit = any(op.op_type == "limit" for op in operations)

        # Fusion is beneficial if we have at least two different types
        # or multiple of the same type that can be combined
        beneficial = (
            (has_filter and has_project)
            or (has_filter and has_limit)
            or (has_project and has_limit)
            or sum(1 for op in operations if op.op_type == "filter") > 1
            or sum(1 for op in operations if op.op_type == "project") > 1
        )

        if not beneficial:
            return plan

        return PhysicalFusedPipeline(
            input=base_input,
            operations=tuple(operations),
            schema=plan.output_schema,
        )

    def _choose_backend_for_exprs(
        self, exprs: tuple[Expr, ...]
    ) -> Literal["auto", "arrow", "numpy"]:
        """
        Choose the best backend for a set of expressions.

        For now, returns "auto". Future versions could analyze
        expressions and choose based on operation requirements.
        """
        # TODO: Implement backend selection based on expression analysis
        return self.preferred_backend


# =============================================================================
# Execution Entry Point
# =============================================================================


def execute_physical_plan(
    plan: PhysicalPlan,
    *,
    preferred_backend: Literal["auto", "arrow", "numpy"] = "auto",
    strict: bool = False,
    preserve_index: bool = False,
    order_relaxed: bool = False,
    spill_config: SpillConfig | None = None,
) -> DataFrame:
    """
    Execute a physical plan and return the result.

    Parameters
    ----------
    plan : PhysicalPlan
        The physical plan to execute.
    preferred_backend : {"auto", "arrow", "numpy"}
        Preferred execution backend.
    strict : bool
        If True, fail on backend fallbacks.
    preserve_index : bool, default False
        If True, preserve the original DataFrame index.
    order_relaxed : bool, default False
        If True, the final output row order is unspecified, which lets
        order-preserving joins route to acero's parallel hash join.
    spill_config : SpillConfig or None, default None
        Configuration for disk spilling under memory pressure.
        When enabled, intermediate results can be spilled to disk.

    Returns
    -------
    DataFrame
        The execution result.
    """
    import pandas as pd

    # Check if adaptive thresholds are enabled
    adaptive_enabled = pd.get_option("compute.lazy.adaptive_thresholds")

    context = ExecutionContext(
        preferred_backend=preferred_backend,
        strict=strict,
        adaptive_thresholds=adaptive_enabled,
        preserve_index=preserve_index,
        order_relaxed=order_relaxed,
        _spill_config=spill_config,
    )

    # Execute through the pipeline engine (docs/ENGINE_DESIGN.md, M1):
    # the plan compiles to an explicit pipeline graph; every node still
    # runs its own execute(), so behavior is identical to the previous
    # direct recursion.
    from pandas.lazy.engine import execute_as_pipelines

    arrays = execute_as_pipelines(plan, context)

    # Convert ArrayDict back to DataFrame with proper index
    # Reconstruct index if preserve_index=True OR user explicitly called set_index()
    should_reconstruct_index = preserve_index or context.user_set_index
    # Pipeline breakers materialize every output column from a fresh
    # take/aggregate, so the result can be assembled without the
    # block-consolidation copy (no column aliases user data).
    materialized_output = isinstance(
        plan,
        (
            PhysicalSort,
            PhysicalHashAggregate,
            PhysicalDistinct,
            PhysicalTopK,
            PhysicalHashJoin,
        ),
    )
    return arrays_to_dataframe(
        arrays,
        index_names=context.index_names,
        index_is_multi=context.index_is_multi,
        preserve_index=should_reconstruct_index,
        schema=plan.output_schema,
        materialized_output=materialized_output,
    )
