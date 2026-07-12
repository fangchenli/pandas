"""Repair DataFusion-produced Substrait into *portable* Substrait.

The Substrait roundtrip probe (`substrait_roundtrip.py`) found DataFusion's
producer emits plans its own consumer tolerates but no second engine will accept
(0/22 TPC-H into Acero) — see docs/upstream/AG17. Two omissions do it:

  1. every ``Expression.ScalarFunction`` is emitted with **no ``output_type``**
     (the required field), so a strict consumer can't type the expression;
  2. every aggregate is emitted with phase ``AGGREGATION_PHASE_UNSPECIFIED``,
     which Acero rejects.

This module is the engine's **local workaround** (keep-the-workaround principle):
post-process the emitted protobuf to fill both in, so our Substrait actually fans
out to Acero (and any other compliant consumer) *now*, without waiting on the
upstream fix. It re-derives exactly what the producer should have: propagate types
bottom-up (reads carry base_schema types; project/filter/aggregate/join transform
them), set each missing ``output_type`` from the function's semantics, and stamp a
concrete aggregation phase.

Scope: the node/expr/function surface the lazy→DataFusion lowering actually emits
(the 22 TPC-H plans — see the enumerated surface in AG17). Unknown functions fall
back to their first argument's type; unknown rels contribute no columns. Purely a
protobuf rewrite — no execution, no DataFusion dependency.
"""

from __future__ import annotations

from substrait.proto import (
    Plan,
    Type,
)

# --- function → result-type category (names are DataFusion's Substrait fn names)
_BOOL_FNS = frozenset(
    {
        "equal", "not_equal", "gt", "lt", "gte", "lte", "and", "or", "not",
        "is_null", "is_not_null", "like", "regexp_like", "starts_with",
        "ends_with", "in", "between", "is_not_distinct_from", "is_distinct_from",
    }
)  # fmt: skip
_ARITH_FNS = frozenset(
    {"add", "subtract", "multiply", "divide", "modulus", "negate", "power"}
)
_STRING_FNS = frozenset(
    {"substr", "substring", "concat", "lower", "upper", "trim", "ltrim", "rtrim"}
)
# numeric promotion rank: arithmetic result takes the widest operand type
_NUM_RANK = {"i8": 1, "i16": 2, "i32": 3, "i64": 4, "fp32": 5, "fp64": 6, "decimal": 7}

_NULLABLE = Type.Nullability.NULLABILITY_NULLABLE
_PHASE_INITIAL_TO_RESULT = 3  # AggregationPhase.AGGREGATION_PHASE_INITIAL_TO_RESULT


def _t(kind: str) -> Type:
    """A fresh nullable Type of a non-parametric kind (bool/i64/fp64/string)."""
    t = Type()
    getattr(t, kind).nullability = _NULLABLE
    return t


def _lit_type(lit) -> Type:
    """Type of a literal (only what our lowering emits: numeric/string/bool/ts)."""
    k = lit.WhichOneof("literal_type")
    simple = {
        "boolean": "bool", "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
        "fp32": "fp32", "fp64": "fp64", "string": "string", "date": "date",
    }  # fmt: skip
    if k in simple:
        return _t(simple[k])
    if k == "precision_timestamp":
        t = Type()
        t.precision_timestamp.nullability = _NULLABLE
        t.precision_timestamp.precision = getattr(
            lit.precision_timestamp, "precision", 9
        )
        return t
    return _t("fp64")  # unreachable for our surface; safe numeric default


class _Fixer:
    def __init__(self, plan: Plan):
        # function anchor -> name
        self.fn = {
            e.extension_function.function_anchor: e.extension_function.name
            for e in plan.extensions
            if e.HasField("extension_function")
        }

    # -- expression: infer type, and set any missing scalar-fn output_type --------
    def infer(self, expr, in_types: list) -> Type:
        k = expr.WhichOneof("rex_type")
        if k == "selection":
            idx = expr.selection.direct_reference.struct_field.field
            return in_types[idx] if 0 <= idx < len(in_types) else _t("fp64")
        if k == "literal":
            return _lit_type(expr.literal)
        if k == "cast":
            self.infer(expr.cast.input, in_types)  # fix nested
            return expr.cast.type
        if k == "scalar_function":
            return self._scalar(expr.scalar_function, in_types)
        if k == "if_then":
            # `if` / `else` are Python keywords -> access via getattr
            it = expr.if_then
            last = _t("bool")
            for clause in it.ifs:
                self.infer(getattr(clause, "if"), in_types)
                last = self.infer(clause.then, in_types)
            if it.HasField("else"):
                last = self.infer(getattr(it, "else"), in_types)
            return last
        if k == "singular_or_list":
            self.infer(expr.singular_or_list.value, in_types)
            for opt in expr.singular_or_list.options:
                self.infer(opt, in_types)
            return _t("bool")
        return _t("bool")

    def _scalar(self, sf, in_types: list) -> Type:
        argts = [
            self.infer(a.value, in_types) for a in sf.arguments if a.HasField("value")
        ]
        name = self.fn.get(sf.function_reference, "?")
        if name in _BOOL_FNS:
            out = _t("bool")
        elif name in _STRING_FNS:
            out = _t("string")
        elif name == "date_part":
            out = _t("fp64")  # DataFusion date_part returns Float64
        elif name in _ARITH_FNS:
            out = self._promote(argts)
        else:
            out = argts[0] if argts else _t("bool")
        if not sf.HasField("output_type"):
            sf.output_type.CopyFrom(out)
        return sf.output_type

    @staticmethod
    def _promote(argts: list) -> Type:
        best = None
        for t in argts:
            r = _NUM_RANK.get(t.WhichOneof("kind"), 0)
            if best is None or r > _NUM_RANK.get(best.WhichOneof("kind"), 0):
                best = t
        return best if best is not None else _t("fp64")

    def _agg_out(self, af, in_types: list) -> Type:
        argts = [
            self.infer(a.value, in_types) for a in af.arguments if a.HasField("value")
        ]
        name = self.fn.get(af.function_reference, "?")
        if name == "count":
            return _t("i64")
        if name == "avg":
            return _t("fp64")
        # sum/min/max preserve the argument's numeric type
        return argts[0] if argts else _t("fp64")

    # -- relation: return output column types, fixing exprs along the way ---------
    def rel_out(self, rel) -> list:
        kind = rel.WhichOneof("rel_type")
        node = getattr(rel, kind)
        if kind == "read":
            pre = list(node.base_schema.struct.types)
        elif kind == "filter":
            pre = self.rel_out(node.input)
            self.infer(node.condition, pre)
        elif kind in ("fetch",):
            pre = self.rel_out(node.input)
        elif kind == "sort":
            pre = self.rel_out(node.input)
            for s in node.sorts:
                self.infer(s.expr, pre)
        elif kind == "project":
            in_types = self.rel_out(node.input)
            expr_types = [self.infer(e, in_types) for e in node.expressions]
            pre = list(in_types) + expr_types
        elif kind == "aggregate":
            in_types = self.rel_out(node.input)
            if node.grouping_expressions:  # deduped form
                gtypes = [self.infer(e, in_types) for e in node.grouping_expressions]
            else:
                gtypes = [
                    self.infer(e, in_types)
                    for g in node.groupings
                    for e in g.grouping_expressions
                ]
            mtypes = []
            for meas in node.measures:
                af = meas.measure
                mt = self._agg_out(af, in_types)
                if af.phase == 0:  # UNSPECIFIED -> INITIAL_TO_RESULT (single-stage)
                    af.phase = _PHASE_INITIAL_TO_RESULT
                if not af.HasField("output_type"):
                    af.output_type.CopyFrom(mt)
                mtypes.append(af.output_type)
            pre = gtypes + mtypes
        elif kind in ("join", "cross"):
            left = self.rel_out(node.left)
            right = self.rel_out(node.right)
            if kind == "join" and node.HasField("expression"):
                self.infer(node.expression, list(left) + list(right))
            pre = list(left) + list(right)
        elif kind == "set":
            pre = self.rel_out(node.inputs[0]) if node.inputs else []
        else:
            pre = []
        common = getattr(node, "common", None)
        if common is not None and common.HasField("emit"):
            pre = [pre[i] for i in common.emit.output_mapping if 0 <= i < len(pre)]
        return pre


def _walk_msg(msg, fn):
    """Depth-first visit every protobuf submessage, applying fn(msg)."""
    from google.protobuf.message import Message

    if not isinstance(msg, Message):
        return
    fn(msg)
    for fd, val in msg.ListFields():
        if fd.label == fd.LABEL_REPEATED:
            for v in val:
                _walk_msg(v, fn)
        else:
            _walk_msg(val, fn)


def _downgrade_precision_timestamp(msg):
    """Rewrite Substrait's newer ``precision_timestamp`` (type + literal) to the
    legacy ``timestamp`` an older Arrow/Acero consumer supports. Type: drop
    precision, keep nullability. Literal: nanoseconds -> microseconds (legacy
    ``timestamp`` is micros). Consumers that DO support precision_timestamp don't
    need this — it's a consumer-version workaround (see AG18), opt-in."""
    nm = msg.DESCRIPTOR.name
    if nm == "Type" and msg.WhichOneof("kind") == "precision_timestamp":
        null = msg.precision_timestamp.nullability
        msg.ClearField("precision_timestamp")
        msg.timestamp.nullability = null
    elif nm == "Literal" and msg.WhichOneof("literal_type") == "precision_timestamp":
        v_ns = msg.precision_timestamp.value
        msg.ClearField("precision_timestamp")
        msg.timestamp = v_ns // 1000  # ns -> us


def _fill_cross_join(msg):
    """Acero rejects a ``JoinRel`` with no expression (how DataFusion emits a
    cross join). Inject a boolean-``true`` condition so it reads as inner-join-on-
    true == cross join."""
    if msg.DESCRIPTOR.name == "JoinRel" and not msg.HasField("expression"):
        msg.expression.literal.boolean = True


def _mirror_fetch_count(msg):
    """SILENT-WRONG-RESULT fix. DataFusion emits ``LIMIT n`` in a ``FetchRel``
    via the newer ``count_expr``/``offset_expr`` (Expression) fields and leaves
    the **deprecated** int64 ``count``/``offset`` at their default 0. A consumer
    that reads the deprecated fields (e.g. Acero on pyarrow 23.0.1) then sees
    ``count == 0`` and returns **zero rows** — no error, just an empty result
    (TPC-H q3/q10/q18/q21). Mirror the literal from ``*_expr`` into the deprecated
    field. Harmless for consumers that read ``*_expr`` (they ignore the int64)."""
    if msg.DESCRIPTOR.name != "FetchRel":
        return
    if msg.count == 0 and msg.HasField("count_expr"):
        e = msg.count_expr
        if (
            e.WhichOneof("rex_type") == "literal"
            and e.literal.WhichOneof("literal_type") == "i64"
        ):
            msg.count = e.literal.i64
    if msg.offset == 0 and msg.HasField("offset_expr"):
        e = msg.offset_expr
        if (
            e.WhichOneof("rex_type") == "literal"
            and e.literal.WhichOneof("literal_type") == "i64"
        ):
            msg.offset = e.literal.i64


def fill_output_types(raw: bytes) -> bytes:
    """Apply ONLY the AG17 fix: fill missing ``output_type`` + aggregate phase
    (no cross-join / fetch / timestamp rewrites). Public entry point for
    isolating AG18 behavior — lets a consumer be tested on an AG17-clean plan
    without the other portability rewrites masking it."""
    plan = Plan()
    plan.ParseFromString(raw)
    fx = _Fixer(plan)
    for rel in plan.relations:
        root = rel.root.input if rel.HasField("root") else None
        if root is not None:
            fx.rel_out(root)
    return plan.SerializeToString()


def fix_plan(raw: bytes, legacy_timestamp: bool = False) -> bytes:
    """DataFusion Substrait bytes -> portable Substrait bytes.

    Always: fill missing ``ScalarFunction``/``AggregateFunction`` ``output_type``
    (AG17), stamp aggregate phase, inject a cross-join condition, and mirror
    ``FetchRel.count_expr`` -> deprecated ``count`` (the silent 0-row limit fix).
    With ``legacy_timestamp``, also downgrade ``precision_timestamp`` ->
    ``timestamp`` for consumers that predate the newer type (AG18)."""
    plan = Plan()
    plan.ParseFromString(raw)
    fx = _Fixer(plan)
    for rel in plan.relations:
        root = rel.root.input if rel.HasField("root") else None
        if root is not None:
            fx.rel_out(root)
    _walk_msg(plan, _fill_cross_join)
    _walk_msg(plan, _mirror_fetch_count)
    if legacy_timestamp:
        _walk_msg(plan, _downgrade_precision_timestamp)
    return plan.SerializeToString()
