"""
Predicate-aware selectivity estimation for cardinality propagation.

Logical nodes' ``estimate_row_count`` feed the physical decision layer — join
build-side selection today, parallelism degree and grace-hash partitioning as
those mature. A ``Filter`` previously multiplied its input by a flat ``0.3``
regardless of predicate; this module refines that into a per-predicate
estimate so a selective equality and a broad range no longer look alike.

Reference: P. Griffiths Selinger, M. M. Astrahan, D. D. Chamberlin, R. A.
Lorie & T. G. Price, "Access Path Selection in a Relational Database
Management System", SIGMOD 1979 — the System R optimizer, which introduced
this constant-selectivity model. The constants below are its textbook
defaults; without column statistics (distinct-value counts, histograms) they
are deliberately rough. Their job is to *rank* plan alternatives, not to
predict exact row counts. Statistics-driven refinement (1/NDV for equality,
Parquet min/max for ranges) is the natural next step — see ROADMAP.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas.lazy.ir import IRNode

# Constant selectivities (System R defaults).
SEL_EQ = 0.1  # col == value
SEL_NEQ = 0.9  # col != value
SEL_RANGE = 1.0 / 3.0  # col <, <=, >, >= value
SEL_IS_NULL = 0.05
SEL_IS_NOT_NULL = 0.95
SEL_STR_MATCH = 0.1  # contains / startswith / endswith
SEL_DEFAULT = 0.3  # unknown predicate (the previous flat estimate)

_RANGE_OPS = frozenset({"less", "less_equal", "greater", "greater_equal"})
_STRING_MATCH_OPS = frozenset({"str_contains", "str_startswith", "str_endswith"})


def estimate_selectivity(ir: IRNode) -> float:
    """Estimate the fraction of rows a boolean predicate keeps, in ``(0, 1]``.

    Boolean combinators compose under the standard independence assumption:
    ``AND`` multiplies, ``OR`` uses inclusion-exclusion, ``NOT`` complements.
    Leaf comparisons map to constant selectivities by operator kind.
    """
    from pandas.lazy.ir import (
        Alias,
        Call,
    )

    if isinstance(ir, Alias):
        return estimate_selectivity(ir.arg)
    if not isinstance(ir, Call):
        return SEL_DEFAULT

    fn = ir.function

    # Boolean combinators recurse into their sub-predicates.
    if fn == "and_":
        sel = 1.0
        for arg in ir.args:
            sel *= estimate_selectivity(arg)  # P(A and B) = sA * sB
        return sel
    if fn == "or_":
        sel = 0.0
        for arg in ir.args:
            sa = estimate_selectivity(arg)
            sel = sel + sa - sel * sa  # P(A or B) = sA + sB - sA*sB
        return sel
    if fn == "invert":
        if not ir.args:
            return SEL_DEFAULT
        return max(0.0, 1.0 - estimate_selectivity(ir.args[0]))

    # Leaf predicates: the constant depends on the operator, not the operands.
    if fn == "equal":
        return SEL_EQ
    if fn == "not_equal":
        return SEL_NEQ
    if fn in _RANGE_OPS:
        return SEL_RANGE
    if fn == "is_null":
        return SEL_IS_NULL
    if fn == "is_not_null":
        return SEL_IS_NOT_NULL
    if fn in _STRING_MATCH_OPS:
        return SEL_STR_MATCH
    if fn == "isin":
        values = (ir.kwargs or {}).get("values")
        k = len(values) if values is not None else 1
        # k disjoint equalities, capped so a huge IN-list never exceeds !=.
        return min(SEL_NEQ, k * SEL_EQ)

    return SEL_DEFAULT
