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

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
)

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

# Rows sampled to estimate per-column statistics at plan time. Exact NDV is a
# full hash pass (~430 ms at 10M on a high-cardinality column) — far too much
# to spend planning; a random sample of this many rows costs <1 ms and is
# exact for the low-cardinality columns that actually flip join build sides.
STATS_SAMPLE_SIZE = 65_536

_RANGE_OPS = frozenset({"less", "less_equal", "greater", "greater_equal"})
_STRING_MATCH_OPS = frozenset({"str_contains", "str_startswith", "str_endswith"})
_RANGE_FLIP = {
    "less": "greater",
    "less_equal": "greater_equal",
    "greater": "less",
    "greater_equal": "less_equal",
}

# A column-statistics lookup: name -> ColumnStats or None when unavailable.
StatsLookup = Callable[[str], "ColumnStats | None"]


@dataclass(frozen=True)
class ColumnStats:
    """Per-column statistics used to refine selectivity estimates.

    Any field may be ``None`` when unknown. ``ndv`` (number of distinct
    values) may be sampled — ``approximate`` flags that — so it ranks plans
    rather than predicting exact counts.
    """

    row_count: int | None = None
    ndv: int | None = None
    min_val: float | None = None
    max_val: float | None = None
    null_count: int | None = None
    approximate: bool = False


def _field_and_literal(ir) -> tuple[str, object, bool] | None:
    """For a binary comparison, return (column, literal, field_is_left)."""
    from pandas.lazy.ir import (
        FieldRef,
        Literal,
    )

    if len(ir.args) != 2:
        return None
    left, right = ir.args
    if isinstance(left, FieldRef) and isinstance(right, Literal):
        return left.name, right.value, True
    if isinstance(left, Literal) and isinstance(right, FieldRef):
        return right.name, left.value, False
    return None


def _range_selectivity(fn, value, field_left, lo, hi) -> float | None:
    """Linear-interpolation selectivity for ``col <op> value`` over [lo, hi]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if lo is None or hi is None or hi <= lo:
        return None
    if not field_left:  # "value <op> col" == "col <flipped-op> value"
        fn = _RANGE_FLIP[fn]
    span = hi - lo
    if fn in ("less", "less_equal"):
        frac = (value - lo) / span
    else:  # greater / greater_equal
        frac = (hi - value) / span
    return min(1.0, max(0.0, frac))


def estimate_selectivity(ir: IRNode, stats: StatsLookup | None = None) -> float:
    """Estimate the fraction of rows a boolean predicate keeps, in ``(0, 1]``.

    Boolean combinators compose under the standard independence assumption:
    ``AND`` multiplies, ``OR`` uses inclusion-exclusion, ``NOT`` complements.
    Leaf comparisons use column statistics when ``stats`` is supplied —
    ``1/NDV`` for equality, min/max interpolation for ranges, the null
    fraction for ``is_null`` — and fall back to System R constants otherwise.
    """
    from pandas.lazy.ir import (
        Alias,
        Call,
    )

    if isinstance(ir, Alias):
        return estimate_selectivity(ir.arg, stats)
    if not isinstance(ir, Call):
        return SEL_DEFAULT

    fn = ir.function

    # Boolean combinators recurse into their sub-predicates.
    if fn == "and_":
        sel = 1.0
        for arg in ir.args:
            sel *= estimate_selectivity(arg, stats)  # P(A and B) = sA * sB
        return sel
    if fn == "or_":
        sel = 0.0
        for arg in ir.args:
            sa = estimate_selectivity(arg, stats)
            sel = sel + sa - sel * sa  # P(A or B) = sA + sB - sA*sB
        return sel
    if fn == "invert":
        if not ir.args:
            return SEL_DEFAULT
        return max(0.0, 1.0 - estimate_selectivity(ir.args[0], stats))

    if fn == "equal":
        if stats is not None:
            fl = _field_and_literal(ir)
            if fl is not None:
                st = stats(fl[0])
                if st is not None and st.ndv:
                    return min(1.0, 1.0 / st.ndv)
        return SEL_EQ
    if fn == "not_equal":
        if stats is not None:
            fl = _field_and_literal(ir)
            if fl is not None:
                st = stats(fl[0])
                if st is not None and st.ndv:
                    return max(0.0, 1.0 - 1.0 / st.ndv)
        return SEL_NEQ
    if fn in _RANGE_OPS:
        if stats is not None:
            fl = _field_and_literal(ir)
            if fl is not None:
                name, value, field_left = fl
                st = stats(name)
                if st is not None:
                    sel = _range_selectivity(
                        fn, value, field_left, st.min_val, st.max_val
                    )
                    if sel is not None:
                        return sel
        return SEL_RANGE
    if fn == "is_null":
        if stats is not None and len(ir.args) == 1:
            from pandas.lazy.ir import FieldRef

            arg = ir.args[0]
            if isinstance(arg, FieldRef):
                st = stats(arg.name)
                if st is not None and st.null_count is not None and st.row_count:
                    return min(1.0, st.null_count / st.row_count)
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


def compute_column_stats(series, sample_size: int = STATS_SAMPLE_SIZE):
    """Sampled ``ColumnStats`` for an in-memory pandas Series.

    NDV is from a random sample (exact for low cardinality; linearly
    extrapolated when the sample saturates, i.e. a high-cardinality column).
    min/max are exact for NumPy numeric dtypes (cheap, and ranges want the
    true span); the null fraction is sampled.
    """
    import numpy as np

    import pandas as pd

    n = len(series)
    if n == 0:
        return ColumnStats(row_count=0)

    values = series.to_numpy() if hasattr(series, "to_numpy") else np.asarray(series)

    # Random sample (not strided): a stride aliases against periodic or sorted
    # data — e.g. ``arange(n) % 7`` strided by 7 samples only zeros. Indices
    # are de-duplicated so ``sample_n`` counts distinct positions; a unique
    # column then saturates exactly (every position a distinct value), which
    # the extrapolation below relies on. A fixed seed keeps planning
    # deterministic.
    if n <= sample_size:
        sample = values
        approximate = False
    else:
        idx = np.unique(np.random.default_rng(0).integers(0, n, sample_size))
        sample = values[idx]
        approximate = True
    sample_n = len(sample)

    try:
        sample_distinct = len(pd.unique(sample))
    except (TypeError, ValueError):
        return ColumnStats(row_count=n)

    # A saturated sample (distinct ~ sample size) means high cardinality;
    # extrapolate linearly. Otherwise the sample saw every distinct value.
    if sample_distinct >= sample_n * 0.95 and approximate:
        ndv = min(n, int(sample_distinct * n / sample_n))
    else:
        ndv = sample_distinct

    min_val = max_val = None
    if values.dtype.kind in ("i", "u", "f"):
        try:
            min_val = float(np.nanmin(values))
            max_val = float(np.nanmax(values))
        except (ValueError, TypeError):
            pass

    null_count = None
    if values.dtype.kind == "f":
        null_count = int(np.isnan(sample).sum() / sample_n * n)

    return ColumnStats(
        row_count=n,
        ndv=max(1, ndv),
        min_val=min_val,
        max_val=max_val,
        null_count=null_count,
        approximate=approximate,
    )
