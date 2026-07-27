"""
Backend routing logic for lazy pandas execution.

Decides whether each expression runs on the Arrow or NumPy backend, based on
the operation (some string/null ops strongly prefer Arrow), the user's
preference, and the input array format.
"""

from functools import lru_cache
from typing import Literal

# =============================================================================
# Operation Classifications
# =============================================================================

# Operations that strongly prefer Arrow (significant performance benefit, or
# Arrow-only kernels). These override the input/preference-based choice.
ARROW_PREFERRED_OPS: frozenset[str] = frozenset(
    {
        # String operations (2-10x faster in Arrow)
        "str_lower",
        "str_upper",
        "str_len",
        "str_strip",
        "str_lstrip",
        "str_rstrip",
        "str_contains",
        "str_startswith",
        "str_endswith",
        "str_replace",
        "str_slice",
        # Arrow-only (no NumPy kernel): must prefer Arrow, else NumPy input
        # routes to a missing kernel and raises NotImplementedError.
        "str_reverse",
        # Null operations (native Arrow support)
        "is_null",
        "is_not_null",
        "fill_null",
        "coalesce",
    }
)


# =============================================================================
# Backend Decision Logic
# =============================================================================


@lru_cache(maxsize=128)
def should_use_arrow(func_name: str) -> bool:
    """Whether an operation should always run on the Arrow backend."""
    return func_name in ARROW_PREFERRED_OPS


@lru_cache(maxsize=256)
def decide_expr_backend(
    func_name: str,
    input_backend: Literal["arrow", "numpy", "auto"],
    preferred_backend: Literal["arrow", "numpy", "auto"],
) -> Literal["arrow", "numpy"]:
    """
    Decide the backend for a single expression/operation.

    Decision priority:
    1. Arrow-preferred ops always use Arrow.
    2. User preference (if not "auto").
    3. Follow the input format.
    4. Default to NumPy (most pandas data starts as NumPy).

    Cached with lru_cache since the same (func_name, input_backend,
    preferred_backend) combinations recur during expression evaluation.
    """
    # 1. Operation override.
    if should_use_arrow(func_name):
        return "arrow"

    # 2. User preference.
    if preferred_backend in ("arrow", "numpy"):
        return preferred_backend

    # 3. Follow input format.
    if input_backend in ("arrow", "numpy"):
        return input_backend

    # 4. Default.
    return "numpy"
