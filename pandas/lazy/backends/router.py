"""
Backend routing logic for lazy pandas execution.

This module decides which backend (Arrow or NumPy) to use for each
physical node based on:
- Operation types (string ops prefer Arrow)
- Current data formats (minimize conversions)
- User preferences

Future Integration Points
-------------------------
The routing system is designed to be extensible for additional backends:

1. **numexpr** (expression acceleration for large NumPy arrays):
   - Add "numexpr" as a backend option in decide_expr_backend()
   - Auto-dispatch in numpy_kernels.py based on array size (>1M elements)
   - Supported ops: +, -, *, /, **, ==, !=, <, <=, >, >=, &, |, ^

2. **numba** (JIT compilation for window/groupby):
   - Reuse existing kernels from pandas/core/_numba/kernels/
   - Add numba kernels for sliding_sum, sliding_mean, etc.
   - Register via @register_kernel("sum", "numba")

3. **Cython** (pandas/_libs/ algorithms):
   - Wrap existing Cython functions: rank_1d, factorize, groupsort_indexer
   - Add "cython" backend for operations like rank, fill
   - Call directly: from pandas._libs import algos
"""

from typing import Literal

from pandas.lazy.backends.types import ArrayDict

# =============================================================================
# Operation Classifications
# =============================================================================

# Operations that strongly prefer Arrow (significant performance benefit)
# These will override the node's default backend
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
        # Null operations (native Arrow support)
        "is_null",
        "is_not_null",
        "fill_null",
        "coalesce",
    }
)

# Operations that strongly prefer NumPy (when data is already NumPy)
# Currently empty - no operations force NumPy
NUMPY_PREFERRED_OPS: frozenset[str] = frozenset()

# Operations that follow input format (neutral)
NEUTRAL_OPS: frozenset[str] = frozenset(
    {
        # Arithmetic
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "modulo",
        "power",
        "negate",
        "abs",
        # Comparison
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        # Logical
        "and_",
        "or_",
        "invert",
        # Aggregations
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "std",
        "var",
        "first",
        "last",
        "n_unique",
    }
)


# =============================================================================
# Backend Decision Logic
# =============================================================================


def is_string_op(func_name: str) -> bool:
    """Check if function is a string operation."""
    return func_name.startswith("str_")


def is_null_op(func_name: str) -> bool:
    """Check if function is a null-handling operation."""
    return func_name in {"is_null", "is_not_null", "fill_null", "coalesce"}


def should_use_arrow(func_name: str) -> bool:
    """
    Check if an operation should use Arrow backend.

    Parameters
    ----------
    func_name : str
        The function name.

    Returns
    -------
    bool
        True if operation prefers Arrow.
    """
    return func_name in ARROW_PREFERRED_OPS


def should_use_numpy(func_name: str) -> bool:
    """
    Check if an operation should use NumPy backend.

    Parameters
    ----------
    func_name : str
        The function name.

    Returns
    -------
    bool
        True if operation prefers NumPy.
    """
    return func_name in NUMPY_PREFERRED_OPS


def infer_backend_from_arrays(
    arrays: ArrayDict,
) -> Literal["arrow", "numpy"]:
    """
    Infer preferred backend from current array formats.

    Uses majority vote - if most columns are Arrow, prefer Arrow.

    Parameters
    ----------
    arrays : ArrayDict
        Current arrays.

    Returns
    -------
    {"arrow", "numpy"}
        Inferred backend.
    """
    import pyarrow as pa

    from pandas.lazy.backends.types import is_index_col

    # Count formats (excluding index columns)
    arrow_count = 0
    numpy_count = 0

    for name, arr in arrays.items():
        if is_index_col(name):
            continue
        if isinstance(arr, (pa.Array, pa.ChunkedArray)):
            arrow_count += 1
        else:
            numpy_count += 1

    # Prefer Arrow if majority or tie
    return "arrow" if arrow_count >= numpy_count else "numpy"


def decide_expr_backend(
    func_name: str,
    input_backend: Literal["arrow", "numpy", "auto"],
    preferred_backend: Literal["arrow", "numpy", "auto"],
) -> Literal["arrow", "numpy"]:
    """
    Decide backend for a single expression/operation.

    Decision priority:
    1. Arrow-preferred ops always use Arrow
    2. NumPy-preferred ops always use NumPy
    3. User preference (if not "auto")
    4. Follow input format

    Parameters
    ----------
    func_name : str
        The function name.
    input_backend : {"arrow", "numpy", "auto"}
        The format of input data.
    preferred_backend : {"arrow", "numpy", "auto"}
        User's preferred backend.

    Returns
    -------
    {"arrow", "numpy"}
        The backend to use.
    """
    # 1. Check operation preferences (overrides)
    if should_use_arrow(func_name):
        return "arrow"
    if should_use_numpy(func_name):
        return "numpy"

    # 2. Check user preference
    if preferred_backend in ("arrow", "numpy"):
        return preferred_backend

    # 3. Follow input format
    if input_backend in ("arrow", "numpy"):
        return input_backend

    # 4. Default to numpy (most pandas data starts as numpy)
    return "numpy"


def decide_node_backend(
    func_names: list[str],
    input_arrays: ArrayDict,
    preferred_backend: Literal["arrow", "numpy", "auto"],
) -> Literal["arrow", "numpy"]:
    """
    Decide backend for a physical node containing multiple operations.

    Per-node decision: all operations in the node use the same backend,
    except for operations with overrides (string/null ops).

    Parameters
    ----------
    func_names : list of str
        Function names used in this node.
    input_arrays : ArrayDict
        Input arrays to the node.
    preferred_backend : {"arrow", "numpy", "auto"}
        User's preferred backend.

    Returns
    -------
    {"arrow", "numpy"}
        The backend for the node.
    """
    # Check if any operation requires Arrow
    has_arrow_ops = any(should_use_arrow(fn) for fn in func_names)
    has_numpy_ops = any(should_use_numpy(fn) for fn in func_names)

    # If node has Arrow-preferred ops, use Arrow
    if has_arrow_ops and not has_numpy_ops:
        return "arrow"

    # If node has NumPy-preferred ops, use NumPy
    if has_numpy_ops and not has_arrow_ops:
        return "numpy"

    # Mixed or neutral - check user preference
    if preferred_backend in ("arrow", "numpy"):
        return preferred_backend

    # Follow input format
    return infer_backend_from_arrays(input_arrays)


def collect_func_names_from_ir(ir_node) -> list[str]:
    """
    Collect all function names from an IR expression tree.

    Parameters
    ----------
    ir_node : IRNode
        The IR node to traverse.

    Returns
    -------
    list of str
        All function names found in the expression.
    """
    from pandas.lazy.ir import (
        Alias,
        Call,
        Cast,
    )

    func_names = []

    def visit(node):
        if isinstance(node, Call):
            func_names.append(node.function)
            for arg in node.args:
                visit(arg)
        elif isinstance(node, Alias):
            visit(node.arg)
        elif isinstance(node, Cast):
            visit(node.arg)
        # FieldRef and Literal are leaf nodes

    visit(ir_node)
    return func_names
