"""
NumExpr expression fusion for lazy pandas.

JIT-compiles arithmetic expressions for significant speedups on large arrays.

Benefits:
- Multi-threaded execution (uses all CPU cores)
- Reduced memory bandwidth (fused operations avoid intermediate arrays)
- Cache-friendly computation
- 2-10x speedup for complex arithmetic expressions

Supported ops: +, -, *, /, **, ==, !=, <, <=, >, >=, &, |, ~
NOT supported: //, % (use numpy for these)

Usage
-----
    from pandas.lazy.backends.numexpr_fusion import can_fuse_expression, fuse_expression

    if can_fuse_expression(ir_node):
        result = fuse_expression(ir_node, arrays)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandas.lazy.backends.types import ArrayDict
    from pandas.lazy.ir import IRNode


# Minimum array size for NumExpr to be beneficial (default value)
# Below this threshold, NumPy is faster due to numexpr overhead
# This value can be overridden via ThresholdConfig
MIN_ELEMENTS_FOR_NUMEXPR = 100_000


def get_numexpr_min_elements() -> int:
    """Get the minimum elements threshold, using config if available."""
    try:
        from pandas.lazy.optimize.config import get_threshold_config

        return get_threshold_config().numexpr_min_elements
    except ImportError:
        return MIN_ELEMENTS_FOR_NUMEXPR


# NumPy dtypes supported by NumExpr
NUMEXPR_SUPPORTED_DTYPES = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "bool",
        "complex64",
        "complex128",
    }
)

# IR function names that can be fused with NumExpr
FUSEABLE_OPS = frozenset(
    {
        # Arithmetic
        "add",
        "subtract",
        "multiply",
        "divide",
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
    }
)

# Mapping from IR function names to NumExpr operators
OP_TO_NUMEXPR = {
    # Binary arithmetic
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "power": "**",
    # Unary
    "negate": "-",
    "abs": "abs",
    # Comparison
    "equal": "==",
    "not_equal": "!=",
    "less": "<",
    "less_equal": "<=",
    "greater": ">",
    "greater_equal": ">=",
    # Logical
    "and_": "&",
    "or_": "|",
    "invert": "~",
}


def is_numexpr_available() -> bool:
    """Check if NumExpr is installed and available."""
    try:
        import numexpr

        return numexpr is not None
    except ImportError:
        return False


def can_fuse_expression(node: IRNode, arrays: ArrayDict | None = None) -> bool:
    """
    Check if an IR expression tree can be fused with NumExpr.

    Parameters
    ----------
    node : IRNode
        The root of the expression tree.
    arrays : ArrayDict, optional
        Current arrays to check dtypes and sizes.
        If provided, also checks that arrays meet size threshold.

    Returns
    -------
    bool
        True if the expression can be fused.
    """
    from pandas.lazy.ir import (
        Alias,
        Call,
        FieldRef,
        Literal,
    )

    if not is_numexpr_available():
        return False

    # Count operations and check if all are fuseable
    op_count = 0

    def check_node(n) -> bool:
        nonlocal op_count

        if isinstance(n, FieldRef):
            # Check dtype if arrays provided
            if arrays is not None and n.name in arrays:
                arr = arrays[n.name]
                if isinstance(arr, np.ndarray):
                    if arr.dtype.name not in NUMEXPR_SUPPORTED_DTYPES:
                        return False
                else:
                    # Not a NumPy array (Arrow) - can't fuse
                    return False
            return True

        elif isinstance(n, Literal):
            # Scalars are fine
            return True

        elif isinstance(n, Alias):
            return check_node(n.arg)

        elif isinstance(n, Call):
            if n.function not in FUSEABLE_OPS:
                return False
            op_count += 1
            return all(check_node(arg) for arg in n.args)

        else:
            return False

    if not check_node(node):
        return False

    # Need at least 2 operations to benefit from fusion
    if op_count < 2:
        return False

    # Check array sizes if arrays provided
    if arrays is not None:
        # Get size from first array
        min_elements = get_numexpr_min_elements()
        for arr in arrays.values():
            if isinstance(arr, np.ndarray):
                if len(arr) < min_elements:
                    return False
                break

    return True


def build_numexpr_string(node: IRNode, var_map: dict[str, str]) -> str:
    """
    Build a NumExpr expression string from an IR tree.

    Parameters
    ----------
    node : IRNode
        The root of the expression tree.
    var_map : dict[str, str]
        Mapping from IR FieldRef names to NumExpr variable names.
        Updated in place with new mappings.

    Returns
    -------
    str
        The NumExpr expression string.
    """
    from pandas.lazy.ir import (
        Alias,
        Call,
        FieldRef,
        Literal,
    )

    if isinstance(node, FieldRef):
        if node.name not in var_map:
            # Create a safe variable name (v0, v1, v2, ...)
            var_map[node.name] = f"v{len(var_map)}"
        return var_map[node.name]

    elif isinstance(node, Literal):
        val = node.value
        if isinstance(val, bool):
            return "True" if val else "False"
        elif isinstance(val, (int, float)):
            return repr(val)
        else:
            raise ValueError(f"Unsupported literal type for NumExpr: {type(val)}")

    elif isinstance(node, Alias):
        return build_numexpr_string(node.arg, var_map)

    elif isinstance(node, Call):
        func = node.function
        op = OP_TO_NUMEXPR[func]
        args = node.args

        if len(args) == 1:
            # Unary operation
            arg_str = build_numexpr_string(args[0], var_map)
            if func == "abs":
                return f"abs({arg_str})"
            else:
                return f"({op}{arg_str})"

        elif len(args) == 2:
            # Binary operation
            left_str = build_numexpr_string(args[0], var_map)
            right_str = build_numexpr_string(args[1], var_map)
            return f"({left_str} {op} {right_str})"

        else:
            raise ValueError(f"Unexpected argument count for {func}: {len(args)}")

    else:
        raise ValueError(f"Unsupported node type for NumExpr: {type(node)}")


def fuse_expression(node: IRNode, arrays: ArrayDict) -> np.ndarray:
    """
    Evaluate a fused expression using NumExpr.

    Parameters
    ----------
    node : IRNode
        The root of the expression tree.
    arrays : ArrayDict
        Dictionary mapping column names to NumPy arrays.

    Returns
    -------
    np.ndarray
        The result of the fused expression.

    Raises
    ------
    ImportError
        If NumExpr is not available.
    ValueError
        If the expression cannot be fused.
    """
    import numexpr as ne

    # Build the expression string and collect variable mappings
    var_map: dict[str, str] = {}
    expr_str = build_numexpr_string(node, var_map)

    # Build the local_dict for NumExpr
    local_dict = {}
    for col_name, var_name in var_map.items():
        arr = arrays[col_name]
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Expected NumPy array for {col_name}, got {type(arr)}")
        local_dict[var_name] = arr

    # Execute the fused expression
    return ne.evaluate(expr_str, local_dict=local_dict)


class NumExprEvaluator:
    """
    Expression evaluator that uses NumExpr for fuseable expressions.

    This evaluator attempts to fuse arithmetic/comparison expressions
    into single NumExpr calls, falling back to element-wise evaluation
    for non-fuseable operations.
    """

    def __init__(
        self,
        arrays: ArrayDict,
        min_elements: int = MIN_ELEMENTS_FOR_NUMEXPR,
    ) -> None:
        """
        Initialize the NumExpr evaluator.

        Parameters
        ----------
        arrays : ArrayDict
            Dictionary mapping column names to arrays.
        min_elements : int, default 100_000
            Minimum array size to use NumExpr.
        """
        self._arrays = arrays
        self._min_elements = min_elements

        # Check array sizes
        self._array_size = 0
        for arr in arrays.values():
            if isinstance(arr, np.ndarray):
                self._array_size = len(arr)
                break

    def should_use_numexpr(self, node: IRNode) -> bool:
        """Check if NumExpr should be used for this expression."""
        if self._array_size < self._min_elements:
            return False
        return can_fuse_expression(node, self._arrays)

    def evaluate(self, node: IRNode) -> np.ndarray:
        """
        Evaluate an expression, using NumExpr fusion when beneficial.

        Parameters
        ----------
        node : IRNode
            The expression to evaluate.

        Returns
        -------
        np.ndarray
            The result of the expression.
        """
        if self.should_use_numexpr(node):
            return fuse_expression(node, self._arrays)
        else:
            # Fall back to regular evaluation
            from pandas.lazy.backends.array_eval import ArrayEvaluator

            evaluator = ArrayEvaluator(self._arrays, preferred_backend="numpy")
            result = evaluator.evaluate(node)
            if not isinstance(result, np.ndarray):
                result = np.asarray(result)
            return result


def count_fuseable_ops(node: IRNode) -> int:
    """
    Count the number of fuseable operations in an expression tree.

    Parameters
    ----------
    node : IRNode
        The root of the expression tree.

    Returns
    -------
    int
        Number of fuseable operations.
    """
    from pandas.lazy.ir import (
        Alias,
        Call,
    )

    count = 0

    def visit(n):
        nonlocal count
        if isinstance(n, Call):
            if n.function in FUSEABLE_OPS:
                count += 1
            for arg in n.args:
                visit(arg)
        elif isinstance(n, Alias):
            visit(n.arg)

    visit(node)
    return count


def estimate_fusion_benefit(node: IRNode, array_size: int) -> float:
    """
    Estimate the speedup from fusing an expression.

    This is a rough heuristic based on:
    - Number of operations (more ops = more benefit)
    - Array size (larger arrays = more benefit)

    Parameters
    ----------
    node : IRNode
        The expression to evaluate.
    array_size : int
        Size of the arrays.

    Returns
    -------
    float
        Estimated speedup factor (>1 means fusion is beneficial).
    """
    op_count = count_fuseable_ops(node)

    if op_count < 2:
        return 0.5  # Fusion not beneficial for single ops

    # Base benefit scales with operation count
    # Each fused op saves ~1 array allocation + memory pass
    base_benefit = 1.0 + 0.3 * (op_count - 1)

    # Size scaling - larger arrays benefit more from fusion
    # Use threshold config if available for size factors
    try:
        from pandas.lazy.optimize.config import get_threshold_config

        config = get_threshold_config()
        min_elements = config.numexpr_min_elements
        large_threshold = config.numexpr_large_array_threshold
        size_factor = config.get_numexpr_size_factor(array_size)
    except ImportError:
        min_elements = MIN_ELEMENTS_FOR_NUMEXPR
        large_threshold = 1_000_000
        if array_size < min_elements:
            size_factor = 0.5  # NumExpr overhead dominates
        elif array_size < large_threshold:
            size_factor = 1.0
        else:
            size_factor = 1.2  # Memory bandwidth becomes bottleneck

    return base_benefit * size_factor
