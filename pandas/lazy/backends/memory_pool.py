"""
Memory-pool configuration for lazy pandas execution.

Provides the NumPy pooling-strategy enum, the Arrow memory-pool backend
selector, and the classifier for which elementwise ops can write into a
reusable ``out=`` buffer -- all consumed by ``ArrayEvaluator``.

NOTE: this only *selects* pooling behavior. Arrow array allocation is handled
by PyArrow's own memory pools (default/mimalloc/jemalloc/system); the NumPy
"scratch" strategy reuses a freshly-computed intermediate as the output buffer
in place (see ArrayEvaluator), avoiding an allocation per elementwise step.
"""

from __future__ import annotations

from enum import Enum

# =============================================================================
# Pooling Strategy Configuration
# =============================================================================


class PoolingStrategy(Enum):
    """
    Strategy for NumPy array output buffers during expression evaluation.

    Arrow operations use PyArrow's built-in memory pools and are unaffected.

    Attributes
    ----------
    NONE : Allocate a new array for every operation.
    SCRATCH : Reuse an ephemeral intermediate as the output buffer in place
        (the default) -- avoids an allocation per elementwise step in a chain.
    ACQUIRE_RELEASE : Reserved for an explicit acquire/release pool.
    """

    NONE = "none"
    SCRATCH = "scratch"
    ACQUIRE_RELEASE = "acquire_release"


class ArrowPoolBackend(Enum):
    """
    Arrow memory pool backend selection.

    PyArrow provides several memory allocator backends. The choice affects
    allocation speed and memory fragmentation characteristics.

    Attributes
    ----------
    DEFAULT : Use PyArrow's default pool
        Usually mimalloc on systems where it's available.
        Best for most workloads.

    MIMALLOC : Microsoft's mimalloc allocator
        Fast, low fragmentation. Default on most systems.
        Best for general-purpose workloads.

    JEMALLOC : Facebook's jemalloc allocator
        Good for long-running processes with varied allocation patterns.
        May have higher overhead for short-lived operations.

    SYSTEM : System malloc
        Uses the system's default allocator (malloc/free).
        Baseline for comparison; usually slower than specialized allocators.
    """

    DEFAULT = "default"
    MIMALLOC = "mimalloc"
    JEMALLOC = "jemalloc"
    SYSTEM = "system"


def get_arrow_memory_pool(backend: ArrowPoolBackend | str = ArrowPoolBackend.DEFAULT):
    """
    Get a PyArrow memory pool for the specified backend.

    Parameters
    ----------
    backend : ArrowPoolBackend or str, default DEFAULT
        Which memory pool backend to use.

    Returns
    -------
    pyarrow.MemoryPool
        The memory pool instance.

    Examples
    --------
    >>> pool = get_arrow_memory_pool("mimalloc")
    >>> pool.backend_name
    'mimalloc'
    """
    import pyarrow as pa

    if isinstance(backend, str):
        backend = ArrowPoolBackend(backend)

    if backend == ArrowPoolBackend.DEFAULT:
        return pa.default_memory_pool()
    elif backend == ArrowPoolBackend.MIMALLOC:
        return pa.mimalloc_memory_pool()
    elif backend == ArrowPoolBackend.JEMALLOC:
        return pa.jemalloc_memory_pool()
    elif backend == ArrowPoolBackend.SYSTEM:
        return pa.system_memory_pool()
    else:
        raise ValueError(f"Unknown Arrow pool backend: {backend}")


# =============================================================================
# Pooled-output classifier
# =============================================================================

# Elementwise NumPy ufuncs that accept an out= buffer (so a scratch/ephemeral
# array can hold the result).
POOLABLE_OPS = frozenset(
    {
        # Arithmetic
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "power",
        "mod",
        "negative",
        "absolute",
        # Comparison (result is bool)
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        # Logical
        "logical_and",
        "logical_or",
        "logical_not",
        "logical_xor",
        # Math
        "sqrt",
        "exp",
        "log",
        "log10",
        "sin",
        "cos",
        "tan",
    }
)


def can_use_pooled_output(func_name: str) -> bool:
    """
    Check if a function supports pooled output arrays.

    Parameters
    ----------
    func_name : str
        IR function name.

    Returns
    -------
    bool
        True if the function supports out= parameter.
    """
    # Map IR names to numpy names
    ir_to_numpy = {
        "add": "add",
        "subtract": "subtract",
        "multiply": "multiply",
        "divide": "divide",
        "floor_divide": "floor_divide",
        "power": "power",
        "modulo": "mod",
        "negate": "negative",
        "abs": "absolute",
        "equal": "equal",
        "not_equal": "not_equal",
        "less": "less",
        "less_equal": "less_equal",
        "greater": "greater",
        "greater_equal": "greater_equal",
        "and_": "logical_and",
        "or_": "logical_or",
        "invert": "logical_not",
    }

    np_name = ir_to_numpy.get(func_name, func_name)
    return np_name in POOLABLE_OPS
