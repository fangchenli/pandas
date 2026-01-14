"""
Memory pooling for lazy pandas NumPy operations.

This module provides array memory pooling to reduce allocation overhead
during physical plan execution. For operations that support in-place
computation (via numpy's out= parameter), pooled arrays can provide
significant speedups by avoiding repeated memory allocation.

NOTE: This pooling is for NumPy arrays only. PyArrow has its own memory
management system through memory pools (pa.default_memory_pool(),
pa.jemalloc_memory_pool(), pa.mimalloc_memory_pool(), etc.) which handle
Arrow array allocations automatically.

Two pool types are provided:

1. ScratchBufferPool (Recommended for expression evaluation)
   - Pre-allocated rotating buffers for expression chains
   - No explicit release needed - buffers rotate automatically
   - ~3x speedup for arithmetic expressions
   - Used internally by ArrayEvaluator for NumPy operations

2. ArrayPool (General purpose)
   - Acquire/release semantics for explicit control
   - Good for single operations where you control the lifecycle
   - Requires explicit release() to return arrays to pool

Benefits:
- Reduces malloc/free overhead for repeated operations
- Improves cache locality by reusing memory regions
- Can reduce peak memory usage by reusing intermediate buffers

Usage
-----
ScratchBufferPool (for expression chains):
    from pandas.lazy.backends.memory_pool import ScratchBufferPool

    pool = ScratchBufferPool(size=1_000_000, dtype=np.float64, num_buffers=2)
    buf1 = pool.get_next()
    np.add(a, b, out=buf1)
    buf2 = pool.get_next()  # Returns different buffer
    np.multiply(buf1, c, out=buf2)
    # buf1 is now safe to reuse

ArrayPool (explicit acquire/release):
    from pandas.lazy.backends.memory_pool import ArrayPool

    pool = ArrayPool()
    arr = pool.acquire(1_000_000, np.float64)
    # ... use arr ...
    pool.release(arr)  # Return to pool for reuse
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Pooling Strategy Configuration
# =============================================================================


class PoolingStrategy(Enum):
    """
    Strategy for NumPy array memory pooling.

    This enum allows configuration of how memory pooling is handled
    for NumPy operations in the lazy evaluation backend.

    Note: Arrow operations use PyArrow's built-in memory pools
    (pa.default_memory_pool(), pa.jemalloc_memory_pool(), etc.)
    and are not affected by this setting.

    Attributes
    ----------
    NONE : No pooling
        Allocate new arrays for every operation. Simple but slower
        for repeated operations on large arrays.

    SCRATCH : Rotating scratch buffers (default, recommended)
        Uses pre-allocated rotating buffers for expression chains.
        Provides ~3x speedup for arithmetic expressions.
        Best for expression evaluation where intermediate results
        are immediately consumed.

    ACQUIRE_RELEASE : Explicit acquire/release pooling
        Maintains a pool of arrays by (size, dtype) buckets.
        Requires explicit release() calls to return arrays to pool.
        Good for manual control over array lifecycle.
    """

    NONE = "none"
    SCRATCH = "scratch"
    ACQUIRE_RELEASE = "acquire_release"


# Default pooling strategy
DEFAULT_POOLING_STRATEGY = PoolingStrategy.SCRATCH


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


# Minimum array size to benefit from pooling
# Below this, allocation overhead is negligible
MIN_POOL_SIZE = 10_000

# Maximum arrays to keep per (size, dtype) bucket
MAX_POOL_PER_BUCKET = 4

# Maximum total memory to hold in the pool (bytes)
MAX_POOL_MEMORY = 256 * 1024 * 1024  # 256 MB


class ArrayPool:
    """
    Pool for reusable NumPy arrays.

    Maintains a pool of arrays organized by (size, dtype) buckets.
    Arrays can be acquired from the pool and released back when no longer needed.

    This pool is NOT thread-safe. Each execution context should have its own pool.
    """

    def __init__(
        self,
        max_memory: int = MAX_POOL_MEMORY,
        max_per_bucket: int = MAX_POOL_PER_BUCKET,
    ) -> None:
        """
        Initialize the array pool.

        Parameters
        ----------
        max_memory : int, default 256 MB
            Maximum total memory to keep in pool.
        max_per_bucket : int, default 4
            Maximum arrays per (size, dtype) bucket.
        """
        self._max_memory = max_memory
        self._max_per_bucket = max_per_bucket

        # Pool organized by (size, dtype_str) -> list of arrays
        self._pool: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)

        # Track total memory in pool
        self._total_bytes = 0

        # Statistics
        self._hits = 0
        self._misses = 0
        self._releases = 0

    def acquire(self, size: int, dtype: np.dtype | type = np.float64) -> np.ndarray:
        """
        Acquire an array from the pool or allocate a new one.

        Parameters
        ----------
        size : int
            Number of elements needed.
        dtype : dtype or type, default np.float64
            Data type of the array.

        Returns
        -------
        np.ndarray
            An uninitialized array of the requested size and dtype.
            Contents are undefined - caller must fill before reading.
        """
        dtype = np.dtype(dtype)
        key = (size, dtype.str)

        bucket = self._pool.get(key)
        if bucket:
            arr = bucket.pop()
            self._total_bytes -= arr.nbytes
            self._hits += 1
            return arr

        self._misses += 1
        return np.empty(size, dtype=dtype)

    def release(self, arr: np.ndarray) -> None:
        """
        Release an array back to the pool.

        Parameters
        ----------
        arr : np.ndarray
            Array to return to the pool. Must be a contiguous array.
        """
        # Skip small arrays - allocation overhead is negligible
        if len(arr) < MIN_POOL_SIZE:
            return

        # Skip non-contiguous arrays
        if not arr.flags.c_contiguous:
            return

        key = (len(arr), arr.dtype.str)
        bucket = self._pool[key]

        # Check bucket limit
        if len(bucket) >= self._max_per_bucket:
            return

        # Check total memory limit
        if self._total_bytes + arr.nbytes > self._max_memory:
            # Could implement LRU eviction here, but for now just skip
            return

        bucket.append(arr)
        self._total_bytes += arr.nbytes
        self._releases += 1

    @contextmanager
    def get_buffer(
        self, size: int, dtype: np.dtype | type = np.float64
    ) -> Generator[np.ndarray]:
        """
        Context manager for temporary array usage.

        Acquires an array and automatically releases it when done.

        Parameters
        ----------
        size : int
            Number of elements needed.
        dtype : dtype or type, default np.float64
            Data type of the array.

        Yields
        ------
        np.ndarray
            Temporary array that will be returned to pool on exit.

        Examples
        --------
        >>> pool = ArrayPool()
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([4, 5, 6])
        >>> with pool.get_buffer(3, np.float64) as tmp:
        ...     np.add(a, b, out=tmp)
        ...     result = tmp * 2  # Can use tmp within context
        """
        arr = self.acquire(size, dtype)
        try:
            yield arr
        finally:
            self.release(arr)

    def clear(self) -> None:
        """Clear all pooled arrays."""
        self._pool.clear()
        self._total_bytes = 0

    @property
    def stats(self) -> dict[str, int]:
        """
        Get pool statistics.

        Returns
        -------
        dict
            Statistics including hits, misses, releases, and memory usage.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "releases": self._releases,
            "total_bytes": self._total_bytes,
            "num_buckets": len(self._pool),
            "num_arrays": sum(len(b) for b in self._pool.values()),
        }

    @property
    def hit_rate(self) -> float:
        """
        Get the pool hit rate.

        Returns
        -------
        float
            Fraction of acquisitions served from pool (0.0 to 1.0).
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"ArrayPool("
            f"arrays={stats['num_arrays']}, "
            f"memory={stats['total_bytes'] / 1024 / 1024:.1f}MB, "
            f"hit_rate={self.hit_rate:.1%})"
        )


def evaluate_with_pool(
    pool: ArrayPool,
    op: str,
    *arrays: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Evaluate a NumPy operation using pooled output array.

    Parameters
    ----------
    pool : ArrayPool
        Array pool to use for output allocation.
    op : str
        NumPy ufunc name (e.g., "add", "multiply", "divide").
    *arrays : np.ndarray
        Input arrays for the operation.
    out : np.ndarray or None, optional
        Pre-allocated output array. If None, acquires from pool.

    Returns
    -------
    np.ndarray
        Result of the operation.

    Examples
    --------
    >>> pool = ArrayPool()
    >>> a = np.array([1.0, 2.0, 3.0])
    >>> b = np.array([4.0, 5.0, 6.0])
    >>> result = evaluate_with_pool(pool, "add", a, b)
    """
    ufunc = getattr(np, op)

    if out is None:
        # Determine output size and dtype
        size = len(arrays[0])
        # Most operations preserve dtype of first float input
        dtype = np.result_type(*arrays)
        out = pool.acquire(size, dtype)

    return ufunc(*arrays, out=out)


# Operations that support out= parameter
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


# =============================================================================
# Scratch Buffer Pool - Optimized for Expression Evaluation
# =============================================================================


class ScratchBufferPool:
    """
    Scratch buffer pool optimized for expression tree evaluation.

    Unlike ArrayPool which requires explicit release(), ScratchBufferPool
    uses a rotating set of scratch buffers that are automatically reused.
    This is ideal for expression evaluation where intermediate results
    are immediately consumed by the next operation.

    The key insight: for a chain like (a+b)*c/d, we only need 2 buffers:
    - Buffer A: holds result of a+b
    - Buffer B: holds result of (a+b)*c, Buffer A can be reused
    - Buffer A: holds result of (a+b)*c/d, Buffer B can be reused

    Usage
    -----
    pool = ScratchBufferPool(size=1_000_000, dtype=np.float64, num_buffers=2)

    # Evaluate (a + b) * c / d using alternating buffers
    buf = pool.get_next()
    np.add(a, b, out=buf)

    buf2 = pool.get_next()  # Returns different buffer
    np.multiply(buf, c, out=buf2)

    buf3 = pool.get_next()  # Returns buf (recycled)
    np.divide(buf2, d, out=buf3)

    result = buf3.copy()  # Copy final result if needed
    """

    def __init__(
        self,
        size: int,
        dtype: np.dtype | type = np.float64,
        num_buffers: int = 2,
    ) -> None:
        """
        Initialize scratch buffer pool.

        Parameters
        ----------
        size : int
            Size of each buffer (number of elements).
        dtype : dtype or type, default np.float64
            Data type for buffers.
        num_buffers : int, default 2
            Number of scratch buffers. 2 is optimal for linear chains.
            Use 3+ for more complex expression trees.
        """
        self._size = size
        self._dtype = np.dtype(dtype)
        self._num_buffers = num_buffers

        # Pre-allocate all buffers
        self._buffers = [np.empty(size, dtype=self._dtype) for _ in range(num_buffers)]
        self._current_idx = 0

        # Statistics
        self._uses = 0

    def get_next(self) -> np.ndarray:
        """
        Get the next scratch buffer.

        Returns a buffer that was least recently used, allowing
        the previous operation's output to be consumed before
        its buffer is recycled.

        Returns
        -------
        np.ndarray
            A scratch buffer. Contents are undefined.
        """
        buf = self._buffers[self._current_idx]
        self._current_idx = (self._current_idx + 1) % self._num_buffers
        self._uses += 1
        return buf

    def get_buffer(self, idx: int) -> np.ndarray:
        """
        Get a specific buffer by index.

        Parameters
        ----------
        idx : int
            Buffer index (0 to num_buffers-1).

        Returns
        -------
        np.ndarray
            The requested buffer.
        """
        return self._buffers[idx % self._num_buffers]

    @property
    def size(self) -> int:
        """Return buffer size."""
        return self._size

    @property
    def dtype(self) -> np.dtype:
        """Return buffer dtype."""
        return self._dtype

    @property
    def num_buffers(self) -> int:
        """Return number of buffers."""
        return self._num_buffers

    @property
    def memory_bytes(self) -> int:
        """Return total memory used by all buffers."""
        return self._size * self._dtype.itemsize * self._num_buffers

    @property
    def uses(self) -> int:
        """Return total number of buffer uses."""
        return self._uses

    def reset(self) -> None:
        """Reset buffer index to start (doesn't clear memory)."""
        self._current_idx = 0

    def __repr__(self) -> str:
        mb = self.memory_bytes / 1024 / 1024
        return (
            f"ScratchBufferPool(size={self._size:,}, dtype={self._dtype}, "
            f"buffers={self._num_buffers}, memory={mb:.1f}MB, uses={self._uses})"
        )


def evaluate_expression_chain(
    ops: list[tuple[str, list[np.ndarray]]],
    pool: ScratchBufferPool | None = None,
) -> np.ndarray:
    """
    Evaluate a chain of operations using scratch buffers.

    This is optimized for linear expression chains where each operation's
    output becomes the input to the next operation.

    Parameters
    ----------
    ops : list of (op_name, args)
        List of operations to apply. Each is (ufunc_name, [arg1, arg2, ...]).
        Use None as a placeholder for the previous result.
    pool : ScratchBufferPool, optional
        Scratch buffer pool. If None, creates one automatically.

    Returns
    -------
    np.ndarray
        Final result (reference to a pool buffer - copy if needed).

    Examples
    --------
    >>> a = np.array([1.0, 2.0, 3.0])
    >>> b = np.array([4.0, 5.0, 6.0])
    >>> c = np.array([2.0, 2.0, 2.0])
    >>> # Compute (a + b) * c
    >>> ops = [
    ...     ("add", [a, b]),  # t1 = a + b
    ...     ("multiply", [None, c]),  # t2 = t1 * c (None = previous result)
    ... ]
    >>> result = evaluate_expression_chain(ops)
    """
    if not ops:
        raise ValueError("ops list cannot be empty")

    # Determine size from first operation's arrays
    size = None
    dtype = np.float64
    for _, args in ops:
        for arg in args:
            if isinstance(arg, np.ndarray):
                size = len(arg)
                dtype = arg.dtype
                break
        if size is not None:
            break

    if size is None:
        raise ValueError("No arrays found in operations")

    # Create pool if needed
    if pool is None:
        pool = ScratchBufferPool(size, dtype, num_buffers=2)

    prev_result = None

    for op_name, args in ops:
        ufunc = getattr(np, op_name)

        # Replace None with previous result
        actual_args = [prev_result if arg is None else arg for arg in args]

        # Get next scratch buffer for output
        out = pool.get_next()

        # Execute operation
        prev_result = ufunc(*actual_args, out=out)

    return prev_result
