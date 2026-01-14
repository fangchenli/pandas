"""
Backend execution system for lazy pandas.

This module provides the kernel registry and dispatch system for
executing operations on Arrow or NumPy arrays.

Usage
-----
Register a kernel:

    from pandas.lazy.backends import register_kernel

    @register_kernel("str_lower", "arrow")
    def arrow_str_lower(arr):
        return pc.utf8_lower(arr)

Dispatch to a kernel:

    from pandas.lazy.backends import dispatch_kernel

    result = dispatch_kernel("str_lower", "arrow", arr)

Extension Points
----------------
The kernel registry supports arbitrary backend names. To add new backends:

1. **numexpr backend** (for large array arithmetic):

    @register_kernel("add", "numexpr")
    def numexpr_add(left, right):
        import numexpr as ne
        return ne.evaluate("left + right")

2. **numba backend** (for window functions):

    @register_kernel("sliding_sum", "numba")
    def numba_sliding_sum(values, start, end, min_periods):
        from pandas.core._numba.kernels import sliding_sum
        return sliding_sum(values, np.float64, start, end, min_periods)

3. **cython backend** (for algorithms):

    @register_kernel("rank", "cython")
    def cython_rank(arr, method="average"):
        from pandas._libs.algos import rank_1d
        return rank_1d(arr, method=method)

The router (router.py) can then be extended to dispatch to these backends
based on operation type, array size, or user preference.
"""

from collections.abc import Callable
from typing import (
    Any,
    Literal,
)

from pandas.lazy.backends.types import (
    ArrayDict,
    ArrayLike,
    PyArrowArray,
)

# =============================================================================
# Kernel Registry
# =============================================================================

# Registry mapping (function_name, backend) -> implementation
_KERNELS: dict[tuple[str, str], Callable[..., Any]] = {}


def register_kernel(func_name: str, backend: Literal["arrow", "numpy"]) -> Callable:
    """
    Decorator to register a kernel implementation.

    Parameters
    ----------
    func_name : str
        The function name (e.g., "str_lower", "add").
    backend : {"arrow", "numpy"}
        The backend this kernel implements.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @register_kernel("str_lower", "arrow")
    ... def arrow_str_lower(arr):
    ...     return pc.utf8_lower(arr)
    """

    def decorator(fn: Callable) -> Callable:
        _KERNELS[(func_name, backend)] = fn
        return fn

    return decorator


def get_kernel(
    func_name: str, backend: Literal["arrow", "numpy"]
) -> Callable[..., Any] | None:
    """
    Get kernel implementation for a function and backend.

    Parameters
    ----------
    func_name : str
        The function name.
    backend : {"arrow", "numpy"}
        The backend.

    Returns
    -------
    Callable or None
        The kernel implementation, or None if not found.
    """
    return _KERNELS.get((func_name, backend))


def has_kernel(func_name: str, backend: Literal["arrow", "numpy"]) -> bool:
    """
    Check if a kernel exists for a function and backend.

    Parameters
    ----------
    func_name : str
        The function name.
    backend : {"arrow", "numpy"}
        The backend.

    Returns
    -------
    bool
        True if kernel exists.
    """
    return (func_name, backend) in _KERNELS


def dispatch_kernel(
    func_name: str,
    backend: Literal["arrow", "numpy"],
    *args: Any,
    **kwargs: Any,
) -> ArrayLike:
    """
    Dispatch to the appropriate kernel implementation.

    Parameters
    ----------
    func_name : str
        The function name.
    backend : {"arrow", "numpy"}
        The backend to use.
    *args : Any
        Positional arguments for the kernel.
    **kwargs : Any
        Keyword arguments for the kernel.

    Returns
    -------
    ArrayLike
        The result of the kernel execution.

    Raises
    ------
    NotImplementedError
        If no kernel exists for the function and backend.
    """
    kernel = get_kernel(func_name, backend)
    if kernel is None:
        raise NotImplementedError(
            f"No {backend} kernel registered for '{func_name}'. "
            f"Available kernels: {list_kernels(backend)}"
        )
    return kernel(*args, **kwargs)


def list_kernels(backend: Literal["arrow", "numpy"] | None = None) -> list[str]:
    """
    List all registered kernel function names.

    Parameters
    ----------
    backend : {"arrow", "numpy"} or None
        Filter by backend. If None, list all.

    Returns
    -------
    list of str
        Function names with registered kernels.
    """
    if backend is None:
        return sorted({name for name, _ in _KERNELS.keys()})
    return sorted(name for name, b in _KERNELS.keys() if b == backend)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ArrayDict",
    "ArrayLike",
    "PyArrowArray",
    "dispatch_kernel",
    "get_kernel",
    "has_kernel",
    "list_kernels",
    "register_kernel",
]


# Import kernel modules to register the kernels
# This must be done after the registry functions are defined
from pandas.lazy.backends import (  # noqa: F401
    arrow,
    numpy,
)
