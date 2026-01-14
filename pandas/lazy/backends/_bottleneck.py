"""
Optional Bottleneck integration for lazy pandas kernels.

Bottleneck provides fast, NaN-aware rolling window functions written in C.
This module handles the optional import and configuration of Bottleneck
for use in lazy pandas kernel implementations.

Usage
-----
>>> from pandas.lazy.backends._bottleneck import use_bottleneck, bn
>>> if use_bottleneck():
...     result = bn.move_sum(arr, window)
"""

from __future__ import annotations

from pandas._config import get_option

from pandas.compat._optional import import_optional_dependency

# Import bottleneck if available (warns if not installed)
bn = import_optional_dependency("bottleneck", errors="warn")
BOTTLENECK_INSTALLED = bn is not None

# Global state for whether to use Bottleneck
_USE_BOTTLENECK = False


def set_use_bottleneck(v: bool = True) -> None:
    """
    Enable or disable Bottleneck acceleration for lazy kernels.

    This respects the pandas global option `compute.use_bottleneck`.
    When Bottleneck is not installed, this is a no-op.

    Parameters
    ----------
    v : bool, default True
        Whether to enable Bottleneck acceleration.
    """
    global _USE_BOTTLENECK
    if BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v


def use_bottleneck() -> bool:
    """
    Check if Bottleneck should be used for kernel operations.

    Returns
    -------
    bool
        True if Bottleneck is installed and enabled via configuration.
    """
    return _USE_BOTTLENECK and BOTTLENECK_INSTALLED


# Initialize from pandas config
# This uses the same config option as pandas core: compute.use_bottleneck
set_use_bottleneck(get_option("compute.use_bottleneck"))
