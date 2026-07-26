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

# Explicit override for the pandas ``compute.use_bottleneck`` option.
#   None  -> follow the live option value (the default)
#   True  -> force on  (still gated on Bottleneck being installed)
#   False -> force off
# Following the option lazily (rather than snapshotting it at import time)
# keeps use_bottleneck() in sync with pd.set_option("compute.use_bottleneck",
# ...) at runtime, matching how pandas core tracks the same option.
_USE_BOTTLENECK_OVERRIDE: bool | None = None


def set_use_bottleneck(v: bool | None = True) -> None:
    """
    Override Bottleneck acceleration for lazy kernels.

    By default the lazy kernels follow the pandas global option
    ``compute.use_bottleneck``. This sets an explicit override that takes
    precedence over the option (useful for benchmarks and tests); pass
    ``None`` to clear the override and resume following the option. When
    Bottleneck is not installed, acceleration stays off regardless.

    Parameters
    ----------
    v : bool or None, default True
        ``True``/``False`` to force acceleration on/off; ``None`` to follow
        the ``compute.use_bottleneck`` option.
    """
    global _USE_BOTTLENECK_OVERRIDE
    _USE_BOTTLENECK_OVERRIDE = None if v is None else bool(v)


def use_bottleneck() -> bool:
    """
    Check if Bottleneck should be used for kernel operations.

    Returns
    -------
    bool
        True if Bottleneck is installed and enabled -- either forced on via
        :func:`set_use_bottleneck` or (when no override is set) by the live
        ``compute.use_bottleneck`` option.
    """
    if not BOTTLENECK_INSTALLED:
        return False
    if _USE_BOTTLENECK_OVERRIDE is not None:
        return _USE_BOTTLENECK_OVERRIDE
    return bool(get_option("compute.use_bottleneck"))
