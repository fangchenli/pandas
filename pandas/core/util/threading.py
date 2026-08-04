"""Common utilities for pandas' own thread pools."""

from __future__ import annotations

import os

from pandas._config import get_option

from pandas.compat import WASM
from pandas.compat._cpu import available_cpu_count


def available_cpus() -> int:
    """
    Number of CPUs this process may actually run on.

    ``os.cpu_count`` reports the CPUs the *machine* has, which is not the same
    thing: a process given a slice of them -- by a CPU affinity mask, or by the
    cgroup quota that ``docker run --cpus`` and Kubernetes CPU limits set --
    would otherwise size its thread pools for hardware it cannot use.
    :func:`pandas.compat._cpu.available_cpu_count` takes the tighter of those
    two, and returns None where neither is probed, which in practice means
    macOS and Windows; the machine count is the best answer there.

    Returns
    -------
    int
        At least 1.

    See Also
    --------
    max_workers : Worker count for a pool, honouring ``mode.max_threads``.

    Examples
    --------
    >>> from pandas.core.util.threading import available_cpus
    >>> available_cpus() >= 1
    True
    """
    return available_cpu_count() or os.cpu_count() or 1


def max_workers(default_cap: int) -> int:
    """
    Upper bound on worker threads for one parallel operation.

    ``mode.max_threads`` takes precedence when set, which is how a caller that
    already parallelizes work -- a Dask or joblib worker, say -- makes pandas
    single-threaded.  ``threadpoolctl`` cannot do that: it limits native OpenMP
    and BLAS pools, not Python-level ones.

    Parameters
    ----------
    default_cap : int
        Ceiling to apply when ``mode.max_threads`` is unset.  Operations differ
        in how far they scale, so each supplies its own.

    Returns
    -------
    int
        Number of workers, 1 meaning "run serially".  Always 1 under WASM,
        which cannot start threads.

    See Also
    --------
    available_cpus : CPUs this process may actually run on.

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas.core.util.threading import max_workers
    >>> with pd.option_context("mode.max_threads", 1):
    ...     max_workers(8)
    1
    """
    if WASM:
        # WASM cannot spawn threads, regardless of mode.max_threads.
        return 1
    requested = get_option("mode.max_threads")
    if requested is not None:
        return requested
    return min(available_cpus(), default_cap)
