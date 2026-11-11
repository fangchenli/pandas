"""Shared helpers for pandas' own thread pools."""

from __future__ import annotations

import concurrent.futures.thread as _futures_thread
import os
import threading
from typing import TYPE_CHECKING

from pandas._config import get_option

from pandas.compat import WASM
from pandas.compat._cpu import available_cpu_count

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import (
        Executor,
        Future,
    )


def interpreter_shutting_down() -> bool:
    """
    Whether interpreter shutdown has begun, so thread pools refuse new work.

    That covers ``atexit`` handlers, threads still running after the main
    thread has exited, and work still running in a ``ThreadPoolExecutor``
    when the script ends.  A pool started then raises ``RuntimeError: cannot
    schedule new futures after interpreter shutdown``.

    Returns
    -------
    bool
        True once shutdown has begun.

    See Also
    --------
    max_workers : Worker count for one parallel operation; 1 during shutdown.

    Examples
    --------
    >>> from pandas.core.util.parallel import interpreter_shutting_down
    >>> interpreter_shutting_down()
    False
    """
    # threading's shutdown first runs concurrent.futures' exit hook, which sets
    # this flag and then waits for busy pool threads, and only afterwards marks
    # the main thread stopped (CPython 3.11 through 3.14).  So the flag covers
    # all three cases, where the main-thread check misses a pool thread still
    # running at exit.  The flag is private: should it ever go, fall back to
    # the public check rather than to never detecting shutdown.
    flag = getattr(_futures_thread, "_shutdown", None)
    if flag is not None:
        return bool(flag)
    return not threading.main_thread().is_alive()


def max_workers(default_cap: int) -> int:
    """
    Number of worker threads for one parallel operation.

    ``mode.max_threads`` takes precedence whenever it is set, in either
    direction: it is how a caller that already parallelizes work -- a Dask or
    joblib worker, say -- makes pandas single-threaded, and how a user raises
    the count past an operation's default cap.  ``threadpoolctl`` cannot do
    either, since it limits native OpenMP and BLAS pools, not Python-level ones.

    Otherwise this is the smallest of the machine's logical CPU count,
    *default_cap*, and the CPUs actually available to the process (CPU affinity
    and cgroup limits, see :func:`pandas.compat._cpu.available_cpu_count`), so
    that a containerised or pinned process does not oversubscribe its
    allocation.

    Parameters
    ----------
    default_cap : int
        Ceiling on the default when ``mode.max_threads`` is unset.  Operations
        differ in how far they scale, so each supplies its own.

    Returns
    -------
    int
        Number of workers; 1 means "run serially".  Always 1, whatever
        ``mode.max_threads`` says, where no thread pool can be started: on
        WebAssembly, and during interpreter shutdown.

    See Also
    --------
    interpreter_shutting_down : Whether thread pools have stopped taking work.
    pandas.compat._cpu.available_cpu_count : CPUs this process may run on.

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas.core.util.parallel import max_workers
    >>> with pd.option_context("mode.max_threads", 1):
    ...     max_workers(8)
    1
    """
    if WASM:
        # WASM cannot spawn threads, regardless of mode.max_threads.
        return 1
    if interpreter_shutting_down():
        # a pool would raise "cannot schedule new futures after interpreter
        # shutdown"
        return 1
    requested = get_option("mode.max_threads")
    if requested is not None:
        return requested
    n_workers = min(os.cpu_count() or 1, default_cap)
    # os.cpu_count() counts the machine's CPUs, not the ones this process may
    # use, so it alone would put default_cap threads on a single-CPU container.
    available = available_cpu_count()
    if available is not None:
        n_workers = min(n_workers, available)
    return n_workers


class PoolRefusal:
    """
    Tell a thread pool refusing work apart from errors raised by the work.

    ``ThreadPoolExecutor.submit`` raises ``RuntimeError`` when the pool cannot
    take work: once interpreter shutdown has begun, which can happen part-way
    through an operation, and when it cannot start a thread ("can't start new
    thread", under a process or thread limit such as a container's pids limit).
    For pandas' own pools, which have no initializer and are not shut down
    while work is submitted, those are the only ``RuntimeError`` it raises, and
    either way the operation can still run serially.

    Submit through :meth:`submit`; in the handler around the pool, fall back to
    serial only if :meth:`is_refusal` says the error is the refusal.  Any other
    ``RuntimeError``, such as one raised by the work itself, is a bug and must
    surface.

    See Also
    --------
    max_workers : Worker count for one parallel operation; 1 during shutdown.

    Examples
    --------
    >>> from concurrent.futures import ThreadPoolExecutor
    >>> from pandas.core.util.parallel import PoolRefusal
    >>> refusal = PoolRefusal()
    >>> try:
    ...     with ThreadPoolExecutor(max_workers=2) as pool:
    ...         total = sum(refusal.submit(pool, abs, x).result() for x in [-1, 2])
    ... except RuntimeError as err:
    ...     if not refusal.is_refusal(err):
    ...         raise
    ...     total = sum(abs(x) for x in [-1, 2])
    >>> total
    3
    """

    def __init__(self) -> None:
        self._error: RuntimeError | None = None

    def submit(
        self, pool: Executor, fn: Callable[..., object], /, *args: object
    ) -> Future[object]:
        """
        ``pool.submit(fn, *args)``, remembering a refusal.

        Parameters
        ----------
        pool : Executor
            The pool to submit to.
        fn : callable
            The work.
        *args
            Arguments for ``fn``.

        Returns
        -------
        Future
            The submitted work.

        Raises
        ------
        RuntimeError
            The pool refused the work; :meth:`is_refusal` is then True for it.
        """
        try:
            return pool.submit(fn, *args)
        except RuntimeError as err:
            self._error = err
            raise

    def is_refusal(self, err: BaseException) -> bool:
        """
        Whether ``err`` is the pool refusing work, not an error from the work.

        Parameters
        ----------
        err : BaseException
            The error caught around the pool.

        Returns
        -------
        bool
            True if ``err`` is the error :meth:`submit` raised.
        """
        return self._error is not None and err is self._error
