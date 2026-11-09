"""Tests for pandas.core.util.parallel, the worker count for pandas' thread pools."""

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from pandas.compat import WASM

import pandas as pd
from pandas.core.util import parallel
from pandas.core.util.parallel import max_workers


@pytest.fixture
def threaded(monkeypatch):
    # WASM is fixed at import, so a Pyodide run needs it patched to exercise
    # the threaded branch at all
    monkeypatch.setattr(parallel, "WASM", False)


@pytest.mark.parametrize(
    "cpu_count, available, expected",
    [
        (2, None, 2),  # unconstrained, below the cap -> logical CPU count
        (16, None, 6),  # cap binds
        (16, 1, 1),  # single-CPU container
        (16, 2, 2),  # cgroup/affinity tighter than the cap
        (16, 8, 6),  # allocation looser than the cap -> cap still binds
        (2, 8, 2),  # allocation looser than the machine
    ],
)
def test_combines_cap_and_allocation(
    threaded, monkeypatch, cpu_count, available, expected
):
    # default = min(logical CPUs, available CPUs, cap)
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(parallel, "available_cpu_count", lambda: available)
    with pd.option_context("mode.max_threads", None):
        assert max_workers(6) == expected


def test_cpu_count_unknown(threaded, monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: None)
    monkeypatch.setattr(parallel, "available_cpu_count", lambda: None)
    with pd.option_context("mode.max_threads", None):
        assert max_workers(6) == 1


def test_wasm_is_serial(monkeypatch):
    monkeypatch.setattr(parallel, "WASM", True)
    # WASM stays serial even if a worker count is requested explicitly.
    with pd.option_context("mode.max_threads", 8):
        assert max_workers(6) == 1


def test_max_threads_exceeds_cap(threaded, monkeypatch):
    # The cap and the availability clamp bound the *default* only.  The
    # mode.max_threads docs promise an explicit setting still wins.
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(parallel, "available_cpu_count", lambda: 1)
    with pd.option_context("mode.max_threads", 32):
        assert max_workers(6) == 32


def test_max_threads_wins(threaded):
    with pd.option_context("mode.max_threads", 3):
        assert max_workers(6) == 3


@pytest.mark.parametrize(
    "futures_flag, main_alive, shutting_down",
    [
        (True, True, True),  # a pool thread still busy at exit
        (True, False, True),
        (False, False, False),  # the flag decides whenever it exists
        (None, False, True),  # flag gone: the main-thread check stands in
        (None, True, False),
    ],
)
def test_shutdown_is_serial(
    threaded, monkeypatch, futures_flag, main_alive, shutting_down
):
    # once shutdown has begun, no pool can be started, so even an explicit
    # worker count must give way.  The module's own references are patched,
    # not the stdlib modules, which other threads may be using.
    flag = {} if futures_flag is None else {"_shutdown": futures_flag}
    main = SimpleNamespace(is_alive=lambda: main_alive)
    monkeypatch.setattr(parallel, "_futures_thread", SimpleNamespace(**flag))
    monkeypatch.setattr(
        parallel, "threading", SimpleNamespace(main_thread=lambda: main)
    )
    assert parallel.interpreter_shutting_down() is shutting_down
    with pd.option_context("mode.max_threads", 8):
        assert max_workers(6) == (1 if shutting_down else 8)


def test_not_shutting_down():
    assert not parallel.interpreter_shutting_down()


@pytest.mark.skipif(WASM, reason="Can't start subprocesses in WASM")
@pytest.mark.single_cpu
@pytest.mark.parametrize("where", ["atexit", "late_thread", "busy_pool"])
def test_no_pool_during_interpreter_shutdown(where):
    # All three are past the point where concurrent.futures refuses new work;
    # a pool started there raises "cannot schedule new futures after
    # interpreter shutdown".  In busy_pool the main thread still reports
    # alive.  Checked against the real interpreter rather than mocks, since
    # the claim is about the shutdown sequence.
    code = textwrap.dedent(
        f"""\
        import atexit, threading, time
        from concurrent.futures import ThreadPoolExecutor
        import concurrent.futures.thread as cft
        import pandas as pd
        from pandas.core.util.parallel import max_workers

        def check():
            with pd.option_context("mode.max_threads", 4):
                # first field: whether shutdown had really begun by the check
                print(cft._shutdown, max_workers(4))

        def late():
            # wait for shutdown to begin, rather than sleep and hope
            deadline = time.monotonic() + 30
            while not cft._shutdown and time.monotonic() < deadline:
                time.sleep(0.01)
            check()

        if "{where}" == "atexit":
            atexit.register(check)
        elif "{where}" == "late_thread":
            threading.Thread(target=late).start()
        else:
            # submitted, not waited for: the script ends while it runs
            ThreadPoolExecutor(1).submit(late)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,  # the returncode is asserted below, with stderr shown
        capture_output=True,
        encoding="utf-8",
        timeout=60,  # a hang at exit must fail, not stall the run
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    # last line only: a verbose editable install prints rebuild output first
    assert lines, proc.stderr
    assert lines[-1] == "True 1", proc.stderr
