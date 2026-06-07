"""
The engine cost model (M2, docs/ENGINE_DESIGN.md).

One home for every execution-decision constant, each with its measured
provenance. Two tiers:

1. **Option-backed runtime thresholds** — the ``compute.lazy.*`` keys —
   live in :class:`pandas.lazy.optimize.config.ThresholdConfig` (built
   from ``pd.get_option`` in ``pandas/core/config_init.py``). They keep
   working unchanged; this module is the documented index to them.
2. **Engine constants** below — gathered from their original scattered
   homes (parallel.py, physical.py, numpy/core.py, keycache.py), which
   now import from here. Calibration tunes this module, not scattered
   if-statements.

This module imports nothing from pandas.lazy (it is imported by kernel
modules during registry construction), so it is cycle-free by design.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Morsel execution (engine/parallel.py)
# ---------------------------------------------------------------------------
# Canonical range per Leis et al. 2014 (~100K) and DuckDB (122,880);
# measured flat above ~10K rows. Tunable via the
# ``compute.lazy.morsel_size`` option (None = use this default).
MORSEL_SIZE = 131_072

# Below this, partitioning overhead beats any parallel gain (measured
# June 2026: a 1M-row/score-3 numeric chain regressed 2x when allowed).
MIN_PARALLEL_ROWS = 2 * MORSEL_SIZE

# 4P+4E Apple Silicon measured ~5.5x at 8 threads on compute-bound
# kernels (spike_morsel_scaling.py); more threads add nothing.
MAX_WORKERS = 8


def get_morsel_size() -> int:
    """Morsel size: the ``compute.lazy.morsel_size`` option when set,
    else the module default (which tests may monkeypatch)."""
    try:
        import pandas as pd

        value = pd.get_option("compute.lazy.morsel_size")
    except (ImportError, KeyError, OSError):
        value = None
    if value is None:
        # Late import so monkeypatched module globals are respected
        from pandas.lazy.engine import parallel

        return parallel.MORSEL_SIZE
    return int(value)


# ---------------------------------------------------------------------------
# Parallel kernels (backends/numpy/core.py, physical.py)
# ---------------------------------------------------------------------------
# Below these, thread spawn + merge overhead exceeds the kernel win
# (calibrated on 4P+4E Apple Silicon, June 2026).
PARALLEL_SORT_MIN_ROWS = 500_000
PARALLEL_TAKE_MIN_ROWS = 500_000

# Arrow's table-level multi-key sort beats lexsort only at scale.
ARROW_MULTIKEY_SORT_MIN_ROWS = 1_000_000

# ---------------------------------------------------------------------------
# Key-encoding cache (backends/keycache.py)
# ---------------------------------------------------------------------------
# Dictionary-encoding a string key costs ~8 ms/M rows; acero's
# dictionary groupby saves ~5 ms/M per query. Below this, the bookkeeping
# isn't worth it (measured June 2026: 10M keys encode 84 ms, group 13 ms
# vs 67 ms raw).
MIN_ENCODE_ROWS = 1_000_000
