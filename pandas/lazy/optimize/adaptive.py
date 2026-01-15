"""
Adaptive threshold system for lazy pandas.

This module provides an adaptive threshold system that adjusts execution
thresholds based on runtime statistics. The system tracks:

1. Backend performance: Arrow vs NumPy execution times per operation type
2. Cache behavior: Hit/miss rates for data conversion
3. Memory pressure: Whether operations are memory-bound

Based on these statistics, the adaptive system adjusts thresholds to
optimize for the actual workload characteristics.

Design
------
The adaptive system uses an exponential moving average (EMA) to smooth
statistics over time, preventing threshold thrashing while still adapting
to sustained workload changes.

Usage
-----
>>> from pandas.lazy.optimize.adaptive import AdaptiveThresholdManager
>>> manager = AdaptiveThresholdManager()
>>>
>>> # Record execution statistics
>>> manager.record_execution("filter", backend="arrow", rows=100_000, time_ms=5.2)
>>> manager.record_execution("filter", backend="numpy", rows=100_000, time_ms=8.1)
>>>
>>> # Get adapted thresholds
>>> config = manager.get_adapted_config()
>>> config.filter_arrow_threshold  # May be lower if Arrow consistently wins
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import (
    dataclass,
    field,
)
from threading import Lock
import time
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from pandas.lazy.optimize.config import ThresholdConfig


@dataclass
class ExecutionStats:
    """
    Statistics for a single execution event.

    Attributes
    ----------
    operation : str
        The operation type (e.g., "filter", "groupby", "projection").
    backend : str
        The backend used ("arrow" or "numpy").
    rows : int
        Number of rows processed.
    time_ms : float
        Execution time in milliseconds.
    timestamp : float
        Unix timestamp of the execution.
    """

    operation: str
    backend: str
    rows: int
    time_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def throughput(self) -> float:
        """Rows per millisecond."""
        if self.time_ms <= 0:
            return float("inf")
        return self.rows / self.time_ms


@dataclass
class BackendComparison:
    """
    Comparison statistics between backends for an operation.

    Uses exponential moving average (EMA) for smooth adaptation.
    """

    # EMA of throughput (rows/ms) for each backend
    arrow_throughput_ema: float = 0.0
    numpy_throughput_ema: float = 0.0

    # Number of samples collected
    arrow_samples: int = 0
    numpy_samples: int = 0

    # EMA smoothing factor (higher = more responsive to recent data)
    alpha: float = 0.2

    # Last crossover point estimate (rows where backends have equal performance)
    crossover_estimate: int | None = None

    # History of recent crossover observations for confidence
    crossover_history: list[int] = field(default_factory=list)

    def update(self, backend: str, rows: int, time_ms: float) -> None:
        """Update statistics with a new execution sample."""
        throughput = rows / max(time_ms, 0.001)  # Avoid division by zero

        if backend == "arrow":
            if self.arrow_samples == 0:
                self.arrow_throughput_ema = throughput
            else:
                self.arrow_throughput_ema = (
                    self.alpha * throughput
                    + (1 - self.alpha) * self.arrow_throughput_ema
                )
            self.arrow_samples += 1
        else:
            if self.numpy_samples == 0:
                self.numpy_throughput_ema = throughput
            else:
                self.numpy_throughput_ema = (
                    self.alpha * throughput
                    + (1 - self.alpha) * self.numpy_throughput_ema
                )
            self.numpy_samples += 1

        # Update crossover estimate if we have samples from both backends
        self._update_crossover(rows, time_ms, backend)

    def _update_crossover(self, rows: int, time_ms: float, backend: str) -> None:
        """Estimate the crossover point where backends have equal performance."""
        if self.arrow_samples < 3 or self.numpy_samples < 3:
            # Not enough data yet
            return

        # Simple heuristic: if Arrow is faster at current size, crossover is below
        # if NumPy is faster, crossover is above
        throughput = rows / max(time_ms, 0.001)

        if backend == "arrow" and throughput > self.numpy_throughput_ema:
            # Arrow was faster, crossover is at or below this size
            self.crossover_history.append(rows)
        elif backend == "numpy" and throughput > self.arrow_throughput_ema:
            # NumPy was faster, crossover is at or above this size
            self.crossover_history.append(rows)

        # Keep only recent history
        if len(self.crossover_history) > 20:
            self.crossover_history = self.crossover_history[-20:]

        # Estimate crossover as median of recent observations
        if len(self.crossover_history) >= 5:
            sorted_history = sorted(self.crossover_history)
            self.crossover_estimate = sorted_history[len(sorted_history) // 2]

    def get_recommended_threshold(self, default: int) -> int:
        """
        Get the recommended threshold based on collected statistics.

        Parameters
        ----------
        default : int
            Default threshold to use if not enough data.

        Returns
        -------
        int
            Recommended threshold.
        """
        if self.crossover_estimate is not None:
            # Apply some margin (10%) below crossover for safety
            return max(1000, int(self.crossover_estimate * 0.9))

        # Not enough data, use default
        return default

    @property
    def confidence(self) -> float:
        """Confidence level in the crossover estimate (0-1)."""
        min_samples = min(self.arrow_samples, self.numpy_samples)
        if min_samples < 3:
            return 0.0
        if min_samples < 10:
            return 0.3
        if min_samples < 30:
            return 0.6
        return 0.9


@dataclass
class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds based on runtime statistics.

    This class collects execution statistics and adjusts thresholds
    to optimize for the observed workload characteristics.

    Parameters
    ----------
    base_config : ThresholdConfig, optional
        Base configuration to start from. If None, uses defaults.
    adaptation_enabled : bool, default True
        Whether adaptation is enabled. If False, always returns base config.
    min_samples_for_adaptation : int, default 10
        Minimum samples needed before adapting thresholds.

    Examples
    --------
    >>> manager = AdaptiveThresholdManager()
    >>> manager.record_execution("filter", "arrow", 100_000, 5.2)
    >>> manager.record_execution("filter", "numpy", 100_000, 8.1)
    >>> config = manager.get_adapted_config()
    """

    base_config: ThresholdConfig | None = None
    adaptation_enabled: bool = True
    min_samples_for_adaptation: int = 10

    # Per-operation backend comparisons
    _comparisons: dict[str, BackendComparison] = field(default_factory=dict)

    # Recent execution history (for debugging/analysis)
    _history: list[ExecutionStats] = field(default_factory=list)
    _max_history: int = 1000

    # Thread safety
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        """Initialize comparisons for known operations."""
        if not self._comparisons:
            self._comparisons = defaultdict(BackendComparison)

    def record_execution(
        self,
        operation: str,
        backend: str,
        rows: int,
        time_ms: float,
    ) -> None:
        """
        Record an execution event for threshold adaptation.

        Parameters
        ----------
        operation : str
            The operation type ("filter", "groupby", "projection", "numexpr").
        backend : str
            The backend used ("arrow" or "numpy").
        rows : int
            Number of rows processed.
        time_ms : float
            Execution time in milliseconds.
        """
        with self._lock:
            # Create stats object
            stats = ExecutionStats(
                operation=operation,
                backend=backend,
                rows=rows,
                time_ms=time_ms,
            )

            # Update comparison stats
            if operation not in self._comparisons:
                self._comparisons[operation] = BackendComparison()
            self._comparisons[operation].update(backend, rows, time_ms)

            # Add to history
            self._history.append(stats)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

    def get_adapted_config(self) -> ThresholdConfig:
        """
        Get the current adapted threshold configuration.

        Returns a ThresholdConfig with thresholds adjusted based on
        the collected execution statistics.

        Returns
        -------
        ThresholdConfig
            Adapted configuration.
        """
        from pandas.lazy.optimize.config import get_threshold_config

        base = self.base_config or get_threshold_config()

        if not self.adaptation_enabled:
            return base

        with self._lock:
            return self._compute_adapted_config(base)

    def _compute_adapted_config(self, base: ThresholdConfig) -> ThresholdConfig:
        """Compute adapted config from current statistics."""
        from pandas.lazy.optimize.config import ThresholdConfig

        # Start with base values
        filter_threshold = base.filter_arrow_threshold
        groupby_row_threshold = base.groupby_arrow_row_threshold
        parallel_threshold = base.parallel_expr_threshold
        numexpr_threshold = base.numexpr_min_elements

        # Adapt filter threshold
        if "filter" in self._comparisons:
            comp = self._comparisons["filter"]
            if comp.confidence >= 0.3:
                filter_threshold = comp.get_recommended_threshold(filter_threshold)

        # Adapt groupby threshold
        if "groupby" in self._comparisons:
            comp = self._comparisons["groupby"]
            if comp.confidence >= 0.3:
                groupby_row_threshold = comp.get_recommended_threshold(
                    groupby_row_threshold
                )

        # Adapt numexpr threshold
        if "numexpr" in self._comparisons:
            comp = self._comparisons["numexpr"]
            if comp.confidence >= 0.3:
                numexpr_threshold = comp.get_recommended_threshold(numexpr_threshold)

        return ThresholdConfig(
            filter_arrow_threshold=filter_threshold,
            groupby_arrow_row_threshold=groupby_row_threshold,
            groupby_arrow_cardinality_threshold=base.groupby_arrow_cardinality_threshold,
            parallel_expr_threshold=parallel_threshold,
            parallel_chunk_size=base.parallel_chunk_size,
            numexpr_min_elements=numexpr_threshold,
            numexpr_min_operations=base.numexpr_min_operations,
            numexpr_large_array_threshold=base.numexpr_large_array_threshold,
            arrow_majority_fraction=base.arrow_majority_fraction,
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get current statistics for inspection.

        Returns
        -------
        dict
            Statistics including per-operation comparisons and confidence levels.
        """
        with self._lock:
            stats: dict[str, Any] = {
                "total_executions": len(self._history),
                "operations": {},
            }

            for op, comp in self._comparisons.items():
                stats["operations"][op] = {
                    "arrow_samples": comp.arrow_samples,
                    "numpy_samples": comp.numpy_samples,
                    "arrow_throughput_ema": comp.arrow_throughput_ema,
                    "numpy_throughput_ema": comp.numpy_throughput_ema,
                    "crossover_estimate": comp.crossover_estimate,
                    "confidence": comp.confidence,
                }

            return stats

    def reset(self) -> None:
        """Reset all collected statistics."""
        with self._lock:
            self._comparisons.clear()
            self._history.clear()


# Global adaptive manager instance
_adaptive_manager: AdaptiveThresholdManager | None = None


def get_adaptive_manager() -> AdaptiveThresholdManager:
    """
    Get the global adaptive threshold manager.

    Returns
    -------
    AdaptiveThresholdManager
        The global manager instance.
    """
    global _adaptive_manager
    if _adaptive_manager is None:
        _adaptive_manager = AdaptiveThresholdManager()
    return _adaptive_manager


def reset_adaptive_manager() -> None:
    """Reset the global adaptive threshold manager."""
    global _adaptive_manager
    _adaptive_manager = None


def record_execution(
    operation: str,
    backend: str,
    rows: int,
    time_ms: float,
) -> None:
    """
    Record an execution event for adaptive threshold tuning.

    This is a convenience function that records to the global manager.

    Parameters
    ----------
    operation : str
        The operation type ("filter", "groupby", "projection", "numexpr").
    backend : str
        The backend used ("arrow" or "numpy").
    rows : int
        Number of rows processed.
    time_ms : float
        Execution time in milliseconds.
    """
    get_adaptive_manager().record_execution(operation, backend, rows, time_ms)


def get_adaptive_config() -> ThresholdConfig:
    """
    Get the current adapted threshold configuration.

    This is a convenience function that gets config from the global manager.

    Returns
    -------
    ThresholdConfig
        The adapted configuration.
    """
    return get_adaptive_manager().get_adapted_config()
