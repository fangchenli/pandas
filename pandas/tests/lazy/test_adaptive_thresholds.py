"""
Tests for adaptive threshold system.
"""

from pandas.lazy.optimize.adaptive import (
    AdaptiveThresholdManager,
    BackendComparison,
    ExecutionStats,
    get_adaptive_config,
    get_adaptive_manager,
    record_execution,
    reset_adaptive_manager,
)
from pandas.lazy.optimize.config import ThresholdConfig


class TestExecutionStats:
    """Tests for ExecutionStats dataclass."""

    def test_throughput(self):
        """Test throughput calculation."""
        stats = ExecutionStats(
            operation="filter",
            backend="arrow",
            rows=100_000,
            time_ms=10.0,
        )
        assert stats.throughput == 10_000  # 100k rows / 10ms

    def test_throughput_zero_time(self):
        """Test throughput with zero time returns inf."""
        stats = ExecutionStats(
            operation="filter",
            backend="arrow",
            rows=100_000,
            time_ms=0.0,
        )
        assert stats.throughput == float("inf")


class TestBackendComparison:
    """Tests for BackendComparison statistics."""

    def test_initial_state(self):
        """Test initial state is zero samples."""
        comp = BackendComparison()
        assert comp.arrow_samples == 0
        assert comp.numpy_samples == 0
        assert comp.arrow_throughput_ema == 0.0
        assert comp.numpy_throughput_ema == 0.0
        assert comp.confidence == 0.0

    def test_update_arrow(self):
        """Test updating with Arrow execution."""
        comp = BackendComparison()
        comp.update("arrow", rows=100_000, time_ms=10.0)

        assert comp.arrow_samples == 1
        assert comp.arrow_throughput_ema == 10_000  # First sample is exact
        assert comp.numpy_samples == 0

    def test_update_numpy(self):
        """Test updating with NumPy execution."""
        comp = BackendComparison()
        comp.update("numpy", rows=100_000, time_ms=20.0)

        assert comp.numpy_samples == 1
        assert comp.numpy_throughput_ema == 5_000  # 100k / 20ms
        assert comp.arrow_samples == 0

    def test_ema_smoothing(self):
        """Test that EMA smooths over multiple samples."""
        comp = BackendComparison(alpha=0.5)  # 50% weight to new sample

        # First sample: throughput = 10_000
        comp.update("arrow", rows=100_000, time_ms=10.0)
        assert comp.arrow_throughput_ema == 10_000

        # Second sample: throughput = 20_000, EMA = 0.5 * 20k + 0.5 * 10k = 15k
        comp.update("arrow", rows=100_000, time_ms=5.0)
        assert comp.arrow_throughput_ema == 15_000

    def test_confidence_increases_with_samples(self):
        """Test that confidence increases as more samples are collected."""
        comp = BackendComparison()

        # No samples - zero confidence
        assert comp.confidence == 0.0

        # Less than 3 samples each - still zero
        comp.update("arrow", rows=1000, time_ms=1.0)
        comp.update("numpy", rows=1000, time_ms=1.0)
        assert comp.confidence == 0.0

        # 3+ samples each - low confidence
        for _ in range(3):
            comp.update("arrow", rows=1000, time_ms=1.0)
            comp.update("numpy", rows=1000, time_ms=1.0)
        assert comp.confidence == 0.3

    def test_recommended_threshold_default(self):
        """Test recommended threshold returns default when no data."""
        comp = BackendComparison()
        assert comp.get_recommended_threshold(50_000) == 50_000

    def test_recommended_threshold_with_crossover(self):
        """Test recommended threshold when crossover is estimated."""
        comp = BackendComparison()
        comp.crossover_estimate = 100_000

        # Should return 90% of crossover (safety margin)
        threshold = comp.get_recommended_threshold(50_000)
        assert threshold == 90_000


class TestAdaptiveThresholdManager:
    """Tests for AdaptiveThresholdManager."""

    def setup_method(self):
        """Reset manager before each test."""
        reset_adaptive_manager()

    def teardown_method(self):
        """Reset manager after each test."""
        reset_adaptive_manager()

    def test_initial_state(self):
        """Test manager starts with no statistics."""
        manager = AdaptiveThresholdManager()
        stats = manager.get_statistics()
        assert stats["total_executions"] == 0
        assert len(stats["operations"]) == 0

    def test_record_execution(self):
        """Test recording execution events."""
        manager = AdaptiveThresholdManager()
        manager.record_execution("filter", "arrow", 100_000, 5.0)

        stats = manager.get_statistics()
        assert stats["total_executions"] == 1
        assert "filter" in stats["operations"]
        assert stats["operations"]["filter"]["arrow_samples"] == 1

    def test_get_adapted_config_no_data(self):
        """Test adapted config with no statistics returns base."""
        manager = AdaptiveThresholdManager()
        config = manager.get_adapted_config()

        # Should match defaults
        assert config.filter_arrow_threshold == 50_000
        assert config.groupby_arrow_row_threshold == 100_000

    def test_get_adapted_config_with_base(self):
        """Test adapted config uses base config."""
        base = ThresholdConfig(filter_arrow_threshold=75_000)
        manager = AdaptiveThresholdManager(base_config=base)
        config = manager.get_adapted_config()

        # Should use base config values
        assert config.filter_arrow_threshold == 75_000

    def test_adaptation_disabled(self):
        """Test that adaptation can be disabled."""
        base = ThresholdConfig(filter_arrow_threshold=75_000)
        manager = AdaptiveThresholdManager(base_config=base, adaptation_enabled=False)

        # Record some executions
        for _ in range(20):
            manager.record_execution("filter", "arrow", 10_000, 1.0)
            manager.record_execution("filter", "numpy", 10_000, 2.0)

        # Should still return base config
        config = manager.get_adapted_config()
        assert config.filter_arrow_threshold == 75_000

    def test_reset(self):
        """Test resetting statistics."""
        manager = AdaptiveThresholdManager()
        manager.record_execution("filter", "arrow", 100_000, 5.0)

        stats = manager.get_statistics()
        assert stats["total_executions"] == 1

        manager.reset()
        stats = manager.get_statistics()
        assert stats["total_executions"] == 0

    def test_thread_safety(self):
        """Test that manager is thread-safe."""
        import threading

        manager = AdaptiveThresholdManager()
        errors = []

        def record_many():
            try:
                for i in range(100):
                    manager.record_execution("filter", "arrow", 1000 * i, 1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = manager.get_statistics()
        assert stats["total_executions"] == 400


class TestGlobalAdaptiveManager:
    """Tests for global adaptive manager functions."""

    def setup_method(self):
        """Reset manager before each test."""
        reset_adaptive_manager()

    def teardown_method(self):
        """Reset manager after each test."""
        reset_adaptive_manager()

    def test_get_adaptive_manager_singleton(self):
        """Test that get_adaptive_manager returns same instance."""
        m1 = get_adaptive_manager()
        m2 = get_adaptive_manager()
        assert m1 is m2

    def test_record_execution_global(self):
        """Test global record_execution function."""
        record_execution("filter", "arrow", 100_000, 5.0)

        manager = get_adaptive_manager()
        stats = manager.get_statistics()
        assert stats["total_executions"] == 1

    def test_get_adaptive_config_global(self):
        """Test global get_adaptive_config function."""
        config = get_adaptive_config()
        assert isinstance(config, ThresholdConfig)


class TestExecutionContextIntegration:
    """Tests for integration with ExecutionContext."""

    def setup_method(self):
        """Reset manager before each test."""
        reset_adaptive_manager()

    def teardown_method(self):
        """Reset manager after each test."""
        reset_adaptive_manager()

    def test_context_adaptive_disabled_by_default(self):
        """Test that adaptive is disabled by default."""
        from pandas.lazy.physical import ExecutionContext

        context = ExecutionContext()
        assert context.adaptive_thresholds is False

    def test_context_adaptive_enabled(self):
        """Test enabling adaptive thresholds on context."""
        from pandas.lazy.physical import ExecutionContext

        context = ExecutionContext(adaptive_thresholds=True)
        assert context.adaptive_thresholds is True

    def test_context_record_execution_when_enabled(self):
        """Test that context records when adaptive is enabled."""
        from pandas.lazy.physical import ExecutionContext

        context = ExecutionContext(adaptive_thresholds=True)
        context.record_execution("filter", "arrow", 100_000, 5.0)

        stats = get_adaptive_manager().get_statistics()
        assert stats["total_executions"] == 1

    def test_context_record_execution_when_disabled(self):
        """Test that context does not record when adaptive is disabled."""
        from pandas.lazy.physical import ExecutionContext

        context = ExecutionContext(adaptive_thresholds=False)
        context.record_execution("filter", "arrow", 100_000, 5.0)

        stats = get_adaptive_manager().get_statistics()
        assert stats["total_executions"] == 0

    def test_context_uses_adaptive_config(self):
        """Test that context uses adaptive config when enabled."""
        from pandas.lazy.physical import ExecutionContext

        # Record some data to create adapted config
        for _ in range(20):
            record_execution("filter", "arrow", 50_000, 1.0)
            record_execution("filter", "numpy", 50_000, 2.0)

        context = ExecutionContext(adaptive_thresholds=True)
        config = context.threshold_config

        # Should be an adapted config (may differ from default)
        assert isinstance(config, ThresholdConfig)


class TestPandasOptionIntegration:
    """Tests for integration with pandas options."""

    def setup_method(self):
        """Reset adaptive manager and options."""
        import pandas as pd

        reset_adaptive_manager()
        pd.reset_option("compute.lazy.adaptive_thresholds")

    def teardown_method(self):
        """Reset adaptive manager and options."""
        import pandas as pd

        reset_adaptive_manager()
        pd.reset_option("compute.lazy.adaptive_thresholds")

    def test_option_default_false(self):
        """Test adaptive option defaults to False."""
        import pandas as pd

        assert pd.get_option("compute.lazy.adaptive_thresholds") is False

    def test_option_can_be_set(self):
        """Test adaptive option can be set."""
        import pandas as pd

        pd.set_option("compute.lazy.adaptive_thresholds", True)
        assert pd.get_option("compute.lazy.adaptive_thresholds") is True

    def test_describe_option(self):
        """Test option has description."""
        import pandas as pd

        desc = pd.describe_option("compute.lazy.adaptive_thresholds", _print_desc=False)
        assert "adaptive" in desc.lower()
        assert "runtime" in desc.lower()
