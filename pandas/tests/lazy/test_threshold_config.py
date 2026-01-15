"""
Tests for the threshold configuration system.
"""

import json
import tempfile

import pytest

from pandas.lazy.optimize.config import (
    ThresholdConfig,
    get_threshold_config,
    reset_threshold_config,
    set_threshold_config,
)


class TestThresholdConfig:
    """Tests for ThresholdConfig class."""

    def test_default_values(self):
        """Test default threshold values."""
        config = ThresholdConfig()

        assert config.filter_arrow_threshold == 50_000
        assert config.groupby_arrow_row_threshold == 100_000
        assert config.groupby_arrow_cardinality_threshold == 100
        assert config.parallel_expr_threshold == 8
        assert config.parallel_chunk_size == 65_536
        assert config.numexpr_min_elements == 100_000
        assert config.numexpr_min_operations == 2
        assert config.numexpr_large_array_threshold == 1_000_000
        assert config.arrow_majority_fraction == 0.5

    def test_custom_values(self):
        """Test custom threshold values."""
        config = ThresholdConfig(
            filter_arrow_threshold=100_000,
            groupby_arrow_row_threshold=200_000,
            parallel_expr_threshold=4,
        )

        assert config.filter_arrow_threshold == 100_000
        assert config.groupby_arrow_row_threshold == 200_000
        assert config.parallel_expr_threshold == 4
        # Other values should be defaults
        assert config.numexpr_min_elements == 100_000

    def test_should_use_arrow_filter(self):
        """Test Arrow filter decision."""
        config = ThresholdConfig(filter_arrow_threshold=50_000)

        # Below threshold
        assert not config.should_use_arrow_filter(10_000)
        assert not config.should_use_arrow_filter(50_000)

        # Above threshold
        assert config.should_use_arrow_filter(50_001)
        assert config.should_use_arrow_filter(100_000)

    def test_should_use_arrow_filter_multiple_conditions(self):
        """Test Arrow filter with multiple conditions lowers threshold."""
        config = ThresholdConfig(filter_arrow_threshold=50_000)

        # With 1 condition: threshold is 50K
        assert not config.should_use_arrow_filter(30_000, n_conditions=1)

        # With 2 conditions: threshold is 25K
        assert config.should_use_arrow_filter(30_000, n_conditions=2)

        # With 5 conditions: threshold is 10K
        assert config.should_use_arrow_filter(15_000, n_conditions=5)

    def test_should_use_arrow_groupby(self):
        """Test Arrow groupby decision."""
        config = ThresholdConfig(
            groupby_arrow_row_threshold=100_000,
            groupby_arrow_cardinality_threshold=100,
        )

        # Below both thresholds
        assert not config.should_use_arrow_groupby(50_000)
        assert not config.should_use_arrow_groupby(50_000, n_groups=50)

        # Above row threshold (groups don't matter)
        assert config.should_use_arrow_groupby(100_001)
        assert config.should_use_arrow_groupby(100_001, n_groups=10)

        # Above cardinality threshold with sufficient rows
        assert config.should_use_arrow_groupby(20_000, n_groups=200)

        # Above cardinality but not enough rows
        assert not config.should_use_arrow_groupby(5_000, n_groups=200)

    def test_should_parallelize_projection(self):
        """Test projection parallelization decision."""
        config = ThresholdConfig(parallel_expr_threshold=8)

        assert not config.should_parallelize_projection(1)
        assert not config.should_parallelize_projection(7)
        assert config.should_parallelize_projection(8)
        assert config.should_parallelize_projection(20)

    def test_should_use_numexpr(self):
        """Test NumExpr fusion decision."""
        config = ThresholdConfig(
            numexpr_min_elements=100_000,
            numexpr_min_operations=2,
        )

        # Need both size and operations
        assert not config.should_use_numexpr(50_000, 5)  # Too small
        assert not config.should_use_numexpr(200_000, 1)  # Too few ops
        assert config.should_use_numexpr(100_000, 2)  # Just right
        assert config.should_use_numexpr(500_000, 5)  # Well above

    def test_get_numexpr_size_factor(self):
        """Test NumExpr size factor calculation."""
        config = ThresholdConfig(
            numexpr_min_elements=100_000,
            numexpr_large_array_threshold=1_000_000,
            numexpr_size_factors={"small": 0.5, "medium": 1.0, "large": 1.2},
        )

        # Small arrays
        assert config.get_numexpr_size_factor(50_000) == 0.5

        # Medium arrays
        assert config.get_numexpr_size_factor(100_000) == 1.0
        assert config.get_numexpr_size_factor(500_000) == 1.0

        # Large arrays
        assert config.get_numexpr_size_factor(1_000_000) == 1.2
        assert config.get_numexpr_size_factor(5_000_000) == 1.2


class TestThresholdConfigSerialization:
    """Tests for config serialization/deserialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ThresholdConfig(
            filter_arrow_threshold=75_000,
            parallel_expr_threshold=10,
        )

        d = config.to_dict()

        assert d["filter_arrow_threshold"] == 75_000
        assert d["parallel_expr_threshold"] == 10
        assert "numexpr_size_factors" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "filter_arrow_threshold": 75_000,
            "groupby_arrow_row_threshold": 150_000,
            "unknown_field": "should be ignored",
        }

        config = ThresholdConfig.from_dict(d)

        assert config.filter_arrow_threshold == 75_000
        assert config.groupby_arrow_row_threshold == 150_000
        # Defaults for unspecified fields
        assert config.parallel_expr_threshold == 8

    def test_to_file_and_from_file(self):
        """Test saving and loading from file."""
        config = ThresholdConfig(
            filter_arrow_threshold=60_000,
            groupby_arrow_row_threshold=80_000,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.to_file(f.name)
            f.flush()

            loaded = ThresholdConfig.from_file(f.name)

        assert loaded.filter_arrow_threshold == 60_000
        assert loaded.groupby_arrow_row_threshold == 80_000

    def test_from_file_calibration_format(self):
        """Test loading from calibration suite output format."""
        calibration_output = {
            "metadata": {
                "timestamp": "2026-01-15T10:30:00Z",
                "hardware": {"cpu": "Test CPU", "cores": 8},
            },
            "recommended_config": {
                "filter_arrow_threshold": 45_000,
                "groupby_arrow_row_threshold": 90_000,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(calibration_output, f)
            f.flush()

            config = ThresholdConfig.from_file(f.name)

        assert config.filter_arrow_threshold == 45_000
        assert config.groupby_arrow_row_threshold == 90_000
        assert config._calibration_metadata["hardware"]["cpu"] == "Test CPU"


class TestGlobalConfig:
    """Tests for global configuration management."""

    def setup_method(self):
        """Reset global config before each test."""
        reset_threshold_config()

    def teardown_method(self):
        """Reset global config after each test."""
        reset_threshold_config()

    def test_get_default_config(self):
        """Test getting default config."""
        config = get_threshold_config()

        assert config is not None
        assert config.filter_arrow_threshold == 50_000

    def test_set_config(self):
        """Test setting custom config."""
        custom = ThresholdConfig(filter_arrow_threshold=75_000)
        set_threshold_config(custom)

        config = get_threshold_config()

        assert config.filter_arrow_threshold == 75_000

    def test_reset_config(self):
        """Test resetting to default."""
        custom = ThresholdConfig(filter_arrow_threshold=75_000)
        set_threshold_config(custom)

        reset_threshold_config()
        config = get_threshold_config()

        assert config.filter_arrow_threshold == 50_000


class TestExecutionContextIntegration:
    """Tests for ExecutionContext integration."""

    def setup_method(self):
        """Reset global config before each test."""
        reset_threshold_config()

    def teardown_method(self):
        """Reset global config after each test."""
        reset_threshold_config()

    def test_context_uses_global_config(self):
        """Test that ExecutionContext uses global config."""
        from pandas.lazy.physical import ExecutionContext

        # Set custom global config
        custom = ThresholdConfig(filter_arrow_threshold=75_000)
        set_threshold_config(custom)

        context = ExecutionContext()

        assert context.threshold_config.filter_arrow_threshold == 75_000

    def test_context_with_explicit_config(self):
        """Test that ExecutionContext can use explicit config."""
        from pandas.lazy.physical import ExecutionContext

        # Set global config
        set_threshold_config(ThresholdConfig(filter_arrow_threshold=75_000))

        # Create context with different config
        explicit = ThresholdConfig(filter_arrow_threshold=25_000)
        context = ExecutionContext(_threshold_config=explicit)

        # Should use explicit config, not global
        assert context.threshold_config.filter_arrow_threshold == 25_000


class TestPandasOptionsIntegration:
    """Tests for integration with pandas global options (pd.set_option)."""

    def setup_method(self):
        """Reset config and options before each test."""
        reset_threshold_config()
        # Reset pandas options to defaults
        import pandas as pd

        pd.reset_option("compute.lazy.filter_arrow_threshold")
        pd.reset_option("compute.lazy.groupby_arrow_row_threshold")
        pd.reset_option("compute.lazy.groupby_arrow_cardinality_threshold")
        pd.reset_option("compute.lazy.parallel_expr_threshold")
        pd.reset_option("compute.lazy.parallel_chunk_size")
        pd.reset_option("compute.lazy.numexpr_min_elements")
        pd.reset_option("compute.lazy.numexpr_min_operations")
        pd.reset_option("compute.lazy.numexpr_large_array_threshold")
        pd.reset_option("compute.lazy.arrow_majority_fraction")

    def teardown_method(self):
        """Reset config and options after each test."""
        reset_threshold_config()
        import pandas as pd

        pd.reset_option("compute.lazy.filter_arrow_threshold")
        pd.reset_option("compute.lazy.groupby_arrow_row_threshold")
        pd.reset_option("compute.lazy.groupby_arrow_cardinality_threshold")
        pd.reset_option("compute.lazy.parallel_expr_threshold")
        pd.reset_option("compute.lazy.parallel_chunk_size")
        pd.reset_option("compute.lazy.numexpr_min_elements")
        pd.reset_option("compute.lazy.numexpr_min_operations")
        pd.reset_option("compute.lazy.numexpr_large_array_threshold")
        pd.reset_option("compute.lazy.arrow_majority_fraction")

    def test_get_option_defaults(self):
        """Test that pandas options have correct defaults."""
        import pandas as pd

        assert pd.get_option("compute.lazy.filter_arrow_threshold") == 50_000
        assert pd.get_option("compute.lazy.groupby_arrow_row_threshold") == 100_000
        assert pd.get_option("compute.lazy.groupby_arrow_cardinality_threshold") == 100
        assert pd.get_option("compute.lazy.parallel_expr_threshold") == 8
        assert pd.get_option("compute.lazy.parallel_chunk_size") == 65_536
        assert pd.get_option("compute.lazy.numexpr_min_elements") == 100_000
        assert pd.get_option("compute.lazy.numexpr_min_operations") == 2
        assert pd.get_option("compute.lazy.numexpr_large_array_threshold") == 1_000_000
        assert pd.get_option("compute.lazy.arrow_majority_fraction") == 0.5

    def test_set_option_updates_threshold_config(self):
        """Test that pd.set_option updates ThresholdConfig."""
        import pandas as pd

        # Change via pandas API
        pd.set_option("compute.lazy.filter_arrow_threshold", 75_000)

        # Should be reflected in get_threshold_config()
        config = get_threshold_config()
        assert config.filter_arrow_threshold == 75_000

    def test_set_multiple_options(self):
        """Test setting multiple options."""
        import pandas as pd

        pd.set_option("compute.lazy.filter_arrow_threshold", 60_000)
        pd.set_option("compute.lazy.groupby_arrow_row_threshold", 200_000)
        pd.set_option("compute.lazy.parallel_expr_threshold", 12)

        config = get_threshold_config()
        assert config.filter_arrow_threshold == 60_000
        assert config.groupby_arrow_row_threshold == 200_000
        assert config.parallel_expr_threshold == 12

    def test_option_context_temporary(self):
        """Test that option_context provides temporary changes."""
        import pandas as pd

        # Default value
        assert get_threshold_config().filter_arrow_threshold == 50_000

        # Temporary change
        with pd.option_context("compute.lazy.filter_arrow_threshold", 99_000):
            assert get_threshold_config().filter_arrow_threshold == 99_000

        # Restored to default
        assert get_threshold_config().filter_arrow_threshold == 50_000

    def test_set_threshold_config_overrides_options(self):
        """Test that set_threshold_config overrides pandas options."""
        import pandas as pd

        # Set via pandas option
        pd.set_option("compute.lazy.filter_arrow_threshold", 80_000)
        assert get_threshold_config().filter_arrow_threshold == 80_000

        # Override with explicit config
        explicit = ThresholdConfig(filter_arrow_threshold=30_000)
        set_threshold_config(explicit)

        # Should use explicit config
        assert get_threshold_config().filter_arrow_threshold == 30_000

        # Reset to use pandas options again
        reset_threshold_config()
        assert get_threshold_config().filter_arrow_threshold == 80_000

    def test_attribute_style_access(self):
        """Test attribute-style access to options."""
        import pandas as pd

        # Read via attribute
        assert pd.options.compute.lazy.filter_arrow_threshold == 50_000

        # Write via attribute
        pd.options.compute.lazy.filter_arrow_threshold = 65_000
        assert get_threshold_config().filter_arrow_threshold == 65_000

    def test_describe_option(self):
        """Test that options have descriptions."""
        import pandas as pd

        desc = pd.describe_option(
            "compute.lazy.filter_arrow_threshold", _print_desc=False
        )
        assert "Arrow filter" in desc
        assert "SIMD" in desc

    def test_invalid_option_value_raises(self):
        """Test that invalid values raise ValueError."""
        import pandas as pd

        with pytest.raises(ValueError, match="Value must be a nonnegative integer"):
            pd.set_option("compute.lazy.filter_arrow_threshold", -100)

        with pytest.raises(ValueError, match="Value must be a nonnegative integer"):
            pd.set_option("compute.lazy.filter_arrow_threshold", "not an int")

    @pytest.mark.parametrize(
        "option_name,test_value",
        [
            ("filter_arrow_threshold", 75_000),
            ("groupby_arrow_row_threshold", 200_000),
            ("groupby_arrow_cardinality_threshold", 150),
            ("parallel_expr_threshold", 12),
            ("parallel_chunk_size", 32_768),
            ("numexpr_min_elements", 50_000),
            ("numexpr_min_operations", 3),
            ("numexpr_large_array_threshold", 500_000),
            ("arrow_majority_fraction", 0.7),
        ],
    )
    def test_all_threshold_options(self, option_name, test_value):
        """Test all threshold options can be set and retrieved."""
        import pandas as pd

        pd.set_option(f"compute.lazy.{option_name}", test_value)
        config = get_threshold_config()
        assert getattr(config, option_name) == test_value
