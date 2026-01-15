"""
Configuration system for lazy pandas execution thresholds.

This module provides a centralized configuration for all threshold-based
decisions in the lazy execution engine. Thresholds can be:

1. Set programmatically via ThresholdConfig
2. Loaded from a calibration file
3. Accessed via pd.set_option("compute.lazy.*")

The configuration system is designed to work with the calibration benchmark
suite, which can generate optimal thresholds for specific hardware.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
import json
from pathlib import Path
from typing import (
    Any,
)


@dataclass
class ThresholdConfig:
    """
    Configuration for all execution thresholds in lazy pandas.

    This class encapsulates all threshold-based decisions in the execution
    engine. Default values are conservative estimates that work well on
    most hardware. For optimal performance, use the calibration benchmark
    suite to generate hardware-specific thresholds.

    Attributes
    ----------
    filter_arrow_threshold : int
        Number of rows above which Arrow filter is used instead of NumPy.
        Arrow filter is faster for large datasets due to SIMD operations.
        Default: 50,000 rows.

    groupby_arrow_row_threshold : int
        Number of rows above which Arrow groupby is used instead of Pandas.
        Arrow has ~10-15ms JIT overhead, so it's only beneficial for large data.
        Default: 100,000 rows.

    groupby_arrow_cardinality_threshold : int
        Number of groups above which Arrow groupby is preferred.
        Arrow scales better with high cardinality.
        Default: 100 groups.

    parallel_expr_threshold : int
        Minimum number of expressions before parallelizing projections.
        ThreadPoolExecutor has ~1-2ms overhead per submission.
        Default: 8 expressions.

    parallel_chunk_size : int
        Batch size for streaming execution. 64K is L3 cache friendly.
        Default: 65,536 rows.

    numexpr_min_elements : int
        Minimum array elements for NumExpr fusion to be beneficial.
        NumExpr has compilation overhead that must be amortized.
        Default: 100,000 elements.

    numexpr_min_operations : int
        Minimum fusible operations for NumExpr to be beneficial.
        Fusion only helps when there are multiple operations to combine.
        Default: 2 operations.

    numexpr_large_array_threshold : int
        Array size where memory bandwidth becomes the bottleneck.
        Above this, NumExpr benefit increases due to cache efficiency.
        Default: 1,000,000 elements.

    arrow_majority_fraction : float
        Fraction of Arrow columns needed to prefer Arrow backend for
        neutral operations. Used in backend voting.
        Default: 0.5 (50%).

    Examples
    --------
    >>> # Use default configuration
    >>> config = ThresholdConfig()

    >>> # Load calibrated configuration
    >>> config = ThresholdConfig.from_file("calibration_results.json")

    >>> # Custom configuration
    >>> config = ThresholdConfig(
    ...     filter_arrow_threshold=100_000,
    ...     groupby_arrow_row_threshold=50_000,
    ... )
    """

    # Backend selection thresholds
    filter_arrow_threshold: int = 50_000
    groupby_arrow_row_threshold: int = 100_000
    groupby_arrow_cardinality_threshold: int = 100

    # Parallelization thresholds
    parallel_expr_threshold: int = 8
    parallel_chunk_size: int = 65_536

    # NumExpr fusion thresholds
    numexpr_min_elements: int = 100_000
    numexpr_min_operations: int = 2
    numexpr_large_array_threshold: int = 1_000_000

    # NumExpr size factors for benefit estimation
    numexpr_size_factors: dict[str, float] = field(
        default_factory=lambda: {
            "small": 0.5,  # Below numexpr_min_elements
            "medium": 1.0,  # Between min and large thresholds
            "large": 1.2,  # Above numexpr_large_array_threshold
        }
    )

    # Conversion cost thresholds
    arrow_majority_fraction: float = 0.5

    # Metadata (populated when loading from file)
    _calibration_metadata: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def should_use_arrow_filter(self, n_rows: int, n_conditions: int = 1) -> bool:
        """
        Determine if Arrow filter should be used.

        Parameters
        ----------
        n_rows : int
            Number of rows in the data.
        n_conditions : int, default 1
            Number of filter conditions.

        Returns
        -------
        bool
            True if Arrow filter should be used.
        """
        # Multiple conditions lower the threshold (Arrow handles compound
        # predicates more efficiently)
        adjusted_threshold = self.filter_arrow_threshold // max(1, n_conditions)
        return n_rows > adjusted_threshold

    def should_use_arrow_groupby(
        self,
        n_rows: int,
        n_groups: int | None = None,
    ) -> bool:
        """
        Determine if Arrow groupby should be used.

        Arrow groupby has ~10-15ms JIT overhead but scales better than
        Pandas Cython for large data or high cardinality.

        Parameters
        ----------
        n_rows : int
            Number of rows in the data.
        n_groups : int or None, default None
            Estimated number of groups. If None, only row count is used.

        Returns
        -------
        bool
            True if Arrow groupby should be used.
        """
        # Use Arrow if either threshold is exceeded
        if n_rows > self.groupby_arrow_row_threshold:
            return True

        if n_groups is not None and n_groups > self.groupby_arrow_cardinality_threshold:
            # Also require minimum rows to amortize Arrow overhead
            return n_rows > 10_000

        return False

    def should_parallelize_projection(self, n_expressions: int) -> bool:
        """
        Determine if projection should be parallelized.

        Parameters
        ----------
        n_expressions : int
            Number of projection expressions.

        Returns
        -------
        bool
            True if projection should be parallelized.
        """
        return n_expressions >= self.parallel_expr_threshold

    def should_use_numexpr(self, array_size: int, n_operations: int) -> bool:
        """
        Determine if NumExpr fusion should be used.

        Parameters
        ----------
        array_size : int
            Size of the arrays involved.
        n_operations : int
            Number of fusible operations.

        Returns
        -------
        bool
            True if NumExpr should be used.
        """
        return (
            array_size >= self.numexpr_min_elements
            and n_operations >= self.numexpr_min_operations
        )

    def get_numexpr_size_factor(self, array_size: int) -> float:
        """
        Get the NumExpr benefit size factor for a given array size.

        Parameters
        ----------
        array_size : int
            Size of the array.

        Returns
        -------
        float
            Size factor for benefit estimation.
        """
        if array_size < self.numexpr_min_elements:
            return self.numexpr_size_factors.get("small", 0.5)
        elif array_size < self.numexpr_large_array_threshold:
            return self.numexpr_size_factors.get("medium", 1.0)
        else:
            return self.numexpr_size_factors.get("large", 1.2)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary.
        """
        return {
            "filter_arrow_threshold": self.filter_arrow_threshold,
            "groupby_arrow_row_threshold": self.groupby_arrow_row_threshold,
            "groupby_arrow_cardinality_threshold": (
                self.groupby_arrow_cardinality_threshold
            ),
            "parallel_expr_threshold": self.parallel_expr_threshold,
            "parallel_chunk_size": self.parallel_chunk_size,
            "numexpr_min_elements": self.numexpr_min_elements,
            "numexpr_min_operations": self.numexpr_min_operations,
            "numexpr_large_array_threshold": self.numexpr_large_array_threshold,
            "numexpr_size_factors": self.numexpr_size_factors.copy(),
            "arrow_majority_fraction": self.arrow_majority_fraction,
        }

    def to_file(self, path: str | Path) -> None:
        """
        Save configuration to JSON file.

        Parameters
        ----------
        path : str or Path
            Path to save configuration.
        """
        path = Path(path)
        data = {
            "config": self.to_dict(),
            "metadata": self._calibration_metadata,
        }
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdConfig:
        """
        Create configuration from dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary.

        Returns
        -------
        ThresholdConfig
            Configuration object.
        """
        # Extract known fields, ignore unknown ones for forward compatibility
        known_fields = {
            "filter_arrow_threshold",
            "groupby_arrow_row_threshold",
            "groupby_arrow_cardinality_threshold",
            "parallel_expr_threshold",
            "parallel_chunk_size",
            "numexpr_min_elements",
            "numexpr_min_operations",
            "numexpr_large_array_threshold",
            "numexpr_size_factors",
            "arrow_majority_fraction",
        }
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> ThresholdConfig:
        """
        Load configuration from JSON file.

        The file should be in the format produced by the calibration
        benchmark suite or `to_file()`.

        Parameters
        ----------
        path : str or Path
            Path to configuration file.

        Returns
        -------
        ThresholdConfig
            Configuration object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        path = Path(path)
        with path.open() as f:
            data = json.load(f)

        # Handle both raw config and wrapped format
        if "config" in data:
            config_data = data["config"]
            metadata = data.get("metadata", {})
        elif "recommended_config" in data:
            # Calibration suite output format
            config_data = data["recommended_config"]
            metadata = data.get("metadata", {})
        else:
            config_data = data
            metadata = {}

        config = cls.from_dict(config_data)
        config._calibration_metadata = metadata
        return config


# Global default configuration
_default_config: ThresholdConfig | None = None


def _get_config_from_pandas_options() -> ThresholdConfig | None:
    """
    Try to build ThresholdConfig from pandas global options.

    Returns None if pandas options are not available (e.g., during import).
    """
    try:
        from pandas import get_option

        return ThresholdConfig(
            filter_arrow_threshold=get_option("compute.lazy.filter_arrow_threshold"),
            groupby_arrow_row_threshold=get_option(
                "compute.lazy.groupby_arrow_row_threshold"
            ),
            groupby_arrow_cardinality_threshold=get_option(
                "compute.lazy.groupby_arrow_cardinality_threshold"
            ),
            parallel_expr_threshold=get_option("compute.lazy.parallel_expr_threshold"),
            parallel_chunk_size=get_option("compute.lazy.parallel_chunk_size"),
            numexpr_min_elements=get_option("compute.lazy.numexpr_min_elements"),
            numexpr_min_operations=get_option("compute.lazy.numexpr_min_operations"),
            numexpr_large_array_threshold=get_option(
                "compute.lazy.numexpr_large_array_threshold"
            ),
            arrow_majority_fraction=get_option("compute.lazy.arrow_majority_fraction"),
        )
    except (ImportError, KeyError):
        # pandas not fully initialized or options not registered yet
        return None


def get_threshold_config() -> ThresholdConfig:
    """
    Get the current threshold configuration.

    Priority:
    1. If set_threshold_config() was called, use that config
    2. Otherwise, read from pandas global options (pd.get_option)
    3. If pandas options not available, use defaults

    Returns
    -------
    ThresholdConfig
        Current threshold configuration.

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas.lazy.optimize.config import get_threshold_config

    >>> # Default config
    >>> config = get_threshold_config()
    >>> config.filter_arrow_threshold
    50000

    >>> # Change via pandas options
    >>> pd.set_option("compute.lazy.filter_arrow_threshold", 100000)
    >>> config = get_threshold_config()
    >>> config.filter_arrow_threshold
    100000
    """
    # If explicitly set, use that
    if _default_config is not None:
        return _default_config

    # Try to get from pandas options
    config = _get_config_from_pandas_options()
    if config is not None:
        return config

    # Fallback to default ThresholdConfig
    return ThresholdConfig()


def set_threshold_config(config: ThresholdConfig | None) -> None:
    """
    Set the global threshold configuration.

    This overrides pandas options. To use pandas options again,
    call reset_threshold_config().

    Parameters
    ----------
    config : ThresholdConfig or None
        Configuration to use, or None to reset to default.

    See Also
    --------
    reset_threshold_config : Reset to use pandas options.
    get_threshold_config : Get current configuration.

    Examples
    --------
    >>> from pandas.lazy.optimize.config import (
    ...     ThresholdConfig,
    ...     set_threshold_config,
    ...     get_threshold_config,
    ... )

    >>> # Set custom config
    >>> custom = ThresholdConfig(filter_arrow_threshold=100000)
    >>> set_threshold_config(custom)
    >>> get_threshold_config().filter_arrow_threshold
    100000
    """
    global _default_config
    _default_config = config


def reset_threshold_config() -> None:
    """
    Reset the global threshold configuration.

    After reset, get_threshold_config() will read from pandas options
    (pd.get_option("compute.lazy.*")).

    See Also
    --------
    set_threshold_config : Set explicit configuration.
    get_threshold_config : Get current configuration.
    """
    global _default_config
    _default_config = None
