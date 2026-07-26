"""
Tests for lazy pandas NumPy miscellaneous kernels.

Tests rolling window operations, shift/lag/lead, and fill NA variants.
"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestRollingSum:
    """Tests for numpy_rolling_sum kernel."""

    def test_basic_rolling_sum(self):
        """Test basic rolling sum."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_sum

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_rolling_sum(arr, window=3)

        # First 2 are NaN (min_periods = window)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 6.0  # 1+2+3
        assert result[3] == 9.0  # 2+3+4
        assert result[4] == 12.0  # 3+4+5

    def test_rolling_sum_with_min_periods(self):
        """Test rolling sum with custom min_periods."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_sum

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_rolling_sum(arr, window=3, min_periods=1)

        # All values should be valid with min_periods=1
        assert result[0] == 1.0
        assert result[1] == 3.0  # 1+2
        assert result[2] == 6.0  # 1+2+3
        assert result[3] == 9.0
        assert result[4] == 12.0

    def test_rolling_sum_with_nan(self):
        """Test rolling sum with NaN values."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_sum

        arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = numpy_rolling_sum(arr, window=3, min_periods=2)

        # Should handle NaN properly
        assert result[2] == 4.0  # 1+3 (nan ignored)

    def test_rolling_sum_empty_array(self):
        """Test rolling sum on empty array."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_sum

        arr = np.array([])
        result = numpy_rolling_sum(arr, window=3)
        assert len(result) == 0

    def test_rolling_sum_window_zero(self):
        """Test rolling sum with zero window."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_sum

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_rolling_sum(arr, window=0)
        assert np.all(np.isnan(result))


class TestRollingMean:
    """Tests for numpy_rolling_mean kernel."""

    def test_basic_rolling_mean(self):
        """Test basic rolling mean."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_mean

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_rolling_mean(arr, window=3)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[3] == 3.0  # (2+3+4)/3
        assert result[4] == 4.0  # (3+4+5)/3

    def test_rolling_mean_with_min_periods(self):
        """Test rolling mean with custom min_periods."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_mean

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_rolling_mean(arr, window=3, min_periods=1)

        assert result[0] == 1.0
        assert result[1] == 1.5  # (1+2)/2
        assert result[2] == 2.0  # (1+2+3)/3

    def test_rolling_mean_with_nan(self):
        """Test rolling mean with NaN values."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_mean

        arr = np.array([1.0, np.nan, 3.0, 4.0])
        result = numpy_rolling_mean(arr, window=3, min_periods=2)

        # (1 + 3) / 2 = 2.0
        assert result[2] == 2.0


class TestRollingMin:
    """Tests for numpy_rolling_min kernel."""

    def test_basic_rolling_min(self):
        """Test basic rolling min."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_min

        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = numpy_rolling_min(arr, window=3)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 1.0  # min(3,1,4)
        assert result[3] == 1.0  # min(1,4,1)
        assert result[4] == 1.0  # min(4,1,5)

    def test_rolling_min_with_min_periods(self):
        """Test rolling min with custom min_periods."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_min

        arr = np.array([3.0, 1.0, 4.0])
        result = numpy_rolling_min(arr, window=3, min_periods=1)

        assert result[0] == 3.0
        assert result[1] == 1.0
        assert result[2] == 1.0


class TestRollingMax:
    """Tests for numpy_rolling_max kernel."""

    def test_basic_rolling_max(self):
        """Test basic rolling max."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_max

        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = numpy_rolling_max(arr, window=3)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 4.0  # max(3,1,4)
        assert result[3] == 4.0  # max(1,4,1)
        assert result[4] == 5.0  # max(4,1,5)


class TestRollingStd:
    """Tests for numpy_rolling_std kernel."""

    def test_basic_rolling_std(self):
        """Test basic rolling std."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_std

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_rolling_std(arr, window=3)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # std of [1,2,3] with ddof=1
        expected_std = np.std([1, 2, 3], ddof=1)
        tm.assert_almost_equal(result[2], expected_std)


class TestRollingVar:
    """Tests for numpy_rolling_var kernel."""

    def test_basic_rolling_var(self):
        """Test basic rolling var."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_var

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_rolling_var(arr, window=3)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        expected_var = np.var([1, 2, 3], ddof=1)
        tm.assert_almost_equal(result[2], expected_var)


class TestRollingCount:
    """Tests for rolling count via helper function."""

    def test_basic_rolling_count(self):
        """Test rolling valid counts via helper."""
        from pandas.lazy.backends.numpy.misc import _get_rolling_valid_counts

        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        counts = _get_rolling_valid_counts(arr, window=3)

        # Counts represent valid values in each window
        assert counts[0] == 1  # [1]
        assert counts[1] == 2  # [1, 2]
        assert counts[2] == 2  # [1, 2, nan] -> 2 valid
        assert counts[3] == 2  # [2, nan, 4] -> 2 valid
        assert counts[4] == 2  # [nan, 4, 5] -> 2 valid


class TestShiftLagLead:
    """Tests for shift/lag/lead operations."""

    def test_shift_positive(self):
        """Test shift with positive periods."""
        from pandas.lazy.backends.numpy.misc import numpy_shift

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_shift(arr, periods=2)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 1.0
        assert result[3] == 2.0
        assert result[4] == 3.0

    def test_shift_negative(self):
        """Test shift with negative periods."""
        from pandas.lazy.backends.numpy.misc import numpy_shift

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_shift(arr, periods=-2)

        assert result[0] == 3.0
        assert result[1] == 4.0
        assert result[2] == 5.0
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_shift_zero(self):
        """Test shift with zero periods."""
        from pandas.lazy.backends.numpy.misc import numpy_shift

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_shift(arr, periods=0)
        tm.assert_numpy_array_equal(result, arr)

    def test_shift_with_fill_value(self):
        """Test shift with custom fill value."""
        from pandas.lazy.backends.numpy.misc import numpy_shift

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_shift(arr, periods=1, fill_value=0)

        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[2] == 2.0

    def test_lag(self):
        """Test lag function."""
        from pandas.lazy.backends.numpy.misc import numpy_lag

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_lag(arr, periods=2)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 1.0
        assert result[3] == 2.0

    def test_lead(self):
        """Test lead function."""
        from pandas.lazy.backends.numpy.misc import numpy_lead

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_lead(arr, periods=2)

        assert result[0] == 3.0
        assert result[1] == 4.0
        assert result[2] == 5.0
        assert np.isnan(result[3])

    def test_lag_with_fill_value(self):
        """Test lag with fill value."""
        from pandas.lazy.backends.numpy.misc import numpy_lag

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_lag(arr, periods=1, fill_value=-1)

        assert result[0] == -1.0
        assert result[1] == 1.0

    def test_lead_with_fill_value(self):
        """Test lead with fill value."""
        from pandas.lazy.backends.numpy.misc import numpy_lead

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_lead(arr, periods=1, fill_value=-1)

        assert result[0] == 2.0
        assert result[1] == 3.0
        assert result[2] == -1.0


class TestFfill:
    """Tests for forward fill operation."""

    def test_basic_ffill(self):
        """Test basic forward fill."""
        from pandas.lazy.backends.numpy.misc import numpy_ffill

        arr = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        result = numpy_ffill(arr)

        assert result[0] == 1.0
        assert result[1] == 1.0
        assert result[2] == 1.0
        assert result[3] == 4.0
        assert result[4] == 4.0

    def test_ffill_with_limit(self):
        """Test forward fill with limit."""
        from pandas.lazy.backends.numpy.misc import numpy_ffill

        arr = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        result = numpy_ffill(arr, limit=1)

        assert result[0] == 1.0
        assert result[1] == 1.0  # Filled
        assert np.isnan(result[2])  # Not filled (exceeds limit)
        assert np.isnan(result[3])  # Not filled
        assert result[4] == 5.0

    def test_ffill_no_nan(self):
        """Test ffill when no NaN values."""
        from pandas.lazy.backends.numpy.misc import numpy_ffill

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_ffill(arr)
        tm.assert_numpy_array_equal(result, arr)

    def test_ffill_empty_array(self):
        """Test ffill on empty array."""
        from pandas.lazy.backends.numpy.misc import numpy_ffill

        arr = np.array([])
        result = numpy_ffill(arr)
        assert len(result) == 0

    def test_ffill_int_array(self):
        """Test ffill converts int to float."""
        from pandas.lazy.backends.numpy.misc import numpy_ffill

        arr = np.array([1, 2, 3])  # int array
        result = numpy_ffill(arr)
        assert result.dtype == np.float64


class TestBfill:
    """Tests for backward fill operation."""

    def test_basic_bfill(self):
        """Test basic backward fill."""
        from pandas.lazy.backends.numpy.misc import numpy_bfill

        arr = np.array([np.nan, 2.0, np.nan, np.nan, 5.0])
        result = numpy_bfill(arr)

        assert result[0] == 2.0
        assert result[1] == 2.0
        assert result[2] == 5.0
        assert result[3] == 5.0
        assert result[4] == 5.0

    def test_bfill_with_limit(self):
        """Test backward fill with limit."""
        from pandas.lazy.backends.numpy.misc import numpy_bfill

        arr = np.array([np.nan, np.nan, np.nan, 4.0])
        result = numpy_bfill(arr, limit=1)

        assert np.isnan(result[0])  # Not filled
        assert np.isnan(result[1])  # Not filled
        assert result[2] == 4.0  # Filled
        assert result[3] == 4.0

    def test_bfill_no_nan(self):
        """Test bfill when no NaN values."""
        from pandas.lazy.backends.numpy.misc import numpy_bfill

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_bfill(arr)
        tm.assert_numpy_array_equal(result, arr)


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_rolling_sum_cumsum_direct(self):
        """Test cumsum-based rolling sum directly."""
        from pandas.lazy.backends.numpy.misc import _rolling_sum_cumsum

        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _rolling_sum_cumsum(arr, window=2, min_periods=1)

        assert result[0] == 1.0
        assert result[1] == 3.0
        assert result[2] == 5.0
        assert result[3] == 7.0

    def test_rolling_mean_cumsum_direct(self):
        """Test cumsum-based rolling mean directly."""
        from pandas.lazy.backends.numpy.misc import _rolling_mean_cumsum

        arr = np.array([2.0, 4.0, 6.0, 8.0])
        result = _rolling_mean_cumsum(arr, window=2, min_periods=1)

        assert result[0] == 2.0
        assert result[1] == 3.0
        assert result[2] == 5.0
        assert result[3] == 7.0

    def test_get_rolling_valid_counts(self):
        """Test rolling valid count calculation."""
        from pandas.lazy.backends.numpy.misc import _get_rolling_valid_counts

        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        counts = _get_rolling_valid_counts(arr, window=3)

        # First position: 1 valid (just [1])
        assert counts[0] == 1
        # Second position: 1 valid ([1, nan])
        assert counts[1] == 1
        # Third position: 2 valid ([1, nan, 3])
        assert counts[2] == 2
        # Fourth position: 1 valid ([nan, 3, nan])
        assert counts[3] == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_window_larger_than_array(self):
        """Test when window is larger than array."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_sum

        arr = np.array([1.0, 2.0])
        result = numpy_rolling_sum(arr, window=5)

        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_all_nan_array(self):
        """Test array of all NaN values."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_mean

        arr = np.array([np.nan, np.nan, np.nan])
        result = numpy_rolling_mean(arr, window=2)

        assert np.all(np.isnan(result))

    def test_single_element_array(self):
        """Test single element array."""
        from pandas.lazy.backends.numpy.misc import numpy_rolling_sum

        arr = np.array([5.0])
        result = numpy_rolling_sum(arr, window=1)

        assert result[0] == 5.0

    def test_shift_larger_than_array(self):
        """Test shift periods larger than array."""
        from pandas.lazy.backends.numpy.misc import numpy_shift

        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_shift(arr, periods=10)

        assert np.all(np.isnan(result))


class TestBottleneckFallbackParity:
    """The pure-NumPy fallback (used when Bottleneck is absent or disabled)
    must match both pandas and the Bottleneck path. Bottleneck is installed and
    enabled by default, so without these the fallback branches are never run."""

    # (kernel name, pandas Series.rolling method, extra kwargs)
    CASES = [
        ("rolling_sum", "sum", {}),
        ("rolling_mean", "mean", {}),
        ("rolling_min", "min", {}),
        ("rolling_max", "max", {}),
        ("rolling_std", "std", {"ddof": 1}),
        ("rolling_var", "var", {"ddof": 1}),
        ("rolling_median", "median", {}),
        ("rolling_count", "count", {}),
    ]

    @pytest.fixture(autouse=True)
    def _restore_bottleneck(self):
        # Restore option-following after each test so the toggle doesn't leak.
        from pandas.lazy.backends._bottleneck import set_use_bottleneck

        yield
        set_use_bottleneck(None)

    @pytest.mark.parametrize("kernel_name,pd_method,kwargs", CASES)
    @pytest.mark.parametrize(
        "enabled", [True, False], ids=["bottleneck", "numpy_fallback"]
    )
    def test_rolling_kernel_matches_pandas(
        self, kernel_name, pd_method, kwargs, enabled
    ):
        from pandas.lazy.backends import get_kernel
        from pandas.lazy.backends._bottleneck import (
            BOTTLENECK_INSTALLED,
            set_use_bottleneck,
        )

        if enabled and not BOTTLENECK_INSTALLED:
            pytest.skip("bottleneck not installed")
        set_use_bottleneck(enabled)

        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.0, 7.0], dtype=float)
        window, min_periods = 3, 1
        kernel = get_kernel(kernel_name, "numpy")
        got = kernel(arr, window, min_periods, **kwargs)

        rolling = pd.Series(arr).rolling(window, min_periods=min_periods)
        expected = getattr(rolling, pd_method)(**kwargs)
        tm.assert_series_equal(
            pd.Series(np.asarray(got, dtype=float)),
            pd.Series(expected.to_numpy(dtype=float)),
            check_dtype=False,
        )

    @pytest.mark.parametrize("kernel_name,pd_method,kwargs", CASES)
    def test_fallback_handles_nan_like_pandas(self, kernel_name, pd_method, kwargs):
        from pandas.lazy.backends import get_kernel
        from pandas.lazy.backends._bottleneck import set_use_bottleneck

        set_use_bottleneck(False)  # force the pure-NumPy fallback
        arr = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0, 7.0], dtype=float)
        window, min_periods = 3, 1
        kernel = get_kernel(kernel_name, "numpy")
        got = kernel(arr, window, min_periods, **kwargs)
        expected = getattr(
            pd.Series(arr).rolling(window, min_periods=min_periods), pd_method
        )(**kwargs)
        tm.assert_series_equal(
            pd.Series(np.asarray(got, dtype=float)),
            pd.Series(expected.to_numpy(dtype=float)),
            check_dtype=False,
        )

    def test_rolling_rank_requires_bottleneck_without_it(self):
        # rolling_rank is the one rolling kernel with no NumPy fallback.
        from pandas.lazy.backends import get_kernel
        from pandas.lazy.backends._bottleneck import set_use_bottleneck

        set_use_bottleneck(False)
        kernel = get_kernel("rolling_rank", "numpy")
        with pytest.raises(NotImplementedError, match="requires Bottleneck"):
            kernel(np.array([1.0, 2.0, 3.0]), 2, 1)

    def test_option_follows_pd_set_option(self):
        from pandas.lazy.backends._bottleneck import (
            BOTTLENECK_INSTALLED,
            set_use_bottleneck,
            use_bottleneck,
        )

        if not BOTTLENECK_INSTALLED:
            pytest.skip("bottleneck not installed")
        set_use_bottleneck(None)  # follow the option
        with pd.option_context("compute.use_bottleneck", False):
            assert use_bottleneck() is False
        with pd.option_context("compute.use_bottleneck", True):
            assert use_bottleneck() is True
        # An explicit override wins over the option.
        set_use_bottleneck(False)
        with pd.option_context("compute.use_bottleneck", True):
            assert use_bottleneck() is False
