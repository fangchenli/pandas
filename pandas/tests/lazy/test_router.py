"""
Tests for lazy pandas backend routing logic.

Tests the router module which decides whether to use Arrow or NumPy backend.
"""

from pandas.lazy.backends.router import (
    ARROW_PREFERRED_OPS,
    decide_expr_backend,
    should_use_arrow,
)


class TestShouldUseArrow:
    """Tests for the Arrow-preference classifier."""

    def test_string_and_null_ops_prefer_arrow(self):
        assert should_use_arrow("str_lower")
        assert should_use_arrow("str_contains")
        assert should_use_arrow("is_null")
        assert should_use_arrow("fill_null")

    def test_neutral_ops_do_not_prefer_arrow(self):
        assert not should_use_arrow("add")
        assert not should_use_arrow("sum")

    def test_arrow_only_str_reverse_prefers_arrow(self):
        # str_reverse has no NumPy kernel; it must route to Arrow.
        assert "str_reverse" in ARROW_PREFERRED_OPS
        assert should_use_arrow("str_reverse")


class TestDecideExprBackend:
    """Tests for single expression backend decision."""

    def test_arrow_preferred_overrides(self):
        # An Arrow-preferred op uses Arrow even with numpy input/preference.
        assert decide_expr_backend("str_lower", "numpy", "numpy") == "arrow"
        assert decide_expr_backend("is_null", "numpy", "numpy") == "arrow"

    def test_user_preference_respected(self):
        assert decide_expr_backend("add", "numpy", "arrow") == "arrow"
        assert decide_expr_backend("add", "arrow", "numpy") == "numpy"

    def test_follows_input_when_auto(self):
        assert decide_expr_backend("add", "numpy", "auto") == "numpy"
        assert decide_expr_backend("add", "arrow", "auto") == "arrow"

    def test_defaults_to_numpy(self):
        assert decide_expr_backend("add", "auto", "auto") == "numpy"


class TestCaching:
    """Tests for LRU cache behavior."""

    def test_decide_expr_backend_cached(self):
        decide_expr_backend.cache_clear()

        result1 = decide_expr_backend("add", "numpy", "auto")
        assert decide_expr_backend.cache_info().misses == 1

        result2 = decide_expr_backend("add", "numpy", "auto")
        assert decide_expr_backend.cache_info().hits == 1
        assert result1 == result2

    def test_should_use_arrow_cached(self):
        should_use_arrow.cache_clear()

        should_use_arrow("str_lower")
        assert should_use_arrow.cache_info().misses == 1

        should_use_arrow("str_lower")
        assert should_use_arrow.cache_info().hits == 1
