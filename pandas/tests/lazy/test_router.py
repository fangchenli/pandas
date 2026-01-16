"""
Tests for lazy pandas backend routing logic.

Tests the router module which decides whether to use Arrow or NumPy backend.
"""

import numpy as np
import pyarrow as pa

from pandas.lazy.backends.router import (
    ARROW_PREFERRED_OPS,
    NEUTRAL_OPS,
    NUMPY_PREFERRED_OPS,
    collect_func_names_from_ir,
    decide_expr_backend,
    decide_node_backend,
    infer_backend_from_arrays,
    is_null_op,
    is_string_op,
    should_use_arrow,
    should_use_numpy,
)


class TestOperationClassification:
    """Tests for operation classification functions."""

    def test_is_string_op(self):
        """Test string operation detection."""
        assert is_string_op("str_lower")
        assert is_string_op("str_upper")
        assert is_string_op("str_contains")
        assert not is_string_op("add")
        assert not is_string_op("sum")

    def test_is_null_op(self):
        """Test null operation detection."""
        assert is_null_op("is_null")
        assert is_null_op("is_not_null")
        assert is_null_op("fill_null")
        assert is_null_op("coalesce")
        assert not is_null_op("add")
        assert not is_null_op("str_lower")

    def test_should_use_arrow(self):
        """Test Arrow preference detection."""
        # String ops prefer Arrow
        assert should_use_arrow("str_lower")
        assert should_use_arrow("str_upper")
        assert should_use_arrow("str_contains")

        # Null ops prefer Arrow
        assert should_use_arrow("is_null")
        assert should_use_arrow("fill_null")

        # Neutral ops don't prefer Arrow
        assert not should_use_arrow("add")
        assert not should_use_arrow("sum")

    def test_should_use_numpy(self):
        """Test NumPy preference detection."""
        # Currently no ops strongly prefer NumPy
        assert not should_use_numpy("add")
        assert not should_use_numpy("sum")
        assert not should_use_numpy("str_lower")

    def test_operation_sets_disjoint(self):
        """Test that operation sets are disjoint."""
        assert len(ARROW_PREFERRED_OPS & NUMPY_PREFERRED_OPS) == 0
        assert len(ARROW_PREFERRED_OPS & NEUTRAL_OPS) == 0


class TestInferBackendFromArrays:
    """Tests for backend inference from arrays."""

    def test_all_numpy_arrays(self):
        """Test inference with all NumPy arrays."""
        arrays = {
            "a": np.array([1, 2, 3]),
            "b": np.array([4, 5, 6]),
        }
        assert infer_backend_from_arrays(arrays) == "numpy"

    def test_all_arrow_arrays(self):
        """Test inference with all Arrow arrays."""
        arrays = {
            "a": pa.array([1, 2, 3]),
            "b": pa.array([4, 5, 6]),
        }
        assert infer_backend_from_arrays(arrays) == "arrow"

    def test_mixed_majority_numpy(self):
        """Test inference with majority NumPy arrays."""
        arrays = {
            "a": np.array([1, 2, 3]),
            "b": np.array([4, 5, 6]),
            "c": pa.array([7, 8, 9]),
        }
        assert infer_backend_from_arrays(arrays) == "numpy"

    def test_mixed_majority_arrow(self):
        """Test inference with majority Arrow arrays."""
        arrays = {
            "a": pa.array([1, 2, 3]),
            "b": pa.array([4, 5, 6]),
            "c": np.array([7, 8, 9]),
        }
        assert infer_backend_from_arrays(arrays) == "arrow"

    def test_tie_prefers_arrow(self):
        """Test that tie goes to Arrow."""
        arrays = {
            "a": np.array([1, 2, 3]),
            "b": pa.array([4, 5, 6]),
        }
        assert infer_backend_from_arrays(arrays) == "arrow"

    def test_excludes_index_columns(self):
        """Test that index columns are excluded from inference."""
        arrays = {
            "a": np.array([1, 2, 3]),
            "__index__": pa.array([0, 1, 2]),  # Index column - excluded
        }
        assert infer_backend_from_arrays(arrays) == "numpy"

    def test_chunked_array_counts_as_arrow(self):
        """Test that ChunkedArray is recognized as Arrow."""
        arrays = {
            "a": pa.chunked_array([[1, 2], [3, 4]]),
        }
        assert infer_backend_from_arrays(arrays) == "arrow"


class TestDecideExprBackend:
    """Tests for single expression backend decision."""

    def test_arrow_preferred_overrides(self):
        """Test that Arrow-preferred ops override other preferences."""
        # String op uses Arrow even with numpy input/preference
        assert decide_expr_backend("str_lower", "numpy", "numpy") == "arrow"
        assert decide_expr_backend("is_null", "numpy", "numpy") == "arrow"

    def test_numpy_preferred_overrides(self):
        """Test that NumPy-preferred ops override (currently empty)."""
        # No ops currently in NUMPY_PREFERRED_OPS

    def test_user_preference_respected(self):
        """Test that user preference is respected for neutral ops."""
        # User prefers arrow
        assert decide_expr_backend("add", "numpy", "arrow") == "arrow"
        # User prefers numpy
        assert decide_expr_backend("add", "arrow", "numpy") == "numpy"

    def test_follows_input_when_auto(self):
        """Test that auto preference follows input."""
        assert decide_expr_backend("add", "numpy", "auto") == "numpy"
        assert decide_expr_backend("add", "arrow", "auto") == "arrow"

    def test_defaults_to_numpy(self):
        """Test default when everything is auto."""
        assert decide_expr_backend("add", "auto", "auto") == "numpy"


class TestDecideNodeBackend:
    """Tests for node-level backend decision."""

    def test_arrow_ops_force_arrow(self):
        """Test that Arrow ops force Arrow backend."""
        arrays = {"a": np.array([1, 2, 3])}
        result = decide_node_backend(["str_lower"], arrays, "numpy")
        assert result == "arrow"

    def test_numpy_ops_force_numpy(self):
        """Test that NumPy ops force NumPy backend."""
        # Currently no NUMPY_PREFERRED_OPS, so this path is not taken

    def test_neutral_ops_follow_preference(self):
        """Test neutral ops follow user preference."""
        arrays = {"a": np.array([1, 2, 3])}

        result = decide_node_backend(["add"], arrays, "arrow")
        assert result == "arrow"

        result = decide_node_backend(["add"], arrays, "numpy")
        assert result == "numpy"

    def test_neutral_ops_follow_input(self):
        """Test neutral ops follow input when auto."""
        numpy_arrays = {"a": np.array([1, 2, 3])}
        arrow_arrays = {"a": pa.array([1, 2, 3])}

        assert decide_node_backend(["add"], numpy_arrays, "auto") == "numpy"
        assert decide_node_backend(["add"], arrow_arrays, "auto") == "arrow"

    def test_multiple_neutral_ops(self):
        """Test with multiple neutral operations."""
        arrays = {"a": np.array([1, 2, 3])}
        result = decide_node_backend(["add", "multiply", "sum"], arrays, "auto")
        assert result == "numpy"


class TestCollectFuncNamesFromIR:
    """Tests for IR function name collection."""

    def test_simple_call(self):
        """Test collecting from simple Call node."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        node = Call("add", args=(FieldRef("a"), FieldRef("b")))
        result = collect_func_names_from_ir(node)
        assert result == ["add"]

    def test_nested_calls(self):
        """Test collecting from nested Call nodes."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        # (a + b) * 2
        inner = Call("add", args=(FieldRef("a"), FieldRef("b")))
        outer = Call("multiply", args=(inner, Literal(2)))

        result = collect_func_names_from_ir(outer)
        assert "multiply" in result
        assert "add" in result

    def test_alias_node(self):
        """Test collecting through Alias node."""
        from pandas.lazy.ir import (
            Alias,
            Call,
            FieldRef,
        )

        call = Call("add", args=(FieldRef("a"), FieldRef("b")))
        alias = Alias(call, "result")

        result = collect_func_names_from_ir(alias)
        assert result == ["add"]

    def test_cast_node(self):
        """Test collecting through Cast node."""
        from pandas.lazy.ir import (
            Call,
            Cast,
            FieldRef,
        )

        call = Call("add", args=(FieldRef("a"), FieldRef("b")))
        cast = Cast(call, "float64")

        result = collect_func_names_from_ir(cast)
        assert result == ["add"]

    def test_field_ref_returns_empty(self):
        """Test that FieldRef returns empty list."""
        from pandas.lazy.ir import FieldRef

        result = collect_func_names_from_ir(FieldRef("a"))
        assert result == []

    def test_literal_returns_empty(self):
        """Test that Literal returns empty list."""
        from pandas.lazy.ir import Literal

        result = collect_func_names_from_ir(Literal(42))
        assert result == []

    def test_complex_expression(self):
        """Test with complex nested expression."""
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        # str_lower(a) + 10 > 5
        lower = Call("str_lower", args=(FieldRef("a"),))
        add = Call("add", args=(lower, Literal(10)))
        compare = Call("greater", args=(add, Literal(5)))

        result = collect_func_names_from_ir(compare)
        assert set(result) == {"greater", "add", "str_lower"}


class TestCaching:
    """Tests for LRU cache behavior."""

    def test_decide_expr_backend_cached(self):
        """Test that decide_expr_backend is cached."""
        # Clear cache
        decide_expr_backend.cache_clear()

        # First call
        result1 = decide_expr_backend("add", "numpy", "auto")
        info1 = decide_expr_backend.cache_info()
        assert info1.misses == 1

        # Second call with same args - should hit cache
        result2 = decide_expr_backend("add", "numpy", "auto")
        info2 = decide_expr_backend.cache_info()
        assert info2.hits == 1
        assert result1 == result2

    def test_should_use_arrow_cached(self):
        """Test that should_use_arrow is cached."""
        should_use_arrow.cache_clear()

        should_use_arrow("str_lower")
        info1 = should_use_arrow.cache_info()
        assert info1.misses == 1

        should_use_arrow("str_lower")
        info2 = should_use_arrow.cache_info()
        assert info2.hits == 1
