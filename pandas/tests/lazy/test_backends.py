"""
Tests for lazy pandas backend execution system.

Tests the kernel registry, format conversion, and backend routing.
"""

import numpy as np
import pyarrow as pa
import pytest

import pandas as pd
import pandas._testing as tm


class TestKernelRegistry:
    """Tests for kernel registration and dispatch."""

    def test_register_and_get_kernel(self):
        """Test registering and retrieving a kernel."""
        from pandas.lazy.backends import (
            get_kernel,
            has_kernel,
            register_kernel,
        )

        @register_kernel("test_func", "arrow")
        def test_impl(arr):
            return arr

        assert has_kernel("test_func", "arrow")
        assert get_kernel("test_func", "arrow") is test_impl

    def test_has_kernel_returns_false_for_missing(self):
        """Test has_kernel returns False for unregistered kernel."""
        from pandas.lazy.backends import has_kernel

        assert not has_kernel("nonexistent_func", "arrow")
        assert not has_kernel("nonexistent_func", "numpy")

    def test_dispatch_kernel(self):
        """Test dispatching to a kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 3])
        result = dispatch_kernel("add", "arrow", arr, 1)
        expected = pa.array([2, 3, 4])
        assert result.equals(expected)

    def test_dispatch_kernel_raises_for_missing(self):
        """Test dispatch raises NotImplementedError for missing kernel."""
        from pandas.lazy.backends import dispatch_kernel

        with pytest.raises(NotImplementedError, match="No arrow kernel"):
            dispatch_kernel("nonexistent_func", "arrow")

    def test_list_kernels(self):
        """Test listing registered kernels."""
        from pandas.lazy.backends import list_kernels

        arrow_kernels = list_kernels("arrow")
        numpy_kernels = list_kernels("numpy")

        # Should have some kernels registered
        assert "add" in arrow_kernels
        assert "add" in numpy_kernels
        assert "str_lower" in arrow_kernels


class TestArrowKernels:
    """Tests for Arrow kernel implementations."""

    def test_str_lower(self):
        """Test Arrow string lowercase."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["HELLO", "WORLD"])
        result = dispatch_kernel("str_lower", "arrow", arr)
        expected = pa.array(["hello", "world"])
        assert result.equals(expected)

    def test_str_upper(self):
        """Test Arrow string uppercase."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["hello", "world"])
        result = dispatch_kernel("str_upper", "arrow", arr)
        expected = pa.array(["HELLO", "WORLD"])
        assert result.equals(expected)

    def test_str_contains(self):
        """Test Arrow string contains."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["hello", "world", "helloworld"])
        result = dispatch_kernel("str_contains", "arrow", arr, "hell", regex=False)
        expected = pa.array([True, False, True])
        assert result.equals(expected)

    def test_is_null(self):
        """Test Arrow is_null."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, None, 3])
        result = dispatch_kernel("is_null", "arrow", arr)
        expected = pa.array([False, True, False])
        assert result.equals(expected)

    def test_fill_null(self):
        """Test Arrow fill_null."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, None, 3])
        result = dispatch_kernel("fill_null", "arrow", arr, 0)
        expected = pa.array([1, 0, 3])
        assert result.equals(expected)

    def test_add(self):
        """Test Arrow addition."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 3])
        result = dispatch_kernel("add", "arrow", arr, 10)
        expected = pa.array([11, 12, 13])
        assert result.equals(expected)

    def test_equal(self):
        """Test Arrow equality comparison."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 3])
        result = dispatch_kernel("equal", "arrow", arr, 2)
        expected = pa.array([False, True, False])
        assert result.equals(expected)

    def test_sum(self):
        """Test Arrow sum aggregation."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 3])
        result = dispatch_kernel("sum", "arrow", arr)
        assert result == 6

    def test_mean(self):
        """Test Arrow mean aggregation."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, 2.0, 3.0])
        result = dispatch_kernel("mean", "arrow", arr)
        assert result == 2.0


class TestNumpyKernels:
    """Tests for NumPy kernel implementations."""

    def test_add(self):
        """Test NumPy addition."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 2, 3])
        result = dispatch_kernel("add", "numpy", arr, 10)
        expected = np.array([11, 12, 13])
        tm.assert_numpy_array_equal(result, expected)

    def test_equal(self):
        """Test NumPy equality comparison."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 2, 3])
        result = dispatch_kernel("equal", "numpy", arr, 2)
        expected = np.array([False, True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_sum(self):
        """Test NumPy sum aggregation."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 2, 3])
        result = dispatch_kernel("sum", "numpy", arr)
        assert result == 6

    def test_mean(self):
        """Test NumPy mean aggregation."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0])
        result = dispatch_kernel("mean", "numpy", arr)
        assert result == 2.0


class TestFormatConversion:
    """Tests for format detection and conversion."""

    def test_is_arrow_backed(self):
        """Test Arrow format detection."""
        from pandas.lazy.backends.convert import is_arrow_backed

        assert is_arrow_backed(pa.array([1, 2, 3]))
        assert is_arrow_backed(pa.chunked_array([[1, 2], [3]]))
        assert not is_arrow_backed(np.array([1, 2, 3]))

    def test_is_numpy_backed(self):
        """Test NumPy format detection."""
        from pandas.lazy.backends.convert import is_numpy_backed

        assert is_numpy_backed(np.array([1, 2, 3]))
        assert not is_numpy_backed(pa.array([1, 2, 3]))

    def test_get_array_backend(self):
        """Test getting array backend type."""
        from pandas.lazy.backends.convert import get_array_backend

        assert get_array_backend(pa.array([1])) == "arrow"
        assert get_array_backend(np.array([1])) == "numpy"

    def test_to_arrow(self):
        """Test converting to Arrow format."""
        from pandas.lazy.backends.convert import to_arrow

        # NumPy to Arrow
        np_arr = np.array([1, 2, 3])
        result = to_arrow(np_arr)
        assert isinstance(result, (pa.Array, pa.ChunkedArray))

        # Arrow stays Arrow
        pa_arr = pa.array([1, 2, 3])
        result = to_arrow(pa_arr)
        assert result is pa_arr

    def test_to_numpy(self):
        """Test converting to NumPy format."""
        from pandas.lazy.backends.convert import to_numpy

        # Arrow to NumPy
        pa_arr = pa.array([1, 2, 3])
        result = to_numpy(pa_arr)
        assert isinstance(result, np.ndarray)
        tm.assert_numpy_array_equal(result, np.array([1, 2, 3]))

        # NumPy stays NumPy
        np_arr = np.array([1, 2, 3])
        result = to_numpy(np_arr)
        assert result is np_arr

    def test_ensure_backend(self):
        """Test ensuring array is in target backend."""
        from pandas.lazy.backends.convert import ensure_backend

        np_arr = np.array([1, 2, 3])
        pa_arr = pa.array([1, 2, 3])

        # NumPy to Arrow
        result = ensure_backend(np_arr, "arrow")
        assert isinstance(result, (pa.Array, pa.ChunkedArray))

        # Arrow to NumPy
        result = ensure_backend(pa_arr, "numpy")
        assert isinstance(result, np.ndarray)


class TestArrayExtraction:
    """Tests for extracting arrays from pandas objects."""

    def test_extract_from_numpy_series(self):
        """Test extracting array from NumPy-backed Series."""
        from pandas.lazy.backends.convert import extract_array

        s = pd.Series([1, 2, 3])
        arr = extract_array(s)
        assert isinstance(arr, np.ndarray)

    def test_extract_from_arrow_series(self):
        """Test extracting array from Arrow-backed Series."""
        from pandas.lazy.backends.convert import extract_array

        s = pd.Series(["a", "b", "c"], dtype="string[pyarrow]")
        arr = extract_array(s)
        assert isinstance(arr, (pa.Array, pa.ChunkedArray))


class TestBackendRouter:
    """Tests for backend routing decisions."""

    def test_should_use_arrow_for_string_ops(self):
        """Test that string ops prefer Arrow."""
        from pandas.lazy.backends.router import should_use_arrow

        assert should_use_arrow("str_lower")
        assert should_use_arrow("str_upper")
        assert should_use_arrow("str_contains")

    def test_should_use_arrow_for_null_ops(self):
        """Test that null ops prefer Arrow."""
        from pandas.lazy.backends.router import should_use_arrow

        assert should_use_arrow("is_null")
        assert should_use_arrow("fill_null")
        assert should_use_arrow("coalesce")

    def test_neutral_ops_dont_force_backend(self):
        """Test that neutral ops don't force a backend."""
        from pandas.lazy.backends.router import (
            should_use_arrow,
            should_use_numpy,
        )

        assert not should_use_arrow("add")
        assert not should_use_numpy("add")

    def test_decide_expr_backend(self):
        """Test expression backend decision."""
        from pandas.lazy.backends.router import decide_expr_backend

        # String ops always use Arrow
        assert decide_expr_backend("str_lower", "numpy", "auto") == "arrow"

        # User preference respected for neutral ops
        assert decide_expr_backend("add", "numpy", "arrow") == "arrow"
        assert decide_expr_backend("add", "arrow", "numpy") == "numpy"

        # Follow input for auto
        assert decide_expr_backend("add", "numpy", "auto") == "numpy"
        assert decide_expr_backend("add", "arrow", "auto") == "arrow"


class TestIndexColumnNaming:
    """Tests for index column naming conventions."""

    def test_index_col_name(self):
        """Test generating index column names."""
        from pandas.lazy.backends.types import index_col_name

        assert index_col_name() == "__index__"
        assert index_col_name(0) == "__index_0__"
        assert index_col_name(1) == "__index_1__"

    def test_is_index_col(self):
        """Test checking if column is an index column."""
        from pandas.lazy.backends.types import is_index_col

        assert is_index_col("__index__")
        assert is_index_col("__index_0__")
        assert is_index_col("__index_1__")
        assert not is_index_col("my_column")
        assert not is_index_col("index")

    def test_parse_index_level(self):
        """Test parsing index level from column name."""
        from pandas.lazy.backends.types import parse_index_level

        assert parse_index_level("__index__") is None
        assert parse_index_level("__index_0__") == 0
        assert parse_index_level("__index_2__") == 2
        assert parse_index_level("my_column") is None


class TestArraysToDataFrame:
    """Tests for converting ArrayDict to DataFrame."""

    def test_simple_conversion(self):
        """Test basic ArrayDict to DataFrame conversion."""
        from pandas.lazy.backends.convert import arrays_to_dataframe

        arrays = {
            "a": np.array([1, 2, 3]),
            "b": np.array([4, 5, 6]),
        }
        result = arrays_to_dataframe(arrays)

        expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_with_arrow_arrays(self):
        """Test conversion with Arrow arrays returns Arrow-backed dtypes by default."""
        from pandas.lazy.backends.convert import arrays_to_dataframe

        arrays = {
            "a": pa.array([1, 2, 3]),
            "b": pa.array([4, 5, 6]),
        }
        result = arrays_to_dataframe(arrays)

        # Default is use_arrow_dtype=True for near-zero-copy conversion
        expected = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3], dtype="int64[pyarrow]"),
                "b": pd.array([4, 5, 6], dtype="int64[pyarrow]"),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_with_arrow_arrays_numpy_output(self):
        """Test conversion with Arrow arrays can return NumPy dtypes."""
        from pandas.lazy.backends.convert import arrays_to_dataframe

        arrays = {
            "a": pa.array([1, 2, 3]),
            "b": pa.array([4, 5, 6]),
        }
        result = arrays_to_dataframe(arrays, use_arrow_dtype=False)

        expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_index_columns_excluded(self):
        """Test that index columns are not included as data columns."""
        from pandas.lazy.backends.convert import arrays_to_dataframe

        arrays = {
            "__index__": np.array([10, 20, 30]),
            "a": np.array([1, 2, 3]),
        }
        result = arrays_to_dataframe(arrays)

        # Index should be RangeIndex, not the __index__ values
        assert list(result.columns) == ["a"]
        assert isinstance(result.index, pd.RangeIndex)


class TestDataFrameToArrays:
    """Tests for converting DataFrame to ArrayDict."""

    def test_simple_conversion(self):
        """Test basic DataFrame to ArrayDict conversion."""
        from pandas.lazy.backends.convert import dataframe_to_arrays
        from pandas.lazy.backends.types import INDEX_COL_NAME

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        arrays, index_names, index_is_multi = dataframe_to_arrays(df)

        assert "a" in arrays
        assert "b" in arrays
        assert INDEX_COL_NAME in arrays
        assert index_names == [None]
        assert not index_is_multi

    def test_with_named_index(self):
        """Test conversion with named index."""
        from pandas.lazy.backends.convert import dataframe_to_arrays

        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx"))
        arrays, index_names, index_is_multi = dataframe_to_arrays(df)

        assert index_names == ["idx"]

    def test_with_multiindex(self):
        """Test conversion with MultiIndex."""
        from pandas.lazy.backends.convert import dataframe_to_arrays

        df = pd.DataFrame(
            {"a": [1, 2, 3]},
            index=pd.MultiIndex.from_tuples(
                [(1, "a"), (1, "b"), (2, "a")], names=["l1", "l2"]
            ),
        )
        arrays, index_names, index_is_multi = dataframe_to_arrays(df)

        assert "__index_0__" in arrays
        assert "__index_1__" in arrays
        assert index_names == ["l1", "l2"]
        assert index_is_multi


class TestArrayEvaluator:
    """Tests for ArrayEvaluator kernel dispatch."""

    def test_evaluate_field_ref(self):
        """Test evaluating a field reference."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import FieldRef

        arrays = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
        evaluator = ArrayEvaluator(arrays)

        result = evaluator.evaluate(FieldRef("a"))
        tm.assert_numpy_array_equal(result, np.array([1, 2, 3]))

    def test_evaluate_literal(self):
        """Test evaluating a literal."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import Literal

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        result = evaluator.evaluate(Literal(42))
        assert result == 42

    def test_evaluate_add_kernel(self):
        """Test evaluating addition via kernel dispatch."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        # a + 10
        node = Call("add", args=(FieldRef("a"), Literal(10)))
        result = evaluator.evaluate(node)
        tm.assert_numpy_array_equal(result, np.array([11, 12, 13]))

    def test_evaluate_comparison_kernel(self):
        """Test evaluating comparison via kernel dispatch."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        # a > 1
        node = Call("greater", args=(FieldRef("a"), Literal(1)))
        result = evaluator.evaluate(node)
        tm.assert_numpy_array_equal(result, np.array([False, True, True]))

    def test_evaluate_arrow_string_kernel(self):
        """Test evaluating string op on Arrow array."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"name": pa.array(["HELLO", "WORLD"])}
        evaluator = ArrayEvaluator(arrays)

        # str_lower(name)
        node = Call("str_lower", args=(FieldRef("name"),))
        result = evaluator.evaluate(node)
        assert result.equals(pa.array(["hello", "world"]))

    def test_evaluate_with_preferred_backend(self):
        """Test that preferred backend is respected."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        # NumPy array with arrow preference
        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays, preferred_backend="arrow")

        # For neutral ops like add, should convert to arrow
        node = Call("add", args=(FieldRef("a"), Literal(1)))
        result = evaluator.evaluate(node)
        # Result should be arrow array
        assert isinstance(result, (pa.Array, pa.ChunkedArray))

    def test_evaluate_alias(self):
        """Test evaluating an Alias node."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Alias,
            FieldRef,
        )

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        # Alias signature: Alias(arg=IRNode, name=str)
        node = Alias(arg=FieldRef("a"), name="b")
        result = evaluator.evaluate(node)
        tm.assert_numpy_array_equal(result, np.array([1, 2, 3]))

    def test_evaluate_chained_operations(self):
        """Test evaluating chained operations."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        # (a + 10) * 2
        add_node = Call("add", args=(FieldRef("a"), Literal(10)))
        mul_node = Call("multiply", args=(add_node, Literal(2)))
        result = evaluator.evaluate(mul_node)
        tm.assert_numpy_array_equal(result, np.array([22, 24, 26]))


class TestSortKernels:
    """Tests for sort-related kernels."""

    def test_arrow_sort_indices(self):
        """Test Arrow sort_indices kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = dispatch_kernel("array_sort_indices", "arrow", arr)
        expected = [1, 3, 6, 0, 2, 4, 7, 5]
        assert result.to_pylist() == expected

    def test_arrow_sort_indices_descending(self):
        """Test Arrow sort_indices with descending order."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = dispatch_kernel("array_sort_indices", "arrow", arr, order="descending")
        expected = [5, 7, 4, 2, 0, 6, 1, 3]
        assert result.to_pylist() == expected

    def test_numpy_sort_indices(self):
        """Test NumPy sort_indices kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = dispatch_kernel("sort_indices", "numpy", arr)
        expected = [1, 3, 6, 0, 2, 4, 7, 5]
        assert result.tolist() == expected

    def test_numpy_sort_indices_descending(self):
        """Test NumPy sort_indices with descending order (stable).

        Lazy sorts are stable by contract: the tied values (1 at indices
        1 and 3) keep their original relative order, matching the Arrow
        backend and eager kind="stable" sorts.
        """
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = dispatch_kernel("sort_indices", "numpy", arr, descending=True)
        expected = [5, 7, 4, 2, 0, 6, 1, 3]
        assert result.tolist() == expected

    def test_numpy_sort_indices_with_nan(self):
        """Test NumPy sort_indices with NaN values."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([3.0, np.nan, 1.0, 4.0])
        result = dispatch_kernel("sort_indices", "numpy", arr, null_placement="at_end")
        # NaN should be at the end
        assert result[-1] == 1  # index of NaN


class TestTakeKernels:
    """Tests for take/gather kernels."""

    def test_arrow_take(self):
        """Test Arrow take kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([10, 20, 30, 40, 50])
        indices = pa.array([0, 2, 4])
        result = dispatch_kernel("take", "arrow", arr, indices)
        assert result.to_pylist() == [10, 30, 50]

    def test_numpy_take(self):
        """Test NumPy take kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([10, 20, 30, 40, 50])
        indices = np.array([0, 2, 4])
        result = dispatch_kernel("take", "numpy", arr, indices)
        tm.assert_numpy_array_equal(result, np.array([10, 30, 50]))


class TestUniqueKernels:
    """Tests for unique/distinct kernels."""

    def test_arrow_unique(self):
        """Test Arrow unique kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 2, 3, 3, 3])
        result = dispatch_kernel("unique", "arrow", arr)
        assert sorted(result.to_pylist()) == [1, 2, 3]

    def test_numpy_unique(self):
        """Test NumPy unique kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 2, 2, 3, 3, 3])
        result = dispatch_kernel("unique", "numpy", arr)
        tm.assert_numpy_array_equal(result, np.array([1, 2, 3]))


class TestIsInKernels:
    """Tests for set membership kernels."""

    def test_arrow_is_in(self):
        """Test Arrow is_in kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 3, 4, 5])
        value_set = pa.array([2, 4])
        result = dispatch_kernel("is_in", "arrow", arr, value_set)
        assert result.to_pylist() == [False, True, False, True, False]

    def test_numpy_is_in(self):
        """Test NumPy is_in kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 2, 3, 4, 5])
        value_set = np.array([2, 4])
        result = dispatch_kernel("is_in", "numpy", arr, value_set)
        tm.assert_numpy_array_equal(result, np.array([False, True, False, True, False]))


class TestSelectKKernels:
    """Tests for TopK selection kernels."""

    def test_arrow_select_k(self):
        """Test Arrow select_k_unstable kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([3, 1, 4, 1, 5, 9, 2, 6])
        # Get indices of 3 smallest
        result = dispatch_kernel("select_k_unstable", "arrow", arr, 3)
        # Values at these indices should be [1, 1, 2]
        values = pa.array([arr[i].as_py() for i in result.to_pylist()])
        assert sorted(values.to_pylist()) == [1, 1, 2]

    def test_numpy_select_k_ascending(self):
        """Test NumPy select_k_unstable with ascending order."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = dispatch_kernel(
            "select_k_unstable", "numpy", arr, 3, order="ascending"
        )
        # Should give indices of 3 smallest, sorted by value
        values = arr[result]
        tm.assert_numpy_array_equal(np.sort(values), np.array([1, 1, 2]))

    def test_numpy_select_k_descending(self):
        """Test NumPy select_k_unstable with descending order."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = dispatch_kernel(
            "select_k_unstable", "numpy", arr, 3, order="descending"
        )
        # Should give indices of 3 largest
        values = arr[result]
        tm.assert_numpy_array_equal(np.sort(values), np.array([5, 6, 9]))


class TestIfElseKernels:
    """Tests for conditional kernels."""

    def test_arrow_if_else(self):
        """Test Arrow if_else kernel."""
        from pandas.lazy.backends import dispatch_kernel

        cond = pa.array([True, False, True, False])
        true_val = pa.array([1, 1, 1, 1])
        false_val = pa.array([0, 0, 0, 0])
        result = dispatch_kernel("if_else", "arrow", cond, true_val, false_val)
        assert result.to_pylist() == [1, 0, 1, 0]

    def test_numpy_if_else(self):
        """Test NumPy if_else kernel."""
        from pandas.lazy.backends import dispatch_kernel

        cond = np.array([True, False, True, False])
        true_val = 100
        false_val = 0
        result = dispatch_kernel("if_else", "numpy", cond, true_val, false_val)
        tm.assert_numpy_array_equal(result, np.array([100, 0, 100, 0]))

    def test_numpy_case_when(self):
        """Test NumPy case_when kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 5, 3, 8, 2])
        # case when arr > 5 then 100, when arr > 2 then 50, else 0
        result = dispatch_kernel("case_when", "numpy", arr > 5, 100, arr > 2, 50, 0)
        tm.assert_numpy_array_equal(result, np.array([0, 50, 50, 100, 0]))


class TestCastKernels:
    """Tests for type casting kernels."""

    def test_arrow_cast(self):
        """Test Arrow cast kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 3])
        result = dispatch_kernel("cast", "arrow", arr, pa.float64())
        assert result.type == pa.float64()
        assert result.to_pylist() == [1.0, 2.0, 3.0]

    def test_numpy_cast(self):
        """Test NumPy cast kernel."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 2, 3], dtype=np.int32)
        result = dispatch_kernel("cast", "numpy", arr, np.float64)
        assert result.dtype == np.float64
        tm.assert_numpy_array_equal(result, np.array([1.0, 2.0, 3.0]))


class TestGroupByKernels:
    """Tests for GroupBy aggregation kernels."""

    # Arrow groupby kernels
    def test_arrow_groupby_sum(self):
        """Test Arrow groupby_sum kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = pa.array(["a", "b", "a", "b", "a"])
        values = pa.array([1, 2, 3, 4, 5])
        result_keys, result_values = dispatch_kernel(
            "groupby_sum", "arrow", keys, values
        )

        # Convert to dict for easier comparison (order may vary)
        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {"a": 9, "b": 6}

    def test_arrow_groupby_mean(self):
        """Test Arrow groupby_mean kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = pa.array(["a", "b", "a", "b", "a"])
        values = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result_keys, result_values = dispatch_kernel(
            "groupby_mean", "arrow", keys, values
        )

        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {"a": 3.0, "b": 3.0}

    def test_arrow_groupby_min(self):
        """Test Arrow groupby_min kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = pa.array(["a", "b", "a", "b", "a"])
        values = pa.array([5, 2, 3, 4, 1])
        result_keys, result_values = dispatch_kernel(
            "groupby_min", "arrow", keys, values
        )

        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {"a": 1, "b": 2}

    def test_arrow_groupby_max(self):
        """Test Arrow groupby_max kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = pa.array(["a", "b", "a", "b", "a"])
        values = pa.array([1, 2, 3, 4, 5])
        result_keys, result_values = dispatch_kernel(
            "groupby_max", "arrow", keys, values
        )

        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {"a": 5, "b": 4}

    def test_arrow_groupby_count(self):
        """Test Arrow groupby_count kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = pa.array(["a", "b", "a", "b", "a"])
        values = pa.array([1, 2, 3, 4, 5])
        result_keys, result_values = dispatch_kernel(
            "groupby_count", "arrow", keys, values
        )

        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {"a": 3, "b": 2}

    def test_arrow_groupby_first(self):
        """Test Arrow groupby_first kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = pa.array(["a", "b", "a", "b"])
        values = pa.array([10, 20, 30, 40])
        result_keys, result_values = dispatch_kernel(
            "groupby_first", "arrow", keys, values
        )

        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {"a": 10, "b": 20}

    def test_arrow_groupby_last(self):
        """Test Arrow groupby_last kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = pa.array(["a", "b", "a", "b"])
        values = pa.array([10, 20, 30, 40])
        result_keys, result_values = dispatch_kernel(
            "groupby_last", "arrow", keys, values
        )

        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {"a": 30, "b": 40}

    def test_arrow_groupby_nunique(self):
        """Test Arrow groupby_nunique kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = pa.array(["a", "b", "a", "b", "a"])
        values = pa.array([1, 2, 1, 3, 2])  # a has {1, 2}, b has {2, 3}
        result_keys, result_values = dispatch_kernel(
            "groupby_nunique", "arrow", keys, values
        )

        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {"a": 2, "b": 2}

    # NumPy groupby kernels
    def test_numpy_groupby_sum(self):
        """Test NumPy groupby_sum kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a", "b", "a"])
        values = np.array([1, 2, 3, 4, 5])
        result_keys, result_values = dispatch_kernel(
            "groupby_sum", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 9, "b": 6}

    def test_numpy_groupby_mean(self):
        """Test NumPy groupby_mean kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a", "b", "a"])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result_keys, result_values = dispatch_kernel(
            "groupby_mean", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 3.0, "b": 3.0}

    def test_numpy_groupby_min(self):
        """Test NumPy groupby_min kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a", "b", "a"])
        values = np.array([5, 2, 3, 4, 1])
        result_keys, result_values = dispatch_kernel(
            "groupby_min", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 1, "b": 2}

    def test_numpy_groupby_max(self):
        """Test NumPy groupby_max kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a", "b", "a"])
        values = np.array([1, 2, 3, 4, 5])
        result_keys, result_values = dispatch_kernel(
            "groupby_max", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 5, "b": 4}

    def test_numpy_groupby_count(self):
        """Test NumPy groupby_count kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a", "b", "a"])
        values = np.array([1, 2, 3, 4, 5])
        result_keys, result_values = dispatch_kernel(
            "groupby_count", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 3, "b": 2}

    def test_numpy_groupby_first(self):
        """Test NumPy groupby_first kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a", "b"])
        values = np.array([10, 20, 30, 40])
        result_keys, result_values = dispatch_kernel(
            "groupby_first", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 10, "b": 20}

    def test_numpy_groupby_last(self):
        """Test NumPy groupby_last kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a", "b"])
        values = np.array([10, 20, 30, 40])
        result_keys, result_values = dispatch_kernel(
            "groupby_last", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 30, "b": 40}

    def test_numpy_groupby_prod(self):
        """Test NumPy groupby_prod kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a"])
        values = np.array([2, 3, 4])
        result_keys, result_values = dispatch_kernel(
            "groupby_prod", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 8, "b": 3}

    def test_numpy_groupby_nunique(self):
        """Test NumPy groupby_nunique kernel."""
        from pandas.lazy.backends import dispatch_kernel

        keys = np.array(["a", "b", "a", "b", "a"])
        values = np.array([1, 2, 1, 3, 2])
        result_keys, result_values = dispatch_kernel(
            "groupby_nunique", "numpy", keys, values
        )

        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {"a": 2, "b": 2}

    # Integer keys tests
    def test_groupby_with_integer_keys(self):
        """Test groupby kernels with integer keys."""
        from pandas.lazy.backends import dispatch_kernel

        # Arrow
        keys = pa.array([1, 2, 1, 2, 1])
        values = pa.array([10, 20, 30, 40, 50])
        result_keys, result_values = dispatch_kernel(
            "groupby_sum", "arrow", keys, values
        )
        result = dict(
            zip(result_keys.to_pylist(), result_values.to_pylist(), strict=True)
        )
        assert result == {1: 90, 2: 60}

        # NumPy
        keys = np.array([1, 2, 1, 2, 1])
        values = np.array([10, 20, 30, 40, 50])
        result_keys, result_values = dispatch_kernel(
            "groupby_sum", "numpy", keys, values
        )
        result = dict(zip(result_keys, result_values, strict=True))
        assert result == {1: 90, 2: 60}

    # Hash aggregate (multi-key) tests
    def test_arrow_hash_aggregate_single_agg(self):
        """Test Arrow hash_aggregate with single aggregation."""
        from pandas.lazy.backends import dispatch_kernel

        keys = [pa.array(["a", "b", "a", "b"])]
        values = [pa.array([1, 2, 3, 4])]
        agg_funcs = ["sum"]

        result_keys, result_values = dispatch_kernel(
            "hash_aggregate", "arrow", keys, values, agg_funcs
        )

        # Verify results
        key_result = dict(
            zip(result_keys[0].to_pylist(), result_values[0].to_pylist(), strict=True)
        )
        assert key_result == {"a": 4, "b": 6}

    def test_numpy_hash_aggregate_multi_key(self):
        """Test NumPy hash_aggregate with multiple keys."""
        from pandas.lazy.backends import dispatch_kernel

        keys = [
            np.array(["a", "a", "b", "b"]),
            np.array([1, 2, 1, 2]),
        ]
        values = [np.array([10, 20, 30, 40])]
        agg_funcs = ["sum"]

        result_keys, result_values = dispatch_kernel(
            "hash_aggregate", "numpy", keys, values, agg_funcs
        )

        # With 2 keys and 4 unique combinations, should have 4 groups
        assert len(result_keys[0]) == 4
        assert len(result_values[0]) == 4


class TestGroupByTableKernels:
    """Tests for Arrow table-based groupby kernel."""

    def test_arrow_group_by_single_agg(self):
        """Test Arrow group_by with single aggregation."""
        from pandas.lazy.backends import dispatch_kernel

        table = pa.table(
            {
                "key": ["a", "b", "a", "b", "a"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        result = dispatch_kernel(
            "group_by", "arrow", table, ["key"], [("total", "value", "sum")]
        )

        # Convert to dict for comparison
        result_dict = {}
        for i in range(len(result.column("key"))):
            key = result.column("key")[i].as_py()
            result_dict[key] = result.column("total")[i].as_py()

        assert result_dict == {"a": 9, "b": 6}

    def test_arrow_group_by_multiple_aggs(self):
        """Test Arrow group_by with multiple aggregations."""
        from pandas.lazy.backends import dispatch_kernel

        table = pa.table(
            {
                "key": ["a", "b", "a", "b", "a"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        result = dispatch_kernel(
            "group_by",
            "arrow",
            table,
            ["key"],
            [("total", "value", "sum"), ("avg", "value", "mean")],
        )

        # Should have key, total, avg columns
        assert set(result.column_names) == {"key", "total", "avg"}

    def test_arrow_group_by_multi_key(self):
        """Test Arrow group_by with multiple grouping keys."""
        from pandas.lazy.backends import dispatch_kernel

        table = pa.table(
            {
                "key1": ["a", "a", "b", "b"],
                "key2": [1, 2, 1, 2],
                "value": [10, 20, 30, 40],
            }
        )

        result = dispatch_kernel(
            "group_by", "arrow", table, ["key1", "key2"], [("total", "value", "sum")]
        )

        # Should have 4 unique (key1, key2) combinations
        assert len(result) == 4


class TestJoinKernels:
    """Tests for Join kernels."""

    # Arrow join tests
    def test_arrow_inner_join(self):
        """Test Arrow inner join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        left = pa.table({"key": [1, 2, 3], "a": [10, 20, 30]})
        right = pa.table({"key": [2, 3, 4], "b": [200, 300, 400]})

        result = dispatch_kernel("inner_join", "arrow", left, right, ["key"])

        assert len(result) == 2  # Only keys 2 and 3 match
        assert set(result.column_names) == {"key", "a", "b"}

    def test_arrow_left_join(self):
        """Test Arrow left join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        left = pa.table({"key": [1, 2, 3], "a": [10, 20, 30]})
        right = pa.table({"key": [2, 3, 4], "b": [200, 300, 400]})

        result = dispatch_kernel("left_join", "arrow", left, right, ["key"])

        assert len(result) == 3  # All left rows preserved
        # Key 1 should have null for b. Assert in Arrow directly:
        # Table.to_pandas goes through pyarrow's deprecated
        # DataFrame(BlockManager) construction on some pyarrow builds.
        import pyarrow.compute as pc

        b_for_key1 = result.filter(pc.equal(result["key"], 1)).column("b")
        assert b_for_key1.null_count == len(b_for_key1)

    def test_arrow_right_join(self):
        """Test Arrow right join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        left = pa.table({"key": [1, 2, 3], "a": [10, 20, 30]})
        right = pa.table({"key": [2, 3, 4], "b": [200, 300, 400]})

        result = dispatch_kernel("right_join", "arrow", left, right, ["key"])

        assert len(result) == 3  # All right rows preserved
        # Key 4 should have null for a (Arrow-native assert, see above)
        import pyarrow.compute as pc

        a_for_key4 = result.filter(pc.equal(result["key"], 4)).column("a")
        assert a_for_key4.null_count == len(a_for_key4)

    def test_arrow_outer_join(self):
        """Test Arrow outer join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        left = pa.table({"key": [1, 2, 3], "a": [10, 20, 30]})
        right = pa.table({"key": [2, 3, 4], "b": [200, 300, 400]})

        result = dispatch_kernel("outer_join", "arrow", left, right, ["key"])

        assert len(result) == 4  # All rows from both tables

    def test_arrow_semi_join(self):
        """Test Arrow semi join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        left = pa.table({"key": [1, 2, 3], "a": [10, 20, 30]})
        right = pa.table({"key": [2, 3, 4], "b": [200, 300, 400]})

        result = dispatch_kernel("semi_join", "arrow", left, right, ["key"])

        assert len(result) == 2  # Only keys 2 and 3 from left
        assert "b" not in result.column_names  # No columns from right

    def test_arrow_anti_join(self):
        """Test Arrow anti join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        left = pa.table({"key": [1, 2, 3], "a": [10, 20, 30]})
        right = pa.table({"key": [2, 3, 4], "b": [200, 300, 400]})

        result = dispatch_kernel("anti_join", "arrow", left, right, ["key"])

        assert len(result) == 1  # Only key 1 (no match in right)
        assert result.column("key").to_pylist() == [1]

    def test_arrow_hash_join_different_keys(self):
        """Test Arrow hash join with different key names."""
        from pandas.lazy.backends import dispatch_kernel

        left = pa.table({"left_key": [1, 2, 3], "a": [10, 20, 30]})
        right = pa.table({"right_key": [2, 3, 4], "b": [200, 300, 400]})

        result = dispatch_kernel(
            "hash_join",
            "arrow",
            left,
            right,
            left_keys=["left_key"],
            right_keys=["right_key"],
            join_type="inner",
        )

        assert len(result) == 2

    def test_arrow_join_with_suffixes(self):
        """Test Arrow join handles column name conflicts."""
        from pandas.lazy.backends import dispatch_kernel

        left = pa.table({"key": [1, 2], "value": [10, 20]})
        right = pa.table({"key": [1, 2], "value": [100, 200]})

        result = dispatch_kernel(
            "inner_join",
            "arrow",
            left,
            right,
            ["key"],
            left_suffix="_left",
            right_suffix="_right",
        )

        assert "value_left" in result.column_names or "value" in result.column_names
        assert "value_right" in result.column_names

    # NumPy join tests
    def test_numpy_inner_join(self):
        """Test NumPy inner join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        left = {"key": np.array([1, 2, 3]), "a": np.array([10, 20, 30])}
        right = {"key": np.array([2, 3, 4]), "b": np.array([200, 300, 400])}

        result = dispatch_kernel("inner_join", "numpy", left, right, ["key"])

        assert len(result["key"]) == 2  # Only keys 2 and 3 match
        assert "a" in result
        assert "b" in result

    def test_numpy_left_join(self):
        """Test NumPy left join kernel."""
        from pandas.lazy.backends import dispatch_kernel

        left = {"key": np.array([1, 2, 3]), "a": np.array([10.0, 20.0, 30.0])}
        right = {"key": np.array([2, 3, 4]), "b": np.array([200.0, 300.0, 400.0])}

        result = dispatch_kernel("left_join", "numpy", left, right, ["key"])

        assert len(result["key"]) == 3  # All left rows preserved
        # Check that unmatched rows have NaN
        assert np.isnan(result["b"]).any()

    def test_numpy_hash_join_different_keys(self):
        """Test NumPy hash join with different key names."""
        from pandas.lazy.backends import dispatch_kernel

        left = {"left_key": np.array([1, 2, 3]), "a": np.array([10, 20, 30])}
        right = {"right_key": np.array([2, 3, 4]), "b": np.array([200, 300, 400])}

        result = dispatch_kernel(
            "hash_join",
            "numpy",
            left,
            right,
            left_keys=["left_key"],
            right_keys=["right_key"],
            join_type="inner",
        )

        assert len(result["left_key"]) == 2

    def test_numpy_join_with_duplicates(self):
        """Test NumPy join with duplicate keys."""
        from pandas.lazy.backends import dispatch_kernel

        left = {"key": np.array([1, 1, 2]), "a": np.array([10, 11, 20])}
        right = {"key": np.array([1, 2, 2]), "b": np.array([100, 200, 201])}

        result = dispatch_kernel("inner_join", "numpy", left, right, ["key"])

        # key=1: 2 left rows x 1 right row = 2
        # key=2: 1 left row x 2 right rows = 2
        # Total = 4
        assert len(result["key"]) == 4


class TestCumulativeKernels:
    """Tests for cumulative/window function kernels."""

    def test_arrow_cumulative_sum(self):
        """Test Arrow cumulative sum."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 3, 4, 5])
        result = dispatch_kernel("cumulative_sum", "arrow", arr)
        expected = pa.array([1, 3, 6, 10, 15])
        assert result.equals(expected)

    def test_numpy_cumulative_sum(self):
        """Test NumPy cumulative sum."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 2, 3, 4, 5])
        result = dispatch_kernel("cumulative_sum", "numpy", arr)
        expected = np.array([1, 3, 6, 10, 15])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_cumulative_max(self):
        """Test Arrow cumulative max."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 3, 2, 5, 4])
        result = dispatch_kernel("cumulative_max", "arrow", arr)
        expected = pa.array([1, 3, 3, 5, 5])
        assert result.equals(expected)

    def test_numpy_cumulative_max(self):
        """Test NumPy cumulative max."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 3, 2, 5, 4])
        result = dispatch_kernel("cumulative_max", "numpy", arr)
        expected = np.array([1, 3, 3, 5, 5])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_cumulative_min(self):
        """Test Arrow cumulative min."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([5, 3, 4, 1, 2])
        result = dispatch_kernel("cumulative_min", "arrow", arr)
        expected = pa.array([5, 3, 3, 1, 1])
        assert result.equals(expected)

    def test_numpy_cumulative_min(self):
        """Test NumPy cumulative min."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([5, 3, 4, 1, 2])
        result = dispatch_kernel("cumulative_min", "numpy", arr)
        expected = np.array([5, 3, 3, 1, 1])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_cumulative_prod(self):
        """Test Arrow cumulative product."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1, 2, 3, 4])
        result = dispatch_kernel("cumulative_prod", "arrow", arr)
        expected = pa.array([1, 2, 6, 24])
        assert result.equals(expected)

    def test_numpy_cumulative_prod(self):
        """Test NumPy cumulative product."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1, 2, 3, 4])
        result = dispatch_kernel("cumulative_prod", "numpy", arr)
        expected = np.array([1, 2, 6, 24])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_cumulative_mean(self):
        """Test Arrow cumulative mean."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, 3.0, 5.0, 7.0])
        result = dispatch_kernel("cumulative_mean", "arrow", arr)
        expected = pa.array([1.0, 2.0, 3.0, 4.0])
        # Compare as numpy due to floating point
        tm.assert_almost_equal(result.to_numpy(), expected.to_numpy())

    def test_numpy_cumulative_mean(self):
        """Test NumPy cumulative mean."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 3.0, 5.0, 7.0])
        result = dispatch_kernel("cumulative_mean", "numpy", arr)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        tm.assert_almost_equal(result, expected)


class TestDatetimeKernels:
    """Tests for datetime extraction kernels."""

    def test_arrow_dt_year(self):
        """Test Arrow year extraction."""
        from pandas.lazy.backends import dispatch_kernel

        # Use pa.array with explicit timestamp type
        arr = pa.array(
            pd.to_datetime(["2020-01-15", "2021-06-30", "2022-12-31"]),
            type=pa.timestamp("ns"),
        )
        result = dispatch_kernel("dt_year", "arrow", arr)
        expected = [2020, 2021, 2022]
        assert result.to_pylist() == expected

    def test_numpy_dt_year(self):
        """Test NumPy year extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(
            ["2020-01-15", "2021-06-30", "2022-12-31"], dtype="datetime64[ns]"
        )
        result = dispatch_kernel("dt_year", "numpy", arr)
        expected = np.array([2020, 2021, 2022])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_dt_month(self):
        """Test Arrow month extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(
            pd.to_datetime(["2020-01-15", "2020-06-30", "2020-12-31"]),
            type=pa.timestamp("ns"),
        )
        result = dispatch_kernel("dt_month", "arrow", arr)
        expected = [1, 6, 12]
        assert result.to_pylist() == expected

    def test_numpy_dt_month(self):
        """Test NumPy month extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(
            ["2020-01-15", "2020-06-30", "2020-12-31"], dtype="datetime64[ns]"
        )
        result = dispatch_kernel("dt_month", "numpy", arr)
        expected = np.array([1, 6, 12])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_arrow_dt_day(self):
        """Test Arrow day extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(
            pd.to_datetime(["2020-01-01", "2020-01-15", "2020-01-31"]),
            type=pa.timestamp("ns"),
        )
        result = dispatch_kernel("dt_day", "arrow", arr)
        expected = [1, 15, 31]
        assert result.to_pylist() == expected

    def test_numpy_dt_day(self):
        """Test NumPy day extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(
            ["2020-01-01", "2020-01-15", "2020-01-31"], dtype="datetime64[ns]"
        )
        result = dispatch_kernel("dt_day", "numpy", arr)
        expected = np.array([1, 15, 31])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_arrow_dt_hour(self):
        """Test Arrow hour extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(
            pd.to_datetime(
                ["2020-01-01 00:30:00", "2020-01-01 12:30:00", "2020-01-01 23:30:00"]
            ),
            type=pa.timestamp("ns"),
        )
        result = dispatch_kernel("dt_hour", "arrow", arr)
        expected = [0, 12, 23]
        assert result.to_pylist() == expected

    def test_numpy_dt_hour(self):
        """Test NumPy hour extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(
            ["2020-01-01T00:30:00", "2020-01-01T12:30:00", "2020-01-01T23:30:00"],
            dtype="datetime64[ns]",
        )
        result = dispatch_kernel("dt_hour", "numpy", arr)
        expected = np.array([0, 12, 23])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_arrow_dt_day_of_week(self):
        """Test Arrow day of week extraction."""
        from pandas.lazy.backends import dispatch_kernel

        # 2020-01-06 is Monday (0), 2020-01-07 is Tuesday (1)
        arr = pa.array(
            pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-12"]),
            type=pa.timestamp("ns"),
        )
        result = dispatch_kernel("dt_day_of_week", "arrow", arr, count_from_zero=True)
        expected = [0, 1, 6]
        assert result.to_pylist() == expected

    def test_numpy_dt_day_of_week(self):
        """Test NumPy day of week extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(
            ["2020-01-06", "2020-01-07", "2020-01-12"], dtype="datetime64[ns]"
        )
        result = dispatch_kernel("dt_day_of_week", "numpy", arr, count_from_zero=True)
        expected = np.array([0, 1, 6])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_arrow_dt_quarter(self):
        """Test Arrow quarter extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(
            pd.to_datetime(["2020-01-15", "2020-04-15", "2020-10-15"]),
            type=pa.timestamp("ns"),
        )
        result = dispatch_kernel("dt_quarter", "arrow", arr)
        expected = [1, 2, 4]
        assert result.to_pylist() == expected

    def test_numpy_dt_quarter(self):
        """Test NumPy quarter extraction."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(
            ["2020-01-15", "2020-04-15", "2020-10-15"], dtype="datetime64[ns]"
        )
        result = dispatch_kernel("dt_quarter", "numpy", arr)
        expected = np.array([1, 2, 4])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_arrow_dt_is_leap_year(self):
        """Test Arrow is_leap_year."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(
            pd.to_datetime(["2020-01-01", "2021-01-01", "2024-01-01"]),
            type=pa.timestamp("ns"),
        )
        result = dispatch_kernel("dt_is_leap_year", "arrow", arr)
        expected = [True, False, True]
        assert result.to_pylist() == expected

    def test_numpy_dt_is_leap_year(self):
        """Test NumPy is_leap_year."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(
            ["2020-01-01", "2021-01-01", "2024-01-01"], dtype="datetime64[ns]"
        )
        result = dispatch_kernel("dt_is_leap_year", "numpy", arr)
        expected = np.array([True, False, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_dt_strftime(self):
        """Test Arrow strftime."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(
            pd.to_datetime(["2020-01-15 12:30:45", "2021-06-30 23:59:59"]),
            type=pa.timestamp("ns"),
        )
        result = dispatch_kernel("dt_strftime", "arrow", arr, format="%Y-%m-%d")
        expected = ["2020-01-15", "2021-06-30"]
        assert result.to_pylist() == expected

    def test_numpy_dt_strftime(self):
        """Test NumPy strftime."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(
            ["2020-01-15T12:30:45", "2021-06-30T23:59:59"], dtype="datetime64[ns]"
        )
        result = dispatch_kernel("dt_strftime", "numpy", arr, format="%Y-%m-%d")
        expected = np.array(["2020-01-15", "2021-06-30"])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)


class TestAdditionalStringKernels:
    """Tests for additional string function kernels."""

    def test_arrow_str_capitalize(self):
        """Test Arrow string capitalize."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["hello", "WORLD", "tEsT"])
        result = dispatch_kernel("str_capitalize", "arrow", arr)
        expected = pa.array(["Hello", "World", "Test"])
        assert result.equals(expected)

    def test_numpy_str_capitalize(self):
        """Test NumPy string capitalize."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["hello", "WORLD", "tEsT"])
        result = dispatch_kernel("str_capitalize", "numpy", arr)
        expected = np.array(["Hello", "World", "Test"])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_title(self):
        """Test Arrow string title."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["hello world", "HELLO WORLD"])
        result = dispatch_kernel("str_title", "arrow", arr)
        expected = pa.array(["Hello World", "Hello World"])
        assert result.equals(expected)

    def test_numpy_str_title(self):
        """Test NumPy string title."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["hello world", "HELLO WORLD"])
        result = dispatch_kernel("str_title", "numpy", arr)
        expected = np.array(["Hello World", "Hello World"])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_swapcase(self):
        """Test Arrow string swapcase."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["Hello", "WORLD", "tEsT"])
        result = dispatch_kernel("str_swapcase", "arrow", arr)
        expected = pa.array(["hELLO", "world", "TeSt"])
        assert result.equals(expected)

    def test_numpy_str_swapcase(self):
        """Test NumPy string swapcase."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["Hello", "WORLD", "tEsT"])
        result = dispatch_kernel("str_swapcase", "numpy", arr)
        expected = np.array(["hELLO", "world", "TeSt"])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_is_alpha(self):
        """Test Arrow string is_alpha."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["Hello", "Hello123", "123"])
        result = dispatch_kernel("str_is_alpha", "arrow", arr)
        expected = pa.array([True, False, False])
        assert result.equals(expected)

    def test_numpy_str_is_alpha(self):
        """Test NumPy string is_alpha."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["Hello", "Hello123", "123"])
        result = dispatch_kernel("str_is_alpha", "numpy", arr)
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_is_digit(self):
        """Test Arrow string is_digit."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["123", "12a", "abc"])
        result = dispatch_kernel("str_is_digit", "arrow", arr)
        expected = pa.array([True, False, False])
        assert result.equals(expected)

    def test_numpy_str_is_digit(self):
        """Test NumPy string is_digit."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["123", "12a", "abc"])
        result = dispatch_kernel("str_is_digit", "numpy", arr)
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_count(self):
        """Test Arrow string count."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["hello", "hello hello", "world"])
        result = dispatch_kernel("str_count", "arrow", arr, pattern="hello")
        expected = [1, 2, 0]
        assert result.to_pylist() == expected

    def test_numpy_str_count(self):
        """Test NumPy string count."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["hello", "hello hello", "world"])
        result = dispatch_kernel("str_count", "numpy", arr, pattern="hello")
        expected = np.array([1, 2, 0])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_find(self):
        """Test Arrow string find."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["hello", "world", "help"])
        result = dispatch_kernel("str_find", "arrow", arr, pattern="el")
        expected = [1, -1, 1]
        assert result.to_pylist() == expected

    def test_numpy_str_find(self):
        """Test NumPy string find."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["hello", "world", "help"])
        result = dispatch_kernel("str_find", "numpy", arr, pattern="el")
        expected = np.array([1, -1, 1])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_center(self):
        """Test Arrow string center."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["a", "ab", "abc"])
        result = dispatch_kernel("str_center", "arrow", arr, width=5)
        expected = pa.array(["  a  ", " ab  ", " abc "])
        assert result.equals(expected)

    def test_numpy_str_center(self):
        """Test NumPy string center."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["a", "ab", "abc"])
        result = dispatch_kernel("str_center", "numpy", arr, width=5)
        # NumPy center puts extra space on the right for odd padding
        expected = np.array(["  a  ", "  ab ", " abc "])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_zfill(self):
        """Test Arrow string zfill."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["1", "12", "123"])
        result = dispatch_kernel("str_zfill", "arrow", arr, width=5)
        expected = pa.array(["00001", "00012", "00123"])
        assert result.equals(expected)

    def test_numpy_str_zfill(self):
        """Test NumPy string zfill."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["1", "12", "123"])
        result = dispatch_kernel("str_zfill", "numpy", arr, width=5)
        expected = np.array(["00001", "00012", "00123"])
        tm.assert_numpy_array_equal(result, expected)

    def test_arrow_str_match(self):
        """Test Arrow string match."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["hello", "world", "helper"])
        result = dispatch_kernel("str_match", "arrow", arr, pattern="hell.*")
        expected = pa.array([True, False, False])
        assert result.equals(expected)

    def test_numpy_str_match(self):
        """Test NumPy string match."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array(["hello", "world", "helper"])
        result = dispatch_kernel("str_match", "numpy", arr, pattern="hell.*")
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)


class TestRollingWindowKernels:
    """Tests for rolling window function kernels."""

    def test_numpy_rolling_sum(self):
        """Test NumPy rolling sum."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("rolling_sum", "numpy", arr, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 6.0
        assert result[3] == 9.0
        assert result[4] == 12.0

    def test_numpy_rolling_mean(self):
        """Test NumPy rolling mean."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("rolling_mean", "numpy", arr, window=3)
        assert result[2] == 2.0
        assert result[3] == 3.0
        assert result[4] == 4.0

    def test_numpy_rolling_min(self):
        """Test NumPy rolling min."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = dispatch_kernel("rolling_min", "numpy", arr, window=3)
        assert result[2] == 1.0
        assert result[3] == 1.0
        assert result[4] == 1.0

    def test_numpy_rolling_max(self):
        """Test NumPy rolling max."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = dispatch_kernel("rolling_max", "numpy", arr, window=3)
        assert result[2] == 4.0
        assert result[3] == 4.0
        assert result[4] == 5.0

    def test_numpy_rolling_std(self):
        """Test NumPy rolling std."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("rolling_std", "numpy", arr, window=3)
        assert abs(result[2] - 1.0) < 1e-10
        assert abs(result[3] - 1.0) < 1e-10
        assert abs(result[4] - 1.0) < 1e-10

    def test_numpy_rolling_var(self):
        """Test NumPy rolling var."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("rolling_var", "numpy", arr, window=3)
        assert abs(result[2] - 1.0) < 1e-10
        assert abs(result[3] - 1.0) < 1e-10
        assert abs(result[4] - 1.0) < 1e-10

    def test_numpy_rolling_with_min_periods(self):
        """Test NumPy rolling with min_periods."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("rolling_sum", "numpy", arr, window=3, min_periods=1)
        assert result[0] == 1.0
        assert result[1] == 3.0
        assert result[2] == 6.0

    def test_numpy_rolling_with_nan(self):
        """Test NumPy rolling with NaN values."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = dispatch_kernel("rolling_sum", "numpy", arr, window=3, min_periods=2)
        assert result[2] == 4.0
        assert result[3] == 7.0


class TestShiftKernels:
    """Tests for shift/lag/lead kernels."""

    def test_arrow_shift_forward(self):
        """Test Arrow shift forward."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("shift", "arrow", arr, periods=2)
        result_list = result.to_pylist()
        assert result_list[0] is None or np.isnan(result_list[0])
        assert result_list[1] is None or np.isnan(result_list[1])
        assert result_list[2] == 1.0
        assert result_list[3] == 2.0
        assert result_list[4] == 3.0

    def test_numpy_shift_forward(self):
        """Test NumPy shift forward."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("shift", "numpy", arr, periods=2)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 1.0
        assert result[3] == 2.0
        assert result[4] == 3.0

    def test_arrow_shift_backward(self):
        """Test Arrow shift backward."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("shift", "arrow", arr, periods=-2)
        result_list = result.to_pylist()
        assert result_list[0] == 3.0
        assert result_list[1] == 4.0
        assert result_list[2] == 5.0
        assert result_list[3] is None or np.isnan(result_list[3])
        assert result_list[4] is None or np.isnan(result_list[4])

    def test_numpy_shift_backward(self):
        """Test NumPy shift backward."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("shift", "numpy", arr, periods=-2)
        assert result[0] == 3.0
        assert result[1] == 4.0
        assert result[2] == 5.0
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_arrow_shift_with_fill(self):
        """Test Arrow shift with fill value."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("shift", "arrow", arr, periods=2, fill_value=0.0)
        result_list = result.to_pylist()
        assert result_list[0] == 0.0
        assert result_list[1] == 0.0
        assert result_list[2] == 1.0

    def test_numpy_shift_with_fill(self):
        """Test NumPy shift with fill value."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("shift", "numpy", arr, periods=2, fill_value=0.0)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 1.0

    def test_arrow_lag(self):
        """Test Arrow lag function."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("lag", "arrow", arr, periods=1)
        result_list = result.to_pylist()
        assert result_list[0] is None or np.isnan(result_list[0])
        assert result_list[1] == 1.0
        assert result_list[2] == 2.0

    def test_numpy_lag(self):
        """Test NumPy lag function."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("lag", "numpy", arr, periods=1)
        assert np.isnan(result[0])
        assert result[1] == 1.0
        assert result[2] == 2.0

    def test_arrow_lead(self):
        """Test Arrow lead function."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("lead", "arrow", arr, periods=1)
        result_list = result.to_pylist()
        assert result_list[0] == 2.0
        assert result_list[1] == 3.0
        assert result_list[4] is None or np.isnan(result_list[4])

    def test_numpy_lead(self):
        """Test NumPy lead function."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dispatch_kernel("lead", "numpy", arr, periods=1)
        assert result[0] == 2.0
        assert result[1] == 3.0
        assert np.isnan(result[4])


class TestFillNAKernels:
    """Tests for fill NA variant kernels."""

    def test_arrow_ffill(self):
        """Test Arrow forward fill."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, None, None, 4.0, None])
        result = dispatch_kernel("ffill", "arrow", arr)
        result_list = result.to_pylist()
        assert result_list[0] == 1.0
        assert result_list[1] == 1.0
        assert result_list[2] == 1.0
        assert result_list[3] == 4.0
        assert result_list[4] == 4.0

    def test_numpy_ffill(self):
        """Test NumPy forward fill."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        result = dispatch_kernel("ffill", "numpy", arr)
        assert result[0] == 1.0
        assert result[1] == 1.0
        assert result[2] == 1.0
        assert result[3] == 4.0
        assert result[4] == 4.0

    def test_arrow_ffill_with_limit(self):
        """Test Arrow forward fill with limit."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, None, None, None, 5.0])
        result = dispatch_kernel("ffill", "arrow", arr, limit=1)
        result_list = result.to_pylist()
        assert result_list[0] == 1.0
        assert result_list[1] == 1.0  # filled
        assert result_list[2] is None or np.isnan(result_list[2])  # not filled
        assert result_list[3] is None or np.isnan(result_list[3])  # not filled
        assert result_list[4] == 5.0

    def test_numpy_ffill_with_limit(self):
        """Test NumPy forward fill with limit."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        result = dispatch_kernel("ffill", "numpy", arr, limit=1)
        assert result[0] == 1.0
        assert result[1] == 1.0  # filled
        assert np.isnan(result[2])  # not filled
        assert np.isnan(result[3])  # not filled
        assert result[4] == 5.0

    def test_arrow_bfill(self):
        """Test Arrow backward fill."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([None, 2.0, None, None, 5.0])
        result = dispatch_kernel("bfill", "arrow", arr)
        result_list = result.to_pylist()
        assert result_list[0] == 2.0
        assert result_list[1] == 2.0
        assert result_list[2] == 5.0
        assert result_list[3] == 5.0
        assert result_list[4] == 5.0

    def test_numpy_bfill(self):
        """Test NumPy backward fill."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([np.nan, 2.0, np.nan, np.nan, 5.0])
        result = dispatch_kernel("bfill", "numpy", arr)
        assert result[0] == 2.0
        assert result[1] == 2.0
        assert result[2] == 5.0
        assert result[3] == 5.0
        assert result[4] == 5.0

    def test_arrow_bfill_with_limit(self):
        """Test Arrow backward fill with limit."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([None, None, None, 4.0, None])
        result = dispatch_kernel("bfill", "arrow", arr, limit=1)
        result_list = result.to_pylist()
        assert result_list[0] is None or np.isnan(result_list[0])  # not filled
        assert result_list[1] is None or np.isnan(result_list[1])  # not filled
        assert result_list[2] == 4.0  # filled
        assert result_list[3] == 4.0
        assert result_list[4] is None or np.isnan(result_list[4])

    def test_numpy_bfill_with_limit(self):
        """Test NumPy backward fill with limit."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([np.nan, np.nan, np.nan, 4.0, np.nan])
        result = dispatch_kernel("bfill", "numpy", arr, limit=1)
        assert np.isnan(result[0])  # not filled
        assert np.isnan(result[1])  # not filled
        assert result[2] == 4.0  # filled
        assert result[3] == 4.0
        assert np.isnan(result[4])

    def test_arrow_interpolate_linear(self):
        """Test Arrow linear interpolation."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([1.0, None, None, 4.0, None])
        result = dispatch_kernel("interpolate_linear", "arrow", arr)
        result_list = result.to_pylist()
        assert result_list[0] == 1.0
        assert result_list[1] == 2.0  # interpolated
        assert result_list[2] == 3.0  # interpolated
        assert result_list[3] == 4.0
        # Trailing NaN should remain NaN
        assert result_list[4] is None or np.isnan(result_list[4])

    def test_numpy_interpolate_linear(self):
        """Test NumPy linear interpolation."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        result = dispatch_kernel("interpolate_linear", "numpy", arr)
        assert result[0] == 1.0
        assert result[1] == 2.0  # interpolated
        assert result[2] == 3.0  # interpolated
        assert result[3] == 4.0
        # Trailing NaN should remain NaN
        assert np.isnan(result[4])

    def test_arrow_interpolate_leading_nan(self):
        """Test Arrow interpolation preserves leading NaN."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array([None, None, 3.0, 4.0, 5.0])
        result = dispatch_kernel("interpolate_linear", "arrow", arr)
        result_list = result.to_pylist()
        # Leading NaN should remain NaN
        assert result_list[0] is None or np.isnan(result_list[0])
        assert result_list[1] is None or np.isnan(result_list[1])
        assert result_list[2] == 3.0
        assert result_list[3] == 4.0
        assert result_list[4] == 5.0

    def test_numpy_interpolate_leading_nan(self):
        """Test NumPy interpolation preserves leading NaN."""
        from pandas.lazy.backends import dispatch_kernel

        arr = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        result = dispatch_kernel("interpolate_linear", "numpy", arr)
        # Leading NaN should remain NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 3.0
        assert result[3] == 4.0
        assert result[4] == 5.0


class TestArrowStringConcatenation:
    """Tests for Arrow string concatenation with type coercion."""

    def test_arrow_string_concat_with_literal(self):
        """Test concatenation of Arrow large_string with Python string literal."""
        from pandas.lazy.backends import dispatch_kernel

        arr = pa.array(["hello", "world"], type=pa.large_string())
        result = dispatch_kernel("add", "arrow", arr, "_suffix")

        # Result should be large_string (same as input)
        assert pa.types.is_large_string(result.type)
        assert result.to_pylist() == ["hello_suffix", "world_suffix"]

    def test_arrow_string_concat_two_arrays(self):
        """Test concatenation of two Arrow string arrays."""
        from pandas.lazy.backends import dispatch_kernel

        arr1 = pa.array(["hello", "world"], type=pa.large_string())
        arr2 = pa.array(["_a", "_b"], type=pa.large_string())
        result = dispatch_kernel("add", "arrow", arr1, arr2)

        assert pa.types.is_large_string(result.type)
        assert result.to_pylist() == ["hello_a", "world_b"]

    def test_arrow_string_concat_mixed_types(self):
        """Test concatenation of large_string with regular string array."""
        from pandas.lazy.backends import dispatch_kernel

        arr1 = pa.array(["hello", "world"], type=pa.large_string())
        arr2 = pa.array(["_a", "_b"], type=pa.string())
        result = dispatch_kernel("add", "arrow", arr1, arr2)

        # Should coerce to large_string (the larger type)
        assert pa.types.is_large_string(result.type)
        assert result.to_pylist() == ["hello_a", "world_b"]

    def test_physical_planner_string_concat_with_literal(self):
        """Test string concat works through physical planner with Arrow-backed data."""
        df = pd.DataFrame(
            {
                "region": pd.array(["North", "South", "East"], dtype="string[pyarrow]"),
                "category": pd.array(["A", "B", "C"], dtype="string[pyarrow]"),
            }
        )

        from pandas.lazy import col

        lf = df.select()
        lf = lf.with_columns((col("region") + "_" + col("category")).alias("combined"))
        result = lf.collect(use_physical_planner=True)

        expected_combined = ["North_A", "South_B", "East_C"]
        assert result["combined"].tolist() == expected_combined

    def test_physical_planner_string_concat_after_str_upper(self):
        """Test string concat after str.upper() works with physical planner."""
        df = pd.DataFrame(
            {
                "region": pd.array(["North", "South"], dtype="string[pyarrow]"),
                "category": pd.array(["a", "b"], dtype="string[pyarrow]"),
            }
        )

        from pandas.lazy import col

        lf = df.select()
        lf = lf.with_columns(
            col("region").str.upper().alias("region_upper"),
        )
        lf = lf.with_columns(
            (col("region_upper") + "_" + col("category")).alias("combined")
        )
        result = lf.collect(use_physical_planner=True)

        expected_combined = ["NORTH_a", "SOUTH_b"]
        assert result["combined"].tolist() == expected_combined


class TestKeyEncodingCache:
    """Dictionary-encoding cache for string aggregation keys: encode on
    second occurrence of the same immutable source buffers, serve the
    cached dictionary thereafter, decode result keys so output dtype is
    unchanged."""

    def _query(self, df):
        from pandas.lazy import col

        return df.select().group_by("g").agg(col("v").sum().alias("s"))

    def test_lifecycle_and_correctness(self):
        import numpy as np

        from pandas.lazy.backends import keycache

        keycache.clear_cache()
        # shrink the threshold so test-sized data exercises the cache
        orig = keycache.MIN_ENCODE_ROWS
        keycache.MIN_ENCODE_ROWS = 100
        try:
            rng = np.random.default_rng(21)
            df = pd.DataFrame(
                {
                    "g": rng.choice(["a", "b", "c"], 5_000),
                    "v": rng.standard_normal(5_000),
                }
            )
            q = self._query(df)
            r1 = q.collect(use_physical_planner=True)
            assert keycache.cache_stats() == {"encoded": 0, "candidates": 1}
            r2 = q.collect(use_physical_planner=True)
            assert keycache.cache_stats() == {"encoded": 1, "candidates": 0}
            r3 = q.collect(use_physical_planner=True)

            eager = df.groupby("g")["v"].sum().sort_index()
            for r in (r1, r2, r3):
                assert str(r["g"].dtype) in ("string", "str")  # decoded back
                got = r.sort_values("g")["s"].to_numpy(dtype="float64")
                assert np.allclose(got, eager.to_numpy())
        finally:
            keycache.MIN_ENCODE_ROWS = orig
            keycache.clear_cache()

    def test_different_sources_do_not_collide(self):
        import numpy as np

        from pandas.lazy.backends import keycache

        keycache.clear_cache()
        orig = keycache.MIN_ENCODE_ROWS
        keycache.MIN_ENCODE_ROWS = 100
        try:
            rng = np.random.default_rng(22)
            df1 = pd.DataFrame(
                {"g": rng.choice(["x", "y"], 1_000), "v": rng.standard_normal(1_000)}
            )
            df2 = pd.DataFrame(
                {"g": rng.choice(["p", "q"], 1_000), "v": rng.standard_normal(1_000)}
            )
            for df in (df1, df2):
                q = self._query(df)
                q.collect(use_physical_planner=True)
                q.collect(use_physical_planner=True)
            # both sources independently encoded, results correct
            assert keycache.cache_stats()["encoded"] == 2
            r2 = self._query(df2).collect(use_physical_planner=True)
            assert set(r2["g"]) == {"p", "q"}
        finally:
            keycache.MIN_ENCODE_ROWS = orig
            keycache.clear_cache()

    def test_below_threshold_never_encodes(self):
        import numpy as np

        from pandas.lazy.backends import keycache

        keycache.clear_cache()
        rng = np.random.default_rng(23)
        df = pd.DataFrame(
            {"g": rng.choice(["x", "y"], 1_000), "v": rng.standard_normal(1_000)}
        )
        q = self._query(df)
        q.collect(use_physical_planner=True)
        q.collect(use_physical_planner=True)
        assert keycache.cache_stats() == {"encoded": 0, "candidates": 0}
