"""
Additional tests for ArrayEvaluator to improve coverage.

Focuses on edge cases and code paths not covered by existing tests.
"""

import numpy as np
import pyarrow as pa
import pytest

import pandas as pd
import pandas._testing as tm


class TestArrayEvaluatorCast:
    """Tests for Cast node evaluation in ArrayEvaluator."""

    def test_cast_numpy_to_float(self):
        """Test casting NumPy int array to float."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Cast,
            FieldRef,
        )

        arrays = {"a": np.array([1, 2, 3], dtype=np.int32)}
        evaluator = ArrayEvaluator(arrays)

        node = Cast(arg=FieldRef("a"), target_dtype="float64")
        result = evaluator.evaluate(node)
        assert result.dtype == np.float64
        tm.assert_numpy_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_cast_numpy_to_int(self):
        """Test casting NumPy float array to int."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Cast,
            FieldRef,
        )

        arrays = {"a": np.array([1.5, 2.5, 3.5])}
        evaluator = ArrayEvaluator(arrays)

        node = Cast(arg=FieldRef("a"), target_dtype="int64")
        result = evaluator.evaluate(node)
        assert result.dtype == np.int64

    def test_cast_arrow_to_float(self):
        """Test casting Arrow int array to float."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Cast,
            FieldRef,
        )

        arrays = {"a": pa.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        node = Cast(arg=FieldRef("a"), target_dtype="float64")
        result = evaluator.evaluate(node)
        assert result.type == pa.float64()

    def test_cast_arrow_to_string(self):
        """Test casting Arrow int array to string."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Cast,
            FieldRef,
        )

        arrays = {"a": pa.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        node = Cast(arg=FieldRef("a"), target_dtype="string")
        result = evaluator.evaluate(node)
        assert result.type == pa.string()

    def test_cast_arrow_chunked_array(self):
        """Test casting Arrow ChunkedArray."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Cast,
            FieldRef,
        )

        arrays = {"a": pa.chunked_array([[1, 2], [3, 4]])}
        evaluator = ArrayEvaluator(arrays)

        node = Cast(arg=FieldRef("a"), target_dtype="float64")
        result = evaluator.evaluate(node)
        assert result.type == pa.float64()


class TestArrayEvaluatorCaseWhen:
    """Tests for case_when evaluation."""

    def test_case_when_numpy_simple(self):
        """Test case_when with NumPy arrays."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"x": np.array([1, 5, 10, 15])}
        evaluator = ArrayEvaluator(arrays)

        # case when x > 10 then 100, when x > 3 then 50, else 0
        node = Call(
            "case_when",
            args=(),
            kwargs={
                "cases": (
                    (
                        Call("greater", args=(FieldRef("x"), Literal(10))),
                        Literal(100),
                    ),
                    (Call("greater", args=(FieldRef("x"), Literal(3))), Literal(50)),
                ),
                "otherwise": Literal(0),
            },
        )
        result = evaluator.evaluate(node)
        expected = np.array([0, 50, 50, 100])
        tm.assert_numpy_array_equal(result, expected)

    def test_case_when_arrow(self):
        """Test case_when with Arrow arrays."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        # Use preferred_backend=arrow to ensure arrow path
        arrays = {"x": pa.array([1, 5, 10, 15])}
        evaluator = ArrayEvaluator(arrays, preferred_backend="arrow")

        node = Call(
            "case_when",
            args=(),
            kwargs={
                "cases": (
                    (
                        Call("greater", args=(FieldRef("x"), Literal(10))),
                        Literal(100),
                    ),
                ),
                "otherwise": FieldRef(
                    "x"
                ),  # Use array as otherwise to ensure arrow path
            },
        )
        result = evaluator.evaluate(node)
        # Result can be arrow or numpy depending on internals
        assert len(result) == 4

    def test_case_when_with_column_value(self):
        """Test case_when with column reference in value."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"x": np.array([1, 5, 10]), "y": np.array([100, 200, 300])}
        evaluator = ArrayEvaluator(arrays)

        # case when x > 3 then y, else 0
        node = Call(
            "case_when",
            args=(),
            kwargs={
                "cases": (
                    (Call("greater", args=(FieldRef("x"), Literal(3))), FieldRef("y")),
                ),
                "otherwise": Literal(0),
            },
        )
        result = evaluator.evaluate(node)
        expected = np.array([0, 200, 300])
        tm.assert_numpy_array_equal(result, expected)


class TestArrayEvaluatorDatetime:
    """Tests for datetime operations in ArrayEvaluator."""

    def test_dt_year_arrow(self):
        """Test datetime year extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-15", "2021-06-30"]), type=pa.timestamp("ns")
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_year", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [2020, 2021]

    def test_dt_month_arrow(self):
        """Test datetime month extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-15", "2020-06-30"]), type=pa.timestamp("ns")
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_month", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [1, 6]

    def test_dt_day_arrow(self):
        """Test datetime day extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-15", "2020-01-30"]), type=pa.timestamp("ns")
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_day", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [15, 30]

    def test_dt_hour_arrow(self):
        """Test datetime hour extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-15 10:30:00", "2020-01-15 23:45:00"]),
            type=pa.timestamp("ns"),
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_hour", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [10, 23]

    def test_dt_minute_arrow(self):
        """Test datetime minute extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-15 10:30:00", "2020-01-15 10:45:00"]),
            type=pa.timestamp("ns"),
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_minute", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [30, 45]

    def test_dt_second_arrow(self):
        """Test datetime second extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-15 10:30:15", "2020-01-15 10:30:45"]),
            type=pa.timestamp("ns"),
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_second", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [15, 45]

    def test_dt_weekday_arrow(self):
        """Test datetime weekday extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        # 2020-01-06 is Monday (0)
        arr = pa.array(
            pd.to_datetime(["2020-01-06", "2020-01-07"]), type=pa.timestamp("ns")
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_weekday", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [0, 1]

    def test_dt_dayofyear_arrow(self):
        """Test datetime day of year extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-01", "2020-12-31"]), type=pa.timestamp("ns")
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_dayofyear", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [1, 366]  # 2020 is leap year

    def test_dt_quarter_arrow(self):
        """Test datetime quarter extraction with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-15", "2020-04-15", "2020-10-15"]),
            type=pa.timestamp("ns"),
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_quarter", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [1, 2, 4]

    def test_dt_is_month_start_arrow(self):
        """Test datetime is_month_start with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-01", "2020-01-15"]), type=pa.timestamp("ns")
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_is_month_start", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [True, False]

    def test_dt_is_year_start_arrow(self):
        """Test datetime is_year_start with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = pa.array(
            pd.to_datetime(["2020-01-01", "2020-02-01"]), type=pa.timestamp("ns")
        )
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_is_year_start", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        assert result.to_pylist() == [True, False]

    def test_dt_with_numpy_array(self):
        """Test datetime extraction with NumPy array."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arr = np.array(["2020-01-15", "2021-06-30"], dtype="datetime64[ns]")
        arrays = {"date": arr}
        evaluator = ArrayEvaluator(arrays)

        node = Call("dt_year", args=(FieldRef("date"),))
        result = evaluator.evaluate(node)
        tm.assert_numpy_array_equal(result, np.array([2020, 2021]))


class TestArrayEvaluatorCumulative:
    """Tests for cumulative operations in ArrayEvaluator."""

    def test_cum_sum_numpy(self):
        """Test cumulative sum with NumPy."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": np.array([1, 2, 3, 4, 5])}
        evaluator = ArrayEvaluator(arrays)

        node = Call("cum_sum", args=(FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = np.array([1, 3, 6, 10, 15])
        tm.assert_numpy_array_equal(result, expected)

    def test_cum_sum_arrow(self):
        """Test cumulative sum with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": pa.array([1, 2, 3, 4, 5])}
        evaluator = ArrayEvaluator(arrays)

        node = Call("cum_sum", args=(FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = pa.array([1, 3, 6, 10, 15])
        assert result.equals(expected)

    def test_cum_min_numpy(self):
        """Test cumulative min with NumPy."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": np.array([5, 3, 4, 1, 2])}
        evaluator = ArrayEvaluator(arrays)

        node = Call("cum_min", args=(FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = np.array([5, 3, 3, 1, 1])
        tm.assert_numpy_array_equal(result, expected)

    def test_cum_min_arrow(self):
        """Test cumulative min with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": pa.array([5, 3, 4, 1, 2])}
        evaluator = ArrayEvaluator(arrays)

        node = Call("cum_min", args=(FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = pa.array([5, 3, 3, 1, 1])
        assert result.equals(expected)

    def test_cum_max_numpy(self):
        """Test cumulative max with NumPy."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": np.array([1, 3, 2, 5, 4])}
        evaluator = ArrayEvaluator(arrays)

        node = Call("cum_max", args=(FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = np.array([1, 3, 3, 5, 5])
        tm.assert_numpy_array_equal(result, expected)

    def test_cum_max_arrow(self):
        """Test cumulative max with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": pa.array([1, 3, 2, 5, 4])}
        evaluator = ArrayEvaluator(arrays)

        node = Call("cum_max", args=(FieldRef("a"),))
        result = evaluator.evaluate(node)
        expected = pa.array([1, 3, 3, 5, 5])
        assert result.equals(expected)


class TestArrayEvaluatorShift:
    """Tests for shift operations (lag/lead) in ArrayEvaluator."""

    def test_lag_numpy(self):
        """Test lag operation with NumPy."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        evaluator = ArrayEvaluator(arrays)

        # Use 'periods' which is the kernel parameter name
        node = Call("lag", args=(FieldRef("a"),), kwargs={"periods": 2})
        result = evaluator.evaluate(node)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 1.0
        assert result[3] == 2.0
        assert result[4] == 3.0

    def test_lag_arrow(self):
        """Test lag operation with Arrow."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": pa.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        evaluator = ArrayEvaluator(arrays)

        # Use 'periods' which is the kernel parameter name
        node = Call("lag", args=(FieldRef("a"),), kwargs={"periods": 2})
        result = evaluator.evaluate(node)
        assert isinstance(result, (pa.Array, pa.ChunkedArray))

    def test_lead_numpy(self):
        """Test lead operation with NumPy."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        evaluator = ArrayEvaluator(arrays)

        # Use 'periods' which is the kernel parameter name
        node = Call("lead", args=(FieldRef("a"),), kwargs={"periods": 2})
        result = evaluator.evaluate(node)
        assert result[0] == 3.0
        assert result[1] == 4.0
        assert result[2] == 5.0
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_lag_with_fill_value(self):
        """Test lag operation with fill_value."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": np.array([1.0, 2.0, 3.0])}
        evaluator = ArrayEvaluator(arrays)

        # Kernels use fill_value, not default
        node = Call(
            "lag", args=(FieldRef("a"),), kwargs={"periods": 1, "fill_value": 0}
        )
        result = evaluator.evaluate(node)
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[2] == 2.0


class TestArrayEvaluatorRowIndex:
    """Tests for row_index operation in ArrayEvaluator."""

    def test_row_index_numpy(self):
        """Test row_index with NumPy backend."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import Call

        arrays = {"a": np.array([10, 20, 30])}
        evaluator = ArrayEvaluator(arrays, preferred_backend="numpy")

        node = Call("row_index", args=(), kwargs={"offset": 0})
        result = evaluator.evaluate(node)
        assert isinstance(result, np.ndarray)
        tm.assert_numpy_array_equal(result, np.array([0, 1, 2]))

    def test_row_index_with_offset(self):
        """Test row_index with offset."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import Call

        arrays = {"a": np.array([10, 20, 30])}
        evaluator = ArrayEvaluator(arrays, preferred_backend="numpy")

        node = Call("row_index", args=(), kwargs={"offset": 100})
        result = evaluator.evaluate(node)
        tm.assert_numpy_array_equal(result, np.array([100, 101, 102]))

    def test_row_index_arrow(self):
        """Test row_index with Arrow backend."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import Call

        arrays = {"a": pa.array([10, 20, 30])}
        evaluator = ArrayEvaluator(arrays, preferred_backend="arrow")

        node = Call("row_index", args=(), kwargs={"offset": 0})
        result = evaluator.evaluate(node)
        assert isinstance(result, (pa.Array, pa.ChunkedArray))
        assert result.to_pylist() == [0, 1, 2]


class TestArrayEvaluatorPoolingStrategy:
    """Tests for different pooling strategies."""

    def test_scratch_pooling(self):
        """Test evaluator with scratch pooling strategy."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.memory_pool import PoolingStrategy
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(
            arrays, preferred_backend="numpy", pooling_strategy=PoolingStrategy.SCRATCH
        )

        node = Call("add", args=(FieldRef("a"), Literal(10)))
        result = evaluator.evaluate(node)
        tm.assert_numpy_array_equal(result, np.array([11, 12, 13]))

    def test_no_pooling(self):
        """Test evaluator with no pooling strategy."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.memory_pool import PoolingStrategy
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(
            arrays, preferred_backend="numpy", pooling_strategy=PoolingStrategy.NONE
        )

        node = Call("add", args=(FieldRef("a"), Literal(10)))
        result = evaluator.evaluate(node)
        tm.assert_numpy_array_equal(result, np.array([11, 12, 13]))

    def test_pooling_strategy_string(self):
        """Test evaluator with string pooling strategy."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(
            arrays, preferred_backend="numpy", pooling_strategy="none"
        )

        node = Call("add", args=(FieldRef("a"), Literal(10)))
        result = evaluator.evaluate(node)
        tm.assert_numpy_array_equal(result, np.array([11, 12, 13]))


class TestArrayEvaluatorUnknownNode:
    """Tests for error handling with unknown nodes."""

    def test_unknown_ir_node_raises(self):
        """Test that unknown IR node raises NotImplementedError."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        class UnknownNode:
            pass

        with pytest.raises(NotImplementedError, match="Unknown IR node type"):
            evaluator.evaluate(UnknownNode())

    def test_unknown_function_raises(self):
        """Test that unknown function raises NotImplementedError."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import (
            Call,
            FieldRef,
        )

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        node = Call("nonexistent_function_xyz", args=(FieldRef("a"),))
        with pytest.raises(NotImplementedError, match="not implemented"):
            evaluator.evaluate(node)


class TestArrayEvaluatorEmptyArrays:
    """Tests for edge cases with empty arrays."""

    def test_row_index_empty_arrays(self):
        """Test row_index with empty arrays dict."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.ir import Call

        arrays = {}
        evaluator = ArrayEvaluator(arrays)

        node = Call("row_index", args=(), kwargs={"offset": 0})
        result = evaluator.evaluate(node)
        assert len(result) == 0


class TestArrayEvaluatorBackendSelection:
    """Tests for backend selection logic."""

    def test_scalar_only_args_default_to_numpy(self):
        """Test that scalar-only args default to numpy backend."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        # This would be unusual but tests the code path
        backend = evaluator._get_backend_for_func("add", [42, 10])
        assert backend == "numpy"  # Default when no array inputs

    def test_mixed_array_types_use_first_found(self):
        """Test that mixed arrays use first found backend."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator

        arrays = {"a": np.array([1, 2, 3])}
        evaluator = ArrayEvaluator(arrays)

        # Arrow array first
        backend = evaluator._get_backend_for_func(
            "add", [pa.array([1, 2, 3]), np.array([4, 5, 6])]
        )
        assert backend == "arrow"

        # NumPy array first
        backend = evaluator._get_backend_for_func(
            "add", [np.array([1, 2, 3]), pa.array([4, 5, 6])]
        )
        assert backend == "numpy"
