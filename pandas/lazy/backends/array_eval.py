"""
Array-based expression evaluation for lazy pandas.

This module provides the ArrayEvaluator class that executes IR nodes
directly on arrays (Arrow or NumPy) using the kernel dispatch system.

Unlike the DataFrame-based Evaluator, this evaluator:
- Works directly on ArrayDict (dict of arrays)
- Dispatches to registered kernels (Arrow or NumPy)
- Avoids pandas overhead for supported operations
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np
import pyarrow as pa

from pandas.lazy.backends import (
    dispatch_kernel,
    has_kernel,
)
from pandas.lazy.backends.convert import (
    ensure_backend,
    get_array_backend,
    is_arrow_backed,
)
from pandas.lazy.backends.router import (
    decide_expr_backend,
)

if TYPE_CHECKING:
    from pandas.lazy.backends.types import (
        ArrayDict,
        ArrayLike,
    )
    from pandas.lazy.ir import IRNode


class ArrayEvaluator:
    """
    Evaluates IR expressions directly on arrays.

    Uses kernel dispatch to execute operations on Arrow or NumPy arrays
    without going through pandas DataFrame operations.
    """

    def __init__(
        self,
        arrays: ArrayDict,
        preferred_backend: Literal["auto", "arrow", "numpy"] = "auto",
    ) -> None:
        """
        Initialize evaluator with arrays.

        Parameters
        ----------
        arrays : ArrayDict
            Dictionary mapping column names to arrays.
        preferred_backend : {"auto", "arrow", "numpy"}
            Preferred execution backend.
        """
        self._arrays = arrays
        self._preferred_backend = preferred_backend

    def evaluate(self, node: IRNode) -> ArrayLike | Any:
        """
        Evaluate an IR node and return the result.

        Parameters
        ----------
        node : IRNode
            The IR node to evaluate.

        Returns
        -------
        ArrayLike or scalar
            The result of the evaluation.
        """
        from pandas.lazy.ir import (
            Alias,
            Call,
            Cast,
            FieldRef,
            Literal,
        )

        if isinstance(node, FieldRef):
            return self._arrays[node.name]

        elif isinstance(node, Literal):
            return node.value

        elif isinstance(node, Alias):
            return self.evaluate(node.arg)

        elif isinstance(node, Call):
            return self._evaluate_call(node)

        elif isinstance(node, Cast):
            result = self.evaluate(node.arg)
            return self._cast_array(result, node.target_dtype)

        else:
            raise NotImplementedError(f"Unknown IR node type: {type(node)}")

    def _get_backend_for_func(
        self, func: str, input_arrays: list[ArrayLike]
    ) -> Literal["arrow", "numpy"]:
        """Determine which backend to use for a function."""
        # Check if any input is an array (vs scalar)
        array_inputs = [
            a
            for a in input_arrays
            if isinstance(a, (np.ndarray, pa.Array, pa.ChunkedArray))
        ]

        if not array_inputs:
            # All scalars - use numpy
            return "numpy"

        # Get current backend of inputs
        input_backend = get_array_backend(array_inputs[0])

        return decide_expr_backend(func, input_backend, self._preferred_backend)

    def _evaluate_call(self, node) -> Any:
        """Evaluate a Call node using kernel dispatch."""
        func = node.function
        args = [self.evaluate(arg) for arg in node.args]

        # Determine backend
        backend = self._get_backend_for_func(func, args)

        # Check if we have a kernel for this function
        if has_kernel(func, backend):
            # Convert inputs to target backend if needed
            converted_args = []
            for arg in args:
                if isinstance(arg, (np.ndarray, pa.Array, pa.ChunkedArray)):
                    converted_args.append(ensure_backend(arg, backend))
                else:
                    # Scalar - keep as is
                    converted_args.append(arg)

            # Dispatch to kernel
            return dispatch_kernel(func, backend, *converted_args, **node.kwargs)

        # Fall back to manual implementation for unsupported operations
        return self._fallback_evaluate_call(node, args, backend)

    def _fallback_evaluate_call(
        self, node, args: list, backend: Literal["arrow", "numpy"]
    ) -> Any:
        """
        Fallback for operations without registered kernels.

        This handles complex operations like window functions, datetime
        accessors, and case_when that don't map cleanly to simple kernels.
        """
        func = node.function

        # Datetime operations - extract component
        if func.startswith("dt_"):
            return self._evaluate_datetime(func, args[0])

        # Conditional operations
        elif func == "case_when":
            return self._evaluate_case_when(node)

        # Window functions
        elif func == "window":
            return self._evaluate_window(node)

        # Rank functions (outside window context)
        elif func == "rank":
            return self._evaluate_rank(args[0], method="min")
        elif func == "dense_rank":
            return self._evaluate_rank(args[0], method="dense")
        elif func == "row_number":
            return self._evaluate_row_number(args[0])

        # Shift operations
        elif func == "lag":
            return self._evaluate_shift(
                args[0], node.kwargs.get("n", 1), node.kwargs.get("default")
            )
        elif func == "lead":
            return self._evaluate_shift(
                args[0], -node.kwargs.get("n", 1), node.kwargs.get("default")
            )

        # Cumulative operations
        elif func == "cum_sum":
            return self._evaluate_cumulative(args[0], "sum")
        elif func == "cum_min":
            return self._evaluate_cumulative(args[0], "min")
        elif func == "cum_max":
            return self._evaluate_cumulative(args[0], "max")

        else:
            raise NotImplementedError(
                f"Function '{func}' not implemented for array evaluation. "
                f"Available kernels: {backend}"
            )

    def _cast_array(self, arr: ArrayLike, target_dtype) -> ArrayLike:
        """Cast array to target dtype."""
        if is_arrow_backed(arr):
            # Arrow casting
            import pyarrow as pa

            pa_type = self._dtype_to_arrow(target_dtype)
            if isinstance(arr, pa.ChunkedArray):
                return arr.cast(pa_type)
            return arr.cast(pa_type)
        else:
            # NumPy casting
            return arr.astype(target_dtype)

    def _dtype_to_arrow(self, dtype):
        """Convert pandas/numpy dtype to Arrow type."""
        import pyarrow as pa

        dtype_str = str(dtype)
        mapping = {
            "int64": pa.int64(),
            "int32": pa.int32(),
            "int16": pa.int16(),
            "int8": pa.int8(),
            "uint64": pa.uint64(),
            "uint32": pa.uint32(),
            "uint16": pa.uint16(),
            "uint8": pa.uint8(),
            "float64": pa.float64(),
            "float32": pa.float32(),
            "bool": pa.bool_(),
            "string": pa.string(),
            "object": pa.string(),
        }
        return mapping.get(dtype_str, pa.string())

    def _evaluate_datetime(self, func: str, arr: ArrayLike) -> ArrayLike:
        """Evaluate datetime accessor operations."""
        import pyarrow.compute as pc

        component = func[3:]  # Remove "dt_" prefix

        if is_arrow_backed(arr):
            # Arrow datetime extraction
            if component == "year":
                return pc.year(arr)
            elif component == "month":
                return pc.month(arr)
            elif component == "day":
                return pc.day(arr)
            elif component == "hour":
                return pc.hour(arr)
            elif component == "minute":
                return pc.minute(arr)
            elif component == "second":
                return pc.second(arr)
            elif component == "weekday":
                return pc.day_of_week(arr)
            elif component == "dayofyear":
                return pc.day_of_year(arr)
            elif component == "quarter":
                return pc.quarter(arr)
            elif component == "is_month_start":
                return pc.equal(pc.day(arr), 1)
            elif component == "is_year_start":
                return pc.and_(pc.equal(pc.month(arr), 1), pc.equal(pc.day(arr), 1))
            else:
                raise NotImplementedError(
                    f"Datetime component not implemented: {component}"
                )
        else:
            # NumPy datetime - convert to pandas for accessor
            import pandas as pd

            s = pd.Series(arr)
            result = getattr(s.dt, component)
            return result.to_numpy()

    def _evaluate_case_when(self, node) -> ArrayLike:
        """Evaluate case_when expression."""
        cases = node.kwargs.get("cases", ())
        otherwise_node = node.kwargs.get("otherwise")

        # Get array length from first array in context
        arr_len = len(next(iter(self._arrays.values())))

        # Evaluate otherwise
        otherwise = self.evaluate(otherwise_node)
        if not isinstance(otherwise, (np.ndarray, pa.Array, pa.ChunkedArray)):
            # Scalar - broadcast
            otherwise = np.full(arr_len, otherwise)

        # Determine backend
        backend = self._get_backend_for_func("case_when", [otherwise])
        result = ensure_backend(otherwise, backend)

        # Apply cases in reverse order
        for condition_node, value_node in reversed(cases):
            condition = self.evaluate(condition_node)
            value = self.evaluate(value_node)

            if backend == "arrow":
                import pyarrow.compute as pc

                # Arrow if_else: if_else(condition, if_true, if_false)
                if not isinstance(value, (pa.Array, pa.ChunkedArray)):
                    value = pa.scalar(value)
                result = pc.if_else(condition, value, result)
            else:
                # NumPy where
                if not isinstance(condition, np.ndarray):
                    condition = np.asarray(condition)
                if not isinstance(value, np.ndarray):
                    value = np.full_like(result, value)
                result = np.where(condition, value, result)

        return result

    def _evaluate_window(self, node) -> ArrayLike:
        """
        Evaluate window function.

        Window functions are complex and currently fall back to pandas.
        Future: implement native Arrow/NumPy window operations.
        """
        # For now, fall back to pandas-based evaluation
        # This is a complex operation that requires groupby + transform
        import pandas as pd
        from pandas.lazy.backends.convert import (
            arrays_to_dataframe,
            extract_array,
        )
        from pandas.lazy.eval import Evaluator

        # Convert arrays to DataFrame for window evaluation
        df = arrays_to_dataframe(self._arrays)
        evaluator = Evaluator(df)
        result = evaluator.evaluate(node)

        # Extract array from result
        if isinstance(result, pd.Series):
            return extract_array(result)
        return result

    def _evaluate_rank(self, arr: ArrayLike, method: str) -> ArrayLike:
        """Evaluate rank operation."""
        if is_arrow_backed(arr):
            # Arrow doesn't have rank - fall back to numpy
            np_arr = arr.to_numpy(zero_copy_only=False)
            from scipy.stats import rankdata

            if method == "min":
                return pa.array(rankdata(np_arr, method="min"))
            elif method == "dense":
                return pa.array(rankdata(np_arr, method="dense"))
            else:
                return pa.array(rankdata(np_arr, method=method))
        else:
            from scipy.stats import rankdata

            return rankdata(arr, method=method)

    def _evaluate_row_number(self, arr: ArrayLike) -> ArrayLike:
        """Evaluate row_number operation."""
        length = len(arr)
        result = np.arange(1, length + 1)
        if is_arrow_backed(arr):
            return pa.array(result)
        return result

    def _evaluate_shift(self, arr: ArrayLike, n: int, default_node) -> ArrayLike:
        """Evaluate lag/lead (shift) operation."""
        if is_arrow_backed(arr):
            # Arrow doesn't have shift - use numpy
            np_arr = arr.to_numpy(zero_copy_only=False)
            result = np.roll(np_arr, n)
            if n > 0:
                result[:n] = np.nan
            elif n < 0:
                result[n:] = np.nan

            if default_node is not None:
                default_val = self.evaluate(default_node)
                mask = np.isnan(result)
                result[mask] = default_val

            return pa.array(result)
        else:
            result = np.roll(arr, n)
            if n > 0:
                result[:n] = np.nan
            elif n < 0:
                result[n:] = np.nan

            if default_node is not None:
                default_val = self.evaluate(default_node)
                mask = np.isnan(result)
                result[mask] = default_val

            return result

    def _evaluate_cumulative(self, arr: ArrayLike, agg: str) -> ArrayLike:
        """Evaluate cumulative operation (cumsum, cummin, cummax)."""
        if is_arrow_backed(arr):
            import pyarrow.compute as pc

            if agg == "sum":
                return pc.cumulative_sum(arr)
            elif agg == "min":
                return pc.cumulative_min(arr)
            elif agg == "max":
                return pc.cumulative_max(arr)
        elif agg == "sum":
            return np.cumsum(arr)
        elif agg == "min":
            return np.minimum.accumulate(arr)
        elif agg == "max":
            return np.maximum.accumulate(arr)
