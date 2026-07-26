"""
Array-based expression evaluation for lazy pandas.

This module provides the ArrayEvaluator class that executes IR nodes
directly on arrays (Arrow or NumPy) using the kernel dispatch system.

Unlike the DataFrame-based Evaluator, this evaluator:
- Works directly on ArrayDict (dict of arrays)
- Dispatches to registered kernels (Arrow or NumPy)
- Avoids pandas overhead for supported operations
- Uses NumExpr expression fusion for large NumPy arrays
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
    get_kernel,
)
from pandas.lazy.backends.convert import (
    ensure_backend,
    is_arrow_backed,
)
from pandas.lazy.backends.memory_pool import (
    ArrowPoolBackend,
    PoolingStrategy,
    get_arrow_memory_pool,
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

# Minimum elements for NumExpr to be beneficial
_NUMEXPR_MIN_ELEMENTS = 100_000


def _condition_to_bool_array(condition) -> np.ndarray:
    """Coerce a when()/case condition to a null-free NumPy bool array.

    Nulls (Arrow null, pandas NA, or ``None`` in an object array) become
    ``False`` so the row falls through to the otherwise branch, matching
    pandas semantics and avoiding ``bool(NA)`` errors in ``np.where``.
    """
    if isinstance(condition, (pa.Array, pa.ChunkedArray)):
        import pyarrow.compute as pc

        condition = pc.fill_null(condition, False).to_numpy(zero_copy_only=False)
    elif hasattr(condition, "fillna"):  # pandas Series / masked ExtensionArray
        condition = condition.fillna(False)
        condition = (
            condition.to_numpy() if hasattr(condition, "to_numpy") else condition
        )
    condition = np.asarray(condition)
    if condition.dtype == object:
        import pandas as pd

        condition = (
            pd.array(condition, dtype="boolean").fillna(False).to_numpy(dtype=bool)
        )
    return condition.astype(bool, copy=False)


def _coerce_temporal_scalar(arg):
    """Coerce a pandas Timestamp/Timedelta scalar to its NumPy equivalent.

    ``np.greater_equal(datetime64_array, pd.Timestamp)`` (and the other
    comparisons) fall to a ~1000x-slower element-wise object path, whereas the
    same against a ``np.datetime64``/``np.timedelta64`` is vectorized. The
    duck-typed ``getattr`` checks are cheap and avoid importing pandas on the
    hot path (a non-temporal scalar returns immediately).
    """
    # A PyArrow scalar (e.g. a scalar aggregate result) is not a NumPy operand;
    # unwrap it to its Python value so the NumPy kernels can use it (otherwise
    # np.multiply(Int64Scalar, DoubleScalar) etc. raise TypeError).
    if isinstance(arg, pa.Scalar):
        arg = arg.as_py()
    to_dt64 = getattr(arg, "to_datetime64", None)
    if to_dt64 is not None:
        return to_dt64()
    to_td64 = getattr(arg, "to_timedelta64", None)
    if to_td64 is not None:
        return to_td64()
    return arg


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
        use_numexpr: bool = True,
        pooling_strategy: PoolingStrategy | str = PoolingStrategy.SCRATCH,
        arrow_pool: ArrowPoolBackend | str = ArrowPoolBackend.DEFAULT,
    ) -> None:
        """
        Initialize evaluator with arrays.

        Parameters
        ----------
        arrays : ArrayDict
            Dictionary mapping column names to arrays.
        preferred_backend : {"auto", "arrow", "numpy"}
            Preferred execution backend.
        use_numexpr : bool, default True
            Whether to use NumExpr for expression fusion when beneficial.
        pooling_strategy : PoolingStrategy or str, default SCRATCH
            Memory pooling strategy for NumPy operations:
            - "scratch" / SCRATCH: Rotating scratch buffers (~3x speedup)
            - "none" / NONE: No pooling (allocate new arrays)
            - "acquire_release" / ACQUIRE_RELEASE: Explicit pool
        arrow_pool : ArrowPoolBackend or str, default DEFAULT
            Arrow memory pool backend for Arrow operations:
            - "default" / DEFAULT: PyArrow's default (usually mimalloc)
            - "mimalloc" / MIMALLOC: Microsoft's fast allocator
            - "jemalloc" / JEMALLOC: Facebook's allocator
            - "system" / SYSTEM: System malloc
        """
        self._arrays = arrays
        self._preferred_backend = preferred_backend
        self._use_numexpr = use_numexpr

        # Normalize pooling strategy for NumPy
        if isinstance(pooling_strategy, str):
            pooling_strategy = PoolingStrategy(pooling_strategy)
        self._pooling_strategy = pooling_strategy
        self._array_pool = None  # Lazy init for acquire/release pool

        # Arrow memory pool (lazy init)
        if isinstance(arrow_pool, str):
            arrow_pool = ArrowPoolBackend(arrow_pool)
        self._arrow_pool_backend = arrow_pool
        self._arrow_pool = None  # Lazy init

        # Determine array size for NumExpr threshold check
        self._array_size = 0
        for arr in arrays.values():
            if isinstance(arr, np.ndarray):
                self._array_size = len(arr)
                break
            elif isinstance(arr, (pa.Array, pa.ChunkedArray)):
                self._array_size = len(arr)
                break

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
            # Try NumExpr fusion for complex arithmetic on large NumPy arrays
            if self._should_try_numexpr(node):
                result = self._try_numexpr_fusion(node)
                if result is not None:
                    return result
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
        # Find first array input to determine backend
        input_backend: Literal["arrow", "numpy"] = "numpy"
        for a in input_arrays:
            if isinstance(a, (pa.Array, pa.ChunkedArray)):
                input_backend = "arrow"
                break
            elif isinstance(a, np.ndarray):
                input_backend = "numpy"
                break
        # If no array inputs found, input_backend stays "numpy"

        return decide_expr_backend(func, input_backend, self._preferred_backend)

    def _evaluate_call(self, node) -> Any:
        """Evaluate a Call node using kernel dispatch."""
        func = node.function
        args = [self.evaluate(arg) for arg in node.args]

        # Determine backend
        backend = self._get_backend_for_func(func, args)

        # Some functions have special kwargs structure and need custom handling
        # rather than direct kernel dispatch
        special_functions = {"case_when"}
        if func in special_functions:
            return self._fallback_evaluate_call(node, args, backend)

        # For Arrow backend, try cached function call for ~10% speedup
        if backend == "arrow":
            result = self._try_arrow_cached_call(func, args, node.kwargs)
            if result is not None:
                return result

        # For NumPy backend with pooling, reuse an ephemeral intermediate as the
        # output buffer to save an allocation. Only a freshly-computed
        # sub-result (a Call/Cast node) is safe to overwrite in place; a source
        # column (FieldRef) is shared and must never be mutated.
        if (
            backend == "numpy"
            and self._pooling_strategy != PoolingStrategy.NONE
            and not node.kwargs
        ):
            ephemeral = [self._is_ephemeral(a) for a in node.args]
            result = self._try_pooled_numpy_call(func, args, ephemeral)
            if result is not None:
                return result

        # Get kernel directly (single lookup instead of has_kernel + dispatch_kernel)
        kernel = get_kernel(func, backend)
        if kernel is not None:
            # Convert inputs to target backend if needed
            converted_args = []
            for arg in args:
                if isinstance(arg, (np.ndarray, pa.Array, pa.ChunkedArray)):
                    converted_args.append(ensure_backend(arg, backend))
                else:
                    # Scalar - keep as is, but coerce pandas temporal scalars
                    # to NumPy: np.greater_equal(datetime64_array, pd.Timestamp)
                    # falls to a ~1000x-slower element-wise object comparison,
                    # where the same against a np.datetime64 is vectorized.
                    converted_args.append(_coerce_temporal_scalar(arg))

            # Call kernel directly
            return kernel(*converted_args, **node.kwargs)

        # If Arrow backend doesn't have kernel, try NumPy kernel with conversion
        if backend == "arrow":
            numpy_kernel = get_kernel(func, "numpy")
            if numpy_kernel is not None:
                # Convert Arrow inputs to NumPy, call kernel, convert back to Arrow
                converted_args = []
                for arg in args:
                    if isinstance(arg, (pa.Array, pa.ChunkedArray)):
                        converted_args.append(ensure_backend(arg, "numpy"))
                    elif isinstance(arg, np.ndarray):
                        converted_args.append(arg)
                    else:
                        # Scalar - keep as is
                        converted_args.append(arg)

                result = numpy_kernel(*converted_args, **node.kwargs)
                # Convert result back to Arrow
                return pa.array(result)

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
        elif func == "cum_mean":
            return self._evaluate_cumulative(args[0], "mean")
        elif func == "cum_prod":
            return self._evaluate_cumulative(args[0], "prod")

        # Row index generation
        elif func == "row_index":
            return self._evaluate_row_index(node.kwargs.get("offset", 0), backend)

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
        import numpy as np
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
        if dtype_str in mapping:
            return mapping[dtype_str]
        # Fall back to Arrow's own NumPy-dtype mapping (datetime64, timedelta64,
        # ...) rather than silently defaulting to pa.string(), which would
        # stringify the column and corrupt a cast to e.g. datetime64[ns].
        try:
            return pa.from_numpy_dtype(np.dtype(dtype))
        except (pa.ArrowNotImplementedError, TypeError, ValueError) as err:
            raise NotImplementedError(
                f"Arrow-backed cast to dtype {dtype!r} is not supported"
            ) from err

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

                # A null condition must fall through to the otherwise branch
                # (treat null as False), matching pandas when() semantics.
                if isinstance(condition, (pa.Array, pa.ChunkedArray)):
                    condition = pc.fill_null(condition, False)
                # Arrow if_else: if_else(condition, if_true, if_false)
                if not isinstance(value, (pa.Array, pa.ChunkedArray)):
                    value = pa.scalar(value)
                result = pc.if_else(condition, value, result)
            else:
                # NumPy where — coerce a nullable-boolean condition so a null
                # is False (falls through to otherwise) rather than raising
                # "boolean value of NA is ambiguous".
                condition = _condition_to_bool_array(condition)
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

    def _evaluate_row_index(
        self, offset: int, backend: Literal["arrow", "numpy"]
    ) -> ArrayLike:
        """Evaluate row_index operation - generates sequence from offset."""
        # Get length from first array in the dict
        if not self._arrays:
            return np.array([], dtype=np.int64)

        first_arr = next(iter(self._arrays.values()))
        length = len(first_arr)
        result = np.arange(offset, offset + length, dtype=np.int64)

        if backend == "arrow":
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
        """Evaluate cumulative operation (cumsum, cummin, cummax, cummean, cumprod).

        Dispatches to the registered per-backend kernels (preserving the input
        backend) instead of a bespoke inline implementation. Both honour pandas
        skipna semantics -- skip the missing value in the running accumulation
        but keep it (NaN/null) at that row: the NumPy kernels handle float NaN,
        and the Arrow kernels use skip_nulls=True (verified to keep the null at
        position, matching pandas). The previous inline code used plain
        np.cumsum / pc.cumulative_* with NaN-propagation, diverging from pandas.
        """
        from pandas.lazy.backends import get_kernel

        backend = "arrow" if is_arrow_backed(arr) else "numpy"
        kernel = get_kernel(f"cumulative_{agg}", backend)
        if kernel is None:
            raise NotImplementedError(f"cumulative_{agg} has no {backend} kernel")
        return kernel(arr)

    # =========================================================================
    # Arrow Memory Pool
    # =========================================================================

    def _get_arrow_pool(self):
        """Get the Arrow memory pool for this evaluator."""
        if self._arrow_pool is None:
            self._arrow_pool = get_arrow_memory_pool(self._arrow_pool_backend)
        return self._arrow_pool

    # =========================================================================
    # Arrow Cached Function Calls
    # =========================================================================

    def _try_arrow_cached_call(
        self, func: str, args: list, kwargs: dict
    ) -> ArrayLike | None:
        """
        Try to evaluate using cached Arrow function reference.

        This provides ~10% speedup by avoiding repeated function lookups,
        and uses the configured memory pool for allocations.

        Returns None if the function is not available in Arrow.
        """
        try:
            from pandas.lazy.backends.arrow.cache import get_arrow_function
        except ImportError:
            return None

        # Only use for basic operations without special kwargs
        if kwargs:
            return None

        # is_null / is_not_null need nan_is_null=True to match pandas (where a
        # float NaN is null); the bare cached pc.is_null does not, so route them
        # to the registered kernels (arrow/core.py) which set that option.
        if func in ("is_null", "is_not_null"):
            return None

        # Reducing aggregates must skip NaN to match pandas: a bare Arrow
        # count/sum/mean treats a float NaN as a present value (count = len -
        # null_count, sum propagates NaN), so a column mixing nulls and NaNs
        # gives wrong, hash-routing-dependent results. Route these to the
        # registered kernels (arrow/core.py) which apply _skipna.
        if func in (
            "count",
            "sum",
            "mean",
            "min",
            "max",
            "std",
            "var",
            "median",
            "product",
        ):
            return None

        fn = get_arrow_function(func)
        if fn is None:
            return None

        try:
            # Use configured memory pool for allocation
            return fn.call(list(args), memory_pool=self._get_arrow_pool())
        except (ValueError, TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
            # Function not available or incompatible args (e.g., string type mismatch)
            # Fall through to kernel-based dispatch which handles type conversion
            return None

    # =========================================================================
    # NumExpr Expression Fusion
    # =========================================================================

    def _should_try_numexpr(self, node) -> bool:
        """Check if NumExpr fusion should be attempted for this node."""
        if not self._use_numexpr:
            return False

        # Only worthwhile for large arrays
        if self._array_size < _NUMEXPR_MIN_ELEMENTS:
            return False

        # Check if preferred backend is numpy (or auto with numpy data)
        if self._preferred_backend == "arrow":
            return False

        # Check if arrays are NumPy (not Arrow)
        has_numpy = False
        for arr in self._arrays.values():
            if isinstance(arr, np.ndarray):
                has_numpy = True
                break
            elif isinstance(arr, (pa.Array, pa.ChunkedArray)):
                # Has Arrow data - NumExpr won't help
                return False

        return has_numpy

    def _try_numexpr_fusion(self, node) -> ArrayLike | None:
        """
        Try to evaluate expression using NumExpr fusion.

        Returns None if fusion is not possible or not beneficial.
        """
        try:
            from pandas.lazy.backends.numexpr_fusion import (
                can_fuse_expression,
                fuse_expression,
            )
        except ImportError:
            return None

        # Check if expression can be fused
        if not can_fuse_expression(node, self._arrays):
            return None

        try:
            return fuse_expression(node, self._arrays)
        except (ValueError, TypeError, KeyError):
            # Fusion failed - fall back to regular evaluation
            return None

    # =========================================================================
    # Pooled NumPy Evaluation
    # =========================================================================

    # IR function to numpy ufunc mapping (shared)
    _IR_TO_NUMPY_UFUNC = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
        "floor_divide": np.floor_divide,
        "power": np.power,
        "modulo": np.mod,
        "negate": np.negative,
        "abs": np.absolute,
        "equal": np.equal,
        "not_equal": np.not_equal,
        "less": np.less,
        "less_equal": np.less_equal,
        "greater": np.greater,
        "greater_equal": np.greater_equal,
        "and_": np.logical_and,
        "or_": np.logical_or,
        "invert": np.logical_not,
    }

    # Operations that return boolean
    _BOOLEAN_OPS = frozenset(
        {
            "equal",
            "not_equal",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "and_",
            "or_",
            "invert",
        }
    )

    @staticmethod
    def _is_ephemeral(node) -> bool:
        """Whether evaluating ``node`` yields a freshly-owned array.

        A ``Call``/``Cast`` produces a single-use intermediate that is safe to
        overwrite in place; a ``FieldRef`` returns a shared source column that
        must never be mutated. ``Alias`` is transparent. Anything else (e.g.
        ``Literal``) is treated as non-ephemeral (conservative = safe).
        """
        from pandas.lazy.ir import (
            Alias,
            Call,
            Cast,
        )

        while isinstance(node, Alias):
            node = node.arg
        return isinstance(node, (Call, Cast))

    def _try_pooled_numpy_call(
        self, func: str, args: list, ephemeral: list[bool]
    ) -> ArrayLike | None:
        """
        Evaluate an elementwise op writing into a reusable output buffer.

        When one of the array inputs is an ephemeral intermediate (a
        freshly-computed sub-result, flagged in ``ephemeral``) of a matching
        dtype, it is reused as the ``out=`` buffer so a chain of operations runs
        in place without per-step allocation. A shared source column is never
        reused. Returns None if pooled evaluation is not possible.
        """
        from pandas.lazy.backends.memory_pool import can_use_pooled_output

        if not can_use_pooled_output(func):
            return None

        # Check if all args are numpy arrays or scalars
        numpy_args = []
        size = None
        for arg in args:
            if isinstance(arg, np.ndarray):
                numpy_args.append(arg)
                if size is None:
                    size = len(arg)
            elif isinstance(arg, (int, float, bool)):
                numpy_args.append(arg)
            else:
                # Unsupported type
                return None

        if size is None:
            return None

        ufunc = self._IR_TO_NUMPY_UFUNC.get(func)
        if ufunc is None:
            return None

        # Determine output dtype
        if func in self._BOOLEAN_OPS:
            out_dtype = np.dtype(np.bool_)
        else:
            out_dtype = np.result_type(
                *[a for a in numpy_args if isinstance(a, np.ndarray)]
            )

        # Reuse an ephemeral input of the right shape/dtype as the output
        # buffer (safe in-place chaining); otherwise allocate a fresh output.
        out = None
        for arg, is_eph in zip(args, ephemeral, strict=True):
            if (
                is_eph
                and isinstance(arg, np.ndarray)
                and arg.dtype == out_dtype
                and arg.shape == (size,)
            ):
                out = arg
                break
        if out is None:
            out = np.empty(size, dtype=out_dtype)

        try:
            return ufunc(*numpy_args, out=out)
        except (TypeError, ValueError):
            # Some edge cases may fail
            return None
