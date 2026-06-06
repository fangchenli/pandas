"""
Format conversion utilities for lazy pandas backend execution.

This module provides utilities for:
- Detecting array formats (Arrow vs NumPy)
- Converting between formats
- Extracting arrays from pandas objects
- Verifying chunk alignment (debug mode)
"""

from typing import Literal

import numpy as np
import pyarrow as pa

from pandas.lazy.backends.types import (
    ArrayDict,
    ArrayLike,
    PyArrowArray,
)

# =============================================================================
# Format Detection
# =============================================================================


def is_arrow_backed(arr: ArrayLike) -> bool:
    """
    Check if array is Arrow-backed (Array, ChunkedArray, or ArrowExtensionArray).

    Parameters
    ----------
    arr : ArrayLike
        The array to check.

    Returns
    -------
    bool
        True if Arrow-backed.
    """
    # Direct PyArrow types
    if isinstance(arr, (pa.Array, pa.ChunkedArray)):
        return True
    # Pandas ArrowExtensionArray (ArrowStringArray, ArrowDtype-backed, etc.)
    # These wrap PyArrow ChunkedArrays internally via _pa_array attribute
    if hasattr(arr, "_pa_array") and isinstance(
        getattr(arr, "_pa_array", None), pa.ChunkedArray
    ):
        return True
    return False


def is_numpy_backed(arr: ArrayLike) -> bool:
    """
    Check if array is NumPy-backed.

    Parameters
    ----------
    arr : ArrayLike
        The array to check.

    Returns
    -------
    bool
        True if NumPy-backed.
    """
    return isinstance(arr, np.ndarray)


def get_array_backend(arr: ArrayLike) -> Literal["arrow", "numpy"]:
    """
    Get the backend type of an array.

    Parameters
    ----------
    arr : ArrayLike
        The array to check.

    Returns
    -------
    {"arrow", "numpy"}
        The backend type.

    Raises
    ------
    TypeError
        If array type is not recognized.
    """
    if is_arrow_backed(arr):
        return "arrow"
    if is_numpy_backed(arr):
        return "numpy"
    raise TypeError(f"Unknown array type: {type(arr)}")


def get_column_formats(arrays: ArrayDict) -> dict[str, Literal["arrow", "numpy"]]:
    """
    Get format for each column in an ArrayDict.

    Parameters
    ----------
    arrays : ArrayDict
        The arrays to check.

    Returns
    -------
    dict
        Mapping of column name to backend type.
    """
    return {name: get_array_backend(arr) for name, arr in arrays.items()}


# =============================================================================
# Array Extraction
# =============================================================================


def extract_array(obj) -> ArrayLike:
    """
    Extract underlying array from pandas object.

    Handles:
    - pandas Series with ArrowExtensionArray
    - ArrowExtensionArray directly
    - NumPy arrays
    - PyArrow arrays (pass through)

    Parameters
    ----------
    obj : Series, ExtensionArray, or array
        The object to extract from.

    Returns
    -------
    ArrayLike
        The underlying array.
    """
    # pandas Series
    if hasattr(obj, "array"):
        arr = obj.array
        # Arrow-backed arrays use _pa_array (ChunkedArray)
        if hasattr(arr, "_pa_array"):
            return arr._pa_array
        # Categorical -> Arrow dictionary, zero-copy on the codes.
        # Without this branch categoricals fell through to np.asarray,
        # materializing an object array of values on EVERY query
        # (322 ms for a 10M groupby key); acero hash-aggregates
        # dictionary keys at ~13 ms - faster than grouping raw strings.
        import pandas as pd

        if isinstance(arr, pd.Categorical):
            codes = arr.codes
            mask = codes == -1
            indices = pa.array(codes, mask=mask if mask.any() else None)
            dictionary = pa.array(arr.categories)
            return pa.DictionaryArray.from_arrays(indices, dictionary)
        # ArrowExtensionArray may store data in _ndarray (ChunkedArray)
        if hasattr(arr, "_ndarray") and isinstance(
            arr._ndarray, (pa.Array, pa.ChunkedArray)
        ):
            return arr._ndarray
        # Other ExtensionArrays - try to get underlying
        # IMPORTANT: Do NOT extract _data from masked arrays (IntegerArray, etc.)
        # because _data is the raw backing array without the null mask.
        # For masked arrays, np.asarray() correctly converts to float64 with NaN.
        if (
            hasattr(arr, "_data")
            and isinstance(arr._data, (np.ndarray, pa.Array, pa.ChunkedArray))
            and not hasattr(arr, "_mask")  # Skip masked arrays
        ):
            return arr._data
        # NumPy-backed - convert to ndarray
        return np.asarray(arr)

    # ArrowExtensionArray directly
    if hasattr(obj, "_pa_array"):
        return obj._pa_array
    if hasattr(obj, "_ndarray") and isinstance(
        obj._ndarray, (pa.Array, pa.ChunkedArray)
    ):
        return obj._ndarray

    # Already an array type
    if isinstance(obj, (np.ndarray, pa.Array, pa.ChunkedArray)):
        return obj

    # Try numpy conversion as last resort
    return np.asarray(obj)


def extract_arrow_array(obj) -> PyArrowArray:
    """
    Extract Arrow array from pandas object.

    Parameters
    ----------
    obj : Series, ExtensionArray, or PyArrow array
        The object to extract from.

    Returns
    -------
    PyArrowArray
        The Arrow array (Array or ChunkedArray).

    Raises
    ------
    TypeError
        If object is not Arrow-backed.
    """
    arr = extract_array(obj)
    if not is_arrow_backed(arr):
        raise TypeError(f"Expected Arrow-backed array, got {type(arr)}")
    return arr


# =============================================================================
# Format Conversion
# =============================================================================


def to_arrow(arr: ArrayLike) -> PyArrowArray:
    """
    Convert array to Arrow format.

    Parameters
    ----------
    arr : ArrayLike
        The array to convert.

    Returns
    -------
    PyArrowArray
        Arrow array (Array or ChunkedArray).
    """
    # Direct PyArrow types - return as-is
    if isinstance(arr, (pa.Array, pa.ChunkedArray)):
        return arr
    # Pandas ArrowExtensionArray - extract underlying PyArrow ChunkedArray
    if hasattr(arr, "_pa_array") and isinstance(
        getattr(arr, "_pa_array", None), pa.ChunkedArray
    ):
        return arr._pa_array
    # NumPy to Arrow. from_pandas=True maps float NaN to null, matching
    # pandas semantics where NaN means missing.
    return pa.array(arr, from_pandas=True)


def mask_nan_to_null(arr: PyArrowArray) -> PyArrowArray:
    """
    Replace floating-point NaN values with null.

    pandas treats NaN as missing data (``isna``/``skipna`` semantics),
    while Arrow treats NaN as an ordinary float value and only skips
    *nulls* in aggregations. Arrow kernels that must match pandas
    aggregation semantics call this on floating-point inputs first.

    Parameters
    ----------
    arr : PyArrowArray
        The array to sanitize. Non-floating arrays are returned as-is.

    Returns
    -------
    PyArrowArray
        Array with NaN replaced by null.
    """
    import pyarrow.compute as pc

    if not pa.types.is_floating(arr.type):
        return arr
    # is_nan returns null for null inputs; treat those as "not NaN" so
    # existing nulls pass through unchanged.
    nan_mask = pc.fill_null(pc.is_nan(arr), False)
    if not pc.any(nan_mask).as_py():
        return arr
    return pc.if_else(nan_mask, pa.scalar(None, type=arr.type), arr)


def to_numpy(arr: ArrayLike) -> np.ndarray:
    """
    Convert array to NumPy format.

    Parameters
    ----------
    arr : ArrayLike
        The array to convert.

    Returns
    -------
    np.ndarray
        NumPy array.
    """
    if is_numpy_backed(arr):
        return arr
    # Pandas ArrowExtensionArray - extract and convert
    if hasattr(arr, "_pa_array") and isinstance(
        getattr(arr, "_pa_array", None), pa.ChunkedArray
    ):
        return arr._pa_array.to_numpy(zero_copy_only=False)
    # Arrow to NumPy
    if isinstance(arr, pa.ChunkedArray):
        return arr.to_numpy(zero_copy_only=False)
    return arr.to_numpy(zero_copy_only=False)


def to_contiguous(arr: PyArrowArray) -> pa.Array:
    """
    Ensure single contiguous Arrow array (unchunk if needed).

    Only call this when an operation requires contiguous memory.

    Parameters
    ----------
    arr : PyArrowArray
        The Arrow array.

    Returns
    -------
    pa.Array
        A single contiguous array.
    """
    if isinstance(arr, pa.ChunkedArray):
        return arr.combine_chunks()
    return arr


def ensure_backend(arr: ArrayLike, backend: Literal["arrow", "numpy"]) -> ArrayLike:
    """
    Ensure array is in the specified backend format.

    Parameters
    ----------
    arr : ArrayLike
        The array to convert if needed.
    backend : {"arrow", "numpy"}
        Target backend.

    Returns
    -------
    ArrayLike
        Array in the target format.
    """
    if backend == "arrow":
        return to_arrow(arr)
    return to_numpy(arr)


# =============================================================================
# Chunk Alignment (Debug)
# =============================================================================


def verify_chunk_alignment(arrays: ArrayDict) -> bool:
    """
    Verify all ChunkedArrays have compatible chunk structures.

    This is a debug utility to catch alignment issues early.
    PyArrow compute functions handle misalignment internally,
    but aligned chunks are more efficient.

    Parameters
    ----------
    arrays : ArrayDict
        The arrays to check.

    Returns
    -------
    bool
        True if all ChunkedArrays are aligned.
    """
    chunked = [arr for arr in arrays.values() if isinstance(arr, pa.ChunkedArray)]

    if len(chunked) < 2:
        return True

    # Check all have same number of chunks
    num_chunks = chunked[0].num_chunks
    if not all(arr.num_chunks == num_chunks for arr in chunked):
        return False

    # Check chunk lengths match
    for i in range(num_chunks):
        lengths = [arr.chunk(i).length for arr in chunked]
        if len(set(lengths)) > 1:
            return False

    return True


def get_chunk_info(arr: PyArrowArray) -> dict:
    """
    Get chunk information for debugging.

    Parameters
    ----------
    arr : PyArrowArray
        The Arrow array.

    Returns
    -------
    dict
        Chunk information including num_chunks and chunk_lengths.
    """
    if isinstance(arr, pa.Array):
        return {"num_chunks": 1, "chunk_lengths": [len(arr)]}
    return {
        "num_chunks": arr.num_chunks,
        "chunk_lengths": [arr.chunk(i).length for i in range(arr.num_chunks)],
    }


# =============================================================================
# ArrayDict to DataFrame Conversion
# =============================================================================


def arrays_to_dataframe(
    arrays: ArrayDict,
    index_names: list[str | None] | None = None,
    index_is_multi: bool = False,
    preserve_index: bool = False,
    use_arrow_dtype: bool = True,
):
    """
    Convert ArrayDict back to pandas DataFrame with proper index.

    This is the final step of physical execution - converting the
    intermediate array representation back to a user-facing DataFrame.

    Parameters
    ----------
    arrays : ArrayDict
        Dictionary mapping column names to arrays.
        Index columns are stored as "__index__" or "__index_N__".
    index_names : list of str or None, optional
        Names for the index level(s). If None, no names are set.
    index_is_multi : bool, default False
        Whether the index is a MultiIndex.
    preserve_index : bool, default False
        If True, reconstruct the DataFrame index from stored index columns.
        If False (default), use a fresh RangeIndex (Polars-style behavior).

        Note: The caller (execute_physical_plan) sets this to True if either:
        1. User passed preserve_index=True to collect(), OR
        2. User explicitly called set_index() (which sets context.user_set_index)

        This ensures that explicit set_index() calls always work regardless
        of the preserve_index parameter passed to collect().
    use_arrow_dtype : bool, default True
        If True and all columns are Arrow-backed, use Arrow-backed pandas
        dtypes (pd.ArrowDtype) for near-zero-copy conversion. This is much
        faster (~15-18x) but returns columns with Arrow dtypes instead of NumPy.

    Returns
    -------
    DataFrame
        The reconstructed DataFrame with proper index.

    Notes
    -----
    Zero-Copy Conversion with Arrow-Backed Dtypes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``use_arrow_dtype=True``, Arrow columns are wrapped per column
    into pandas extension arrays without copying:

    1. string/large_string columns become ``ArrowStringArray`` — pandas'
       default Arrow-backed ``str`` dtype, matching the eager path
    2. other Arrow columns become ``ArrowExtensionArray`` (``ArrowDtype``),
       which stores a reference to the ``ChunkedArray`` in ``_pa_array``
    3. the underlying Arrow memory buffers are shared, not copied
    4. NumPy columns pass through untouched (floats containing NaN are
       converted to nullable ``Float64`` for ``pd.NA`` semantics)

    ``pa.Table.to_pandas`` is deliberately NOT used here: pyarrow's
    ``table_to_dataframe`` constructs ``DataFrame(BlockManager)``, which
    is deprecated in pandas and varies by installed pyarrow build.

    With ``use_arrow_dtype=False``, Arrow columns are converted per column
    via ``arr.to_pandas()`` (copies into NumPy-backed arrays).

    References
    ~~~~~~~~~~
    - Apache Arrow Python Pandas Integration:
      https://arrow.apache.org/docs/python/pandas.html
    - pandas PyArrow Functionality:
      https://pandas.pydata.org/docs/user_guide/pyarrow.html
    """
    import pandas as pd
    from pandas.lazy.backends.types import (
        INDEX_COL_NAME,
        index_col_name,
        is_index_col,
    )

    # Separate index columns from data columns
    data_cols = {name: arr for name, arr in arrays.items() if not is_index_col(name)}
    index_cols = {name: arr for name, arr in arrays.items() if is_index_col(name)}

    if not data_cols:
        # Empty DataFrame
        return pd.DataFrame()

    # One unified per-column conversion path for every backend mix.
    #
    # Two earlier designs were rejected for cause:
    # - pa.Table.to_pandas(types_mapper) for all-Arrow outputs: pyarrow's
    #   table_to_dataframe constructs DataFrame(BlockManager), which is
    #   deprecated in pandas (Pandas4Warning, soon an error) and varies by
    #   installed pyarrow build (broke CI while passing locally).
    # - arr.to_pandas() per Arrow column: round-tripped pass-through
    #   string columns through object arrays, which the DataFrame
    #   constructor re-inferred back to Arrow - two full copies of data
    #   no operation ever touched (~320 ms at 10M rows).
    #
    # The extension-array wraps below are zero-copy and never enter
    # pyarrow's pandas_compat.
    def to_pandas_array(arr):
        """Convert array to pandas-compatible format with proper null handling."""
        if isinstance(arr, (pa.Array, pa.ChunkedArray)):
            if not use_arrow_dtype:
                return arr.to_pandas()
            chunked = pa.chunked_array([arr]) if isinstance(arr, pa.Array) else arr
            if pa.types.is_dictionary(chunked.type):
                # Dictionary -> Categorical (mirror of the zero-copy
                # input wrap): category dtype in, category dtype out,
                # matching the eager path.
                combined = chunked.combine_chunks()
                return pd.Categorical.from_codes(
                    np.asarray(combined.indices.fill_null(-1)),
                    categories=combined.dictionary.to_pylist(),
                )
            if pa.types.is_string(chunked.type) or pa.types.is_large_string(
                chunked.type
            ):
                # Match pandas' default Arrow-backed str dtype
                # (zero-copy wrap of the existing buffers)
                try:
                    from pandas.core.arrays.string_arrow import (
                        ArrowStringArray,
                    )

                    return ArrowStringArray(chunked)
                except (TypeError, ValueError):
                    pass
            return pd.arrays.ArrowExtensionArray(chunked)

        # For numpy arrays, convert float arrays with NaN to nullable dtypes
        # This ensures we use pd.NA instead of np.nan
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "f":
            # Check if array has any NaN values
            mask = np.isnan(arr)
            if mask.any():
                # Convert to nullable dtype with pd.NA
                return pd.array(arr, dtype=pd.Float64Dtype())
        return arr

    pandas_data = {name: to_pandas_array(arr) for name, arr in data_cols.items()}
    df = pd.DataFrame(pandas_data)

    # Index reconstruction
    # Reconstruct index if preserve_index=True AND index columns exist
    # The caller (execute_physical_plan) sets preserve_index=True if either:
    # 1. User passed preserve_index=True to collect(), OR
    # 2. User explicitly called set_index() (context.user_set_index=True)
    if preserve_index and index_cols:
        # Convert index arrays to pandas-compatible format
        def to_pandas_index_array(arr):
            if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                return arr.to_pandas()
            return arr

        if index_is_multi:
            # Reconstruct MultiIndex from multiple index columns
            levels = []
            names = index_names if index_names else []
            for i in range(len(names) if names else 0):
                col_name = index_col_name(i)
                if col_name in index_cols:
                    levels.append(to_pandas_index_array(index_cols[col_name]))
            if levels:
                df.index = pd.MultiIndex.from_arrays(levels, names=names)
            else:
                df.index = pd.RangeIndex(len(df))
        else:
            # Reconstruct single Index
            idx_arr = index_cols.get(INDEX_COL_NAME)
            if idx_arr is not None:
                idx_name = index_names[0] if index_names else None
                df.index = pd.Index(to_pandas_index_array(idx_arr), name=idx_name)
            else:
                df.index = pd.RangeIndex(len(df))
    else:
        # No index reconstruction - use fresh RangeIndex (Polars-style behavior)
        df.index = pd.RangeIndex(len(df))

    return df


def dataframe_to_arrays(df) -> tuple[ArrayDict, list[str | None], bool]:
    """
    Convert DataFrame to ArrayDict, extracting index.

    This is the inverse of arrays_to_dataframe, useful for testing.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to convert.

    Returns
    -------
    tuple
        (arrays, index_names, index_is_multi)
    """
    import pandas as pd
    from pandas.lazy.backends.types import index_col_name

    arrays: ArrayDict = {}

    # Extract data columns
    for col in df.columns:
        arrays[col] = extract_array(df[col])

    # Extract index
    if isinstance(df.index, pd.MultiIndex):
        index_is_multi = True
        index_names = list(df.index.names)
        for i in range(df.index.nlevels):
            col_name = index_col_name(i)
            arrays[col_name] = df.index.get_level_values(i).to_numpy()
    else:
        index_is_multi = False
        index_names = [df.index.name]
        arrays[index_col_name()] = df.index.to_numpy()

    return arrays, index_names, index_is_multi
