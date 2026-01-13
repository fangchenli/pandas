"""
Type definitions for lazy pandas backend execution.

This module defines the core types used throughout the backend system:
- ArrayLike: Union of array types we operate on
- ArrayDict: The intermediate data structure passed between physical nodes
- Index column naming conventions
"""

from typing import TypeAlias

import numpy as np
import pyarrow as pa

# =============================================================================
# Array Types
# =============================================================================

# Array types we work with
# NOTE: pandas Arrow-backed columns use pa.ChunkedArray internally
# (accessed via ArrowExtensionArray._ndarray)
ArrayLike: TypeAlias = np.ndarray | pa.Array | pa.ChunkedArray

# For type checking any PyArrow array type
PyArrowArray: TypeAlias = pa.Array | pa.ChunkedArray

# =============================================================================
# Intermediate Data Structure
# =============================================================================

# The intermediate data structure passed between physical nodes
# Maps column names to their array data
ArrayDict: TypeAlias = dict[str, ArrayLike]

# =============================================================================
# Index Column Naming Convention
# =============================================================================

# Prefix for all index columns
INDEX_COL_PREFIX = "__index"

# Single index column name
INDEX_COL_NAME = "__index__"

# MultiIndex format: __index_0__, __index_1__, etc.
# Use index_col_name(level) to generate


def index_col_name(level: int | None = None) -> str:
    """
    Generate index column name.

    Parameters
    ----------
    level : int or None
        For MultiIndex, the level number (0-indexed).
        For single index, pass None.

    Returns
    -------
    str
        The index column name.

    Examples
    --------
    >>> index_col_name()
    '__index__'
    >>> index_col_name(0)
    '__index_0__'
    >>> index_col_name(1)
    '__index_1__'
    """
    if level is None:
        return INDEX_COL_NAME
    return f"__index_{level}__"


def is_index_col(name: str) -> bool:
    """
    Check if a column name is an index column.

    Parameters
    ----------
    name : str
        Column name to check.

    Returns
    -------
    bool
        True if this is an index column.

    Examples
    --------
    >>> is_index_col("__index__")
    True
    >>> is_index_col("__index_0__")
    True
    >>> is_index_col("my_column")
    False
    """
    return name.startswith(INDEX_COL_PREFIX)


def parse_index_level(name: str) -> int | None:
    """
    Parse index level from column name.

    Parameters
    ----------
    name : str
        Index column name.

    Returns
    -------
    int or None
        The level number for MultiIndex, None for single index.
        Returns None if not an index column.

    Examples
    --------
    >>> parse_index_level("__index__")
    None
    >>> parse_index_level("__index_0__")
    0
    >>> parse_index_level("__index_2__")
    2
    >>> parse_index_level("my_column")
    None
    """
    if name == INDEX_COL_NAME:
        return None
    if name.startswith("__index_") and name.endswith("__"):
        try:
            return int(name[8:-2])
        except ValueError:
            return None
    return None
