"""
Lazy pandas - opt-in lazy query mode for pandas DataFrames.

This module provides a lazy evaluation mode for pandas, allowing users
to build query plans that are optimized before execution.

Entry Points
------------
**From DataFrame**: Use DataFrame.select() to enter lazy mode:

    >>> import pandas as pd
    >>> from pandas.lazy import col, lit
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> ldf = df.select()  # All columns
    >>> ldf = df.select("a", "b")  # Specific columns
    >>> ldf = df.select(col("a"), col("b"))  # Using expressions

**From Files**: Use scan() to lazily read files:

    >>> from pandas.lazy import scan, col
    >>> ldf = scan("data.parquet")
    >>> ldf = scan("data/*.parquet")  # Glob patterns
    >>> ldf = scan("s3://bucket/data.parquet")  # URLs

Expression Building
-------------------
col(name)
    Reference a column by name.
lit(value)
    Create a literal value.

Query Execution
---------------
collect()
    Execute the query and return a DataFrame.
explain()
    Show the query plan without executing.

Examples
--------
>>> import pandas as pd
>>> from pandas.lazy import col, lit
>>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

# Simple column selection
>>> result = df.select("a", "b").collect()

# Column with alias
>>> result = df.select(col("a").alias("new_a")).collect()

# View the query plan
>>> print(df.select("a", "b").explain())

# Lazy file scanning with predicate pushdown
>>> from pandas.lazy import scan, col
>>> ldf = scan("data.parquet")
>>> result = ldf.filter(col("value") > 100).collect(use_physical_planner=True)
"""

from pandas.lazy.expr import (
    coalesce,
    col,
    lit,
    when,
)
from pandas.lazy.frame import LazyDataFrame
from pandas.lazy.scan import scan

__all__ = [
    "LazyDataFrame",
    "coalesce",
    "col",
    "lit",
    "scan",
    "when",
]
