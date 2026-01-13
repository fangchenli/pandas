"""
Lazy pandas - opt-in lazy query mode for pandas DataFrames.

This module provides a lazy evaluation mode for pandas, allowing users
to build query plans that are optimized before execution.

Entry Point
-----------
Use DataFrame.select() to enter lazy mode:

    >>> import pandas as pd
    >>> from pandas.lazy import col, lit
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> ldf = df.select()  # All columns
    >>> ldf = df.select("a", "b")  # Specific columns
    >>> ldf = df.select(col("a"), col("b"))  # Using expressions

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
"""

from pandas.lazy.expr import (
    coalesce,
    col,
    lit,
    when,
)
from pandas.lazy.frame import LazyDataFrame

__all__ = [
    "LazyDataFrame",
    "coalesce",
    "col",
    "lit",
    "when",
]
