"""
Arrow Join kernel implementations.

This module contains join operations (inner, left, right, outer, cross).
"""

import pyarrow as pa

from pandas.lazy.backends import register_kernel

# Join Operations
# =============================================================================


@register_kernel("hash_join", "arrow")
def arrow_hash_join(
    left_table: pa.Table,
    right_table: pa.Table,
    *,
    keys: list[str] | None = None,
    left_keys: list[str] | None = None,
    right_keys: list[str] | None = None,
    join_type: str = "inner",
    left_suffix: str = "",
    right_suffix: str = "_right",
    coalesce_keys: bool = True,
) -> pa.Table:
    """
    Perform a hash join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str] or None
        Column names to join on (used for both tables).
    left_keys : list[str] or None
        Column names from left table to join on.
    right_keys : list[str] or None
        Column names from right table to join on.
    join_type : str, default "inner"
        Type of join: "inner", "left", "right", "outer", "left semi",
        "right semi", "left anti", "right anti".
    left_suffix : str, default ""
        Suffix to add to left column names for disambiguation.
    right_suffix : str, default "_right"
        Suffix to add to right column names for disambiguation.
    coalesce_keys : bool, default True
        Whether to coalesce duplicate key columns.

    Returns
    -------
    pa.Table
        Result of the join operation.
    """
    # Map pandas join types to PyArrow join types
    join_type_map = {
        "inner": "inner",
        "left": "left outer",
        "right": "right outer",
        "outer": "full outer",
        "cross": None,  # Special case - handled separately
        "left_semi": "left semi",
        "right_semi": "right semi",
        "left_anti": "left anti",
        "right_anti": "right anti",
    }

    pa_join_type = join_type_map.get(join_type, join_type)

    if pa_join_type is None:
        raise ValueError(f"Join type '{join_type}' not supported by Arrow kernel")

    # Determine keys
    if keys is not None:
        join_keys = keys if isinstance(keys, list) else [keys]
        right_join_keys = None
    else:
        join_keys = left_keys if isinstance(left_keys, list) else [left_keys]
        right_join_keys = right_keys if isinstance(right_keys, list) else [right_keys]

    # Perform join
    result = left_table.join(
        right_table,
        keys=join_keys,
        right_keys=right_join_keys,
        join_type=pa_join_type,
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=coalesce_keys,
    )

    return result


@register_kernel("inner_join", "arrow")
def arrow_inner_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> pa.Table:
    """
    Perform an inner join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    pa.Table
        Result of the inner join.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="inner",
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=True,
    )


@register_kernel("left_join", "arrow")
def arrow_left_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> pa.Table:
    """
    Perform a left outer join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    pa.Table
        Result of the left join.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="left outer",
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=True,
    )


@register_kernel("right_join", "arrow")
def arrow_right_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> pa.Table:
    """
    Perform a right outer join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    pa.Table
        Result of the right join.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="right outer",
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=True,
    )


@register_kernel("outer_join", "arrow")
def arrow_outer_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
    *,
    left_suffix: str = "",
    right_suffix: str = "_right",
) -> pa.Table:
    """
    Perform a full outer join between two Arrow tables.

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.
    left_suffix : str, default ""
        Suffix to add to left column names.
    right_suffix : str, default "_right"
        Suffix to add to right column names.

    Returns
    -------
    pa.Table
        Result of the full outer join.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="full outer",
        left_suffix=left_suffix if left_suffix else None,
        right_suffix=right_suffix if right_suffix else None,
        coalesce_keys=True,
    )


@register_kernel("semi_join", "arrow")
def arrow_semi_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
) -> pa.Table:
    """
    Perform a left semi join (filter left by existence in right).

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.

    Returns
    -------
    pa.Table
        Rows from left table that have matches in right table.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="left semi",
        coalesce_keys=True,
    )


@register_kernel("anti_join", "arrow")
def arrow_anti_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: list[str],
) -> pa.Table:
    """
    Perform a left anti join (filter left by non-existence in right).

    Parameters
    ----------
    left_table : pa.Table
        Left table in the join.
    right_table : pa.Table
        Right table in the join.
    keys : list[str]
        Column names to join on.

    Returns
    -------
    pa.Table
        Rows from left table that have no matches in right table.
    """
    return left_table.join(
        right_table,
        keys=keys,
        join_type="left anti",
        coalesce_keys=True,
    )


# =============================================================================
