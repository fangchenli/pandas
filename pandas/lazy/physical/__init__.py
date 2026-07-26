"""
Physical plan representation for lazy pandas.

This package defines physical plan nodes (concrete execution strategies) and the
planner that converts an optimized logical plan into one. It was split out of a
single ~8k-line ``physical.py`` module; the submodules are:

- ``base`` — :class:`PhysicalPlan`, :class:`ExecutionContext`,
  :class:`PhysicalMaterialize`, shared helpers.
- ``scans`` — scan nodes (DataFrame/Parquet/CSV).
- ``project_filter`` — projection and filter nodes.
- ``groupby`` — hash-aggregate node and the group-by kernels/toggles.
- ``join`` — hash/sort-merge join, join chains, join helpers.
- ``reshape`` — convert, set/reset index, concat, cached subplan.
- ``fused`` — fused pipeline / filter-agg / join-agg nodes.
- ``sort_limit`` — sort, top-k, limit, distinct, group-by-head.
- ``planner`` — :class:`PhysicalPlanner` and :func:`execute_physical_plan`.

Everything public is re-exported here, so ``from pandas.lazy.physical import X``
keeps working unchanged.
"""

from __future__ import annotations

from pandas.lazy.physical.base import (
    ExecutionContext,
    PhysicalMaterialize,
    PhysicalPlan,
)
from pandas.lazy.physical.fused import (
    FusedOperation,
    PhysicalFusedFilterAgg,
    PhysicalFusedJoinAgg,
    PhysicalFusedPipeline,
)
from pandas.lazy.physical.groupby import (
    PhysicalHashAggregate,
    groupby_prefers_arrow,
)
from pandas.lazy.physical.join import (
    PhysicalHashJoin,
    PhysicalJoinChain,
    PhysicalSortMergeJoin,
)
from pandas.lazy.physical.planner import (
    PhysicalPlanner,
    execute_physical_plan,
)
from pandas.lazy.physical.project_filter import (
    PhysicalFilter,
    PhysicalProject,
)
from pandas.lazy.physical.reshape import (
    PhysicalCachedSubplan,
    PhysicalConcat,
    PhysicalConvert,
    PhysicalResetIndex,
    PhysicalSetIndex,
)
from pandas.lazy.physical.scans import (
    PhysicalCSVScan,
    PhysicalParquetScan,
    PhysicalScan,
)
from pandas.lazy.physical.sort_limit import (
    PhysicalDistinct,
    PhysicalGroupByHead,
    PhysicalLimit,
    PhysicalSort,
    PhysicalTopK,
)

__all__ = [
    "ExecutionContext",
    "FusedOperation",
    "PhysicalCSVScan",
    "PhysicalCachedSubplan",
    "PhysicalConcat",
    "PhysicalConvert",
    "PhysicalDistinct",
    "PhysicalFilter",
    "PhysicalFusedFilterAgg",
    "PhysicalFusedJoinAgg",
    "PhysicalFusedPipeline",
    "PhysicalGroupByHead",
    "PhysicalHashAggregate",
    "PhysicalHashJoin",
    "PhysicalJoinChain",
    "PhysicalLimit",
    "PhysicalMaterialize",
    "PhysicalParquetScan",
    "PhysicalPlan",
    "PhysicalPlanner",
    "PhysicalProject",
    "PhysicalResetIndex",
    "PhysicalScan",
    "PhysicalSetIndex",
    "PhysicalSort",
    "PhysicalSortMergeJoin",
    "PhysicalTopK",
    "execute_physical_plan",
    "groupby_prefers_arrow",
]
