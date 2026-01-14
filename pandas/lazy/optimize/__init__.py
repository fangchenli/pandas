"""
Query optimization for lazy pandas.

This module implements a rule-based optimizer with multiple transformation passes.
Each pass traverses the logical plan tree and applies transformations.

Pass Ordering (critical for correctness):
1. ConstantFolding - Evaluate constant expressions first
2. FilterFusion - Combine consecutive filters
3. PredicatePushdown - Push filters toward data sources
4. ProjectionPruning - Remove unused columns
5. LimitPushdown - Push limits toward sources
6. SortLimitToTopK - Combine Sort+Limit into TopK
7. EngineSelection - Insert Convert nodes for backend decisions
8. ConversionElimination - Remove redundant conversions
"""

from pandas.lazy.optimize.base import (
    OptimizationPass,
    Optimizer,
    PlanVisitor,
)
from pandas.lazy.optimize.engine import (
    BackendRequirements,
    ConversionElimination,
    EngineSelection,
    analyze_backend_requirements,
)
from pandas.lazy.optimize.passes import (
    AggregatePushdown,
    CommonSubexpressionElimination,
    ConstantFolding,
    DeadCodeElimination,
    FilterFusion,
    LimitPushdown,
    PredicatePushdown,
    ProjectionPruning,
    SortLimitToTopK,
)
from pandas.lazy.optimize.utils import (
    JoinColumnMapping,
    build_join_column_mapping,
    build_project_lineage,
    can_push_predicate_through_join,
    can_push_predicate_through_project,
    extract_output_name,
    get_referenced_columns,
    get_source_column_name,
    is_pass_through_column,
    substitute_columns,
)

__all__ = [
    "AggregatePushdown",
    "BackendRequirements",
    "CommonSubexpressionElimination",
    "ConstantFolding",
    "ConversionElimination",
    "DeadCodeElimination",
    "EngineSelection",
    "FilterFusion",
    "JoinColumnMapping",
    "LimitPushdown",
    "OptimizationPass",
    "Optimizer",
    "PlanVisitor",
    "PredicatePushdown",
    "ProjectionPruning",
    "SortLimitToTopK",
    "analyze_backend_requirements",
    "build_join_column_mapping",
    "build_project_lineage",
    "can_push_predicate_through_join",
    "can_push_predicate_through_project",
    "extract_output_name",
    "get_referenced_columns",
    "get_source_column_name",
    "is_pass_through_column",
    "substitute_columns",
]
