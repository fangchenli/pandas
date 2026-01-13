"""
Tests for pandas.lazy.plan module - Logical plan nodes.
"""

import pandas as pd
from pandas.lazy.expr import col
from pandas.lazy.plan import (
    DataFrameSource,
    Filter,
    LogicalPlan,
    Project,
)


class TestDataFrameSource:
    """Tests for DataFrameSource plan node."""

    def test_creation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        assert source.df is df

    def test_resolve_schema(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        source = DataFrameSource(df)
        schema = source.resolve_schema()
        assert "a" in schema
        assert "b" in schema
        assert schema["a"].is_numeric()
        assert schema["b"].is_string()

    def test_children_empty(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        assert source.children() == []

    def test_repr(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        result = repr(source)
        assert "DataFrameSource" in result
        assert "columns" in result
        assert "rows" in result

    def test_is_logical_plan(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        assert isinstance(source, LogicalPlan)


class TestProject:
    """Tests for Project plan node."""

    def test_creation(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        exprs = (col("a"), col("b"))
        project = Project(source, exprs)
        assert project.input is source
        assert project.exprs == exprs

    def test_resolve_schema(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        exprs = (col("a"),)
        project = Project(source, exprs)
        schema = project.resolve_schema()
        assert schema.names == ["a"]

    def test_resolve_schema_with_alias(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        exprs = (col("a").alias("new_a"),)
        project = Project(source, exprs)
        schema = project.resolve_schema()
        assert schema.names == ["new_a"]

    def test_children(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        project = Project(source, (col("a"),))
        children = project.children()
        assert len(children) == 1
        assert children[0] is source

    def test_repr(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        project = Project(source, (col("a"),))
        result = repr(project)
        assert "Project" in result

    def test_nested_project(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        project1 = Project(source, (col("a"), col("b")))
        project2 = Project(project1, (col("a"),))
        schema = project2.resolve_schema()
        assert schema.names == ["a"]


class TestFilter:
    """Tests for Filter plan node."""

    def test_creation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        source = DataFrameSource(df)
        predicate = col("a")  # Placeholder, actual predicate would be comparison
        filter_node = Filter(source, predicate)
        assert filter_node.input is source
        assert filter_node.predicate is predicate

    def test_resolve_schema_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        predicate = col("a")
        filter_node = Filter(source, predicate)
        # Filter doesn't change schema
        schema = filter_node.resolve_schema()
        assert schema.names == ["a", "b"]

    def test_children(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        filter_node = Filter(source, col("a"))
        children = filter_node.children()
        assert len(children) == 1
        assert children[0] is source

    def test_repr(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        filter_node = Filter(source, col("a"))
        result = repr(filter_node)
        assert "Filter" in result


class TestPlanComposition:
    """Tests for composing plan nodes."""

    def test_source_project(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        project = Project(source, (col("a"),))

        assert project.children() == [source]
        assert project.resolve_schema().names == ["a"]

    def test_source_project_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        source = DataFrameSource(df)
        project = Project(source, (col("a"), col("b")))
        filter_node = Filter(project, col("a"))

        assert filter_node.children() == [project]
        assert filter_node.resolve_schema().names == ["a", "b"]

    def test_plan_tree_depth(self):
        df = pd.DataFrame({"a": [1]})
        source = DataFrameSource(df)
        p1 = Project(source, (col("a"),))
        p2 = Project(p1, (col("a").alias("b"),))
        p3 = Project(p2, (col("b").alias("c"),))

        # Walk up the tree
        assert p3.input == p2
        assert p2.input == p1
        assert p1.input == source
        assert source.children() == []
