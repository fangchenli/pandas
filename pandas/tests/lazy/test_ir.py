"""
Tests for pandas.lazy.ir module - IR node types.
"""

import pytest

from pandas.lazy.ir import (
    Alias,
    Call,
    Cast,
    FieldRef,
    IRNode,
    Literal,
)


class TestFieldRef:
    """Tests for FieldRef IR node."""

    def test_creation(self):
        node = FieldRef("column_name")
        assert node.name == "column_name"

    def test_repr(self):
        node = FieldRef("col")
        assert repr(node) == "FieldRef('col')"

    def test_is_ir_node(self):
        node = FieldRef("col")
        assert isinstance(node, IRNode)

    def test_frozen(self):
        node = FieldRef("col")
        with pytest.raises(AttributeError):
            node.name = "other"

    def test_equality(self):
        node1 = FieldRef("col")
        node2 = FieldRef("col")
        node3 = FieldRef("other")
        assert node1 == node2
        assert node1 != node3

    def test_hash(self):
        node1 = FieldRef("col")
        node2 = FieldRef("col")
        assert hash(node1) == hash(node2)
        # Can use in sets/dicts
        s = {node1, node2}
        assert len(s) == 1


class TestLiteral:
    """Tests for Literal IR node."""

    def test_creation_int(self):
        node = Literal(42)
        assert node.value == 42
        assert node.dtype is None

    def test_creation_with_dtype(self):
        from pandas.lazy.types import LazyDtype

        dtype = LazyDtype("numeric", None, None, False)
        node = Literal(42, dtype)
        assert node.value == 42
        assert node.dtype == dtype

    def test_creation_string(self):
        node = Literal("hello")
        assert node.value == "hello"

    def test_creation_float(self):
        node = Literal(3.14)
        assert node.value == 3.14

    def test_creation_bool(self):
        node = Literal(True)
        assert node.value is True

    def test_repr_without_dtype(self):
        node = Literal(42)
        assert repr(node) == "Literal(42)"

    def test_repr_with_dtype(self):
        from pandas.lazy.types import LazyDtype

        dtype = LazyDtype("numeric", None, None, False)
        node = Literal(42, dtype)
        assert "dtype=" in repr(node)

    def test_frozen(self):
        node = Literal(42)
        with pytest.raises(AttributeError):
            node.value = 100


class TestAlias:
    """Tests for Alias IR node."""

    def test_creation(self):
        inner = FieldRef("col")
        node = Alias(inner, "new_name")
        assert node.arg == inner
        assert node.name == "new_name"

    def test_repr(self):
        inner = FieldRef("col")
        node = Alias(inner, "new_name")
        assert "Alias" in repr(node)
        assert "new_name" in repr(node)

    def test_nested_alias(self):
        inner = FieldRef("col")
        alias1 = Alias(inner, "name1")
        alias2 = Alias(alias1, "name2")
        assert alias2.name == "name2"
        assert alias2.arg.name == "name1"


class TestCall:
    """Tests for Call IR node."""

    def test_creation_simple(self):
        node = Call("add", ())
        assert node.function == "add"
        assert node.args == ()
        assert node.kwargs == {}
        assert node.is_aggregate is False

    def test_creation_with_args(self):
        arg1 = FieldRef("a")
        arg2 = FieldRef("b")
        node = Call("add", (arg1, arg2))
        assert node.function == "add"
        assert len(node.args) == 2
        assert node.args[0] == arg1
        assert node.args[1] == arg2

    def test_creation_with_kwargs(self):
        arg = FieldRef("col")
        node = Call("str_contains", (arg,), {"regex": True, "case": False})
        assert node.kwargs["regex"] is True
        assert node.kwargs["case"] is False

    def test_creation_aggregate(self):
        arg = FieldRef("col")
        node = Call("sum", (arg,), is_aggregate=True)
        assert node.is_aggregate is True

    def test_repr(self):
        arg1 = FieldRef("a")
        arg2 = Literal(10)
        node = Call("add", (arg1, arg2))
        result = repr(node)
        assert "Call" in result
        assert "add" in result


class TestCast:
    """Tests for Cast IR node."""

    def test_creation(self):
        inner = FieldRef("col")
        node = Cast(inner, "float64")
        assert node.arg == inner
        assert node.target_dtype == "float64"

    def test_repr(self):
        inner = FieldRef("col")
        node = Cast(inner, "float64")
        result = repr(node)
        assert "Cast" in result
        assert "float64" in result
