"""
Tests for lazy pandas memory pooling system.

Tests the ArrayPool, ScratchBufferPool, and related utilities.
"""

import numpy as np
import pytest

import pandas._testing as tm


class TestPoolingStrategy:
    """Tests for PoolingStrategy enum."""

    def test_pooling_strategy_values(self):
        """Test PoolingStrategy enum values."""
        from pandas.lazy.backends.memory_pool import PoolingStrategy

        assert PoolingStrategy.NONE.value == "none"
        assert PoolingStrategy.SCRATCH.value == "scratch"
        assert PoolingStrategy.ACQUIRE_RELEASE.value == "acquire_release"

    def test_pooling_strategy_from_string(self):
        """Test creating PoolingStrategy from string."""
        from pandas.lazy.backends.memory_pool import PoolingStrategy

        assert PoolingStrategy("none") == PoolingStrategy.NONE
        assert PoolingStrategy("scratch") == PoolingStrategy.SCRATCH
        assert PoolingStrategy("acquire_release") == PoolingStrategy.ACQUIRE_RELEASE


class TestArrowPoolBackend:
    """Tests for ArrowPoolBackend enum and get_arrow_memory_pool."""

    def test_arrow_pool_backend_values(self):
        """Test ArrowPoolBackend enum values."""
        from pandas.lazy.backends.memory_pool import ArrowPoolBackend

        assert ArrowPoolBackend.DEFAULT.value == "default"
        assert ArrowPoolBackend.MIMALLOC.value == "mimalloc"
        assert ArrowPoolBackend.JEMALLOC.value == "jemalloc"
        assert ArrowPoolBackend.SYSTEM.value == "system"

    def test_get_arrow_memory_pool_default(self):
        """Test getting default Arrow memory pool."""
        import pyarrow as pa

        from pandas.lazy.backends.memory_pool import get_arrow_memory_pool

        pool = get_arrow_memory_pool("default")
        assert pool is not None
        assert isinstance(pool, pa.MemoryPool)

    def test_get_arrow_memory_pool_system(self):
        """Test getting system Arrow memory pool."""
        import pyarrow as pa

        from pandas.lazy.backends.memory_pool import get_arrow_memory_pool

        pool = get_arrow_memory_pool("system")
        assert pool is not None
        assert isinstance(pool, pa.MemoryPool)

    def test_get_arrow_memory_pool_mimalloc(self):
        """Test getting mimalloc Arrow memory pool."""
        import pyarrow as pa

        from pandas.lazy.backends.memory_pool import get_arrow_memory_pool

        try:
            pool = get_arrow_memory_pool("mimalloc")
            assert pool is not None
            assert isinstance(pool, pa.MemoryPool)
        except NotImplementedError:
            # mimalloc may not be available on all systems
            pytest.skip("mimalloc not available")

    def test_get_arrow_memory_pool_jemalloc(self):
        """Test getting jemalloc Arrow memory pool."""
        import pyarrow as pa

        from pandas.lazy.backends.memory_pool import get_arrow_memory_pool

        try:
            pool = get_arrow_memory_pool("jemalloc")
            assert pool is not None
            assert isinstance(pool, pa.MemoryPool)
        except NotImplementedError:
            # jemalloc may not be available on all systems
            pytest.skip("jemalloc not available")

    def test_get_arrow_memory_pool_enum(self):
        """Test getting Arrow memory pool with enum."""
        import pyarrow as pa

        from pandas.lazy.backends.memory_pool import (
            ArrowPoolBackend,
            get_arrow_memory_pool,
        )

        pool = get_arrow_memory_pool(ArrowPoolBackend.DEFAULT)
        assert isinstance(pool, pa.MemoryPool)


class TestCanUsePooledOutput:
    """Tests for can_use_pooled_output function."""

    def test_poolable_arithmetic_ops(self):
        """Test that arithmetic ops are poolable."""
        from pandas.lazy.backends.memory_pool import can_use_pooled_output

        assert can_use_pooled_output("add")
        assert can_use_pooled_output("subtract")
        assert can_use_pooled_output("multiply")
        assert can_use_pooled_output("divide")

    def test_poolable_comparison_ops(self):
        """Test that comparison ops are poolable."""
        from pandas.lazy.backends.memory_pool import can_use_pooled_output

        assert can_use_pooled_output("equal")
        assert can_use_pooled_output("not_equal")
        assert can_use_pooled_output("less")
        assert can_use_pooled_output("greater")

    def test_poolable_logical_ops(self):
        """Test that logical ops are poolable."""
        from pandas.lazy.backends.memory_pool import can_use_pooled_output

        assert can_use_pooled_output("and_")
        assert can_use_pooled_output("or_")
        assert can_use_pooled_output("invert")

    def test_non_poolable_ops(self):
        """Test that some ops are not poolable."""
        from pandas.lazy.backends.memory_pool import can_use_pooled_output

        assert not can_use_pooled_output("sum")
        assert not can_use_pooled_output("mean")
        assert not can_use_pooled_output("unknown_op")


class TestPoolingIntegration:
    """Integration tests for pooling with ArrayEvaluator."""

    def test_evaluator_uses_scratch_pool(self):
        """Test that ArrayEvaluator uses scratch pool."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.memory_pool import PoolingStrategy
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"a": np.array([1.0, 2.0, 3.0] * 10000)}
        evaluator = ArrayEvaluator(
            arrays, preferred_backend="numpy", pooling_strategy=PoolingStrategy.SCRATCH
        )

        # Evaluate a + 10
        node = Call("add", args=(FieldRef("a"), Literal(10)))
        result = evaluator.evaluate(node)

        expected = arrays["a"] + 10
        tm.assert_numpy_array_equal(result, expected)

    def test_evaluator_no_pooling(self):
        """Test ArrayEvaluator with no pooling."""
        from pandas.lazy.backends.array_eval import ArrayEvaluator
        from pandas.lazy.backends.memory_pool import PoolingStrategy
        from pandas.lazy.ir import (
            Call,
            FieldRef,
            Literal,
        )

        arrays = {"a": np.array([1.0, 2.0, 3.0] * 10000)}
        evaluator = ArrayEvaluator(
            arrays, preferred_backend="numpy", pooling_strategy=PoolingStrategy.NONE
        )

        node = Call("add", args=(FieldRef("a"), Literal(10)))
        result = evaluator.evaluate(node)

        expected = arrays["a"] + 10
        tm.assert_numpy_array_equal(result, expected)
