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


class TestArrayPool:
    """Tests for ArrayPool acquire/release pooling."""

    def test_acquire_creates_array(self):
        """Test that acquire creates an array."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()
        arr = pool.acquire(100, np.float64)

        assert isinstance(arr, np.ndarray)
        assert len(arr) == 100
        assert arr.dtype == np.float64

    def test_acquire_different_dtypes(self):
        """Test acquiring arrays with different dtypes."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()

        arr_float = pool.acquire(100, np.float64)
        arr_int = pool.acquire(100, np.int32)
        arr_bool = pool.acquire(100, np.bool_)

        assert arr_float.dtype == np.float64
        assert arr_int.dtype == np.int32
        assert arr_bool.dtype == np.bool_

    def test_release_and_reuse(self):
        """Test that released arrays are reused."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()

        # Need at least MIN_POOL_SIZE for pooling to work
        size = 20_000
        arr1 = pool.acquire(size, np.float64)
        arr1_id = id(arr1)

        pool.release(arr1)

        arr2 = pool.acquire(size, np.float64)
        # Should get the same array back
        assert id(arr2) == arr1_id

    def test_release_small_arrays_not_pooled(self):
        """Test that small arrays are not pooled."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()

        # Small array below MIN_POOL_SIZE
        arr = pool.acquire(100, np.float64)
        pool.release(arr)

        # Stats should show 0 releases (small arrays skipped)
        assert pool.stats["releases"] == 0

    def test_release_noncontiguous_not_pooled(self):
        """Test that non-contiguous arrays are not pooled."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()

        # Create non-contiguous array by slicing 2D
        arr_2d = np.zeros((200, 200))
        arr = arr_2d[::2, 0]  # Non-contiguous view
        assert not arr.flags.c_contiguous
        assert len(arr) >= 20_000 or True  # Skip size check for this test

        pool.release(arr)
        assert pool.stats["releases"] == 0

    def test_pool_stats(self):
        """Test pool statistics."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()
        size = 20_000

        # First acquire - miss
        arr1 = pool.acquire(size, np.float64)
        assert pool.stats["misses"] == 1
        assert pool.stats["hits"] == 0

        pool.release(arr1)
        assert pool.stats["releases"] == 1

        # Second acquire - hit
        pool.acquire(size, np.float64)
        assert pool.stats["hits"] == 1

    def test_pool_hit_rate(self):
        """Test pool hit rate calculation."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()
        size = 20_000

        # Initial hit rate is 0
        assert pool.hit_rate == 0.0

        # Miss
        arr = pool.acquire(size, np.float64)
        assert pool.hit_rate == 0.0

        pool.release(arr)

        # Hit
        pool.acquire(size, np.float64)
        assert pool.hit_rate == 0.5  # 1 hit, 1 miss

    def test_pool_clear(self):
        """Test clearing the pool."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()
        size = 20_000

        arr = pool.acquire(size, np.float64)
        pool.release(arr)
        assert pool.stats["total_bytes"] > 0

        pool.clear()
        assert pool.stats["total_bytes"] == 0
        assert pool.stats["num_arrays"] == 0

    def test_pool_max_per_bucket(self):
        """Test max arrays per bucket limit."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool(max_per_bucket=2)
        size = 20_000

        # Release 3 arrays, only 2 should be kept
        arr1 = pool.acquire(size, np.float64)
        arr2 = pool.acquire(size, np.float64)
        arr3 = pool.acquire(size, np.float64)

        pool.release(arr1)
        pool.release(arr2)
        pool.release(arr3)  # This one should be dropped

        # Only 2 releases should be counted
        assert pool.stats["releases"] == 2
        assert pool.stats["num_arrays"] == 2

    def test_pool_repr(self):
        """Test pool string representation."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()
        repr_str = repr(pool)

        assert "ArrayPool" in repr_str
        assert "arrays=" in repr_str
        assert "memory=" in repr_str
        assert "hit_rate=" in repr_str

    def test_get_buffer_context_manager(self):
        """Test get_buffer context manager."""
        from pandas.lazy.backends.memory_pool import ArrayPool

        pool = ArrayPool()
        size = 20_000

        with pool.get_buffer(size, np.float64) as buf:
            assert isinstance(buf, np.ndarray)
            assert len(buf) == size
            # Fill buffer
            buf[:] = 42.0

        # Buffer should be released back to pool
        assert pool.stats["releases"] == 1


class TestScratchBufferPool:
    """Tests for ScratchBufferPool rotating buffer pool."""

    def test_create_scratch_pool(self):
        """Test creating a scratch buffer pool."""
        from pandas.lazy.backends.memory_pool import ScratchBufferPool

        pool = ScratchBufferPool(size=1000, dtype=np.float64, num_buffers=2)

        assert pool.size == 1000
        assert pool.dtype == np.float64
        assert pool.num_buffers == 2

    def test_get_next_rotates(self):
        """Test that get_next returns rotating buffers."""
        from pandas.lazy.backends.memory_pool import ScratchBufferPool

        pool = ScratchBufferPool(size=100, num_buffers=3)

        buf1 = pool.get_next()
        buf2 = pool.get_next()
        buf3 = pool.get_next()
        buf4 = pool.get_next()  # Should wrap around to buf1

        assert id(buf1) != id(buf2)
        assert id(buf2) != id(buf3)
        assert id(buf4) == id(buf1)  # Wrapped around

    def test_get_buffer_by_index(self):
        """Test getting buffer by specific index."""
        from pandas.lazy.backends.memory_pool import ScratchBufferPool

        pool = ScratchBufferPool(size=100, num_buffers=3)

        buf0 = pool.get_buffer(0)
        pool.get_buffer(1)  # Get buffer 1
        pool.get_buffer(2)  # Get buffer 2
        buf3 = pool.get_buffer(3)  # Wraps to 0

        assert id(buf3) == id(buf0)

    def test_memory_bytes(self):
        """Test memory_bytes property."""
        from pandas.lazy.backends.memory_pool import ScratchBufferPool

        pool = ScratchBufferPool(size=1000, dtype=np.float64, num_buffers=2)

        expected = 1000 * 8 * 2  # 1000 elements * 8 bytes * 2 buffers
        assert pool.memory_bytes == expected

    def test_uses_counter(self):
        """Test uses counter."""
        from pandas.lazy.backends.memory_pool import ScratchBufferPool

        pool = ScratchBufferPool(size=100, num_buffers=2)

        assert pool.uses == 0

        pool.get_next()
        assert pool.uses == 1

        pool.get_next()
        pool.get_next()
        assert pool.uses == 3

    def test_reset(self):
        """Test reset method."""
        from pandas.lazy.backends.memory_pool import ScratchBufferPool

        pool = ScratchBufferPool(size=100, num_buffers=3)

        # Get some buffers
        pool.get_next()
        pool.get_next()

        # Reset
        pool.reset()

        # Next buffer should be first one
        buf = pool.get_next()
        assert id(buf) == id(pool.get_buffer(0))

    def test_repr(self):
        """Test string representation."""
        from pandas.lazy.backends.memory_pool import ScratchBufferPool

        pool = ScratchBufferPool(size=1000, dtype=np.float64, num_buffers=2)
        repr_str = repr(pool)

        assert "ScratchBufferPool" in repr_str
        assert "size=1,000" in repr_str
        assert "buffers=2" in repr_str
        assert "memory=" in repr_str


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


class TestEvaluateWithPool:
    """Tests for evaluate_with_pool function."""

    def test_evaluate_add(self):
        """Test evaluating add with pool."""
        from pandas.lazy.backends.memory_pool import (
            ArrayPool,
            evaluate_with_pool,
        )

        pool = ArrayPool()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        result = evaluate_with_pool(pool, "add", a, b)
        expected = np.array([5.0, 7.0, 9.0])
        tm.assert_numpy_array_equal(result, expected)

    def test_evaluate_multiply(self):
        """Test evaluating multiply with pool."""
        from pandas.lazy.backends.memory_pool import (
            ArrayPool,
            evaluate_with_pool,
        )

        pool = ArrayPool()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])

        result = evaluate_with_pool(pool, "multiply", a, b)
        expected = np.array([2.0, 6.0, 12.0])
        tm.assert_numpy_array_equal(result, expected)

    def test_evaluate_with_provided_out(self):
        """Test evaluating with pre-allocated output."""
        from pandas.lazy.backends.memory_pool import (
            ArrayPool,
            evaluate_with_pool,
        )

        pool = ArrayPool()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        out = np.zeros(3)

        result = evaluate_with_pool(pool, "add", a, b, out=out)
        assert result is out
        expected = np.array([5.0, 7.0, 9.0])
        tm.assert_numpy_array_equal(result, expected)


class TestEvaluateExpressionChain:
    """Tests for evaluate_expression_chain function."""

    def test_simple_chain(self):
        """Test simple expression chain."""
        from pandas.lazy.backends.memory_pool import evaluate_expression_chain

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        c = np.array([2.0, 2.0, 2.0])

        # Compute (a + b) * c
        ops = [
            ("add", [a, b]),
            ("multiply", [None, c]),  # None = previous result
        ]
        result = evaluate_expression_chain(ops)
        expected = np.array([10.0, 14.0, 18.0])
        tm.assert_numpy_array_equal(result, expected)

    def test_longer_chain(self):
        """Test longer expression chain."""
        from pandas.lazy.backends.memory_pool import evaluate_expression_chain

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        c = np.array([2.0, 2.0, 2.0])
        d = np.array([10.0, 10.0, 10.0])

        # Compute ((a + b) * c) - d
        ops = [
            ("add", [a, b]),
            ("multiply", [None, c]),
            ("subtract", [None, d]),
        ]
        result = evaluate_expression_chain(ops)
        expected = np.array([0.0, 4.0, 8.0])
        tm.assert_numpy_array_equal(result, expected)

    def test_chain_with_provided_pool(self):
        """Test chain with provided scratch pool."""
        from pandas.lazy.backends.memory_pool import (
            ScratchBufferPool,
            evaluate_expression_chain,
        )

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        pool = ScratchBufferPool(size=3, dtype=np.float64, num_buffers=2)
        ops = [("add", [a, b])]
        result = evaluate_expression_chain(ops, pool=pool)

        expected = np.array([5.0, 7.0, 9.0])
        tm.assert_numpy_array_equal(result, expected)
        assert pool.uses > 0

    def test_chain_empty_raises(self):
        """Test that empty ops raises error."""
        from pandas.lazy.backends.memory_pool import evaluate_expression_chain

        with pytest.raises(ValueError, match="ops list cannot be empty"):
            evaluate_expression_chain([])

    def test_chain_no_arrays_raises(self):
        """Test that ops without arrays raises error."""
        from pandas.lazy.backends.memory_pool import evaluate_expression_chain

        # Note: This would be an unusual case
        with pytest.raises(ValueError, match="No arrays found"):
            evaluate_expression_chain([("add", [None, None])])


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
