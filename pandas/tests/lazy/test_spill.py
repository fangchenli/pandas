"""
Tests for lazy pandas spill manager.

Tests the disk spilling infrastructure for handling memory pressure
during physical plan execution.
"""

from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa

import pandas._testing as tm


class TestSpillConfig:
    """Tests for SpillConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from pandas.lazy.backends.spill import SpillConfig

        config = SpillConfig()
        assert not config.enabled
        assert config.threshold_mb == 2048
        assert config.operator_budget_mb == 512
        assert config.spill_dir is None

    def test_threshold_bytes(self):
        """Test threshold conversion to bytes."""
        from pandas.lazy.backends.spill import SpillConfig

        config = SpillConfig(threshold_mb=1024)
        assert config.threshold_bytes == 1024 * 1024 * 1024

    def test_custom_spill_dir(self):
        """Test custom spill directory."""
        from pandas.lazy.backends.spill import SpillConfig

        config = SpillConfig(spill_dir="/tmp/test_spill")
        assert config.spill_dir == Path("/tmp/test_spill")


class TestMemoryTracking:
    """Tests for memory tracking utilities."""

    def test_get_array_bytes_numpy(self):
        """Test getting bytes for NumPy array."""
        from pandas.lazy.backends.spill import get_array_bytes

        arr = np.zeros(1000, dtype=np.float64)
        assert get_array_bytes(arr) == 1000 * 8

    def test_get_array_bytes_arrow(self):
        """Test getting bytes for Arrow array."""
        from pandas.lazy.backends.spill import get_array_bytes

        arr = pa.array([1, 2, 3, 4, 5])
        assert get_array_bytes(arr) > 0

    def test_get_array_bytes_chunked(self):
        """Test getting bytes for ChunkedArray."""
        from pandas.lazy.backends.spill import get_array_bytes

        arr = pa.chunked_array([[1, 2], [3, 4, 5]])
        assert get_array_bytes(arr) > 0

    def test_get_arrays_bytes(self):
        """Test getting total bytes for ArrayDict."""
        from pandas.lazy.backends.spill import get_arrays_bytes

        arrays = {
            "a": np.zeros(1000, dtype=np.float64),
            "b": np.zeros(1000, dtype=np.int32),
        }
        expected = 1000 * 8 + 1000 * 4
        assert get_arrays_bytes(arrays) == expected


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_register_and_track(self):
        """Test registering arrays for tracking."""
        from pandas.lazy.backends.spill import (
            MemoryTracker,
            SpillConfig,
        )

        config = SpillConfig(enabled=True, threshold_mb=1)
        tracker = MemoryTracker(config)

        arrays = {"a": np.zeros(1000, dtype=np.float64)}
        size = tracker.register("test", arrays)

        assert size == 8000
        assert tracker.tracked_mb > 0
        assert tracker.get_tracked("test") is not None

    def test_unregister(self):
        """Test unregistering arrays."""
        from pandas.lazy.backends.spill import (
            MemoryTracker,
            SpillConfig,
        )

        config = SpillConfig(enabled=True)
        tracker = MemoryTracker(config)

        arrays = {"a": np.zeros(1000, dtype=np.float64)}
        tracker.register("test", arrays)
        tracker.unregister("test")

        assert tracker.get_tracked("test") is None
        assert tracker.tracked_mb == 0

    def test_should_spill_when_over_threshold(self):
        """Test spill trigger when over threshold."""
        from pandas.lazy.backends.spill import (
            MemoryTracker,
            SpillConfig,
        )

        # Set tiny threshold (1 KB)
        config = SpillConfig(enabled=True, threshold_mb=0.001)
        tracker = MemoryTracker(config)

        # Register array larger than threshold
        arrays = {"a": np.zeros(10000, dtype=np.float64)}  # 80 KB
        tracker.register("test", arrays)

        assert tracker.should_spill()

    def test_should_not_spill_when_disabled(self):
        """Test no spill trigger when disabled."""
        from pandas.lazy.backends.spill import (
            MemoryTracker,
            SpillConfig,
        )

        config = SpillConfig(enabled=False, threshold_mb=0.001)
        tracker = MemoryTracker(config)

        arrays = {"a": np.zeros(10000, dtype=np.float64)}
        tracker.register("test", arrays)

        assert not tracker.should_spill()


class TestSpillFileManager:
    """Tests for SpillFileManager class."""

    def test_spill_dir_creation(self):
        """Test spill directory is created."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillFileManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(spill_dir=tmpdir)
            manager = SpillFileManager(config)

            # Access spill_dir to trigger creation
            spill_dir = manager.spill_dir
            assert spill_dir.exists()

            manager.cleanup()

    def test_get_spill_path(self):
        """Test spill path generation."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillFileManager,
            SpillFormat,
        )

        config = SpillConfig(format=SpillFormat.ARROW_IPC)
        manager = SpillFileManager(config)

        path = manager.get_spill_path("test_data")
        assert path.suffix == ".arrow"

        manager.cleanup()


class TestSpillOperations:
    """Tests for spill and reload operations."""

    def test_spill_arrays_ipc(self):
        """Test spilling arrays to Arrow IPC."""
        from pandas.lazy.backends.spill import (
            reload_arrays_ipc,
            spill_arrays_ipc,
        )

        arrays = {
            "a": np.array([1, 2, 3, 4, 5]),
            "b": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.arrow"
            spill_file = spill_arrays_ipc(arrays, path)

            assert spill_file.exists()
            assert spill_file.num_rows == 5
            assert spill_file.num_columns == 2
            assert set(spill_file.columns) == {"a", "b"}

            # Reload
            reloaded = reload_arrays_ipc(path)
            assert set(reloaded.keys()) == {"a", "b"}
            assert len(reloaded["a"]) == 5

    def test_spill_arrays_parquet(self):
        """Test spilling arrays to Parquet."""
        from pandas.lazy.backends.spill import (
            reload_arrays_parquet,
            spill_arrays_parquet,
        )

        arrays = {
            "a": np.array([1, 2, 3, 4, 5]),
            "b": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            spill_file = spill_arrays_parquet(arrays, path)

            assert spill_file.exists()
            assert spill_file.num_rows == 5

            # Reload
            reloaded = reload_arrays_parquet(path)
            assert set(reloaded.keys()) == {"a", "b"}
            assert len(reloaded["a"]) == 5


class TestSpillManager:
    """Tests for SpillManager class."""

    def test_register_and_get(self):
        """Test registering and getting arrays."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                arrays = {"a": np.array([1, 2, 3])}
                manager.register("test", arrays)

                result = manager.get("test")
                assert result is not None
                tm.assert_numpy_array_equal(result["a"], arrays["a"])

    def test_spill_and_reload(self):
        """Test spilling and reloading arrays."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                arrays = {"a": np.array([1, 2, 3, 4, 5])}
                manager.register("test", arrays)

                # Spill
                spill_file = manager.spill("test")
                assert spill_file is not None
                assert spill_file.exists()

                # Reload
                reloaded = manager.reload("test")
                assert reloaded is not None
                assert len(reloaded["a"]) == 5

    def test_spill_largest(self):
        """Test spilling largest in-memory data."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                # Register two arrays of different sizes
                small = {"a": np.zeros(100)}
                large = {"b": np.zeros(10000)}

                manager.register("small", small)
                manager.register("large", large)

                # Spill largest
                spill_file = manager.spill_largest()
                assert spill_file is not None
                assert "large" in spill_file.name

    def test_check_memory_pressure(self):
        """Test automatic spilling under memory pressure."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Very low threshold to trigger spilling
            config = SpillConfig(enabled=True, threshold_mb=0.0001, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                arrays = {"a": np.zeros(10000, dtype=np.float64)}
                manager.register("test", arrays)

                # Check pressure - should spill
                spilled = manager.check_memory_pressure()
                assert len(spilled) > 0

    def test_stats(self):
        """Test statistics reporting."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                arrays = {"a": np.zeros(1000)}
                manager.register("test", arrays)

                stats = manager.stats
                assert "tracked_mb" in stats
                assert "num_spill_files" in stats

    def test_unregister_deletes_spill_file(self):
        """Test that unregister deletes spill files."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                arrays = {"a": np.array([1, 2, 3])}
                manager.register("test", arrays)

                spill_file = manager.spill("test")
                assert spill_file.exists()

                manager.unregister("test")
                assert not spill_file.exists()


class TestPartitionedSpilling:
    """Tests for partitioned spilling (Grace hash join support)."""

    def test_spill_partitioned(self):
        """Test partitioned spilling by hash key."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                # Create data with partition key
                arrays = {
                    "key": np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                    "value": np.array([10, 20, 30, 40, 50, 60, 70, 80]),
                }

                # Partition into 4 partitions
                spill_files = manager.spill_partitioned(
                    "test", arrays, partition_key="key", num_partitions=4
                )

                # Should have created some partition files
                assert len(spill_files) > 0

                # Each partition should have partition_id set
                for sf in spill_files:
                    assert sf.partition_id is not None


class TestSortedRunSpilling:
    """Tests for sorted run spilling (external sort support)."""

    def test_spill_sorted_run(self):
        """Test spilling a sorted run."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                # Create sorted arrays
                sorted_arrays = {
                    "key": np.array([1, 2, 3, 4, 5]),
                    "value": np.array([10, 20, 30, 40, 50]),
                }

                spill_file = manager.spill_sorted_run("sort", sorted_arrays, run_id=0)

                assert spill_file is not None
                assert spill_file.partition_id == 0
                assert "_run_" in spill_file.name

    def test_get_sorted_runs(self):
        """Test getting all sorted runs."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                # Create multiple runs
                for i in range(3):
                    arrays = {"key": np.array([i, i + 1, i + 2])}
                    manager.spill_sorted_run("sort", arrays, run_id=i)

                runs = manager.get_sorted_runs("sort")
                assert len(runs) == 3


class TestExternalSorter:
    """Tests for external merge sort."""

    def test_single_batch_sort(self):
        """Test sorting a single batch (no spilling needed)."""
        from pandas.lazy.backends.spill import (
            ExternalSorter,
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                sorter = ExternalSorter(manager, run_size_mb=1)

                arrays = {
                    "key": np.array([3, 1, 4, 1, 5, 9, 2, 6]),
                    "value": np.array([30, 10, 40, 11, 50, 90, 20, 60]),
                }

                sorter.add_batch(arrays, sort_keys=["key"])
                result = sorter.finish(sort_keys=["key"])

                # Should be sorted by key
                # Result arrays may be Arrow arrays, so convert to list
                key_arr = result["key"]
                key_list = (
                    key_arr.to_pylist()
                    if hasattr(key_arr, "to_pylist")
                    else list(key_arr)
                )
                assert key_list == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_multi_batch_sort(self):
        """Test sorting multiple batches with spilling."""
        from pandas.lazy.backends.spill import (
            ExternalSorter,
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                # Small run size to force multiple runs
                sorter = ExternalSorter(manager, run_size_mb=0.0001)

                # Add multiple batches
                batch1 = {"key": np.array([5, 3, 1]), "value": np.array([50, 30, 10])}
                batch2 = {"key": np.array([6, 4, 2]), "value": np.array([60, 40, 20])}

                sorter.add_batch(batch1, sort_keys=["key"])
                sorter.add_batch(batch2, sort_keys=["key"])

                result = sorter.finish(sort_keys=["key"])

                # Should be sorted
                # Result arrays may be Arrow arrays, so convert to list
                key_arr = result["key"]
                key_list = (
                    key_arr.to_pylist()
                    if hasattr(key_arr, "to_pylist")
                    else list(key_arr)
                )
                assert key_list == [1, 2, 3, 4, 5, 6]


class TestGraceHashJoiner:
    """Tests for Grace hash join."""

    def test_partition_and_join(self):
        """Test partitioning and joining."""
        from pandas.lazy.backends.spill import (
            GraceHashJoiner,
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                joiner = GraceHashJoiner(manager, num_partitions=4)

                left = {
                    "key": np.array([1, 2, 3, 4]),
                    "left_val": np.array([10, 20, 30, 40]),
                }
                right = {
                    "key": np.array([2, 3, 4, 5]),
                    "right_val": np.array([200, 300, 400, 500]),
                }

                joiner.partition_left(left, ["key"])
                joiner.partition_right(right, ["key"])

                result = joiner.join(how="inner")

                # Should have joined results
                assert len(result) > 0
                assert "key" in result
                assert "left_val" in result
                assert "right_val" in result


class TestExecutionContextIntegration:
    """Tests for ExecutionContext integration with spill manager."""

    def test_spill_manager_property(self):
        """Test spill_manager property creates manager when config set."""
        from pandas.lazy.backends.spill import SpillConfig
        from pandas.lazy.physical import ExecutionContext

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)
            context = ExecutionContext(_spill_config=config)

            manager = context.spill_manager
            assert manager is not None

            # Cleanup
            manager.close()

    def test_spill_enabled_property(self):
        """Test spill_enabled property."""
        from pandas.lazy.backends.spill import SpillConfig
        from pandas.lazy.physical import ExecutionContext

        # Disabled by default
        context1 = ExecutionContext()
        assert not context1.spill_enabled

        # Enabled when config is set
        config = SpillConfig(enabled=True)
        context2 = ExecutionContext(_spill_config=config)
        assert context2.spill_enabled

        # Disabled when config.enabled=False
        config_disabled = SpillConfig(enabled=False)
        context3 = ExecutionContext(_spill_config=config_disabled)
        assert not context3.spill_enabled

    def test_check_memory_pressure(self):
        """Test check_memory_pressure method."""
        from pandas.lazy.backends.spill import SpillConfig
        from pandas.lazy.physical import ExecutionContext

        with tempfile.TemporaryDirectory() as tmpdir:
            # Low threshold to trigger spilling
            config = SpillConfig(enabled=True, threshold_mb=0.0001, spill_dir=tmpdir)
            context = ExecutionContext(_spill_config=config)

            # Register some data
            manager = context.spill_manager
            arrays = {"a": np.zeros(10000)}
            manager.register("test", arrays)

            # Check pressure - should spill
            spilled = context.check_memory_pressure()
            assert spilled

            # Cleanup
            manager.close()


class TestPhysicalSortWithSpill:
    """Tests for PhysicalSort with spill-enabled external merge sort."""

    def test_sort_with_spill_enabled(self):
        """Test that sort works with spill enabled (small data, no actual spill)."""
        import pandas as pd

        df = pd.DataFrame(
            {"a": [3, 1, 4, 1, 5, 9, 2, 6], "b": [30, 10, 40, 10, 50, 90, 20, 60]}
        )

        # Sort using lazy pandas
        result = df.select().sort("a").collect(use_physical_planner=True)

        # Verify sorted
        assert list(result["a"]) == [1, 1, 2, 3, 4, 5, 6, 9]
        assert list(result["b"]) == [10, 10, 20, 30, 40, 50, 60, 90]

    def test_sort_descending_with_spill(self):
        """Test descending sort with spill enabled."""
        import pandas as pd

        df = pd.DataFrame({"a": [3, 1, 4, 1, 5], "b": [30, 10, 40, 10, 50]})

        result = (
            df.select().sort("a", descending=True).collect(use_physical_planner=True)
        )

        # Verify sorted descending
        assert list(result["a"]) == [5, 4, 3, 1, 1]


class TestPhysicalHashJoinWithSpill:
    """Tests for PhysicalHashJoin with spill-enabled Grace hash join."""

    def test_join_with_spill_enabled(self):
        """Test that join works with spill enabled (small data, no actual spill)."""
        import pandas as pd

        left = pd.DataFrame({"key": [1, 2, 3], "left_val": [10, 20, 30]})
        right = pd.DataFrame({"key": [2, 3, 4], "right_val": [200, 300, 400]})

        # Inner join
        result = (
            left.select()
            .join(right.select(), on="key", how="inner")
            .collect(use_physical_planner=True)
        )

        # Verify join result
        assert len(result) == 2  # Keys 2 and 3 match
        assert set(result["key"]) == {2, 3}

    def test_left_join_with_spill(self):
        """Test left join with spill enabled."""
        import pandas as pd

        left = pd.DataFrame({"key": [1, 2, 3], "left_val": [10, 20, 30]})
        right = pd.DataFrame({"key": [2, 3, 4], "right_val": [200, 300, 400]})

        result = (
            left.select()
            .join(right.select(), on="key", how="left")
            .collect(use_physical_planner=True)
        )

        # Verify left join result - all left rows preserved
        assert len(result) == 3
        assert set(result["key"]) == {1, 2, 3}

    def test_grace_join_is_size_triggered(self, monkeypatch):
        """Grace hash join engages by size, not merely by spill being enabled.

        With spill enabled and a normal budget, a join that fits stays on the
        fast in-memory pd.merge path; only when the materialized inputs exceed
        the operator budget does it partition/spill.
        """
        import warnings

        import numpy as np

        import pandas as pd
        from pandas.lazy.backends.spill import SpillConfig
        from pandas.lazy.physical import PhysicalHashJoin

        left = pd.DataFrame(
            {"key": np.arange(1_000_000) % 5000, "lv": np.arange(1_000_000)}
        )
        right = pd.DataFrame({"key": np.arange(5000), "rv": np.arange(5000) * 10})

        calls = {"grace": 0}
        original = PhysicalHashJoin._execute_grace_hash_join

        def spy(self, context, left_arrays=None, right_arrays=None):
            calls["grace"] += 1
            return original(self, context, left_arrays, right_arrays)

        monkeypatch.setattr(PhysicalHashJoin, "_execute_grace_hash_join", spy)

        def run(budget_mb):
            calls["grace"] = 0
            config = SpillConfig(enabled=True, operator_budget_mb=budget_mb)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = (
                    left.select()
                    .join(right.select(), on="key", how="inner")
                    .collect(use_physical_planner=True, spill_config=config)
                )
            return len(result), calls["grace"]

        fits_rows, fits_grace = run(512)  # comfortably fits the budget
        spill_rows, spill_grace = run(1)  # 1 MB budget < ~16 MB inputs

        assert fits_grace == 0  # in-memory pd.merge, no spill
        assert spill_grace == 1  # size exceeded -> grace
        assert fits_rows == spill_rows  # identical result either way


class TestForcedSpillingBehavior:
    """
    Tests that actually exercise spilling by setting very low memory thresholds.

    These tests verify that the spill infrastructure works correctly end-to-end:
    - Data is correctly spilled to disk
    - Data is correctly reloaded from disk
    - Final results are correct after spill/reload cycle
    """

    def test_external_sort_forces_multiple_runs(self):
        """Test external sort with tiny run_size_mb to force multiple runs."""
        from pandas.lazy.backends.spill import (
            ExternalSorter,
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                # Use extremely small run size (1 byte) to force each batch to spill
                sorter = ExternalSorter(
                    manager, run_size_mb=0.000001, name="force_sort"
                )

                # Add several batches - each should become a separate run
                batch1 = {
                    "key": np.array([30, 10, 50]),
                    "value": np.array([300, 100, 500]),
                }
                batch2 = {
                    "key": np.array([20, 40, 60]),
                    "value": np.array([200, 400, 600]),
                }
                batch3 = {
                    "key": np.array([15, 25, 35]),
                    "value": np.array([150, 250, 350]),
                }

                sorter.add_batch(batch1, sort_keys=["key"])
                sorter.add_batch(batch2, sort_keys=["key"])
                sorter.add_batch(batch3, sort_keys=["key"])

                # Check that runs were created
                runs = manager.get_sorted_runs("force_sort")
                assert len(runs) >= 2, (
                    f"Expected multiple runs, got {len(runs)}. "
                    f"Spilling should have been forced."
                )

                # Finish sort and verify result
                result = sorter.finish(sort_keys=["key"])

                # Convert to list for comparison
                key_arr = result["key"]
                key_list = (
                    key_arr.to_pylist()
                    if hasattr(key_arr, "to_pylist")
                    else list(key_arr)
                )
                value_arr = result["value"]
                value_list = (
                    value_arr.to_pylist()
                    if hasattr(value_arr, "to_pylist")
                    else list(value_arr)
                )

                # Verify sorted order
                assert key_list == [10, 15, 20, 25, 30, 35, 40, 50, 60]
                assert value_list == [100, 150, 200, 250, 300, 350, 400, 500, 600]

    def test_external_sort_spill_files_exist_during_sort(self):
        """Verify that spill files are actually created on disk during external sort."""
        from pandas.lazy.backends.spill import (
            ExternalSorter,
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                sorter = ExternalSorter(manager, run_size_mb=0.000001, name="file_test")

                # Add batches to trigger spilling
                for i in range(5):
                    batch = {
                        "key": np.array([i * 10, i * 10 + 1, i * 10 + 2]),
                        "value": np.array([i, i + 1, i + 2]),
                    }
                    sorter.add_batch(batch, sort_keys=["key"])

                # Check that spill files exist on disk
                spill_dir = manager._file_manager.spill_dir
                arrow_files = list(spill_dir.glob("*.arrow"))
                assert len(arrow_files) >= 2, (
                    f"Expected spill files on disk, found {len(arrow_files)}"
                )

                # Files should have non-zero size
                for f in arrow_files:
                    assert f.stat().st_size > 0, f"Spill file {f} is empty"

                # Finish and verify correctness
                result = sorter.finish(sort_keys=["key"])
                assert len(result["key"]) == 15  # 5 batches * 3 rows each

    def test_grace_hash_join_forces_partitioning(self):
        """Test Grace hash join with small num_partitions."""
        from pandas.lazy.backends.spill import (
            GraceHashJoiner,
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                # Use small number of partitions
                joiner = GraceHashJoiner(manager, num_partitions=4, name="force_join")

                # Create larger dataset to ensure partitions are non-empty
                left = {
                    "key": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "left_val": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                }
                right = {
                    "key": np.array([2, 4, 6, 8, 10, 12]),
                    "right_val": np.array([200, 400, 600, 800, 1000, 1200]),
                }

                # Partition both sides - this should create spill files
                left_files = joiner.partition_left(left, ["key"])
                right_files = joiner.partition_right(right, ["key"])

                # Verify partitions were created
                assert len(left_files) > 0, "Left side should have partitions"
                assert len(right_files) > 0, "Right side should have partitions"

                # Check partition files exist
                for sf in left_files + right_files:
                    assert sf.exists(), f"Partition file {sf.path} should exist"
                    assert sf.partition_id is not None, "Partition ID should be set"

                # Perform join and verify results
                result = joiner.join(how="inner")

                assert "key" in result
                assert "left_val" in result
                assert "right_val" in result

                # Inner join should have 5 matching keys: 2, 4, 6, 8, 10
                assert len(result["key"]) == 5
                assert set(result["key"].tolist()) == {2, 4, 6, 8, 10}

    def test_spill_reload_data_integrity(self):
        """Test that data survives spill-reload cycle without corruption."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpillConfig(enabled=True, spill_dir=tmpdir)

            with SpillManager(config) as manager:
                # Create data with various dtypes
                original = {
                    "int_col": np.array([1, 2, 3, 4, 5], dtype=np.int64),
                    "float_col": np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
                    "str_col": np.array(["a", "b", "c", "d", "e"]),
                }

                # Register and spill
                manager.register("integrity_test", original)
                spill_file = manager.spill("integrity_test")

                assert spill_file is not None
                assert spill_file.exists()

                # Reload
                reloaded = manager.reload("integrity_test")

                assert reloaded is not None

                # Verify data integrity for each column
                for col, orig_data in original.items():
                    reload_data = reloaded[col]

                    # Convert Arrow to numpy if needed
                    if hasattr(reload_data, "to_numpy"):
                        reload_data = reload_data.to_numpy(zero_copy_only=False)
                    elif hasattr(reload_data, "to_pylist"):
                        reload_data = np.array(reload_data.to_pylist())

                    if orig_data.dtype.kind in ("U", "O"):  # String types
                        orig_list = list(orig_data)
                        reload_list = list(reload_data)
                        assert orig_list == reload_list, (
                            f"Column {col} mismatch: {orig_list} vs {reload_list}"
                        )
                    else:
                        tm.assert_numpy_array_equal(
                            orig_data,
                            reload_data,
                        )

    def test_memory_pressure_triggers_spill(self):
        """Test that memory pressure automatically triggers spilling."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set extremely low threshold (1 KB) to guarantee spilling
            config = SpillConfig(
                enabled=True,
                threshold_mb=0.001,
                spill_dir=tmpdir,  # 1 KB
            )

            with SpillManager(config) as manager:
                # Register several arrays that exceed threshold
                for i in range(3):
                    arrays = {
                        "data": np.zeros(10000, dtype=np.float64)  # 80 KB each
                    }
                    manager.register(f"large_data_{i}", arrays)

                # Check memory pressure - should trigger spills
                assert manager.should_spill(), "Should detect memory pressure"

                # Let it spill
                spilled_files = manager.check_memory_pressure()

                assert len(spilled_files) > 0, (
                    "Should have spilled at least one file under memory pressure"
                )

                # Verify stats
                stats = manager.stats
                assert stats["spill_count"] > 0, "Should have recorded spills"
                assert stats["num_spill_files"] > 0, "Should have spill files"

    def test_operator_budget_triggers_spill_check(self):
        """Test that operator budget threshold is correctly checked."""
        from pandas.lazy.backends.spill import (
            SpillConfig,
            SpillManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set low operator budget
            config = SpillConfig(
                enabled=True,
                threshold_mb=1000,  # High global threshold
                operator_budget_mb=0.001,  # 1 KB operator budget
                spill_dir=tmpdir,
            )

            with SpillManager(config) as manager:
                # Register data exceeding operator budget
                large_arrays = {"data": np.zeros(10000, dtype=np.float64)}  # 80 KB
                manager.register("big_operator", large_arrays)

                # Should trigger on operator budget
                assert manager.should_spill_operator("big_operator"), (
                    "Should detect that operator exceeds its budget"
                )

                # Global threshold should NOT be exceeded
                assert not manager.should_spill(), (
                    "Global threshold should not be exceeded"
                )


class TestCollectSpillConfigAPI:
    """Tests for spill_config parameter in LazyDataFrame.collect()."""

    def test_spill_config_requires_physical_planner(self):
        """Test that spill_config raises error without use_physical_planner=True."""
        import pytest

        import pandas as pd
        from pandas.lazy.backends.spill import SpillConfig

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select()

        config = SpillConfig(enabled=True)

        with pytest.raises(
            ValueError, match="spill_config requires use_physical_planner=True"
        ):
            ldf.collect(spill_config=config)

    def test_spill_config_with_physical_planner(self):
        """Test that spill_config works with use_physical_planner=True."""
        import pandas as pd
        from pandas.lazy.backends.spill import SpillConfig

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select()

        config = SpillConfig(enabled=True, threshold_mb=2048)

        # Should work without error
        result = ldf.collect(use_physical_planner=True, spill_config=config)

        tm.assert_frame_equal(result, df)

    def test_spill_config_with_streaming(self):
        """Test that spill_config works with streaming=True."""
        import pandas as pd
        from pandas.lazy.backends.spill import SpillConfig

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select()

        config = SpillConfig(enabled=True, threshold_mb=2048)

        # Should work without error
        batches = list(
            ldf.collect(
                streaming=True,
                use_physical_planner=True,
                spill_config=config,
            )
        )

        # Combine batches
        result = pd.concat(batches, ignore_index=True)
        tm.assert_frame_equal(result, df)

    def test_spill_config_none_is_default(self):
        """Test that spill_config=None (default) works correctly."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ldf = df.select()

        # Should work without error (default behavior)
        result = ldf.collect(use_physical_planner=True)

        tm.assert_frame_equal(result, df)
