"""
Spill Manager for lazy pandas execution.

This module provides disk spilling infrastructure to handle memory pressure
during physical plan execution. When memory usage exceeds configured thresholds,
intermediate results are spilled to disk using Arrow IPC format for efficient
zero-copy reload via memory mapping.

Design Philosophy
-----------------
Following Polars and DuckDB patterns:
- Spill at operator level (partitions, sorted runs), not data structure level
- Use Arrow IPC format for zero-copy memory mapping on reload
- Partition data by hash for joins to enable out-of-core processing
- Use external merge sort for large sorts

Key Components
--------------
SpillConfig
    Configuration for spill behavior (thresholds, directory, format).

SpillManager
    Core manager that handles spilling and reloading data.

MemoryTracker
    Tracks current memory usage and triggers spills when needed.

ExternalSort
    External merge sort implementation using sorted runs.

GraceHashPartitioner
    Partitions data by hash key for out-of-core joins.

Integration Points
------------------
1. ExecutionContext: Add spill_manager field
2. PhysicalHashAggregate: Partition-based spilling for large groupby
3. PhysicalHashJoin: Grace hash join with partition spilling
4. PhysicalSort: External merge sort for large sorts

Usage
-----
>>> from pandas.lazy.backends.spill import SpillConfig, SpillManager
>>> config = SpillConfig(
...     enabled=True,
...     threshold_mb=2048,  # Spill when >2GB used
...     spill_dir="/tmp/pandas_spill",
... )
>>> manager = SpillManager(config)
>>> # Register arrays that may need spilling
>>> manager.register("join_build", arrays)
>>> # Check memory and spill if needed
>>> manager.check_memory_pressure()
>>> # Reload spilled data
>>> arrays = manager.reload("join_build")
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
import os
from pathlib import Path
import shutil
import tempfile
from typing import (
    TYPE_CHECKING,
    Any,
)
import uuid

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas.lazy.backends.types import ArrayDict


# =============================================================================
# Configuration
# =============================================================================


class SpillFormat(Enum):
    """
    Format for spilled data.

    ARROW_IPC is strongly recommended for best performance.
    """

    ARROW_IPC = "arrow_ipc"  # Zero-copy via mmap (recommended)
    PARQUET = "parquet"  # Compressed, slower reload


class SpillTrigger(Enum):
    """
    When to trigger spilling.

    MEMORY_PRESSURE: Spill when total memory exceeds threshold
    OPERATOR_SIZE: Spill when single operator exceeds its budget
    BOTH: Spill on either condition
    """

    MEMORY_PRESSURE = "memory_pressure"
    OPERATOR_SIZE = "operator_size"
    BOTH = "both"


@dataclass
class SpillConfig:
    """
    Configuration for spill manager.

    Parameters
    ----------
    enabled : bool, default False
        Whether spilling is enabled. When False, operations that would
        spill raise MemoryError instead.

    threshold_mb : int, default 2048
        Memory threshold in MB. When total tracked memory exceeds this,
        spilling is triggered.

    operator_budget_mb : int, default 512
        Per-operator memory budget in MB. Individual operators (join,
        sort, groupby) should spill when they exceed this budget.

    spill_dir : str or Path, optional
        Directory for spill files. If None, uses system temp directory.
        Created if it doesn't exist.

    format : SpillFormat, default ARROW_IPC
        Format for spilled data. ARROW_IPC recommended for zero-copy.

    trigger : SpillTrigger, default BOTH
        When to trigger spilling.

    compression : str or None, default None
        Compression for spilled data. None recommended for Arrow IPC
        (compression adds overhead, mmap benefits lost).

    max_spill_files : int, default 1024
        Maximum number of spill files before cleanup is forced.

    cleanup_on_exit : bool, default True
        Whether to clean up spill files when manager is closed.
    """

    enabled: bool = False
    threshold_mb: int = 2048
    operator_budget_mb: int = 512
    spill_dir: str | Path | None = None
    format: SpillFormat = SpillFormat.ARROW_IPC
    trigger: SpillTrigger = SpillTrigger.BOTH
    compression: str | None = None
    max_spill_files: int = 1024
    cleanup_on_exit: bool = True

    def __post_init__(self):
        if self.spill_dir is not None:
            self.spill_dir = Path(self.spill_dir)

    @property
    def threshold_bytes(self) -> int:
        """Threshold in bytes."""
        return self.threshold_mb * 1024 * 1024

    @property
    def operator_budget_bytes(self) -> int:
        """Per-operator budget in bytes."""
        return self.operator_budget_mb * 1024 * 1024


# =============================================================================
# Memory Tracking
# =============================================================================


def get_array_bytes(arr: Any) -> int:
    """
    Get memory size of an array in bytes.

    Handles NumPy arrays, Arrow arrays, and ChunkedArrays.
    """
    import numpy as np
    import pyarrow as pa

    if isinstance(arr, np.ndarray):
        return arr.nbytes
    elif isinstance(arr, pa.Array):
        return arr.nbytes
    elif isinstance(arr, pa.ChunkedArray):
        return sum(chunk.nbytes for chunk in arr.chunks)
    else:
        # Fallback: estimate 8 bytes per element
        return len(arr) * 8


def get_arrays_bytes(arrays: ArrayDict) -> int:
    """Get total memory size of an ArrayDict."""
    return sum(get_array_bytes(arr) for arr in arrays.values())


def get_process_memory_mb() -> float:
    """
    Get current process memory usage in MB.

    Uses psutil if available, falls back to resource module.
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    try:
        import resource

        # On Unix, get RSS in KB
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB
    except (ImportError, AttributeError):
        pass

    return 0.0  # Unknown


def get_available_memory_mb() -> float:
    """
    Get available system memory in MB.

    Returns 0 if unable to determine.
    """
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        return 0.0


@dataclass
class MemoryTracker:
    """
    Tracks memory usage for spill decisions.

    Tracks both:
    - Registered arrays (explicit tracking)
    - Process memory (via psutil/resource)
    """

    config: SpillConfig

    # Tracked arrays: name -> (arrays, size_bytes)
    _tracked: dict[str, tuple[ArrayDict, int]] = field(default_factory=dict)

    # Total tracked bytes
    _total_tracked_bytes: int = 0

    # Statistics
    _peak_tracked_bytes: int = 0
    _spill_count: int = 0

    def register(self, name: str, arrays: ArrayDict) -> int:
        """
        Register arrays for tracking.

        Returns the size in bytes.
        """
        size = get_arrays_bytes(arrays)
        self._tracked[name] = (arrays, size)
        self._total_tracked_bytes += size
        self._peak_tracked_bytes = max(
            self._peak_tracked_bytes, self._total_tracked_bytes
        )
        return size

    def unregister(self, name: str) -> None:
        """Unregister arrays from tracking."""
        if name in self._tracked:
            _, size = self._tracked.pop(name)
            self._total_tracked_bytes -= size

    def get_tracked(self, name: str) -> ArrayDict | None:
        """Get tracked arrays by name."""
        if name in self._tracked:
            return self._tracked[name][0]
        return None

    def should_spill(self) -> bool:
        """Check if spilling should be triggered."""
        if not self.config.enabled:
            return False

        trigger = self.config.trigger

        # Check total memory
        if trigger in (SpillTrigger.MEMORY_PRESSURE, SpillTrigger.BOTH):
            if self._total_tracked_bytes > self.config.threshold_bytes:
                return True

        return False

    def should_spill_operator(self, size_bytes: int) -> bool:
        """Check if a specific operator should spill based on its size."""
        if not self.config.enabled:
            return False

        trigger = self.config.trigger

        if trigger in (SpillTrigger.OPERATOR_SIZE, SpillTrigger.BOTH):
            if size_bytes > self.config.operator_budget_bytes:
                return True

        return False

    @property
    def tracked_mb(self) -> float:
        """Total tracked memory in MB."""
        return self._total_tracked_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        """Peak tracked memory in MB."""
        return self._peak_tracked_bytes / (1024 * 1024)

    @property
    def stats(self) -> dict[str, Any]:
        """Get tracking statistics."""
        return {
            "tracked_mb": self.tracked_mb,
            "peak_mb": self.peak_mb,
            "num_tracked": len(self._tracked),
            "spill_count": self._spill_count,
            "process_mb": get_process_memory_mb(),
            "available_mb": get_available_memory_mb(),
        }


# =============================================================================
# Spill File Management
# =============================================================================


@dataclass
class SpillFile:
    """
    Represents a spilled file on disk.

    Tracks metadata needed for reload and cleanup.
    """

    path: Path
    name: str  # Logical name (e.g., "join_build_partition_3")
    format: SpillFormat
    num_rows: int
    num_columns: int
    size_bytes: int
    columns: list[str]

    # For partitioned data
    partition_id: int | None = None
    partition_key: Any = None

    def exists(self) -> bool:
        """Check if spill file exists."""
        return self.path.exists()

    def delete(self) -> bool:
        """Delete spill file. Returns True if deleted."""
        if self.path.exists():
            self.path.unlink()
            return True
        return False


class SpillFileManager:
    """
    Manages spill files on disk.

    Handles creation, tracking, and cleanup of spill files.
    """

    def __init__(self, config: SpillConfig):
        self.config = config
        self._files: dict[str, SpillFile] = {}
        self._spill_dir: Path | None = None
        self._session_id = uuid.uuid4().hex[:8]

    @property
    def spill_dir(self) -> Path:
        """Get or create spill directory."""
        if self._spill_dir is None:
            if self.config.spill_dir is not None:
                self._spill_dir = Path(self.config.spill_dir)
            else:
                self._spill_dir = Path(tempfile.gettempdir()) / "pandas_spill"

            # Add session subdirectory to avoid conflicts
            self._spill_dir = self._spill_dir / f"session_{self._session_id}"
            self._spill_dir.mkdir(parents=True, exist_ok=True)

        return self._spill_dir

    def get_spill_path(self, name: str) -> Path:
        """Generate path for a spill file."""
        ext = ".arrow" if self.config.format == SpillFormat.ARROW_IPC else ".parquet"
        return self.spill_dir / f"{name}{ext}"

    def register_file(self, spill_file: SpillFile) -> None:
        """Register a spill file."""
        self._files[spill_file.name] = spill_file

    def get_file(self, name: str) -> SpillFile | None:
        """Get a spill file by name."""
        return self._files.get(name)

    def get_files_by_prefix(self, prefix: str) -> list[SpillFile]:
        """Get all spill files with a given prefix."""
        return [f for name, f in self._files.items() if name.startswith(prefix)]

    def delete_file(self, name: str) -> bool:
        """Delete a spill file."""
        if name in self._files:
            spill_file = self._files.pop(name)
            return spill_file.delete()
        return False

    def delete_by_prefix(self, prefix: str) -> int:
        """Delete all spill files with a given prefix. Returns count."""
        to_delete = [name for name in self._files if name.startswith(prefix)]
        count = 0
        for name in to_delete:
            if self.delete_file(name):
                count += 1
        return count

    def cleanup(self) -> None:
        """Delete all spill files and directory."""
        for name in list(self._files.keys()):
            self.delete_file(name)

        if self._spill_dir is not None and self._spill_dir.exists():
            shutil.rmtree(self._spill_dir, ignore_errors=True)
            self._spill_dir = None

    @property
    def num_files(self) -> int:
        """Number of spill files."""
        return len(self._files)

    @property
    def total_bytes(self) -> int:
        """Total size of all spill files."""
        return sum(f.size_bytes for f in self._files.values())

    def __del__(self):
        """Cleanup on garbage collection."""
        if self.config.cleanup_on_exit:
            self.cleanup()


# =============================================================================
# Core Spill/Reload Operations
# =============================================================================


def spill_arrays_ipc(arrays: ArrayDict, path: Path) -> SpillFile:
    """
    Spill arrays to Arrow IPC file.

    Uses Arrow's IPC format which supports zero-copy reads via mmap.
    """
    import pyarrow as pa

    from pandas.lazy.backends.convert import to_arrow

    # Convert all arrays to Arrow
    arrow_arrays = {}
    for name, arr in arrays.items():
        arrow_arrays[name] = to_arrow(arr)

    # Create table
    table = pa.table(arrow_arrays)

    # Write IPC file (streaming format for mmap support)
    with pa.ipc.new_file(str(path), table.schema) as writer:
        writer.write_table(table)

    return SpillFile(
        path=path,
        name=path.stem,
        format=SpillFormat.ARROW_IPC,
        num_rows=table.num_rows,
        num_columns=table.num_columns,
        size_bytes=path.stat().st_size,
        columns=list(arrays.keys()),
    )


def reload_arrays_ipc(path: Path, memory_map: bool = True) -> ArrayDict:
    """
    Reload arrays from Arrow IPC file.

    Parameters
    ----------
    path : Path
        Path to IPC file.
    memory_map : bool, default True
        Whether to use memory mapping for zero-copy reads.

    Returns
    -------
    ArrayDict
        Reloaded arrays (Arrow format if memory_map=True).
    """
    import pyarrow as pa

    # Read with memory mapping for zero-copy
    source = pa.memory_map(str(path), "r") if memory_map else str(path)
    reader = pa.ipc.open_file(source)
    table = reader.read_all()

    # Return as ArrayDict (Arrow arrays)
    return {name: table.column(name).combine_chunks() for name in table.column_names}


def spill_arrays_parquet(
    arrays: ArrayDict, path: Path, compression: str | None = None
) -> SpillFile:
    """
    Spill arrays to Parquet file.

    Parquet is compressed but slower to reload than IPC.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pandas.lazy.backends.convert import to_arrow

    # Convert all arrays to Arrow
    arrow_arrays = {}
    for name, arr in arrays.items():
        arrow_arrays[name] = to_arrow(arr)

    table = pa.table(arrow_arrays)

    pq.write_table(table, path, compression=compression)

    return SpillFile(
        path=path,
        name=path.stem,
        format=SpillFormat.PARQUET,
        num_rows=table.num_rows,
        num_columns=table.num_columns,
        size_bytes=path.stat().st_size,
        columns=list(arrays.keys()),
    )


def reload_arrays_parquet(path: Path) -> ArrayDict:
    """Reload arrays from Parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {name: table.column(name).combine_chunks() for name in table.column_names}


# =============================================================================
# Spill Manager
# =============================================================================


class SpillManager:
    """
    Core spill manager for lazy pandas execution.

    Coordinates memory tracking, spilling, and reloading of intermediate
    results during physical plan execution.

    Usage
    -----
    >>> config = SpillConfig(enabled=True, threshold_mb=2048)
    >>> manager = SpillManager(config)
    >>>
    >>> # Register data that may need spilling
    >>> manager.register("groupby_input", arrays)
    >>>
    >>> # Check memory and spill if needed
    >>> if manager.should_spill():
    ...     manager.spill("groupby_input")
    >>>
    >>> # Reload when needed
    >>> arrays = manager.reload("groupby_input")
    >>>
    >>> # Cleanup
    >>> manager.close()
    """

    def __init__(self, config: SpillConfig | None = None):
        self.config = config or SpillConfig()
        self._tracker = MemoryTracker(self.config)
        self._file_manager = SpillFileManager(self.config)

        # Track what's in memory vs spilled
        self._in_memory: set[str] = set()
        self._spilled: set[str] = set()

    # -------------------------------------------------------------------------
    # Registration and Tracking
    # -------------------------------------------------------------------------

    def register(self, name: str, arrays: ArrayDict) -> int:
        """
        Register arrays for potential spilling.

        Parameters
        ----------
        name : str
            Unique name for this data (e.g., "join_build", "sort_input").
        arrays : ArrayDict
            The arrays to track.

        Returns
        -------
        int
            Size in bytes.
        """
        size = self._tracker.register(name, arrays)
        self._in_memory.add(name)
        return size

    def unregister(self, name: str) -> None:
        """
        Unregister arrays (when no longer needed).

        Also deletes any spill files for this name.
        """
        self._tracker.unregister(name)
        self._in_memory.discard(name)
        self._spilled.discard(name)
        self._file_manager.delete_file(name)

    def get(self, name: str) -> ArrayDict | None:
        """
        Get arrays by name.

        If spilled, reloads from disk. If in memory, returns directly.
        """
        if name in self._in_memory:
            return self._tracker.get_tracked(name)
        elif name in self._spilled:
            return self.reload(name)
        return None

    # -------------------------------------------------------------------------
    # Spill Operations
    # -------------------------------------------------------------------------

    def should_spill(self) -> bool:
        """Check if spilling should be triggered based on memory pressure."""
        return self._tracker.should_spill()

    def should_spill_operator(self, name: str) -> bool:
        """Check if a specific operator should spill."""
        tracked = self._tracker.get_tracked(name)
        if tracked is None:
            return False
        size = get_arrays_bytes(tracked)
        return self._tracker.should_spill_operator(size)

    def spill(self, name: str) -> SpillFile | None:
        """
        Spill registered arrays to disk.

        Parameters
        ----------
        name : str
            Name of the registered arrays to spill.

        Returns
        -------
        SpillFile or None
            The spill file, or None if not found/already spilled.
        """
        if name not in self._in_memory:
            return None

        arrays = self._tracker.get_tracked(name)
        if arrays is None:
            return None

        # Determine spill path
        path = self._file_manager.get_spill_path(name)

        # Spill based on format
        if self.config.format == SpillFormat.ARROW_IPC:
            spill_file = spill_arrays_ipc(arrays, path)
        else:
            spill_file = spill_arrays_parquet(arrays, path, self.config.compression)

        # Update tracking
        self._file_manager.register_file(spill_file)
        self._tracker.unregister(name)
        self._in_memory.discard(name)
        self._spilled.add(name)
        self._tracker._spill_count += 1

        return spill_file

    def spill_largest(self) -> SpillFile | None:
        """
        Spill the largest in-memory data.

        Useful when memory pressure is detected but specific target unknown.
        """
        if not self._in_memory:
            return None

        # Find largest
        largest_name = None
        largest_size = 0

        for name in self._in_memory:
            tracked = self._tracker.get_tracked(name)
            if tracked is not None:
                size = get_arrays_bytes(tracked)
                if size > largest_size:
                    largest_size = size
                    largest_name = name

        if largest_name is not None:
            return self.spill(largest_name)

        return None

    def check_memory_pressure(self) -> list[SpillFile]:
        """
        Check memory and spill as needed until under threshold.

        Returns list of spilled files.
        """
        spilled = []

        while self.should_spill() and self._in_memory:
            spill_file = self.spill_largest()
            if spill_file is not None:
                spilled.append(spill_file)
            else:
                break  # Nothing left to spill

        return spilled

    # -------------------------------------------------------------------------
    # Reload Operations
    # -------------------------------------------------------------------------

    def reload(self, name: str, keep_file: bool = False) -> ArrayDict | None:
        """
        Reload spilled arrays from disk.

        Parameters
        ----------
        name : str
            Name of the spilled data.
        keep_file : bool, default False
            Whether to keep the spill file after reload.

        Returns
        -------
        ArrayDict or None
            The reloaded arrays, or None if not found.
        """
        if name not in self._spilled:
            # Maybe it's still in memory
            return self._tracker.get_tracked(name)

        spill_file = self._file_manager.get_file(name)
        if spill_file is None or not spill_file.exists():
            return None

        # Reload based on format
        if spill_file.format == SpillFormat.ARROW_IPC:
            arrays = reload_arrays_ipc(spill_file.path, memory_map=True)
        else:
            arrays = reload_arrays_parquet(spill_file.path)

        # Update tracking
        if not keep_file:
            self._file_manager.delete_file(name)
            self._spilled.discard(name)

        # Re-register in memory
        self._tracker.register(name, arrays)
        self._in_memory.add(name)

        return arrays

    def reload_iterator(self, prefix: str) -> Iterator[tuple[str, ArrayDict]]:
        """
        Iterate over spilled files with a given prefix.

        Useful for processing partitioned data one partition at a time.

        Yields
        ------
        (name, arrays) : tuple
            Name and arrays for each spill file.
        """
        files = self._file_manager.get_files_by_prefix(prefix)
        for spill_file in sorted(files, key=lambda f: f.partition_id or 0):
            arrays = self.reload(spill_file.name, keep_file=True)
            if arrays is not None:
                yield spill_file.name, arrays

    # -------------------------------------------------------------------------
    # Partitioned Spilling (for joins)
    # -------------------------------------------------------------------------

    def spill_partitioned(
        self,
        name: str,
        arrays: ArrayDict,
        partition_key: str,
        num_partitions: int,
    ) -> list[SpillFile]:
        """
        Spill arrays partitioned by hash of a key column.

        Used for Grace hash join: partition both sides by join key,
        then process matching partitions together.

        Parameters
        ----------
        name : str
            Base name for partitions (e.g., "join_left").
        arrays : ArrayDict
            The arrays to partition and spill.
        partition_key : str
            Column name to partition by (usually join key).
        num_partitions : int
            Number of partitions to create.

        Returns
        -------
        list of SpillFile
            One spill file per partition.
        """
        import numpy as np
        import pyarrow as pa

        from pandas.lazy.backends.convert import to_arrow

        # Get the partition key array
        key_arr = arrays[partition_key]

        # Convert to Arrow if needed for hashing
        if isinstance(key_arr, np.ndarray):
            key_arr = to_arrow(key_arr)

        # Compute partition assignments using hash
        # Arrow doesn't have a direct hash function, use Python hash
        if isinstance(key_arr, pa.ChunkedArray):
            key_arr = key_arr.combine_chunks()

        # For numeric types, use modulo; for strings, use hash
        key_list = key_arr.to_pylist()
        partition_ids = np.array(
            [hash(k) % num_partitions for k in key_list], dtype=np.int32
        )

        # Partition each array
        spill_files = []
        for part_id in range(num_partitions):
            mask = partition_ids == part_id
            if not mask.any():
                continue

            # Extract partition
            part_arrays = {}
            for col_name, arr in arrays.items():
                if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                    if isinstance(arr, pa.ChunkedArray):
                        arr = arr.combine_chunks()
                    part_arrays[col_name] = arr.filter(pa.array(mask))
                else:
                    part_arrays[col_name] = arr[mask]

            # Spill partition
            part_name = f"{name}_part_{part_id:04d}"
            path = self._file_manager.get_spill_path(part_name)
            spill_file = spill_arrays_ipc(part_arrays, path)
            spill_file.partition_id = part_id
            self._file_manager.register_file(spill_file)
            self._spilled.add(part_name)
            spill_files.append(spill_file)

        return spill_files

    # -------------------------------------------------------------------------
    # Sorted Run Spilling (for external sort)
    # -------------------------------------------------------------------------

    def spill_sorted_run(
        self,
        name: str,
        arrays: ArrayDict,
        run_id: int,
    ) -> SpillFile:
        """
        Spill a sorted run for external merge sort.

        Parameters
        ----------
        name : str
            Base name (e.g., "sort_run").
        arrays : ArrayDict
            Pre-sorted arrays.
        run_id : int
            Run identifier.

        Returns
        -------
        SpillFile
            The spill file for this run.
        """
        run_name = f"{name}_run_{run_id:04d}"
        path = self._file_manager.get_spill_path(run_name)
        spill_file = spill_arrays_ipc(arrays, path)
        spill_file.partition_id = run_id
        self._file_manager.register_file(spill_file)
        self._spilled.add(run_name)
        return spill_file

    def get_sorted_runs(self, name: str) -> list[SpillFile]:
        """Get all sorted runs for external merge sort."""
        return self._file_manager.get_files_by_prefix(f"{name}_run_")

    # -------------------------------------------------------------------------
    # Cleanup and Stats
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close manager and cleanup all spill files."""
        self._file_manager.cleanup()
        self._in_memory.clear()
        self._spilled.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Get spill manager statistics."""
        return {
            **self._tracker.stats,
            "num_spill_files": self._file_manager.num_files,
            "spill_bytes_on_disk": self._file_manager.total_bytes,
            "in_memory_count": len(self._in_memory),
            "spilled_count": len(self._spilled),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"SpillManager("
            f"tracked={stats['tracked_mb']:.1f}MB, "
            f"spilled={stats['spill_count']}, "
            f"files={stats['num_spill_files']})"
        )


# =============================================================================
# External Sort Support
# =============================================================================


class ExternalSorter:
    """
    External merge sort using spill manager.

    For datasets larger than memory, sorts by:
    1. Reading chunks that fit in memory
    2. Sorting each chunk (creating sorted "runs")
    3. Spilling sorted runs to disk
    4. K-way merging runs to produce final sorted output

    Usage
    -----
    >>> sorter = ExternalSorter(spill_manager, run_size_mb=256)
    >>> for batch in batches:
    ...     sorter.add_batch(batch)
    >>> result = sorter.finish(sort_keys=["col1", "col2"])
    """

    def __init__(
        self,
        spill_manager: SpillManager,
        run_size_mb: int = 256,
        name: str = "external_sort",
    ):
        self.spill_manager = spill_manager
        self.run_size_bytes = run_size_mb * 1024 * 1024
        self.name = name

        self._current_batch: ArrayDict = {}
        self._current_size: int = 0
        self._run_count: int = 0
        self._total_rows: int = 0

    def add_batch(self, arrays: ArrayDict, sort_keys: list[str]) -> None:
        """
        Add a batch of data.

        If current accumulated data exceeds run size, sorts and spills.
        """

        batch_size = get_arrays_bytes(arrays)
        batch_rows = len(next(iter(arrays.values())))

        # If adding would exceed run size, flush current
        if (
            self._current_size > 0
            and self._current_size + batch_size > self.run_size_bytes
        ):
            self._flush_run(sort_keys)

        # Accumulate batch
        if not self._current_batch:
            self._current_batch = {k: [v] for k, v in arrays.items()}
        else:
            for k, v in arrays.items():
                self._current_batch[k].append(v)

        self._current_size += batch_size
        self._total_rows += batch_rows

    def _flush_run(self, sort_keys: list[str]) -> None:
        """Sort current batch and spill as a run."""
        import numpy as np

        if not self._current_batch:
            return

        # Concatenate accumulated batches
        combined = {}
        for k, arrays_list in self._current_batch.items():
            if len(arrays_list) == 1:
                combined[k] = arrays_list[0]
            else:
                combined[k] = np.concatenate(arrays_list)

        # Sort
        sorted_arrays = self._sort_arrays(combined, sort_keys)

        # Spill
        self.spill_manager.spill_sorted_run(self.name, sorted_arrays, self._run_count)
        self._run_count += 1

        # Reset
        self._current_batch = {}
        self._current_size = 0

    def _sort_arrays(self, arrays: ArrayDict, sort_keys: list[str]) -> ArrayDict:
        """Sort arrays by keys."""
        import numpy as np

        # Get sort indices using lexsort (sorts by last key first)
        keys_for_sort = [arrays[k] for k in reversed(sort_keys)]
        indices = np.lexsort(keys_for_sort)

        # Apply sort order to all columns
        return {k: v[indices] for k, v in arrays.items()}

    def finish(self, sort_keys: list[str]) -> ArrayDict:
        """
        Finish sorting and return merged result.

        Performs k-way merge of all sorted runs.
        """

        import numpy as np

        # Flush any remaining data
        if self._current_batch:
            self._flush_run(sort_keys)

        # If only one run (or fits in memory), just reload it
        runs = self.spill_manager.get_sorted_runs(self.name)
        if len(runs) == 0:
            return {}
        if len(runs) == 1:
            return self.spill_manager.reload(runs[0].name) or {}

        # K-way merge
        # For simplicity, load all runs and merge
        # (A production implementation would use streaming merge)
        all_arrays = []
        for run in runs:
            arrays = self.spill_manager.reload(run.name)
            if arrays:
                all_arrays.append(arrays)

        if not all_arrays:
            return {}

        # Concatenate all runs
        combined = {}
        for k in all_arrays[0].keys():
            combined[k] = np.concatenate([a[k] for a in all_arrays])

        # Final sort
        return self._sort_arrays(combined, sort_keys)


# =============================================================================
# Grace Hash Join Support
# =============================================================================


class GraceHashJoiner:
    """
    Grace hash join using spill manager.

    For joins larger than memory:
    1. Partition both sides by hash of join key
    2. Spill partitions to disk
    3. Join matching partitions (which now fit in memory)
    4. Concatenate results

    Usage
    -----
    >>> joiner = GraceHashJoiner(spill_manager, num_partitions=64)
    >>> joiner.partition_left(left_arrays, join_keys)
    >>> joiner.partition_right(right_arrays, join_keys)
    >>> result = joiner.join(how="inner")
    """

    def __init__(
        self,
        spill_manager: SpillManager,
        num_partitions: int = 64,
        name: str = "grace_join",
        max_partition_skew_ratio: float = 10.0,
    ):
        self.spill_manager = spill_manager
        self.num_partitions = num_partitions
        self.name = name
        self.max_partition_skew_ratio = max_partition_skew_ratio

        self._left_partitioned = False
        self._right_partitioned = False
        self._join_keys: list[str] = []

        # Statistics for pathological detection
        self._left_partition_sizes: list[int] = []
        self._right_partition_sizes: list[int] = []
        self._total_spill_bytes = 0
        self._empty_partitions = 0
        self._max_partition_size = 0

    def partition_left(
        self, arrays: ArrayDict, join_keys: list[str]
    ) -> list[SpillFile]:
        """Partition and spill left side."""
        self._join_keys = join_keys
        # Use first join key for partitioning
        files = self.spill_manager.spill_partitioned(
            f"{self.name}_left", arrays, join_keys[0], self.num_partitions
        )
        self._left_partitioned = True

        # Track partition sizes for pathological detection
        self._left_partition_sizes = [f.size_bytes for f in files]
        self._total_spill_bytes += sum(self._left_partition_sizes)
        self._update_partition_stats(self._left_partition_sizes)

        return files

    def partition_right(
        self, arrays: ArrayDict, join_keys: list[str]
    ) -> list[SpillFile]:
        """Partition and spill right side."""
        files = self.spill_manager.spill_partitioned(
            f"{self.name}_right", arrays, join_keys[0], self.num_partitions
        )
        self._right_partitioned = True

        # Track partition sizes for pathological detection
        self._right_partition_sizes = [f.size_bytes for f in files]
        self._total_spill_bytes += sum(self._right_partition_sizes)
        self._update_partition_stats(self._right_partition_sizes)

        return files

    def _update_partition_stats(self, sizes: list[int]) -> None:
        """Update statistics from partition sizes."""
        for size in sizes:
            if size == 0:
                self._empty_partitions += 1
            if size > self._max_partition_size:
                self._max_partition_size = size

    def is_pathological(self, operator_budget_bytes: int) -> bool:
        """
        Check if the hash join has become pathological.

        Pathological conditions indicate sort-merge join may be better:
        1. Very skewed partitions (max >> average) - indicates bad hash distribution
        2. Many empty partitions (> 50%) - wasted work
        3. Max partition still exceeds memory budget - will cause recursive spills

        Parameters
        ----------
        operator_budget_bytes : int
            Memory budget for the operator in bytes.

        Returns
        -------
        bool
            True if spill behavior is pathological and fallback recommended.
        """
        if not self._left_partitioned or not self._right_partitioned:
            return False

        # Check 1: Max partition exceeds budget (will cause recursive spills)
        if self._max_partition_size > operator_budget_bytes:
            return True

        # Check 2: Very skewed partitions
        all_sizes = self._left_partition_sizes + self._right_partition_sizes
        non_empty = [s for s in all_sizes if s > 0]
        if non_empty:
            avg_size = sum(non_empty) / len(non_empty)
            skew_threshold = avg_size * self.max_partition_skew_ratio
            if avg_size > 0 and self._max_partition_size > skew_threshold:
                return True

        # Check 3: Too many empty partitions (> 50%)
        total_partitions = 2 * self.num_partitions
        if self._empty_partitions > total_partitions * 0.5:
            return True

        return False

    def get_stats(self) -> dict:
        """Get statistics about the join for debugging/monitoring."""
        all_sizes = self._left_partition_sizes + self._right_partition_sizes
        non_empty = [s for s in all_sizes if s > 0]
        avg_size = sum(non_empty) / len(non_empty) if non_empty else 0

        return {
            "num_partitions": self.num_partitions,
            "total_spill_bytes": self._total_spill_bytes,
            "empty_partitions": self._empty_partitions,
            "max_partition_size": self._max_partition_size,
            "avg_partition_size": avg_size,
            "skew_ratio": self._max_partition_size / avg_size if avg_size > 0 else 0,
        }

    def join(self, how: str = "inner") -> ArrayDict:
        """
        Perform join across all partition pairs.

        Parameters
        ----------
        how : str
            Join type: "inner", "left", "right", "outer".

        Returns
        -------
        ArrayDict
            Joined result.
        """
        import numpy as np

        if not self._left_partitioned or not self._right_partitioned:
            raise RuntimeError("Must partition both sides before joining")

        results = []

        # Process each partition pair
        for part_id in range(self.num_partitions):
            left_name = f"{self.name}_left_part_{part_id:04d}"
            right_name = f"{self.name}_right_part_{part_id:04d}"

            left = self.spill_manager.reload(left_name)
            right = self.spill_manager.reload(right_name)

            if left is None and right is None:
                continue

            # Perform in-memory join for this partition
            if left is not None and right is not None:
                part_result = self._join_partition(left, right, how)
                if part_result:
                    results.append(part_result)
            elif how in ("left", "outer") and left is not None:
                # Left side with nulls for right
                results.append(self._pad_with_nulls(left, "right"))
            elif how in ("right", "outer") and right is not None:
                # Right side with nulls for left
                results.append(self._pad_with_nulls(right, "left"))

        # Concatenate all partition results
        if not results:
            return {}

        combined = {}
        for k in results[0].keys():
            combined[k] = np.concatenate([r[k] for r in results])

        return combined

    def _join_partition(
        self, left: ArrayDict, right: ArrayDict, how: str
    ) -> ArrayDict | None:
        """Join a single partition pair in memory."""
        # This is a simplified implementation
        # Production would use hash join kernel

        from pandas.lazy.backends.convert import arrays_to_dataframe

        # Convert to DataFrames for join
        left_df = arrays_to_dataframe(left)
        right_df = arrays_to_dataframe(right)

        # Perform join
        result_df = left_df.merge(right_df, on=self._join_keys, how=how)

        if len(result_df) == 0:
            return None

        # Convert back to arrays
        return {col: result_df[col].values for col in result_df.columns}

    def _pad_with_nulls(self, arrays: ArrayDict, side: str) -> ArrayDict:
        """Pad arrays with null columns for missing join side."""
        # This would add null columns for the missing side
        # Simplified: just return the arrays we have
        return arrays
