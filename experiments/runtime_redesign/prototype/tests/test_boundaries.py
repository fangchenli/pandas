import numpy as np
import pyarrow as pa
import pytest

from pdruntime import NA, Runtime
from pdruntime.values import Physical


@pytest.mark.parametrize("strategy", ["eager", "compact", "tree"])
@pytest.mark.parametrize("backend", ["numpy", "arrow"])
def test_empty_filter_assignment_and_export(strategy, backend):
    runtime = Runtime(strategy=strategy)
    array = runtime.array([], dtype="int64", backend=backend)
    assert array.filter([]).to_list() == []
    array.set_values([], [])
    assert len(array.to_arrow()) == 0
    assert array.add(1).to_list() == []
    assert array.to_numpy()[0].dtype == np.dtype("int64")
    full = runtime.full(2, NA, dtype="boolean", backend=backend)
    assert full.eq(False).to_list() == [NA, NA]
    assert full.compact().to_list() == [NA, NA]


def test_materialization_failure_can_retry(monkeypatch):
    runtime = Runtime(strategy="tree")
    array = runtime.array([1, 2, 3], dtype="int64", backend="numpy").take([2, 0])
    state = array.cell.state
    adapter = runtime.backend("numpy")
    select = adapter.select

    def fail(storage, row_map):
        raise MemoryError("injected allocation failure")

    monkeypatch.setattr(adapter, "select", fail)
    with pytest.raises(MemoryError, match="injected"):
        array.materialize()
    assert array.cell.state is state
    assert array.scalar_at(0) == 3
    monkeypatch.setattr(adapter, "select", select)
    assert array.to_list() == [3, 1]
    assert isinstance(array.cell.state, Physical)


def test_source_materialized_between_descendant_reads():
    runtime = Runtime(strategy="tree")
    source = runtime.array([10, None, 30, 40], dtype="int64", backend="arrow")
    middle = source.take([3, 1, 0])
    child = middle.take([2, 1, 0])
    middle.materialize()
    assert child.to_list() == [10, NA, 40]


def test_runtime_and_type_boundaries():
    runtime = Runtime()
    other = Runtime()
    array = runtime.array([1], dtype="int64", backend="numpy")
    foreign = other.array([1], dtype="int64", backend="numpy")
    with pytest.raises(TypeError, match="same runtime"):
        array.add(foreign)
    with pytest.raises(TypeError):
        runtime.frame({"a": foreign})
    with pytest.raises(TypeError):
        runtime.full(2, "a", dtype="utf8", backend="numpy")
    with pytest.raises(TypeError):
        runtime.array(pa.array([True]), dtype="int64", backend="arrow")
    with pytest.raises(TypeError):
        array.sum(skipna=1)
    with pytest.raises(ValueError):
        array.sum(min_count=-1)
    with pytest.raises(TypeError):
        array.add(runtime.array([True], dtype="boolean", backend="numpy"))
    frame = runtime.frame({"a": array})
    with pytest.raises(ValueError):
        frame.project(["a", "a"])
    with pytest.raises(KeyError):
        frame.column("missing")
    with pytest.raises(TypeError):
        frame.filter(frame.column("a"))
    assert runtime.frame({}, index=["a", "b"]).to_pandas().shape == (2, 0)


@pytest.mark.parametrize("strategy", ["eager", "compact", "tree"])
def test_string_null_exports_and_snapshots(strategy):
    runtime = Runtime(strategy=strategy)
    array = runtime.array(["a", None, "c"], dtype="utf8", backend="arrow")
    selected = array.take([1, 0, 0])
    values, valid = selected.to_numpy()
    assert values.tolist() == ["", "a", "a"]
    assert valid.tolist() == [False, True, True]
    assert selected.eq("a").to_list() == [NA, True, True]
    values[:] = "new"
    assert selected.to_list() == [NA, "a", "a"]
    array.set_at(0, "changed")
    assert selected.to_list() == [NA, "a", "a"]
