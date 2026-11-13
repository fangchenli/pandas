import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("strategy", ["eager", "compact", "tree"])
@pytest.mark.parametrize("backend", ["numpy", "arrow"])
def test_scalar_operands_broadcast_without_constant_arrays(strategy, backend):
    runtime = Runtime(strategy=strategy, metrics=True)
    array = runtime.array([1, None, 3], dtype="int64", backend=backend)
    runtime.metrics.clear()
    assert array.add(1).to_list() == [2, NA, 4]
    assert array.lt(2).to_list() == [True, NA, False]
    assert runtime.metrics.count("materialization") == 0
    assert array.add(NA).to_list() == [NA, NA, NA]
    assert array.eq(NA).dtype == array.eq(1).dtype
    strings = runtime.array(["a", None], dtype="utf8", backend="arrow")
    assert strings.eq("a").to_list() == [True, NA]
    assert strings.eq(NA).to_list() == [NA, NA]


@pytest.mark.parametrize("backend", ["numpy", "arrow"])
def test_filter_is_not_an_export(backend):
    runtime = Runtime(metrics=True)
    array = runtime.array([1, 2, 3], dtype="int64", backend=backend)
    predicate = array.lt(3)
    runtime.metrics.clear()
    assert array.filter(predicate).to_list() == [1, 2]
    assert runtime.metrics.count("export_copy") == 0


@pytest.mark.parametrize("strategy", ["eager", "compact", "tree"])
def test_replace_column_aligns_series_labels(strategy):
    runtime = Runtime(strategy=strategy)
    values = runtime.array([1, 2, 3], dtype="int64", backend="numpy")
    frame = runtime.frame({"v": values, "w": values}, index=["a", "b", "c"])
    oracle = frame.to_pandas()

    reversed_series = frame.take([2, 1, 0]).column("v")
    frame.replace_column("v", reversed_series)
    oracle["v"] = oracle["v"].iloc[[2, 1, 0]]
    assert frame.column("v").to_list() == [1, 2, 3]

    partial = frame.take([2, 0]).column("w")  # labels c, a
    frame.replace_column("w", partial)
    oracle["w"] = oracle["w"].iloc[[2, 0]]
    assert frame.column("w").to_list() == [1, NA, 3]
    pd.testing.assert_frame_equal(frame.to_pandas(), oracle)

    frame.replace_column("v", runtime.array([7, 8, 9], dtype="int64", backend="arrow"))
    assert frame.column("v").to_list() == [7, 8, 9]  # Arrays stay positional

    duplicated = runtime.frame({"v": values}, index=["a", "a", "b"]).column("v")
    with pytest.raises(ValueError, match="duplicate"):
        frame.replace_column("v", duplicated)


@pytest.mark.parametrize("strategy", ["eager", "compact", "tree"])
def test_string_pandas_export_wraps_arrow(strategy):
    runtime = Runtime(strategy=strategy)
    array = runtime.array(["a", None, "ccc"], dtype="utf8", backend="arrow")
    frame = runtime.frame({"s": array}, index=[10, 20, 30])
    actual = frame.take([2, 1, 0]).to_pandas()["s"]
    expected = pd.Series(["ccc", None, "a"], index=[30, 20, 10], dtype=pd.StringDtype())
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    empty = runtime.frame({"s": array}).take([]).to_pandas()
    assert empty["s"].dtype == pd.StringDtype()


def test_missing_singleton_survives_copy_and_pickle():
    import copy
    import pickle

    assert copy.copy(NA) is NA
    assert copy.deepcopy([NA])[0] is NA
    assert pickle.loads(pickle.dumps(NA)) is NA
    assert type(NA)() is NA


def test_logical_types_have_named_validated_parameters():
    from pdruntime import Boolean, LogicalType, SignedInteger, Utf8, boolean, int64

    assert int64() == SignedInteger(bits=64) == SignedInteger()
    assert int64() != boolean() and hash(int64()) != hash(boolean())
    assert Boolean() == boolean() and Utf8() != Boolean()
    assert str(int64()) == "int64" and repr(int64()) == "SignedInteger(bits=64)"
    assert int64().key == ("signed_integer", (("bits", 64),), "NA")
    assert boolean().key == ("boolean", (), "NA")
    runtime = Runtime()
    array = runtime.array([1], dtype=SignedInteger(bits=64), backend="numpy")
    assert array.dtype is int64()  # normalized to the canonical instance
    with pytest.raises(TypeError, match="width"):
        SignedInteger(bits=32)  # declared types must be fully supported
    with pytest.raises(TypeError):
        SignedInteger(64)  # parameters are keyword-only
    with pytest.raises(TypeError, match="abstract"):
        LogicalType()
    with pytest.raises(AttributeError):
        int64().bits = 8  # frozen


@pytest.mark.parametrize("backend", ["numpy", "arrow"])
def test_core_rejects_pandas_missing_value(backend):
    runtime = Runtime()
    array = runtime.array([1, None, 3], dtype="int64", backend=backend)
    assert array.to_list() == [1, NA, 3]  # None remains a missing input
    with pytest.raises(TypeError):
        runtime.array([1, pd.NA], dtype="int64", backend=backend)
    with pytest.raises(TypeError):
        runtime.full(2, pd.NA, dtype="int64", backend=backend)
    with pytest.raises(TypeError):
        array.add(pd.NA)
    with pytest.raises(TypeError):
        array.set_at(0, pd.NA)
    with pytest.raises(TypeError, match="neither a scalar nor a sequence"):
        array.set_values([0, 1], pd.NA)
    with pytest.raises(TypeError):
        array.set_values([0, 1], [5, pd.NA])
    assert array.to_list() == [1, NA, 3]  # failed writes changed nothing
    # Export still maps the runtime's NA to pd.NA at the pandas boundary.
    frame = runtime.frame({"a": array})
    assert frame.to_pandas()["a"].iloc[1] is pd.NA


def test_type_layer_does_not_import_pandas():
    # The package still imports pandas for axes; the type modules must not.
    import ast
    from pathlib import Path

    import pdruntime

    for name in ("types.py", "type_rules.py"):
        tree = ast.parse((Path(pdruntime.__file__).parent / name).read_text())
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "pandas" not in imported, name
