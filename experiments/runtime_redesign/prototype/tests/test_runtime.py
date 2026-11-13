"""Semantic, ownership and representation conformance for M0-M3."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdruntime import NA, Runtime, boolean, int64
from pdruntime.maps import Affine, NumpyMaps

STRATEGIES = ("eager", "compact", "tree")
BACKENDS = ("numpy", "arrow")


def plain(array):
    return [None if value is NA else value for value in array.to_list()]


@pytest.fixture(params=STRATEGIES)
def runtime(request):
    return Runtime(strategy=request.param, metrics=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_opening_example(runtime, backend):
    frame = runtime.frame(
        {
            "price": runtime.array([10, 20, 30], dtype="int64", backend=backend),
            "name": runtime.array(["a", "b", "c"], dtype="utf8", backend="arrow"),
        },
        index=["x", "y", "z"],
    )
    selected = frame.take([1, 1, 2])
    short = selected.slice(slice(0, 2))
    assert plain(short.column("price").add(1)) == [21, 21]
    short.set_at(row=0, column="price", value=99)
    assert plain(short.column("price")) == [99, 20]
    assert plain(selected.column("price")) == [20, 20, 30]
    assert plain(frame.column("price")) == [10, 20, 30]
    assert list(short.index) == ["y", "y"]
    pd.testing.assert_frame_equal(short.compact().to_pandas(), short.to_pandas())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strategy", STRATEGIES)
@settings(max_examples=50, deadline=None)
@given(st.lists(st.one_of(st.none(), st.integers(-1000, 1000)), max_size=30), st.data())
def test_random_selection_mutation(strategy, backend, values, data):
    runtime = Runtime(strategy=strategy)
    original = runtime.array(values, dtype="int64", backend=backend)
    result = original
    oracle = pd.Series(values, dtype="Int64")
    for _ in range(5):
        if len(oracle):
            positions = data.draw(
                st.lists(st.integers(-len(oracle), len(oracle) - 1), max_size=25)
            )
        else:
            positions = []
        result = result.take(positions)
        oracle = oracle.iloc[positions]
    expected = [None if value is pd.NA else int(value) for value in oracle]
    assert plain(result) == expected
    if len(result):
        result.set_at(0, 17)
        expected[0] = 17
        assert plain(result) == expected
    assert plain(original) == values


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "slice_value",
    [
        slice(None, None, -1),
        slice(None, None, -2),
        slice(1, 4),
        slice(-100, 100, 3),
        slice(2, 2),
        slice(4, 0, -2),
    ],
)
def test_slices(runtime, backend, slice_value):
    values = [1, None, 3, 4, 5]
    array = runtime.array(values, dtype="int64", backend=backend)
    result = array.slice(slice_value).slice(slice(None, None, -1))
    assert plain(result) == values[slice_value][::-1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_selection_and_empty_source(runtime, backend):
    array = runtime.array([1, None, 3], dtype="int64", backend=backend)
    row_map = runtime.maps.positions(
        3, [2, 999, 1], [True, False, True], negative=False
    )
    result = array._select_map(row_map).take([2, 1, 0, 1])
    assert plain(result) == [None, None, 3, None]
    empty = runtime.array([], dtype="int64", backend=backend)
    absent = runtime.maps.positions(0, [999, -200], [False, False], negative=False)
    assert plain(empty._select_map(absent)) == [None, None]


@pytest.mark.parametrize("backend", BACKENDS)
def test_integer_kernels_and_null_reductions(runtime, backend):
    array = runtime.array([2**63 - 1, None, 1], dtype=int64(), backend=backend)
    assert plain(array.add(1)) == [-(2**63), None, 2]
    assert array.sum() == -(2**63)
    assert array.sum(skipna=False) is NA
    assert array.sum(min_count=3) is NA
    assert plain(array.eq(array)) == [True, None, True]
    assert plain(array.lt(2)) == [False, None, True]
    assert array.eq(array).dtype == boolean()
    for values in ([], [None, None]):
        empty = runtime.array(values, dtype="int64", backend=backend)
        assert empty.sum() == 0
        assert empty.sum(min_count=1) is NA


def test_mixed_backend_and_alignment(runtime):
    left = runtime.frame(
        {"a": runtime.array([1, 2], dtype="int64", backend="numpy")}, index=["b", "a"]
    )
    right = runtime.frame(
        {"a": runtime.array([10, 20], dtype="int64", backend="arrow")}, index=["c", "a"]
    )
    actual = left.column("a").add(right.column("a"))
    expected = left.to_pandas()["a"] + right.to_pandas()["a"]
    pd.testing.assert_series_equal(actual.to_pandas(), expected)
    assert actual.storage_info()["backend"] == "numpy"
    assert actual.dtype == int64()


def test_duplicate_axis_and_axis_ownership(runtime):
    labels = np.array([1, 1, 2])
    array = runtime.array([10, 20, 30], dtype="int64", backend="numpy")
    frame = runtime.frame({"a": array}, index=labels)
    labels[0] = 99
    exposed = frame.index
    exported_labels = exposed.values
    exported_labels.flags.writeable = True
    exported_labels[0] = 77
    assert list(frame.index) == [1, 1, 2]
    assert plain(frame.column("a").add(frame.column("a"))) == [20, 40, 60]
    with pytest.raises(ValueError, match="unique"):
        frame.column("a").add(frame.take([2, 1, 0]).column("a"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_numpy_ingress_snapshot_and_indexer_ownership(runtime, backend):
    values = np.array([1, 2, 3])
    valid = np.array([True, False, True])
    array = runtime.array(values, validity=valid, dtype="int64", backend=backend)
    indices = np.array([2, 0])
    child = array.take(indices)
    values[:] = 99
    valid[:] = True
    indices[:] = 1
    assert plain(array) == [1, None, 3]
    assert plain(child) == [3, 1]


@pytest.mark.parametrize(
    "dtype,arrow_type,raw",
    [
        ("int64", pa.int64(), np.array([10, 20, 30])),
        ("boolean", pa.bool_(), np.array([0b00000101], dtype="uint8")),
    ],
)
def test_arrow_external_buffer_snapshot(runtime, dtype, arrow_type, raw):
    raw = raw.copy()
    valid = np.array([0b00000101], dtype="uint8")
    source = pa.Array.from_buffers(
        arrow_type, 3, [pa.py_buffer(valid), pa.py_buffer(raw)]
    )
    array = runtime.array(source, dtype=dtype, backend="arrow")
    raw[:] = 0
    valid[:] = 0
    assert plain(array) == ([10, None, 30] if dtype == "int64" else [True, None, True])


def test_arrow_string_buffer_snapshot_and_chunks(runtime):
    offsets = np.array([0, 1, 3, 6], dtype="int32")
    payload = bytearray(b"abbccc")
    source = pa.Array.from_buffers(
        pa.string(), 3, [None, pa.py_buffer(offsets), pa.py_buffer(payload)]
    )
    chunks = pa.chunked_array([source.slice(0, 1), source.slice(1)])
    array = runtime.array(chunks, dtype="utf8", backend="arrow")
    payload[:] = b"xxxxxx"
    offsets[1] = 0
    assert plain(array.take([2, 0, 1])) == ["ccc", "a", "bb"]
    assert plain(array.eq("bb")) == [False, True, False]
    child = array.take([1, 1, 0])
    child.set_at(0, None)
    assert plain(child) == [None, "bb", "a"]
    assert plain(array) == ["a", "bb", "ccc"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_exports_are_detached(runtime, backend):
    array = runtime.array([1, None, 3], dtype="int64", backend=backend)
    values, valid = array.to_numpy()
    values[:] = 99
    valid[:] = True
    assert plain(array) == [1, None, 3]
    exported = array.to_arrow().chunk(0)
    # IPC exports may be immutable; if writable they must still be independent.
    if exported.buffers()[1].is_mutable:
        np.frombuffer(exported.buffers()[1], dtype="int64")[:] = 77
    assert plain(array) == [1, None, 3]
    frame = runtime.frame({"a": array})
    external = frame.to_pandas()
    external.iloc[0, 0] = 999
    assert plain(frame.column("a")) == [1, None, 3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutation_isolation_and_transaction(runtime, backend):
    array = runtime.array([1, 2, 3], dtype="int64", backend=backend)
    frame = runtime.frame({"a": array, "b": array})
    child = frame.take([1, 1, 2])
    series = child.column("a")
    series.set_at(0, 100)
    assert plain(child.column("a")) == [2, 2, 3]
    frame.set_values(rows=[0, 2], column="a", values=[10, None])
    assert plain(array) == [1, 2, 3]
    assert plain(frame.column("b")) == [1, 2, 3]
    assert plain(frame.column("a")) == [10, 2, None]
    for rows, values, error in [
        ([0, 0], [3, 4], ValueError),
        ([0, 9], [3, 4], IndexError),
        ([0], [3, 4], ValueError),
        ([0], ["bad"], TypeError),
    ]:
        with pytest.raises(error):
            frame.set_values(rows=rows, column="a", values=values)
        assert plain(frame.column("a")) == [10, 2, None]
    frame.set_values(rows=[0, 0], column="a", values=7)
    assert plain(frame.column("a")) == [7, 2, None]
    replacement = runtime.array([True, None, False], dtype="boolean", backend=backend)
    frame.replace_column("a", replacement)
    replacement.set_at(0, False)
    assert plain(frame.column("a")) == [True, None, False]


def test_shared_maps_and_single_leaf_gather():
    for strategy in ("compact", "tree"):
        runtime = Runtime(strategy=strategy, metrics=True)
        frame = runtime.frame(
            {
                name: runtime.array(np.arange(10), dtype="int64", backend="numpy")
                for name in ("a", "b", "c")
            }
        )
        child = frame.take([8, 2, 4, 2]).take([3, 0, 1])
        expected = 1 if strategy == "compact" else 0
        assert runtime.metrics.count("map_composition") == expected
        assert runtime.metrics.count("payload_gather") == 0
        child.materialize()
        assert runtime.metrics.count("map_composition") == 1
        assert runtime.metrics.count("payload_gather") == 3
        child.materialize()
        assert runtime.metrics.count("payload_gather") == 3


def test_deep_tree_scalar_and_concurrent_materialization():
    runtime = Runtime(strategy="tree", metrics=True)
    array = runtime.array([1, 2, 3], dtype="int64", backend="numpy")
    for _ in range(1200):
        array = array.take([2, 1, 0])
    assert array.scalar_at(0) == 1
    assert runtime.metrics.count("payload_gather") == 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: array.sum(), range(12)))
    assert results == [6] * 12
    assert runtime.metrics.count("payload_gather") == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_source_release_empty_and_compaction(backend):
    runtime = Runtime(strategy="tree")
    array = runtime.array(np.arange(1000), dtype="int64", backend=backend)
    source = weakref.ref(array.cell)
    empty = array.take([])
    child = array.take([1, 5])
    del array
    gc.collect()
    assert source() is not None
    child.materialize()
    gc.collect()
    assert source() is None
    assert plain(empty) == []
    assert child.memory_report()["map_bytes"] == 0
    parent = runtime.array(np.arange(10000), dtype="int64", backend=backend)
    small = parent.slice(slice(0, 2))
    detached = small.compact()
    assert plain(detached) == [0, 1]
    assert detached.memory_report()["known_retained_payload_bytes"] < 1000


@pytest.mark.parametrize("backend", BACKENDS)
def test_filter_constants_and_dtype_rules(runtime, backend):
    array = runtime.array([1, 2, 3], dtype="int64", backend=backend)
    assert plain(array.filter([True, None, False])) == [1]
    full = runtime.full(3, 7, dtype="int64", backend=backend)
    row_map = runtime.maps.positions(3, [2, 999], [True, False], negative=False)
    assert plain(full._select_map(row_map)) == [7, None]
    assert hash(array.dtype) == hash(array.take([0]).to_backend("arrow").dtype)
    with pytest.raises(TypeError):
        bool(NA)
    with pytest.raises(TypeError):
        array.add(True)
    with pytest.raises(TypeError):
        array.add(2**63)
    with pytest.raises(TypeError):
        runtime.array([True, 1], dtype="int64", backend=backend)
    with pytest.raises(TypeError):
        runtime.array([1.0], dtype="int64", backend=backend)
    with pytest.raises(TypeError):
        array.to_backend("jax")
    with pytest.raises(TypeError):
        array.to_backend("numpy", device="cuda:0")
    with pytest.raises(TypeError):
        array.take([True, 1])
    with pytest.raises(IndexError):
        array.take([2**64 - 1])
    with pytest.raises(ValueError):
        array.slice(slice(None, None, 0))


@settings(max_examples=150, deadline=None)
@given(st.data())
def test_missing_map_composition(data):
    maps = NumpyMaps()
    n = data.draw(st.integers(0, 10))
    first = data.draw(
        st.lists(
            st.one_of(st.none(), st.integers(0, n - 1)) if n else st.none(), max_size=20
        )
    )
    second = data.draw(
        st.lists(
            st.one_of(st.none(), st.integers(0, len(first) - 1))
            if first
            else st.none(),
            max_size=20,
        )
    )
    inner = maps.positions(
        n,
        [0 if x is None else x for x in first],
        [x is not None for x in first],
        negative=False,
    )
    outer = maps.positions(
        len(first),
        [0 if x is None else x for x in second],
        [x is not None for x in second],
        negative=False,
    )
    composed = maps.compose(inner, outer)
    expected = [None if x is None else first[x] for x in second]
    assert [maps.at(composed, i) for i in range(len(expected))] == expected


def test_affine_composition():
    maps = NumpyMaps()
    inner = maps.slice(10, slice(None, None, -2))
    outer = maps.slice(inner.output_length, slice(1, None, 2))
    result = maps.compose(inner, outer)
    assert isinstance(result, Affine)
    assert maps.dense(result)[0].tolist() == list(range(10))[::-2][1::2]
