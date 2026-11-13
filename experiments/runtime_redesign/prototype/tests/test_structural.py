"""M4 structural operations: sort, reindex, limited join and concatenation.

pandas is the oracle for values, dtypes and labels. Every case runs under all
three strategies; the representation assertions check what each one builds.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdruntime import NA, Runtime
from pdruntime.values import Physical, Segments, Select

STRATEGIES = ("eager", "compact", "tree")
BACKENDS = ("numpy", "arrow")
STRING = pd.StringDtype()


def plain(array):
    return [None if value is NA else value for value in array.to_list()]


@pytest.fixture(params=STRATEGIES)
def runtime(request):
    return Runtime(strategy=request.param, metrics=True)


def make_frame(runtime, data, *, backend="numpy", index=None):
    """A frame from plain lists: str-valued columns are UTF-8, others Int64."""
    columns = {}
    for name, values in data.items():
        if any(isinstance(value, str) for value in values):
            columns[name] = runtime.array(values, dtype="utf8", backend="arrow")
        else:
            columns[name] = runtime.array(values, dtype="int64", backend=backend)
    return runtime.frame(columns, index=index)


def oracle(data, index=None):
    columns = {
        name: pd.array(
            values,
            dtype=STRING if any(isinstance(v, str) for v in values) else "Int64",
        )
        for name, values in data.items()
    }
    return pd.DataFrame(columns, index=index)


# Sorting


@pytest.mark.parametrize("backend", BACKENDS)
def test_sort_values_is_stable_with_missing_last(runtime, backend):
    data = {"k": [3, None, 1, 3, None, 1], "s": ["a", "b", "c", "d", "e", "f"]}
    index = ["u", "v", "w", "x", "y", "z"]
    frame = make_frame(runtime, data, backend=backend, index=index)
    result = frame.sort_values("k")
    expected = oracle(data, index).sort_values("k", kind="stable")
    pd.testing.assert_frame_equal(result.to_pandas(), expected)
    assert plain(frame.column("k")) == data["k"]  # the parent is unchanged
    assert runtime.metrics.count("key_values") == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strategy", STRATEGIES)
@settings(max_examples=40, deadline=None)
@given(st.lists(st.one_of(st.none(), st.integers(-5, 5)), max_size=25))
def test_sort_values_matches_pandas(strategy, backend, keys):
    runtime = Runtime(strategy=strategy)
    data = {"k": keys, "v": list(range(len(keys)))}
    result = make_frame(runtime, data, backend=backend).sort_values("k")
    expected = oracle(data).sort_values("k", kind="stable")
    pd.testing.assert_frame_equal(result.to_pandas(), expected)


def test_sort_values_shares_one_permutation(runtime):
    frame = make_frame(runtime, {"k": [2, 1, 3], "a": [1, 2, 3], "s": ["x", "y", "z"]})
    runtime.metrics.clear()
    result = frame.sort_values("k")
    if runtime.execution.strategy == "eager":
        assert runtime.metrics.count("payload_gather") == 3
        return
    maps = {id(cell.state.row_map) for cell in result.cells}
    assert len(maps) == 1
    assert runtime.metrics.count("payload_gather") == 0


def test_sort_values_rejects_unsupported_keys(runtime):
    frame = make_frame(runtime, {"k": [1, 2], "s": ["a", "b"]})
    with pytest.raises(TypeError, match="Int64"):
        frame.sort_values("s")
    with pytest.raises(KeyError):
        frame.sort_values("missing")


# Reindexing


@pytest.mark.parametrize("backend", BACKENDS)
def test_reindex_introduces_missing_rows(runtime, backend):
    data = {"a": [1, None, 3], "s": ["x", "y", None]}
    frame = make_frame(runtime, data, backend=backend, index=["p", "q", "r"])
    flags = runtime.array([True, False, None], dtype="boolean", backend=backend)
    frame = runtime.frame(
        {**{n: frame.column(n).array for n in frame.columns}, "b": flags},
        index=["p", "q", "r"],
    )
    target = ["r", "zz", "p", "r", "q"]
    result = frame.reindex(target)
    expected = oracle(data, ["p", "q", "r"])
    expected["b"] = pd.array([True, False, None], dtype="boolean")
    pd.testing.assert_frame_equal(result.to_pandas(), expected.reindex(target))
    assert result.column("a").scalar_at(1) is NA
    assert list(frame.index) == ["p", "q", "r"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_reindex_then_labeled_arithmetic(runtime, backend):
    """W04: reindex with missing labels, then align, add and export."""
    frame = make_frame(runtime, {"a": [1, 2, 3]}, backend=backend, index=[10, 20, 30])
    other = make_frame(runtime, {"a": [100, 200]}, backend=backend, index=[30, 40])
    left = frame.reindex([30, 40, 50]).column("a")
    result = left.add(other.column("a"))
    expected = (
        oracle({"a": [1, 2, 3]}, [10, 20, 30])
        .reindex([30, 40, 50])["a"]
        .add(oracle({"a": [100, 200]}, [30, 40])["a"])
    )
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_reindex_requires_unique_source(runtime):
    frame = make_frame(runtime, {"a": [1, 2]}, index=["x", "x"])
    with pytest.raises(ValueError, match="duplicate"):
        frame.reindex(["x"])
    unique = make_frame(runtime, {"a": [1, 2]}, index=["x", "y"])
    assert len(unique.reindex([]).to_pandas()) == 0


# Join


LEFT = {"k": [3, 1, 4, 1, 5], "v": [10, 20, 30, 40, 50], "s": ["a", "b", "c", "d", "e"]}
RIGHT = {"k": [5, 1, 9, 3], "v": [100, None, 300, 400], "t": ["w", "x", None, "z"]}


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_join_matches_pandas_merge(runtime, backend, how):
    left = make_frame(runtime, LEFT, backend=backend, index=list("abcde"))
    right = make_frame(runtime, RIGHT, backend=backend)
    result = left.join(right, on="k", how=how, suffixes=("_l", "_r"))
    expected = oracle(LEFT, list("abcde")).merge(
        oracle(RIGHT), on="k", how=how, suffixes=("_l", "_r")
    )
    pd.testing.assert_frame_equal(result.to_pandas(), expected)
    assert result.columns == ("k", "v_l", "s", "v_r", "t")


def test_join_keeps_independent_side_maps(runtime):
    left = make_frame(runtime, LEFT)
    right = make_frame(runtime, RIGHT)
    runtime.metrics.clear()
    joined = left.join(right, on="k", how="left")
    assert runtime.metrics.count("key_match") == 1
    if runtime.execution.strategy == "eager":
        assert all(isinstance(cell.state, Physical) for cell in joined.cells)
    else:
        assert runtime.metrics.count("payload_gather") == 0
        # The left join keeps left cells as they are; right cells share one map.
        assert joined.cells[:3] == left.cells
        right_maps = {id(cell.state.row_map) for cell in joined.cells[3:]}
        assert len(right_maps) == 1
    # Selecting after the join composes each side's map separately.
    short = joined.take([3, 0])
    expected = oracle(LEFT).merge(oracle(RIGHT), on="k", how="left").take([3, 0])
    pd.testing.assert_frame_equal(short.to_pandas(), expected)
    # A replaced column is in output order and inherits neither side's map.
    short.replace_column(
        "t", runtime.array(["new", None], dtype="utf8", backend="arrow")
    )
    assert plain(short.column("t")) == ["new", None]
    assert plain(joined.column("t")) == ["z", "x", None, "x", "w"]


def test_join_with_no_matches_and_empty_sides(runtime):
    left = make_frame(runtime, {"k": [1, 2], "v": [5, 6]})
    right = make_frame(runtime, {"k": [7], "w": [8]})
    assert len(left.join(right, on="k")) == 0
    assert plain(left.join(right, on="k", how="left").column("w")) == [None, None]
    empty = make_frame(runtime, {"k": [], "w": []})
    assert plain(left.join(empty, on="k", how="left").column("w")) == [None, None]


def test_join_rejects_unsupported_cases(runtime):
    left = make_frame(runtime, LEFT)
    with pytest.raises(ValueError, match="unique right keys"):
        left.join(left, on="k")
    with pytest.raises(ValueError, match="how"):
        left.join(make_frame(runtime, RIGHT), on="k", how="outer")
    with pytest.raises(ValueError, match="duplicate column names"):
        left.join(make_frame(runtime, RIGHT), on="k", suffixes=("", ""))
    with pytest.raises(TypeError, match="Int64"):
        left.join(make_frame(runtime, RIGHT), on="s")
    nulls = make_frame(runtime, {"k": [1, None]})
    with pytest.raises(ValueError, match="missing"):
        nulls.join(make_frame(runtime, RIGHT), on="k")
    with pytest.raises(TypeError, match="same runtime"):
        left.join(make_frame(Runtime(), RIGHT), on="k")


# Concatenation


@pytest.mark.parametrize("backend", BACKENDS)
def test_concat_matches_pandas(runtime, backend):
    pieces = [
        {"a": [1, None], "s": ["x", "y"]},
        {"a": [], "s": []},
        {"a": [3, 4, 5], "s": [None, "z", "w"]},
    ]
    indexes = [[0, 1], [], [0, 1, 2]]
    frames = [
        make_frame(runtime, piece, backend=backend, index=index)
        if piece["a"]
        else runtime.frame(
            {
                "a": runtime.array([], dtype="int64", backend=backend),
                "s": runtime.array([], dtype="utf8", backend="arrow"),
            },
            index=[],
        )
        for piece, index in zip(pieces, indexes, strict=True)
    ]
    result = runtime.concat(frames)
    described = result.explain_representation()["a"]
    if runtime.execution.strategy == "eager":
        assert described["leaf"] == "Physical"
    else:
        assert described["leaf"] == "Segments" and described["segments"] == 2
    expected = pd.concat(
        [
            oracle(piece, index).astype({"a": "Int64", "s": STRING})
            for piece, index in zip(pieces, indexes, strict=True)
        ]
    )
    pd.testing.assert_frame_equal(result.to_pandas(), expected)
    assert [result.column("a").scalar_at(i) for i in range(5)] == [1, NA, 3, 4, 5]


@pytest.mark.parametrize("backend", BACKENDS)
def test_take_across_segments(runtime, backend):
    """W10: a gather crossing segments, out of order, with duplicates."""
    pieces = [[1, 2, 3], [None, 5], [6, 7, 8, 9]]
    frames = [make_frame(runtime, {"a": piece}, backend=backend) for piece in pieces]
    combined = runtime.concat(frames)
    flat = [value for piece in pieces for value in piece]
    positions = [8, 0, 3, 3, -1, 4, 1]
    assert plain(combined.take(positions).column("a")) == [flat[p] for p in positions]
    assert plain(combined.slice(slice(2, 7)).column("a")) == flat[2:7]
    assert plain(combined.slice(slice(None, None, -2)).column("a")) == flat[::-2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_aware_read_through_segments(runtime, backend):
    left = make_frame(runtime, {"k": [4, 9, 1, 2]}, backend=backend)
    parts = [
        make_frame(runtime, {"k": [1, 2], "w": [10, None]}, backend=backend),
        make_frame(runtime, {"k": [3, 4], "w": [30, 40]}, backend=backend),
    ]
    right = runtime.concat(parts)
    joined = left.join(right, on="k", how="left")
    assert plain(joined.column("w")) == [40, None, 10, None]


def test_nested_concat_flattens_only_in_compact(runtime):
    pieces = [make_frame(runtime, {"a": [k, k + 1]}) for k in range(0, 8, 2)]
    nested = runtime.concat([runtime.concat(pieces[:2]), runtime.concat(pieces[2:])])
    described = nested.explain_representation()["a"]
    expected = {"eager": None, "compact": 4, "tree": 2}[runtime.execution.strategy]
    assert described.get("segments") == expected
    assert plain(nested.column("a")) == list(range(8))


@pytest.mark.parametrize("backend", BACKENDS)
def test_concat_snapshot_and_mutation_isolation(runtime, backend):
    first = make_frame(runtime, {"a": [1, 2]}, backend=backend)
    second = make_frame(runtime, {"a": [3]}, backend=backend)
    combined = runtime.concat([first, second])
    combined.set_at(row=2, column="a", value=30)
    first.set_at(row=0, column="a", value=10)
    assert plain(combined.column("a")) == [1, 2, 30]
    assert plain(first.column("a")) == [10, 2]
    assert plain(second.column("a")) == [3]
    report = combined.memory_report()
    assert report["value_cells"] == 1  # the write detached the column


def test_concat_rejects_mismatched_frames(runtime):
    ints = make_frame(runtime, {"a": [1]})
    with pytest.raises(ValueError, match="at least one"):
        runtime.concat([])
    with pytest.raises(ValueError, match="identical ordered columns"):
        runtime.concat([ints, make_frame(runtime, {"b": [1]})])
    with pytest.raises(TypeError, match="dtypes"):
        runtime.concat([ints, make_frame(runtime, {"a": ["x"]})])
    with pytest.raises(TypeError, match="destination backend"):
        runtime.concat([ints, make_frame(runtime, {"a": [1]}, backend="arrow")])
    with pytest.raises(TypeError, match="label type"):
        runtime.concat([ints, make_frame(runtime, {"a": [2]}, index=["x"])])
    with pytest.raises(TypeError, match="this runtime"):
        runtime.concat([ints, make_frame(Runtime(), {"a": [1]})])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strategy", STRATEGIES)
@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.lists(st.one_of(st.none(), st.integers(-9, 9)), max_size=6), max_size=5
    ),
    st.data(),
)
def test_random_concat_selection(strategy, backend, pieces, data):
    runtime = Runtime(strategy=strategy)
    frames = [make_frame(runtime, {"a": piece}, backend=backend) for piece in pieces]
    if not frames:
        return
    flat = [value for piece in pieces for value in piece]
    combined = runtime.concat(frames)
    # Concatenate again, then select twice: exercises nesting and composition.
    combined = runtime.concat([combined, frames[0]])
    flat = flat + pieces[0]
    for _ in range(2):
        positions = data.draw(
            st.lists(st.integers(0, len(flat) - 1), max_size=12)
            if flat
            else st.just([])
        )
        combined = combined.take(positions)
        flat = [flat[p] for p in positions]
    assert plain(combined.column("a")) == flat
    for position, value in enumerate(flat):
        assert combined.column("a").scalar_at(position) == (
            NA if value is None else value
        )


def test_segments_descriptor_records_offsets():
    runtime = Runtime(strategy="tree")
    frames = [
        runtime.frame({"a": runtime.array([1, 2], dtype="int64", backend="numpy")})
    ]
    frames.append(
        runtime.frame({"a": runtime.array([3], dtype="int64", backend="numpy")})
    )
    combined = runtime.concat(frames)
    state = combined.cells[0].state
    assert isinstance(state, Segments)
    assert list(state.offsets) == [0, 2, 3]
    assert combined.memory_report()["map_bytes"] == state.offsets.nbytes
    selected = combined.take([2, 0])
    assert isinstance(selected.cells[0].state, Select)
    assert np.array_equal(selected.column("a").to_numpy()[0], [3, 1])


# Review regressions


def test_long_tree_append_chain_reads_without_recursion():
    runtime = Runtime(strategy="tree")
    piece = runtime.frame({"a": runtime.array([1], dtype="int64", backend="numpy")})
    result = piece
    for _ in range(1500):
        result = runtime.concat([result, piece])
    assert result.column("a").take([0, 1500, 7]).sum() == 3
    assert result.column("a").sum() == 1501


def test_columns_share_segment_offsets_and_routing(runtime):
    pieces = [
        make_frame(runtime, {"a": [1, 2, 3], "b": [4, 5, 6], "s": ["x", "y", "z"]}),
        make_frame(runtime, {"a": [7, 8], "b": [9, 10], "s": ["u", "v"]}),
    ]
    combined = runtime.concat(pieces)
    if runtime.execution.strategy == "eager":
        return
    offsets = {id(cell.state.offsets) for cell in combined.cells}
    assert len(offsets) == 1
    assert (
        combined.memory_report()["map_bytes"] == combined.cells[0].state.offsets.nbytes
    )
    selected = combined.take([4, 0, 3])
    runtime.metrics.clear()
    selected.materialize()
    assert runtime.metrics.count("segment_routing") == 1
    assert plain(selected.column("s")) == ["v", "x", "u"]


def test_concat_rejects_promoting_integer_axes(runtime):
    unsigned = make_frame(
        runtime, {"a": [1, 2]}, index=np.array([1, 2], dtype=np.uint64)
    )
    signed = make_frame(runtime, {"a": [3, 4]}, index=[-1, 3])
    with pytest.raises(TypeError, match="label type"):
        runtime.concat([unsigned, signed])


def test_axis_length_is_checked_before_label_type(runtime):
    values = runtime.array([1, 2, 3], dtype="int64", backend="numpy")
    with pytest.raises(ValueError, match="length mismatch"):
        runtime.frame({"a": values}, index=[1.5, 2.5])


@pytest.mark.parametrize("backend", BACKENDS)
def test_key_values_are_read_without_copying(backend):
    runtime = Runtime()
    keys = runtime.array([3, 1, 2], dtype="int64", backend=backend)
    storage, adapter = keys._physical()
    values, valid = adapter.key_values(storage)
    assert valid is None and not values.flags.writeable
    source = storage.data if backend == "numpy" else storage.data.chunk(0).to_numpy()
    assert np.shares_memory(values, source)


def test_segments_compare_by_identity():
    runtime = Runtime(strategy="tree")
    frames = [make_frame(runtime, {"a": [k]}) for k in range(2)]
    first, second = (runtime.concat(frames).cells[0].state for _ in range(2))
    assert first != second and first == first
    assert hash(first) != hash(second)
