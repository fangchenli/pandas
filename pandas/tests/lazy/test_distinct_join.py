"""
Tests for pandas.lazy distinct and join operations.
"""

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.lazy import col
from pandas.lazy.plan import (
    Distinct,
    Join,
)


class TestDistinctAPI:
    """Tests for the distinct() method API."""

    def test_distinct_plan_node(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        ldf = df.select()
        plan = ldf.distinct()._plan
        assert isinstance(plan, Distinct)
        assert plan.subset is None

    def test_distinct_with_subset_plan_node(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 2]})
        ldf = df.select()
        plan = ldf.distinct("a")._plan
        assert isinstance(plan, Distinct)
        assert plan.subset == ("a",)

    def test_distinct_multiple_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 2], "c": [1, 2, 3]})
        ldf = df.select()
        plan = ldf.distinct("a", "b")._plan
        assert plan.subset == ("a", "b")


class TestDistinctEvaluation:
    """Tests for distinct() evaluation."""

    def test_distinct_all_columns(self):
        df = pd.DataFrame(
            {
                "a": [1, 1, 2, 2, 3],
                "b": [1, 1, 2, 2, 3],
            }
        )
        result = df.select().distinct().collect()
        expected = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1, 2, 3],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_distinct_subset(self):
        df = pd.DataFrame(
            {
                "a": [1, 1, 2, 2, 3],
                "b": [1, 2, 3, 4, 5],
            }
        )
        result = df.select().distinct("a").collect()
        # Keeps first occurrence of each unique 'a'
        expected = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1, 3, 5],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_distinct_multiple_subset_columns(self):
        df = pd.DataFrame(
            {
                "a": [1, 1, 1, 2],
                "b": [1, 1, 2, 1],
                "c": [10, 20, 30, 40],
            }
        )
        result = df.select().distinct("a", "b").collect()
        # Unique combinations of (a, b): (1,1), (1,2), (2,1)
        expected = pd.DataFrame(
            {
                "a": [1, 1, 2],
                "b": [1, 2, 1],
                "c": [10, 30, 40],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_distinct_no_duplicates(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        result = df.select().distinct().collect()
        expected = df.copy()
        tm.assert_frame_equal(result, expected)

    def test_distinct_all_duplicates(self):
        df = pd.DataFrame(
            {
                "a": [1, 1, 1],
                "b": [2, 2, 2],
            }
        )
        result = df.select().distinct().collect()
        expected = pd.DataFrame(
            {
                "a": [1],
                "b": [2],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_distinct_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        result = df.select().distinct().collect()
        assert len(result) == 0
        assert list(result.columns) == ["a"]

    def test_distinct_with_filter(self):
        df = pd.DataFrame(
            {
                "a": [1, 1, 2, 2, 3, 3],
                "b": [1, 2, 1, 2, 1, 2],
            }
        )
        result = df.select().filter(col("a") > 1).distinct().collect()
        expected = (
            pd.DataFrame(
                {
                    "a": [2, 2, 3, 3],
                    "b": [1, 2, 1, 2],
                }
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(result, expected)


class TestJoinAPI:
    """Tests for join() method API."""

    def test_join_plan_node_on(self):
        df1 = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"id": [1, 2], "b": [100, 200]})
        ldf1 = df1.select()
        ldf2 = df2.select()
        plan = ldf1.join(ldf2, on="id")._plan
        assert isinstance(plan, Join)
        assert plan.on == ("id",)
        assert plan.how == "inner"

    def test_join_plan_node_left_right_on(self):
        df1 = pd.DataFrame({"id1": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"id2": [1, 2], "b": [100, 200]})
        ldf1 = df1.select()
        ldf2 = df2.select()
        plan = ldf1.join(ldf2, left_on="id1", right_on="id2")._plan
        assert plan.left_on == ("id1",)
        assert plan.right_on == ("id2",)

    def test_join_requires_on_or_left_right(self):
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [1]})
        with pytest.raises(ValueError, match="Must provide either"):
            df1.select().join(df2.select())

    def test_join_left_right_length_mismatch(self):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"c": [1]})
        with pytest.raises(ValueError, match="same length"):
            df1.select().join(df2.select(), left_on=["a", "b"], right_on=["c"])


class TestJoinEvaluation:
    """Tests for join() evaluation."""

    @pytest.fixture
    def left_df(self):
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "left_val": ["a", "b", "c", "d"],
            }
        )

    @pytest.fixture
    def right_df(self):
        return pd.DataFrame(
            {
                "id": [2, 3, 4, 5],
                "right_val": ["x", "y", "z", "w"],
            }
        )

    def test_inner_join(self, left_df, right_df):
        result = left_df.select().join(right_df.select(), on="id").collect()
        expected = pd.DataFrame(
            {
                "id": [2, 3, 4],
                "left_val": ["b", "c", "d"],
                "right_val": ["x", "y", "z"],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_left_join(self, left_df, right_df):
        result = left_df.select().join(right_df.select(), on="id", how="left").collect()
        expected = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "left_val": ["a", "b", "c", "d"],
                "right_val": [None, "x", "y", "z"],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_right_join(self, left_df, right_df):
        result = (
            left_df.select().join(right_df.select(), on="id", how="right").collect()
        )
        expected = pd.DataFrame(
            {
                "id": [2, 3, 4, 5],
                "left_val": ["b", "c", "d", None],
                "right_val": ["x", "y", "z", "w"],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_outer_join(self, left_df, right_df):
        result = (
            left_df.select().join(right_df.select(), on="id", how="outer").collect()
        )
        # Outer join includes all rows from both sides
        assert len(result) == 5
        assert set(result["id"]) == {1, 2, 3, 4, 5}

    def test_join_different_column_names(self):
        left = pd.DataFrame({"id_left": [1, 2, 3], "val": ["a", "b", "c"]})
        right = pd.DataFrame({"id_right": [2, 3, 4], "data": [100, 200, 300]})
        result = (
            left.select()
            .join(right.select(), left_on="id_left", right_on="id_right")
            .collect()
        )
        expected = pd.DataFrame(
            {
                "id_left": [2, 3],
                "val": ["b", "c"],
                "id_right": [2, 3],
                "data": [100, 200],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_join_multiple_keys(self):
        left = pd.DataFrame(
            {
                "k1": [1, 1, 2, 2],
                "k2": ["a", "b", "a", "b"],
                "val": [10, 20, 30, 40],
            }
        )
        right = pd.DataFrame(
            {
                "k1": [1, 2],
                "k2": ["a", "b"],
                "data": [100, 200],
            }
        )
        result = left.select().join(right.select(), on=["k1", "k2"]).collect()
        expected = pd.DataFrame(
            {
                "k1": [1, 2],
                "k2": ["a", "b"],
                "val": [10, 40],
                "data": [100, 200],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_join_with_suffix(self):
        left = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        right = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        result = (
            left.select()
            .join(right.select(), on="id", suffix=("_left", "_right"))
            .collect()
        )
        assert "value_left" in result.columns
        assert "value_right" in result.columns
        assert list(result["value_left"]) == [10, 20]
        assert list(result["value_right"]) == [100, 200]

    def test_cross_join(self):
        left = pd.DataFrame({"a": [1, 2]})
        right = pd.DataFrame({"b": ["x", "y"]})
        result = left.select().join(right.select(), how="cross").collect()
        assert len(result) == 4
        assert list(result["a"]) == [1, 1, 2, 2]
        assert list(result["b"]) == ["x", "y", "x", "y"]

    def test_join_empty_left(self):
        left = pd.DataFrame(
            {"id": pd.Series([], dtype="int64"), "val": pd.Series([], dtype="str")}
        )
        right = pd.DataFrame({"id": [1, 2], "data": ["a", "b"]})
        result = left.select().join(right.select(), on="id").collect()
        assert len(result) == 0

    def test_join_empty_right(self):
        left = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        right = pd.DataFrame(
            {"id": pd.Series([], dtype="int64"), "data": pd.Series([], dtype="str")}
        )
        result = left.select().join(right.select(), on="id").collect()
        assert len(result) == 0

    def test_join_with_dataframe_auto_convert(self):
        """Test that join accepts a regular DataFrame and auto-converts it."""
        left = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        right = pd.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})
        # Pass DataFrame directly instead of LazyDataFrame
        result = left.select().join(right, on="id").collect()
        expected = pd.DataFrame(
            {
                "id": [2, 3],
                "a": [20, 30],
                "b": [200, 300],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_join_with_dataframe_left_join(self):
        """Test left join with auto-converted DataFrame."""
        left = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        right = pd.DataFrame({"id": [2, 3], "b": [200, 300]})
        result = left.select().join(right, on="id", how="left").collect()
        expected = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "a": [10, 20, 30],
                "b": [None, 200.0, 300.0],
            }
        )
        tm.assert_frame_equal(result, expected)


class TestJoinChaining:
    """Tests for chaining joins with other operations."""

    def test_join_then_filter(self):
        left = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        right = pd.DataFrame({"id": [1, 2, 3], "category": ["A", "B", "A"]})
        result = (
            left.select()
            .join(right.select(), on="id")
            .filter(col("category") == "A")
            .collect()
        )
        expected = pd.DataFrame(
            {
                "id": [1, 3],
                "val": [10, 30],
                "category": ["A", "A"],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_join_then_select(self):
        left = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
        right = pd.DataFrame({"id": [1, 2], "b": [100, 200]})
        result = (
            left.select()
            .join(right.select(), on="id")
            .with_columns((col("a") + col("b")).alias("total"))
            .collect()
        )
        assert "total" in result.columns
        assert list(result["total"]) == [110, 220]

    def test_filter_then_join(self):
        left = pd.DataFrame({"id": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
        right = pd.DataFrame({"id": [1, 2, 3, 4], "data": ["a", "b", "c", "d"]})
        result = (
            left.select()
            .filter(col("val") > 15)
            .join(right.select(), on="id")
            .collect()
        )
        expected = pd.DataFrame(
            {
                "id": [2, 3, 4],
                "val": [20, 30, 40],
                "data": ["b", "c", "d"],
            }
        )
        tm.assert_frame_equal(result, expected)


class TestSemiAntiJoinKernels:
    """Tests for semi and anti join kernel implementations."""

    def test_semi_join_basic(self):
        """Test basic semi join - return left rows that exist in right."""
        import numpy as np

        from pandas.lazy.backends.numpy.join import numpy_semi_join

        left = {"id": np.array([1, 2, 3, 4]), "val": np.array(["a", "b", "c", "d"])}
        right = {"id": np.array([2, 3, 5]), "other": np.array([10, 20, 30])}

        result = numpy_semi_join(left, right, keys=["id"])

        # Only ids 2 and 3 exist in right
        assert list(result["id"]) == [2, 3]
        assert list(result["val"]) == ["b", "c"]
        # Right columns should NOT be in result
        assert "other" not in result

    def test_anti_join_basic(self):
        """Test basic anti join - return left rows that DON'T exist in right."""
        import numpy as np

        from pandas.lazy.backends.numpy.join import numpy_anti_join

        left = {"id": np.array([1, 2, 3, 4]), "val": np.array(["a", "b", "c", "d"])}
        right = {"id": np.array([2, 3, 5]), "other": np.array([10, 20, 30])}

        result = numpy_anti_join(left, right, keys=["id"])

        # Only ids 1 and 4 do NOT exist in right
        assert list(result["id"]) == [1, 4]
        assert list(result["val"]) == ["a", "d"]
        assert "other" not in result

    def test_semi_join_empty_right(self):
        """Semi join with empty right table returns empty result."""
        import numpy as np

        from pandas.lazy.backends.numpy.join import numpy_semi_join

        left = {"id": np.array([1, 2, 3]), "val": np.array(["a", "b", "c"])}
        right = {
            "id": np.array([], dtype=np.int64),
            "other": np.array([], dtype=np.int64),
        }

        result = numpy_semi_join(left, right, keys=["id"])
        assert len(result["id"]) == 0

    def test_anti_join_empty_right(self):
        """Anti join with empty right table returns all left rows."""
        import numpy as np

        from pandas.lazy.backends.numpy.join import numpy_anti_join

        left = {"id": np.array([1, 2, 3]), "val": np.array(["a", "b", "c"])}
        right = {
            "id": np.array([], dtype=np.int64),
            "other": np.array([], dtype=np.int64),
        }

        result = numpy_anti_join(left, right, keys=["id"])
        assert list(result["id"]) == [1, 2, 3]
        assert list(result["val"]) == ["a", "b", "c"]

    def test_semi_join_all_match(self):
        """Semi join where all left rows match returns all left rows."""
        import numpy as np

        from pandas.lazy.backends.numpy.join import numpy_semi_join

        left = {"id": np.array([1, 2]), "val": np.array(["a", "b"])}
        right = {"id": np.array([1, 2, 3]), "other": np.array([10, 20, 30])}

        result = numpy_semi_join(left, right, keys=["id"])
        assert list(result["id"]) == [1, 2]
        assert list(result["val"]) == ["a", "b"]

    def test_anti_join_all_match(self):
        """Anti join where all left rows match returns empty result."""
        import numpy as np

        from pandas.lazy.backends.numpy.join import numpy_anti_join

        left = {"id": np.array([1, 2]), "val": np.array(["a", "b"])}
        right = {"id": np.array([1, 2, 3]), "other": np.array([10, 20, 30])}

        result = numpy_anti_join(left, right, keys=["id"])
        assert len(result["id"]) == 0

    def test_semi_join_composite_keys(self):
        """Test semi join with multiple key columns."""
        import numpy as np

        from pandas.lazy.backends.numpy.join import numpy_semi_join

        left = {
            "k1": np.array([1, 1, 2, 2]),
            "k2": np.array(["a", "b", "a", "b"]),
            "val": np.array([10, 20, 30, 40]),
        }
        right = {
            "k1": np.array([1, 2]),
            "k2": np.array(["a", "b"]),
            "other": np.array([100, 200]),
        }

        result = numpy_semi_join(left, right, keys=["k1", "k2"])
        # Only (1, 'a') and (2, 'b') exist in right
        assert list(result["k1"]) == [1, 2]
        assert list(result["k2"]) == ["a", "b"]
        assert list(result["val"]) == [10, 40]


class TestJoinCardinalityEstimation:
    """Tests for join cardinality estimation in logical plan nodes."""

    def test_dataframe_source_row_count(self):
        """DataFrameSource should return exact row count."""
        from pandas.lazy.plan import DataFrameSource

        df = pd.DataFrame({"a": range(100)})
        source = DataFrameSource(df)
        assert source.estimate_row_count() == 100

    def test_filter_reduces_row_count(self):
        """Filter should estimate reduced row count."""
        df = pd.DataFrame({"a": range(1000)})
        ldf = df.select().filter(col("a") > 500)
        estimate = ldf._plan.estimate_row_count()
        # Should be less than 1000 but not None
        assert estimate is not None
        assert 0 < estimate < 1000

    def test_limit_caps_row_count(self):
        """Limit should cap row count at n."""
        df = pd.DataFrame({"a": range(1000)})
        ldf = df.select().head(10)
        estimate = ldf._plan.estimate_row_count()
        assert estimate == 10

    def test_inner_join_row_count(self):
        """Inner join should estimate min of both sides."""
        left = pd.DataFrame({"id": range(1000), "a": range(1000)})
        right = pd.DataFrame({"id": range(500), "b": range(500)})
        ldf = left.select().join(right.select(), on="id")
        estimate = ldf._plan.estimate_row_count()
        # Inner join estimate should be min(left, right)
        assert estimate == 500

    def test_left_join_row_count(self):
        """Left join should preserve left side row count."""
        left = pd.DataFrame({"id": range(1000), "a": range(1000)})
        right = pd.DataFrame({"id": range(500), "b": range(500)})
        ldf = left.select().join(right.select(), on="id", how="left")
        estimate = ldf._plan.estimate_row_count()
        # Left join should preserve left count
        assert estimate == 1000

    def test_cross_join_row_count(self):
        """Cross join should be product of both sides."""
        left = pd.DataFrame({"a": range(100)})
        right = pd.DataFrame({"b": range(50)})
        ldf = left.select().join(right.select(), how="cross")
        estimate = ldf._plan.estimate_row_count()
        assert estimate == 5000

    def test_row_count_propagates_through_plan(self):
        """Row count should propagate through filter->join chain."""
        left = pd.DataFrame({"id": range(10000), "a": range(10000)})
        right = pd.DataFrame({"id": range(5000), "b": range(5000)})

        # Filter left, then join
        ldf = left.select().filter(col("a") > 5000).join(right.select(), on="id")
        estimate = ldf._plan.estimate_row_count()

        # Filter estimate ~30% of 10000 = 3000
        # Join with right (5000 rows) -> min = 3000
        assert estimate is not None
        assert 0 < estimate < 10000


class TestBuildProbeOptimization:
    """Tests for hash join build/probe side optimization."""

    def test_physical_hash_join_has_row_estimates(self):
        """PhysicalHashJoin should receive row count estimates from planner."""
        from pandas.lazy.optimize import Optimizer
        from pandas.lazy.physical import PhysicalPlanner

        left = pd.DataFrame({"id": range(100), "a": range(100)})
        right = pd.DataFrame({"id": range(1000), "b": range(1000)})
        ldf = left.select().join(right.select(), on="id")

        # Optimize and plan
        optimizer = Optimizer()
        optimized = optimizer.optimize(ldf._plan)
        planner = PhysicalPlanner(preferred_backend="numpy")
        physical = planner.plan(optimized)

        # Check that row estimates are set
        assert physical.left_rows_estimate == 100
        assert physical.right_rows_estimate == 1000

    def test_build_probe_swap_when_left_is_smaller(self):
        """Build/probe should swap to build on smaller side for inner join."""
        import numpy as np

        # Create data where LEFT is SMALLER than right
        # The implementation should swap to put smaller side in right position
        left_data = {"id": np.array([2, 3]), "val": np.array(["b", "c"])}
        right_data = {
            "id": np.array([1, 2, 3, 4, 5]),
            "other": np.array([10, 20, 30, 40, 50]),
        }

        from pandas.lazy.physical import PhysicalHashJoin
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
        )

        # Create schema using LazyDtype.from_pandas_dtype
        schema = Schema(
            {
                "id": LazyDtype.from_pandas_dtype(np.dtype("int64")),
                "val": LazyDtype.from_pandas_dtype(np.dtype("O")),
                "other": LazyDtype.from_pandas_dtype(np.dtype("int64")),
            }
        )

        # Create a mock plan with left SMALLER than right
        join = PhysicalHashJoin(
            left=None,  # Not used for _maybe_swap_for_build_probe
            right=None,
            on=("id",),
            left_on=None,
            right_on=None,
            how="inner",
            suffix=("", "_right"),
            schema=schema,
            left_rows_estimate=2,  # Left is smaller
            right_rows_estimate=5,
        )

        # Should swap for inner join to put smaller side in build (right) position
        swapped_left, swapped_right = join._maybe_swap_for_build_probe(
            left_data, right_data
        )

        # Left (smaller) should now be swapped to right position for building hash table
        assert len(swapped_right["id"]) == 2  # Was left, now right (build side)
        assert len(swapped_left["id"]) == 5  # Was right, now left (probe side)

    def test_build_probe_no_swap_when_right_already_smaller(self):
        """Build/probe should not swap when right is already smaller."""
        import numpy as np

        # Create data where right is already smaller
        left_data = {
            "id": np.array([1, 2, 3, 4, 5]),
            "val": np.array(["a", "b", "c", "d", "e"]),
        }
        right_data = {"id": np.array([2, 3]), "other": np.array([10, 20])}

        from pandas.lazy.physical import PhysicalHashJoin
        from pandas.lazy.types import (
            LazyDtype,
            Schema,
        )

        schema = Schema(
            {
                "id": LazyDtype.from_pandas_dtype(np.dtype("int64")),
                "val": LazyDtype.from_pandas_dtype(np.dtype("O")),
                "other": LazyDtype.from_pandas_dtype(np.dtype("int64")),
            }
        )

        join = PhysicalHashJoin(
            left=None,
            right=None,
            on=("id",),
            left_on=None,
            right_on=None,
            how="inner",
            suffix=("", "_right"),
            schema=schema,
            left_rows_estimate=5,  # Left is larger
            right_rows_estimate=2,  # Right is already smaller - no swap needed
        )

        # Should NOT swap since right is already smaller (optimal for build)
        swapped_left, swapped_right = join._maybe_swap_for_build_probe(
            left_data, right_data
        )

        # No swap - left stays left, right stays right
        assert len(swapped_left["id"]) == 5  # Left unchanged
        assert len(swapped_right["id"]) == 2  # Right unchanged (already build side)
