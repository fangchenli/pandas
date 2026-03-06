"""
Integration tests: pandas code → JIT tracing → Substrait IR → backend execution.

Each test compiles a pandas workflow to a Substrait plan, executes it on
PyArrow's Acero engine or DataFusion, and compares the result against the
equivalent pandas computation.
"""

from __future__ import annotations

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

pa = pytest.importorskip("pyarrow")
pas = pytest.importorskip("pyarrow.substrait")

from pandas.jit.compiler import (
    PandasBackend,
    SubstraitCompiler,
    infer_schema,
)
from pandas.jit.ir import (
    AddColumn,
    Aggregate,
    BinOp,
    ColRef,
    DType,
    Filter,
    Join,
    Limit,
    Literal,
    Project,
    ReadTable,
    Sort,
)
from pandas.jit.jit import (
    Tracer,
    compilable,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_on_acero(
    ir_node,
    tables: dict[str, DataFrame],
) -> DataFrame:
    """Compile IR to Substrait, run on Acero, return pandas DataFrame."""
    compiler = SubstraitCompiler()
    plan = compiler.compile(ir_node)
    plan_bytes = plan.SerializeToString()

    pa_tables = {name: pa.Table.from_pandas(df) for name, df in tables.items()}

    def table_provider(names, _schema):
        return pa_tables[names[0]]

    reader = pas.run_query(plan_bytes, table_provider=table_provider)
    return reader.read_all().to_pandas()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df():
    return DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "region": ["East", "West", "East", "West", "East", "West"],
            "product": ["A", "B", "A", "A", "B", "B"],
            "price": [100.0, 250.0, 150.0, 300.0, 50.0, 175.0],
            "quantity": [10, 5, 8, 3, 20, 7],
        }
    )


@pytest.fixture
def regions_df():
    return DataFrame(
        {
            "region": ["East", "West"],
            "manager": ["Alice", "Bob"],
        }
    )


@pytest.fixture
def nullable_df():
    return DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "score": [10.0, np.nan, 30.0, np.nan, 50.0],
            "label": ["a", "b", None, "d", "e"],
        }
    )


@pytest.fixture
def datetime_df():
    return DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2024-01-15 10:30:00",
                    "2024-03-20 14:45:00",
                    "2023-07-04 08:00:00",
                    "2024-12-25 23:59:00",
                    "2023-06-15 12:00:00",
                ]
            ),
            "value": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture
def string_df():
    return DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "city": ["  New York  ", "London", "Paris", "Tokyo", "Berlin"],
            "score": [90, 85, 95, 70, 88],
        }
    )


# ---------------------------------------------------------------------------
# IR → Acero: individual operations
# ---------------------------------------------------------------------------


class TestAceroReadTable:
    def test_full_scan(self, sales_df):
        schema = infer_schema(sales_df)
        node = ReadTable("sales", schema)
        result = _run_on_acero(node, {"sales": sales_df})
        assert len(result) == 6
        assert list(result.columns) == list(sales_df.columns)


class TestAceroFilter:
    def test_simple_gt(self, sales_df):
        schema = infer_schema(sales_df)
        node = Filter(
            ReadTable("sales", schema),
            BinOp("gt", ColRef("price"), Literal(100.0, DType.FLOAT64)),
        )
        result = _run_on_acero(node, {"sales": sales_df})
        expected = sales_df[sales_df["price"] > 100].reset_index(drop=True)
        assert len(result) == len(expected)
        assert all(result["price"] > 100)

    def test_compound_predicate(self, sales_df):
        schema = infer_schema(sales_df)
        pred = BinOp(
            "and",
            BinOp("gt", ColRef("price"), Literal(100.0, DType.FLOAT64)),
            BinOp("lt", ColRef("price"), Literal(300.0, DType.FLOAT64)),
        )
        node = Filter(ReadTable("sales", schema), pred)
        result = _run_on_acero(node, {"sales": sales_df})
        expected = sales_df[(sales_df["price"] > 100) & (sales_df["price"] < 300)]
        assert len(result) == len(expected)


class TestAceroProject:
    def test_select_columns(self, sales_df):
        schema = infer_schema(sales_df)
        node = Project(ReadTable("sales", schema), ["id", "region", "price"])
        result = _run_on_acero(node, {"sales": sales_df})
        assert list(result.columns) == ["id", "region", "price"]
        assert len(result) == 6


class TestAceroAddColumn:
    def test_computed_column(self, sales_df):
        schema = infer_schema(sales_df)
        base = ReadTable("sales", schema)
        expr = BinOp("mul", ColRef("price"), ColRef("quantity"))
        node = AddColumn(base, "revenue", expr, DType.FLOAT64)
        result = _run_on_acero(node, {"sales": sales_df})
        assert "revenue" in result.columns
        expected_revenue = sales_df["price"] * sales_df["quantity"]
        tm.assert_numpy_array_equal(result["revenue"].values, expected_revenue.values)


class TestAceroSort:
    def test_sort_descending(self, sales_df):
        schema = infer_schema(sales_df)
        node = Sort(ReadTable("sales", schema), [("price", False)])
        result = _run_on_acero(node, {"sales": sales_df})
        assert list(result["price"]) == sorted(sales_df["price"], reverse=True)

    def test_sort_ascending(self, sales_df):
        schema = infer_schema(sales_df)
        node = Sort(ReadTable("sales", schema), [("price", True)])
        result = _run_on_acero(node, {"sales": sales_df})
        assert list(result["price"]) == sorted(sales_df["price"])


class TestAceroLimit:
    def test_take_n(self, sales_df):
        schema = infer_schema(sales_df)
        node = Limit(ReadTable("sales", schema), 3)
        result = _run_on_acero(node, {"sales": sales_df})
        assert len(result) == 3


class TestAceroAggregate:
    def test_sum_by_region(self, sales_df):
        schema = infer_schema(sales_df)
        node = Aggregate(
            ReadTable("sales", schema),
            group_keys=["region"],
            agg_specs=[("total_price", "price", "sum")],
        )
        result = _run_on_acero(node, {"sales": sales_df})
        assert set(result.columns) == {"region", "total_price"}
        assert len(result) == 2

        expected = sales_df.groupby("region")["price"].sum()
        result_sorted = result.sort_values("region").reset_index(drop=True)
        tm.assert_numpy_array_equal(
            result_sorted["total_price"].values,
            np.array([expected["East"], expected["West"]]),
        )

    def test_count(self, sales_df):
        schema = infer_schema(sales_df)
        node = Aggregate(
            ReadTable("sales", schema),
            group_keys=["region"],
            agg_specs=[("cnt", "id", "count")],
        )
        result = _run_on_acero(node, {"sales": sales_df})
        assert set(result.columns) == {"region", "cnt"}
        result_sorted = result.sort_values("region").reset_index(drop=True)
        assert list(result_sorted["cnt"]) == [3, 3]

    def test_multiple_aggs(self, sales_df):
        schema = infer_schema(sales_df)
        node = Aggregate(
            ReadTable("sales", schema),
            group_keys=["region"],
            agg_specs=[
                ("total", "price", "sum"),
                ("avg_price", "price", "avg"),
                ("cnt", "id", "count"),
            ],
        )
        result = _run_on_acero(node, {"sales": sales_df})
        assert set(result.columns) == {"region", "total", "avg_price", "cnt"}


class TestAceroJoin:
    def test_inner_join(self, sales_df, regions_df):
        left_schema = infer_schema(sales_df)
        right_schema = infer_schema(regions_df)
        node = Join(
            ReadTable("sales", left_schema),
            ReadTable("regions", right_schema),
            "region",
            "region",
        )
        result = _run_on_acero(node, {"sales": sales_df, "regions": regions_df})
        assert "manager" in result.columns
        assert len(result) == 6
        # All East rows → Alice, all West rows → Bob
        east_rows = result[result["region"] == "East"]
        assert all(east_rows["manager"] == "Alice")
        west_rows = result[result["region"] == "West"]
        assert all(west_rows["manager"] == "Bob")


# ---------------------------------------------------------------------------
# Chained IR → Acero: multi-step pipelines
# ---------------------------------------------------------------------------


class TestAceroPipeline:
    def test_filter_add_sort_limit(self, sales_df):
        """filter → add column → sort → limit — full chain on Acero."""
        schema = infer_schema(sales_df)
        base = ReadTable("sales", schema)

        pipeline = Limit(
            Sort(
                AddColumn(
                    Filter(
                        base,
                        BinOp("gt", ColRef("price"), Literal(50.0, DType.FLOAT64)),
                    ),
                    "revenue",
                    BinOp("mul", ColRef("price"), ColRef("quantity")),
                    DType.FLOAT64,
                ),
                [("revenue", False)],
            ),
            3,
        )

        result = _run_on_acero(pipeline, {"sales": sales_df})
        assert len(result) == 3
        assert "revenue" in result.columns
        # Should be sorted descending by revenue
        assert list(result["revenue"]) == sorted(result["revenue"], reverse=True)

    def test_join_then_aggregate(self, sales_df, regions_df):
        """Join → filter → aggregate on Acero."""
        left_schema = infer_schema(sales_df)
        right_schema = infer_schema(regions_df)

        joined = Join(
            ReadTable("sales", left_schema),
            ReadTable("regions", right_schema),
            "region",
            "region",
        )
        filtered = Filter(
            joined,
            BinOp("gt", ColRef("price"), Literal(100.0, DType.FLOAT64)),
        )
        agged = Aggregate(
            filtered,
            group_keys=["manager"],
            agg_specs=[("total", "price", "sum"), ("cnt", "id", "count")],
        )

        result = _run_on_acero(agged, {"sales": sales_df, "regions": regions_df})
        assert set(result.columns) == {"manager", "total", "cnt"}
        assert len(result) == 2

    def test_project_then_sort(self, sales_df):
        """Project → sort on Acero."""
        schema = infer_schema(sales_df)
        base = ReadTable("sales", schema)
        pipeline = Sort(
            Project(base, ["id", "price"]),
            [("price", True)],
        )
        result = _run_on_acero(pipeline, {"sales": sales_df})
        assert list(result.columns) == ["id", "price"]
        assert list(result["price"]) == sorted(sales_df["price"])


# ---------------------------------------------------------------------------
# Correctness: PandasBackend vs Acero — same IR, same results
# ---------------------------------------------------------------------------


class TestPandasVsAcero:
    def test_filter_matches(self, sales_df):
        schema = infer_schema(sales_df)
        node = Filter(
            ReadTable("sales", schema),
            BinOp("gte", ColRef("price"), Literal(150.0, DType.FLOAT64)),
        )
        pandas_result = PandasBackend().execute(node, {"sales": sales_df})
        acero_result = _run_on_acero(node, {"sales": sales_df})

        pandas_sorted = pandas_result.sort_values("id").reset_index(drop=True)
        acero_sorted = acero_result.sort_values("id").reset_index(drop=True)
        tm.assert_frame_equal(pandas_sorted, acero_sorted)

    def test_add_column_matches(self, sales_df):
        schema = infer_schema(sales_df)
        base = ReadTable("sales", schema)
        node = AddColumn(
            base,
            "revenue",
            BinOp("mul", ColRef("price"), ColRef("quantity")),
            DType.FLOAT64,
        )
        pandas_result = PandasBackend().execute(node, {"sales": sales_df})
        acero_result = _run_on_acero(node, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.sort_values("id").reset_index(drop=True),
            acero_result.sort_values("id").reset_index(drop=True),
        )

    def test_aggregate_matches(self, sales_df):
        schema = infer_schema(sales_df)
        node = Aggregate(
            ReadTable("sales", schema),
            group_keys=["region"],
            agg_specs=[("total", "price", "sum"), ("cnt", "id", "count")],
        )
        pandas_result = PandasBackend().execute(node, {"sales": sales_df})
        acero_result = _run_on_acero(node, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.sort_values("region").reset_index(drop=True),
            acero_result.sort_values("region").reset_index(drop=True),
        )

    def test_sort_limit_matches(self, sales_df):
        schema = infer_schema(sales_df)
        node = Limit(
            Sort(ReadTable("sales", schema), [("price", False)]),
            3,
        )
        pandas_result = PandasBackend().execute(node, {"sales": sales_df})
        acero_result = _run_on_acero(node, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.reset_index(drop=True),
            acero_result.reset_index(drop=True),
        )

    def test_join_matches(self, sales_df, regions_df):
        left_schema = infer_schema(sales_df)
        right_schema = infer_schema(regions_df)
        node = Join(
            ReadTable("sales", left_schema),
            ReadTable("regions", right_schema),
            "region",
            "region",
        )
        pandas_result = PandasBackend().execute(
            node, {"sales": sales_df, "regions": regions_df}
        )
        acero_result = _run_on_acero(node, {"sales": sales_df, "regions": regions_df})

        tm.assert_frame_equal(
            pandas_result.sort_values("id").reset_index(drop=True),
            acero_result.sort_values("id").reset_index(drop=True),
        )

    def test_full_pipeline_matches(self, sales_df):
        """Complex pipeline: filter → add col → sort → limit."""
        schema = infer_schema(sales_df)
        base = ReadTable("sales", schema)

        pipeline = Limit(
            Sort(
                AddColumn(
                    Filter(
                        base,
                        BinOp("gt", ColRef("price"), Literal(50.0, DType.FLOAT64)),
                    ),
                    "revenue",
                    BinOp("mul", ColRef("price"), ColRef("quantity")),
                    DType.FLOAT64,
                ),
                [("revenue", False)],
            ),
            3,
        )

        pandas_result = PandasBackend().execute(pipeline, {"sales": sales_df})
        acero_result = _run_on_acero(pipeline, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.reset_index(drop=True),
            acero_result.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# JIT decorator → Acero: end-to-end from @compilable to Substrait execution
# ---------------------------------------------------------------------------


class TestJitToAcero:
    def test_jit_plan_runs_on_acero(self, sales_df):
        """Trace with @compilable, extract Substrait plan, run on Acero."""

        @compilable(backend=PandasBackend())
        def pipeline(df):
            filtered = df[df["price"] > 100]
            filtered["revenue"] = filtered["price"] * filtered["quantity"]
            return filtered[["region", "product", "revenue"]].sort_values(
                "revenue", ascending=False
            )

        # Run on PandasBackend to get reference result
        pandas_result = pipeline(sales_df)

        # Now trace and extract the Substrait plan
        ctx = pipeline.trace(sales_df)
        plans = []
        for seg in ctx.segments:
            from pandas.jit.compiler import CompiledSegment

            if isinstance(seg, CompiledSegment):
                compiler = SubstraitCompiler()
                plans.append((seg, compiler.compile(seg.ir_node)))

        # Execute last compiled segment on Acero
        assert len(plans) > 0
        last_seg, last_plan = plans[-1]
        plan_bytes = last_plan.SerializeToString()

        pa_tables = {name: pa.Table.from_pandas(df) for name, df in ctx.tables.items()}

        def provider(names, _schema):
            return pa_tables[names[0]]

        acero_result = (
            pas.run_query(plan_bytes, table_provider=provider).read_all().to_pandas()
        )

        assert set(acero_result.columns) == set(pandas_result.columns)
        assert len(acero_result) == len(pandas_result)

    def test_tracer_plan_runs_on_acero(self, sales_df):
        """Tracer context manager → extract plans → Acero execution."""
        with Tracer(backend=PandasBackend()) as t:
            df = t.input(sales_df, "sales")
            expensive = df[df["price"] > 100]
            result = expensive.sort_values("price", ascending=False)
            t.output(result)

        pandas_result = t.result()

        # Extract Substrait plans from the context
        plans = t.to_substrait_plans()
        assert len(plans) >= 1

        # Execute the last plan on Acero
        plan_bytes = plans[-1].SerializeToString()
        pa_tables = {
            name: pa.Table.from_pandas(df) for name, df in t._ctx.tables.items()
        }

        def provider(names, _schema):
            return pa_tables[names[0]]

        acero_result = (
            pas.run_query(plan_bytes, table_provider=provider).read_all().to_pandas()
        )

        # Both should have the same rows (sorted by price desc)
        assert len(acero_result) == len(pandas_result)
        assert all(acero_result["price"] > 100)


# ---------------------------------------------------------------------------
# Dtype round-trips through Acero
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# User scenarios: @compilable decorator — how users actually use the API
# ---------------------------------------------------------------------------


class TestCompileDecorator:
    """End-to-end tests using @compilable with the default AceroBackend."""

    def test_filter_rows(self, sales_df):
        @compilable
        def get_expensive(df):
            return df[df["price"] > 100]

        result = get_expensive(sales_df)
        expected = sales_df[sales_df["price"] > 100].reset_index(drop=True)
        tm.assert_frame_equal(
            result.sort_values("id").reset_index(drop=True),
            expected.sort_values("id").reset_index(drop=True),
        )

    def test_select_columns(self, sales_df):
        @compilable
        def select_cols(df):
            return df[["id", "region", "price"]]

        result = select_cols(sales_df)
        assert list(result.columns) == ["id", "region", "price"]
        assert len(result) == len(sales_df)

    def test_add_computed_column(self, sales_df):
        @compilable
        def add_revenue(df):
            df["revenue"] = df["price"] * df["quantity"]
            return df

        result = add_revenue(sales_df)
        assert "revenue" in result.columns
        expected = sales_df.copy()
        expected["revenue"] = expected["price"] * expected["quantity"]
        tm.assert_numpy_array_equal(
            result.sort_values("id").reset_index(drop=True)["revenue"].values,
            expected.sort_values("id").reset_index(drop=True)["revenue"].values,
        )

    def test_filter_and_sort(self, sales_df):
        @compilable
        def top_expensive(df):
            expensive = df[df["price"] > 100]
            return expensive.sort_values("price", ascending=False)

        result = top_expensive(sales_df)
        assert all(result["price"] > 100)
        assert list(result["price"]) == sorted(result["price"], reverse=True)

    def test_sort_and_head(self, sales_df):
        @compilable
        def top_3(df):
            return df.sort_values("price", ascending=False).head(3)

        result = top_3(sales_df)
        assert len(result) == 3
        assert list(result["price"]) == sorted(result["price"], reverse=True)

    def test_groupby_sum(self, sales_df):
        @compilable
        def regional_totals(df):
            return df.groupby("region").sum()

        result = regional_totals(sales_df)
        assert "region" in result.columns
        # TracedGroupBy.sum() only sums numeric columns
        result_sorted = result.sort_values("region").reset_index(drop=True)
        expected = (
            sales_df.groupby("region", as_index=False)[["id", "price", "quantity"]]
            .sum()
            .sort_values("region")
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(result_sorted, expected, check_dtype=False)

    def test_groupby_count(self, sales_df):
        @compilable
        def product_counts(df):
            return df.groupby("product").count()

        result = product_counts(sales_df)
        assert "product" in result.columns
        assert len(result) == 2  # products A and B

    def test_groupby_std(self, sales_df):
        @compilable
        def f(df):
            return df.groupby("region").std()

        result = f(sales_df)
        assert "region" in result.columns
        # Std values should be non-negative floats
        result_sorted = result.sort_values("region").reset_index(drop=True)
        assert all(result_sorted["price"] >= 0)
        assert result_sorted["price"].dtype == np.float64

    def test_groupby_var(self, sales_df):
        @compilable
        def f(df):
            return df.groupby("region").var()

        result = f(sales_df)
        assert "region" in result.columns
        # Var values should be non-negative floats
        result_sorted = result.sort_values("region").reset_index(drop=True)
        assert all(result_sorted["price"] >= 0)
        assert result_sorted["price"].dtype == np.float64

    def test_groupby_series_std(self, sales_df):
        @compilable
        def f(df):
            return df.groupby("region")["price"].std()

        result = f(sales_df)
        assert "price" in result.columns
        assert "region" in result.columns

    def test_groupby_first(self, sales_df):
        @compilable
        def f(df):
            return df.sort_values("price").groupby("region").first()

        result = f(sales_df)
        assert "region" in result.columns
        assert len(result) == 2

    def test_groupby_last(self, sales_df):
        @compilable
        def f(df):
            return df.sort_values("price").groupby("region").last()

        result = f(sales_df)
        assert "region" in result.columns
        assert len(result) == 2

    def test_series_std(self, sales_df):
        @compilable
        def f(df):
            s = df["price"].std()
            return df[df["price"] > s]

        result = f(sales_df)
        assert all(result["price"] > sales_df["price"].std())

    def test_series_var(self, sales_df):
        @compilable
        def f(df):
            v = df["price"].var()
            return df[df["price"] > v]

        result = f(sales_df)
        assert all(result["price"] > sales_df["price"].var())

    def test_filter_then_groupby(self, sales_df):
        @compilable
        def expensive_by_region(df):
            expensive = df[df["price"] > 100]
            return expensive.groupby("region").sum()

        result = expensive_by_region(sales_df)
        assert "region" in result.columns
        # All prices in the input should be > 100

    def test_multi_step_pipeline(self, sales_df):
        @compilable
        def pipeline(df):
            df["revenue"] = df["price"] * df["quantity"]
            high_rev = df[df["revenue"] > 500]
            return high_rev.sort_values("revenue", ascending=False).head(3)

        result = pipeline(sales_df)
        assert "revenue" in result.columns
        assert all(result["revenue"] > 500)
        assert len(result) <= 3
        assert list(result["revenue"]) == sorted(result["revenue"], reverse=True)

    def test_nlargest(self, sales_df):
        @compilable
        def top_by_price(df):
            return df.nlargest(3, "price")

        result = top_by_price(sales_df)
        assert len(result) == 3
        assert list(result["price"]) == sorted(result["price"], reverse=True)

    def test_drop_columns(self, sales_df):
        @compilable
        def drop_qty(df):
            return df.drop(columns=["quantity"])

        result = drop_qty(sales_df)
        assert "quantity" not in result.columns
        assert "price" in result.columns

    def test_query_string(self, sales_df):
        @compilable
        def query_filter(df):
            return df.query("price > 100")

        result = query_filter(sales_df)
        assert all(result["price"] > 100)

    def test_rename_columns(self, sales_df):
        @compilable
        def rename_cols(df):
            return df.rename(columns={"price": "cost", "quantity": "qty"})

        result = rename_cols(sales_df)
        assert "cost" in result.columns
        assert "qty" in result.columns
        assert "price" not in result.columns
        assert "quantity" not in result.columns

    def test_assign(self, sales_df):
        @compilable
        def with_discount(df):
            return df.assign(discount=df["price"] * 0.1)

        result = with_discount(sales_df)
        assert "discount" in result.columns
        r = result.sort_values("id").reset_index(drop=True)
        expected = sales_df.sort_values("id").reset_index(drop=True)["price"] * 0.1
        tm.assert_numpy_array_equal(r["discount"].values, expected.values)

    def test_nsmallest(self, sales_df):
        @compilable
        def bottom_by_price(df):
            return df.nsmallest(3, "price")

        result = bottom_by_price(sales_df)
        assert len(result) == 3
        assert list(result["price"]) == sorted(result["price"])

    def test_merge(self, sales_df, regions_df):
        @compilable
        def with_manager(df, regions):
            return df.merge(regions, on="region")

        result = with_manager(sales_df, regions_df)
        assert "manager" in result.columns
        assert len(result) == len(sales_df)

    def test_isin_filter(self, sales_df):
        @compilable
        def east_west(df):
            return df[df["product"].isin(["A"])]

        result = east_west(sales_df)
        assert all(result["product"] == "A")

    def test_dropna(self, nullable_df):
        @compilable
        def drop_nulls(df):
            return df.dropna(subset=["score"])

        result = drop_nulls(nullable_df)
        assert result["score"].notna().all()
        assert len(result) == 3

    def test_isna_filter(self, nullable_df):
        @compilable
        def get_missing(df):
            return df[df["score"].isna()]

        result = get_missing(nullable_df)
        assert len(result) == 2
        assert result["score"].isna().all()

    def test_notna_filter(self, nullable_df):
        @compilable
        def get_present(df):
            return df[df["score"].notna()]

        result = get_present(nullable_df)
        assert len(result) == 3
        assert result["score"].notna().all()

    def test_fillna_scalar(self, nullable_df):
        @compilable
        def fill_zeros(df):
            df["score"] = df["score"].fillna(0.0)
            return df

        result = fill_zeros(nullable_df)
        assert result["score"].notna().all()
        filled = result.sort_values("id").reset_index(drop=True)
        assert filled.loc[1, "score"] == 0.0
        assert filled.loc[3, "score"] == 0.0

    def test_fillna_dataframe(self, nullable_df):
        @compilable
        def fill_df(df):
            return df.fillna({"score": -1.0})

        result = fill_df(nullable_df)
        assert result["score"].notna().all()
        filled = result.sort_values("id").reset_index(drop=True)
        assert filled.loc[1, "score"] == -1.0

    def test_series_abs(self, sales_df):
        @compilable
        def price_dist(df):
            df["dist"] = (df["price"] - 200.0).abs()
            return df

        result = price_dist(sales_df)
        assert "dist" in result.columns
        assert all(result["dist"] >= 0)

    def test_series_negation(self, sales_df):
        @compilable
        def neg_price(df):
            df["neg"] = -df["price"]
            return df

        result = neg_price(sales_df)
        assert all(result["neg"] < 0)
        r = result.sort_values("id").reset_index(drop=True)
        tm.assert_numpy_array_equal(
            r["neg"].values,
            -sales_df.sort_values("id").reset_index(drop=True)["price"].values,
        )

    def test_between_filter(self, sales_df):
        @compilable
        def mid_range(df):
            return df[df["price"].between(100.0, 200.0)]

        result = mid_range(sales_df)
        assert all(result["price"] >= 100)
        assert all(result["price"] <= 200)

    def test_compound_boolean_filter(self, sales_df):
        @compilable
        def compound(df):
            return df[(df["price"] > 100) & (df["region"] == "East")]

        result = compound(sales_df)
        assert all(result["price"] > 100)
        assert all(result["region"] == "East")

    def test_cache_reuse(self, sales_df):
        @compilable
        def f(df):
            return df[df["price"] > 100]

        r1 = f(sales_df)
        r2 = f(sales_df)
        assert len(f._cached_plans) == 1
        tm.assert_frame_equal(
            r1.sort_values("id").reset_index(drop=True),
            r2.sort_values("id").reset_index(drop=True),
        )

    def test_explain_output(self, sales_df):
        @compilable
        def f(df):
            return df[df["price"] > 100]

        plan_str = f.explain(sales_df)
        assert "ExecutionPlan" in plan_str
        assert "COMPILED" in plan_str

    def test_cache_invalidation_on_scalar_change(self, sales_df):
        """Changing a scalar parameter must re-trace, not reuse stale plan."""

        @compilable
        def f(df, threshold):
            return df[df["price"] > threshold]

        r1 = f(sales_df, 100)
        assert all(r1["price"] > 100)

        r2 = f(sales_df, 200)
        assert all(r2["price"] > 200)

        # Two different scalar args → two cached plans
        assert len(f._cached_plans) == 2

    def test_cache_hit_same_scalar(self, sales_df):
        """Same scalar parameter should reuse the cached plan."""

        @compilable
        def f(df, threshold):
            return df[df["price"] > threshold]

        f(sales_df, 100)
        f(sales_df, 100)
        assert len(f._cached_plans) == 1

    def test_cache_invalidation_on_kwarg_change(self, sales_df):
        """Changing a keyword argument must re-trace."""

        @compilable
        def f(df, ascending=True):
            return df.sort_values("price", ascending=ascending)

        r1 = f(sales_df, ascending=True)
        r2 = f(sales_df, ascending=False)

        # Different kwargs → two cached plans
        assert len(f._cached_plans) == 2

        # Results should differ in order
        assert r1.iloc[0]["price"] != r2.iloc[0]["price"]

    def test_cache_invalidation_on_multiple_scalars(self, sales_df):
        """Multiple scalar args must all match for cache hit."""

        @compilable
        def f(df, lo, hi):
            return df[df["price"].between(lo, hi)]

        r1 = f(sales_df, 50.0, 150.0)
        r2 = f(sales_df, 100.0, 300.0)

        assert len(f._cached_plans) == 2
        assert all(r1["price"] >= 50)
        assert all(r1["price"] <= 150)
        assert all(r2["price"] >= 100)
        assert all(r2["price"] <= 300)

    def test_multi_df_join(self):
        """Two DataFrame arguments are traced and joined."""
        orders = DataFrame(
            {
                "order_id": [1, 2, 3],
                "product_id": [10, 20, 10],
                "qty": [5, 3, 2],
            }
        )
        products = DataFrame(
            {
                "product_id": [10, 20],
                "name": ["Widget", "Gadget"],
                "price": [9.99, 19.99],
            }
        )

        @compilable
        def join_two(df_orders, df_products):
            return df_orders.merge(df_products, on="product_id")

        result = join_two(orders, products)
        assert len(result) == 3
        assert set(result.columns) == {
            "order_id",
            "product_id",
            "qty",
            "name",
            "price",
        }
        # Verify join correctness
        assert result.loc[result["order_id"] == 2, "name"].iloc[0] == "Gadget"

    def test_multi_df_join_with_filter(self):
        """Join two DataFrames and then filter the result."""
        orders = DataFrame(
            {
                "order_id": [1, 2, 3],
                "product_id": [10, 20, 10],
                "qty": [5, 3, 2],
            }
        )
        products = DataFrame(
            {
                "product_id": [10, 20],
                "name": ["Widget", "Gadget"],
                "price": [9.99, 19.99],
            }
        )

        @compilable
        def expensive_orders(df_orders, df_products):
            merged = df_orders.merge(df_products, on="product_id")
            return merged[merged["price"] > 15]

        result = expensive_orders(orders, products)
        assert len(result) == 1
        assert all(result["price"] > 15)
        assert result.iloc[0]["name"] == "Gadget"

    def test_composite_key_join(self):
        """Join two DataFrames on multiple columns."""
        orders = DataFrame(
            {
                "year": [2024, 2024, 2025],
                "region": ["E", "W", "E"],
                "revenue": [100, 200, 300],
            }
        )
        targets = DataFrame(
            {
                "year": [2024, 2025],
                "region": ["E", "E"],
                "target": [150, 250],
            }
        )

        @compilable
        def with_target(df_orders, df_targets):
            return df_orders.merge(df_targets, on=["year", "region"])

        result = with_target(orders, targets)
        assert "target" in result.columns
        assert len(result) == 2
        assert set(result["revenue"]) == {100, 300}

    def test_concat_then_filter(self):
        a = DataFrame({"id": [1, 2], "val": [10, 20]})
        b = DataFrame({"id": [3, 4], "val": [30, 40]})

        @compilable
        def stacked_and_filtered(df1, df2):
            combined = pd.concat([df1, df2])
            return combined[combined["val"] > 15]

        result = stacked_and_filtered(a, b)
        assert len(result) == 3
        assert all(result["val"] > 15)

    def test_filter_then_iloc_slice(self):
        df = DataFrame({"id": range(20), "val": range(20)})

        @compilable
        def top_filtered(df):
            big = df[df["val"] >= 5]
            return big.iloc[:3]

        result = top_filtered(df)
        assert len(result) == 3
        assert list(result["id"]) == [5, 6, 7]

    def test_sort_then_iloc_offset(self):
        df = DataFrame({"id": [3, 1, 4, 1, 5, 9], "val": [30, 10, 40, 10, 50, 90]})

        @compilable
        def middle_sorted(df):
            sorted_df = df.sort_values("val")
            return sorted_df.iloc[2:4]

        result = middle_sorted(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# User scenarios: @compilable with graph breaks
# ---------------------------------------------------------------------------


class TestCompileGraphBreaks:
    """Tests for @compilable with operations that force materialization."""

    def test_len_graph_break(self, sales_df):
        @compilable
        def f(df):
            filtered = df[df["price"] > 100]
            if len(filtered) > 0:
                return filtered
            return df

        result = f(sales_df)
        assert all(result["price"] > 100)

    def test_shape_graph_break(self, sales_df):
        @compilable
        def f(df):
            filtered = df[df["price"] > 100]
            n_rows, _ = filtered.shape
            return filtered.head(min(n_rows, 2))

        result = f(sales_df)
        assert len(result) == 2

    def test_iterrows_graph_break(self, sales_df):
        @compilable
        def f(df):
            sorted_df = df.sort_values("price", ascending=False)
            top = sorted_df.head(2)
            ids = []
            for _, row in top.iterrows():
                ids.append(row["id"])
            return top

        result = f(sales_df)
        assert len(result) == 2

    def test_drop_duplicates_traced(self, sales_df):
        @compilable
        def f(df):
            return df[["region", "product"]].drop_duplicates()

        result = f(sales_df)
        assert len(result) <= len(sales_df)
        # No duplicate region+product pairs
        assert not result.duplicated().any()

    def test_filter_then_drop_duplicates(self, sales_df):
        @compilable
        def f(df):
            big = df[df["price"] > 100]
            return big[["region"]].drop_duplicates()

        result = f(sales_df)
        assert not result.duplicated().any()
        assert "region" in result.columns


# ---------------------------------------------------------------------------
# User scenarios: Tracer context manager
# ---------------------------------------------------------------------------


class TestTracerScenarios:
    """End-to-end tests using the Tracer context manager."""

    def test_basic_filter(self, sales_df):
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            result = df[df["price"] > 100]
            t.output(result)

        result_df = t.result()
        assert all(result_df["price"] > 100)
        assert len(result_df) == len(sales_df[sales_df["price"] > 100])

    def test_add_column_and_sort(self, sales_df):
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            df["revenue"] = df["price"] * df["quantity"]
            sorted_df = df.sort_values("revenue", ascending=False)
            t.output(sorted_df)

        result = t.result()
        assert "revenue" in result.columns
        assert list(result["revenue"]) == sorted(result["revenue"], reverse=True)

    def test_filter_sort_head(self, sales_df):
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            expensive = df[df["price"] > 50]
            top = expensive.sort_values("price", ascending=False).head(3)
            t.output(top)

        result = t.result()
        assert len(result) == 3
        assert all(result["price"] > 50)

    def test_explain_shows_plan(self, sales_df):
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            t.output(df[df["price"] > 100])

        plan = t.explain()
        assert "CompiledSegment" in plan

    def test_graph_break_then_continue(self, sales_df):
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            n = len(df)
            assert n == 6
            t.output(df.head(n - 1))

        result = t.result()
        assert len(result) == 5

    def test_dropna(self, nullable_df):
        with Tracer() as t:
            df = t.input(nullable_df, "data")
            clean = df.dropna(subset=["score"])
            t.output(clean)

        result = t.result()
        assert result["score"].notna().all()
        assert len(result) == 3

    def test_fillna_and_filter(self, nullable_df):
        with Tracer() as t:
            df = t.input(nullable_df, "data")
            df["score"] = df["score"].fillna(0.0)
            result = df[df["score"] > 0]
            t.output(result)

        result_df = t.result()
        assert result_df["score"].notna().all()
        assert all(result_df["score"] > 0)

    def test_between(self, sales_df):
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            mid = df[df["price"].between(100.0, 200.0)]
            t.output(mid)

        result = t.result()
        assert all(result["price"] >= 100)
        assert all(result["price"] <= 200)


# ---------------------------------------------------------------------------
# User scenarios: top-level jit access
# ---------------------------------------------------------------------------


class TestPdJitScenarios:
    """End-to-end tests using compilable as users would."""

    def test_pd_jit_filter(self, sales_df):
        @compilable
        def f(df):
            return df[df["price"] > 100]

        result = f(sales_df)
        assert all(result["price"] > 100)

    def test_pd_jit_pipeline(self, sales_df):
        @compilable
        def pipeline(df):
            df["revenue"] = df["price"] * df["quantity"]
            return df.sort_values("revenue", ascending=False).head(3)

        result = pipeline(sales_df)
        assert len(result) == 3
        assert "revenue" in result.columns

    def test_pd_jit_groupby(self, sales_df):
        @compilable
        def agg(df):
            return df.groupby("region").sum()

        result = agg(sales_df)
        assert "region" in result.columns


# ---------------------------------------------------------------------------
# Substrait export
# ---------------------------------------------------------------------------


class TestSubstraitExport:
    """Tests for Substrait plan export on CompiledFunction and Tracer."""

    def test_compiled_function_to_substrait(self, sales_df):
        @compilable
        def f(df):
            return df[df["price"] > 100]

        plans = f.to_substrait(sales_df)
        assert len(plans) >= 1
        # Plan is a protobuf with required fields
        plan = plans[0]
        assert plan.version.major_number >= 0
        assert len(plan.relations) >= 1

    def test_compiled_function_to_substrait_serializes(self, sales_df):
        @compilable
        def f(df):
            return df[df["price"] > 100]

        plans = f.to_substrait(sales_df)
        plan_bytes = plans[0].SerializeToString()
        assert isinstance(plan_bytes, bytes)
        assert len(plan_bytes) > 0

    def test_compiled_function_to_substrait_json(self, sales_df):
        @compilable
        def f(df):
            return df[df["price"] > 100]

        json_str = f.to_substrait_json(sales_df)
        import json

        parsed = json.loads(json_str)
        assert "version" in parsed
        assert "relations" in parsed

    def test_tracer_to_substrait(self, sales_df):
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            t.output(df[df["price"] > 100])

        plans = t.to_substrait()
        assert len(plans) >= 1
        plan_bytes = plans[0].SerializeToString()
        assert isinstance(plan_bytes, bytes)
        assert len(plan_bytes) > 0

    def test_tracer_to_substrait_json(self, sales_df):
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            t.output(df[df["price"] > 100])

        json_str = t.to_substrait_json()
        import json

        parsed = json.loads(json_str)
        assert "version" in parsed
        assert "relations" in parsed

    def test_tracer_backward_compat_alias(self, sales_df):
        """to_substrait_plans() still works as an alias."""
        with Tracer() as t:
            df = t.input(sales_df, "sales")
            t.output(df[df["price"] > 100])

        plans = t.to_substrait_plans()
        assert len(plans) >= 1
        assert plans[0].SerializeToString() == t.to_substrait()[0].SerializeToString()

    def test_multi_segment_export(self, sales_df):
        """Graph break produces multiple segments; only compiled ones export."""

        @compilable
        def f(df):
            filtered = df[df["price"] > 100]
            n = len(filtered)  # graph break
            return filtered.head(min(n, 3))

        plans = f.to_substrait(sales_df)
        # Should have at least one plan from the compiled segments
        assert len(plans) >= 1

    def test_substrait_plan_runs_on_acero(self, sales_df):
        """Exported Substrait plan can be executed on Acero."""

        @compilable
        def f(df):
            return df[df["price"] > 100]

        plans = f.to_substrait(sales_df)
        plan_bytes = plans[0].SerializeToString()

        pa_tables = {"df": pa.Table.from_pandas(sales_df)}

        def provider(names, _schema):
            return pa_tables[names[0]]

        acero_result = (
            pas.run_query(plan_bytes, table_provider=provider).read_all().to_pandas()
        )
        assert all(acero_result["price"] > 100)
        assert len(acero_result) == len(sales_df[sales_df["price"] > 100])


# ---------------------------------------------------------------------------
# Dtype round-trips through Acero
# ---------------------------------------------------------------------------


class TestAceroDtypeRoundTrip:
    @pytest.mark.parametrize(
        "col_data, np_dtype, expected_ir_dtype",
        [
            (np.array([1, 2, 3], dtype="int8"), "int8", DType.INT8),
            (np.array([1, 2, 3], dtype="int16"), "int16", DType.INT16),
            (np.array([1, 2, 3], dtype="int32"), "int32", DType.INT32),
            (np.array([1, 2, 3], dtype="int64"), "int64", DType.INT64),
            (np.array([1.0, 2.0], dtype="float32"), "float32", DType.FLOAT32),
            (np.array([1.0, 2.0], dtype="float64"), "float64", DType.FLOAT64),
        ],
    )
    def test_numeric_types_survive_acero(self, col_data, np_dtype, expected_ir_dtype):
        df = DataFrame({"x": col_data, "y": col_data * 2})
        schema = infer_schema(df)
        assert schema.columns["x"] is expected_ir_dtype

        # Filter x > 1 and run on Acero
        node = Filter(
            ReadTable("t", schema),
            BinOp("gt", ColRef("x"), Literal(1, expected_ir_dtype)),
        )
        result = _run_on_acero(node, {"t": df})
        assert len(result) > 0
        assert all(result["x"] > 1)


# ---------------------------------------------------------------------------
# Datetime accessor (.dt)
# ---------------------------------------------------------------------------


class TestDatetimeAccessor:
    def test_dt_year_filter(self, datetime_df):
        @compilable
        def f(df):
            return df[df["ts"].dt.year == 2024]

        result = f(datetime_df)
        expected = datetime_df[datetime_df["ts"].dt.year == 2024]
        assert len(result) == len(expected)

    def test_dt_month_add_column(self, datetime_df):
        @compilable
        def f(df):
            df["month"] = df["ts"].dt.month
            return df

        result = f(datetime_df)
        assert "month" in result.columns
        result_sorted = result.sort_values("value").reset_index(drop=True)
        expected_sorted = datetime_df.sort_values("value").reset_index(drop=True)
        expected_months = expected_sorted["ts"].dt.month.values
        tm.assert_numpy_array_equal(
            result_sorted["month"].values.astype(int), expected_months.astype(int)
        )

    def test_dt_day_of_week(self, datetime_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["dow"] = df["ts"].dt.dayofweek
            return df

        result = f(datetime_df)
        assert "dow" in result.columns
        assert all(result["dow"] >= 0)
        assert all(result["dow"] <= 6)

    def test_dt_quarter(self, datetime_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["q"] = df["ts"].dt.quarter
            return df

        result = f(datetime_df)
        assert "q" in result.columns
        assert all(result["q"] >= 1)
        assert all(result["q"] <= 4)

    def test_dt_hour(self, datetime_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["h"] = df["ts"].dt.hour
            return df

        result = f(datetime_df)
        assert "h" in result.columns
        expected = datetime_df["ts"].dt.hour.values
        result_sorted = result.sort_values("value").reset_index(drop=True)
        tm.assert_numpy_array_equal(
            result_sorted["h"].values.astype(int), expected.astype(int)
        )

    def test_dt_multiple_components(self, datetime_df):
        @compilable
        def f(df):
            df["year"] = df["ts"].dt.year
            df["month"] = df["ts"].dt.month
            df["day"] = df["ts"].dt.day
            return df

        result = f(datetime_df)
        assert {"year", "month", "day"}.issubset(result.columns)


# ---------------------------------------------------------------------------
# String accessor (.str) — PandasBackend
# ---------------------------------------------------------------------------


class TestStringAccessor:
    def test_str_upper(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["upper_name"] = df["name"].str.upper()
            return df

        result = f(string_df)
        assert all(result["upper_name"] == string_df["name"].str.upper())

    def test_str_lower(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["lower_name"] = df["name"].str.lower()
            return df

        result = f(string_df)
        assert all(result["lower_name"] == string_df["name"].str.lower())

    def test_str_contains_filter(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["name"].str.contains("li", regex=False)]

        result = f(string_df)
        assert len(result) == 2  # Alice, Charlie

    def test_str_startswith(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["name"].str.startswith("A")]

        result = f(string_df)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_str_endswith(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            return df[df["name"].str.endswith("e")]

        result = f(string_df)
        assert all(r.endswith("e") for r in result["name"])

    def test_str_strip(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["clean_city"] = df["city"].str.strip()
            return df

        result = f(string_df)
        assert result.iloc[0]["clean_city"] == "New York"

    def test_str_len(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["name_len"] = df["name"].str.len()
            return df

        result = f(string_df)
        expected_lens = string_df["name"].str.len()
        result_sorted = result.sort_values("name").reset_index(drop=True)
        expected_sorted = expected_lens[
            np.argsort(string_df["name"].values)
        ].reset_index(drop=True)
        tm.assert_series_equal(
            result_sorted["name_len"], expected_sorted, check_names=False
        )

    def test_str_replace(self, string_df):
        @compilable(backend=PandasBackend())
        def f(df):
            df["replaced"] = df["name"].str.replace("e", "X", regex=False)
            return df

        result = f(string_df)
        assert result.iloc[0]["replaced"] == "AlicX"  # Alice -> AlicX


# ---------------------------------------------------------------------------
# Graph-break operations: rolling, expanding, reshape, astype, where, mask
# ---------------------------------------------------------------------------


class TestRolling:
    def test_rolling_mean(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("price")
            return nums.rolling(2).mean()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("price").rolling(2).mean()
        )
        assert len(result) == len(expected)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_rolling_sum(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.rolling(3).sum()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").rolling(3).sum()
        )
        assert len(result) == len(expected)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_rolling_std(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.rolling(3).std()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").rolling(3).std()
        )
        assert len(result) == len(expected)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_rolling_then_filter(self, sales_df):
        """Graph break from rolling, then continue tracing with filter."""

        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            rolled = nums.rolling(2).mean()
            return rolled[rolled["price"] > 100]

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").rolling(2).mean()
        )
        expected = expected[expected["price"] > 100].reset_index(drop=True)
        assert len(result) == len(expected)
        assert all(result["price"] > 100)

    def test_rolling_min(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.rolling(2).min()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").rolling(2).min()
        )
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_rolling_max(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.rolling(2).max()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").rolling(2).max()
        )
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_rolling_var(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.rolling(3).var()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").rolling(3).var()
        )
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_rolling_count(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.rolling(2).count()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").rolling(2).count()
        )
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


class TestExpanding:
    def test_expanding_sum(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.expanding().sum()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").expanding().sum()
        )
        assert len(result) == len(expected)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_expanding_mean(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.expanding().mean()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").expanding().mean()
        )
        assert len(result) == len(expected)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_expanding_min(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.expanding().min()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").expanding().min()
        )
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_expanding_max(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.expanding().max()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").expanding().max()
        )
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_expanding_count(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.expanding().count()

        result = f(sales_df)
        expected = (
            sales_df[["id", "price", "quantity"]].sort_values("id").expanding().count()
        )
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


class TestCumulative:
    def test_cumsum_end_to_end(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.cumsum()

        result = f(sales_df)
        expected = sales_df[["id", "price", "quantity"]].sort_values("id").cumsum()
        assert len(result) == len(expected)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_cummax_end_to_end(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.cummax()

        result = f(sales_df)
        expected = sales_df[["id", "price", "quantity"]].sort_values("id").cummax()
        assert len(result) == len(expected)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_cumprod_end_to_end(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.cumprod()

        result = f(sales_df)
        expected = sales_df[["id", "price", "quantity"]].sort_values("id").cumprod()
        assert len(result) == len(expected)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_series_cumsum_assign(self, sales_df):
        @compilable
        def f(df):
            df = df.sort_values("id")
            df["price_cumsum"] = df["price"].cumsum()
            return df

        result = f(sales_df)
        expected = sales_df.sort_values("id").copy()
        expected["price_cumsum"] = expected["price"].cumsum()
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_series_cumsum_then_filter(self, sales_df):
        @compilable
        def f(df):
            df = df.sort_values("id")
            df["running_total"] = df["price"].cumsum()
            return df[df["running_total"] <= 500]

        result = f(sales_df)
        expected = sales_df.sort_values("id").copy()
        expected["running_total"] = expected["price"].cumsum()
        expected = expected[expected["running_total"] <= 500]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


class TestPivotTable:
    def test_pivot_table(self, sales_df):
        @compilable
        def f(df):
            return df.pivot_table(values="price", index="region", aggfunc="mean")

        result = f(sales_df)
        expected = sales_df.pivot_table(
            values="price", index="region", aggfunc="mean"
        ).reset_index()
        assert "region" in result.columns
        assert "price" in result.columns
        tm.assert_frame_equal(
            result.sort_values("region").reset_index(drop=True),
            expected.sort_values("region").reset_index(drop=True),
            check_dtype=False,
        )

    def test_pivot_then_filter(self, sales_df):
        """Graph break from pivot_table, then continue tracing."""

        @compilable
        def f(df):
            pivoted = df.pivot_table(values="price", index="region", aggfunc="mean")
            return pivoted[pivoted["price"] > 150]

        result = f(sales_df)
        expected = sales_df.pivot_table(
            values="price", index="region", aggfunc="mean"
        ).reset_index()
        expected = expected[expected["price"] > 150].reset_index(drop=True)
        assert len(result) == len(expected)


class TestMelt:
    def test_melt(self, sales_df):
        @compilable
        def f(df):
            return df[["id", "price", "quantity"]].melt(
                id_vars=["id"], value_vars=["price", "quantity"]
            )

        result = f(sales_df)
        expected = sales_df[["id", "price", "quantity"]].melt(
            id_vars=["id"], value_vars=["price", "quantity"]
        )
        assert set(result.columns) == {"id", "variable", "value"}
        assert len(result) == len(expected)


class TestAstype:
    def test_astype_dataframe(self, sales_df):
        @compilable
        def f(df):
            return df[["id", "price"]].astype({"price": "int64"})

        result = f(sales_df)
        assert result["price"].dtype == np.int64

    def test_astype_then_filter(self, sales_df):
        """astype graph break, then continue tracing."""

        @compilable
        def f(df):
            casted = df[["id", "price"]].astype({"price": "int64"})
            return casted[casted["price"] > 100]

        result = f(sales_df)
        assert result["price"].dtype == np.int64
        assert all(result["price"] > 100)


class TestWhereMask:
    def test_where_dataframe(self, sales_df):
        @compilable
        def f(df):
            return df[["id", "price"]].where(sales_df[["id", "price"]] > 100, other=-1)

        result = f(sales_df)
        assert len(result) == len(sales_df)
        # Values ≤ 100 should be -1 (for price column)
        # id column: values > 100 are kept, rest are -1
        # Since id values are 1-6, all ≤ 100, so all id = -1
        # price: 100 → -1, 250 → 250, ...

    def test_mask_dataframe(self, sales_df):
        @compilable
        def f(df):
            return df[["id", "price"]].mask(sales_df[["id", "price"]] > 100, other=0)

        result = f(sales_df)
        assert len(result) == len(sales_df)

    def test_where_with_pandas_condition(self):
        """where with a plain pandas boolean condition (not traced)."""
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        cond = df > 2

        @compilable
        def f(d):
            return d.where(cond, other=-1)

        result = f(df)
        expected = df.where(cond, other=-1)
        tm.assert_frame_equal(result, expected)

    def test_where_traced_condition(self, sales_df):
        """where with a traced boolean series builds IR (no graph break)."""

        @compilable(backend=PandasBackend())
        def f(df):
            return df[["id", "price"]].where(df["price"] > 100, other=-1)

        result = f(sales_df)
        expected = sales_df[["id", "price"]].where(sales_df["price"] > 100, other=-1)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_mask_traced_condition(self, sales_df):
        """mask with a traced boolean series builds IR (no graph break)."""

        @compilable(backend=PandasBackend())
        def f(df):
            return df[["id", "price"]].mask(df["price"] > 200, other=0)

        result = f(sales_df)
        expected = sales_df[["id", "price"]].mask(sales_df["price"] > 200, other=0)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


# ---------------------------------------------------------------------------
# reset_index / set_index integration tests
# ---------------------------------------------------------------------------


class TestResetSetIndex:
    def test_groupby_reset_index_end_to_end(self):
        """groupby().sum().reset_index(drop=True) — common pattern.

        JIT Aggregate keeps group keys as columns (SQL semantics),
        so reset_index(drop=True) is a no-op. Both JIT and pandas
        produce the same columns when accounting for this.
        """
        df = DataFrame({"group": ["a", "a", "b", "b"], "value": [10, 20, 30, 40]})

        @compilable
        def f(df):
            return df.groupby("group").sum().reset_index(drop=True)

        result = f(df)
        # JIT keeps group keys as columns; pandas puts them in index.
        # reset_index(drop=False) on pandas gives same shape as JIT.
        expected = df.groupby("group").sum().reset_index(drop=False)
        # Backend may not preserve sort order.
        tm.assert_frame_equal(
            result.sort_values("group").reset_index(drop=True),
            expected.sort_values("group").reset_index(drop=True),
        )

    def test_set_index_end_to_end(self):
        """set_index() sets a column as the index."""
        df = DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

        @compilable
        def f(df):
            return df.set_index("id")

        result = f(df)
        expected = df.set_index("id")
        tm.assert_frame_equal(result, expected)

    def test_set_index_then_reset(self):
        """set_index() → reset_index() round-trip."""
        df = DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

        @compilable
        def f(df):
            return df.set_index("id").reset_index()

        result = f(df)
        expected = df.set_index("id").reset_index()
        tm.assert_frame_equal(result, expected)

    def test_series_reset_index_drop_false(self):
        """Series.reset_index(drop=False) returns a DataFrame."""
        df = DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="idx"))

        @compilable
        def f(df):
            return df["a"].reset_index(drop=False)

        result = f(df)
        expected = df["a"].reset_index(drop=False)
        tm.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# shift / diff integration tests
# ---------------------------------------------------------------------------


class TestShiftDiff:
    def test_shift_end_to_end(self):
        """shift(1) produces correct lagged values."""
        df = DataFrame({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0, 40.0]})

        @compilable
        def f(df):
            return df.shift(1)

        result = f(df)
        expected = df.shift(1)
        tm.assert_frame_equal(result, expected)

    def test_diff_end_to_end(self):
        """diff(1) produces correct first differences."""
        df = DataFrame({"a": [1, 3, 6, 10], "b": [10.0, 20.0, 30.0, 40.0]})

        @compilable
        def f(df):
            return df.diff(1)

        result = f(df)
        expected = df.diff(1)
        tm.assert_frame_equal(result, expected)

    def test_shift_then_filter(self):
        """shift + downstream filter operation."""
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            shifted = df.shift(1)
            return shifted[shifted["a"] > 2]

        result = f(df)
        expected = df.shift(1)
        expected = expected[expected["a"] > 2]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


class TestRank:
    def test_rank_end_to_end(self):
        """Series rank with method='min' end-to-end."""
        df = DataFrame({"a": [3, 1, 4, 1, 5]})

        @compilable
        def f(df):
            return df.assign(r=df["a"].rank(method="min"))

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="min"))
        tm.assert_frame_equal(result, expected)

    def test_groupby_rank_end_to_end(self):
        """GroupBy series rank end-to-end."""
        df = DataFrame(
            {
                "g": ["a", "a", "b", "b", "a"],
                "v": [3, 1, 4, 2, 5],
            }
        )

        @compilable
        def f(df):
            return df.assign(r=df.groupby("g")["v"].rank(method="dense"))

        result = f(df)
        expected = df.assign(r=df.groupby("g")["v"].rank(method="dense"))
        tm.assert_frame_equal(result, expected)

    def test_rank_then_filter(self):
        """Rank followed by downstream filter operation."""
        df = DataFrame({"a": [3, 1, 4, 1, 5], "b": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            ranked = df.assign(r=df["a"].rank(method="min"))
            return ranked[ranked["r"] <= 2]

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="min"))
        expected = expected[expected["r"] <= 2]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


class TestCrossIRComposition:
    def test_rank_assign_pipeline(self):
        """Full pipeline: rank → assign → filter."""
        df = DataFrame({"a": [3, 1, 4, 1, 5], "b": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            ranked = df.assign(r=df["a"].rank(method="min"))
            return ranked[ranked["r"] <= 2]

        result = f(df)
        expected = df.assign(r=df["a"].rank(method="min"))
        expected = expected[expected["r"] <= 2]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_groupby_rank_filter(self):
        """Groupby rank → assign → filter pipeline."""
        df = DataFrame(
            {
                "g": ["a", "a", "b", "b", "a"],
                "v": [3, 1, 4, 2, 5],
            }
        )

        @compilable
        def f(df):
            ranked = df.assign(r=df.groupby("g")["v"].rank(method="dense"))
            return ranked[ranked["r"] == 1]

        result = f(df)
        expected = df.assign(r=df.groupby("g")["v"].rank(method="dense"))
        expected = expected[expected["r"] == 1]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_shift_assign_pipeline(self):
        """Series shift → assign → sort pipeline."""
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            return df.assign(a_prev=df["a"].shift(1)).sort_values("b")

        result = f(df)
        expected = df.assign(a_prev=df["a"].shift(1)).sort_values("b")
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# DataFusion backend integration tests
# ---------------------------------------------------------------------------

datafusion_mod = pytest.importorskip("datafusion")
pytest.importorskip("datafusion.substrait")


def _run_on_datafusion(
    ir_node,
    tables: dict[str, DataFrame],
) -> DataFrame:
    """Compile IR to Substrait, run on DataFusion, return pandas DataFrame."""
    from datafusion import SessionContext
    from datafusion.substrait import (
        Consumer,
        Serde,
    )

    compiler = SubstraitCompiler()
    plan = compiler.compile(ir_node)
    plan_bytes = plan.SerializeToString()

    ctx = SessionContext()
    for name, df in tables.items():
        arrow_table = pa.Table.from_pandas(df)
        # Cast large_string/large_binary to utf8/binary for Substrait compat
        new_fields = []
        needs_cast = False
        for field in arrow_table.schema:
            if field.type == pa.large_string():
                new_fields.append(pa.field(field.name, pa.utf8()))
                needs_cast = True
            elif field.type == pa.large_binary():
                new_fields.append(pa.field(field.name, pa.binary()))
                needs_cast = True
            else:
                new_fields.append(field)
        if needs_cast:
            arrow_table = arrow_table.cast(pa.schema(new_fields))
        ctx.register_record_batches(name, [arrow_table.to_batches()])

    substrait_plan = Serde.deserialize_bytes(plan_bytes)
    logical_plan = Consumer.from_substrait_plan(ctx, substrait_plan)
    result_df = ctx.create_dataframe_from_logical_plan(logical_plan)
    return result_df.to_pandas()


class TestPandasVsDataFusion:
    def test_filter_matches(self, sales_df):
        schema = infer_schema(sales_df)
        node = Filter(
            ReadTable("sales", schema),
            BinOp("gte", ColRef("price"), Literal(150.0, DType.FLOAT64)),
        )
        pandas_result = PandasBackend().execute(node, {"sales": sales_df})
        df_result = _run_on_datafusion(node, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.sort_values("id").reset_index(drop=True),
            df_result.sort_values("id").reset_index(drop=True),
        )

    def test_add_column_matches(self, sales_df):
        schema = infer_schema(sales_df)
        base = ReadTable("sales", schema)
        node = AddColumn(
            base,
            "revenue",
            BinOp("mul", ColRef("price"), ColRef("quantity")),
            DType.FLOAT64,
        )
        pandas_result = PandasBackend().execute(node, {"sales": sales_df})
        df_result = _run_on_datafusion(node, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.sort_values("id").reset_index(drop=True),
            df_result.sort_values("id").reset_index(drop=True),
        )

    def test_aggregate_matches(self, sales_df):
        schema = infer_schema(sales_df)
        node = Aggregate(
            ReadTable("sales", schema),
            group_keys=["region"],
            agg_specs=[("total", "price", "sum"), ("cnt", "id", "count")],
        )
        pandas_result = PandasBackend().execute(node, {"sales": sales_df})
        df_result = _run_on_datafusion(node, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.sort_values("region").reset_index(drop=True),
            df_result.sort_values("region").reset_index(drop=True),
        )

    def test_sort_limit_matches(self, sales_df):
        schema = infer_schema(sales_df)
        node = Limit(
            Sort(ReadTable("sales", schema), [("price", False)]),
            3,
        )
        pandas_result = PandasBackend().execute(node, {"sales": sales_df})
        df_result = _run_on_datafusion(node, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.reset_index(drop=True),
            df_result.reset_index(drop=True),
        )

    def test_join_matches(self, sales_df, regions_df):
        left_schema = infer_schema(sales_df)
        right_schema = infer_schema(regions_df)
        node = Join(
            ReadTable("sales", left_schema),
            ReadTable("regions", right_schema),
            "region",
            "region",
        )
        pandas_result = PandasBackend().execute(
            node, {"sales": sales_df, "regions": regions_df}
        )
        df_result = _run_on_datafusion(node, {"sales": sales_df, "regions": regions_df})

        tm.assert_frame_equal(
            pandas_result.sort_values("id").reset_index(drop=True),
            df_result.sort_values("id").reset_index(drop=True),
        )

    def test_full_pipeline_matches(self, sales_df):
        """Complex pipeline: filter -> add col -> sort -> limit."""
        schema = infer_schema(sales_df)
        base = ReadTable("sales", schema)

        pipeline = Limit(
            Sort(
                AddColumn(
                    Filter(
                        base,
                        BinOp("gt", ColRef("price"), Literal(50.0, DType.FLOAT64)),
                    ),
                    "revenue",
                    BinOp("mul", ColRef("price"), ColRef("quantity")),
                    DType.FLOAT64,
                ),
                [("revenue", False)],
            ),
            3,
        )

        pandas_result = PandasBackend().execute(pipeline, {"sales": sales_df})
        df_result = _run_on_datafusion(pipeline, {"sales": sales_df})

        tm.assert_frame_equal(
            pandas_result.reset_index(drop=True),
            df_result.reset_index(drop=True),
        )


class TestCompileDecoratorDataFusion:
    """End-to-end: @compilable(backend=DataFusionBackend()) decorator tests."""

    def test_filter_rows(self, sales_df):
        from pandas.jit.compiler import DataFusionBackend

        @compilable(backend=DataFusionBackend())
        def f(df):
            return df[df["price"] > 100]

        result = f(sales_df)
        expected = sales_df[sales_df["price"] > 100]
        tm.assert_frame_equal(
            result.sort_values("id").reset_index(drop=True),
            expected.sort_values("id").reset_index(drop=True),
        )

    def test_groupby_sum(self, sales_df):
        from pandas.jit.compiler import DataFusionBackend

        @compilable(backend=DataFusionBackend())
        def f(df):
            return df.groupby("region")["price"].sum().reset_index()

        result = f(sales_df)
        expected = sales_df.groupby("region")["price"].sum().reset_index()
        tm.assert_frame_equal(
            result.sort_values("region").reset_index(drop=True),
            expected.sort_values("region").reset_index(drop=True),
            check_dtype=False,
        )

    def test_multi_step_pipeline(self, sales_df):
        from pandas.jit.compiler import DataFusionBackend

        @compilable(backend=DataFusionBackend())
        def f(df):
            df = df[df["price"] > 50]
            df["revenue"] = df["price"] * df["quantity"]
            return df.sort_values("revenue", ascending=False).head(3)

        result = f(sales_df)
        expected = sales_df[sales_df["price"] > 50].copy()
        expected["revenue"] = expected["price"] * expected["quantity"]
        expected = expected.sort_values("revenue", ascending=False).head(3)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Clip and abs — end-to-end
# ---------------------------------------------------------------------------


class TestClipAbs:
    @pytest.fixture
    def df(self):
        return DataFrame({"a": [-5, -2, 0, 3, 7], "b": [10, 20, 30, 40, 50]})

    def test_clip_in_pipeline(self, df):
        @compilable
        def f(df):
            return df.clip(lower=0, upper=40)

        result = f(df)
        expected = df.clip(lower=0, upper=40)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_abs_in_pipeline(self, df):
        @compilable
        def f(df):
            df["diff"] = df["a"] - df["b"]
            return df.abs()

        result = f(df)
        expected = df.copy()
        expected["diff"] = expected["a"] - expected["b"]
        expected = expected.abs()
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_series_clip_assign(self, df):
        @compilable
        def f(df):
            df["a_clipped"] = df["a"].clip(lower=-1, upper=4)
            return df

        result = f(df)
        expected = df.copy()
        expected["a_clipped"] = expected["a"].clip(lower=-1, upper=4)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_clip_then_filter(self, df):
        @compilable
        def f(df):
            df = df.clip(lower=0, upper=40)
            return df[df["a"] > 0]

        result = f(df)
        expected = df.clip(lower=0, upper=40)
        expected = expected[expected["a"] > 0]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Diff — end-to-end
# ---------------------------------------------------------------------------


class TestDiff:
    def test_series_diff_pipeline(self, sales_df):
        @compilable
        def f(df):
            df = df.sort_values("id")
            df["price_diff"] = df["price"].diff(1)
            return df

        result = f(sales_df)
        expected = sales_df.sort_values("id").copy()
        expected["price_diff"] = expected["price"].diff(1)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_dataframe_diff_pipeline(self, sales_df):
        @compilable
        def f(df):
            nums = df[["id", "price", "quantity"]].sort_values("id")
            return nums.diff(1)

        result = f(sales_df)
        expected = sales_df[["id", "price", "quantity"]].sort_values("id").diff(1)
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


# ---------------------------------------------------------------------------
# GroupBy cumulative — end-to-end
# ---------------------------------------------------------------------------


class TestGroupByCumulative:
    def test_groupby_cumsum_pipeline(self, sales_df):
        @compilable
        def f(df):
            df["running_price"] = df.groupby("region")["price"].cumsum()
            return df

        result = f(sales_df)
        expected = sales_df.copy()
        expected["running_price"] = expected.groupby("region")["price"].cumsum()
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


# ---------------------------------------------------------------------------
# Complex multi-operation pipelines
# ---------------------------------------------------------------------------


class TestComplexPipelines:
    def test_etl_pipeline(self):
        """ETL: clean -> transform -> aggregate -> sort."""
        df = DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-02",
                        "2024-01-03",
                    ]
                ),
                "category": ["A", "B", "A", "B", "A"],
                "amount": [100.0, np.nan, 300.0, 400.0, 500.0],
                "region": ["  East  ", "West", "East", " West ", "East"],
            }
        )

        @compilable
        def etl(df):
            df["region"] = df["region"].str.strip()
            df = df.dropna(subset=["amount"])
            df["month"] = df["date"].dt.month
            result = df.groupby(["region", "category"]).agg({"amount": "sum"})
            return result.sort_values("amount", ascending=False)

        result = etl(df)
        assert len(result) >= 2
        assert all(r.strip() == r for r in result["region"])

    def test_feature_engineering(self):
        """Feature engineering: diff + pct_change + normalize."""
        df = DataFrame(
            {
                "price": [100.0, 102.0, 101.0, 105.0, 103.0, 108.0],
                "volume": [1000.0, 1100.0, 900.0, 1200.0, 1050.0, 1300.0],
            }
        )

        @compilable
        def features(df):
            df["price_change"] = df["price"].diff(1)
            df["price_pct"] = df["price"].pct_change(1)
            df["vol_ratio"] = df["volume"] / df["volume"].mean()
            return df

        result = features(df)
        expected = df.copy()
        expected["price_change"] = expected["price"].diff(1)
        expected["price_pct"] = expected["price"].pct_change(1)
        expected["vol_ratio"] = expected["volume"] / expected["volume"].mean()
        tm.assert_frame_equal(result, expected, check_dtype=False, atol=1e-10)

    def test_multi_table_join(self):
        """Join 2 tables -> filter -> aggregate."""
        orders = DataFrame(
            {
                "order_id": range(1, 6),
                "customer_id": [1, 2, 1, 3, 2],
                "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
            }
        )
        customers = DataFrame(
            {
                "customer_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        @compilable
        def pipeline(orders, customers):
            merged = orders.merge(customers, on="customer_id")
            high_value = merged[merged["amount"] > 150]
            return (
                high_value.groupby("name")
                .agg({"amount": "sum"})
                .sort_values("amount", ascending=False)
            )

        result = pipeline(orders, customers)
        expected = orders.merge(customers, on="customer_id")
        expected = expected[expected["amount"] > 150]
        expected = (
            expected.groupby("name")["amount"]
            .sum()
            .reset_index()
            .sort_values("amount", ascending=False)
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(
            result.sort_values("name").reset_index(drop=True),
            expected.sort_values("name").reset_index(drop=True),
            check_dtype=False,
        )

    def test_string_cleaning_pipeline(self):
        """String: strip -> lower -> contains filter -> replace."""
        df = DataFrame(
            {"name": ["  Alice  ", " BOB ", "charlie", "  DAVE  ", "  eve  "]}
        )

        @compilable
        def clean(df):
            df["name"] = df["name"].str.strip().str.lower()
            df = df[df["name"].str.contains("e")]
            df["name"] = df["name"].str.replace("e", "E")
            return df

        result = clean(df)
        expected = df.copy()
        expected["name"] = expected["name"].str.strip().str.lower()
        expected = expected[expected["name"].str.contains("e")]
        expected["name"] = expected["name"].str.replace("e", "E")
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_math_normalization(self):
        """Math: sqrt + log -> normalize by mean."""
        df = DataFrame({"x": [1.0, 4.0, 9.0, 16.0, 25.0]})

        @compilable
        def normalize(df):
            df["sqrt_x"] = df["x"].sqrt()
            df["log_x"] = df["x"].log()
            df["norm"] = df["sqrt_x"] / df["sqrt_x"].mean()
            return df

        result = normalize(df)
        expected = df.copy()
        expected["sqrt_x"] = np.sqrt(expected["x"])
        expected["log_x"] = np.log(expected["x"])
        expected["norm"] = expected["sqrt_x"] / expected["sqrt_x"].mean()
        tm.assert_frame_equal(result, expected, check_dtype=False, atol=1e-10)

    def test_cumulative_with_groupby_filter(self):
        """Cumulative sum per group, then filter."""
        df = DataFrame(
            {
                "g": ["a", "b", "a", "b", "a", "b"],
                "v": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )

        @compilable
        def f(df):
            df["cumv"] = df.groupby("g")["v"].cumsum()
            return df[df["cumv"] > 30]

        result = f(df)
        expected = df.copy()
        expected["cumv"] = expected.groupby("g")["v"].cumsum()
        expected = expected[expected["cumv"] > 30]
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_assign_chain(self):
        """5+ chained assigns with mixed operations."""
        df = DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})

        @compilable
        def f(df):
            df["b"] = df["a"] * 2
            df["c"] = df["b"] + 10
            df["d"] = df["c"] / df["a"]
            df["e"] = df["a"].sqrt()
            df["f"] = df["d"].round(1)
            return df

        result = f(df)
        expected = df.copy()
        expected["b"] = expected["a"] * 2
        expected["c"] = expected["b"] + 10
        expected["d"] = expected["c"] / expected["a"]
        expected["e"] = np.sqrt(expected["a"])
        expected["f"] = expected["d"].round(1)
        tm.assert_frame_equal(result, expected, check_dtype=False, atol=1e-10)

    def test_window_ranking(self):
        """Rank within dataset."""
        df = DataFrame(
            {
                "dept": ["A", "A", "A", "B", "B", "B"],
                "salary": [50, 70, 60, 80, 55, 65],
            }
        )

        @compilable
        def f(df):
            df["rnk"] = df["salary"].rank(method="min")
            return df.sort_values("rnk")

        result = f(df)
        expected = df.copy()
        expected["rnk"] = expected["salary"].rank(method="min")
        expected = expected.sort_values("rnk")
        tm.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


# ---------------------------------------------------------------------------
# NaN handling integration
# ---------------------------------------------------------------------------


class TestNaNIntegration:
    def test_nan_propagation_through_pipeline(self):
        df = DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})

        @compilable
        def f(df):
            df["c"] = df["a"] + df["b"]
            return df[df["c"].isna() | (df["c"] > 0)]

        result = f(df)
        assert len(result) == 3

    def test_fillna_then_aggregate(self):
        df = DataFrame({"g": ["a", "a", "b"], "v": [1.0, np.nan, 3.0]})

        @compilable
        def f(df):
            df["v"] = df["v"].fillna(0)
            return df.groupby("g")[["v"]].sum()

        result = f(df)
        result = result.sort_values("g").reset_index(drop=True)
        assert result.iloc[0]["v"] == 1.0
        assert result.iloc[1]["v"] == 3.0

    def test_dropna_then_sort(self):
        df = DataFrame({"x": [3.0, np.nan, 1.0, np.nan, 2.0]})

        @compilable
        def f(df):
            df = df.dropna(subset=["x"])
            return df.sort_values("x")

        result = f(df)
        assert list(result["x"]) == [1.0, 2.0, 3.0]

    def test_nan_in_window(self):
        """Window cumsum skips NaN (treated as 0), unlike pandas."""
        df = DataFrame({"x": [1.0, np.nan, 3.0, 4.0]})

        @compilable
        def f(df):
            df["cs"] = df["x"].cumsum()
            return df

        result = f(df)
        # JIT cumsum uses window sum which skips NaN
        assert result["cs"].iloc[0] == 1.0
        assert result["cs"].iloc[2] == 4.0
        assert result["cs"].iloc[3] == 8.0

    def test_nan_in_deferred_scalar(self):
        df = DataFrame({"x": [1.0, np.nan, 3.0]})

        @compilable
        def f(df):
            df["norm"] = df["x"] / df["x"].mean()
            return df

        result = f(df)
        expected = df.copy()
        expected["norm"] = expected["x"] / expected["x"].mean()
        tm.assert_frame_equal(result, expected, check_dtype=False)


# ---------------------------------------------------------------------------
# Transform Integration Tests (Phase 37)
# ---------------------------------------------------------------------------


class TestTransformIntegration:
    """Integration tests for groupby().transform() and optimization passes."""

    def test_transform_then_filter(self):
        """transform('sum') → filter → sort pipeline."""
        df = DataFrame({"g": ["a", "a", "b", "b", "b"], "v": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            df["group_sum"] = df.groupby("g")["v"].transform("sum")
            df = df[df["group_sum"] > 50]
            return df.sort_values("v")

        result = f(df)
        assert all(result["group_sum"] > 50)
        assert list(result["v"]) == sorted(result["v"])

    def test_transform_normalization(self):
        """Transform sum used in downstream computation."""
        df = DataFrame({"g": ["a", "a", "b", "b"], "v": [10.0, 20.0, 30.0, 40.0]})

        @compilable
        def f(df):
            df["group_sum"] = df.groupby("g")["v"].transform("sum")
            df = df[df["group_sum"] > 50]
            return df

        result = f(df)
        # Group a sum=30 (<50), group b sum=70 (>50)
        assert len(result) == 2
        assert all(result["g"] == "b")

    def test_multi_optimization_pass(self):
        """Pipeline where predicate pushdown + projection pruning both apply."""
        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": ["x", "y", "z", "w", "v"],
            }
        )

        @compilable
        def f(df):
            df["d"] = df["a"] + df["b"]
            df = df[df["a"] > 2]
            return df[["a", "d"]]

        result = f(df)
        assert all(result["a"] > 2)
        assert list(result.columns) == ["a", "d"]

    def test_transform_with_deferred_scalar(self):
        """transform + DeferredScalar composition."""
        df = DataFrame({"g": ["a", "a", "b", "b"], "v": [10.0, 20.0, 30.0, 40.0]})

        @compilable
        def f(df):
            df["group_mean"] = df.groupby("g")["v"].transform("mean")
            df["norm"] = df["v"] / df["v"].mean()
            return df

        result = f(df)
        assert "group_mean" in result.columns
        assert "norm" in result.columns
        overall_mean = df["v"].mean()
        for i in range(len(result)):
            assert abs(result["norm"].iloc[i] - df["v"].iloc[i] / overall_mean) < 1e-10


# ---------------------------------------------------------------------------
# Phase 38-42 integration tests
# ---------------------------------------------------------------------------


class TestPhase38To42Integration:
    def test_select_dtypes_then_aggregate(self):
        """select_dtypes('number') → groupby → sum."""
        df = DataFrame(
            {
                "cat": ["a", "a", "b", "b"],
                "x": [1, 2, 3, 4],
                "y": [10, 20, 30, 40],
                "label": ["p", "q", "r", "s"],
            }
        )

        @compilable
        def f(df):
            numeric = df.select_dtypes(include="number")
            return numeric

        result = f(df)
        assert set(result.columns) == {"x", "y"}
        assert len(result) == 4

    def test_replace_then_groupby(self):
        """replace({0: 1}) → groupby → sum."""
        df = DataFrame({"g": ["a", "a", "b", "b"], "v": [0, 2, 3, 0]})

        @compilable
        def f(df):
            df = df.replace(0, 1)
            return df

        result = f(df)
        assert 0 not in result["v"].values
        assert list(result["v"]) == [1, 2, 3, 1]

    def test_drop_duplicates_in_pipeline(self):
        """filter → drop_duplicates(subset) → sort → head."""
        df = DataFrame(
            {
                "cat": ["a", "a", "b", "b", "c"],
                "val": [10, 20, 30, 40, 50],
            }
        )

        @compilable
        def f(df):
            df = df[df["val"] > 10]
            df = df.drop_duplicates(subset=["cat"])
            df = df.sort_values("val")
            return df.head(2)

        result = f(df)
        assert len(result) == 2

    def test_replace_preserves_types(self):
        """replace doesn't change dtypes for int→int."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        @compilable
        def f(df):
            return df.replace(1, 99)

        result = f(df)
        assert result["a"].iloc[0] == 99
        assert result["b"].dtype == df["b"].dtype

    def test_select_dtypes_empty(self):
        """select_dtypes with no matching columns returns empty DataFrame."""
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        @compilable
        def f(df):
            return df.select_dtypes(include="datetime")

        result = f(df)
        assert len(result.columns) == 0

    def test_df_aggregation_after_filter(self):
        """filter → df.sum(), verify correct results."""
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        @compilable
        def f(df):
            df = df[df["a"] > 2]
            return df.sum()

        result = f(df)
        assert result["a"] == 12  # 3+4+5
        assert result["b"] == 120  # 30+40+50

    def test_string_split_rejoin(self):
        """str.split → extract column → filter."""
        df = DataFrame({"s": ["hello-world", "foo-bar", "test-case"]})

        @compilable
        def f(df):
            parts = df["s"].str.split("-", expand=True)
            return parts

        result = f(df)
        assert result.shape == (3, 2)

    def test_timedelta_filtering(self):
        """Create timedelta column, filter by .dt.days > 1."""
        df = DataFrame(
            {
                "name": ["a", "b", "c"],
                "duration": pd.to_timedelta([1, 3, 0], unit="D"),
            }
        )

        @compilable
        def f(df):
            df["days"] = df["duration"].dt.days
            return df[df["days"] > 1]

        result = f(df)
        assert len(result) == 1
        assert result["name"].iloc[0] == "b"

    def test_full_pipeline_with_new_features(self):
        """Combines df replace, select_dtypes, dedup, head."""
        df = DataFrame(
            {
                "cat": ["a", "a", "b", "b"],
                "x": [0, 2, 3, 0],
                "y": [10, 20, 30, 40],
                "label": ["p", "q", "r", "s"],
            }
        )

        @compilable
        def f(df):
            df = df.replace(0, 1)
            numeric = df.select_dtypes(include="number")
            numeric = numeric.drop_duplicates()
            return numeric

        result = f(df)
        assert set(result.columns) == {"x", "y"}
        assert 0 not in result["x"].values

    def test_filter_groupby_median_rename(self):
        """Filter rows, groupby median, rename columns."""
        df = DataFrame(
            {
                "region": ["east", "east", "west", "west", "east"],
                "sales": [100.0, 200.0, 150.0, 250.0, 300.0],
                "cost": [50.0, 80.0, 60.0, 90.0, 110.0],
            }
        )

        @compilable
        def f(df):
            df = df[df["sales"] > 120]
            result = df.groupby("region").median()
            result = result.rename(columns={"sales": "median_sales"})
            return result

        result = f(df)
        assert "median_sales" in result.columns

    def test_comparison_dropna_to_frame(self):
        """Chain comparison methods, dropna, and to_frame."""
        df = DataFrame({"x": [1.0, float("nan"), 3.0, 4.0, float("nan")]})

        @compilable
        def f(df):
            s = df["x"].dropna()
            big = s.gt(2.0)
            return big.to_frame(name="is_big")

        result = f(df)
        assert "is_big" in result.columns
        assert result["is_big"].sum() == 2

    def test_ffill_pct_change_duplicated(self):
        """Forward fill, pct_change, then check duplicated."""
        df = DataFrame(
            {
                "a": [1.0, float("nan"), 3.0, 3.0],
                "b": [10.0, 20.0, 20.0, 40.0],
            }
        )

        @compilable
        def f(df):
            df = df.ffill()
            pct = df.pct_change()
            return pct

        result = f(df)
        assert len(result) == 4

    def test_groupby_nunique_then_filter(self):
        """Groupby nunique then filter groups with >1 unique."""
        df = DataFrame(
            {
                "g": ["a", "a", "b", "b", "b"],
                "x": [1, 1, 2, 3, 4],
            }
        )

        @compilable
        def f(df):
            counts = df.groupby("g").nunique()
            return counts[counts["x"] > 1]

        result = f(df)
        assert len(result) == 1
        assert result["g"].iloc[0] == "b"

    def test_insert_sort_head(self):
        """Insert column, sort, then take head."""
        df = DataFrame({"name": ["c", "a", "b"], "score": [3, 1, 2]})

        @compilable
        def f(df):
            df.insert(1, "rank", [3, 1, 2])
            df = df.sort_values("rank")
            return df.head(2)

        result = f(df)
        assert len(result) == 2
        assert result["name"].iloc[0] == "a"

    def test_dt_week_groupby_sum(self):
        """Extract week from datetime, groupby week, sum."""
        df = DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-08", "2024-01-09"]
                ),
                "value": [10, 20, 30, 40],
            }
        )

        @compilable
        def f(df):
            df["week"] = df["date"].dt.week
            return df.groupby("week").sum()

        result = f(df)
        assert len(result) >= 1

    def test_groupby_shift_diff_corr(self):
        """GroupBy shift, diff, then compute correlation."""
        df = DataFrame(
            {
                "g": ["a", "a", "a", "b", "b", "b"],
                "x": [1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
                "y": [10.0, 20.0, 40.0, 5.0, 10.0, 20.0],
            }
        )

        @compilable
        def f(df):
            diffs = df.groupby("g").diff()
            return diffs[["x", "y"]].corr()

        result = f(df)
        assert "x" in result.columns

    def test_median_quantile_pipeline(self):
        """Compute median and quantile on filtered data."""
        df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})

        @compilable
        def f(df):
            df = df[df["x"] > 3]
            med = df["x"].median()
            q25 = df["x"].quantile(0.25)
            return DataFrame({"median": [med], "q25": [q25]})

        result = f(df)
        assert result["median"].iloc[0] == 7.0

    def test_strftime_groupby_count(self):
        """Format dates, group by month string, count."""
        df = DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-15", "2024-01-20", "2024-02-10", "2024-02-25"]
                ),
                "value": [1, 2, 3, 4],
            }
        )

        @compilable
        def f(df):
            df["month"] = df["date"].dt.strftime("%Y-%m")
            return df.groupby("month").count()

        result = f(df)
        assert len(result) == 2

    def test_idxmax_join_pipeline(self):
        """Find idxmax, then use result."""
        df = DataFrame(
            {
                "category": ["a", "b", "a", "b"],
                "score": [10.0, 30.0, 50.0, 20.0],
            }
        )

        @compilable
        def f(df):
            best_idx = df["score"].idxmax()
            best_row = df.iloc[[best_idx]]
            return best_row

        result = f(df)
        assert result["score"].iloc[0] == 50.0

    def test_groupby_first_pct_change(self):
        """Get first per group, then pct_change across groups."""
        df = DataFrame(
            {
                "g": ["a", "a", "b", "b", "c", "c"],
                "x": [100.0, 200.0, 50.0, 100.0, 75.0, 150.0],
            }
        )

        @compilable
        def f(df):
            firsts = df.groupby("g").first()
            return firsts

        result = f(df)
        assert len(result) == 3

    def test_combine_first_then_stats(self):
        """Combine two DataFrames, then compute stats."""
        df1 = DataFrame({"x": [1.0, float("nan"), 3.0], "y": [float("nan"), 5.0, 6.0]})
        df2 = DataFrame({"x": [10.0, 20.0, 30.0], "y": [40.0, 50.0, 60.0]})

        @compilable
        def f(df):
            filled = df.combine_first(df2)
            return filled

        result = f(df1)
        assert result["x"].iloc[1] == 20.0
        assert result["y"].iloc[0] == 40.0


class TestPhase53_57Integration:
    """Integration tests combining features from Phases 53-57."""

    def test_groupby_prod_cumcount(self):
        """GroupBy prod + cumcount pipeline."""
        df = DataFrame({"g": ["a", "a", "a", "b", "b"], "x": [2, 3, 4, 5, 6]})

        @compilable
        def f(df):
            df["cc"] = df.groupby("g").cumcount()
            prods = df.groupby("g").prod()
            return prods

        result = f(df)
        vals = dict(zip(result["g"], result["x"], strict=True))
        assert vals["a"] == 24  # 2*3*4
        assert vals["b"] == 30  # 5*6

    def test_series_properties_chain(self):
        """Series hasnans -> filter -> is_unique."""
        df = DataFrame({"x": [1.0, 2.0, float("nan"), 3.0, 3.0]})

        @compilable
        def f(df):
            has_na = df["x"].hasnans
            if has_na:
                clean = df.dropna()
                return clean["x"].is_unique
            return True

        result = f(df)
        assert result is False  # 3.0 appears twice

    def test_rolling_median_expanding_quantile(self):
        """Rolling median + expanding quantile pipeline."""
        df = DataFrame({"x": [1.0, 5.0, 3.0, 7.0, 2.0, 8.0]})

        @compilable
        def f(df):
            _ = df.rolling(3).median()
            eq = df.expanding().quantile(0.75)
            return eq

        result = f(df)
        assert len(result) == 6

    def test_df_filter_reindex_equals(self):
        """DataFrame filter + reindex + equals."""
        df = DataFrame({"col_a": [1, 2], "col_b": [3, 4], "other": [5, 6]})

        @compilable
        def f(df):
            filtered = df.filter(regex=r"^col_")
            reindexed = df.reindex(columns=["col_a", "col_b"])
            return filtered.equals(reindexed)

        assert f(df) is True

    def test_gbs_shift_diff_first(self):
        """GroupBySeries shift + diff + first."""
        df = DataFrame(
            {
                "g": ["a", "a", "a", "b", "b", "b"],
                "x": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            }
        )

        @compilable
        def f(df):
            df["d"] = df.groupby("g")["x"].diff(1)
            firsts = df.groupby("g")["x"].first().to_frame()
            return firsts

        result = f(df)
        vals = sorted(result["x"].tolist())
        assert vals == [10.0, 100.0]

    def test_full_pipeline(self):
        """Combine groupby extras, series props, rolling, df utils."""
        df = DataFrame(
            {
                "category": ["A", "A", "A", "B", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )

        @compilable
        def f(df):
            filtered = df.filter(items=["category", "value"])
            result = filtered.groupby("category").prod()
            return result

        result = f(df)
        vals = dict(zip(result["category"], result["value"], strict=True))
        assert vals["A"] == 6000.0  # 10*20*30
        assert vals["B"] == 120000.0  # 40*50*60


class TestPhase58_62Integration:
    """Integration tests combining features from Phases 58-62."""

    def test_groupby_multi_shift_diff_first(self):
        """GroupByMulti shift + diff + first pipeline."""
        df = DataFrame(
            {
                "g": ["a", "a", "a", "b", "b", "b"],
                "x": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        @compilable
        def f(df):
            multi = df.groupby("g")[["x", "y"]]
            _ = multi.shift(1)
            firsts = multi.first()
            return firsts

        result = f(df)
        vals_x = sorted(result["x"].tolist())
        assert vals_x == [10.0, 100.0]

    def test_dataframe_arithmetic_any_all(self):
        """DataFrame arithmetic + any/all pipeline."""
        df = DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

        @compilable
        def f(df):
            scaled = df * 2
            above = scaled["x"] > 4
            has_any = bool(above.any())
            return has_any

        result = f(df)
        assert result is True

    def test_groupby_rank_rolling_agg(self):
        """GroupBy rank + rolling agg pipeline."""
        df = DataFrame(
            {
                "g": ["a", "a", "a", "b", "b", "b"],
                "x": [30.0, 10.0, 20.0, 60.0, 40.0, 50.0],
            }
        )

        @compilable
        def f(df):
            ranked = df.groupby("g").rank(method="min")
            rolled = ranked.rolling(2).agg("mean")
            return rolled

        result = f(df)
        assert len(result) == 6

    def test_series_conversion_conditional(self):
        """Series conversion in conditional pipeline."""
        df = DataFrame({"x": [1, 2, 3, 4, 5]})

        @compilable
        def f(df):
            arr = df["x"].to_numpy()
            total = arr.sum()
            if total > 10:
                d = df["x"].to_dict()
                return len(d)
            return 0

        result = f(df)
        assert result == 5  # sum=15 > 10, so returns len(dict)=5

    def test_dataframe_iter_filter(self):
        """DataFrame iter + filter pipeline."""
        df = DataFrame({"col_a": [1, 2], "col_b": [3, 4], "other": [5, 6]})

        @compilable
        def f(df):
            _ = list(df)
            filtered = df.filter(regex=r"^col_")
            agged = filtered.agg("sum")
            return agged

        result = f(df)
        assert result["col_a"] == 3
        assert result["col_b"] == 7

    def test_full_pipeline_phases58_62(self):
        """Full pipeline combining arithmetic, map, groupby multi, iter."""
        df = DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
                "code": [1, 2, 1, 2],
            }
        )

        @compilable
        def f(df):
            _ = df * 1  # arithmetic (makes a copy)
            _ = df["code"].map({1: 100, 2: 200})  # series map dict
            _ = list(df)  # DataFrame iter
            totals = df.agg({"value": "sum"})
            return totals

        result = f(df)
        assert result["value"] == 100  # 10+20+30+40
