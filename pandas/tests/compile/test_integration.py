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

from pandas.compile.compiler import (
    PandasBackend,
    SubstraitCompiler,
    infer_schema,
)
from pandas.compile.ir import (
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
from pandas.compile.jit import (
    Tracer,
    compile,
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
# JIT decorator → Acero: end-to-end from @compile to Substrait execution
# ---------------------------------------------------------------------------


class TestJitToAcero:
    def test_jit_plan_runs_on_acero(self, sales_df):
        """Trace with @compile, extract Substrait plan, run on Acero."""

        @compile(backend=PandasBackend())
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
            from pandas.compile.compiler import CompiledSegment

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
# User scenarios: @compile decorator — how users actually use the API
# ---------------------------------------------------------------------------


class TestCompileDecorator:
    """End-to-end tests using @compile with the default AceroBackend."""

    def test_filter_rows(self, sales_df):
        @compile
        def get_expensive(df):
            return df[df["price"] > 100]

        result = get_expensive(sales_df)
        expected = sales_df[sales_df["price"] > 100].reset_index(drop=True)
        tm.assert_frame_equal(
            result.sort_values("id").reset_index(drop=True),
            expected.sort_values("id").reset_index(drop=True),
        )

    def test_select_columns(self, sales_df):
        @compile
        def select_cols(df):
            return df[["id", "region", "price"]]

        result = select_cols(sales_df)
        assert list(result.columns) == ["id", "region", "price"]
        assert len(result) == len(sales_df)

    def test_add_computed_column(self, sales_df):
        @compile
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
        @compile
        def top_expensive(df):
            expensive = df[df["price"] > 100]
            return expensive.sort_values("price", ascending=False)

        result = top_expensive(sales_df)
        assert all(result["price"] > 100)
        assert list(result["price"]) == sorted(result["price"], reverse=True)

    def test_sort_and_head(self, sales_df):
        @compile
        def top_3(df):
            return df.sort_values("price", ascending=False).head(3)

        result = top_3(sales_df)
        assert len(result) == 3
        assert list(result["price"]) == sorted(result["price"], reverse=True)

    def test_groupby_sum(self, sales_df):
        @compile
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
        @compile
        def product_counts(df):
            return df.groupby("product").count()

        result = product_counts(sales_df)
        assert "product" in result.columns
        assert len(result) == 2  # products A and B

    def test_groupby_std(self, sales_df):
        @compile
        def f(df):
            return df.groupby("region").std()

        result = f(sales_df)
        assert "region" in result.columns
        # Std values should be non-negative floats
        result_sorted = result.sort_values("region").reset_index(drop=True)
        assert all(result_sorted["price"] >= 0)
        assert result_sorted["price"].dtype == np.float64

    def test_groupby_var(self, sales_df):
        @compile
        def f(df):
            return df.groupby("region").var()

        result = f(sales_df)
        assert "region" in result.columns
        # Var values should be non-negative floats
        result_sorted = result.sort_values("region").reset_index(drop=True)
        assert all(result_sorted["price"] >= 0)
        assert result_sorted["price"].dtype == np.float64

    def test_groupby_series_std(self, sales_df):
        @compile
        def f(df):
            return df.groupby("region")["price"].std()

        result = f(sales_df)
        assert "price" in result.columns
        assert "region" in result.columns

    def test_groupby_first(self, sales_df):
        @compile
        def f(df):
            return df.sort_values("price").groupby("region").first()

        result = f(sales_df)
        assert "region" in result.columns
        assert len(result) == 2

    def test_groupby_last(self, sales_df):
        @compile
        def f(df):
            return df.sort_values("price").groupby("region").last()

        result = f(sales_df)
        assert "region" in result.columns
        assert len(result) == 2

    def test_series_std(self, sales_df):
        @compile
        def f(df):
            s = df["price"].std()
            return df[df["price"] > s]

        result = f(sales_df)
        assert all(result["price"] > sales_df["price"].std())

    def test_series_var(self, sales_df):
        @compile
        def f(df):
            v = df["price"].var()
            return df[df["price"] > v]

        result = f(sales_df)
        assert all(result["price"] > sales_df["price"].var())

    def test_filter_then_groupby(self, sales_df):
        @compile
        def expensive_by_region(df):
            expensive = df[df["price"] > 100]
            return expensive.groupby("region").sum()

        result = expensive_by_region(sales_df)
        assert "region" in result.columns
        # All prices in the input should be > 100

    def test_multi_step_pipeline(self, sales_df):
        @compile
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
        @compile
        def top_by_price(df):
            return df.nlargest(3, "price")

        result = top_by_price(sales_df)
        assert len(result) == 3
        assert list(result["price"]) == sorted(result["price"], reverse=True)

    def test_drop_columns(self, sales_df):
        @compile
        def drop_qty(df):
            return df.drop(columns=["quantity"])

        result = drop_qty(sales_df)
        assert "quantity" not in result.columns
        assert "price" in result.columns

    def test_query_string(self, sales_df):
        @compile
        def query_filter(df):
            return df.query("price > 100")

        result = query_filter(sales_df)
        assert all(result["price"] > 100)

    def test_rename_columns(self, sales_df):
        @compile
        def rename_cols(df):
            return df.rename(columns={"price": "cost", "quantity": "qty"})

        result = rename_cols(sales_df)
        assert "cost" in result.columns
        assert "qty" in result.columns
        assert "price" not in result.columns
        assert "quantity" not in result.columns

    def test_assign(self, sales_df):
        @compile
        def with_discount(df):
            return df.assign(discount=df["price"] * 0.1)

        result = with_discount(sales_df)
        assert "discount" in result.columns
        r = result.sort_values("id").reset_index(drop=True)
        expected = sales_df.sort_values("id").reset_index(drop=True)["price"] * 0.1
        tm.assert_numpy_array_equal(r["discount"].values, expected.values)

    def test_nsmallest(self, sales_df):
        @compile
        def bottom_by_price(df):
            return df.nsmallest(3, "price")

        result = bottom_by_price(sales_df)
        assert len(result) == 3
        assert list(result["price"]) == sorted(result["price"])

    def test_merge(self, sales_df, regions_df):
        @compile
        def with_manager(df, regions):
            return df.merge(regions, on="region")

        result = with_manager(sales_df, regions_df)
        assert "manager" in result.columns
        assert len(result) == len(sales_df)

    def test_isin_filter(self, sales_df):
        @compile
        def east_west(df):
            return df[df["product"].isin(["A"])]

        result = east_west(sales_df)
        assert all(result["product"] == "A")

    def test_dropna(self, nullable_df):
        @compile
        def drop_nulls(df):
            return df.dropna(subset=["score"])

        result = drop_nulls(nullable_df)
        assert result["score"].notna().all()
        assert len(result) == 3

    def test_isna_filter(self, nullable_df):
        @compile
        def get_missing(df):
            return df[df["score"].isna()]

        result = get_missing(nullable_df)
        assert len(result) == 2
        assert result["score"].isna().all()

    def test_notna_filter(self, nullable_df):
        @compile
        def get_present(df):
            return df[df["score"].notna()]

        result = get_present(nullable_df)
        assert len(result) == 3
        assert result["score"].notna().all()

    def test_fillna_scalar(self, nullable_df):
        @compile
        def fill_zeros(df):
            df["score"] = df["score"].fillna(0.0)
            return df

        result = fill_zeros(nullable_df)
        assert result["score"].notna().all()
        filled = result.sort_values("id").reset_index(drop=True)
        assert filled.loc[1, "score"] == 0.0
        assert filled.loc[3, "score"] == 0.0

    def test_fillna_dataframe(self, nullable_df):
        @compile
        def fill_df(df):
            return df.fillna({"score": -1.0})

        result = fill_df(nullable_df)
        assert result["score"].notna().all()
        filled = result.sort_values("id").reset_index(drop=True)
        assert filled.loc[1, "score"] == -1.0

    def test_series_abs(self, sales_df):
        @compile
        def price_dist(df):
            df["dist"] = (df["price"] - 200.0).abs()
            return df

        result = price_dist(sales_df)
        assert "dist" in result.columns
        assert all(result["dist"] >= 0)

    def test_series_negation(self, sales_df):
        @compile
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
        @compile
        def mid_range(df):
            return df[df["price"].between(100.0, 200.0)]

        result = mid_range(sales_df)
        assert all(result["price"] >= 100)
        assert all(result["price"] <= 200)

    def test_compound_boolean_filter(self, sales_df):
        @compile
        def compound(df):
            return df[(df["price"] > 100) & (df["region"] == "East")]

        result = compound(sales_df)
        assert all(result["price"] > 100)
        assert all(result["region"] == "East")

    def test_cache_reuse(self, sales_df):
        @compile
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
        @compile
        def f(df):
            return df[df["price"] > 100]

        plan_str = f.explain(sales_df)
        assert "ExecutionPlan" in plan_str
        assert "COMPILED" in plan_str

    def test_cache_invalidation_on_scalar_change(self, sales_df):
        """Changing a scalar parameter must re-trace, not reuse stale plan."""

        @compile
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

        @compile
        def f(df, threshold):
            return df[df["price"] > threshold]

        f(sales_df, 100)
        f(sales_df, 100)
        assert len(f._cached_plans) == 1

    def test_cache_invalidation_on_kwarg_change(self, sales_df):
        """Changing a keyword argument must re-trace."""

        @compile
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

        @compile
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

        @compile
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

        @compile
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

        @compile
        def with_target(df_orders, df_targets):
            return df_orders.merge(df_targets, on=["year", "region"])

        result = with_target(orders, targets)
        assert "target" in result.columns
        assert len(result) == 2
        assert set(result["revenue"]) == {100, 300}

    def test_concat_then_filter(self):
        a = DataFrame({"id": [1, 2], "val": [10, 20]})
        b = DataFrame({"id": [3, 4], "val": [30, 40]})

        @compile
        def stacked_and_filtered(df1, df2):
            combined = pd.concat([df1, df2])
            return combined[combined["val"] > 15]

        result = stacked_and_filtered(a, b)
        assert len(result) == 3
        assert all(result["val"] > 15)

    def test_filter_then_iloc_slice(self):
        df = DataFrame({"id": range(20), "val": range(20)})

        @compile
        def top_filtered(df):
            big = df[df["val"] >= 5]
            return big.iloc[:3]

        result = top_filtered(df)
        assert len(result) == 3
        assert list(result["id"]) == [5, 6, 7]

    def test_sort_then_iloc_offset(self):
        df = DataFrame({"id": [3, 1, 4, 1, 5, 9], "val": [30, 10, 40, 10, 50, 90]})

        @compile
        def middle_sorted(df):
            sorted_df = df.sort_values("val")
            return sorted_df.iloc[2:4]

        result = middle_sorted(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# User scenarios: @compile with graph breaks
# ---------------------------------------------------------------------------


class TestCompileGraphBreaks:
    """Tests for @compile with operations that force materialization."""

    def test_len_graph_break(self, sales_df):
        @compile
        def f(df):
            filtered = df[df["price"] > 100]
            if len(filtered) > 0:
                return filtered
            return df

        result = f(sales_df)
        assert all(result["price"] > 100)

    def test_shape_graph_break(self, sales_df):
        @compile
        def f(df):
            filtered = df[df["price"] > 100]
            n_rows, _ = filtered.shape
            return filtered.head(min(n_rows, 2))

        result = f(sales_df)
        assert len(result) == 2

    def test_iterrows_graph_break(self, sales_df):
        @compile
        def f(df):
            sorted_df = df.sort_values("price", ascending=False)
            top = sorted_df.head(2)
            ids = []
            for _, row in top.iterrows():
                ids.append(row["id"])
            return top

        result = f(sales_df)
        assert len(result) == 2

    def test_drop_duplicates_graph_break(self, sales_df):
        @compile
        def f(df):
            return df[["region", "product"]].drop_duplicates()

        result = f(sales_df)
        assert len(result) <= len(sales_df)
        # No duplicate region+product pairs
        assert not result.duplicated().any()


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
# User scenarios: top-level compile access
# ---------------------------------------------------------------------------


class TestPdCompileScenarios:
    """End-to-end tests using compile as users would."""

    def test_pd_compile_filter(self, sales_df):
        @compile
        def f(df):
            return df[df["price"] > 100]

        result = f(sales_df)
        assert all(result["price"] > 100)

    def test_pd_compile_pipeline(self, sales_df):
        @compile
        def pipeline(df):
            df["revenue"] = df["price"] * df["quantity"]
            return df.sort_values("revenue", ascending=False).head(3)

        result = pipeline(sales_df)
        assert len(result) == 3
        assert "revenue" in result.columns

    def test_pd_compile_groupby(self, sales_df):
        @compile
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
        @compile
        def f(df):
            return df[df["price"] > 100]

        plans = f.to_substrait(sales_df)
        assert len(plans) >= 1
        # Plan is a protobuf with required fields
        plan = plans[0]
        assert plan.version.major_number >= 0
        assert len(plan.relations) >= 1

    def test_compiled_function_to_substrait_serializes(self, sales_df):
        @compile
        def f(df):
            return df[df["price"] > 100]

        plans = f.to_substrait(sales_df)
        plan_bytes = plans[0].SerializeToString()
        assert isinstance(plan_bytes, bytes)
        assert len(plan_bytes) > 0

    def test_compiled_function_to_substrait_json(self, sales_df):
        @compile
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

        @compile
        def f(df):
            filtered = df[df["price"] > 100]
            n = len(filtered)  # graph break
            return filtered.head(min(n, 3))

        plans = f.to_substrait(sales_df)
        # Should have at least one plan from the compiled segments
        assert len(plans) >= 1

    def test_substrait_plan_runs_on_acero(self, sales_df):
        """Exported Substrait plan can be executed on Acero."""

        @compile
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
        @compile
        def f(df):
            return df[df["ts"].dt.year == 2024]

        result = f(datetime_df)
        expected = datetime_df[datetime_df["ts"].dt.year == 2024]
        assert len(result) == len(expected)

    def test_dt_month_add_column(self, datetime_df):
        @compile
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
        @compile(backend=PandasBackend())
        def f(df):
            df["dow"] = df["ts"].dt.dayofweek
            return df

        result = f(datetime_df)
        assert "dow" in result.columns
        assert all(result["dow"] >= 0)
        assert all(result["dow"] <= 6)

    def test_dt_quarter(self, datetime_df):
        @compile(backend=PandasBackend())
        def f(df):
            df["q"] = df["ts"].dt.quarter
            return df

        result = f(datetime_df)
        assert "q" in result.columns
        assert all(result["q"] >= 1)
        assert all(result["q"] <= 4)

    def test_dt_hour(self, datetime_df):
        @compile(backend=PandasBackend())
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
        @compile
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
        @compile(backend=PandasBackend())
        def f(df):
            df["upper_name"] = df["name"].str.upper()
            return df

        result = f(string_df)
        assert all(result["upper_name"] == string_df["name"].str.upper())

    def test_str_lower(self, string_df):
        @compile(backend=PandasBackend())
        def f(df):
            df["lower_name"] = df["name"].str.lower()
            return df

        result = f(string_df)
        assert all(result["lower_name"] == string_df["name"].str.lower())

    def test_str_contains_filter(self, string_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["name"].str.contains("li", regex=False)]

        result = f(string_df)
        assert len(result) == 2  # Alice, Charlie

    def test_str_startswith(self, string_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["name"].str.startswith("A")]

        result = f(string_df)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_str_endswith(self, string_df):
        @compile(backend=PandasBackend())
        def f(df):
            return df[df["name"].str.endswith("e")]

        result = f(string_df)
        assert all(r.endswith("e") for r in result["name"])

    def test_str_strip(self, string_df):
        @compile(backend=PandasBackend())
        def f(df):
            df["clean_city"] = df["city"].str.strip()
            return df

        result = f(string_df)
        assert result.iloc[0]["clean_city"] == "New York"

    def test_str_len(self, string_df):
        @compile(backend=PandasBackend())
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
        @compile(backend=PandasBackend())
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
        @compile
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
        @compile
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
        @compile
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

        @compile
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


class TestExpanding:
    def test_expanding_sum(self, sales_df):
        @compile
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
        @compile
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


class TestPivotTable:
    def test_pivot_table(self, sales_df):
        @compile
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

        @compile
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
        @compile
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
        @compile
        def f(df):
            return df[["id", "price"]].astype({"price": "int64"})

        result = f(sales_df)
        assert result["price"].dtype == np.int64

    def test_astype_then_filter(self, sales_df):
        """astype graph break, then continue tracing."""

        @compile
        def f(df):
            casted = df[["id", "price"]].astype({"price": "int64"})
            return casted[casted["price"] > 100]

        result = f(sales_df)
        assert result["price"].dtype == np.int64
        assert all(result["price"] > 100)


class TestWhereMask:
    def test_where_dataframe(self, sales_df):
        @compile
        def f(df):
            return df[["id", "price"]].where(sales_df[["id", "price"]] > 100, other=-1)

        result = f(sales_df)
        assert len(result) == len(sales_df)
        # Values ≤ 100 should be -1 (for price column)
        # id column: values > 100 are kept, rest are -1
        # Since id values are 1-6, all ≤ 100, so all id = -1
        # price: 100 → -1, 250 → 250, ...

    def test_mask_dataframe(self, sales_df):
        @compile
        def f(df):
            return df[["id", "price"]].mask(sales_df[["id", "price"]] > 100, other=0)

        result = f(sales_df)
        assert len(result) == len(sales_df)

    def test_where_with_pandas_condition(self):
        """where with a plain pandas boolean condition (not traced)."""
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        cond = df > 2

        @compile
        def f(d):
            return d.where(cond, other=-1)

        result = f(df)
        expected = df.where(cond, other=-1)
        tm.assert_frame_equal(result, expected)

    def test_where_traced_condition(self, sales_df):
        """where with a traced boolean series builds IR (no graph break)."""

        @compile(backend=PandasBackend())
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

        @compile(backend=PandasBackend())
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
    """End-to-end: @compile(backend=DataFusionBackend()) decorator tests."""

    def test_filter_rows(self, sales_df):
        from pandas.compile.compiler import DataFusionBackend

        @compile(backend=DataFusionBackend())
        def f(df):
            return df[df["price"] > 100]

        result = f(sales_df)
        expected = sales_df[sales_df["price"] > 100]
        tm.assert_frame_equal(
            result.sort_values("id").reset_index(drop=True),
            expected.sort_values("id").reset_index(drop=True),
        )

    def test_groupby_sum(self, sales_df):
        from pandas.compile.compiler import DataFusionBackend

        @compile(backend=DataFusionBackend())
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
        from pandas.compile.compiler import DataFusionBackend

        @compile(backend=DataFusionBackend())
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
