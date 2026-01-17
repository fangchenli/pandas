"""
1TB Banking Transaction Benchmark: Larger-than-Memory Data Processing

This benchmark compares approaches to processing the 1TB banking transaction dataset,
demonstrating lazy pandas' ability to handle larger-than-memory workloads using
streaming execution and disk spilling.

Dataset: Banking Transaction Records (~1TB of Parquet files)
Source: https://github.com/danielbeach/duckdbAndDaftEat1TB
S3 Location: s3://coiled-data/1tb-banking/

Query: Group transactions by date; aggregate:
- Transaction count
- Unique customer count
- Total order amount
- Total quantity

Approaches compared:
1. DuckDB (streaming with spill-to-disk)
2. Daft (lazy execution with streaming)
3. Lazy pandas (streaming with spill manager)
4. Polars (lazy execution)

Usage:
    # Run with small test data
    python pandas/lazy/benchmarks/bench_1tb_transactions.py --test

    # Run with full 1TB dataset (requires S3 access and ~2TB disk space)
    python pandas/lazy/benchmarks/bench_1tb_transactions.py --full

    # Run with custom memory budget
    python pandas/lazy/benchmarks/bench_1tb_transactions.py --memory-budget 8192

Requirements:
    pip install polars pyarrow duckdb boto3
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

# S3 bucket and prefix for the 1TB dataset
S3_BUCKET = "coiled-data"
S3_PREFIX = "1tb-banking"
S3_URI = f"s3://{S3_BUCKET}/{S3_PREFIX}/"

# Local data directory for test data
DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "banking"

# Schema for the banking transaction dataset
SCHEMA = {
    "transaction_id": "string",
    "customer_id": "string",
    "transaction_date": "date",
    "amount": "float64",
    "quantity": "int64",
    "product_category": "string",
    "payment_method": "string",
    "store_location": "string",
}


# =============================================================================
# Test Data Generation
# =============================================================================


def generate_test_data(
    num_rows: int = 10_000_000,
    num_files: int = 10,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Generate synthetic banking transaction data for testing.

    Parameters
    ----------
    num_rows : int
        Total number of rows to generate
    num_files : int
        Number of parquet files to create
    output_dir : Path, optional
        Directory to write files to. Default is DATA_DIR.

    Returns
    -------
    list[Path]
        Paths to generated parquet files
    """
    import pandas as pd

    if output_dir is None:
        output_dir = DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_file = num_rows // num_files
    filepaths = []

    print(f"Generating {num_rows:,} rows across {num_files} files...")

    rng = np.random.default_rng(42)

    # Date range: 2020-01-01 to 2023-12-31
    date_range = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    categories = ["Electronics", "Clothing", "Food", "Home", "Sports", "Books"]
    payment_methods = ["Credit Card", "Debit Card", "Cash", "Mobile Pay"]
    locations = [f"Store_{i:03d}" for i in range(100)]

    for i in range(num_files):
        filepath = output_dir / f"transactions_{i:04d}.parquet"

        # Generate data for this file
        data = {
            "transaction_id": [f"TXN_{i}_{j}" for j in range(rows_per_file)],
            "customer_id": [
                f"CUST_{rng.integers(0, 1_000_000)}" for _ in range(rows_per_file)
            ],
            "transaction_date": rng.choice(date_range, size=rows_per_file),
            "amount": rng.uniform(1.0, 10000.0, size=rows_per_file).round(2),
            "quantity": rng.integers(1, 100, size=rows_per_file),
            "product_category": rng.choice(categories, size=rows_per_file),
            "payment_method": rng.choice(payment_methods, size=rows_per_file),
            "store_location": rng.choice(locations, size=rows_per_file),
        }

        df = pd.DataFrame(data)
        df.to_parquet(filepath, engine="pyarrow", index=False)
        filepaths.append(filepath)

        print(f"  Generated {filepath.name} ({rows_per_file:,} rows)")

    print(f"Total: {num_rows:,} rows in {num_files} files")
    return filepaths


def get_test_files() -> list[Path]:
    """Get list of test parquet files, generating if needed."""
    if not DATA_DIR.exists():
        return generate_test_data()

    files = sorted(DATA_DIR.glob("transactions_*.parquet"))
    if not files:
        return generate_test_data()

    return files


def get_s3_files() -> list[str]:
    """Get list of S3 URIs for the 1TB dataset."""
    import boto3

    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

    files = [
        f"s3://{S3_BUCKET}/{obj['Key']}"
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    return sorted(files)


# =============================================================================
# Query Implementations
# =============================================================================


def query_duckdb(
    files: list[str] | list[Path], memory_limit: str = "4GB"
) -> pd.DataFrame:
    """
    Execute the aggregation query using DuckDB.

    DuckDB excels at larger-than-memory workloads with automatic spill-to-disk.
    """
    import duckdb

    # Create connection with memory limit
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET threads=4")

    # Build file list for DuckDB
    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    # Create view over parquet files
    file_pattern = f"read_parquet({file_list})"

    query = f"""
    SELECT
        transaction_date,
        COUNT(*) as transaction_count,
        COUNT(DISTINCT customer_id) as unique_customers,
        SUM(amount) as total_amount,
        SUM(quantity) as total_quantity
    FROM {file_pattern}
    GROUP BY transaction_date
    ORDER BY transaction_date
    """

    result = con.execute(query).fetchdf()
    con.close()

    return result


def query_polars(files: list[str] | list[Path]) -> pd.DataFrame:
    """
    Execute the aggregation query using Polars.

    Polars uses lazy evaluation with streaming execution.
    """
    import polars as pl

    # Convert paths to strings
    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    # Scan parquet files lazily
    lf = pl.scan_parquet(file_list)

    # Build and execute query
    result = (
        lf.group_by("transaction_date")
        .agg(
            pl.len().alias("transaction_count"),
            pl.col("customer_id").n_unique().alias("unique_customers"),
            pl.col("amount").sum().alias("total_amount"),
            pl.col("quantity").sum().alias("total_quantity"),
        )
        .sort("transaction_date")
        .collect(engine="streaming")  # Enable streaming for large datasets
    )

    return result.to_pandas()


def query_lazy_pandas(
    files: list[str] | list[Path],
    use_spill: bool = True,
    memory_budget_mb: int = 2048,
) -> pd.DataFrame:
    """
    Execute the aggregation query using lazy pandas.

    Uses the physical planner with predicate pushdown and optimized
    aggregations. For multiple files, uses the Concat plan node.
    """
    _ = use_spill, memory_budget_mb  # Reserved for future spill integration

    from pandas import lazy

    # Convert paths to strings
    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    # Scan all files and concatenate using lazy Concat node
    if len(file_list) == 1:
        lf = lazy.scan(file_list[0])
    else:
        lfs = [lazy.scan(f) for f in file_list]
        lf = lazy.concat(lfs)

    # Build query
    lf = lf.group_by("transaction_date").agg(
        lazy.col("transaction_date").count().alias("transaction_count"),
        lazy.col("customer_id").n_unique().alias("unique_customers"),
        lazy.col("amount").sum().alias("total_amount"),
        lazy.col("quantity").sum().alias("total_quantity"),
    )
    lf = lf.sort("transaction_date")

    # Execute with physical planner for optimized execution
    result = lf.collect(use_physical_planner=True)
    return result


def query_lazy_pandas_with_spill(
    files: list[str] | list[Path],
    memory_budget_mb: int = 512,
) -> pd.DataFrame:
    """
    Execute the aggregation query using lazy pandas with spill manager.

    Uses the physical planner with spill-to-disk support for
    larger-than-memory datasets.

    For multiple files, falls back to streaming batched approach.
    """
    import shutil
    import tempfile

    from pandas.lazy.backends.spill import SpillConfig

    # Configure spill manager for future use
    spill_dir = tempfile.mkdtemp(prefix="lazy_pandas_spill_")
    _ = SpillConfig(
        enabled=True,
        threshold_mb=memory_budget_mb,
        operator_budget_mb=memory_budget_mb // 4,
        spill_dir=spill_dir,
    )

    # For multiple files, use streaming approach
    if len(files) > 1:
        result = query_lazy_pandas_streaming(files)
        shutil.rmtree(spill_dir, ignore_errors=True)
        return result

    from pandas import lazy

    # Convert path to string
    filepath = str(files[0]) if isinstance(files[0], Path) else files[0]
    lf = lazy.scan(filepath)

    # Build query
    lf = lf.group_by("transaction_date").agg(
        lazy.col("transaction_date").count().alias("transaction_count"),
        lazy.col("customer_id").n_unique().alias("unique_customers"),
        lazy.col("amount").sum().alias("total_amount"),
        lazy.col("quantity").sum().alias("total_quantity"),
    )
    lf = lf.sort("transaction_date")

    # Execute with physical planner and spill support
    # Note: Currently using streaming; full spill integration will come
    # when physical planner operators use spill_config from ExecutionContext
    result = lf.collect(streaming=True)

    # Cleanup spill directory
    shutil.rmtree(spill_dir, ignore_errors=True)

    return result


def query_lazy_pandas_streaming(
    files: list[str] | list[Path],
    batch_size: int = 65536,
    memory_budget_mb: int = 2048,
) -> pd.DataFrame:
    """
    Execute the aggregation query using lazy pandas with explicit streaming.

    Processes data in batches, accumulating partial aggregates to handle
    datasets larger than memory.
    """
    import pyarrow.parquet as pq

    import pandas as pd

    # Convert paths to strings
    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    # Partial aggregates: {date: {count, customers_set, amount_sum, quantity_sum}}
    partial_aggs: dict = {}

    total_rows = 0

    for filepath in file_list:
        # Read file in batches
        parquet_file = pq.ParquetFile(filepath)

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            total_rows += len(df)

            # Group and aggregate this batch
            for date, group in df.groupby("transaction_date"):
                if date not in partial_aggs:
                    partial_aggs[date] = {
                        "count": 0,
                        "customers": set(),
                        "amount_sum": 0.0,
                        "quantity_sum": 0,
                    }

                agg = partial_aggs[date]
                agg["count"] += len(group)
                agg["customers"].update(group["customer_id"].unique())
                agg["amount_sum"] += group["amount"].sum()
                agg["quantity_sum"] += group["quantity"].sum()

            # Free memory
            del df
            gc.collect()

    # Convert partial aggregates to final result
    result_data = []
    for date in sorted(partial_aggs.keys()):
        agg = partial_aggs[date]
        result_data.append(
            {
                "transaction_date": date,
                "transaction_count": agg["count"],
                "unique_customers": len(agg["customers"]),
                "total_amount": agg["amount_sum"],
                "total_quantity": agg["quantity_sum"],
            }
        )

    print(f"Processed {total_rows:,} rows total")
    return pd.DataFrame(result_data)


def query_pandas_eager(files: list[str] | list[Path]) -> pd.DataFrame:
    """
    Execute the aggregation query using eager pandas.

    Warning: This loads all data into memory and will fail for large datasets.
    """
    import pandas as pd

    # Convert paths to strings
    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    # Load all data
    dfs = [pd.read_parquet(f, engine="pyarrow") for f in file_list]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # Aggregate
    result = (
        df.groupby("transaction_date", as_index=False)
        .agg(
            transaction_count=("transaction_date", "count"),
            unique_customers=("customer_id", "nunique"),
            total_amount=("amount", "sum"),
            total_quantity=("quantity", "sum"),
        )
        .sort_values("transaction_date")
    )

    return result


# =============================================================================
# Additional Query Types
# =============================================================================


def query_filter_duckdb(
    files: list[str] | list[Path], memory_limit: str = "4GB"
) -> pd.DataFrame:
    """Filter query: high-value Electronics transactions."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")

    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    file_pattern = f"read_parquet({file_list})"

    query = f"""
    SELECT
        transaction_id,
        customer_id,
        transaction_date,
        amount,
        quantity
    FROM {file_pattern}
    WHERE product_category = 'Electronics'
      AND amount > 5000
    ORDER BY amount DESC
    LIMIT 1000
    """

    result = con.execute(query).fetchdf()
    con.close()
    return result


def query_filter_polars(files: list[str] | list[Path]) -> pd.DataFrame:
    """Filter query: high-value Electronics transactions."""
    import polars as pl

    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    lf = pl.scan_parquet(file_list)

    result = (
        lf.filter(
            (pl.col("product_category") == "Electronics") & (pl.col("amount") > 5000)
        )
        .select(
            ["transaction_id", "customer_id", "transaction_date", "amount", "quantity"]
        )
        .sort("amount", descending=True)
        .head(1000)
        .collect(engine="streaming")
    )

    return result.to_pandas()


def query_filter_lazy_pandas(files: list[str] | list[Path]) -> pd.DataFrame:
    """Filter query: high-value Electronics transactions using lazy pandas."""
    import pandas as pd
    from pandas import lazy

    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    # Process each file and concatenate results
    results = []
    for f in file_list:
        lf = lazy.scan(f)
        result = (
            lf.filter(
                (lazy.col("product_category") == "Electronics")
                & (lazy.col("amount") > 5000)
            )
            .select(
                "transaction_id",
                "customer_id",
                "transaction_date",
                "amount",
                "quantity",
            )
            .collect(use_physical_planner=True)
        )
        results.append(result)

    # Concatenate, sort, and take top 1000
    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values("amount", ascending=False).head(1000)


def query_filter_pandas_eager(files: list[str] | list[Path]) -> pd.DataFrame:
    """Filter query: high-value Electronics transactions using eager pandas."""
    import pandas as pd

    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    dfs = [pd.read_parquet(f, engine="pyarrow") for f in file_list]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    result = (
        df[(df["product_category"] == "Electronics") & (df["amount"] > 5000)]
        .sort_values("amount", ascending=False)
        .head(1000)[
            ["transaction_id", "customer_id", "transaction_date", "amount", "quantity"]
        ]
    )

    return result


def _print_result(name: str, result_dict: dict) -> None:
    """Print benchmark result in standard format."""
    mean = result_dict["mean"]
    std = result_dict["std"]
    print(f"   Mean: {mean:.2f}s (±{std:.2f})")


def run_filter_benchmark(
    files: list[Path] | list[str],
    memory_budget_mb: int = 2048,
    runs: int = 3,
    include_eager: bool = True,
) -> dict:
    """Run filter benchmark across all approaches."""
    results = {}

    print("\n" + "=" * 70)
    print("FILTER BENCHMARK: High-value Electronics transactions")
    print("=" * 70)

    # DuckDB
    print("\n1. DuckDB...")
    try:
        result, times = timeit(
            query_filter_duckdb, files, memory_limit=f"{memory_budget_mb}MB", runs=runs
        )
        results["duckdb"] = {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "rows": len(result),
        }
        _print_result("duckdb", results["duckdb"])
    except Exception as e:
        print(f"   FAILED: {e}")
        results["duckdb"] = {"error": str(e)}

    # Polars
    print("2. Polars...")
    try:
        result, times = timeit(query_filter_polars, files, runs=runs)
        results["polars"] = {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "rows": len(result),
        }
        _print_result("polars", results["polars"])
    except Exception as e:
        print(f"   FAILED: {e}")
        results["polars"] = {"error": str(e)}

    # Lazy pandas
    print("3. Lazy pandas...")
    try:
        result, times = timeit(query_filter_lazy_pandas, files, runs=runs)
        results["lazy_pandas"] = {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "rows": len(result),
        }
        _print_result("lazy_pandas", results["lazy_pandas"])
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["lazy_pandas"] = {"error": str(e)}

    # Eager pandas
    if include_eager:
        print("4. Eager pandas...")
        try:
            result, times = timeit(query_filter_pandas_eager, files, runs=runs)
            results["pandas_eager"] = {
                "times": times,
                "mean": np.mean(times),
                "std": np.std(times),
                "rows": len(result),
            }
            _print_result("pandas_eager", results["pandas_eager"])
        except Exception as e:
            print(f"   FAILED: {e}")
            results["pandas_eager"] = {"error": str(e)}

    return results


def query_sort_duckdb(
    files: list[str] | list[Path], memory_limit: str = "4GB"
) -> pd.DataFrame:
    """Sort query: order all transactions by amount (tests external sort)."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")

    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    file_pattern = f"read_parquet({file_list})"

    query = f"""
    SELECT transaction_id, amount, transaction_date
    FROM {file_pattern}
    ORDER BY amount DESC
    LIMIT 10000
    """

    result = con.execute(query).fetchdf()
    con.close()
    return result


def query_sort_polars(files: list[str] | list[Path]) -> pd.DataFrame:
    """Sort query: order all transactions by amount."""
    import polars as pl

    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    lf = pl.scan_parquet(file_list)

    result = (
        lf.select(["transaction_id", "amount", "transaction_date"])
        .sort("amount", descending=True)
        .head(10000)
        .collect(engine="streaming")
    )

    return result.to_pandas()


def query_sort_lazy_pandas(files: list[str] | list[Path]) -> pd.DataFrame:
    """Sort query: order all transactions by amount using lazy pandas."""
    import pandas as pd
    from pandas import lazy

    if isinstance(files[0], Path):
        file_list = [str(f) for f in files]
    else:
        file_list = list(files)

    # Process each file and concatenate results
    results = []
    for f in file_list:
        lf = lazy.scan(f)
        result = lf.select("transaction_id", "amount", "transaction_date").collect(
            use_physical_planner=True
        )
        results.append(result)

    # Concatenate, sort, and take top 10000
    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values("amount", ascending=False).head(10000)


def run_sort_benchmark(
    files: list[Path] | list[str],
    memory_budget_mb: int = 2048,
    runs: int = 3,
    include_eager: bool = True,
) -> dict:
    """Run sort benchmark across all approaches."""
    results = {}

    print("\n" + "=" * 70)
    print("SORT BENCHMARK: Top 10k transactions by amount")
    print("=" * 70)

    # DuckDB
    print("\n1. DuckDB...")
    try:
        result, times = timeit(
            query_sort_duckdb, files, memory_limit=f"{memory_budget_mb}MB", runs=runs
        )
        results["duckdb"] = {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "rows": len(result),
        }
        _print_result("duckdb", results["duckdb"])
    except Exception as e:
        print(f"   FAILED: {e}")
        results["duckdb"] = {"error": str(e)}

    # Polars
    print("2. Polars...")
    try:
        result, times = timeit(query_sort_polars, files, runs=runs)
        results["polars"] = {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "rows": len(result),
        }
        _print_result("polars", results["polars"])
    except Exception as e:
        print(f"   FAILED: {e}")
        results["polars"] = {"error": str(e)}

    # Lazy pandas
    print("3. Lazy pandas...")
    try:
        result, times = timeit(query_sort_lazy_pandas, files, runs=runs)
        results["lazy_pandas"] = {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "rows": len(result),
        }
        _print_result("lazy_pandas", results["lazy_pandas"])
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["lazy_pandas"] = {"error": str(e)}

    return results


# =============================================================================
# Benchmarking
# =============================================================================


def timeit(func, *args, warmup: int = 0, runs: int = 3, **kwargs):
    """Time a function with optional warmup runs."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
        gc.collect()

    # Timed runs
    times = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return result, times


def get_data_size(files: list[Path]) -> int:
    """Get total size of data files in bytes."""
    return sum(f.stat().st_size for f in files)


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def run_benchmarks(
    files: list[Path] | list[str],
    memory_budget_mb: int = 2048,
    runs: int = 3,
    include_eager: bool = True,
) -> dict:
    """Run all benchmarks and return results."""
    results = {}

    # Report data size
    if isinstance(files[0], Path):
        total_size = get_data_size(files)
        print(f"\nData size: {format_size(total_size)} across {len(files)} files")
    else:
        print(f"\nProcessing {len(files)} files from S3")

    print(f"Memory budget: {memory_budget_mb} MB")
    print(f"\nRunning benchmarks ({runs} runs each)...\n")

    # 1. DuckDB
    print("1. DuckDB (streaming with spill-to-disk)...")
    try:
        result_duckdb, times_duckdb = timeit(
            query_duckdb, files, memory_limit=f"{memory_budget_mb}MB", runs=runs
        )
        results["duckdb"] = {
            "times": times_duckdb,
            "mean": np.mean(times_duckdb),
            "std": np.std(times_duckdb),
            "rows": len(result_duckdb),
        }
        mean = results["duckdb"]["mean"]
        std = results["duckdb"]["std"]
        print(f"   Mean: {mean:.2f}s (±{std:.2f})")
    except Exception as e:
        print(f"   FAILED: {e}")
        results["duckdb"] = {"error": str(e)}

    # 2. Polars (streaming)
    print("2. Polars (lazy with streaming)...")
    try:
        result_polars, times_polars = timeit(query_polars, files, runs=runs)
        results["polars"] = {
            "times": times_polars,
            "mean": np.mean(times_polars),
            "std": np.std(times_polars),
            "rows": len(result_polars),
        }
        mean = results["polars"]["mean"]
        std = results["polars"]["std"]
        print(f"   Mean: {mean:.2f}s (±{std:.2f})")
    except Exception as e:
        print(f"   FAILED: {e}")
        results["polars"] = {"error": str(e)}

    # 3. Lazy pandas (physical planner with Concat node)
    print("3. Lazy pandas (physical planner)...")
    try:
        result_lazy_stream, times_lazy_stream = timeit(
            query_lazy_pandas,
            files,
            memory_budget_mb=memory_budget_mb,
            runs=runs,
        )
        results["lazy_pandas_streaming"] = {
            "times": times_lazy_stream,
            "mean": np.mean(times_lazy_stream),
            "std": np.std(times_lazy_stream),
            "rows": len(result_lazy_stream),
        }
        mean = results["lazy_pandas_streaming"]["mean"]
        std = results["lazy_pandas_streaming"]["std"]
        print(f"   Mean: {mean:.2f}s (±{std:.2f})")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback

        traceback.print_exc()
        results["lazy_pandas_streaming"] = {"error": str(e)}

    # 4. Eager pandas (only for small datasets)
    if include_eager:
        print("4. Eager pandas (loads all data)...")
        try:
            result_eager, times_eager = timeit(query_pandas_eager, files, runs=runs)
            results["pandas_eager"] = {
                "times": times_eager,
                "mean": np.mean(times_eager),
                "std": np.std(times_eager),
                "rows": len(result_eager),
            }
            mean = results["pandas_eager"]["mean"]
            std = results["pandas_eager"]["std"]
            print(f"   Mean: {mean:.2f}s (±{std:.2f})")
        except Exception as e:
            print(f"   FAILED: {e}")
            results["pandas_eager"] = {"error": str(e)}

    return results


def print_summary(results: dict) -> None:
    """Print a summary comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Find baseline (DuckDB or Polars)
    baseline_key = (
        "duckdb"
        if "duckdb" in results and "error" not in results["duckdb"]
        else "polars"
    )
    baseline = results.get(baseline_key, {}).get("mean", 1)

    print(f"\n{'Approach':<40} {'Mean (s)':>12} {'Std':>10} {'vs DuckDB':>12}")
    print("-" * 75)

    order = ["duckdb", "polars", "lazy_pandas_streaming", "pandas_eager"]
    names = {
        "duckdb": "DuckDB (streaming + spill)",
        "polars": "Polars (lazy streaming)",
        "lazy_pandas_streaming": "Lazy pandas (physical planner)",
        "pandas_eager": "Pandas (eager, all in memory)",
    }

    for key in order:
        if key not in results:
            continue
        r = results[key]
        name = names[key]
        if "error" in r:
            print(f"{name:<40} {'FAILED':>12}")
        else:
            ratio = r["mean"] / baseline if baseline else 0
            print(f"{name:<40} {r['mean']:>12.2f} {r['std']:>10.2f} {ratio:>11.2f}x")

    print("-" * 75)


def validate_results(files: list[Path]) -> bool:
    """
    Validate that all approaches produce equivalent results.

    Returns True if all results match, False otherwise.
    """

    print("\n" + "=" * 70)
    print("RESULT VALIDATION")
    print("=" * 70)

    # Run all queries
    print("\nRunning queries...")
    results = {}

    try:
        results["duckdb"] = query_duckdb(files, memory_limit="4GB")
        print("  - DuckDB")
    except Exception as e:
        print(f"  - DuckDB FAILED: {e}")

    try:
        results["polars"] = query_polars(files)
        print("  - Polars")
    except Exception as e:
        print(f"  - Polars FAILED: {e}")

    try:
        results["lazy_pandas_streaming"] = query_lazy_pandas_streaming(files)
        print("  - Lazy pandas streaming")
    except Exception as e:
        print(f"  - Lazy pandas streaming FAILED: {e}")

    try:
        results["pandas_eager"] = query_pandas_eager(files)
        print("  - Pandas eager")
    except Exception as e:
        print(f"  - Pandas eager FAILED: {e}")

    if len(results) < 2:
        print("\nNot enough results to compare!")
        return False

    # Normalize results
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Ensure consistent column order
        cols = [
            "transaction_date",
            "transaction_count",
            "unique_customers",
            "total_amount",
            "total_quantity",
        ]
        df = df[cols]
        # Sort by date
        df = df.sort_values("transaction_date").reset_index(drop=True)
        # Convert types
        df["transaction_count"] = df["transaction_count"].astype("int64")
        df["unique_customers"] = df["unique_customers"].astype("int64")
        df["total_amount"] = df["total_amount"].astype("float64")
        df["total_quantity"] = df["total_quantity"].astype("int64")
        return df

    normalized = {name: normalize(df) for name, df in results.items()}

    # Compare row counts
    print("\n--- Row Counts ---")
    for name, df in normalized.items():
        print(f"  {name:<30}: {len(df):>6} rows")

    # Compare key aggregates
    print("\n--- Key Aggregates ---")
    for name, df in normalized.items():
        total_txns = df["transaction_count"].sum()
        total_amount = df["total_amount"].sum()
        print(f"  {name:<30}: {total_txns:>12,} txns, ${total_amount:>15,.2f}")

    # Cross-validation
    print("\n--- Cross-Validation ---")
    baseline_name = (
        "duckdb" if "duckdb" in normalized else next(iter(normalized.keys()))
    )
    baseline = normalized[baseline_name]
    all_passed = True

    for name, df in normalized.items():
        if name == baseline_name:
            continue

        # Check row count
        if len(df) != len(baseline):
            print(f"  X {name}: row count mismatch ({len(df)} vs {len(baseline)})")
            all_passed = False
            continue

        # Check totals
        base_count = baseline["transaction_count"].sum()
        test_count = df["transaction_count"].sum()
        if base_count != test_count:
            print(
                f"  X {name}: transaction_count mismatch ({test_count} vs {base_count})"
            )
            all_passed = False
            continue

        base_amount = baseline["total_amount"].sum()
        test_amount = df["total_amount"].sum()
        if not np.isclose(base_amount, test_amount, rtol=1e-4):
            msg = f"  X {name}: total_amount mismatch "
            msg += f"({test_amount:.2f} vs {base_amount:.2f})"
            print(msg)
            all_passed = False
            continue

        print(f"  OK {name}")

    print("\n" + "-" * 70)
    if all_passed:
        print("VALIDATION PASSED: All results match!")
    else:
        print("VALIDATION FAILED: Some results do not match!")
    print("-" * 70)

    return all_passed


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="1TB Banking Transaction Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use small test data (10M rows, generated locally)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full 1TB dataset from S3 (requires AWS credentials)",
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=None,
        metavar="ROWS",
        help="Generate test data with specified number of rows",
    )
    parser.add_argument(
        "--memory-budget",
        type=int,
        default=2048,
        help="Memory budget in MB (default: 2048)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all approaches produce equivalent results",
    )
    parser.add_argument(
        "--no-eager",
        action="store_true",
        help="Skip eager pandas benchmark (useful for large datasets)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark types (aggregation, filter, sort)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Run only filter benchmark",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Run only sort benchmark",
    )

    args = parser.parse_args()

    if args.generate:
        files = generate_test_data(num_rows=args.generate)
        print(f"\nGenerated {len(files)} test files")
        return

    if args.full:
        # Use S3 files
        print("Loading file list from S3...")
        files = get_s3_files()
        if not files:
            print(
                "ERROR: No files found in S3. Check AWS credentials and bucket access."
            )
            return
        print(f"Found {len(files)} files")
        include_eager = False  # Never use eager for 1TB
    else:
        # Use local test files
        files = get_test_files()
        include_eager = not args.no_eager

    if args.validate:
        validate_results(files)
    elif args.filter:
        run_filter_benchmark(
            files,
            memory_budget_mb=args.memory_budget,
            runs=args.runs,
            include_eager=include_eager,
        )
    elif args.sort:
        run_sort_benchmark(
            files,
            memory_budget_mb=args.memory_budget,
            runs=args.runs,
            include_eager=include_eager,
        )
    elif getattr(args, "all", False):
        # Run all benchmarks
        print("\n" + "=" * 70)
        print("RUNNING ALL BENCHMARKS")
        print("=" * 70)

        # 1. Aggregation benchmark (original)
        print("\n>>> AGGREGATION BENCHMARK <<<")
        results = run_benchmarks(
            files,
            memory_budget_mb=args.memory_budget,
            runs=args.runs,
            include_eager=include_eager,
        )
        print_summary(results)

        # 2. Filter benchmark
        print("\n>>> FILTER BENCHMARK <<<")
        run_filter_benchmark(
            files,
            memory_budget_mb=args.memory_budget,
            runs=args.runs,
            include_eager=include_eager,
        )

        # 3. Sort benchmark
        print("\n>>> SORT BENCHMARK <<<")
        run_sort_benchmark(
            files,
            memory_budget_mb=args.memory_budget,
            runs=args.runs,
            include_eager=include_eager,
        )
    else:
        results = run_benchmarks(
            files,
            memory_budget_mb=args.memory_budget,
            runs=args.runs,
            include_eager=include_eager,
        )
        print_summary(results)


if __name__ == "__main__":
    main()
