#!/usr/bin/env python
"""
Benchmark: String operations

Compares Arrow vs NumPy string operations.
String ops are a key area where Arrow excels.
"""

from collections.abc import Callable
import time

import numpy as np

import pandas as pd


def timeit(func: Callable, n_runs: int = 5, warmup: int = 1) -> tuple[float, float]:
    """Run function multiple times and return (mean, std) in milliseconds."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def create_string_data(n_rows: int, use_arrow: bool = False) -> pd.DataFrame:
    """Create DataFrame with string columns."""
    rng = np.random.default_rng(42)

    # Generate random strings
    prefixes = ["foo", "bar", "baz", "qux"]
    suffixes = ["_a", "_b", "_c", "_d"]

    strings = [
        rng.choice(prefixes) + str(i % 1000) + rng.choice(suffixes)
        for i in range(n_rows)
    ]

    df = pd.DataFrame({"text": strings, "id": np.arange(n_rows)})

    if use_arrow:
        df = df.astype({"text": "string[pyarrow]", "id": "int64[pyarrow]"})

    return df


def bench_eager_str_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: lowercase strings."""
    result = df.copy()
    result["lower"] = df["text"].str.lower()
    return result


def bench_lazy_str_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: lowercase strings."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(col("text").str.lower().alias("lower"))
    return lf.collect()


def bench_eager_str_upper(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: uppercase strings."""
    result = df.copy()
    result["upper"] = df["text"].str.upper()
    return result


def bench_lazy_str_upper(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: uppercase strings."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(col("text").str.upper().alias("upper"))
    return lf.collect()


def bench_eager_str_contains(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: check if string contains pattern."""
    return df[df["text"].str.contains("foo")]


def bench_lazy_str_contains(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: check if string contains pattern."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("text").str.contains("foo"))
    return lf.collect()


def bench_eager_str_startswith(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: check if string starts with pattern."""
    return df[df["text"].str.startswith("foo")]


def bench_lazy_str_startswith(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: check if string starts with pattern."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.filter(col("text").str.startswith("foo"))
    return lf.collect()


def bench_eager_str_len(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: get string length."""
    result = df.copy()
    result["length"] = df["text"].str.len()
    return result


def bench_lazy_str_len(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: get string length."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(col("text").str.len().alias("length"))
    return lf.collect()


def bench_eager_str_replace(df: pd.DataFrame) -> pd.DataFrame:
    """Eager: replace pattern in string."""
    result = df.copy()
    result["replaced"] = df["text"].str.replace("foo", "FOO", regex=False)
    return result


def bench_lazy_str_replace(df: pd.DataFrame) -> pd.DataFrame:
    """Lazy: replace pattern in string."""
    from pandas.lazy import col

    lf = df.select()
    lf = lf.with_columns(
        col("text").str.replace("foo", "FOO", regex=False).alias("replaced")
    )
    return lf.collect()


def run_benchmarks():
    """Run all string benchmarks."""
    sizes = [10_000, 100_000, 500_000]  # Smaller sizes for strings (more expensive)

    print("=" * 70)
    print("STRING OPERATION BENCHMARKS")
    print("=" * 70)
    print()
    print("NOTE: String operations typically show Arrow's advantage most clearly.")
    print()

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset size: {n_rows:,} rows")
        print("=" * 70)

        df_numpy = create_string_data(n_rows, use_arrow=False)
        df_arrow = create_string_data(n_rows, use_arrow=True)

        # str.lower()
        print("\n--- str.lower() ---")

        mean, std = timeit(lambda: bench_eager_str_lower(df_numpy))
        print(f"Eager (NumPy/object):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_lower(df_numpy))
        print(f"Lazy (NumPy/object):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_str_lower(df_arrow))
        print(f"Eager (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_lower(df_arrow))
        print(f"Lazy (Arrow):          {mean:8.2f} ms ± {std:.2f} ms")

        # str.contains()
        print("\n--- str.contains('foo') ---")

        mean, std = timeit(lambda: bench_eager_str_contains(df_numpy))
        print(f"Eager (NumPy/object):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_contains(df_numpy))
        print(f"Lazy (NumPy/object):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_str_contains(df_arrow))
        print(f"Eager (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_contains(df_arrow))
        print(f"Lazy (Arrow):          {mean:8.2f} ms ± {std:.2f} ms")

        # str.startswith()
        print("\n--- str.startswith('foo') ---")

        mean, std = timeit(lambda: bench_eager_str_startswith(df_numpy))
        print(f"Eager (NumPy/object):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_startswith(df_numpy))
        print(f"Lazy (NumPy/object):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_str_startswith(df_arrow))
        print(f"Eager (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_startswith(df_arrow))
        print(f"Lazy (Arrow):          {mean:8.2f} ms ± {std:.2f} ms")

        # str.len()
        print("\n--- str.len() ---")

        mean, std = timeit(lambda: bench_eager_str_len(df_numpy))
        print(f"Eager (NumPy/object):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_len(df_numpy))
        print(f"Lazy (NumPy/object):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_str_len(df_arrow))
        print(f"Eager (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_len(df_arrow))
        print(f"Lazy (Arrow):          {mean:8.2f} ms ± {std:.2f} ms")

        # str.replace()
        print("\n--- str.replace('foo', 'FOO') ---")

        mean, std = timeit(lambda: bench_eager_str_replace(df_numpy))
        print(f"Eager (NumPy/object):  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_replace(df_numpy))
        print(f"Lazy (NumPy/object):   {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_eager_str_replace(df_arrow))
        print(f"Eager (Arrow):         {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: bench_lazy_str_replace(df_arrow))
        print(f"Lazy (Arrow):          {mean:8.2f} ms ± {std:.2f} ms")


if __name__ == "__main__":
    run_benchmarks()
