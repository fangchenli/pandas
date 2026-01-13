#!/usr/bin/env python
"""
Benchmark: Array format conversions

Measures the cost of converting between Arrow and NumPy formats.
This helps understand when format conversion is worth it.
"""

from collections.abc import Callable
import time

import numpy as np
import pyarrow as pa

import pandas as pd


def timeit(func: Callable, n_runs: int = 10, warmup: int = 2) -> tuple[float, float]:
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


def run_benchmarks():
    """Run conversion benchmarks."""
    from pandas.lazy.backends.convert import (
        to_arrow,
        to_numpy,
    )

    sizes = [10_000, 100_000, 1_000_000, 5_000_000]

    print("=" * 70)
    print("ARRAY CONVERSION BENCHMARKS")
    print("=" * 70)
    print()
    print("Measures cost of converting between Arrow and NumPy formats.")
    print()

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Array size: {n_rows:,} elements")
        print("=" * 70)

        rng = np.random.default_rng(42)

        # Numeric arrays
        np_float = rng.random(n_rows)
        np_int = rng.integers(0, 1000, n_rows)
        pa_float = pa.array(np_float)
        pa_int = pa.array(np_int)

        # String arrays
        strings = [f"value_{i % 1000}" for i in range(n_rows)]
        np_str = np.array(strings, dtype=object)
        pa_str = pa.array(strings)

        print("\n--- Float64 arrays ---")

        mean, std = timeit(lambda: to_arrow(np_float))
        print(f"NumPy → Arrow:  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: to_numpy(pa_float))
        print(f"Arrow → NumPy:  {mean:8.2f} ms ± {std:.2f} ms")

        # Zero-copy if possible
        mean, std = timeit(lambda: pa_float.to_numpy(zero_copy_only=False))
        print(f"Arrow → NumPy (to_numpy):  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n--- Int64 arrays ---")

        mean, std = timeit(lambda: to_arrow(np_int))
        print(f"NumPy → Arrow:  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: to_numpy(pa_int))
        print(f"Arrow → NumPy:  {mean:8.2f} ms ± {std:.2f} ms")

        print("\n--- String arrays ---")

        mean, std = timeit(lambda: to_arrow(np_str))
        print(f"NumPy → Arrow:  {mean:8.2f} ms ± {std:.2f} ms")

        mean, std = timeit(lambda: to_numpy(pa_str))
        print(f"Arrow → NumPy:  {mean:8.2f} ms ± {std:.2f} ms")

        # DataFrame conversion
        print("\n--- DataFrame conversion (3 float columns) ---")

        df_numpy = pd.DataFrame(
            {
                "a": rng.random(n_rows),
                "b": rng.random(n_rows),
                "c": rng.random(n_rows),
            }
        )

        df_arrow = df_numpy.astype(
            {"a": "double[pyarrow]", "b": "double[pyarrow]", "c": "double[pyarrow]"}
        )

        mean, std = timeit(
            lambda: df_numpy.astype(
                {"a": "double[pyarrow]", "b": "double[pyarrow]", "c": "double[pyarrow]"}
            )
        )
        print(f"DataFrame NumPy → Arrow:  {mean:8.2f} ms ± {std:.2f} ms")

        # Arrow to NumPy via to_numpy() on each column
        def arrow_df_to_numpy():
            return pd.DataFrame(
                {col: df_arrow[col].to_numpy() for col in df_arrow.columns}
            )

        mean, std = timeit(arrow_df_to_numpy)
        print(f"DataFrame Arrow → NumPy:  {mean:8.2f} ms ± {std:.2f} ms")

    # Cost analysis
    print("\n" + "=" * 70)
    print("CONVERSION COST ANALYSIS")
    print("=" * 70)
    print()
    print("Key insights:")
    print("1. Float/Int conversions are relatively cheap (may be zero-copy)")
    print("2. String conversions are expensive (memory allocation)")
    print("3. Only convert when the operation benefit exceeds conversion cost")
    print("4. For strings: Arrow ops are 2-10x faster, worth converting")
    print("5. For numerics: stay in original format unless doing many ops")


if __name__ == "__main__":
    run_benchmarks()
