#!/usr/bin/env python
"""
Benchmark: Kernel dispatch overhead

Measures the overhead of the kernel dispatch system compared to direct calls.
This helps quantify the cost of the abstraction layer.
"""

from collections.abc import Callable
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def timeit(func: Callable, n_runs: int = 100, warmup: int = 10) -> tuple[float, float]:
    """Run function multiple times and return (mean, std) in microseconds."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)  # Convert to microseconds

    return np.mean(times), np.std(times)


def create_arrays(n_rows: int) -> tuple[np.ndarray, pa.Array]:
    """Create test arrays."""
    rng = np.random.default_rng(42)
    np_arr = rng.random(n_rows)
    pa_arr = pa.array(np_arr)
    return np_arr, pa_arr


def run_benchmarks():
    """Run kernel overhead benchmarks."""
    from pandas.lazy.backends import dispatch_kernel

    sizes = [1_000, 10_000, 100_000, 1_000_000]

    print("=" * 70)
    print("KERNEL DISPATCH OVERHEAD BENCHMARKS")
    print("=" * 70)
    print()
    print("Comparing direct function calls vs dispatch_kernel() calls.")
    print("Times are in microseconds (μs).")
    print()

    for n_rows in sizes:
        print(f"\n{'=' * 70}")
        print(f"Array size: {n_rows:,} elements")
        print("=" * 70)

        np_arr, pa_arr = create_arrays(n_rows)
        scalar = 2.0

        # NumPy addition
        print("\n--- NumPy addition (arr + scalar) ---")

        mean, std = timeit(lambda: np.add(np_arr, scalar))
        print(f"Direct np.add:         {mean:10.2f} μs ± {std:.2f} μs")

        mean, std = timeit(lambda: dispatch_kernel("add", "numpy", np_arr, scalar))
        print(f"dispatch_kernel:       {mean:10.2f} μs ± {std:.2f} μs")

        # Arrow addition
        print("\n--- Arrow addition (arr + scalar) ---")

        mean, std = timeit(lambda: pc.add(pa_arr, scalar))
        print(f"Direct pc.add:         {mean:10.2f} μs ± {std:.2f} μs")

        mean, std = timeit(lambda: dispatch_kernel("add", "arrow", pa_arr, scalar))
        print(f"dispatch_kernel:       {mean:10.2f} μs ± {std:.2f} μs")

        # NumPy comparison
        print("\n--- NumPy comparison (arr > scalar) ---")

        mean, std = timeit(lambda: np.greater(np_arr, 0.5))
        print(f"Direct np.greater:     {mean:10.2f} μs ± {std:.2f} μs")

        mean, std = timeit(lambda: dispatch_kernel("greater", "numpy", np_arr, 0.5))
        print(f"dispatch_kernel:       {mean:10.2f} μs ± {std:.2f} μs")

        # Arrow comparison
        print("\n--- Arrow comparison (arr > scalar) ---")

        mean, std = timeit(lambda: pc.greater(pa_arr, 0.5))
        print(f"Direct pc.greater:     {mean:10.2f} μs ± {std:.2f} μs")

        mean, std = timeit(lambda: dispatch_kernel("greater", "arrow", pa_arr, 0.5))
        print(f"dispatch_kernel:       {mean:10.2f} μs ± {std:.2f} μs")

        # NumPy multiply
        print("\n--- NumPy multiply (arr * scalar) ---")

        mean, std = timeit(lambda: np.multiply(np_arr, scalar))
        print(f"Direct np.multiply:    {mean:10.2f} μs ± {std:.2f} μs")

        mean, std = timeit(lambda: dispatch_kernel("multiply", "numpy", np_arr, scalar))
        print(f"dispatch_kernel:       {mean:10.2f} μs ± {std:.2f} μs")

        # Arrow multiply
        print("\n--- Arrow multiply (arr * scalar) ---")

        mean, std = timeit(lambda: pc.multiply(pa_arr, scalar))
        print(f"Direct pc.multiply:    {mean:10.2f} μs ± {std:.2f} μs")

        mean, std = timeit(lambda: dispatch_kernel("multiply", "arrow", pa_arr, scalar))
        print(f"dispatch_kernel:       {mean:10.2f} μs ± {std:.2f} μs")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The overhead of dispatch_kernel() should be minimal (a few μs).")
    print("For large arrays, this overhead is negligible compared to compute time.")
    print("For very small arrays, direct calls may be preferred.")


if __name__ == "__main__":
    run_benchmarks()
