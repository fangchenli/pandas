"""
Parallel driver for the sorted merge-join indexer in ``pandas._libs.join``.

``libjoin.inner_join_indexer`` walks two monotonic arrays in lockstep, so the
work can be split into independent sub-joins.  Splitting naively and
concatenating the per-chunk outputs is slower than not splitting at all -- the
concatenation costs more than the parallel walk saves.  Instead the chunks are
counted first, the counts are prefix-summed into output offsets, the three
output arrays are allocated once, and each chunk fills its own disjoint slice.

Split points must land on a *key* boundary rather than a raw index: a run of
equal keys straddling a split would make the equal-key advance in the kernel
emit a different number of pairs than the serial walk does.

The callers that benefit are :meth:`Index.join` with ``how="inner"`` and
:meth:`Index.intersection`, both of which are dominated by this merge.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from pandas._config import get_option

from pandas._libs import join as libjoin

if TYPE_CHECKING:
    from pandas._typing import npt

# The size at which splitting a join across threads starts to pay for the
# thread dispatch was measured between ~50k and ~500k rows, varying with the
# number of chunks, how many keys match, and above all whether the thread pool
# is warm.  A pool is created per call here (as ``read_csv`` does), which is the
# expensive end of that range, and the measurements come from a single machine,
# so the threshold is set well above the worst observed crossover.
_PARALLEL_JOIN_MIN_ROWS = 1_000_000

# Parallel joins see diminishing returns beyond a handful of workers -- the
# output-writing half is memory-bound and stops scaling well before the scan
# does -- and a low default avoids oversubscribing the machine.  Users who want
# more can opt in explicitly via ``mode.max_threads``.
_MAX_DEFAULT_WORKERS = 4


def _n_workers() -> int:
    """Number of worker threads to use, honouring ``mode.max_threads``."""
    if sys.platform == "emscripten":
        # WASM cannot spawn threads, regardless of mode.max_threads.
        return 1
    max_threads = get_option("mode.max_threads")
    if max_threads is not None:
        return max_threads
    return min(os.cpu_count() or 1, _MAX_DEFAULT_WORKERS)


def _can_parallelize(left: np.ndarray, right: np.ndarray) -> bool:
    """
    Whether these inputs are worth splitting.

    Deliberately cheap, and checked *before* _n_workers(): the overwhelming
    majority of joins are small, and reading ``mode.max_threads`` plus
    ``os.cpu_count()`` costs more than the whole serial merge does at that size.
    """
    if min(left.shape[0], right.shape[0]) < _PARALLEL_JOIN_MIN_ROWS:
        return False
    if left.dtype != right.dtype or left.dtype.kind not in "iuf":
        # object dtype holds the GIL in the kernel, so threads would not help
        return False
    if left.ndim != 1 or right.ndim != 1:
        return False
    return bool(left.flags.c_contiguous and right.flags.c_contiguous)


def _key_boundaries(left: np.ndarray, n_chunks: int) -> npt.NDArray[np.intp]:
    """
    Split points in *left* that never cut a run of equal keys.

    Aims for *n_chunks* equal-sized chunks, then pushes each split to the end of
    whatever run of equal keys it landed in.  Returns strictly increasing
    indices starting at 0 and ending at ``len(left)``, so chunk ``i`` is
    ``left[bounds[i]:bounds[i + 1]]``.  Fewer chunks than requested come back
    when keys repeat heavily; a single-key array yields ``[0, len(left)]``.
    """
    n = left.shape[0]
    approx = np.linspace(0, n, n_chunks + 1)[1:-1].astype(np.intp)
    keys = left[approx]
    # Either end of the run containing a split point is a valid boundary.
    # Prefer the end of the run, but fall back to its start when the end would
    # land past the last element -- otherwise a split point landing anywhere in
    # the final run is discarded and the whole join silently runs serially.
    after = np.searchsorted(left, keys, side="right")
    before = np.searchsorted(left, keys, side="left")
    snapped = np.where(after < n, after, before)
    inner = np.unique(snapped[(snapped > 0) & (snapped < n)])
    return np.concatenate(
        [
            np.zeros(1, dtype=np.intp),
            inner.astype(np.intp),
            np.array([n], dtype=np.intp),
        ]
    )


def _chunk_ranges(
    left: np.ndarray, right: np.ndarray, bounds: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    For each left chunk, the half-open sub-range of *right* that can match it.

    Only right values within ``[first key, last key]`` of a chunk can pair with
    it, and an inner join discards everything else, so the sub-ranges are
    disjoint -- with gaps wherever *right* holds keys that *left* does not.
    """
    starts = bounds[:-1]
    stops = bounds[1:]
    rstart = np.searchsorted(right, left[starts], side="left")
    rstop = np.searchsorted(right, left[stops - 1], side="right")
    return rstart.astype(np.intp), rstop.astype(np.intp)


def inner_join_indexer(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Equivalent of ``libjoin.inner_join_indexer``, parallel for large inputs.

    Both *left* and *right* must be monotonic increasing.  The result is
    identical to the serial indexer for every input; the parallel path is used
    only when it is expected to pay off, and falls back silently otherwise.
    """
    if not _can_parallelize(left, right):
        return libjoin.inner_join_indexer(left, right)

    n_workers = _n_workers()
    if n_workers < 2:
        return libjoin.inner_join_indexer(left, right)

    bounds = _key_boundaries(left, n_workers)
    if bounds.shape[0] <= 2:
        # a single run of equal keys spans the whole of left; nothing to split
        return libjoin.inner_join_indexer(left, right)

    rstart, rstop = _chunk_ranges(left, right, bounds)
    lviews = [left[a:b] for a, b in pairwise(bounds)]
    rviews = [right[a:b] for a, b in zip(rstart, rstop, strict=True)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        counts = np.fromiter(
            executor.map(libjoin.inner_join_count_range, lviews, rviews),
            dtype=np.intp,
            count=len(lviews),
        )

        offsets = np.zeros(counts.shape[0], dtype=np.intp)
        np.cumsum(counts[:-1], out=offsets[1:])
        total = int(counts.sum())

        result = np.empty(total, dtype=left.dtype)
        lindexer = np.empty(total, dtype=np.intp)
        rindexer = np.empty(total, dtype=np.intp)

        def fill(i: int) -> None:
            written = libjoin.inner_join_fill_range(
                lviews[i],
                rviews[i],
                result,
                lindexer,
                rindexer,
                int(offsets[i]),
                int(bounds[i]),
                int(rstart[i]),
            )
            if written != counts[i]:
                # the count and fill passes disagree, which can only happen if
                # the inputs changed underneath us
                raise ValueError(
                    f"parallel join chunk {i} wrote {written} pairs, "
                    f"expected {counts[i]}"
                )

        # consume the iterator so worker exceptions propagate here
        list(executor.map(fill, range(len(lviews))))

    return result, lindexer, rindexer
