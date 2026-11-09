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

Where those points go is :func:`_plan_chunks`, which scores two schemes against
the work each leaves its chunks and declines when neither divides it.  See that
function; the constants below decide only whether and how far to split.

The callers that benefit are :meth:`Index.join` with ``how="inner"`` and
:meth:`Index.intersection`, both of which are dominated by this merge.

Tuning
------
Measured on a four performance-core laptop, where repeated runs of the same
binary spread 8-16%; smaller differences are not resolvable here and none are
claimed.  Figures are ratios against the serial kernel on the same machine.

``_PARALLEL_JOIN_MIN_ROWS`` is the size below which splitting never happens.
The crossover moves by an order of magnitude with how many keys match --
measured at ~40k rows when half of them do and ~350k when all do -- so the
threshold tracks the match-dense end.  Why density matters that much has not
been pinned down; note that the serial indexer is itself two-pass, so the
counting pass is not extra work the split has to earn back.  The threshold also
keeps small joins from starting pools at all.

``_ROWS_PER_CHUNK_SCALE`` sets how the chunk count grows, as
``isqrt(n // this)``, fitted to optima of 4 chunks at 2M rows, 8 at 10M and 16
at 40M.  The curve is flat near its peak, so landing within a factor of two
costs little; a fixed count does not, and gave up about half the available
speedup at 40M.

``_MAX_DEFAULT_WORKERS`` bounds the chunk count -- not the thread count, since
``ThreadPoolExecutor`` starts workers only as tasks arrive, so a pool sized for
a 192-core host still spawns just a handful.  It engages only above 32M rows
and only where more cores than this are available, so its value is a policy
choice rather than a measured optimum; from 8 chunks to 32 the curve is flat
here.

``_MIN_USEFUL_SPEEDUP`` is the least a split must be worth for
:func:`_plan_chunks` to accept it.  Some inputs cannot be divided usefully
under either scheme, and splitting those pays for a pool and buys nothing.

``_DISPATCH_COST_ITERS`` floors what a chunk is costed at, so that a split
whose chunks clip down to almost nothing is still charged for being dispatched.
Planning and the pool come to ~82us, and a walk step that only advances an
index measures 1.27ns, hence ~65k iterations.  Steps that emit are slower, so
this errs high and declines marginal splits.  Without it a join the serial walk
finishes in one step -- a tiny right side lying below every left key -- scores
infinitely well and gets a thread pool.

See GH#43313 for the wider question of how pandas should expose parallelism at
all; ``mode.max_threads`` is the only control over these pools, as
``threadpoolctl`` does not see Python-level ones.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import join as libjoin

from pandas.core.util.parallel import (
    interpreter_shutting_down,
    max_workers,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
    )

    from pandas._typing import npt

# Below this, the counting pass costs more than the split saves; see Tuning.
_PARALLEL_JOIN_MIN_ROWS = 1_000_000

# Ceiling on chunks per join; only engages above 32M rows.  See Tuning.
_MAX_DEFAULT_WORKERS = 16

# Chunk count is isqrt(n // this), fitted to the measured optima.
_ROWS_PER_CHUNK_SCALE = 125_000

# Least speedup worth a pool, against the ceiling the largest chunk sets.
_MIN_USEFUL_SPEEDUP = 1.05

# Floor on what a chunk costs, in walk iterations; see Tuning.
_DISPATCH_COST_ITERS = 65_000

# What the chunked kernels are specialised for (numeric_t in join.pyx), in
# native byte order.  Checked exactly, not by dtype kind, so that float16 or a
# byte-swapped array is turned away here rather than after planning, by a
# kernel raising inside the pool.
_KERNEL_DTYPES = frozenset(
    np.dtype(t)
    for t in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    )
)


def _n_chunks(n_rows: int, n_workers: int) -> int:
    """
    How many chunks to split a join of *n_rows* into.

    Parameters
    ----------
    n_rows : int
        Length of the left side, which the split is taken over.
    n_workers : int
        Upper bound from :func:`pandas.core.util.parallel.max_workers`.

    Returns
    -------
    int
        At least 2, never more than *n_workers*.
    """
    return max(2, min(n_workers, math.isqrt(n_rows // _ROWS_PER_CHUNK_SCALE)))


def _can_parallelize(left: np.ndarray, right: np.ndarray) -> bool:
    """
    Whether these inputs are worth splitting at all.

    Deliberately cheap, and checked *before* resolving a worker count: the
    overwhelming majority of joins are small, and reading ``mode.max_threads``
    plus a CPU count costs more than the whole serial merge does at that size.

    Parameters
    ----------
    left, right : ndarray
        The join targets, monotonic increasing and free of NAs.

    Returns
    -------
    bool
    """
    # Only left is measured against the threshold, because only left is
    # divided; each chunk clips right to its own key span, so a right side
    # orders of magnitude smaller still splits usefully -- 20M against 100k
    # measured 3.7x.  Where it does not, :func:`_plan_chunks` declines.
    if left.shape[0] < _PARALLEL_JOIN_MIN_ROWS or right.shape[0] == 0:
        return False
    if left.dtype != right.dtype or left.dtype not in _KERNEL_DTYPES:
        # object dtype holds the GIL in the kernel, so threads would not help;
        # anything else has no chunked kernel to run
        return False
    # Strides are deliberately not checked: the kernel indexes through
    # ``diminfo[0].strides``, so a sliced index joins correctly and still gains.
    return left.ndim == 1 and right.ndim == 1


def _snap_to_keys(left: np.ndarray, keys: np.ndarray) -> npt.NDArray[np.intp]:
    """
    Turn cut keys into strictly increasing bounds over *left*.

    Cutting at a run boundary keeps every occurrence of that key on one side of
    the boundary, which the kernel's equal-key advance requires.  Either end of
    the run will do; prefer the end, but fall back to the start when the end
    would land past the last element -- otherwise a cut inside the final run is
    discarded and the join silently runs serially.

    Parameters
    ----------
    left : ndarray
        The left join target, monotonic increasing.
    keys : ndarray
        Candidate cut keys, of ``left.dtype``.  Duplicates and keys outside
        *left* are fine; both drop out.

    Returns
    -------
    ndarray of intp
        Strictly increasing bounds beginning at 0 and ending at ``len(left)``,
        so chunk ``i`` is ``left[bounds[i]:bounds[i + 1]]``.  Just
        ``[0, len(left)]`` when no key yields a usable cut.

    See Also
    --------
    _even_boundaries : Cuts from equal position ranges.
    _merge_path_boundaries : Cuts from equal shares of the merge.
    """
    n = left.shape[0]
    after = left.searchsorted(keys, side="right")
    before = left.searchsorted(keys, side="left")
    cuts = np.where(after < n, after, before)
    inner = np.unique(cuts[(cuts > 0) & (cuts < n)])
    return np.concatenate(
        [
            np.zeros(1, dtype=np.intp),
            inner.astype(np.intp, copy=False),
            np.array([n], dtype=np.intp),
        ]
    )


def _even_boundaries(left: np.ndarray, n_chunks: int) -> npt.NDArray[np.intp]:
    """
    Cuts that divide *left* into equal position ranges.

    Right is only ever clipped to each chunk's key span, so when the two sides
    barely overlap this leaves most chunks with nothing to do -- which is the
    cheapest possible outcome, not a failure.  It balances badly when right is
    lopsided; see :func:`_merge_path_boundaries`.

    Parameters
    ----------
    left : ndarray
        The left join target, monotonic increasing.
    n_chunks : int
        How many chunks to aim for.

    Returns
    -------
    ndarray of intp
        Bounds over *left*, as :func:`_snap_to_keys` returns.  Fewer chunks
        than asked for come back when keys repeat heavily.

    See Also
    --------
    _merge_path_boundaries : The other scheme :func:`_plan_chunks` weighs.
    """
    approx = np.linspace(0, left.shape[0], n_chunks + 1)[1:-1].astype(np.intp)
    return _snap_to_keys(left, left[approx])


def _merge_path_boundaries(
    left: np.ndarray, right: np.ndarray, n_chunks: int
) -> npt.NDArray[np.intp]:
    """
    Cuts that divide the *merge* of both sides into equal shares.

    The walk takes one step per element from either side, so cutting the merge
    into equal-length runs balances the work by construction [1]_.  A cut is
    the point ``(i, j)`` with ``i + j == diag`` where taking from left gives way
    to taking from right; that switch is monotonic along the diagonal, so a
    binary search finds it in ``O(log min(n, m))``.  This is what
    handles a lopsided right, where dividing left evenly hands one chunk the
    whole of a dominant run and the rest nothing -- the naive scheme's defining
    failure, per the same paper.

    It is not universally better here, because our chunks discard right rows
    outside their key span while Merge Path charges for them: two disjoint
    indexes cost nothing to join and everything to merge, so it packs them into
    one chunk.  :func:`_plan_chunks` costs both and takes the better.

    Parameters
    ----------
    left, right : ndarray
        The join targets, monotonic increasing.
    n_chunks : int
        How many chunks to aim for.

    Returns
    -------
    ndarray of intp
        Bounds over *left*, as :func:`_snap_to_keys` returns.

    See Also
    --------
    _even_boundaries : The other scheme :func:`_plan_chunks` weighs.

    References
    ----------
    .. [1] S. Odeh, O. Green, Z. Mwassi, O. Shmueli, Y. Birk. `Merge Path -
       Parallel Merging Made Simple
       <https://ieeexplore.ieee.org/document/6270834>`_.  IEEE IPDPSW, 2012,
       pp. 1611-1618.  A later write-up by a subset of the authors is freely
       available as `arXiv:1406.2628 <https://arxiv.org/abs/1406.2628>`_.
    """
    n, m = left.shape[0], right.shape[0]
    total = n + m
    keys = []
    for p in range(1, n_chunks):
        diag = p * total // n_chunks
        lo, hi = max(0, diag - m), min(diag, n)
        while lo < hi:
            mid = (lo + hi) // 2
            if left.item(mid) > right.item(diag - mid - 1):
                hi = mid
            else:
                lo = mid + 1
        i, j = lo, diag - lo
        # the next key the merge would consume, which is where to cut
        if i < n and j < m:
            keys.append(min(left.item(i), right.item(j)))
        elif i < n:
            keys.append(left.item(i))
        elif j < m:
            keys.append(right.item(j))
    if not keys:
        return np.array([0, n], dtype=np.intp)
    return _snap_to_keys(left, np.array(keys, dtype=left.dtype))


def _chunk_ranges(
    left: np.ndarray, right: np.ndarray, bounds: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    For each left chunk, the half-open sub-range of *right* that can match it.

    Only right values within ``[first key, last key]`` of a chunk can pair with
    it, and an inner join discards everything else, so the sub-ranges are
    disjoint -- with gaps wherever *right* holds keys that *left* does not.
    Those gaps are the whole reason a barely overlapping join is cheap to split.

    Parameters
    ----------
    left, right : ndarray
        The join targets, monotonic increasing.
    bounds : ndarray of intp
        Split points in *left*, from one of the boundary functions.

    Returns
    -------
    rstart, rstop : ndarray of intp
        Half-open sub-range of *right* per chunk, one entry shorter than
        *bounds*.  ``rstart == rstop`` marks a chunk with nothing to match.
    """
    starts = bounds[:-1]
    stops = bounds[1:]
    rstart = right.searchsorted(left[starts], side="left")
    rstop = right.searchsorted(left[stops - 1], side="right")
    # searchsorted already returns intp; the cast only pins it for typing
    return rstart.astype(np.intp, copy=False), rstop.astype(np.intp, copy=False)


def _walk_cost(
    left: np.ndarray,
    right: np.ndarray,
    lstart: npt.NDArray[np.intp],
    lstop: npt.NDArray[np.intp],
    rstart: npt.NDArray[np.intp],
    rstop: npt.NDArray[np.intp],
) -> npt.NDArray[np.intp]:
    """
    Iterations the kernel takes over each of several sub-range pairs.

    The walk advances one side per iteration and stops as soon as either side
    runs out, so it never reaches keys above ``min(last left, last right)``.
    Charging both sides in full instead -- the obvious estimate -- is wrong
    whenever one side ends early, and wrong by enough to matter: a right side
    of a million rows that are all below every left key costs the walk a
    million steps, while the same rows split across chunks are clipped away and
    cost nothing.

    Applied to whole arrays it estimates the serial cost, and to chunks the
    parallel one, so the two are comparable by construction.

    Parameters
    ----------
    left, right : ndarray
        The join targets, monotonic increasing.
    lstart, lstop, rstart, rstop : ndarray of intp
        Half-open sub-ranges, one per chunk.

    Returns
    -------
    ndarray of intp
        Iterations per chunk; zero where either side of the pair is empty.
    """
    nonempty = (rstop > rstart) & (lstop > lstart)
    # last key each side offers, clamped so empty ranges stay in bounds
    lmax = left[np.maximum(lstop - 1, 0)]
    rmax = right[np.maximum(rstop - 1, 0)]
    hi = np.minimum(lmax, rmax)
    lwork = np.clip(left.searchsorted(hi, side="right"), lstart, lstop) - lstart
    rwork = np.clip(right.searchsorted(hi, side="right"), rstart, rstop) - rstart
    return np.where(nonempty, lwork + rwork, 0)


def _plan_chunks(
    left: np.ndarray, right: np.ndarray, n_chunks: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]] | None:
    """
    Choose how to split, or decline.

    Both boundary schemes are costed and the better taken; neither wins
    everywhere, for reasons given in each.

    The score is the estimated serial cost over the largest chunk's, which is
    the speedup a split can reach, since the largest chunk bounds the wall clock
    on its own.  Scoring a split against *itself* -- total chunk work over the
    largest chunk's share, a load-balance measure -- looks equivalent and is
    not: clipping removes right rows the serial walk would have had to scan, so
    one chunk doing a hundred iterations where serial does a million is a win
    however lopsided it looks.

    Parameters
    ----------
    left, right : ndarray
        The join targets.
    n_chunks : int
        How many chunks to aim for.

    Returns
    -------
    tuple of ndarray, or None
        ``(bounds, rstart, rstop)`` for the better split, or None when neither
        divides the work enough to pay for a pool.
    """

    whole = np.array([0], dtype=np.intp)
    serial = int(
        _walk_cost(
            left,
            right,
            whole,
            np.array([left.shape[0]], dtype=np.intp),
            whole,
            np.array([right.shape[0]], dtype=np.intp),
        )[0]
    )

    def score(
        bounds: npt.NDArray[np.intp],
    ) -> tuple[
        tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]] | None,
        float,
    ]:
        if bounds.shape[0] <= 2:
            # one run of equal keys spans everything; nothing to split
            return None, 0.0
        rstart, rstop = _chunk_ranges(left, right, bounds)
        work = _walk_cost(left, right, bounds[:-1], bounds[1:], rstart, rstop)
        # no chunk costs less than dispatching it, however little it clips down
        # to, or a join the serial walk finishes in one step would win a pool
        peak = max(int(work.max()), _DISPATCH_COST_ITERS)
        return (bounds, rstart, rstop), serial / peak

    even, even_score = score(_even_boundaries(left, n_chunks))
    # Even division is cheaper and usually good enough, so only pay for the
    # merge-path search when it leaves headroom.  k chunks cannot beat serial by
    # more than k, so this asks whether it is already within a tenth of that.
    # k is the count asked for, not what is left after snapping to keys: runs
    # of equal keys can collapse the even split to a few near-perfect chunks,
    # and merge-path may still find many more.
    if even is not None and even_score >= 0.9 * n_chunks:
        return even

    path, path_score = score(_merge_path_boundaries(left, right, n_chunks))
    if path_score > even_score:
        best, best_score = path, path_score
    else:
        best, best_score = even, even_score
    return best if best is not None and best_score >= _MIN_USEFUL_SPEEDUP else None


def inner_join_indexer(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Equivalent of ``libjoin.inner_join_indexer``, parallel for large inputs.

    The result is identical to the serial indexer for every input.  The
    parallel path is taken only where it is expected to pay off, and the serial
    one is used otherwise.

    Parameters
    ----------
    left, right : ndarray
        Monotonic increasing and free of NAs, which is what
        ``Index._can_use_libjoin`` guarantees of the callers.

    Returns
    -------
    result : ndarray
        The joined key values, of ``left.dtype``.
    lindexer, rindexer : ndarray of intp
        Positions in *left* and *right* of each emitted pair.

    See Also
    --------
    pandas._libs.join.inner_join_indexer : The serial equivalent.
    """
    if not _can_parallelize(left, right):
        return libjoin.inner_join_indexer(left, right)

    n_workers = max_workers(_MAX_DEFAULT_WORKERS)
    if n_workers < 2:
        return libjoin.inner_join_indexer(left, right)

    plan = _plan_chunks(left, right, _n_chunks(left.shape[0], n_workers))
    if plan is None:
        return libjoin.inner_join_indexer(left, right)
    bounds, rstart, rstop = plan

    # Clipping can leave chunks with no right rows at all; they emit nothing,
    # so they are dropped rather than dispatched.
    active = np.flatnonzero(rstop > rstart)
    if active.shape[0] == 0:
        return (
            np.empty(0, dtype=left.dtype),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )
    lviews = [left[bounds[i] : bounds[i + 1]] for i in active]
    rviews = [right[rstart[i] : rstop[i]] for i in active]
    lbases = bounds[active]
    rbases = rstart[active]

    if len(lviews) == 1:
        # one chunk has anything to match: the clipping still saves the
        # serial walk's scan, but there is no concurrency to buy with a pool
        return _count_and_fill(map, left.dtype, lviews, rviews, lbases, rbases)

    # The pool refuses work once interpreter shutdown begins, which can happen
    # mid-join; the refusal is kept so the handler below can tell it from any
    # other RuntimeError.
    refused: list[RuntimeError] = []
    try:
        # created and joined here, so no worker threads are live across a fork
        with ThreadPoolExecutor(max_workers=n_workers) as executor:

            def pool_map(
                fn: Callable[..., object], *iterables: Iterable[object]
            ) -> Iterator[object]:
                # executor.map, keeping the refusal
                try:
                    futures = [
                        executor.submit(fn, *args)
                        for args in zip(*iterables, strict=True)
                    ]
                except RuntimeError as err:
                    if interpreter_shutting_down():
                        refused.append(err)
                    raise
                return (future.result() for future in futures)

            return _count_and_fill(pool_map, left.dtype, lviews, rviews, lbases, rbases)
    except RuntimeError as err:
        if not refused or err is not refused[0]:
            raise
        # Shutdown began after max_workers looked, and the pool refused work
        # part-way through.  Leaving the ``with`` block waited for the chunks it
        # had accepted, and those only wrote to outputs the serial pass below
        # reallocates.
        return _count_and_fill(map, left.dtype, lviews, rviews, lbases, rbases)


def _count_and_fill(
    mapper: Callable[..., Iterator[object]],
    dtype: np.dtype,
    lviews: list[np.ndarray],
    rviews: list[np.ndarray],
    lbases: npt.NDArray[np.intp],
    rbases: npt.NDArray[np.intp],
) -> tuple[np.ndarray, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Count every chunk, allocate the output once, then fill each chunk's slice.

    Parameters
    ----------
    mapper : callable
        ``map`` to run the chunks inline, or an executor's ``map`` to run them
        on a pool.
    dtype : np.dtype
        Dtype of the joined key values.
    lviews, rviews : list of ndarray
        Each chunk's sub-ranges of left and right.
    lbases, rbases : ndarray of intp
        Where each chunk's sub-ranges start in the full arrays.

    Returns
    -------
    result : ndarray
    lindexer, rindexer : ndarray of intp
    """
    counts = np.fromiter(
        mapper(libjoin.inner_join_count_range, lviews, rviews),
        dtype=np.intp,
        count=len(lviews),
    )

    offsets = np.zeros(counts.shape[0], dtype=np.intp)
    np.cumsum(counts[:-1], out=offsets[1:])
    total = int(counts.sum())

    result = np.empty(total, dtype=dtype)
    lindexer = np.empty(total, dtype=np.intp)
    rindexer = np.empty(total, dtype=np.intp)

    def fill(i: int) -> None:
        # a slice per chunk, not a shared array and an offset: the kernel
        # writes with bounds checking off, and disjoint slices keep a chunk
        # that miscounted from reaching its neighbour's region
        lo = int(offsets[i])
        hi = lo + int(counts[i])
        written = libjoin.inner_join_fill_range(
            lviews[i],
            rviews[i],
            result[lo:hi],
            lindexer[lo:hi],
            rindexer[lo:hi],
            int(lbases[i]),
            int(rbases[i]),
        )
        if written != counts[i]:
            # the count and fill passes disagree, which can only happen if
            # the inputs changed underneath us
            raise ValueError(
                f"parallel join chunk {i} wrote {written} pairs, expected {counts[i]}"
            )

    # consume the iterator so worker exceptions propagate here
    list(mapper(fill, range(len(lviews))))

    return result, lindexer, rindexer
