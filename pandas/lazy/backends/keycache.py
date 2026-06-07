"""
Dictionary-encoding cache for string aggregation keys.

acero hash-aggregates dictionary keys ~5x faster than raw strings
(13 ms vs 67 ms at 10M rows), but dictionary-encoding costs ~84 ms —
losing for a single query, winning from the second on. This cache
amortizes the encode across queries against the same source column.

Safety rests on Arrow immutability: a ``pa.ChunkedArray`` /
``pa.Array`` never changes after construction, so an identity-keyed
entry can never serve stale data. Entries hold a strong reference to
the source array, which both keeps the dictionary's parent alive and
prevents ``id()`` reuse; the cache is a small bounded LRU so the
strong references cannot grow unboundedly.

Policy: encode on the SECOND occurrence of the same key array at or
above MIN_ENCODE_ROWS. One-shot queries never pay the encode; repeated
workloads (dashboards, notebooks re-running cells, benchmark loops)
hit the fast path from the second query on.
"""

from __future__ import annotations

from collections import OrderedDict

import pyarrow as pa
import pyarrow.compute as pc

MIN_ENCODE_ROWS = 1_000_000
MAX_ENTRIES = 32

# Identity is keyed on BUFFER ADDRESSES, not Python object identity:
# pa.Table.column() returns a fresh Python wrapper per call around the
# same immutable C++ buffers, so id() differs across queries while the
# data is provably identical. A buffer address cannot be recycled while
# an entry pins its source array, so (length, first-buffer-address) is
# collision-free for live entries.
_cache: OrderedDict[tuple, tuple[object, object]] = OrderedDict()
_seen: OrderedDict[tuple, object] = OrderedDict()


def _identity_key(arr) -> tuple | None:
    first = arr.chunk(0) if isinstance(arr, pa.ChunkedArray) else arr
    if len(first) == 0:
        return None
    addr = next((b.address for b in first.buffers() if b is not None), None)
    if addr is None:
        return None
    return (len(arr), addr)


def maybe_dictionary_encode(arr):
    """Return a dictionary-encoded version of a string key array when
    the cache policy says it pays; otherwise return ``arr`` unchanged.
    """
    if not isinstance(arr, (pa.Array, pa.ChunkedArray)):
        return arr
    if not (pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)):
        return arr
    if pa.types.is_dictionary(arr.type):
        return arr
    if len(arr) < MIN_ENCODE_ROWS:
        return arr
    if isinstance(arr, pa.ChunkedArray) and arr.num_chunks == 0:
        return arr

    key = _identity_key(arr)
    if key is None:
        return arr

    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit[1]

    if key not in _seen:
        # First occurrence: register the candidate, pinning the source
        # so the buffer address stays valid for the identity check.
        _seen[key] = arr
        while len(_seen) > MAX_ENTRIES:
            _seen.popitem(last=False)
        return arr

    # Second occurrence: encode and cache.
    encoded = pc.dictionary_encode(arr)
    source = _seen.pop(key)
    _cache[key] = (source, encoded)
    while len(_cache) > MAX_ENTRIES:
        _cache.popitem(last=False)
    return encoded


def cache_stats() -> dict[str, int]:
    """Introspection for tests and diagnostics."""
    return {"encoded": len(_cache), "candidates": len(_seen)}


def clear_cache() -> None:
    _cache.clear()
    _seen.clear()
