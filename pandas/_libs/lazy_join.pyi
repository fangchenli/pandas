import numpy as np

def build_join_table_i8(
    keys: np.ndarray,  # const int64_t[::1]
) -> tuple[
    np.ndarray,  # slot_key   int64
    np.ndarray,  # slot_gid   int64
    np.ndarray,  # counts     int64 (per group)
    np.ndarray,  # offsets    int64 (per group, +1)
    np.ndarray,  # group_rows int64 (right rows, grouped, right order)
]: ...
def probe_count_chunk(
    keys: np.ndarray,  # const int64_t[::1]
    lo: int,
    hi: int,
    slot_key: np.ndarray,
    slot_gid: np.ndarray,
    counts: np.ndarray,
) -> int: ...
def probe_fill_chunk(
    keys: np.ndarray,  # const int64_t[::1]
    lo: int,
    hi: int,
    slot_key: np.ndarray,
    slot_gid: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    group_rows: np.ndarray,
    out_left: np.ndarray,  # int64_t[::1]
    out_right: np.ndarray,  # int64_t[::1]
    out_start: int,
) -> None: ...
