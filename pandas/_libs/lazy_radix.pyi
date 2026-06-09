import numpy as np

def radix_argsort_u64(
    keys_in: np.ndarray,  # const uint64_t[::1]
) -> np.ndarray: ...  # np.ndarray[np.int64]
def histogram_chunk(
    keys: np.ndarray,  # const uint64_t[::1]
    lo: int,
    hi: int,
    shift: int,
    mask: int,
    hist: np.ndarray,  # int64_t[::1]
) -> None: ...
def scatter_chunk(
    keys: np.ndarray,  # const uint64_t[::1]
    idx: np.ndarray,  # const int64_t[::1]
    lo: int,
    hi: int,
    shift: int,
    mask: int,
    cursor: np.ndarray,  # int64_t[::1]
    kb: np.ndarray,  # uint64_t[::1]
    ib: np.ndarray,  # int64_t[::1]
) -> None: ...
