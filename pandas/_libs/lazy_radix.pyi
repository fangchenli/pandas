import numpy as np

def radix_argsort_u64(
    keys_in: np.ndarray,  # const uint64_t[::1]
) -> np.ndarray: ...  # np.ndarray[np.int64]
