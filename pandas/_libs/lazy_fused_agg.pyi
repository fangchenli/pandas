import numpy as np

def fused_filter_aggs(
    i64_cols: list[np.ndarray],
    i64_lo: np.ndarray,
    i64_hi: np.ndarray,
    f64_cols: list[np.ndarray],
    f64_lo: np.ndarray,
    f64_hi: np.ndarray,
    agg_kinds: np.ndarray,
    agg_a: list[np.ndarray],
    agg_b: list[np.ndarray | None],
    start: int,
    end: int,
) -> tuple[np.ndarray, int]: ...
