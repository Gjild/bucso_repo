from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np


def db_to_lin(db: float) -> float:
    return 10 ** (db / 10.0)


def lin_to_db(lin: float) -> float:
    if lin <= 0:
        return -300.0
    return 10.0 * np.log10(lin)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def power_sum_db(levels_db: Iterable[float]) -> float:
    return lin_to_db(sum(db_to_lin(v) for v in levels_db if v is not None))


@dataclass(frozen=True)
class Band:
    center_hz: float
    bw_hz: float

    @property
    def f_lo(self) -> float:
        return self.center_hz - self.bw_hz / 2

    @property
    def f_hi(self) -> float:
        return self.center_hz + self.bw_hz / 2

    def contains(self, f: float) -> bool:
        return self.f_lo <= f <= self.f_hi


def coalesce_bins(freqs: np.ndarray, levels_db: np.ndarray, bin_width_hz: float):
    """
    Vectorized fixed-bin coalescing to avoid order-dependent clustering.
    Each frequency is assigned to bin index = floor(f / bin_width_hz).
    Returns (bin_representative_freqs, summed_levels_db).
    Deterministic & stable (mergesort on bin ids).
    """
    if freqs is None or levels_db is None:
        return np.array([]), np.array([])
    if len(freqs) == 0:
        return np.array([]), np.array([])

    freqs = np.asarray(freqs, float)
    levels_db = np.asarray(levels_db, float)

    bins = np.floor(freqs / bin_width_hz).astype(np.int64)
    order = np.argsort(bins, kind="mergesort")

    bins_s = bins[order]
    f_s = freqs[order]
    Llin_s = 10.0 ** (levels_db[order] / 10.0)

    # indices where a new bin starts
    start = np.empty_like(bins_s, dtype=bool)
    start[0] = True
    start[1:] = bins_s[1:] != bins_s[:-1]
    idx = np.flatnonzero(start)

    sums = np.add.reduceat(Llin_s, idx)
    # representative frequency per bin: lowest frequency within the bin (stable)
    f_rep = f_s[idx]

    return f_rep, 10.0 * np.log10(sums)