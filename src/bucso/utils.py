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
    """Return (bin_centers, summed_levels_db) with simple sliding coalescing."""
    if freqs is None or levels_db is None or len(freqs) == 0:
        return np.array([]), np.array([])
    order = np.argsort(freqs)
    freqs = freqs[order]
    levels_db = levels_db[order]
    bins_f = [freqs[0]]
    bins_lin = [db_to_lin(levels_db[0])]
    last_f = freqs[0]
    for f, L in zip(freqs[1:], levels_db[1:]):
        if abs(f - last_f) <= bin_width_hz:
            bins_lin[-1] += db_to_lin(L)
        else:
            bins_f.append(f)
            bins_lin.append(db_to_lin(L))
        last_f = f
    return np.array(bins_f), np.array([lin_to_db(v) for v in bins_lin])
