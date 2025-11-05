from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass
class RFBPF:
    freq_hz: np.ndarray
    s21_db: np.ndarray
    id: str

    # cached members (not shown in repr)
    _x: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)
    _interp: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)

    @classmethod
    def from_csv(cls, path: str) -> "RFBPF":
        df = pd.read_csv(path)
        # Ensure sorted by frequency for deterministic, correct interpolation
        df = df.sort_values("freq_hz", kind="mergesort")
        f = df["freq_hz"].to_numpy(dtype=float)
        s = df["s21_db"].to_numpy(dtype=float)
        obj = cls(freq_hz=f, s21_db=s, id=path)
        # Build interpolator once (log-frequency, linear dB) with extrapolation
        obj._x = np.log10(obj.freq_hz)
        obj._y = obj.s21_db
        obj._interp = interp1d(
            obj._x, obj._y,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        return obj

    def attn_at(self, f_hz):
        """
        Return attenuation at f_hz (Hz). Accepts scalar or ndarray.
        Always returns an ndarray of dtype float (shape follows input).
        NOTE: Do NOT clamp here; we rely on extrapolation of the log-frequency interpolator.
        """
        fx = np.log10(np.asarray(f_hz, dtype=float))
        return np.asarray(self._interp(fx), dtype=float)


@dataclass
class IF2Parametric:
    center_hz: float
    bw_hz: float
    passband_il_db: float
    stop_floor_db: float
    rolloff_db_per_dec: float  # positive number (magnitude)
    # symmetric_powerlaw

    def attn_at(self, f_hz: float) -> float:
        """
        Flat IL inside passband; outside, slope to floor (negative).
        (Scalar implementation — cheap and fine for current call sites.)
        """
        edge = self.bw_hz / 2.0
        df = abs(f_hz - self.center_hz)
        if df <= edge:
            return -abs(self.passband_il_db)
        # decades beyond edge
        decades = np.log10(max(df / edge, 1e-9))
        attn = -abs(self.passband_il_db) - self.rolloff_db_per_dec * decades
        return max(attn, self.stop_floor_db)

    def contains_desired(self, center: float, bw: float) -> bool:
        # require desired band fully inside passband rectangle
        return (abs(center - self.center_hz) + bw / 2.0) <= (self.bw_hz / 2.0)
