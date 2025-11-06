from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
import yaml
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
    def _from_arrays(cls, freq_hz: np.ndarray, s21_db: np.ndarray, ident: str) -> "RFBPF":
        # Ensure sorted by frequency for deterministic, correct interpolation
        order = np.argsort(freq_hz, kind="mergesort")
        f = np.asarray(freq_hz, dtype=float)[order]
        s = np.asarray(s21_db, dtype=float)[order]
        obj = cls(freq_hz=f, s21_db=s, id=ident)
        # Build interpolator once (log-frequency, linear dB) with extrapolation
        # Guard log10(0) via clamp inside attn_at()
        obj._x = np.log10(np.maximum(obj.freq_hz, 1.0))
        obj._y = obj.s21_db
        obj._interp = interp1d(
            obj._x, obj._y,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        return obj

    @classmethod
    def from_csv(cls, path: str) -> "RFBPF":
        df = pd.read_csv(path)
        f = df["freq_hz"].to_numpy(dtype=float)
        s = df["s21_db"].to_numpy(dtype=float)
        return cls._from_arrays(f, s, ident=path)

    @classmethod
    def from_yaml(cls, path: str) -> "RFBPF":
        y = yaml.safe_load(open(path, "r"))
        # Support either dict with lists or list of {freq_hz, s21_db}
        if isinstance(y, dict) and "freq_hz" in y and "s21_db" in y:
            f = np.array(y["freq_hz"], dtype=float)
            s = np.array(y["s21_db"], dtype=float)
        elif isinstance(y, list):
            f = np.array([row["freq_hz"] for row in y], dtype=float)
            s = np.array([row["s21_db"] for row in y], dtype=float)
        else:
            raise ValueError(f"Unsupported RF BPF YAML format in {path!r}")
        return cls._from_arrays(f, s, ident=path)

    @classmethod
    def from_path(cls, path: str) -> "RFBPF":
        p = str(path).lower()
        if p.endswith(".csv"):
            return cls.from_csv(path)
        if p.endswith(".yaml") or p.endswith(".yml"):
            return cls.from_yaml(path)
        # Best effort: try CSV first, then YAML
        try:
            return cls.from_csv(path)
        except Exception:
            return cls.from_yaml(path)

    def attn_at(self, f_hz):
        """
        Return attenuation at f_hz (Hz). Accepts scalar or ndarray.
        Always returns an ndarray of dtype float (shape follows input).

        Robust to f_hz <= 0 by clamping to 1 Hz before log10().
        """
        fx = np.asarray(f_hz, dtype=float)
        # Avoid log10(0) / negatives by clamping to >= 1 Hz
        fx = np.maximum(fx, 1.0)
        fx = np.log10(fx)
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
