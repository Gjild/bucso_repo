from __future__ import annotations
from typing import Dict
from functools import lru_cache
import numpy as np
from .models import MixerModel
from .utils import clamp


def _bilinear_clamped(z: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, x: float, y: float) -> float:
    """
    Bilinear interpolation over a rect grid with clamp-at-edges behavior.
    z shape must be (len(x_axis), len(y_axis)) in the same orientation used in the YAML.
    """
    # Validate shape early to avoid silent mis-indexing
    if z.shape != (len(x_axis), len(y_axis)):
        raise ValueError(
            f"Mixer spur grid shape {z.shape} does not match axes "
            f"({len(x_axis)}, {len(y_axis)}). Check YAML grids."
        )
    # map value->index space (scalar)
    xi = np.interp(x, x_axis, np.arange(len(x_axis)))
    yi = np.interp(y, y_axis, np.arange(len(y_axis)))
    i0 = int(clamp(np.floor(xi + 1e-9), 0, len(x_axis) - 2))
    j0 = int(clamp(np.floor(yi + 1e-9), 0, len(y_axis) - 2))
    tx = xi - i0
    ty = yi - j0
    return float(
        z[i0, j0] * (1 - tx) * (1 - ty)
        + z[i0 + 1, j0] * (tx) * (1 - ty)
        + z[i0, j0 + 1] * (1 - tx) * (ty)
        + z[i0 + 1, j0 + 1] * (tx) * (ty)
    )


class Mixer:
    """
    Mixer rejection lookup with:
      - scalar entries for specific (m,n)
      - optional per-(m,n) grids (grids_by_order: {"m,n": {lo_hz, if_hz, rej_dbc}})
      - optional legacy single grid assumed to be (1,1)
      - configurable fallback for unknown orders
    Also provides simple drive-derate and order-aware LO family scaling.
    """

    def __init__(self, mdl: MixerModel):
        self.mdl = mdl

        st: Dict[str, object] = mdl.spur_table or {}

        # New: per-(m,n) grids
        self._grids_by_order: Dict[str, Dict[str, np.ndarray]] = {}
        if "grids_by_order" in st and isinstance(st["grids_by_order"], dict):
            for key, g in st["grids_by_order"].items():
                lo = np.array((g.get("lo_hz") or []), float)
                ifv = np.array((g.get("if_hz") or []), float)
                rej = np.array((g.get("rej_dbc") or []), float)
                if rej.ndim != 2:
                    raise ValueError(f"Mixer '{mdl.name}' per-order grid rej_dbc must be 2D; got shape {rej.shape}.")
                # shape validation will be re-checked in _bilinear_clamped
                self._grids_by_order[str(key)] = {"lo": lo, "if": ifv, "rej": rej}

        # Legacy: a single grid interpreted as the (1,1) grid.
        self._grid_default = None
        if "grids" in st and isinstance(st["grids"], dict):
            g = st["grids"]
            lo = np.array((g.get("lo_hz") or []), float)
            ifv = np.array((g.get("if_hz") or []), float)
            rej = np.array((g.get("rej_dbc") or []), float)
            if rej.size and (rej.ndim != 2):
                raise ValueError(f"Mixer '{mdl.name}' legacy grid rej_dbc must be 2D; got shape {rej.shape}.")
            self._grid_default = {"lo": lo, "if": ifv, "rej": rej}

        # Scalar entries
        self._entries = st.get("entries", []) or []

        # Quantization for cache keys (Hz)
        self._q_hz = 1e6

    def set_cache_quantum_hz(self, q_hz: float):
        self._q_hz = float(q_hz)
        self._rej_cached.cache_clear()

    def clear_cache(self):
        self._rej_cached.cache_clear()

    @lru_cache(maxsize=262144)
    def _rej_cached(self, m: int, n: int, lo_q: float, if_q: float) -> float:
        # 1) Exact scalar entry wins
        for e in self._entries:
            if e.get("m") == m and e.get("n") == n:
                return float(e["rej_dbc"])

        # 2) Per-(m,n) grid if present
        key = f"{int(m)},{int(n)}"
        grid = self._grids_by_order.get(key)
        if grid and grid["rej"].size and grid["lo"].size and grid["if"].size:
            return _bilinear_clamped(grid["rej"], grid["lo"], grid["if"], lo_q, if_q)

        # 3) Legacy default grid only for (1,1)
        if (m == 1 and n == 1) and (self._grid_default is not None):
            gd = self._grid_default
            if gd["rej"].size and gd["lo"].size and gd["if"].size:
                return _bilinear_clamped(gd["rej"], gd["lo"], gd["if"], lo_q, if_q)

        # 4) Conservative fallback
        return float(self.mdl.fallback_rej_dbc)

    def rejection_dbc(self, m: int, n: int, lo_hz: float, if_hz: float) -> float:
        q = self._q_hz
        return self._rej_cached(m, n, round(lo_hz / q) * q, round(if_hz / q) * q)

    def drive_derate_db(self, delivered_dbm: float) -> float:
        reqmin = self.mdl.required_lo_drive_dbm["min"]
        if delivered_dbm >= reqmin:
            return 0.0
        delta = reqmin - delivered_dbm
        slope = self.mdl.drive_derate.slope_db_per_db
        cap = self.mdl.drive_derate.max_derate_db
        return min(cap, slope * delta)

    def family_scale_db(self, order_abs: int, lo_rel_dBc: float) -> float:
        slope = self.mdl.lo_family_scaling.get("default_slope_db_per_db", 1.0)
        cap = self.mdl.lo_family_scaling.get("cap_db", 12.0)
        val = slope * abs(order_abs) * lo_rel_dBc
        return max(-cap, min(cap, val))
