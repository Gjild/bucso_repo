from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional
import numpy as np
from .models import LOMdl, Mode


@dataclass(frozen=True)
class LOCarrier:
    freq_hz: float
    rel_dBc: float  # relative to main tone at mixer input
    tag: str        # "main", "harm2", "pfd_k1", ...


@dataclass
class LOSolution:
    name: str
    mode: str
    f_out_hz: float
    divider: str
    delivered_dbm: float
    lock_time_ms: float
    pad_db: float  # which pad choice produced delivered_dbm (for legality)
    # PFD info (P2): derived from ref and pfd_divider legality
    pfd_hz: float
    pfd_divider: int


class Synth:
    def __init__(self, mdl: LOMdl):
        self.mdl = mdl

    def output_power_dbm(self, f_out_hz: float, divider: str) -> float:
        t = self.mdl.output_power_model.table
        fv = np.array(t["freq_hz"], float)
        pv = np.array(t["p_out_dbm"], float)
        p = float(np.interp(f_out_hz, fv, pv))
        p += self.mdl.output_power_model.divider_adjust_db.get(divider, 0.0)
        return p

    def delivered_drive_dbm(self, f_out_hz: float, divider: str, path: str, pad_db: float) -> float:
        p = self.output_power_dbm(f_out_hz, divider)
        loss = self.mdl.distribution.path_losses_db.get(path, 0.0)
        return p - loss - pad_db

    def equivalent_carriers(
        self,
        f_out_hz: float,
        mode: Mode,
        divider: str,
        pfd_hz: float | None = None,
    ) -> List[LOCarrier]:
        """
        Build equivalent LO carriers at the *output* (post-divider) with relative levels.
        Apply divider-spectrum harmonic deltas to harmonics.
        """
        carriers: List[LOCarrier] = [LOCarrier(f_out_hz, 0.0, "main")]

        harm_delta = self.mdl.divider_spectrum.get(divider, None)
        harm_boost = 0.0 if harm_delta is None else float(harm_delta.harm_delta_dBc)

        # Harmonics at output (datasheet gives relative at output already).
        for h in (mode.harmonics_at_output or []):
            # Apply divider-spectrum delta to reflect folding behavior
            rel = float(h.rel_dBc) + harm_boost
            carriers.append(LOCarrier(h.k * f_out_hz, rel, f"harm{h.k}"))

        # Basic PFD families, if present.
        fams = None
        if isinstance(mode.pfd_spurs_at_output, dict):
            fams = mode.pfd_spurs_at_output.get("families", [])
        if fams and pfd_hz and pfd_hz > 0:
            for fam in fams:
                for comp in fam.components:
                    for sgn in (+1, -1):
                        f = f_out_hz + sgn * comp.k * float(pfd_hz)
                        carriers.append(LOCarrier(f, float(comp.base_rel_dBc), f"pfd{comp.k}"))

        # Optional fractional-N boundary spur envelope (crude envelope)
        env = getattr(mode, "frac_boundary_spurs", None)
        if isinstance(env, dict) and env.get("enabled", False) and pfd_hz and pfd_hz > 0:
            amp = float(env.get("amplitude_at_eps0p5_rel_dBc", -58))
            for sgn in (+1, -1):
                carriers.append(LOCarrier(f_out_hz + sgn * 0.5 * float(pfd_hz), amp, "frac_boundary"))

        return carriers

    def _best_pad_within_drive(
        self,
        f_out_hz: float,
        divider: str,
        path: str,
        min_dbm: float,
        max_dbm: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Return (pad_db, delivered_dbm) choosing a pad that keeps delivered drive within [min,max].
        If not possible, return None.
        """
        best: Optional[Tuple[float, float]] = None
        for pad in self.mdl.distribution.pad_options_db:
            delivered = self.delivered_drive_dbm(f_out_hz, divider, path, pad)
            if (delivered >= min_dbm) and (delivered <= max_dbm):
                # Choose the *largest* pad that still meets min (to keep headroom)
                if (best is None) or (pad > best[0]):
                    best = (float(pad), float(delivered))
        return best

    def _choose_pfd(self, mode: Mode) -> tuple[float, int] | None:
        """Pick a legal PFD given ref constraints and mode pfd_dividers and range (choose largest PFD within range)."""
        refs = list(self.mdl.ref_constraints.get("allowed_refs_hz", []) or [])
        if not refs:
            return None
        # Highest PFD within range is often better (fewer frac spurs)
        best: tuple[float, int] | None = None
        for ref in refs:
            for div in (mode.pfd_dividers or []):
                f_pfd = float(ref) / float(div)
                lo, hi = float(mode.pfd_hz_range[0]), float(mode.pfd_hz_range[1])
                if lo <= f_pfd <= hi:
                    if (best is None) or (f_pfd > best[0]):
                        best = (f_pfd, int(div))
        return best

    def legal_settings(
        self, *, name_filter: Iterable[str] | None, f_min: float, f_max: float, path: str,
        drive_min_dbm: float | None = None, drive_max_dbm: float | None = None
    ) -> List[LOSolution]:
        """Enumerate legal LO grid points (post-divider) with available VCO dividers from mode.vco_dividers.
        Enforce PFD legality vs allowed refs and mode pfd_hz_range.
        """
        out: List[LOSolution] = []
        step = float(self.mdl.step_hz)
        f0 = max(self.mdl.freq_range_hz[0], f_min)
        f1 = min(self.mdl.freq_range_hz[1], f_max)
        if f0 > f1:
            return out

        for mode in self.mdl.modes:
            if not mode.enabled:
                continue

            pfd_sel = self._choose_pfd(mode)
            if pfd_sel is None:
                continue
            f_pfd, pfd_divider = pfd_sel

            vco_divs = mode.vco_dividers or [1]
            for vdiv in vco_divs:
                divider = f"/{int(vdiv)}"
                f = float(np.ceil(f0 / step) * step)
                while f <= f1 + 1e-9:
                    if (drive_min_dbm is not None) and (drive_max_dbm is not None):
                        pad_sel = self._best_pad_within_drive(f, divider, path, drive_min_dbm, drive_max_dbm)
                        if pad_sel is None:
                            f += step
                            continue
                        pad_db, delivered = pad_sel
                    else:
                        # fall back: pick pad giving maximum delivered
                        pad_db = max(self.mdl.distribution.pad_options_db or [0.0])
                        delivered = self.delivered_drive_dbm(f, divider, path, pad_db)
                    out.append(LOSolution(
                        name=self.mdl.name, mode=mode.name, f_out_hz=f,
                        divider=divider, delivered_dbm=float(delivered),
                        lock_time_ms=float(mode.lock_time_model.base_ms),
                        pad_db=float(pad_db),
                        pfd_hz=float(f_pfd), pfd_divider=int(pfd_divider)
                    ))
                    f += step
        return out

    def snap_to_legal(
        self, f_target: float, path: str, window_steps: int = 2,
        drive_min_dbm: float | None = None, drive_max_dbm: float | None = None
    ) -> LOSolution | None:
        """Snap a target to nearest legal grid ±window_steps over all modes/dividers, enforcing PFD legality."""
        step = float(self.mdl.step_hz)
        if step <= 0:
            return None
        grid = [round(f_target / step) * step + k * step for k in range(-window_steps, window_steps + 1)]

        best: Optional[LOSolution] = None
        best_del = -1e9

        for mode in self.mdl.modes:
            if not mode.enabled:
                continue
            pfd_sel = self._choose_pfd(mode)
            if pfd_sel is None:
                continue
            f_pfd, pfd_divider = pfd_sel

            vco_divs = mode.vco_dividers or [1]
            for vdiv in vco_divs:
                divider = f"/{int(vdiv)}"
                for f in grid:
                    if not (self.mdl.freq_range_hz[0] <= f <= self.mdl.freq_range_hz[1]):
                        continue
                    if (drive_min_dbm is not None) and (drive_max_dbm is not None):
                        pad_sel = self._best_pad_within_drive(f, divider, path, drive_min_dbm, drive_max_dbm)
                        if pad_sel is None:
                            continue
                        pad_db, delivered = pad_sel
                    else:
                        pad_db = max(self.mdl.distribution.pad_options_db or [0.0])
                        delivered = self.delivered_drive_dbm(f, divider, path, pad_db)

                    if delivered > best_del:
                        best_del = delivered
                        best = LOSolution(
                            name=self.mdl.name, mode=mode.name, f_out_hz=float(f),
                            divider=divider, delivered_dbm=float(delivered),
                            lock_time_ms=float(mode.lock_time_model.base_ms),
                            pad_db=float(pad_db),
                            pfd_hz=float(f_pfd), pfd_divider=int(pfd_divider)
                        )
        return best
