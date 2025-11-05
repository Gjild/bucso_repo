from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .utils import Band, coalesce_bins
from .filters import IF2Parametric, RFBPF
from .mixer import Mixer
from .synth import LOSolution, LOCarrier


@dataclass
class BinEntry:
    __slots__ = ("f_rf_hz", "level_dbc", "inband", "info")
    f_rf_hz: float
    level_dbc: float
    inband: bool
    info: dict


@dataclass
class TileSummary:
    worst_margin_db: float
    bins: List[BinEntry]
    desired_rf_band: Band


def rbw_width(cfg_rb, bw_hz: float, freq_hz: float) -> float:
    return max(cfg_rb.rbw_hz, cfg_rb.rbw_frac_of_bw * bw_hz, cfg_rb.rbw_ppm_of_freq * freq_hz * 1e-6)


def desired_paths(tile_if1: Band, lo1: float, lo2: float, inj1: int, s2_sign: int) -> tuple[Band, Band]:
    """
    Desired path with explicit signs:
      Stage-1 desired: IF2 = | inj1*LO1  -  IF1 |
      Stage-2 desired:  RF  = | s2_sign*LO2  +  IF2 |
    Centers are magnitudes.
    """
    if2_center = abs(inj1 * lo1 - tile_if1.center_hz)
    if2 = Band(center_hz=if2_center, bw_hz=tile_if1.bw_hz)
    rf_center = abs(s2_sign * lo2 + if2.center_hz)
    rf = Band(center_hz=rf_center, bw_hz=if2.bw_hz)
    return if2, rf


def _in_if2_passband(f_hz: float, if2: IF2Parametric) -> bool:
    return abs(f_hz - if2.center_hz) <= (if2.bw_hz / 2.0)


def _interp_mask_flat_or_table(default_dbc: float, table, x: np.ndarray, *, is_oob_abs_freq: bool) -> np.ndarray:
    """
    Returns a vector of limits starting from default_dbc, optionally overridden by a table.
    For in-band: x is offset-from-edge (>=0). For OOB: x is absolute frequency.
    IMPORTANT: np.interp left/right must be scalars, not arrays.
    """
    limits = np.full_like(x, float(default_dbc), dtype=float)
    if not table:
        return limits
    try:
        axis = np.array([e.offset_hz for e in table], float)
        vals = np.array([e.limit_dbc for e in table], float)
    except Exception:
        # tolerate dict-like entries
        axis = np.array([e.get("offset_hz") for e in table], float)
        vals = np.array([e.get("limit_dbc") for e in table], float)
    if axis.size and vals.size:
        left = float(vals[0])
        right = float(vals[-1])
        lim = np.interp(x, axis, vals, left=left, right=right)
        return lim
    return limits


def enumerate_spurs(
    tile_if1: Band,
    lo1_sol: LOSolution,
    lo2_sol: LOSolution,
    if2win: IF2Parametric,
    rf_filter: RFBPF,
    mix1: Mixer,
    mix2: Mixer,
    cfg,
    inj1_sign: int,
    s2_sign: int,
    carriers_lo1: List[LOCarrier],
    carriers_lo2: List[LOCarrier],
    *,
    rf_center_override_hz: Optional[float] = None,
    rf_bw_override_hz: Optional[float] = None,
) -> TileSummary:
    """
    Main spur enumeration with band-affine widths:
      - LO family carriers + order-aware family scaling (both stages)
      - Stage-1 specials (LO1 feedthrough, IF1 leakage) propagated through Stage-2
      - LO2 feedthroughs (fundamental + harmonics) at RF
      - Skips the *desired* mechanism from being counted as a spur
      - Supports flat/tabled masks
      - Correct ± signs on IF terms in both stages
      - Optional rf_center/bw override to assess robustness to RF-request perturbations
      - OOB evaluation uses cfg RF band (S21 extrapolation allowed)
      - **Band-affine ΔA**: uses worst-of-edges attenuation deltas for conservative scoring
    """
    # Desired bands
    if2_des, rf_des_nom = desired_paths(tile_if1, lo1_sol.f_out_hz, lo2_sol.f_out_hz, inj1_sign, s2_sign)

    # Optional override for mask/inband reference (robustness sweeps)
    rf_des = rf_des_nom
    if rf_center_override_hz is not None:
        rf_des = Band(center_hz=float(rf_center_override_hz), bw_hz=rf_bw_override_hz or rf_des_nom.bw_hz)

    # RBW bin width
    bin_w = rbw_width(cfg.rbw_binning, rf_des.bw_hz, rf_des.center_hz)

    # Bind hot callables
    rf_attn = rf_filter.attn_at
    if2_attn = if2win.attn_at
    rej1 = mix1.rejection_dbc
    der1 = mix1.drive_derate_db
    rej2 = mix2.rejection_dbc
    der2 = mix2.drive_derate_db
    fam1 = mix1.family_scale_db
    fam2 = mix2.family_scale_db

    rf_attn_des = float(rf_attn(rf_des.center_hz))
    if2_attn_des = float(if2_attn(if2_des.center_hz))

    # RF span guard (use config RF band; S21 extrapolates beyond file)
    rf_lo = float(cfg.bands.rf_hz.min)
    rf_hi = float(cfg.bands.rf_hz.max)

    freq_list: list[float] = []
    level_list: list[float] = []

    def worst_delta(attn_func, center: float, half_width: float, des_attn: float) -> float:
        """
        Conservative ΔA: worst (largest) attenuation at the two band edges minus desired attenuation.
        """
        if half_width <= 0:
            return float(attn_func(abs(center))) - des_attn
        a_lo = float(attn_func(abs(center - half_width)))
        a_hi = float(attn_func(abs(center + half_width)))
        return max(a_lo, a_hi) - des_attn

    # Stage-1 spur indices (exclude n1=0 per model; treat images via signs)
    m1s = [m for m in range(-cfg.orders.m1n1_max_abs, cfg.orders.m1n1_max_abs + 1) if m != 0]
    n1s = range(1, cfg.orders.m1n1_max_abs + 1)

    for m1 in m1s:
        for n1 in n1s:
            base_L1 = rej1(m1, n1, lo1_sol.f_out_hz, tile_if1.center_hz) + der1(lo1_sol.delivered_dbm)
            for c1 in carriers_lo1:
                for sgn1 in (+1, -1):  # include ± on IF1 term
                    # Stage-1 band parameters
                    f_if2_c_raw = m1 * c1.freq_hz + sgn1 * n1 * tile_if1.center_hz
                    f_if2_c = abs(f_if2_c_raw)
                    w_if2 = abs(n1) * (tile_if1.bw_hz * 0.5)

                    # ΔA at IF2: worst of the edges minus desired
                    dA_if2 = float(worst_delta(if2_attn, f_if2_c_raw, w_if2, if2_attn_des))
                    # Family scaling (order-aware, capped)
                    L1 = base_L1 + fam1(abs(m1), c1.rel_dBc)

                    for m2 in range(-cfg.orders.m2n2_max_abs, cfg.orders.m2n2_max_abs + 1):
                        if m2 == 0:
                            continue
                        for n2 in range(1, cfg.orders.m2n2_max_abs + 1):
                            if abs(m1) + abs(n1) + abs(m2) + abs(n2) > cfg.orders.cross_stage_sum_max:
                                continue

                            base_L2 = rej2(m2, n2, lo2_sol.f_out_hz, f_if2_c) + der2(lo2_sol.delivered_dbm)
                            for c2 in carriers_lo2:
                                for sgn2 in (+1, -1):  # include ± on IF2 term for stage-2
                                    L2 = base_L2 + fam2(abs(m2), c2.rel_dBc)
                                    # Stage-2 band parameters
                                    f_rf_c_raw = m2 * c2.freq_hz + sgn2 * n2 * f_if2_c_raw
                                    f_rf_c = abs(f_rf_c_raw)
                                    w_rf = abs(n2) * w_if2

                                    # Skip outside a padded RF span for speed
                                    if (f_rf_c < rf_lo - 2 * bin_w) or (f_rf_c > rf_hi + 2 * bin_w):
                                        continue

                                    # *** Skip the exact desired mechanism counted as spur ***
                                    if (
                                        (m1 == inj1_sign)
                                        and (n1 == 1)
                                        and (c1.tag == "main")
                                        and (m2 == s2_sign)
                                        and (n2 == 1)
                                        and (c2.tag == "main")
                                        and (abs(f_rf_c - rf_des_nom.center_hz) <= bin_w * 0.5)
                                    ):
                                        continue

                                    # ΔA at RF: worst-of-edges minus desired
                                    dA_rf = float(worst_delta(rf_attn, f_rf_c_raw, w_rf, rf_attn_des))

                                    freq_list.append(f_rf_c)
                                    level_list.append(L1 + dA_if2 + L2 + dA_rf)

    # Stage-1 specials at IF2 output → propagate through Stage-2
    # LO1 feedthrough at IF2 out (treat as a tone; no band)
    f_lo1 = lo1_sol.f_out_hz
    L_lo1_if2 = mix1.mdl.isolation.lo_to_rf_db + (float(if2_attn(f_lo1)) - if2_attn_des)
    for m2 in range(-cfg.orders.m2n2_max_abs, cfg.orders.m2n2_max_abs + 1):
        if m2 == 0:
            continue
        for n2 in range(1, cfg.orders.m2n2_max_abs + 1):
            if abs(m2) + n2 > cfg.orders.cross_stage_sum_max:
                continue
            base_L2 = rej2(m2, n2, lo2_sol.f_out_hz, f_lo1) + der2(lo2_sol.delivered_dbm)
            for c2 in carriers_lo2:
                for sgn2 in (+1, -1):
                    L2 = base_L2 + fam2(abs(m2), c2.rel_dBc)
                    f_rf = abs(m2 * c2.freq_hz + sgn2 * n2 * (+f_lo1))
                    if (f_rf < rf_lo - 2 * bin_w) or (f_rf > rf_hi + 2 * bin_w):
                        continue
                    dA_rf = float(rf_attn(f_rf) - rf_attn_des)
                    freq_list.append(f_rf)
                    level_list.append(L_lo1_if2 + L2 + dA_rf)

    # IF1 leakage at IF2 out (center only)
    f_if1 = tile_if1.center_hz
    L_if1_if2 = mix1.mdl.isolation.if_to_rf_db + (float(if2_attn(f_if1)) - if2_attn_des)
    for m2 in range(-cfg.orders.m2n2_max_abs, cfg.orders.m2n2_max_abs + 1):
        if m2 == 0:
            continue
        for n2 in range(1, cfg.orders.m2n2_max_abs + 1):
            if abs(m2) + n2 > cfg.orders.cross_stage_sum_max:
                continue
            base_L2 = rej2(m2, n2, lo2_sol.f_out_hz, f_if1) + der2(lo2_sol.delivered_dbm)
            for c2 in carriers_lo2:
                for sgn2 in (+1, -1):
                    L2 = base_L2 + fam2(abs(m2), c2.rel_dBc)
                    f_rf = abs(m2 * c2.freq_hz + sgn2 * n2 * (+f_if1))
                    if (f_rf < rf_lo - 2 * bin_w) or (f_rf > rf_hi + 2 * bin_w):
                        continue
                    dA_rf = float(rf_attn(f_rf) - rf_attn_des)
                    freq_list.append(f_rf)
                    level_list.append(L_if1_if2 + L2 + dA_rf)

    # LO2 feedthroughs at RF: include main + harmonics (use isolation plus carrier rel_dBc)
    for c2 in carriers_lo2:
        f = c2.freq_hz
        if (rf_lo - 2 * bin_w) <= f <= (rf_hi + 2 * bin_w):
            L = mix2.mdl.isolation.lo_to_rf_db + c2.rel_dBc + (float(rf_attn(f)) - rf_attn_des)
            freq_list.append(f)
            level_list.append(L)

    # Coalesce by RBW window
    freqs = np.asarray(freq_list, dtype=float)
    levs = np.asarray(level_list, dtype=float)
    cf, cL = coalesce_bins(freqs, levs, bin_w)

    # Margins vs masks (flat or offset-dependent)
    inband_mask = np.abs(cf - rf_des.center_hz) <= (rf_des.bw_hz / 2.0 + bin_w / 2.0)

    # In-band limits: default or table vs offset-from-edge (>=0 inside band is 0)
    edge = rf_des.bw_hz / 2.0
    offsets_edge = np.clip(np.abs(cf - rf_des.center_hz) - edge, 0.0, None)
    lim_in = _interp_mask_flat_or_table(
        cfg.masks.inband.default_dbc, cfg.masks.inband.table, offsets_edge, is_oob_abs_freq=False
    )

    # Out-of-band limits: default or table vs absolute frequency
    lim_oob = _interp_mask_flat_or_table(
        cfg.masks.outofband.default_dbc, cfg.masks.outofband.table, cf, is_oob_abs_freq=True
    )

    limits = np.where(inband_mask, lim_in, lim_oob).astype(float)
    margins = limits - cL - cfg.constraints.guard_margin_db
    worst = float(np.min(margins)) if margins.size else -999.0

    final_bins: list[BinEntry] = [
        BinEntry(float(f), float(L), bool(inb), {"limit_dbc": float(lim), "margin_db": float(mar)})
        for f, L, inb, lim, mar in zip(cf, cL, inband_mask, limits, margins)
    ]

    return TileSummary(worst_margin_db=worst, bins=final_bins, desired_rf_band=rf_des)
