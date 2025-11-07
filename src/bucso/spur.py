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


def _interp_mask_flat_or_table(default_dbc: float, table, x: np.ndarray) -> np.ndarray:
    """
    Returns a vector of limits starting from default_dbc, optionally overridden by a table.
    x can be absolute frequency OR offset-from-edge depending on caller.
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
    Spur enumeration with full provenance per mechanism and component-aware RBW coalescing.

    For every RF bin we now store:
      - combined level (dBc)
      - inband flag, mask limit, final margin
      - components[]: each with family, per-stage orders, LO carrier tags, filter deltas,
                      mixer rejection sources (scalar / grid_mn / grid_11_legacy / fallback)
      - summary flags: families_present[], has_missing_table_data (any fallback), etc. (in info)
    """
    # Desired bands (with optional RF override for robustness)
    if2_des, rf_des_nom = desired_paths(tile_if1, lo1_sol.f_out_hz, lo2_sol.f_out_hz, inj1_sign, s2_sign)
    rf_des = rf_des_nom if rf_center_override_hz is None else Band(float(rf_center_override_hz), rf_bw_override_hz or rf_des_nom.bw_hz)

    # RBW bin width
    bin_w = rbw_width(cfg.rbw_binning, rf_des.bw_hz, rf_des.center_hz)

    # Hot callables
    rf_attn = rf_filter.attn_at
    if2_attn = if2win.attn_at
    rf_attn_des = float(rf_attn(rf_des.center_hz))
    if2_attn_des = float(if2_attn(if2_des.center_hz))

    # RF span guard
    rf_lo = float(cfg.bands.rf_hz.min)
    rf_hi = float(cfg.bands.rf_hz.max)

    # Conservative ΔA helper
    def worst_delta(attn_func, center_raw: float, half_width: float, des_attn: float) -> float:
        if half_width <= 0:
            return float(attn_func(abs(center_raw))) - des_attn
        a_lo = float(attn_func(abs(center_raw - half_width)))
        a_hi = float(attn_func(abs(center_raw + half_width)))
        return max(a_lo, a_hi) - des_attn

    # Collect raw mechanisms before binning
    # Each element: (f_rf_abs, level_dbc, component_dict)
    mechanisms: list[tuple[float, float, dict]] = []

    # --- Stage-1 × Stage-2 mixing families -----------------------------------
    m1s = [m for m in range(-cfg.orders.m1n1_max_abs, cfg.orders.m1n1_max_abs + 1) if m != 0]
    n1s = range(1, cfg.orders.m1n1_max_abs + 1)
    m2s = [m for m in range(-cfg.orders.m2n2_max_abs, cfg.orders.m2n2_max_abs + 1) if m != 0]
    n2s = range(1, cfg.orders.m2n2_max_abs + 1)

    for m1 in m1s:
        for n1 in n1s:
            rej1_nom, src1 = mix1.rejection_with_meta(m1, n1, lo1_sol.f_out_hz, tile_if1.center_hz)
            base_L1 = rej1_nom + mix1.drive_derate_db(lo1_sol.delivered_dbm)
            for c1 in carriers_lo1:
                fam1_term = mix1.family_scale_db(abs(m1), c1.rel_dBc)
                for sgn1 in (+1, -1):  # sign on IF1 term
                    f_if2_raw = m1 * c1.freq_hz + sgn1 * n1 * tile_if1.center_hz
                    f_if2_c = abs(f_if2_raw)
                    w_if2 = abs(n1) * (tile_if1.bw_hz * 0.5)
                    dA_if2 = float(worst_delta(if2_attn, f_if2_raw, w_if2, if2_attn_des))
                    L1_eff = base_L1 + fam1_term

                    for m2 in m2s:
                        for n2 in n2s:
                            if (abs(m1) + abs(n1) + abs(m2) + abs(n2)) > cfg.orders.cross_stage_sum_max:
                                continue

                            rej2_nom, src2 = mix2.rejection_with_meta(m2, n2, lo2_sol.f_out_hz, f_if2_c)
                            base_L2 = rej2_nom + mix2.drive_derate_db(lo2_sol.delivered_dbm)
                            for c2 in carriers_lo2:
                                fam2_term = mix2.family_scale_db(abs(m2), c2.rel_dBc)
                                for sgn2 in (+1, -1):
                                    f_rf_raw = m2 * c2.freq_hz + sgn2 * n2 * f_if2_raw
                                    f_rf = abs(f_rf_raw)
                                    if (f_rf < rf_lo - 2 * bin_w) or (f_rf > rf_hi + 2 * bin_w):
                                        continue

                                    # Skip exact desired first-order (m,n)=(±1,1) on main carriers inside the desired bin
                                    if (
                                        (abs(m1) == 1) and (n1 == 1) and (c1.tag == "main") and
                                        (abs(m2) == 1) and (n2 == 1) and (c2.tag == "main") and
                                        (abs(f_rf - rf_des_nom.center_hz) <= bin_w * 0.5)
                                    ):
                                        continue

                                    dA_rf = float(worst_delta(rf_attn, f_rf_raw, abs(n2) * w_if2, rf_attn_des))
                                    L_total = L1_eff + dA_if2 + (base_L2 + fam2_term) + dA_rf

                                    comp = {
                                        "family": "mixing",
                                        "stage1": {
                                            "m": m1, "n": n1, "sign_if": sgn1,
                                            "lo1_carrier_tag": c1.tag, "lo1_rel_dBc": float(c1.rel_dBc),
                                            "rej_dbc": float(rej1_nom), "rej_source": src1,
                                            "drive_derate_db": float(mix1.drive_derate_db(lo1_sol.delivered_dbm)),
                                            "family_scale_db": float(fam1_term),
                                        },
                                        "stage2": {
                                            "m": m2, "n": n2, "sign_if": sgn2,
                                            "lo2_carrier_tag": c2.tag, "lo2_rel_dBc": float(c2.rel_dBc),
                                            "rej_dbc": float(rej2_nom), "rej_source": src2,
                                            "drive_derate_db": float(mix2.drive_derate_db(lo2_sol.delivered_dbm)),
                                            "family_scale_db": float(fam2_term),
                                        },
                                        "dA_if2_db": float(dA_if2),
                                        "dA_rf_db": float(dA_rf),
                                        "f_rf_hz_raw": float(f_rf_raw),
                                        "f_if2_hz_raw": float(f_if2_raw),
                                    }
                                    mechanisms.append((f_rf, float(L_total), comp))

    # --- Stage-1 specials propagated ------------------------------------------------
    # LO1 feedthrough at IF2 output
    f_lo1 = lo1_sol.f_out_hz
    dA_if2_lo = float(if2_attn(f_lo1) - if2_attn_des)
    L_lo1_if2 = mix1.mdl.isolation.lo_to_rf_db + dA_if2_lo
    for m2 in m2s:
        for n2 in n2s:
            if (abs(m2) + n2) > cfg.orders.cross_stage_sum_max:
                continue
            rej2_nom, src2 = mix2.rejection_with_meta(m2, n2, lo2_sol.f_out_hz, f_lo1)
            base_L2 = rej2_nom + mix2.drive_derate_db(lo2_sol.delivered_dbm)
            for c2 in carriers_lo2:
                fam2_term = mix2.family_scale_db(abs(m2), c2.rel_dBc)
                for sgn2 in (+1, -1):
                    f_rf_raw = m2 * c2.freq_hz + sgn2 * n2 * (+f_lo1)
                    f_rf = abs(f_rf_raw)
                    if (f_rf < rf_lo - 2 * bin_w) or (f_rf > rf_hi + 2 * bin_w):
                        continue
                    dA_rf = float(rf_attn(f_rf) - rf_attn_des)
                    L_total = L_lo1_if2 + base_L2 + fam2_term + dA_rf
                    comp = {
                        "family": "lo1_feedthrough",
                        "stage2": {
                            "m": m2, "n": n2, "sign_if": sgn2,
                            "lo2_carrier_tag": c2.tag, "lo2_rel_dBc": float(c2.rel_dBc),
                            "rej_dbc": float(rej2_nom), "rej_source": src2,
                            "drive_derate_db": float(mix2.drive_derate_db(lo2_sol.delivered_dbm)),
                            "family_scale_db": float(fam2_term),
                        },
                        "lo1": {"freq_hz": float(f_lo1), "isolation_db": float(mix1.mdl.isolation.lo_to_rf_db)},
                        "dA_if2_db": float(dA_if2_lo),
                        "dA_rf_db": float(dA_rf),
                        "f_rf_hz_raw": float(f_rf_raw),
                    }
                    mechanisms.append((f_rf, float(L_total), comp))

    # IF1 leakage at IF2 output
    f_if1c = tile_if1.center_hz
    dA_if2_if = float(if2_attn(f_if1c) - if2_attn_des)
    L_if1_if2 = mix1.mdl.isolation.if_to_rf_db + dA_if2_if
    for m2 in m2s:
        for n2 in n2s:
            if (abs(m2) + n2) > cfg.orders.cross_stage_sum_max:
                continue
            rej2_nom, src2 = mix2.rejection_with_meta(m2, n2, lo2_sol.f_out_hz, f_if1c)
            base_L2 = rej2_nom + mix2.drive_derate_db(lo2_sol.delivered_dbm)
            for c2 in carriers_lo2:
                fam2_term = mix2.family_scale_db(abs(m2), c2.rel_dBc)
                for sgn2 in (+1, -1):
                    f_rf_raw = m2 * c2.freq_hz + sgn2 * n2 * (+f_if1c)
                    f_rf = abs(f_rf_raw)
                    if (f_rf < rf_lo - 2 * bin_w) or (f_rf > rf_hi + 2 * bin_w):
                        continue
                    dA_rf = float(rf_attn(f_rf) - rf_attn_des)
                    L_total = L_if1_if2 + base_L2 + fam2_term + dA_rf
                    comp = {
                        "family": "if1_leakage",
                        "stage2": {
                            "m": m2, "n": n2, "sign_if": sgn2,
                            "lo2_carrier_tag": c2.tag, "lo2_rel_dBc": float(c2.rel_dBc),
                            "rej_dbc": float(rej2_nom), "rej_source": src2,
                            "drive_derate_db": float(mix2.drive_derate_db(lo2_sol.delivered_dbm)),
                            "family_scale_db": float(fam2_term),
                        },
                        "if1": {"center_hz": float(f_if1c), "isolation_db": float(mix1.mdl.isolation.if_to_rf_db)},
                        "dA_if2_db": float(dA_if2_if),
                        "dA_rf_db": float(dA_rf),
                        "f_rf_hz_raw": float(f_rf_raw),
                    }
                    mechanisms.append((f_rf, float(L_total), comp))

    # LO2 feedthroughs at RF (main + harmonics/PFD/boundary carriers)
    for c2 in carriers_lo2:
        f = float(c2.freq_hz)
        dA_rf = float(rf_attn(f) - rf_attn_des)
        L = float(mix2.mdl.isolation.lo_to_rf_db) + float(c2.rel_dBc) + dA_rf
        comp = {
            "family": "lo2_feedthrough",
            "lo2_carrier_tag": c2.tag,
            "lo2_rel_dBc": float(c2.rel_dBc),
            "lo2_freq_hz": f,
            "isolation_db": float(mix2.mdl.isolation.lo_to_rf_db),
            "dA_rf_db": float(dA_rf),
        }
        mechanisms.append((f, L, comp))

    # ---- Component-aware coalescing into RBW bins -----------------------------
    if not mechanisms:
        return TileSummary(worst_margin_db=-999.0, bins=[], desired_rf_band=rf_des)

    # deterministic sort
    mechanisms.sort(key=lambda t: t[0])

    bins_freq: list[float] = []
    bins_pow: list[float] = []      # linear power sum
    bins_components: list[list[dict]] = []

    def db_to_lin(db): return 10.0 ** (db / 10.0)
    def lin_to_db(lin): return -999.0 if lin <= 0 else 10.0 * np.log10(lin)

    current_center = mechanisms[0][0]
    current_lin = 0.0
    current_comps: list[dict] = []

    def flush():
        if current_comps:
            bins_freq.append(current_center)
            bins_pow.append(current_lin)
            bins_components.append(list(current_comps))

    for f, L, comp in mechanisms:
        if abs(f - current_center) <= (bin_w * 0.5):
            current_lin += db_to_lin(L)
            current_comps.append(comp)
        else:
            flush()
            current_center = f
            current_lin = db_to_lin(L)
            current_comps = [comp]
    flush()

    cf = np.asarray(bins_freq, float)
    cL = np.asarray([lin_to_db(p) for p in bins_pow], float)

    # In-band classification & limits (unchanged math)
    inband_mask = np.abs(cf - rf_des.center_hz) <= (rf_des.bw_hz / 2.0 + bin_w / 2.0)
    edge = rf_des.bw_hz / 2.0
    dist_from_edge = np.clip(np.abs(cf - rf_des.center_hz) - edge, 0.0, None)

    # Inband mask
    lim_in = _interp_mask_flat_or_table(cfg.masks.inband.default_dbc, cfg.masks.inband.table, dist_from_edge)

    # OOB mask
    if (getattr(cfg.masks.outofband, "mode", "absolute") == "edge_relative"):
        oob_arg = dist_from_edge
    else:
        oob_arg = cf
    lim_oob = _interp_mask_flat_or_table(cfg.masks.outofband.default_dbc, cfg.masks.outofband.table, oob_arg)

    limits = np.where(inband_mask, lim_in, lim_oob).astype(float)
    margins = limits - cL - cfg.constraints.guard_margin_db
    worst = float(np.min(margins)) if margins.size else -999.0

    final_bins: list[BinEntry] = []
    for f, Ldb, inb, lim, mar, comps in zip(cf, cL, inband_mask, limits, margins, bins_components):
        families = sorted({c.get("family", "mixing") for c in comps})
        has_fallback = any(
            (c.get("stage1", {}).get("rej_source") == "fallback") or
            (c.get("stage2", {}).get("rej_source") == "fallback")
            for c in comps
        )
        info = {
            "limit_dbc": float(lim),
            "margin_db": float(mar),
            "components": comps,                 # <-- full provenance
            "families_present": families,
            "has_missing_table_data": bool(has_fallback),
        }
        final_bins.append(BinEntry(float(f), float(Ldb), bool(inb), info))

    return TileSummary(worst_margin_db=worst, bins=final_bins, desired_rf_band=rf_des)