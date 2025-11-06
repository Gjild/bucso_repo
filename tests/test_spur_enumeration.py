# tests/test_spur_enumeration.py
from __future__ import annotations
import math
import numpy as np
import pytest

from bucso.spur import enumerate_spurs, rbw_width, desired_paths
from bucso.utils import Band, coalesce_bins
from bucso.filters import IF2Parametric


# --------------------------
# Helpers
# --------------------------

def _pick_snap(synth, f_target, path, mxd_min=11.0, mxd_max=17.0):
    """Snap a synthesizer to a legal grid point near f_target."""
    sol = synth.snap_to_legal(
        f_target=float(f_target),
        path=path,
        window_steps=2,
        drive_min_dbm=mxd_min,
        drive_max_dbm=mxd_max,
    )
    assert sol is not None, "Failed to snap LO to legal grid near target"
    return sol


def _nearest_bin(bins, f0):
    """Return the bin from TileSummary.bins whose center is nearest f0."""
    return min(bins, key=lambda b: abs(b.f_rf_hz - float(f0)))


# --------------------------
# Core spur-enumeration tests
# --------------------------

def test_desired_path_is_excluded_from_spur_bins(
    tiny_config, simple_tile_if1, rf_filter_flat_pass, if2_window, mixer1_model, mixer2_model,
    lo1_synth, lo2_synth
):
    """
    The desired mechanism must not be counted as a spur bin.
    We verify no returned bin sits within 0.5*RBW of the desired RF center.
    """
    inj1 = +1   # high-side IF2 = | +LO1 - IF1 |
    s2   = +1   # stage-2 sum: RF = | +LO2 + IF2 |

    # Choose an illustrative plan close to the design doc example
    # IF1c=1.55 GHz, IF2c≈4.65 GHz -> LO1≈6.20 GHz; RFc=29.25 GHz -> LO2≈24.60 GHz
    if1 = simple_tile_if1
    if2 = if2_window
    lo1 = _pick_snap(lo1_synth, f_target=6.20e9, path="lo1",
                     mxd_min=mixer1_model.mdl.required_lo_drive_dbm["min"],
                     mxd_max=mixer1_model.mdl.required_lo_drive_dbm["max"])
    lo2 = _pick_snap(lo2_synth, f_target=24.60e9, path="lo2",
                     mxd_min=mixer2_model.mdl.required_lo_drive_dbm["min"],
                     mxd_max=mixer2_model.mdl.required_lo_drive_dbm["max"])

    # Equivalent carriers (include harmonics but desired path uses "main")
    mode1 = next(m for m in lo1_synth.mdl.modes if m.name == lo1.mode)
    mode2 = next(m for m in lo2_synth.mdl.modes if m.name == lo2.mode)
    C1 = lo1_synth.equivalent_carriers(lo1.f_out_hz, mode1, lo1.divider, pfd_hz=lo1.pfd_hz)
    C2 = lo2_synth.equivalent_carriers(lo2.f_out_hz, mode2, lo2.divider, pfd_hz=lo2.pfd_hz)

    # Run enumeration
    summ = enumerate_spurs(
        tile_if1=if1,
        lo1_sol=lo1,
        lo2_sol=lo2,
        if2win=if2,
        rf_filter=rf_filter_flat_pass,
        mix1=mixer1_model,
        mix2=mixer2_model,
        cfg=tiny_config,
        inj1_sign=inj1,
        s2_sign=s2,
        carriers_lo1=C1,
        carriers_lo2=C2,
    )

    # Desired RF band (used for mask/in-band classification)
    _, rf_des = desired_paths(if1, lo1.f_out_hz, lo2.f_out_hz, inj1, s2)
    bin_w = rbw_width(tiny_config.rbw_binning, rf_des.bw_hz, rf_des.center_hz)

    # Assert no spur bin falls within the RBW window around the desired center
    assert all(abs(b.f_rf_hz - rf_des.center_hz) > 0.5 * bin_w for b in summ.bins), \
        "Desired mechanism leaked into spur bin list"


def test_lo2_feedthrough_is_present_and_level_matches_isolation_delta(
    tiny_config, simple_tile_if1, rf_filter_flat_pass, if2_window, mixer1_model, mixer2_model,
    lo1_synth, lo2_synth
):
    """
    LO2 feedthrough (fundamental and possibly harmonics) at RF should appear as spur bins.
    For the fundamental, its level should be approx:
        mix2.isolation.lo_to_rf_db + (A_RF(f_LO2) - A_RF(desired_RF_center))
    ignoring small differences from bin coalescing.
    """
    inj1, s2 = +1, +1
    if1 = simple_tile_if1
    if2 = if2_window

    lo1 = _pick_snap(lo1_synth, 6.20e9, "lo1",
                     mixer1_model.mdl.required_lo_drive_dbm["min"],
                     mixer1_model.mdl.required_lo_drive_dbm["max"])
    lo2 = _pick_snap(lo2_synth, 24.60e9, "lo2",
                     mixer2_model.mdl.required_lo_drive_dbm["min"],
                     mixer2_model.mdl.required_lo_drive_dbm["max"])

    mode1 = next(m for m in lo1_synth.mdl.modes if m.name == lo1.mode)
    mode2 = next(m for m in lo2_synth.mdl.modes if m.name == lo2.mode)
    C1 = lo1_synth.equivalent_carriers(lo1.f_out_hz, mode1, lo1.divider, pfd_hz=lo1.pfd_hz)
    C2 = lo2_synth.equivalent_carriers(lo2.f_out_hz, mode2, lo2.divider, pfd_hz=lo2.pfd_hz)

    summ = enumerate_spurs(
        if1, lo1, lo2, if2, rf_filter_flat_pass, mixer1_model, mixer2_model, tiny_config,
        inj1_sign=inj1, s2_sign=s2, carriers_lo1=C1, carriers_lo2=C2
    )

    # Expected LO2 feedthrough level (relative to desired) at f = LO2
    _, rf_des = desired_paths(if1, lo1.f_out_hz, lo2.f_out_hz, inj1, s2)
    a_des = float(rf_filter_flat_pass.attn_at(rf_des.center_hz))
    a_at_lo2 = float(rf_filter_flat_pass.attn_at(lo2.f_out_hz))
    expected = mixer2_model.mdl.isolation.lo_to_rf_db + (a_at_lo2 - a_des)

    # Find the nearest bin to LO2 frequency
    b = _nearest_bin(summ.bins, lo2.f_out_hz)

    # Within a couple dB tolerance to allow binning/summing with any neighbors
    assert abs(b.f_rf_hz - lo2.f_out_hz) <= rbw_width(tiny_config.rbw_binning, rf_des.bw_hz, b.f_rf_hz)
    assert pytest.approx(b.level_dbc, abs=2.0) == expected


def test_if2_window_tightening_improves_margin(
    tiny_config, simple_tile_if1, rf_filter_flat_pass, mixer1_model, mixer2_model, lo1_synth, lo2_synth
):
    """
    If we make the IF2 window tighter (same center), spurs outside IF2 should see larger ΔA_IF2,
    leading to *better* worst-case spur margin (or at least not worse).
    """
    inj1, s2 = +1, +1
    if1 = simple_tile_if1

    # Choose a stable LO pair near the illustrative example
    lo1 = _pick_snap(lo1_synth, 6.20e9, "lo1",
                     mixer1_model.mdl.required_lo_drive_dbm["min"],
                     mixer1_model.mdl.required_lo_drive_dbm["max"])
    lo2 = _pick_snap(lo2_synth, 24.60e9, "lo2",
                     mixer2_model.mdl.required_lo_drive_dbm["min"],
                     mixer2_model.mdl.required_lo_drive_dbm["max"])

    mode1 = next(m for m in lo1_synth.mdl.modes if m.name == lo1.mode)
    mode2 = next(m for m in lo2_synth.mdl.modes if m.name == lo2.mode)
    C1 = lo1_synth.equivalent_carriers(lo1.f_out_hz, mode1, lo1.divider, pfd_hz=lo1.pfd_hz)
    C2 = lo2_synth.equivalent_carriers(lo2.f_out_hz, mode2, lo2.divider, pfd_hz=lo2.pfd_hz)

    # Compute desired IF2 center to align windows
    if2_des_c = abs(inj1 * lo1.f_out_hz - if1.center_hz)

    # Wide IF2 (e.g., 700 MHz)
    if2_wide = IF2Parametric(center_hz=if2_des_c, bw_hz=0.7e9,
                             passband_il_db=1.0, stop_floor_db=-80.0, rolloff_db_per_dec=40.0)
    # Tight IF2 (e.g., 500 MHz)
    if2_tight = IF2Parametric(center_hz=if2_des_c, bw_hz=0.5e9,
                              passband_il_db=1.0, stop_floor_db=-80.0, rolloff_db_per_dec=40.0)

    # Sanity: both contain the desired band
    assert if2_wide.contains_desired(if2_des_c, if1.bw_hz)
    assert if2_tight.contains_desired(if2_des_c, if1.bw_hz)

    # Evaluate both
    summ_wide = enumerate_spurs(
        if1, lo1, lo2, if2_wide, rf_filter_flat_pass, mixer1_model, mixer2_model, tiny_config,
        inj1_sign=inj1, s2_sign=s2, carriers_lo1=C1, carriers_lo2=C2
    )
    summ_tight = enumerate_spurs(
        if1, lo1, lo2, if2_tight, rf_filter_flat_pass, mixer1_model, mixer2_model, tiny_config,
        inj1_sign=inj1, s2_sign=s2, carriers_lo1=C1, carriers_lo2=C2
    )

    # Tighter IF2 should not degrade worst-case margin and typically improves it
    assert summ_tight.worst_margin_db >= summ_wide.worst_margin_db - 1e-6


def test_inband_vs_oob_mask_selection_changes_worst_margin(
    tiny_config, simple_tile_if1, rf_filter_flat_pass, if2_window, mixer1_model, mixer2_model,
    lo1_synth, lo2_synth
):
    """
    Make the OOB mask much stricter than in-band so that the worst-case margin is driven by OOB,
    then relax OOB and verify the worst-case margin increases accordingly.
    """
    cfg = tiny_config.model_copy()

    # Force OOB much stricter at first
    cfg.masks.inband.default_dbc = -20.0   # permissive in-band
    cfg.masks.outofband.default_dbc = -90.0  # very strict OOB
    cfg.constraints.guard_margin_db = 0.0

    inj1, s2 = +1, +1
    if1 = simple_tile_if1
    if2 = if2_window

    lo1 = _pick_snap(lo1_synth, 6.20e9, "lo1",
                     mixer1_model.mdl.required_lo_drive_dbm["min"],
                     mixer1_model.mdl.required_lo_drive_dbm["max"])
    lo2 = _pick_snap(lo2_synth, 24.60e9, "lo2",
                     mixer2_model.mdl.required_lo_drive_dbm["min"],
                     mixer2_model.mdl.required_lo_drive_dbm["max"])

    mode1 = next(m for m in lo1_synth.mdl.modes if m.name == lo1.mode)
    mode2 = next(m for m in lo2_synth.mdl.modes if m.name == lo2.mode)
    C1 = lo1_synth.equivalent_carriers(lo1.f_out_hz, mode1, lo1.divider, pfd_hz=lo1.pfd_hz)
    C2 = lo2_synth.equivalent_carriers(lo2.f_out_hz, mode2, lo2.divider, pfd_hz=lo2.pfd_hz)

    # Strict-OOB pass
    s_strict = enumerate_spurs(
        if1, lo1, lo2, if2, rf_filter_flat_pass, mixer1_model, mixer2_model, cfg,
        inj1_sign=inj1, s2_sign=s2, carriers_lo1=C1, carriers_lo2=C2
    )

    # Now relax the OOB mask notably
    cfg_relaxed = cfg.model_copy()
    cfg_relaxed.masks.outofband.default_dbc = -40.0

    s_relaxed = enumerate_spurs(
        if1, lo1, lo2, if2, rf_filter_flat_pass, mixer1_model, mixer2_model, cfg_relaxed,
        inj1_sign=inj1, s2_sign=s2, carriers_lo1=C1, carriers_lo2=C2
    )

    # Worst-case margin should improve (become less negative / larger) when OOB mask is relaxed
    assert s_relaxed.worst_margin_db >= s_strict.worst_margin_db - 1e-6


# --------------------------
# Small, direct checks that support enumerate_spurs behavior
# --------------------------

def test_rbw_width_and_fixed_bin_coalescing_behave_as_expected():
    """Sanity-check rbw width formula and coalescing power sum behavior."""
    class RBW:
        rbw_hz = 10e3
        rbw_frac_of_bw = 0.0025
        rbw_ppm_of_freq = 0.0

    bw = 400e6
    f0 = 29.25e9
    w = rbw_width(RBW, bw, f0)
    # With the defaults from conftest, the fractional term dominates: 0.0025 * 400 MHz = 1 MHz
    assert w == pytest.approx(1e6, rel=0, abs=1e-6)

    # Two tones inside one fixed bin should power-sum to ~ +3 dB
    freqs = np.array([10.000e9 + 0.10e6, 10.000e9 + 0.50e6])  # both inside the same 1 MHz bin
    levs_db = np.array([-65.0, -65.0])
    bf, bL = coalesce_bins(freqs, levs_db, bin_width_hz=1e6)
    assert len(bf) == 1
    # 2 equal powers combine to +3.0103 dB
    assert bL[0] == pytest.approx(-65.0 + 10*np.log10(2.0), abs=1e-3)


def test_desired_paths_math_matches_sign_conventions(simple_tile_if1):
    """Quick functional check of desired_paths() sign conventions."""
    inj1, s2 = +1, +1
    lo1 = 6.20e9
    lo2 = 24.60e9
    if2, rf = desired_paths(simple_tile_if1, lo1, lo2, inj1, s2)
    assert if2.center_hz == pytest.approx(abs(inj1*lo1 - simple_tile_if1.center_hz))
    assert rf.center_hz == pytest.approx(abs(s2*lo2 + if2.center_hz))
    # BW is preserved across stages
    assert if2.bw_hz == pytest.approx(simple_tile_if1.bw_hz)
    assert rf.bw_hz == pytest.approx(simple_tile_if1.bw_hz)
