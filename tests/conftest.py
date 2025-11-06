import math
import numpy as np
import pytest

from bucso.models import (
    Project, Bands, BandLimits, Grids, Masks, MaskSpec, RBWBinning,
    Orders, Constraints, EarlyReject, RuntimePolicy, Targets,
    Config, MixerModel, MixerIsolation, MixerDriveDerate, LOMdl,
    Mode, LockTimeModel, OutputPowerTable, Distribution, DividerSpectrum,
    Search,  # <-- added
)
from bucso.filters import RFBPF, IF2Parametric
from bucso.mixer import Mixer
from bucso.synth import Synth
from bucso.utils import Band


# ---------- generic small helpers ----------

@pytest.fixture
def rf_filter_flat_pass():
    # Very simple RF BPF: deep stop, flat-ish pass around 29–30 GHz
    f = np.array([26.0e9, 27.5e9, 28.0e9, 29.0e9, 30.0e9, 31.0e9, 32.0e9])
    s = np.array([-40, -15, -5, -1, -1, -5, -15], dtype=float)
    return RFBPF._from_arrays(f, s, ident="TEST_RF")


@pytest.fixture
def if2_window():
    # ~500 MHz window around 4.5 GHz, 1 dB IL, 40 dB/dec, -80 dB floor
    return IF2Parametric(
        center_hz=4.5e9, bw_hz=0.5e9,
        passband_il_db=1.0, stop_floor_db=-80.0, rolloff_db_per_dec=40.0
    )


@pytest.fixture
def mixer1_model():
    # Spur table with a few entries + legacy grid for (1,1)
    return Mixer(MixerModel(
        name="MX1",
        type="double-balanced",
        if_range_hz=(50e6, 8e9),
        lo_range_hz=(100e6, 12e9),
        rf_range_hz=(50e6, 12e9),
        required_lo_drive_dbm={"min": 11.0, "max": 17.0},
        drive_derate=MixerDriveDerate(nominal_dbm=13.0, slope_db_per_db=1.0, max_derate_db=6.0),
        spur_table={
            "entries": [
                {"m": 1, "n": 1, "rej_dbc": -35},
                {"m": 1, "n": 2, "rej_dbc": -45},
                {"m": 2, "n": 1, "rej_dbc": -38},
            ],
            "grids": {
                "lo_hz": [0.5e9, 2e9, 6e9],
                "if_hz": [0.1e9, 0.5e9, 1.0e9],
                "rej_dbc": [
                    [-32, -34, -36],
                    [-33, -35, -37],
                    [-34, -36, -38],
                ],
            },
        },
        isolation=MixerIsolation(lo_to_rf_db=-40, if_to_rf_db=-60),
        lo_family_scaling={"default_slope_db_per_db": 1.0, "cap_db": 12.0},
        fallback_rej_dbc=-20.0,
    ))


@pytest.fixture
def mixer2_model():
    return Mixer(MixerModel(
        name="MX2",
        type="double-balanced",
        if_range_hz=(50e6, 12e9),
        lo_range_hz=(100e6, 26e9),
        rf_range_hz=(50e6, 32e9),
        required_lo_drive_dbm={"min": 11.0, "max": 17.0},
        drive_derate=MixerDriveDerate(nominal_dbm=13.0, slope_db_per_db=1.0, max_derate_db=6.0),
        spur_table={
            "entries": [
                {"m": 1, "n": 1, "rej_dbc": -30},
                {"m": 1, "n": 2, "rej_dbc": -38},
                {"m": 2, "n": 1, "rej_dbc": -36},
            ],
        },
        isolation=MixerIsolation(lo_to_rf_db=-38, if_to_rf_db=-55),
        lo_family_scaling={"default_slope_db_per_db": 1.0, "cap_db": 12.0},
        fallback_rej_dbc=-20.0,
    ))


@pytest.fixture
def lo1_synth():
    # Simple LO with harmonics, divider spectrum, single mode
    return Synth(LOMdl(
        name="LO1",
        ref_constraints={"allowed_refs_hz": [10e6]},
        freq_range_hz=(10e6, 20e9),
        step_hz=1e6,
        output_power_model=OutputPowerTable(
            table={"freq_hz": [5e9, 10e9, 15e9, 20e9], "p_out_dbm": [18, 18, 18, 18]},
            divider_adjust_db={"/1": 0, "/2": -1, "/4": -2},
        ),
        distribution=Distribution(path_losses_db={"lo1": 3.0}, pad_options_db=[0.0, 3.0, 6.0]),
        modes=[
            Mode(
                name="fracN", enabled=True,
                pfd_hz_range=(10e6, 200e6),
                pfd_dividers=[1, 2, 4, 8],
                vco_dividers=[1, 2, 4],
                lock_time_model=LockTimeModel(base_ms=0.4, per_mhz_ms=0.002, mode_penalties_ms={}),
                harmonics_at_output=[{"k": 2, "rel_dBc": -25}, {"k": 3, "rel_dBc": -35}],
                pfd_spurs_at_output={"families": []},
                frac_boundary_spurs={"enabled": True, "amplitude_at_eps0p5_rel_dBc": -58, "rolloff_slope_db_per_dec": 10},
            )
        ],
        divider_spectrum={"/1": DividerSpectrum(harm_delta_dBc=0), "/4": DividerSpectrum(harm_delta_dBc=12)},
    ))


@pytest.fixture
def lo2_synth():
    return Synth(LOMdl(
        name="LO2",
        ref_constraints={"allowed_refs_hz": [10e6]},
        freq_range_hz=(10e6, 26e9),
        step_hz=1e6,
        output_power_model=OutputPowerTable(
            table={"freq_hz": [5e9, 15e9, 26e9], "p_out_dbm": [15, 15, 15]},
            divider_adjust_db={"/1": 0},
        ),
        distribution=Distribution(path_losses_db={"lo2": 4.0}, pad_options_db=[0.0, 3.0, 6.0]),
        modes=[
            Mode(
                name="fracN", enabled=True,
                pfd_hz_range=(10e6, 200e6),
                pfd_dividers=[1, 2, 4, 8],
                vco_dividers=[1],
                lock_time_model=LockTimeModel(base_ms=0.5, per_mhz_ms=0.002, mode_penalties_ms={}),
                harmonics_at_output=[{"k": 2, "rel_dBc": -25}],
                pfd_spurs_at_output={"families": []},
            )
        ],
        divider_spectrum={"/1": DividerSpectrum(harm_delta_dBc=0)},
    ))


@pytest.fixture
def tiny_config():
    # Provide a minimal but valid Search object (Config.search can't be None)
    search = Search(
        lo1_candidates=[],
        lo2_candidates=[],
        mixer1_candidates=[],
        mixer2_candidates=[],
        rf_bpf_choices=[],
        if2_filter_model="",
    )

    return Config(
        project=Project(name="TestProj", seed=123, reference_10mhz_hz=10_000_000.0),
        bands=Bands(
            if1_hz=BandLimits(min=950e6, max=2150e6),
            rf_hz=BandLimits(min=27.5e9, max=31.0e9),
            required_bandwidths_hz=[400e6],
        ),
        grids=Grids(
            if1_center_step_hz=100e6,
            rf_center_step_hz=250e6,
            bw_grid_hz=[400e6],
            snap_if1_rf_to_grid=True,
        ),
        masks=Masks(
            inband=MaskSpec(default_dbc=-60.0, table=[]),
            outofband=MaskSpec(default_dbc=-60.0, table=[], mode="absolute"),
        ),
        rbw_binning=RBWBinning(rbw_hz=10e3, rbw_frac_of_bw=0.0025, rbw_ppm_of_freq=0.0),
        search=search,  # <-- fixed (was None)
        orders=Orders(m1n1_max_abs=3, m2n2_max_abs=3, cross_stage_sum_max=8),
        constraints=Constraints(enforce_desired_mn11_only=True, guard_margin_db=2.0, desired_stage1_sign=+1, desired_stage2_sign=+1),
        early_reject=EarlyReject(image_in_if2_passband=True, loft_in_if2_or_rf_passbands=True,
                                 rf_first_order_image_in_passband=True),
        runtime_policy=RuntimePolicy(hysteresis_hz=10e6, prefer_fewer_retunes=True,
                                     lock_time_penalty_weight=1.0, markov_transition_matrix_csv=None),
        targets=Targets(min_margin_db=0.0, alt_within_db=3.0),
    )


@pytest.fixture
def simple_tile_if1():
    # IF1: 1.55 GHz, BW 400 MHz
    return Band(center_hz=1.55e9, bw_hz=400e6)
