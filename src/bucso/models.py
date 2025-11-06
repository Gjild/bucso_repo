from __future__ import annotations
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field
import numpy as np

Hz = float
dB = float


class Project(BaseModel):
    name: str = "KaBUC-SpurPlan"
    seed: int = 42
    reference_10mhz_hz: Hz = 10_000_000.0


class BandLimits(BaseModel):
    min: Hz
    max: Hz


class Bands(BaseModel):
    if1_hz: BandLimits
    rf_hz: BandLimits
    required_bandwidths_hz: List[Hz] = [400e6]


class Grids(BaseModel):
    if1_center_step_hz: Hz = 25e6
    rf_center_step_hz: Hz = 25e6
    bw_grid_hz: List[Hz] = [400e6]
    snap_if1_rf_to_grid: bool = True


class MaskTableEntry(BaseModel):
    offset_hz: Hz
    limit_dbc: dB


class MaskSpec(BaseModel):
    default_dbc: dB = -60.0
    # safer than a bare [] default
    table: List[MaskTableEntry] = Field(default_factory=list)
    # NEW: how to interpret table for OOB: "absolute" (abs freq) or "edge_relative" (offset from RF-band edge)
    # For inband masks this field is ignored.
    mode: str = "absolute"


class Masks(BaseModel):
    inband: MaskSpec = MaskSpec()
    outofband: MaskSpec = MaskSpec()


class RBWBinning(BaseModel):
    rbw_hz: Hz = 10e3
    rbw_frac_of_bw: float = 0.0025
    rbw_ppm_of_freq: float = 0.0


class Orders(BaseModel):
    m1n1_max_abs: int = 7
    m2n2_max_abs: int = 7
    cross_stage_sum_max: int = 12


class Constraints(BaseModel):
    enforce_desired_mn11_only: bool = True
    guard_margin_db: dB = 2.0
    # Optional sign locks for the desired mechanism:
    # +1 = "sum / high-side", -1 = "difference / low-side"
    desired_stage1_sign: int | None = None
    desired_stage2_sign: int | None = None


class EarlyReject(BaseModel):
    image_in_if2_passband: bool = True
    loft_in_if2_or_rf_passbands: bool = True
    rf_first_order_image_in_passband: bool = True


class RuntimePolicy(BaseModel):
    hysteresis_hz: Hz = 10e6
    prefer_fewer_retunes: bool = True
    lock_time_penalty_weight: float = 1.0
    markov_transition_matrix_csv: Optional[str] = None


class Targets(BaseModel):
    min_margin_db: dB = 0.0
    alt_within_db: dB = 3.0


class Search(BaseModel):
    lo1_candidates: List[str]
    lo2_candidates: List[str]
    mixer1_candidates: List[str]
    mixer2_candidates: List[str]
    rf_bpf_choices: List[str]
    if2_filter_model: str

    # -------- NEW (fast snapped LO search) --------
    # Defaults preserve current behavior but enable the fast path out of the box.
    enable_snapped_lo_search: bool = True             # set False to force legacy wide sweeps
    lo_snap_window_steps: int = 2                     # initial ±N around snap
    lo_snap_expand_schedule: List[int] = [2, 4, 8]    # widening schedule if empty
    lo_snap_max_window_steps: int = 8                 # hard cap for widening
    include_lo2_alt_form: bool = False                # also try s2*(-RFc - IF2c)

    # Optional: cheaper search (winner is re-scored at full orders for parity)
    search_orders_mn_abs: int | None = None           # e.g., 3 or 4
    search_cross_sum_max: int | None = None           # e.g., 8..10

    # Optional: cap per-LO candidate list before Cartesian pairing
    per_lo_candidate_cap: int | None = 128


class Config(BaseModel):
    project: Project
    bands: Bands
    grids: Grids
    masks: Masks = Masks()
    rbw_binning: RBWBinning = RBWBinning()
    search: Search
    orders: Orders = Orders()
    constraints: Constraints = Constraints()
    early_reject: EarlyReject = EarlyReject()
    runtime_policy: RuntimePolicy = RuntimePolicy()
    targets: Targets = Targets()


# Mixer & LO YAML models (simplified for starter)
class MixerSpurEntry(BaseModel):
    m: int
    n: int
    rej_dbc: dB


class MixerIsolation(BaseModel):
    lo_to_rf_db: dB = -40.0
    if_to_rf_db: dB = -60.0


class MixerDriveDerate(BaseModel):
    nominal_dbm: float = 13.0
    slope_db_per_db: float = 1.0
    max_derate_db: float = 6.0


class MixerModel(BaseModel):
    name: str
    type: str = "double-balanced"
    if_range_hz: Tuple[Hz, Hz]
    lo_range_hz: Tuple[Hz, Hz]
    rf_range_hz: Tuple[Hz, Hz]
    required_lo_drive_dbm: Dict[str, float] = {"min": 11.0, "max": 17.0}
    drive_derate: MixerDriveDerate = MixerDriveDerate()
    # spur_table supports:
    #   - entries: [ {m,n,rej_dbc}, ... ]
    #   - grids_by_order: {"m,n": {lo_hz, if_hz, rej_dbc}}
    #   - (legacy) grids: {lo_hz, if_hz, rej_dbc} interpreted as (1,1)
    spur_table: Dict[str, object]
    isolation: MixerIsolation = MixerIsolation()
    lo_family_scaling: Dict[str, float] = {"default_slope_db_per_db": 1.0, "cap_db": 12.0}
    notes: Optional[str] = None
    # configurable fallback for missing table entries (keep but consider tightening in config)
    fallback_rej_dbc: dB = -20.0


class Harmonic(BaseModel):
    k: int
    rel_dBc: dB


class PfdComponent(BaseModel):
    k: int
    base_rel_dBc: dB
    rolloff_dB_per_dec: float
    # NEW: optional corner to express roll-off vs (k*fPFD / corner_hz)
    corner_hz: Optional[Hz] = None


class PfdFamily(BaseModel):
    name: str
    components: List[PfdComponent]


class LockTimeModel(BaseModel):
    base_ms: float = 0.4
    per_mhz_ms: float = 0.002
    mode_penalties_ms: Dict[str, float] = {}


class Mode(BaseModel):
    name: str
    enabled: bool = True
    pfd_hz_range: Tuple[Hz, Hz]
    pfd_dividers: List[int]
    vco_dividers: List[int]
    lock_time_model: LockTimeModel
    harmonics_at_output: List[Harmonic] = []
    pfd_spurs_at_output: Dict[str, List[PfdFamily]] | Dict[str, object] | None = None
    frac_boundary_spurs: Dict[str, object] | None = None


class DividerSpectrum(BaseModel):
    harm_delta_dBc: dB = 0.0


class OutputPowerTable(BaseModel):
    table: Dict[str, List[float]]
    divider_adjust_db: Dict[str, float] = {}


class Distribution(BaseModel):
    path_losses_db: Dict[str, float]
    pad_options_db: List[float] = [0.0, 3.0, 6.0]


class LOMdl(BaseModel):
    name: str
    ref_constraints: Dict[str, List[Hz]]
    freq_range_hz: Tuple[Hz, Hz]
    step_hz: Hz = 1e3
    output_power_model: OutputPowerTable
    distribution: Distribution
    modes: List[Mode]
    divider_spectrum: Dict[str, DividerSpectrum]
