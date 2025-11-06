from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Optional, Tuple
import os, yaml, math, hashlib, importlib.metadata
from contextlib import contextmanager
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .models import Config, MixerModel, LOMdl, Mode
from .filters import IF2Parametric, RFBPF
from .mixer import Mixer
from .synth import Synth
from .tiling import make_tiles
from .spur import enumerate_spurs, desired_paths
from .utils import Band

# ------------------- process-global model cache -------------------
_MODELS: dict | None = None  # set by _init_models_once()


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(root: str, name_or_path: str):
    path = name_or_path if os.path.isabs(name_or_path) else os.path.join(root, name_or_path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _init_models_once(models_dir: str, cfg_dict: dict):
    """
    Process initializer: load heavy models only once per process and stash
    them in module globals to avoid per-tile I/O/parse overhead.
    Also sets mixer cache quantization for better hit rate.
    """
    global _MODELS
    if _MODELS is not None:
        return

    cfg = Config(**cfg_dict)

    # Load mixer models (construct Mixer objects)
    mixer1 = [Mixer(MixerModel(**_load_yaml(models_dir, p))) for p in cfg.search.mixer1_candidates]
    mixer2 = [Mixer(MixerModel(**_load_yaml(models_dir, p))) for p in cfg.search.mixer2_candidates]
    # Improve cache reuse (quantize lookups more coarsely, clear LRUs)
    for m in (mixer1 + mixer2):
        m.set_cache_quantum_hz(5e6)  # ~5 MHz; good vs 25–250 MHz grids
        m.clear_cache()

    # Load synthesizers
    lo1s = [Synth(LOMdl(**_load_yaml(models_dir, p))) for p in cfg.search.lo1_candidates]
    lo2s = [Synth(LOMdl(**_load_yaml(models_dir, p))) for p in cfg.search.lo2_candidates]

    # RF filters (CSV/YAML auto)
    rfs = []
    for rp in cfg.search.rf_bpf_choices:
        path = rp if os.path.isabs(rp) else os.path.join(models_dir, rp)
        rfs.append(RFBPF.from_path(path))

    # IF2 parametric YAML (constants)
    if2yaml = _load_yaml(models_dir, cfg.search.if2_filter_model)

    _MODELS = dict(
        cfg_dict=cfg_dict,
        mixer1=mixer1,
        mixer2=mixer2,
        lo1s=lo1s,
        lo2s=lo2s,
        rfs=rfs,
        if2yaml=if2yaml,
        models_dir=models_dir,
    )


@contextmanager
def _progress_ctx(total: int, desc: str, provided_cb: Optional[Callable[[int], None]]):
    """
    Yield a callable `update(n)` that advances progress by n.
    If `provided_cb` is given, we just yield that.
    Otherwise, we enable a Rich progress bar when BUCSO_PROGRESS is truthy.
    Fallback to a no-op if Rich isn't available or progress is disabled.
    """
    if provided_cb is not None:
        # External progress handler (e.g., CLI) – just forward updates
        yield provided_cb
        return

    use_env = os.getenv("BUCSO_PROGRESS", "").strip().lower()
    enabled = use_env in ("1", "true", "yes", "on", "y", "t")

    if not enabled:
        # No progress – return a no-op updater
        def _noop(_n: int) -> None:
            pass
        yield _noop
        return

    try:
        from rich.progress import (
            Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn,
            MofNCompleteColumn, SpinnerColumn
        )
        columns = (
            SpinnerColumn(),
            f"[bold]{desc}[/bold]",
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        with Progress(*columns) as progress:
            task_id = progress.add_task(desc, total=total)

            def _update(n: int) -> None:
                if n:
                    progress.update(task_id, advance=int(n))

            yield _update
    except Exception:
        # Graceful fallback
        def _noop(_n: int) -> None:
            pass
        yield _noop


@dataclass
class PlanRow:
    tile_id: int
    if1_center_hz: float
    bw_hz: float
    rf_center_hz: float
    lo1_name: str
    lo1_hz: float
    lo1_mode: str
    lo1_divider: str
    lo1_pad_db: float
    lo1_lock_ms: float
    lo2_name: str
    lo2_hz: float
    lo2_mode: str
    lo2_divider: str
    lo2_pad_db: float
    lo2_lock_ms: float
    if2_center_hz: float
    if2_bw_hz: float
    rf_bpf_id: str
    spur_margin_db: float
    brittleness_db_per_step: float
    # Traceability: desired signs used
    desired_stage1_sign: int
    desired_stage2_sign: int


def _overlap_1d(a_center: float, a_bw: float, b_center: float, b_bw: float) -> bool:
    a_lo = a_center - a_bw / 2.0
    a_hi = a_center + a_bw / 2.0
    b_lo = b_center - b_bw / 2.0
    b_hi = b_center + b_bw / 2.0
    return not (a_hi < b_lo or b_hi < a_lo)


def _early_reject_gate(
    tile_if1: Band,
    lo1_hz: float,
    lo2_hz: float,
    inj1: int,
    s2: int,
    if2: IF2Parametric,
    rf_bpf: RFBPF,
    cfg: Config,
) -> bool:
    """
    Stronger early rejects with band-overlap checks.
    Returns True if candidate should be rejected.
    """
    if not (
        cfg.early_reject.image_in_if2_passband
        or cfg.early_reject.loft_in_if2_or_rf_passbands
        or cfg.early_reject.rf_first_order_image_in_passband
    ):
        return False

    # Desired bands for equivalence tests
    if2_des, rf_des = desired_paths(tile_if1, lo1_hz, lo2_hz, inj1, s2)

    # quick desired-IF2 containment precheck to avoid wasting work
    if not if2.contains_desired(if2_des.center_hz, if2_des.bw_hz):
        return True

    # Stage-1 images (sum & diff) against IF2 passband — don't reject if it's the desired path
    if cfg.early_reject.image_in_if2_passband:
        for sgn in (+1, -1):
            img2_c = abs(lo1_hz + sgn * tile_if1.center_hz)
            is_desired = abs(img2_c - if2_des.center_hz) <= 1e-6
            if (not is_desired) and _overlap_1d(img2_c, tile_if1.bw_hz, if2.center_hz, if2.bw_hz):
                return True

    # LO1 feedthrough inside IF2 passband
    if cfg.early_reject.loft_in_if2_or_rf_passbands:
        if _overlap_1d(lo1_hz, 0.0, if2.center_hz, if2.bw_hz):
            return True

    # Stage-2 first-order images (±LO2 ± IF2) overlapping RF desired band
    if cfg.early_reject.rf_first_order_image_in_passband:
        for sgn1 in (+1, -1):
            for sgn2 in (+1, -1):
                f_c = abs(sgn1 * lo2_hz + sgn2 * if2_des.center_hz)
                if _overlap_1d(f_c, if2_des.bw_hz, rf_des.center_hz, rf_des.bw_hz):
                    is_desired = (sgn1 == s2) and (sgn2 == +1)
                    is_desired_equiv = (sgn1 == -s2) and (sgn2 == -1)  # |·| symmetry
                    # Reject only if a different first-order image overlaps
                    if not (is_desired or is_desired_equiv):
                        return True

    # LO2 feedthrough / quick harmonics in RF passband (cheap reject)
    if cfg.early_reject.loft_in_if2_or_rf_passbands:
        # main tone
        if _overlap_1d(lo2_hz, 0.0, rf_des.center_hz, rf_des.bw_hz):
            return True
        # quick harmonic screen — include k=2,3,4 (very cheap)
        for k in (2, 3, 4):
            f_h = k * lo2_hz
            if _overlap_1d(f_h, 0.0, rf_des.center_hz, rf_des.bw_hz):
                return True

    return False


# --- Helpers -------------------------------------------------------------

def _rf_band_ok(rf_filter: RFBPF, rf_band_center: float, rf_bw: float, max_il_db: float = 3.0) -> bool:
    """Require the desired RF band (center ± BW/2) to be inside the passband by IL >= -max_il_db.
    Sample densely across the band to avoid optimistic passes."""
    half = rf_bw * 0.5
    fs = np.linspace(rf_band_center - half, rf_band_center + half, 17)
    attn = np.array([float(rf_filter.attn_at(f)) for f in fs], float)
    return np.min(attn) >= -max_il_db


# --- fast snapped LO neighborhoods ---------------------------------------

def _neighborhood_solutions(
    synth: Synth, *,
    f_targets: list[float],
    path: str,
    drive_min: float,
    drive_max: float,
    base_N: int,
    expand_sched: list[int],
    hard_max: int,
    cap: int | None,
    bw_hz_for_fallback: float,
):
    """
    For each analytic target:
      1) snap to legal grid and probe ±N grid steps
      2) widen using expand_sched up to hard_max if needed
      3) final fallback: bounded legacy legal_settings() within ±3×BW
    Returns deterministic, de-duplicated LOSolution entries sorted by proximity.
    """
    step = float(synth.mdl.step_hz)

    def _probe(N: int):
        offsets = [0] + [off for k in range(1, max(0, N) + 1) for off in (k, -k)]
        sols = []
        for tgt in f_targets:
            for off in offsets:
                sol = synth.snap_to_legal(
                    f_target=float(tgt + off * step),
                    path=path,
                    window_steps=0,
                    drive_min_dbm=drive_min,
                    drive_max_dbm=drive_max,
                )
                if sol is not None:
                    sols.append(sol)

        # Deterministic de-dup + sort by |f - nearest target|
        def key(s): return (s.name, s.mode, s.divider, round(s.f_out_hz, 0))
        dedup = {key(s): s for s in sols}
        out = list(dedup.values())

        def dist(s): return min(abs(s.f_out_hz - ft) for ft in f_targets)
        out.sort(key=dist)

        if cap is not None and len(out) > cap:
            out = out[:cap]
        return out

    # Initial probe
    sols = _probe(int(base_N))
    if sols:
        return sols

    # Expand schedule
    for N in (expand_sched or []):
        N = int(min(max(0, N), hard_max))
        sols = _probe(N)
        if sols:
            return sols

    # Final bounded fallback (legacy semantics, but centered on targets)
    sols = []
    for tgt in f_targets:
        fmin = max(1.0, float(tgt) - 3.0 * bw_hz_for_fallback)
        fmax = float(tgt) + 3.0 * bw_hz_for_fallback
        sols.extend(synth.legal_settings(
            name_filter=None, f_min=fmin, f_max=fmax, path=path,
            drive_min_dbm=drive_min, drive_max_dbm=drive_max,
        ))

    def key(s): return (s.name, s.mode, s.divider, round(s.f_out_hz, 0))
    dedup = {key(s): s for s in sols}
    out = list(dedup.values())

    def dist(s): return min(abs(s.f_out_hz - ft) for ft in f_targets)
    out.sort(key=dist)

    if cap is not None and len(out) > cap:
        out = out[:cap]
    return out


# --- IF2 refinement (deterministic coordinate descent + memo) ------------------------

def _coordinate_descent_if2(
    if2_const: dict,
    seed_center_hz: float,
    seed_bw_hz: float,
    tile_if1: Band,
    lo1,
    lo2,
    rf_filter: RFBPF,
    mixer1: Mixer,
    mixer2: Mixer,
    cfg: Config,
    inj1: int,
    s2: int,
    carriers_lo1,
    carriers_lo2,
    max_iters: int,
    step_frac_center: float,
    step_frac_bw: float,
    bw_limits: tuple[float, float],
    center_limits: tuple[float, float],
) -> IF2Parametric:
    """
    Internal helper that actually runs the loop. if2_const has keys:
      passband_il_db, stop_floor_db, rolloff_db_per_dec
    Enhanced with memoization and an early-stop after two consecutive no-improve rounds.
    """
    c = float(seed_center_hz)
    b = float(seed_bw_hz)
    best_win = IF2Parametric(
        center_hz=c, bw_hz=b,
        passband_il_db=if2_const["passband_il_db"],
        stop_floor_db=if2_const["stop_floor_db"],
        rolloff_db_per_dec=if2_const["rolloff_db_per_dec"],
    )

    memo: dict[tuple[float, float], float] = {}

    def score(win: IF2Parametric) -> float:
        key = (round(win.center_hz, 1), round(win.bw_hz, 1))
        if key in memo:
            return memo[key]
        if2_des_center = abs(inj1 * lo1.f_out_hz - tile_if1.center_hz)
        if not win.contains_desired(center=if2_des_center, bw=tile_if1.bw_hz):
            memo[key] = -1e9
            return memo[key]
        summ = enumerate_spurs(
            tile_if1, lo1, lo2, win, rf_filter, mixer1, mixer2, cfg,
            inj1_sign=inj1, s2_sign=s2, carriers_lo1=carriers_lo1, carriers_lo2=carriers_lo2
        )
        memo[key] = float(summ.worst_margin_db)
        return memo[key]

    best = score(best_win)
    dc = max(1.0, step_frac_center) * (tile_if1.bw_hz * 0.5)  # start with half-BW scaled step
    db = max(1.0, step_frac_bw) * (tile_if1.bw_hz * 0.3)

    no_improve = 0
    for _ in range(max_iters):
        improved = False
        # Try center +/- dc
        for sign in (+1, -1):
            c_try = float(np.clip(c + sign * dc, center_limits[0], center_limits[1]))
            win = IF2Parametric(c_try, b, if2_const["passband_il_db"], if2_const["stop_floor_db"], if2_const["rolloff_db_per_dec"])
            s = score(win)
            if s > best + 1e-6:
                best, c, best_win = s, c_try, win
                improved = True
        # Try BW +/- db
        for sign in (+1, -1):
            b_try = float(np.clip(b + sign * db, bw_limits[0], bw_limits[1]))
            win = IF2Parametric(c, b_try, if2_const["passband_il_db"], if2_const["stop_floor_db"], if2_const["rolloff_db_per_dec"])
            s = score(win)
            if s > best + 1e-6:
                best, b, best_win = s, b_try, win
                improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 2:
                break

        # Shrink steps
        dc *= 0.5
        db *= 0.5

    return best_win


def _robustness_score(
    base_summary,
    tile_if1: Band,
    lo1, lo2,
    if2_use: IF2Parametric,
    rf_filter: RFBPF,
    mixer1: Mixer,
    mixer2: Mixer,
    cfg: Config,
    inj1: int,
    s2: int,
    carriers_lo1,
    carriers_lo2,
    rf_center_nominal: float,
    if1_step: float,
    rf_step: float,
) -> Tuple[float, float]:
    """
    Evaluate worst-case margin under ± half-step perturbations of IF1 and RF centers.
    We keep the LO/IF2 settings fixed and only change the evaluation reference.
    Returns (robust_min_margin, brittleness_db_per_step).
    """
    margins = [float(base_summary.worst_margin_db)]

    # IF1 ± half step (true recomputation, because IF2 desired depends on IF1)
    for sign in (+1, -1):
        if1_pert = Band(tile_if1.center_hz + sign * 0.5 * if1_step, tile_if1.bw_hz)
        # Robustness guard: if desired IF2 drifts outside the IF2 passband, treat as hard fail
        if2_des_center = abs(inj1 * lo1.f_out_hz - if1_pert.center_hz)
        if not if2_use.contains_desired(center=if2_des_center, bw=if1_pert.bw_hz):
            margins.append(-1e9)
            continue
        summ = enumerate_spurs(
            if1_pert, lo1, lo2, if2_use, rf_filter, mixer1, mixer2, cfg,
            inj1_sign=inj1, s2_sign=s2, carriers_lo1=carriers_lo1, carriers_lo2=carriers_lo2
        )
        margins.append(float(summ.worst_margin_db))

    # RF ± half step (classification/limit reference moves; hardware fixed)
    for sign in (+1, -1):
        rf_ref = rf_center_nominal + sign * 0.5 * rf_step
        summ = enumerate_spurs(
            tile_if1, lo1, lo2, if2_use, rf_filter, mixer1, mixer2, cfg,
            inj1_sign=inj1, s2_sign=s2, carriers_lo1=carriers_lo1, carriers_lo2=carriers_lo2,
            rf_center_override_hz=rf_ref, rf_bw_override_hz=tile_if1.bw_hz,
        )
        margins.append(float(summ.worst_margin_db))

    robust = float(min(margins))
    drop = float(max(margins) - robust) if margins else 0.0
    # Keep unit as "drop in dB"
    brittleness = drop
    return robust, brittleness


def _eval_tile(task_args) -> tuple[PlanRow | None, list[dict], dict] | None:
    """
    Worker function: evaluate one tile with carriers, early rejects, RF-BPF choice,
    robustness, and alternatives within Δ dB of best.
    Returns (best_row, alternatives_list, ledger_for_best) or None if no candidate.
    """
    t, cfg_dict, models_dir = task_args

    # Use process-global cache (initializer populates this)
    global _MODELS
    if _MODELS is None:
        _init_models_once(models_dir, cfg_dict)
    cache = _MODELS

    cfg = Config(**cache["cfg_dict"])

    # Load models from cache
    mixer1_mdls: List[Mixer] = cache["mixer1"]
    mixer2_mdls: List[Mixer] = cache["mixer2"]
    lo1_synths: List[Synth] = cache["lo1s"]
    lo2_synths: List[Synth] = cache["lo2s"]
    rf_filters: List[RFBPF] = cache["rfs"]
    if2_yaml = cache["if2yaml"]

    # ---- Optional: "search-order thinning" (keep parity via re-score later) ----
    orig_cfg = Config(**cfg_dict)  # exact original for parity/scoring
    used_thin_orders = False
    search_mn = getattr(cfg.search, "search_orders_mn_abs", None)
    search_sum = getattr(cfg.search, "search_cross_sum_max", None)
    if (search_mn is not None) or (search_sum is not None):
        cfg = cfg.model_copy()
        if search_mn is not None:
            cfg.orders.m1n1_max_abs = int(search_mn)
            cfg.orders.m2n2_max_abs = int(search_mn)
            used_thin_orders = True
        if search_sum is not None:
            cfg.orders.cross_stage_sum_max = int(search_sum)
            used_thin_orders = True

    def _f(x, default=None):
        if x is None:
            return float(default) if default is not None else None
        try:
            return float(x)
        except (TypeError, ValueError):
            return float(str(x))

    if2_pass_il = _f(if2_yaml.get("passband_il_db", 1.0), 1.0)
    if2_floor = _f(if2_yaml.get("stop_floor_db", -80.0), -80.0)
    if2_roll = _f(if2_yaml.get("rolloff_db_per_dec", 40.0), 40.0)
    if2_min_bw = _f(if2_yaml.get("min_bw_hz", 500e6), 500e6)
    if2_max_bw = _f(if2_yaml.get("max_bw_hz", 6000e6), 6000e6)
    _cr = if2_yaml.get("center_range_hz", [500e6, 9000e6]) or [500e6, 9000e6]
    if2_center_range = (_f(_cr[0], 500e6), _f(_cr[1], 9000e6))

    # Search parameters (if present)
    search_cfg = (if2_yaml.get("search") or {})
    # seeds_per_tile controls center/BW seed densities deterministically
    seeds_per_tile = int(search_cfg.get("seeds_per_tile", 8) or 8)
    refine_kind = str(search_cfg.get("local_refinement", "coordinate_descent"))
    max_refine_iters = int(search_cfg.get("max_refine_iters", 12) or 12)

    tile_if1 = Band(t.if1_center_hz, t.bw_hz)
    best_row: PlanRow | None = None
    best_summary = None
    best_score = -1e9
    alternatives: list[dict] = []
    best_ledger: dict = {}

    # --- Derive IF2 seeds: local neighborhood + global anchors ---
    num_c = max(3, int(seeds_per_tile))
    center_lo_mul, center_hi_mul = 0.7, 1.3
    local_center_seeds = [float(np.clip(m * t.if1_center_hz, if2_center_range[0], if2_center_range[1]))
                          for m in np.linspace(center_lo_mul, center_hi_mul, num_c)]
    # A few coarse global anchors spanning the allowed IF2 center range
    global_anchors = list(np.linspace(if2_center_range[0], if2_center_range[1], 4))
    if2_center_seeds = sorted(set(map(float, [*local_center_seeds, *global_anchors])))

    # IF2 BW seeds around required BW
    base_bw_low, base_bw_high = 1.05, 1.5
    n_bw = int(max(2, min(5, math.ceil(seeds_per_tile / 3))))
    if n_bw == 2:
        bw_seed_muls = (base_bw_low, base_bw_high)
    else:
        bw_seed_muls = tuple(np.linspace(base_bw_low, base_bw_high, n_bw))
    bw_seed_list = bw_seed_muls

    # Fast/legacy toggle
    use_fast = bool(getattr(cfg.search, "enable_snapped_lo_search", True))
    base_N = int(getattr(cfg.search, "lo_snap_window_steps", 2))
    expand_sched = list(getattr(cfg.search, "lo_snap_expand_schedule", [2, 4, 8]))
    hard_max = int(getattr(cfg.search, "lo_snap_max_window_steps", 8))
    cap_per_lo = getattr(cfg.search, "per_lo_candidate_cap", None)
    include_lo2_alt = bool(getattr(cfg.search, "include_lo2_alt_form", False))

    # respect optional desired sign locks (±1) for desired path
    desired_s1 = getattr(cfg.constraints, "desired_stage1_sign", None)
    desired_s2 = getattr(cfg.constraints, "desired_stage2_sign", None)
    inj1_opts = ([int(desired_s1)] if desired_s1 in (-1, 1) else (+1, -1))
    s2_opts = ([int(desired_s2)] if desired_s2 in (-1, 1) else (+1, -1))

    for mixer1 in mixer1_mdls:
        for mixer2 in mixer2_mdls:
            for inj1 in inj1_opts:
                for if2c_seed in if2_center_seeds:
                    for bw_mul in bw_seed_list:
                        if2_bw_seed = float(np.clip(bw_mul * t.bw_hz, if2_min_bw, if2_max_bw))
                        if2_template = IF2Parametric(if2c_seed, if2_bw_seed, if2_pass_il, if2_floor, if2_roll)

                        # Stage-1 target LO1s from seed center; include (IF2+IF1) and (IF2-IF1) if >0
                        lo1_targets = [inj1 * (if2c_seed + t.if1_center_hz)]
                        alt_t = inj1 * (if2c_seed - t.if1_center_hz)
                        if alt_t > 0:
                            lo1_targets.append(alt_t)

                        # Enumerate LO1 candidates via snapped neighborhoods (or legacy)
                        lo1_settings = []
                        for lo1_synth in lo1_synths:
                            if use_fast:
                                lo1_settings.extend(_neighborhood_solutions(
                                    synth=lo1_synth,
                                    f_targets=lo1_targets,
                                    path="lo1",
                                    drive_min=mixer1.mdl.required_lo_drive_dbm["min"],
                                    drive_max=mixer1.mdl.required_lo_drive_dbm["max"],
                                    base_N=base_N,
                                    expand_sched=expand_sched,
                                    hard_max=hard_max,
                                    cap=cap_per_lo,
                                    bw_hz_for_fallback=t.bw_hz,
                                ))
                            else:
                                fmin1 = max(1.0, min(lo1_targets) - 3 * t.bw_hz)
                                fmax1 = max(lo1_targets) + 3 * t.bw_hz
                                lo1_settings.extend(lo1_synth.legal_settings(
                                    name_filter=None,
                                    f_min=fmin1,
                                    f_max=fmax1,
                                    path="lo1",
                                    drive_min_dbm=mixer1.mdl.required_lo_drive_dbm["min"],
                                    drive_max_dbm=mixer1.mdl.required_lo_drive_dbm["max"],
                                ))
                        if not lo1_settings:
                            continue

                        for lo1 in lo1_settings:
                            # Require desired IF2 band to be inside the IF2 passband
                            if2_des_center = abs(inj1 * lo1.f_out_hz - t.if1_center_hz)
                            if not if2_template.contains_desired(if2_des_center, t.bw_hz):
                                continue

                            for s2 in s2_opts:
                                # Stage-2 targets
                                cand2 = [s2 * (t.rf_center_hz - if2c_seed)]
                                if include_lo2_alt:
                                    cand2.append(s2 * (-t.rf_center_hz - if2c_seed))

                                lo2_settings = []
                                for lo2_synth in lo2_synths:
                                    if use_fast:
                                        lo2_settings.extend(_neighborhood_solutions(
                                            synth=lo2_synth,
                                            f_targets=cand2,
                                            path="lo2",
                                            drive_min=mixer2.mdl.required_lo_drive_dbm["min"],
                                            drive_max=mixer2.mdl.required_lo_drive_dbm["max"],
                                            base_N=base_N,
                                            expand_sched=expand_sched,
                                            hard_max=hard_max,
                                            cap=cap_per_lo,
                                            bw_hz_for_fallback=t.bw_hz,
                                        ))
                                    else:
                                        fmin2 = min(cand2) - 3 * t.bw_hz
                                        fmax2 = max(cand2) + 3 * t.bw_hz
                                        lo2_settings.extend(lo2_synth.legal_settings(
                                            name_filter=None,
                                            f_min=max(1.0, fmin2),
                                            f_max=fmax2,
                                            path="lo2",
                                            drive_min_dbm=mixer2.mdl.required_lo_drive_dbm["min"],
                                            drive_max_dbm=mixer2.mdl.required_lo_drive_dbm["max"],
                                        ))
                                if not lo2_settings:
                                    continue

                                for lo2 in lo2_settings:
                                    # Verify desired RF placement under these LOs
                                    _, rf_des_chk = desired_paths(tile_if1, lo1.f_out_hz, lo2.f_out_hz, inj1, s2)
                                    if abs(rf_des_chk.center_hz - t.rf_center_hz) > (cfg.grids.rf_center_step_hz * 0.5 + 1e3):
                                        continue

                                    # Build equivalent carriers (with divider-spectrum behavior) using chosen PFD
                                    mode1 = next((m for m in lo1_synths[0].mdl.modes if m.name == lo1.mode), None)
                                    if mode1 is None:
                                        # fetch from the specific synth used to create lo1
                                        for s in lo1_synths:
                                            if s.mdl.name == lo1.name:
                                                mode1 = next((m for m in s.mdl.modes if m.name == lo1.mode), s.mdl.modes[0])
                                                break
                                    mode2 = next((m for m in lo2_synths[0].mdl.modes if m.name == lo2.mode), None)
                                    if mode2 is None:
                                        for s in lo2_synths:
                                            if s.mdl.name == lo2.name:
                                                mode2 = next((m for m in s.mdl.modes if m.name == lo2.mode), s.mdl.modes[0])
                                                break
                                    # use the synth that matches the LO name for carriers
                                    s1_for_car = next((s for s in lo1_synths if s.mdl.name == lo1.name), lo1_synths[0])
                                    s2_for_car = next((s for s in lo2_synths if s.mdl.name == lo2.name), lo2_synths[0])
                                    C1 = s1_for_car.equivalent_carriers(lo1.f_out_hz, mode1, lo1.divider, pfd_hz=lo1.pfd_hz)
                                    C2 = s2_for_car.equivalent_carriers(lo2.f_out_hz, mode2, lo2.divider, pfd_hz=lo2.pfd_hz)

                                    # Desired RF band must be inside RF BPF passband
                                    for rf_filter in rf_filters:
                                        if not _rf_band_ok(rf_filter, rf_des_chk.center_hz, t.bw_hz, max_il_db=3.0):
                                            continue

                                        # Early reject gates
                                        if _early_reject_gate(tile_if1, lo1.f_out_hz, lo2.f_out_hz, inj1, s2, if2_template, rf_filter, cfg):
                                            continue

                                        # --- IF2 refinement (coordinate descent + memo) ---
                                        if refine_kind == "coordinate_descent" and max_refine_iters > 0:
                                            if2_const = {
                                                "passband_il_db": if2_pass_il,
                                                "stop_floor_db": if2_floor,
                                                "rolloff_db_per_dec": if2_roll,
                                            }
                                            if2_use = _coordinate_descent_if2(
                                                if2_const=if2_const,
                                                seed_center_hz=if2c_seed,
                                                seed_bw_hz=if2_bw_seed,
                                                tile_if1=tile_if1,
                                                lo1=lo1,
                                                lo2=lo2,
                                                rf_filter=rf_filter,
                                                mixer1=mixer1,
                                                mixer2=mixer2,
                                                cfg=cfg,
                                                inj1=inj1,
                                                s2=s2,
                                                carriers_lo1=C1,
                                                carriers_lo2=C2,
                                                max_iters=max_refine_iters,
                                                step_frac_center=0.1,
                                                step_frac_bw=0.15,
                                                bw_limits=(if2_min_bw, if2_max_bw),
                                                center_limits=if2_center_range,
                                            )
                                        else:
                                            if2_use = if2_template

                                        # Base evaluation (possibly under thinned orders)
                                        summ = enumerate_spurs(
                                            tile_if1,
                                            lo1,
                                            lo2,
                                            if2_use,
                                            rf_filter,
                                            mixer1,
                                            mixer2,
                                            cfg,
                                            inj1_sign=inj1,
                                            s2_sign=s2,
                                            carriers_lo1=C1,
                                            carriers_lo2=C2,
                                        )

                                        # Robustness evaluation (± half-steps) with IF2 containment guard
                                        robust_margin, brittleness = _robustness_score(
                                            base_summary=summ,
                                            tile_if1=tile_if1,
                                            lo1=lo1,
                                            lo2=lo2,
                                            if2_use=if2_use,
                                            rf_filter=rf_filter,
                                            mixer1=mixer1,
                                            mixer2=mixer2,
                                            cfg=cfg,
                                            inj1=inj1,
                                            s2=s2,
                                            carriers_lo1=C1,
                                            carriers_lo2=C2,
                                            rf_center_nominal=t.rf_center_hz,
                                            if1_step=cfg.grids.if1_center_step_hz,
                                            rf_step=cfg.grids.rf_center_step_hz,
                                        )

                                        score = robust_margin

                                        # Construct row for this candidate
                                        row = PlanRow(
                                            tile_id=t.id,
                                            if1_center_hz=t.if1_center_hz,
                                            bw_hz=t.bw_hz,
                                            rf_center_hz=t.rf_center_hz,
                                            lo1_name=lo1.name,
                                            lo1_hz=lo1.f_out_hz,
                                            lo1_mode=lo1.mode,
                                            lo1_divider=lo1.divider,
                                            lo1_pad_db=lo1.pad_db,
                                            lo1_lock_ms=lo1.lock_time_ms,
                                            lo2_name=lo2.name,
                                            lo2_hz=lo2.f_out_hz,
                                            lo2_mode=lo2.mode,
                                            lo2_divider=lo2.divider,
                                            lo2_pad_db=lo2.pad_db,
                                            lo2_lock_ms=lo2.lock_time_ms,
                                            if2_center_hz=if2_use.center_hz,
                                            if2_bw_hz=if2_use.bw_hz,
                                            rf_bpf_id=rf_filter.id,
                                            spur_margin_db=score,
                                            brittleness_db_per_step=brittleness,
                                            desired_stage1_sign=inj1,
                                            desired_stage2_sign=s2,
                                        )

                                        if (best_row is None) or (score > best_score + 1e-9):
                                            # New best: reset alternatives
                                            best_row = row
                                            best_summary = summ
                                            best_score = score
                                            alternatives = []
                                            # capture ledger (bins) for the current best
                                            best_ledger = {
                                                "rf_center_hz": float(summ.desired_rf_band.center_hz),
                                                "rf_bw_hz": float(summ.desired_rf_band.bw_hz),
                                                "bins": [
                                                    {
                                                        "f_rf_hz": b.f_rf_hz,
                                                        "combined_level_dbc": b.level_dbc,
                                                        "inband": b.inband,
                                                        "limit_dbc": b.info.get("limit_dbc"),
                                                        "margin_db": b.info.get("margin_db"),
                                                    }
                                                    for b in summ.bins
                                                ],
                                            }
                                        else:
                                            # Track alternatives within Δ dB of best
                                            if best_row is not None and (best_score - score) <= float(cfg.targets.alt_within_db + 1e-9):
                                                alternatives.append({
                                                    "delta_margin_db": float(best_score - score),
                                                    "row": asdict(row),
                                                })

    if best_row is None:
        return None

    # If we used thinned orders for speed, re-score the selected BEST at full/original orders for parity.
    if used_thin_orders and best_summary is not None:
        full_cfg = orig_cfg
        # Recompute carriers (same synths) for the chosen LOs
        lo1_synth = next((s for s in lo1_synths if s.mdl.name == best_row.lo1_name), lo1_synths[0])
        lo2_synth = next((s for s in lo2_synths if s.mdl.name == best_row.lo2_name), lo2_synths[0])
        mode1 = next((m for m in lo1_synth.mdl.modes if m.name == best_row.lo1_mode), lo1_synth.mdl.modes[0])
        mode2 = next((m for m in lo2_synth.mdl.modes if m.name == best_row.lo2_mode), lo2_synth.mdl.modes[0])
        # Minimal LOSolution-like shims using best_row fields
        class _Shim:
            def __init__(self, name, mode, f_out_hz, divider, delivered_dbm, lock_time_ms, pad_db, pfd_hz):
                self.name=name; self.mode=mode; self.f_out_hz=f_out_hz; self.divider=divider
                self.delivered_dbm=delivered_dbm; self.lock_time_ms=lock_time_ms; self.pad_db=pad_db; self.pfd_hz=pfd_hz
        lo1 = _Shim(best_row.lo1_name, best_row.lo1_mode, best_row.lo1_hz, best_row.lo1_divider, 0.0, best_row.lo1_lock_ms, best_row.lo1_pad_db, 10e6)
        lo2 = _Shim(best_row.lo2_name, best_row.lo2_mode, best_row.lo2_hz, best_row.lo2_divider, 0.0, best_row.lo2_lock_ms, best_row.lo2_pad_db, 10e6)
        C1 = lo1_synth.equivalent_carriers(lo1.f_out_hz, mode1, lo1.divider, pfd_hz=lo1.pfd_hz)
        C2 = lo2_synth.equivalent_carriers(lo2.f_out_hz, mode2, lo2.divider, pfd_hz=lo2.pfd_hz)
        rf_filter = next((rf for rf in rf_filters if rf.id == best_row.rf_bpf_id), rf_filters[0])
        if2_use = IF2Parametric(best_row.if2_center_hz, best_row.if2_bw_hz,
                                if2_pass_il, if2_floor, if2_roll)
        mixer1 = mixer1_mdls[0] if mixer1_mdls else None
        mixer2 = mixer2_mdls[0] if mixer2_mdls else None
        if mixer1 and mixer2:
            summ_full = enumerate_spurs(
                Band(best_row.if1_center_hz, best_row.bw_hz),
                lo1, lo2, if2_use, rf_filter, mixer1, mixer2, full_cfg,
                inj1_sign=best_row.desired_stage1_sign,
                s2_sign=best_row.desired_stage2_sign,
                carriers_lo1=C1, carriers_lo2=C2
            )
            # overwrite margin (and ledger) with full-order result
            best_row.spur_margin_db = float(summ_full.worst_margin_db)
            best_ledger = {
                "rf_center_hz": float(summ_full.desired_rf_band.center_hz),
                "rf_bw_hz": float(summ_full.desired_rf_band.bw_hz),
                "bins": [
                    {
                        "f_rf_hz": b.f_rf_hz,
                        "combined_level_dbc": b.level_dbc,
                        "inband": b.inband,
                        "limit_dbc": b.info.get("limit_dbc"),
                        "margin_db": b.info.get("margin_db"),
                    }
                    for b in summ_full.bins
                ],
            }

    # Rank alternatives (smallest delta first)
    alternatives.sort(key=lambda d: d.get("delta_margin_db", 999.0))
    # Attach rank
    for i, a in enumerate(alternatives, 1):
        a["rank"] = i

    return best_row, alternatives, best_ledger


# ---- Retune/lock accounting with Δf and penalties ----------------------------

def _mode_lookup(lo_mdl: LOMdl, name: str) -> Optional[Mode]:
    for m in lo_mdl.modes:
        if m.name == name:
            return m
    return lo_mdl.modes[0] if lo_mdl.modes else None


def _lock_time_for_hop(prev_hz: float, prev_mode: str, prev_divider: str,
                       curr_hz: float, curr_mode: str, curr_divider: str,
                       lo_mdl: LOMdl) -> float:
    mode = _mode_lookup(lo_mdl, curr_mode)
    if mode is None:
        return 0.0
    base = float(mode.lock_time_model.base_ms)
    slope = float(getattr(mode.lock_time_model, "per_mhz_ms", 0.0))
    df_mhz = abs(curr_hz - prev_hz) / 1e6
    t = base + slope * df_mhz
    # crude penalties for mode/divider changes (if provided)
    penalties = mode.lock_time_model.mode_penalties_ms or {}
    if prev_mode != curr_mode:
        t += float(penalties.get("mode_change", penalties.get("int_to_frac", 0.0)))
    if prev_divider != curr_divider:
        t += float(penalties.get("divider_change", 0.0))
    return t


def _retune_accounting(rows: List[PlanRow], lo_models_by_name: Dict[str, LOMdl]) -> dict:
    """
    Uniform traversal retune/lock accounting using hop size and penalties.
    Works even if the LO type changes between rows by using the *destination* row model.
    """
    if not rows:
        return {"total_lock_ms": 0.0, "retunes": 0, "avg_lock_ms_per_hop": 0.0}

    total_lock = 0.0
    retunes = 0
    hops = 0

    prev = rows[0]
    for r in rows[1:]:
        hops += 1
        changed = False
        # LO1 hop
        if abs(r.lo1_hz - prev.lo1_hz) > 0.5 or (r.lo1_mode != prev.lo1_mode) or (r.lo1_divider != prev.lo1_divider) or (r.lo1_name != prev.lo1_name):
            mdl = lo_models_by_name.get(r.lo1_name)
            if mdl is not None:
                total_lock += _lock_time_for_hop(prev.lo1_hz, prev.lo1_mode, prev.lo1_divider,
                                                 r.lo1_hz, r.lo1_mode, r.lo1_divider, mdl)
            changed = True
        # LO2 hop
        if abs(r.lo2_hz - prev.lo2_hz) > 0.5 or (r.lo2_mode != prev.lo2_mode) or (r.lo2_divider != prev.lo2_divider) or (r.lo2_name != prev.lo2_name):
            mdl = lo_models_by_name.get(r.lo2_name)
            if mdl is not None:
                total_lock += _lock_time_for_hop(prev.lo2_hz, prev.lo2_mode, prev.lo2_divider,
                                                 r.lo2_hz, r.lo2_mode, r.lo2_divider, mdl)
            changed = True
        if changed:
            retunes += 1
        prev = r

    avg = total_lock / max(1, hops)
    return {"total_lock_ms": float(total_lock), "retunes": int(retunes), "avg_lock_ms_per_hop": float(avg)}


# ---- Merge/smoothing (spans) -------------------------------------------------

def _same_settings(a: PlanRow, b: PlanRow, eps_hz=500.0) -> bool:
    return (
        a.lo1_name == b.lo1_name and
        a.lo2_name == b.lo2_name and
        abs(a.lo1_hz - b.lo1_hz) < eps_hz and
        abs(a.lo2_hz - b.lo2_hz) < eps_hz and
        a.lo1_mode == b.lo1_mode and
        a.lo2_mode == b.lo2_mode and
        a.lo1_divider == b.lo1_divider and
        a.lo2_divider == b.lo2_divider and
        abs(a.if2_center_hz - b.if2_center_hz) < 1e3 and
        abs(a.if2_bw_hz - b.if2_bw_hz) < 1e3 and
        a.rf_bpf_id == b.rf_bpf_id and
        a.desired_stage1_sign == b.desired_stage1_sign and
        a.desired_stage2_sign == b.desired_stage2_sign
    )

def _merge_adjacent(rows: list[PlanRow]) -> list[PlanRow]:
    """
    Simple pass: along the sorted tile order, merge consecutive rows with identical settings
    by keeping the one with better margin (acts as a coalescer rather than producing spans).
    """
    if not rows:
        return rows
    merged = [rows[0]]
    for r in rows[1:]:
        if _same_settings(merged[-1], r):
            if r.spur_margin_db > merged[-1].spur_margin_db:
                merged[-1] = r
        else:
            merged.append(r)
    return merged

def _emit_spans(rows: list[PlanRow]) -> list[dict]:
    """
    Produce RF spans for each IF1/BW where settings are identical across consecutive RF tiles.
    This is a 1D run-length merge along RF for each IF1 center; rectangular spans across IF1
    are left for later sophistication.
    """
    spans: list[dict] = []
    if not rows:
        return spans
    # Group by (if1_center, bw)
    from collections import defaultdict
    groups: dict[tuple[float,float], list[PlanRow]] = defaultdict(list)
    for r in rows:
        groups[(r.if1_center_hz, r.bw_hz)].append(r)
    for (if1c, bw), gr in groups.items():
        gr_sorted = sorted(gr, key=lambda x: x.rf_center_hz)
        start = gr_sorted[0]
        last = gr_sorted[0]
        for r in gr_sorted[1:]:
            if _same_settings(last, r):
                # continue span
                last = r
            else:
                spans.append({
                    "if1_center_hz": float(if1c),
                    "bw_hz": float(bw),
                    "rf_from_hz": float(start.rf_center_hz),
                    "rf_to_hz": float(last.rf_center_hz),
                    "settings": {
                        "lo1_name": start.lo1_name, "lo1_hz": start.lo1_hz, "lo1_mode": start.lo1_mode, "lo1_divider": start.lo1_divider, "lo1_pad_db": start.lo1_pad_db,
                        "lo2_name": start.lo2_name, "lo2_hz": start.lo2_hz, "lo2_mode": start.lo2_mode, "lo2_divider": start.lo2_divider, "lo2_pad_db": start.lo2_pad_db,
                        "if2_center_hz": start.if2_center_hz, "if2_bw_hz": start.if2_bw_hz, "rf_bpf_id": start.rf_bpf_id,
                        "desired_stage1_sign": start.desired_stage1_sign, "desired_stage2_sign": start.desired_stage2_sign
                    }
                })
                start = r; last = r
        # flush last span
        spans.append({
            "if1_center_hz": float(if1c),
            "bw_hz": float(bw),
            "rf_from_hz": float(start.rf_center_hz),
            "rf_to_hz": float(last.rf_center_hz),
            "settings": {
                "lo1_name": start.lo1_name, "lo1_hz": start.lo1_hz, "lo1_mode": start.lo1_mode, "lo1_divider": start.lo1_divider, "lo1_pad_db": start.lo1_pad_db,
                "lo2_name": start.lo2_name, "lo2_hz": start.lo2_hz, "lo2_mode": start.lo2_mode, "lo2_divider": start.lo2_divider, "lo2_pad_db": start.lo2_pad_db,
                "if2_center_hz": start.if2_center_hz, "if2_bw_hz": start.if2_bw_hz, "rf_bpf_id": start.rf_bpf_id,
                "desired_stage1_sign": start.desired_stage1_sign, "desired_stage2_sign": start.desired_stage2_sign
            }
        })
    return spans


# ---- Driver ------------------------------------------------------------------

def optimize(cfg: Config, models_dir: str, input_files: list[str] | None = None,
             progress_cb: Optional[Callable[[int], None]] = None) -> Dict:
    """
    Optimize policy across all tiles in parallel.
    Adds: alternatives-within-ΔdB, robustness perturbations, multi-device search,
    RF passband edge checks, retune accounting with Δf and penalties.
    Export spur ledgers for best per-tile and coverage gaps; broaden LO search windows (P2);
    generate RF spans per IF1/BW (P2); include file hashes and package versions (P1).
    **NEW**: process-global model cache; IF2 refinement memoization; wider LO2 harmonic quick-screen.
    """
    # Deterministic tiles
    tiles = make_tiles(
        cfg.bands.if1_hz.min,
        cfg.bands.if1_hz.max,
        cfg.bands.rf_hz.min,
        cfg.bands.rf_hz.max,
        cfg.grids.bw_grid_hz,
        cfg.grids.if1_center_step_hz,
        cfg.grids.rf_center_step_hz,
    )

    # Parallel evaluation
    tasks = [(t, cfg.model_dump(), models_dir) for t in tiles]
    rows: List[PlanRow] = []
    alts_by_tile: dict[int, list[dict]] = {}
    ledgers_by_tile: dict[int, dict] = {}
    gaps: list[dict] = []

    # Wrap executor loop with an auto progress context when no callback was given
    with _progress_ctx(total=len(tasks), desc="Tiles", provided_cb=progress_cb) as _update:
        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=_init_models_once,
            initargs=(models_dir, cfg.model_dump()),
        ) as ex:
            futs = [ex.submit(_eval_tile, args) for args in tasks]
            for f in as_completed(futs):
                try:
                    res = f.result()
                    if res is not None:
                        best_row, alts, ledger = res
                        rows.append(best_row)
                        if alts:
                            alts_by_tile[best_row.tile_id] = alts
                        if ledger:
                            ledgers_by_tile[best_row.tile_id] = ledger
                    else:
                        # No legal candidate for this tile
                        pass
                finally:
                    try:
                        _update(1)
                    except Exception:
                        pass

    rows.sort(key=lambda r: r.tile_id)

    # Identify missing tiles (coverage gaps) and tiles below target margin
    present_ids = {r.tile_id for r in rows}
    all_ids = {t.id for t in tiles}
    missing = sorted(all_ids - present_ids)
    for tid in missing:
        # Recover basic tile info
        t = next(tt for tt in tiles if tt.id == tid)
        gaps.append({
            "tile_id": tid,
            "if1_center_hz": float(t.if1_center_hz),
            "bw_hz": float(t.bw_hz),
            "rf_center_hz": float(t.rf_center_hz),
            "reason": "no_legal_candidate"
        })
    # Low margin
    for r in rows:
        if r.spur_margin_db < float(cfg.targets.min_margin_db):
            gaps.append({
                "tile_id": int(r.tile_id),
                "if1_center_hz": float(r.if1_center_hz),
                "bw_hz": float(r.bw_hz),
                "rf_center_hz": float(r.rf_center_hz),
                "reason": "margin_below_target",
                "margin_db": float(r.spur_margin_db),
                "target_db": float(cfg.targets.min_margin_db)
            })

    # ---- second-pass policy-level selection (tie-break by retune/lock and brittleness) ----
    # Build LO model lookup by name for hop computations (also used later for accounting)
    lo_models_by_name: Dict[str, LOMdl] = {}
    for path in (cfg.search.lo1_candidates + cfg.search.lo2_candidates):
        mdl = LOMdl(**_load_yaml(models_dir, path))
        lo_models_by_name[mdl.name] = mdl

    # Create candidate set per tile: best + alternatives (within Δ dB)
    from collections import defaultdict
    candidates_by_tile: dict[int, list[PlanRow]] = defaultdict(list)
    for r in rows:
        candidates_by_tile[r.tile_id].append(r)
    for tid, alts in (alts_by_tile or {}).items():
        for a in alts:
            try:
                candidates_by_tile[int(tid)].append(PlanRow(**a["row"]))
            except Exception:
                # Ignore malformed alternative rows
                pass

    # Greedy sweep in tile order minimizing cost = -(margin) + λ * hop_lock_ms
    # λ tuned by lock_time_penalty_weight; scale: ~0.1 → 10ms ≈ 1 dB
    lam = max(0.0, float(cfg.runtime_policy.lock_time_penalty_weight))
    lam_scale = 0.1  # ms-to-dB scale
    chosen: list[PlanRow] = []
    prev: PlanRow | None = None
    for t in sorted(candidates_by_tile.keys()):
        candset = candidates_by_tile[t]
        # Stable order: highest margin first; on near ties prefer lower brittleness
        candset.sort(key=lambda c: (-float(c.spur_margin_db), float(c.brittleness_db_per_step),
                                    c.lo1_name, c.lo2_name, c.if2_center_hz, c.if2_bw_hz))
        best_c = None
        best_score = -1e18
        for c in candset:
            hop_ms = 0.0
            if prev is not None:
                # LO1 hop cost
                mdl1 = lo_models_by_name.get(c.lo1_name)
                if mdl1 is not None:
                    hop_ms += _lock_time_for_hop(prev.lo1_hz, prev.lo1_mode, prev.lo1_divider,
                                                 c.lo1_hz, c.lo1_mode, c.lo1_divider, mdl1)
                # LO2 hop cost
                mdl2 = lo_models_by_name.get(c.lo2_name)
                if mdl2 is not None:
                    hop_ms += _lock_time_for_hop(prev.lo2_hz, prev.lo2_mode, prev.lo2_divider,
                                                 c.lo2_hz, c.lo2_mode, c.lo2_divider, mdl2)
            score = float(c.spur_margin_db) - lam_scale * lam * hop_ms
            # Soft preference for lower brittleness when score is effectively equal
            if (score > best_score + 1e-12) or (abs(score - best_score) <= 1e-12 and best_c and
                                                c.brittleness_db_per_step < best_c.brittleness_db_per_step):
                best_score = score
                best_c = c
        if best_c is None:
            # Fallback to the original per-tile best if something went wrong
            best_c = candset[0]
        chosen.append(best_c)
        prev = best_c

    # Light merge/smoothing (keeps determinism)
    rows = _merge_adjacent(sorted(chosen, key=lambda r: r.tile_id))

    # Retune/lock-time accounting (uniform traversal) with Δf and penalties
    retune_meta = _retune_accounting(rows, lo_models_by_name)

    # RF spans (per IF1/BW, along RF)
    spans = _emit_spans(rows)

    # Hashes and versions
    file_hashes = {}
    for p in (input_files or []):
        try:
            file_hashes[p] = _file_hash(p)
        except Exception:
            file_hashes[p] = "unavailable"
    pkg_list = ["numpy", "pandas", "scipy", "pydantic", "pyyaml", "jinja2"]
    versions = {}
    for pkg in pkg_list:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "n/a"

    # ---- IF2 window export (unique windows used in the chosen policy) ----
    # We reload IF2 YAML to get constants; windows differ by center/BW here.
    if2_yaml = _load_yaml(models_dir, cfg.search.if2_filter_model)
    pass_il = float(if2_yaml.get("passband_il_db", 1.0))
    floor_db = float(if2_yaml.get("stop_floor_db", -80.0))
    roll_db_dec = float(if2_yaml.get("rolloff_db_per_dec", 40.0))

    from collections import Counter
    # Round to 1 kHz for uniqueness (consistent with earlier eps used)
    def q1k(x: float) -> float:
        return round(float(x) / 1e3) * 1e3

    counts = Counter((q1k(r.if2_center_hz), q1k(r.if2_bw_hz)) for r in rows)
    if2_windows = [{
        "center_hz": float(k[0]),
        "bw_hz": float(k[1]),
        "passband_il_db": pass_il,
        "stop_floor_db": floor_db,
        "rolloff_db_per_dec": roll_db_dec,
        "use_count": int(v),
    } for k, v in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1]))]

    policy = {
        "project": cfg.project.model_dump(),
        "grids": cfg.grids.model_dump(),
        "orders": cfg.orders.model_dump(),
        "targets": cfg.targets.model_dump(),
        "rows": [asdict(r) for r in rows],
        "alternatives": {int(k): v for k, v in alts_by_tile.items() if v},
        "ledgers": ledgers_by_tile,  # per-tile best spur bins
        "spans": spans,
        "coverage_gaps": gaps,
        "if2_windows": if2_windows,
        "meta": {
            "deterministic_seed": cfg.project.seed,
            "retune_accounting": retune_meta,
            "guard_margin_db": cfg.constraints.guard_margin_db,
            "file_hashes": file_hashes,
            "package_versions": versions,
        },
    }
    return policy