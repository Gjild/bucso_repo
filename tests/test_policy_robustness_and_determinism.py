import os
import yaml
from pathlib import Path
import copy

from bucso.policy import optimize
from bucso.tiling import make_tiles
from bucso.filters import RFBPF
from bucso.models import Config

def test_optimize_small_single_tile(tmp_path, tiny_config, mixer1_model, mixer2_model,
                                    lo1_synth, lo2_synth, rf_filter_flat_pass, if2_window, monkeypatch):
    """
    End-to-end smoke test:
    - Build a tiny models_dir with YAML files constructed from our fixtures
    - Run optimize on a single tile
    - Check output structure & determinism (run twice)
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- write minimal YAMLs the optimizer expects by file path ---
    import yaml as _yaml
    def dump_yaml(obj, path):
        with open(path, "w") as f:
            _yaml.safe_dump(obj, f, sort_keys=False)

    # Mixer1 YAML
    m1y = mixer1_model.mdl.model_dump()
    dump_yaml(m1y, models_dir / "MX1.yaml")
    # Mixer2 YAML
    m2y = mixer2_model.mdl.model_dump()
    dump_yaml(m2y, models_dir / "MX2.yaml")
    # LO1 YAML
    l1y = lo1_synth.mdl.model_dump()
    dump_yaml(l1y, models_dir / "LO1.yaml")
    # LO2 YAML
    l2y = lo2_synth.mdl.model_dump()
    dump_yaml(l2y, models_dir / "LO2.yaml")
    # IF2 YAML
    if2y = {
        "name": "IF2_Model",
        "type": "symmetric_powerlaw",
        "passband_il_db": if2_window.passband_il_db,
        "stop_floor_db": if2_window.stop_floor_db,
        "rolloff_db_per_dec": if2_window.rolloff_db_per_dec,
        "min_bw_hz": 300e6,
        "max_bw_hz": 1200e6,
        "center_range_hz": [1.0e9, 9.0e9],
        "search": {"seeds_per_tile": 4, "local_refinement": "coordinate_descent", "max_refine_iters": 4},
    }
    dump_yaml(if2y, models_dir / "IF2.yaml")
    # RF S21 CSV
    with open(models_dir / "RF.csv", "w") as f:
        f.write("freq_hz,s21_db\n26.0e9,-40\n27.5e9,-15\n28.0e9,-5\n29.0e9,-1\n30.0e9,-1\n31.0e9,-5\n")

    # --- tiny config pointing to those YAMLs ---
    cfg: Config = copy.deepcopy(tiny_config)
    cfg.grids.if1_center_step_hz = 200e6     # keep tile count tiny
    cfg.grids.rf_center_step_hz = 500e6
    cfg.search = cfg.search.__class__(  # Search dataclass
        lo1_candidates=["LO1.yaml"],
        lo2_candidates=["LO2.yaml"],
        mixer1_candidates=["MX1.yaml"],
        mixer2_candidates=["MX2.yaml"],
        rf_bpf_choices=["RF.csv"],
        if2_filter_model="IF2.yaml",
    )

    # Limit tiles to 1 by picking a narrow window manually
    tiles = make_tiles(
        cfg.bands.if1_hz.min, cfg.bands.if1_hz.max,
        cfg.bands.rf_hz.min, cfg.bands.rf_hz.max,
        cfg.grids.bw_grid_hz, cfg.grids.if1_center_step_hz, cfg.grids.rf_center_step_hz
    )
    assert len(tiles) >= 1

    policy1 = optimize(cfg, str(models_dir), input_files=[])
    policy2 = optimize(cfg, str(models_dir), input_files=[])

    # Structure & minimal sanity
    assert "rows" in policy1 and isinstance(policy1["rows"], list)
    assert len(policy1["rows"]) >= 1
    # Determinism: policy dicts should be equal except possibly package_versions order/values
    p1 = copy.deepcopy(policy1); p2 = copy.deepcopy(policy2)
    # normalize versions to avoid environment differences
    if "meta" in p1 and "package_versions" in p1["meta"]:
        p1["meta"]["package_versions"] = {}
        p2["meta"]["package_versions"] = {}
    assert p1 == p2
