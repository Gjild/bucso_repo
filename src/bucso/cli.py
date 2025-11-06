from __future__ import annotations
import os, sys, json, yaml, hashlib
from pathlib import Path
import typer
from rich import print
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

from .models import Config
from .policy import optimize as optimize_policy
from .tiling import make_tiles

app = typer.Typer(no_args_is_help=True)

@app.command()
def init_stubs(dst: str = "examples"):
    """Write example YAML/CSV stubs."""
    p = Path(dst)
    p.mkdir(parents=True, exist_ok=True)
    (p/"config.yaml").write_text(EXAMPLE_CONFIG)
    (p/"MXR1.yaml").write_text(EXAMPLE_MIXER)
    (p/"MXR2.yaml").write_text(EXAMPLE_MIXER2)
    (p/"LMX2595_A.yaml").write_text(EXAMPLE_LO)
    (p/"LMX2592_A.yaml").write_text(EXAMPLE_LO2)
    (p/"IF2_Model_01.yaml").write_text(EXAMPLE_IF2)
    (p/"RF_S21_28to31GHz.csv").write_text(EXAMPLE_RF_CSV)
    print(f"[green]Stub inputs written to {p}[/green]")

def _resolve_model_paths(cfg: Config, cfg_path: str) -> list[str]:
    """Collect absolute file paths for hashing/metadata."""
    base = Path(cfg_path).parent.resolve()
    files: list[str] = [str(Path(cfg_path).resolve())]

    def _abspath(p):
        return str((p if os.path.isabs(p) else base / p).resolve())

    files.extend(_abspath(p) for p in cfg.search.mixer1_candidates)
    files.extend(_abspath(p) for p in cfg.search.mixer2_candidates)
    files.extend(_abspath(p) for p in cfg.search.lo1_candidates)
    files.extend(_abspath(p) for p in cfg.search.lo2_candidates)
    files.extend(_abspath(p) for p in cfg.search.rf_bpf_choices)
    files.append(_abspath(cfg.search.if2_filter_model))
    # de-dup and keep deterministic order
    files = sorted(set(files))
    return files

def _write_runtime_selector(dst_py: Path, hysteresis_hz: float):
    """Emit a simple runtime selector π that loads policy.csv and applies hysteresis."""
    code = f'''# Auto-generated runtime selector (π)
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path

HYSTERESIS_HZ = {hysteresis_hz:.1f}

@dataclass(frozen=True)
class Row:
    if1_center_hz: float
    bw_hz: float
    rf_center_hz: float
    lo1_name: str
    lo1_hz: float
    lo1_mode: str
    lo1_divider: str
    lo1_pad_db: float
    lo2_name: str
    lo2_hz: float
    lo2_mode: str
    lo2_divider: str
    lo2_pad_db: float
    if2_center_hz: float
    if2_bw_hz: float
    rf_bpf_id: str
    spur_margin_db: float
    desired_stage1_sign: int
    desired_stage2_sign: int

def load_policy_csv(path: str | Path) -> list[Row]:
    rows: list[Row] = []
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(Row(
                if1_center_hz=float(r["if1_center_hz"]),
                bw_hz=float(r["bw_hz"]),
                rf_center_hz=float(r["rf_center_hz"]),
                lo1_name=r["lo1_name"],
                lo1_hz=float(r["lo1_hz"]),
                lo1_mode=r["lo1_mode"],
                lo1_divider=r["lo1_divider"],
                lo1_pad_db=float(r["lo1_pad_db"]),
                lo2_name=r["lo2_name"],
                lo2_hz=float(r["lo2_hz"]),
                lo2_mode=r["lo2_mode"],
                lo2_divider=r["lo2_divider"],
                lo2_pad_db=float(r["lo2_pad_db"]),
                if2_center_hz=float(r["if2_center_hz"]),
                if2_bw_hz=float(r["if2_bw_hz"]),
                rf_bpf_id=r["rf_bpf_id"],
                spur_margin_db=float(r["spur_margin_db"]),
                desired_stage1_sign=int(r.get("desired_stage1_sign","1")),
                desired_stage2_sign=int(r.get("desired_stage2_sign","1")),
            ))
    return rows

def select_settings(rows: list[Row], if1_center_hz: float, bw_hz: float, rf_center_hz: float, last: Row | None = None) -> Row:
    # Hysteresis: if last exists and requested RF is within ±HYSTERESIS_HZ, keep last
    if last and abs(rf_center_hz - last.rf_center_hz) <= HYSTERESIS_HZ and abs(if1_center_hz - last.if1_center_hz) <= HYSTERESIS_HZ and abs(bw_hz - last.bw_hz) < 1:
        return last
    # Nearest neighbor on (if1_center, bw, rf_center)
    best = None; best_d = 1e99
    for r in rows:
        if abs(r.bw_hz - bw_hz) > 1:  # require exact BW
            continue
        d = abs(r.if1_center_hz - if1_center_hz) + abs(r.rf_center_hz - rf_center_hz)
        if d < best_d:
            best, best_d = r, d
    return best if best else rows[0]
'''
    dst_py.parent.mkdir(parents=True, exist_ok=True)
    dst_py.write_text(code, encoding="utf-8")

@app.command()
def validate(cfg_path: str):
    """Validate config YAML loads and basic constraints."""
    cfg = Config(**yaml.safe_load(Path(cfg_path).read_text()))
    assert cfg.bands.if1_hz.min < cfg.bands.if1_hz.max
    assert cfg.bands.rf_hz.min < cfg.bands.rf_hz.max
    print("[green]Config validated.[/green]")

@app.command()
def optimize(cfg_path: str, out: str = "out", models_dir: str | None = None):
    """Run optimization and write policy YAML/CSV + runtime selector. Shows a progress bar."""
    cfg = Config(**yaml.safe_load(Path(cfg_path).read_text()))
    mdir = models_dir or str(Path(cfg_path).parent)

    # Compute tiles up-front so we can size the progress bar deterministically
    tiles = make_tiles(
        cfg.bands.if1_hz.min, cfg.bands.if1_hz.max,
        cfg.bands.rf_hz.min, cfg.bands.rf_hz.max,
        cfg.grids.bw_grid_hz, cfg.grids.if1_center_step_hz, cfg.grids.rf_center_step_hz
    )
    total_tiles = len(tiles)

    # Progress bar
    progress_columns = (
        SpinnerColumn(),
        "[bold]Optimizing[/bold]",
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    # File list (for hashes)
    input_files = _resolve_model_paths(cfg, cfg_path)

    with Progress(*progress_columns) as progress:
        task_id = progress.add_task("Optimizing", total=total_tiles)

        def _progress_cb(done: int):
            # called once per tile completion from the policy layer
            progress.update(task_id, advance=done)

        policy = optimize_policy(cfg, mdir, input_files=input_files, progress_cb=_progress_cb)

    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    (outp/"policy.yaml").write_text(yaml.safe_dump(policy, sort_keys=False))
    # Compact CSV
    rows = policy["rows"]
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(outp/"policy.csv", index=False)
    print(f"[green]Wrote {len(rows)} policy rows to {outp}[/green]")

    # Spur ledgers (optional, compact JSONL per tile)
    ledgers = policy.get("ledgers", {})
    if ledgers:
        (outp/"ledgers.jsonl").write_text(
            "\n".join(json.dumps({"tile_id": int(k), "bins": v}, ensure_ascii=False) for k, v in ledgers.items()),
            encoding="utf-8"
        )

    # Spans & coverage gaps
    if policy.get("spans"):
        (outp/"spans.yaml").write_text(yaml.safe_dump(policy["spans"], sort_keys=False), encoding="utf-8")
    if policy.get("coverage_gaps"):
        (outp/"coverage_gaps.yaml").write_text(yaml.safe_dump(policy["coverage_gaps"], sort_keys=False), encoding="utf-8")

    # NEW: IF2 windows export
    if policy.get("if2_windows"):
        (outp/"if2_windows.yaml").write_text(yaml.safe_dump(policy["if2_windows"], sort_keys=False), encoding="utf-8")
        print(f"[green]Wrote unique IF2 windows to {outp/'if2_windows.yaml'}[/green]")

    # Runtime selector (π)
    _write_runtime_selector(outp/"runtime_selector.py", hysteresis_hz=cfg.runtime_policy.hysteresis_hz)

    print(f"[green]Wrote runtime selector to {outp/'runtime_selector.py'}[/green]")

@app.command()
def report(policy_yaml: str, html: str = "out/summary.html"):
    """Very light HTML summary."""
    policy = yaml.safe_load(Path(policy_yaml).read_text())
    from jinja2 import Template
    tpl = Template("""
    <html><head><meta charset="utf-8"><title>BUCSO Summary</title>
    <style>table {border-collapse: collapse} td,th{border:1px solid #ccc;padding:4px}</style>
    </head><body>
    <h1>{{proj.name}}</h1>
    <p>Rows: {{rows|length}} | Seed: {{meta.deterministic_seed}}</p>
    <p>Total lock: {{meta.retune_accounting.total_lock_ms}} ms | Retunes: {{meta.retune_accounting.retunes}} | Avg/step: {{meta.retune_accounting.avg_lock_ms_per_hop}} ms</p>
    <p><strong>Note:</strong> IF2 search now uses local + global anchors; RF S21 supports CSV or YAML; S21 lookups are robust near DC.</p>
    <p>Unique IF2 windows: {{ (policy.if2_windows or []) | length }}</p>
    <p>Input hashes:</p>
    <pre style="font-size:12px">{{ meta.file_hashes | tojson(indent=2) }}</pre>
    <table>
      <tr>
        <th>tile</th><th>IF1c</th><th>BW</th><th>RFc</th>
        <th>LO1</th><th>LO2</th><th>IF2c</th><th>IF2BW</th><th>RF BPF</th><th>Margin (dB)</th><th>Brittleness</th>
      </tr>
      {% for r in rows %}
      <tr>
        <td>{{r.tile_id}}</td>
        <td>{{"%.3f"%(r.if1_center_hz/1e9)}} GHz</td>
        <td>{{"%.0f"%(r.bw_hz/1e6)}} MHz</td>
        <td>{{"%.3f"%(r.rf_center_hz/1e9)}} GHz</td>
        <td>{{"%.3f"%(r.lo1_hz/1e9)}} GHz</td>
        <td>{{"%.3f"%(r.lo2_hz/1e9)}} GHz</td>
        <td>{{"%.3f"%(r.if2_center_hz/1e9)}} GHz</td>
        <td>{{"%.0f"%(r.if2_bw_hz/1e6)}} MHz</td>
        <td>{{r.rf_bpf_id}}</td>
        <td>{{"%.1f"%r.spur_margin_db}}</td>
        <td>{{"%.2f"%r.brittleness_db_per_step}}</td>
      </tr>
      {% endfor %}
    </table>
    </body></html>""")
    html_text = tpl.render(proj=policy["project"], rows=policy["rows"], meta=policy["meta"], policy=policy)
    Path(html).parent.mkdir(parents=True, exist_ok=True)
    Path(html).write_text(html_text, encoding="utf-8")
    print(f"[green]Wrote {html}[/green]")

# --- Example stubs embedded so `init-stubs` needs no downloads ---

EXAMPLE_CONFIG = """\
project:
  name: "KaBUC-SpurPlan"
  seed: 42
  reference_10mhz_hz: 10000000

bands:
  if1_hz: {min: 950e6, max: 2150e6}
  rf_hz: {min: 27.5e9, max: 31.0e9}
  required_bandwidths_hz: [400e6]

grids:
  if1_center_step_hz: 100e6
  rf_center_step_hz: 250e6
  bw_grid_hz: [400e6]
  snap_if1_rf_to_grid: true

masks:
  inband: {default_dbc: -60, table: []}
  outofband: {default_dbc: -60, table: []}

rbw_binning:
  rbw_hz: 10e3
  rbw_frac_of_bw: 0.0025
  rbw_ppm_of_freq: 0
  # NOTE: The tool computes the coalescing window as max(rbw_hz, rbw_frac_of_bw*BW, rbw_ppm_of_freq*freq).
  #       A string expression knob (coalesce_window) from the design doc is not implemented on purpose.

search:
  lo1_candidates: ["LMX2595_A.yaml"]
  lo2_candidates: ["LMX2592_A.yaml"]
  mixer1_candidates: ["MXR1.yaml"]
  mixer2_candidates: ["MXR2.yaml"]
  rf_bpf_choices: ["RF_S21_28to31GHz.csv"]
  if2_filter_model: "IF2_Model_01.yaml"

orders:
  m1n1_max_abs: 7
  m2n2_max_abs: 7
  cross_stage_sum_max: 12

constraints:
  enforce_desired_mn11_only: true
  guard_margin_db: 2.0
  desired_stage1_sign: +1
  desired_stage2_sign: +1

early_reject:
  image_in_if2_passband: true
  loft_in_if2_or_rf_passbands: true
  rf_first_order_image_in_passband: true

runtime_policy:
  hysteresis_hz: 10e6
  prefer_fewer_retunes: true
  lock_time_penalty_weight: 1.0
  markov_transition_matrix_csv: null

targets:
  min_margin_db: 0
  alt_within_db: 3.0
"""

EXAMPLE_MIXER = """\
name: "MXR1"
type: "double-balanced"
if_range_hz: [50e6, 8e9]
lo_range_hz: [100e6, 18e9]
rf_range_hz: [50e6, 18e9]

required_lo_drive_dbm: {min: 11, max: 17}
drive_derate: {nominal_dbm: 13, slope_db_per_db: 1.0, max_derate_db: 6}

spur_table:
  entries:
    - {m: 1, n: 1, rej_dbc: -35}
    - {m: 1, n: 2, rej_dbc: -45}
    - {m: 2, n: 1, rej_dbc: -40}
  grids:
    lo_hz: [500e6, 2e9, 6e9, 12e9]
    if_hz: [100e6, 500e6, 1e9]
    rej_dbc: [[-32,-34,-36], [-33,-35,-37], [-35,-37,-40], [-36,-38,-42]]

isolation:
  lo_to_rf_db: -40
  if_to_rf_db: -60

lo_family_scaling:
  default_slope_db_per_db: 1.0
  cap_db: 12
"""

EXAMPLE_MIXER2 = """\
name: "MXR2"
type: "double-balanced"
if_range_hz: [50e6, 12e9]
lo_range_hz: [100e6, 26e9]
rf_range_hz: [50e6, 32e9]

required_lo_drive_dbm: {min: 11, max: 17}
drive_derate: {nominal_dbm: 13, slope_db_per_db: 1.0, max_derate_db: 6}

spur_table:
  entries:
    - {m: 1, n: 1, rej_dbc: -30}
    - {m: 1, n: 2, rej_dbc: -38}
    - {m: 2, n: 1, rej_dbc: -36}
  grids:
    lo_hz: [2e9, 10e9, 20e9, 26e9]
    if_hz: [100e6, 2e9, 6e9]
    rej_dbc: [[-28,-30,-32], [-29,-31,-33], [-31,-33,-35], [-32,-34,-36]]

isolation:
  lo_to_rf_db: -38
  if_to_rf_db: -55

lo_family_scaling:
  default_slope_db_per_db: 1.0
  cap_db: 12
"""

EXAMPLE_LO = """\
name: "LMX2595_A"
ref_constraints:
  allowed_refs_hz: [10e6]

freq_range_hz: [10e6, 20e9]
step_hz: 1e6

output_power_model:
  table:
    freq_hz: [5e9, 10e9, 15e9, 20e9]
    p_out_dbm: [19, 19, 19, 19]
  divider_adjust_db:
    "/1": 0
    "/2": -1
    "/4": -2
    "/8": -3

distribution:
  path_losses_db: {lo1: 3.0}
  pad_options_db: [0, 3, 6]

modes:
  - name: "fracN"
    enabled: true
    pfd_hz_range: [10e6, 200e6]
    pfd_dividers: [1,2,4,8]
    vco_dividers: [1,2,4,8,16]
    lock_time_model:
      base_ms: 0.40
      per_mhz_ms: 0.002
      mode_penalties_ms: {int_to_frac: 0.30, frac_to_int: 0.20}
    harmonics_at_output:
      - {k: 2, rel_dBc: -25}
      - {k: 3, rel_dBc: -35}
    pfd_spurs_at_output:
      families: []

divider_spectrum:
  "/1": { harm_delta_dBc: 0 }
  "/2": { harm_delta_dBc: 6 }
  "/4": { harm_delta_dBc: 12 }
  "/8": { harm_delta_dBc: 18 }
"""

EXAMPLE_LO2 = """\
name: "LMX2592_A"
ref_constraints:
  allowed_refs_hz: [10e6]

freq_range_hz: [10e6, 26e9]
step_hz: 1e6

output_power_model:
  table:
    freq_hz: [5e9, 10e9, 20e9, 26e9]
    p_out_dbm: [15, 15, 15, 15]
  divider_adjust_db:
    "/1": 0

distribution:
  path_losses_db: {lo2: 4.0}
  pad_options_db: [0, 3, 6]

modes:
  - name: "fracN"
    enabled: true
    pfd_hz_range: [10e6, 200e6]
    pfd_dividers: [1,2,4,8]
    vco_dividers: [1,2,4,8,16]
    lock_time_model:
      base_ms: 0.50
      per_mhz_ms: 0.002
      mode_penalties_ms: {int_to_frac: 0.30, frac_to_int: 0.20}
    harmonics_at_output:
      - {k: 2, rel_dBc: -25}
    pfd_spurs_at_output:
      families: []

divider_spectrum:
  "/1": { harm_delta_dBc: 0 }
"""

EXAMPLE_IF2 = """\
name: "IF2_Model_01"
type: "symmetric_powerlaw"
passband_il_db: 1.0
stop_floor_db: -80.0
rolloff_db_per_dec: 40.0
min_bw_hz: 500e6
max_bw_hz: 6000e6
center_range_hz: [500e6, 9000e6]
search:
  # seeds_per_tile CONTROLS IF2 SEARCH DENSITY deterministically (both center and BW seeds).
  # Larger = broader exploration; keep modest to bound runtime.
  seeds_per_tile: 8
  local_refinement: "coordinate_descent"
  max_refine_iters: 12
  bounds_guard_hz: 0.0
"""

EXAMPLE_RF_CSV = """\
freq_hz,s21_db
2.60e10,-30
2.75e10,-5
2.90e10,-1
3.10e10,-1
3.25e10,-10
"""