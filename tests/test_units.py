from __future__ import annotations
import yaml, json
from pathlib import Path
from bucso.models import Config
from bucso.policy import optimize

def test_config_and_opt(tmp_path):
    examples = tmp_path/"examples"; examples.mkdir()
    from bucso.cli import EXAMPLE_CONFIG, EXAMPLE_IF2, EXAMPLE_MIXER, EXAMPLE_MIXER2, EXAMPLE_LO, EXAMPLE_LO2, EXAMPLE_RF_CSV
    (examples/"config.yaml").write_text(EXAMPLE_CONFIG)
    (examples/"IF2_Model_01.yaml").write_text(EXAMPLE_IF2)
    (examples/"MXR1.yaml").write_text(EXAMPLE_MIXER)
    (examples/"MXR2.yaml").write_text(EXAMPLE_MIXER2)
    (examples/"LMX2595_A.yaml").write_text(EXAMPLE_LO)
    (examples/"LMX2592_A.yaml").write_text(EXAMPLE_LO2)
    (examples/"RF_S21_28to31GHz.csv").write_text(EXAMPLE_RF_CSV)

    cfg = Config(**yaml.safe_load((examples/"config.yaml").read_text()))
    policy = optimize(cfg, str(examples))
    assert "rows" in policy
    assert isinstance(policy["rows"], list)
    # There should be at least 1 row (coarse settings may be sparse but nonzero)
    assert len(policy["rows"]) >= 1
