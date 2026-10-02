"""Stage 1 repository setup: development profile, config loader, logging helper, Makefile."""
import logging
import os
import subprocess

import pytest

from src.config import DEFAULT_CONFIG, load_config, load_default, stage_settings
from src.logging_utils import get_logger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGETS = ["data", "baseline", "train", "fuse", "figures", "replay", "test", "all"]


def test_default_profile_loads_with_requested_values():
    st = stage_settings(load_default())
    assert st.seed == 42
    assert st.imu_rate_hz == 100.0
    assert (st.window_s, st.stride_s) == (2.0, 0.1)
    assert (st.window_samples, st.stride_samples) == (200, 10)
    assert st.outage_lengths_s == (30.0, 60.0, 120.0)
    assert st.paths["data"] == "data" and st.paths["results"] == "results"


def test_default_profile_inherits_pipeline_and_paths_agree():
    cfg = load_default()
    assert "filter" in cfg and "split" in cfg                     # inherited from base.yaml
    assert cfg["paths"]["raw"] == cfg["data"]["root"]
    assert cfg["paths"]["cache"] == cfg["data"]["cache"]
    assert list(cfg["outages"]["lengths_s"]) == list(cfg["evaluate"]["durations_s"])


def test_mvp_config_is_unchanged_by_the_new_profile():
    base = load_config(DEFAULT_CONFIG)
    mvp = load_config(os.path.join(ROOT, "configs", "sih_mvp.yaml"))
    for cfg in (base, mvp):
        assert cfg["seed"] == 7 and cfg["model"]["window"] == 50
        assert cfg["model"]["path"] == "results/models/motionnet.pt"
        assert "imu" not in cfg and "windows" not in cfg


def test_overrides_and_validation():
    st = stage_settings(load_default(["imu.rate_hz=200", "windows.stride_s=0.05"]))
    assert (st.window_samples, st.stride_samples) == (400, 10)
    with pytest.raises(ValueError):                                # 0.105 s is not whole samples at 100 Hz
        stage_settings(load_default(["windows.stride_s=0.105"]))
    with pytest.raises(ValueError):
        stage_settings(load_default(["windows.stride_s=3.0"]))      # stride longer than the window
    with pytest.raises(ValueError):
        stage_settings(load_default(["outages.lengths_s=[]"]))
    with pytest.raises(KeyError):
        stage_settings(load_config(DEFAULT_CONFIG))                 # base.yaml has no stage sections


def test_logger_writes_once_per_destination(tmp_path):
    path = tmp_path / "logs" / "run.log"
    log = get_logger("avirat.test", log_file=str(path))
    log = get_logger("avirat.test", log_file=str(path))              # second call adds no handlers
    assert len(log.handlers) == 2 and log.level == logging.INFO
    log.info("median drift %.2f%%", 9.91)
    for h in log.handlers:
        h.flush()
    lines = path.read_text().splitlines()
    assert len(lines) == 1 and "avirat.test: median drift 9.91%" in lines[0]


def test_makefile_has_all_targets_and_protects_the_mvp_model():
    text = open(os.path.join(ROOT, "Makefile")).read()
    for t in TARGETS:
        assert f"\n{t}:" in text, t
    dry = subprocess.run(["make", "-n", *TARGETS], cwd=ROOT, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert "model.path=outputs/models/motionnet_candidate.pt" in dry.stdout
    refused = subprocess.run(["make", "train", "NEW_MODEL=results/models/motionnet.pt"], cwd=ROOT,
                             capture_output=True, text=True)
    assert refused.returncode != 0 and "refusing to overwrite" in refused.stdout
