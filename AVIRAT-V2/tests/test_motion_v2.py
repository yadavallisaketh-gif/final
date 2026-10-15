"""MotionNet v2: causality, train-only normalisation, streaming == offline, safety, metrics, ONNX."""
import os

import numpy as np
import pytest
import torch

from src.models.motion_classic import row_features
from src.models.motion_data import WindowSet, end_indices
from src.models.motion_eval import error_correlation, recalibrate_conformal, speed_metrics
from src.models.motion_features import V2_CHANNELS, FeatureStream, VibrationEnergy, session_features_causal
from src.models.motion_v2 import ORIGINAL_CHECKPOINT, CausalTCN, GRUBaseline, MotionModelV2, MultiRateNet, MultiScaleGRU
from src.preprocess import fit_alignment
from src.synthetic import make_drive, to_frame


def _drive(seconds=900, seed=0):
    return to_frame(make_drive(seconds, seed=seed))


def _ws(n=3000, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, len(V2_CHANNELS))) + offset).astype(np.float32)
    v = np.abs(rng.normal(10, 4, n)).astype(np.float32)
    ends = np.arange(700, n)
    return WindowSet(X, v, np.zeros(n, np.float32), np.full(n, 2), ends, np.zeros(n, int), ["syn"], np.arange(n) * 0.1)


# --------------------------------------------------------------------------- causality
def test_feature_pipeline_is_causal(cfg):
    """Perturbing the raw IMU after time t must not change any feature at or before t."""
    g = _drive(900)
    a = session_features_causal(g, cfg)
    g2 = g.copy()
    cut = 700.0
    late = g2["t"] > cut
    g2.loc[late, ["ax", "ay", "az", "gx", "gy", "gz"]] += 3.0
    b = session_features_causal(g2, cfg)
    early = (g["t"] <= cut).to_numpy() & a["valid"]
    assert early.sum() > 1000
    assert np.allclose(a["X"][early], b["X"][early], equal_nan=True)


def test_chunk_alignment_uses_only_the_previous_chunk(cfg):
    """Chunk k's features come from an alignment fitted on chunk k-1: changing GNSS inside
    chunk k (the 'future' of that alignment) leaves chunk k's features unchanged."""
    g = _drive(900)
    a = session_features_causal(g, cfg)
    g2 = g.copy()
    in_chunk1 = (g2["t"] >= 300) & (g2["t"] < 600)
    g2.loc[in_chunk1, "gnss_speed"] *= 1.5
    b = session_features_causal(g2, cfg)
    m = in_chunk1.to_numpy() & a["valid"]
    assert np.allclose(a["X"][m], b["X"][m])
    later = ((g["t"] >= 600) & (g["t"] < 900)).to_numpy() & a["valid"]   # chunk 2 is fitted on chunk 1 -> may differ
    assert later.sum() > 0


@pytest.mark.parametrize("net", [GRUBaseline(7), MultiScaleGRU(7), CausalTCN(7)])
def test_networks_use_the_newest_sample_and_nothing_outside_their_context(net):
    torch.manual_seed(0)
    net.eval()
    x = torch.randn(2, 600, 7)
    y0 = net(x)["mu"]
    x1 = x.clone()
    x1[:, -1] += 5.0
    assert not torch.allclose(net(x1)["mu"], y0)                   # 'now' matters
    if isinstance(net, CausalTCN):
        x2 = x.clone()
        x2[:, : 600 - net.receptive] += 5.0                          # outside the receptive field
        assert torch.allclose(net(x2)["mu"], y0, atol=1e-5)


def test_multirate_net_refuses_low_rate_and_runs_at_200hz():
    with pytest.raises(ValueError, match="needs a >= 50 Hz IMU"):
        MultiRateNet(7, raw_rate_hz=10.0)
    net = MultiRateNet(7, raw_rate_hz=200.0).eval()
    x, raw = torch.randn(2, 600, 7), torch.randn(2, 600 * 20, 6)
    y0 = net(x, raw)["mu"]
    raw2 = raw.clone()
    raw2[:, -1] += 5.0
    assert not torch.allclose(net(x, raw2)["mu"], y0)               # the newest raw sample is used
    assert y0.shape == (2,)


def test_row_features_are_causal():
    ws = _ws()
    idx = np.arange(50, 60)
    f0 = row_features(ws, idx)
    ws.X[ws.ends[idx].max() + 1:] += 100.0
    assert np.allclose(row_features(ws, idx), f0)


# --------------------------------------------------------------------------- streaming == offline
def test_vibration_energy_stream_equals_block():
    x = np.random.default_rng(0).normal(9.81, 0.3, 500)
    a, b = VibrationEnergy(10.0, 1e-4), VibrationEnergy(10.0, 1e-4)
    assert np.allclose([a.update(v) for v in x], b.block(x))


def test_feature_stream_equals_offline_chunk(cfg):
    g = _drive(900)
    off = session_features_causal(g, cfg)
    t = g["t"].to_numpy()
    c = t[0] + 600.0                                                 # third chunk
    al = fit_alignment(g[(t >= c - 300) & (t < c)], cfg)
    fs = FeatureStream(al, cfg, 10.0)
    run = np.nonzero((t >= c - 60) & (t < c + 300))[0]
    stream = np.array([fs.step(g[["ax", "ay", "az"]].to_numpy()[i], g[["gx", "gy", "gz"]].to_numpy()[i]) for i in run])
    keep = t[run] >= c + 10                                          # energy EMAs started 60 s earlier: settled
    ref = off["X"][run][keep]
    assert np.allclose(stream[keep][:, :5], ref[:, :5], atol=1e-5)
    assert np.allclose(stream[keep][:, 5:], ref[:, 5:], atol=1e-4)


# --------------------------------------------------------------------------- protocol and safety
def test_normalisation_comes_from_training_windows_only(cfg):
    from src.train_motion_v2 import EXPERIMENTS, train
    exp = dict(EXPERIMENTS["B1"], name="t")
    tr, va = _ws(seed=1), _ws(seed=2, offset=50.0)                   # validation lives far away
    m, _ = train(cfg, exp, tr, va, epochs=1, per_epoch=512, batch=256, log=lambda *_: None)
    assert np.allclose(m.mean, tr.X[tr.ends].mean(0), atol=1e-5)
    assert np.all(np.abs(m.mean) < 1.0)                              # untouched by the +50 validation offset


def test_original_checkpoint_cannot_be_overwritten(tmp_path):
    m = MotionModelV2(MultiScaleGRU(7), "msgru", {}, V2_CHANNELS, 600, np.zeros(7), np.ones(7))
    with pytest.raises(PermissionError):
        m.save(ORIGINAL_CHECKPOINT)
    m.save(str(tmp_path / "x.pt"))
    assert MotionModelV2.load(str(tmp_path / "x.pt")).n_params == m.n_params


def test_end_indices_need_full_valid_history_in_one_session():
    valid = np.ones(30, bool)
    valid[10] = False
    session = np.r_[np.zeros(20, int), np.ones(10, int)]
    e = end_indices(valid, session, history=5)
    assert e.tolist() == [4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29]   # gap at 10, new session at 20


# --------------------------------------------------------------------------- metrics
def test_error_correlation_recovers_ar1():
    rng = np.random.default_rng(0)
    rho = 0.95
    e = np.zeros(20000)
    for i in range(1, len(e)):
        e[i] = rho * e[i - 1] + rng.standard_normal()
    r = error_correlation(e, [np.arange(len(e))])
    assert abs(r["rho_0.1s"] - rho) < 0.01
    assert abs(r["integral_time_s"] - 0.1 * (1 + rho) / (2 * (1 - rho))) < 0.3


def test_calibrated_gaussian_has_nominal_coverage_and_conformal_q():
    rng = np.random.default_rng(0)
    sd = rng.uniform(0.5, 3, 50000)
    y = rng.normal(0, sd)
    m = speed_metrics(np.zeros_like(y), sd, y)
    assert abs(m["cov90"] - 0.9) < 0.01 and abs(m["z_rms"] - 1) < 0.02 and m["ece"] < 0.01
    assert abs(recalibrate_conformal(np.zeros_like(y), sd, y, 0.9) - 1.645) < 0.03


def test_onnx_export_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    m = MotionModelV2(MultiScaleGRU(7), "msgru", {}, V2_CHANNELS, 600, np.zeros(7, np.float32),
                      np.ones(7, np.float32), {"sigma_scale": 1.5})
    p = str(tmp_path / "m.pt")
    m.save(p)
    from scripts.export_onnx_v2 import main
    main([p, "--out", str(tmp_path / "m.onnx")])
    import json
    meta = json.load(open(tmp_path / "m.json"))
    assert max(meta["parity_max_abs_diff"].values()) < 1e-4
    assert meta["host_cpu_latency_ms"]["note"].startswith("host x86 CPU")
    assert os.path.getsize(tmp_path / "m.onnx") < 200_000
