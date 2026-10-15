"""Judge question: "Are you secretly using GPS speed?" - these tests say no."""
import numpy as np
import pytest
import torch

from src.blackout import BlackoutWindow, EstimatorInput, LeakageError, apply_blackout
from src.engine import VARIANTS, NavigationEngine
from src.fusion.ekf2d import EKF2D, GnssDisabledError
from src.models.motion_net import MotionModel, MotionGRU, assert_allowed_features
from src.preprocess import fit_alignment
from src.sensors import ReplaySource
from src.synthetic import make_drive, to_frame


def _setup(cfg, t0=200.0, t1=260.0):
    df = to_frame(make_drive(400))
    w = BlackoutWindow("synthetic", 0, t0, t1)
    est, truth = apply_blackout(df, w)
    al = fit_alignment(est.df[est.df.t < t0], cfg)
    return df, w, est, truth, al


def _tiny_model(window=20):
    feats = ["a_f", "a_l", "a_u", "w_u", "w_h"]
    net = MotionGRU(len(feats), 8)
    return MotionModel(net, "gru", feats, window, np.zeros(5), np.ones(5), hidden=8)


def test_estimator_input_has_no_reference_columns(cfg):
    df, w, est, truth, al = _setup(cfg)
    assert not any(c.startswith("ref_") for c in est.df.columns)
    with pytest.raises(LeakageError):
        EstimatorInput(df, w)                       # full table incl. truth is refused


def test_gnss_masked_inside_blackout(cfg):
    df, w, est, truth, al = _setup(cfg)
    e = est.df
    inside = w.contains(e.t)
    assert not e.loc[inside, "gnss_healthy"].any()
    assert e.loc[inside, ["gnss_x", "gnss_y", "gnss_speed", "gnss_yaw"]].isna().all().all()
    assert e.loc[~inside, "gnss_healthy"].all()
    # the truth is still complete for scoring
    assert np.isfinite(truth.x[inside]).all()


def test_tampering_with_the_mask_is_detected(cfg):
    df, w, est, truth, al = _setup(cfg)
    e = est.df
    e.loc[w.contains(e.t), "gnss_healthy"] = True
    with pytest.raises(LeakageError):
        EstimatorInput(e, w)


def test_replay_source_emits_no_fix_inside_blackout(cfg):
    df, w, est, truth, al = _setup(cfg)
    for s in ReplaySource(est):
        if w.contains(s.t):
            assert s.gnss is None


@pytest.mark.parametrize("key", list(VARIANTS))
def test_no_gnss_update_during_blackout_any_variant(cfg, key):
    df, w, est, truth, al = _setup(cfg)
    v = VARIANTS[key]
    from src.constraints.map_match import MapMatcher, RoadNetwork
    net = RoadNetwork.from_traces([(truth.x, truth.y)])
    eng = NavigationEngine(cfg, al, v, 10.0, _tiny_model() if v.use_motion else None,
                           MapMatcher(net, cfg) if v.use_map else None)
    eng.run(ReplaySource(est))
    inside = eng.ekf.counts(w.t_start, w.t_end, accepted_only=False)
    assert not any(k.startswith("gnss") for k in inside), inside
    before = eng.ekf.counts(0, w.t_start)
    assert before.get("gnss_pos", 0) > 100               # GNSS was genuinely used when healthy
    after = eng.ekf.counts(w.t_end, np.inf, accepted_only=False)
    assert after.get("gnss_pos", 0) > 0                   # ... and again after reacquisition
    if v.use_motion:
        assert inside.get("motionnet", 0) > 0


def test_gnss_toggle_off_removes_all_gnss_updates(cfg):
    """Even if fixes are present in the stream, gnss_enabled=False ignores them."""
    df = to_frame(make_drive(300))
    est, truth = apply_blackout(df, None)                  # no blackout: every sample has a fix
    al = fit_alignment(est.df, cfg)
    eng = NavigationEngine(cfg, al, VARIANTS["B"], 10.0)
    samples = list(ReplaySource(est))
    for s in samples[:1500]:
        eng.step(s)
    eng.gnss_enabled = False
    n_before = len([r for r in eng.ekf.log if r.source.startswith("gnss")])
    for i, s in enumerate(samples[1500:]):
        assert s.gnss is not None
        mode = eng.step(s)
        if i > 10 * cfg["filter"]["gnss_timeout_s"] + 1:
            assert mode == "DEAD RECKONING"
    n_after = len([r for r in eng.ekf.log if r.source.startswith("gnss")])
    assert n_after == n_before


def test_ekf_refuses_gnss_when_disabled(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0, 0, 0, 10, 0)
    ekf.gnss_enabled = False
    for call in (lambda: ekf.update_gnss_position(1, 1, 3), lambda: ekf.update_gnss_speed(10),
                 lambda: ekf.update_gnss_heading(0.1)):
        with pytest.raises(GnssDisabledError):
            call()


@pytest.mark.parametrize("bad", ["gnss_speed", "ref_speed", "gps_lat", "wheel_speed", "ref_x", "speed"])
def test_forbidden_features_rejected(bad):
    with pytest.raises(AssertionError):
        assert_allowed_features(["a_f", bad])
    with pytest.raises(AssertionError):
        MotionModel(MotionGRU(2, 4), "gru", ["a_f", bad], 10, np.zeros(2), np.ones(2))


def test_model_input_shape_is_enforced():
    m = _tiny_model()
    with pytest.raises(AssertionError):
        m.predict(np.zeros((20, 6)))                       # a sneaked-in extra column
    mu, sd = m.predict(np.zeros((20, 5)))
    assert mu >= 0 and sd > 0


def test_model_checkpoint_roundtrip(tmp_path):
    m = _tiny_model()
    p = str(tmp_path / "m.pt")
    m.save(p)
    m2 = MotionModel.load(p)
    x = np.random.default_rng(0).normal(size=(3, 20, 5)).astype(np.float32)
    assert np.allclose(m.predict_batch(x)[0], m2.predict_batch(x)[0])
    assert m2.features == m.features
    torch.manual_seed(0)
