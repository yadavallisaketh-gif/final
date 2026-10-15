import numpy as np
import pandas as pd

from src.blackout import BlackoutWindow, apply_blackout, make_blackout_windows, path_length, segment_for_window
from src.metrics import blackout_metrics
from src.synthetic import make_drive, to_frame


def test_windows_are_deterministic_and_inside_one_session(cfg):
    df = to_frame(make_drive(900))
    df.loc[df.t > 600, "session"] = 1
    w1 = make_blackout_windows("syn", df, cfg)
    w2 = make_blackout_windows("syn", df, cfg)
    assert w1 == w2 and len(w1) > 0
    for w in w1:
        seg = segment_for_window(df, w, cfg)
        assert seg["session"].nunique() == 1
        assert seg["t"].min() >= w.t_start - cfg["evaluate"]["warmup_s"] - 1e-9
        assert w.duration in cfg["evaluate"]["durations_s"]


def test_windows_skip_unlabelled_and_stationary(cfg):
    df = to_frame(make_drive(900))
    df["ref_valid"] = False
    assert make_blackout_windows("syn", df, cfg) == []
    df = to_frame(make_drive(900))
    df["ref_speed"] = 0.0
    df["ref_x"] = 0.0
    df["ref_y"] = 0.0
    assert make_blackout_windows("syn", df, cfg) == []


def test_hidden_truth_is_a_copy(cfg):
    df = to_frame(make_drive(300))
    w = BlackoutWindow("syn", 0, 100, 160)
    est, truth = apply_blackout(df, w)
    truth.x[:] = 0
    assert df["ref_x"].abs().sum() > 0          # the original is untouched
    e = est.df
    e["gnss_x"] = 1.0
    assert np.isnan(est.df.loc[w.contains(est.df.t), "gnss_x"]).all()   # stored input untouched


def test_drift_percent_definition():
    t = np.arange(0, 20, 0.1)
    w = BlackoutWindow("syn", 0, 5.0, 15.0)
    truth_x, truth_y = t * 10.0, np.zeros_like(t)
    from src.blackout import HiddenTruth
    truth = HiddenTruth(t, truth_x, truth_y, np.full_like(t, 10.0), np.zeros_like(t), np.ones_like(t, bool))
    est = pd.DataFrame({"t": t, "x": truth_x, "y": truth_y + np.where(w.contains(t), (t - 5.0) * 2.0, 0.0),
                        "v_f": 10.0, "v_l": 0.0, "yaw": 0.0})
    m = blackout_metrics(est, truth, w)
    inside = w.contains(t)
    dist = path_length(truth_x[inside], truth_y[inside])
    assert np.isclose(m["distance_m"], dist)
    assert np.isclose(m["endpoint_error_m"], (t[inside][-1] - 5.0) * 2.0)
    assert np.isclose(m["drift_percent"], 100 * m["endpoint_error_m"] / dist)
