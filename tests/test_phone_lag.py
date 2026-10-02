"""Phone-GNSS latency: online lag estimator, rollback buffer and the engine with a known 2 s lag."""
import numpy as np
import pytest

from src.config import load_config
from src.engine import MODE_GNSS, VARIANTS, NavigationEngine
from src.fusion.ekf2d import EKF2D, X, Y, YAW
from src.fusion.lag import DelayBuffer, PhoneLagEstimator
from src.preprocess import fit_alignment
from src.sensors import GnssFix, SensorSample
from src.synthetic import make_drive, to_frame


def _pc(cfg, **kw):
    pc = dict(cfg["phone_gnss"])
    pc.update(kw)
    return pc


def _lagged_fixes(d, lag_s: float, every_s: float):
    """Fix k is stamped at t_k but carries the truth at t_k - lag (phone-style latency)."""
    t = d.t
    out = []
    for tk in np.arange(t[0] + lag_s + every_s, t[-1], every_s):
        tm = tk - lag_s
        out.append((tk, float(np.interp(tm, t, d.x)), float(np.interp(tm, t, d.y)),
                    float(np.interp(tm, t, d.speed)), float(np.interp(tm, t, np.unwrap(d.yaw)))))
    return out


@pytest.mark.parametrize("every_s", [1.0, 9.0])
def test_estimator_recovers_a_known_2s_lag(cfg, every_s):
    d = make_drive(600, seed=3)
    est = PhoneLagEstimator(_pc(cfg, lag_window_s=120 if every_s == 1 else 300))
    a_f = np.gradient(d.speed, d.t)
    w = np.gradient(np.unwrap(d.yaw), d.t)
    fixes = _lagged_fixes(d, 2.0, every_s)
    taus = []
    j = 0
    for i, ti in enumerate(d.t):
        est.add_imu(ti, w[i], a_f[i])
        while j < len(fixes) and fixes[j][0] <= ti:
            e = est.add_fix(fixes[j][0], fixes[j][3], fixes[j][4])
            if e.source == "online" and fixes[j][0] > 300:
                taus.append(e.tau)
            j += 1
    assert len(taus) > 5
    assert np.median(taus) == pytest.approx(2.0, abs=0.2)


def test_estimator_falls_back_without_information(cfg):
    est = PhoneLagEstimator(_pc(cfg, lag_default_s=0.1))
    for i in range(3000):                                  # straight line, constant speed: nothing to correlate
        t = i / 10
        est.add_imu(t, 0.0, 0.0)
        if i % 10 == 0 and t > 15:
            e = est.add_fix(t, 15.0, 0.3)
    assert e.source == "default" and e.tau == 0.1 and e.confidence == 0.0


def _run_ops(ekf, ops):
    for op in ops:
        if op[0] == "predict":
            ekf.predict(*op[1:])
        elif op[0] == "gnss":
            ekf.update_gnss_position(op[1], op[2], 3.0)


def test_rollback_reproduces_an_on_time_update_exactly(cfg):
    """Filter A gets the fix at its measurement time; filter B gets it 2 s later and rolls back.
    After replay both must be in the same state (same predicts, same update, same order)."""
    rng = np.random.default_rng(0)
    steps = [("predict", 0.1 * k, rng.normal(0, 0.3), rng.normal(0, 0.3), rng.normal(0, 0.05)) for k in range(1, 101)]
    fix = ("gnss", 3.0, -2.0)
    a, b = EKF2D(cfg), EKF2D(cfg)
    for e in (a, b):
        e.initialise(0.0, 0.0, 0.0, 10.0, 0.2, 3.0)
    buf = DelayBuffer(15.0)
    run = lambda op: _run_ops(b, [op])
    for k, st in enumerate(steps, start=1):
        _run_ops(a, [st])
        if k == 40:
            _run_ops(a, [fix])                              # on time, at t = 4.0 s
        run(st)
        buf.record(st)
        if k == 60:                                         # arrives at t = 6.0 s, measured at 4.0 s
            idx = buf.rollback_index(4.0)
            buf.replay(b, idx, fix, run)
        buf.commit(st[1], b, dr=False)
    assert np.allclose(a.s, b.s, atol=1e-10) and np.allclose(a.P, b.P, atol=1e-10)
    assert buf.rollbacks == 1


def test_rollback_never_crosses_dead_reckoning(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0, 3.0)
    buf = DelayBuffer(15.0)
    for k in range(1, 31):
        ekf.predict(0.1 * k, 0.0, 0.0, 0.0)
        buf.commit(0.1 * k, ekf, dr=(k == 25))              # one dead-reckoning step at 2.5 s
    assert buf.rollback_index(2.6) is not None
    assert buf.rollback_index(2.0) is None                  # would replay through the DR step


def _engine_run(cfg, lag_comp, lag_s=2.0, every_s=1.0, default=2.0):
    d = make_drive(420, seed=5, gyro_noise=0.001, acc_noise=0.02)
    df = to_frame(d)
    c = load_config(None, ["phone_gnss.event_updates=true", f"phone_gnss.lag_comp={lag_comp}",
                           f"phone_gnss.lag_default_s={default}", "anomaly.enabled=false"])
    al = fit_alignment(df.query("t < 120"), c)
    eng = NavigationEngine(c, al, VARIANTS["B"], 10.0)
    fixes = _lagged_fixes(d, lag_s, every_s)
    held, j, err = None, 0, []
    acc, gyr = df[["ax", "ay", "az"]].to_numpy(), df[["gx", "gy", "gz"]].to_numpy()
    for i, t in enumerate(df.t.to_numpy()):
        while j < len(fixes) and fixes[j][0] <= t + 1e-9:
            tk, x, y, v, yaw = fixes[j]
            held = GnssFix(x, y, v, yaw, 3.0)              # repeated on every row until the next fix
            j += 1
        mode = eng.step(SensorSample(t, acc[i], gyr[i], held))
        if t > 150 and mode == MODE_GNSS:
            err.append((np.hypot(eng.ekf.s[X] - d.x[i], eng.ekf.s[Y] - d.y[i]),
                        abs((eng.ekf.s[YAW] - d.yaw[i] + np.pi) % (2 * np.pi) - np.pi)))
    return np.array(err), eng


def test_engine_with_a_known_2s_lag_compensated_vs_not():
    off, e0 = _engine_run(load_config(), "off")
    fixed, e1 = _engine_run(load_config(), "fixed")
    online, e2 = _engine_run(load_config(), "online", default=0.0)
    # uncompensated: a 2 s old fix at ~12 m/s pulls the estimate ~24 m behind the car
    assert np.median(off[:, 0]) > 10.0
    assert np.median(fixed[:, 0]) < 0.3 * np.median(off[:, 0])
    assert np.median(online[:, 0]) < 0.5 * np.median(off[:, 0])
    assert e1.delay.rollbacks > 100 and all(r for *_, r in e1.fix_log[5:])
    taus = np.array([x[1] for x in e2.fix_log if x[4] == "online"])
    assert len(taus) > 20 and np.median(taus) == pytest.approx(2.0, abs=0.3)
    # one update per fix (after the filter initialises), not one per 10 Hz row carrying the held fix
    n_fixes = len(_lagged_fixes(make_drive(420, seed=5), 2.0, 1.0))
    assert 0.9 * n_fixes <= len(e0.fix_log) <= n_fixes
    assert e0.ekf.counts()["gnss_pos"] <= n_fixes


def test_lag_comp_requires_event_updates():
    c = load_config(None, ["phone_gnss.lag_comp=fixed"])
    d = make_drive(60)
    with pytest.raises(ValueError, match="event_updates"):
        NavigationEngine(c, fit_alignment(to_frame(d), c), VARIANTS["B"], 10.0)
