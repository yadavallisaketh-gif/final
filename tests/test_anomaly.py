import numpy as np

from src.anomaly_detector import AnomalyDetector
from src.engine import VARIANTS, NavigationEngine
from src.preprocess import fit_alignment
from src.sensors import SensorSample
from src.synthetic import make_drive, to_frame


def _rot_x(deg):
    a = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _run(cfg, acc, gyr, rate=10.0):
    det = AnomalyDetector(cfg, rate)
    for i, (a, g) in enumerate(zip(acc, gyr)):
        det.update(i / rate, a, g)
    return det


def test_clean_drive_with_turns_raises_nothing(cfg):
    d = make_drive(400)
    det = _run(cfg, d.acc, d.gyro)
    assert det.counts() == {"shock": 0, "transient": 0, "slip": 0}


def test_pothole_spike_is_a_shock(cfg):
    d = make_drive(200)
    acc = d.acc.copy()
    acc[1000, 2] += 25.0                                   # one-sample vertical impact at 10 Hz
    det = _run(cfg, acc, d.gyro)
    assert det.counts()["shock"] == 1 and det.counts()["slip"] == 0
    assert abs(det.events[0].t - 100.0) < 0.15


def test_mount_slip_is_confirmed_and_relevels(cfg):
    d = make_drive(200)
    acc, gyr = d.acc.copy(), d.gyro.copy()
    R = _rot_x(15.0)                                       # the phone tips 15 deg in its mount at t = 100 s
    acc[1000:] = acc[1000:] @ R.T
    gyr[1000:] = gyr[1000:] @ R.T
    gyr[999:1001, 0] += 6.0                                # the rotation itself: a fast burst
    det = _run(cfg, acc, gyr)
    slips = [e for e in det.events if e.kind == "slip"]
    assert len(slips) == 1 and abs(slips[0].magnitude - 15.0) < 2.0
    true_up = R @ np.array([0.0, 0.0, 1.0])
    new_up = slips[0].rotation @ np.array([0.0, 0.0, 1.0])     # applied to the old (level) gravity
    assert np.degrees(np.arccos(np.clip(new_up @ true_up, -1, 1))) < 2.0


def test_gyro_burst_without_tilt_change_is_a_transient(cfg):
    d = make_drive(200)
    gyr = d.gyro.copy()
    gyr[1000, 0] += 6.0                                    # a jolt, phone settles back where it was
    det = _run(cfg, d.acc, gyr)
    assert det.counts()["transient"] == 1 and det.counts()["slip"] == 0


def test_engine_inflates_accel_noise_after_a_shock_and_relevels_after_a_slip(cfg):
    d = make_drive(300)
    df = to_frame(d)
    al = fit_alignment(df.query("t < 150"), cfg)
    eng = NavigationEngine(cfg, al, VARIANTS["B"], 10.0)
    acc, gyr = df[["ax", "ay", "az"]].to_numpy(), df[["gx", "gy", "gz"]].to_numpy()
    acc[2000, 2] += 25.0                                   # shock at t = 200 s
    from src.sensors import GnssFix
    scales = {}
    for i, t in enumerate(df.t.to_numpy()):
        fix = GnssFix(d.x[i], d.y[i], d.speed[i], d.yaw[i], 2.0)
        eng.step(SensorSample(t, acc[i], gyr[i], fix))
        scales[round(t, 1)] = eng.ekf.accel_noise_scale
    assert scales[200.5] == cfg["anomaly"]["shock_q_scale"] and scales[201.5] == 1.0
    assert [a[1] for a in eng.anomalies] == ["shock"]
    # raw-INS baseline stays naive: no detector
    assert NavigationEngine(cfg, al, VARIANTS["A"], 10.0).detector is None
