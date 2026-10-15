"""Filter, preprocessing, NHC and sensor-agnostic engine behaviour on synthetic drives."""
import numpy as np

from src.blackout import BlackoutWindow, apply_blackout
from src.engine import VARIANTS, NavigationEngine
from src.fusion.ekf2d import EKF2D, VL
from src.metrics import blackout_metrics
from src.preprocess import ImuPreprocessor, fit_alignment, preprocess_frame
from src.sensors import ReplaySource, SyntheticSource
from src.synthetic import make_drive, to_frame


def test_alignment_recovers_mount_angle(cfg):
    for mount in (-70.0, 0.0, 35.0, 120.0):
        df = to_frame(make_drive(400, mount_yaw_deg=mount))
        al = fit_alignment(df, cfg)
        err = (al.mount_yaw_deg - mount + 180) % 360 - 180
        assert abs(err) < 3.0, (mount, al.mount_yaw_deg)
        assert al.lateral_corr > 0.8 and al.yaw_rate_corr > 0.8


def test_streaming_equals_block_preprocessing(cfg):
    df = to_frame(make_drive(60))
    al = fit_alignment(df, cfg)
    block = preprocess_frame(df, al, cfg).to_numpy()
    pre = ImuPreprocessor(al, cfg, 10.0)
    stream = []
    for a, g, v in zip(df[["ax", "ay", "az"]].to_numpy(), df[["gx", "gy", "gz"]].to_numpy(), df["gnss_speed"].to_numpy()):
        pre.set_speed(v)                       # the engine feeds its speed estimate the same way
        stream.append(pre.process(a, g))
    stream = np.array(stream)
    assert np.allclose(block, stream, atol=1e-9)


def test_ekf_tracks_straight_line(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    for k in range(1, 601):
        t = k * 0.1
        ekf.predict(t, 0.0, 0.0, 0.0)
        if k % 10 == 0:
            ekf.update_gnss_position(10.0 * t, 0.0, 2.0)
    assert abs(ekf.s[0] - 600.0) < 2 and abs(ekf.s[1]) < 2 and abs(ekf.s[2] - 10.0) < 0.2


def test_ekf_estimates_gyro_bias(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    bias = 0.01
    for k in range(1, 3001):
        t = k * 0.1
        ekf.predict(t, 0.0, 0.0, bias)            # true yaw rate 0, gyro reads bias
        if k % 10 == 0:
            ekf.update_gnss_position(10.0 * t, 0.0, 2.0)
            ekf.update_gnss_heading(0.0)
    assert abs(ekf.s[6] - bias) < 0.003


def test_nhc_suppresses_lateral_drift_without_inventing_a_turn(cfg):
    """A lateral accelerometer error must be absorbed by b_l, not read as a gyro bias."""
    res = {}
    for use in (False, True):
        ekf = EKF2D(cfg)
        ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
        for k in range(1, 601):
            ekf.predict(k * 0.1, 0.0, 0.3, 0.0)   # a biased lateral accelerometer, straight road
            if use:
                ekf.update_nhc(cfg["filter"]["nhc_sigma"])
        res[use] = (abs(ekf.s[VL]), abs(ekf.s[1]), abs(ekf.s[4]))
    assert res[True][0] < 0.2 * res[False][0]
    assert res[True][1] < 0.2 * res[False][1]
    assert res[True][2] < np.deg2rad(10)          # no phantom turn


def test_nhc_observes_gyro_bias_during_blackout(cfg):
    """Straight drive with a biased gyro and no GNSS: NHC must pull b_g towards the truth."""
    bias = 0.01
    out = {}
    for freeze in (True, False):
        cfg["filter"]["freeze_bias_in_dr"] = freeze
        ekf = EKF2D(cfg)
        ekf.initialise(0.0, 0.0, 0.0, 5.0, 0.0)
        ekf.gnss_enabled = False                  # the whole run is a blackout
        v = 5.0
        for k in range(1, 1201):                  # 120 s, speed varies 5..20 m/s
            t = k * 0.1
            a = 0.5 * np.cos(2 * np.pi * t / 60.0) * np.pi / 3
            v += a * 0.1
            ekf.predict(t, a, 0.0, bias)          # true yaw rate 0, gyro reads the bias
            ekf.update_speed(v, 0.3)
            ekf.update_nhc(cfg["filter"]["nhc_sigma"])
        out[freeze] = (ekf.s[6], abs(ekf.s[4]))
    assert out[True][0] == 0.0                     # frozen: never learns
    assert out[False][0] > 0.3 * bias              # observable: learns a large part of it
    assert out[False][1] < 0.7 * out[True][1]      # and the heading error is clearly smaller


def test_zaru_estimates_gyro_bias_at_a_stop(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 0.0, 0.0)
    for k in range(1, 101):                        # 10 s standing still
        ekf.predict(k * 0.1, 0.0, 0.0, 0.004)
        ekf.update_zaru(0.004, cfg["filter"]["zaru_sigma"])
    assert abs(ekf.s[6] - 0.004) < 0.001


def test_process_noise_is_rate_invariant(cfg):
    P = {}
    for rate in (10.0, 200.0):
        ekf = EKF2D(cfg)
        ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
        for k in range(1, int(rate * 5) + 1):
            ekf.predict(k / rate, 0.0, 0.0, 0.0)
        P[rate] = np.diag(ekf.P)
    rel = np.abs(P[10.0] - P[200.0]) / np.maximum(P[10.0], 1e-12)
    assert rel[[2, 3, 4, 6]].max() < 0.1, rel      # velocity, heading, gyro bias


def test_nhc_information_is_rate_invariant(cfg):
    """Engine scales NHC R with the rate: the lateral-velocity variance after 2 s matches."""
    from src.preprocess import Alignment
    var = {}
    for rate in (10.0, 200.0):
        al = Alignment(np.eye(3), 9.81, 0, 0, 1, 1, 1, 0)
        eng = NavigationEngine(cfg, al, VARIANTS["B"].__class__(**{**VARIANTS["B"].__dict__, "use_nhc": True}), rate)
        eng.ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
        eng.initialised = True
        eng._last_fix_t = -1e9
        from src.sensors import SensorSample
        for k in range(1, int(rate * 2) + 1):
            eng.step(SensorSample(k / rate, np.array([0, 0, 9.81]), np.zeros(3)))
        var[rate] = eng.ekf.P[3, 3]
    assert abs(var[10.0] - var[200.0]) / var[10.0] < 0.25, var


def test_filter_dimensions_through_a_full_cycle(cfg):
    """Every predict/update type with the b_v and lever-arm states enabled: no shape errors."""
    cfg["filter"].update(motion_bias_state=True, estimate_lever_arm=True)
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 1.0, 2.0, 10.0, 0.3)
    assert ekf.P[9, 9] > 0 and ekf.P[8, 8] > 0            # both augmented states are live
    for k in range(1, 51):
        ekf.predict(k * 0.1, 0.2, 0.1, 0.05)
        assert ekf.s.shape == (10,) and ekf.P.shape == (10, 10)
    ekf.update_gnss_position(5.0, 3.0, 3.0)
    ekf.update_gnss_speed(10.5)
    ekf.update_gnss_heading(0.35)
    ekf.update_speed(10.0, 1.0)
    ekf.gnss_enabled = False                               # GNSS-denied branch of every update
    ekf.update_speed(10.2, 1.0)
    ekf.update_nhc(0.15)
    ekf.update_nhc(0.15, w=0.1, a_l=0.5)
    ekf.update_zupt(0.2)
    ekf.update_zaru(0.001, 0.003)
    ekf.update_road(5.0, 3.0, 0.3, 8.0, np.deg2rad(6))
    ekf.predict(5.1, 0.0, 0.0, 0.0)
    assert ekf.s.shape == (10,) and ekf.P.shape == (10, 10)
    assert np.isfinite(ekf.P).all() and np.allclose(ekf.P, ekf.P.T, atol=1e-9)
    assert np.linalg.eigvalsh(ekf.P).min() > -1e-9


def test_motionnet_bias_is_gauss_markov(cfg):
    cfg["filter"].update(motion_bias_state=True, motion_bias_tau_s=4.7, motion_bias_sigma=3.0)
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    ekf.s[9] = 2.0
    for k in range(1, 48):                                 # 4.7 s = one time constant
        ekf.predict(k * 0.1, 0.0, 0.0, 0.0)
    assert abs(ekf.s[9] - 2.0 * np.exp(-1.0)) < 1e-6
    for k in range(48, 1000):
        ekf.predict(k * 0.1, 0.0, 0.0, 0.0)
    assert abs(np.sqrt(ekf.P[9, 9]) - 3.0) < 0.05           # stationary std = sigma_b


def test_motionnet_bias_calibrated_before_blackout_and_removed_during_it(cfg):
    out = {}
    for state in (False, True):
        cfg["filter"].update(motion_bias_state=state, motion_bias_tau_s=600.0, motion_bias_sigma=3.0)
        ekf = EKF2D(cfg)
        ekf.initialise(0.0, 0.0, 0.0, 12.0, 0.0)
        k = 0
        for k in range(1, 601):                            # 60 s with GNSS speed: calibrate
            ekf.predict(k * 0.1, 0.0, 0.0, 0.0)
            ekf.update_gnss_speed(12.0)
            ekf.update_speed(14.0, 0.5)                    # MotionNet reads +2 m/s too fast
        bv_cal = ekf.s[9]
        ekf.gnss_enabled = False
        for k in range(601, 1201):                         # 60 s blackout, MotionNet only
            ekf.predict(k * 0.1, 0.0, 0.0, 0.0)
            ekf.update_speed(14.0, 0.5)
        out[state] = (bv_cal, ekf.s[2])
    assert abs(out[True][0] - 2.0) < 0.2                   # b_v converged on the offset
    assert abs(out[True][1] - 12.0) < 0.5                  # speed stays right through the blackout
    assert abs(out[False][1] - 14.0) < 0.5                 # without the state: the bias passes through


def test_accel_bias_clamped_in_gnss_denial(cfg):
    cfg["filter"].update(freeze_accel_bias_in_dr=True, freeze_bias_in_dr=False)
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    ekf.s[5] = 0.05
    for k in range(1, 21):
        ekf.predict(k * 0.1, 0.3, 0.2, 0.02)
    ekf.gnss_enabled = False
    for k in range(21, 121):
        ekf.predict(k * 0.1, 0.3, 0.2, 0.02)
        ekf.update_speed(9.0, 1.0)
        ekf.update_nhc(0.15)
        ekf.update_zupt(0.2)
        ekf.update_road(k * 1.0, 0.0, 0.0, 8.0, np.deg2rad(6))
    assert ekf.s[5] == 0.05                                # untouched by every DR update


def test_nhc_never_changes_position_or_forward_speed(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 12.0, 0.3)
    for k in range(1, 51):                         # build up cross-correlations first
        ekf.predict(k * 0.1, 0.2, 0.5, 0.05)
    vf, xy = ekf.s[2], ekf.s[:2].copy()
    ekf.update_nhc(cfg["filter"]["nhc_sigma"])
    assert ekf.s[2] == vf and (ekf.s[:2] == xy).all()
    assert abs(ekf.s[VL]) < 1.0


def test_dynamic_attitude_removes_grade_leakage(cfg):
    """Hills put g*sin(grade) on the forward axis; static levelling keeps it, dynamic tracking removes most."""
    d = make_drive(600, grade_amp=0.05, mount_yaw_deg=30.0, acc_noise=0.02, gyro_noise=0.0005)
    df = to_frame(d)
    true_af = np.gradient(d.speed, d.t)
    err = {}
    for mode, gyro in (("static", False), ("dynamic", True)):   # synthetic data is body-frame: gyro valid
        cfg["preprocess"].update(attitude=mode, attitude_use_gyro=gyro, attitude_tau_s=30.0, attitude_acc_gate=0.15)
        al = fit_alignment(df.query("t < 300"), cfg)
        f = preprocess_frame(df, al, cfg)
        e = (f["a_f"].to_numpy() - true_af)[3000:]          # evaluate on the second half
        err[mode] = float(np.sqrt(np.mean(e ** 2)))
    assert err["static"] > 0.25                           # ~g * 5 % / sqrt(2) leaks without tracking
    # The floor is physical: a slow hill and a slow acceleration look alike to
    # the accelerometer, so the correction cannot remove all of it.
    assert err["dynamic"] < 0.8 * err["static"], err


def test_attitude_filter_tracks_pitch_with_gyro(cfg):
    """Pure rotation, no linear acceleration: gyro propagation follows the pitch exactly."""
    from src.preprocess import AttitudeFilter
    cfg["preprocess"].update(attitude_use_gyro=True, attitude_tau_s=1e9)
    att = AttitudeFilter(np.array([0.0, 0.0, 1.0]), 9.81, cfg)
    dt, theta = 0.01, 0.0
    for k in range(1000):                                 # pitch up at 0.05 rad/s for 10 s
        theta += 0.05 * dt
        g = att.step(9.81 * np.array([np.sin(theta), 0, np.cos(theta)]), np.array([0.0, -0.05, 0.0]), dt)
    assert abs(np.arctan2(g[0], g[2]) - theta) < 1e-3


def _steady_turn(cfg, r_x_true, r_x_model, om=0.3, v=10.0, seconds=20.0, freeze=False):
    """Phone r_x_true ahead of the rear axle; the axle moves at v with yaw rate om.
    Phone-point kinematics: v_l = om * r_x, a_l = om * v_f, a_f = -om * v_l."""
    cfg["filter"]["freeze_bias_in_dr"] = freeze
    cfg["filter"]["lever_arm_x"] = r_x_model
    ekf = EKF2D(cfg)
    vl = om * r_x_true
    ekf.initialise(0.0, 0.0, 0.0, v, 0.0)
    ekf.s[VL] = vl
    nis = []
    for k in range(1, int(seconds * 10) + 1):
        ekf.predict(k * 0.1, -om * vl, om * v, om)
        ekf.update_nhc(cfg["filter"]["nhc_sigma"])
        nis.append(ekf.log[-1].nis)
    return ekf, np.array(nis)


def test_lever_arm_removes_phantom_lateral_innovation_in_a_turn(cfg):
    ekf, nis = _steady_turn(cfg, 1.8, 1.8)
    assert np.nanmax(nis) < 1e-3                       # innovation ~ 0: nothing to "correct"
    assert abs(ekf.s[VL] - 0.3 * 1.8) < 0.01           # the phone's true sideways motion survives
    assert abs(ekf.s[6]) < 1e-4                        # no phantom gyro bias


def test_without_lever_arm_the_same_turn_injects_false_corrections(cfg):
    ekf_ok, nis_ok = _steady_turn(cfg, 1.8, 1.8)
    ekf_bad, nis_bad = _steady_turn(cfg, 1.8, 0.0)
    # (the turn-rate inflation of R already softens it: NIS stays < 1, but it is
    # persistent and one-sided, so it still biases the states it may touch)
    assert np.nanmean(nis_bad) > 10 * max(np.nanmean(nis_ok), 1e-9)
    assert abs(ekf_bad.s[6]) > 10 * max(abs(ekf_ok.s[6]), 1e-6)   # leaks into the gyro bias
    assert abs(ekf_bad.s[VL] - 0.54) > 0.1                          # and erodes the true v_l


def test_nhc_is_skipped_in_aggressive_cornering(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 15.0, 0.0)
    ekf.predict(0.1, 0.0, 7.5, 0.5)                    # 0.5 rad/s > nhc_max_yaw_rate
    before = ekf.s.copy()
    assert ekf.update_nhc(cfg["filter"]["nhc_sigma"]) is False
    assert (ekf.s == before).all()
    assert ekf.log[-1].source == "nhc" and not ekf.log[-1].accepted


def test_nhc_noise_inflates_with_turn_rate_and_lateral_accel(cfg):
    gains = []
    for w, al in ((0.0, 0.0), (0.2, 0.0), (0.0, 2.0)):
        ekf = EKF2D(cfg)
        ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
        ekf.P[VL, VL] = 1.0
        ekf.s[VL] = 0.5 + w * cfg["filter"]["lever_arm_x"]   # same axle-level violation (0.5 m/s)
        ekf.update_nhc(0.3, w=w, a_l=al)
        gains.append(0.5 + w * cfg["filter"]["lever_arm_x"] - ekf.s[VL])
    # R x5 in both cases: the correction shrinks from K = 1/1.09 to 1/1.45 of the innovation
    assert gains[1] < 0.8 * gains[0] and gains[2] < 0.8 * gains[0]


def test_motionnet_update_rate_and_r_inflation(cfg):
    from src.preprocess import Alignment
    from src.models.motion_net import MotionModel, MotionGRU
    al = Alignment(np.eye(3), 9.81, 0, 0, 1, 1, 1, 0)
    m = MotionModel(MotionGRU(5, 8), "gru", ["a_f", "a_l", "a_u", "w_u", "w_h"], 20, np.zeros(5), np.ones(5), hidden=8)
    cfg["filter"].update(motion_update_hz=1.0, motion_err_tau_s=6.7)   # decorrelated mode
    eng = NavigationEngine(cfg, al, VARIANTS["C"], 10.0, m)
    assert eng.motion_every_n == 10                     # 1 Hz at a 10 Hz IMU
    rho = np.exp(-1.0 / cfg["filter"]["motion_err_tau_s"])
    assert np.isclose(eng.motion_r_inflation, (1 + rho) / (1 - rho))
    cfg["filter"].update(motion_update_hz=10.0, motion_err_tau_s=0)
    eng10 = NavigationEngine(cfg, al, VARIANTS["C"], 10.0, m)
    assert eng10.motion_every_n == 1 and eng10.motion_r_inflation == 1.0   # old behaviour reproducible


def test_engine_blackout_filtered_beats_raw_with_bias(cfg):
    d = make_drive(500, gyro_bias=0.004, acc_bias=0.08)
    df = to_frame(d)
    w = BlackoutWindow("syn", 0, 300.0, 360.0)
    est, truth = apply_blackout(df, w)
    al = fit_alignment(est.df[est.df.t < w.t_start], cfg)
    drift = {}
    for key in ("A", "B"):
        eng = NavigationEngine(cfg, al, VARIANTS[key], 10.0)
        traj = eng.run(ReplaySource(est))
        drift[key] = blackout_metrics(traj, truth, w)["drift_percent"]
    assert drift["B"] < drift["A"], drift


def test_reacquisition_is_gradual(cfg):
    d = make_drive(400, gyro_bias=0.01, acc_bias=0.2)
    df = to_frame(d)
    w = BlackoutWindow("syn", 0, 200.0, 290.0)
    est, truth = apply_blackout(df, w)
    al = fit_alignment(est.df[est.df.t < w.t_start], cfg)
    al.gyro_bias[:] = 0
    al.acc_bias[:] = 0
    eng = NavigationEngine(cfg, al, VARIANTS["A"], 10.0)
    traj = eng.run(ReplaySource(est))
    m = blackout_metrics(traj, truth, w)
    assert m["endpoint_error_m"] > 20                       # there was a real error to correct
    assert m["reacq_max_step_m"] < 0.5 * m["endpoint_error_m"]  # ... corrected over several fixes
    assert m["reacq_time_to_5m_s"] < 20


def test_same_engine_runs_on_a_200hz_external_imu(cfg):
    """SensorSource abstraction: a 200 Hz IMU with 1 Hz GNSS, no engine changes."""
    d = make_drive(400)
    w = BlackoutWindow("syn", 0, 250.0, 280.0)
    src = SyntheticSource(d, rate_hz=200.0, gnss_hz=1.0, blackout=w)
    # calibrate on the pre-blackout part of the 10 Hz version of the same drive
    al = fit_alignment(to_frame(d).query("t < 250"), cfg)
    eng = NavigationEngine(cfg, al, VARIANTS["C+NHC"].__class__(**{**VARIANTS["B"].__dict__, "use_nhc": True}), 200.0)
    traj = eng.run(src)
    assert abs(np.median(np.diff(traj.t)) - 0.005) < 1e-6
    inside = w.contains(traj.t.to_numpy())
    settled = inside & (traj.t.to_numpy() > w.t_start + cfg["filter"]["gnss_timeout_s"])
    assert (traj["mode"][settled] == "DEAD RECKONING").all()
    assert (traj["mode"][traj.t.to_numpy() < w.t_start] != "DEAD RECKONING").all()   # 1 Hz GNSS is not "lost"
    assert not any(k.startswith("gnss") for k in eng.ekf.counts(w.t_start, w.t_end, accepted_only=False))
    end = np.nonzero(inside)[0][-1]
    err = np.hypot(traj.x[end] - np.interp(traj.t[end], d.t, d.x), traj.y[end] - np.interp(traj.t[end], d.t, d.y))
    assert err < 30.0, err


def test_blackout_speed_decoupled_from_forward_accelerometer(cfg):
    """dr_accel_mode=decouple: a biased forward accelerometer cannot drag v_f in a
    blackout; MotionNet alone sets the speed. With GNSS it is used as normal."""
    out = {}
    for mode in ("normal", "decouple", "inflate"):
        cfg["filter"]["dr_accel_mode"] = mode
        ekf = EKF2D(cfg)
        ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
        ekf.denied, ekf.gnss_enabled = True, False           # blackout
        for k in range(1, 601):                              # 60 s, a_f reads a spurious -0.3 m/s^2
            ekf.predict(k * 0.1, -0.3, 0.0, 0.0)
            ekf.update_speed(10.0, 2.0)                      # MotionNet: 10 m/s, noisy
        out[mode] = ekf.s[2]
    assert abs(out["decouple"] - 10.0) < 0.3
    assert abs(out["inflate"] - 10.0) < 0.5
    assert out["normal"] < out["decouple"] - 0.5             # the accelerometer bias leaks without decoupling
    cfg["filter"]["dr_accel_mode"] = "decouple"
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)                 # GNSS healthy: accelerometer still integrated
    for k in range(1, 11):
        ekf.predict(k * 0.1, 1.0, 0.0, 0.0)
    assert abs(ekf.s[2] - 11.0) < 1e-6


def test_config_profile_inherits_base():
    from src.config import load_config
    c = load_config("configs/sih_mvp.yaml")
    b = load_config()
    assert c["data"]["gnss_source"] == "vehicle" and c["filter"]["dr_accel_mode"] in ("normal", "decouple", "inflate")
    assert c["filter"]["lever_arm_x"] == b["filter"]["lever_arm_x"]    # everything else from base
    assert c["split"] == b["split"]
