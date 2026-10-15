"""Physics-first observations (constraints/physics_obs.py, docs/PHYSICS_OBSERVABILITY.md).

Measurement equations and Jacobians, the observability claims on controlled scenarios,
every context gate, the calibrations, and the rules that stop the same sensor data from
being used twice. Scenario tests run the real evaluation path (src/physics_eval.py).
"""
import numpy as np
import pytest

from src.config import copy_config, load_config
from src.constraints.physics_obs import (BumpPairDetector, PhysicsObserver, bump_pair_min_rate,
                                         calibrate_centripetal, calibrate_gyro_turns, lean_observation,
                                         physics_enabled)
from src.engine import VARIANTS, NavigationEngine
from src.fusion.ekf2d import BG, BL, EKF2D, N, RX, VF, X, Y, YAW
from src.physics_eval import INS_NHC, INS_NHC_MAP, run_scenario, scenario_map
from src.preprocess import Alignment, fit_alignment
from src.scenarios import SensorModel, make_scenario


@pytest.fixture
def pcfg():
    """The scenario configuration: IO-VNBD defaults (no gyro-propagated attitude)."""
    return load_config()


def _on(cfg, **kw):
    return copy_config(cfg, **{k.replace("__", "."): v for k, v in kw.items()})


# --------------------------------------------------------------------------- defaults and rules
def test_every_physics_feature_is_off_by_default(pcfg):
    assert not physics_enabled(pcfg)
    al = Alignment(np.eye(3), 9.81, 0, 0, 1, 1, 1, 0)
    assert NavigationEngine(pcfg, al, VARIANTS["B"], 10.0).phys is None
    f = pcfg["filter"]
    assert f["zupt_verifier"] == "motionnet" and f["still_detector"] == "raw" and f["zaru_period_s"] == 0.0


def test_turn_speed_and_lean_speed_are_mutually_exclusive(pcfg):
    cfg = _on(pcfg, physics__turn_speed__enabled=True, physics__lean_speed__enabled=True)
    with pytest.raises(ValueError, match="same centripetal"):
        PhysicsObserver(cfg, 10.0, None)


def test_turn_angle_disables_absolute_map_heading(pcfg):
    obs = PhysicsObserver(_on(pcfg, physics__turn_angle__enabled=True), 10.0, None)
    assert obs.disable_map_heading
    assert not PhysicsObserver(_on(pcfg, physics__map_anchor__enabled=True), 10.0, None).disable_map_heading


def test_nhc_never_moves_forward_speed_so_turn_speed_is_the_only_centripetal_path(cfg):
    """The NHC + v_l propagation would carry the same a_l = w v information; the NHC is
    forbidden to change v_f, so with turn_speed on it reaches v_f exactly once."""
    e = EKF2D(cfg)
    e.initialise(0.0, 0.0, 0.0, 12.0, 0.0)
    e.P[VF, 3] = e.P[3, VF] = 0.5        # strong v_f - v_l correlation
    e.s[3] = 1.0
    vf = e.s[VF]
    e.update_nhc(0.15, w=0.0, a_l=0.0)
    assert e.s[VF] == vf


# --------------------------------------------------------------------------- measurement equations
def test_turn_speed_jacobian_matches_finite_differences(cfg):
    cfg["filter"]["estimate_lever_arm"] = True
    e = EKF2D(cfg)
    e.initialise(0.0, 0.0, 0.0, 11.0, 0.2)
    e.s[[BG, BL, RX]] = 0.003, 0.05, 1.7
    w, w_dot, k = 0.18, 0.02, 1.04
    h = lambda s: k * ((w - s[BG]) * s[VF] + w_dot * s[RX]) + s[BL]
    num = np.array([(h(e.s + d) - h(e.s - d)) / 2e-6 for d in np.eye(N) * 1e-6])
    expected = np.zeros(N)
    expected[VF], expected[BG], expected[BL], expected[RX] = k * (w - e.s[BG]), -k * e.s[VF], 1.0, k * w_dot
    assert np.allclose(num, expected, atol=1e-7)
    # the EKF update moves v_f towards a_l / (k w): truth 14 m/s, filter 11 m/s
    e.gnss_enabled = False
    e.P[VF, VF] = 4.0
    a_l = k * (w - e.s[BG]) * 14.0 + e.s[BL] + k * w_dot * e.s[RX]
    assert e.update_turn_speed(a_l, w, w_dot, k, 0.15, 0.01, 0.03, None)
    assert 11.0 < e.s[VF] < 14.0 and e.s[VF] > 12.5


def test_along_track_anchor_observes_only_the_along_road_component(cfg):
    e = EKF2D(cfg)
    e.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    e.gnss_enabled = False
    x0, y0 = e.s[X], e.s[Y]
    e.update_along_track(0.0, 7.0, 1.0, 0.0, 2.0, None)     # a purely cross-road offset (road along +x)
    assert np.isclose(e.s[X], x0) and np.isclose(e.s[Y], y0)
    pxx = e.P[X, X]
    e.update_along_track(6.0, 0.0, 1.0, 0.0, 2.0, None)     # along-road offset: x moves, y does not
    assert np.isclose(e.s[X], x0 + 6.0 * pxx / (pxx + 4.0)) and np.isclose(e.s[Y], y0)


def test_yaw_clone_is_exact_for_ordinary_updates(cfg):
    """Augmenting with a clone must not change any ordinary update (exact marginal)."""
    a, b = EKF2D(cfg), EKF2D(cfg)
    for e in (a, b):
        e.initialise(0.0, 0.0, 0.0, 12.0, 0.3)
    b.start_yaw_clone()
    for k in range(1, 50):
        for e in (a, b):
            e.predict(0.1 * k, 0.1, 0.4, 0.05)
            e.update_nhc(0.15)
            e.update_gnss_position(1.2 * k, 0.1 * k, 3.0)
    assert np.allclose(a.s, b.s) and np.allclose(a.P, b.P)


def test_heading_change_attributes_the_residual_to_gyro_bias(cfg):
    """Clone, turn 90 deg in 10 s with an unknown 0.01 rad/s gyro bias, then observe the
    true turn angle: b_g moves towards the truth although the absolute heading is unknown."""
    e = EKF2D(cfg)
    e.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    e.P[BG, BG] = 0.01 ** 2
    e.gnss_enabled, e.denied = False, True
    bias = 0.01
    e.start_yaw_clone()
    for k in range(1, 101):
        e.predict(0.1 * k, 0.0, 0.0, np.pi / 2 / 10 + bias)      # gyro reads truth + bias
    assert e.update_heading_change(np.pi / 2, np.deg2rad(2.0), frozen=(VF, 3, 5, 7, 8))
    assert 0.4 * bias < e.s[BG] < 1.2 * bias
    assert abs(e.s[YAW] - np.pi / 2) < abs(np.pi / 2 + bias * 10 - np.pi / 2)


# --------------------------------------------------------------------------- observability on controlled scenarios
def test_straight_constant_speed_has_no_speed_observation(pcfg):
    """Ideal straight constant-speed driving: a_f = 0 and w = 0 carry no information on v.
    The gate must refuse every turn-speed update - nothing may be invented."""
    cfg = _on(pcfg, physics__turn_speed__enabled=True)
    r = run_scenario(make_scenario("straight", 0), cfg)
    assert r["calib_valid"] and r["n_turn_speed"] == 0
    assert set(k for k in r["reject"] if k.startswith("turn_speed")) == {"turn_speed:low_turn_rate"}


def test_steady_turn_makes_speed_observable(pcfg):
    base = run_scenario(make_scenario("steady_turn", 1), pcfg)
    phys = run_scenario(make_scenario("steady_turn", 1), _on(pcfg, physics__turn_speed__enabled=True))
    assert phys["n_turn_speed"] > 50
    assert phys["speed_mae"] < 0.8 * base["speed_mae"]
    assert 1.0 < phys["calib_k"] < 1.1                            # body roll leaks gravity: k > 1


def test_changing_radius_turn_entry_transients_are_rejected(pcfg):
    r = run_scenario(make_scenario("changing_radius", 0), _on(pcfg, physics__turn_speed__enabled=True))
    assert r["reject"].get("turn_speed:transient", 0) > 0 and r["n_turn_speed"] > 0


def test_context_gate_reasons(pcfg):
    cfg = _on(pcfg, physics__turn_speed__enabled=True)
    from src.constraints.physics_obs import CentripetalCalibration, PhysicsCalibration
    obs = PhysicsObserver(cfg, 10.0, PhysicsCalibration(CentripetalCalibration(True, 1.0, 0.2, 0.02, 0.3, 0.95, 500)))
    e = EKF2D(cfg)
    e.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    g = lambda **kw: obs._gate_turn(e, **{**dict(w=0.2, w_dot=0.0, a_l=2.0, a_f_max=0.2, shock=False), **kw})
    assert g() == ""
    assert g(w=0.05) == "low_turn_rate"
    assert g(w_dot=0.2) == "transient"
    assert g(a_l=5.0) == "slip"
    assert g(a_f_max=3.0) == "braking"
    assert g(shock=True) == "shock"
    assert g(a_l=-2.0) == "sign"
    e.s[VF] = 1.0
    assert g() == "low_speed"
    obs.calib.centripetal.valid = False
    assert g() == "not_calibrated"


def test_centripetal_calibration_refuses_inconsistent_data(pcfg):
    """A phone whose lateral accelerometer does not follow w * v (IO-VNBD drive Vta16) must
    not produce turn-speed observations: the pre-blackout consistency check rejects it."""
    cfg = _on(pcfg, physics__turn_speed__enabled=True)
    sc = make_scenario("urban", 0)
    calib = sc.df[sc.df.t < 300].copy()
    al = fit_alignment(calib, cfg)
    good = calibrate_centripetal(calib, al, cfg)
    assert good.valid and good.corr > 0.9
    bad = calib.copy()
    rng = np.random.default_rng(0)
    bad["gnss_speed"] = rng.permutation(bad["gnss_speed"].to_numpy())        # speed no longer explains a_l
    assert not calibrate_centripetal(bad, al, cfg).valid


def test_gyro_turn_calibration_recovers_scale(pcfg):
    sc = make_scenario("urban", 0, sensors=SensorModel(gyro_scale_z=1.03))
    calib = sc.df[sc.df.t < 300]
    cal = calibrate_gyro_turns(calib, fit_alignment(calib, pcfg), pcfg)
    assert cal.valid and cal.n_turns >= 3 and abs(cal.scale - 1.03) < 0.006


def test_gyro_turn_calibration_reduces_heading_drift_in_a_long_turn(pcfg):
    sc = make_scenario("steady_turn", 0, sensors=SensorModel(gyro_scale_z=1.03))
    base = run_scenario(sc, pcfg)
    cal = run_scenario(sc, _on(pcfg, physics__gyro_turn_cal__enabled=True))
    assert cal["heading_final_deg"] < 0.3 * base["heading_final_deg"]


# --------------------------------------------------------------------------- stops
def _stop_cfg(pcfg):
    return _on(pcfg, filter__zupt_verifier="physics", filter__still_detector="smoothed", filter__zaru_period_s=1.0)


def test_idle_stop_detected_despite_engine_vibration(pcfg):
    sc = make_scenario("idling", 1)
    base = run_scenario(sc, pcfg)
    stop = run_scenario(sc, _stop_cfg(pcfg))
    assert base["n_zupt"] == 0 and stop["n_zupt"] > 50
    assert stop["drift_pct"] < base["drift_pct"]


def test_physics_verifier_rejects_a_quiet_constant_speed_cruise(pcfg):
    """A perfectly smooth cruise looks still to the vibration detector (V1 warning); the
    physics verifier must refuse because v = 0 is inconsistent with the filter's speed."""
    sm = SensorModel(drive_vib=0.0, acc_noise=0.01)
    sc = make_scenario("straight", 0, sensors=sm)
    r = run_scenario(sc, _on(pcfg, filter__zupt_verifier="physics"))
    assert r["n_zupt"] == 0


def test_physics_verifier_cannot_separate_a_crawl_from_a_stop(pcfg):
    """Documented physical limit (PHYSICS_OBSERVABILITY.md): at 10 Hz a 1-2.5 m/s crawl with
    road vibration and an idle stop are both 'gravity + broadband noise'. Only the filter's own
    confidence separates them: with a loose speed-sigma bound false ZUPTs occur in a crawl; the
    default bound (0.8 m/s) suppresses them - and with them the late stops of a long blackout.
    This is why zupt_verifier stays 'motionnet' by default."""
    sc = make_scenario("crawl", 0)
    loose = run_scenario(sc, _on(_stop_cfg(pcfg), filter__zupt_physics_sigma_max=1.5))
    strict = run_scenario(sc, _stop_cfg(pcfg))
    assert loose["n_zupt"] > 50                                  # false stops: the car never stops
    assert strict["n_zupt"] == 0


# --------------------------------------------------------------------------- bump pair and lean
def test_bump_pair_needs_a_fast_imu():
    assert 40 < bump_pair_min_rate(30.0, 2.7, 0.1) < 55
    cfg = load_config()
    cfg["physics"]["bump_pair"]["enabled"] = True
    assert PhysicsObserver(cfg, 10.0, None).bump is None                        # refuses at 10 Hz
    assert PhysicsObserver(cfg, 200.0, None).bump is not None


@pytest.mark.parametrize("rate,expect", [(10.0, 0), (200.0, 9)])
def test_bump_pair_speed_at_10_and_200_hz(rate, expect):
    cfg = load_config()
    sm = SensorModel(rough_acc=0.3, bumps=list(310 + np.arange(0, 45, 5.0)))
    sc = make_scenario("rough_road", 1, rate_hz=rate, sensors=sm)
    det = BumpPairDetector(cfg, rate)
    obs = [(o, v) for t, az, v in zip(sc.df.t, sc.df.az, sc.df.ref_speed) if (o := det.update(t, az))]
    assert len(obs) == expect
    for (v_hat, sig), v in obs:
        assert abs(v_hat - v) < max(3 * sig, 0.05 * v)


def test_lean_observation_geometry():
    g, v, w = 9.81, 12.0, 0.15
    phi = np.arctan2(v * w, g)
    f = np.array([0.0, 0.0, g / np.cos(phi)])                    # coordinated: along the leaning body z
    gyro = np.array([0.0, -np.sin(phi) * w, np.cos(phi) * w])
    a_c, om, lean = lean_observation(f, gyro, np.zeros(3), g)
    assert np.isclose(a_c / om, v) and np.isclose(lean, phi)


# --------------------------------------------------------------------------- maps
def test_map_turn_observations_fire_on_a_mapped_urban_route(pcfg):
    sc = make_scenario("urban", 2)
    net = scenario_map(sc, seed=2)
    cfg = _on(pcfg, physics__turn_angle__enabled=True, physics__map_anchor__enabled=True)
    r = run_scenario(sc, cfg, INS_NHC_MAP, network=net)
    assert r["n_turn_angle"] >= 1 and r["n_map_along"] >= 1
    assert r["drift_pct"] < 5.0


def test_simulator_obeys_rigid_body_centripetal_kinematics():
    """Scenario sanity: in the steady part of the turn, |horizontal specific force| ~ v w."""
    sc = make_scenario("steady_turn", 0, sensors=SensorModel(acc_noise=0.0, gyro_noise=0.0, drive_vib=0.0,
                                                           acc_bias=(0, 0, 0), gyro_bias=(0, 0, 0)))
    d = sc.df[(sc.df.t > 320) & (sc.df.t < 350)]
    acc = d[["ax", "ay", "az"]].to_numpy()
    w = d["ref_yaw_rate"].to_numpy()
    v = d["ref_wheel_speed"].to_numpy()
    horiz = np.sqrt(np.maximum((acc ** 2).sum(1) - 9.81 ** 2, 0))
    assert np.allclose(horiz, np.abs(v * w), rtol=0.08)
