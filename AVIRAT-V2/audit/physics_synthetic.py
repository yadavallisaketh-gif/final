"""Monte-Carlo study of the physics-first observations on controlled scenarios.

    python -B AVIRAT-V2/audit/physics_synthetic.py AVIRAT-V2 [seeds]

Every seed draws a different phone: gyro z scale ~ N(1, 0.015), accelerometer bias
~ N(0, 0.05) m/s^2 per horizontal axis, gyro bias ~ N(0, 0.002) rad/s per axis, and its own
noise. All estimators are ML-free (the IO-VNBD MotionNet means nothing on a synthetic IMU).
Writes phys_synthetic.csv and phys_synthetic_summary.md next to this script.
"""
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
REPO = sys.argv[1]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
sys.path.insert(0, REPO)
OUT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import copy_config, load_config  # noqa: E402
from src.physics_eval import INS_NHC, INS_NHC_MAP, run_scenario, scenario_map  # noqa: E402
from src.scenarios import SensorModel, make_scenario  # noqa: E402

BASE = load_config()
STOP = {"filter.zupt_verifier": "physics", "filter.still_detector": "smoothed", "filter.zaru_period_s": 1.0}
CONFIGS = {
    "orig": {},
    "turn_speed": {"physics.turn_speed.enabled": True},
    "gyro_turn_cal": {"physics.gyro_turn_cal.enabled": True},
    "stop_physics": STOP,
    "all_car": {"physics.turn_speed.enabled": True, "physics.gyro_turn_cal.enabled": True, **STOP},
}
MAP_CONFIGS = {
    "map": {},
    "map+turn_angle": {"physics.turn_angle.enabled": True},
    "map+anchor": {"physics.map_anchor.enabled": True},
    "map+both": {"physics.turn_angle.enabled": True, "physics.map_anchor.enabled": True},
}
CAR = ["straight", "steady_turn", "changing_radius", "braking", "idling", "crawl", "urban", "rough_road", "remount"]


def phone(seed: int, **kw) -> SensorModel:
    r = np.random.default_rng(seed + 500)
    return SensorModel(gyro_scale_z=float(1 + 0.015 * r.standard_normal()),
                       acc_bias=tuple(np.r_[0.05 * r.standard_normal(2), 0.0]),
                       gyro_bias=tuple(0.002 * r.standard_normal(3)), seed=seed, **kw)


def job(args):
    kind, name, seed = args
    rows = []
    if kind == "car":
        sc = make_scenario(name, seed, sensors=phone(seed, **_extras(name, seed)))
        for tag, o in CONFIGS.items():
            rows.append(dict(run_scenario(sc, copy_config(BASE, **o)), config=tag))
    elif kind == "map":
        sc = make_scenario("urban", seed, sensors=phone(seed))
        net = scenario_map(sc, seed=seed)
        for tag, o in MAP_CONFIGS.items():
            rows.append(dict(run_scenario(sc, copy_config(BASE, **o), INS_NHC_MAP, network=net), config=tag,
                             scenario="urban_mapped"))
    elif kind == "bump":
        for rate in (10.0, 200.0):
            sm = phone(seed, rough_acc=0.5, bumps=list(300 + np.sort(np.random.default_rng(seed).uniform(2, 58, 8))))
            sc = make_scenario("rough_road", seed, rate_hz=rate, sensors=sm)
            for tag, o in (("orig", {}), ("bump_pair", {"physics.bump_pair.enabled": True})):
                rows.append(dict(run_scenario(sc, copy_config(BASE, **o)), config=f"{tag}@{rate:.0f}Hz",
                                 scenario="bumps"))
    elif kind == "bike":
        bike = load_config("configs/two_wheeler_experimental.yaml")
        sc = make_scenario("two_wheeler", seed, sensors=phone(seed))
        for tag, c in (("car_profile", BASE), ("bike_lean_off", copy_config(bike, **{"physics.lean_speed.enabled": False})),
                       ("bike_lean_on", bike)):
            rows.append(dict(run_scenario(sc, c), config=tag))
    for r in rows:
        r.pop("reject", None)
    return rows


def _extras(name, seed):
    r = np.random.default_rng(seed)
    return {"rough_road": dict(rough_acc=1.0, rough_tilt_deg=0.6, bumps=list(300 + np.sort(r.uniform(2, 58, 6)))),
            "idling": dict(idle_acc=0.3, idle_gyro=0.012), "remount": dict(remount_t=320.0)}.get(name, {})


def summarise(df: pd.DataFrame) -> str:
    df = df.assign(fail=(df.drift_pct > 30) | (df.speed_mae > 3) | ~np.isfinite(df.drift_pct))
    g = df.groupby(["scenario", "config"])
    t = g.agg(runs=("seed", "size"), speed_mae=("speed_mae", "mean"), speed_rmse=("speed_rmse", "mean"),
              heading_deg=("heading_final_deg", "median"), drift_med=("drift_pct", "median"),
              drift_mean=("drift_pct", "mean"), endpoint_m=("endpoint_m", "median"),
              speed_nees=("speed_nees", "median"), pos_nees=("pos_nees_end", "median"), fail_rate=("fail", "mean"),
              n_turn=("n_turn_speed", "mean"), n_zupt=("n_zupt", "mean"), n_bump=("n_bump_pair", "mean"),
              n_lean=("n_lean_speed", "mean"), n_turn_angle=("n_turn_angle", "mean"),
              n_anchor=("n_map_along", "mean")).reset_index()
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    jobs = [("car", n, s) for n in CAR for s in range(SEEDS)] + [("map", "urban", s) for s in range(SEEDS)] \
        + [("bump", "rough_road", s) for s in range(SEEDS)] + [("bike", "two_wheeler", s) for s in range(SEEDS)]
    with Pool(4) as p:
        rows = [r for rr in p.map(job, jobs, chunksize=1) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phys_synthetic.csv"), index=False)
    md = summarise(df)
    with open(os.path.join(OUT, "phys_synthetic_summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)
