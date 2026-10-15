import sys, os, cProfile, pstats, io, time
sys.dont_write_bytecode = True
REPO = sys.argv[1]; sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, torch
torch.set_num_threads(1)
from src.config import load_config
from src.data_io import load_drive
from src.blackout import make_blackout_windows, segment_for_window, apply_blackout
from src.evaluate import build_network, calibration_alignment
from src.engine import VARIANTS, NavigationEngine
from src.constraints.map_match import MapMatcher
from src.models.motion_net import MotionModel
from src.sensors import ReplaySource
cfg = load_config("configs/sih_mvp.yaml"); model = MotionModel.load(cfg["model"]["path"])
drive = load_drive("S1", cfg); net = build_network(cfg, drive.origin, {"S1"})
w = [x for x in make_blackout_windows("S1", drive.df, cfg) if x.duration == 60][0]
est, truth = apply_blackout(segment_for_window(drive.df, w, cfg), w)
al = calibration_alignment(est.df, w.t_start, cfg)
def run():
    eng = NavigationEngine(cfg, al, VARIANTS["D"], 10.0, model, MapMatcher(net, cfg)); eng.run(ReplaySource(est)); return eng
t0 = time.perf_counter(); run(); print(f"steps {len(est)}  wall {time.perf_counter()-t0:.2f}s")
pr = cProfile.Profile(); pr.enable(); eng = run(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14); print(s.getvalue()[:3500])
