import sys, os, time
sys.dont_write_bytecode = True
os.chdir(sys.argv[1]); sys.path.insert(0, ".")
from multiprocessing import Pool
import numpy as np
from src.config import load_config
from src.models.motion_data import drive_features, end_indices
cfg = load_config("configs/sih_mvp.yaml")
drives = cfg["split"]["train"] + cfg["split"]["val"] + cfg["split"]["test"]
def f(d):
    t = time.time(); x = drive_features(d, cfg)
    n = len(end_indices(x["valid"], x["session"])) if len(x["t"]) else 0
    return d, len(x["t"]), int(x["valid"].sum()) if len(x["t"]) else 0, n, round(time.time() - t, 1)
with Pool(4) as p:
    for r in p.imap_unordered(f, sorted(drives, key=lambda d: -os.path.getsize(f"data/iovnbd/{d}/S.csv"))): print(*r, flush=True)
