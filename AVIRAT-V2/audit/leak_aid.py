"""Does MotionNet depend on the label-derived velocity aid used to build its training features?"""
import sys, os, json
sys.dont_write_bytecode = True
REPO = sys.argv[1]; sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from src.config import load_config
from src.data_io import load_drive
from src.dataset import make_windows, SessionData
from src.models.motion_net import MotionModel
from src.preprocess import fit_alignment, ImuPreprocessor, FEATURE_COLUMNS, gnss_speed
cfg = load_config("configs/sih_mvp.yaml"); model = MotionModel.load(cfg["model"]["path"])
cols = [FEATURE_COLUMNS.index(f) for f in model.features]
res = {}
for split in ("val", "test"):
    acc = {"truth_aid": [], "no_aid": [], "y": []}
    for d in cfg["split"][split]:
        drive = load_drive(d, cfg)
        for sid, g in drive.sessions():
            if g["ref_valid"].sum() < 100 or len(g) < 300: continue
            al = fit_alignment(g, cfg)
            fs = 1.0 / np.median(np.diff(g["t"].to_numpy()))
            A, Gy = g[["ax","ay","az"]].to_numpy(), g[["gx","gy","gz"]].to_numpy()
            for name, spd in (("truth_aid", gnss_speed(g)), ("no_aid", np.full(len(g), np.nan))):
                f = ImuPreprocessor(al, cfg, fs).process_block(A, Gy, spd)[:, cols].astype(np.float32)
                s = SessionData(d, int(sid), g["t"].to_numpy(), f, g["ref_speed"].to_numpy(np.float32), g["ref_valid"].to_numpy(bool), {})
                X, y, _ = make_windows([s], model.window, 5)
                if len(y) == 0: continue
                p, _ = model.predict_batch(X)
                acc[name].append(p)
                if name == "truth_aid": acc["y"].append(y)
    y = np.concatenate(acc["y"])
    res[split] = {k: float(np.sqrt(np.mean((np.concatenate(acc[k]) - y) ** 2))) for k in ("truth_aid", "no_aid")}
    res[split]["windows"] = int(len(y))
print(json.dumps(res, indent=1))
