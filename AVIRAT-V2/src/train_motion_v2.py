"""Train MotionNet v2 candidates (study: docs/MOTIONNET_V2_REPORT.md).

    python -m src.train_motion_v2 --exp B3 [--fold official|device_A|route_Vw] [--out results/models_v2]

Protocol (all enforced here):
  * causal v2 features (motion_features.session_features_causal), split by complete drive;
  * normalisation statistics from TRAINING rows only;
  * model selection (early stopping) and sigma calibration on the fold's VALIDATION drives only;
  * the test drives (S1, M) are never loaded by this module;
  * the original checkpoint results/models/motionnet.pt is never written (MotionModelV2.save refuses).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .config import load_config
from .models.motion_data import WindowSet, drive_meta
from .models.motion_features import CONTEXT_CLASSES, V2_CHANNELS
from .models.motion_v2 import ARCHS, MotionModelV2

V1_FEATURES = ["a_f", "a_l", "a_u", "w_u", "w_h"]
RAW7 = V1_FEATURES + ["e_acc", "e_gyr"]                 # raw (un-normalised) vibration energy
NORM7 = V1_FEATURES + ["e_acc_n", "e_gyr_n"]            # device-normalised vibration energy
AUG_OFF = {"rot_deg": 0.0, "tilt_deg": 0.0, "gain": 0.0, "e_shift": 0.0}

# The experiment grid. Each row changes one thing relative to its parent (ablation chain).
EXPERIMENTS = {
    "A1": dict(arch="gru", arch_kw=dict(hidden=64, short=50, acc=False, ctx=False), features=V1_FEATURES,
               window=50, loss="nll", balance=False, aug=AUG_OFF, note="V1 architecture + recipe, causal features"),
    "A2": dict(arch="gru", arch_kw=dict(hidden=64, short=50, acc=False, ctx=False), features=RAW7,
               window=50, loss="nll", balance=False, aug=AUG_OFF, note="A1 + vibration-energy channels"),
    "B1": dict(arch="msgru", arch_kw=dict(hidden=32, long_hidden=16, short=50, pool=10, acc=True, ctx=True),
               features=RAW7, window=600, loss="nll", balance=False, aug=AUG_OFF,
               note="compact multi-scale GRU (5 s @ 10 Hz + 60 s @ 1 Hz), multi-head"),
    "B2": dict(arch="msgru", arch_kw=dict(hidden=32, long_hidden=16, short=50, pool=10, acc=True, ctx=True),
               features=RAW7, window=600, loss="beta_nll", balance=True, aug=AUG_OFF,
               note="B1 + beta-NLL (beta 0.5) + speed-balanced weights"),
    "B3": dict(arch="msgru", arch_kw=dict(hidden=32, long_hidden=16, short=50, pool=10, acc=True, ctx=True),
               features=RAW7, window=600, loss="beta_nll", balance=True,
               aug={"rot_deg": 5.0, "tilt_deg": 1.5, "gain": 0.05, "e_shift": 0.5},
               note="B2 + mount / device augmentation"),
    "B0": dict(arch="msgru", arch_kw=dict(hidden=32, long_hidden=16, short=50, pool=10, acc=True, ctx=True),
               features=V1_FEATURES, window=600, loss="beta_nll", balance=True,
               aug={"rot_deg": 5.0, "tilt_deg": 1.5, "gain": 0.05, "e_shift": 0.0},
               note="B3 without the vibration-energy channels (ablation)"),
    "B4": dict(arch="msgru", arch_kw=dict(hidden=32, long_hidden=16, short=50, pool=10, acc=True, ctx=True),
               features=NORM7, window=600, loss="nll",
               balance=False, aug=AUG_OFF, note="B1 with device-normalised vibration energy (calibration-segment offset)"),
    "B5": dict(arch="msgru", arch_kw=dict(hidden=32, long_hidden=16, short=50, pool=10, acc=True, ctx=True),
               features=NORM7, window=600, loss="nll",
               balance=False, aug={"rot_deg": 5.0, "tilt_deg": 1.5, "gain": 0.05, "e_shift": 0.0},
               note="B4 + mount augmentation (no energy shift: offsets are already normalised)"),
    "B6": dict(arch="msgru", arch_kw=dict(hidden=32, long_hidden=16, short=50, pool=10, acc=True, ctx=True),
               features=V2_CHANNELS, window=600, loss="nll", balance=False, aug=AUG_OFF,
               note="B1 with raw AND device-normalised energy (9 channels)"),
    "C1": dict(arch="tcn", arch_kw=dict(ch=24, k=3, acc=True, ctx=True), features=RAW7, window=600,
               loss="beta_nll", balance=True, aug={"rot_deg": 5.0, "tilt_deg": 1.5, "gain": 0.05, "e_shift": 0.5},
               note="causal depthwise-separable TCN, 51 s receptive field, B3 recipe"),
}

FOLDS = {
    # name: (train drives, validation drives) - test drives are never part of any fold
    "official": None,
    "device_A": "hold out every driver-A drive (phone A) of train+val; validate on them",
    "route_Vw": "hold out every Vw drive of train+val; validate on them",
}


def fold_drives(cfg: dict, fold: str) -> tuple[list[str], list[str]]:
    tr, va = list(cfg["split"]["train"]), list(cfg["split"]["val"])
    if fold == "official":
        return tr, va
    meta = drive_meta(cfg)
    pool = tr + va
    if fold == "device_A":
        held = [d for d in pool if meta[d]["device"] == "A"]
    elif fold == "route_Vw":
        held = [d for d in pool if meta[d]["group"] == "Vw"]
    else:
        raise ValueError(fold)
    return [d for d in pool if d not in held], held


def _augment(x: np.ndarray, cols: list[str], aug: dict, rng: np.random.Generator) -> np.ndarray:
    """Per-window mount and device perturbations on raw feature values (B, W, C)."""
    B = x.shape[0]
    x = x.copy()
    ci = {c: i for i, c in enumerate(cols)}
    if aug["rot_deg"] and "a_f" in ci and "a_l" in ci:            # residual mount-yaw error
        a = np.deg2rad(rng.uniform(-aug["rot_deg"], aug["rot_deg"], B))[:, None]
        af, al = x[:, :, ci["a_f"]].copy(), x[:, :, ci["a_l"]].copy()
        x[:, :, ci["a_f"]] = np.cos(a) * af - np.sin(a) * al
        x[:, :, ci["a_l"]] = np.sin(a) * af + np.cos(a) * al
    if aug["tilt_deg"] and "a_f" in ci:                              # residual pitch: gravity leaks into a_f
        x[:, :, ci["a_f"]] += 9.81 * np.sin(np.deg2rad(rng.normal(0, aug["tilt_deg"], B)))[:, None]
    if aug["gain"]:
        for c in ("a_f", "a_l", "a_u", "w_u", "w_h"):
            if c in ci:
                x[:, :, ci[c]] *= (1 + rng.normal(0, aug["gain"], B))[:, None]
    if aug["e_shift"]:                                               # device sensitivity: log-energy offset
        for c in ("e_acc", "e_gyr", "e_acc_n", "e_gyr_n"):
            if c in ci:
                x[:, :, ci[c]] += rng.normal(0, aug["e_shift"], B)[:, None]
    return x


def _nll(mu, logvar, y, beta: float, w):
    var = logvar.exp()
    l = 0.5 * (logvar + (y - mu) ** 2 / var)
    if beta > 0:
        l = l * var.detach() ** beta                                # beta-NLL (Seitzer et al., 2022)
    return (l * w).sum() / w.sum()


def train(cfg: dict, exp: dict, train_ws: WindowSet, val_ws: WindowSet, epochs: int = 10, per_epoch: int = 100_000,
          batch: int = 512, lr: float = 2e-3, seed: int = 7, patience: int = 3, log=print) -> tuple[MotionModelV2, list]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(os.cpu_count() or 1, 1))
    cols = list(exp["features"])
    ci = [V2_CHANNELS.index(c) for c in cols]
    W = exp["window"]
    mean = train_ws.X[train_ws.ends][:, ci].mean(0)                 # training windows only
    std = train_ws.X[train_ws.ends][:, ci].std(0) + 1e-6
    net = ARCHS[exp["arch"]](len(cols), **exp["arch_kw"])
    model = MotionModelV2(net, exp["arch"], exp["arch_kw"], cols, W, mean, std)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    v_all = train_ws.v[train_ws.ends]
    if exp["balance"]:                                               # inverse frequency over 2.5 m/s bins, capped
        b = np.minimum((v_all / 2.5).astype(int), 13)
        cnt = np.bincount(b, minlength=14).astype(float)
        wts = 1.0 / np.maximum(cnt[b], 1)
        wts = np.minimum(wts / wts.mean(), 5.0)
        wts /= wts.mean()
    else:
        wts = np.ones(len(v_all))
    ctx_w = torch.ones(len(CONTEXT_CLASSES))
    val_idx = np.arange(0, len(val_ws.ends), 3)
    vt = val_ws.targets(val_idx)
    vx = val_ws.windows(val_idx, W)[:, :, ci]
    hist, best = [], (np.inf, None, -1)
    huber_epochs = 2
    for ep in range(epochs):
        net.train()
        t0 = time.time()
        perm = rng.choice(len(train_ws.ends), min(per_epoch, len(train_ws.ends)), replace=False)
        tot = 0.0
        for i in range(0, len(perm), batch):
            ib = perm[i:i + batch]
            x = train_ws.windows(ib, W)[:, :, ci]
            x = _augment(x, cols, exp["aug"], rng)
            xt = torch.from_numpy((x - mean) / std).float()
            tg = train_ws.targets(ib)
            y = torch.from_numpy(tg["v"]).float()
            w = torch.from_numpy(wts[ib]).float()
            o = net(xt)
            if ep < huber_epochs or exp["loss"] == "huber":
                loss = (torch.nn.functional.huber_loss(o["mu"], y, delta=1.0, reduction="none") * w).sum() / w.sum() \
                    + 0.01 * o["logvar"].pow(2).mean()
            else:
                loss = _nll(o["mu"], o["logvar"], y, 0.5 if exp["loss"] == "beta_nll" else 0.0, w)
            if "acc_mu" in o:
                ya = torch.from_numpy(np.clip(tg["acc_corr"], -5, 5)).float()
                loss = loss + 0.5 * _nll(o["acc_mu"], o["acc_logvar"], ya, 0.5, torch.ones_like(ya))
            if "ctx_logits" in o:
                loss = loss + 0.2 * torch.nn.functional.cross_entropy(o["ctx_logits"], torch.from_numpy(tg["ctx"]),
                                                                      weight=ctx_w, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(ib)
        sched.step()
        net.eval()
        model.sigma_scale = 1.0
        p = model.predict_all(vx)
        rmse = float(np.sqrt(np.mean((p["mu"] - vt["v"]) ** 2)))
        hist.append(dict(epoch=ep + 1, loss=tot / len(perm), val_rmse=rmse, seconds=round(time.time() - t0, 1)))
        log(f"[{exp.get('name', '?')}] epoch {ep + 1} loss {tot / len(perm):.3f} val RMSE {rmse:.3f} ({time.time() - t0:.0f}s)")
        if ep >= huber_epochs and rmse < best[0]:
            best = (rmse, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1)
        if ep >= huber_epochs and ep + 1 - best[2] >= patience:
            break
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()
    # sigma calibration on validation drives: E[z^2] = 1, separately for each head
    model.sigma_scale = model.acc_sigma_scale = 1.0
    p = model.predict_all(vx)
    model.sigma_scale = float(np.sqrt(np.mean(((p["mu"] - vt["v"]) / p["sd"]) ** 2)))
    if "acc_mu" in p:
        ya = np.clip(vt["acc_corr"], -5, 5)
        model.acc_sigma_scale = float(np.sqrt(np.mean(((p["acc_mu"] - ya) / p["acc_sd"]) ** 2)))
    xe = train_ws.X[train_ws.ends]
    offset_prior = [float(np.nanmedian(xe[:, 5] - xe[:, 7])), float(np.nanmedian(xe[:, 6] - xe[:, 8]))]
    model.meta = dict(exp, history=hist, energy_offset_prior=offset_prior, best_epoch=best[2], sigma_scale=model.sigma_scale,
                      acc_sigma_scale=model.acc_sigma_scale, params=model.n_params,
                      train_drives=train_ws.drives, val_drives=val_ws.drives,
                      normalisation="fitted on training windows only")
    return model, hist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sih_mvp.yaml")
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--fold", default="official", choices=sorted(FOLDS))
    ap.add_argument("--out", default="results/models_v2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--per-epoch", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    tr, va = fold_drives(cfg, a.fold)
    assert not (set(tr) | set(va)) & set(cfg["split"]["test"]), "test drives must never be used here"
    train_ws, val_ws = WindowSet.build(tr, cfg), WindowSet.build(va, cfg)
    exp = dict(EXPERIMENTS[a.exp], name=a.exp, fold=a.fold, seed=a.seed)
    print(f"{a.exp} on fold {a.fold}: {len(train_ws.ends)} train / {len(val_ws.ends)} val windows")
    model, _ = train(cfg, exp, train_ws, val_ws, epochs=a.epochs, per_epoch=a.per_epoch, seed=a.seed)
    path = os.path.join(a.out, f"{a.exp}_{a.fold}_s{a.seed}.pt")
    model.save(path)
    print("saved", path, json.dumps({k: model.meta[k] for k in ("best_epoch", "sigma_scale", "params")}))


if __name__ == "__main__":
    main()
