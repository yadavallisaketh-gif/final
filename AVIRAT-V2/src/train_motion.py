"""Train MotionNet (forward speed from smartphone IMU windows).

    python -m src.train_motion                       # GRU, config defaults
    python -m src.train_motion --set model.type=tcn  # TCN variant
    python -m src.train_motion --compare             # GRU vs TCN on the same split

Inputs are IMU features only; the reference speed is used strictly as the
training target. Split is by complete drive (configs/base.yaml).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import load_config
from .dataset import check_split, drive_sessions, make_windows
from .models.motion_net import MotionModel, assert_allowed_features, build_net


def _metrics(pred, y):
    e = pred - y
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "n": int(len(y))}


def load_split_windows(cfg, split):
    mc = cfg["model"]
    out, sessions = {}, {}
    for name, drives in split.items():
        ss = []
        for d in drives:
            try:
                ss += drive_sessions(d, cfg, mc["features"])
            except FileNotFoundError:
                print(f"  (skipping {d}: not downloaded)")
        stride = mc["stride"] if name == "train" else 1
        out[name] = make_windows(ss, mc["window"], stride)
        sessions[name] = ss
        print(f"{name:5s}: {len(ss)} sessions from {len(drives)} drives -> {len(out[name][1])} windows")
    return out, sessions


def train_one(cfg, data, tag: str, out_dir: str):
    mc = cfg["model"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(max(os.cpu_count() or 1, 1))
    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    assert_allowed_features(mc["features"])
    # normalisation from the training split only
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6

    net = build_net(mc["type"], Xtr.shape[-1], mc["hidden"], mc["layers"])
    model = MotionModel(net, mc["type"], mc["features"], mc["window"], mean, std, hidden=mc["hidden"], layers=mc["layers"])
    opt = torch.optim.Adam(net.parameters(), lr=mc["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=mc["epochs"])
    Xt = torch.from_numpy((Xtr - mean) / std).float()
    yt = torch.from_numpy(ytr).float()
    history, best = [], (np.inf, None)
    for ep in range(mc["epochs"]):
        net.train()
        perm = torch.randperm(len(yt))
        t0, tot = time.time(), 0.0
        use_nll = ep >= mc["nll_after_epoch"]
        for i in range(0, len(perm), mc["batch_size"]):
            b = perm[i:i + mc["batch_size"]]
            xb = Xt[b]
            if mc.get("augment_noise", 0) > 0:  # robustness to sensor noise / other phones
                xb = xb * (1 + mc["augment_scale"] * torch.randn(len(b), 1, xb.shape[-1])) \
                    + mc["augment_noise"] * torch.randn_like(xb)
            mu, logvar = net(xb)
            if use_nll:  # Gaussian NLL -> calibrated uncertainty head
                loss = 0.5 * (logvar + (yt[b] - mu) ** 2 / logvar.exp()).mean()
            else:
                loss = torch.nn.functional.huber_loss(mu, yt[b], delta=1.0) + 0.01 * logvar.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        net.eval()
        ptr, _ = model.predict_batch(Xtr[:: max(1, len(Xtr) // 20000)])
        pva, sva = model.predict_batch(Xva)
        mtr = _metrics(ptr, ytr[:: max(1, len(Xtr) // 20000)])
        mva = _metrics(pva, yva)
        history.append(dict(epoch=ep + 1, loss=tot / len(yt), train_rmse=mtr["rmse"], val_rmse=mva["rmse"],
                            val_mae=mva["mae"], seconds=round(time.time() - t0, 1)))
        print(f"[{tag}] epoch {ep + 1:2d} loss {tot / len(yt):.3f} train RMSE {mtr['rmse']:.2f} "
              f"val RMSE {mva['rmse']:.2f} m/s ({time.time() - t0:.0f}s)")
        # model selection on validation drives only (the uncertainty head is only
        # meaningful once the NLL phase has started)
        if use_nll and mva["rmse"] < best[0]:
            best = (mva["rmse"], {k: v.clone() for k, v in net.state_dict().items()})
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()

    # calibrate the predicted std on validation drives: scale so that E[z^2] = 1
    pva, sva = model.predict_batch(Xva)
    scale = float(np.sqrt(np.mean(((pva - yva) / sva) ** 2)))
    model.sigma_scale = scale
    return model, history


def evaluate_model(model: MotionModel, data, sessions, cfg, tag, out_dir):
    res = {}
    for name in ("train", "val", "test"):
        X, y, groups = data[name]
        if len(y) == 0:
            continue
        p, s = model.predict_batch(X)
        m = _metrics(p, y)
        z = (p - y) / s
        m["within_1sigma"] = float(np.mean(np.abs(z) < 1))
        m["within_2sigma"] = float(np.mean(np.abs(z) < 2))
        per = {}
        for g in sorted(set(groups)):
            idx = np.array([gg == g for gg in groups])
            per[g] = _metrics(p[idx], y[idx])
        m["per_session"] = per
        res[name] = m

    # plots: predicted vs reference speed on the longest test session
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    test_sessions = sorted(sessions.get("test", []), key=lambda s: -len(s.t))
    if test_sessions:
        s = test_sessions[0]
        X, y, _ = make_windows([s], model.window, 1)
        p, sd = model.predict_batch(X)
        tt = np.arange(len(y)) * 0.1 / 60
        n = min(len(y), 9000)
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw=dict(height_ratios=[2, 1]))
        ax[0].plot(tt[:n], y[:n], "k", lw=1, label="reference speed (label)")
        ax[0].plot(tt[:n], p[:n], "C0", lw=1, label="MotionNet (IMU only)")
        ax[0].fill_between(tt[:n], (p - 2 * sd)[:n], (p + 2 * sd)[:n], color="C0", alpha=0.15, label="±2σ")
        ax[0].set_ylabel("speed (m/s)")
        ax[0].legend(loc="upper right")
        ax[0].set_title(f"MotionNet on held-out drive {s.drive_id} (session {s.session})")
        ax[1].plot(tt[:n], (p - y)[:n], "C3", lw=0.8)
        ax[1].set_ylabel("error (m/s)")
        ax[1].set_xlabel("time (min)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plots", f"motionnet_{tag}_test_speed.png"), dpi=120)
        plt.close(fig)
    return res


def plot_history(history, tag, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_rmse"] for h in history], "o-", label="train")
    ax.plot(ep, [h["val_rmse"] for h in history], "o-", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("speed RMSE (m/s)")
    ax.legend()
    ax.set_title(f"MotionNet ({tag}) learning curve")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plots", f"motionnet_{tag}_curves.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--compare", action="store_true", help="train GRU and TCN on the same split")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    split = check_split(cfg)
    out_dir = cfg["evaluate"]["output_dir"]
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    data, sessions = load_split_windows(cfg, split)

    kinds = ["gru", "tcn"] if a.compare else [cfg["model"]["type"]]
    summary = {}
    for kind in kinds:
        cfg["model"]["type"] = kind
        model, history = train_one(cfg, data, kind, out_dir)
        res = evaluate_model(model, data, sessions, cfg, kind, out_dir)
        plot_history(history, kind, out_dir)
        info = {
            "kind": kind, "params": model.n_params, "latency_ms": round(model.latency_ms(), 3),
            "sigma_scale": model.sigma_scale, "split": split,
            "normalisation": "fitted on training windows only",
            "windows": {k: int(len(v[1])) for k, v in data.items()},
            "history": history, "metrics": res,
        }
        model.meta = {k: info[k] for k in ("kind", "params", "latency_ms", "sigma_scale", "split", "normalisation", "windows")}
        model.meta["test_rmse"] = res.get("test", {}).get("rmse")
        model.meta["val_rmse"] = res.get("val", {}).get("rmse")
        path = cfg["model"]["path"] if not a.compare else cfg["model"]["path"].replace(".pt", f"_{kind}.pt")
        model.save(path)
        with open(os.path.join(out_dir, "metrics", f"motionnet_{kind}.json"), "w") as f:
            json.dump(info, f, indent=2)
        summary[kind] = {"params": info["params"], "latency_ms": info["latency_ms"],
                         "val_rmse": res["val"]["rmse"], "test_rmse": res.get("test", {}).get("rmse"),
                         "test_mae": res.get("test", {}).get("mae")}
        print(f"[{kind}] saved {path}: params={info['params']} latency={info['latency_ms']} ms "
              f"val RMSE={res['val']['rmse']:.2f} test RMSE={res.get('test', {}).get('rmse', float('nan')):.2f} m/s")
    if a.compare:
        with open(os.path.join(out_dir, "metrics", "motionnet_compare.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
