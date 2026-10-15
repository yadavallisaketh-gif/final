"""Export a MotionNet v2 checkpoint to ONNX, verify parity, size and host-CPU latency.

    python scripts/export_onnx_v2.py results/models_v2/motionnet_v2.pt [--out results/models_v2/motionnet_v2.onnx]

Graph: input `features` float32 (1, window, C) raw v2 features (motion_features.V2_CHANNELS
order as stored in the sidecar), normalisation and sigma calibration baked in. Outputs:
speed, sigma (m/s), acc_corr, acc_sigma (m/s^2), ctx_prob (context classes). Fixed shapes.

Latency is measured with ONNX Runtime on THIS machine's CPU only. It is not an Android
measurement; on-device latency must be measured on the phone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.motion_v2 import ORIGINAL_CHECKPOINT, MotionModelV2  # noqa: E402


class Mobile(nn.Module):
    def __init__(self, m: MotionModelV2):
        super().__init__()
        self.net = m.net
        self.register_buffer("mean", torch.as_tensor(m.mean))
        self.register_buffer("std", torch.as_tensor(m.std))
        self.s, self.sa = float(m.sigma_scale), float(m.acc_sigma_scale)

    def forward(self, features):
        o = self.net((features - self.mean) / self.std)
        outs = [o["mu"], torch.exp(0.5 * o["logvar"]) * self.s]
        if "acc_mu" in o:
            outs += [o["acc_mu"], torch.exp(0.5 * o["acc_logvar"]) * self.sa]
        if "ctx_logits" in o:
            outs.append(torch.softmax(o["ctx_logits"], 1))
        return tuple(outs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args(argv)
    out = a.out or os.path.splitext(a.model)[0] + ".onnx"
    if os.path.abspath(out).startswith(os.path.abspath(os.path.dirname(ORIGINAL_CHECKPOINT)) + os.sep + "motionnet"):
        raise PermissionError("refusing to write next to the original MotionNet export")
    m = MotionModelV2.load(a.model)
    mob = Mobile(m).eval()
    shape = (1, m.window, len(m.features))
    names = ["speed", "sigma"] + (["acc_corr", "acc_sigma"] if m.net.heads.acc is not None else []) \
        + (["ctx_prob"] if m.net.heads.ctx is not None else [])
    torch.onnx.export(mob, torch.zeros(shape), out, input_names=["features"], output_names=names,
                      opset_version=a.opset, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(out))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = (m.mean + m.std * rng.standard_normal((64,) + shape[1:])).astype(np.float32)
    pt = m.predict_all(x)
    keys = {"speed": "mu", "sigma": "sd", "acc_corr": "acc_mu", "acc_sigma": "acc_sd", "ctx_prob": "ctx_prob"}
    diffs = {n: 0.0 for n in names}
    for i in range(len(x)):
        o = sess.run(None, {"features": x[i:i + 1]})
        for n, v in zip(names, o):
            diffs[n] = max(diffs[n], float(np.max(np.abs(v[0] - pt[keys[n]][i]))))
    t0 = time.perf_counter()
    for i in range(200):
        sess.run(None, {"features": x[i % 64:i % 64 + 1]})
    lat = (time.perf_counter() - t0) / 200 * 1000
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    for i in range(50):
        m.predict_all(x[i:i + 1])
    lat_pt = (time.perf_counter() - t0) / 50 * 1000
    meta = {"source": a.model, "source_sha256": hashlib.sha256(open(a.model, "rb").read()).hexdigest(),
            "onnx_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(), "opset": a.opset,
            "input": {"name": "features", "shape": list(shape), "dtype": "float32", "rate_hz": 10,
                      "columns": m.features}, "outputs": names, "params": m.n_params,
            "size_bytes": os.path.getsize(out), "parity_max_abs_diff": diffs,
            "host_cpu_latency_ms": {"onnxruntime_1_thread": round(lat, 3), "pytorch_1_thread": round(lat_pt, 3),
                                    "machine": f"{platform.machine()} {platform.processor() or ''}".strip(),
                                    "note": "host x86 CPU in a cloud container - NOT an Android measurement"},
            "normalisation": "baked into the graph", "sigma": "validation-calibrated, baked into the graph"}
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    if max(diffs.values()) > 1e-4:
        raise SystemExit("parity check FAILED")


if __name__ == "__main__":
    main()
