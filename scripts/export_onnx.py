"""Export MotionNet to ONNX for on-device (edge) inference.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --model results/models/motionnet.pt --out results/models/motionnet_mobile.onnx

The exported graph is self-contained, so the phone passes raw features:
  input   features  float32 (1, window, 5)  vehicle-frame IMU features at 10 Hz, oldest first,
                                            columns in the order stored in the sidecar JSON
                                            (a_f, a_l, a_u, w_u, w_h)
  outputs speed     float32 (1,)  forward speed, m/s
          sigma     float32 (1,)  1-sigma uncertainty, m/s (validation-calibrated)

The feature normalisation and the validation sigma scale are baked into the graph.
The batch and sequence dimensions are fixed (no dynamic axes) so mobile runtimes
(ONNX Runtime Mobile / NNAPI / CoreML) can plan memory once. After export the
script runs the ONNX file with ONNX Runtime and checks it matches PyTorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.motion_net import MotionModel  # noqa: E402


class MobileMotionNet(nn.Module):
    """Normalisation + GRU + calibrated sigma in one graph."""

    def __init__(self, model: MotionModel):
        super().__init__()
        self.net = model.net
        self.register_buffer("mean", torch.as_tensor(model.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(model.std, dtype=torch.float32))
        self.sigma_scale = float(model.sigma_scale)

    def forward(self, features):
        mu, logvar = self.net((features - self.mean) / self.std)
        return mu, torch.exp(0.5 * logvar) * self.sigma_scale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="results/models/motionnet.pt")
    ap.add_argument("--out", default="results/models/motionnet_mobile.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args(argv)

    model = MotionModel.load(a.model)
    mobile = MobileMotionNet(model).eval()
    shape = (1, model.window, len(model.features))
    print(f"MotionNet: {model.kind}, {model.n_params} params, features {model.features}, "
          f"fixed input {shape} ({model.window / 10:.0f} s at 10 Hz)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.onnx.export(mobile, torch.zeros(shape), a.out, input_names=["features"],
                      output_names=["speed", "sigma"], opset_version=a.opset, dynamo=False)

    # parity check against the PyTorch model the engine uses
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(a.out))
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = (model.mean + model.std * rng.standard_normal((64,) + shape[1:])).astype(np.float32)
    mu_pt, sd_pt = model.predict_batch(x)
    mu_ox = np.array([sess.run(None, {"features": x[i:i + 1]})[0][0] for i in range(len(x))])
    sd_ox = np.array([sess.run(None, {"features": x[i:i + 1]})[1][0] for i in range(len(x))])
    d_mu, d_sd = float(np.max(np.abs(mu_pt - mu_ox))), float(np.max(np.abs(sd_pt - sd_ox)))
    print(f"ONNX Runtime vs PyTorch on 64 random windows: max |speed diff| {d_mu:.2e} m/s, "
          f"max |sigma diff| {d_sd:.2e} m/s")
    if d_mu > 1e-4 or d_sd > 1e-4:
        raise SystemExit("parity check FAILED")

    meta = {"source": a.model, "opset": a.opset, "input": {"name": "features", "shape": list(shape),
            "dtype": "float32", "rate_hz": 10, "columns": model.features},
            "outputs": {"speed": "m/s, forward speed", "sigma": "m/s, 1-sigma (validation-calibrated)"},
            "normalisation": "baked into the graph", "params": model.n_params,
            "parity_max_abs_diff": {"speed": d_mu, "sigma": d_sd}}
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {a.out} ({os.path.getsize(a.out) / 1024:.1f} KB) and {os.path.splitext(a.out)[0]}.json")


if __name__ == "__main__":
    main()
