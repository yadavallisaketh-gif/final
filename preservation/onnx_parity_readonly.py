"""Read-only parity check: committed ONNX vs committed PyTorch checkpoint.

Same procedure as scripts/export_onnx.py (64 random windows, seed 0) but it never
re-exports or writes anything, so the archived ONNX file is the thing under test.

    python onnx_parity_readonly.py <repo_root>
"""
import hashlib
import json
import sys

sys.dont_write_bytecode = True  # never leave __pycache__ inside the archive (python -I ignores the env var)

import numpy as np  # noqa: E402

root = sys.argv[1]
sys.path.insert(0, root)
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from src.models.motion_net import MotionModel  # noqa: E402

pt = f"{root}/results/models/motionnet.pt"
ox = f"{root}/results/models/motionnet_mobile.onnx"
model = MotionModel.load(pt)
onnx.checker.check_model(onnx.load(ox))
sess = ort.InferenceSession(ox, providers=["CPUExecutionProvider"])
shape = (1, model.window, len(model.features))
rng = np.random.default_rng(0)
x = (model.mean + model.std * rng.standard_normal((64,) + shape[1:])).astype(np.float32)
mu_pt, sd_pt = model.predict_batch(x)
out = [sess.run(None, {"features": x[i:i + 1]}) for i in range(len(x))]
mu_ox = np.array([o[0][0] for o in out])
sd_ox = np.array([o[1][0] for o in out])
res = {
    "onnx_sha256": hashlib.sha256(open(ox, "rb").read()).hexdigest(),
    "pt_sha256": hashlib.sha256(open(pt, "rb").read()).hexdigest(),
    "kind": model.kind, "params": model.n_params, "window": model.window,
    "features": list(model.features), "input_shape": list(shape),
    "max_abs_diff_speed": float(np.max(np.abs(mu_pt - mu_ox))),
    "max_abs_diff_sigma": float(np.max(np.abs(sd_pt - sd_ox))),
}
res["pass"] = res["max_abs_diff_speed"] <= 1e-4 and res["max_abs_diff_sigma"] <= 1e-4
print(json.dumps(res, indent=2))
sys.exit(0 if res["pass"] else 1)
