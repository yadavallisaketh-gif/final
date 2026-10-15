"""Figures for docs/MOTIONNET_V2_REPORT.md from run_eval JSON files (audit only).

    python -B AVIRAT-V2/audit/motion/plots.py <eval.json> <out_dir> <title> MODEL ...
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

path, out, title, models = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
r = json.load(open(path))
os.makedirs(out, exist_ok=True)
tag = os.path.splitext(os.path.basename(path))[0]

fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
nominal = [0.5, 0.8, 0.9, 0.95]
ax[0].plot([0, 1], [0, 1], "k:", lw=1)
for m in models:
    s = r[m]["speed"]
    ax[0].plot(nominal, [s[f"cov{int(c * 100)}"] for c in nominal], "o-", label=m)
ax[0].set(xlabel="nominal coverage", ylabel="empirical coverage", title="reliability (central intervals)",
          xlim=(0.4, 1), ylim=(0.4, 1))
ax[0].legend(fontsize=8)
for m in models:
    ax[1].step(np.arange(10) / 10, r[m]["speed"]["pit_hist"], where="post", label=m)
ax[1].axhline(0.1, color="k", ls=":")
ax[1].set(xlabel="PIT", ylabel="fraction", title="PIT histogram (flat = calibrated)")
bins = [b["bin"] for b in r[models[0]]["speed"]["bins"]]
w = 0.8 / len(models)
for i, m in enumerate(models):
    d = {b["bin"]: b for b in r[m]["speed"]["bins"]}
    ax[2].bar(np.arange(len(bins)) + i * w, [d[b]["bias"] for b in bins], w, label=m)
    ax[3].bar(np.arange(len(bins)) + i * w, [d[b]["rmse"] for b in bins], w, label=m)
for a, lab in ((ax[2], "bias (m/s)"), (ax[3], "RMSE (m/s)")):
    a.set_xticks(np.arange(len(bins)) + 0.4 - w / 2)
    a.set_xticklabels(bins, fontsize=8)
    a.set(xlabel="true speed bin (m/s)", ylabel=lab, title=lab + " by speed")
    a.axhline(0, color="k", lw=0.8)
ax[3].legend(fontsize=8)
fig.suptitle(title)
fig.tight_layout()
fig.savefig(os.path.join(out, f"{tag}_calibration_bins.png"), dpi=110)
plt.close(fig)

fig, a = plt.subplots(figsize=(7, 4.2))
for m in models:
    acf = r[m]["error_correlation"]["acf"]
    a.plot(np.arange(len(acf)), acf, label=f"{m} (T_int {r[m]['error_correlation']['integral_time_s']:.0f} s)")
a.axhline(0, color="k", lw=0.8)
a.axhline(np.exp(-1), color="k", ls=":", lw=0.8)
a.set(xlabel="lag (s)", ylabel="autocorrelation of speed error", title=f"temporal error correlation - {title}")
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, f"{tag}_error_acf.png"), dpi=110)
plt.close(fig)
print("saved", out)
