"""Compare navigation runs from nav_eval.py on identical windows (audit only).

    python -B AVIRAT-V2/audit/motion/nav_compare.py <out.md> REF=ref.csv NAME=run.csv ...

Per variant and blackout duration: median / P90 / worst drift, and against REF (paired,
same windows): windows better / worse, Wilcoxon signed-rank p, bootstrap 95 % CI of the
median paired difference (cluster bootstrap by drive, then window). Consistency:
  q_end = |e_end|^2 / tr(P_xy)  - for a consistent 2-D filter the median is ~0.69 (chi2_2 median / 2)
  speed NEES (mean over blackout) - ~1 when consistent
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

out = sys.argv[1]
runs = dict(a.split("=", 1) for a in sys.argv[2:])
names = list(runs)
ref = names[0]
K = ["drive", "session", "t_start", "duration_s", "variant"]
D = {n: pd.read_csv(p) for n, p in runs.items()}
rng = np.random.default_rng(0)


def boot_median_diff(diff: np.ndarray, drive: np.ndarray, B: int = 5000) -> tuple[float, float]:
    groups = {d: diff[drive == d] for d in np.unique(drive)}
    keys = list(groups)
    meds = []
    for _ in range(B):
        ks = rng.choice(keys, len(keys))
        s = np.concatenate([groups[k][rng.integers(0, len(groups[k]), len(groups[k]))] for k in ks])
        meds.append(np.median(s))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


lines = ["# Navigation-level comparison (validation drives S3c, Vta16, Vfa01; 18 windows per duration)", "",
         f"Reference for paired tests: **{ref}**. Drift in % of distance; diff = run - reference (negative = better).", ""]
for var in ("D", "C+NHC", "C"):
    lines += [f"## Variant {var}", "",
              "| Run | Dur | Median | P90 | Worst | Better/worse vs ref | Median diff [95 % CI] | Wilcoxon p | q_end median | q_end > 3.0 | Speed NEES median | MotionNet updates/window |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        d = D[n][D[n].variant == var]
        for dur, g in d.groupby("duration_s"):
            r = D[ref][(D[ref].variant == var) & (D[ref].duration_s == dur)]
            m = g.merge(r, on=K, suffixes=("", "_ref"))
            diff = (m.drift_percent - m.drift_percent_ref).to_numpy()
            if n == ref:
                cmp_txt, ci_txt, p_txt = "-", "-", "-"
            else:
                better, worse = int((diff < -1e-6).sum()), int((diff > 1e-6).sum())
                lo, hi = boot_median_diff(diff, m.drive.to_numpy())
                p = wilcoxon(diff).pvalue if np.any(np.abs(diff) > 1e-9) else 1.0
                cmp_txt, ci_txt, p_txt = f"{better}/{worse}", f"{np.median(diff):+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{p:.3f}"
            q = g.pos_nees_end
            lines.append(f"| {n} | {int(dur)} | {g.drift_percent.median():.2f} | {g.drift_percent.quantile(0.9):.2f} | "
                         f"{g.drift_percent.max():.2f} | {cmp_txt} | {ci_txt} | {p_txt} | {q.median():.1f} | "
                         f"{(q > 3.0).mean():.2f} | {g.speed_nees.median():.1f} | {g.n_motion.mean():.0f} |")
    lines.append("")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
