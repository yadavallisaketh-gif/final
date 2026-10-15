import sys, json, pandas as pd, numpy as np
rep = pd.read_csv(sys.argv[1]); out = {}
for name in sys.argv[2:]:
    arc = pd.read_csv(name)
    key = ["drive","session","t_start","duration_s","variant"]
    m = rep.merge(arc, on=key, suffixes=("_rep","_arc"))
    cols = [c for c in arc.columns if c not in key and c in rep.columns and pd.api.types.is_numeric_dtype(arc[c]) and c not in ("ms_per_step","update_rate_hz")]
    r = {"rows_reproduced": len(rep), "rows_archived": len(arc), "rows_matched": len(m)}
    exact = [c for c in cols if np.array_equal(m[c+"_rep"].values, m[c+"_arc"].values)]
    diff = {}
    for c in cols:
        if c in exact: continue
        d = (m[c+"_rep"]-m[c+"_arc"]).abs()
        diff[c] = {"max_abs_diff": float(d.max()), "rows_different": int((d>1e-9).sum())}
    r["columns_bit_identical"] = len(exact); r["columns_compared"] = len(cols); r["columns_different"] = diff
    dd = m[m.variant=="D"]
    r["D_drift_max_abs_diff_pct"] = float((dd.drift_percent_rep-dd.drift_percent_arc).abs().max())
    r["by_variant_drift_rows_different"] = {v: int(((g.drift_percent_rep-g.drift_percent_arc).abs()>1e-9).sum()) for v,g in m.groupby("variant")}
    out[name.split("/")[-1]] = r
print(json.dumps(out, indent=1))
