import pandas as pd, numpy as np, sys, json
k=["drive","session","t_start","duration_s","variant"]
split=sys.argv[1]; tags=sys.argv[2].split(",")
b=pd.read_csv(f"phys_{split}_base.csv")
for tag in ["base"]+tags:
    f=pd.read_csv(f"phys_{split}_{tag}.csv")
    for v,g in f.groupby("variant"):
        m=g.groupby("duration_s").drift_percent.median().round(2).to_dict()
        bb=b[b.variant==v].merge(g,on=k,suffixes=("_b",""))
        better=(bb.drift_percent<bb.drift_percent_b-1e-6).sum(); worse=(bb.drift_percent>bb.drift_percent_b+1e-6).sum()
        extra=""
        for c in ["n_turn_speed","n_zupt","n_turn_angle","n_map_along"]:
            if c in g and g[c].sum()>0: extra+=f" {c[2:]} {g[c].mean():.1f}/win"
        print(f"{tag:10s} {v:8s} med {m} mean60 {g[g.duration_s==60].drift_percent.mean():.1f} spdMAE {g.speed_mae.median():.2f} hdg {g.heading_final_deg.median():.1f} fail>50% {(g.drift_percent>50).mean():.2f} | better {better} worse {worse} of {len(bb)}{extra}")
