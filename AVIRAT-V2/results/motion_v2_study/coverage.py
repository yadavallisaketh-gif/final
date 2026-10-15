import sys, os, time
sys.dont_write_bytecode = True
os.chdir(sys.argv[1]); sys.path.insert(0, ".")
from multiprocessing import Pool
import csv, numpy as np
from src.config import load_config
from src.data_io import load_drive
cfg = load_config("configs/sih_mvp.yaml")
def f(d):
    t=time.time()
    try:
        dr = load_drive(d, cfg); df = dr.df
        return d, len(df)/36000, float(df.ref_valid.mean()), float(np.nanmean(df.ref_speed)), round(time.time()-t,1)
    except Exception as e:
        return d, 0, 0, 0, str(e)[:40]
rows=[r["drive_id"] for r in csv.DictReader(open("configs/iovnbd_manifest.csv"))]
with Pool(4) as p:
    for r in p.imap(f, rows): print(*[round(x,2) if isinstance(x,float) else x for x in r], flush=True)
