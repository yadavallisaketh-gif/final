"""Download drivable OSM roads for a bounding box via Overpass (run once, offline use after).

    python -m src.constraints.fetch_osm --bbox 52.36 -1.65 52.46 -1.45 --out data/osm/coventry.json

The build environment used for the reported results could not reach any OSM
endpoint, so evaluation there fell back to the trace-derived road proxy. With
a downloaded file, pass `--set map.osm_path=data/osm/coventry.json` to evaluate.
"""
from __future__ import annotations

import argparse
import os
import urllib.parse
import urllib.request

QUERY = """[out:json][timeout:120];
way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|_link|living_street|service"]({s},{w},{n},{e});
(._;>;);
out body;"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("SOUTH", "WEST", "NORTH", "EAST"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="https://overpass-api.de/api/interpreter")
    a = ap.parse_args(argv)
    s, w, n, e = a.bbox
    body = urllib.parse.urlencode({"data": QUERY.format(s=s, w=w, n=n, e=e)}).encode()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with urllib.request.urlopen(a.endpoint, data=body, timeout=180) as r, open(a.out, "wb") as f:
        f.write(r.read())
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
