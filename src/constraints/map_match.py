"""Feature 3 - simple offline road matching.

Road geometry is a set of straight segments in the local ENU frame. Sources:

  * OpenStreetMap: Overpass JSON, .osm XML or GeoJSON LineStrings
    (`RoadNetwork.from_file`; download with `python -m src.constraints.fetch_osm`)
  * a trace-derived proxy built from *training-drive* reference tracks
    (`RoadNetwork.from_traces`), used when OSM is unavailable offline

Matching never looks at the hidden GNSS track: candidates are found around the
*estimated* position and scored by perpendicular distance, heading agreement
and continuity with the previously matched segment. Implausible matches are
rejected rather than snapped.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..data_io import latlon_to_local

DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
            "living_street", "service", "road"}


@dataclass
class MatchResult:
    segment: int
    px: float
    py: float
    road_yaw: float         # segment direction closest to the vehicle heading
    distance: float
    heading_error: float    # rad
    score: float
    streak: int = 1         # consecutive matches on the same or a connected segment


class RoadNetwork:
    def __init__(self, segments: np.ndarray, node_ids: np.ndarray | None = None, oneway: np.ndarray | None = None,
                 source: str = "unknown", connect_radius: float = 3.0):
        """segments: (M,4) [x0,y0,x1,y1]; node_ids: (M,2) endpoint node ids (optional)."""
        self.seg = np.asarray(segments, float).reshape(-1, 4)
        self.source = source
        self.oneway = np.zeros(len(self.seg), bool) if oneway is None else np.asarray(oneway, bool)
        d = self.seg[:, 2:] - self.seg[:, :2]
        self.length = np.hypot(d[:, 0], d[:, 1])
        keep = self.length > 0.1
        self.seg, self.oneway, self.length = self.seg[keep], self.oneway[keep], self.length[keep]
        if node_ids is not None:
            node_ids = np.asarray(node_ids)[keep]
        self.dir = (self.seg[:, 2:] - self.seg[:, :2]) / self.length[:, None]
        self.yaw = np.arctan2(self.dir[:, 1], self.dir[:, 0])
        mids = 0.5 * (self.seg[:, :2] + self.seg[:, 2:])
        self._mid_tree = cKDTree(mids) if len(mids) else None
        self._half_len = 0.5 * self.length
        self._build_adjacency(node_ids, connect_radius)

    def __len__(self):
        return len(self.seg)

    def _build_adjacency(self, node_ids, radius):
        m = len(self.seg)
        self.neighbours: list[set[int]] = [set() for _ in range(m)]
        if m == 0:
            return
        if node_ids is not None:
            by_node: dict = {}
            for i, (a, b) in enumerate(node_ids):
                by_node.setdefault(a, []).append(i)
                by_node.setdefault(b, []).append(i)
            groups = by_node.values()
        else:  # geometric connectivity for trace maps
            ends = np.r_[self.seg[:, :2], self.seg[:, 2:]]
            owner = np.r_[np.arange(m), np.arange(m)]
            tree = cKDTree(ends)
            groups = ([owner[j] for j in grp] for grp in tree.query_ball_point(ends, radius))
        for grp in groups:
            for i in grp:
                self.neighbours[i].update(grp)

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_traces(cls, traces: list[tuple[np.ndarray, np.ndarray]], spacing: float = 8.0) -> "RoadNetwork":
        """Polyline segments from reference tracks (lists of x, y arrays in ENU)."""
        segs = []
        for x, y in traces:
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) < 2:
                continue
            d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
            s = np.arange(0, d[-1], spacing)
            if len(s) < 2:
                continue
            xs, ys = np.interp(s, d, x), np.interp(s, d, y)
            segs.append(np.c_[xs[:-1], ys[:-1], xs[1:], ys[1:]])
        seg = np.concatenate(segs) if segs else np.zeros((0, 4))
        return cls(seg, source="trace-derived (training drives)", connect_radius=spacing * 0.6)

    @classmethod
    def from_file(cls, path: str, origin: tuple[float, float]) -> "RoadNetwork":
        if path.endswith((".geojson", ".json")):
            with open(path) as f:
                data = json.load(f)
            if "elements" in data:
                return cls._from_overpass(data, origin)
            return cls._from_geojson(data, origin)
        return cls._from_osm_xml(path, origin)

    @classmethod
    def _from_ways(cls, nodes: dict, ways: list[tuple[list, bool]], origin, source: str) -> "RoadNetwork":
        segs, ids, oneway = [], [], []
        lat0, lon0 = origin
        for refs, ow in ways:
            pts = [(r, nodes[r]) for r in refs if r in nodes]
            for (ra, (la, lo)), (rb, (lb, lob)) in zip(pts[:-1], pts[1:]):
                xa, ya = latlon_to_local(la, lo, lat0, lon0)
                xb, yb = latlon_to_local(lb, lob, lat0, lon0)
                segs.append([float(xa), float(ya), float(xb), float(yb)])
                ids.append((ra, rb))
                oneway.append(ow)
        return cls(np.array(segs) if segs else np.zeros((0, 4)), np.array(ids, dtype=object) if ids else None,
                   np.array(oneway, bool), source=source)

    @classmethod
    def _from_overpass(cls, data: dict, origin) -> "RoadNetwork":
        nodes = {e["id"]: (e["lat"], e["lon"]) for e in data["elements"] if e["type"] == "node"}
        ways = [(e["nodes"], e.get("tags", {}).get("oneway") == "yes") for e in data["elements"]
                if e["type"] == "way" and e.get("tags", {}).get("highway") in DRIVABLE]
        return cls._from_ways(nodes, ways, origin, "OpenStreetMap (Overpass)")

    @classmethod
    def _from_osm_xml(cls, path: str, origin) -> "RoadNetwork":
        root = ET.parse(path).getroot()
        nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in root.iter("node")}
        ways = []
        for w in root.iter("way"):
            tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
            if tags.get("highway") in DRIVABLE:
                ways.append(([nd.get("ref") for nd in w.iter("nd")], tags.get("oneway") == "yes"))
        return cls._from_ways(nodes, ways, origin, "OpenStreetMap (XML)")

    @classmethod
    def _from_geojson(cls, data: dict, origin) -> "RoadNetwork":
        nodes, ways = {}, []
        for f in data.get("features", []):
            geom = f.get("geometry", {})
            lines = [geom["coordinates"]] if geom.get("type") == "LineString" else (
                geom["coordinates"] if geom.get("type") == "MultiLineString" else [])
            for line in lines:
                refs = []
                for lon, lat, *_ in line:
                    key = (round(lat, 7), round(lon, 7))
                    nodes[key] = (lat, lon)
                    refs.append(key)
                ways.append((refs, f.get("properties", {}).get("oneway") == "yes"))
        return cls._from_ways(nodes, ways, origin, "GeoJSON")

    # ------------------------------------------------------------------ queries
    def candidates(self, x: float, y: float, radius: float) -> np.ndarray:
        if self._mid_tree is None:
            return np.zeros(0, int)
        reach = radius + float(self._half_len.max())
        idx = np.asarray(self._mid_tree.query_ball_point([x, y], reach), int)
        return idx

    def project(self, idx: np.ndarray, x: float, y: float):
        a = self.seg[idx, :2]
        u = np.clip(((np.array([x, y]) - a) * self.dir[idx]).sum(1), 0.0, self.length[idx])
        p = a + self.dir[idx] * u[:, None]
        return p, np.hypot(p[:, 0] - x, p[:, 1] - y)


class MapMatcher:
    def __init__(self, network: RoadNetwork, cfg: dict):
        mc = cfg["map"]
        self.net = network
        self.radius = mc["radius_m"]
        self.heading_gate = np.deg2rad(mc["heading_gate_deg"])
        self.continuity_bonus = mc["continuity_bonus"]
        self.max_score = mc["max_score"]
        self.sigma_d = mc["sigma_across_m"] * 2.0
        self.ambiguity_margin = mc.get("ambiguity_margin", 2.0)
        self.branch_angle = np.deg2rad(mc.get("branch_angle_deg", 20.0))
        self.prev: int | None = None
        self.streak = 0
        self.decisions: list[dict] = []

    def reset(self):
        self.prev = None
        self.streak = 0

    def match(self, x: float, y: float, yaw: float, t: float = 0.0) -> MatchResult | None:
        idx = self.net.candidates(x, y, self.radius)
        if len(idx) == 0:
            self.prev = None
            self.streak = 0
            return None
        p, dist = self.net.project(idx, x, y)
        near = dist <= self.radius
        if not near.any():
            self.prev = None
            self.streak = 0
            return None
        idx, p, dist = idx[near], p[near], dist[near]
        seg_yaw = self.net.yaw[idx]
        fwd = np.abs((yaw - seg_yaw + np.pi) % (2 * np.pi) - np.pi)
        bwd = np.abs((yaw - seg_yaw) % (2 * np.pi) - np.pi)
        use_bwd = (bwd < fwd) & ~self.net.oneway[idx]
        herr = np.where(use_bwd, bwd, fwd)
        road_yaw = np.where(use_bwd, seg_yaw + np.pi, seg_yaw)
        score = (dist / self.sigma_d) ** 2 + (herr / (self.heading_gate / 2)) ** 2
        if self.prev is not None:
            linked = np.array([i == self.prev or i in self.net.neighbours[self.prev] for i in idx])
            score = score - self.continuity_bonus * linked
        score[herr > self.heading_gate] = np.inf
        k = int(np.argmin(score))
        if not np.isfinite(score[k]) or score[k] > self.max_score:
            self.decisions.append(dict(t=t, matched=False, candidates=int(len(idx))))
            self.prev = None
            self.streak = 0
            return None
        # Junction ambiguity: another plausible road with a clearly different
        # direction scores almost as well -> do not guess, skip this update.
        other = np.abs((road_yaw - road_yaw[k] + np.pi) % (2 * np.pi) - np.pi) > self.branch_angle
        if (other & (score < score[k] + self.ambiguity_margin)).any():
            self.decisions.append(dict(t=t, matched=False, ambiguous=True, candidates=int(len(idx))))
            return None
        k_seg = int(idx[k])
        linked_prev = self.prev is not None and (k_seg == self.prev or k_seg in self.net.neighbours[self.prev])
        self.streak = self.streak + 1 if linked_prev else 1
        self.prev = k_seg
        res = MatchResult(k_seg, float(p[k, 0]), float(p[k, 1]), float(road_yaw[k]),
                          float(dist[k]), float(herr[k]), float(score[k]), self.streak)
        self.decisions.append(dict(t=t, matched=True, segment=res.segment, distance=round(res.distance, 1),
                                   heading_err_deg=round(float(np.degrees(res.heading_error)), 1)))
        return res
