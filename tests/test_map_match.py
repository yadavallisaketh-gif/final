import json

import numpy as np

from src.constraints.map_match import MapMatcher, RoadNetwork


def _grid():
    # two parallel east-west roads 40 m apart and one north-south road crossing them
    segs = []
    for y in (0.0, 40.0):
        xs = np.arange(-200, 201, 20.0)
        segs += [[a, y, b, y] for a, b in zip(xs[:-1], xs[1:])]
    ys = np.arange(-100, 141, 20.0)
    segs += [[0.0, a, 0.0, b] for a, b in zip(ys[:-1], ys[1:])]
    return RoadNetwork(np.array(segs), source="test")


def test_matches_nearest_heading_consistent_road(cfg):
    m = MapMatcher(_grid(), cfg)
    r = m.match(55.0, 6.0, 0.02)                 # driving east, 6 m north of road y=0
    assert r is not None and abs(r.py - 0.0) < 1e-6 and abs(r.road_yaw) < 1e-6


def test_heading_disambiguates_crossing(cfg):
    m = MapMatcher(_grid(), cfg)
    r = m.match(3.0, 4.0, np.pi / 2)             # near the junction but heading north
    assert r is not None and abs(r.px) < 1e-6
    assert abs(r.road_yaw - np.pi / 2) < 1e-6


def test_reverse_direction_on_two_way_road(cfg):
    m = MapMatcher(_grid(), cfg)
    r = m.match(55.0, 2.0, np.pi - 0.05)         # driving west
    assert r is not None and abs(abs(r.road_yaw) - np.pi) < 1e-6


def test_rejects_far_or_crosswise_positions(cfg):
    m = MapMatcher(_grid(), cfg)
    assert m.match(100.0, 20.0 + 0.0, np.pi / 2) is None   # between roads, heading across them
    assert m.match(1000.0, 1000.0, 0.0) is None            # no road nearby


def test_continuity_prefers_previous_road(cfg):
    m = MapMatcher(_grid(), cfg)
    first = m.match(50.0, 2.0, 0.0)
    assert abs(first.py) < 1e-6
    # halfway between the two parallel roads: continuity keeps us on the previous one
    r = m.match(70.0, 19.0, 0.0)
    assert r is not None and abs(r.py) < 1e-6


def test_load_overpass_json(tmp_path):
    data = {"elements": [
        {"type": "node", "id": 1, "lat": 52.4000, "lon": -1.5000},
        {"type": "node", "id": 2, "lat": 52.4000, "lon": -1.4990},
        {"type": "node", "id": 3, "lat": 52.4010, "lon": -1.4990},
        {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "residential"}},
        {"type": "way", "id": 11, "nodes": [1, 3], "tags": {"highway": "footway"}},
    ]}
    p = tmp_path / "roads.json"
    p.write_text(json.dumps(data))
    net = RoadNetwork.from_file(str(p), (52.4, -1.5))
    assert len(net) == 2                        # footway ignored
    assert 1 in net.neighbours[0]               # connected through node 2
    assert abs(net.seg[0, 2] - 68.0) < 1.0      # ~68 m east


def test_trace_network(cfg):
    x = np.linspace(0, 500, 200)
    net = RoadNetwork.from_traces([(x, np.zeros_like(x))], spacing=10.0)
    assert len(net) >= 45
    assert MapMatcher(net, cfg).match(250.0, 3.0, 0.0) is not None


def test_ambiguous_junction_is_skipped(cfg):
    # a Y-junction: two roads leave the origin 30 degrees apart
    a = [[0, 0, 100 * np.cos(0.0), 0.0]]
    b = [[0, 0, 100 * np.cos(np.radians(30)), 100 * np.sin(np.radians(30))]]
    net = RoadNetwork(np.array(a + b, float), source="test")
    m = MapMatcher(net, cfg)
    assert m.match(8.0, 2.0, np.radians(15)) is None        # right at the fork, heading between both
    assert m.decisions[-1].get("ambiguous")
    r = m.match(80.0, 1.0, 0.0)                             # well past the fork on road a
    assert r is not None and abs(r.py) < 1e-6
