import numpy as np
import pytest

from src.dataset import SessionData, check_split, make_windows, window_indices


def test_configured_split_is_disjoint(cfg):
    sp = check_split(cfg)
    assert set(sp["test"]).isdisjoint(sp["train"]) and set(sp["test"]).isdisjoint(sp["val"])


def test_overlapping_split_is_rejected(cfg):
    cfg["split"]["val"] = list(cfg["split"]["val"]) + [cfg["split"]["test"][0]]
    with pytest.raises(AssertionError):
        check_split(cfg)


def test_windows_never_cross_unlabelled_gaps():
    valid = np.array([1] * 10 + [0] + [1] * 10, bool)
    ends = window_indices(valid, 5, 1)
    for e in ends:
        assert valid[e - 4:e + 1].all()
    assert 10 not in ends and 11 not in ends and 14 not in ends


def test_window_target_is_last_sample_no_future():
    n = 50
    feats = np.arange(n, dtype=np.float32)[:, None]
    target = np.arange(n, dtype=np.float32) * 10
    s = SessionData("d", 0, np.arange(n) * 0.1, feats, target, np.ones(n, bool), {})
    X, y, groups = make_windows([s], 8, 1)
    assert np.allclose(y, X[:, -1, 0] * 10)       # label aligned with the newest input sample
    assert (X[:, :, 0].max(1) <= y / 10).all()    # no input sample from the future


def test_windows_from_different_drives_stay_apart():
    a = SessionData("A", 0, np.arange(20) * 0.1, np.zeros((20, 1), np.float32), np.zeros(20, np.float32), np.ones(20, bool), {})
    b = SessionData("B", 0, np.arange(20) * 0.1, np.ones((20, 1), np.float32), np.ones(20, np.float32), np.ones(20, bool), {})
    X, y, g = make_windows([a, b], 5, 1)
    for xi, gi in zip(X, g):
        assert len(set(xi[:, 0])) == 1           # a window never mixes sessions
