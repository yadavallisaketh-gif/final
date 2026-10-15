"""Architecture E: non-neural speed models on causal window statistics.

Row features at a window end e: for every v2 channel, the mean and standard deviation over the
last 1 / 5 / 20 / 60 s (10 / 50 / 200 / 600 samples) plus the current value: 7 x 9 = 63
features, all causal (only rows <= e).

  RidgeSpeed    linear baseline, homoscedastic sigma (validation RMS)
  GBMSpeed      LightGBM mean model + LightGBM model of log squared residuals (out-of-fold by
                drive on the training set, so residuals are not in-sample optimistic); sigma
                scaled to E[z^2] = 1 on validation; optional LightGBM context classifier
"""
from __future__ import annotations

import pickle

import numpy as np

from .motion_data import WindowSet
from .motion_features import CONTEXT_CLASSES, V2_CHANNELS

HORIZONS = (10, 50, 200, 600)


def row_features(ws: WindowSet, idx: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(ws.X.astype(np.float64))
    c1 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    c2 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X ** 2, 0)])
    e = ws.ends[idx] + 1
    feats = [X[e - 1]]
    for h in HORIZONS:
        m = (c1[e] - c1[e - h]) / h
        v = np.maximum((c2[e] - c2[e - h]) / h - m ** 2, 0)
        feats += [m, np.sqrt(v)]
    return np.hstack(feats).astype(np.float32)


def feature_names() -> list[str]:
    names = [f"{c}_now" for c in V2_CHANNELS]
    for h in HORIZONS:
        names += [f"{c}_mean{h}" for c in V2_CHANNELS] + [f"{c}_std{h}" for c in V2_CHANNELS]
    return names


class RidgeSpeed:
    kind = "ridge"

    def fit(self, tr: WindowSet, va: WindowSet, alpha: float = 10.0):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        itr = np.arange(len(tr.ends))
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(row_features(tr, itr), tr.v[tr.ends])
        iva = np.arange(len(va.ends))
        r = self.m.predict(row_features(va, iva)) - va.v[va.ends]
        self.sigma = float(np.sqrt(np.mean(r ** 2)))
        self.n_params = int(self.m[-1].coef_.size + 1)
        return self

    def predict_rows(self, ws, idx):
        mu = np.maximum(self.m.predict(row_features(ws, idx)), 0.0)
        return {"mu": mu, "sd": np.full(len(mu), self.sigma)}


class GBMSpeed:
    kind = "lgbm"

    def fit(self, tr: WindowSet, va: WindowSet, rounds: int = 800, with_ctx: bool = True, seed: int = 7):
        import lightgbm as lgb
        p = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                 seed=seed, num_threads=4)
        itr, iva = np.arange(len(tr.ends)), np.arange(len(va.ends))
        Ftr, Fva = row_features(tr, itr), row_features(va, iva)
        ytr, yva = tr.v[tr.ends], va.v[va.ends]
        dtr = lgb.Dataset(Ftr, ytr)
        dva = lgb.Dataset(Fva, yva, reference=dtr)
        self.mean = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        # out-of-fold residuals by drive (3 folds) -> log-variance model
        d = tr.drive[tr.ends]
        folds = np.array_split(np.random.default_rng(seed).permutation(np.unique(d)), 3)
        oof = np.zeros(len(ytr))
        for f in folds:
            m = np.isin(d, f)
            b = lgb.train(p, lgb.Dataset(Ftr[~m], ytr[~m]), self.mean.best_iteration or rounds)
            oof[m] = b.predict(Ftr[m])
        q = dict(p, learning_rate=0.05, num_leaves=31)
        self.var = lgb.train(q, lgb.Dataset(Ftr, np.log((ytr - oof) ** 2 + 1e-2)), 300)
        self.scale, self.ctx = 1.0, None
        pr = self.predict_rows(va, iva)
        self.scale = float(np.sqrt(np.mean(((pr["mu"] - yva) / pr["sd"]) ** 2)))
        if with_ctx:
            c = dict(p, objective="multiclass", num_class=len(CONTEXT_CLASSES))
            self.ctx = lgb.train(c, lgb.Dataset(Ftr, tr.ctx[tr.ends]), 300,
                                 valid_sets=[lgb.Dataset(Fva, va.ctx[va.ends])],
                                 callbacks=[lgb.early_stopping(30, verbose=False)])
        self.n_trees = self.mean.num_trees()
        self.n_params = int(sum(t["num_leaves"] for t in self.mean.dump_model()["tree_info"]))
        return self

    def predict_rows(self, ws, idx):
        F = row_features(ws, idx)
        mu = np.maximum(self.mean.predict(F), 0.0)
        sd = np.sqrt(np.exp(self.var.predict(F))) * self.scale
        out = {"mu": mu, "sd": sd}
        if self.ctx is not None:
            out["ctx_prob"] = self.ctx.predict(F)
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean.model_to_string(), "var": self.var.model_to_string(),
                         "ctx": self.ctx.model_to_string() if self.ctx else None, "scale": self.scale}, f)
