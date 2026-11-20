"""Signal fusion with L2-regularised logistic regression (the only trained component of UG-RAV)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLACKBOX_CLOSED = ["se", "one_minus_sc", "one_minus_agreement"]
BLACKBOX_FULL = BLACKBOX_CLOSED + ["se_rag", "delta_se", "support_rag", "contradiction_rag", "support_closed",
                                   "contradiction_closed", "agreement", "one_minus_sc_rag"]
CLOSED_FEATURES = ["se", "one_minus_sc", "one_minus_agreement", "one_minus_p_true", "neg_log_min_tok", "ln_pe"]
FULL_FEATURES = CLOSED_FEATURES + ["se_rag", "delta_se", "support_rag", "contradiction_rag", "support_closed",
                                   "contradiction_closed", "agreement", "one_minus_sc_rag"]


def feature_row(rec, names: list[str]) -> list[float]:
    c, r = rec.closed, rec.rag
    vals = {
        "se": c.se,
        "one_minus_sc": 1 - c.self_consistency,
        "one_minus_agreement": 1 - c.greedy_agreement,
        "one_minus_p_true": 1 - c.p_true if c.p_true is not None else np.nan,
        "neg_log_min_tok": -np.log(max(c.min_token_prob, 1e-6)) if c.min_token_prob is not None else np.nan,
        "ln_pe": c.ln_pe if c.ln_pe is not None else np.nan,
    }
    if r is not None:
        vals.update({
            "se_rag": r.se_rag, "delta_se": r.delta_se, "support_rag": r.support_rag,
            "contradiction_rag": r.contradiction_rag, "support_closed": r.support_closed,
            "contradiction_closed": r.contradiction_closed, "agreement": float(r.agreement),
            "one_minus_sc_rag": 1 - r.self_consistency_rag,
        })
    return [float(vals.get(n, np.nan)) for n in names]


def available(records, names: list[str]) -> list[str]:
    """Drop features that are missing for any record (e.g. log-probs from an API that hides them)."""
    X = np.array([feature_row(r, names) for r in records], dtype=float)
    return [n for j, n in enumerate(names) if not np.isnan(X[:, j]).any()]


@dataclass
class LogisticFusion:
    """r̂ = σ(wᵀx + b), fitted by Newton–Raphson (IRLS) with an L2 penalty on standardised features."""

    l2: float = 1.0
    max_iter: int = 50

    def fit(self, X, y) -> LogisticFusion:
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu_) / self.sd_]
        w = np.zeros(Z.shape[1])
        reg = np.eye(Z.shape[1]) * self.l2
        reg[0, 0] = 0.0
        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + reg @ w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + reg
            step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.coef_ = w
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        Z = np.c_[np.ones(len(X)), (X - self.mu_) / self.sd_]
        return 1 / (1 + np.exp(-Z @ self.coef_))

    def to_dict(self, names: list[str]) -> dict:
        return {"features": names, "mu": self.mu_.tolist(), "sd": self.sd_.tolist(), "coef": self.coef_.tolist(),
                "l2": self.l2}

    @classmethod
    def from_dict(cls, d: dict) -> LogisticFusion:
        m = cls(l2=d.get("l2", 1.0))
        m.mu_, m.sd_, m.coef_ = np.array(d["mu"]), np.array(d["sd"]), np.array(d["coef"])
        m.features = d["features"]
        return m

    def weights(self, names: list[str]) -> dict[str, float]:
        """Weights on the standardised scale (comparable across features)."""
        return {"bias": float(self.coef_[0]), **{n: float(v) for n, v in zip(names, self.coef_[1:])}}


def out_of_fold(X, y, groups, n_folds: int = 5, l2: float = 1.0, seed: int = 0) -> np.ndarray:
    """5-fold cross-validated risk predictions; folds are split by ``groups`` (question id) to avoid leakage."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    folds = np.array([fold_of[g] for g in groups])
    out = np.zeros(len(y))
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(set(y[tr])) < 2:
            out[te] = y[tr].mean() if tr.any() else 0.5
            continue
        out[te] = LogisticFusion(l2=l2).fit(X[tr], y[tr]).predict_proba(X[te])
    return out
