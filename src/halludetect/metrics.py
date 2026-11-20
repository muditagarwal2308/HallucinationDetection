"""Evaluation metrics for hallucination detectors and selective-prediction policies.

Conventions
-----------
* ``scores``: detector output, **higher = more likely hallucinated** (risk).
* ``labels``: 1 if the answer is a hallucination / incorrect, else 0.
* ``correct``: 1 if the answer is correct (= 1 - label).

All functions are implemented with NumPy only so the metrics are transparent and
testable (see ``tests/test_metrics.py`` for the worked examples from the notes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def auroc(scores, labels) -> float:
    """Area under the ROC curve via the Mann–Whitney U statistic (ties count 1/2).

    Equals the probability that a randomly chosen hallucinated answer receives a
    higher risk score than a randomly chosen correct one.
    """
    s, y = _arr(scores), _arr(labels).astype(int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i: j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_of = np.empty(len(allv))
    rank_of[order] = ranks
    r_pos = rank_of[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def roc_curve(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) with thresholds in decreasing order."""
    s, y = _arr(scores), _arr(labels).astype(int)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, len(s) - 1]
    tps = np.cumsum(y)[idx]
    fps = (idx + 1) - tps
    P, N = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / N]
    thr = np.r_[np.inf, s[idx]]
    return fpr, tpr, thr


def precision_recall_curve(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s, y = _arr(scores), _arr(labels).astype(int)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    precision = tp / k
    recall = tp / max(y.sum(), 1)
    return np.r_[1.0, precision], np.r_[0.0, recall]


def auprc(scores, labels) -> float:
    """Average precision (step-wise area under the precision–recall curve) for the hallucination class."""
    s, y = _arr(scores), _arr(labels).astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def ece(confidences, correct, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins over [0, 1].

    ECE = Σ_m |B_m|/n · |acc(B_m) − conf(B_m)|; 0 means perfectly calibrated.
    """
    c, a = _arr(confidences), _arr(correct)
    if len(c) == 0:
        return float("nan")
    bins = np.clip((c * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(a[m].mean() - c[m].mean())
    return float(total)


def reliability_diagram(confidences, correct, n_bins: int = 10) -> dict[str, list[float]]:
    c, a = _arr(confidences), _arr(correct)
    bins = np.clip((c * n_bins).astype(int), 0, n_bins - 1)
    out = {"bin_center": [], "accuracy": [], "confidence": [], "count": []}
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out["bin_center"].append((b + 0.5) / n_bins)
            out["accuracy"].append(float(a[m].mean()))
            out["confidence"].append(float(c[m].mean()))
            out["count"].append(int(m.sum()))
    return out


def risk_coverage_curve(risk_scores, correct) -> tuple[np.ndarray, np.ndarray]:
    """Answer the most confident questions first; return (coverage, selective risk).

    At coverage c the system answers the ⌈c·n⌉ lowest-risk questions and abstains
    on the rest; selective risk is the error rate among the answered questions.
    """
    r, a = _arr(risk_scores), _arr(correct)
    order = np.argsort(r, kind="mergesort")
    a = a[order]
    k = np.arange(1, len(a) + 1)
    coverage = k / len(a)
    risk = 1.0 - np.cumsum(a) / k
    return coverage, risk


def aurc(risk_scores, correct) -> float:
    """Area under the risk–coverage curve (lower is better)."""
    _, risk = risk_coverage_curve(risk_scores, correct)
    return float(risk.mean())


def aurac(risk_scores, correct) -> float:
    """Area under the rejection–accuracy curve (higher is better).

    Accuracy of the retained answers averaged over all rejection rates; used in
    the semantic-entropy papers (Kuhn et al., 2023; Farquhar et al., 2024).
    """
    return 1.0 - aurc(risk_scores, correct)


def accuracy_at_coverage(risk_scores, correct, coverage: float) -> float:
    r, a = _arr(risk_scores), _arr(correct)
    n = max(1, int(np.ceil(coverage * len(a))))
    order = np.argsort(r, kind="mergesort")[:n]
    return float(a[order].mean())


def oracle_aurac(correct) -> float:
    """AURAC of a perfect detector (all errors rejected first) — an upper bound."""
    a = np.sort(_arr(correct))[::-1]
    return aurac(-a, a)


@dataclass
class DetectionReport:
    auroc: float
    auprc: float
    ece: float
    aurac: float
    acc_at_80: float
    acc_at_90: float
    base_accuracy: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def detection_report(risk_scores, correct, confidence=None) -> DetectionReport:
    """Bundle the detection and selective-prediction metrics used in the reports.

    ``confidence`` (probability of being correct) is needed for ECE; if omitted
    it is derived as ``1 - risk`` when the risk is already a probability.
    """
    r, a = _arr(risk_scores), _arr(correct)
    if confidence is None:
        conf = 1.0 - r if (r.min() >= 0 and r.max() <= 1) else None
    else:
        conf = _arr(confidence)
    return DetectionReport(
        auroc=auroc(r, 1 - a),
        auprc=auprc(r, 1 - a),
        ece=ece(conf, a) if conf is not None else float("nan"),
        aurac=aurac(r, a),
        acc_at_80=accuracy_at_coverage(r, a, 0.8),
        acc_at_90=accuracy_at_coverage(r, a, 0.9),
        base_accuracy=float(a.mean()),
        n=int(len(a)),
    )


def bootstrap_ci(metric, *arrays, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``metric(*arrays)``."""
    rng = np.random.default_rng(seed)
    arrs = [_arr(x) for x in arrays]
    n = len(arrs[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric(*[x[idx] for x in arrs])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def precision_recall_f1(pred, gold) -> tuple[float, float, float]:
    p, g = _arr(pred).astype(bool), _arr(gold).astype(bool)
    tp = float((p & g).sum())
    prec = tp / p.sum() if p.sum() else 0.0
    rec = tp / g.sum() if g.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
