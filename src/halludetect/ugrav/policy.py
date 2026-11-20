"""UG-RAV decision policy (Algorithm 1) and the baselines it is compared with.

Every policy maps each question to one *decision* — answer from the closed-book
model, answer from retrieved evidence, or abstain — plus a risk score used to
draw a risk–coverage curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import metrics


@dataclass
class PolicyArrays:
    """Column view of the cached per-question records used by all policies."""

    closed_correct: np.ndarray
    rag_correct: np.ndarray
    se: np.ndarray
    se_rag: np.ndarray
    sc: np.ndarray
    min_tok: np.ndarray
    support_rag: np.ndarray
    r0: np.ndarray        # fused closed-book risk (out-of-fold)
    r_rag: np.ndarray     # fused risk of the evidence-grounded answer (out-of-fold)

    @classmethod
    def from_records(cls, records, r0, r_rag) -> PolicyArrays:
        mt = [r.closed.min_token_prob if r.closed.min_token_prob is not None else np.nan for r in records]
        return cls(
            closed_correct=np.array([r.closed_correct for r in records], float),
            rag_correct=np.array([r.rag_correct for r in records], float),
            se=np.array([r.closed.se for r in records]), se_rag=np.array([r.rag.se_rag for r in records]),
            sc=np.array([r.closed.self_consistency for r in records]), min_tok=np.array(mt, float),
            support_rag=np.array([r.rag.support_rag for r in records]), r0=np.asarray(r0), r_rag=np.asarray(r_rag),
        )

    def subset(self, idx) -> PolicyArrays:
        return PolicyArrays(**{k: v[idx] for k, v in self.__dict__.items()})


@dataclass
class Decisions:
    """source: 0 = closed-book answer, 1 = evidence-grounded answer, -1 = abstain."""

    name: str
    source: np.ndarray
    risk: np.ndarray
    retrieved: np.ndarray

    def correct(self, a: PolicyArrays) -> np.ndarray:
        return np.where(self.source == 1, a.rag_correct, a.closed_correct)

    def summary(self, a: PolicyArrays) -> dict:
        answered = self.source >= 0
        corr = self.correct(a)
        n = len(self.source)
        n_ans = answered.sum()
        right = float((corr * answered).sum())
        wrong = float(((1 - corr) * answered).sum())
        curve_correct = corr  # risk ordering over the routed answer of every question
        return {
            "coverage": float(n_ans / n),
            "accuracy": right / n,
            "selective_accuracy": right / n_ans if n_ans else float("nan"),
            "hallucination_rate": wrong / n,
            "utility": (right - wrong) / n,
            "retrieval_rate": float(self.retrieved.mean()),
            "aurac": metrics.aurac(self.risk, curve_correct),
            "acc_at_80": metrics.accuracy_at_coverage(self.risk, curve_correct, 0.8),
            "acc_at_90": metrics.accuracy_at_coverage(self.risk, curve_correct, 0.9),
        }


# ------------------------------------------------------------------ policies
def never_retrieve(a: PolicyArrays) -> Decisions:
    n = len(a.se)
    return Decisions("never_retrieve", np.zeros(n, int), a.se, np.zeros(n, bool))


def always_retrieve(a: PolicyArrays) -> Decisions:
    n = len(a.se)
    return Decisions("always_retrieve", np.ones(n, int), a.se_rag, np.ones(n, bool))


def sc_abstention(a: PolicyArrays, theta: float) -> Decisions:
    src = np.where(a.sc >= theta, 0, -1)
    return Decisions("sc_abstention", src, 1 - a.sc, np.zeros(len(src), bool))


def flare_gate(a: PolicyArrays, theta: float) -> Decisions:
    mt = np.nan_to_num(a.min_tok, nan=1.0)
    retrieve = mt < theta
    risk = np.where(retrieve, a.se_rag, a.se)
    return Decisions("flare_style", retrieve.astype(int), risk, retrieve)


def se_gate(a: PolicyArrays, tau: float) -> Decisions:
    retrieve = a.se > tau
    risk = np.where(retrieve, a.se_rag, a.se)
    return Decisions("se_gated_no_verification", retrieve.astype(int), risk, retrieve)


def ugrav(a: PolicyArrays, tau_low: float, tau_high: float, min_support: float = 0.5, name: str = "ug_rav",
          r_rag: np.ndarray | None = None) -> Decisions:
    """Algorithm 1: accept a0 if r̂0 < τ_low; else retrieve and answer from evidence if r̂ < τ_high and entailed."""
    rr = a.r_rag if r_rag is None else r_rag
    accept = a.r0 < tau_low
    grounded = (~accept) & (rr < tau_high) & (a.support_rag >= min_support)
    src = np.where(accept, 0, np.where(grounded, 1, -1))
    risk = np.where(accept, a.r0, rr)
    return Decisions(name, src, risk, ~accept)


def oracle(a: PolicyArrays) -> Decisions:
    src = np.where(a.closed_correct == 1, 0, np.where(a.rag_correct == 1, 1, -1))
    risk = np.where(src >= 0, 0.0, 1.0)
    return Decisions("oracle", src, risk, src == 1)


# -------------------------------------------------------------------- tuning
GRID = np.round(np.linspace(0.05, 0.95, 19), 3)


def tune_ugrav(a: PolicyArrays, r_rag: np.ndarray | None = None, retrieval_cost: float = 0.05) -> dict:
    """Grid-search τ_low, τ_high and the support floor maximising (correct − wrong)/n − cost · retrieval rate."""
    best = None
    for lo in GRID:
        for hi in GRID:
            for sup in (0.0, 0.3, 0.5):
                d = ugrav(a, lo, hi, sup, r_rag=r_rag)
                u = d.summary(a)["utility"] - retrieval_cost * float(d.retrieved.mean())
                key = (round(u, 6), -d.retrieved.mean())
                if best is None or key > best[0]:
                    best = (key, {"tau_low": float(lo), "tau_high": float(hi), "min_support": sup})
    return best[1]


def tune_sc(a: PolicyArrays) -> float:
    grid = np.linspace(0.0, 1.0, 21)
    return float(max(grid, key=lambda t: (sc_abstention(a, t).summary(a)["utility"], -t)))


def match_rate(values: np.ndarray, rate: float, retrieve_if_below: bool) -> float:
    """Threshold that retrieves for the requested fraction of questions."""
    v = np.nan_to_num(values, nan=1.0 if retrieve_if_below else 0.0)
    if rate <= 0:
        return -np.inf if retrieve_if_below else np.inf
    if rate >= 1:
        return np.inf if retrieve_if_below else -np.inf
    q = np.quantile(v, rate if retrieve_if_below else 1 - rate)
    return float(q + 1e-12) if retrieve_if_below else float(q - 1e-12)


def gate_budget_curve(a_dev: PolicyArrays, a_test: PolicyArrays, gates: dict[str, tuple[np.ndarray, np.ndarray]],
                      budgets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    """Accuracy of "retrieve if the gate fires, otherwise answer closed-book" at matched retrieval budgets.

    ``gates`` maps a name to (dev_scores, test_scores) where a *higher* score means "retrieve".
    The threshold for each budget is the dev-split quantile, so all gates spend the same budget.
    """
    out: dict[str, list[dict]] = {}
    for name, (s_dev, s_test) in gates.items():
        rows = []
        for b in budgets:
            if b <= 0:
                thr = np.inf
            elif b >= 1:
                thr = -np.inf
            else:
                thr = float(np.quantile(s_dev, 1 - b))
            retrieve = s_test > thr if np.isfinite(thr) else np.full(len(s_test), thr < 0)
            corr = np.where(retrieve, a_test.rag_correct, a_test.closed_correct)
            rows.append({"budget": b, "retrieval_rate": float(retrieve.mean()), "accuracy": float(corr.mean())})
        out[name] = rows
    return out
