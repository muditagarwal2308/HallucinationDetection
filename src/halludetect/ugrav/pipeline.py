"""Single-question UG-RAV pipeline with a full, human-readable trace (used by the app and CLI)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..llm.base import LLM
from ..nli.base import NLIModel
from ..paths import RESULTS
from ..text import contains_match
from .fusion import LogisticFusion, feature_row
from .signals import QuestionRecord, closed_book_signals, rag_signals

DEFAULT_THRESHOLDS = {"tau_low": 0.3, "tau_high": 0.5, "min_support": 0.5}


class UGRAV:
    """Uncertainty-Gated Retrieval-Augmented Verification for one question at a time.

    Parameters
    ----------
    llm, nli, retriever : the generator, the NLI model and a retriever with ``search(query, k)``.
    k, temperature : number of samples and sampling temperature for semantic entropy.
    fusion_path : JSON produced by the benchmark (``results/ugrav/fusion_model.json``) holding the
        fitted logistic-regression weights and tuned thresholds. Without it a transparent
        fallback risk based on semantic entropy and evidence support is used.
    """

    def __init__(self, llm: LLM, nli: NLIModel, retriever, k: int = 10, temperature: float = 1.0, top_k: int = 3,
                 fusion_path: Path | None = None, thresholds: dict | None = None):
        self.llm, self.nli, self.retriever = llm, nli, retriever
        self.k, self.temperature, self.top_k = k, temperature, top_k
        self.models: dict[str, LogisticFusion] = {}
        path = fusion_path or RESULTS / "ugrav" / "fusion_model.json"
        th = dict(DEFAULT_THRESHOLDS)
        if Path(path).exists():
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.models = {k_: LogisticFusion.from_dict(v) for k_, v in payload["models"].items()}
            th.update(payload.get("thresholds", {}))
        th.update(thresholds or {})
        self.thresholds = th

    def _risk(self, rec: QuestionRecord, kind: str) -> tuple[float, str]:
        for key in (kind, f"{kind}_blackbox"):
            m = self.models.get(key)
            if m is None:
                continue
            row = feature_row(rec, m.features)
            if not np.isnan(row).any():
                return float(m.predict_proba([row])[0]), key
        c = rec.closed
        if kind == "closed":
            z = 2.2 * c.se - 3.0 * (c.self_consistency - 0.5)
        else:
            r = rec.rag
            z = 1.5 * r.se_rag - 3.0 * (r.support_rag - 0.5) + 1.5 * r.contradiction_rag - 0.5 * float(r.agreement)
        return float(1 / (1 + np.exp(-z))), "heuristic"

    def run(self, question: str, aliases: list[str] | None = None) -> dict:
        t0 = time.time()
        steps = []
        closed, clusters = closed_book_signals(question, self.llm, self.nli, k=self.k, temperature=self.temperature)
        rec = QuestionRecord("live", question, aliases or [], {}, closed, None)
        r0, model0 = self._risk(rec, "closed")
        steps.append({"step": "closed_book", "greedy": closed.greedy, "semantic_entropy": closed.se,
                      "self_consistency": closed.self_consistency, "p_true": closed.p_true,
                      "risk_r0": r0, "fusion_model": model0})
        th = self.thresholds
        decision = {"retrieved": False}
        if r0 < th["tau_low"]:
            decision.update(action="accept", answer=closed.greedy,
                            rationale=f"Closed-book risk r̂0 = {r0:.2f} < τ_low = {th['tau_low']:.2f}: the model is "
                                      "consistent, so no retrieval is needed.")
            rag = None
        else:
            rag = rag_signals(question, closed, clusters, self.llm, self.nli, self.retriever, k=self.k,
                              temperature=self.temperature, top_k=self.top_k)
            rec.rag = rag
            r_hat, model1 = self._risk(rec, "rag")
            steps.append({"step": "retrieval_and_verification", "passages": rag.passages, "rag_answer": rag.greedy,
                          "semantic_entropy_rag": rag.se_rag, "delta_se": rag.delta_se,
                          "support": rag.support_rag, "contradiction": rag.contradiction_rag,
                          "agreement": rag.agreement, "risk_r_hat": r_hat, "fusion_model": model1})
            decision["retrieved"] = True
            if r_hat < th["tau_high"] and rag.support_rag >= th["min_support"]:
                decision.update(action="answer_from_evidence", answer=rag.greedy,
                                rationale=f"Fused risk r̂ = {r_hat:.2f} < τ_high = {th['tau_high']:.2f} and the "
                                          f"evidence entails the answer (support {rag.support_rag:.2f}).")
            else:
                why = []
                if r_hat >= th["tau_high"]:
                    why.append(f"fused risk r̂ = {r_hat:.2f} ≥ τ_high = {th['tau_high']:.2f}")
                if rag.support_rag < th["min_support"]:
                    why.append(f"evidence support {rag.support_rag:.2f} < {th['min_support']:.2f}")
                decision.update(action="abstain", answer="I don't know.",
                                rationale="Neither the model nor the evidence can be trusted: " + "; ".join(why) + ".")
        if aliases:
            decision["closed_correct"] = contains_match(closed.greedy, aliases)
            if rag is not None:
                decision["rag_correct"] = contains_match(rag.greedy, aliases)
            if decision["action"] != "abstain":
                decision["final_correct"] = contains_match(decision["answer"], aliases)
        return {"question": question, "decision": decision, "steps": steps, "closed": asdict(closed),
                "rag": asdict(rag) if rag else None, "thresholds": th, "seconds": round(time.time() - t0, 3)}
