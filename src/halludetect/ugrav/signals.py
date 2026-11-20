"""Per-question signal extraction for UG-RAV (closed-book and retrieval-augmented)."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from ..llm.base import LLM, Generation
from ..nli.base import NLIModel
from ..text import answer_to_statement, contains_match
from ..uncertainty import (
    SemanticClusters,
    cluster_answers,
    length_normalised_predictive_entropy,
    lexical_entropy,
    predictive_entropy,
    self_consistency,
    semantic_entropy,
)


@dataclass
class ClosedBookSignals:
    greedy: str
    samples: list[str]
    clusters: list[int]
    se: float
    se_weighted: float | None
    pe: float | None
    ln_pe: float | None
    lexical_entropy: float
    self_consistency: float
    greedy_agreement: float
    n_clusters: int
    p_true: float | None
    min_token_prob: float | None
    nli_calls: int = 0


@dataclass
class RAGSignals:
    passage_ids: list[str]
    passages: list[str]
    greedy: str
    samples: list[str]
    clusters: list[int]
    se_rag: float
    self_consistency_rag: float
    delta_se: float
    support_rag: float
    contradiction_rag: float
    support_closed: float
    contradiction_closed: float
    agreement: bool
    nli_calls: int = 0


@dataclass
class QuestionRecord:
    id: str
    question: str
    aliases: list[str]
    meta: dict = field(default_factory=dict)
    closed: ClosedBookSignals | None = None
    rag: RAGSignals | None = None
    closed_correct: bool | None = None
    rag_correct: bool | None = None
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> QuestionRecord:
        rec = cls(d["id"], d["question"], d["aliases"], d.get("meta", {}))
        if d.get("closed"):
            rec.closed = ClosedBookSignals(**d["closed"])
        if d.get("rag"):
            rec.rag = RAGSignals(**d["rag"])
        rec.closed_correct, rec.rag_correct = d.get("closed_correct"), d.get("rag_correct")
        rec.seconds = d.get("seconds", 0)
        return rec


def _logprob_or_none(gens: list[Generation], fn):
    if not all(g.token_logprobs for g in gens):
        return None
    v = fn(gens)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def closed_book_signals(question: str, llm: LLM, nli: NLIModel, k: int = 10, temperature: float = 1.0,
                        use_p_true: bool = True) -> tuple[ClosedBookSignals, SemanticClusters]:
    greedy = llm.answer(question, n=1, temperature=0.0)[0]
    samples = llm.answer(question, n=k, temperature=temperature)
    texts = [s.text for s in samples]
    cl = cluster_answers(texts, nli, question)
    se = semantic_entropy(cl)
    se_w = _logprob_or_none(samples, lambda g: semantic_entropy(cl, [x.mean_logprob for x in g]))
    pe = _logprob_or_none(samples, lambda g: predictive_entropy([x.seq_logprob for x in g]))
    ln_pe = _logprob_or_none(samples, lambda g: length_normalised_predictive_entropy([x.token_logprobs for x in g]))
    from ..text import normalize_answer

    g_norm = normalize_answer(greedy.text)
    rep_idx = None
    for i, t in enumerate(texts):
        if normalize_answer(t) == g_norm:
            rep_idx = i
            break
    calls = cl.nli_calls
    if rep_idx is not None:
        g_cluster = cl.assignments[rep_idx]
    else:
        g_cluster = -1
        for i, t in enumerate(cl.representatives):
            calls += 1
            if nli.equivalent(t, greedy.text, context=question, strict=getattr(nli, "default_strict", True)):
                g_cluster = i
                break
    agreement = sum(1 for c in cl.assignments if c == g_cluster) / max(1, len(texts))
    p_true = llm.p_true(question, greedy.text, texts) if use_p_true else None
    mtp = greedy.min_token_prob if greedy.token_logprobs else None
    sig = ClosedBookSignals(greedy.text, texts, cl.assignments, se, se_w, pe, ln_pe, lexical_entropy(texts),
                            self_consistency(cl), agreement, cl.n_clusters, p_true, mtp, calls)
    return sig, cl


def evidence_support(question: str, answer: str, passages: list[str], nli: NLIModel) -> tuple[float, float]:
    """Sup(a) = max_i P_NLI(entailment | d_i, q ⊕ a) and the matching max contradiction."""
    if not passages or not answer.strip():
        return 0.0, 0.0
    hyp = answer_to_statement(question, answer)
    res = nli.predict_batch([(p, hyp) for p in passages])
    return max(r.entailment for r in res), max(r.contradiction for r in res)


def rag_signals(question: str, closed: ClosedBookSignals, closed_clusters: SemanticClusters, llm: LLM, nli: NLIModel,
                retriever, k: int = 10, temperature: float = 1.0, top_k: int = 3) -> RAGSignals:
    hits = retriever.search(question, k=top_k)
    ids = [h[0] for h in hits]
    passages = [h[1] for h in hits]
    greedy = llm.answer(question, passages=passages, n=1, temperature=0.0)[0]
    samples = llm.answer(question, passages=passages, n=k, temperature=temperature)
    texts = [s.text for s in samples]
    cl = cluster_answers(texts, nli, question)
    se_rag = semantic_entropy(cl)
    sup_r, con_r = evidence_support(question, greedy.text, passages, nli)
    sup_c, con_c = evidence_support(question, closed.greedy, passages, nli)
    maj_closed = closed_clusters.majority_answer()
    maj_rag = cl.majority_answer()
    agree = bool(maj_closed and maj_rag and nli.equivalent(maj_closed, maj_rag, context=question,
                                                          strict=getattr(nli, "default_strict", True)))
    calls = cl.nli_calls + 2 * len(passages) + 2
    return RAGSignals(ids, passages, greedy.text, texts, cl.assignments, se_rag, self_consistency(cl),
                      closed.se - se_rag, sup_r, con_r, sup_c, con_c, agree, calls)


def build_record(qid: str, question: str, aliases: list[str], llm: LLM, nli: NLIModel, retriever, k: int = 10,
                 temperature: float = 1.0, top_k: int = 3, meta: dict | None = None, use_p_true: bool = True
                 ) -> QuestionRecord:
    t = time.time()
    closed, cl = closed_book_signals(question, llm, nli, k=k, temperature=temperature, use_p_true=use_p_true)
    rag = rag_signals(question, closed, cl, llm, nli, retriever, k=k, temperature=temperature, top_k=top_k)
    rec = QuestionRecord(qid, question, aliases, meta or {}, closed, rag)
    rec.closed_correct = contains_match(closed.greedy, aliases)
    rec.rag_correct = contains_match(rag.greedy, aliases)
    rec.seconds = round(time.time() - t, 3)
    return rec
