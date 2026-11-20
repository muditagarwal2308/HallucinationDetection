"""Uncertainty measures for hallucination detection.

Implements the measures compared in the dissertation (Objective O1):

* semantic clustering by bidirectional entailment and **semantic entropy**
  (Kuhn et al., ICLR 2023; Farquhar et al., Nature 2024) — discrete
  (black-box) and probability-weighted (white-box) estimators;
* **predictive entropy** and its length-normalised variant (Malinin & Gales, 2021);
* **lexical entropy** over normalised strings (to illustrate the paraphrase problem);
* **self-consistency** (share of the largest meaning cluster) and agreement
  of the greedy answer with the samples;
* **min-token-probability** (the FLARE retrieval trigger).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..nli.base import NLIModel
from ..text import normalize_answer


@dataclass
class SemanticClusters:
    """Partition of sampled answers into meaning-equivalence classes."""

    answers: list[str]
    assignments: list[int]
    nli_calls: int = 0
    representatives: list[str] = field(default_factory=list)

    @property
    def clusters(self) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for i, c in enumerate(self.assignments):
            groups.setdefault(c, []).append(i)
        return [groups[k] for k in sorted(groups)]

    @property
    def sizes(self) -> list[int]:
        return [len(c) for c in self.clusters]

    @property
    def n_clusters(self) -> int:
        return len(set(self.assignments))

    def largest(self) -> list[int]:
        return max(self.clusters, key=len) if self.answers else []

    def majority_answer(self) -> str:
        idx = self.largest()
        return self.answers[idx[0]] if idx else ""

    def cluster_of(self, i: int) -> int:
        return self.assignments[i]


def cluster_answers(answers: list[str], nli: NLIModel, question: str = "", strict: bool | None = None
                    ) -> SemanticClusters:
    """Greedy bidirectional-entailment clustering (Algorithm 1 of Kuhn et al., 2023).

    Each answer is compared with the representative (first member) of every
    existing cluster and joins the first cluster it is equivalent to; otherwise
    it starts a new cluster. Answers that are identical after normalisation are
    merged without an NLI call, which removes most of the K(K−1)/2 comparisons
    in practice. ``strict=None`` uses the NLI back-end's default criterion.
    """
    if strict is None:
        strict = getattr(nli, "default_strict", True)
    assignments: list[int] = []
    reps: list[str] = []
    rep_norm: list[str] = []
    norm_to_cluster: dict[str, int] = {}
    calls = 0
    for ans in answers:
        key = normalize_answer(ans)
        if key in norm_to_cluster:
            assignments.append(norm_to_cluster[key])
            continue
        found = -1
        for cid, rep in enumerate(reps):
            calls += 1
            if nli.equivalent(rep, ans, context=question, strict=strict):
                found = cid
                break
        if found < 0:
            found = len(reps)
            reps.append(ans)
            rep_norm.append(key)
        norm_to_cluster[key] = found
        assignments.append(found)
    return SemanticClusters(list(answers), assignments, calls, reps)


def _entropy(probs) -> float:
    return max(0.0, float(-sum(p * math.log(p) for p in probs if p > 0)))  # avoid -0.0


def semantic_entropy(clusters: SemanticClusters, logprobs: list[float] | None = None) -> float:
    """Semantic entropy SE(x) = −Σ_c p(c|x) log p(c|x).

    * ``logprobs is None`` → **discrete** estimator, p(c|x) = |c| / K (black-box).
    * otherwise → **weighted** estimator: p(c|x) ∝ Σ_{s∈c} p(s|x), using the
      (preferably length-normalised) sequence log-probabilities of the samples.
    """
    if not clusters.answers:
        return 0.0
    if logprobs is None:
        k = len(clusters.answers)
        return _entropy([n / k for n in clusters.sizes])
    mx = max(logprobs)
    weights = [math.exp(lp - mx) for lp in logprobs]
    total = sum(weights)
    mass: dict[int, float] = {}
    for c, w in zip(clusters.assignments, weights):
        mass[c] = mass.get(c, 0.0) + w / total
    return _entropy(mass.values())


def lexical_entropy(answers: list[str]) -> float:
    """Entropy over normalised strings — overstates uncertainty when answers are paraphrases."""
    if not answers:
        return 0.0
    counts = Counter(normalize_answer(a) for a in answers)
    k = len(answers)
    return _entropy([n / k for n in counts.values()])


def predictive_entropy(sequence_logprobs: list[float]) -> float:
    """Monte-Carlo predictive entropy: −(1/K) Σ_i log p(s_i | x)."""
    if not sequence_logprobs:
        return float("nan")
    return float(-sum(sequence_logprobs) / len(sequence_logprobs))


def length_normalised_predictive_entropy(token_logprobs: list[list[float]]) -> float:
    """−(1/K) Σ_i (1/|s_i|) Σ_t log p(s_i,t | s_i,<t, x)."""
    vals = [sum(t) / len(t) for t in token_logprobs if t]
    if not vals:
        return float("nan")
    return float(-sum(vals) / len(vals))


def self_consistency(clusters: SemanticClusters) -> float:
    """Share of samples in the largest meaning cluster (1.0 = all samples agree)."""
    if not clusters.answers:
        return 0.0
    return max(clusters.sizes) / len(clusters.answers)


def greedy_agreement(greedy: str, samples: list[str], nli: NLIModel, question: str = "") -> float:
    """Fraction of samples that are meaning-equivalent to the greedy answer."""
    if not samples:
        return 0.0
    g = normalize_answer(greedy)
    agree = sum(1 for s in samples if normalize_answer(s) == g or nli.equivalent(greedy, s, context=question))
    return agree / len(samples)


def min_token_prob(token_logprobs: list[float]) -> float:
    """Lowest token probability in an answer (FLARE retrieves when it falls below θ)."""
    return float(math.exp(min(token_logprobs))) if token_logprobs else float("nan")


def max_entropy(k: int) -> float:
    """Upper bound ln K reached when all K samples have different meanings."""
    return math.log(k) if k > 0 else 0.0


__all__ = [
    "SemanticClusters", "cluster_answers", "semantic_entropy", "lexical_entropy", "predictive_entropy",
    "length_normalised_predictive_entropy", "self_consistency", "greedy_agreement", "min_token_prob",
    "max_entropy",
]
