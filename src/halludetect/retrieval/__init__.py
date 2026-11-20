"""Sparse retrieval: Okapi BM25 (Robertson & Zaragoza, 2009) and reciprocal-rank fusion."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..text import content_tokens


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    rank: int


class BM25:
    """In-memory Okapi BM25 index with an inverted file.

    Parameters follow the common defaults (k1 = 1.5, b = 0.75). ``tokenizer``
    defaults to stemmed content words, so "burneth" in a 1922 translation
    matches "burn" in a modern question.
    """

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75,
                 tokenizer: Callable[[str], list[str]] | None = None):
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or content_tokens
        self.doc_ids = list(docs)
        self.doc_len: dict[str, int] = {}
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for did, text in docs.items():
            tf = Counter(self.tokenizer(text))
            self.doc_len[did] = sum(tf.values())
            for tok, n in tf.items():
                self.postings[tok].append((did, n))
        self.n_docs = max(len(self.doc_ids), 1)
        self.avgdl = sum(self.doc_len.values()) / self.n_docs if self.doc_len else 0.0
        self.idf = {
            tok: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for tok, p in self.postings.items()
        }

    def scores(self, query: str) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for tok in set(self.tokenizer(query)):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for did, tf in self.postings[tok]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[did] / (self.avgdl or 1))
                out[did] += idf * tf * (self.k1 + 1) / denom
        return out

    def search(self, query: str, k: int = 5) -> list[Hit]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(did, float(s), i + 1) for i, (did, s) in enumerate(ranked)]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = 60, top: int = 5) -> list[Hit]:
    """Fuse several ranked lists: score(d) = Σ_r 1 / (k + rank_r(d)) (Cormack et al., 2009)."""
    fused: dict[str, float] = defaultdict(float)
    for hits in rankings:
        for h in hits:
            fused[h.doc_id] += 1.0 / (k + h.rank)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [Hit(did, float(s), i + 1) for i, (did, s) in enumerate(ranked)]


__all__ = ["BM25", "Hit", "reciprocal_rank_fusion"]
