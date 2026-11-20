"""Loading the open-domain QA world (questions + passage corpus)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import ODQA_DATA
from ..retrieval import BM25


@dataclass
class WorldItem:
    id: str
    question: str
    answer: str
    aliases: list[str]
    group: str
    answer_type: str
    tier: str
    popularity: int
    kind: str = "standard"
    misconception: str | None = None
    distractors: list[str] = field(default_factory=list)
    passage: str = ""
    in_corpus: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> WorldItem:
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


def load_world(path: Path | None = None) -> list[WorldItem]:
    p = path or ODQA_DATA / "world.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    return [WorldItem.from_dict(json.loads(line)) for line in lines if line.strip()]


def load_passages(path: Path | None = None) -> dict[str, str]:
    p = path or ODQA_DATA / "passages.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r["text"] for r in rows}


class PassageRetriever:
    """BM25 over a passage collection; returns ``(passage_id, text, score)``."""

    def __init__(self, passages: dict[str, str]):
        self.passages = passages
        self.index = BM25(passages)

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        return [(h.doc_id, self.passages[h.doc_id], h.score) for h in self.index.search(query, k=k)]
