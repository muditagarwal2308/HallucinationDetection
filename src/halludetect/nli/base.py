"""NLI interface shared by all back-ends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        best = max(
            ("entailment", self.entailment), ("neutral", self.neutral), ("contradiction", self.contradiction),
            key=lambda kv: kv[1],
        )
        return best[0]

    def as_dict(self) -> dict[str, float]:
        return {"entailment": round(self.entailment, 4), "neutral": round(self.neutral, 4),
                "contradiction": round(self.contradiction, 4), "label": self.label}


class NLIModel(ABC):
    """Premise/hypothesis classifier into entailment / neutral / contradiction."""

    name: str = "nli"
    #: Clustering criterion used when callers do not specify one: strict bidirectional
    #: entailment, or the relaxed "no contradiction, not both neutral" rule.
    default_strict: bool = True

    @abstractmethod
    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Classify a batch of ``(premise, hypothesis)`` pairs."""

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        return self.predict_batch([(premise, hypothesis)])[0]

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.predict(premise, hypothesis).label == "entailment"

    def equivalent(self, a: str, b: str, context: str = "", strict: bool = True) -> bool:
        """Bidirectional entailment test used for semantic clustering (Kuhn et al., 2023).

        The question is prepended as context so that short answers are judged in
        meaning ("Canberra" vs "The capital is Canberra"). With ``strict=False``
        the relaxed criterion of the reference implementation is used: no
        contradiction in either direction and not both neutral.
        """
        ctx = f"{context.strip()} " if context else ""
        r1, r2 = self.predict_batch([(ctx + a, ctx + b), (ctx + b, ctx + a)])
        if strict:
            return r1.label == "entailment" and r2.label == "entailment"
        labels = {r1.label, r2.label}
        return "contradiction" not in labels and labels != {"neutral"}
