"""Abstract LLM interface used by both frameworks."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from . import prompts


@dataclass
class Generation:
    """One generated answer with (optional) per-token log-probabilities."""

    text: str
    token_logprobs: list[float] = field(default_factory=list)

    @property
    def seq_logprob(self) -> float:
        return float(sum(self.token_logprobs)) if self.token_logprobs else float("nan")

    @property
    def mean_logprob(self) -> float:
        return float(sum(self.token_logprobs) / len(self.token_logprobs)) if self.token_logprobs else float("nan")

    @property
    def min_token_prob(self) -> float:
        return float(math.exp(min(self.token_logprobs))) if self.token_logprobs else float("nan")


class LLM(ABC):
    """Minimal text-generation interface.

    Sub-classes implement :meth:`generate`; :meth:`answer` and :meth:`p_true`
    build the prompts used in the dissertation on top of it.
    """

    name: str = "llm"
    supports_logprobs: bool = False

    @abstractmethod
    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0, max_tokens: int = 48,
                 system: str | None = None) -> list[Generation]:
        """Return ``n`` completions of ``prompt``."""

    def answer(self, question: str, passages: list[str] | None = None, n: int = 1,
               temperature: float = 0.0, max_tokens: int = 24) -> list[Generation]:
        """Short-answer QA, closed-book (``passages=None``) or retrieval-augmented."""
        if passages:
            prompt = prompts.rag_prompt(question, passages)
        else:
            prompt = prompts.closed_book_prompt(question)
        gens = self.generate(prompt, n=n, temperature=temperature, max_tokens=max_tokens, system=prompts.QA_SYSTEM)
        return [Generation(prompts.clean_answer(g.text), g.token_logprobs) for g in gens]

    def p_true(self, question: str, answer: str, samples: list[str] | None = None) -> float | None:
        """P(True) self-evaluation (Kadavath et al., 2022); ``None`` if unsupported."""
        return None

    def describe(self) -> str:
        return self.name
