"""Offline answer generator for Gita RAG demos, with controllable hallucination injection."""

from __future__ import annotations

import random
import re

from ..llm.base import LLM, Generation
from . import perturb
from .corpus import GitaCorpus, Verse

MODES = ["none", "fabricated_citation", "misattributed_citation", "misquotation", "wrong_speaker",
         "unsupported_claim", "random"]

_ARCHAIC = [
    (r"\bthou art\b", "you are"), (r"\bthou\b", "you"), (r"\bthee\b", "you"), (r"\bthy\b", "your"),
    (r"\bthine\b", "your"), (r"\bhath\b", "has"), (r"\bdoth\b", "does"), (r"\bdost\b", "do"), (r"\bart\b", "are"),
    (r"\bshalt\b", "shall"), (r"\bwilt\b", "will"), (r"\bye\b", "you"), (r"\bunto\b", "to"),
]


def _third_person(stem: str) -> str:
    """burn → burns, dri → dries, perish → perishes, tak → takes, becom → becomes."""
    low = stem.lower()
    if low.endswith("i"):
        return stem + "es"
    if low.endswith(("sh", "ch", "x", "s", "z", "v", "c", "g")):
        return stem + "es"
    if len(low) >= 3 and low[-1] in "kmd" and low[-2] in "aeiou" and low[-3] not in "aeiou":
        return stem + "es"
    return stem + "s"


def _eth(m: re.Match) -> str:
    return _third_person(m.group(1))


def modernise(text: str) -> str:
    out = text
    for pat, rep in _ARCHAIC:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return re.sub(r"\b(\w{3,}?)eth\b", _eth, out)


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


class SimulatedGitaLLM(LLM):
    """Writes a cited answer from the retrieved verses, then optionally injects one hallucination type.

    It deliberately answers *every* question (it never refuses), so false-premise
    and out-of-scope questions show how VGHD catches premise acceptance.
    """

    name = "simulator:gita"

    def __init__(self, corpus: GitaCorpus, mode: str = "none", seed: int = 0):
        self.corpus, self.mode, self.seed = corpus, mode, seed

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0, max_tokens: int = 200,
                 system: str | None = None) -> list[Generation]:
        raise NotImplementedError("Use compose().")

    def compose(self, question: str, verses: list[Verse], mode: str | None = None) -> tuple[str, str]:
        mode = mode or self.mode
        rng = random.Random(f"{self.seed}|{question}|{mode}")
        if not verses:
            return "I could not find relevant verses.", "none"
        v = verses[0]
        body = modernise(v.english).rstrip(".;,")
        if v.speaker == "Krishna":
            answer = f"Krishna says that {_lower_first(body)} (BG {v.ref})."
        elif v.speaker in ("Arjuna", "Sanjaya", "Dhritarashtra"):
            answer = f"{v.speaker} says, \"{v.english.rstrip('.;,')}\" (BG {v.ref})."
        else:
            answer = f"The Gita says that {_lower_first(body)} (BG {v.ref})."
        if len(verses) > 1 and verses[1].chapter == v.chapter and abs(verses[1].verse - v.verse) == 1:
            nxt = verses[1]
            answer += f" It continues: {modernise(nxt.english).rstrip('.;,')} (BG {nxt.ref})."
        if mode == "random":
            mode = rng.choice(MODES[1:-1])
        injected = mode
        if mode == "fabricated_citation":
            answer = perturb.fabricate_citation(answer, self.corpus, rng) or answer
        elif mode == "misattributed_citation":
            answer = perturb.misattribute_citation(answer, self.corpus, rng) or answer
        elif mode == "misquotation":
            answer = perturb.misquote(answer, self.corpus, rng, v.ref) or answer
        elif mode == "wrong_speaker":
            answer = perturb.wrong_speaker(answer, v.speaker, rng) or answer
        elif mode == "unsupported_claim":
            answer = perturb.unsupported_claim(answer, rng, v.ref)
        else:
            injected = "none"
        return answer, injected
