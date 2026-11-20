"""A stochastic, mechanistic simulator of LLM question answering (closed-book and RAG).

Purpose: make the full UG-RAV pipeline runnable offline and reproducibly, with
behaviour that reproduces the phenomena the dissertation targets — **not** to
stand in for real model results (see ``scripts/run_real_experiment.py``).

Behaviour model (per question, fixed by ``seed``):

* **Knowledge** — the probability that the model knows a fact grows with entity
  popularity, p_know = σ(slope · (log10 popularity − mid)), mirroring the
  long-tail effect reported for PopQA (Mallen et al., 2023).
* **Confident errors** — if it does not know, it may have firmly memorised a
  wrong answer (more likely when a common misconception exists, cf. TruthfulQA):
  samples then agree and semantic entropy stays low — the blind spot of
  uncertainty-only detection.
* **Confabulation** — otherwise probability mass is spread over plausible
  distractors (high semantic entropy).
* **Ambiguity** — questions with many valid answers spread mass over correct
  answers (high entropy without error: aleatoric false alarms).
* **Retrieval** — the model *reads* the passages it is given: if the gold
  answer appears it usually adopts it (but may stubbornly keep a confident
  wrong belief — knowledge conflict); if only a different candidate appears it
  may be misled (external hallucination); otherwise it becomes less certain.
* **Surface forms** — answers are paraphrased ("Premchand", "It was Munshi
  Premchand.") so lexical and semantic uncertainty differ, and token
  log-probabilities are emitted so predictive entropy, P(True) and FLARE-style
  triggers can be computed.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from ..llm.base import LLM, Generation
from ..text import contains_match, normalize_answer
from .data import WorldItem

TEMPLATES = [("{a}", 0.45), ("{a}.", 0.2), ("It is {a}.", 0.12), ("The answer is {a}.", 0.13), ("It was {a}.", 0.1)]
YEAR_TEMPLATES = [("{a}", 0.55), ("In {a}.", 0.3), ("The year was {a}.", 0.15)]


@dataclass
class Profile:
    name: str = "sim-7b"
    slope: float = 1.6
    mid: float = 3.4
    p_confident_wrong_misconception: float = 0.6
    p_confident_wrong_other: float = 0.2
    p_read_gold: float = 0.92
    p_stubborn: float = 0.35
    p_misled: float = 0.55


PROFILES = {
    "sim-7b": Profile(),
    "sim-8b": Profile(name="sim-8b", slope=1.7, mid=3.2, p_confident_wrong_misconception=0.55, p_stubborn=0.3),
}


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _dirichlet(rng: random.Random, n: int, alpha: float = 1.0) -> list[float]:
    xs = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    s = sum(xs) or 1.0
    return [x / s for x in xs]


class SimulatedQALLM(LLM):
    """LLM simulator over the ODQA world; implements :meth:`answer` and :meth:`p_true`."""

    supports_logprobs = True

    def __init__(self, world: list[WorldItem], seed: int = 0, profile: str | Profile = "sim-7b"):
        self.world = world
        self.by_question = {normalize_answer(w.question): w for w in world}
        self.seed = seed
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.name = f"simulator:{self.profile.name}:seed{seed}"
        self.vocab: dict[str, set[str]] = {}
        for w in world:
            if w.answer_type != "ambiguous":
                self.vocab.setdefault(w.answer_type, set()).add(w.answer)
                self.vocab[w.answer_type].update(w.distractors)
        self._calls = 0

    # ------------------------------------------------------------ beliefs
    def item(self, question: str) -> WorldItem | None:
        return self.by_question.get(normalize_answer(question))

    def closed_belief(self, item: WorldItem) -> tuple[str, dict[str, float]]:
        """Return (knowledge state, distribution over answer meanings)."""
        rng = _rng(self.seed, self.profile.name, item.id, "closed")
        p = self.profile
        if item.kind == "ambiguous":
            options = item.aliases[:]
            rng.shuffle(options)
            options = options[: rng.randint(4, min(6, len(options)))]
            w = _dirichlet(rng, len(options), 1.5)
            return "ambiguous", dict(zip(options, w))
        p_know = 1.0 / (1.0 + math.exp(-p.slope * (math.log10(max(item.popularity, 1)) - p.mid)))
        distractors = item.distractors or ["unknown"]
        if rng.random() < p_know:
            m = rng.uniform(0.75, 0.97)
            rest = distractors[: rng.randint(1, 2)]
            w = _dirichlet(rng, len(rest))
            return "knows", {item.answer: m, **{d: (1 - m) * x for d, x in zip(rest, w)}}
        p_cw = p.p_confident_wrong_misconception if item.misconception else p.p_confident_wrong_other
        if rng.random() < p_cw:
            wrong = item.misconception or distractors[0]
            m = rng.uniform(0.7, 0.95)
            c = rng.uniform(0.0, 0.1) * (1 - m)
            others = [d for d in distractors if d != wrong][:2]
            w = _dirichlet(rng, len(others)) if others else []
            dist = {wrong: m, item.answer: c}
            for d, x in zip(others, w):
                dist[d] = dist.get(d, 0.0) + (1 - m - c) * x
            return "confident_wrong", dist
        c = rng.uniform(0.05, 0.4)
        opts = distractors[: rng.randint(3, min(4, len(distractors)))] if len(distractors) >= 3 else distractors
        w = _dirichlet(rng, len(opts), 0.8)
        dist = {item.answer: c}
        for d, x in zip(opts, w):
            dist[d] = dist.get(d, 0.0) + (1 - c) * x
        return "confused", dist

    def rag_belief(self, item: WorldItem, passages: list[str]) -> tuple[str, dict[str, float]]:
        state, closed = self.closed_belief(item)
        rng = _rng(self.seed, self.profile.name, item.id, "rag", "|".join(sorted(passages)))
        p = self.profile
        text = " ".join(passages)
        if item.kind == "ambiguous":
            found = [a for a in item.aliases if contains_match(text, [a])]
            if found:
                w = _dirichlet(rng, len(found), 2.0)
                return "ambiguous_grounded", dict(zip(found, w))
            return "ambiguous", closed
        if contains_match(text, item.aliases):
            if state == "confident_wrong" and rng.random() < p.p_stubborn:
                mix = {k: 0.7 * v for k, v in closed.items()}
                mix[item.answer] = mix.get(item.answer, 0.0) + 0.3
                return "knowledge_conflict", mix
            if rng.random() < p.p_read_gold:
                m = rng.uniform(0.85, 0.98)
                dist = {k: (1 - m) * v for k, v in closed.items() if k != item.answer}
                z = sum(dist.values()) or 1.0
                dist = {k: v * (1 - m) / z for k, v in dist.items()}
                dist[item.answer] = m
                return "grounded", dist
            return "ignored_evidence", closed
        candidates = [c for c in sorted(self.vocab.get(item.answer_type, set()))
                      if c != item.answer and contains_match(text, [c])]
        if candidates and rng.random() < p.p_misled:
            wrong = candidates[0]
            m = rng.uniform(0.6, 0.9)
            dist = {k: (1 - m) * v for k, v in closed.items()}
            dist[wrong] = dist.get(wrong, 0.0) + m
            return "misled", dist
        flat = {k: v ** 0.6 for k, v in closed.items()}
        z = sum(flat.values())
        return "unhelpful_evidence", {k: v / z for k, v in flat.items()}

    # ---------------------------------------------------------- sampling
    def _render(self, item: WorldItem, meaning: str, rng: random.Random, greedy: bool) -> tuple[str, float]:
        templates = YEAR_TEMPLATES if item.answer_type == "year" else TEMPLATES
        if (meaning == item.answer or meaning in item.aliases) and item.kind != "ambiguous":
            # surface variants of the same name ("Premchand" / "Munshi Premchand"), not disjoint aliases
            head = set(normalize_answer(meaning).split())
            variants = [a for a in item.aliases if a != meaning and head & set(normalize_answer(a).split())]
            forms = [meaning] + variants[:2]
        else:
            forms = [meaning]
        if greedy:
            return templates[0][0].format(a=forms[0]), math.log(0.9)
        f_idx = 0 if rng.random() < 0.7 or len(forms) == 1 else rng.randrange(1, len(forms))
        r, acc = rng.random(), 0.0
        tpl, tp = templates[-1]
        for t, w in templates:
            acc += w
            if r <= acc:
                tpl, tp = t, w
                break
        p_form = 0.7 if f_idx == 0 else 0.3 / max(1, len(forms) - 1)
        return tpl.format(a=forms[f_idx]), math.log(tp) + math.log(p_form)

    def _generate(self, item: WorldItem, dist: dict[str, float], n: int, temperature: float,
                  tag: str) -> list[Generation]:
        self._calls += 1
        rng = _rng(self.seed, self.profile.name, item.id, tag, n, temperature, self._calls)
        meanings = list(dist)
        probs = [max(dist[m], 1e-9) for m in meanings]
        out = []
        if temperature <= 0:
            best = max(range(len(meanings)), key=lambda i: probs[i])
            text, lp_form = self._render(item, meanings[best], rng, greedy=True)
            out.append(Generation(text, self._token_logprobs(text, math.log(probs[best]), lp_form, rng)))
            return out * n
        scaled = [p ** (1.0 / temperature) for p in probs]
        z = sum(scaled)
        scaled = [s / z for s in scaled]
        for _ in range(n):
            r, acc, idx = rng.random(), 0.0, len(meanings) - 1
            for i, s in enumerate(scaled):
                acc += s
                if r <= acc:
                    idx = i
                    break
            text, lp_form = self._render(item, meanings[idx], rng, greedy=False)
            out.append(Generation(text, self._token_logprobs(text, math.log(probs[idx]), lp_form, rng)))
        return out

    @staticmethod
    def _token_logprobs(text: str, lp_meaning: float, lp_form: float, rng: random.Random) -> list[float]:
        """The content (entity) token carries the meaning probability; formatting spreads over the other tokens."""
        n_rest = max(1, len(text.split()))
        return [lp_meaning] + [lp_form / n_rest - abs(rng.gauss(0.03, 0.02)) for _ in range(n_rest)]

    # ------------------------------------------------------- LLM interface
    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0, max_tokens: int = 48,
                 system: str | None = None) -> list[Generation]:
        raise NotImplementedError("The simulator answers structured questions; call answer().")

    def answer(self, question: str, passages: list[str] | None = None, n: int = 1, temperature: float = 0.0,
               max_tokens: int = 24) -> list[Generation]:
        item = self.item(question)
        if item is None:
            return [Generation("I don't know.", [math.log(0.2), -0.1, -0.1])] * n
        if passages:
            _, dist = self.rag_belief(item, passages)
            tag = "rag"
        else:
            _, dist = self.closed_belief(item)
            tag = "closed"
        return self._generate(item, dist, n, temperature, tag)

    def p_true(self, question: str, answer: str, samples: list[str] | None = None) -> float | None:
        item = self.item(question)
        if item is None:
            return 0.5
        _, dist = self.closed_belief(item)
        belief = 0.0
        for meaning, p in dist.items():
            forms = item.aliases if meaning == item.answer else [meaning]
            if contains_match(answer, forms):
                belief = max(belief, p)
        rng = _rng(self.seed, item.id, "ptrue", answer)
        conf = 0.05 + 0.9 * belief ** 0.6 + rng.gauss(0, 0.15)  # noisy, over-confident self-evaluation
        return float(min(0.99, max(0.01, conf)))
