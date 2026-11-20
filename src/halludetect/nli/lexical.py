"""A transparent, dependency-free NLI approximation.

``LexicalNLI`` estimates entailment from the weighted proportion of the
hypothesis' content words that are covered by the premise, after stemming,
archaic-English folding ("burneth" → "burn") and a concept lexicon
("cleave" → "cut", "Partha" → "Arjuna"); see :data:`halludetect.text.SYNONYMS`.
Contradiction is signalled by three cues:

* **entity substitution** – proper nouns/numbers carrying ≥ 30 % of the
  hypothesis weight are missing while the premise names other entities
  ("Tagore wrote Godan" vs "Godan ... by Premchand");
* **number mismatch** – both sides contain numbers and none agree;
* **scope-aware negation** – a content word is negated on one side but asserted
  on the other ("never with its fruits" vs "only for its fruits").

It is **not** a substitute for a trained NLI model. It exists so that every
pipeline in this repository runs offline (tests, CI, demo app); the reports
compare it with DeBERTa-large-MNLI.
"""

from __future__ import annotations

import math
import re

from ..text import STOPWORDS, STRONG_NEGATIONS, canon, numbers_in, stem, strip_accents
from .base import NLIModel, NLIResult

# Relational and meta words carry little identifying information in factual statements.
LIGHT = frozenset(stem(w) for w in (
    "write build invent discover found born die located capital city country name call known famous "
    "first year said says say tell tells told describe described state states stated teach teaches "
    "explain explains verse chapter gita bhagavad text".split()
))

_TOKEN = re.compile(r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF]+")
_CLAUSE_BREAK = frozenset(",;:.!?")


class LexicalNLI(NLIModel):
    """Weighted lexical-coverage NLI with contradiction cues (offline, deterministic).

    Parameters
    ----------
    entail_threshold : coverage at which entailment probability crosses 0.5.
    idf : optional token → IDF map (e.g. from the verse corpus) to up-weight rare words.
    light_terms : extra words to down-weight (e.g. character names in a scripture).
    """

    name = "lexical"

    def __init__(self, entail_threshold: float = 0.7, idf: dict[str, float] | None = None, sharpness: float = 14.0,
                 light_terms: tuple[str, ...] | list[str] = ()):
        self.entail_threshold = entail_threshold
        self.sharpness = sharpness
        self.idf = idf
        self.light = LIGHT | {canon(t) for t in light_terms}
        if idf:
            vals = sorted(idf.values())
            self._idf_mid = vals[len(vals) // 2]
            self._idf_max = vals[-1]

    # ------------------------------------------------------------------ tokens
    def tokens(self, text: str) -> list[tuple[str, bool]]:
        """Return ``(canonical_token, is_strong)`` for every content word.

        Strong tokens are numbers and capitalised words that are not sentence-initial.
        """
        out = []
        for m in _TOKEN.finditer(text):
            raw = m.group(0)
            if len(raw) < 2 and not raw.isdigit():
                continue
            low = strip_accents(raw.lower())
            if low in STOPWORDS or low in STRONG_NEGATIONS or low == "without":
                continue
            prev = text[: m.start()].rstrip()
            sentence_start = not prev or prev[-1] in ".!?:;\"“("
            strong = raw.isdigit() or (raw[0].isupper() and not sentence_start)
            out.append((canon(raw), strong))
        return out

    def _scoped(self, text: str, back: int = 3, forward: int = 5) -> tuple[set[str], set[str]]:
        """Split content tokens into those asserted and those inside a negation scope.

        The scope is symmetric within a clause (``back`` words before and ``forward``
        words after the negation) because translations of verse often post-pose the
        negation: "Weapons cleave him not", "flickereth not".
        """
        text = re.sub(r"n['’]t\b", " not", text)
        affirmed: set[str] = set()
        negated: set[str] = set()
        for clause in re.split(r"[;:.!?]", text):
            seq: list[str | None] = []  # None marks a negation
            for m in re.finditer(r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF]+", clause):
                low = strip_accents(m.group(0).lower())
                if low in STRONG_NEGATIONS:
                    seq.append(None)
                elif low not in STOPWORDS and (len(low) >= 2 or low.isdigit()):
                    seq.append(canon(m.group(0)))
            content_idx = [i for i, t in enumerate(seq) if t is not None]
            neg_pos = [i for i, t in enumerate(seq) if t is None]
            for i in content_idx:
                is_neg = False
                for j in neg_pos:
                    n_after = sum(1 for k in content_idx if j < k <= i)   # content words between negation and i
                    n_before = sum(1 for k in content_idx if i <= k < j)  # content words between i and negation
                    if (i > j and n_after <= forward) or (i < j and n_before <= back):
                        is_neg = True
                        break
                (negated if is_neg else affirmed).add(seq[i])
        return affirmed, negated

    def _weight(self, tok: str, strong: bool) -> float:
        if tok in self.light:
            w = 0.5
        else:
            w = 2.0 if strong else 1.0
        if self.idf:
            idf = self.idf.get(tok, self._idf_max)
            w *= min(2.0, max(0.5, idf / self._idf_mid))
        return w

    # ---------------------------------------------------------------- predict
    def _score(self, premise: str, hypothesis: str) -> NLIResult:
        p_toks = self.tokens(premise)
        h_toks = self.tokens(hypothesis)
        p_set = {t for t, _ in p_toks}
        if not h_toks:
            return NLIResult(0.34, 0.66, 0.0)
        weights: dict[str, float] = {}
        strong_h: set[str] = set()
        for t, s in h_toks:
            weights[t] = max(weights.get(t, 0.0), self._weight(t, s))
            if s and t not in self.light:
                strong_h.add(t)
        total = sum(weights.values())
        covered = sum(w for t, w in weights.items() if t in p_set) / total

        contra = 0.0
        missing_strong = sum(weights[t] for t in strong_h if t not in p_set)
        strong_p_extra = {t for t, s in p_toks if s and t not in self.light} - set(weights)
        if missing_strong / total >= 0.3 and strong_p_extra and covered >= 0.3:
            contra += 0.55
        h_nums, p_nums = numbers_in(hypothesis), numbers_in(premise)
        if h_nums and p_nums and not (h_nums & p_nums):
            contra += 0.5
        if covered >= 0.5:
            h_aff, h_neg = self._scoped(hypothesis)
            p_aff, p_neg = self._scoped(premise)
            only_neg_h, only_aff_h = h_neg - h_aff, h_aff - h_neg
            only_neg_p, only_aff_p = p_neg - p_aff, p_aff - p_neg
            conflict = (only_neg_h & only_aff_p) | (only_aff_h & only_neg_p)
            if any(weights.get(t, 0.0) >= 0.75 for t in conflict):
                contra += 0.5
        contra = min(contra, 0.95)

        ent_raw = 1.0 / (1.0 + math.exp(-self.sharpness * (covered - self.entail_threshold)))
        ent = ent_raw * (1.0 - contra)
        neutral = max(0.0, 1.0 - ent - contra)
        z = ent + neutral + contra
        return NLIResult(ent / z, neutral / z, contra / z)

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        return [self._score(p, h) for p, h in pairs]

    def coverage(self, premise: str, hypothesis: str) -> float:
        """Weighted fraction of hypothesis content covered by the premise (exposed for diagnostics)."""
        p_set = {t for t, _ in self.tokens(premise)}
        h = self.tokens(hypothesis)
        if not h:
            return 0.0
        w: dict[str, float] = {}
        for t, s in h:
            w[t] = max(w.get(t, 0.0), self._weight(t, s))
        return sum(v for t, v in w.items() if t in p_set) / sum(w.values())

    # ------------------------------------------------------------ equivalence
    def equivalent(self, a: str, b: str, context: str = "", strict: bool = True) -> bool:
        """Meaning equivalence of two short answers given the question.

        Question words are removed so only the answer content is compared;
        answers are equivalent when one content set contains the other with at
        most one extra token (``"Premchand"`` ≡ ``"Munshi Premchand"``) or, in
        relaxed mode, when their Jaccard overlap is at least one third.
        """
        q = {t for t, _ in self.tokens(context)}
        ta = {t for t, _ in self.tokens(a)} - q
        tb = {t for t, _ in self.tokens(b)} - q
        if not ta or not tb:
            na = " ".join(strip_accents(a.lower()).split())
            nb = " ".join(strip_accents(b.lower()).split())
            return bool(ta) == bool(tb) and na == nb
        na_, nb_ = numbers_in(a), numbers_in(b)
        if (na_ or nb_) and na_ != nb_:
            return False
        neg_a = bool(set(re.findall(r"[a-z']+", a.lower())) & STRONG_NEGATIONS)
        neg_b = bool(set(re.findall(r"[a-z']+", b.lower())) & STRONG_NEGATIONS)
        if neg_a != neg_b:
            return False
        inter = ta & tb
        if not inter:
            return False
        if strict:
            small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
            return small <= big and len(big - small) <= 1
        return len(inter) / len(ta | tb) >= 1 / 3
