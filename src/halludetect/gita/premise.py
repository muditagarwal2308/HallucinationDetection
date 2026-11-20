"""Question analysis: non-existent references, out-of-scope topics and false premises."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..nli.base import NLIModel
from ..text import concept_tokens
from .citations import extract_citations
from .claims import SPEAKER_ALIASES, extract_quotes
from .corpus import GitaCorpus, Verse

# Topics that cannot be addressed by a text composed two millennia ago.
ANACHRONISMS = frozenset(
    """internet smartphone smartphones phone phones mobile computer computers software cryptocurrency bitcoin crypto
    tax taxes income cricket football electric electricity car cars airplane aeroplane television tv vaccine vaccines
    covid democracy parliament election elections mumbai delhi bangalore stock stocks shares bank banking loan
    payasam recipe biryani pizza email website app apps chatgpt ai robot robots""".split()
)
GENERIC = frozenset(
    concept_tokens(
        "bhagavad gita gītā krishna arjuna sanjaya lord verse verses chapter chapters text scripture say says said "
        "tell tells told teach teaches teaching explain explains describe describes mention mentions according "
        "state states declare declares advise advises speak speaks"
    )
)
REFUSAL = re.compile(
    r"\b(?:does not|doesn't|do not|did not|is not|isn't|no verse|not (?:found|mentioned|addressed|stated|said)|"
    r"there is no|cannot be found|outside (?:the )?scope|not discuss|nowhere|false premise|in fact|actually|"
    r"on the contrary|rather|i (?:do not|don't) know|cannot answer|has only|contains only|only has|"
    r"no such (?:verse|chapter)|not (?:part|a part) of|the opposite|only \d+ (?:verses|chapters))\b",
    re.IGNORECASE,
)

PRESUPPOSITION_PATTERNS = [
    re.compile(r"^(?:which|what)\s+(?:verse|shloka|sloka|passage)(?:\s+of\s+(?:the\s+)?(?:bhagavad\s+)?g[iī]t[aā])?"
               r"[^?]*?\b(?:says?|states?|teach(?:es)?|declares?|mentions?|tells?|describes?|explains?)\s+"
               r"(?:us\s+)?(?:that\s+)?(?P<x>.+?)\??$", re.IGNORECASE),
    re.compile(r"^(?:in\s+which|where|which)\s+(?:verse|chapter|shloka)?[^?]*?\bdoes\s+(?P<s>[a-z ]{3,20}?)\s+"
               r"(?P<v>say|tell|declare|state|advise|teach|ask|urge|promise|instruct)\s+(?:arjuna\s+|krishna\s+)?"
               r"(?:that\s+)?(?P<x>.+?)\??$", re.IGNORECASE),
    re.compile(r"^why\s+does\s+(?P<s>(?:the\s+)?(?:bhagavad\s+)?g[iī]t[aā]|[a-z ]{3,20}?)\s+"
               r"(?P<v>say|tell|declare|state|advise|teach|ask|urge|claim|insist|promise)\s+(?:arjuna\s+|krishna\s+)?"
               r"(?:that\s+)?(?P<x>.+?)\??$", re.IGNORECASE),
]


@dataclass
class PremiseAnalysis:
    kind: str = "ok"  # ok | nonexistent_reference | out_of_scope | false_premise
    presupposition: str | None = None
    presupposed_speaker: str | None = None
    message: str = ""
    evidence: list[dict] = field(default_factory=list)
    support: float = 1.0

    @property
    def is_problematic(self) -> bool:
        return self.kind != "ok"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "presupposition": self.presupposition, "presupposed_speaker":
                self.presupposed_speaker, "message": self.message, "support": round(self.support, 3),
                "evidence": self.evidence}


def is_refusal(answer: str) -> bool:
    """Does the answer reject the premise / decline to answer?"""
    return bool(REFUSAL.search(answer or ""))


class PremiseChecker:
    def __init__(self, corpus: GitaCorpus, nli: NLIModel, support_threshold: float = 0.5,
                 coverage_threshold: float = 0.6):
        self.corpus = corpus
        self.nli = nli
        self.support_threshold = support_threshold
        self.coverage_threshold = coverage_threshold

    def window(self, verse: Verse) -> str:
        """Verse with its neighbours: presuppositions are often spread over consecutive verses."""
        parts = [v for v in (self.corpus.get(verse.chapter, verse.verse - 1), verse,
                             self.corpus.get(verse.chapter, verse.verse + 1)) if v]
        return " ".join(f"{v.speaker} said: {v.english}" for v in parts)

    def best_support(self, statement: str, k: int = 8) -> tuple[Verse | None, float, float, list[dict]]:
        hits = self.corpus.search(statement, k=k)
        best, best_e, best_c, ev = None, 0.0, 0.0, []
        cov_fn = getattr(self.nli, "coverage", None)
        for v, _ in hits:
            prem = self.window(v)
            r = self.nli.predict(prem, statement)
            e = r.entailment
            if cov_fn is not None:
                e = max(e, 1.0 if cov_fn(prem, statement) >= self.coverage_threshold and r.contradiction < 0.4 else 0.0)
            ev.append({"ref": v.ref, "entailment": round(r.entailment, 3), "contradiction": round(r.contradiction, 3)})
            if e > best_e:
                best, best_e = v, e
            best_c = max(best_c, r.contradiction)
        return best, best_e, best_c, ev

    def analyze(self, question: str) -> PremiseAnalysis:
        q = question.strip()
        # 1. references to verses or chapters that do not exist
        for c in extract_citations(q):
            for ch, v in c.refs:
                if not self.corpus.exists(ch, v):
                    n = self.corpus.chapter_lengths.get(ch)
                    msg = (f"The Bhagavad Gita has no verse {ch}.{v}: chapter {ch} has {n} verses." if n else
                           f"The Bhagavad Gita has 18 chapters; there is no chapter {ch}.")
                    return PremiseAnalysis("nonexistent_reference", message=msg, support=0.0)
        for m in re.finditer(r"\bchapter\s+(\d{1,3})\b", q, re.IGNORECASE):
            if int(m.group(1)) not in self.corpus.chapter_lengths:
                return PremiseAnalysis("nonexistent_reference", support=0.0,
                                       message=f"The Bhagavad Gita has 18 chapters; there is no chapter {m.group(1)}.")
        # 2. topics outside the text
        toks = [t for t in concept_tokens(q) if t not in GENERIC]
        raw = {w.lower() for w in re.findall(r"[A-Za-z]+", q)}
        if raw & ANACHRONISMS:
            return PremiseAnalysis("out_of_scope", support=0.0,
                                   message="The question concerns a topic the Bhagavad Gita does not address "
                                           f"({', '.join(sorted(raw & ANACHRONISMS))}).")
        toks = [t for t in toks if not t.isdigit()]
        cites_verse = bool(extract_citations(q))
        if toks and not cites_verse and not re.search(r"[\u0900-\u097F]", q) and not self._looks_sanskrit(q):
            known = [t for t in toks if t in self.corpus.english_vocabulary]
            if len(toks) >= 2 and len(known) / len(toks) < 0.25:
                return PremiseAnalysis("out_of_scope", support=0.0,
                                       message="None of the verses discusses this topic.")
        # 3. presuppositions ("Which verse says X", "Why does Krishna say X")
        for pat in PRESUPPOSITION_PATTERNS:
            m = pat.match(q)
            if not m:
                continue
            x = m.group("x").strip()
            speaker = None
            if "s" in m.groupdict() and m.group("s"):
                speaker = SPEAKER_ALIASES.get(m.group("s").strip().lower())
            quotes = extract_quotes(q)
            if quotes:
                return self._check_quote_premise(quotes[0], speaker)
            if len(concept_tokens(x)) < 2:
                return PremiseAnalysis()
            best, e, c, ev = self.best_support(x)
            if best is not None and e >= self.support_threshold:
                if speaker and speaker != best.speaker and best.speaker != "Sanjaya":
                    return PremiseAnalysis(
                        "false_premise", presupposition=x, presupposed_speaker=speaker, evidence=ev[:3], support=e,
                        message=f"The words are spoken by {best.speaker}, not {speaker} (BG {best.ref}).")
                return PremiseAnalysis("ok", presupposition=x, presupposed_speaker=speaker, evidence=ev[:3], support=e)
            msg = "No verse of the Bhagavad Gita states this."
            if best is not None and c >= 0.4:
                msg += f" The closest verse (BG {ev[0]['ref']}) says something different."
            return PremiseAnalysis("false_premise", presupposition=x, presupposed_speaker=speaker,
                                   evidence=ev[:3], support=e, message=msg)
        return PremiseAnalysis()

    def _looks_sanskrit(self, text: str) -> bool:
        """True if most character 4-grams of the (romanised) text occur in the IAST verse index."""
        from .corpus import char_ngrams

        grams = char_ngrams(text)
        if len(grams) < 6:
            return False
        vocab = self.corpus.iast_index.idf
        return sum(g in vocab for g in grams) / len(grams) >= 0.8

    def _check_quote_premise(self, quote: str, speaker: str | None) -> PremiseAnalysis:
        found = self.corpus.find_quote(quote, k=1)
        if not found or found[0][1] < 0.8:
            return PremiseAnalysis("false_premise", presupposition=quote, presupposed_speaker=speaker, support=0.0,
                                   message="The quoted words do not occur in the Bhagavad Gita.")
        verse, score = found[0]
        ev = [{"ref": verse.ref, "quote_match": round(score, 3)}]
        if speaker and verse.speaker not in (speaker, "Sanjaya"):
            return PremiseAnalysis("false_premise", presupposition=quote, presupposed_speaker=speaker, evidence=ev,
                                   support=0.0, message=f"These words are spoken by {verse.speaker}, not {speaker} "
                                                        f"(BG {verse.ref}).")
        return PremiseAnalysis("ok", presupposition=quote, presupposed_speaker=speaker, evidence=ev, support=score)
