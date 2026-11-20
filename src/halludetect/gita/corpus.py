"""The verse-aligned Bhagavad Gita corpus and its retrieval indexes."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import cached_property
from pathlib import Path

from ..paths import GITA_DATA
from ..retrieval import BM25, Hit, reciprocal_rank_fusion
from ..text import concept_tokens, strip_accents

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Words that describe the request rather than its content ("Which verse of the Gita says ...").
QUERY_NOISE = re.compile(
    r"\b(?:which|what|where|when|why|how|does|do|is|are|verse|verses|shloka|sloka|chapter|chapters|bhagavad|g[iī]t[aā]|"
    r"says?|said|tells?|told|describes?|mentions?|states?|teach(?:es)?|explains?|according)\b",
    re.IGNORECASE,
)

# Hindi → English concept glossary for cross-lingual retrieval (Hindi questions over an
# English/Sanskrit index). Keys are stems; a query token matches if it starts with the key.
HINDI_GLOSSARY: dict[str, str] = {
    "आत्म": "soul self", "शरीर": "body", "देह": "body", "कर्म": "action work", "फल": "fruit",
    "धर्म": "duty righteousness", "अधर्म": "unrighteousness decay righteousness", "युद्ध": "fight battle war",
    "योग": "yoga", "भक्ति": "devotion", "भक्त": "devotee devotion", "ज्ञान": "wisdom knowledge", "मन": "mind",
    "इच्छा": "desire", "कामना": "desire", "काम": "desire lust", "क्रोध": "anger wrath", "लोभ": "greed",
    "शांति": "peace", "शान्ति": "peace", "मृत्यु": "death", "मर": "death die", "जन्म": "birth born",
    "अर्जुन": "arjuna", "कृष्ण": "krishna", "शस्त्र": "weapons", "अग्नि": "fire", "आग": "fire", "जल": "water burn",
    "पानी": "water", "वायु": "wind air", "हवा": "wind", "काट": "cut cleave", "वस्त्र": "garments",
    "कपड": "garments", "पुरान": "worn-out old", "नए": "new", "नय": "new", "त्याग": "renounce abandon relinquish",
    "संन्यास": "renunciation", "शरण": "refuge shelter", "पाप": "sin", "मोक्ष": "liberate liberation",
    "मुक्त": "liberate free", "भोजन": "food", "आहार": "food", "श्रद्धा": "faith", "यज्ञ": "sacrifice",
    "ध्यान": "meditation", "गुरु": "teacher", "सुख": "pleasure happiness", "दुःख": "pain", "दुख": "pain",
    "समत्व": "equilibrium balanced", "सम": "equal balanced", "स्थितप्रज्ञ": "stable mind sage",
    "प्रकृति": "nature matter", "पुरुष": "spirit", "गुण": "qualities", "अवतार": "come forth born age",
    "विश्वरूप": "form universal", "काल": "time", "ब्रह्म": "eternal brahman", "नरक": "hell", "द्वार": "gate",
    "तीन": "three triple", "चार": "four", "वेद": "veda vedas", "कुशल": "skill", "कौशल": "skill",
    "चंचल": "restless", "वश": "curb control", "अभ्यास": "practice", "वैराग्य": "dispassion",
    "इंद्रिय": "senses", "इन्द्रिय": "senses", "बुद्धि": "reason understanding", "मोह": "delusion",
    "स्मृति": "memory", "संशय": "doubt", "पत्र": "leaf", "पुष्प": "flower", "फूल": "flower",
    "दीप": "lamp", "दीपक": "lamp", "कछुआ": "tortoise", "कमल": "lotus", "वृक्ष": "tree", "पेड़": "tree",
    "सूर्य": "sun", "चंद्र": "moon", "चन्द्र": "moon", "शंख": "conch", "संजय": "sanjaya", "सञ्जय": "sanjaya",
    "धृतराष्ट्र": "dhritarashtra",
}


@dataclass(frozen=True)
class Verse:
    ref: str
    chapter: int
    verse: int
    ref_700: str | None
    speaker: str
    sanskrit: str
    iast: str
    english: str

    @property
    def label(self) -> str:
        return f"BG {self.ref}"

    def premise(self) -> str:
        """Text used as NLI premise: the speaker plus Besant's translation."""
        return f"{self.speaker} said: {self.english}"


def fold(text: str) -> str:
    """ASCII-fold and lower-case (IAST ``kṛṣṇa`` → ``krsna``), keep letters and spaces."""
    t = strip_accents(text.lower()).replace("'", "")
    return re.sub(r"[^a-z0-9 ]+", " ", t)


def char_ngrams(text: str, n: int = 4) -> list[str]:
    t = re.sub(r"\s+", "", fold(text))
    return [t[i: i + n] for i in range(max(0, len(t) - n + 1))]


class GitaCorpus:
    """701 verses (Besant numbering) with existence checks, edition mapping and search."""

    def __init__(self, verses: list[Verse], chapters: list[dict]):
        self.verses: dict[str, Verse] = {v.ref: v for v in verses}
        self.ordered: list[Verse] = sorted(verses, key=lambda v: (v.chapter, v.verse))
        self.chapters = {c["chapter"]: c for c in chapters}
        self.chapter_lengths = {c["chapter"]: c["verses"] for c in chapters}
        self.chapter_lengths_700 = {c["chapter"]: c["verses_700"] for c in chapters}
        self._by_700 = {v.ref_700: v for v in verses if v.ref_700}

    @classmethod
    def load(cls, directory: Path | str | None = None) -> GitaCorpus:
        d = Path(directory) if directory else GITA_DATA
        verses = [Verse(**v) for v in json.loads((d / "verses.json").read_text(encoding="utf-8"))]
        chapters = json.loads((d / "chapters.json").read_text(encoding="utf-8"))
        return cls(verses, chapters)

    def __len__(self) -> int:
        return len(self.verses)

    # --------------------------------------------------------------- lookup
    def exists(self, chapter: int, verse: int) -> bool:
        """True if BG chapter.verse exists in either the 701-verse or the 700-verse numbering."""
        if chapter not in self.chapter_lengths:
            return False
        return 1 <= verse <= self.chapter_lengths[chapter]

    def get(self, chapter: int | str, verse: int | None = None) -> Verse | None:
        if verse is None:
            return self.verses.get(str(chapter))
        return self.verses.get(f"{chapter}.{verse}")

    def get_700(self, chapter: int, verse: int) -> Verse | None:
        """Resolve a reference given in the common 700-verse numbering."""
        return self._by_700.get(f"{chapter}.{verse}")

    def candidates(self, chapter: int, verse: int) -> list[Verse]:
        """All verses a citation may denote under the two numbering conventions (deduplicated)."""
        out = []
        for v in (self.get(chapter, verse), self.get_700(chapter, verse)):
            if v is not None and v not in out:
                out.append(v)
        return out

    def neighbours(self, verse: Verse, width: int = 1) -> list[Verse]:
        out = []
        for d in range(-width, width + 1):
            if d == 0:
                continue
            v = self.get(verse.chapter, verse.verse + d)
            if v:
                out.append(v)
        return out

    # ------------------------------------------------------------- indexes
    @cached_property
    def english_index(self) -> BM25:
        return BM25({v.ref: f"{v.english} {v.speaker}" for v in self.ordered}, tokenizer=concept_tokens)

    @cached_property
    def iast_index(self) -> BM25:
        return BM25({v.ref: v.iast for v in self.ordered}, tokenizer=char_ngrams)

    @cached_property
    def devanagari_index(self) -> BM25:
        def dev_ngrams(t: str) -> list[str]:
            s = re.sub(r"[\s।॥]+", "", t)
            return [s[i: i + 3] for i in range(max(0, len(s) - 2))]

        return BM25({v.ref: v.sanskrit for v in self.ordered}, tokenizer=dev_ngrams)

    @cached_property
    def idf(self) -> dict[str, float]:
        """Inverse document frequency of content tokens over the English verses (for LexicalNLI)."""
        df: Counter[str] = Counter()
        for v in self.ordered:
            df.update(set(concept_tokens(v.english)))
        n = len(self.ordered)
        return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}

    @cached_property
    def english_vocabulary(self) -> set[str]:
        return {t for v in self.ordered for t in concept_tokens(v.english)}

    # -------------------------------------------------------------- search
    def expand_hindi(self, query: str) -> str:
        extra = []
        for tok in re.findall(r"[\u0900-\u097F]+", query):
            for key, eng in HINDI_GLOSSARY.items():
                if tok.startswith(key) or (len(tok) >= 3 and key.startswith(tok)):
                    extra.append(eng)
        return " ".join(extra)

    def search(self, query: str, k: int = 5) -> list[tuple[Verse, float]]:
        """Hybrid search: English BM25 + (Hindi glossary | IAST n-grams | Devanagari n-grams), fused by RRF."""
        rankings: list[list[Hit]] = []
        english_query = query
        if DEVANAGARI.search(query):
            latin_part = re.sub(r"[\u0900-\u097F]+", " ", query)
            expansion = self.expand_hindi(query)
            english_query = f"{latin_part} {expansion}"
            # Character n-grams over the Sanskrit text help for Sanskrit quotations, but add noise for
            # Hindi questions, which the glossary already maps to English concepts.
            if "।" in query or "॥" in query or not expansion.strip():
                dev = self.devanagari_index.search(query, k=20)
                if dev:
                    rankings.append(dev)
        eng = self.english_index.search(QUERY_NOISE.sub(" ", english_query), k=20)
        if eng:
            rankings.append(eng)
        toks = concept_tokens(fold(query))
        unknown = [t for t in toks if t not in self.english_vocabulary]
        if len(unknown) >= 2 and len(unknown) >= 0.5 * max(len(toks), 1):
            iast = self.iast_index.search(query, k=20)
            if iast:
                rankings.append(iast)
        if not rankings:
            return []
        if len(rankings) == 1:
            hits = rankings[0][:k]
        else:
            hits = reciprocal_rank_fusion(rankings, top=k)
        return [(self.verses[h.doc_id], h.score) for h in hits]

    def max_bm25(self, query: str) -> float:
        hits = self.english_index.search(query, k=1)
        return hits[0].score if hits else 0.0

    # --------------------------------------------------------------- quotes
    @staticmethod
    def partial_ratio(needle: str, haystack: str) -> float:
        """Best SequenceMatcher ratio of ``needle`` against any equally long window of ``haystack``."""
        a, b = fold(needle).split(), fold(haystack).split()
        if not a or not b:
            return 0.0
        n = len(a)
        if len(b) <= n:
            return SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
        best = 0.0
        sa = " ".join(a)
        for i in range(len(b) - n + 1):
            r = SequenceMatcher(None, sa, " ".join(b[i: i + n])).ratio()
            if r > best:
                best = r
        return best

    def quote_match(self, quote: str, verse: Verse) -> float:
        """Similarity of a quotation to a verse (max over English, IAST and Devanagari)."""
        if DEVANAGARI.search(quote):
            q = re.sub(r"[\s।॥]+", "", quote)
            s = re.sub(r"[\s।॥]+", "", verse.sanskrit)
            if not q:
                return 0.0
            best = 0.0
            for i in range(0, max(1, len(s) - len(q) + 1)):
                r = SequenceMatcher(None, q, s[i: i + len(q)]).ratio()
                best = max(best, r)
            return best
        cands = [self.partial_ratio(quote, verse.english)]
        cands.append(self.partial_ratio(quote, verse.iast))
        # IAST in the corpus keeps sandhi-joined compounds; compare without spaces as well.
        q2 = re.sub(r"\s+", "", fold(quote))
        s2 = re.sub(r"\s+", "", fold(verse.iast))
        if q2 and len(q2) <= len(s2):
            best = max(SequenceMatcher(None, q2, s2[i: i + len(q2)]).ratio()
                       for i in range(0, len(s2) - len(q2) + 1, 2))
            cands.append(best)
        return max(cands)

    def find_quote(self, quote: str, k: int = 3) -> list[tuple[Verse, float]]:
        """Locate the verses that best match a quotation anywhere in the corpus."""
        if DEVANAGARI.search(quote):
            pool = [self.verses[h.doc_id] for h in self.devanagari_index.search(quote, k=25)]
        else:
            pool_ids = [h.doc_id for h in self.english_index.search(quote, k=15)]
            pool_ids += [h.doc_id for h in self.iast_index.search(quote, k=15)]
            pool = [self.verses[i] for i in dict.fromkeys(pool_ids)]
        scored = sorted(((v, self.quote_match(quote, v)) for v in pool), key=lambda x: -x[1])
        return scored[:k]
