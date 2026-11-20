"""Segmentation of an answer into claims with their citations, quotations and speaker attributions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text import content_tokens, split_sentences
from .citations import Citation, extract_citations, strip_citations

SPEAKER_ALIASES = {
    "krishna": "Krishna", "krsna": "Krishna", "kṛṣṇa": "Krishna", "lord krishna": "Krishna", "the lord": "Krishna",
    "the blessed lord": "Krishna", "bhagavan": "Krishna", "shri krishna": "Krishna", "sri krishna": "Krishna",
    "vasudeva": "Krishna", "keshava": "Krishna", "arjuna": "Arjuna", "partha": "Arjuna", "sanjaya": "Sanjaya",
    "dhritarashtra": "Dhritarashtra", "the king": "Dhritarashtra",
}
_SPK = "|".join(sorted((re.escape(k) for k in SPEAKER_ALIASES), key=len, reverse=True))
_VERB = (r"says|said|tells|told|declares|declared|states|stated|asks|asked|replies|replied|teaches|taught|"
         r"explains|explained|instructs|instructed|advises|advised|proclaims|proclaimed|exclaims|exclaimed|"
         r"assures|assured|urges|urged|reveals|revealed|describes|described|speaks|spoke|promises|promised|"
         r"reports|reported|narrates|narrated|answers|answered|agrees|agreed|names|named|calls|called")
SPEAKER_PATTERNS = [
    re.compile(rf"\b({_SPK})\s+(?:{_VERB})\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_VERB})\s+({_SPK})\b", re.IGNORECASE),
    re.compile(rf"\b(?:according to|in the words of|spoken by|said by|uttered by)\s+({_SPK})\b", re.IGNORECASE),
]
QUOTE = re.compile(r"[\"“”]([^\"“”]{8,400})[\"“”]|‘([^‘’]{8,300})’|(?:(?<=\s)|^)'([^']{8,300})'(?=[\s.,;:!?)]|$)")
POINTER = re.compile(
    r"^(?:this|that|it|these|those|the same|such|as|see|cf|refer)\b.{0,60}\b(?:described|stated|said|mentioned|found|"
    r"taught|given|expressed|explained|appears|declared|told|verse|verses)\b|^(?:see|cf\.?|refer to)\b",
    re.IGNORECASE,
)
REFUSAL_PREFIX = re.compile(
    r"^(?:no verse (?:says|states) (?:this|that)|(?:the\s+)?(?:bhagavad\s+)?g[iī]t[aā] does not say (?:this|that)|"
    r"\w+ does not say (?:this|that|these words)|(?:the\s+)?(?:bhagavad\s+)?(?:g[iī]t[aā]|\w+) says the opposite|"
    r"on the contrary|in fact|yes)\s*[;:,.]?\s*",
    re.IGNORECASE,
)
ATTRIBUTION_PREFIX = re.compile(
    rf"^(?:(?:in|according to)\s+(?:the\s+)?(?:bhagavad\s+)?g[iī]t[aā],?\s*|(?:the\s+)?(?:bhagavad\s+)?g[iī]t[aā]\s+"
    rf"(?:says|states|teaches|explains|declares|tells us)\s*(?:that)?\s*|(?:{_SPK})\s+(?:{_VERB})\s+(?:to\s+)?"
    rf"(?:arjuna|krishna)?\s*(?:that)?\s*,?\s*)",
    re.IGNORECASE,
)


@dataclass
class Claim:
    index: int
    sentence: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    speaker: str | None = None
    inherited_citation: bool = False

    @property
    def core(self) -> str:
        """Claim text without speaker attribution prefixes (what the verse must entail)."""
        core = self.text
        for _ in range(2):
            core = REFUSAL_PREFIX.sub("", core).strip()
            core = ATTRIBUTION_PREFIX.sub("", core).strip()
        return core[:1].upper() + core[1:] if core else core


def speaker_of(sentence: str) -> str | None:
    for pat in SPEAKER_PATTERNS:
        m = pat.search(sentence)
        if m:
            return SPEAKER_ALIASES.get(m.group(1).lower())
    return None


def extract_quotes(sentence: str) -> list[str]:
    out = []
    for m in QUOTE.finditer(sentence):
        q = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if len(q.split()) >= 3:
            out.append(q)
    return out


def segment_claims(answer: str) -> list[Claim]:
    """Split an answer into claims.

    A sentence that only points to a source ("This is described in BG 2.20.")
    contributes its citations to the previous claim instead of forming a claim
    of its own — exactly the pattern in the BG 2.20 / BG 2.23 example.
    """
    claims: list[Claim] = []
    for sent in split_sentences(answer):
        cits = extract_citations(sent)
        text = strip_citations(sent)
        is_pointer = bool(cits) and (len(content_tokens(text)) <= 4 or bool(POINTER.search(text)))
        if is_pointer and claims:
            claims[-1].citations.extend(cits)
            claims[-1].inherited_citation = True
            continue
        if not content_tokens(text) and not cits:
            continue
        claims.append(Claim(index=len(claims), sentence=sent, text=text, citations=list(cits),
                            quotes=extract_quotes(sent), speaker=speaker_of(sent)))
    return claims
