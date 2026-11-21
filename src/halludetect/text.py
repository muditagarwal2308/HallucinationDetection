"""Text normalisation, tokenisation and answer-matching utilities."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from functools import lru_cache

ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which while who whom why will
    with would you your yours yourself yourselves also thus hence verily even let shall may might must
    o oh ye unto upon one ones whoever whatever whosoever alone""".split()
)

# Archaic forms that appear in 19th/early-20th century translations (Besant, Telang).
ARCHAIC = {
    "thou": "you", "thee": "you", "thy": "your", "thine": "your", "ye": "you",
    "hath": "has", "hast": "have", "doth": "does", "dost": "do", "art": "are",
    "wilt": "will", "shalt": "shall", "canst": "can", "spake": "spoke", "saith": "says",
    "knoweth": "knows", "goeth": "goes", "seeth": "sees", "cometh": "comes",
    "unto": "to", "whence": "where", "wherefore": "why", "nay": "no", "aught": "anything",
    "naught": "nothing", "fain": "gladly",
}

NEGATIONS = frozenset({"not", "no", "never", "nor", "neither", "none", "cannot", "nothing", "nobody", "without"})
# Negations that flip the meaning of what follows (used for scope-aware contradiction).
STRONG_NEGATIONS = frozenset({"not", "no", "never", "nor", "neither", "none", "cannot", "nothing", "nobody"})


def strip_accents(text: str) -> str:
    """Remove diacritics (useful for IAST: ``kṛṣṇa`` -> ``krsna``)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lower-case, drop punctuation, articles and extra spaces."""
    text = strip_accents(text.lower())
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, aliases: list[str] | tuple[str, ...]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(a) for a in aliases)


def contains_match(prediction: str, aliases: list[str] | tuple[str, ...]) -> bool:
    """True if any gold alias occurs as a whole-word span of the prediction.

    This is the standard lenient criterion for free-form LLM answers such as
    "It was written by Munshi Premchand." (gold alias "Premchand").
    """
    pred = f" {normalize_answer(prediction)} "
    return any(f" {normalize_answer(a)} " in pred for a in aliases if normalize_answer(a))


def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def max_token_f1(prediction: str, aliases: list[str] | tuple[str, ...]) -> float:
    return max((token_f1(prediction, a) for a in aliases), default=0.0)


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """A small, deterministic suffix stripper (Porter-lite) that also folds archaic -eth/-est."""
    w = ARCHAIC.get(word, word)
    if len(w) <= 3:
        return w
    for suffix, repl in (
        ("ieth", "y"), ("eth", ""), ("est", ""), ("ies", "y"), ("ied", "y"), ("ing", ""),
        ("edly", ""), ("ness", ""), ("ment", ""), ("ful", ""), ("ly", ""), ("ed", ""),
        ("es", ""), ("s", ""),
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)] + repl
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]  # cutting -> cutt -> cut
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


_WORD = re.compile(r"[a-z0-9\u0900-\u097F]{2,}|\d")


def words(text: str) -> list[str]:
    return _WORD.findall(strip_accents(text.lower()))


def content_tokens(text: str, keep_negations: bool = False) -> list[str]:
    """Lower-cased, stemmed tokens with stop words removed."""
    out = []
    for w in words(text):
        w = ARCHAIC.get(w, w)
        if w in NEGATIONS:
            if keep_negations:
                out.append("not")
            continue
        if w in STOPWORDS:
            continue
        out.append(stem(w))
    return out


def has_negation(text: str) -> bool:
    toks = set(words(text))
    return bool(toks & NEGATIONS) or "n't" in text.lower()


_REFUSAL = re.compile(
    r"^\W*(?:i\s+(?:do\s+not|don['’]t)\s+know|i\s+am\s+not\s+sure|i['’]m\s+not\s+sure|unknown|"
    r"i\s+(?:cannot|can['’]t)\s+(?:answer|say|determine)|(?:there\s+is\s+)?no\s+(?:answer|information)|"
    r"unanswerable|not\s+enough\s+information)\b",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """True for short-answer refusals such as "I don't know." (an abstention, not a factual claim)."""
    return bool(_REFUSAL.search(answer or ""))


_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[]?[A-Z0-9\u0900-\u097F])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking verse references such as ``BG 2.47``."""
    text = " ".join(text.split())
    if not text:
        return []
    parts = _SENT.split(text)
    return [p.strip() for p in parts if p.strip()]


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def answer_to_statement(question: str, answer: str) -> str:
    """Turn a (question, short answer) pair into a declarative hypothesis for NLI.

    NLI models work on sentences, not bare answers ("Premchand"), so we combine
    the two (q ⊕ a in the dissertation notation).
    """
    q = question.strip().rstrip("?").strip()
    a = answer.strip().rstrip(".").strip()
    if not a:
        return q + "."
    if len(a.split()) > 6:
        return a + "."  # already a full sentence
    m = re.match(r"(?i)^who\s+(.+)$", q)
    if m:
        return f"{a} {m.group(1)}."
    m = re.match(r"(?i)^what\s+(is|was|are|were)\s+(.+)$", q)
    if m:
        rest = m.group(2)
        return f"{rest[0].upper() + rest[1:]} {m.group(1).lower()} {a}."
    m = re.match(r"(?i)^(?:in\s+)?which\s+(\w+)\s+(is|was|are|were)\s+(.+)$", q)
    if m:
        return f"{m.group(3)[0].upper() + m.group(3)[1:]} {m.group(2).lower()} {a}."
    return f"{q}? {a}."


# Concept lexicon: folds synonyms and transliterated Sanskrit terms onto one canonical token
# so that modern paraphrases match 19th/20th-century translations ("cut" ~ "cleave").
_SYNONYM_GROUPS: dict[str, list[str]] = {
    "write": ["wrote", "written", "writes", "writing", "author", "authored", "authors", "penned"],
    "build": ["built", "builds", "constructed", "construct", "construction", "commissioned"],
    "invent": ["invented", "inventor", "invention", "devised"],
    "discover": ["discovered", "discoverer", "discovery"],
    "found": ["founded", "founder", "founding", "established"],
    "kill": ["slay", "slain", "slayeth", "slayer", "killed", "kills", "slaughtered", "slaughter", "killing"],
    "die": ["died", "dies", "death", "dead", "dying", "perish", "perisheth", "perishes"],
    "born": ["birth", "births", "borne"],
    "eternal": ["everlasting", "perpetual", "imperishable", "indestructible", "undying", "immortal",
                "eternally", "unending", "changeless"],
    "soul": ["self", "spirit", "atman", "atma", "dweller", "embodied"],
    "cut": ["cleave", "cleaves", "cleaveth", "cleft", "cleaved", "cuts", "cutting"],
    "burn": ["burnt", "burned", "burneth", "burns", "burning"],
    "wet": ["wets", "wetted", "moisten", "moistens", "drench"],
    "dry": ["dries", "drieth", "dried", "wither", "withers"],
    "action": ["actions", "act", "acts", "deed", "deeds", "work", "works", "karma", "activity", "activities"],
    "fruit": ["fruits", "result", "results", "reward", "rewards", "outcome", "outcomes", "consequence",
              "consequences"],
    "duty": ["duties", "dharma", "svadharma", "swadharma"],
    "devotion": ["devotee", "devotees", "devoted", "bhakti", "devotional"],
    "mind": ["minds", "manas"],
    "wisdom": ["knowledge", "jnana", "gnana", "wise"],
    "sage": ["sages", "muni", "rishi", "seer", "seers"],
    "desire": ["desires", "craving", "cravings", "lust", "kama", "longing", "longings"],
    "anger": ["wrath", "rage", "krodha"],
    "peace": ["tranquillity", "tranquility", "serenity", "calm", "shanti"],
    "fight": ["fighting", "fought", "battle", "war", "combat", "warfare", "battlefield"],
    "renounce": ["renunciation", "renounced", "renounces", "abandon", "abandoning", "abandoned", "abandons",
                 "relinquish", "relinquishment", "relinquishing", "sannyasa", "tyaga"],
    "attach": ["attachment", "attached", "attachments", "clinging", "cling"],
    "supreme": ["highest", "ultimate"],
    "weapon": ["weapons", "sword", "swords"],
    "meditate": ["meditation", "dhyana", "meditating"],
    "grieve": ["grief", "mourn", "mourning", "lament", "lamentation", "sorrow", "sorrowing", "grievest"],
    "happy": ["happiness", "joy", "bliss", "delight", "sukha", "pleasure", "pleasures"],
    "pain": ["pains", "suffering", "duhkha", "misery"],
    "eat": ["eating", "eaten", "food", "foods", "diet"],
    "faith": ["shraddha", "belief"],
    "sacrifice": ["sacrifices", "yajna", "offering", "offerings"],
    "yoga": ["yogi", "yogin", "yogis", "yogins"],
    "refuge": ["shelter", "surrender"],
    "sin": ["sins", "sinful", "papa"],
    "liberate": ["liberation", "liberates", "freedom", "moksha", "release", "deliverance", "salvation"],
    "teacher": ["teachers", "guru", "preceptor", "preceptors", "acharya"],
    "kinsmen": ["relatives", "kin", "kinsfolk", "kindred", "relations"],
    "garment": ["garments", "clothes", "clothing", "robes"],
    "capital": ["capitals"],
    "decay": ["decline", "declines", "declining", "decays", "decayed", "wane", "wanes"],
    "concern": ["business", "concerns", "right", "rights", "entitled", "entitlement"],
    "asvattha": ["ashvattha", "ashwattha", "asvatta", "peepal", "pipal"],
    "wake": ["wakefulness", "wakeful", "awake", "waking", "vigil"],
    "krishna": ["keshava", "janardana", "govinda", "madhusudana", "hrishikesha", "vasudeva", "varshneya",
                "achyuta", "madhava", "krsna"],
    "arjuna": ["partha", "kaunteya", "dhananjaya", "gudakesha", "parantapa", "savyasachi"],
    "kurukshetra": ["kuru"],
}
SYNONYMS: dict[str, str] = {}
for _concept, _variants in _SYNONYM_GROUPS.items():
    SYNONYMS[_concept] = _concept
    for _v in _variants:
        SYNONYMS[_v] = _concept



def canon(word: str) -> str:
    """Canonical concept token: accent-folded, synonym-mapped, stemmed."""
    w = strip_accents(word.lower())
    w = SYNONYMS.get(w, w)
    return stem(w)


def concept_tokens(text: str) -> list[str]:
    """Content words mapped to canonical concepts (used for Gita retrieval and lexical NLI)."""
    out = []
    for w in words(text):
        w = ARCHAIC.get(w, w)
        if w in STOPWORDS or w in NEGATIONS:
            continue
        out.append(canon(w))
    return out
