"""Controlled hallucination injection for Gita answers.

Used to (i) build the labelled VGHD test set from reference answers and
(ii) drive the offline answer simulator in the demo app. Every function takes a
correct, cited answer and returns a perturbed answer exhibiting exactly one
hallucination type, so detection can be evaluated per type.
"""

from __future__ import annotations

import random
import re

from ..text import STOPWORDS
from .citations import extract_citations
from .corpus import GitaCorpus

UNSUPPORTED_TEACHINGS = [
    "one must fast on every Ekadashi to attain liberation",
    "a pilgrimage to Kashi washes away all sins",
    "worship should be performed only at sunrise",
    "the soul is reborn exactly seven times before liberation",
    "charity must be given only to temples",
    "one should chant a mantra one hundred and eight times every day",
    "a householder must give up all his property at the age of fifty",
    "only Brahmanas are permitted to practise meditation",
    "the planets decide the fruits of every action",
    "Arjuna had been a sage in his previous birth",
    "eating onions and garlic destroys one's merit",
    "one should never travel during the rainy season",
]
SWAP_WORDS = ["always", "gold", "mountain", "river", "kingdom", "silence", "darkness", "joyfully", "seven",
              "temple", "forever", "anger", "wealth", "music"]


def _replace_citation(answer: str, new_ref: str, index: int = 0) -> str:
    cits = extract_citations(answer)
    if not cits:
        return answer
    c = cits[min(index, len(cits) - 1)]
    raw = c.raw
    m = re.search(r"\d{1,2}\s*[.:]\s*\d{1,3}(?:\s*[-–—]\s*\d{1,3})?", raw)
    new_raw = raw[: m.start()] + new_ref + raw[m.end():] if m else f"BG {new_ref}"
    return answer[: c.span[0]] + new_raw + answer[c.span[1]:]


def fabricate_citation(answer: str, corpus: GitaCorpus, rng: random.Random) -> str | None:
    cits = extract_citations(answer)
    if not cits:
        return None
    ch = cits[0].chapter
    n = corpus.chapter_lengths.get(ch, 20)
    if rng.random() < 0.8:
        ref = f"{ch}.{n + rng.randint(3, 25)}"
    else:
        ref = f"{rng.randint(19, 24)}.{rng.randint(1, 40)}"
    return _replace_citation(answer, ref)


def misattribute_citation(answer: str, corpus: GitaCorpus, rng: random.Random, min_distance: int = 4) -> str | None:
    """Cite a real verse of the same chapter, far enough not to share the claim."""
    cits = extract_citations(answer)
    if not cits:
        return None
    c = cits[0]
    gold = {v for _, v in c.refs}
    n = corpus.chapter_lengths[c.chapter] if c.chapter in corpus.chapter_lengths else 0
    choices = [v for v in range(1, n + 1) if all(abs(v - g) >= min_distance for g in gold)]
    if not choices:
        return None
    return _replace_citation(answer, f"{c.chapter}.{rng.choice(choices)}")


def wrong_speaker(answer: str, gold_speaker: str, rng: random.Random) -> str | None:
    wrong = {"Krishna": "Arjuna", "Arjuna": "Krishna", "Sanjaya": "Krishna", "Dhritarashtra": "Arjuna"}[gold_speaker]
    names = r"\b(Krishna|Arjuna|Sanjaya|Dhritarashtra)\b(?=\s+(?:says|said|tells|told|declares|asks|replies|teaches|"\
            r"explains|urges|promises|answers|agrees|names|calls|reports))"
    if re.search(names, answer):
        return re.sub(names, wrong, answer, count=1)
    if not extract_citations(answer):
        return None
    first = answer[0].lower() + answer[1:]
    return f"{wrong} says that {first}"


def _alter_words(text: str, rng: random.Random, k: int = 3) -> str:
    toks = text.split()
    idx = [i for i, t in enumerate(toks) if re.sub(r"\W", "", t).lower() not in STOPWORDS and len(t) > 3]
    rng.shuffle(idx)
    for i in idx[:k]:
        punct = re.sub(r"[\w'’-]", "", toks[i][-1:]) if toks[i] else ""
        toks[i] = rng.choice(SWAP_WORDS) + punct
    return " ".join(toks)


def misquote(answer: str, corpus: GitaCorpus, rng: random.Random, gold_ref: str) -> str | None:
    """Alter an existing quotation, or add a corrupted quotation of the gold verse."""
    m = re.search(r"[\"“]([^\"”]{8,})[\"”]", answer)
    if m:
        bad = _alter_words(m.group(1), rng, k=3)
        return answer[: m.start(1)] + bad + answer[m.end(1):]
    verse = corpus.get(gold_ref)
    if verse is None:
        return None
    words = verse.english.split()
    if len(words) < 8:
        return None
    start = rng.randint(0, max(0, len(words) - 10))
    fragment = " ".join(words[start: start + 10]).strip(",;:")
    bad = _alter_words(fragment, rng, k=3)
    return f"{answer.rstrip()} In the words of the text, \"{bad}\" (BG {gold_ref})."


def unsupported_claim(answer: str, rng: random.Random, gold_ref: str) -> str:
    return f"{answer.rstrip()} The Gita also teaches that {rng.choice(UNSUPPORTED_TEACHINGS)} (BG {gold_ref})."


def accept_false_premise(question: str, presupposition: str, evidence_ref: str) -> str:
    p = presupposition.rstrip("?. ")
    return f"Yes. In BG {evidence_ref} the Gita states that {p[0].lower() + p[1:]}."


def answer_out_of_scope(topic: str, rng: random.Random) -> str:
    refs = ["2.47", "3.35", "6.16", "18.66", "2.14"]
    return (f"The Gita teaches that {topic} should be approached without attachment to results, "
            f"since one's concern is with action alone (BG {rng.choice(refs)}).")


def answer_nonexistent(ref_text: str) -> str:
    return f"{ref_text} teaches that the seeker should meditate at dawn with a calm and steady mind."
