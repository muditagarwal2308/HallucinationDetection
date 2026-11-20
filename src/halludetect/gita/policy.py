"""Repair-or-abstain policy driven by the verse-level verification report."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .citations import extract_citations
from .corpus import GitaCorpus
from .verifier import IssueType, VerificationReport


@dataclass
class Decision:
    action: str  # accept | repair | regenerate | abstain
    answer: str
    rationale: str
    edits: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"action": self.action, "answer": self.answer, "rationale": self.rationale, "edits": self.edits}


def _retarget_citation(text: str, raw: str, new_ref: str, anchor: str = "") -> str:
    """Replace the citation ``raw`` that belongs to the claim starting at ``anchor``.

    The search starts at the claim's sentence and runs forward, because a citation may sit in a
    following pointer sentence ("This is described in BG 2.20."); other occurrences of the same
    reference elsewhere in the answer are left untouched.
    """
    if not raw:
        return text
    start = max(text.find(anchor), 0) if anchor else 0
    pos = text.find(raw, start)
    if pos < 0:
        pos = text.find(raw)
    if pos < 0:
        return text
    m = re.search(r"\d{1,2}\s*[.:]\s*\d{1,3}(?:\s*[-–—]\s*\d{1,3})?", raw)
    new_raw = (raw[: m.start()] + new_ref + raw[m.end():]) if m else f"BG {new_ref}"
    return text[:pos] + new_raw + text[pos + len(raw):]


def _best_window(quote: str, source: str) -> str:
    """The passage of ``source`` that the (mis)quotation was taken from.

    The best-matching window of the same length is located and then widened to clause boundaries,
    so the restored quotation is a complete phrase of the verse; when the quotation covers most of
    the verse, the whole verse is returned.
    """
    from difflib import SequenceMatcher

    q, words = quote.split(), source.split()
    n = len(q)
    if len(words) <= n or n >= 0.6 * len(words):
        return source.strip().rstrip(" ,;:")
    best, best_i = -1.0, 0
    for i in range(len(words) - n + 1):
        r = SequenceMatcher(None, quote.lower(), " ".join(words[i: i + n]).lower()).ratio()
        if r > best:
            best, best_i = r, i
    start, end = best_i, best_i + n
    while start > 0 and not words[start - 1].endswith((".", ";", ":", "!", "?", ",")):
        start -= 1
    while end < len(words) and not words[end - 1].endswith((".", ";", ":", "!", "?")):
        end += 1
    return " ".join(words[start:end]).strip(" ,;:")


def _fix_speaker(sentence: str, wrong: str, correct: str) -> str:
    """Correct the speaker of a claim; "Arjuna tells Krishna" becomes "Krishna tells Arjuna"."""
    if re.search(rf"\b{re.escape(correct)}\b", sentence):
        tmp = re.sub(rf"\b{re.escape(wrong)}\b", "\x00", sentence)
        tmp = re.sub(rf"\b{re.escape(correct)}\b", wrong, tmp)
        return tmp.replace("\x00", correct)
    return re.sub(rf"\b{re.escape(wrong)}\b", correct, sentence, count=1)


class RepairPolicy:
    """Turns a :class:`VerificationReport` into an action.

    * problematic premise accepted → **abstain** with an explanation;
    * citation / quotation / speaker errors → **repair** deterministically
      (retarget the citation, restore the quotation, correct the speaker);
    * claims that no verse supports → removed; if nothing remains → **abstain**
      (or **regenerate** when an LLM is available, see :mod:`halludetect.gita.pipeline`);
    * otherwise → **accept**.
    """

    def __init__(self, corpus: GitaCorpus):
        self.corpus = corpus

    def decide(self, report: VerificationReport) -> Decision:
        types = {i.type for i in report.issues}
        if report.verdict == "correct_abstention":
            return Decision("accept", report.answer, "The answer correctly declines a problematic question.")
        if IssueType.FALSE_PREMISE_ACCEPTED in types or IssueType.OUT_OF_SCOPE_ANSWERED in types:
            msg = report.premise.message or "The question cannot be answered from the Bhagavad Gita."
            ev = report.premise.evidence[0]["ref"] if report.premise.evidence else None
            if ev and report.premise.kind == "false_premise":
                v = self.corpus.get(ev)
                msg += f" See BG {ev}: \"{v.english}\"" if v else ""
            return Decision("abstain", msg, "The answer accepted a false premise or an out-of-scope question.")
        if not report.hallucinated and IssueType.IMPRECISE_CITATION not in types:
            return Decision("accept", report.answer, "Every claim is supported by the verses it cites.")

        text = report.answer
        edits: list[str] = []
        drop_claims: set[int] = set()
        claims = {c.claim.index: c.claim for c in report.claims}

        def anchor(idx: int | None) -> str:
            if idx is None or idx not in claims:
                return ""
            return claims[idx].sentence if claims[idx].sentence in text else claims[idx].text[:40]

        for issue in report.issues:
            rep = issue.replacement or {}
            if issue.type in (IssueType.MISATTRIBUTED_CITATION, IssueType.IMPRECISE_CITATION,
                              IssueType.FABRICATED_CITATION):
                if rep.get("ref") and rep.get("citation"):
                    text = _retarget_citation(text, rep["citation"], rep["ref"], anchor(issue.claim_index))
                    edits.append(f"citation {rep['citation']} → BG {rep['ref']}")
                elif issue.type == IssueType.FABRICATED_CITATION and issue.claim_index is not None:
                    drop_claims.add(issue.claim_index)
            elif issue.type == IssueType.WRONG_SPEAKER and rep.get("speaker") and issue.claim_index is not None:
                sent = claims[issue.claim_index].sentence
                fixed = _fix_speaker(sent, rep["speaker"], rep["correct"].split("/")[0])
                text = text.replace(sent, fixed, 1)
                edits.append(f"speaker {rep['speaker']} → {rep['correct']}")
            elif issue.type == IssueType.MISQUOTATION:
                if rep.get("quote") and rep.get("ref"):
                    v = self.corpus.get(rep["ref"])
                    is_iast = bool(re.search(r"[āīūṛṝḷṃṅñṭḍṇśṣḥ]", rep["quote"]))
                    correct = _best_window(rep["quote"], v.iast if is_iast else v.english)
                    text = text.replace(rep["quote"], correct, 1)
                    idx = issue.claim_index
                    cits = extract_citations(claims[idx].sentence) if idx is not None else []
                    if cits and all(r != tuple(map(int, v.ref.split("."))) for r in cits[0].refs):
                        text = _retarget_citation(text, cits[0].raw, v.ref, anchor(issue.claim_index))
                    edits.append(f"quotation restored from BG {v.ref}")
                elif issue.claim_index is not None:
                    drop_claims.add(issue.claim_index)
            elif issue.type == IssueType.UNSUPPORTED_CLAIM and issue.claim_index is not None:
                drop_claims.add(issue.claim_index)
        for idx in sorted(drop_claims, reverse=True):
            sent = claims[idx].sentence
            text = " ".join(text.replace(sent, "", 1).split())
            edits.append(f"removed unsupported claim: “{claims[idx].text[:80]}”")
        if not extract_citations(text) and drop_claims:
            return Decision("abstain", "The verses do not support an answer to this question.",
                            "All claims were unsupported by the text.", edits)
        return Decision("repair", text, "Verse-level errors were corrected.", edits)
