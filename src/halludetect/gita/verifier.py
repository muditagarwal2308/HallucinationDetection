"""Verse-level verification of an answer (the core of VGHD)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from ..nli import LexicalNLI
from ..nli.base import NLIModel
from .citations import Citation
from .claims import Claim, segment_claims
from .corpus import GitaCorpus, Verse
from .premise import PremiseAnalysis, PremiseChecker, is_refusal


class IssueType(str, Enum):
    FABRICATED_CITATION = "fabricated_citation"
    MISATTRIBUTED_CITATION = "misattributed_citation"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    MISQUOTATION = "misquotation"
    WRONG_SPEAKER = "wrong_speaker"
    FALSE_PREMISE_ACCEPTED = "false_premise_accepted"
    OUT_OF_SCOPE_ANSWERED = "out_of_scope_answered"
    IMPRECISE_CITATION = "imprecise_citation"
    UNCITED_CLAIM = "uncited_claim"


SEVERITY = {
    IssueType.FABRICATED_CITATION: "critical", IssueType.FALSE_PREMISE_ACCEPTED: "critical",
    IssueType.OUT_OF_SCOPE_ANSWERED: "critical", IssueType.MISATTRIBUTED_CITATION: "major",
    IssueType.UNSUPPORTED_CLAIM: "major", IssueType.MISQUOTATION: "major", IssueType.WRONG_SPEAKER: "major",
    IssueType.IMPRECISE_CITATION: "minor", IssueType.UNCITED_CLAIM: "minor",
}
HALLUCINATION_TYPES = {t for t, s in SEVERITY.items() if s in ("critical", "major")}
# Names and meta words that appear in claims but rarely in the verse text itself.
GITA_LIGHT_TERMS = ("gita", "bhagavad", "krishna", "arjuna", "sanjaya", "dhritarashtra", "lord", "verse", "text",
                    "end", "ends", "present")


@dataclass
class Issue:
    type: IssueType
    message: str
    claim_index: int | None = None
    citation: str | None = None
    suggestion: str | None = None
    replacement: dict | None = None

    @property
    def severity(self) -> str:
        return SEVERITY[self.type]

    def as_dict(self) -> dict:
        return {"type": self.type.value, "severity": self.severity, "message": self.message,
                "claim_index": self.claim_index, "citation": self.citation, "suggestion": self.suggestion}


@dataclass
class CitedVerseCheck:
    citation: str
    ref: str
    exists: bool
    entailment: float = 0.0
    contradiction: float = 0.0
    speaker: str | None = None
    english: str | None = None


@dataclass
class QuoteCheck:
    quote: str
    match_cited: float
    best_ref: str | None
    best_match: float
    status: str  # verified | misquoted | misattributed | not_found


@dataclass
class ClaimCheck:
    claim: Claim
    cited: list[CitedVerseCheck] = field(default_factory=list)
    support: float = 0.0
    best_ref: str | None = None
    best_support: float = 0.0
    quotes: list[QuoteCheck] = field(default_factory=list)
    speaker_ok: bool | None = None
    status: str = "supported"
    risk: float = 0.0

    def as_dict(self) -> dict:
        return {
            "index": self.claim.index, "claim": self.claim.text, "core": self.claim.core,
            "citations": [c.label for c in self.claim.citations], "speaker": self.claim.speaker,
            "cited": [c.__dict__ for c in self.cited], "support": round(self.support, 3),
            "best_ref": self.best_ref, "best_support": round(self.best_support, 3),
            "quotes": [q.__dict__ for q in self.quotes], "speaker_ok": self.speaker_ok,
            "status": self.status, "risk": round(self.risk, 3),
        }


@dataclass
class VerificationReport:
    question: str
    answer: str
    premise: PremiseAnalysis
    claims: list[ClaimCheck]
    issues: list[Issue]
    score: float
    verdict: str  # faithful | hallucinated | correct_abstention | over_abstention

    @property
    def hallucinated(self) -> bool:
        return any(i.type in HALLUCINATION_TYPES for i in self.issues)

    @property
    def types(self) -> set[str]:
        return {i.type.value for i in self.issues if i.type in HALLUCINATION_TYPES}

    def as_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "verdict": self.verdict,
                "score": round(self.score, 4), "hallucinated": self.hallucinated, "types": sorted(self.types),
                "premise": self.premise.as_dict(), "claims": [c.as_dict() for c in self.claims],
                "issues": [i.as_dict() for i in self.issues]}


class VerseVerifier:
    """Checks citations, claim support, quotations, speakers and premises of a Gita answer.

    Parameters
    ----------
    corpus : the verse corpus (701 verses, Besant numbering, 700-verse mapping).
    nli : NLI model; defaults to :class:`LexicalNLI` weighted by corpus IDF.
    support_threshold : entailment probability at which a verse is taken to support a claim.
    """

    def __init__(self, corpus: GitaCorpus, nli: NLIModel | None = None, support_threshold: float = 0.5,
                 quote_threshold: float = 0.92, search_k: int = 12, check_premise: bool = True,
                 check_quotes: bool = True, check_speaker: bool = True):
        self.corpus = corpus
        self.nli = nli or LexicalNLI(idf=corpus.idf, light_terms=GITA_LIGHT_TERMS)
        self.support_threshold = support_threshold
        self.quote_threshold = quote_threshold
        self.search_k = search_k
        self.check_premise = check_premise
        self.check_quotes = check_quotes
        self.check_speaker = check_speaker
        self.premise_checker = PremiseChecker(corpus, self.nli, support_threshold)

    # ---------------------------------------------------------------- helpers
    def _support(self, verses: list[Verse], statement: str) -> tuple[float, float]:
        """Max entailment over the cited verses individually and jointly (a claim may span verses)."""
        if not verses:
            return 0.0, 0.0
        premises = [v.premise() for v in verses]
        if len(verses) > 1:
            premises.append(" ".join(v.premise() for v in verses))
        res = self.nli.predict_batch([(p, statement) for p in premises])
        return max(r.entailment for r in res), max(r.contradiction for r in res)

    def _best_verse(self, statement: str, extra: list[Verse]) -> tuple[Verse | None, float]:
        pool = [v for v, _ in self.corpus.search(statement, k=self.search_k)]
        for v in extra:
            if v not in pool:
                pool.append(v)
        if not pool:
            return None, 0.0
        res = self.nli.predict_batch([(v.premise(), statement) for v in pool])
        i = max(range(len(pool)), key=lambda j: res[j].entailment)
        return pool[i], res[i].entailment

    def _resolve(self, citation: Citation) -> list[tuple[str, list[Verse]]]:
        out = []
        for ch, v in citation.refs:
            out.append((f"{ch}.{v}", self.corpus.candidates(ch, v)))
        return out

    def _check_quote(self, quote: str, cited: list[Verse]) -> QuoteCheck:
        m_cited = max((self.corpus.quote_match(quote, v) for v in cited), default=0.0)
        found = self.corpus.find_quote(quote, k=1)
        best_ref, best = (found[0][0].ref, found[0][1]) if found else (None, 0.0)
        if cited and m_cited >= self.quote_threshold:
            status = "verified"
        elif best >= self.quote_threshold and (not cited or best_ref not in {v.ref for v in cited}):
            status = "misattributed" if cited else "verified"
        elif max(m_cited, best) >= 0.6:
            status = "misquoted"
        else:
            status = "not_found"
        return QuoteCheck(quote, round(m_cited, 3), best_ref, round(best, 3), status)

    # ------------------------------------------------------------------ main
    def verify(self, question: str, answer: str, premise: PremiseAnalysis | None = None) -> VerificationReport:
        if premise is None:
            premise = self.premise_checker.analyze(question) if self.check_premise else PremiseAnalysis()
        issues: list[Issue] = []
        checks: list[ClaimCheck] = []
        refused = is_refusal(answer)

        if premise.is_problematic:
            if refused:
                verdict = "correct_abstention"
                return VerificationReport(question, answer, premise, [], [], 0.05, verdict)
            itype = IssueType.OUT_OF_SCOPE_ANSWERED if premise.kind == "out_of_scope" \
                else IssueType.FALSE_PREMISE_ACCEPTED
            issues.append(Issue(itype, f"The answer accepts a problematic question: {premise.message}",
                                suggestion="Abstain or correct the premise."))

        for claim in segment_claims(answer):
            checks.append(self._check_claim(claim, issues))

        score = max([c.risk for c in checks] + [0.0])
        if premise.is_problematic and not refused:
            score = max(score, 0.95)
        if refused and not premise.is_problematic and not any(c.cited for c in checks):
            verdict = "over_abstention"
        elif any(i.type in HALLUCINATION_TYPES for i in issues):
            verdict = "hallucinated"
        else:
            verdict = "faithful"
        return VerificationReport(question, answer, premise, checks, issues, score, verdict)

    def _check_claim(self, claim: Claim, issues: list[Issue]) -> ClaimCheck:
        check = ClaimCheck(claim)
        statement = claim.core or claim.text
        cited_verses: list[Verse] = []
        fabricated = False
        for cit in claim.citations:
            for ref, cands in self._resolve(cit):
                if not cands:
                    fabricated = True
                    ch, v = (int(x) for x in ref.split("."))
                    n = self.corpus.chapter_lengths.get(ch)
                    why = f"chapter {ch} has only {n} verses" if n else "the Gita has only 18 chapters"
                    check.cited.append(CitedVerseCheck(cit.label, ref, False))
                    issues.append(Issue(IssueType.FABRICATED_CITATION, f"BG {ref} does not exist ({why}).",
                                        claim.index, cit.raw))
                    continue
                cited_verses.extend(cands)
                for verse in cands:
                    e, c = self._support([verse], statement)
                    check.cited.append(CitedVerseCheck(cit.label, verse.ref, True, round(e, 3), round(c, 3),
                                                       verse.speaker, verse.english))

        support, _ = self._support(cited_verses, statement)
        check.support = support

        # quotations: a verified quotation is itself evidence for the citation
        for q in (claim.quotes if self.check_quotes else []):
            qc = self._check_quote(q, cited_verses)
            check.quotes.append(qc)
            if qc.status == "verified" and cited_verses:
                check.support = max(check.support, qc.match_cited)
            elif qc.status == "misattributed":
                issues.append(Issue(IssueType.MISATTRIBUTED_CITATION,
                                    f"The quotation comes from BG {qc.best_ref}, not from the cited verse.",
                                    claim.index, suggestion=f"BG {qc.best_ref}",
                                    replacement={"citation": claim.citations[0].raw if claim.citations else None,
                                                 "ref": qc.best_ref}))
                check.status = "misattributed"
            elif qc.status in ("misquoted", "not_found"):
                # The claim cites a verse and misquotes it: restore from the cited verse that matches best;
                # fall back to the best match anywhere only when no verse is cited.
                if cited_verses:
                    src = max(cited_verses, key=lambda v: self.corpus.quote_match(q, v))
                    where = f" of BG {src.ref} (similarity {qc.match_cited:.2f})."
                else:
                    src = self.corpus.get(qc.best_ref) if qc.best_ref else None
                    where = f" (closest: BG {qc.best_ref}, similarity {qc.best_match:.2f})." if src else "."
                issues.append(Issue(IssueType.MISQUOTATION, "The quotation does not match the text" + where,
                                    claim.index, suggestion=src.english if src else None,
                                    replacement={"quote": q, "ref": src.ref} if src else None))
                check.status = "misquoted"

        best, best_support = self._best_verse(statement, cited_verses)
        check.best_ref, check.best_support = (best.ref if best else None), best_support
        supported = check.support >= self.support_threshold

        if fabricated:
            check.status, check.risk = "fabricated_citation", 1.0
            if best is not None and best_support >= self.support_threshold:
                issues[-1].suggestion = f"BG {best.ref}"
                issues[-1].replacement = {"citation": claim.citations[0].raw, "ref": best.ref}
        elif cited_verses and not supported and check.status == "supported":
            near = best is not None and any(best.chapter == v.chapter and abs(best.verse - v.verse) <= 2
                                            for v in cited_verses)
            if best is not None and best_support >= self.support_threshold and best not in cited_verses:
                if near:
                    check.status, check.risk = "imprecise_citation", 0.35
                    issues.append(Issue(IssueType.IMPRECISE_CITATION,
                                        f"The claim is stated in the neighbouring verse BG {best.ref}.",
                                        claim.index, suggestion=f"BG {best.ref}",
                                        replacement={"citation": claim.citations[0].raw, "ref": best.ref}))
                else:
                    check.status = "misattributed"
                    check.risk = 0.75 + 0.25 * (1 - check.support)
                    cited = ", ".join(f"BG {v.ref}" for v in cited_verses)
                    issues.append(Issue(IssueType.MISATTRIBUTED_CITATION,
                                        f"{cited} does not support this claim; BG {best.ref} does.",
                                        claim.index, suggestion=f"BG {best.ref}",
                                        replacement={"citation": claim.citations[0].raw, "ref": best.ref}))
            else:
                check.status = "unsupported"
                check.risk = 0.6 + 0.4 * (1 - max(check.support, best_support))
                issues.append(Issue(IssueType.UNSUPPORTED_CLAIM,
                                    "No verse of the Gita supports this claim; it may be an interpretation or "
                                    "commentary presented as the text.", claim.index))
        elif not cited_verses and not (claim.quotes and self.check_quotes):
            attributes_text = claim.speaker is not None or any(
                w in claim.text.lower() for w in ("gita", "gītā", "verse", "teach", "krishna", "arjuna")) or bool(
                re.search(r"\b(?:he|she|it|the lord|the text)\s+(?:also\s+)?(?:says|adds|states|teaches|declares|"
                          r"explains|tells|promises|asserts|instructs|insists)\b", claim.text, re.IGNORECASE))
            if best is not None and best_support >= self.support_threshold:
                check.status, check.risk = "uncited", 0.2
                issues.append(Issue(IssueType.UNCITED_CLAIM, f"Supported by BG {best.ref}, but no verse is cited.",
                                    claim.index, suggestion=f"BG {best.ref}"))
            elif attributes_text:
                check.status = "unsupported"
                check.risk = 0.6 + 0.4 * (1 - best_support)
                issues.append(Issue(IssueType.UNSUPPORTED_CLAIM,
                                    "This statement is attributed to the Gita, but no verse supports it.",
                                    claim.index))
            else:
                check.status, check.risk = "context", 0.1
        else:
            check.risk = max(check.risk, 0.3 * (1 - check.support))
            if check.status in ("misquoted", "misattributed"):
                check.risk = max(check.risk, 0.8)

        # speaker attribution
        if self.check_speaker and claim.speaker and cited_verses:
            speakers = {v.speaker for v in cited_verses}
            narrated = any(v.speaker == "Sanjaya" and '"' in v.english for v in cited_verses)
            check.speaker_ok = claim.speaker in speakers or narrated
            if not check.speaker_ok:
                actual = "/".join(sorted(speakers))
                issues.append(Issue(IssueType.WRONG_SPEAKER,
                                    f"{claim.speaker} is named as the speaker, but BG "
                                    f"{', '.join(v.ref for v in cited_verses)} is spoken by {actual}.",
                                    claim.index, suggestion=actual,
                                    replacement={"speaker": claim.speaker, "correct": actual}))
                check.risk = max(check.risk, 0.8)
                if check.status == "supported":
                    check.status = "wrong_speaker"
        return check
